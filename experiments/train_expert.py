"""Train ONE single-task Atari expert (Impala-CNN) to convergence, fully logged.

Unconstrained standard PPO on a single game, from a shared init, with NO
iteration/frame cap beyond a generous safety bound: it stops when the greedy
score plateaus. Everything needed to reuse or debug the model is stored under
``<out>/``:

  best_model.pt   -- weights at the best greedy-100 score (use this as the expert)
  last_model.pt   -- final weights
  checkpoints/    -- periodic snapshots (ckpt_<frames>.pt)
  train_log.csv   -- per-iteration: frames / agent-steps / episodes / wall-time,
                     smoothed training return, and greedy-100 score on eval iters
  meta.json       -- frames/iters/episodes-at-best + totals (for fair budgeting)
  reward_curve.png

Reported score = greedy (argmax) rollouts, per the project rule (never stochastic).

    python -m experiments.train_expert --game Pong --out experts/Pong \
        --init experts/_shared_init.pt --seed 0
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from crl.config import EnvConfig, PPOConfig, PolicyConfig
from crl.envs import make_family
from crl.policies import make_policy
from crl.ppo.collector import RolloutCollector
from crl.ppo.evaluate import evaluate_value_and_score
from crl.ppo.trainer import PPOTrainer

# Fixed PPO recipe shared by ALL experts (identical arch/optimizer/hypers).
HYPERS = dict(n_envs=16, n_steps=128, ppo_epochs=4, num_minibatches=4,
              clip_ratio=0.1, gae_lambda=0.95, vf_coef=0.5, ent_coef=0.01,
              max_grad_norm=0.5, lr=2.5e-4, normalize_advantage=True)
ENV_PARAMS = dict(gamma=0.99, max_steps=0, frame_skip=4, frame_stack=4,
                  noop_max=30, terminal_on_life_loss=False,
                  repeat_action_probability=0.0, clip_rewards=True)


def build(game, device):
    env = EnvConfig(family="atari", params=dict(ENV_PARAMS),
                    tasks=[{"game": game, "threshold": 0.0}])
    fam = make_family(env)
    pol = make_policy(PolicyConfig(kind="impala_ac", hidden_sizes=[512],
                                   task_conditioned=False), fam).to(device)
    return fam, pol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default="")           # shared init state_dict
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--eval-interval", type=int, default=100)   # iters between greedy-100 evals
    ap.add_argument("--eval-episodes", type=int, default=100)   # greedy rollouts (project rule)
    ap.add_argument("--patience", type=int, default=6)          # non-improving evals -> stop
    ap.add_argument("--min-frames", type=float, default=5e6)    # floor before plateau can stop
    ap.add_argument("--max-frames", type=float, default=60e6)   # generous safety cap
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device in ("auto", "cuda")
                                     and torch.cuda.is_available()) else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out = Path(args.out); (out / "checkpoints").mkdir(parents=True, exist_ok=True)

    fam, policy = build(args.game, device)
    task = fam.tasks[0]
    init_hash = ""
    if args.init:
        sd = torch.load(args.init, map_location=device)
        policy.load_state_dict(sd)
        init_hash = str(hash(tuple(float(v.sum()) for v in sd.values())))
    ppo = PPOConfig(**HYPERS)
    trainer = PPOTrainer(ppo, device, logger=None, log_every=1)
    opt = trainer._new_optimizer(policy)
    collector = RolloutCollector(task, ppo.n_envs, ppo.n_steps, device, args.seed)
    frame_skip = ENV_PARAMS["frame_skip"]
    steps_per_iter = ppo.n_envs * ppo.n_steps

    cfg_dump = {"game": args.game, "seed": args.seed, "arch": "impala_ac large (32,64,64)->512",
                "hypers": HYPERS, "env_params": ENV_PARAMS, "eval_episodes": args.eval_episodes,
                "init": args.init, "init_hash": init_hash,
                "n_params": sum(p.numel() for p in policy.parameters())}
    (out / "config.json").write_text(json.dumps(cfg_dump, indent=2))

    log_path = out / "train_log.csv"
    fh = open(log_path, "w", newline="")
    writer = csv.writer(fh)
    writer.writerow(["iter", "frames", "agent_steps", "episodes", "wall_s",
                     "train_return_mean", "greedy_score", "pg_loss", "v_loss",
                     "entropy", "approx_kl", "clipfrac"])

    def greedy_eval():
        _, score, std, n = evaluate_value_and_score(
            policy, task, args.eval_episodes, 16, device, seed=100_000, greedy=True)
        return score, std, n

    recent = deque(maxlen=100)   # smoothed training-return window
    agent_steps = episodes = 0
    best = -1e18; since_best = 0; best_meta = {}
    t0 = time.time()
    stop_reason = "max_frames"
    it = 0
    while True:
        batch = collector.collect(policy, ppo.gae_lambda)
        stats = trainer.optimize_batches(policy, opt, [batch], [1.0])
        agent_steps += steps_per_iter
        frames = agent_steps * frame_skip
        episodes += len(batch.ep_returns)
        recent.extend(batch.ep_returns)
        train_ret = float(np.mean(recent)) if recent else float("nan")

        gscore = ""
        if it % args.eval_interval == 0:
            gscore_val, _, _ = greedy_eval()
            gscore = gscore_val
            torch.save(policy.state_dict(), out / "checkpoints" / f"ckpt_{int(frames)}.pt")
            improved = gscore_val > best * 1.02 + 0.5
            if improved:
                best = gscore_val; since_best = 0
                torch.save(policy.state_dict(), out / "best_model.pt")
                best_meta = {"best_greedy": gscore_val, "frames_at_best": int(frames),
                             "iters_at_best": it, "episodes_at_best": episodes}
            else:
                since_best += 1
            print(f"[{args.game}] it={it} frames={frames/1e6:.1f}M greedy={gscore_val:.1f} "
                  f"best={best:.1f} since_best={since_best} train_ret={train_ret:.1f}")
            if frames >= args.min_frames and since_best >= args.patience:
                stop_reason = "plateau";
                writer.writerow([it, int(frames), agent_steps, episodes,
                                 round(time.time()-t0, 1), round(train_ret, 3), gscore,
                                 round(stats["pg_loss"], 4), round(stats["v_loss"], 4),
                                 round(stats["entropy"], 4), round(stats["approx_kl"], 5),
                                 round(stats["clipfrac"], 4)])
                break
        writer.writerow([it, int(frames), agent_steps, episodes, round(time.time()-t0, 1),
                         round(train_ret, 3), gscore, round(stats["pg_loss"], 4),
                         round(stats["v_loss"], 4), round(stats["entropy"], 4),
                         round(stats["approx_kl"], 5), round(stats["clipfrac"], 4)])
        fh.flush()
        if frames >= args.max_frames:
            break
        it += 1
    collector.close(); fh.close()

    final_score, final_std, _ = greedy_eval()
    torch.save(policy.state_dict(), out / "last_model.pt")
    meta = {**cfg_dump, **best_meta, "final_greedy": final_score, "final_std": final_std,
            "total_frames": int(agent_steps * frame_skip), "total_agent_steps": agent_steps,
            "total_episodes": episodes, "total_iters": it + 1,
            "wall_s": round(time.time() - t0, 1), "stop_reason": stop_reason,
            "random_score": __import__("crl.envs.atari", fromlist=["RANDOM_SCORES"]).RANDOM_SCORES.get(args.game)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    # reward curve: official greedy-100 (sparse) + smoothed training return (dense)
    rows = list(csv.DictReader(open(log_path)))
    def col(name):
        return np.array([float(r[name]) if r[name] not in ("", "nan") else np.nan
                         for r in rows])
    fr = col("frames") / 1e6
    ev_fr, ev_sc = [], []
    for r in rows:
        if r["greedy_score"] not in ("", "nan"):
            ev_fr.append(float(r["frames"]) / 1e6); ev_sc.append(float(r["greedy_score"]))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(fr, col("train_return_mean"), color="#bbb", lw=1,
            label="training return (stochastic, smoothed)")
    ax.plot(ev_fr, ev_sc, color="#1b9e77", marker="o", label="greedy-100 (reported)")
    ax.set_xlabel("frames (millions)"); ax.set_ylabel("raw game score")
    ax.set_title(f"{args.game} expert — reward curve  (best greedy {best:.0f} @ "
                 f"{best_meta.get('frames_at_best',0)/1e6:.1f}M frames)")
    ax.legend()
    fig.savefig(out / "reward_curve.png", dpi=130, bbox_inches="tight")
    print(f"[{args.game}] DONE best={best:.1f} final={final_score:.1f} "
          f"frames={meta['total_frames']/1e6:.1f}M episodes={episodes} reason={stop_reason}")


if __name__ == "__main__":
    main()
