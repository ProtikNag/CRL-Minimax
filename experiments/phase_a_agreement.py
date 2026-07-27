"""Phase A: cheap mid/late-game policy-agreement diagnostic (no state-restore).

Question: does the consolidated GLOBAL policy match the LOCAL/expert policy only
at the START of an episode (the 50 no-op starts we currently evaluate), or also
in the MIDDLE and END? If agreement is high early and collapses late, that
explains why the discounted-V / no-op-start metric looks OK while the full-episode
game score does not -- and justifies Phase B (evaluating from restored mid-game
states).

Method (forward passes only, no env-state restore):
  1. Roll the EXPERT greedily from several deterministic no-op starts -> its own
     successful trajectory (states it actually visits, no dead/invalid states).
  2. Per visited state record (obs, expert greedy action, clipped reward).
  3. Discounted return-to-go rtg_t = r_t + gamma*rtg_{t+1}; KEEP ONLY states with
     rtg > 0 (a state with no future reward carries no signal -- point 6).
  4. Bucket each kept state by fractional episode position: early/mid/late thirds.
  5. For each global checkpoint (just-learned head vs FINAL, post-forgetting),
     PER STATE compare its greedy action to the expert's, then report the
     agreement rate + mean prob-on-expert-action PER BUCKET (case by case, not one
     global average -- point 5).

    python -m experiments.phase_a_agreement --run results/consolidate10_seed0 \
        --games Breakout Qbert
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from crl.config import EnvConfig, PolicyConfig
from crl.envs import make_family
from crl.policies import make_policy

BUCKETS = ["early", "mid", "late"]


def _bucket(frac: float) -> int:
    return 0 if frac < 1 / 3 else (1 if frac < 2 / 3 else 2)


@torch.no_grad()
def _expert_trajectories(expert, task, device, gamma, n_starts, cap):
    """Greedy expert rollouts from no-op counts 1..n_starts. Returns a list of
    per-episode dicts: obs (T,4,84,84) uint8, act (T,), rtg (T,), frac (T,)."""
    episodes = []
    for noop in range(1, n_starts + 1):
        env = task.make_env(clip_rewards=True)          # training-scale reward
        obs, _ = env.reset(seed=10_000 + noop)
        for _ in range(noop):                            # deterministic no-op prefix
            obs, _, term, trunc, _ = env.step(0)
            if term or trunc:
                obs, _ = env.reset(seed=10_000 + noop)
        O, A, R = [], [], []
        for _ in range(cap):
            ot = torch.as_tensor(np.asarray(obs), device=device).unsqueeze(0)
            a = int(expert.dist(ot, 0).logits.argmax(-1).item())
            nobs, r, term, trunc, _ = env.step(a)
            O.append(np.asarray(obs, dtype=np.uint8)); A.append(a); R.append(float(r))
            obs = nobs
            if term or trunc:
                break
        env.close()
        T = len(R)
        if T == 0:
            continue
        rtg = np.zeros(T)
        acc = 0.0
        for t in range(T - 1, -1, -1):
            acc = R[t] + gamma * acc
            rtg[t] = acc
        frac = (np.arange(T) + 0.5) / T
        episodes.append({"obs": np.stack(O), "act": np.array(A),
                         "rtg": rtg, "frac": frac})
    return episodes


@torch.no_grad()
def _agreement(global_pol, task_id, obs_u8, expert_act, device, bs=512):
    """Per-state: global greedy action, prob on expert's action, and critic value."""
    gact = np.empty(len(obs_u8), dtype=np.int64)
    pexp = np.empty(len(obs_u8), dtype=np.float32)
    gval = np.empty(len(obs_u8), dtype=np.float32)
    for i in range(0, len(obs_u8), bs):
        ot = torch.as_tensor(obs_u8[i:i + bs], device=device)
        dist, val = global_pol.dist_value(ot, task_id)
        logits = dist.logits
        gact[i:i + bs] = logits.argmax(-1).cpu().numpy()
        probs = torch.softmax(logits, -1)
        ea = torch.as_tensor(expert_act[i:i + bs], device=device).unsqueeze(-1)
        pexp[i:i + bs] = probs.gather(-1, ea).squeeze(-1).cpu().numpy()
        gval[i:i + bs] = val.cpu().numpy()
    return gact, pexp, gval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--games", nargs="+", required=True)
    ap.add_argument("--n_starts", type=int, default=8)
    ap.add_argument("--cap", type=int, default=3000)
    ap.add_argument("--out", default="diagnostics/phase_a")
    args = ap.parse_args()
    run = Path(args.run)
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    all_games = [t["game"] for t in cfg["env"]["tasks"]]
    gamma = float(cfg["env"]["params"]["gamma"])
    expert_dir = cfg["ppo"]["expert_dir"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # the global (multi-head) checkpoints: FINAL (post-forgetting) + per-task
    def load_global(path):
        fam = make_family(EnvConfig(family="atari", params=dict(cfg["env"]["params"]),
                                    tasks=cfg["env"]["tasks"]))
        pol = make_policy(PolicyConfig(kind="impala_ac_multihead", hidden_sizes=[512],
                                       task_conditioned=False), fam).to(device)
        pol.load_state_dict(torch.load(path, map_location=device))
        pol.eval()
        return pol
    final_global = load_global(run / "final_policy.pt")

    report = {}
    for game in args.games:
        tid = all_games.index(game)
        # expert
        env = EnvConfig(family="atari", params=dict(cfg["env"]["params"]),
                        tasks=[{"game": game, "threshold": 0.0}])
        fam = make_family(env)
        task = fam.tasks[0]
        expert = make_policy(PolicyConfig(kind="impala_ac", hidden_sizes=[512],
                                          task_conditioned=False), fam).to(device)
        expert.load_state_dict(torch.load(f"{expert_dir}/{game}/best_model.pt",
                                          map_location=device))
        expert.eval()
        just = load_global(run / f"global_after_task{tid + 1}.pt")  # right after learning

        eps = _expert_trajectories(expert, task, device, gamma, args.n_starts, args.cap)
        obs = np.concatenate([e["obs"] for e in eps])
        act = np.concatenate([e["act"] for e in eps])
        rtg = np.concatenate([e["rtg"] for e in eps])
        frac = np.concatenate([e["frac"] for e in eps])
        keep = rtg > 0                                   # point 6: drop zero-future states
        obs, act, frac, rtg = obs[keep], act[keep], frac[keep], rtg[keep]
        buck = np.array([_bucket(f) for f in frac])

        game_rep = {"n_states_kept": int(keep.sum()), "n_states_total": int(keep.size),
                    "mean_ep_len": float(np.mean([len(e["act"]) for e in eps])),
                    "checkpoints": {}}
        for name, pol in [("just_learned", just), ("final", final_global)]:
            gact, pexp, gval = _agreement(pol, tid, obs, act, device)
            agree = (gact == act)
            # per-state value ratio global_value / expert_return_to_go (case by case,
            # point 5); rtg>0 already guaranteed. Clip to [0,2] to tame outliers.
            vratio = np.clip(gval / rtg, 0.0, 2.0)
            per_bucket = {}
            for b, bn in enumerate(BUCKETS):
                m = buck == b
                per_bucket[bn] = {
                    "n": int(m.sum()),
                    "action_agree": float(agree[m].mean()) if m.any() else float("nan"),
                    "prob_on_expert_action": float(pexp[m].mean()) if m.any() else float("nan"),
                    "mean_global_value": float(gval[m].mean()) if m.any() else float("nan"),
                    "mean_expert_rtg": float(rtg[m].mean()) if m.any() else float("nan"),
                    "median_value_ratio": float(np.median(vratio[m])) if m.any() else float("nan"),
                }
            game_rep["checkpoints"][name] = per_bucket
        report[game] = game_rep
        print(f"\n=== {game} (global head {tid}) — kept {game_rep['n_states_kept']}/"
              f"{game_rep['n_states_total']} states, mean ep len {game_rep['mean_ep_len']:.0f} ===")
        for metric, lab, sc in [("action_agree", "action-agree", 100.0),
                                ("median_value_ratio", "V_G/rtg (median)", 1.0)]:
            print(f"  [{lab}]")
            print(f"    {'checkpoint':<14}" + "".join(f"{b:>14}" for b in BUCKETS))
            for name in ("just_learned", "final"):
                pb = game_rep["checkpoints"][name]
                suff = "%" if sc == 100.0 else ""
                print(f"    {name:<14}" + "".join(
                    f"{pb[b][metric]*sc:>13.2f}{suff}" for b in BUCKETS))

    outd = Path(args.out) / run.name
    outd.mkdir(parents=True, exist_ok=True)
    (outd / "phase_a_agreement.json").write_text(json.dumps(report, indent=2))

    # figure: 2 rows (action-agree, value ratio) x games; per checkpoint per bucket
    ng = len(args.games)
    x = np.arange(3)
    fig, axes = plt.subplots(2, ng, figsize=(4.2 * ng, 6.4), squeeze=False)
    rows = [("action_agree", "action agreement w/ expert", (0, 1.02), None),
            ("median_value_ratio", "median V_global / expert return-to-go", (0, 1.6), 1.0)]
    for r, (metric, ylab, ylim, hline) in enumerate(rows):
        for j, game in enumerate(args.games):
            ax = axes[r][j]
            for name, c in [("just_learned", "C0"), ("final", "C3")]:
                pb = report[game]["checkpoints"][name]
                ax.plot(x, [pb[b][metric] for b in BUCKETS], "-o", c=c, label=name)
            if hline is not None:
                ax.axhline(hline, ls="--", c="0.5", lw=1)  # V_G == expert (perfect)
            ax.set_xticks(x); ax.set_xticklabels(BUCKETS)
            ax.set_ylim(*ylim); ax.grid(alpha=0.3)
            if r == 0:
                ax.set_title(f"{game}\n(kept {report[game]['n_states_kept']} states)", fontsize=9)
            ax.set_ylabel(ylab, fontsize=8)
            ax.set_xlabel("episode phase")
            if r == 0 and j == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Phase A — does the global match the expert in mid/late game? "
                 "(over expert-visited states, rtg>0)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("png", "svg"):
        (outd / ext).mkdir(exist_ok=True)
        fig.savefig(outd / ext / ("phase_a_agreement." + ext), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[phase A] report -> {outd}/phase_a_agreement.json ; figures -> {outd}/{{png,svg}}/")


if __name__ == "__main__":
    main()
