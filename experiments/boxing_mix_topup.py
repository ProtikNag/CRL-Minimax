"""Rebalanced Boxing top-up (DISCARDABLE experiment, v2).

Supersedes the pure-Boxing top-up (which recovered Boxing +36 but crashed the
other 4 because Boxing's unbounded min-max constraint coeff dwarfed retention).
Here: take the FINAL global (final_policy.pt) and run 100 rehearsal iters where
each iter picks ONE task and does a plain PPO step on it -- 40% Boxing, 60% split
across the other 4 (15% each), interleaved randomly. This gives the other games
real rehearsal every few iters while still oversampling Boxing. Eval all 5 before
and after. Writes to results/atari5_rev_ours_seed0_boxmix/ (original untouched).
"""
from __future__ import annotations
import numpy as np
import torch
from crl.config import load_config
from crl.envs import make_family
from crl.policies import make_policy
from crl.logging_utils import RunLogger
from crl.seeding import set_seed
from crl.ppo_continual import PPOAlternationTrainer
from crl.ppo.collector import RolloutCollector

SRC = "results/atari5_rev_ours_seed0"
CFG = "configs/atari5_reversed.yaml"
FINAL = f"{SRC}/final_policy.pt"
N_ITERS = 100
BOX = 1                       # Boxing task index (0=SI,1=Boxing,2=Breakout,3=Pong,4=Qbert)
P_BOX = 0.40                  # 40% Boxing, 60% uniform over the other 4

def main():
    cfg = load_config(CFG)
    set_seed(cfg.experiment.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    family = make_family(cfg.env)
    policy = make_policy(cfg.policy, family).to(dev)
    logger = RunLogger(cfg.experiment.results_dir,
                       f"{cfg.experiment.name}_seed{cfg.experiment.seed}_boxmix",
                       cfg.to_dict())
    tr = PPOAlternationTrainer(cfg, family, policy, logger)
    tr.global_policy.load_state_dict(torch.load(FINAL, map_location=dev))
    tr.global_policy.to(dev)
    gp = tr.global_policy
    tasks = family.tasks
    games = [t.spec.name for t in tasks]
    n = len(tasks)

    def eval_all(tag):
        row = [tr._eval_report(gp, t)[0] for t in tasks]
        print(f"[{tag}] " + " ".join(f"{g.split('-')[-1]}={s:.1f}" for g, s in zip(games, row)), flush=True)
        return row

    before = eval_all("BEFORE")

    collectors = [RolloutCollector(t, cfg.ppo.n_envs, cfg.ppo.n_steps, dev,
                                   cfg.experiment.seed + 700 + i,
                                   async_mode=cfg.ppo.async_envs)
                  for i, t in enumerate(tasks)]
    opt = tr.local_trainer._new_optimizer(gp)
    # per-iter task sampling probs: 40% Boxing, 15% each of the other 4
    probs = np.full(n, (1.0 - P_BOX) / (n - 1)); probs[BOX] = P_BOX
    rng = np.random.default_rng(cfg.experiment.seed + 123)
    counts = np.zeros(n, dtype=int)
    try:
        for it in range(N_ITERS):
            idx = int(rng.choice(n, p=probs))
            counts[idx] += 1
            stream = collectors[idx].collect(gp, cfg.ppo.gae_lambda)
            tr.local_trainer.optimize_batches(gp, opt, [stream], [1.0])
            if (it + 1) % 20 == 0:
                print(f"[mix] it={it+1} counts={dict(zip([g.split('-')[-1] for g in games], counts.tolist()))}", flush=True)
    finally:
        for c in collectors:
            c.close()
    print(f"[mix] task step counts: {dict(zip(games, counts.tolist()))}", flush=True)

    after = eval_all("AFTER")
    print("\n=== BOXING MIX TOP-UP (40% Boxing / 60% rest, 100 iters): before -> after ===")
    for g, b, a in zip(games, before, after):
        print(f"  {g:20s} {b:8.1f} -> {a:8.1f}   ({a-b:+.1f})")
    torch.save(gp.state_dict(), logger.run_dir / "boxmix_after.pt")
    logger.save_json("boxmix_eval.json",
                     {"games": games, "before": before, "after": after,
                      "step_counts": counts.tolist(), "p_box": P_BOX})
    print(f"\nsaved -> {logger.run_dir}/boxmix_after.pt (discardable)")

if __name__ == "__main__":
    main()
