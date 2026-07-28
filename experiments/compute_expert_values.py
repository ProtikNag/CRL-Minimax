"""Compute + cache the fixed expert reference values at a given discount factor.

The consolidation run needs each expert's greedy discounted value V_L as the
constraint reference. Changing gamma (0.99 -> 0.999) invalidates the cached
gamma=0.99 values, so recompute them once here (exact no-op enumeration, same
3000-step eval cap the run uses) and write a cache the run can reuse via
ppo.expert_refs_path -- and so we can pick constraint_tau from the real value
distribution before launching.

    python -m experiments.compute_expert_values --config configs/consolidate10_strict_g999.yaml \
        --out results/expert_refs_g999.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from crl.config import EnvConfig, PolicyConfig
from crl.envs import make_family
from crl.policies import make_policy
from crl.ppo.evaluate import evaluate_value_and_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    params = dict(cfg["env"]["params"])
    games = [t["game"] for t in cfg["env"]["tasks"]]
    gamma = float(params["gamma"])
    expert_dir = cfg["ppo"]["expert_dir"]
    n_envs = int(cfg["ppo"]["n_envs"])
    eval_seed = int(cfg["ppo"]["eval_seed"])
    cap = int(cfg["ppo"].get("eval_max_ep_steps", 3000))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[expert-values] gamma={gamma} cap={cap} device={device} games={len(games)}")

    values, scores = [], []
    for g in games:
        env = EnvConfig(family="atari", params=params, tasks=[{"game": g, "threshold": 0.0}])
        fam = make_family(env)
        task = fam.tasks[0]
        pol = make_policy(PolicyConfig(kind="impala_ac", hidden_sizes=[512],
                                       task_conditioned=False), fam).to(device)
        pol.load_state_dict(torch.load(f"{expert_dir}/{g}/best_model.pt", map_location=device))
        pol.eval()
        v, s, _, n = evaluate_value_and_score(
            pol, task, 100, n_envs, device, seed=eval_seed, greedy=True,
            noop_enumerate=True, max_ep_steps=cap)
        values.append(float(v)); scores.append(float(s))
        print(f"[expert-values] {g:14} V_L={v:.4f}  score={s:.1f}  (n={n})", flush=True)
        # incremental save so progress is visible
        Path(args.out).write_text(json.dumps(
            {"games": games[:len(values)], "gamma": gamma,
             "expert_values": values, "expert_scores": scores}, indent=2))
    print(f"[expert-values] -> {args.out}")


if __name__ == "__main__":
    main()
