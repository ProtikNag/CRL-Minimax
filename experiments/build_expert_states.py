"""Precompute (and smoke-test) the windowed expert-agreement state sets.

Protocol: docs/expert_value_agreement_eval.md. For each game we roll its frozen
single-task expert, snapshot consequential near-/mid-/early states, and pickle the
reusable set to ``--out/<Game>.pkl``. These sets are then scored against any
consolidated global checkpoint by ``crl.ppo.expert_eval.evaluate_expert_agreement``.

Full precompute (5000 states x 10 games) should be run via sbatch on a COMPUTE
node, not the login node. ``--smoke`` runs a tiny single-game self-consistency
check (expert-vs-its-own-set shortfall must be ~0) suitable for a quick local run.

Examples:
  python -m experiments.build_expert_states --smoke --game Pong --config configs/atari4_minmax.yaml
  python -m experiments.build_expert_states --config configs/atari5_ppo_v5.yaml --out expert_states
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from crl.config import EnvConfig, PolicyConfig, load_config
from crl.envs import make_family
from crl.ppo.expert_eval import (build_expert_state_set, evaluate_expert_agreement,
                                 save_state_set)


def _load_expert(game: str, env_params: dict, expert_dir: str, device):
    """Load a frozen single-task Impala expert + a single-game task to roll it in."""
    env = EnvConfig(family="atari", params=dict(env_params),
                    tasks=[{"game": game, "threshold": 0.0}])
    fam = make_family(env)
    from crl.policies import make_policy
    pol = make_policy(PolicyConfig(kind="impala_ac", hidden_sizes=[512],
                                   task_conditioned=False), fam).to(device)
    sd = torch.load(f"{expert_dir}/{game}/best_model.pt", map_location=device)
    pol.load_state_dict(sd)
    for p in pol.parameters():
        p.requires_grad_(False)
    pol.eval()
    return pol, fam.tasks[0]


def _check_selfmanaged_stack(task, device) -> bool:
    """The self-managed 4-frame stack must match FrameStackObservation exactly."""
    from collections import deque
    full = task.make_env(clip_rewards=False)          # has FrameStackObservation
    snap = task.make_snapshot_env(clip_rewards=False)  # single frame; we stack
    fo, _ = full.reset(seed=7)
    so, _ = snap.reset(seed=7)
    fs = task.frame_stack
    stack = deque([np.asarray(so, np.uint8)] * fs, maxlen=fs)
    ok = np.array_equal(np.asarray(fo), np.stack(stack, 0))
    for a in [2, 3, 0, 2, 4, 3]:
        fo, *_ = full.step(a)
        so, *_ = snap.step(a)
        stack.append(np.asarray(so, np.uint8))
        ok = ok and np.array_equal(np.asarray(fo), np.stack(stack, 0))
    full.close(); snap.close()
    return bool(ok)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="config for env.params + tasks")
    ap.add_argument("--game", default=None, help="single game (default: all env.tasks)")
    ap.add_argument("--expert-dir", default="experts")
    ap.add_argument("--out", default="expert_states")
    ap.add_argument("--n-source-starts", type=int, default=20)
    ap.add_argument("--states-per-task", type=int, default=5000)
    ap.add_argument("--smoke", action="store_true", help="tiny single-game self-check")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_params = dict(cfg.env.params)

    if args.smoke:
        game = args.game or cfg.env.tasks[0]["game"]
        print(f"[smoke] game={game} device={device}")
        assert _check_selfmanaged_stack(
            _load_expert(game, env_params, args.expert_dir, device)[1], device), \
            "self-managed stack != FrameStackObservation"
        print("[smoke] self-managed stack matches FrameStackObservation: OK")
        expert, task = _load_expert(game, env_params, args.expert_dir, device)
        ss = build_expert_state_set(expert, task, device, n_source_starts=2,
                                    states_per_task=150)
        print(f"[smoke] H_i={ss.horizon}  kept={len(ss.states)}  "
              f"categories={ss.meta['category_counts']}  "
              f"consequential_frac={ {k: round(v,3) for k,v in ss.meta['consequential_frac_by_H'].items()} }")
        # Expert vs its OWN set -> shortfall must be ~0 (same greedy policy from same state).
        res = evaluate_expert_agreement(expert, task, ss, device, task_id=0, n_envs=8)
        print(f"[smoke] expert-vs-own-set: relative_gap={res['relative_gap']:.4g} "
              f"agreement_gap={res['agreement_gap']:.4g} (both must be ~0)")
        assert res["relative_gap"] < 1e-6, "expert self-agreement should be ~0!"
        print("[smoke] PASS")
        return

    games = [args.game] if args.game else [t["game"] for t in cfg.env.tasks]
    os.makedirs(args.out, exist_ok=True)
    for tid, game in enumerate(games):
        expert, task = _load_expert(game, env_params, args.expert_dir, device)
        ss = build_expert_state_set(expert, task, device, expert_task_id=0,
                                    n_source_starts=args.n_source_starts,
                                    states_per_task=args.states_per_task)
        path = os.path.join(args.out, f"{game}.pkl")
        save_state_set(ss, path)
        print(f"[{game}] H_i={ss.horizon} kept={len(ss.states)} "
              f"cats={ss.meta['category_counts']} -> {path}")


if __name__ == "__main__":
    main()
