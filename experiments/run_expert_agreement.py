"""Score a finished min-max run's checkpoints with the windowed expert-agreement
metric (docs/expert_value_agreement_eval.md) + gather the data the figures need.

For each game it (a) builds/loads the reusable 5000-state expert set, (b) records
the expert's greedy-100 game score (for normalized / %-expert figures), then for
each ``global_after_task{k}.pt`` scores the windowed one-sided relative shortfall
on every seen task i<=k (head i-1). Writes ``expert_agreement.json`` into the run
dir: expert scores, per-game horizons, and the lower-triangular agreement matrix
(relative_gap per (k,i)) that mirrors the forgetting matrix.

  python -m experiments.run_expert_agreement --config configs/atari4_v6_full.yaml \
      --run-dir results/atari4_v6_full_seed0
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from crl.config import EnvConfig, PolicyConfig, load_config
from crl.envs import make_family
from crl.policies import make_policy
from crl.ppo.evaluate import evaluate_value_and_score
from crl.ppo.expert_eval import (build_expert_state_set, evaluate_expert_agreement,
                                 load_state_set, save_state_set)


def _load_expert(game, env_params, expert_dir, device):
    env = EnvConfig(family="atari", params=dict(env_params),
                    tasks=[{"game": game, "threshold": 0.0}])
    fam = make_family(env)
    pol = make_policy(PolicyConfig(kind="impala_ac", hidden_sizes=[512],
                                   task_conditioned=False), fam).to(device)
    pol.load_state_dict(torch.load(f"{expert_dir}/{game}/best_model.pt",
                                   map_location=device, weights_only=False))
    for p in pol.parameters():
        p.requires_grad_(False)
    pol.eval()
    return pol, fam.tasks[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--expert-dir", default="experts")
    ap.add_argument("--states-dir", default="expert_states")
    ap.add_argument("--states-per-task", type=int, default=5000)
    ap.add_argument("--n-source-starts", type=int, default=20)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_params = dict(cfg.env.params)
    games = [t["game"] for t in cfg.env.tasks]
    K = len(games)
    fam = make_family(cfg.env)                       # K-head multihead family
    os.makedirs(args.states_dir, exist_ok=True)

    # (a) per-game: reusable expert state set + expert greedy-100 score --------
    sets, expert_scores, horizons = [], [], []
    for idx, game in enumerate(games):
        expert, etask = _load_expert(game, env_params, args.expert_dir, device)
        path = os.path.join(args.states_dir, f"{game}.pkl")
        if os.path.exists(path):
            ss = load_state_set(path)
            print(f"[{game}] loaded state set ({len(ss.states)} states, H={ss.horizon})")
        else:
            ss = build_expert_state_set(expert, etask, device,
                                        n_source_starts=args.n_source_starts,
                                        states_per_task=args.states_per_task)
            save_state_set(ss, path)
            print(f"[{game}] built {len(ss.states)} states, H={ss.horizon} -> {path}")
        _, escore, _, _ = evaluate_value_and_score(
            expert, etask, cfg.ppo.eval_episodes, cfg.ppo.n_envs, device,
            seed=cfg.ppo.eval_seed, greedy=True, max_ep_steps=cfg.ppo.eval_max_ep_steps)
        sets.append(ss); expert_scores.append(float(escore)); horizons.append(ss.horizon)
        print(f"[{game}] expert greedy-100 score = {escore:.1f}")

    # (b) each checkpoint k: windowed agreement on seen tasks i<=k -------------
    agreement = [[None] * K for _ in range(K)]       # agreement[k-1][i] = relative_gap
    for k in range(1, K + 1):
        ckpt = os.path.join(args.run_dir, f"global_after_task{k}.pt")
        if not os.path.exists(ckpt):
            print(f"[ckpt T{k}] MISSING {ckpt} -- skipping"); continue
        gpol = make_policy(cfg.policy, fam).to(device)
        gpol.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
        gpol.eval()
        for i in range(k):                           # seen tasks (head i)
            res = evaluate_expert_agreement(gpol, fam.tasks[i], sets[i], device,
                                            task_id=i, n_envs=cfg.ppo.n_envs)
            agreement[k - 1][i] = res["relative_gap"]
            print(f"[ckpt T{k}] {games[i]:9s} relative_gap={res['relative_gap']:.3f} "
                  f"(H={res['horizon']}, {res['n_states']} states)")

    out = {
        "games": games, "expert_greedy_score": expert_scores, "horizons": horizons,
        "agreement_matrix_relative_gap": agreement,       # lower-triangular, None above diag
        "note": "relative_gap in [0,1], lower=better (0=matches/beats expert). "
                "Report alongside the greedy-100 forgetting matrix, never alone.",
    }
    path = os.path.join(args.run_dir, "expert_agreement.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {path}")

    # Also emit figure_data.json (forgetting matrix + expert/random scores) so
    # experiments/make_figures.py can regenerate the whole figure set unattended
    # (the v2 auto-visualization pipeline).
    from crl.envs.atari import RANDOM_SCORES
    em = os.path.join(args.run_dir, "eval_matrix.json")
    fig = {"games": games,
           "thresholds": [t.get("threshold") for t in cfg.env.tasks],
           "forgetting_matrix": (json.load(open(em)) if os.path.exists(em) else None),
           "expert_scores": expert_scores,
           "random_scores": [RANDOM_SCORES.get(g, 0.0) for g in games],
           "note": "forgetting_matrix rows=after task k, cols=greedy-100 score on task i "
                   "(lower-triangular)"}
    fpath = os.path.join(args.run_dir, "figure_data.json")
    with open(fpath, "w") as f:
        json.dump(fig, f, indent=2)
    print(f"wrote {fpath}")


if __name__ == "__main__":
    main()
