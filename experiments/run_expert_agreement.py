"""Score a finished min-max run's global checkpoints with the windowed value-
agreement metric, referenced to the run's OWN LOCAL models.

Part A = experts NOT stored, so the per-task reference/"specialist" is the LOCAL
model (it IS the constraint target V_k^L) -- NOT an external single-task expert.
Task 1 has no local phase, so its reference is global_after_task1. Both references
are the multi-head policy; task i uses head i.

Emits into the run dir:
  - expert_agreement.json : local reference greedy-100 scores + per-game horizons
    + lower-triangular relative_gap matrix (global head i vs LOCAL i, on task i).
  - figure_data.json      : forgetting matrix + reference (local) scores + random
    baselines, for experiments/make_figures.py.

  python -m experiments.run_expert_agreement --config configs/atari5_v2_full.yaml \
      --run-dir results/atari5_v2_full_seed0
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from crl.config import load_config
from crl.envs import make_family
from crl.envs.atari import RANDOM_SCORES
from crl.policies import make_policy
from crl.ppo.evaluate import evaluate_value_and_score
from crl.ppo.expert_eval import (build_expert_state_set, evaluate_expert_agreement,
                                 load_state_set, save_state_set)


def _load_multihead(cfg, fam, ckpt, device):
    p = make_policy(cfg.policy, fam).to(device)
    p.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
    for q in p.parameters():
        q.requires_grad_(False)
    p.eval()
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--states-per-task", type=int, default=5000)
    ap.add_argument("--n-source-starts", type=int, default=20)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    games = [t["game"] for t in cfg.env.tasks]
    K = len(games)
    fam = make_family(cfg.env)
    sdir = os.path.join(args.run_dir, "ref_states")
    os.makedirs(sdir, exist_ok=True)

    # reference model for task i (0-based): the LOCAL specialist; task 1 -> global.
    def ref_ckpt(i: int) -> str:
        name = "global_after_task1.pt" if i == 0 else f"local_after_task{i + 1}.pt"
        return os.path.join(args.run_dir, name)

    # (a) per task: LOCAL reference state set (head i) + reference greedy-100 score
    sets, ref_scores, horizons = [], [], []
    for i, game in enumerate(games):
        ck = ref_ckpt(i)
        if not os.path.exists(ck):
            print(f"[{game}] MISSING reference {ck} -- aborting"); return
        ref = _load_multihead(cfg, fam, ck, device)
        spath = os.path.join(sdir, f"{game}.pkl")
        ss = (load_state_set(spath) if os.path.exists(spath)
              else build_expert_state_set(ref, fam.tasks[i], device, expert_task_id=i,
                                          n_source_starts=args.n_source_starts,
                                          states_per_task=args.states_per_task))
        if not os.path.exists(spath):
            save_state_set(ss, spath)
        _, rs, _, _ = evaluate_value_and_score(
            ref, fam.tasks[i], cfg.ppo.eval_episodes, cfg.ppo.n_envs, device,
            seed=cfg.ppo.eval_seed, greedy=True, max_ep_steps=cfg.ppo.eval_max_ep_steps)
        sets.append(ss); ref_scores.append(round(float(rs), 2)); horizons.append(ss.horizon)
        print(f"[{game}] ref={os.path.basename(ck)} local greedy-100={rs:.1f} "
              f"H={ss.horizon} states={len(ss.states)}", flush=True)

    # (b) each global checkpoint k: windowed agreement (head i vs LOCAL i), i<=k
    agreement = [[None] * K for _ in range(K)]
    for k in range(1, K + 1):
        gk = os.path.join(args.run_dir, f"global_after_task{k}.pt")
        if not os.path.exists(gk):
            print(f"[T{k}] MISSING {gk}"); continue
        g = _load_multihead(cfg, fam, gk, device)
        for i in range(k):
            res = evaluate_expert_agreement(g, fam.tasks[i], sets[i], device,
                                            task_id=i, n_envs=cfg.ppo.n_envs)
            agreement[k - 1][i] = res["relative_gap"]
            print(f"[T{k}] {games[i]:12s} relative_gap(vs local)={res['relative_gap']:.3f}",
                  flush=True)

    thr = [t.get("threshold") for t in cfg.env.tasks]
    em = os.path.join(args.run_dir, "eval_matrix.json")
    fm = json.load(open(em)) if os.path.exists(em) else None
    with open(os.path.join(args.run_dir, "expert_agreement.json"), "w") as f:
        json.dump({"games": games,
                   "reference": "LOCAL model (Part A specialist); task1=global_after_task1",
                   "reference_greedy_score": ref_scores, "horizons": horizons,
                   "agreement_matrix_relative_gap": agreement,
                   "note": "relative_gap in [0,1], lower=better (0 = global matches/beats "
                           "the LOCAL). Global head i vs local model i on task i. Report "
                           "with the greedy-100 forgetting matrix."}, f, indent=2)
    with open(os.path.join(args.run_dir, "figure_data.json"), "w") as f:
        json.dump({"games": games, "thresholds": thr, "forgetting_matrix": fm,
                   "reference_scores": ref_scores, "reference_label": "local model (pi_L)",
                   "random_scores": [RANDOM_SCORES.get(g, 0.0) for g in games],
                   "note": "reference = per-task LOCAL model (Part A specialist), NOT a stored "
                           "expert. forgetting_matrix rows=after task k, cols=greedy-100 score "
                           "on task i (lower-triangular)."}, f, indent=2)
    print(f"\nwrote expert_agreement.json + figure_data.json in {args.run_dir}", flush=True)


if __name__ == "__main__":
    main()
