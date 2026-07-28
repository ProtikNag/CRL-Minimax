"""Precompute + cache mid/late-game intermediate states per game (for Run B).

For each game, re-simulate the expert's greedy trajectory and sample N random
intermediate positions (rtg>0, excluding the very start/end). We cache, per game,
the expert's action sequence + reset seed (so the trainer can reach state s_t by
EXACT deterministic re-simulation -- replay actions 0..t-1), the sampled step
indices t, and the expert's discounted return-to-go V_L(s_t) at each. The
constraint then compares the global's true rollout return from s_t against V_L(s_t).

    python -m experiments.compute_intermediate_states \
        --config configs/consolidate10_strict_g999_interm.yaml \
        --out results/intermediate_states_g999.json --n 30
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from crl.config import EnvConfig, PolicyConfig
from crl.envs import make_family
from crl.policies import make_policy


@torch.no_grad()
def _expert_traj(expert, task, device, gamma, seed, cap):
    env = task.make_env(clip_rewards=True)
    obs, _ = env.reset(seed=seed)
    A, R = [], []
    for _ in range(cap):
        ot = torch.as_tensor(np.asarray(obs), device=device).unsqueeze(0)
        a = int(expert.dist(ot, 0).logits.argmax(-1).item())
        obs, r, term, trunc, _ = env.step(a)
        A.append(a); R.append(float(r))
        if term or trunc:
            break
    env.close()
    T = len(R)
    rtg = np.zeros(T); acc = 0.0
    for t in range(T - 1, -1, -1):
        acc = R[t] + gamma * acc
        rtg[t] = acc
    return A, rtg, T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=30, help="intermediate states per game")
    ap.add_argument("--n_traj", type=int, default=2)
    ap.add_argument("--cap", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=30000)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    params = dict(cfg["env"]["params"])
    games = [t["game"] for t in cfg["env"]["tasks"]]
    gamma = float(params["gamma"])
    expert_dir = cfg["ppo"]["expert_dir"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    print(f"[interm] gamma={gamma} n/game={args.n} device={device}")

    out = {"gamma": gamma, "cap": args.cap, "games": {}}
    for g in games:
        env = EnvConfig(family="atari", params=params, tasks=[{"game": g, "threshold": 0.0}])
        fam = make_family(env)
        task = fam.tasks[0]
        pol = make_policy(PolicyConfig(kind="impala_ac", hidden_sizes=[512],
                                       task_conditioned=False), fam).to(device)
        pol.load_state_dict(torch.load(f"{expert_dir}/{g}/best_model.pt", map_location=device))
        pol.eval()
        trajs = []
        for ti in range(args.n_traj):
            s = args.seed + 911 * ti
            A, rtg, T = _expert_traj(pol, task, device, gamma, s, args.cap)
            if T > 5:
                trajs.append({"seed": s, "actions": A, "rtg": rtg.tolist(), "T": T})
        cands = [(j, t) for j, tr in enumerate(trajs)
                 for t in range(1, tr["T"] - 1) if tr["rtg"][t] > 0]
        pick = rng.choice(len(cands), size=min(args.n, len(cands)), replace=False)
        starts = [{"traj": int(j), "t": int(t), "frac": float(t / trajs[j]["T"]),
                   "expert_rtg": float(trajs[j]["rtg"][t])} for j, t in sorted(cands[i] for i in pick)]
        out["games"][g] = {
            "trajs": [{"seed": tr["seed"], "actions": tr["actions"]} for tr in trajs],
            "starts": starts,
        }
        Path(args.out).write_text(json.dumps(out))
        print(f"[interm] {g:14} {len(starts)} states from {len(trajs)} trajs "
              f"(ep_len~{int(np.mean([tr['T'] for tr in trajs]))})", flush=True)
    print(f"[interm] -> {args.out}")


if __name__ == "__main__":
    main()
