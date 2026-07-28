"""Phase B: mid/late-game retention via ACTUAL rollout returns (no critic estimate).

For each game, from RANDOM intermediate points along the expert's own greedy
trajectory, measure the GLOBAL policy's *true* discounted return-to-go (a real
rollout) and compare it to the expert's return-to-go. Answers, with actual
returns, whether the global matches the expert not just at the episode start but
mid/late -- the question Phase A raised with a (cheap) critic proxy.

Reaching an intermediate state (exact, no emulator-state surgery): the env is
deterministic (repeat_action_probability=0, fixed reset seed), so we REPLAY the
expert's recorded actions 0..t-1 to land exactly on state s_t with the frame-stack
buffers correctly primed -- then roll the GLOBAL greedily from s_t to the end.

Applies to ALL games identically (no long/short special-casing). Writes results
incrementally (per game) so progress is checkable mid-run.

    python -m experiments.phase_b_rollout --run results/consolidate10_seed0 \
        --games Pong Breakout Qbert --k 45
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
def _expert_traj(expert, task, device, gamma, seed, cap):
    """One greedy expert rollout. Returns actions, discounted return-to-go, T."""
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
    return np.array(A, dtype=np.int64), rtg, T


@torch.no_grad()
def _rollout_from(policy, task_id, task, device, gamma, seed, replay_actions, cap):
    """Reset, REPLAY replay_actions to reach s_t (exact), then roll `policy`
    greedily to episode end. Returns its discounted return-to-go from s_t.
    If policy is None, replay the given actions verbatim (self-check)."""
    env = task.make_env(clip_rewards=True)
    obs, _ = env.reset(seed=seed)
    for a in replay_actions:                       # deterministic replay to s_t
        obs, _, term, trunc, _ = env.step(int(a))
        if term or trunc:
            env.close()
            return None                            # trajectory ended during replay
    disc = 0.0; g = 1.0
    for _ in range(cap):
        ot = torch.as_tensor(np.asarray(obs), device=device).unsqueeze(0)
        a = int(policy.dist(ot, task_id).logits.argmax(-1).item())
        obs, r, term, trunc, _ = env.step(a)
        disc += g * float(r); g *= gamma
        if term or trunc:
            break
    env.close()
    return disc


def _load_multihead(cfg, path, device):
    fam = make_family(EnvConfig(family="atari", params=dict(cfg["env"]["params"]),
                                tasks=cfg["env"]["tasks"]))
    pol = make_policy(PolicyConfig(kind="impala_ac_multihead", hidden_sizes=[512],
                                   task_conditioned=False), fam).to(device)
    pol.load_state_dict(torch.load(path, map_location=device))
    pol.eval()
    return pol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--games", nargs="+", required=True)
    ap.add_argument("--k", type=int, default=45, help="intermediate starts per game")
    ap.add_argument("--n_traj", type=int, default=3, help="expert trajectories to pool")
    ap.add_argument("--cap", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20000)
    ap.add_argument("--out", default="diagnostics/phase_b")
    args = ap.parse_args()
    run = Path(args.run)
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    all_games = [t["game"] for t in cfg["env"]["tasks"]]
    gamma = float(cfg["env"]["params"]["gamma"])
    expert_dir = cfg["ppo"]["expert_dir"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    final_global = _load_multihead(cfg, run / "final_policy.pt", device)
    outd = Path(args.out) / run.name
    outd.mkdir(parents=True, exist_ok=True)
    report = {"gamma": gamma, "k": args.k, "n_traj": args.n_traj, "games": {}}

    for game in args.games:
        tid = all_games.index(game)
        env = EnvConfig(family="atari", params=dict(cfg["env"]["params"]),
                        tasks=[{"game": game, "threshold": 0.0}])
        fam = make_family(env)
        task = fam.tasks[0]
        expert = make_policy(PolicyConfig(kind="impala_ac", hidden_sizes=[512],
                                          task_conditioned=False), fam).to(device)
        expert.load_state_dict(torch.load(f"{expert_dir}/{game}/best_model.pt",
                                          map_location=device))
        expert.eval()

        # expert trajectories (pooled)
        trajs = []
        for ti in range(args.n_traj):
            s = args.seed + 137 * ti
            acts, rtg, T = _expert_traj(expert, task, device, gamma, s, args.cap)
            if T > 3:
                trajs.append({"seed": s, "acts": acts, "rtg": rtg, "T": T})
        # candidate intermediate starts: rtg>0, exclude t=0 and the last step
        cands = [(j, t) for j, tr in enumerate(trajs)
                 for t in range(1, tr["T"] - 1) if tr["rtg"][t] > 0]
        pick = rng.choice(len(cands), size=min(args.k, len(cands)), replace=False)
        starts = sorted(cands[i] for i in pick)

        # self-check on the first 2 starts: replaying the EXPERT from s_t must
        # reproduce the expert's own return-to-go (proves the replay is exact).
        checks = []
        for j, t in starts[:2]:
            tr = trajs[j]
            r = _rollout_from(expert, 0, task, device, gamma, tr["seed"],
                              tr["acts"][:t], args.cap - t)
            checks.append({"t": int(t), "expert_rtg": float(tr["rtg"][t]),
                           "expert_replayed": (None if r is None else float(r))})

        rows = []
        for j, t in starts:
            tr = trajs[j]
            g_rtg = _rollout_from(final_global, tid, task, device, gamma,
                                  tr["seed"], tr["acts"][:t], args.cap - t)
            if g_rtg is None:
                continue
            e_rtg = float(tr["rtg"][t])
            rows.append({"traj": j, "t": int(t), "frac": float(t / tr["T"]),
                         "expert_rtg": e_rtg, "global_rtg": float(g_rtg),
                         "ratio": float(g_rtg / e_rtg)})

        per_bucket = {}
        for b, bn in enumerate(BUCKETS):
            rr = [x for x in rows if _bucket(x["frac"]) == b]
            if rr:
                ratios = np.clip([x["ratio"] for x in rr], -0.5, 2.0)
                per_bucket[bn] = {
                    "n": len(rr),
                    "median_ratio": float(np.median(ratios)),
                    "mean_global_rtg": float(np.mean([x["global_rtg"] for x in rr])),
                    "mean_expert_rtg": float(np.mean([x["expert_rtg"] for x in rr])),
                }
            else:
                per_bucket[bn] = {"n": 0, "median_ratio": float("nan"),
                                  "mean_global_rtg": float("nan"),
                                  "mean_expert_rtg": float("nan")}
        report["games"][game] = {"tid": tid, "n_starts": len(rows),
                                 "mean_ep_len": float(np.mean([tr["T"] for tr in trajs])),
                                 "self_check": checks, "buckets": per_bucket, "rows": rows}
        # incremental write + print (progress-checkable mid-run)
        (outd / "phase_b_rollout.json").write_text(json.dumps(report, indent=2))
        chk = checks[0] if checks else {}
        print(f"\n=== {game} (head {tid}) — {len(rows)} starts, ep_len "
              f"{report['games'][game]['mean_ep_len']:.0f} ===")
        if chk:
            print(f"  self-check t={chk.get('t')}: expert_rtg={chk.get('expert_rtg'):.3f} "
                  f"replayed={chk.get('expert_replayed')}  (should match)")
        print(f"  {'bucket':<8}{'n':>5}{'median G/E':>13}{'mean G':>10}{'mean E':>10}")
        for bn in BUCKETS:
            pb = per_bucket[bn]
            print(f"  {bn:<8}{pb['n']:>5}{pb['median_ratio']:>13.3f}"
                  f"{pb['mean_global_rtg']:>10.3f}{pb['mean_expert_rtg']:>10.3f}")

    # figure: median true-return ratio vs bucket, per game
    ng = len(args.games)
    fig, ax = plt.subplots(figsize=(1.6 + 2.0 * ng, 4))
    x = np.arange(3)
    for game in args.games:
        pb = report["games"][game]["buckets"]
        ax.plot(x, [pb[b]["median_ratio"] for b in BUCKETS], "-o", label=game)
    ax.axhline(1.0, ls="--", c="0.5", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(BUCKETS)
    ax.set_ylim(0, 1.4); ax.grid(alpha=0.3)
    ax.set_ylabel("median  global true return / expert return")
    ax.set_xlabel("episode phase of the intermediate start")
    ax.set_title("Phase B — actual-rollout retention from random intermediate states")
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "svg"):
        (outd / ext).mkdir(exist_ok=True)
        fig.savefig(outd / ext / ("phase_b_rollout." + ext), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[phase B] report -> {outd}/phase_b_rollout.json ; figure -> {outd}/{{png,svg}}/")


if __name__ == "__main__":
    main()
