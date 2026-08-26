"""Eval-only diagnostic: is a task's greedy score a bimodal launch pathology?

Loads a saved global checkpoint and evaluates ONE task under greedy (argmax) vs
stochastic actions, reporting the full per-episode score distribution (not just
the mean). If greedy is ~0 with a few high outliers while stochastic is high, the
"0" in the forgetting matrix is a greedy-determinism EVAL artifact (e.g. Breakout
never firing the ball from most no-op starts), NOT catastrophic forgetting.

Usage:
  python scripts/diag_eval_checkpoint.py CONFIG.yaml CHECKPOINT.pt TASK_INDEX [n_episodes]
"""
from __future__ import annotations
import sys
import torch

from crl.config import load_config
from crl.envs import make_family
from crl.policies import make_policy


def rollout_scores(policy, task, n_episodes, n_envs, device, seed, greedy):
    tid = task.spec.task_id
    venv = task.make_vector_env(n_envs, clip_rewards=False)
    scores = []
    try:
        obs, _ = venv.reset(seed=seed)
        obs = torch.as_tensor(obs, device=device)
        raw = torch.zeros(n_envs)
        cap = n_episodes * 40_000 // max(1, n_envs) + 20_000
        steps = 0
        while len(scores) < n_episodes and steps < cap:
            steps += 1
            with torch.no_grad():
                logits = policy.dist(obs, tid).logits
            act = (logits.argmax(-1) if greedy
                   else torch.distributions.Categorical(logits=logits).sample())
            obs_np, reward, term, trunc, _ = venv.step(act.to("cpu").numpy())
            raw += torch.as_tensor(reward, dtype=torch.float32)
            done = term | trunc
            for i in range(n_envs):
                if bool(done[i]):
                    scores.append(float(raw[i])); raw[i] = 0.0
            obs = torch.as_tensor(obs_np, device=device)
    finally:
        venv.close()
    return scores[:n_episodes]


def summarize(name, s):
    import statistics as st
    n = len(s)
    zeros = sum(1 for x in s if x <= 1.0)
    hi = sum(1 for x in s if x >= 50.0)
    mean = sum(s) / n if n else 0.0
    sd = st.pstdev(s) if n > 1 else 0.0
    print(f"  {name:11s} n={n:3d}  mean={mean:7.1f}  std={sd:6.1f}  "
          f"min={min(s):5.0f}  max={max(s):5.0f}  "
          f"|  <=1pt: {zeros:3d} ({100*zeros/n:4.0f}%)   >=50pt: {hi:3d} ({100*hi/n:4.0f}%)")


def main():
    cfg_path, ckpt, ti = sys.argv[1], sys.argv[2], int(sys.argv[3])
    n_ep = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    cfg = load_config(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    family = make_family(cfg.env)
    policy = make_policy(cfg.policy, family).to(device)
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    missing, unexpected = policy.load_state_dict(sd, strict=False)
    policy.eval()
    task = family.tasks[ti]
    seed = cfg.ppo.eval_seed
    print(f"checkpoint: {ckpt}")
    print(f"task[{ti}] = {task.spec.name}  (task_id={task.spec.task_id})  "
          f"device={device}  seed={seed}  episodes={n_ep}")
    print(f"load: {len(missing)} missing / {len(unexpected)} unexpected params")
    print(f"--- per-episode score distribution ({task.spec.name}) ---")
    summarize("greedy", rollout_scores(policy, task, n_ep, cfg.ppo.n_envs, device, seed, True))
    summarize("stochastic", rollout_scores(policy, task, n_ep, cfg.ppo.n_envs, device, seed, False))


if __name__ == "__main__":
    main()
