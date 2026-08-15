"""Micro-benchmark the per-iteration value-estimation cost (exploring option a).

Loads global_after_task1.pt and times each value estimator used in the global
loop, so we can see where 577s/iter goes and how cheap the alternatives are.
"""
import time, torch
from crl.config import load_config
from crl.envs import make_family
from crl.policies import make_policy
from crl.ppo.evaluate import (evaluate_greedy_noop_enumerated,
                              evaluate_intermediate_values_vec,
                              evaluate_value_and_score)
import json

cfg = load_config("configs/consolidate4_brute.yaml")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fam = make_family(cfg.env)
pol = make_policy(cfg.policy, fam).to(dev)
sd = torch.load("results/consolidate4_brute_seed0/global_after_task1.pt", map_location=dev)
pol.load_state_dict(sd); pol.eval()
interm = json.load(open(cfg.ppo.constraint_intermediate_path))
games = [t.spec.name.replace("atari-", "") for t in fam.tasks]
n_envs = cfg.ppo.n_envs
cap = cfg.ppo.eval_max_ep_steps

def timeit(fn, label, reps=1):
    t = time.time()
    out = None
    for _ in range(reps):
        out = fn()
    dt = (time.time() - t) / reps
    print(f"  {label:52s} {dt:7.2f}s")
    return dt, out

for ti in (1, 2):  # Boxing (current@task2) and Pong (past@task2)
    task = fam.tasks[ti]; g = games[ti]
    print(f"\n=== {g} (task_id={task.spec.task_id}) ===")
    d1, o1 = timeit(lambda: evaluate_greedy_noop_enumerated(
        pol, task, n_envs, dev, seed=cfg.ppo.eval_seed, max_ep_steps=cap),
        "noop_enumerate greedy (50 eps, full episode)")
    print(f"      -> V={o1[0]:.3f} score={o1[1]:.1f} n={o1[3]}")
    d2, o2 = timeit(lambda: evaluate_intermediate_values_vec(
        pol, task, task.spec.task_id, interm["games"][g], dev,
        n_envs=n_envs, max_ep_steps=cap),
        "intermediate_vec (50 states, full episode)")
    print(f"      -> mean V_g={sum(o2[0])/len(o2[0]):.3f} n={len(o2[0])}")
    d3, o3 = timeit(lambda: evaluate_value_and_score(
        pol, task, 16, n_envs, dev, seed=cfg.ppo.eval_seed, greedy=False),
        "CHEAP stochastic MC (16 eps, full episode)")
    print(f"      -> V={o3[0]:.3f} score={o3[1]:.1f} n={o3[3]}")
    d4, o4 = timeit(lambda: evaluate_value_and_score(
        pol, task, 16, n_envs, dev, seed=cfg.ppo.eval_seed, greedy=False,
        max_ep_steps=500),
        "CHEAP stochastic MC (16 eps, 500-step cap)")
    print(f"      -> V={o4[0]:.3f} score={o4[1]:.1f} n={o4[3]}")
    print(f"  per-task full cost (noop+interm) = {d1+d2:.1f}s ; cheap16 = {d3:.1f}s")

print("\nDONE")
