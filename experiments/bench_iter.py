"""Profile one global iteration: where does the ~12 s/iter go?"""
import time, torch, numpy as np
from crl.config import load_config
from crl.envs import make_family
from crl.policies import make_policy
from crl.ppo.collector import RolloutCollector
from crl.ppo.trainer import GlobalTrainer
from crl.ppo.evaluate import cache_eval_obs, critic_values
import json

cfg = load_config("configs/consolidate4_ratio_critic.yaml")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fam = make_family(cfg.env)
pol = make_policy(cfg.policy, fam).to(dev)
sd = torch.load("results/consolidate4_ratio_critic_seed0/global_after_task1.pt", map_location=dev)
pol.load_state_dict(sd); pol.eval()
gt = GlobalTrainer(cfg.ppo, dev, None, 1)
opt = gt._new_optimizer(pol)

boxing, pong = fam.tasks[1], fam.tasks[0]
cb = RolloutCollector(boxing, cfg.ppo.n_envs, cfg.ppo.n_steps, dev, 7)
pb = RolloutCollector(pong, cfg.ppo.n_envs, cfg.ppo.n_steps, dev, 101)

def t(fn, label, reps=5):
    fn()  # warmup
    s = time.time()
    for _ in range(reps): out = fn()
    dt = (time.time()-s)/reps
    print(f"  {label:40s} {dt*1000:8.1f} ms")
    return dt, out

print("=== per-iteration component timing (Boxing current, Pong past) ===")
d_cur, cur = t(lambda: cb.collect(pol, cfg.ppo.gae_lambda), "collect() current rollout")
d_past, past = t(lambda: pb.collect(pol, cfg.ppo.gae_lambda), "collect() past rollout")
d_opt, _ = t(lambda: gt.optimize_batches(pol, opt, [past, cur], [0.5, 1.0]), "optimize_batches (2 streams)")

interm = json.load(open(cfg.ppo.constraint_intermediate_path))
ec = cache_eval_obs(boxing, 1, interm["games"]["Boxing"], dev, 50, seed=cfg.ppo.eval_seed)
d_cv, _ = t(lambda: (critic_values(pol, ec["noop_obs"], 1), critic_values(pol, ec["interm_obs"], 1)), "critic eval (100 states)")

tot = d_cur + d_past + d_opt + d_cv
print(f"\n  SUM(collect_cur+collect_past+optimize+critic) = {tot:.2f}s")
print(f"  collect share = {(d_cur+d_past)/tot*100:.0f}%  optimize = {d_opt/tot*100:.0f}%  critic = {d_cv/tot*100:.0f}%")
cb.close(); pb.close()
print("DONE")
