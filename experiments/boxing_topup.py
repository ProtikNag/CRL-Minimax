"""Boxing top-up probe (DISCARDABLE experiment).

Take the FINAL global model (after Qbert consolidation, final_policy.pt) and run
100 more global-consolidation iters with BOXING as the current task every iter:
its constraint floor (Boxing's local reference V_L) pushes the forgotten Boxing
value back up, while the other 4 games are retained as 'past'. Eval all 5 games
greedy-100 before and after. Original results dir is NOT touched; this writes to
results/atari5_rev_ours_seed0_boxtopup/ and can be deleted.

Run on a compute node (sbatch), never the login node.
"""
from __future__ import annotations
import os
import torch
from crl.config import load_config
from crl.envs import make_family
from crl.policies import make_policy
from crl.logging_utils import RunLogger
from crl.seeding import set_seed
from crl.ppo_continual import PPOAlternationTrainer

SRC = "results/atari5_rev_ours_seed0"
CFG = "configs/atari5_reversed.yaml"
N_ITERS = int(os.environ.get("BOX_ITERS", "100"))
BOX = 1          # 0-based task index for Boxing (task 2, 1-based)
BOX_LOCAL = f"{SRC}/local_after_task2.pt"
FINAL = f"{SRC}/final_policy.pt"

def main():
    cfg = load_config(CFG)
    set_seed(cfg.experiment.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    family = make_family(cfg.env)
    policy = make_policy(cfg.policy, family).to(dev)
    logger = RunLogger(cfg.experiment.results_dir,
                       f"{cfg.experiment.name}_seed{cfg.experiment.seed}_boxtopup{N_ITERS}",
                       cfg.to_dict())
    tr = PPOAlternationTrainer(cfg, family, policy, logger)

    # load the FINAL global (after Qbert)
    tr.global_policy.load_state_dict(torch.load(FINAL, map_location=dev))
    tr.global_policy.to(dev)
    games = [t.spec.name for t in family.tasks]

    def eval_all(tag):
        row = [tr._eval_report(tr.global_policy, t)[0] for t in family.tasks]
        print(f"[{tag}] " + " ".join(f"{g.split('-')[-1]}={s:.1f}" for g, s in zip(games, row)), flush=True)
        return row

    before = eval_all("BEFORE")

    # Boxing local reference (the constraint floor V_L)
    box_local = make_policy(cfg.policy, family).to(dev)
    box_local.load_state_dict(torch.load(BOX_LOCAL, map_location=dev))
    for p in box_local.parameters():
        p.requires_grad_(False)
    box_local.eval()
    box = family.tasks[BOX]
    past = [family.tasks[i] for i in range(len(family.tasks)) if i != BOX]  # other 4
    ref_current = tr._eval_value(box_local, box)
    print(f"[ref] Boxing local V_L = {ref_current:.3f}", flush=True)

    tr.mu_ctrl.reset()
    omega = [1.0 / len(family.tasks)] * len(past)   # 1/5 each, 4 past tasks
    summ = tr.global_trainer.train(
        tr.global_policy, box, past,
        ref_current=ref_current, mu_ctrl=tr.mu_ctrl, omega=omega,
        eps=tr._eps(), num_iters=N_ITERS, seed=cfg.experiment.seed + 99,
        current_task=BOX + 1, local_policy=box_local,
        retention_refs=None,   # run the full 100 iters, no early stop
    )
    print(f"[train] {summ}", flush=True)

    after = eval_all("AFTER")

    print("\n=== BOXING TOP-UP: before -> after (100 iters) ===")
    for g, b, a in zip(games, before, after):
        print(f"  {g:20s} {b:8.1f} -> {a:8.1f}   ({a-b:+.1f})")
    torch.save(tr.global_policy.state_dict(), logger.run_dir / "boxtopup_after.pt")
    logger.save_json("boxtopup_eval.json", {"games": games, "before": before, "after": after})
    print(f"\nsaved -> {logger.run_dir}/boxtopup_after.pt (discardable)")

if __name__ == "__main__":
    main()
