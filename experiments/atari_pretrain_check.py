"""Per-game local-PPO sanity check.

Trains standard PPO (the LocalTrainer, i.e. the exact optimizer used by the
continual method's local phase) on ONE Atari game and reports the raw game score
over training. Use this to confirm each of the 5 games reaches a decent score and
to calibrate the per-phase iteration budgets before the full continual run.

    python -m experiments.atari_pretrain_check --game Pong --iters 600 --device auto

Writes a score curve to ``results/pretrain_<game>_seed<seed>/logs.jsonl`` and
prints the final mean score.
"""

from __future__ import annotations

import argparse

import torch

from crl.config import EnvConfig, PolicyConfig, PPOConfig
from crl.envs import make_family
from crl.logging_utils import RunLogger
from crl.policies import make_policy
from crl.ppo.evaluate import evaluate_value_and_score
from crl.ppo.trainer import LocalTrainer
from crl.seeding import set_seed
from experiments.run import resolve_device


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--game", required=True, help="Atari game (e.g. Pong, Breakout).")
    p.add_argument("--iters", type=int, default=600, help="PPO iterations.")
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--n-steps", type=int, default=128)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=50, help="Eval cadence (iters).")
    p.add_argument("--max-steps", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--results-dir", default="results")
    args = p.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    family = make_family(
        EnvConfig(
            family="atari",
            params={"gamma": 0.99, "max_steps": args.max_steps,
                    "repeat_action_probability": 0.0, "clip_rewards": True},
            tasks=[{"game": args.game}],
        )
    )
    policy = make_policy(PolicyConfig(kind="cnn_ac", hidden_sizes=[512]), family).to(device)
    task = family.tasks[0]

    run_name = f"pretrain_{args.game}_seed{args.seed}"
    logger = RunLogger(args.results_dir, run_name, {"game": args.game, "iters": args.iters})
    cfg = PPOConfig(n_envs=args.n_envs, n_steps=args.n_steps)
    trainer = LocalTrainer(cfg, device, logger, log_every=max(1, args.eval_every))

    print(f"[pretrain] game={args.game} device={device} iters={args.iters} "
          f"params={sum(p.numel() for p in policy.parameters())}")

    counter = {"it": 0}

    def probe(phase_type: str, current_task: int) -> None:
        counter["it"] += 1
        if args.eval_every and counter["it"] % args.eval_every == 0:
            _, score, n = evaluate_value_and_score(
                policy, task, args.eval_episodes, args.n_envs, device
            )
            logger.log({"phase": "score", "iter": counter["it"], "score": score,
                        "episodes": n})
            print(f"[pretrain] {args.game} it={counter['it']:4d} score={score:.2f} "
                  f"(n={n})")

    trainer.train(policy, task, num_iters=args.iters, seed=args.seed + 1,
                  current_task=1, phase_type="pretrain", probe=probe)

    value, score, n = evaluate_value_and_score(
        policy, task, max(args.eval_episodes, 20), args.n_envs, device
    )
    logger.log({"phase": "final", "score": score, "value": value, "episodes": n})
    logger.close()
    torch.save(policy.state_dict(), logger.run_dir / "final_policy.pt")
    print(f"[pretrain] {args.game} FINAL score={score:.2f} value={value:.3f} (n={n})")


if __name__ == "__main__":
    main()
