"""Generate the shared init every expert (and the global) forks from.

Deterministic (seed 12345) so it is exactly reproducible. Single-head Impala-CNN
(task_conditioned=False) — identical architecture across all experts, so this
one state_dict loads into any of them.

    python -m experiments.make_shared_init            # -> experts/_shared_init.pt
"""
from __future__ import annotations

from pathlib import Path

import torch

from crl.config import EnvConfig, PolicyConfig
from crl.envs import make_family
from crl.policies import make_policy

SEED = 12345
ENV_PARAMS = dict(gamma=0.99, max_steps=4000, frame_skip=4, frame_stack=4,
                  noop_max=30, terminal_on_life_loss=False,
                  repeat_action_probability=0.0, clip_rewards=True)


def main(out="experts/_shared_init.pt"):
    torch.manual_seed(SEED)
    env = EnvConfig(family="atari", params=dict(ENV_PARAMS),
                    tasks=[{"game": "Pong", "threshold": 0.0}])  # arch is game-agnostic
    fam = make_family(env)
    pol = make_policy(PolicyConfig(kind="impala_ac", hidden_sizes=[512],
                                   task_conditioned=False), fam)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(pol.state_dict(), out)
    print(f"saved {out} | {sum(p.numel() for p in pol.parameters())} params (seed {SEED})")


if __name__ == "__main__":
    main()
