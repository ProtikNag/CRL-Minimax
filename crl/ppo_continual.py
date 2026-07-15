"""PPO continual-learning orchestrator (min-max local/global alternation).

This is the PPO analogue of :class:`crl.trainer.AlternationTrainer`. The
continual-learning framework is identical to the document -- only the optimizer
is PPO instead of REINFORCE:

* task 1        -- standard PPO trains the global model (no past tasks).
* task k >= 2   -- for each cycle: a *local* phase (standard PPO on the current
                   task from theta^0 = phi) produces the frozen reference V_k^L;
                   a *global* phase (PPO + the actor-only mu constraint) pushes
                   the global model's past-task value up while keeping
                   V_k^G >= V_k^L via the one-sided squared shortfall.

Uniform weights omega_i = 1/k, a single persistent mu controller (reset per
global phase, warm-started by config), replay-free fresh rollouts. The eval
matrix and probes report the raw game score (task performance), while the
constraint uses the discounted return (V), exactly as in the REINFORCE trainer.

``method: finetune`` runs naive sequential standard PPO on one shared net (no
local phase, no constraint) -- the catastrophic-forgetting baseline.
"""

from __future__ import annotations

import torch

from crl.config import Config
from crl.duals import make_dual
from crl.envs.base import TaskFamily
from crl.logging_utils import RunLogger
from crl.policies.base import Policy, clone_policy
from crl.ppo.evaluate import evaluate_value_and_score
from crl.ppo.trainer import GlobalTrainer, LocalTrainer


class PPOAlternationTrainer:
    """Runs the full task sequence with PPO; records the eval matrix + probes."""

    def __init__(
        self,
        config: Config,
        family: TaskFamily,
        global_policy: Policy,
        logger: RunLogger,
    ) -> None:
        self.cfg = config.trainer
        self.ppo = config.ppo
        self.dual_cfg = config.duals
        self.family = family
        self.global_policy = global_policy
        self.logger = logger
        self.device = next(global_policy.parameters()).device
        self.seed = config.experiment.seed
        self.method = config.ppo.method
        log_every = config.experiment.log_every

        self.local_trainer = LocalTrainer(self.ppo, self.device, logger, log_every)
        self.global_trainer = GlobalTrainer(self.ppo, self.device, logger, log_every)
        self.mu_ctrl = make_dual(self.dual_cfg)

        self.eval_matrix: list[list[float]] = []
        self.cumulative_step = 0

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #

    def _eps(self) -> float:
        e = self.cfg.eps
        return float(e) if isinstance(e, (int, float)) else float(e[0])

    def _eval(self, policy: Policy, task, num_episodes: int) -> tuple[float, float]:
        """Return ``(discounted_value, raw_score)`` for ``policy`` on ``task``."""
        value, score, _ = evaluate_value_and_score(
            policy, task, num_episodes, self.ppo.n_envs, self.device
        )
        return value, score

    def _evaluate_row(self, k: int) -> list[float]:
        """Raw-score row of the forgetting matrix after finishing task k."""
        last = len(self.family) if self.cfg.eval_all_tasks else k
        row = []
        for i in range(last):
            _, score = self._eval(
                self.global_policy, self.family.tasks[i], self.ppo.eval_episodes
            )
            row.append(score)
        return row

    def _probe(self, phase_type: str, current_task: int) -> None:
        """Record the global policy's score on every task vs cumulative iters."""
        self.cumulative_step += 1
        every = self.ppo.eval_every
        if not every or self.cumulative_step % every != 0:
            return
        values = []
        for i in range(len(self.family)):
            _, score = self._eval(
                self.global_policy, self.family.tasks[i], self.ppo.eval_episodes
            )
            values.append(score)
        self.logger.log(
            {
                "phase": "probe",
                "cumulative_step": self.cumulative_step,
                "current_task": current_task,
                "phase_type": phase_type,
                "values": values,
            }
        )

    # ------------------------------------------------------------------ #
    # main loop
    # ------------------------------------------------------------------ #

    def _train_first_task(self) -> None:
        """Standard PPO on task 1 (the global model; no past tasks, no constraint)."""
        self.local_trainer.train(
            self.global_policy,
            self.family.tasks[0],
            num_iters=self.ppo.task1_iters,
            seed=self.seed + 1000,
            current_task=1,
            phase_type="task1",
            probe=self._probe,
        )

    def _finetune_task(self, k: int) -> None:
        """Naive baseline: keep fine-tuning the one shared net on task k."""
        self.local_trainer.train(
            self.global_policy,
            self.family.tasks[k - 1],
            num_iters=self.ppo.local_iters + self.ppo.global_iters,
            seed=self.seed + 1000 * k,
            current_task=k,
            phase_type="finetune",
            probe=self._probe,
        )

    def _constrained_task(self, k: int) -> None:
        task_k = self.family.tasks[k - 1]
        past_tasks = [self.family.tasks[i] for i in range(k - 1)]
        for cycle in range(self.cfg.cycles_per_task):
            # ---- local phase: theta^0 = phi, standard PPO on task k ---------
            local_policy = clone_policy(self.global_policy, trainable=True)
            self.local_trainer.train(
                local_policy, task_k,
                num_iters=self.ppo.local_iters,
                seed=self.seed + 1000 * k + 13 * cycle,
                current_task=k, phase_type="local", probe=self._probe,
            )
            frozen_local = clone_policy(local_policy, trainable=False)
            ref_current, _ = self._eval(
                frozen_local, task_k, self.ppo.constraint_episodes
            )

            # ---- global phase: PPO + actor-only mu constraint ---------------
            self.mu_ctrl.reset()
            omega = [1.0 / k] * (k - 1)  # uniform weights omega_i = 1/k
            self.global_trainer.train(
                self.global_policy, task_k, past_tasks,
                ref_current=ref_current, mu_ctrl=self.mu_ctrl, omega=omega,
                eps=self._eps(),
                num_iters=self.ppo.global_iters,
                seed=self.seed + 1000 * k + 13 * cycle,
                current_task=k, probe=self._probe,
            )
            self.logger.log(
                {"phase": "gaps", "task": k, "cycle": cycle,
                 "V_k_ref_local": ref_current}
            )

    def run(self) -> list[list[float]]:
        self._train_first_task()
        self.eval_matrix.append(self._evaluate_row(1))
        self.logger.log({"phase": "eval", "task": 1, "values": self.eval_matrix[-1]})

        for k in range(2, len(self.family) + 1):
            if self.method == "finetune":
                self._finetune_task(k)
            elif self.method == "constrained":
                self._constrained_task(k)
            else:
                raise KeyError(
                    f"Unknown ppo.method '{self.method}'; available: "
                    "constrained, finetune"
                )
            self.eval_matrix.append(self._evaluate_row(k))
            self.logger.log(
                {"phase": "eval", "task": k, "values": self.eval_matrix[-1]}
            )

        self.logger.save_json("eval_matrix.json", self.eval_matrix)
        torch.save(
            self.global_policy.state_dict(), self.logger.run_dir / "final_policy.pt"
        )
        return self.eval_matrix
