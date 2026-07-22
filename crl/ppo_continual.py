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

        # CLEAR baseline: single replay store + trainer (created lazily on use).
        self._clear_trainer = None
        self._clear_replay = None
        self._log_every = log_every

        self.eval_matrix: list[list[float]] = []
        self.cumulative_step = 0

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #

    def _eps(self) -> float:
        e = self.cfg.eps
        return float(e) if isinstance(e, (int, float)) else float(e[0])

    def _eval_report(self, policy: Policy, task) -> tuple[float, float]:
        """Reported (raw) game score for ``policy`` on ``task``, pooled over a
        mix of GREEDY and STOCHASTIC episodes (fixed seeds -> reproducible).

        Of ``eval_episodes`` total, ``eval_greedy_episodes`` use argmax actions
        and the rest sample; the reported score is the pooled mean over ALL
        episodes (a blend of best-case and on-policy behaviour, not best-of).
        Returns ``(pooled_mean_score, pooled_std)``."""
        total = self.ppo.eval_episodes
        n_greedy = min(self.ppo.eval_greedy_episodes, total) if self.ppo.eval_greedy else 0
        n_stoch = total - n_greedy

        groups = []  # (mean, std, n)
        if n_greedy > 0:
            _, m, s, n = evaluate_value_and_score(
                policy, task, n_greedy, self.ppo.n_envs, self.device,
                seed=self.ppo.eval_seed, greedy=True,
            )
            groups.append((m, s, n))
        if n_stoch > 0:
            _, m, s, n = evaluate_value_and_score(
                policy, task, n_stoch, self.ppo.n_envs, self.device,
                seed=self.ppo.eval_seed + 1, greedy=False,
            )
            groups.append((m, s, n))

        n_tot = sum(n for _, _, n in groups)
        if n_tot == 0:
            return 0.0, 0.0
        pooled_mean = sum(m * n for m, _, n in groups) / n_tot
        # Exact pooled population variance across the (possibly two) groups.
        pooled_var = sum(n * (s * s + (m - pooled_mean) ** 2)
                         for m, s, n in groups) / n_tot
        return pooled_mean, pooled_var ** 0.5

    def _eval_value(self, policy: Policy, task) -> float:
        """On-policy STOCHASTIC discounted value V^pi (the constraint reference)."""
        value, _, _, _ = evaluate_value_and_score(
            policy, task, self.ppo.constraint_episodes, self.ppo.n_envs,
            self.device, greedy=False,
        )
        return value

    def _evaluate_row(self, k: int) -> tuple[list[float], list[float]]:
        """Raw-score row (+ per-game std) of the forgetting matrix after task k."""
        last = len(self.family) if self.cfg.eval_all_tasks else k
        row, stds = [], []
        for i in range(last):
            score, std = self._eval_report(self.global_policy, self.family.tasks[i])
            row.append(score)
            stds.append(std)
        return row, stds

    def _probe(self, phase_type: str, current_task: int) -> None:
        """Record the global policy's score on every task vs cumulative iters."""
        self.cumulative_step += 1
        every = self.ppo.eval_every
        if not every or self.cumulative_step % every != 0:
            return
        values = [self._eval_report(self.global_policy, self.family.tasks[i])[0]
                  for i in range(len(self.family))]
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

    def _clear_task(self, k: int) -> None:
        """CLEAR baseline: PPO on task k + replay/behavioral/value cloning on the
        past, then snapshot task k's behavior into the replay store."""
        task_k = self.family.tasks[k - 1]
        self._clear_trainer.train(
            self.global_policy, task_k, self._clear_replay,
            num_iters=self.ppo.local_iters + self.ppo.global_iters,
            seed=self.seed + 1000 * k, current_task=k, probe=self._probe,
        )
        self._clear_trainer.snapshot(self.global_policy, task_k, self._clear_replay)

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
            ref_current = self._eval_value(frozen_local, task_k)

            # ---- global phase: PPO + actor-only mu constraint ---------------
            self.mu_ctrl.reset()
            omega = [1.0 / k] * (k - 1)  # uniform weights omega_i = 1/k
            if self.ppo.global_probe_head_only:
                # EXPERIMENT: consolidate on the local's (current-task) trunk,
                # frozen, moving only the per-task heads.
                self.global_policy.load_state_dict(local_policy.state_dict())
                for name, p in self.global_policy.named_parameters():
                    p.requires_grad_(not name.startswith("trunk."))
            self.global_trainer.train(
                self.global_policy, task_k, past_tasks,
                ref_current=ref_current, mu_ctrl=self.mu_ctrl, omega=omega,
                eps=self._eps(),
                num_iters=self.ppo.global_iters,
                seed=self.seed + 1000 * k + 13 * cycle,
                current_task=k, probe=self._probe,
            )
            if self.ppo.global_probe_head_only:  # restore full trainability
                for p in self.global_policy.parameters():
                    p.requires_grad_(True)
            self.logger.log(
                {"phase": "gaps", "task": k, "cycle": cycle,
                 "V_k_ref_local": ref_current}
            )

    def run(self) -> list[list[float]]:
        if self.method == "clear":
            from crl.ppo.clear import ClearTrainer, ReplayStore
            self._clear_trainer = ClearTrainer(self.ppo, self.device, self.logger,
                                               self._log_every)
            self._clear_replay = ReplayStore()

        self._train_first_task()
        if self.method == "clear":  # store task-1 behavior as a cloning target
            self._clear_trainer.snapshot(self.global_policy, self.family.tasks[0],
                                         self._clear_replay)
        row, stds = self._evaluate_row(1)
        self.eval_matrix.append(row)
        self.logger.log({"phase": "eval", "task": 1, "values": row, "stds": stds})

        for k in range(2, len(self.family) + 1):
            if self.method == "finetune":
                self._finetune_task(k)
            elif self.method == "clear":
                self._clear_task(k)
            elif self.method == "constrained":
                self._constrained_task(k)
            else:
                raise KeyError(
                    f"Unknown ppo.method '{self.method}'; available: "
                    "constrained, finetune, clear"
                )
            row, stds = self._evaluate_row(k)
            self.eval_matrix.append(row)
            self.logger.log({"phase": "eval", "task": k, "values": row, "stds": stds})

        self.logger.save_json("eval_matrix.json", self.eval_matrix)
        torch.save(
            self.global_policy.state_dict(), self.logger.run_dir / "final_policy.pt"
        )
        return self.eval_matrix
