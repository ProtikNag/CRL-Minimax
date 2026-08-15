"""Stored-expert consolidation trainer -- Formulation B (NO min-max).

This is the second, separate formulation from the objective document's section
*"When we have the expert models stored"* (eqs 34-46). It is deliberately kept
apart from the min-max :class:`~crl.ppo.trainer.GlobalTrainer`:

* **Min-max (Formulation A, `GlobalTrainer`)** -- the local and global policies
  are each other's *moving* reference; the global maximizes its past-task lead
  while a dual multiplier ``mu`` constrains it on the current task. Saddle point.

* **Stored-expert (Formulation B, THIS FILE)** -- every task's expert is frozen
  and its value ``V*_i`` is a fixed *ceiling*. There is no adversary and nothing
  to trade, so the min-max collapses (objective doc, Q2): consolidation becomes a
  plain one-sided-shortfall **regression** toward the ceilings. **No multiplier,
  no min-max, no local phase.** Past and current tasks are treated *symmetrically*
  -- every task gets the same gap-weighted coefficient.

Per-task actor coefficient (eq 46, at the start-distribution / scalar level):

    coeff_i = omega_i * 2 * max(0, V*_i - V^{pi_phi}_i)

``V*_i`` is the frozen expert's Monte-Carlo value; ``V^{pi_phi}_i`` is the
global policy's **fresh** Monte-Carlo value on task ``i``. Reading the global's
value off its own critic *head* -- which the actor update drags along on the
shared trunk -- was the bug the objective doc calls out (Q1); we therefore
estimate it from rollouts. Both ``V*_i`` and ``V^{pi_phi}_i`` come from the SAME
evaluator (:func:`evaluate_value_and_score`), so they share reward scaling,
clipping and horizon -- this dissolves the scale-mismatch silent bug (Q4).

At the ceiling the gap is zero, the coefficient vanishes, and the actor stops
moving on that task (eq 41/46) -- exactly the intended fixed point. The critic /
GAE / value loss / entropy are standard PPO, inherited unchanged from
:class:`~crl.ppo.trainer.PPOTrainer`.
"""

from __future__ import annotations

import torch

from crl.envs.base import Task
from crl.policies.base import Policy
from crl.ppo.collector import RolloutCollector
from crl.ppo.evaluate import evaluate_value_and_score
from crl.ppo.trainer import ProbeHook, PPOTrainer


class StoredExpertTrainer(PPOTrainer):
    """Gap-weighted consolidation toward frozen expert ceilings (eq 46)."""

    def _global_value(self, policy: Policy, task: Task) -> tuple[float, float]:
        """Fresh Monte-Carlo value ``V^{pi_phi}_i`` (+ raw score) of the global on
        one task, using the SAME evaluator/settings the frozen expert ``V*_i`` was
        computed with (so the two are directly comparable). Never the critic head."""
        v, s, _, _ = evaluate_value_and_score(
            policy, task, self.ppo.constraint_episodes, self.ppo.n_envs,
            self.device, seed=self.ppo.eval_seed, greedy=self.ppo.constraint_greedy,
            noop_enumerate=self.ppo.eval_noop_enumerate,
            max_ep_steps=self.ppo.eval_max_ep_steps)
        return v, s

    def train(
        self,
        global_policy: Policy,
        tasks: list[Task],
        expert_values: list[float],
        omega: list[float],
        num_iters: int,
        seed: int,
        current_task: int,
        expert_scores: list[float] | None = None,
        probe: ProbeHook | None = None,
    ) -> None:
        """One consolidation phase over ``tasks`` (all seen so far, 1..k).

        ``expert_values[i]`` is the frozen ceiling ``V*_i``; ``omega[i]`` its
        weight (uniform ``1/k``). Every ``constraint_every`` iters the global's
        fresh value on each task is re-estimated and the per-task gap coefficient
        ``omega_i * 2 * max(0, V*_i - V_i^G)`` refreshed; between refreshes the
        coefficients are held fixed (slow-timescale, mirroring the mu cadence in
        the min-max trainer). No dual multiplier is involved."""
        cfg = self.ppo
        k = len(tasks)
        optimizer = self._new_optimizer(global_policy)
        # One persistent collector per task -- replay-free fresh rollouts in every
        # (past + current) environment. Formulation B has no past/current split.
        collectors = [
            RolloutCollector(t, cfg.n_envs, cfg.n_steps, self.device, seed + 101 + i,
                             async_mode=cfg.async_envs)
            for i, t in enumerate(tasks)
        ]
        gaps = [0.0] * k
        vg = [float("nan")] * k
        sg = [float("nan")] * k
        coeffs = [0.0] * k
        try:
            for it in range(num_iters):
                streams = [c.collect(global_policy, cfg.gae_lambda) for c in collectors]

                # Refresh the gap coefficients on the slow timescale.
                if it % max(1, cfg.constraint_every) == 0:
                    for i, t in enumerate(tasks):
                        vg[i], sg[i] = self._global_value(global_policy, t)
                        gaps[i] = max(0.0, expert_values[i] - vg[i])
                    coeffs = [omega[i] * 2.0 * gaps[i] for i in range(k)]

                stats = self.optimize_batches(global_policy, optimizer, streams, coeffs)

                if probe is not None:
                    probe("stored_expert", current_task)
                if it % self.log_every == 0:
                    ratios = [(vg[i] / expert_values[i]) if expert_values[i] else float("nan")
                              for i in range(k)]
                    self.logger.log({
                        "phase": "stored_expert", "task": current_task, "step": it,
                        "V_global": list(vg), "V_expert": list(expert_values),
                        "score_global": list(sg),
                        "score_expert": list(expert_scores) if expert_scores else None,
                        "gap": list(gaps), "coeff": list(coeffs),
                        "ratio_v": ratios, **stats,
                    })
                    gstr = " ".join(f"{r:.2f}" for r in ratios)
                    print(f"[stored_expert k={current_task}] it={it:4d} "
                          f"V/V*=[{gstr}] pg={stats['pg_loss']:.3f} "
                          f"v={stats['v_loss']:.3f}")
        finally:
            for c in collectors:
                c.close()
