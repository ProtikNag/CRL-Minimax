"""OURS. Constrained two-policy min-max continual RL.

Task 1        standard PPO on the global policy. No past tasks, no constraint.
Task k >= 2   per cycle, two phases:

  LOCAL   theta^0 = phi (clone the global), then standard PPO on task k alone.
          The result is frozen. Its value is V_k^L, the constraint target.
          It is NOT an externally trained expert: the local model IS the
          specialist, so results normalise against it.

  GLOBAL  maximise the past-task objective  sum_{i<k} omega_i V_i
          subject to the current task not falling behind its own local run:

              F_k = [V_k^L - V_k^G]_+^2  <=  eps

          Actor coefficient on the current-task stream is the differentiated
          hinge, coeff_k = 2 mu [V_k^L - V_k^G]_+ , and mu follows projected
          dual ascent. Uniform weights omega_i = 1/k. Replay-free: every past
          environment is rolled out fresh each iteration.
"""

from src.duals import ProjectedAscentDual
from src.policy import clone_policy
from src.ppo import RolloutCollector, evaluate, ppo_update


class MinMaxTrainer:
    def __init__(self, cfg, family, policy):
        self.cfg, self.family, self.policy = cfg, family, policy
        self.mu_ctrl = ProjectedAscentDual(cfg.dual)
        self.eval_matrix = []
        self.local_refs = {}      # task -> greedy score of its frozen local model

    def run(self):
        self.train_first_task()
        self.eval_matrix.append(self.evaluate_seen(1))
        for k in range(2, len(self.family) + 1):
            for _cycle in range(self.cfg.cycles_per_task):
                frozen_local, v_k_local = self.local_phase(k)
                self.global_phase(k, frozen_local, v_k_local)
            self.eval_matrix.append(self.evaluate_seen(k))
        return self.eval_matrix

    # -- local phase -------------------------------------------------------
    def local_phase(self, k):
        task_k = self.family[k - 1]
        local = clone_policy(self.policy, trainable=True)   # theta^0 = phi
        plain_ppo(local, task_k, self.cfg.iters_for(task_k, phase="local"), self.cfg)

        frozen = clone_policy(local, trainable=False)
        self.local_refs[task_k] = evaluate(frozen, task_k, greedy=True).score
        v_k_local = evaluate(frozen, task_k).value       # discounted return, = V_k^L
        return frozen, v_k_local

    # -- global phase ------------------------------------------------------
    def global_phase(self, k, frozen_local, v_k_local):
        task_k = self.family[k - 1]
        past = self.family[: k - 1]
        omega = [1.0 / k] * (k - 1)                      # uniform weights
        self.mu_ctrl.reset()                             # unless warm_start

        # One collector per seen task. Replay-free: these roll out the real past
        # environments again, they are not a stored buffer.
        past_collectors = [RolloutCollector(t, self.cfg) for t in past]
        cur_collector = RolloutCollector(task_k, self.cfg)

        mu, shortfall, met = self.mu_ctrl.value, 0.0, 0
        for it in range(self.cfg.global_iters):
            if self.cfg.past_task_sampling == "sample" and past:
                # One random past task per iteration, rescaled by the count.
                # Unbiased estimate of the full sum over past tasks.
                j = random_index(len(past))
                past_batches = [past_collectors[j].collect(self.policy)]
                past_coeffs = [omega[j] * len(past)]
            else:
                past_batches = [c.collect(self.policy) for c in past_collectors]
                past_coeffs = list(omega)
            cur_batch = cur_collector.collect(self.policy)

            # Dual on a slower timescale than the primal: refresh every
            # constraint_every iterations, hold coeff_k fixed in between.
            if it % self.cfg.constraint_every == 0:
                v_k_global = self.estimate_v(self.policy, task_k, frozen_local)
                shortfall = max(0.0, v_k_local - v_k_global)   # [V_k^L - V_k^G]_+
                constraint = shortfall ** 2                    # F_k
                mu = self.mu_ctrl.update(constraint, self.cfg.eps)
            coeff_k = mu * 2.0 * shortfall                     # d/dphi of the hinge

            # The actor loss is a weighted sum over all seen tasks; the critic
            # loss is the plain value MSE on every stream. ppo_update normalises
            # the actor coefficients so an unbounded mu cannot starve the critic.
            ppo_update(
                self.policy,
                streams=past_batches + [cur_batch],
                actor_coeffs=past_coeffs + [coeff_k],
                cfg=self.cfg,
            )

            # `patience` CONSECUTIVE passing checks, not one. A single passing
            # check can be noise from a lucky eval.
            met = met + 1 if self.retention_reached(k) else 0
            if met >= self.cfg.patience and it >= self.cfg.min_iters:
                break

    def estimate_v(self, policy, task_k, frozen_local):
        """V_k^G, the current task's value under the global policy.

        mc         fresh on-policy Monte-Carlo rollouts. Used for Atari and
                   GridWorld (the PPO backend), where episodes are short enough
                   to roll out to termination.
        bootstrap  windowed critic value-gap: a frozen local critic evaluates
                   both ends of an H-step window under the global policy's own
                   rollout, with a gamma^H bootstrap. Used for Meta-World, where
                   full episodes are too expensive. This is a biased proxy for
                   the MC value and is instrumented against MC in that run.
        """
        if self.cfg.value_estimator == "mc":
            return evaluate(policy, task_k).value
        return windowed_critic_gap(policy, frozen_local, task_k, self.cfg)

    def retention_reached(self, k):
        """Stop the global phase only when EVERY seen task is back to a fraction
        of its own local reference; otherwise run to the iteration cap.

        Stopping on the current task alone would let the global phase quit while
        past tasks are still depressed, which is the thing it exists to fix.
        """
        for task in self.family[:k]:
            # environment call: greedy score of the global policy on this task
            score = evaluate(self.policy, task, greedy=True).score
            if score < self.cfg.retention_frac * self.local_refs[task]:
                return False
        return True

    def evaluate_seen(self, k):
        """Row k of the eval matrix: raw game score on every task seen so far.

        The matrix reports the RAW score. The constraint above uses the
        DISCOUNTED return. Two different quantities, deliberately.
        """
        return [evaluate(self.policy, t, greedy=True).score for t in self.family[:k]]


def plain_ppo(policy, task, n_iters, cfg):
    """Standard PPO on one task. Used for task 1, for every local phase, and by
    the fine-tuning and from-scratch baselines."""
    collector = RolloutCollector(task, cfg)
    for it in range(n_iters):
        batch = collector.collect(policy)
        ppo_update(policy, streams=[batch], actor_coeffs=[1.0], cfg=cfg)
        # environment call: greedy score every stop_eval_every iters, ~10 episodes.
        # Early-stop once it clears the task threshold for `patience` checks in a
        # row. Thresholds are set to the jointly-trained model's score, so a task
        # is not left undertrained; thresholds that were too low silently capped
        # the global policy on those tasks for the rest of the sequence.
        if reached_threshold(policy, task, cfg):   # counts consecutive passes
            break


# ---------------------------------------------------------------------------
# environment-facing helpers, stubbed: this branch is about the method
def random_index(n): ...
def reached_threshold(policy, task, cfg): ...
def windowed_critic_gap(policy, frozen_local, task, cfg): ...
