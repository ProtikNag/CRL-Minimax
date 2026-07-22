"""Reusable PPO core, specialized into Local and Global trainers.

``PPOTrainer`` is the *entire* standard PPO optimizer: it turns a set of collected
rollout streams into gradient steps via the clipped surrogate, a standard
value-MSE critic loss, an entropy bonus, GAE advantages, and global grad-norm
clipping. Nothing continual lives here.

Two specializations:

* :class:`LocalTrainer` -- standard PPO on the current task (one stream, actor
  coefficient 1). No constraint, no past-task rollouts.
* :class:`GlobalTrainer` -- the SAME PPO core, but the actor receives the
  continual-learning term: past tasks enter with weight ``omega_i`` and the
  current task enters with coefficient ``mu * 2 * shortfall`` (the differentiated
  one-sided squared shortfall, eqs 30-32). It owns the replay-free fresh
  past-task rollouts, the Monte-Carlo shortfall estimate, and the ``mu`` dual
  update. The critic / GAE / value loss / entropy are inherited unchanged --
  only the ACTOR is constrained.

The only thing that differs between the two is the per-stream **actor
coefficient**; the optimization routine (:meth:`PPOTrainer.optimize_batches`) is
shared verbatim.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from crl.config import PPOConfig
from crl.duals.controllers import DualController
from crl.envs.base import Task
from crl.policies.base import Policy
from crl.ppo.collector import RolloutBatch, RolloutCollector
from crl.ppo.evaluate import evaluate_value_and_score

# Callback invoked once per PPO iteration: (phase_type, current_task) -> None.
ProbeHook = Callable[[str, int], None]


class PPOTrainer:
    """Standard clipped-PPO optimizer shared by the local and global phases."""

    def __init__(self, cfg: PPOConfig, device: torch.device, logger, log_every: int) -> None:
        self.ppo = cfg
        self.device = device
        self.logger = logger
        self.log_every = max(1, log_every)

    def _new_optimizer(self, policy: Policy) -> torch.optim.Optimizer:
        # Adam over the trainable params (whole model normally; only the heads
        # when the trunk is frozen for the head-only consolidation probe).
        params = [p for p in policy.parameters() if p.requires_grad]
        return torch.optim.Adam(params, lr=self.ppo.lr, eps=1e-5)

    def _stop_score(self, policy: Policy, task: Task, iters_done: int) -> float | None:
        """Greedy score for the early-stop check on this iteration, or None if
        this iteration is not a check point (or the task has no threshold)."""
        cfg = self.ppo
        thr = float(getattr(task, "threshold", float("inf")))
        if thr == float("inf") or cfg.stop_eval_every <= 0:
            return None
        if iters_done < cfg.min_iters or iters_done % cfg.stop_eval_every != 0:
            return None
        _, score, _, _ = evaluate_value_and_score(
            policy, task, cfg.stop_eval_episodes, cfg.n_envs, self.device,
            seed=cfg.eval_seed, greedy=True,
        )
        return score

    def optimize_batches(
        self,
        policy: Policy,
        optimizer: torch.optim.Optimizer,
        streams: list[RolloutBatch],
        actor_coeffs: list[float],
    ) -> dict[str, float]:
        """One PPO update (``ppo_epochs`` x minibatches) over the given streams.

        Each stream contributes ``actor_coeff * clipped_surrogate`` to the actor
        loss and a standard ``vf_coef * value_MSE`` to the critic loss, plus the
        standard entropy bonus. All streams share the minibatch schedule (equal
        size ``N = n_envs * n_steps``).
        """
        cfg = self.ppo
        n = streams[0].obs.shape[0]
        mb_size = max(1, n // cfg.num_minibatches)
        # Normalize the actor coefficients so the actor-loss scale stays O(1)
        # regardless of the dual multiplier's magnitude (PPO-Lagrangian style).
        # The dual coefficient mu*2*shortfall is unbounded; without this, a large
        # mu makes the actor gradient dominate the shared global grad-norm clip
        # and STARVES the shared critic's value-loss gradient, breaking GAE
        # advantages so the actor can no longer improve. Dividing every actor
        # coefficient by their sum is a positive rescaling of the ascent
        # direction -- it preserves the relative past/current weighting (and thus
        # the primal-dual fixed point), only bounding the step magnitude. For the
        # local phase (coeffs == [1.0]) this is a no-op.
        coeff_sum = float(sum(actor_coeffs))
        if coeff_sum > 0:
            actor_coeffs = [c / coeff_sum for c in actor_coeffs]
        pg_acc = v_acc = ent_acc = kl_acc = clip_acc = 0.0
        count = 0
        for _ in range(cfg.ppo_epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, mb_size):
                idx = perm[start : start + mb_size]
                total_loss = torch.zeros((), device=self.device)
                for coeff, batch in zip(actor_coeffs, streams):
                    dist, value = policy.dist_value(batch.obs[idx], batch.task_id)
                    new_logp = dist.log_prob(batch.actions[idx])
                    logratio = new_logp - batch.logprobs[idx]
                    ratio = logratio.exp()
                    adv = batch.advantages[idx]
                    if cfg.normalize_advantage:
                        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                    pg_loss = torch.max(
                        -adv * ratio,
                        -adv * torch.clamp(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio),
                    ).mean()
                    v_loss = 0.5 * (value - batch.returns[idx]).pow(2).mean()
                    entropy = dist.entropy().mean()
                    total_loss = (
                        total_loss
                        + coeff * pg_loss
                        + cfg.vf_coef * v_loss
                        - cfg.ent_coef * entropy
                    )
                    with torch.no_grad():
                        pg_acc += float(pg_loss)
                        v_acc += float(v_loss)
                        ent_acc += float(entropy)
                        kl_acc += float((ratio - 1 - logratio).mean())
                        clip_acc += float(
                            ((ratio - 1.0).abs() > cfg.clip_ratio).float().mean()
                        )
                        count += 1
                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                optimizer.step()
        c = max(1, count)
        return {
            "pg_loss": pg_acc / c,
            "v_loss": v_acc / c,
            "entropy": ent_acc / c,
            "approx_kl": kl_acc / c,
            "clipfrac": clip_acc / c,
        }


class LocalTrainer(PPOTrainer):
    """Standard PPO on the current task (the min-max *local* player)."""

    def train(
        self,
        policy: Policy,
        task: Task,
        num_iters: int,
        seed: int,
        current_task: int,
        phase_type: str = "local",
        probe: ProbeHook | None = None,
    ) -> None:
        optimizer = self._new_optimizer(policy)
        collector = RolloutCollector(
            task, self.ppo.n_envs, self.ppo.n_steps, self.device, seed
        )
        thr = float(getattr(task, "threshold", float("inf")))
        met = 0
        try:
            for it in range(num_iters):
                batch = collector.collect(policy, self.ppo.gae_lambda)
                stats = self.optimize_batches(policy, optimizer, [batch], [1.0])
                if probe is not None:
                    probe(phase_type, current_task)
                gscore = self._stop_score(policy, task, it + 1)
                if gscore is not None:
                    met = met + 1 if gscore >= thr else 0
                if it % self.log_every == 0 or gscore is not None:
                    ep = batch.ep_returns
                    ep_mean = sum(ep) / len(ep) if ep else float("nan")
                    self.logger.log(
                        {
                            "phase": phase_type, "task": current_task, "step": it,
                            "ep_return_clipped": ep_mean, "greedy_score": gscore,
                            **stats,
                        }
                    )
                    gs = f" greedy={gscore:.1f}/{thr:.0f}" if gscore is not None else ""
                    print(
                        f"[{phase_type} k={current_task}] it={it:4d} "
                        f"epR(clip)={ep_mean:.3f} pg={stats['pg_loss']:.3f} "
                        f"v={stats['v_loss']:.3f} ent={stats['entropy']:.3f}{gs}"
                    )
                if met >= self.ppo.patience:
                    print(f"[{phase_type} k={current_task}] EARLY STOP it={it+1} "
                          f"greedy={gscore:.1f} >= thr={thr:.0f}")
                    break
        finally:
            collector.close()


class GlobalTrainer(PPOTrainer):
    """PPO consolidation with the actor-only continual-learning constraint.

    Maximizes the past-task objective ``sum_i omega_i V_i`` while the single
    multiplier ``mu`` enforces the current-task floor ``V_k^G >= V_k^L`` via the
    one-sided squared shortfall (eqs 11-14, 30-32). Fresh rollouts in every task's
    environment each iteration (replay-free).
    """

    def _actor_pg(self, policy: Policy, batch: RolloutBatch) -> torch.Tensor:
        """Clipped-surrogate actor loss on one full stream (for grad diagnostics)."""
        cfg = self.ppo
        dist, _ = policy.dist_value(batch.obs, batch.task_id)
        ratio = (dist.log_prob(batch.actions) - batch.logprobs).exp()
        adv = batch.advantages
        if cfg.normalize_advantage:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return torch.max(
            -adv * ratio,
            -adv * torch.clamp(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio),
        ).mean()

    def _grad_decomposition(self, policy, past_batches, cur_batch, omega, coeff_k):
        """Split the actor update into the past-task objective gradient g_old =
        grad[sum_i omega_i * pg_i] and the current-task constraint gradient
        g_new = grad[coeff_k * pg_k] (coeffs normalized as in the real update).
        Returns (||g_new||, ||g_old||, ||g_new||/||g_old||, cos(g_old, g_new))."""
        params = [p for p in policy.parameters() if p.requires_grad]
        coeff_sum = float(sum(omega[: len(past_batches)]) + coeff_k)
        scale = (1.0 / coeff_sum) if coeff_sum > 0 else 1.0

        def flat(loss):
            gs = torch.autograd.grad(loss, params, allow_unused=True, retain_graph=False)
            return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                              for g, p in zip(gs, params)])

        g_new = flat(coeff_k * scale * self._actor_pg(policy, cur_batch))
        if past_batches:
            loss_old = sum(w * scale * self._actor_pg(policy, b)
                           for w, b in zip(omega, past_batches))
            g_old = flat(loss_old)
        else:
            g_old = torch.zeros_like(g_new)
        nn_new, nn_old = float(g_new.norm()), float(g_old.norm())
        ratio = nn_new / nn_old if nn_old > 0 else float("inf")
        denom = g_new.norm() * g_old.norm()
        cos = float(torch.dot(g_new, g_old) / denom) if float(denom) > 0 else float("nan")
        return nn_new, nn_old, ratio, cos

    def _log_diagnostics(self, policy, task_k, past_tasks, ref_current, mu, shortfall,
                         constraint, eps, coeff_k, v_k_g, past_batches, cur_batch,
                         omega, current_task, step):
        """Emit one rich 'global_diag' row (see PPOConfig.diagnostics)."""
        cfg = self.ppo
        # current-task greedy performance (trajectory point) + fresh stochastic V_k^G
        _, cur_score, _, _ = evaluate_value_and_score(
            policy, task_k, cfg.diag_score_episodes, cfg.n_envs, self.device,
            seed=cfg.eval_seed, greedy=True)
        # each past task individually: value V_i^G (stochastic) + greedy score
        past = []
        for t in past_tasks:
            vi, _, _, _ = evaluate_value_and_score(
                policy, t, cfg.constraint_episodes, cfg.n_envs, self.device, greedy=False)
            _, si, _, _ = evaluate_value_and_score(
                policy, t, cfg.diag_score_episodes, cfg.n_envs, self.device,
                seed=cfg.eval_seed, greedy=True)
            past.append({"name": t.spec.name, "task_id": t.spec.task_id,
                         "V_i_global": vi, "greedy_score": si})
        nn_new, nn_old, ratio, cos = self._grad_decomposition(
            policy, past_batches, cur_batch, omega, coeff_k)
        self.logger.log({
            "phase": "global_diag", "task": current_task, "step": step,
            "cur_name": task_k.spec.name,
            "mu": mu, "coeff_k": coeff_k,
            "V_k_local": ref_current, "V_k_global": v_k_g,
            "V_gap": ref_current - v_k_g,           # V_L - V_G (oscillation)
            "F_G": constraint, "eps": eps,
            "constraint_active": bool(shortfall > 0.0),
            "cur_greedy_score": cur_score,          # current-task perf trajectory
            "grad_norm_new": nn_new, "grad_norm_old": nn_old,
            "grad_ratio_new_over_old": ratio,       # <1 => old-task objective wins
            "grad_cos_new_old": cos,                # alignment / conflict
            "past": past,                           # per-old-task value + score
        })
        print(f"[diag k={current_task}] it={step:4d} Vgap={ref_current-v_k_g:+.3f} "
              f"mu={mu:.2f} F_G={constraint:.4f} |g_new|/|g_old|={ratio:.3f} "
              f"cos={cos:+.3f} cur={cur_score:.1f}")

    def train(
        self,
        global_policy: Policy,
        task_k: Task,
        past_tasks: list[Task],
        ref_current: float,
        mu_ctrl: DualController,
        omega: list[float],
        eps: float,
        num_iters: int,
        seed: int,
        current_task: int,
        probe: ProbeHook | None = None,
    ) -> None:
        """``omega`` is the per-stream weight list aligned to
        ``past_tasks + [task_k]`` order is NOT used; past weights are
        ``omega[:len(past_tasks)]`` and the current task's PG weight is the
        constraint coefficient (computed here)."""
        optimizer = self._new_optimizer(global_policy)
        cfg = self.ppo
        # One persistent collector per task (current + all past): replay-free
        # fresh rollouts in the old environments.
        past_collectors = [
            RolloutCollector(t, cfg.n_envs, cfg.n_steps, self.device, seed + 101 + i)
            for i, t in enumerate(past_tasks)
        ]
        cur_collector = RolloutCollector(
            task_k, cfg.n_envs, cfg.n_steps, self.device, seed + 7
        )
        mu = mu_ctrl.value
        shortfall = 0.0
        constraint = 0.0
        v_k_g = float("nan")
        thr = float(getattr(task_k, "threshold", float("inf")))
        met = 0
        try:
            for it in range(num_iters):
                past_batches = [
                    c.collect(global_policy, cfg.gae_lambda) for c in past_collectors
                ]
                cur_batch = cur_collector.collect(global_policy, cfg.gae_lambda)

                # Slower dual timescale: refresh V_k^G / mu every constraint_every.
                if it % max(1, cfg.constraint_every) == 0:
                    v_k_g, _, _, _ = evaluate_value_and_score(
                        global_policy, task_k, cfg.constraint_episodes,
                        cfg.n_envs, self.device,
                    )
                    shortfall = max(0.0, ref_current - v_k_g)  # V_k^L - V_k^G
                    constraint = shortfall * shortfall  # squared hinge (eq 32)
                    mu = mu_ctrl.update(constraint, eps)
                coeff_k = mu * 2.0 * shortfall  # differentiated shortfall (eq 30)

                if cfg.diagnostics and it % max(1, cfg.diag_every) == 0:
                    self._log_diagnostics(
                        global_policy, task_k, past_tasks, ref_current, mu, shortfall,
                        constraint, eps, coeff_k, v_k_g, past_batches, cur_batch,
                        omega, current_task, it)

                streams = past_batches + [cur_batch]
                coeffs = list(omega[: len(past_tasks)]) + [coeff_k]
                stats = self.optimize_batches(global_policy, optimizer, streams, coeffs)

                if probe is not None:
                    probe("global", current_task)
                # Early stop: the global has consolidated the current game to its
                # greedy threshold (past tasks are maintained by the omega term).
                gscore = self._stop_score(global_policy, task_k, it + 1)
                if gscore is not None:
                    met = met + 1 if gscore >= thr else 0
                if it % self.log_every == 0 or gscore is not None:
                    self.logger.log(
                        {
                            "phase": "global", "task": current_task, "step": it,
                            "F_G": constraint, "mu": mu, "V_k_global": v_k_g,
                            "V_k_ref_local": ref_current, "coeff_k": coeff_k,
                            "greedy_score": gscore, **stats,
                        }
                    )
                    gs = f" greedy={gscore:.1f}/{thr:.0f}" if gscore is not None else ""
                    print(
                        f"[global k={current_task}] it={it:4d} "
                        f"Vk_G={v_k_g:.3f} refL={ref_current:.3f} "
                        f"F_G={constraint:.5f} mu={mu:.3f} pg={stats['pg_loss']:.3f}{gs}"
                    )
                if met >= self.ppo.patience:
                    print(f"[global k={current_task}] EARLY STOP it={it+1} "
                          f"greedy={gscore:.1f} >= thr={thr:.0f}")
                    break
        finally:
            for c in past_collectors:
                c.close()
            cur_collector.close()
