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
from crl.ppo.evaluate import (critic_values, evaluate_intermediate_values_vec,
                              evaluate_value_and_score)

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
        bc: tuple | None = None,
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
                n_streams = len(streams)
                for si, (coeff, batch) in enumerate(zip(actor_coeffs, streams)):
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
                    # EXPERIMENT 2: behavioral-cloning term on the CURRENT task
                    # (the last stream): coef * KL(pi_local || pi_global).
                    if bc is not None and si == n_streams - 1:
                        local_policy, bc_coef = bc
                        with torch.no_grad():
                            pl = local_policy.dist(batch.obs[idx], batch.task_id)
                        total_loss = total_loss + bc_coef * torch.distributions.kl_divergence(
                            pl, dist).mean()
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
            task, self.ppo.n_envs, self.ppo.n_steps, self.device, seed,
            async_mode=self.ppo.async_envs,
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
        """Decompose the actor update into the current-task constraint gradient
        g_new = grad[coeff_k * pg_k] and EACH past-task gradient g_i =
        grad[omega_i * pg_i] separately (coeffs normalized as in the real update).

        The aggregate cos(g_new, sum_i g_i) can read ~0 while individual tasks
        strongly align/conflict and cancel in the sum, so we return the PER-TASK
        alignment cos(g_new, g_i) as well. Returns
        (||g_new||, ||g_old||, ||g_new||/||g_old||, cos_aggregate, per_task) where
        per_task[i] = {grad_norm_i, grad_cos_new_i} aligned to past_batches."""
        params = [p for p in policy.parameters() if p.requires_grad]
        coeff_sum = float(sum(omega[: len(past_batches)]) + coeff_k)
        scale = (1.0 / coeff_sum) if coeff_sum > 0 else 1.0

        def flat(loss):
            gs = torch.autograd.grad(loss, params, allow_unused=True, retain_graph=False)
            return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                              for g, p in zip(gs, params)])

        def cosine(a, b):
            d = a.norm() * b.norm()
            return float(torch.dot(a, b) / d) if float(d) > 0 else float("nan")

        g_new = flat(coeff_k * scale * self._actor_pg(policy, cur_batch))
        per_task = []                       # aligned to past_batches order
        g_old = torch.zeros_like(g_new)
        for w, b in zip(omega, past_batches):
            g_i = flat(w * scale * self._actor_pg(policy, b))
            per_task.append({"grad_norm_i": float(g_i.norm()),
                             "grad_cos_new_i": cosine(g_new, g_i)})
            g_old = g_old + g_i
        nn_new, nn_old = float(g_new.norm()), float(g_old.norm())
        ratio = nn_new / nn_old if nn_old > 0 else float("inf")
        return nn_new, nn_old, ratio, cosine(g_new, g_old), per_task

    def _log_diagnostics(self, policy, task_k, past_tasks, ref_current, mu, shortfall,
                         constraint, eps, coeff_k, v_k_g, past_batches, cur_batch,
                         omega, current_task, step, local_policy=None):
        """Emit one rich 'global_diag' row (see PPOConfig.diagnostics)."""
        cfg = self.ppo
        # EXPERIMENT 2: behavioral gap between the frozen local and the global on
        # current-task states. If the value gap -> 0 while this KL stays large,
        # the scalar value constraint is satisfied by a behaviorally-different
        # policy (the value constraint is too weak).
        kl_lg = tv_lg = float("nan")
        if local_policy is not None:
            with torch.no_grad():
                pl = local_policy.dist(cur_batch.obs, cur_batch.task_id)
                pg = policy.dist(cur_batch.obs, cur_batch.task_id)
                kl_lg = float(torch.distributions.kl_divergence(pl, pg).mean())
                tv_lg = float(0.5 * (pl.probs - pg.probs).abs().sum(-1).mean())
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
        nn_new, nn_old, ratio, cos, per_task = self._grad_decomposition(
            policy, past_batches, cur_batch, omega, coeff_k)
        for pd, gd in zip(past, per_task):  # attach per-task grad alignment
            pd.update(gd)
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
            "kl_local_global": kl_lg,               # behavioral gap (value too weak?)
            "tv_local_global": tv_lg,
            "past": past,                           # per-old-task value + score
        })
        print(f"[diag k={current_task}] it={step:4d} Vgap={ref_current-v_k_g:+.3f} "
              f"mu={mu:.2f} F_G={constraint:.4f} |g_new|/|g_old|={ratio:.3f} "
              f"cos={cos:+.3f} cur={cur_score:.1f}")

    def _monitor_past(self, global_policy, past_tasks, past_caches, past_expert_values,
                      current_task, step, past_eval_caches=None):
        """Per-past-task retention probe: for each past task, the global's value
        V_i^G over the 50 no-op + 50 intermediate states, logged with the ratio vs
        the fixed expert value. Diagnostic only (not in the update). Uses the
        CRITIC on cached obs when enabled (cheap, so it can run every iteration);
        otherwise falls back to a full MC rollout."""
        cfg = self.ppo
        rows = []
        for i, t in enumerate(past_tasks):
            ec = past_eval_caches[i] if past_eval_caches else None
            if cfg.constraint_use_critic and ec is not None:
                vn = critic_values(global_policy, ec["noop_obs"], t.spec.task_id)
                vi = critic_values(global_policy, ec["interm_obs"], t.spec.task_id)
                v_noop = (sum(vn) / len(vn)) if vn else float("nan")
                v_interm = (sum(vi) / len(vi)) if vi else float("nan")
                s_noop = float("nan")
            else:
                v_noop, s_noop, _, _ = evaluate_value_and_score(
                    global_policy, t, cfg.constraint_episodes, cfg.n_envs, self.device,
                    seed=cfg.eval_seed, greedy=cfg.constraint_greedy,
                    noop_enumerate=cfg.eval_noop_enumerate,
                    max_ep_steps=cfg.eval_max_ep_steps)
                v_interm = float("nan")
                cache_i = past_caches[i] if past_caches else None
                if cache_i is not None and cfg.constraint_intermediate_states > 0:
                    vg_i, _ = evaluate_intermediate_values_vec(
                        global_policy, t, t.spec.task_id, cache_i, self.device,
                        n_envs=cfg.n_envs, max_ep_steps=cfg.eval_max_ep_steps)
                    v_interm = (sum(vg_i) / len(vg_i)) if vg_i else float("nan")
            ve = (past_expert_values[i] if past_expert_values else float("nan"))
            rows.append({"name": t.spec.name, "task_id": t.spec.task_id,
                         "V_noop": v_noop, "V_interm": v_interm, "score_noop": s_noop,
                         "V_expert": ve,
                         "ratio_noop": (v_noop / ve) if ve else float("nan"),
                         "ratio_interm": (v_interm / ve) if ve else float("nan")})
        self.logger.log({"phase": "past_monitor", "task": current_task, "step": step,
                         "past": rows})
        if rows:
            summ = " ".join(f"{r['name'].replace('atari-','')}:"
                            f"{r['ratio_noop']:.2f}/{r['ratio_interm']:.2f}" for r in rows)
            print(f"[past_mon k={current_task}] it={step:4d} (noop/interm ratio) {summ}")

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
        local_policy: Policy | None = None,
        ref_score: float = float("nan"),
        intermediate_cache: dict | None = None,
        past_intermediate_caches: list[dict | None] | None = None,
        past_expert_values: list[float] | None = None,
        monitor_every: int = 0,
        eval_cache: dict | None = None,
        past_eval_caches: list[dict | None] | None = None,
    ) -> None:
        """``omega`` is the per-stream weight list aligned to
        ``past_tasks + [task_k]`` order is NOT used; past weights are
        ``omega[:len(past_tasks)]`` and the current task's PG weight is the
        constraint coefficient (computed here).

        ``monitor_every`` > 0 turns on the brute-force past-task retention
        monitor: every that-many iters, each past task's global value V_i^G is
        re-estimated over 50 no-op + 50 intermediate states (same eval as the
        current-task shortfall) and logged as a "past_monitor" row. Diagnostic
        only -- it does NOT change Eq 29's update."""
        optimizer = self._new_optimizer(global_policy)
        cfg = self.ppo
        # One persistent collector per task (current + all past): replay-free
        # fresh rollouts in the old environments.
        past_collectors = [
            RolloutCollector(t, cfg.n_envs, cfg.n_steps, self.device, seed + 101 + i,
                             async_mode=cfg.async_envs)
            for i, t in enumerate(past_tasks)
        ]
        cur_collector = RolloutCollector(
            task_k, cfg.n_envs, cfg.n_steps, self.device, seed + 7,
            async_mode=cfg.async_envs,
        )
        mu = mu_ctrl.value
        shortfall = 0.0
        constraint = 0.0
        v_k_g = float("nan")
        s_k_g = float("nan")  # global's raw (undiscounted) score, same rollouts as V_k_g
        sf_int = 0.0          # mean per-state intermediate shortfall (Run B)
        int_vg = float("nan")  # mean global value over intermediate states
        thr = float(getattr(task_k, "threshold", float("inf")))
        met = 0
        try:
            for it in range(num_iters):
                if cfg.past_task_sampling == "sample" and past_collectors:
                    # Minibatch over past tasks: one random past task per iter,
                    # rescaled by the count -> unbiased estimate of the full sum.
                    j = int(torch.randint(len(past_collectors), (1,)))
                    past_batches = [past_collectors[j].collect(global_policy, cfg.gae_lambda)]
                    past_coeffs = [omega[j] * len(past_collectors)]
                else:
                    past_batches = [c.collect(global_policy, cfg.gae_lambda)
                                    for c in past_collectors]
                    past_coeffs = list(omega[: len(past_tasks)])
                cur_batch = cur_collector.collect(global_policy, cfg.gae_lambda)

                # Refresh V_k^G / mu every constraint_every iters. With the CRITIC
                # value (constraint_use_critic) this is ~ms, so constraint_every=1
                # ("update the value every iteration") is affordable.
                if it % max(1, cfg.constraint_every) == 0:
                    use_critic = (cfg.constraint_use_critic and eval_cache is not None)
                    # --- current-task value on the 50 no-op START states ---
                    if use_critic and eval_cache.get("noop_obs") is not None:
                        vn = critic_values(global_policy, eval_cache["noop_obs"],
                                           task_k.spec.task_id)
                        v_k_g = (sum(vn) / len(vn)) if vn else float("nan")
                    else:
                        # MC fallback: enumeration returns the value AND raw score.
                        v_k_g, s_k_g, _, _ = evaluate_value_and_score(
                            global_policy, task_k, cfg.constraint_episodes,
                            cfg.n_envs, self.device, seed=cfg.eval_seed,
                            greedy=cfg.constraint_greedy,
                            noop_enumerate=cfg.eval_noop_enumerate,
                            max_ep_steps=cfg.eval_max_ep_steps)
                    # --- current-task value on the 50 INTERMEDIATE states ---
                    if use_critic and eval_cache.get("interm_obs") is not None:
                        vg_i = critic_values(global_policy, eval_cache["interm_obs"],
                                             task_k.spec.task_id)
                        ve_i = eval_cache["interm_rtg"]
                    elif (intermediate_cache is not None
                          and cfg.constraint_intermediate_states > 0):
                        vg_i, ve_i = evaluate_intermediate_values_vec(
                            global_policy, task_k, task_k.spec.task_id,
                            intermediate_cache, self.device, n_envs=cfg.n_envs,
                            max_ep_steps=cfg.eval_max_ep_steps)
                    else:
                        vg_i, ve_i = [], []

                    # Per-state one-sided shortfall. "ratio" (relative) form:
                    #   sf = max(0, V_L - V_G) / max(|V_L|, tau)   -- delta lives in
                    # the dual threshold eps (=delta^2), and mu is capped by the dual
                    # controller's max_value. "floored" form keeps the additive hinge
                    # (delta inside, dual threshold 0).
                    def _sf(vl, vg):
                        if cfg.constraint_form == "floored":
                            return max(0.0, (vl - vg)
                                       - cfg.constraint_delta * max(abs(vl), cfg.constraint_tau))
                        return max(0.0, vl - vg) / max(abs(vl), cfg.constraint_tau)
                    dual_eps = 0.0 if cfg.constraint_form == "floored" else eps

                    sf_start = _sf(ref_current, v_k_g)
                    if vg_i:
                        sfs = [_sf(ve, vg) for vg, ve in zip(vg_i, ve_i)]
                        sf_int = sum(sfs) / len(sfs)
                        int_sq = sum(s * s for s in sfs) / len(sfs)
                        int_vg = sum(vg_i) / len(vg_i)
                        shortfall = 0.5 * sf_start + 0.5 * sf_int
                        constraint = 0.5 * sf_start * sf_start + 0.5 * int_sq
                    else:
                        sf_int = 0.0; int_vg = float("nan")
                        shortfall = sf_start
                        constraint = sf_start * sf_start
                    mu = mu_ctrl.update(constraint, dual_eps)

                    # Periodic MC calibration: check the critic value vs a true MC
                    # rollout (drift diagnostic; also refreshes the logged score).
                    if (cfg.constraint_calibrate_every
                            and it % cfg.constraint_calibrate_every == 0):
                        v_mc, s_mc, _, _ = evaluate_value_and_score(
                            global_policy, task_k, cfg.constraint_episodes, cfg.n_envs,
                            self.device, seed=cfg.eval_seed, greedy=cfg.constraint_greedy,
                            noop_enumerate=cfg.eval_noop_enumerate,
                            max_ep_steps=cfg.eval_max_ep_steps)
                        vgi_mc, _ = (evaluate_intermediate_values_vec(
                            global_policy, task_k, task_k.spec.task_id, intermediate_cache,
                            self.device, n_envs=cfg.n_envs, max_ep_steps=cfg.eval_max_ep_steps)
                            if intermediate_cache is not None else ([], []))
                        s_k_g = s_mc
                        self.logger.log({
                            "phase": "calibration", "task": current_task, "step": it,
                            "V_k_critic": v_k_g, "V_k_mc": v_mc, "score_k_mc": s_mc,
                            "V_interm_critic": int_vg,
                            "V_interm_mc": (sum(vgi_mc) / len(vgi_mc)) if vgi_mc else float("nan")})
                coeff_k = mu * 2.0 * shortfall  # differentiated shortfall (eq 30)

                if cfg.diagnostics and it % max(1, cfg.diag_every) == 0:
                    self._log_diagnostics(
                        global_policy, task_k, past_tasks, ref_current, mu, shortfall,
                        constraint, eps, coeff_k, v_k_g, past_batches, cur_batch,
                        omega, current_task, it, local_policy)

                streams = past_batches + [cur_batch]
                coeffs = past_coeffs + [coeff_k]
                bc = ((local_policy, self.ppo.global_bc_coef)
                      if local_policy is not None and self.ppo.global_bc_coef > 0 else None)
                stats = self.optimize_batches(global_policy, optimizer, streams, coeffs, bc=bc)

                if probe is not None:
                    probe("global", current_task)
                if monitor_every > 0 and past_tasks and it % monitor_every == 0:
                    self._monitor_past(global_policy, past_tasks,
                                       past_intermediate_caches, past_expert_values,
                                       current_task, it, past_eval_caches=past_eval_caches)
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
                            "score_k_global": s_k_g, "score_k_ref": ref_score,
                            "sf_intermediate": sf_int, "V_k_intermediate": int_vg,
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
