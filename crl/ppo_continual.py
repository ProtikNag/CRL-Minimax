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
from crl.ppo.evaluate import cache_eval_obs, evaluate_value_and_score
from crl.ppo.stored_expert import StoredExpertTrainer
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
        self._config = config
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
        # Expert references (populated by the stored-expert / consolidate paths;
        # left empty otherwise -- _probe treats an empty list as "no experts").
        self._expert_scores: list[float] = []

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
        if self.ppo.eval_noop_enumerate:  # exact greedy via no-op enumeration
            _, score, std, _ = evaluate_value_and_score(
                policy, task, 0, self.ppo.n_envs, self.device,
                seed=self.ppo.eval_seed, noop_enumerate=True,
                max_ep_steps=self.ppo.eval_max_ep_steps)
            return score, std
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
        # Live retention probe: greedy score on the SEEN tasks only (past + current;
        # future tasks aren't learned yet, so evaluating them wastes eval time).
        # current_task is 1-based, so range(current_task) = tasks 0..k-1 = all seen.
        seen = range(current_task)
        values = [self._eval_report(self.global_policy, self.family.tasks[i])[0]
                  for i in seen]
        exp = [self._expert_scores[i] for i in seen] if self._expert_scores else [0.0] * current_task
        ratio = [values[i] / exp[i] if exp[i] else float("nan") for i in seen]
        self.logger.log(
            {
                "phase": "probe",
                "cumulative_step": self.cumulative_step,
                "current_task": current_task,
                "phase_type": phase_type,
                "values": values,             # greedy scores, seen tasks (past + current)
                "expert_scores": exp,
                "ratio": ratio,               # global/expert per seen task = live retention
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
                local_policy=frozen_local,  # for KL-gap logging + optional BC term
            )
            if self.ppo.global_probe_head_only:  # restore full trainability
                for p in self.global_policy.parameters():
                    p.requires_grad_(True)
            self.logger.log(
                {"phase": "gaps", "task": k, "cycle": cycle,
                 "V_k_ref_local": ref_current}
            )

    def _train_joint(self) -> None:
        """EXPERIMENT 1 (feasibility upper bound). First train a FRESH single-task
        model per game (standard PPO) = the per-task ceiling; then train ONE
        shared-trunk model on ALL games with mixed batches (equal weight, no
        constraint). Compare joint per-task performance to the ceilings: if joint
        reaches them, a feasible shared theta EXISTS."""
        from crl.policies import make_policy
        from crl.ppo.collector import RolloutCollector

        cfg = self.ppo
        tasks = self.family.tasks
        n = len(tasks)

        # --- per-task ceilings (fresh model each; only its head + trunk move) ---
        ceilings = []
        for i, t in enumerate(tasks):
            m = make_policy(self._config.policy, self.family).to(self.device)
            self.local_trainer.train(
                m, t, num_iters=cfg.joint_single_iters, seed=self.seed + 500 + i,
                current_task=i + 1, phase_type="joint_single")
            score, _ = self._eval_report(m, t)
            ceilings.append(score)
            self.logger.log({"phase": "joint_ceiling", "task": i + 1,
                             "game": t.spec.name, "score": score})
            print(f"[joint ceiling] {t.spec.name}: {score:.1f}")

        # --- joint model: one shared trunk, all games mixed, no constraint ---
        joint = make_policy(self._config.policy, self.family).to(self.device)
        collectors = [RolloutCollector(t, cfg.n_envs, cfg.n_steps, self.device,
                                       self.seed + 700 + i) for i, t in enumerate(tasks)]
        opt = self.local_trainer._new_optimizer(joint)
        try:
            for it in range(cfg.joint_iters):
                streams = [c.collect(joint, cfg.gae_lambda) for c in collectors]
                self.local_trainer.optimize_batches(joint, opt, streams, [1.0] * n)
                if cfg.eval_every and (it + 1) % cfg.eval_every == 0:
                    scores = [self._eval_report(joint, t)[0] for t in tasks]
                    self.logger.log({"phase": "joint_probe", "step": it + 1,
                                     "scores": scores})
                    print(f"[joint] it={it+1} scores={['%.0f'%s for s in scores]}")
        finally:
            for c in collectors:
                c.close()

        joint_scores = [self._eval_report(joint, t)[0] for t in tasks]
        thresholds = [float(getattr(t, "threshold", 0.0)) for t in tasks]
        result = {"games": [t.spec.name for t in tasks], "ceilings": ceilings,
                  "joint": joint_scores, "thresholds": thresholds}
        self.eval_matrix = [joint_scores]
        self.logger.log({"phase": "joint_final", **result})
        self.logger.save_json("eval_matrix.json", self.eval_matrix)
        self.logger.save_json("joint_result.json", result)
        torch.save(joint.state_dict(), self.logger.run_dir / "final_policy.pt")
        print(f"[joint] ceilings={['%.0f'%c for c in ceilings]} "
              f"joint={['%.0f'%s for s in joint_scores]}")

    # ------------------------------------------------------------------ #
    # consolidation with precomputed experts as fixed references
    # ------------------------------------------------------------------ #

    def _eval_value_greedy(self, policy, task) -> float:
        """Discounted value V estimated with `constraint_episodes` rollouts
        (greedy iff `constraint_greedy`), fixed seed. The constraint reference."""
        v, _, _, _ = evaluate_value_and_score(
            policy, task, self.ppo.constraint_episodes, self.ppo.n_envs,
            self.device, seed=self.ppo.eval_seed, greedy=self.ppo.constraint_greedy,
            noop_enumerate=self.ppo.eval_noop_enumerate,
            max_ep_steps=self.ppo.eval_max_ep_steps)
        return v

    def _load_expert(self, game: str):
        """Load a frozen single-task Impala expert (`<expert_dir>/<game>/best_model.pt`)
        and a matching single-game task to roll it out in."""
        from crl.config import EnvConfig, PolicyConfig
        from crl.envs import make_family
        from crl.policies import make_policy
        env = EnvConfig(family="atari", params=dict(self._config.env.params),
                        tasks=[{"game": game, "threshold": 0.0}])
        fam = make_family(env)
        pol = make_policy(PolicyConfig(kind="impala_ac", hidden_sizes=[512],
                                       task_conditioned=False), fam).to(self.device)
        sd = torch.load(f"{self.ppo.expert_dir}/{game}/best_model.pt",
                        map_location=self.device)
        pol.load_state_dict(sd)
        for p in pol.parameters():
            p.requires_grad_(False)
        pol.eval()
        return pol, fam.tasks[0]

    def _init_global_from_expert(self, expert) -> None:
        """Global (multi-head, task_conditioned=False) inits from the task-1
        expert: identical trunk + expert's ACTOR head into head 0. The critic head
        is NOT seeded -- it is not part of the actor min-max (eq 29) and a copied
        critic head is invalid on the shared trunk anyway; PPO's value loss trains
        it from scratch on the shared representation."""
        g = self.global_policy
        g.trunk.load_state_dict(expert.trunk.state_dict())
        g.actors[0].load_state_dict(expert.actor.state_dict())

    def _save_ckpt(self, k: int) -> None:
        torch.save(self.global_policy.state_dict(),
                   self.logger.run_dir / f"global_after_task{k}.pt")

    def _log_retention(self, k: int, games: list[str]) -> None:
        """After consolidating task k: global raw score on every SEEN game
        (greedy), logged with the fixed expert scores + ratio. Read on demand."""
        gscores = [self._eval_report(self.global_policy, self.family.tasks[i])[0]
                   for i in range(k)]
        self.eval_matrix.append(gscores)
        self.logger.log({"phase": "retention", "task": k, "games": games[:k],
                         "global_scores": gscores,
                         "expert_scores": self._expert_scores[:k],
                         "ratio": [gscores[i] / self._expert_scores[i]
                                   if self._expert_scores[i] else float("nan")
                                   for i in range(k)]})
        self.logger.save_json("eval_matrix.json", self.eval_matrix)
        print(f"[consolidate] after T{k}: global={[round(s,1) for s in gscores]} "
              f"expert={[round(s,1) for s in self._expert_scores[:k]]}")

    def _load_all_experts(self, games: list[str]) -> None:
        """Load every game's frozen single-task expert and its fixed reference
        value ``V*`` (greedy) + raw score. Populates ``self._experts /
        _expert_values / _expert_scores``. Reuses a cached ``expert_refs.json`` if
        provided (refs are deterministic -> identical across variants), skipping
        the expensive enumeration setup; the expert POLICIES are always loaded
        (needed for the global init / current-head warm-start)."""
        import json as _json
        import os as _os
        cache = None
        if self.ppo.expert_refs_path and _os.path.exists(self.ppo.expert_refs_path):
            c = _json.load(open(self.ppo.expert_refs_path))
            if c.get("games") == games:
                cache = c
                print(f"[expert ref] reusing cached refs from {self.ppo.expert_refs_path}")
        self._experts, self._expert_values, self._expert_scores = [], [], []
        for i, g in enumerate(games):
            epol, etask = self._load_expert(g)   # policy always needed (init/warm-start/BC)
            if cache is not None:
                v, s = cache["expert_values"][i], cache["expert_scores"][i]
            else:
                v = self._eval_value_greedy(epol, etask)
                s = self._eval_report(epol, etask)[0]
            self._experts.append(epol)
            self._expert_values.append(v)
            self._expert_scores.append(s)
            self.logger.log({"phase": "expert_ref", "game": g, "V_L": v, "score": s,
                             "cached": cache is not None})
            print(f"[expert ref] {g}: V*={v:.3f} score={s:.1f}"
                  f"{' (cached)' if cache is not None else ''}")
        self.logger.save_json("expert_refs.json",
                              {"games": games, "expert_values": self._expert_values,
                               "expert_scores": self._expert_scores})

    def _stored_expert(self) -> None:
        """Formulation B: stored-expert consolidation, NO min-max (objective doc,
        'When we have the expert models stored', eqs 34-46).

        Each task's expert value ``V*_i`` is a frozen ceiling; the global regresses
        toward every ceiling at once via the gap-weighted policy gradient
        ``coeff_i = omega_i * 2 * max(0, V*_i - V_i^G)`` -- no multiplier, no
        adversary, past and current tasks symmetric. The global inits from the
        task-1 expert and is consolidated incrementally over tasks 2..K, saving a
        retention row + checkpoint after each phase so the confusion matrix is
        available on demand. See :class:`crl.ppo.stored_expert.StoredExpertTrainer`."""
        games = [t.spec.name.replace("atari-", "") for t in self.family.tasks]
        K = len(self.family)

        self._load_all_experts(games)

        # Global inits from the task-1 expert; retention after T1.
        self._init_global_from_expert(self._experts[0])
        self._log_retention(1, games)
        self._save_ckpt(1)

        trainer = StoredExpertTrainer(self.ppo, self.device, self.logger, self._log_every)
        for k in range(2, K + 1):
            tasks_k = [self.family.tasks[i] for i in range(k)]      # all seen tasks
            expert_vals = [self._expert_values[i] for i in range(k)]
            omega = [1.0 / k] * k                                   # uniform, ALL tasks
            if self.ppo.warmstart_head_from_expert:
                # Seed the new task's ACTOR head from its expert so the gap starts
                # small and consolidation preserves rather than re-learns. The
                # critic head is left to PPO's value loss (a copied critic is
                # invalid on the shared trunk).
                ek = self._experts[k - 1]
                self.global_policy.actors[k - 1].load_state_dict(ek.actor.state_dict())
            trainer.train(
                self.global_policy, tasks_k, expert_vals, omega,
                num_iters=self.ppo.global_iters, seed=self.seed + 1000 * k,
                current_task=k, expert_scores=self._expert_scores[:k], probe=self._probe)
            self.logger.log({"phase": "gaps", "task": k,
                             "V_expert": expert_vals})
            self._log_retention(k, games)
            self._save_ckpt(k)

        torch.save(self.global_policy.state_dict(),
                   self.logger.run_dir / "final_policy.pt")

    def _consolidate(self) -> None:
        games = [t.spec.name.replace("atari-", "") for t in self.family.tasks]
        K = len(self.family)

        # 1) load experts; fixed reference value V_L (greedy) + raw score per game.
        self._load_all_experts(games)

        # 2) global inits from task-1 expert; retention after T1
        self._init_global_from_expert(self._experts[0])
        self._log_retention(1, games)
        self._save_ckpt(1)

        # Intermediate-state cache (Run B): per-game mid/late states the constraint
        # also holds. Loaded once; the current game's slice is passed per task.
        interm = None
        if self.ppo.constraint_intermediate_states > 0 and self.ppo.constraint_intermediate_path:
            interm = _json.load(open(self.ppo.constraint_intermediate_path))
            print(f"[interm] loaded {len(interm['games'])} games from "
                  f"{self.ppo.constraint_intermediate_path}")

        # Critic-eval cache: reach the 50 no-op + 50 intermediate states ONCE per
        # game and store their observation tensors, so the constraint value V_G can
        # be read from the critic (one forward pass) every iteration.
        self._eval_cache = {}
        if self.ppo.constraint_use_critic:
            for i, g in enumerate(games):
                task_i = self.family.tasks[i]
                cg = interm["games"][g] if interm else None
                noop_max = int(getattr(task_i, "_noop_max", 50))
                self._eval_cache[g] = cache_eval_obs(
                    task_i, task_i.spec.task_id, cg, self.device, noop_max,
                    seed=self.ppo.eval_seed)
                n_no = 0 if self._eval_cache[g]["noop_obs"] is None else len(self._eval_cache[g]["noop_obs"])
                n_in = 0 if self._eval_cache[g]["interm_obs"] is None else len(self._eval_cache[g]["interm_obs"])
                print(f"[eval-cache] {g}: {n_no} no-op + {n_in} intermediate states cached")

        # 3) consolidate tasks 2..K (min-max; expert_k is the fixed local ref)
        for k in range(2, K + 1):
            task_k = self.family.tasks[k - 1]
            past_tasks = [self.family.tasks[i] for i in range(k - 1)]
            ref_current = self.ppo.ref_fraction * self._expert_values[k - 1]
            # (B) warm-start the current task's head from its expert (vs random)
            if self.ppo.warmstart_head_from_expert:
                # Seed ONLY the current task's ACTOR head (useful behavior the
                # constraint acts on). The critic head is left to PPO's value loss:
                # a copied critic is invalid on the shared trunk and isn't in eq 29.
                ek = self._experts[k - 1]
                self.global_policy.actors[k - 1].load_state_dict(ek.actor.state_dict())
            # (A) dedicated UNCONSTRAINED learning of the current task first, so the
            # global actually masters it before the constrained consolidation.
            if self.ppo.adapt_iters > 0:
                self.local_trainer.train(
                    self.global_policy, task_k, num_iters=self.ppo.adapt_iters,
                    seed=self.seed + 500 * k, current_task=k, phase_type="adapt",
                    probe=self._probe)
            self.mu_ctrl.reset()
            omega = [1.0 / k] * (k - 1)
            # Brute-force retention monitor: each past task's intermediate-state
            # cache + fixed expert value, so the global's V_i^G can be re-checked
            # over no-op + intermediate states every `past_monitor_every` iters.
            past_caches = ([interm["games"][games[i]] for i in range(k - 1)]
                           if interm else None)
            past_expert_values = [self._expert_values[i] for i in range(k - 1)]
            eval_cache = self._eval_cache.get(games[k - 1])
            past_eval_caches = ([self._eval_cache.get(games[i]) for i in range(k - 1)]
                                if self._eval_cache else None)
            self.global_trainer.train(
                self.global_policy, task_k, past_tasks, ref_current=ref_current,
                mu_ctrl=self.mu_ctrl, omega=omega, eps=self._eps(),
                num_iters=self.ppo.global_iters, seed=self.seed + 1000 * k,
                current_task=k, probe=self._probe, local_policy=self._experts[k - 1],
                ref_score=self._expert_scores[k - 1],
                intermediate_cache=(interm["games"][games[k - 1]] if interm else None),
                past_intermediate_caches=past_caches,
                past_expert_values=past_expert_values,
                monitor_every=self.ppo.past_monitor_every,
                eval_cache=eval_cache, past_eval_caches=past_eval_caches)
            self.logger.log({"phase": "gaps", "task": k, "V_k_ref_local": ref_current})
            self._log_retention(k, games)
            self._save_ckpt(k)

        torch.save(self.global_policy.state_dict(),
                   self.logger.run_dir / "final_policy.pt")

    def run(self) -> list[list[float]]:
        if self.method == "stored_expert":
            self._stored_expert()
            return self.eval_matrix
        if self.method == "consolidate":
            self._consolidate()
            return self.eval_matrix
        if self.method == "joint":
            self._train_joint()
            return self.eval_matrix
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
                    "constrained, finetune, clear, joint, consolidate, stored_expert"
                )
            row, stds = self._evaluate_row(k)
            self.eval_matrix.append(row)
            self.logger.log({"phase": "eval", "task": k, "values": row, "stds": stds})

        self.logger.save_json("eval_matrix.json", self.eval_matrix)
        torch.save(
            self.global_policy.state_dict(), self.logger.run_dir / "final_policy.pt"
        )
        return self.eval_matrix
