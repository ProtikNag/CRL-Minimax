"""Double-DQN realization of the min-max continual method.

The derivation (docs/Objective_for_Continual_Reinforcement_Learning.pdf) is a
constrained saddle problem on policy *values* V_i, solved by alternating a local
and a global policy. This module keeps that structure verbatim but replaces each
value-gradient step with one **Double-DQN gradient step** -- descending the
Bellman loss on task i's data is a policy-improvement operator that raises V_i,
so it stands in for grad V_i. Concretely:

* task 1   -- plain DDQN on task 1 into the global network phi.
* local    -- theta <- phi, then UNCONSTRAINED DDQN on the current task only
              (full plasticity: learn the new game as well as possible).
* global   -- minimize   sum_{i<k} omega_i * L_DDQN^i(phi)
                        + mu * 2 * s * L_DDQN^k(phi),
              where s = max(0, V_k^theta - V_k^phi) is the current-task
              shortfall (greedy-rollout values) and mu is dual-ascended on
              F_G = s^2 <= eps. This is eq 30's structure: push past-task value
              (the omega_i terms) while the mu term pulls the current task up to
              the local until the constraint is met. Past-task data is fresh
              interaction with the old environments (replay-free across tasks).

Target networks are ordinary DQN plumbing: each learner keeps its own lagged
copy, hard-synced every ``target_sync_every`` steps, used only to form the
Double-DQN bootstrap y = r + gamma * Q_target(s', argmax_a Q_online(s', a)).
They have no role in the constraint theory.

Rewards are scaled by each task's ``reward_scale`` for training and for the
value/constraint (so eps is balanced across games, as in the PG method); the
reported eval matrix is the RAW game score (unscaled greedy-rollout return).
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from crl.config import Config
from crl.duals import make_dual
from crl.envs.base import Task, TaskFamily
from crl.logging_utils import RunLogger
from crl.policies.base import Policy


# --------------------------------------------------------------------------- #
# replay buffer
# --------------------------------------------------------------------------- #
class ReplayBuffer:
    """Fixed-capacity ring buffer of transitions (one task)."""

    def __init__(self, capacity: int, obs_shape: tuple[int, int, int]) -> None:
        self.capacity = capacity
        self.obs = torch.zeros((capacity, *obs_shape), dtype=torch.float32)
        self.next_obs = torch.zeros((capacity, *obs_shape), dtype=torch.float32)
        self.actions = torch.zeros(capacity, dtype=torch.long)
        self.rewards = torch.zeros(capacity, dtype=torch.float32)
        self.dones = torch.zeros(capacity, dtype=torch.float32)
        self._pos = 0
        self._full = False

    def add(self, obs, action, reward, next_obs, done) -> None:
        i = self._pos
        self.obs[i] = obs
        self.next_obs[i] = next_obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = float(done)
        self._pos = (i + 1) % self.capacity
        self._full = self._full or self._pos == 0

    def __len__(self) -> int:
        return self.capacity if self._full else self._pos

    def sample(self, batch_size: int, device) -> dict[str, torch.Tensor]:
        n = len(self)
        idx = torch.randint(0, n, (batch_size,))
        return {
            "obs": self.obs[idx].to(device),
            "next_obs": self.next_obs[idx].to(device),
            "actions": self.actions[idx].to(device),
            "rewards": self.rewards[idx].to(device),
            "dones": self.dones[idx].to(device),
        }


# --------------------------------------------------------------------------- #
# environment pool (persistent parallel envs for one task)
# --------------------------------------------------------------------------- #
def _pad(state: np.ndarray, max_channels: int) -> torch.Tensor:
    chw = np.transpose(state.astype(np.float32), (2, 0, 1))
    if chw.shape[0] < max_channels:
        pad = np.zeros((max_channels - chw.shape[0], *chw.shape[1:]), dtype=np.float32)
        chw = np.concatenate([chw, pad], axis=0)
    return torch.from_numpy(chw)


class EnvPool:
    """``num_envs`` persistent MinAtar envs for one task; epsilon-greedy collect."""

    def __init__(self, task: Task, num_envs: int) -> None:
        from minatar import Environment
        self.task_id = task.spec.task_id
        self._scale = task._reward_scale
        self._maxc = task._max_channels
        self._make = lambda: Environment(
            task._game, sticky_action_prob=task._sticky,
            difficulty_ramping=task._ramping)
        base = int(torch.randint(0, 2**31 - 1, (1,)).item())
        self.envs = []
        for i in range(num_envs):
            e = self._make(); e.seed(base + i); e.reset()
            self.envs.append(e)
        self.states = [_pad(e.state(), self._maxc) for e in self.envs]

    @torch.no_grad()
    def collect(self, qnet: Policy, epsilon: float, num_transitions: int,
                buffer: ReplayBuffer, device) -> None:
        pushed = 0
        n = len(self.envs)
        while pushed < num_transitions:
            obs = torch.stack(self.states).to(device)
            q = qnet.q_values(obs, self.task_id)
            greedy = q.argmax(dim=-1).to("cpu")
            for i, env in enumerate(self.envs):
                if torch.rand(()) < epsilon:
                    a = int(torch.randint(0, q.shape[-1], (1,)).item())
                else:
                    a = int(greedy[i])
                reward, done = env.act(a)
                s = self.states[i]
                if done:
                    env.reset()
                s_next = _pad(env.state(), self._maxc)
                buffer.add(s, a, float(reward) * self._scale, s_next, done)
                self.states[i] = s_next
                pushed += 1


# --------------------------------------------------------------------------- #
# greedy value / raw score (constraint + reporting)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def greedy_value(qnet: Policy, task: Task, num_episodes: int, gamma: float,
                 device, epsilon: float = 0.0) -> tuple[float, float]:
    """Return (discounted scaled value V, undiscounted raw game score) for the
    greedy policy of ``qnet`` on ``task``, averaged over episodes.

    V (scaled, discounted) feeds the constraint; the raw score is the reported
    game performance. Batched: one Q forward per timestep over alive episodes.
    """
    from minatar import Environment
    task_id = task.spec.task_id
    scale = task._reward_scale
    maxc = task._max_channels
    base = int(torch.randint(0, 2**31 - 1, (1,)).item())
    envs = []
    for i in range(num_episodes):
        e = Environment(task._game, sticky_action_prob=task._sticky,
                        difficulty_ramping=task._ramping)
        e.seed(base + i); e.reset()
        envs.append(e)
    alive = [True] * num_episodes
    disc = np.zeros(num_episodes)
    raw = np.zeros(num_episodes)
    discount = np.ones(num_episodes)
    for _ in range(task._max_steps):
        active = [i for i in range(num_episodes) if alive[i]]
        if not active:
            break
        obs = torch.stack([_pad(envs[i].state(), maxc) for i in active]).to(device)
        q = qnet.q_values(obs, task_id)
        greedy = q.argmax(dim=-1).to("cpu")
        for j, i in enumerate(active):
            if epsilon > 0 and torch.rand(()) < epsilon:
                a = int(torch.randint(0, q.shape[-1], (1,)).item())
            else:
                a = int(greedy[j])
            reward, done = envs[i].act(a)
            disc[i] += discount[i] * reward * scale
            raw[i] += reward
            discount[i] *= gamma
            if done:
                alive[i] = False
    return float(disc.mean()), float(raw.mean())


# --------------------------------------------------------------------------- #
# Double-DQN loss
# --------------------------------------------------------------------------- #
def ddqn_loss(online: Policy, target: Policy, batch: dict, task_id: int,
              gamma: float) -> torch.Tensor:
    """Double-DQN Huber loss for one minibatch (action from online, value from
    target): y = r + gamma * (1 - done) * Q_target(s', argmax_a Q_online(s', a))."""
    q = online.q_values(batch["obs"], task_id).gather(
        1, batch["actions"].unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_act = online.q_values(batch["next_obs"], task_id).argmax(dim=1)
        next_q = target.q_values(batch["next_obs"], task_id).gather(
            1, next_act.unsqueeze(1)).squeeze(1)
        y = batch["rewards"] + gamma * (1.0 - batch["dones"]) * next_q
    return F.smooth_l1_loss(q, y)


def _sync(target: Policy, online: Policy) -> None:
    target.load_state_dict(online.state_dict())


def _epsilon(step: int, cfg) -> float:
    frac = min(1.0, step / max(1, cfg.eps_decay_steps))
    return cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)


# --------------------------------------------------------------------------- #
# continual trainer
# --------------------------------------------------------------------------- #
class DQNContinualTrainer:
    """Alternating local/global Double-DQN with the current-task constraint."""

    def __init__(self, config: Config, family: TaskFamily, global_qnet: Policy,
                 logger: RunLogger, device) -> None:
        self.cfg = config.dqn
        self.dual_cfg = config.duals
        self.family = family
        self.gamma = family.tasks[0].gamma
        self.q = global_qnet
        self.q_target = copy.deepcopy(global_qnet)
        self.logger = logger
        self.device = device
        n = len(family)
        self.omega = ([1.0 / n] * n if self.cfg.omega is None
                      else [float(w) for w in self.cfg.omega])
        self.mu_ctrl = make_dual(self.dual_cfg)
        self.buffers: dict[int, ReplayBuffer] = {}
        self.pools: dict[int, EnvPool] = {}
        self.eval_matrix: list[list[float]] = []
        self.grad_step = 0

    # -- infrastructure ----------------------------------------------------- #
    def _buffer(self, task: Task) -> ReplayBuffer:
        tid = task.spec.task_id
        if tid not in self.buffers:
            self.buffers[tid] = ReplayBuffer(self.cfg.buffer_capacity,
                                             self.family.obs_shape)
            self.pools[tid] = EnvPool(task, self.cfg.num_envs)
        return self.buffers[tid]

    def _opt(self, qnet: Policy) -> torch.optim.Optimizer:
        return torch.optim.Adam(qnet.parameters(), lr=self.cfg.lr)

    def _clip(self, qnet: Policy) -> None:
        if self.cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(qnet.parameters(), self.cfg.grad_clip)

    def _raw_scores(self, qnet: Policy, upto: int) -> list[float]:
        return [greedy_value(qnet, self.family.tasks[i], self.cfg.eval_episodes,
                             self.gamma, self.device)[1] for i in range(upto)]

    # -- phases ------------------------------------------------------------- #
    def _warmup(self, qnet: Policy, task: Task) -> None:
        buf = self._buffer(task)
        while len(buf) < self.cfg.warmup_transitions:
            self.pools[task.spec.task_id].collect(
                qnet, 1.0, self.cfg.num_envs * 4, buf, self.device)

    def _ddqn_unconstrained(self, qnet: Policy, task: Task, num_steps: int,
                            tag: str) -> None:
        """Standard DDQN training of ``qnet`` on a single task (task1 / local /
        expert). No constraint -- pure plasticity."""
        target = copy.deepcopy(qnet)
        opt = self._opt(qnet)
        buf = self._buffer(task)
        pool = self.pools[task.spec.task_id]
        self._warmup(qnet, task)
        for step in range(num_steps):
            eps = _epsilon(step, self.cfg)
            pool.collect(qnet, eps, self.cfg.collect_per_step, buf, self.device)
            batch = buf.sample(self.cfg.batch_size, self.device)
            loss = ddqn_loss(qnet, target, batch, task.spec.task_id, self.gamma)
            opt.zero_grad(); loss.backward(); self._clip(qnet); opt.step()
            if step % self.cfg.target_sync_every == 0:
                _sync(target, qnet)
            self.grad_step += 1
            if step % self.cfg.log_every == 0:
                v, score = greedy_value(qnet, task, self.cfg.value_episodes,
                                        self.gamma, self.device)
                self.logger.log({"phase": tag, "task": task.spec.task_id + 1,
                                 "step": step, "loss": float(loss), "eps": eps,
                                 "V": v, "score": score})
                print(f"[{tag} t{task.spec.task_id+1}] step={step:6d} "
                      f"loss={float(loss):.4f} eps={eps:.2f} "
                      f"V={v:.3f} score={score:.2f}")

    def _global_phase(self, k: int, cycle: int, local_qnet: Policy) -> None:
        """Constrained global update (eq 30 with DDQN losses)."""
        task_k = self.family.tasks[k - 1]
        past = list(range(k - 1))
        self.mu_ctrl.reset()
        target = copy.deepcopy(self.q)
        opt = self._opt(self.q)
        # ensure buffers/pools exist and are warm for current + past tasks
        for i in past + [k - 1]:
            self._warmup(self.q, self.family.tasks[i])
        v_local, _ = greedy_value(local_qnet, task_k, self.cfg.value_episodes,
                                  self.gamma, self.device)
        mu = self.mu_ctrl.value
        shortfall = 0.0
        for step in range(self.cfg.global_steps):
            eps = _epsilon(step, self.cfg)
            # sample the active past task(s) for this step
            if self.cfg.past_task_sampling == "sample" and len(past) > 1:
                active = [past[int(torch.randint(len(past), (1,)).item())]]
                scale = float(len(past))
            else:
                active = list(past)
                scale = 1.0

            loss = torch.zeros((), device=self.device)
            past_losses = {}
            for i in active:
                task_i = self.family.tasks[i]
                self.pools[i].collect(self.q, eps, self.cfg.collect_per_step,
                                      self.buffers[i], self.device)
                batch = self.buffers[i].sample(self.cfg.batch_size, self.device)
                li = ddqn_loss(self.q, target, batch, i, self.gamma)
                loss = loss + self.omega[i] * scale * li
                past_losses[f"L_past_{i}"] = float(li)

            # current-task term, weighted by mu * 2 * shortfall
            self.pools[k - 1].collect(self.q, eps, self.cfg.collect_per_step,
                                      self.buffers[k - 1], self.device)
            batch_k = self.buffers[k - 1].sample(self.cfg.batch_size, self.device)
            lk = ddqn_loss(self.q, target, batch_k, k - 1, self.gamma)
            coeff_k = mu * 2.0 * shortfall
            loss = loss + coeff_k * lk

            opt.zero_grad(); loss.backward(); self._clip(self.q); opt.step()
            if step % self.cfg.target_sync_every == 0:
                _sync(target, self.q)
            self.grad_step += 1

            # periodically re-measure the shortfall and update mu (dual ascent)
            if step % self.cfg.value_every == 0:
                v_global, score_k = greedy_value(self.q, task_k,
                                                 self.cfg.value_episodes,
                                                 self.gamma, self.device)
                shortfall = max(0.0, v_local - v_global)
                constraint = shortfall * shortfall
                mu = self.mu_ctrl.update(constraint, self.cfg.eps_constraint)
                if step % self.cfg.log_every == 0:
                    self.logger.log({"phase": "global", "task": k, "cycle": cycle,
                                     "step": step, "loss": float(loss),
                                     "F_G": constraint, "mu": mu, "shortfall": shortfall,
                                     "V_k_local": v_local, "V_k_global": v_global,
                                     "score_k": score_k, **past_losses})
                    print(f"[global k={k} c={cycle}] step={step:6d} "
                          f"loss={float(loss):.4f} s={shortfall:.3f} "
                          f"mu={mu:.2f} F_G={constraint:.4f} "
                          f"Vk_l={v_local:.3f} Vk_g={v_global:.3f}")

    # -- main loop ---------------------------------------------------------- #
    def run(self) -> list[list[float]]:
        print("=== task 1: plain DDQN into the global network ===")
        self._ddqn_unconstrained(self.q, self.family.tasks[0],
                                 self.cfg.task1_steps, tag="task1")
        self.eval_matrix.append(self._raw_scores(self.q, 1))
        self.logger.log({"phase": "eval", "task": 1, "values": self.eval_matrix[-1]})
        print(f"[eval t1] {self.eval_matrix[-1]}")

        for k in range(2, len(self.family) + 1):
            for cycle in range(self.cfg.cycles_per_task):
                print(f"\n=== task {k} cycle {cycle}: local (unconstrained) ===")
                local = copy.deepcopy(self.q)          # theta <- phi (eq 2)
                for p in local.parameters():
                    p.requires_grad_(True)
                self._ddqn_unconstrained(local, self.family.tasks[k - 1],
                                         self.cfg.local_steps, tag="local")
                local.eval()
                print(f"=== task {k} cycle {cycle}: global (constrained) ===")
                self._global_phase(k, cycle, local)
            self.eval_matrix.append(self._raw_scores(self.q, k))
            self.logger.log({"phase": "eval", "task": k,
                             "values": self.eval_matrix[-1]})
            print(f"[eval t{k}] {self.eval_matrix[-1]}")

        self.logger.save_json("eval_matrix.json", self.eval_matrix)
        torch.save(self.q.state_dict(), self.logger.run_dir / "final_policy.pt")
        return self.eval_matrix
