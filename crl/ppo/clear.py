"""CLEAR baseline (Rolnick et al. 2019; CORA's rehearsal baseline) on PPO.

CLEAR is the recent replay-based SOTA continual-RL baseline used by CORA (Powers
et al., CoLLAs 2022) on sequential Atari. Its anti-forgetting mechanism is:

* a **replay buffer** of past-task experience,
* **behavioral cloning** on replay: KL(behavior_stored || current policy),
* **value cloning** on replay: MSE(current value, stored value),
* a 50-50 novel/replay batch ratio.

Original CLEAR is off-policy (IMPALA + V-trace). We port it to the same PPO
backbone the min-max method uses (exactly as REINFORCE->PPO was ported): the
*novel* stream is standard clipped PPO on fresh current-task rollouts; the
V-trace off-policy correction is replaced by PPO clipping. The cloning losses are
CORA's Atari values (policy_clone=0.01, value_clone=0.005). This isolates the
anti-forgetting *mechanism* (replay + cloning) on an identical architecture,
budget, and evaluation protocol -- a fair comparison to the replay-free min-max
constraint and to naive fine-tuning.

Replay is stored as end-of-task snapshots of the trained policy's behavior
(obs, logits, value) per task, which are the cloning targets while later tasks
train.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical, kl_divergence

from crl.policies.base import Policy
from crl.ppo.collector import RolloutCollector
from crl.ppo.trainer import PPOTrainer


class ReplayStore:
    """Per-task snapshots of (obs, behavior logits, value) held on CPU."""

    def __init__(self) -> None:
        self._obs: dict[int, torch.Tensor] = {}
        self._logits: dict[int, torch.Tensor] = {}
        self._values: dict[int, torch.Tensor] = {}

    def add(self, task_id: int, obs, logits, values) -> None:
        obs = obs.to("cpu"); logits = logits.to("cpu"); values = values.to("cpu")
        if task_id in self._obs:
            self._obs[task_id] = torch.cat([self._obs[task_id], obs])
            self._logits[task_id] = torch.cat([self._logits[task_id], logits])
            self._values[task_id] = torch.cat([self._values[task_id], values])
        else:
            self._obs[task_id] = obs
            self._logits[task_id] = logits
            self._values[task_id] = values

    def task_ids(self) -> list[int]:
        return sorted(self._obs)

    def sample(self, task_id: int, n: int, device):
        m = self._obs[task_id].shape[0]
        idx = torch.randint(0, m, (min(n, m),))
        return (self._obs[task_id][idx].to(device),
                self._logits[task_id][idx].to(device),
                self._values[task_id][idx].to(device))


class ClearTrainer(PPOTrainer):
    """PPO on the current task + CLEAR replay/behavioral/value cloning on the past."""

    def _optimize_clear(self, policy, optimizer, batch, replay, past_ids):
        cfg = self.ppo
        n = batch.obs.shape[0]
        mb = max(1, n // cfg.num_minibatches)
        pg_acc = v_acc = clone_p_acc = clone_v_acc = 0.0
        count = 0
        for _ in range(cfg.ppo_epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, mb):
                idx = perm[start:start + mb]
                # --- novel stream: standard clipped PPO on the current task -----
                dist, value = policy.dist_value(batch.obs[idx], batch.task_id)
                logratio = dist.log_prob(batch.actions[idx]) - batch.logprobs[idx]
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
                loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy

                # --- replay stream: behavioral + value cloning (50-50 ratio) ----
                clone_p = torch.zeros((), device=self.device)
                clone_v = torch.zeros((), device=self.device)
                if past_ids:
                    picks = past_ids if cfg.clear_replay_task_per_step >= len(past_ids) \
                        else [past_ids[int(torch.randint(len(past_ids), (1,)))]
                              for _ in range(cfg.clear_replay_task_per_step)]
                    for tid in picks:
                        r_obs, r_logits, r_val = replay.sample(tid, mb, self.device)
                        r_dist, r_value = policy.dist_value(r_obs, tid)
                        behavior = Categorical(logits=r_logits)
                        clone_p = clone_p + kl_divergence(behavior, r_dist).mean()
                        clone_v = clone_v + 0.5 * (r_value - r_val).pow(2).mean()
                    clone_p = clone_p / len(picks)
                    clone_v = clone_v / len(picks)
                    loss = loss + cfg.clear_policy_clone_cost * clone_p \
                        + cfg.clear_value_clone_cost * clone_v

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                optimizer.step()
                with torch.no_grad():
                    pg_acc += float(pg_loss); v_acc += float(v_loss)
                    clone_p_acc += float(clone_p); clone_v_acc += float(clone_v)
                    count += 1
        c = max(1, count)
        return {"pg_loss": pg_acc / c, "v_loss": v_acc / c,
                "clone_policy": clone_p_acc / c, "clone_value": clone_v_acc / c}

    @torch.no_grad()
    def snapshot(self, policy, task, replay: ReplayStore) -> None:
        """Store end-of-task behavior (obs, logits, value) as cloning targets."""
        cfg = self.ppo
        coll = RolloutCollector(task, cfg.n_envs, cfg.n_steps, self.device,
                                seed=cfg.eval_seed + task.spec.task_id)
        try:
            for _ in range(cfg.clear_snapshot_batches):
                b = coll.collect(policy, cfg.gae_lambda)
                dist, value = policy.dist_value(b.obs, b.task_id)
                replay.add(task.spec.task_id, b.obs, dist.logits, value)
        finally:
            coll.close()

    def train(self, policy: Policy, task, replay: ReplayStore, num_iters: int,
              seed: int, current_task: int, probe=None) -> None:
        optimizer = self._new_optimizer(policy)
        collector = RolloutCollector(task, self.ppo.n_envs, self.ppo.n_steps,
                                     self.device, seed)
        past_ids = replay.task_ids()
        thr = float(getattr(task, "threshold", float("inf")))
        met = 0
        try:
            for it in range(num_iters):
                batch = collector.collect(policy, self.ppo.gae_lambda)
                stats = self._optimize_clear(policy, optimizer, batch, replay, past_ids)
                if probe is not None:
                    probe("clear", current_task)
                gscore = self._stop_score(policy, task, it + 1)
                if gscore is not None:
                    met = met + 1 if gscore >= thr else 0
                if it % self.log_every == 0 or gscore is not None:
                    self.logger.log({"phase": "clear", "task": current_task,
                                     "step": it, "greedy_score": gscore, **stats})
                    gs = f" greedy={gscore:.1f}/{thr:.0f}" if gscore is not None else ""
                    print(f"[clear k={current_task}] it={it:4d} pg={stats['pg_loss']:.3f}"
                          f" clone_p={stats['clone_policy']:.4f}"
                          f" clone_v={stats['clone_value']:.3f}{gs}")
                if met >= self.ppo.patience:
                    print(f"[clear k={current_task}] EARLY STOP it={it+1} "
                          f"greedy={gscore:.1f} >= {thr:.0f}")
                    break
        finally:
            collector.close()
