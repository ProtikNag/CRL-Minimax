"""Monte-Carlo evaluation for the PPO backend.

Two distinct quantities, from full-episode rollouts:

* ``value``  -- mean discounted return on the *training* reward scale (sign-clipped
                iff the task clips rewards). This is V_i^pi used for the shortfall
                F-hat, the frozen references, and the constraint. It is measured on
                the *stochastic* policy (sampled actions), faithful to the theory's
                V^pi. Both come from one rollout on an unclipped env; the clipped
                value is re-derived with ``sign()`` to avoid info-plumbing.
* ``score``  -- mean *raw* (unclipped) episode return: the reported game score.

For REPORTING (eval matrix, probes, early-stopping thresholds) we use
``greedy=True`` (argmax actions), ``num_episodes`` large (e.g. 50), and a FIXED
``seed`` so the measurement is low-variance and reproducible across methods,
seeds and checkpoints. For the CONSTRAINT value we use ``greedy=False`` (the
on-policy stochastic value). ``std`` (of the raw score across episodes) is
returned for error bars.
"""

from __future__ import annotations

import numpy as np
import torch

from crl.envs.base import Task
from crl.policies.base import Policy


@torch.no_grad()
def evaluate_greedy_noop_enumerated(
    policy: Policy,
    task: Task,
    n_envs: int,
    device: torch.device,
    seed: int | None = None,
    max_ep_steps: int = 0,
) -> tuple[float, float, float, int]:
    """EXACT greedy value/score by ENUMERATING every no-op start.

    Greedy actions on ``repeat_action_probability=0`` Atari are deterministic
    given the initial no-op count, so the expected greedy return over the
    standard random-start distribution (1..noop_max no-ops) is *exactly* the mean
    of one rollout per no-op count -- no sampling, no variance. Runs those
    ``noop_max`` rollouts in parallel across ``n_envs`` envs (built with
    ``noop_override=0`` so we apply the no-op prefix ourselves), accumulating the
    discounted value only AFTER the no-ops so the semantics match training
    (episode starts post-no-op). Returns ``(mean_value, mean_score, score_std, n)``.
    """
    import numpy as np
    noop_max = int(getattr(task, "_noop_max", 0))
    if noop_max <= 0:
        return evaluate_value_and_score(policy, task, 100, n_envs, device,
                                        seed=seed, greedy=True)
    clip = bool(getattr(task, "clip_rewards", False))
    gamma = task.gamma
    tid = task.spec.task_id
    counts = list(range(1, noop_max + 1))          # the noop_max distinct starts
    # Per-episode TimeLimit on the EVAL env only (safe: no GAE here). Discounted V
    # is unchanged (gamma kills late steps); bounds cost on long-episode experts.
    venv = task.make_vector_env(n_envs, clip_rewards=False, noop_override=0,
                                max_steps_override=(max_ep_steps or None))
    try:
        obs, _ = venv.reset(seed=seed)
        obs = torch.as_tensor(obs, device=device)
        queue = list(counts)
        assigned = [-1] * n_envs           # noop count this env is currently evaluating
        noops_left = [0] * n_envs
        disc = np.zeros(n_envs); gpow = np.ones(n_envs); raw = np.zeros(n_envs)
        for j in range(n_envs):
            if queue:
                assigned[j] = queue.pop(0); noops_left[j] = assigned[j]
        res_disc: dict[int, float] = {}; res_raw: dict[int, float] = {}
        steps = 0
        cap = noop_max * 200_000 // max(1, n_envs) + 10_000
        while any(a != -1 for a in assigned) and steps < cap:
            steps += 1
            g_act = policy.dist(obs, tid).logits.argmax(dim=-1).to("cpu").numpy()
            actions = np.zeros(n_envs, dtype=np.int64)
            for j in range(n_envs):
                if assigned[j] != -1 and noops_left[j] == 0:
                    actions[j] = g_act[j]        # else NOOP (0): inactive or no-op prefix
            obs_np, reward, term, trunc, _ = venv.step(actions)
            done = term | trunc
            for j in range(n_envs):
                if assigned[j] == -1:
                    continue
                if noops_left[j] > 0:            # still applying the no-op prefix
                    noops_left[j] -= 1
                    continue
                r = float(reward[j])
                disc[j] += gpow[j] * (float(np.sign(r)) if clip else r)
                gpow[j] *= gamma; raw[j] += r
                if bool(done[j]):
                    res_disc[assigned[j]] = disc[j]; res_raw[assigned[j]] = raw[j]
                    disc[j] = 0.0; gpow[j] = 1.0; raw[j] = 0.0
                    if queue:                   # SAME_STEP autoreset already gave fresh obs
                        assigned[j] = queue.pop(0); noops_left[j] = assigned[j]
                    else:
                        assigned[j] = -1
            obs = torch.as_tensor(obs_np, device=device)
    finally:
        venv.close()
    if not res_raw:
        return 0.0, 0.0, 0.0, 0
    vals = np.array([res_disc[c] for c in sorted(res_disc)])
    scs = np.array([res_raw[c] for c in sorted(res_raw)])
    return float(vals.mean()), float(scs.mean()), float(scs.std()), len(scs)


@torch.no_grad()
def evaluate_intermediate_values(
    policy: Policy,
    task: Task,
    task_id: int,
    cache_game: dict,
    device: torch.device,
    max_ep_steps: int = 3000,
) -> tuple[list[float], list[float]]:
    """Per-state true rollout return of ``policy`` from cached intermediate states.

    Each cached start is reached by EXACT deterministic re-simulation: reset with
    the trajectory's seed and replay the expert's recorded actions ``0..t-1`` to
    land on ``s_t`` (frame-stack buffers correctly primed), then roll ``policy``
    greedily to the episode end for its discounted return-to-go. Returns
    ``(v_globals, v_experts)`` aligned to the kept ``cache_game['starts']`` -- a
    start whose re-simulation ends during the prefix is dropped from both lists.
    """
    gamma = task.gamma
    clip = bool(getattr(task, "clip_rewards", False))
    trajs = cache_game["trajs"]
    v_g: list[float] = []
    v_e: list[float] = []
    for st in cache_game["starts"]:
        tr = trajs[st["traj"]]
        acts = tr["actions"]
        t = int(st["t"])
        env = task.make_env(clip_rewards=True)
        obs, _ = env.reset(seed=int(tr["seed"]))
        broke = False
        for a in acts[:t]:                       # deterministic re-simulation to s_t
            obs, _, term, trunc, _ = env.step(int(a))
            if term or trunc:
                broke = True
                break
        if broke:
            env.close()
            continue
        disc = 0.0; g = 1.0
        for _ in range(max(1, max_ep_steps - t)):
            ot = torch.as_tensor(np.asarray(obs), device=device).unsqueeze(0)
            act = int(policy.dist(ot, task_id).logits.argmax(-1).item())
            obs, r, term, trunc, _ = env.step(act)
            rr = float(np.sign(r)) if clip else float(r)
            disc += g * rr; g *= gamma
            if term or trunc:
                break
        env.close()
        v_g.append(disc)
        v_e.append(float(st["expert_rtg"]))
    return v_g, v_e


@torch.no_grad()
def evaluate_intermediate_values_vec(
    policy: Policy,
    task: Task,
    task_id: int,
    cache_game: dict,
    device: torch.device,
    n_envs: int = 16,
    max_ep_steps: int = 3000,
) -> tuple[list[float], list[float]]:
    """Vectorized ``evaluate_intermediate_values``: identical semantics, but runs
    the intermediate-state rollouts in banks of ``n_envs`` with BATCHED forward
    passes instead of one state at a time.

    Per-env seeding (a list of the trajectory seeds) makes each sub-env reach the
    same deterministic state, and one unified loop drives every env: while
    ``s < t_i`` the env replays its recorded prefix action (return not
    accumulated); at ``s >= t_i`` it takes the global's greedy action and
    accumulates the discounted reward until ``done``. Because prefix+rollout is
    capped at ``max_ep_steps``, all envs finish by step ``max_ep_steps``. Returns
    ``(v_globals, v_experts)`` for the kept starts (a state whose prefix hits
    ``done`` is dropped, matching the scalar version).
    """
    gamma = task.gamma
    clip = bool(getattr(task, "clip_rewards", False))
    trajs = cache_game["trajs"]
    starts = cache_game["starts"]
    v_g: list[float] = []
    v_e: list[float] = []
    for c0 in range(0, len(starts), n_envs):
        chunk = starts[c0:c0 + n_envs]
        B = len(chunk)
        seeds = [int(trajs[st["traj"]]["seed"]) for st in chunk]
        prefixes = [trajs[st["traj"]]["actions"] for st in chunk]
        tlens = [int(st["t"]) for st in chunk]
        venv = task.make_vector_env(B, clip_rewards=True)
        try:
            obs, _ = venv.reset(seed=seeds)          # per-env seeds -> same s_t
            obs = np.asarray(obs)
            disc = np.zeros(B); gpow = np.ones(B)
            finished = np.zeros(B, dtype=bool); invalid = np.zeros(B, dtype=bool)
            result = np.zeros(B)
            for s in range(max_ep_steps):
                if (finished | invalid).all():
                    break
                gact = policy.dist(torch.as_tensor(obs, device=device),
                                   task_id).logits.argmax(-1).cpu().numpy()
                actions = np.zeros(B, dtype=np.int64)
                in_prefix = np.zeros(B, dtype=bool)
                for i in range(B):
                    if finished[i] or invalid[i]:
                        continue
                    if s < tlens[i]:
                        actions[i] = prefixes[i][s]; in_prefix[i] = True
                    else:
                        actions[i] = gact[i]
                obs_np, reward, term, trunc, _ = venv.step(actions)
                done = term | trunc
                for i in range(B):
                    if finished[i] or invalid[i]:
                        continue
                    if in_prefix[i]:
                        if bool(done[i]):            # ended during prefix -> drop
                            invalid[i] = True
                        continue
                    r = float(reward[i])
                    disc[i] += gpow[i] * (float(np.sign(r)) if clip else r)
                    gpow[i] *= gamma
                    if bool(done[i]):
                        result[i] = disc[i]; finished[i] = True
                obs = np.asarray(obs_np)
            for i in range(B):
                if invalid[i]:
                    continue
                v_g.append(float(result[i] if finished[i] else disc[i]))
                v_e.append(float(chunk[i]["expert_rtg"]))
        finally:
            venv.close()
    return v_g, v_e


@torch.no_grad()
def evaluate_value_and_score(
    policy: Policy,
    task: Task,
    num_episodes: int,
    n_envs: int,
    device: torch.device,
    seed: int | None = None,
    greedy: bool = False,
    max_env_steps: int = 200_000,
    noop_enumerate: bool = False,
    max_ep_steps: int = 0,
) -> tuple[float, float, float, int]:
    if noop_enumerate:
        return evaluate_greedy_noop_enumerated(policy, task, n_envs, device,
                                               seed=seed, max_ep_steps=max_ep_steps)
    """Return ``(mean_value, mean_score, score_std, n_episodes_used)``.

    ``mean_value`` = discounted return on the training reward scale (stochastic
    policy semantics regardless of ``greedy`` for the clipped-value accounting).
    ``mean_score`` / ``score_std`` = mean and std of the raw undiscounted episode
    return (game score). ``greedy`` selects argmax actions (for reporting);
    otherwise actions are sampled. A fixed ``seed`` makes the rollout reproducible.
    """
    clip = bool(getattr(task, "clip_rewards", False))
    gamma = task.gamma
    tid = task.spec.task_id
    # With no per-episode TimeLimit, strong agents can have long episodes; give the
    # eval enough total steps to actually finish `num_episodes` (scales with the
    # request, ~30k agent-steps/episode budget), keeping the fixed floor.
    max_env_steps = max(max_env_steps, num_episodes * 30_000 // max(1, n_envs))
    venv = task.make_vector_env(n_envs, clip_rewards=False)  # raw rewards
    try:
        obs, _ = venv.reset(seed=seed)
        obs = torch.as_tensor(obs, device=device)
        disc = torch.zeros(n_envs)  # running discounted (clipped-scale) return
        gpow = torch.ones(n_envs)  # gamma^t per env
        raw = torch.zeros(n_envs)  # running raw return
        values: list[float] = []
        scores: list[float] = []

        steps = 0
        while len(scores) < num_episodes and steps < max_env_steps:
            steps += 1
            logits = policy.dist(obs, tid).logits
            if greedy:
                action = logits.argmax(dim=-1)
            else:
                action = torch.distributions.Categorical(logits=logits).sample()
            action = action.to("cpu").numpy()
            obs_np, reward, term, trunc, _ = venv.step(action)
            reward_t = torch.as_tensor(reward, dtype=torch.float32)
            train_r = torch.sign(reward_t) if clip else reward_t
            disc += gpow * train_r
            raw += reward_t
            gpow *= gamma
            done = term | trunc
            for i in range(n_envs):
                if bool(done[i]):
                    values.append(float(disc[i]))
                    scores.append(float(raw[i]))
                    disc[i] = 0.0
                    raw[i] = 0.0
                    gpow[i] = 1.0
            obs = torch.as_tensor(obs_np, device=device)
    finally:
        venv.close()

    if not scores:  # no episode finished within the step budget
        return 0.0, 0.0, 0.0, 0
    n = min(len(scores), num_episodes)
    sc = torch.tensor(scores[:n])
    mean_value = sum(values[:n]) / n
    return mean_value, float(sc.mean()), float(sc.std(unbiased=False)), n
