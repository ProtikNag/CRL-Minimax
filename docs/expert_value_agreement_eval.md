# Expert Value-Agreement Evaluation (windowed, off-policy)

Status: design locked 2026-08-18 (`feature/updated-objective`). This document is
the authoritative description of the evaluation protocol and the engineering
choices behind it, written so the choices can be reported directly in the paper.

---

## 1. Purpose and scope

We want a **cheap, discriminative** measure of whether a consolidated global
policy `π_φ` on a past task `i` **achieves the same value as that task's expert
`π_i*`** — i.e. *"can the new model play as well as the expert?"* — without paying
for many full-episode rollouts, and without letting easy/near-random early-game
play inflate the score.

This metric measures **value agreement on the expert's own state distribution**.
It deliberately does **not** measure:

- **On-policy forgetting / deployment competence.** A policy can match the expert
  from expert-visited states yet still fail on its own because it never *reaches*
  those states (the *reachability gap*). Therefore this metric is **complementary
  to**, not a replacement for, the on-policy anchor (§6).
- **Behavioural imitation.** Equal windowed value does not imply the same action
  distribution (two action sequences can score the same). We report value only;
  behaviour agreement is out of scope by choice (see §3, decision D2).

This is the evaluation-side analogue of the doc `Updated_Objective_for_CRL.pdf`
Part B state-set idea (`D_i`, eqs 50–51) and its Q6 caveat, applied to *reporting*
rather than to the training objective.

---

## 2. Definitions

For a task `i` with expert `π_i*` and discount `γ`, let `s` be a state the expert
visits, and let `H_i` be the game-specific evaluation horizon (§3, D4).

- **Expert windowed value** `V*_win(s)`: the expert's realized discounted return
  over the next `min(H_i, steps-to-terminal)` agent-steps from `s`, recorded from
  the expert's own trajectory (free — no extra rollout).
- **Policy windowed value** `Vπ_win(s)`: the discounted return obtained by
  restoring the emulator to `s` and rolling **the policy** greedily for the same
  window (capped at terminal).
- **Per-state one-sided shortfall** (decision D5):
  `sf(s) = max(0, V*_win(s) − Vπ_win(s))` — underperformance only; beating the
  expert contributes 0.
- **Per-state relative shortfall**: `sf_rel(s) = sf(s) / V*_win(s)` (safe because
  the consequential filter, D3, guarantees `V*_win(s)` is bounded away from 0).

**Task metric** (per task `i`, per checkpoint):
```
Agreement gap_i     = mean over kept states s of sf(s)           # raw units
Relative gap_i      = mean over kept states s of sf_rel(s)       # in [0, 1], cross-game comparable
```
`Relative gap_i = 0` ⇒ the policy matches or beats the expert on every kept state;
`= 1` ⇒ it captures none of the expert's windowed value. Lower is better.

Both the discounted value (for `V`) and the raw windowed score are recorded so the
metric can be read on either scale; the discounted-clipped value is primary
(matches the training value semantics), raw score is secondary/reporting.

---

## 3. Design decisions and rationale (paper-facing)

**D1 — Off-policy, expert-state evaluation (deliberate).** States are drawn from
the *expert's* occupancy, not the evaluated policy's. This answers the intended
question ("can it play like the expert?") and lets the state set be **computed
once per task and reused across every checkpoint and method** (the dominant cost
saving). The known consequence — it can look good while failing on-policy — is
handled by pairing it with the on-policy anchor (§6), never by using it alone.

**D2 — Value, not behaviour.** We compare returns, not action distributions. This
is a scope choice: we care about *outcome parity with the expert*. (If behavioural
absorption is later needed, add action-agreement / KL along expert rollouts per
Part-B Q6; the state set already supports it.)

**D3 — Consequential-window filter.** Over a short horizon most Atari windows
contain no reward, so a naive average is dominated by vacuous `0 ≈ 0` "perfect
agreement". We therefore **keep only consequential states**: the expert's RAW
window score `≥ consequential_thresh` (it scored non-trivially) **and** its
discounted sign-clipped value `V*_win ≥ v_floor > 0` (so the relative-shortfall
denominator is strictly bounded away from 0 — a raw-positive window can still be
net-negative in discounted-clipped value, e.g. `+1` early then several `−1`). This
removes the vacuity and focuses on the deciding states. **Disclosed selection
bias:** the filter conditions on the *expert's* reward (an oracle quantity), so it
can *undercount* a policy that achieves the same episode return by scoring at
*different* states (this interacts with D2: value ≠ behaviour). The net direction
is game-dependent — report the kept-state count and, when in doubt, an unfiltered
aggregate alongside.

**D4 — Per-game adaptive horizon `H_i` (not per-game frame-skip).** Sparse games
need a longer window to contain a scoring event; dense games a shorter one. We
adapt the **window length**, chosen from a candidate grid
`G = {15, 20, 25, 30, 40, 60}` agent-steps: `H_i` is the smallest `H ∈ G` for
which at least a fraction `τ = 0.5` of sampled expert windows are consequential
(else the largest `H`, with a logged warning). We do **not** change the agent's
frame-skip at eval time: the policy was trained at `frame_skip = 4`, and altering
its action cadence would evaluate it under a *different MDP* than it was trained
on, biasing the result for reasons unrelated to retention.

**D5 — One-sided shortfall (not symmetric).** The expert is the ceiling; we care
whether the policy *reaches* it, not whether it stays exactly on it — exceeding
the expert on a state is fine and must not be penalised. One-sided also matches
the training hinge `[V_k^L − V_k^G]_+` (eq 26), so evaluation and objective use the
same notion of "shortfall".

**D6 — Efficient state restore via emulator snapshots.** States are captured with
ALE `cloneState` **during the expert rollout** (O(1) each; the `ALEState` is
picklable, so the set serialises to disk) and restored with `restoreState` (O(1)),
rather than by replaying a seed+action prefix (which for a near-terminal state
costs an entire episode). The policy's 4-frame stack — which lives in
`FrameStackObservation`, not in the ALE core — is snapshotted as the stored stacked
observation and **re-managed by the evaluator** on restore; the `AtariPreprocessing`
max-pool buffer needs no snapshot because each agent step fully overwrites it.
Validated: post-`restoreState` observations and rewards reproduce bit-for-bit — but
this holds **only at `repeat_action_probability = 0`**, which `make_snapshot_env`
now **asserts** (sticky actions would make restored rollouts stochastic and break
the comparison).

---

## 4. Algorithm

### 4a. Precompute the expert state set (once per task, reusable)
```
build_expert_state_set(expert π_i*, task i, γ, grid G, τ=0.5,
                       n_source_starts, states_per_task N_i):
  roll π_i* greedily from several distinct no-op starts -> expert trajectories
  # each trajectory records, per agent-step t: ale_state = cloneState() (snapshot
  #   BEFORE stepping), stacked_obs (4,84,84 uint8), reward_t, steps_to_terminal
  choose H_i = smallest H in G with frac(consequential windows) >= tau (else max G)
  for sampled states s across the trajectories (mix of near-terminal / mid / noop):
      n_win(s)  = min(H_i, steps_to_term)
      V*_win(s) = discounted sum of expert rewards over n_win(s) steps
      keep s iff CONSEQUENTIAL: raw_win(s) >= consequential_thresh AND V*_win(s) >= v_floor
  store {ale_state, stacked_obs, V*_win, raw_win, n_win, steps_to_term, category}, H_i
  pickle to expert_states/<game>.pkl  (ALEState is picklable)
```

### 4b. Evaluate a policy against the set (per checkpoint, per task)
```
evaluate_expert_agreement(policy π_φ, task i, expert_set):
  build a preprocessing-only eval env (ALE + AtariPreprocessing; no frame stack,
      no TimeLimit, unclipped rewards, terminal_on_life_loss=False)
  for each kept state s (vectorized in banks of n_envs):
      restoreState(ale, stored ale_state)         # O(1) restore
      init a 4-frame stack from stored stacked_obs
      roll π_φ GREEDILY for n_win(s) = min(H_i, steps_to_term) steps (cap at terminal),
          managing the stack, so Vπ_win(s) covers the SAME window length as V*_win(s),
          on the same (discounted, sign-clipped) scale
      sf(s)     = max(0, V*_win(s) - Vπ_win(s))
      sf_rel(s) = sf(s) / V*_win(s)               # V*_win(s) >= v_floor > 0 by the filter
  return mean sf, mean sf_rel  (+ per-category breakdown)
```

Cost: expert set amortised across checkpoints; each eval is
`|kept states| × H_i` agent-steps (≈ a few full episodes' worth).

---

## 5. Hyperparameters (defaults)

| name | default | meaning |
|---|---|---|
| `horizon_grid` | `{15,20,25,30,40,60}` | candidate window lengths (agent-steps) |
| `consequential_frac τ` | `0.5` | min fraction of consequential windows to accept `H_i` |
| `consequential_thresh` | `1.0` (raw) | min expert RAW window score to keep a state |
| `v_floor` | `0.1` | min expert discounted-clipped `V*_win` to keep (denominator safety) |
| `states_per_task N_i` | `5000` | kept expert states evaluated per task |
| `n_source_starts` | `20` | distinct expert no-op starts generating source trajectories |
| `noop_anchor_rollouts` | `10` | on-policy full greedy rollouts kept alongside (§6) |
| eval action selection | greedy (argmax) | matches the project greedy-eval rule |

---

## 6. Limitations and the mandatory on-policy anchor

1. **Reachability gap.** This metric conditions on expert-reached states; it says
   nothing about whether `π_φ` can get there on its own. **Always report it
   alongside the on-policy anchor: 10 no-op full greedy rollouts** (the deployment
   / forgetting number). Never substitute one for the other.
2. **Value ≠ behaviour** (D2): equal value can hide different behaviour.
3. **Windowed, not full value.** For mid-episode states the horizon truncates the
   tail; two policies matching on the window can diverge afterward. Mitigated but
   not eliminated by the consequential filter and adaptive `H_i`.
4. **Off-distribution by construction** (D1): optimistic about deployment; that is
   the intended trade for cheap, expert-anchored value agreement.
5. **Cross-game comparability.** Because `H_i` and the reward scale differ per game,
   the *raw* `agreement_gap` is **not** comparable across tasks; only
   `relative_gap ∈ [0,1]` is. `H_i` is frozen inside the `ExpertStateSet` and reused
   for every checkpoint/method — the same measuring stick — and must **never** be
   recomputed per policy.
6. **Selection bias (D3).** Conditioning on the expert's scoring windows can
   undercount a policy that scores at different states; disclose it, and report an
   unfiltered aggregate if in doubt.
7. **Sticky actions off.** The whole comparison requires `repeat_action_probability
   = 0` (asserted in `make_snapshot_env`).

The metric is a **diagnostic of expert-value parity**, reported *with* the
on-policy anchor — not the standalone retention/forgetting number.
