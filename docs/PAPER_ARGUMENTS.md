# Standing arguments for the paper

Rebuttals and framings agreed on 2026-09-16. When a draft exists, check it
against this file. Every item here is a claim the paper must either make or
deliberately decline to make.

---

## 1. The spine

> Standard continual-RL evaluation suppresses the interference it claims to
> measure. Remove either of the two mechanisms doing the suppressing and the
> field's ranking inverts.

The paper is **not** "our method is SOTA". Under that framing, being third on
the CKA-RL benchmark is a liability. Under the spine above, it is the first
piece of evidence.

**The single strongest number.** On the CKA-RL benchmark ours and naive
fine-tuning differ by **0.0011** (0.8670 vs FT-N 0.8659). Remove the per-task
heads and the gap is **0.335** (0.596 vs 0.261).

## 2. The two crutches, as a factorial

| Tier | Head | Task relatedness |
|---|---|---|
| CKA-RL benchmark | per-task | modes of one game |
| GridWorld, 50 tasks | **shared** | related layouts |
| Atari, 5 games | per-task | **different games** |

Each of our tiers removes exactly one property and holds the other fixed. This
is a designed ablation of the benchmark, not "we made it harder". Either crutch
alone is sufficient to break the tier-1 ranking, which is far harder to dismiss
than one broken benchmark.

## 3. The inversion

| Method | Tier 1 rank | Tier 1 PERF | Tier 2 rank | Tier 2 PERF |
|---|---:|---:|---:|---:|
| CKA-RL | 1 | 0.8925 | 2 | 0.470 |
| CompoNet | 2 | 0.8729 | excluded, grows | — |
| **Ours** | **3** | 0.8670 | **1** | **0.596** |
| Fine-tuning | 4 | 0.8659 | 4 | 0.261 |
| CReLUs | 5 | 0.8354 | 3 | 0.328 |
| CbpNet | 6 | 0.8035 | 5 | 0.258 |

CKA-RL leads ours by 0.026 at tier 1. Ours leads CKA-RL by 0.126 at tier 2,
five times larger and in the opposite direction. CbpNet falls below plain
fine-tuning at tier 2 on both performance and forgetting.

## 4. Environment access, the rebuttal

Order these arguments as written. The second is the strongest.

1. **Persistent simulators are standard practice** in sim, games and robotics.
   The setting is well populated, not exotic.
2. **The environments are retained for evaluation regardless.** No method can
   fill in a forgetting matrix without re-instantiating past tasks after every
   new one, so past-task access is already present in the experimental setup
   for CKA-RL, CompoNet, CbpNet, CReLUs and fine-tuning alike. The only
   question is whether the algorithm may use what the harness already holds.
3. **Access alone does not explain the gain.** CLEAR holds past-task data via
   replay and still loses in the reversed Atari order. Joint holds all data from
   the start and ours matches or exceeds it at small scale.

**Expected counter.** The evaluation harness sits outside the agent, so granting
the agent that access changes the problem class. This counter is fair.

**Response.** Name the setting rather than claim parity. "Continual RL with
persistent task simulators" is a legitimate setting. Never claim our assumption
is weaker than or equal to replay. It is stronger, and saying so plainly is what
makes the rest credible. State it in the abstract, not in an appendix.

**Precision required on argument 3.** Two separate pieces of evidence, unequal
strength.
- "Matches or exceeds Joint" is from the 3 to 6 task GridWorld with the exact
  estimator. Solid, small scale.
- On Atari ours reaches 83% of the Joint ceiling reversed and 43% canonical. It
  does not surpass Joint in either order.

## 5. Order sensitivity

| Order | Ours | CLEAR |
|---|---:|---:|
| Canonical | 43% | **88%** |
| Reversed | **83%** | 43% |

Ours swings 40 points, CLEAR 45. Ours is **not** more order-robust, and
order-averaged the two are tied (63% vs 65.5%).

**Placement, Protik's call.** The limitations section, framed as a failure mode
common to every method tested and explicitly out of scope for this paper. The
planned figure extends the demonstration to CKA-RL and CompoNet and **includes
ours alongside them**. Omitting ours would read as concealment.

**Consequence.** The Atari tier cannot carry "we beat CLEAR", since that holds
in the reversed order only. Report both orders wherever Atari appears.

## 6. Division of labour across tiers

| Tier | Carries |
|---|---|
| GridWorld, 50 tasks, 3 seeds, 6 methods | **The method claim.** No interval overlap, 0.126 ahead, 91% against 13% |
| CKA-RL benchmark | The saturation diagnosis |
| Atari, 5 games, both orders | Interference under genuinely different tasks, plus the order-sensitivity limitation |

GridWorld is the only tier with real seeds and it is the one where we dominate.
Atari single-seed and order-contingent cannot carry a method claim.

## 7. Other attacks, pre-empted

- **Ours learns each task to 0.25 where others reach 0.65**, then recovers.
  Positive backward transfer is partly earned and partly structural. Defence is
  the headroom-normalised table, which is positive for ours in every bin and
  negative for every baseline in every bin. Put it in the main paper.
- **Forward transfer, last of six.** Local budgets are unmatched (150 to 250
  against 150 to 500) and the from-scratch reference is being recomputed. Either
  match the budgets or state the asymmetry plainly.
- **Breakout value-vs-score gap.** The constraint was satisfied on `V` while the
  greedy score fell 396 to 77.6. A real limitation of constraining value rather
  than score. Volunteer it.
- **Compute.** 244 min against 136 to 171. Between 1.4x and 1.8x. State plainly.
- **Seeds.** Tier 1 and Atari are single-seed. Footnote promising rebuttal-period
  standard deviations is the standard and acceptable move.

## 8. The one blocking control

A **Joint multi-task run at the 50-task shared-head GridWorld scale**. It is the
only control that isolates the objective from the access at scale with measured
seeds, now that the CLEAR comparison is known to be order-dependent.
`joint_multitask()` exists at `crl/baselines.py:104`; no joint family exists in
`reports/gridworld_sharedhead/`.

Prediction worth stating in advance. With one shared head and 50 tasks trained
simultaneously, Joint may underperform through capacity contention with no
curriculum. If ours exceeds Joint there, that is a headline in its own right.
