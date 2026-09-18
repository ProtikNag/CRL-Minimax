# Standing arguments for the paper

Rebuttals and framings agreed on 2026-09-16. When a draft exists, check it
against this file. Every item here is a claim the paper must either make or
deliberately decline to make.

---

## 1. The spine

**Revised 2026-09-17. The algorithm is the contribution.** An earlier draft of
this file made the benchmark diagnosis the spine and the method a consequence.
That is now inverted at Protik's direction.

> Continual RL still loses old tasks whenever capacity is shared. We state
> retention as an explicit constraint, that the deployed policy stay within
> epsilon of an expert on every task it has seen, and solve the resulting
> min-max problem by primal-dual alternation.

The evaluation contribution is **subordinate and instrumental**. The standard
benchmark cannot show what the constraint does, so we build two settings that
can. It motivates the experimental design in Section 5 and returns as a finding
in Section 7; it is not the opening claim.

**The number that justifies the extra settings.** On the CKA-RL benchmark ours
and naive fine-tuning differ by **0.0011** (0.8670 vs FT-N 0.8659). Remove the
per-task heads and the gap is **0.335** (0.596 vs 0.261). Used to explain why a
third-place finish on a saturated benchmark is uninformative either way, not to
argue that benchmarks are the paper's subject.

## 1b. Claims already cut as false, do not reintroduce

Two formulations of the gap were drafted and killed. Both are the kind a
reviewer refutes in one line, so they are recorded here rather than left to be
rediscovered.

**"Existing methods never check whether an old task is still done well."**
Flatly untrue. Every continual-RL paper evaluates old tasks and reports backward
transfer, average performance and forward transfer off a forgetting matrix. The
distinction the paper actually draws is between **measuring** retention after
training and **requiring** a performance level during it.

**"They decide in advance how much to protect old tasks."** Overclaims.
Adaptive schemes exist and Fisher weighting is data-driven, so a blanket "in
advance" invites a counterexample.

**What survives, and why it holds.** Retention is pursued indirectly, by
penalising weight movement, imitating behaviour from stored data, or expanding
the architecture, and none of these makes performance on an old task a
constraint the agent is required to meet. That last clause covers architectural
methods too, because freezing guarantees no **change**, which is not the same as
requiring a **level**, and is why those methods cannot improve an old task
either.

## 1c. Plain language in the abstract and introduction

No reader should need to already know what primal-dual means, what a one-sided
constraint is, or what a shared head is. Each is stated as what literally
happens.

| Term | What to write instead |
|---|---|
| primal-dual alternation | the correction grows with how far a task has fallen |
| one-sided constraint | corrected only when it scores below an expert, never pulled back when it scores above |
| one shared head | the network cannot expand to give each task its own parameters |

The failure mode to avoid is swapping jargon for vague abstraction, which is
worse. An earlier draft wrote "the pressure applied for each task" and "anchors
the deployed policy", neither of which names anything a reader can picture. The
technical vocabulary is correct and belongs in Section 4.

## 1d. The multiplier adapts, and this is the evidence

The abstract and the method section both claim each task's multiplier is set by
the optimisation rather than chosen in advance. The dual traces support it
directly, and across two tiers in opposite directions.

| Tier | Multiplier | Backward transfer |
|---|---|---:|
| GridWorld, 50 tasks | zero in ~98% of logged steps, peak 0.007 to 0.02 | **+0.358** |
| Atari, 5 games | active in all 88 logged steps of all three consolidated tasks, saturated at its cap of 5.0 | −0.26 |

Slack where the constraint is easily satisfied, pinned to the ceiling where it
is not. A fixed penalty weight cannot do both.

**State it as a sentence, not a figure.** Four figures were built from
`reports/final/atari_reversed/consolidation_dynamics.json` and all four cut; the
reasons are recorded in that folder's README. The short version is that a
one-sided hinge releasing above zero is true by construction, and the tier that
shows the mechanism is not the tier that shows the payoff.

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

**The Atari claim needs its qualifier.** CompoNet retains the earlier games
better than we do, 0.83 against 0.67 of ceiling, by expanding the architecture
and by never learning the final game, which it scores 0.0 on. Ours is the only
method **without an expanding architecture** that both retains and learns, with
CLEAR at 0.43 on prior games and CKA-RL at −0.25. Unqualified, the claim is
refutable from our own table.

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
