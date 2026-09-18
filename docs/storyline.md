# Paper storyline

The working storyline document for the paper. It is the single source of truth
for what the paper claims, in what order, and what the headline contribution is.
It has been through one round of outside review, and the decisions that came out
of that round are in Section 0.

It is written to be self-contained, so it can be handed to a reader with no
prior context on the project.

**Venue.** ICLR 2026.

> A few final numbers are still landing from the cluster. Treat the figures
> below as current best values. Nothing about the storyline depends on the last
> decimal place.

---

## 0. Settled decisions

| Decision | Resolution |
|---|---|
| Framing | **Method-first.** The algorithm is the headline contribution; the evaluation settings exist so it can be seen working |
| Method name | **DUEL**, Dual-Policy Expert-Anchored Learning |
| Title | *DUEL: Primal-Dual Retention Constraints for Continual Reinforcement Learning* |
| Results order | Established benchmark, then GridWorld, then Atari |
| Order sensitivity | Limitations section, framed as field-wide and out of scope |
| Scope and cost | Stated in the abstract, not buried in limitations |

**On the results order.** An outside reviewer argued for leading with GridWorld,
so the reader's first empirical impression would be the strongest evidence rather
than a third-place table. We kept the established benchmark first for two
reasons. Table 1 first poses a question, that the benchmark cannot separate this
method from naive fine-tuning, which the GridWorld section then answers. And it
establishes standing on the field's own benchmark before asking the reader to
accept two settings of our own.

Emphasis on GridWorld is carried by four other levers instead.

1. **The abstract quotes GridWorld's numbers.** The established benchmark gets
   one clause.
2. **The per-task scatter is Figure 1**, placed in the introduction. Reviewers
   read figures before sections, which makes this the strongest lever.
3. **The benchmark subsection runs a third of a page** against a full page for
   GridWorld. Length signals importance more reliably than position.
4. **Its title and opening sentence pre-frame it** as a compatibility check,
   closing with a pointer forward.

---

## 1. The problem

Continual reinforcement learning asks an agent to learn a sequence of tasks and
still perform well on the earlier ones at the end. The failure mode is
catastrophic forgetting, and it worsens the more the tasks share parameters.

Existing approaches fall into three families, each paying a different price.

| Family | Examples | What it pays |
|---|---|---|
| Replay | CLEAR | Storage. Keeps a buffer of past trajectories |
| Regularisation | EWC and descendants | A penalty weight fixed in advance, before you know how much any given task will be hurt |
| Modular growth | ProgNet, PackNet, CompoNet, CKA-RL | Parameters. Adds or freezes capacity per task |

None of the three states retention as a **requirement the optimiser has to
satisfy**. They state it as a penalty, a buffer, or a partition. That gap is what
the method below is for.

---

## 2. The method, DUEL

**Dual-Policy Expert-Anchored Learning (DUEL)**. Two policies, and the deployed
one is anchored to an expert on every task it has seen.

**A note on vocabulary.** The paper itself avoids "primal-dual", "one-sided
constraint" and "shared head", because no reader of an abstract or introduction
should need to already know those terms. They are used freely in this document
and in Section 4 of the paper, but the abstract states each as what literally
happens. See `paper/sections/abstract.tex`.

**Two policies.** A *local* policy specialises on whatever task has just arrived.
A *global* policy is the one deployed and evaluated. At the start of each task
the local policy is initialised from the global, so the expert is always
reachable from where the deployed policy currently sits.

**The constraint.** For each task *k* the agent has seen, define the shortfall of
the deployed global policy against that task's expert

```
F_k = [ V_k(local) − V_k(global) ]_+ ^ 2  ≤  ε
```

The deployed policy must stay within ε of an expert on **every** task it has
seen. Two details in that form matter.

- **One-sided.** The hinge `[·]_+` means beating the expert is not a
  violation. A two-sided constraint would cap the method at expert
  performance, throwing away the positive backward transfer that turns out to be
  the main empirical result.
- **Squared.** Gives a smooth gradient as the shortfall approaches zero.

**How it is solved.** The Lagrangian gives a min-max objective, solved by
primal-dual alternation with projected gradient ascent on a dual multiplier μ.
The actor coefficient is the differentiated hinge, `2·μ·[V_local − V_global]_+`.
Because the hinge is one-sided, a task already inside ε contributes **exactly
zero** gradient, so in principle optimisation effort concentrates on whichever
past tasks are at risk. This is the intended difference from uniform joint
training, which spends equally on every task including those needing nothing.

**No replay buffer.** Rather than storing trajectories, the method re-enters past
task environments and collects fresh rollouts during consolidation. This is an
assumption and an important one. See objection 1.

> **Open verification item.** In the logged GridWorld runs the global multiplier
> μ sits at zero in roughly 98% of recorded steps across all three seeds, peaking
> around 0.007 to 0.02. The shortfall itself averages 0.089, so the policies do
> differ, but the constraint is on the *squared* shortfall and 0.089² sits under
> ε, leaving the constraint slack. Those logs record the global multiplier only,
> not the per-past-task local multipliers, so this may not be the whole picture.
> Until it is resolved, any claim that the multipliers actively steer effort
> toward endangered tasks should be treated as design intent rather than as a
> measured property. See Section 8.

---

## 3. The three experiment settings, and how they differ

Continual RL benchmarks vary along two axes that strongly affect how much
interference is even possible.

- **Output head.** Does each task get its own output head, or do all tasks share
  one? With a per-task head, the head cannot be overwritten by later tasks, so
  forgetting can happen only through the shared trunk. Interference is
  structurally suppressed.
- **Task relatedness.** Are the tasks *modes of the same game*, or wholly
  different environments? Modes of one game share states and dynamics, so
  transfer is high and interference is low.

The established benchmark has **both** properties. Our two additional settings
each remove **one** of them and hold the other fixed. It is a factorial design
over the benchmark's own structure, not simply "we made it harder".

| Setting | Output head | Task relatedness | Tasks | Seeds |
|---|---|---|---|---|
| **A. CKA-RL benchmark** (NeurIPS 2025) | per-task | modes of one game | Meta-World CW20, SpaceInvaders, Freeway | 1 (ours), 10 (paper rows) |
| **B. GridWorld** | **shared** | related layouts | 50 distinct layouts | 3 |
| **C. Atari sequence** | per-task | **different games** | 5 games | 1 |

**A** is the published benchmark we add a row to. It is the compatibility check.

**B** removes the per-task head. Fifty 50×50 layouts varying obstacle density and
type, slipperiness, and reward and penalty magnitudes, under one shared
task-conditioned head. Capacity is fixed and interference is forced. The
strongest evidence lives here, because it is the only setting with multiple
seeds.

**C** removes task relatedness. Five different Atari games in sequence,
SpaceInvaders → Boxing → Breakout → Pong → Q\*bert. Disjoint state spaces and
unrelated dynamics, so almost nothing transfers and the shared trunk is under
maximum pressure.

---

## 4. Results

### Setting A, the established benchmark

Rows transcribed from the CKA-RL paper. The average is recomputed over
SpaceInvaders and Freeway for every row, including the paper's own, so all rows
are comparable.

| Rank | Method | Avg performance | Avg forward transfer |
|---:|---|---:|---:|
| 1 | CKA-RL | 0.8925 | 0.7589 |
| 2 | CompoNet | 0.8729 | 0.7039 |
| **3** | **DUEL (ours)** | **0.8670** | **0.6555** |
| 4 | Fine-tuning (FT-N) | 0.8659 | 0.6900 |
| 5 | CReLUs | 0.8354 | 0.6305 |
| 6 | CbpNet | 0.8035 | 0.6022 |
| 7 | Baseline | 0.3780 | 0.0000 |
| 8 | ProgNet | 0.3441 | 0.0931 |
| 9 | FT-1 | 0.2962 | 0.6900 |
| 10 | PackNet | 0.2533 | 0.0610 |
| 11 | MaskNet | 0.0322 | −0.2185 |

**Third of eleven, winning no individual column, 0.026 behind the best.** Two
things matter here beyond the rank. The top six span a band of only **0.089**,
and **naive sequential fine-tuning sits 0.0011 behind us** in fourth place. On
this benchmark a method built to prevent forgetting is not separable from one
that does nothing about it. Per-task heads and same-game tasks are why.

### Setting B, GridWorld, 50 tasks, shared head, three seeds

| | Performance | Forgetting | Backward transfer | Forward transfer |
|---|---:|---:|---:|---:|
| **DUEL (ours)** | **0.596** ±0.007 | **0.112** ±0.008 | **+0.358** ±0.028 | 0.077 ±0.063 |
| CKA-RL | 0.470 ±0.040 | 0.237 ±0.028 | −0.187 ±0.039 | 0.162 ±0.037 |
| CReLUs | 0.328 ±0.043 | 0.391 ±0.057 | −0.359 ±0.057 | **0.179** ±0.066 |
| Fine-tuning | 0.261 ±0.051 | 0.430 ±0.048 | −0.390 ±0.053 | 0.142 ±0.020 |
| CbpNet | 0.258 ±0.030 | 0.467 ±0.033 | −0.430 ±0.033 | 0.156 ±0.029 |
| From-scratch | 0.158 ±0.062 | 0.559 ±0.066 | −0.506 ±0.070 | 0 (reference) |

Normalised as `(score − random) / (ceiling − random)`, so 0 is a random policy
and 1 is a solved task.

We lead by **0.126** and no confidence interval comes close to touching. **Ours
is the only method among those evaluated with positive backward transfer**,
+0.358 against −0.187 for the nearest. Every other method ends below where it
started.

The clearest single statistic. Of all 150 task-seed pairs, did each end better or
worse than when it was first learned?

| | Tasks that ended **better** |
|---|---:|
| **DUEL (ours)** | **91%** |
| CKA-RL | 13% |
| Fine-tuning | 8% |
| From-scratch | 6% |
| CReLUs | 5% |
| CbpNet | 4% |

Our point cloud sits above the no-change diagonal. Every other method's sits
below it. A difference in kind rather than in degree, and this scatter is
Figure 1.

**Contrast with Setting A.** Ours and fine-tuning differ by 0.001 on the
established benchmark. Remove the per-task heads and the same two methods differ
by 0.335.

### Setting C, five different Atari games

Normalised by a jointly-trained model's score on each game.

| | Backward transfer | Forgetting | Average, all tasks | Average, before the last task |
|---|---:|---:|---:|---:|
| **DUEL (ours)** | **−0.21** | **0.29** | 0.77 | 0.70 |
| CLEAR (replay) | −0.62 | 0.62 | 1.07 | 0.43 |
| CKA-RL | −0.99 | 0.99 | 0.01 | −0.25 |
| CompoNet | +0.00 | 0.00 | 0.66 | **0.83** |

The last column excludes the final task, which has had nothing trained after it
and so measures capacity rather than retention. Including it lets a method that
simply overfits the last game post a flattering average, which is what happens to
CLEAR.

The aggregate undersells what is interesting. **The two strongest baselines fail
in opposite directions, and between them they draw the stability-plasticity
dilemma about as starkly as a benchmark can.** This should be shown at task
level, not only as aggregates.

- **CKA-RL forgot almost everything.** Final scores are SpaceInvaders 212.6,
  Boxing −15.5, Breakout 6.7, Pong −21.0, Q\*bert 4420.5. **Boxing and Pong
  finish below a random policy.** Only the last game learned survives. A pure
  retention failure.
- **CompoNet forgot nothing and stopped learning.** Components freeze, so
  backward transfer is exactly 0.00 on every task by construction. The price is
  on the diagonal. Breakout reached only 99.2 against a threshold of 285, and
  **Q\*bert scored 0.0, never learned at all.** A pure plasticity failure.
- **CLEAR's 1.07 is an artefact** of overfitting the final task. Its Q\*bert
  score is 3.6× the joint reference while everything before decayed, which is why
  it falls to 0.43 once the last task is excluded.

**CompoNet posts the best retention in the table, 0.83 against our 0.67**,
precisely because frozen components cannot be overwritten. It buys that by
expanding the architecture, and it never learns the final game at all, scoring
0.0, which is why its all-task average of 0.66 sits below ours at 0.78.

**State the claim with its qualifier.** Among methods that do **not** expand the
architecture, ours is the only one that both retains the earlier games and
learns the last: CLEAR reaches 0.43 on prior games and CKA-RL −0.25, against our
0.67. Dropping "without an expanding architecture" turns a true claim into one a
reviewer can refute from our own table.

---

## 5. Limitations we will state

**Task order sensitivity, out of scope for this paper.** Running the same tasks
in a different order changes which method comes out ahead. **Ours is order
dependent, and so is every baseline we tested**, by comparable margins. We have
run both orders and will present the numbers with ourselves included rather than
quietly omitting ourselves from the figure. The paper states the problem as
field-wide and declares it out of scope. Diagnosing or fixing it is future work.

**The constraint is on value, not on score, and they can come apart.** On
Breakout the local policy peaked at 396, the value shortfall reached
approximately zero, and the greedy score still fell to 77.6 during consolidation
before recovering to 144.5.

**Compute.** Consolidation re-simulates past environments, so we run roughly 1.4×
to 1.8× the wall-clock of the cheapest baseline. Stated in the abstract.

**Forward transfer.** Last of six in setting B, by design, since the local phase
is deliberately short.

---

## 6. Objections we expect, and our answers

### Objection 1. Live past-task environment access is an unfair advantage

We re-enter past environments during consolidation. That is a **stronger**
assumption than a replay buffer, not a weaker one, and we say so in the abstract
rather than bury it.

Three answers.

1. **The environments are retained for evaluation regardless.** No continual RL
   paper can fill in a forgetting matrix without re-instantiating past tasks
   after every new one, so past-task access is already present in the
   experimental setup for *every* method in the comparison. The only question is
   whether the algorithm may use what the harness already holds.
2. **Persistent simulators are standard practice** in robotics sim, games, and
   recommender sandboxes. The setting is well populated, not exotic.
3. **Access alone does not explain the gain.** CLEAR holds past-task data via
   replay and is beaten. Joint multi-task training holds all data from the start
   and ours matches or exceeds it at small scale.

The counter we expect is that the evaluation harness sits outside the agent, so
granting the agent that access changes the problem class. That counter is fair.
Our response is to **name the setting** ("continual RL with persistent task
simulators") rather than claim parity with replay-free methods.

### Objection 2. Is the gain from the objective or just from the access?

This is the attribution question, and it needs the two experiments in Section 7.

### Objection 3. You are only third on the established benchmark

That benchmark packs six methods into 0.089 and puts naive fine-tuning 0.001
behind us, so it cannot separate methods on retention. Being third there is a
check that the method is not broken, not a result in either direction.

### Objection 4. The positive backward transfer is a headroom artefact

A task left at a low score has more room to improve, so "improved" could in
principle be mechanical. We measure the gain as a share of the headroom still
available, within bins of initial score. Ours is positive in every bin. Every
baseline is negative in every bin, so they do not merely fail to improve, they
give ground back regardless of where the task started.

---

## 7. The plan

**Contributions**, most to least important.

1. **The method.** Retention stated as an explicit per-task constraint on the
   deployed policy's shortfall against an expert, solved by primal-dual
   alternation, rather than approximated by a fixed penalty, a replay buffer, or
   a parameter partition. The constraint is one-sided, so the deployed policy is
   protected from falling behind experts while remaining free to exceed them.
2. **Two evaluation settings** that remove, one at a time, the two properties
   that suppress interference in the established benchmark, holding the other
   fixed in each.
3. **Empirical findings.** Method ranking does not transfer between the
   established benchmark and the shared-head setting. Plasticity methods do not
   address retention, and CbpNet is in fact worse than plain fine-tuning at fifty
   tasks. In the unrelated-game sequence the two strongest baselines fail in
   opposite directions and ours avoids both.

**Section plan, nine pages.**

| § | Section | Purpose |
|---|---|---|
| 1 | Introduction | Make the reader want a retention constraint before they see one. Figure 1 lives here |
| 2 | Related work | Retention families, plasticity methods, constrained RL and primal-dual |
| 3 | Problem setting | Notation, the persistent-simulator assumption stated up front, metrics |
| 4 | **Method** (2.25 pages) | Two policies, the constraint, primal-dual solution, why a constraint and not a penalty, implementation |
| 5 | Experimental setup | Three settings and why the established one is not enough, baselines and budgets |
| 6 | Results | A, then B, then C |
| 7 | Analysis | Where the advantage comes from, what the benchmark cannot show, plasticity versus retention |
| 8 | Limitations | Order sensitivity and the rest |
| 9 | Conclusion | |

**Narrative flow.** Motivate the constraint from the gap in existing work →
present the method as the centre of the paper → explain why the established
benchmark alone cannot demonstrate it → show parity there, then dominance in the
two settings that permit interference → analyse why → concede the limitations.

**Two experiments still required before submission.**

1. **Matched-access control.** A comparison with past-task simulator access and a
   comparable rollout budget but without the retention constraint, that is, joint
   multi-task training at the full fifty-task scale. Answers whether the gain
   comes from the objective or merely from revisiting old environments.
2. **Multiplier-zero ablation.** Same pipeline, same access, same local and
   global design, multipliers fixed at zero. The direct test that the constraints
   themselves matter.

**Claim discipline.** Say that per-task heads and task relatedness *can
substantially mask* interference, rather than that conventional benchmarks do not
measure forgetting. Say *among the evaluated methods* rather than field-wide.
Describe the established-benchmark result as parity, never as dominance.

---

## 8. Claude's opinion, offered separately

*This section is the opinion of Claude, the assistant helping prepare this
document, kept apart so it can be disagreed with without unpicking the facts.
First person below refers to Claude.*

**On the multiplier traces, which is what I would resolve first.** The
contribution list leans on the constraint doing observable work, and the method
section exists partly to defend choosing a constraint over a penalty. The logged
runs do not currently support that. μ is zero in 23 of 1470, 69 of 1550 and 31 of
1510 recorded steps across the three seeds, and the mean actor coefficient lies
between 2e-5 and 1.2e-4. If the per-task local multipliers turn out to be equally
quiet, the multiplier-zero ablation will come back indistinguishable from the
full method, and the GridWorld result becomes attributable to consolidation
rather than to the constraint. Two outcomes are survivable but they want
different papers. Either the constraint is a guarantee that rarely binds, which
is a weaker but still real claim, or ε is set too loose and tightening it is a
cheap experiment that makes the story land as intended. Better known now than
after a reviewer asks.

**On what is strongest.** The factorial framing of the two extra settings is the
most defensible methodological asset. "We made it harder" invites a reviewer to
say we picked settings that suit us. "We removed the two specific properties that
suppress interference, one at a time, holding the other fixed" is a designed
experiment and far harder to dismiss. State it explicitly rather than leaving it
implicit.

**On the Atari result.** The opposite failures of CKA-RL and CompoNet are the
most quotable thing in the paper and the current plan does not give them enough
room. One baseline forgets everything except the last task; the other never
forgets and never learns the last task. Ours sits between them. That is the
stability-plasticity dilemma as a single figure, and it argues for the method
better than any aggregate metric.

**On the results order.** I preferred leading with GridWorld and was overruled.
Having thought about the counter-argument, I think the decision is right. Table 1
first turns a third-place finish into the setup for a question, which is a
stronger arc than opening with the answer. The four emphasis levers in Section 0
matter more than the order does, and the first of them, making the scatter
Figure 1, is the one I would act on immediately.

**On honesty as strategy.** We report several things that do not flatter us,
namely last place on forward transfer, the low immediate learning score, the
order sensitivity, the compute cost, and the stronger-than-replay assumption. I
think this helps rather than hurts, because each is something a reviewer would
otherwise find and weight more heavily than we would.

---

## 9. Open questions

1. **Are the local multipliers active?** The item in Section 2 and the first
   entry in Section 8. Much of the contribution list depends on it.
2. **Where is this plan most exposed** to a sceptical reviewer, beyond the
   attribution question the two required experiments address?
3. **What have we not thought of?** Framings, analyses, figures, further cuts of
   the results already in hand, or experiments not considered. Speculative
   suggestions welcome.
4. **Anything being undersold or oversold** in the results as presented here?
