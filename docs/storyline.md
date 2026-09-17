# Storyline brief, for a second opinion

## What this document is

I am writing a paper for **ICLR 2026** on a continual reinforcement learning
algorithm. The experiments are done or nearly done. What I am still deciding is
the **storyline**, that is, what the paper claims, in what order, and what the
headline contribution is.

This document gives you everything you need to form an independent view. It
describes the algorithm, the three experiment sets and how they differ from each
other, the results, the objections I expect from reviewers and my prepared
answers, and my current plan for the paper's structure. Section 8 carries the
opinion of **Claude**, the assistant that helped prepare this brief, kept
separate so you can disagree with it without unpicking the facts. Section 9 says
what I would like you to produce.

I am collecting opinions from several models before committing. Please be
direct. If you think the framing is wrong, say so and say why.

> Minor note. A few of the final numbers are still landing from the cluster.
> Treat the figures below as current best values. Nothing about the storyline
> question depends on the last decimal place.

---

## 1. The problem

Continual reinforcement learning asks an agent to learn a sequence of tasks and
still perform well on the earlier ones at the end. The failure mode is
catastrophic forgetting, and it gets worse the more the tasks have to share the
same parameters.

Existing approaches fall into three families, and each pays a different price.

| Family | Examples | What it pays |
|---|---|---|
| Replay | CLEAR | Storage. Keeps a buffer of past trajectories |
| Regularisation | EWC and descendants | A penalty weight fixed in advance, before you know how much any given task will be hurt |
| Modular growth | ProgNet, PackNet, CompoNet, CKA-RL | Parameters. Adds or freezes capacity per task |

None of the three states retention as a **requirement the optimiser has to
satisfy**. They state it as a penalty, a buffer, or a partition. That gap is
what the algorithm below is for.

---

## 2. The algorithm

Working name **SPARC**, for Specialist-Anchored Regret Constraint. It is a
constrained two-policy min-max method.

**Two policies.** A *local* policy specialises on whatever task has just
arrived. A *global* policy is the one actually deployed and evaluated. At the
start of each task the local policy is initialised from the global, so the
specialist is always reachable from where the deployed policy currently sits.

**The constraint.** For each task *k* the agent has seen, define the shortfall
of the deployed global policy against that task's specialist

```
F_k = [ V_k(local) − V_k(global) ]_+ ^ 2  ≤  ε
```

The deployed policy is required to stay within ε of a specialist on **every**
task it has seen. Two details in that form matter.

- **One-sided.** The hinge `[·]_+` means beating the specialist is not a
  violation. A two-sided constraint would cap the method at specialist
  performance, which would throw away the positive backward transfer that turns
  out to be the method's main empirical result.
- **Squared.** Gives a smooth gradient as the shortfall approaches zero.

**How it is solved.** The Lagrangian of the constrained problem gives a min-max
objective. It is solved by primal-dual alternation, with projected gradient
ascent on a dual multiplier μ. The actor coefficient is the differentiated
hinge, `2·μ·[V_local − V_global]_+`. Because the hinge is one-sided, a task
already inside ε contributes **exactly zero** gradient, so optimisation effort
concentrates automatically on whichever past tasks are actually at risk. This is
the main difference from uniform joint training, which spends equally on every
task including the ones that need nothing.

**No replay buffer.** Instead of storing trajectories, the method re-enters past
task environments and collects fresh rollouts during consolidation. This is an
assumption, and an important one. See objection 1 in Section 6.

---

## 3. The three experiment sets, and how they differ

This is the part I most want a view on, so read it carefully.

Continual RL benchmarks vary along two axes that strongly affect how much
interference is even possible.

- **Output head.** Does each task get its own output head, or do all tasks share
  one? With a per-task head, the head cannot be overwritten by later tasks, so
  forgetting can only happen through the shared trunk. Interference is
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

**A** is the published benchmark we add a row to. It is the sanity check.

**B** removes the per-task head. Fifty different 50×50 layouts varying obstacle
density and type, slipperiness, and reward and penalty magnitudes, all under one
shared task-conditioned head. Capacity is fixed and interference is forced. This
is where the strongest evidence lives, because it is the only setting with
multiple seeds.

**C** removes task relatedness. Five different Atari games in sequence,
SpaceInvaders → Boxing → Breakout → Pong → Q\*bert. The games have disjoint
state spaces and unrelated dynamics, so almost nothing transfers and the shared
trunk is under maximum pressure.

---

## 4. Results

### Setting A, the established benchmark

Rows are transcribed from the CKA-RL paper. The average is recomputed over
SpaceInvaders and Freeway for every row, including the paper's own rows, so all
rows are comparable.

| Rank | Method | Avg performance | Avg forward transfer |
|---:|---|---:|---:|
| 1 | CKA-RL | 0.8925 | 0.7589 |
| 2 | CompoNet | 0.8729 | 0.7039 |
| **3** | **SPARC (ours)** | **0.8670** | **0.6555** |
| 4 | Fine-tuning (FT-N) | 0.8659 | 0.6900 |
| 5 | CReLUs | 0.8354 | 0.6305 |
| 6 | CbpNet | 0.8035 | 0.6022 |
| 7 | Baseline | 0.3780 | 0.0000 |
| 8 | ProgNet | 0.3441 | 0.0931 |
| 9 | FT-1 | 0.2962 | 0.6900 |
| 10 | PackNet | 0.2533 | 0.0610 |
| 11 | MaskNet | 0.0322 | −0.2185 |

**We are third of eleven and win no individual column.** We are 0.026 behind the
best. Two things are worth noticing about this table beyond our rank.

The top six methods span a band of only **0.089**. And **naive sequential
fine-tuning is 0.0011 behind us**, in fourth place. A benchmark on which a
method designed to prevent forgetting is statistically indistinguishable from a
method that does nothing to prevent forgetting is not measuring forgetting.
Per-task heads and same-game tasks are why.

### Setting B, GridWorld, 50 tasks, shared head, three seeds

| | Performance | Forgetting | Backward transfer | Forward transfer |
|---|---:|---:|---:|---:|
| **SPARC (ours)** | **0.596** ±0.007 | **0.112** ±0.008 | **+0.358** ±0.028 | 0.077 ±0.063 |
| CKA-RL | 0.470 ±0.040 | 0.237 ±0.028 | −0.187 ±0.039 | 0.162 ±0.037 |
| CReLUs | 0.328 ±0.043 | 0.391 ±0.057 | −0.359 ±0.057 | **0.179** ±0.066 |
| Fine-tuning | 0.261 ±0.051 | 0.430 ±0.048 | −0.390 ±0.053 | 0.142 ±0.020 |
| CbpNet | 0.258 ±0.030 | 0.467 ±0.033 | −0.430 ±0.033 | 0.156 ±0.029 |
| From-scratch | 0.158 ±0.062 | 0.559 ±0.066 | −0.506 ±0.070 | 0 (reference) |

Normalisation is `(score − random) / (ceiling − random)`, so 0 is a random
policy and 1 is a solved task.

We lead by **0.126**, and no confidence interval comes close to touching.
**Ours is the only method with positive backward transfer**, +0.358 against
−0.187 for the nearest competitor. Every other method ends below where it
started.

The clearest single statistic. Ask of every one of the 150 task-seed pairs
whether it ended better or worse than when it was first learned.

| | Tasks that ended **better** |
|---|---:|
| **SPARC (ours)** | **91%** |
| CKA-RL | 13% |
| Fine-tuning | 8% |
| From-scratch | 6% |
| CReLUs | 5% |
| CbpNet | 4% |

Our point cloud sits above the no-change diagonal. Every other method's sits
below it. That is a difference in kind rather than in degree.

**Note the contrast with Setting A.** Ours and fine-tuning differ by 0.001 on
the established benchmark. Remove the per-task heads and the same two methods
differ by 0.335.

### Setting C, five different Atari games

Scores are normalised by a jointly-trained model's score on each game.

| | Backward transfer | Forgetting | Average, all tasks | Average, before the last task |
|---|---:|---:|---:|---:|
| **SPARC (ours)** | **−0.21** | **0.29** | 0.77 | 0.70 |
| CLEAR (replay) | −0.62 | 0.62 | 1.07 | 0.43 |
| CKA-RL | −0.99 | 0.99 | 0.01 | −0.25 |
| CompoNet | +0.00 | 0.00 | 0.66 | **0.83** |

The last column excludes the final task in the sequence, which has had nothing
trained after it and therefore measures capacity rather than retention.
Including it lets a method that simply overfits the last game post a flattering
average, which is exactly what happens to CLEAR.

The aggregate table undersells what is interesting here. **The two strongest
baselines fail in opposite directions, and between them they draw the
stability-plasticity dilemma about as starkly as a benchmark can.**

- **CKA-RL forgot almost everything.** Its final scores are SpaceInvaders 212.6,
  Boxing −15.5, Breakout 6.7, Pong −21.0, Q\*bert 4420.5. **Boxing and Pong
  finish below a random policy.** Only the last game it learned survives. This
  is a pure retention failure.
- **CompoNet forgot nothing and stopped learning.** Its components freeze, so
  backward transfer is exactly 0.00 on every task by construction. The price
  appears on the diagonal. Breakout reached only 99.2 against a threshold of
  285, and **Q\*bert scored 0.0, never learned at all.** This is a pure
  plasticity failure.
- **CLEAR's 1.07 average is an artefact** of overfitting the final task. Its
  Q\*bert score is 3.6× the joint reference while everything before it decayed,
  which is why it falls to 0.43 once the last task is excluded.

Note that CompoNet posts the **best** score in the table on tasks before the
last one, 0.83, precisely because its frozen components cannot be overwritten.
Its 0.66 overall is what that costs, and the whole of the difference is the game
it never learned.

Ours is the only method in the comparison that both retains old tasks and keeps
learning new ones.

---

## 5. Limitations we will state

**Task order sensitivity, and it is out of scope for this paper.** Running the
same tasks in a different order changes which method comes out ahead. **Ours is
order dependent, and so is every baseline we tested**, by comparable margins. We
have run both orders and have the numbers to show it, and we will present them
with ourselves included rather than quietly omitting ourselves from the figure.
The paper will state the problem as field-wide and declare it out of scope.
Diagnosing or fixing order sensitivity is future work, not this paper.

**Other limitations we will state.**

- The constraint is on **value**, not on score, and they can come apart. On
  Breakout the local policy peaked at 396, the value shortfall reached
  approximately zero, and the greedy score still fell to 77.6 during
  consolidation before recovering to 144.5.
- **Compute.** Consolidation re-simulates past environments, so we are roughly
  1.4× to 1.8× the wall-clock of the cheapest baseline.
- **Forward transfer.** Last of six in setting B, by design, since our local
  phase is deliberately short.

---

## 6. Objections we expect, and our answers

### Objection 1. Live past-task environment access is an unfair advantage

We re-enter past environments during consolidation. That is a **stronger**
assumption than a replay buffer, not a weaker one, and we will say so in the
abstract rather than bury it.

Three answers.

1. **The environments are retained for evaluation regardless.** No continual RL
   paper can fill in a forgetting matrix without re-instantiating past tasks
   after every new one. So past-task access is already present in the
   experimental setup for *every* method in the comparison. The only question is
   whether the algorithm may use what the evaluation harness already holds.
2. **Persistent simulators are standard practice** in robotics sim, games, and
   recommender sandboxes. The setting is well populated, not exotic.
3. **Access alone does not explain the gain.** CLEAR holds past-task data via
   replay and is beaten. Joint multi-task training holds all data from the start
   and ours matches or exceeds it at small scale.

The counter we expect is that the evaluation harness sits outside the agent, so
granting the agent that access changes the problem class. We think that counter
is fair, and our response is to **name the setting** ("continual RL with
persistent task simulators") rather than claim parity with replay-free methods.

### Objection 2. Is the gain from the objective or just from the access?

The honest answer is that this needs a joint multi-task control at the full
fifty-task scale with matched access and budget, which is the last experiment on
our list. We will also ablate the constraint itself by fixing the multiplier at
zero.

### Objection 3. You are only third on the established benchmark

Our answer is that the established benchmark packs six methods into 0.089 and
puts naive fine-tuning 0.001 behind us, so it cannot separate methods on
retention at all. Being third there is a check that the method is not broken, not
a result in either direction.

### Objection 4. The positive backward transfer is a headroom artefact

A task left at a low score has more room to improve, so "improved" could in
principle be mechanical. We measure the gain as a share of the headroom still
available, within bins of initial score. Ours is positive in every bin. Every
baseline is negative in every bin, so they do not merely fail to improve, they
give ground back regardless of where the task started.

---

## 7. My current plan for the paper

**Target venue.** ICLR 2026.

**Contribution priority, most to least important.**

1. **The algorithm.** Retention stated as an explicit per-task constraint and
   solved by primal-dual alternation, rather than approximated by a penalty, a
   buffer, or a partition.
2. **Two evaluation settings** that remove, one at a time, the two structural
   properties that suppress interference in the established benchmark.
3. **Empirical findings**, including that method ranking does not transfer
   between the established benchmark and the shared-head setting, and that
   plasticity methods (CbpNet, CReLUs) do not address retention. CbpNet is in
   fact *worse* than plain fine-tuning at fifty tasks.

**Proposed title.** *Bounding Forgetting in Continual Reinforcement Learning.*
The word "bounding" is literally what the constraint does, and it implies the
quantity is currently unbounded, which hints at the state of the field without
spending a clause on it.

**Section plan, nine pages.**

| § | Section | Purpose |
|---|---|---|
| 1 | Introduction | Make the reader want a retention constraint before they see one |
| 2 | Related work | Retention families, plasticity methods, constrained RL and primal-dual |
| 3 | Problem setting | Notation, the persistent-simulator assumption stated up front, metrics |
| 4 | **Method** (2.25 pages) | Two policies, the constraint, primal-dual solution, why a constraint and not a penalty, implementation |
| 5 | Experimental setup | Three settings and why the established one is not enough, baselines and budgets |
| 6 | Results | A, then B, then C, in that order |
| 7 | Analysis | Where the advantage comes from, what the benchmark cannot show, plasticity versus retention |
| 8 | Limitations | Order sensitivity and the rest |
| 9 | Conclusion | |

**Narrative flow.** Motivate the constraint from the gap in existing work →
present the algorithm as the centre of the paper → explain why the established
benchmark alone cannot demonstrate it → show parity on that benchmark, then
dominance in the two settings that permit interference → analyse why → concede
the limitations.

---

## 8. Claude's opinion, offered separately

*This section is the opinion of Claude, the assistant that helped prepare this
brief. It is not mine and it is not settled fact. First person below refers to
Claude. It is the part I would most like you to push back on.*

**On the framing risk.** There are two coherent ways to tell this story and they
are in tension.

- *Method-first*, which is the plan above. Reads as a normal method paper.
  Its exposure is that Section 6.1 presents a third-place finish before the
  reader has seen anything impressive, so Section 5.1 has to do a lot of work in
  half a page.
- *Diagnosis-first*, where the paper's thesis is that continual RL benchmarks
  suppress the interference they claim to measure, and the algorithm is the
  existence proof of what you find once you stop suppressing it. Under that
  framing the third-place finish stops being a liability and becomes the opening
  evidence.

Method-first is the decision and this is not an attempt to reopen it. It is
recorded so you know what the choice costs, because the single strongest fact in
the project belongs to the other framing. **On the established benchmark this
method and naive sequential fine-tuning differ by 0.001. Remove the per-task
heads and the gap is 0.335.** That one sentence justifies the entire
experimental program, and under method-first it lands in a setup section rather
than in the opening. Whoever writes Section 5.1 should treat it as the sentence
that section exists to carry.

**On what is strongest.** The factorial framing of the two extra settings is, I
think, the most defensible methodological asset. "We made it harder" invites a
reviewer to say we picked settings that suit us. "We removed the two specific
properties that suppress interference, one at a time, holding the other fixed"
is a designed experiment and far harder to dismiss. It should be stated
explicitly rather than left implicit.

**On what is weakest.** The environment-access assumption combined with the
missing joint control at full scale. If a reviewer believes the gain comes from
access rather than from the objective, everything else follows from that doubt.
I would prioritise that ablation above any other remaining experiment.

**On the Atari result.** The opposite failures of CKA-RL and CompoNet are the
most quotable thing in the paper and I do not think the current plan gives them
enough room. One baseline forgets everything except the last task, the other
never forgets and never learns the last task. Ours sits between them. That is
the stability-plasticity dilemma as a single figure, and it argues for the method
better than any aggregate metric does.

**On honesty as strategy.** We are reporting several things that do not flatter
us, namely last place on forward transfer, the low immediate learning score, the
order sensitivity, the compute cost, and the stronger-than-replay assumption. I
think this is correct and will help rather than hurt, because each of them is
something a reviewer would otherwise find and weight more heavily than we would.

---

## 9. What I would like from you

**Do not grade what is above. Build on it.** One decision is settled and I am not
looking to reopen it. **The paper is method-first.** The algorithm is the
headline contribution, and the evaluation settings exist so that it can be seen
working. Please work inside that.

Write your own version of the following four, from scratch rather than as edits
to mine.

1. **A contribution list.** Ordered, most important first, phrased the way you
   would put it in an introduction.
2. **A storyline.** The argument the paper makes from start to finish, in a
   paragraph or two. What should a reader believe by the end, and what sequence
   of claims gets them there?
3. **A section flow.** Sections and subsections, each with a line on what it is
   for and how it hands off to the next.
4. **Title candidates.** Several of them. Foreground the algorithm and hint at
   the state of the field. Propose your own rather than commenting on mine.

Then, more loosely:

5. **Brainstorm what we have missed.** Framings, analyses, figures, further cuts
   of the results we already have, or experiments we have not considered.
   Speculative ideas are welcome, this is the part where I want range rather
   than caution.
6. **Where is this plan most exposed** to a sceptical reviewer?
7. **Anything in the results being undersold or oversold?**
