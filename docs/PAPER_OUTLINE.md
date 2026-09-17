# Paper outline

ICLR 2026. Structure loosely follows NeurIPS convention. Each entry carries a
one-line goal and a flow. Claims referenced here are held in
[`PAPER_ARGUMENTS.md`](PAPER_ARGUMENTS.md).

Working method name **SPARC** (SPecialist-Anchored Regret Constraint).

---

## 1. Introduction

**goal** Convince the reader the benchmark is blind before we mention our method.

**flow** Continual RL exists to prevent forgetting → on the standard benchmark
every method scores near ceiling → ours and naive fine-tuning differ by 0.001 →
name the two structural causes, per-task heads and tasks as modes of one game →
we relax each separately → the ranking inverts both times → introduce SPARC,
competitive under both crutches and first without either → contributions.

## 2. Related Work

**goal** Place us among retention methods and place the critique among evaluation work.

### 2.1 Retention in continual RL
**goal** Show the three existing families and the gap a constraint fills.
**flow** Replay (CLEAR) → regularisation → modular growth (ProgNet, PackNet,
CompoNet, CKA-RL) → all three pay in storage or capacity → none formulates
retention as an explicit constraint.

### 2.2 Plasticity methods
**goal** Establish plasticity and retention as different problems, setting up our negative result.
**flow** Loss of plasticity as a distinct failure → CbpNet, CReLUs → they target
learning ability, not interference → we test whether the distinction survives
empirically.

### 2.3 Evaluation protocols
**goal** Show benchmark critique is an accepted genre and that the head confound is unexamined in RL.
**flow** Saturation critiques in supervised continual learning → continual RL
suites and CORA → per-task heads inherited from supervised multi-head protocols
→ no prior work isolates the head as the confound in RL.

## 3. Problem Setting

### 3.1 Continual RL with a shared policy
**goal** Fix notation and make clear what is deployed.
**flow** Task sequence → shared parameters → the deployed policy is the global →
per-task heads named as a protocol choice, flagged now because it becomes the
experimental variable.

### 3.2 Metrics
**goal** Define PERF, forgetting, BWT and FWT once, with normalisation.
**flow** The forgetting matrix → four scalars read off it → normalisation
against random and ceiling → why FWT needs a from-scratch reference.

### 3.3 The persistent simulator assumption
**goal** State our extra assumption up front and show it is cheap.
**flow** We re-enter past environments during consolidation → this is stronger
than replay, say so plainly → but every method's evaluation already
re-instantiates past tasks to build the matrix → so the access exists in the
setup and the only question is whether the algorithm may use it → name the
setting.

## 4. Method

### 4.1 Local and global policies
**goal** Motivate two policies rather than one.
**flow** A single policy must both learn and retain → split the roles → local
specialises on the current task, global is deployed → local initialises from
global each phase.

### 4.2 The specialist shortfall constraint
**goal** Give the constraint and justify its form.
**flow** Define the shortfall → one-sided because beating the specialist is not
a violation → squared for smooth gradients near zero → epsilon in squared-value
units → the global must stay within epsilon of every specialist it has seen.

### 4.3 Primal-dual optimisation
**goal** Show how the constraint is enforced.
**flow** Lagrangian → dual ascent on the multiplier → alternation between local
and global phases → the actor coefficient as a differentiated hinge → the
multiplier resets each phase.

### 4.4 Implementation
**goal** Record the two choices that are load-bearing.
**flow** Actor-coefficient renormalisation and why an unbounded multiplier
starves the shared critic through the grad-norm clip → PPO details, KL early
stop on the local only → pseudocode box.

## 5. Evaluation Design

### 5.1 What the standard benchmark cannot separate
**goal** Quantify the saturation.
**flow** Ceilings near 0.99 → six methods inside a 0.089 band → the 0.001 gap
between third place and naive fine-tuning → conclude the benchmark orders
methods without separating them.

### 5.2 Two controlled relaxations
**goal** Present the tiers as a factorial, not as difficulty.
**flow** Name the two properties → tier 2 removes the head and holds relatedness
→ tier 3 removes relatedness and holds the head → factorial table → state the
prediction, that either relaxation alone should re-expose forgetting.

## 6. Experiments

### 6.1 Setup
**goal** Get budgets, seeds and reimplementation disclosures out of the way once.
**flow** Three tiers, networks, budgets → seed counts per tier stated plainly →
which baselines are our reimplementations → the per-task budget asymmetry.

### 6.2 The standard benchmark
**goal** Establish that we are competitive and that competitive means little here.
**flow** Table 1 with our row → third of eleven, winning no column → the band is
0.089 wide → fine-tuning sits 0.001 behind us → read this tier as calibration,
not as a result.

### 6.3 Shared head, fifty tasks
**goal** The method claim, and the first inversion.
**flow** Lead with learned-vs-retained, 91% against 13% → aggregate table, ours
ahead by 0.126 with no interval overlap → retention curve, flat for ours and
falling for every baseline → concede ours learns each task to 0.25 against 0.65
→ defend with the headroom table, positive for ours in every bin and negative
for every baseline in every bin → note CbpNet falling below fine-tuning.

### 6.4 Five different games
**goal** Show the second inversion holds when tasks share nothing.
**flow** Five games, disjoint state spaces, one head each → both orders reported
→ forgetting matrices for four methods → ours against CLEAR, CKA-RL, CompoNet →
conclude interference reappears exactly where tasks stop being modes.

### 6.5 Ablations
**goal** Isolate the objective from the environment access.
**flow** Joint multi-task at 50 tasks with matched access and budget → the
constraint removed, multiplier fixed at zero → epsilon sweep → conclude access
is necessary but not sufficient.

## 7. Discussion

### 7.1 Ranking instability
**goal** State the finding that does not depend on our method.
**flow** Rank correlation between tiers is near zero → only fine-tuning holds
its rank, and it does so by being mediocre everywhere → implication, a
single-benchmark claim in continual RL does not transfer.

### 7.2 Plasticity does not buy retention
**goal** Report the negative result cleanly.
**flow** CbpNet and CReLUs both improve on baseline at tier 1 → at 50 tasks
CbpNet is below fine-tuning on both performance and forgetting → CReLUs is a
modest real gain → neither improves backward transfer → the two problems are
separate, and our constraint makes no plasticity claim.

## 8. Limitations

**goal** Own every attack before a reviewer finds it.

**flow** Order sensitivity, universal across every method tested including ours,
documented and out of scope → the value-versus-score gap on Breakout → unmatched
per-task budgets and what they cost us on forward transfer → compute at 1.4x to
1.8x → single seed at two tiers, with the rebuttal-period commitment → the
persistent simulator assumption restated.

## 9. Conclusion

**goal** Leave the reader with the diagnosis, not the method.

**flow** Benchmarks hid forgetting → two controlled relaxations re-expose it →
the ranking inverts under either → SPARC survives both → evaluation design, not
method design, is the field's current bottleneck.

---

## Appendix

Derivation of the constraint and the dual update; full hyperparameters per tier;
per-task score tables; complete forgetting matrices for every method and order;
reimplementation details for CKA-RL, CbpNet and CReLUs in the shared-head
setting; the full order-sensitivity study.
