# Paper outline

ICLR 2026. Structure loosely follows NeurIPS convention. Each entry carries a
one-line goal and a flow. Claims referenced here are held in
[`PAPER_ARGUMENTS.md`](PAPER_ARGUMENTS.md).

Working method name **DUEL**, Dual-Policy Expert-Anchored Learning.

**Revised 2026-09-17.** The algorithm is the contribution. The evaluation
design is instrumental, it exists so the constraint can be seen working, and it
returns as a finding in Section 7 rather than opening the paper. The previous
diagnosis-first outline is in git history.

Rough budget for nine pages: intro 1, related 0.75, setting 0.75, method
**2.25**, setup 0.5, results 2.5, analysis 0.75, limitations 0.4, conclusion 0.2.

---

## 1. Introduction

**goal** Make the reader want a retention constraint before they see one.

**flow** Continual RL is supposed to keep what it learned → it does not, once a
task's solution has to share capacity with the next one → existing answers
either store data (replay) or spend parameters (modular growth), and neither
states retention as a requirement the optimiser must satisfy → we do, as an
explicit per-task constraint on the deployed policy's shortfall against a
expert → the resulting min-max problem is solved by primal-dual alternation
→ one honest complication, the standard benchmark cannot show this working,
because per-task heads and same-game tasks suppress the interference, so we
also evaluate where interference is real → contributions.

## 2. Related Work

**goal** Position the constraint against the two existing families and against
constrained RL.

### 2.1 Retention in continual RL
**goal** Show what the three existing families pay, and what none of them states.
**flow** Replay (CLEAR) pays in storage → regularisation pays in a fixed penalty
weight chosen before you know how much each task will be hurt → modular growth
(ProgNet, PackNet, CompoNet, CKA-RL) pays in parameters → none of the three
makes retention a constraint the optimiser has to satisfy.

### 2.2 Plasticity methods
**goal** Separate the problem we solve from the one they solve.
**flow** Loss of plasticity as a distinct failure → CbpNet, CReLUs → they restore
the ability to learn, not the ability to hold → we make no plasticity claim, and
we test whether the distinction survives at fifty tasks.

### 2.3 Constrained RL and primal-dual methods
**goal** Ground the machinery so the method reads as standard, not exotic.
**flow** Constrained MDPs and Lagrangian RL → safety constraints as the usual
application → our departure, the constraint is against a *learned, per-task
comparator* that moves, not a fixed budget.

## 3. Problem Setting

### 3.1 Continual RL with a shared policy
**goal** Fix notation and name what is deployed.
**flow** Task sequence → shared parameters → the deployed policy is the global
one → per-task heads flagged as a protocol choice, because it later becomes the
experimental variable.

### 3.2 The persistent simulator assumption
**goal** State the extra assumption up front and show it is cheap.
**flow** We re-enter past environments during consolidation → stronger than
replay, say so plainly → but every method's evaluation already re-instantiates
past tasks to build the forgetting matrix, so the access is in the setup for
everyone → the only question is whether the algorithm may use it → name the
setting.

### 3.3 Metrics
**goal** Define PERF, forgetting, BWT and FWT once, with normalisation.
**flow** The forgetting matrix → four scalars off it → normalisation against
random and ceiling → why FWT needs a from-scratch reference.

## 4. Method

The centre of the paper. Everything before it is setup and everything after is
evidence.

### 4.1 Two policies, local and global
**goal** Motivate splitting the roles rather than tuning one policy harder.
**flow** A single policy must both acquire and hold, and the two pull opposite
ways → split them → local specialises on the current task, global is deployed →
local initialises from global each phase, so the expert is always reachable.

### 4.2 The expert shortfall constraint
**goal** Give the constraint and justify every choice in its form.
**flow** Define the shortfall of the global against task *k*'s expert →
one-sided, because beating the expert is not a violation and penalising it
would cap the method at the expert → squared, for a smooth gradient near
zero → epsilon in squared-value units → the feasible set is every policy within
epsilon of every expert seen so far.

### 4.3 Primal-dual optimisation
**goal** Show how the constraint is enforced, not merely stated.
**flow** Lagrangian → projected dual ascent on the multiplier → alternation
between local and global phases → the actor coefficient as a differentiated
hinge → the multiplier resets each phase, and why.

### 4.4 Why a constraint and not a penalty
**goal** Defend the central design choice, which is the reviewer's first question.
**flow** A fixed regularisation weight is chosen before you know how much each
task will be hurt → the dual multiplier is the weight the problem selects for
itself → the one-sided hinge means a task already within epsilon contributes
exactly nothing, so effort concentrates where the shortfall is → contrast with
uniform joint training, which spends equally on tasks that need nothing.

### 4.5 Implementation
**goal** Record the choices that are load-bearing rather than incidental.
**flow** Actor-coefficient renormalisation, and why an unbounded multiplier
starves the shared critic through the grad-norm clip → PPO details, KL early
stop on the local only → pseudocode box → cost, and what consolidation adds.

## 5. Experimental Setup

**goal** Justify three settings in half a page, without arguing benchmark reform.

### 5.1 Three settings, and why the standard one is not enough
**goal** Explain why a third-place finish on the usual benchmark settles nothing.
**flow** The established benchmark gives each task its own head and draws tasks
as modes of one game, so interference is structurally suppressed → its ceilings
sit near 0.99 and six methods land inside 0.089, with third place 0.001 from
naive fine-tuning → a constraint on retention cannot be seen working there →
so we add two settings, one removing the shared-head suppression and one
removing the same-game suppression, holding the other fixed in each.

### 5.2 Baselines, budgets and seeds
**goal** Get every disclosure out of the way once, in one place.
**flow** Baselines per setting → which are our reimplementations → seed counts,
stated plainly per tier → the per-task budget asymmetry and the frame counts →
compute.

## 6. Results

**Order settled 2026-09-17.** Established benchmark first, then GridWorld, then
Atari, over a suggestion to lead with GridWorld. Table 1 first sets up a
question, that the benchmark cannot separate this method from naive fine-tuning,
and 6.2 is the payoff that answers it. It also establishes standing on the
field's own benchmark before we ask the reader to accept two settings of our own.

Emphasis on GridWorld is carried by four other levers rather than by order.
- **The abstract quotes GridWorld's numbers**, 91% against 13% and the only
  positive backward transfer in the comparison. Table 1 gets one clause.
- **The learned-vs-retained scatter is Figure 1**, placed in the introduction.
  Reviewers read figures before sections, which makes this the strongest lever.
- **6.1 runs a third of a page** against 6.2's full page. Length signals
  importance more reliably than position does.
- **The section title and first sentence pre-frame it** as a compatibility
  check, and it closes with an explicit pointer forward to 6.2.

### 6.1 Compatibility with the established benchmark
**goal** Establish standing, and pose the question 6.2 answers.
**flow** We first confirm the method is sound where the field measures → Table 1
with our row, third of eleven, within 0.026 of the best, winning no column → the
top six span 0.089 and naive fine-tuning sits 0.0011 behind us → so this
benchmark cannot separate a method built to prevent forgetting from one that
does nothing about it → point forward to 6.2, which runs the same two methods
with the heads shared.

### 6.2 Shared head, fifty tasks
**goal** The main result.
**flow** Lead with learned-vs-retained, 91% of tasks end better against 13% for
the nearest baseline → aggregate table, ahead by 0.126 with no interval overlap
across three seeds → retention curve flat for ours and falling for every
baseline → concede ours learns each task to 0.25 against 0.65 → defend with the
headroom table, positive for ours in every bin and negative for every baseline
in every bin → note CbpNet falling below plain fine-tuning.

### 6.3 Five different games
**goal** Show the constraint holds when tasks share no structure at all.
**flow** Five games, disjoint state spaces, one head each → both orders reported
→ retention matrices for four methods → the two baseline failures are opposite,
CKA-RL forgets to below random while CompoNet freezes and never learns Q\*bert →
ours is the only method **without an expanding architecture** that both retains
and keeps learning. CompoNet retains better, 0.83 against our 0.67, by expanding
and by never learning the final game. Keep the qualifier.

### 6.4 Ablations
**goal** Isolate the objective from the environment access.
**flow** Joint multi-task at fifty tasks with matched access and budget → the
constraint removed, multiplier fixed at zero → epsilon sweep → conclude the
access is necessary but not sufficient, and the constraint is what converts it.

## 7. Analysis

### 7.1 Where the advantage comes from
**goal** Explain the mechanism, not just the margin.
**flow** Consolidation does most of the learning, the local phase is short →
with related tasks there is transfer to harvest and the constraint decides where
to spend → read the dual multiplier traces against the tasks that were at risk.

### 7.2 What the standard benchmark cannot show
**goal** Report the evaluation finding, now as a result rather than a thesis.
**flow** Rank correlation between the benchmark and the shared-head setting is
near zero → only fine-tuning holds its rank, by being mediocre in both → so a
single-benchmark claim in continual RL does not transfer, which is why Section 5
added two settings.

### 7.3 Plasticity and retention are different problems
**goal** Report the negative result cleanly and briefly.
**flow** CbpNet and CReLUs both improve on baseline in the standard setting → at
fifty tasks CbpNet is below fine-tuning on performance and on forgetting →
CReLUs is a modest real gain → neither improves backward transfer.

## 8. Limitations

**goal** Own every attack before a reviewer finds it.

**flow** Order sensitivity, universal across every method tested including ours,
documented and out of scope → the value-versus-score gap on Breakout, where the
constraint was satisfied on V while the greedy score collapsed → unmatched
per-task budgets and what they cost on forward transfer → compute → single seed
at two tiers, with the rebuttal-period commitment → the persistent simulator
assumption restated.

## 9. Conclusion

**goal** Leave the reader with the constraint.

**flow** Retention stated as a constraint rather than hoped for as a side effect
→ primal-dual alternation solves it without replay → it holds fifty shared-head
tasks and five unrelated games where the alternatives do not → and the settings
needed to see that are themselves worth having.

---

## Appendix

Derivation of the constraint and the dual update; full hyperparameters per tier;
per-task score tables; complete retention matrices for every method and order;
reimplementation details for CKA-RL, CbpNet and CReLUs in the shared-head
setting; the full order-sensitivity study.
