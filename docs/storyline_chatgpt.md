# ICLR storyline feedback and method-name notes

This note collects all suggestions made in the discussion. The current authoritative method description is the one in `storyline.md`; the older objective sketch is intentionally not used here.

## Overall assessment

The paper has a coherent method-first story:

> SPARC (or its eventual new name) makes retention an explicit, adaptive per-task constraint rather than a fixed regularization trade-off. Its advantage becomes visible when the benchmark actually permits parameter interference.

The proposed method is a local-adaptation/global-consolidation procedure. A local policy specializes on the incoming task; the deployed global policy is then consolidated under one-sided specialist-referenced retention constraints. Primal-dual updates make only the at-risk constraints active. The central empirical claim is not universal superiority on every existing benchmark: it is that conventional task heads and highly related tasks can mask interference, whereas the shared-head setting reveals a large retention advantage.

## Contribution list

1. Introduce a constrained continual-RL method that separates fast local task adaptation from global consolidation, and uses primal-dual optimization to make retention an active requirement rather than a uniformly weighted regularization term.
2. Use one-sided, task-specific retention constraints so the global policy is protected from falling behind task specialists but remains free to exceed them and obtain positive backward transfer.
3. Use adaptive dual multipliers to direct corrective effort toward tasks that are currently at risk rather than treating all prior tasks uniformly.
4. Evaluate in controlled continual-RL settings that separately remove per-task output heads and task relatedness, two benchmark properties that can suppress interference.
5. Show that the method is competitive on the established benchmark and, under shared-head interference, substantially improves retention and is uniquely associated with positive backward transfer.
6. Show in the unrelated Atari sequence that it avoids the two opposite failure modes of severe forgetting and no-longer-learning plasticity.

## Recommended paper storyline

Continual RL requires adaptation to incoming tasks while preserving prior competence. Existing approaches encode that balance indirectly: replay stores history, regularization uses a fixed penalty weight, and modular methods protect knowledge by adding or freezing capacity. The proposed method instead treats retention as an explicit task-wise requirement. A local policy learns the new task; a global policy consolidates it while staying within an allowed gap of prior task specialists. The dual variables increase pressure only on violated tasks, focusing consolidation where retention is actually threatened.

The experiments establish why this mechanism should be evaluated in settings with real interference. The method is competitive on the conventional benchmark but that benchmark barely separates it from naive fine-tuning. Removing per-task heads produces a sharp separation: the proposed method retains more, improves more past tasks, and yields positive backward transfer while every comparison method has negative backward transfer. The Atari sequence then illustrates the stability-plasticity dilemma task by task: CKA-RL forgets, CompoNet preserves but can fail to learn, and the proposed method balances both.

## Section flow

| Section | Purpose and handoff |
|---|---|
| Introduction | Motivate adaptive retention constraints and state the persistent-simulator assumption early. |
| Related work | Organize replay, regularization, modularity, and constrained RL; identify task-specific adaptive constraint enforcement as the gap. |
| Problem setting | Define local/global policies, specialists, retention, metrics, and the persistent-simulator setting. |
| Method | Present the constraint, one-sided hinge, Lagrangian, primal-dual optimization, and implementation. |
| Why benchmark structure matters | Explain how per-task heads and task relatedness can suppress interference; introduce the factorial evaluation logic. |
| Experimental protocol | Specify settings, metrics, baselines, budgets, and seed counts. |
| Results: shared-head GridWorld | Lead with the central fifty-task result: performance, forgetting, backward transfer, and the 91% task-improvement statistic. |
| Results: established benchmark | Present this as a compatibility/sanity result, not the headline empirical test. |
| Results: Atari | Show the stability-plasticity failure modes with task-level outcomes, not only aggregate averages. |
| Ablations and analysis | Attribute gains to the constraint rather than merely old-task access; show task-order and cost results. |
| Limitations and conclusion | State simulator access, compute, order sensitivity, and forward-transfer limitations precisely. |

## Results order

Keep the paper **method-first**, but lead the Results section with shared-head GridWorld, then the established benchmark, then Atari. This is not a change of framing. It ensures the reader’s first empirical impression is the strongest evidence rather than a third-place table.

## Mandatory work before the deadline

Only these items were identified as essential:

1. **Matched-access control.** Compare against a method with old-task simulator access and a comparable rollout/compute budget but without the proposed retention-constraint mechanism. This answers whether the gain comes from the objective or merely from revisiting old environments.
2. **Multiplier-zero ablation.** Keep the pipeline, simulator access, and local/global design fixed; set the multipliers to zero. This is the direct, inexpensive test that the constraints themselves matter.
3. **Scope and cost up front.** State prominently that the method assumes past simulator access, stores no replay trajectories, and costs roughly 1.4x to 1.8x the cheapest baseline. Do not bury this in limitations.

If Atari remains single-seed, keep it as a diagnostic illustration of distinct failure modes rather than treating it as definitive ranking evidence.

## Reviewer-risk wording

The main reviewer risk is causal attribution: simulator access versus the constrained objective. The matched-access control and multiplier-zero ablation directly answer it.

Avoid the broad claim that conventional benchmarks "do not measure forgetting." Safer wording:

> In our experiments, per-task heads and task relatedness can substantially mask interference that becomes visible under a shared-head design.

Other claim-discipline notes:

- Say the method is the only one to both retain and keep learning **among the evaluated methods**, rather than field-wide.
- Describe the established-benchmark result as competitive/parity rather than as evidence of dominance.
- Present value, score, compute, order sensitivity, and lower forward transfer candidly. They are manageable limitations when scoped early.

## Results and figures to emphasize

- The shared-head result is the empirical centerpiece, especially **91%** of task-seed pairs ending better versus **13%** for the nearest competitor.
- Make the per-task scatter around the no-change diagonal a prominent figure.
- Show a task-by-stage forgetting matrix for SPARC, fine-tuning, CKA-RL, and CompoNet if space permits.
- In Atari, show CKA-RL’s retention failure and CompoNet’s Q*bert plasticity failure at task level. This is more memorable than the aggregate table.
- If available, show multiplier trajectories or active constraints to support the claim that optimization concentrates on endangered tasks.

## Title candidates

- *Constrained Retention for Continual Reinforcement Learning*
- *Adaptive Retention Constraints for Continual Reinforcement Learning*
- *Specialist-Anchored Retention Constraints for Continual Reinforcement Learning*
- *Bounding Forgetting with Adaptive Constraints in Continual Reinforcement Learning*
- *Local Adaptation, Global Retention in Continual Reinforcement Learning*
- *When Forgetting Becomes a Constraint: Continual Reinforcement Learning with Specialist Anchors*

## Method naming discussion

### Current favorite: DUEL

**DUEL** is concise, memorable, and accurately evokes the method’s local/global two-policy design. Its limitation is that it names the architecture more than the primal-dual constraint mechanism.

Possible expansions and titles:

- **DUEL** - *Dual-Policy Equilibrium Learning*
- *DUEL: Dual-Policy Equilibrium Learning for Continual Reinforcement Learning*
- *DUEL: Constrained Dual-Policy Learning for Continual Reinforcement Learning*
- *DUEL: Primal-Dual Retention Constraints for Continual Reinforcement Learning*

The last subtitle is the recommended way to preserve the memorable DUEL name while stating the mechanism precisely.

### Architecture-first alternatives

| Name | Expansion | Emphasis |
|---|---|---|
| **TETHER** | Task-wise Expert-Tethered Retention | The global policy remains tied to specialists without being capped by them. |
| **SENTRY** | Specialist-Referenced Retention | Protection of prior specialist performance. |
| **ANCHOR** | Adaptive Network Constraints for Historical-Policy Retention | Specialist references as retention anchors. |
| **CLASP** | Constrained Local Adaptation and Specialist Preservation | Local adaptation followed by preservation. |

### Mechanism-first alternatives

| Name | Expansion | Emphasis |
|---|---|---|
| **SADDLE** | Specialist-Anchored Dual Dynamics for Lifelong Reinforcement Learning | Saddle-point/primal-dual optimization. |
| **PACE** | Primal-dual Adaptive Constraint Enforcement | Adaptive enforcement of violated constraints. |
| **LACE** | Lagrangian Adaptive Constraint Enforcement | Lagrangian mechanism. |
| **MACE** | Min-max Adaptive Constraint Enforcement | Explicit max-min mechanism. |
| **PACT** | Primal-dual Adaptive Constraint Tuning | A maintained agreement with specialists. |
| **TRACE** | Task-wise Retention through Adaptive Constraint Enforcement | Task-specific retention. |
| **GUARD** | Global Updates with Adaptive Retention Dynamics | Dynamic protection in global consolidation. |
| **AEGIS** | Adaptive Expert-Guided Interference Suppression | Protection from destructive interference. |
| **EQUIP** | EQUilibrium-based Interference Protection | Stability-plasticity equilibrium. |
| **PRIME** | PRImal-dual Memoryless Equilibrium learning | Primal-dual optimization; avoid "memoryless" if simulator access is central. |

### Mechanism-first titles

- *SADDLE: Specialist-Anchored Dual Dynamics for Continual Reinforcement Learning*
- *PACE: Primal-dual Adaptive Constraint Enforcement for Continual Reinforcement Learning*
- *TETHER: Constrained Specialist Retention in Continual Reinforcement Learning*

## Bottom-line recommendation

Use **DUEL** if it remains the name you prefer. It is stronger than the mechanism-first acronyms as a memorable paper identity. Let the title/subtitle carry the full technical content:

> **DUEL: Primal-Dual Retention Constraints for Continual Reinforcement Learning**
