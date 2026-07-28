# Atari continual-RL findings log (paper reference)

Living record of the **interesting findings** on the Atari track of the constrained
two-policy min-max continual-RL method, the **experiment** that produced each, and the
**measurement** we used. Written to be mined when drafting the paper. Newest structural
findings toward the end. Numbers are greedy, deterministic, and reproducible unless noted.

Method in one line: a *local* policy learns the current task; a *global* policy
consolidates all past tasks under a one-sided squared **value-shortfall** constraint
(dual μ), replay-free (fresh rollouts in old envs), ω_i = 1/k, θ⁰ = φ. PPO is the
optimizer backend; only the actor is constrained.

---

## Part 1 — Experimental setup and methodological choices ("measures taken")

These are the design decisions that make the numbers trustworthy; several are findings in
their own right (marked ★) because they were forced by a bug or a cost wall.

| # | Choice | Why it matters |
|---|---|---|
| S1 | **Impala-CNN (large), ~4.36M params.** 3 conv sequences (32,64,64), each conv3×3 → maxpool3×3/2 → 2 residual blocks, FC 512. Obs `(4,84,84)`. | Enough capacity that under-consolidation is not trivially a size problem. |
| S2 | **Shared trunk + per-task heads** (`impala_ac_multihead`, `task_conditioned=False`). Full 18-action ALE space shared by all games. | The **shared conv trunk is where cross-game forgetting lives**; the constraint's job is to protect it. Heads are per-task. |
| S3 | **10 games, one shared init**: Pong, Breakout, Boxing, Freeway, SpaceInvaders, Qbert, Assault, Krull, Seaquest, BeamRider. | Fixed continual order; diverse episode lengths (short Pong → long Breakout/Qbert). |
| S4 | **Precomputed single-task experts**, trained once to convergence with **no iteration cap**, reused as *fixed local references* (policy + value). | Removes local-training variance from the study; experts are a stable yardstick. Cached in `expert_refs.json`. |
| S5 ★ | **Greedy-only evaluation** (argmax), never stochastic, everywhere (matrix, probes, thresholds). | Low-variance, reproducible across methods/seeds/checkpoints. Binding project rule. |
| S6 ★ | **Deterministic no-op enumeration.** With `repeat_action_probability=0` + greedy, a rollout is *deterministic given its no-op count*; enumerating no-op counts 1…50 gives the **exact** expected greedy value, **zero variance**, in ~50 rollouts. | Replaces hundreds of sampled episodes; exact, cheap, reproducible. `evaluate_greedy_noop_enumerated`. |
| S7 ★ | **No per-episode step cap in training** (`max_steps=0`). | A `TimeLimit` + `done = term \| trunc` makes GAE **bootstrap 0 at truncation**, poisoning the value target — this silently broke Breakout training. ALE-v5 has no internal limit. |
| S8 | **Eval-only episode cap (3000 steps).** `TimeLimit` on the *eval* env only. | Safe (no GAE at eval); with γ<1 the discounted V is unchanged (γ has already killed the tail); bounds enumeration cost on long-episode experts. |
| S9 | **Relative constraint** `V_G / V_L ≥ 1−δ` (shortfall = gap/\|ref\|, ε = δ²). | Scale-normalizes across games so a 20% drop on Pong and on Qbert count equally (see F7). |
| S10 | **Past-task sampling**: one random past task per iter, rescaled by count. | Unbiased O(1) estimate of the Σ_{i<k} past-task term; turns an O(k) collection into O(1). |
| S11 | **Reward sign-clipping** for the *training* stream; **raw** score preserved (`RecordEpisodeStatistics` below the clip) for reporting. | Keeps discounted values O(1)-comparable across games; reported metric stays the true game score. |
| S12 | **PPO port is optimizer-only.** One reusable `PPOTrainer` → Local/Global; the CL framework (constraint, duals, fresh past-task rollouts) is identical to the REINFORCE version; only the actor surrogate is constrained; critic/GAE/entropy standard. | The method is unchanged; PPO only lowers gradient variance. |

---

## Part 2 — Findings

### F1. The min-max **under-consolidates**: forgetting during later tasks, not failure to learn
- **Finding:** After all 10 tasks, only the most-recent games (and Krull) are retained; the
  early games collapse to ~random (Pong −2%, Boxing −0%, Breakout 2%, Qbert 2%). Crucially,
  the **diagonal is strong** — each game *is* learned well when it is the current task
  (Breakout 90, SpaceInvaders 735, Qbert 3456, Krull 8508, Seaquest 1432). So the loss is
  **forgetting during subsequent consolidation**, which the value constraint is supposed to
  prevent and does not.
- **Experiment:** full 10-task baseline consolidation run (`consolidate10_seed0`).
- **Measure:** lower-triangular retention matrix, greedy no-op-enumerated raw scores, global
  score / expert score per seen game. (See `diagnostics/consolidation/consolidate10_seed0/`.)

Final retention (global / expert, %):

| Pong | Breakout | Boxing | Freeway | SpaceInv | Qbert | Assault | Krull | Seaquest | BeamRider |
|---|---|---|---|---|---|---|---|---|---|
| −2% | 2% | −0% | 37% | 21% | 2% | 19% | **121%** | 13% | 37% |

### F2. ★ The **value-vs-score mismatch** (the central finding)
- **Finding:** The constraint protects the **discounted value** V (γ=0.99), but the reported
  metric is the **undiscounted full-episode score**. The global can *satisfy the V constraint*
  while *failing the score*: on Breakout it drove `V_G / ref → 0.98` (constraint essentially
  met) while the greedy game score was ~56 vs the expert's 257. The gap is **largest on
  long-episode games** and small on short ones.
- **Experiment:** traced the adapt variant on task-2 Breakout through consolidation.
- **Measure:** compared the discounted **V-ratio** (`V_G/ref`) against the raw **score-ratio**
  over training. They diverge.
- **Implication:** the metric we optimize (discounted V) and the metric we report (score) are
  not the same quantity on long horizons. This reframes the whole problem.

### F3. γ sets an **effective horizon** that explains F2
- **Finding:** γ=0.99 ⇒ effective horizon 1/(1−γ) = **100 agent-steps** (~400 frames), reward
  half-life ~69 steps, 1% weight by ~460 steps. Episodes run *thousands* of steps, so V only
  "sees" the opening. γ=0.999 ⇒ horizon **1000 steps** (~4000 frames), spanning essentially a
  full (3000-capped) episode.
- **Measure:** analytic (discount weighting); corroborated by which games mismatch (F4).

### F4. Short-episode games are retained; long-episode games bleed
- **Finding:** **Pong is retained near-perfectly (100%) through task 9**, collapsing only at
  task 10. Because Pong is short, its discounted V ≈ its full score, so the constraint that
  holds V also holds the score. Long games (Breakout, Qbert) decay steadily
  (Breakout 35→24→…→2%; Qbert 19→4→2→1% down the retention column).
- **Measure:** retention-matrix columns over tasks.
- **Implication:** direct empirical signature of F2/F3 — the mismatch is episode-length-driven.

### F5. Krull is **anomalously over-retained** (>expert)
- **Finding:** Krull holds 199%→134%→**121%** of its own expert across later tasks — above
  expert. Its reward structure evidently aligns with whatever representation the shared trunk
  settles into.
- **Measure:** retention-matrix Krull column. (Flagged as an outlier worth a sentence.)

### F6. The "adapt" variant retains **less**, not more
- **Finding:** Adding an unconstrained adapt phase + expert head warm-start + looser ε made
  retention *worse* (e.g. **Freeway 90% → 4% in a single subsequent task**). Two mechanisms:
  (a) **looser ε (30% vs 5%)** explicitly permits a 6× larger drop before the constraint bites;
  (b) the **1000-iter unconstrained adapt phase overwrites the shared trunk before any
  protection applies**, and the loose consolidation never restores it.
- **Experiment:** adapt run (`consolidate10_adapt_seed0`) vs baseline, through task 5.
- **Measure:** per-task retention rows.
- **Implication:** retention is governed by (i) how tightly the constraint binds (ε) and
  (ii) never leaving the shared trunk unprotected — not by how well the *current* head starts.

### F7. The relative constraint fixes **scale**, not retention
- **Finding:** `V_G − V_L → V_G / V_L` only equalizes cross-game priority (Qbert's 18000 vs
  Pong's 20). It does **not** strengthen the leash or change *what* is measured. Both baseline
  and adapt runs already used it; it was never the retention lever.
- **Measure:** conceptual + config audit (`constraint_relative: true` in both).

### F8. Head warm-start helps **learning**, not **retention**
- **Finding:** Warm-starting task-k's head from its expert improves the *diagonal*
  (faster mastery of the current task) but cannot protect past tasks: (a) forgetting lives in
  the **shared trunk**, which no head touches; (b) the expert head was trained for the
  **expert's** trunk, so on the global's shared trunk it is a representation mismatch whose
  advantage decays immediately.
- **Measure:** adapt-run diagonal (good) vs its retention (poor).

### F9. ★ **Action-agreement is uninformative in Atari** — measure *value*, not actions
- **Finding:** Over states the expert visits, the global's greedy action matches the expert's
  only **~6%** of the time (chance for 18 actions) — *even the just-learned global that scores
  90 on Breakout*. Most Atari frames are **"don't-care"** (ball mid-flight; any action → same
  outcome), so two competent policies disagree on nearly every frame. Policy *match* must be
  measured by **value/return**, not action overlap.
- **Experiment:** Phase A — roll expert greedily over its own trajectory; per visited state
  compare global's greedy action to expert's.
- **Measure:** fraction of states with matching argmax, bucketed early/mid/late. Flat ~random.

### F10. ★ Retention failure is **concentrated in the mid/late game**
- **Finding:** The global's value tracks the expert at the **start** of an episode and
  **progressively falls short** deeper in. Breakout median `V_global / expert-return-to-go`
  decays **0.74 → 0.41 → 0.14** (early → mid → late; both just-learned and final checkpoints).
  Because our standard eval uses **50 no-op *starts*** (early game only), it samples exactly the
  region where the global looks best (~0.75) and **structurally misses** the late-game failure
  (~0.14). This is the mechanistic explanation of F2.
- **Experiment:** Phase A — same expert-trajectory states, bucketed early/mid/late, filtered to
  return-to-go > 0; compare global critic value to expert discounted return-to-go **per state**.
- **Measure:** median per-state value ratio per bucket. (`diagnostics/phase_a/consolidate10_seed0/`.)
- **Caveats:** value here is the global's **critic estimate** (Phase B will confirm with the
  global's *true* rollout return from restored states); the clean early→late gradient is shown
  on **Breakout** (partially retained). **Qbert-final** is *uniformly* collapsed (~3–11%) — it
  is simply fully forgotten, so no gradient remains.

Phase A value ratios `V_global / expert-rtg` (median):

| game / ckpt | early | mid | late |
|---|---|---|---|
| Breakout · just-learned | 0.74 | 0.41 | **0.14** |
| Breakout · final | 0.77 | 0.51 | **0.25** |
| Qbert · just-learned | 0.14 | 0.17 | 0.52 |
| Qbert · final | 0.06 | 0.03 | 0.11 |

### F11. Intermediate states cannot be **averaged** into the ratio constraint
- **Finding:** If mid/late states are added to the constraint's V by averaging, two problems
  arise: **(a) heterogeneity** — early states carry large return-to-go, late states small, so a
  ratio-of-means is dominated by early states and re-hides the late game; **(b) near-zero
  denominators** — terminal-adjacent states have expert rtg ≈ 0, so per-state ratios blow up.
  Intermediate-state retention must be measured **case-by-case, per game-phase bucket, filtered
  to rtg > 0** — not folded into one averaged V.
- **Measure:** analysis of the relative shortfall under a mixed state set. Guided the Phase A/B
  design (diagnostic-first, bucketed).

### F12. (Earlier diagnostics, pre-consolidation) the constraint is **too weak** + trunk interference
- **Finding:** A scalar value constraint is insufficient to pin behavior (value can be matched
  while the policy differs — EXP2, closed by behavior cloning); and the shared trunk causes
  cross-task interference (EXP1). These motivated the value-vs-score investigation above.
- **Experiment:** EXP1 (joint feasibility / capacity), EXP2 (value-constraint sufficiency, KL /
  behavioral-cloning), head-only probe (trunk vs objective), per-task gradient cosine.
- **Measure:** KL gap local↔global; per-task cos(g_new, g_i); joint upper bound.
- **Note:** BC/KL deliberately deferred; the current thread is fixing the mismatch **inside the
  value formulation** first (γ, score reference).

### F14. ★ Actual-rollout confirmation (Phase B) + ratio instability shows up in *measurement*
- **Finding:** Using the global's **true discounted return** from random intermediate states
  (reached by exact deterministic re-simulation to s_t; replay self-check **bit-exact**), the
  early→late decay is confirmed on **partially-retained** games:
  Pong **0.33 → −0.20 → −0.50** (goes *negative* — the global loses points the expert wins),
  Breakout **0.63 → 0.58 → 0.26**. **Fully-forgotten Qbert** is uniformly low (global return
  ~2–8 vs expert ~9–24 everywhere); its late-bucket ratio (0.31) is **inflated by the smaller
  expert return-to-go at the tail**, not real skill — an empirical instance of F11: **ratios
  destabilize even as a measurement** when the denominator shrinks.
- **Experiment:** Phase B (`experiments/phase_b_rollout.py`), 45 random intermediate starts per
  game, global's actual rollout return vs expert return-to-go, bucketed early/mid/late, rtg>0.
- **Measure:** median per-state ratio **and** mean absolute returns per bucket — prefer the
  **absolute means**; per-state ratios are noisy/skewed.
- **Implication:** (a) the retention gap is genuinely concentrated mid/late where a game is
  *partially* held (uniform collapse once fully forgotten); (b) report **absolute** per-bucket
  returns, not ratios; (c) independently reinforces F15's switch away from ratio constraints.

### F15. Ratio constraint is numerically unstable → floored additive hinge
- **Finding:** The normalized-shortfall constraint `(V_L−V_G)/max(|V_L|, ε)` carries **1/|V_L|
  in the gradient**, so near-zero-value tasks explode: with Pong's `V_L = 0.031` the constraint
  gradient coefficient is **34×** a unit-value game (→ 1000× if `|V_L|` hits the 10⁻³ floor).
  Replacing it with an **additive floored hinge**
  `loss = (max(0, (V_L−V_G) − δ·max(|V_L|, τ)))²` keeps the **same feasibility boundary**
  (allow a δ fractional drop) but puts the gradient coefficient in **raw value units — no
  division**, so it is stable. Trade-off: gradient magnitude now scales with raw value
  (Qbert coeff 12× Pong), i.e. bounded scale-dependence instead of unbounded explosion.
- **Experiment:** analytic + numeric check of the coefficient at the measured `V_L` values;
  identified `V_L(Pong)=0.031` as the live failure case.
- **Measure:** gradient coefficient `μ·2·shortfall` under both forms.
- **Implication:** default constraint switched to `constraint_form="floored"` (δ=0.05, τ=0.5);
  ε/relative fields become legacy. Prompted by supervisor feedback.

### F16. ★★ Past-task starvation — the current-task constraint erases old tasks
- **Finding (the pivotal one):** During task k the global actor loss is
  `Σ_{i<k} ω_i·L_i^CLIP + μ·2·shortfall·L_k^CLIP` — past tasks on a **fixed** ω_i=1/k,
  only the **current** task carrying the V-shortfall (+ intermediate states). This
  breaks retention **two ways**: (a) past-task retention has **no adaptive shortfall
  and no intermediate states** — just a blind fixed-weight push; (b) worse, `coeff_k =
  μ·2·shortfall` is **unbounded** (μ up to 1000, shortfall O(30)), and after we
  normalize the actor coefficients by their sum (the F13 critic-starvation fix), a large
  `coeff_k` drives the past-task weight to **~0**. Measured live on the interm run: at
  task-2 step 400, `coeff_k = 18,750`, μ climbing (223→297→447), so **Pong received
  0.003% of the actor gradient — actively forgotten while Breakout was learned**
  (and `V_k_global` oscillated 18.4→3.7, unstable). Almost certainly the mechanism behind
  the baseline's early-task collapse (F1), compounding across 10 tasks.
- **Experiment:** the `consolidate10_strict_g999_interm` run; read `coeff_k`, μ, and the
  implied gradient split `ω / (ω + coeff_k)` live at steps 0/200/400.
- **Measure:** `coeff_k`, μ, normalized past/current gradient share.
- **Implication:** retention cannot work while the current-task constraint can starve
  past tasks. Proposed fix — put the floored shortfall (incl. intermediate states) on
  **all** tasks and weight by **relative** shortfall `w_i = shortfall_i / Σ_j shortfall_j`
  (bounded, sums to 1 → true min-max, worst-retained task pushed hardest, **no
  starvation**). Fixed ω and the unbounded `coeff_k` both go away. Redesign under review.

### F13. Shared-critic starvation (PPO port detail)
- **Finding:** An unbounded dual coefficient can starve the shared critic; normalizing the
  actor coefficients by their sum stabilizes it.
- **Measure:** value-loss / training stability during the port.

---

## Part 3 — Candidate fixes on the table (not yet decided)

Staying **inside the value formulation** (BC/KL deferred):
1. **γ = 0.999** — lengthen the horizon so discounted V spans the whole episode (targets F2/F3).
2. **Score / near-undiscounted reference** — constrain the quantity we actually report.
3. **More constrained optimization, no unconstrained phase** — never leave the trunk unprotected
   (from F6); ε back to 5% (from F6/F7).
4. **Intermediate-state constraint** (from Phase B, F10) — hold the current task at 50
   re-simulated mid/late states (true rollout returns; per-state floored shortfall). Built and
   vectorized (`evaluate_intermediate_values_vec`, exact to 2e-14).
5. **Relative-shortfall min-max over ALL tasks (the F16 fix, under review)** — per-task floored
   shortfall incl. intermediate states for every task (past + current); weight each task's
   surrogate by `w_i = shortfall_i / Σ_j shortfall_j` (bounded, sums to 1). Makes retention
   adaptive + mid/late-aware AND removes the unbounded-`coeff_k` starvation.

**Run history:**
- `consolidate10_strict_g999_interm` (γ=0.999, floored constraint, 50 no-op + 50 intermediate
  states, vectorized eval, live past-task probe) was launched and **stopped at task-2 step 400**
  after F16 was diagnosed live (past-task starvation: `coeff_k`=18,750 → Pong at 0.003% of the
  gradient). γ=0.999 expert refs + intermediate-state caches are precomputed
  (`results/expert_refs_g999.json`, `results/intermediate_states_g999.json`) and reusable.
- **Next:** redesign per fix 5 (relative-shortfall min-max on all tasks) — pending decision.

---

## Artifacts (for reproducibility / figures)

- Configs: `configs/consolidate10.yaml` (baseline), `configs/consolidate10_adapt.yaml`,
  `configs/consolidate10_strict_g999.yaml`, `configs/consolidate10_strict_g999_interm.yaml`.
- Consolidation figures + retention table: `diagnostics/consolidation/consolidate10_seed0/`
  (`retention_matrix`, `retention_curves`, `final_retention_bars`, `consolidation_dynamics`,
  `mu_constraint`, `ppo_health`, `retention_triangular.md`).
- Phase A: `experiments/phase_a_agreement.py`, `diagnostics/phase_a/consolidate10_seed0/`.
- Phase B: `experiments/phase_b_rollout.py`, `diagnostics/phase_b/consolidate10_seed0/`.
- Precompute: `experiments/compute_expert_values.py` (γ=0.999 refs),
  `experiments/compute_intermediate_states.py` (per-game mid/late states).
- Report generators: `experiments/baseline_report.py`, `experiments/consolidation_retention.py`.
- Eval core: `crl/ppo/evaluate.py` — `evaluate_greedy_noop_enumerated`,
  `evaluate_intermediate_values` / `_vec` (per-state true-return from re-simulated states).
- Trainer / orchestrator: `crl/ppo/trainer.py` (GlobalTrainer, floored + intermediate
  constraint), `crl/ppo_continual.py` (live past-task probe).
- Paper-side changes to reflect: `docs/paper_updates_needed.md`.
