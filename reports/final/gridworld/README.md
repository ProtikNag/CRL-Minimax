# GridWorld: 50 tasks, one shared head

The tier that removes the two crutches the CKA-RL benchmark leans on.

There, the tasks are **modes of one game** and every method gets **its own head
per task**. Both make the problem easier than it looks: performance ceilings sit
near 0.99, so there is almost nothing left to separate methods by, and a
per-task head means a large part of each task's solution never has to share
capacity with anything else.

Here there are **50 genuinely different layouts** — varying obstacle density and
type, slipperiness, and reward and penalty magnitudes on a 50×50 grid — and
**one shared task-conditioned head**. Capacity is fixed, interference is forced,
and forgetting has somewhere to live.

## Status: partial

| Run | Tasks done |
|---|---|
| ours seed 0 | 45 / 50 |
| ours seed 1 | 44 / 50 |
| ours seed 2 | **50 / 50** |
| CKA-RL seed 0 | 41 / 50 |
| Fine-tuning seed 0 | **50 / 50** |
| From-scratch seed 0 | **50 / 50** |

Every figure draws each method to its own last completed phase and says so.
Forgetting and backward transfer for the unfinished runs are **lower bounds**
that will edge up as the last tasks land. Re-running the script picks up new rows
automatically — nothing needs editing when the runs finish.

## Figures

| Stem | What it shows |
|---|---|
| `retention_trajectory` | Mean retention of all prior tasks as the sequence grows. The headline. |
| `headline_metrics` | PERF, forgetting, backward and forward transfer, four methods |
| `forgetting_matrices` | The full 50×50 lower triangle per method |
| `retention_by_age` | Retention against phases-since-learned — the decay curve |

All in `png/` (300 dpi) and `svg/` (vector, verified zero embedded raster).

## What the numbers say

Normalisation is `(score − random) / (1 − random)`, so **0 is a random policy and
1 is a solved task**. Random is per-task and sits around 0.72 raw, which is why
the normalised span is the honest one to read.

| | PERF | Forgetting | BWT | FWT |
|---|---:|---:|---:|---:|
| **Min-Max (ours)** | **0.61** | **0.10** | **+0.37** | 0.09 |
| CKA-RL | 0.39 | 0.30 | −0.26 | **0.18** |
| Fine-tuning | 0.31 | 0.39 | −0.34 | 0.16 |
| From-scratch | 0.18 | 0.52 | −0.47 | 0 (reference) |

Ours roughly **doubles** the next-best average performance and is the only
method with **positive backward transfer**. Every other method ends below where
it started on old tasks; ours ends above.

### The trade, stated plainly

`retention_by_age` is the figure that explains the rest, and it is not flattering
in every direction:

| | just learned (age 0) | age ≥ 10 | change |
|---|---:|---:|---:|
| **Min-Max (ours)** | 0.25 | **0.63** | **+0.38** |
| CKA-RL | 0.64 | 0.42 | −0.22 |
| Fine-tuning | 0.65 | 0.31 | −0.34 |
| From-scratch | 0.64 | 0.17 | −0.47 |

**Ours learns each new task to less than half the immediate level the others
reach**, and then climbs past all of them within two or three phases. The other
three start where a specialist would and decay from there.

So ours' positive backward transfer is partly earned and partly structural: it is
easier to improve on a task you left at 0.25 than one you left at 0.65. That is
worth saying in the paper rather than letting a reviewer find it. The claim that
survives it is the one about **final state** — after 50 tasks ours holds 0.61
against 0.39, 0.31 and 0.18 — and about the shape of the curve, which is flat for
ours and falling for everyone else.

The likely mechanism is that the local phase here is short (150–250 iterations)
and the global consolidation, which maximises return over *all* seen tasks, does
most of the learning. With 50 related layouts there is a great deal of positive
transfer available, and consolidation is what harvests it.

### Forward transfer

Ours is last on FWT (0.09 against CKA-RL's 0.18), and the age-0 column above is
why: forward transfer measures how fast a task is learned in its own phase, and
ours deliberately spends less there. Two things belong in the caption.

**The per-task budgets are not matched.** Ours' local phase runs 150–250
iterations; fine-tuning, from-scratch and CKA-RL run 150–500. Ours uses *more*
total optimisation (≈17.5k iterations against 14k, 15k and 10k) but less of it
inside any single task's own learning phase.

**From-scratch is the FWT reference**, so its value is 0 by construction and it
is omitted from that panel rather than drawn as a competitor.

## Disclosures for any caption

- **Ours re-simulates past environments during consolidation.** CKA-RL,
  fine-tuning and from-scratch train only on the current task. This is live
  past-task environment access and is a stronger assumption than a replay buffer,
  not a weaker one.
- **CKA-RL here is our own reimplementation**, adapted to the shared-head setting
  the original does not target. Say so in the paper.
- **CompoNet is excluded** because it grows the network; every method compared
  here has a fixed footprint (ours a fixed shared head, CKA-RL a fixed trunk plus
  a bounded pool of 5 and a small per-task α).
- **Seeds:** ours has 3, everything else has 1. `headline_metrics` lends ours'
  standard deviation to the single-seed methods as a **placeholder**, drawn
  dotted and uncapped and marked `†` so it cannot be mistaken for a measurement.
  Replace it as their seeds land.

## Rebuild

```bash
python reports/final/gridworld/make_figures.py
```

Reads `reports/gridworld_sharedhead/*/` per `docs/LOGGING_CONTRACT.md`. Nothing
is transcribed or duplicated into this folder.
