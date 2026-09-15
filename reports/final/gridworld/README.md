# GridWorld: 50 tasks, one shared head

The tier that removes the two things making the CKA-RL benchmark easy.

There, the tasks are **modes of one game** and every method gets **its own head
per task**. Both make the problem easier than it looks: ceilings sit near 0.99,
so there is almost nothing left to separate methods by, and a per-task head means
much of each task's solution never competes for capacity.

Here there are **50 different layouts** — varying obstacle density and type,
slipperiness, and reward and penalty magnitudes on a 50×50 grid — under **one
shared task-conditioned head**. Capacity is fixed and interference is forced.

## Figures

| Stem | What it shows |
|---|---|
| `learned_vs_retained` | Every task's score when learned against its score at the end, 2×2. **The separator.** |
| `retention_curve` | Mean score of everything learned so far, as the sequence grows |
| `headline_metrics` | PERF, forgetting, backward and forward transfer, as a table |
| `raw_vs_normalised` | Final per-task score on both scales, and why the choice matters |
| `compute_cost` | Wall-clock for one complete 50-task run |

All in `png/` (300 dpi) and `svg/` (vector, verified zero embedded raster).

Normalisation is `(score − random) / (1 − random)`, so **0 is a random policy and
1 is a solved task**. The random floor is per task and averages ≈0.72 raw.

## Status

Complete 50-task runs: **Min-Max 3, CKA-RL 2, fine-tuning 3, from-scratch 2**.
Every method now has a measured standard deviation; no interval in this folder is
borrowed any more.

Partial runs are **excluded from every aggregate** rather than averaged in —
mixing a 50-task run with a 24-task one produces a number belonging to neither.
Which runs count is read from `metrics.json`, and every caption's seed counts and
tallies are derived from the data rather than typed, so a re-run after new seeds
land updates the figures and their captions together.

## What the numbers say

| | PERF | Forgetting | BWT | FWT |
|---|---:|---:|---:|---:|
| **Min-Max (ours)** | **0.596** ±0.007 | **0.112** ±0.008 | **+0.358** ±0.028 | 0.080 ±0.064 |
| CKA-RL | 0.490 ±0.027 | 0.226 ±0.029 | −0.168 ±0.026 | **0.181** ±0.022 |
| Fine-tuning | 0.261 ±0.051 | 0.430 ±0.048 | −0.390 ±0.053 | 0.147 ±0.022 |
| From-scratch | 0.193 ±0.012 | 0.520 ±0.006 | −0.466 ±0.001 | 0 (reference) |

Ours leads average performance by **0.106**, and the two intervals are nowhere
near touching. It halves CKA-RL's forgetting and is the only method with positive
backward transfer.

### Why `learned_vs_retained` is the figure to lead with

On average performance ours and CKA-RL sit 0.09 apart, which is easy to wave
away. Ask a yes/no question of every task instead — *did it end better or worse
than when it was learned* — and the methods separate completely:

| | tasks that ended **better** |
|---|---:|
| **Min-Max (ours)** | **91%** |
| CKA-RL | 13% |
| Fine-tuning | 8% |
| From-scratch | 8% |

Ours' point cloud sits above the no-change diagonal; every other method's sits
below it. That is a difference in kind, not in degree, and it is the same fact
the backward-transfer column summarises — read one task at a time.

#### The shaded quadrants

Two dotted lines per panel bound one quadrant, and the two quadrants are mirror
images of each other: **0.25** is barely off a random policy, **0.60** is most of
the way to solved.

- **Blue** (ours): learned below 0.25, ended above 0.60 — brought back.
- **Red** (baselines): learned above 0.60, ended below 0.25 — lost.

Each panel shades only the quadrant that characterises its method.

| | points | rescued | lost |
|---|---:|---:|---:|
| **Min-Max (ours)** | 150 | **32** | **0** |
| CKA-RL | 100 | 0 | 6 |
| Fine-tuning | 150 | 0 | 45 |
| From-scratch | 100 | 0 | 40 |

No baseline rescues a single task. Ours loses none. CKA-RL's low count in the
red quadrant is not retention — its final scores mostly sit above 0.25, so it
lands between the two boxes rather than in either; its weakness shows up in the
aggregate metrics instead.

**It is not a headroom artefact.** A task left at 0.2 has more room to improve
than one left at 0.9, so "improved" could in principle be mechanical. Measuring
the gain as a fraction of the headroom still available (`1 − learned`) rules it
out:

| | learned < 0.2 | 0.2–0.5 | > 0.5 |
|---|---:|---:|---:|
| **Min-Max (ours)** | **+41%** | **+57%** | +16% |
| CKA-RL | −8% | −24% | −123% |
| Fine-tuning | −7% | −40% | −285% |
| From-scratch | −8% | −53% | −282% |

Ours captures a large share of the room it has left. Every baseline is negative
in every bin: they do not merely fail to improve, they give ground back
regardless of where the task started.

### The trade, stated plainly

The per-task detail behind the aggregate is not flattering in every direction:

| | mean score when just learned | mean score at the end |
|---|---:|---:|
| **Min-Max (ours)** | 0.25 | **0.60** |
| CKA-RL | 0.65 | 0.49 |
| Fine-tuning | 0.64 | 0.26 |
| From-scratch | 0.65 | 0.19 |

**Ours learns each new task to less than half the immediate level the others
reach**, then climbs past all of them within two or three phases. So its positive
backward transfer is partly earned and partly structural: it is easier to improve
a task left at 0.25 than one left at 0.65. Better to say so than to let a
reviewer find it. The claims that survive are the **final state** after 50 tasks
and the **shape** — flat for ours, falling for everyone else.

The likely mechanism: the local phase here is short (150–250 iterations) and the
global consolidation, which maximises return over *all* seen tasks, does most of
the learning. With 50 related layouts there is a lot of positive transfer
available, and consolidation is what harvests it.

### `retention_curve`

After finishing task *k*, the mean score over the *k−1* tasks learned before it.
The just-learned task is excluded: including it lets a method that merely learns
the newest task well post a flattering curve.

This is the readable form of the forgetting matrix. At 50 tasks the triangle is
far too dense to see anything in, but its row means are not, and they answer the
question the matrix was there for — *does the method still work at 50 tasks*.
Ours is flat at ≈0.60 while the others sit at 0.45, 0.30 and 0.17.

The shaded band is the min-max over ours' 3 complete seeds. The other three
methods have one complete seed each so far, so they carry no band; a band over a
single run would be an invented interval.

### Raw against normalised

A random policy already collects most of the raw discounted return on this grid,
so raw scores crowd into [0.59, 0.99] and every method looks close. The random
floor also varies per task (0.55 to 0.87), so the same raw number is a different
achievement on different tasks. Median final score:

| | raw | normalised |
|---|---:|---:|
| **Min-Max (ours)** | 0.92 | **0.67** |
| CKA-RL | 0.88 | 0.51 |
| Fine-tuning | 0.80 | 0.20 |
| From-scratch | 0.78 | 0.11 |

Showing both is the point: the separation normalisation exposes is real, not an
artefact of the normaliser.

### Forward transfer, and the budget asymmetry behind it

Ours is last on FWT (0.080 against CKA-RL's 0.181), and the table above is why: forward transfer measures how fast a task is learned in its own phase, and
ours deliberately spends less there.

**The per-task budgets are not matched.** Ours' local phase runs 150–250
iterations; the others run 150–500. Ours uses *more* total optimisation but less
of it inside any single task's own learning phase. From-scratch is the FWT
reference, so its value is 0 by construction and it is omitted from that panel.

### Compute

| | wall-clock | vs ours |
|---|---:|---:|
| **Min-Max (ours)** | 244 min | — |
| CKA-RL | 155 min | 0.63× |
| Fine-tuning | 136 min | 0.56× |
| From-scratch | 151 min | 0.62× |

Ours is the most expensive because consolidation re-simulates past environments,
spending time on tasks it has already learned. Measured on a shared, contended
cluster, so read it as indicative rather than exact.

## Disclosures for any caption

- **Ours re-simulates past environments during consolidation.** The others train
  only on the current task. This is live past-task environment access — a
  stronger assumption than a replay buffer, not a weaker one.
- **CKA-RL here is our own reimplementation**, adapted to the shared-head setting
  the original does not target. Say so in the paper.
- **CompoNet is excluded** because it grows the network; every method here has a
  fixed footprint (ours a fixed shared head, CKA-RL a fixed trunk plus a bounded
  pool of 5 and a small per-task α).
- **Seeds:** Min-Max 3, CKA-RL 2, fine-tuning 3, from-scratch 2, all complete.
  Every interval shown is measured; the borrowed-standard-deviation placeholder
  is no longer used anywhere and its `†` note disappears from the table
  automatically when it is not needed.

## Rebuild

```bash
python reports/final/gridworld/make_figures.py
```

Reads `reports/gridworld_sharedhead/*/` per `docs/LOGGING_CONTRACT.md`. Nothing
is transcribed or duplicated into this folder, and no edits are needed when the
remaining seeds land.
