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
| `learned_vs_retained` | Every task's score when learned against its score at the end, 2×3. **The separator.** |
| `retention_curve` | Mean score of everything learned so far, as the sequence grows |
| `headline_metrics` | PERF, forgetting, backward and forward transfer, as a table |
| `raw_vs_normalised` | Final per-task score on both scales, and why the choice matters |
| `compute_cost` | Wall-clock for one complete 50-task run |

All in `png/` (300 dpi) and `svg/` (vector, verified zero embedded raster).

Normalisation is `(score − random) / (1 − random)`, so **0 is a random policy and
1 is a solved task**. The random floor is per task and averages ≈0.72 raw.

## Status

Six methods, all with at least one complete 50-task run.

| Method | Complete seeds | In flight |
|---|---:|---|
| Min-Max (ours) | 3 | — |
| CKA-RL | 3 | — |
| CbpNet | 1 | seeds 1 and 2, at 24/50 and 28/50 |
| CReLUs | 1 | seed 1, at 4/50 |
| Fine-tuning | 3 | — |
| From-scratch | 3 | — |

CbpNet and CReLUs carry a **borrowed standard deviation** (ours', marked `†`)
until their own seeds land. A lent interval is not a measurement, and the table
says so; the marker and its footnote disappear on their own once the seeds
arrive. Their per-task panels also hold 50 points rather than 150, so read their
quadrant counts as rates.

Partial runs are **excluded from every aggregate** rather than averaged in —
mixing a 50-task run with a 24-task one produces a number belonging to neither.
Which runs count is read from `metrics.json`, and every caption's seed counts,
tallies and quoted ranges are derived from the data rather than typed.

## What the numbers say

| | PERF | Forgetting | BWT | FWT |
|---|---:|---:|---:|---:|
| **Min-Max (ours)** | **0.596** ±0.007 | **0.112** ±0.008 | **+0.358** ±0.028 | 0.077 ±0.063 |
| CKA-RL | 0.470 ±0.040 | 0.237 ±0.028 | −0.187 ±0.039 | 0.162 ±0.037 |
| CReLUs | 0.329 † | 0.407 † | −0.371 † | **0.251** † |
| CbpNet | 0.284 † | 0.431 † | −0.394 † | 0.175 † |
| Fine-tuning | 0.261 ±0.051 | 0.430 ±0.048 | −0.390 ±0.053 | 0.142 ±0.020 |
| From-scratch | 0.158 ±0.062 | 0.559 ±0.066 | −0.506 ±0.070 | 0 (reference) |

Ours leads average performance by **0.126** over the nearest method, and no
interval comes close to touching.

### The plasticity baselines do exactly what the plasticity literature predicts

This is the useful result from adding them, and it is worth stating explicitly
because it is a clean division of labour rather than a horse race.

**They take the top two forward-transfer scores** (CReLUs 0.251, CbpNet 0.175),
beating CKA-RL and every other method including ours. **They fix nothing about
retention**: backward transfer is −0.371 and −0.394, statistically
indistinguishable from plain fine-tuning's −0.390, and forgetting sits at 0.41
and 0.43 against fine-tuning's 0.43.

That is the expected shape. Both methods target *loss of plasticity*, the
network's decaying ability to learn anything new after long training. Neither
targets interference between tasks. So they buy speed on each new task and give
back the same ground on the old ones. The papers are in `docs/papers/`
(`2023_Abbas_*` for CReLUs, `2024_Dohare_*` for continual backprop), and CbpNet
is one of CKA-RL's own baselines.

The reading for the paper: **our constraint and their activation or reset trick
address different failures and compose rather than compete.** Ours makes no
plasticity claim, and the forward-transfer column is where that shows.

### Why `learned_vs_retained` is the figure to lead with

On average performance ours and CKA-RL sit 0.13 apart, which is easy to wave
away. Ask a yes/no question of every task instead — *did it end better or worse
than when it was learned* — and the methods separate completely:

| | tasks that ended **better** |
|---|---:|
| **Min-Max (ours)** | **91%** |
| CKA-RL | 13% |
| CbpNet | 8% |
| CReLUs | 8% |
| Fine-tuning | 8% |
| From-scratch | 6% |

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

| | points | rescued | lost | lost as a rate |
|---|---:|---:|---:|---:|
| **Min-Max (ours)** | 150 | **32** | **0** | **0%** |
| CKA-RL | 150 | 0 | 12 | 8% |
| CbpNet | 50 | 0 | 13 | 26% |
| CReLUs | 50 | 0 | 17 | 34% |
| Fine-tuning | 150 | 0 | 45 | 30% |
| From-scratch | 150 | 0 | 57 | 38% |

**Compare the rate, not the count** — CbpNet and CReLUs contribute one seed each,
so 50 points against 150. By rate they lose tasks about as often as plain
fine-tuning, and more often than CKA-RL.

No baseline rescues a single task. Ours loses none. CKA-RL's low rate in the red
quadrant is not retention — its final scores mostly sit above 0.25, so it lands
between the two boxes rather than in either; its weakness shows up in the
aggregate metrics instead.

**It is not a headroom artefact.** A task left at 0.2 has more room to improve
than one left at 0.9, so "improved" could in principle be mechanical. Measuring
the gain as a share of the headroom still available (`1 − learned`) rules it out:

| | learned < 0.2 | 0.2–0.5 | > 0.5 |
|---|---:|---:|---:|
| **Min-Max (ours)** | **+42%** | **+57%** | **+19%** |
| CKA-RL | −6% | −24% | −86% |
| CbpNet | −7% | −30% | −184% |
| Fine-tuning | −7% | −38% | −182% |
| CReLUs | −11% | −39% | −201% |
| From-scratch | −8% | −56% | −233% |

Ours captures a large share of the room it has left, in every bin. Every baseline
is negative in every bin: they do not merely fail to improve, they give ground
back regardless of where the task started.

> Read as a **pooled** ratio, `Σ(final − learned) / Σ(1 − learned)` within each
> bin, not a mean of per-task ratios. Per task the denominator can be zero —
> CKA-RL has two tasks learned to exactly 1.00 — and the mean of those ratios
> diverges. The pooled form is the same quantity without the singularity.

### The trade, stated plainly

The per-task detail behind the aggregate is not flattering in every direction:

| | mean score when just learned | mean score at the end |
|---|---:|---:|
| **Min-Max (ours)** | 0.25 | **0.60** |
| CKA-RL | 0.65 | 0.47 |
| CReLUs | 0.69 | 0.33 |
| CbpNet | 0.67 | 0.28 |
| Fine-tuning | 0.64 | 0.26 |
| From-scratch | 0.65 | 0.16 |

**Ours learns each new task to less than half the immediate level the others
reach**, then climbs past all of them within two or three phases. So its positive
backward transfer is partly earned and partly structural: it is easier to improve
a task left at 0.25 than one left at 0.65. Better to say so than to let a
reviewer find it. The claims that survive are the **final state** after 50 tasks
and the **shape** — flat for ours, falling for everyone else.

Note CReLUs has the **highest** immediate score of any method (0.69) and still
ends at 0.33. Learning each task well is not the binding constraint here;
holding it is.

The likely mechanism for ours: the local phase is short (150–250 iterations) and
the global consolidation, which maximises return over *all* seen tasks, does most
of the learning. With 50 related layouts there is a lot of positive transfer
available, and consolidation is what harvests it.

### `retention_curve`

After finishing task *k*, the mean score over the *k−1* tasks learned before it.
The just-learned task is excluded: including it lets a method that merely learns
the newest task well post a flattering curve.

This is the readable form of the forgetting matrix. At 50 tasks the triangle is
far too dense to see anything in, but its row means are not, and they answer the
question the matrix was there for — *does the method still work at 50 tasks*.
Ours ends flat at ≈0.61; the rest sit at 0.46 (CKA-RL), 0.32 (CReLUs), 0.28
(CbpNet), 0.25 (fine-tuning) and 0.15 (from-scratch).

Ours' band never touches CKA-RL's at any point in the sequence. CbpNet and
CReLUs carry no band, having one complete seed each.

### Raw against normalised

A random policy already collects most of the raw discounted return on this grid,
so raw scores crowd into [0.59, 1.00] and every method looks close. The random
floor also varies per task (0.55 to 0.83), so the same raw number is a different
achievement on different tasks. Median final score:

| | raw | normalised |
|---|---:|---:|
| **Min-Max (ours)** | 0.92 | **0.67** |
| CKA-RL | 0.87 | 0.49 |
| CbpNet | 0.81 | 0.26 |
| CReLUs | 0.80 | 0.23 |
| Fine-tuning | 0.80 | 0.20 |
| From-scratch | 0.77 | 0.10 |

Showing both is the point: the separation normalisation exposes is real, not an
artefact of the normaliser. Note that CbpNet, CReLUs and fine-tuning are
indistinguishable on the raw scale (0.80–0.81) and still ordered on the
normalised one.

### Forward transfer, and the budget asymmetry behind it

Ours is **last of six** on FWT (0.077, against CReLUs' 0.251), and the trade
table above is why: forward transfer measures how fast a task is learned in its
own phase, and ours deliberately spends less there.

**The per-task budgets are not matched.** Ours' local phase runs 150–250
iterations; the others run 150–500. Ours uses *more* total optimisation but less
of it inside any single task's own learning phase. From-scratch is the FWT
reference, so its value is 0 by construction and it is omitted from that column.

Expect a reviewer to press on this. The answer is the division of labour above:
the plasticity methods win this column and lose every retention column, which is
what their own papers claim they do.

### Compute

| | wall-clock | vs ours |
|---|---:|---:|
| **Min-Max (ours)** | 244 min | — |
| CKA-RL | 161 min | 0.66× |
| CbpNet | 170 min | 0.70× |
| CReLUs | 168 min | 0.69× |
| Fine-tuning | 136 min | 0.56× |
| From-scratch | 171 min | 0.70× |

Ours is the most expensive because consolidation re-simulates past environments,
spending time on tasks it has already learned. Measured on a shared, contended
cluster, so read it as indicative rather than exact. Fine-tuning's three seeds
agree to within 7 minutes, so its interval hides behind its marker.

## Disclosures for any caption

- **Ours re-simulates past environments during consolidation.** The others train
  only on the current task. This is live past-task environment access — a
  stronger assumption than a replay buffer, not a weaker one.
- **CKA-RL, CbpNet and CReLUs here are our own reimplementations**, adapted to
  the shared-head setting. CKA-RL's original does not target it. Say so.
- **CompoNet is excluded** because it grows the network; every method here has a
  fixed footprint (ours a fixed shared head, CKA-RL a fixed trunk plus a bounded
  pool of 5 and a small per-task α, CbpNet fixed with unit resets, CReLUs fixed
  with a CReLU activation).
- **Seeds:** three complete for ours, CKA-RL, fine-tuning and from-scratch; one
  each for CbpNet and CReLUs, whose intervals are borrowed and marked `†`.

## Rebuild

```bash
python reports/final/gridworld/make_figures.py
```

Reads `reports/gridworld_sharedhead/*/` per `docs/LOGGING_CONTRACT.md`. Nothing
is transcribed or duplicated into this folder. Method count is read from the
data, and the script checks its own output for clipped captions and rasterised
SVG, exiting non-zero on either.
