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
| `learned_vs_retained` | Every task's score when learned against its score at the end. **The separator.** |
| `task_life` | Two views of the sequence: system state as it grows, and the life of one task |
| `headline_metrics` | PERF, forgetting, backward and forward transfer |
| `raw_vs_normalised` | Final per-task score on both scales, and why the choice matters |
| `compute_cost` | Wall-clock and environment frames |

All in `png/` (300 dpi) and `svg/` (vector, verified zero embedded raster).

Normalisation is `(score − random) / (1 − random)`, so **0 is a random policy and
1 is a solved task**. The random floor is per task and averages ≈0.72 raw.

## Status

Complete 50-task runs: **ours ×3 seeds**, and one seed each of CKA-RL,
fine-tuning and from-scratch. Their seeds 1–2 are still running. Partial runs are
**excluded from every aggregate** rather than averaged in — mixing a 50-task run
with a 24-task one produces a number that belongs to neither. Which runs count is
read from `metrics.json`, so finished seeds join automatically on a re-run.

## What the numbers say

| | PERF | Forgetting | BWT | FWT |
|---|---:|---:|---:|---:|
| **Min-Max (ours)** | **0.60** | **0.11** | **+0.36** | 0.05 |
| CKA-RL | 0.51 | 0.21 | −0.15 | **0.20** |
| Fine-tuning | 0.31 | 0.39 | −0.34 | 0.16 |
| From-scratch | 0.18 | 0.52 | −0.47 | 0 (reference) |

### Why `learned_vs_retained` is the figure to lead with

On average performance ours and CKA-RL sit 0.09 apart, which is easy to wave
away. Ask a yes/no question of every task instead — *did it end better or worse
than when it was learned* — and the methods separate completely:

| | tasks that ended **better** |
|---|---:|
| **Min-Max (ours)** | **91%** |
| CKA-RL | 12% |
| Fine-tuning | 4% |
| From-scratch | 8% |

Ours' point cloud sits above the no-change diagonal; every other method's sits
below it. That is a difference in kind, not in degree, and it is the same fact
the backward-transfer column summarises — read one task at a time.

### The trade, stated plainly

`task_life`'s right panel is not flattering in every direction:

| | just learned | age ≥ 10 |
|---|---:|---:|
| **Min-Max (ours)** | 0.25 | **0.63** |
| CKA-RL | 0.64 | 0.42 |
| Fine-tuning | 0.65 | 0.31 |
| From-scratch | 0.64 | 0.17 |

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

### The two panels of `task_life` are not the same plot

They are easy to confuse, which is why they share a figure:

- **Left** indexes by **position in the sequence**. After finishing task *k*, how
  is the model doing on the *k−1* tasks behind it? This is the state of the whole
  system as the sequence grows — *does the method still work at 50 tasks*.
- **Right** indexes by **time since a task was learned**, pooling every task that
  is *n* phases old whenever it happened. This is the life of a single task.

A method can be flat on the left and still decaying on the right, if the later
tasks happened to be easier. The right panel is where ours differs in kind: it
starts *lowest* at age 0 and is the only curve that rises.

### Raw against normalised

A random policy already collects most of the raw discounted return on this grid,
so raw scores crowd into [0.59, 0.99] and every method looks close. The random
floor also varies per task (0.55 to 0.87), so the same raw number is a different
achievement on different tasks. Median final score:

| | raw | normalised |
|---|---:|---:|
| **Min-Max (ours)** | 0.92 | **0.67** |
| CKA-RL | 0.88 | 0.55 |
| Fine-tuning | 0.81 | 0.21 |
| From-scratch | 0.78 | 0.10 |

Showing both is the point: the separation normalisation exposes is real, not an
artefact of the normaliser.

### Forward transfer, and the budget asymmetry behind it

Ours is last on FWT (0.05 against CKA-RL's 0.20), and the age-0 column above is
why: forward transfer measures how fast a task is learned in its own phase, and
ours deliberately spends less there.

**The per-task budgets are not matched.** Ours' local phase runs 150–250
iterations; the others run 150–500. Ours uses *more* total optimisation but less
of it inside any single task's own learning phase. From-scratch is the FWT
reference, so its value is 0 by construction and it is omitted from that panel.

### Compute

| | wall-clock | frames | vs ours |
|---|---:|---:|---:|
| **Min-Max (ours)** | 244 min | 18.1 M | — |
| CKA-RL | 177 min | 13.1 M | 0.73× |
| Fine-tuning | 140 min | 14.3 M | 0.79× |
| From-scratch | 156 min | 15.4 M | 0.85× |

Ours is the most expensive and the frames panel says why: consolidation
re-simulates past environments, so it spends frames on tasks it already learned.
Wall-clock came off a shared, contended cluster and is the softer of the two;
**frames is the number to quote**.

## Disclosures for any caption

- **Ours re-simulates past environments during consolidation.** The others train
  only on the current task. This is live past-task environment access — a
  stronger assumption than a replay buffer, not a weaker one.
- **CKA-RL here is our own reimplementation**, adapted to the shared-head setting
  the original does not target. Say so in the paper.
- **CompoNet is excluded** because it grows the network; every method here has a
  fixed footprint (ours a fixed shared head, CKA-RL a fixed trunk plus a bounded
  pool of 5 and a small per-task α).
- **Seeds:** ours 3 complete, the others 1 each so far. `headline_metrics` lends
  ours' standard deviation to the single-seed methods as a **placeholder**, drawn
  dotted and uncapped and marked `†` so it cannot be mistaken for a measurement.

## Rebuild

```bash
python reports/final/gridworld/make_figures.py
```

Reads `reports/gridworld_sharedhead/*/` per `docs/LOGGING_CONTRACT.md`. Nothing
is transcribed or duplicated into this folder, and no edits are needed when the
remaining seeds land.
