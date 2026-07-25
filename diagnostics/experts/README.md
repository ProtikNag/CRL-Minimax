# Trained experts (Impala-CNN large, single-task, shared init)

All greedy-100 scores (project rule). `frames_at_best` etc. are the training
budget to reach the best model — use these to give CLEAR a matched budget.

| game | best greedy | random | frames→best | episodes→best | iters→best | total frames | stop |
|---|---|---|---|---|---|---|---|
| Assault | 2105 | 222.4 | 17.2M | 5367 | 2100 | 22.1M | plateau |
| BeamRider | 2802 | 363.9 | 21.3M | 3103 | 2600 | 26.2M | plateau |
| Boxing | 72 | 0.1 | 21.3M | 2976 | 2600 | 26.2M | plateau |
| Breakout | 279 | 1.7 | 11.5M | 3516 | 1400 | 16.4M | plateau |
| Freeway | 32 | 0.0 | 13.1M | 1600 | 1600 | 18.0M | plateau |
| Krull | 4198 | 1598.0 | 3.3M | 602 | 400 | 8.2M | plateau |
| Pong | 20 | -20.7 | 2.5M | 310 | 300 | 7.4M | plateau |
| Qbert | 18000 | 163.9 | 21.3M | 6216 | 2600 | 26.2M | plateau |
| Seaquest | 1738 | 68.4 | 5.7M | 1103 | 700 | 10.7M | plateau |
| SpaceInvaders | 1079 | 148.0 | 11.5M | 3806 | 1400 | 16.4M | plateau |
