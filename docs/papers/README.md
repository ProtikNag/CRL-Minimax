# CRL papers — bookkeeping

Reference papers for the constrained min-max continual-RL project. PDFs + extracted
`.txt` (via `pdftotext`) live in this folder.

| File | Paper | Year / venue | Domain | Relevance |
|------|-------|--------------|--------|-----------|
| `2018_Schwarz_Progress_and_Compress.pdf` | Progress & Compress (P&C) — Schwarz, Luketina, Czarnecki, … Pascanu, Hadsell | ICML 2018 | Atari + 3D maze | The classic distill-then-consolidate baseline. **Old** (2018); active column → knowledge base via online-EWC distillation. |
| `2022_Powers_CORA_benchmark.pdf` | CORA, benchmarks and baselines for continual RL, by Powers, Xing, Kolve, Mottaghi, Gupta | CoLLAs 2022 (arXiv 2110.10067) | Atari, Procgen, MiniHack, CHORES | The standard sequential-Atari CRL platform. Task sequences, metrics (continual evaluation, isolated forgetting, zero-shot forward transfer), baselines (CLEAR, Online-EWC/P&C, IMPALA fine-tune). IMPALA-based, not PPO. Code, `github.com/AGI-Labs/continual_rl`. |
| `2023_Abbas_Loss_of_Plasticity_CReLU.pdf` | Loss of Plasticity in Continual Deep Reinforcement Learning, by Abbas, Zhao, Modayil, White, Machado | CoLLAs 2023 (PMLR v232) | Atari, cycled game sequences | The **plasticity** line. Value-based agents cycling through Atari games stop being able to learn; the activation footprint goes sparse and gradients shrink. Mitigation is a one-line change, **Concatenated ReLUs**. Orthogonal to our constraint, it fixes capacity loss rather than forgetting, and the two failure modes are separable. Relevant because our global policy trains for a long time on a growing task set, exactly the regime where plasticity decays. |
| `2024_Dohare_Loss_of_Plasticity_Continual_Backprop.pdf` | Loss of plasticity in deep continual learning, by Dohare, Hernandez-Garcia, Lan, Rahman, Mahmood, Sutton | Nature 2024 (vol. 632, 768–774) | Continual ImageNet + RL (ant locomotion) | The **CbpNet** source. Same failure as Abbas et al. shown at scale across architectures and optimisers, with the remedy being **continual backpropagation**, continually reinitialising a small fraction of low-utility units. Listed as a CKA-RL baseline, so a likely reviewer ask. Note it needs a random, non-gradient component, which our primal-dual scheme does not have. |
| `2024_Malagon_CompoNet_Self_Composing_Policies.pdf` | Self-Composing Policies for Scalable Continual RL (CompoNet), by Malagon, Ceberio, Lozano | ICML 2024 | Atari (Freeway, SpaceInvaders), Meta-World | Growing-network parameter isolation, each new policy is a module that attends over the frozen previous ones. A CKA-RL baseline. **Excluded from our GridWorld tier** because it grows the network while every method there has a fixed footprint. |
| `2025_Erden_Autoencoder_Task_Recognition_CRL.pdf` | Continual RL via Autoencoder-Driven Task and New Environment Recognition — Erden, Gasmi, Faltings (EPFL) | 2025 (arXiv 2505.09003) | MiniGrid + **Atari** (Breakout, Pong, BeamRider) | Recent; task-free — autoencoders detect new tasks/environments (no external task signal), new subnetwork per environment. Online. |
| `2025_Hu_CKA_RL_Continual_Knowledge_Adaptation.pdf` | Continual Knowledge Adaptation for RL (CKA-RL) — Hu, Lian, Wen, … Xiao, Tan (SCUT) | NeurIPS 2025 | Meta-World + **Atari** (Freeway, SpaceInvaders) | Recent parameter-isolation line: a task-specific *knowledge-vector pool* reused and adapted on each new task, plus Adaptive Knowledge Merging to cap memory growth. Reports average performance and **forward transfer**, not forgetting/BWT — different metric axis from ours. Baselines are ProgNet/PackNet/MaskNet/CompoNet/CbpNet; **no CLEAR, no CORA sequence**. 1M steps/task. Code: `github.com/Fhujinwu/CKA-RL`. |
| `2025_Pan_Survey_of_Continual_RL.pdf` | A Survey of Continual Reinforcement Learning — Pan, Yang, Li, … | 2025 (IEEE TPAMI; arXiv 2506.21872) | Survey | Taxonomy (replay / regularization / parameter-isolation / knowledge-transfer), benchmarks (Atari, Procgen, Continual World, CORA, HackAtari), metrics. |

## The plasticity pair, and why both are here

Abbas et al. and Dohare et al. are the same finding at two scales, and CKA-RL's
`CbpNet` row is the second one. Keeping both matters for one reason. Every
baseline in our tables is judged on forgetting and transfer, whereas these two
measure whether the network can still learn *at all* after long training. A
reviewer can reasonably ask which failure our constraint addresses. It addresses
forgetting; we make no plasticity claim, and neither CReLU nor continual
backprop conflicts with the min-max objective, so both compose with it.

**Our 50-task GridWorld runs bear this out.** Both take the top forward-transfer
scores of any method we ran (CReLUs 0.251, CbpNet 0.175, against ours at 0.077)
and neither improves retention at all: backward transfer −0.371 and −0.394,
indistinguishable from plain fine-tuning's −0.390. They buy speed on each new
task and give back the same ground on the old ones, which is what their papers
claim they do. Numbers and figures in `reports/final/gridworld/`.

## External references worth comparing against (not in this folder)

- **WMAR** (Yang, Kuhlmann, Kowadlo, 2024; arXiv 2401.16650) — online model-based
  (DreamerV3) continual RL on Atari (no shared structure), replay-based, forgetting +
  transfer metrics.
- **CLEAR** (Rolnick et al., NeurIPS 2019) — experience replay + behavioral cloning; the
  de-facto SOTA baseline on Atari that CORA and most Atari CRL papers compare to.
