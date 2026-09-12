# CRL papers — bookkeeping

Reference papers for the constrained min-max continual-RL project. PDFs + extracted
`.txt` (via `pdftotext`) live in this folder.

| File | Paper | Year / venue | Domain | Relevance |
|------|-------|--------------|--------|-----------|
| `2018_Schwarz_Progress_and_Compress.pdf` | Progress & Compress (P&C) — Schwarz, Luketina, Czarnecki, … Pascanu, Hadsell | ICML 2018 | Atari + 3D maze | The classic distill-then-consolidate baseline. **Old** (2018); active column → knowledge base via online-EWC distillation. |
| `2025_Erden_Autoencoder_Task_Recognition_CRL.pdf` | Continual RL via Autoencoder-Driven Task and New Environment Recognition — Erden, Gasmi, Faltings (EPFL) | 2025 (arXiv 2505.09003) | MiniGrid + **Atari** (Breakout, Pong, BeamRider) | Recent; task-free — autoencoders detect new tasks/environments (no external task signal), new subnetwork per environment. Online. |
| `2025_Hu_CKA_RL_Continual_Knowledge_Adaptation.pdf` | Continual Knowledge Adaptation for RL (CKA-RL) — Hu, Lian, Wen, … Xiao, Tan (SCUT) | NeurIPS 2025 | Meta-World + **Atari** (Freeway, SpaceInvaders) | Recent parameter-isolation line: a task-specific *knowledge-vector pool* reused and adapted on each new task, plus Adaptive Knowledge Merging to cap memory growth. Reports average performance and **forward transfer**, not forgetting/BWT — different metric axis from ours. Baselines are ProgNet/PackNet/MaskNet/CompoNet/CbpNet; **no CLEAR, no CORA sequence**. 1M steps/task. Code: `github.com/Fhujinwu/CKA-RL`. |
| `2025_Pan_Survey_of_Continual_RL.pdf` | A Survey of Continual Reinforcement Learning — Pan, Yang, Li, … | 2025 (IEEE TPAMI; arXiv 2506.21872) | Survey | Taxonomy (replay / regularization / parameter-isolation / knowledge-transfer), benchmarks (Atari, Procgen, Continual World, CORA, HackAtari), metrics. |

## External references worth comparing against (found via web search, not in this folder)

- **CORA** (Powers et al., CoLLAs 2022; arXiv 2110.10067; code: github.com/AGI-Labs/continual_rl) —
  the standard sequential-**Atari** CRL platform. Task sequences + metrics (Continual
  Evaluation, Isolated Forgetting, Zero-shot Forward Transfer) + baselines (**CLEAR**,
  Online-EWC/P&C, IMPALA fine-tune). IMPALA-based (not PPO).
- **WMAR** (Yang, Kuhlmann, Kowadlo, 2024; arXiv 2401.16650) — online model-based
  (DreamerV3) continual RL on Atari (no shared structure), replay-based, forgetting +
  transfer metrics.
- **CLEAR** (Rolnick et al., NeurIPS 2019) — experience replay + behavioral cloning; the
  de-facto SOTA baseline on Atari that CORA and most Atari CRL papers compare to.
