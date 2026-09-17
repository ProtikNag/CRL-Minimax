# Paper source

ICLR 2026. Working title *DUEL: Primal-Dual Retention Constraints for
Continual Reinforcement Learning*.

```
paper/
  main.tex              skeleton, \input's each section
  references.bib        bibliography, with a VERIFY block at the bottom
  sections/
    related_work.tex    done
```

The section plan, the goal and flow of each section, and the decisions behind
them live in [`../docs/PAPER_OUTLINE.md`](../docs/PAPER_OUTLINE.md). The
storyline and the standing rebuttals live in
[`../docs/storyline.md`](../docs/storyline.md) and
[`../docs/PAPER_ARGUMENTS.md`](../docs/PAPER_ARGUMENTS.md).

## Conventions

- One file per section under `sections/`, `\input` from `main.tex`. Nothing in
  `sections/` depends on the document class, so the ICLR style files can be
  dropped into `main.tex` without touching section content.
- `\citep` / `\citet` via natbib.
- Related work is grouped thematically rather than chronologically, following
  Hu et al. (CKA-RL, NeurIPS 2025), and every family paragraph closes on what
  that family leaves unaddressed.

## Before submitting

`references.bib` ends with a block marked **VERIFY BEFORE SUBMISSION**. Those
entries are cited by CKA-RL, and what `related_work.tex` says about them comes
from CKA-RL's own description of them, which is reliable. Their bibliographic
details are not. Check each against the CKA-RL reference list or drop the
citation.

## Build

```bash
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```
