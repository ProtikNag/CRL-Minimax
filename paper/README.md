# Paper source

ICLR 2026. Working title *Forgetting Returns Without Parameter Isolation:
DUEL for Continual Reinforcement Learning*.

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
  Hu et al. (CKA-RL, NeurIPS 2025), and every family closes on what it leaves
  unaddressed.

## Page budget

Nine pages, and the results section is the heavy one. Related work is held to
roughly half a page, naming only the canonical representative of each family
and letting the survey carry the rest. Evaluation protocols (CORA, Continual
World) belong in `sections/setup.tex`, where the benchmark-structure argument
is made anyway, rather than being covered twice.

## Before submitting

`references.bib` ends with a block marked **CURRENTLY UNCITED. VERIFY BEFORE
USING.** Those entries survive from a longer draft of the related work and
nothing cites them now, so none reaches the rendered bibliography. Their
bibliographic details are guesses. Check each against the CKA-RL reference list
before restoring any citation to them.

## Build

```bash
cd paper
rm -f main.aux main.bbl main.blg
pdflatex main; bibtex main; pdflatex main; pdflatex main
```

Separate the steps with `;` rather than `&&`. `pdflatex` exits non-zero on
warnings alone, so an `&&` chain silently skips the later passes and leaves
no PDF. Clear the auxiliaries first. Running over a stale `main.aux` makes BibTeX skip
the bibliography and leaves every citation undefined, which looks like a
citation bug rather than a build one. Check `main.blg` for BibTeX errors; the
LaTeX log will not show them.
