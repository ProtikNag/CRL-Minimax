"""Render pseudocode JSON specs to paper-style PDFs (ruled algorithm blocks).

Why not LaTeX: this cluster's TeX Live is incomplete (no CM .tfm font metrics,
no pdflatex format file), so algorithm2e/algpseudocode cannot compile. We instead
typeset with matplotlib, whose *mathtext* engine renders LaTeX-style math with its
own bundled Computer Modern fonts -- fully independent of the broken system TeX.

Output style mimics the LaTeX `algorithm2e` ruled float: a thick top rule, a bold
"Algorithm N: Caption" line, a thin rule, italic Input/Output, then numbered,
indented body lines, closed by a thick bottom rule. Prose renders in serif; inline
`$...$` spans render as CM math; control keywords are auto-bolded.

INPUT: a JSON document (see pseudocode/specs/*.json):
  { "title": "...", "subtitle": "...",
    "sections": [
      {"kind":"heading","text":"..."},
      {"kind":"prose","text":"paragraph, may contain $math$"},
      {"kind":"algorithm","name":"Caption",
       "inputs":["..."], "outputs":["..."],
       "body":[ {"lvl":0,"text":"...","num":true}, ... ]}
    ]}

MATHTEXT SUBSET (author within this -- matplotlib mathtext is not full LaTeX):
  OK   : \leftarrow \rightarrow \sum \prod \nabla \partial \hat{} \bar{} \tilde{}
         \frac{}{} \left[ \right] \le \ge \in \sum_{}^{} greek \mathbf \mathrm \max \min
  AVOID: \gets (use \leftarrow), \big/\Big (use \left \right), \tfrac (use \frac),
         \text (use \mathrm), \! spacing.

Usage:  python render_pseudocode.py specs/04_partA.json 04_partA_minmax_no_stored_experts.pdf
"""
from __future__ import annotations

import json
import re
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

plt.rcParams.update({
    "mathtext.fontset": "cm",      # Computer Modern math (matplotlib's own fonts)
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "pdf.fonttype": 42,            # embed TrueType so the PDF is portable
})

# page geometry (letter portrait), in figure fraction
LEFT, RIGHT, TOP, BOT = 0.11, 0.92, 0.94, 0.07
BODY_FS, HEAD_FS, TITLE_FS = 10.5, 12.5, 17.0
DY = 0.026                         # vertical step per body line
GUTTER = 0.045                     # width reserved for line numbers
INDENT = 0.030                     # horizontal step per nesting level

# control-flow keywords bolded ONLY inside algorithm bodies (never in prose)
_KW = (r"end for|end if|end while|for each|foreach|for|while|if|else if|else|"
       r"then|do|return|repeat|until|break|continue|initialize")
_KW_RE = re.compile(r"(?<![\\A-Za-z])(" + _KW + r")(?![A-Za-z])")
# tall math that extends below its baseline -> give the next line extra room
_TALL_RE = re.compile(r"\\sum|\\prod|\\int|\\frac|\\left|\\binom")

# matplotlib mathtext is not full LaTeX; normalize common aliases so authors can
# write natural TeX. Applied ONLY inside $...$ spans.
_MATH_ALIAS = [
    (re.compile(r"\\gets(?![A-Za-z])"), r"\\leftarrow"),
    (re.compile(r"\\ge(?![A-Za-z])"), r"\\geq"),
    (re.compile(r"\\le(?![A-Za-z])"), r"\\leq"),
    (re.compile(r"\\tfrac(?![A-Za-z])"), r"\\frac"),
    (re.compile(r"\\[Bb]igg?(?![A-Za-z])"), r""),   # drop manual delimiter sizing
]


def _normalize_math(span):
    inner = span[1:-1]
    for rx, rep in _MATH_ALIAS:
        inner = rx.sub(rep, inner)
    return "$" + inner + "$"


def _emit_line(ax, y, text, *, x=LEFT, fs=BODY_FS, weight="normal", style="normal",
               bold_kw=False):
    """Render one line: prose in serif, `$...$` spans as CM math. Control keywords
    are bolded only when ``bold_kw`` (algorithm-body lines), never in prose."""
    parts = re.split(r"(\$[^$]*\$)", text)
    out = []
    for p in parts:
        if p.startswith("$") and p.endswith("$"):
            out.append(_normalize_math(p))
        elif bold_kw:
            out.append(_KW_RE.sub(
                lambda m: r"$\mathbf{" + m.group(1).replace(" ", r"\ ") + "}$", p))
        else:
            out.append(p)
    ax.text(x, y, "".join(out), fontsize=fs, family="serif", weight=weight,
            style=style, transform=ax.transAxes, va="top", ha="left")


class Pager:
    def __init__(self, pdf):
        self.pdf, self.fig, self.ax, self.y = pdf, None, None, TOP
        self._new()

    def _new(self):
        if self.fig is not None:
            self.pdf.savefig(self.fig); plt.close(self.fig)
        self.fig = plt.figure(figsize=(8.5, 11))
        self.ax = self.fig.add_axes([0, 0, 1, 1]); self.ax.axis("off")
        self.ax.set_xlim(0, 1); self.ax.set_ylim(0, 1)
        self.y = TOP

    def space(self, k=1.0):
        self.y -= DY * k

    def need(self, k=1.0):
        if self.y - DY * k < BOT:
            self._new()

    def rule(self, lw=1.4):
        self.need(0.6)
        self.ax.plot([LEFT, RIGHT], [self.y, self.y], color="black", lw=lw,
                     transform=self.ax.transAxes, solid_capstyle="butt")
        self.y -= DY * 0.55

    def close(self):
        if self.fig is not None:
            self.pdf.savefig(self.fig); plt.close(self.fig); self.fig = None


def _wrap_prose(s, width=95):
    return textwrap.wrap(s, width=width) or [""]


def render(spec_path, out_path):
    spec = json.load(open(spec_path, encoding="utf-8"))
    with PdfPages(out_path) as pdf:
        pg = Pager(pdf)
        if spec.get("title"):
            pg.ax.text(LEFT, pg.y, spec["title"], fontsize=TITLE_FS, family="serif",
                       fontweight="bold", transform=pg.ax.transAxes, va="top")
            pg.space(1.8)
        if spec.get("subtitle"):
            _emit_line(pg.ax, pg.y, spec["subtitle"], fs=BODY_FS, style="italic")
            pg.space(1.6)

        alg_no = 0
        for sec in spec.get("sections", []):
            kind = sec["kind"]
            if kind == "heading":
                pg.space(0.6); pg.need(2)
                pg.ax.text(LEFT, pg.y, sec["text"], fontsize=HEAD_FS, family="serif",
                           fontweight="bold", transform=pg.ax.transAxes, va="top")
                pg.space(1.5)
            elif kind == "prose":
                for ln in _wrap_prose(sec["text"]):
                    pg.need(1); _emit_line(pg.ax, pg.y, ln); pg.space(1.0)
                pg.space(0.4)
            elif kind == "algorithm":
                alg_no += 1
                pg.space(0.5); pg.need(4)
                pg.rule(1.6)                                   # thick top rule
                _emit_line(pg.ax, pg.y, f"Algorithm {alg_no}: {sec['name']}",
                           fs=HEAD_FS, weight="bold"); pg.space(1.15)
                pg.rule(0.8)                                   # thin rule
                for tag, items in (("Input", sec.get("inputs")),
                                   ("Output", sec.get("outputs"))):
                    for it in (items or []):
                        pg.need(1)
                        _emit_line(pg.ax, pg.y, f"$\\mathit{{{tag}}}$: {it}", fs=BODY_FS)
                        pg.space(1.0)
                n = 0
                for step in sec["body"]:
                    tall = bool(_TALL_RE.search(step["text"]))
                    pg.need(1.9 if tall else 1)
                    lvl = step.get("lvl", 0)
                    x = LEFT + GUTTER + lvl * INDENT
                    if step.get("num", True):
                        n += 1
                        pg.ax.text(LEFT + GUTTER - 0.008, pg.y, f"{n}", fontsize=BODY_FS - 1,
                                   family="serif", color="black", transform=pg.ax.transAxes,
                                   va="top", ha="right")
                    _emit_line(pg.ax, pg.y, step["text"], x=x, fs=BODY_FS, bold_kw=True)
                    pg.space(1.9 if tall else 1.0)   # extra room below tall math
                pg.space(0.15); pg.rule(1.6)                   # thick bottom rule
                pg.space(0.6)
        pg.close()
    print(f"[render_pseudocode] wrote {out_path} from {spec_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python render_pseudocode.py SPEC.json OUT.pdf")
    render(sys.argv[1], sys.argv[2])
