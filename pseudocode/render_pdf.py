"""Render a pseudocode markdown-ish text file to a paginated PDF (matplotlib).

Used because this cluster's TeX install is broken (no pdflatex format file). This
is the fallback typesetting route; on a machine with working LaTeX, prefer a real
algorithm2e/algpseudocode document instead.

Line grammar of the input `.md`/`.txt`:
  `# Title`     -> large bold header (first one also sets the PDF title)
  `## Section`  -> bold sub-header
  blank line    -> vertical gap
  anything else -> monospace body line; indentation is preserved and inline
                   `$...$` segments are rendered as math (matplotlib mathtext).

Usage:
  python render_pdf.py INPUT.md OUTPUT.pdf
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

LINES_PER_PAGE = 46          # letter portrait at ~10pt monospace
LEFT = 0.06                  # left margin in axes fraction
TOP = 0.97
LINE_DY = (TOP - 0.05) / LINES_PER_PAGE


def render(inp: str, outp: str) -> None:
    lines = open(inp, encoding="utf-8").read().splitlines()
    with PdfPages(outp) as pdf:
        fig = ax = None
        row = 0

        def new_page():
            nonlocal fig, ax, row
            fig = plt.figure(figsize=(8.5, 11))
            ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
            row = 0

        def close_page():
            if fig is not None:
                pdf.savefig(fig); plt.close(fig)

        new_page()
        for raw in lines:
            if row >= LINES_PER_PAGE:
                close_page(); new_page()
            y = TOP - row * LINE_DY
            if raw.startswith("# "):
                ax.text(LEFT, y, raw[2:], fontsize=15, fontweight="bold",
                        family="serif", transform=ax.transAxes, va="top")
                row += 2
            elif raw.startswith("## "):
                ax.text(LEFT, y, raw[3:], fontsize=12, fontweight="bold",
                        family="serif", transform=ax.transAxes, va="top")
                row += 1
            elif raw.strip() == "":
                row += 1
            else:
                # monospace body; matplotlib renders $...$ spans as math inline.
                ax.text(LEFT, y, raw.replace(" ", " "), fontsize=9.5,
                        family="monospace", transform=ax.transAxes, va="top")
                row += 1
        close_page()
    print(f"[render_pdf] wrote {outp} from {inp}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python render_pdf.py INPUT.md OUTPUT.pdf")
    render(sys.argv[1], sys.argv[2])
