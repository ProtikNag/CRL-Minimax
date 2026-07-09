"""Shared figure styling and IO for all analysis plots.

Academic palette (blue/amber/green/... on white), Tufte spine, faint
horizontal grid. Every figure is saved in BOTH formats into split
subfolders: ``<figures>/png/<name>.png`` and ``<figures>/svg/<name>.svg``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica", "Arial", "DejaVu Sans"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#E9ECEF",
        "grid.linewidth": 0.7,
        "axes.edgecolor": "#495057",
        "axes.labelcolor": "#212529",
        "axes.titlecolor": "#212529",
        "axes.titleweight": "600",
        "text.color": "#212529",
        "xtick.color": "#6C757D",
        "ytick.color": "#6C757D",
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

# Academic categorical palette (never Anthropic brand colors, never rainbow).
AC = {
    "blue": "#2563EB", "amber": "#D97706", "green": "#059669", "red": "#DC2626",
    "violet": "#7C3AED", "teal": "#0891B2", "rose": "#BE185D", "sienna": "#92400E",
    "surface": "#F8F9FA", "border": "#DEE2E6", "axis": "#495057",
    "grid": "#E9ECEF", "text": "#212529", "muted": "#6C757D", "faint": "#ADB5BD",
}
AC_SERIES = [AC["blue"], AC["amber"], AC["green"], AC["red"],
             AC["violet"], AC["teal"], AC["rose"], AC["sienna"]]

# Method colors used consistently across every comparison figure.
METHOD_COLORS = {
    "constrained": AC["green"],     # ours
    "finetune": AC["red"],          # naive sequential fine-tuning (forgets old)
    "unconstrained": AC["amber"],   # two-policy ablation (forgets newest)
    "joint": AC["violet"],          # joint multi-task upper bound
}
METHOD_LABELS = {
    "constrained": "Constrained min-max (ours)",
    "finetune": "Naive fine-tuning (single net)",
    "unconstrained": "Unconstrained ablation (duals off)",
    "joint": "Joint multi-task (upper bound)",
}


def save_figure(fig: plt.Figure, figures_dir: str | Path, name: str) -> None:
    """Write ``fig`` to ``<figures_dir>/png/<name>.png`` and ``.../svg/<name>.svg``."""
    figures_dir = Path(figures_dir)
    for subdir, ext in (("png", "png"), ("svg", "svg")):
        target = figures_dir / subdir
        target.mkdir(parents=True, exist_ok=True)
        fig.savefig(target / f"{name}.{ext}", format=ext)
    plt.close(fig)


def blue_sequential():
    """Colorblind-safe sequential map: surface white -> primary blue."""
    return mpl.colors.LinearSegmentedColormap.from_list(
        "ac_blue", [AC["surface"], AC["blue"]]
    )
