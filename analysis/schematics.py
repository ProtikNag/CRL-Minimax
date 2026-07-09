"""Conceptual (data-independent) figures: the design-space map and the method
schematic. These are the paper's Fig 1 and Fig 2.

Usage:
    python -m analysis.schematics --out <figures_dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from analysis.style import AC, save_figure
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def design_space_map(out_dir: Path) -> None:
    """Fig 1: where a constrained local/global min-max method sits.

    x = consolidation mechanism (fine-tune -> ... -> hard Lagrangian constraint);
    y = memory strategy (raw replay buffer -> ... -> no stored data). Our method
    occupies the hard-constraint / no-data corner.
    """
    fig, ax = plt.subplots(figsize=(9, 6.5))

    # (x, y, label, is_ours, ha, x_off, y_off) -- offsets in points.
    points = [
        (0.5, 0.15, "CLEAR\n(Rolnick 2019)", False, "left", 8, 6),
        (1.0, 0.52, "EWC\n(Kirkpatrick 2017)", False, "left", 8, 6),
        (2.0, 0.72, "Progressive Nets\n(Rusu 2016)", False, "left", 8, 6),
        (3.0, 0.45, "Policy Distillation\n(Rusu 2016)", False, "right", -8, 6),
        (3.15, 0.28, "RePR\n(Atkinson 2021)", False, "left", 8, 0),
        (3.15, 0.70, "Progress & Compress\n(Schwarz 2018)", False, "left", 8, 6),
        (3.7, 0.52, "Policy Consolidation\n(Kaplanis 2019)", False, "right", -8, 6),
        (4.3, 0.88, "Ours: constrained\nlocal/global min-max", True, "center", 0, 16),
    ]
    for x, y, label, ours, ha, dx, dy in points:
        if ours:
            ax.scatter([x], [y], s=360, marker="*", color=AC["green"],
                       edgecolor=AC["axis"], linewidth=1.0, zorder=5)
            ax.annotate(label, (x, y), textcoords="offset points", xytext=(dx, dy),
                        ha=ha, va="bottom", fontsize=10, fontweight="600",
                        color=AC["green"])
        else:
            ax.scatter([x], [y], s=70, color=AC["muted"], zorder=4)
            ax.annotate(label, (x, y), textcoords="offset points", xytext=(dx, dy),
                        ha=ha, fontsize=8.5, color=AC["muted"])

    ax.axhspan(0.0, 0.3, color=AC["red"], alpha=0.05)
    ax.text(5.0, 0.02, "replay-dependent", color=AC["red"], fontsize=9, ha="right")
    ax.text(5.0, 1.0, "replay-free", color=AC["green"], fontsize=9, ha="right")

    ax.set_xlim(-0.1, 5.1)
    ax.set_ylim(-0.02, 1.1)
    ax.set_xticks([0, 1, 2, 3, 4],
                  ["Fine-tune", "Regularization\n(EWC)", "Architectural\nisolation",
                   "Distillation", "Hard constraint\n(Lagrangian)"])
    ax.set_yticks([0.15, 0.55, 0.9],
                  ["Raw replay\nbuffer", "Less stored\ndata", "No stored\ndata"])
    ax.set_xlabel("Consolidation mechanism  →  more structured / data-free")
    ax.set_ylabel("Memory strategy  →  less stored data")
    ax.set_title("Design space: a constrained local/global min-max method fills the "
                 "hard-constraint, no-data corner")
    ax.grid(True, alpha=0.4)
    save_figure(fig, out_dir, "design_space_map")


def method_schematic(out_dir: Path) -> None:
    """Fig 2: the local <-> global alternation with the Lagrangian constraint."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)

    def box(x, y, w, h, text, color):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                               linewidth=1.5, edgecolor=color, facecolor="white")
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=10, color=AC["text"])

    box(0.6, 4.2, 3.4, 1.8,
        "LOCAL  $\\pi_\\theta$\nmax current-task lead\n"
        "s.t. per-task $\\lambda_i$ hold\npast tasks (squared hinge)", AC["blue"])
    box(6.0, 4.2, 3.4, 1.8,
        "GLOBAL  $\\pi_\\phi$\nmax past-task lead\n"
        "s.t. $\\mu$ holds the\ncurrent task", AC["amber"])

    # Alternation arrows.
    ax.add_patch(FancyArrowPatch((4.0, 5.4), (6.0, 5.4), arrowstyle="-|>",
                 mutation_scale=18, color=AC["axis"], lw=1.5))
    ax.text(5.0, 5.7, "freeze local,\ntrain global", ha="center", fontsize=8.5,
            color=AC["muted"])
    ax.add_patch(FancyArrowPatch((6.0, 4.6), (4.0, 4.6), arrowstyle="-|>",
                 mutation_scale=18, color=AC["axis"], lw=1.5))
    ax.text(5.0, 4.0, "freeze global,\ntrain local ($\\theta^{(0)}=\\phi$)",
            ha="center", fontsize=8.5, color=AC["muted"])

    # Constraint band in the middle.
    box(3.1, 2.4, 3.8, 1.0,
        "one-sided squared constraint\n$F=\\max(0, V_{ref}-V)^2 \\leq \\epsilon$",
        AC["green"])
    ax.add_patch(FancyArrowPatch((2.3, 4.2), (3.5, 3.4), arrowstyle="-|>",
                 mutation_scale=14, color=AC["green"], lw=1.2))
    ax.add_patch(FancyArrowPatch((7.7, 4.2), (6.5, 3.4), arrowstyle="-|>",
                 mutation_scale=14, color=AC["green"], lw=1.2))

    # Deployment note.
    box(3.1, 0.6, 3.8, 1.0,
        "deploy GLOBAL $\\pi_\\phi$\n(consolidates all tasks)", AC["violet"])
    ax.add_patch(FancyArrowPatch((7.7, 4.2), (6.0, 1.6), arrowstyle="-|>",
                 mutation_scale=14, color=AC["muted"], lw=1.0, linestyle="--"))

    ax.set_title("Method: alternating local/global primal-dual updates under a "
                 "min-max value constraint", fontweight="600")
    save_figure(fig, out_dir, "method_schematic")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Figures dir.")
    args = parser.parse_args()
    out_dir = Path(args.out)
    design_space_map(out_dir)
    method_schematic(out_dir)
    print(f"[schematics] {out_dir}/png and {out_dir}/svg written")


if __name__ == "__main__":
    main()
