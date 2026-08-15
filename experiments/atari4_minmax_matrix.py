"""Visualize ONE continual-Atari run as a lower-triangular, color-coded score matrix.

Rows = training stage (state AFTER learning game k); columns = evaluated game.
Cell (i, j), i >= j = the greedy game score on game j after game i was learned.
The upper triangle (games not yet seen) is masked out. Cells are annotated with
the raw game score and colored by the per-game *normalized* score
``(raw - random) / (target - random)`` so colors are comparable across games of
very different scales (0 = random play, 1 = the per-game target/expert ceiling).

Usage:
    python -m experiments.atari4_minmax_matrix \
        --run results/atari4_minmax_seed0 --out reports/atari4_minmax \
        --title "Min-max (constrained) — 4-game continual Atari"
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

from analysis.continual_metrics import cl_metrics, normalize_matrix
from crl.envs.atari import RANDOM_SCORES


def _load(run_dir: Path):
    rows = json.load(open(run_dir / "eval_matrix.json"))
    T = len(rows)
    M = np.full((T, T), np.nan, dtype=float)
    for i, r in enumerate(rows):
        for j, v in enumerate(r[:T]):
            M[i, j] = v
    cfg = yaml.safe_load(open(run_dir / "config.yaml"))
    tasks = cfg["env"].get("tasks") or cfg["env"]["params"].get("tasks")
    games = [t["game"] for t in tasks][:T]
    targets = [float(t.get("threshold", np.nan)) for t in tasks][:T]
    return M, games, targets


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="run dir with eval_matrix.json + config.yaml")
    ap.add_argument("--out", default="reports/atari4_minmax", help="output dir")
    ap.add_argument("--title", default="Continual Atari — forgetting matrix")
    args = ap.parse_args()

    run = Path(args.run)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    M, games, targets = _load(run)
    T = len(games)

    # Normalized scores drive the cell color (comparable across games); raw scores
    # are the printed annotation. Mask the upper triangle (game not yet learned).
    Mn = normalize_matrix(M, games, targets)
    tri = np.tri(T, dtype=bool)                     # lower triangle incl. diagonal = seen
    disp = np.ma.masked_where(~tri | ~np.isfinite(Mn), Mn)

    fig, ax = plt.subplots(figsize=(1.7 * T + 2.6, 1.45 * T + 1.8))
    # Diverging, colorblind-safe map (RdBu is CVD-safe) centered at the RANDOM
    # baseline (normalized 0). vmin/vmax span the ACTUAL data range so nothing is
    # silently clipped: below-random (Boxing -27 vs -5.6) and above-target (Pong)
    # cells stay distinguishable.
    finite = Mn[tri & np.isfinite(Mn)]
    lo = float(min(finite.min(), 0.0)) - 0.05
    hi = float(max(finite.max(), 1.0)) + 0.05
    norm = TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="#e9e9e9")                   # masked (future) cells -> light grey
    im = ax.imshow(disp, cmap=cmap, norm=norm, aspect="auto")

    for i in range(T):
        for j in range(T):
            if not tri[i, j]:
                continue
            raw = M[i, j]
            nv = Mn[i, j]
            txt = f"{raw:.1f}"
            # contrast: white text on dark cells, black on light (per-cell luminance)
            r, g, b, _ = cmap(norm(nv)) if np.isfinite(nv) else (1.0, 1.0, 1.0, 1.0)
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            color = "white" if lum < 0.5 else "black"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=11,
                    fontweight="bold")

    xlabels = [f"{g}\n({RANDOM_SCORES.get(g, 0.0):g}→{t:.0f})"
               for g, t in zip(games, targets)]
    ax.set_xticks(range(T)); ax.set_xticklabels(xlabels, rotation=0, fontsize=9)
    ax.set_yticks(range(T))
    ax.set_yticklabels([f"after learning\nT{i+1}: {games[i]}" for i in range(T)], fontsize=9)
    ax.set_xlabel("evaluated game", fontsize=11)
    ax.set_ylabel("training stage", fontsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    raw_m = cl_metrics(M)
    norm_m = cl_metrics(Mn)
    sub = (f"AP={raw_m['avg_performance']:.1f}  Forgetting={raw_m['forgetting']:.1f}  "
           f"BWT={raw_m['bwt']:+.1f}   |   normalized: AP={norm_m['avg_performance']:.2f}  "
           f"F={norm_m['forgetting']:.2f}  BWT={norm_m['bwt']:+.2f}")
    ax.set_title(f"{args.title}\n{sub}", fontsize=11)

    # outline the "just-learned" diagonal (what forgetting/BWT are measured against)
    for i in range(T):
        ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                               edgecolor="black", lw=2.2))

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, ticks=[lo, 0.0, 1.0, hi])
    cbar.set_label("normalized score: 0 = random baseline, 1 = per-game target\n"
                   "(diverging at random; full data range shown, no clipping)",
                   fontsize=8)
    fig.text(0.5, 0.005,
             "Single seed. 'Forgetting=0' means no game dropped below its running max "
             "in this run — not proof of an anti-forgetting mechanism.",
             ha="center", fontsize=7.5, style="italic", color="#444444")
    fig.tight_layout(rect=[0, 0.03, 1, 1])

    png = out / "score_matrix.png"
    fig.savefig(png, dpi=160, bbox_inches="tight")

    # Also dump a CSV + markdown table of the raw scores (lower triangle).
    with open(out / "score_matrix.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage \\ game"] + games)
        for i in range(T):
            w.writerow([f"after_T{i+1}_{games[i]}"] +
                       [f"{M[i, j]:.2f}" if j <= i else "" for j in range(T)])

    md = ["| training stage \\ game | " + " | ".join(games) + " |",
          "|" + "---|" * (T + 1)]
    for i in range(T):
        cells = [f"{M[i, j]:.1f}" if j <= i else "" for j in range(T)]
        md.append(f"| after T{i+1}: {games[i]} | " + " | ".join(cells) + " |")
    (out / "score_matrix.md").write_text("\n".join(md) + "\n")

    json.dump({"games": games, "targets": targets, "matrix": M.tolist(),
               "raw_metrics": raw_m, "norm_metrics": norm_m},
              open(out / "score_matrix.json", "w"), indent=2)

    print(f"[minmax matrix] wrote {png}")
    print(f"  + score_matrix.csv / .md / .json in {out}")
    print("\nLower-triangular raw-score matrix (rows = after learning Tk, cols = game):")
    print("            " + "".join(f"{g[:8]:>9s}" for g in games))
    for i in range(T):
        cells = "".join(f"{M[i, j]:9.1f}" if j <= i else f"{'—':>9s}" for j in range(T))
        print(f"after T{i+1} {games[i][:7]:<8s}{cells}")


if __name__ == "__main__":
    main()
