#!/usr/bin/env python3
"""Finalised paper figures: the 50-task shared-head GridWorld tier.

The point of this tier is that it removes the crutch the CKA-RL benchmark leans
on. There, tasks are **modes of one game** and every method gets **its own head
per task**, so the ceiling sits near 0.99 and there is almost nothing to
separate. Here there are 50 genuinely different layouts and **one shared
task-conditioned head**, so capacity is fixed, interference is forced, and
forgetting has somewhere to live.

Reads the tidy CSVs written under ``reports/gridworld_sharedhead/`` per the
logging contract (``docs/LOGGING_CONTRACT.md``). Nothing is duplicated here.

**These runs are partial.** Ours seeds 0 and 1 and CKA-RL had not reached 50
tasks at the time of writing; every figure draws each method to its own last
completed phase and says so. Re-running this script after the runs finish picks
the new rows up automatically.

Usage::

    python reports/final/gridworld/make_figures.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
sys.path.insert(0, str(REPO / "report"))

from acviz import (  # noqa: E402
    AC, FONT_MONO, FONT_UI, W_FULL, W_ONE_HALF, export_figure, height_for,
    hex_to_rgba, install_template,
)

RUNS = REPO / "reports" / "gridworld_sharedhead"
PNG_DIR, SVG_DIR = HERE / "png", HERE / "svg"
N_TASKS = 50

# ── Series identity, fixed across every figure in this folder ───────────────
# Ours is primary; CKA-RL is the method being compared against; fine-tuning is
# the catastrophic-forgetting case; the from-scratch baseline is a reference, so
# it stays neutral rather than competing for attention.
METHODS = [
    ("ours",     "Min-Max (ours)",   AC["blue"],
     ["biggrid50_sh_ours_seed0", "biggrid50_sh_ours_seed1",
      "biggrid50_sh_ours_seed2"]),
    ("cka_rl",   "CKA-RL",           AC["amber"],  ["biggrid50_cka_rl_seed0"]),
    ("finetune", "Fine-tuning",      AC["red"],    ["biggrid50_sh_finetune_seed0"]),
    ("baseline", "From-scratch",     AC["text_faint"], ["biggrid50_sh_baseline_seed0"]),
]


# ── Loading ─────────────────────────────────────────────────────────────────
def load_matrix(run: str) -> np.ndarray:
    """`[after_task, eval_task]` normalised scores; NaN where never evaluated."""
    grid = np.full((N_TASKS, N_TASKS), np.nan)
    with (RUNS / run / "forgetting_matrix.csv").open() as handle:
        for row in csv.DictReader(handle):
            grid[int(row["after_task"]), int(row["eval_task"])] = float(row["normalized"])
    return grid


def load_curves(run: str) -> dict[int, list[tuple[int, float]]]:
    """Within-phase learning curve per task: `{task: [(iter, normalized), ...]}`."""
    curves: dict[int, list[tuple[int, float]]] = defaultdict(list)
    with (RUNS / run / "learning_curves.csv").open() as handle:
        for row in csv.DictReader(handle):
            curves[int(row["task"])].append((int(row["iter"]), float(row["normalized"])))
    for task in curves:
        curves[task].sort()
    return curves


def load_phases(run: str) -> list[dict]:
    with (RUNS / run / "phases.csv").open() as handle:
        return list(csv.DictReader(handle))


def metrics() -> dict:
    return json.loads((RUNS / "metrics_partial.json").read_text(encoding="utf-8"))


def completed(run: str) -> int:
    """Number of end-of-task rows this run actually has."""
    return int(np.isfinite(load_matrix(run)).any(axis=1).sum())


def prior_retention(grid: np.ndarray) -> np.ndarray:
    """Mean normalised score over the tasks learned *before* each phase.

    The just-learned diagonal is excluded: it measures plasticity, not
    retention, and including it lets a method that merely learns the newest task
    well post a flattering curve.
    """
    out = np.full(N_TASKS, np.nan)
    for after in range(1, N_TASKS):
        prior = grid[after, :after]
        if np.isfinite(prior).any():
            out[after] = np.nanmean(prior)
    return out


# ── Figure 1: retention trajectory ──────────────────────────────────────────
def figure_retention_trajectory() -> None:
    """Mean retention of everything learned so far, as the sequence grows.

    The headline figure for this tier. With 50 tasks a full matrix is unreadable,
    but its row means are not: a method that forgets shows a falling curve, and
    one that consolidates shows a flat or rising one.
    """
    fig = go.Figure()
    ends: list[tuple[str, float, float, str]] = []

    for key, label, color, runs in METHODS:
        series = [prior_retention(load_matrix(run)) for run in runs]
        stack = np.vstack(series)
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(stack, axis=0)
            low, high = np.nanmin(stack, axis=0), np.nanmax(stack, axis=0)
        n_seeds = np.isfinite(stack).sum(axis=0)

        # Spread band only where more than one seed reports; drawing a band over
        # a single run would invent an interval that does not exist.
        if len(runs) > 1:
            band = np.where(n_seeds >= 2)[0]
            if band.size:
                xs = band + 1
                fig.add_trace(go.Scatter(
                    x=np.concatenate([xs, xs[::-1]]),
                    y=np.concatenate([high[band], low[band][::-1]]),
                    fill="toself", fillcolor=hex_to_rgba(color, 0.18),
                    line=dict(width=0), hoverinfo="skip", showlegend=False,
                ))

        valid = np.where(np.isfinite(mean))[0]
        fig.add_trace(go.Scatter(
            x=valid + 1, y=mean[valid], mode="lines",
            line=dict(color=color, width=2.2, shape="spline", smoothing=0.4),
            hovertemplate=f"{label}<br>%{{x}} tasks: %{{y:.3f}}<extra></extra>",
            showlegend=False,
        ))
        ends.append((label, float(valid[-1] + 1), float(mean[valid[-1]]), color))

    # Direct end-of-line labels rather than a legend box: four series.
    spread = {label: y for label, _, y, _ in ends}
    nudged = dict(spread)
    order = sorted(spread, key=spread.get)
    for lower, upper in zip(order, order[1:]):
        if nudged[upper] - nudged[lower] < 0.045:
            nudged[upper] = nudged[lower] + 0.045
    for label, x_end, _, color in ends:
        fig.add_annotation(
            x=x_end, y=nudged[label], text=f"<b>{label}</b>", showarrow=False,
            xanchor="left", xshift=9, yanchor="middle",
            font=dict(family=FONT_UI, size=11.5, color=color),
        )

    fig.update_xaxes(
        title=dict(text="tasks learned so far",
                   font=dict(family=FONT_UI, size=11.5, color=AC["text_primary"])),
        range=[0, N_TASKS + 1], dtick=10, showgrid=False, zeroline=False,
        showline=True, linecolor=AC["axis"], linewidth=1.2, ticklen=4,
        tickfont=dict(family=FONT_MONO, size=10, color=AC["text_muted"]),
    )
    fig.update_yaxes(
        title=dict(text="mean retention of prior tasks",
                   font=dict(family=FONT_UI, size=11.5, color=AC["text_primary"])),
        # Range covers From-scratch's dip below zero rather than clipping it:
        # zero here is random-policy performance, so going under it is real and
        # is exactly the kind of thing a truncated axis would hide.
        range=[-0.10, 0.78], showgrid=True, gridcolor=AC["grid"], gridwidth=0.6,
        zeroline=True, zerolinecolor=AC["border"], zerolinewidth=1.0,
        showline=True, linecolor=AC["axis"], linewidth=1.2,
        ticklen=4, tickfont=dict(family=FONT_MONO, size=10, color=AC["text_muted"]),
    )
    fig.add_annotation(
        x=0, y=0.0, xref="paper", yref="y", xshift=-4,
        text="random", showarrow=False, xanchor="right", yanchor="middle",
        font=dict(family=FONT_UI, size=8.5, color=AC["text_faint"]),
    )
    fig.add_annotation(
        x=0, y=-0.20, xref="paper", yref="paper", xshift=-58,
        text=("Mean over every task learned <i>before</i> each phase, in units of "
              "(score − random) / (1 − random), so 0 is a random policy and 1 is "
              "a perfect one. The just-learned<br>"
              "diagonal is excluded because it measures plasticity, not "
              "retention. Band is the min-max over 3 seeds where 2 or more "
              "report; no band is drawn for the single-seed<br>"
              "methods. <b>Partial runs:</b> ours is 1 seed past task 44, and "
              "CKA-RL, ours-seed0 and ours-seed1 each stop at their last "
              "completed phase."),
        showarrow=False, xanchor="left", yanchor="top", align="left",
        font=dict(family=FONT_UI, size=8.5, color=AC["text_muted"]),
    )
    fig.update_layout(title=None, showlegend=False,
                      margin=dict(l=72, r=132, t=16, b=104))
    export_pair(fig, "retention_trajectory", W_FULL, height_for(W_FULL, 1.58))


# ── Figure 2: headline metrics ──────────────────────────────────────────────
def figure_headline_metrics() -> None:
    """The four continual-learning metrics, one panel each.

    Lollipops rather than bars: four values per panel, and the thin stem leaves
    the panel quiet enough to carry the seed spread.
    """
    data = metrics()
    panels = [
        ("perf", "Average performance", "higher is better"),
        ("forgetting", "Forgetting", "lower is better"),
        ("bwt", "Backward transfer", "higher is better"),
        ("fwt", "Forward transfer", "higher is better"),
    ]

    fig = make_subplots(rows=1, cols=4, horizontal_spacing=0.055,
                        subplot_titles=[t for _, t, _ in panels])

    # Ours is the only method with repeats, so its per-metric standard deviation
    # is lent to the single-seed methods as a PLACEHOLDER until their own seeds
    # land. A borrowed interval is not a measurement, so it is drawn differently
    # — dotted and uncapped, against ours' solid capped bar — and daggered.
    ours_runs = next(runs for key, _l, _c, runs in METHODS if key == "ours")

    def value_of(runs: list[str], metric: str) -> list[float]:
        return [data[r][metric] for r in runs
                if data.get(r, {}).get(metric) is not None]

    # ONE row order for every panel, best average performance first. Sorting each
    # panel independently would put a different method on each row while the
    # labels are drawn once on the left, which silently mislabels three panels.
    order = sorted(METHODS, key=lambda m: -float(np.mean(value_of(m[3], "perf"))))
    row_of = {m[0]: i for i, m in enumerate(order)}

    for col, (metric, _title, _sense) in enumerate(panels, start=1):
        ours_values = value_of(ours_runs, metric)
        imputed_sd = float(np.std(ours_values, ddof=1)) if len(ours_values) > 1 else 0.0

        rows = []
        for key, label, color, runs in order:
            values = value_of(runs, metric)
            if not values:
                # No value for this metric: the from-scratch run is forward
                # transfer's own reference, so it is 0 by construction there.
                fig.add_annotation(
                    x=0, y=row_of[key], text="0, by construction", showarrow=False,
                    xanchor="left", yanchor="middle", xshift=6,
                    font=dict(family=FONT_UI, size=8.5, color=AC["text_faint"]),
                    row=1, col=col,
                )
                continue
            mean = float(np.mean(values))
            if len(values) > 1:
                sd, measured = float(np.std(values, ddof=1)), True
            else:
                sd, measured = imputed_sd, False
            rows.append((row_of[key], color, mean, sd, measured))

        for index, color, mean, sd, measured in rows:
            fig.add_trace(go.Scatter(
                x=[0, mean], y=[index, index], mode="lines",
                line=dict(color=hex_to_rgba(color, 0.40), width=3),
                hoverinfo="skip", showlegend=False,
            ), row=1, col=col)
            if sd > 1e-9:
                fig.add_trace(go.Scatter(
                    x=[mean - sd, mean + sd], y=[index, index], mode="lines",
                    line=dict(color=color, width=2 if measured else 1.4,
                              dash=None if measured else "dot"),
                    hoverinfo="skip", showlegend=False,
                ), row=1, col=col)
                if measured:
                    for edge in (mean - sd, mean + sd):
                        fig.add_trace(go.Scatter(
                            x=[edge, edge], y=[index - 0.13, index + 0.13],
                            mode="lines", line=dict(color=color, width=2),
                            hoverinfo="skip", showlegend=False,
                        ), row=1, col=col)
            fig.add_trace(go.Scatter(
                x=[mean], y=[index], mode="markers",
                marker=dict(color=color, size=10,
                            line=dict(color=AC["bg"], width=1.6)),
                hovertemplate="%{x:.3f}<extra></extra>", showlegend=False,
            ), row=1, col=col)
            value = f"{mean:+.2f}" if metric == "bwt" else f"{mean:.2f}"
            dagger = "" if measured else (
                f"<span style=\"font-size:8px;color:{AC['text_muted']}\">†</span>")
            fig.add_annotation(
                x=mean + sd, y=index, text=f"<b>{value}</b>{dagger}",
                showarrow=False, xanchor="left", yanchor="middle", xshift=8,
                font=dict(family=FONT_MONO, size=9.5, color=AC["text_primary"]),
                row=1, col=col,
            )

        span = [r[2] + r[3] for r in rows] + [r[2] - r[3] for r in rows]
        lo, hi = min(0.0, min(span)), max(span)
        pad = (hi - lo) * 0.40
        fig.update_xaxes(range=[lo - pad * 0.12, hi + pad], row=1, col=col,
                         showgrid=True, gridcolor=AC["grid"], gridwidth=0.6,
                         zeroline=True, zerolinecolor=AC["border"],
                         zerolinewidth=1.0, showline=False, ticklen=0, nticks=4,
                         tickfont=dict(family=FONT_MONO, size=8.5,
                                       color=AC["text_muted"]))
        fig.update_yaxes(
            tickmode="array", tickvals=list(range(len(order))),
            ticktext=[m[1] for m in order] if col == 1 else ["" for _ in order],
            range=[len(order) - 0.5, -0.5], showgrid=False, zeroline=False,
            showline=False, ticklen=0,
            tickfont=dict(family=FONT_UI, size=10, color=AC["text_primary"]),
            row=1, col=col,
        )

    for annotation, (_m, _t, sense) in zip(fig.layout.annotations[:4], panels):
        annotation.font = dict(family=FONT_UI, size=11, color=AC["text_primary"])
        annotation.y = 1.10
    for col, (_m, _t, sense) in enumerate(panels, start=1):
        fig.add_annotation(
            x=0.5, y=1.015, xref=f"x{col if col > 1 else ''} domain",
            yref="paper", text=sense, showarrow=False, xanchor="center",
            yanchor="bottom",
            font=dict(family=FONT_UI, size=8.5, color=AC["text_muted"]),
        )

    fig.add_annotation(
        x=0, y=-0.17, xref="paper", yref="paper", xshift=-100,
        text=("Rows keep one order in every panel, best average performance "
              "first. Ours is the mean over 3 seeds, its bar ±1 s.d. across "
              "them — measured, solid, capped.<br>"
              "<b>† marks a placeholder.</b> The other three have one seed each, "
              "so they carry ours' s.d. on that metric until their own seeds "
              "land. A lent interval is not a measurement,<br>"
              "which is why it is dotted and uncapped; do not read it as one.<br>"
              "<b>Partial runs:</b> ours seeds 0 and 1 stop at 45 and 44 tasks, "
              "CKA-RL at 41, so their forgetting and backward transfer are lower "
              "bounds."),
        showarrow=False, xanchor="left", yanchor="top", align="left",
        font=dict(family=FONT_UI, size=8.5, color=AC["text_muted"]),
    )
    fig.update_layout(title=None, showlegend=False, plot_bgcolor=AC["bg"],
                      margin=dict(l=104, r=20, t=54, b=92))
    export_pair(fig, "headline_metrics", W_FULL, height_for(W_FULL, 2.35))


# ── Figure 3: forgetting matrices ───────────────────────────────────────────
# Sequential, not diverging: this is "how much survived", which has a floor at
# random (0) and a ceiling at 1, and no meaningful midpoint to pivot about.
RETENTION_SCALE = [
    [0.00, "#DC2626"], [0.20, "#E8736F"], [0.40, "#F4B3AE"],
    [0.55, "#F2F3F5"], [0.72, "#B9CDF6"], [0.87, "#7CA2F0"], [1.00, "#2563EB"],
]


def sample_scale(scale: list, fraction: float) -> str:
    """Sample a colourscale at ``fraction`` in [0, 1], blending in sRGB."""
    fraction = min(max(fraction, 0.0), 1.0)
    for (low, low_hex), (high, high_hex) in zip(scale, scale[1:]):
        if fraction <= high:
            span = high - low
            weight = 0.0 if span == 0 else (fraction - low) / span
            left = [int(low_hex[i:i + 2], 16) for i in (1, 3, 5)]
            right = [int(high_hex[i:i + 2], 16) for i in (1, 3, 5)]
            return "#{:02X}{:02X}{:02X}".format(
                *[round(a + (b - a) * weight) for a, b in zip(left, right)])
    return scale[-1][1]


def figure_forgetting_matrices() -> None:
    """The full 50x50 lower triangle for each method.

    Cells are drawn as vector rectangles rather than as a ``go.Heatmap``, which
    rasterises into an embedded bitmap. No numbers: at 50x50 they would be
    illegible, and the texture is the point — where a method loses tasks, and
    whether the loss is a band, a column or the whole triangle.
    """
    panels = [(label, color, runs[-1] if key != "ours" else "biggrid50_sh_ours_seed2")
              for key, label, color, runs in METHODS]

    fig = make_subplots(rows=1, cols=4, horizontal_spacing=0.035,
                        subplot_titles=[p[0] for p in panels])

    # Shapes are collected and assigned to the layout in one go. add_shape
    # revalidates the whole layout on every call, which turns ~4,500 cells into
    # minutes; one assignment makes it a second.
    shapes: list[dict] = []
    for col, (label, _color, run) in enumerate(panels, start=1):
        grid = load_matrix(run)
        suffix = "" if col == 1 else str(col)
        for after in range(N_TASKS):
            for task in range(after + 1):
                value = grid[after, task]
                if not np.isfinite(value):
                    continue
                shapes.append(dict(
                    type="rect", layer="below",
                    x0=task, x1=task + 1, y0=after, y1=after + 1,
                    xref=f"x{suffix}", yref=f"y{suffix}",
                    fillcolor=sample_scale(RETENTION_SCALE, value),
                    line=dict(width=0),
                ))
        done = completed(run)
        if done < N_TASKS:
            fig.add_annotation(
                x=N_TASKS * 0.52, y=done + 3, xref=f"x{suffix}", yref=f"y{suffix}",
                text=f"stops at {done}", showarrow=False,
                xanchor="left", yanchor="top",
                font=dict(family=FONT_UI, size=8.5, color=AC["text_faint"]),
            )

    fig.update_layout(shapes=shapes)

    # One colourbar for all four panels, on an invisible marker trace so it
    # stays vector.
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers", hoverinfo="skip", showlegend=False,
        marker=dict(color=[0], colorscale=RETENTION_SCALE, cmin=0.0, cmax=1.0,
                    showscale=True, opacity=0,
                    colorbar=dict(
                        title=dict(text="retained", side="right",
                                   font=dict(family=FONT_UI, size=10,
                                             color=AC["text_muted"])),
                        tickmode="array", tickvals=[0.0, 0.5, 1.0],
                        ticktext=["0 = random", "0.5", "1 = solved"],
                        thickness=9, len=0.72, y=0.5, yanchor="middle", x=1.015,
                        outlinewidth=0, ticklen=3, tickcolor=AC["border"],
                        tickfont=dict(family=FONT_MONO, size=8.5,
                                      color=AC["text_muted"]))),
    ), row=1, col=4)

    for annotation in fig.layout.annotations[:4]:
        annotation.font = dict(family=FONT_UI, size=11, color=AC["text_primary"])
        annotation.y = 1.035

    ticks = [0.5, 9.5, 19.5, 29.5, 39.5, 49.5]
    labels = ["1", "10", "20", "30", "40", "50"]
    fig.update_xaxes(range=[0, N_TASKS], tickmode="array", tickvals=ticks,
                     ticktext=labels, showgrid=False, zeroline=False,
                     showline=False, ticklen=0,
                     tickfont=dict(family=FONT_MONO, size=8.5,
                                   color=AC["text_muted"]))
    fig.update_yaxes(range=[N_TASKS, 0], tickmode="array", tickvals=ticks,
                     ticktext=labels, showgrid=False, zeroline=False,
                     showline=False, ticklen=0,
                     tickfont=dict(family=FONT_MONO, size=8.5,
                                   color=AC["text_muted"]))
    fig.update_yaxes(title=dict(text="after consolidating task",
                                font=dict(family=FONT_UI, size=10,
                                          color=AC["text_muted"])),
                     row=1, col=1)
    # One shared x-axis title under the row rather than four, which would
    # repeat the same words and crowd the footnote.
    fig.add_annotation(
        x=0.44, y=-0.075, xref="paper", yref="paper", text="evaluated on task",
        showarrow=False, xanchor="center", yanchor="top",
        font=dict(family=FONT_UI, size=10, color=AC["text_muted"]),
    )
    fig.add_annotation(
        x=0, y=-0.19, xref="paper", yref="paper", xshift=-46,
        text=("Every end-of-task evaluation, in units of (score − random) / "
              "(1 − random). Rows are training phases, columns are tasks in "
              "learning order;<br>"
              "the upper triangle is never evaluated. Ours is seed 2, the one "
              "complete run.<br>"
              "Read down a column for one task's fate: ours keeps its columns "
              "blue, while fine-tuning and from-scratch bleed to red within a "
              "few phases."),
        showarrow=False, xanchor="left", yanchor="top", align="left",
        font=dict(family=FONT_UI, size=8.5, color=AC["text_muted"]),
    )
    fig.update_layout(title=None, showlegend=False, plot_bgcolor=AC["bg"],
                      margin=dict(l=56, r=104, t=42, b=104))
    export_pair(fig, "forgetting_matrices", W_FULL, height_for(W_FULL, 2.05))


# ── Figure 4: retention by task age ─────────────────────────────────────────
def figure_retention_by_age() -> None:
    """How much a task retains as a function of how long ago it was learned.

    The trajectory figure shows retention against sequence position; this shows
    it against *elapsed phases*, pooling every task at a given age. It is the
    decay curve, and it separates a method that forgets slowly from one that
    forgets immediately and then holds.
    """
    fig = go.Figure()
    ends = []

    for key, label, color, runs in METHODS:
        per_age: dict[int, list[float]] = defaultdict(list)
        for run in runs:
            grid = load_matrix(run)
            for after in range(N_TASKS):
                for task in range(after + 1):
                    if np.isfinite(grid[after, task]):
                        per_age[after - task].append(grid[after, task])
        ages = sorted(a for a in per_age if len(per_age[a]) >= 5)
        means = [float(np.mean(per_age[a])) for a in ages]
        fig.add_trace(go.Scatter(
            x=ages, y=means, mode="lines",
            line=dict(color=color, width=2.2, shape="spline", smoothing=0.5),
            hovertemplate=f"{label}<br>age %{{x}}: %{{y:.3f}}<extra></extra>",
            showlegend=False,
        ))
        ends.append((label, ages[-1], means[-1], color))

    spread = {label: y for label, _, y, _ in ends}
    nudged = dict(spread)
    for lower, upper in zip(sorted(spread, key=spread.get),
                            sorted(spread, key=spread.get)[1:]):
        if nudged[upper] - nudged[lower] < 0.05:
            nudged[upper] = nudged[lower] + 0.05
    for label, x_end, _, color in ends:
        fig.add_annotation(
            x=x_end, y=nudged[label], text=f"<b>{label}</b>", showarrow=False,
            xanchor="left", xshift=9, yanchor="middle",
            font=dict(family=FONT_UI, size=11.5, color=color),
        )

    fig.update_xaxes(
        title=dict(text="phases since the task was learned",
                   font=dict(family=FONT_UI, size=11.5, color=AC["text_primary"])),
        showgrid=False, zeroline=False, showline=True, linecolor=AC["axis"],
        linewidth=1.2, ticklen=4, dtick=10,
        tickfont=dict(family=FONT_MONO, size=10, color=AC["text_muted"]))
    fig.update_yaxes(
        title=dict(text="mean retention",
                   font=dict(family=FONT_UI, size=11.5, color=AC["text_primary"])),
        showgrid=True, gridcolor=AC["grid"], gridwidth=0.6,
        zeroline=True, zerolinecolor=AC["border"], zerolinewidth=1.0,
        showline=True, linecolor=AC["axis"], linewidth=1.2, ticklen=4,
        tickfont=dict(family=FONT_MONO, size=10, color=AC["text_muted"]))
    fig.add_annotation(
        x=0, y=-0.19, xref="paper", yref="paper", xshift=-58,
        text=("Every matrix cell pooled by age, where age 0 is the task just "
              "learned. Ages with fewer than 5 contributing cells are dropped, "
              "so the tail is not one noisy task.<br>"
              "Ours pools 3 seeds, the rest 1 each.<br>"
              "<b>The crossing near age 2 is the whole trade.</b> Ours learns a "
              "new task to a markedly lower immediate level, then climbs and "
              "holds; the others start higher and decay.<br>"
              "Plasticity is what ours pays, retention is what it buys."),
        showarrow=False, xanchor="left", yanchor="top", align="left",
        font=dict(family=FONT_UI, size=8.5, color=AC["text_muted"]))
    fig.update_layout(title=None, showlegend=False,
                      margin=dict(l=66, r=132, t=16, b=112))
    export_pair(fig, "retention_by_age", W_FULL, height_for(W_FULL, 1.72))


# ── Export ──────────────────────────────────────────────────────────────────
def export_pair(fig: go.Figure, stem: str, width: int, height: int) -> None:
    """Write `png/<stem>.png` and `svg/<stem>.svg`, both verified non-empty."""
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    SVG_DIR.mkdir(parents=True, exist_ok=True)
    export_figure(fig, PNG_DIR / stem, width, height, label=stem)
    (PNG_DIR / f"{stem}.svg").replace(SVG_DIR / f"{stem}.svg")
    print(f"  png/{stem}.png  svg/{stem}.svg")


def main() -> int:
    install_template()
    print("reports/final/gridworld")
    figure_retention_trajectory()
    figure_headline_metrics()
    figure_forgetting_matrices()
    figure_retention_by_age()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
