#!/usr/bin/env python3
"""Finalised paper figures: the 50-task shared-head GridWorld tier.

This tier removes the two things that make the CKA-RL benchmark easy. There the
tasks are **modes of one game** and every method gets **its own head per task**,
so ceilings sit near 0.99 and a large part of each task's solution never shares
capacity. Here there are 50 different layouts and **one shared task-conditioned
head**: capacity is fixed and interference is forced.

Reads the tidy CSVs under ``reports/gridworld_sharedhead/`` per the logging
contract. Which runs count as complete is read from ``metrics.json``, so partial
seeds join the aggregates automatically as they finish — nothing here needs
editing when a run lands.

Usage::

    python reports/final/gridworld/make_figures.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
sys.path.insert(0, str(REPO / "report"))

from acviz import (  # noqa: E402
    AC, FONT_MONO, FONT_UI, W_FULL, export_figure, hex_to_rgba, install_template,
)

RUNS = REPO / "reports" / "gridworld_sharedhead"
PNG_DIR, SVG_DIR = HERE / "png", HERE / "svg"
N_TASKS = 50

# Type scale, one step larger than the Atari set. Those were legible on screen
# but small on paper; these are sized for a column in a two-column layout.
FS_PANEL, FS_AXIS, FS_TICK = 13.0, 12.0, 10.5
FS_VALUE, FS_NOTE = 11.0, 9.5

METHODS = [
    ("ours",     "Min-Max (ours)", AC["blue"],       "biggrid50_sh_ours_"),
    ("cka_rl",   "CKA-RL",         AC["amber"],      "biggrid50_cka_rl_"),
    ("finetune", "Fine-tuning",    AC["red"],        "biggrid50_sh_finetune_"),
    ("baseline", "From-scratch",   AC["text_faint"], "biggrid50_sh_baseline_"),
]


# ── Loading ─────────────────────────────────────────────────────────────────
def metrics() -> dict:
    return json.loads((RUNS / "metrics.json").read_text(encoding="utf-8"))


def complete_runs(prefix: str) -> list[str]:
    """Runs of this method that reached all 50 tasks.

    Partial runs are excluded from every aggregate: averaging a 50-task run with
    a 24-task one produces a number that belongs to neither.
    """
    return sorted(name for name, row in metrics().items()
                  if name.startswith(prefix) and row.get("complete"))


def load_matrix(run: str, column: str = "normalized") -> np.ndarray:
    grid = np.full((N_TASKS, N_TASKS), np.nan)
    with (RUNS / run / "forgetting_matrix.csv").open() as handle:
        for row in csv.DictReader(handle):
            grid[int(row["after_task"]), int(row["eval_task"])] = float(row[column])
    return grid


def load_phases(run: str) -> list[dict]:
    with (RUNS / run / "phases.csv").open() as handle:
        return list(csv.DictReader(handle))


def last_phase(grid: np.ndarray) -> int:
    return int(np.isfinite(grid).any(axis=1).sum()) - 1


def learned_and_final(run: str, column: str = "normalized") -> tuple[np.ndarray, np.ndarray]:
    """Per task: its score when just learned, and at the end of the sequence."""
    grid = load_matrix(run, column)
    end = last_phase(grid)
    learned = np.array([grid[i, i] for i in range(end + 1)])
    final = grid[end, :end + 1]
    keep = np.isfinite(learned) & np.isfinite(final)
    return learned[keep], final[keep]


def add_footnote(fig: go.Figure, text: str, left_margin: int,
                 clear_axis_title: bool = True) -> None:
    """Place a footnote below the plot, measured in pixels rather than fractions.

    Anchoring at a paper fraction fails on short figures: -0.2 of a shallow plot
    area is only a few pixels, so the note lands on the x-axis title. Pixels are
    what actually need clearing, so pixels are what this uses.
    """
    fig.add_annotation(
        x=0, y=0, xref="paper", yref="paper",
        xshift=-left_margin + 4, yshift=-(64 if clear_axis_title else 34),
        text=text, showarrow=False, xanchor="left", yanchor="top", align="left",
        font=dict(family=FONT_UI, size=FS_NOTE, color=AC["text_muted"]))


# ── Figure 1: learned vs retained ───────────────────────────────────────────
def figure_learned_vs_retained() -> None:
    """Each task's score when learned against its score at the end.

    The separator. Average performance puts ours and CKA-RL within 0.09 of each
    other, which is easy to wave away; this asks a yes/no question of every task
    — did it end better or worse than when it was learned — and the answer
    splits the methods completely. The diagonal is "no change": above it a task
    improved after the model moved on, below it the task was forgotten.

    Laid out 2x2 rather than 1x4 so each panel has room for 50-150 points, and
    the headline percentage sits above the panel rather than on top of the data.
    """
    fig = make_subplots(rows=2, cols=2, horizontal_spacing=0.13,
                        vertical_spacing=0.20)

    for index, (key, label, color, prefix) in enumerate(METHODS):
        row, col = index // 2 + 1, index % 2 + 1
        runs = complete_runs(prefix)
        pairs = [learned_and_final(r) for r in runs]
        learned = np.concatenate([p[0] for p in pairs])
        final = np.concatenate([p[1] for p in pairs])
        improved = float(np.mean(final > learned))

        fig.add_trace(go.Scatter(
            x=[-0.6, 1.05], y=[-0.6, 1.05], mode="lines",
            line=dict(color=AC["border"], width=1.3),
            hoverinfo="skip", showlegend=False), row=row, col=col)
        fig.add_trace(go.Scatter(
            x=learned, y=final, mode="markers",
            marker=dict(color=hex_to_rgba(color, 0.55), size=6.5,
                        line=dict(color=color, width=0.7)),
            hovertemplate="learned %{x:.2f} → final %{y:.2f}<extra></extra>",
            showlegend=False), row=row, col=col)

        # Title and headline live above the panel, clear of every point.
        axis = f"{index + 1 if index else ''}"
        fig.add_annotation(
            x=0, y=1.19, xref=f"x{axis} domain", yref=f"y{axis} domain",
            text=f"<b>{label}</b>", showarrow=False,
            xanchor="left", yanchor="bottom",
            font=dict(family=FONT_UI, size=FS_PANEL, color=AC["text_primary"]))
        fig.add_annotation(
            x=0, y=1.035, xref=f"x{axis} domain", yref=f"y{axis} domain",
            text=f"<b>{improved:.0%}</b> of tasks ended better than when learned",
            showarrow=False, xanchor="left", yanchor="bottom",
            font=dict(family=FONT_UI, size=FS_TICK, color=color))

    fig.update_xaxes(range=[-0.6, 1.05], dtick=0.5, showgrid=True,
                     gridcolor=AC["grid"], gridwidth=0.6, zeroline=False,
                     showline=True, linecolor=AC["axis"], linewidth=1.2,
                     ticklen=4, tickfont=dict(family=FONT_MONO, size=FS_TICK,
                                              color=AC["text_muted"]))
    fig.update_yaxes(range=[-0.6, 1.05], dtick=0.5, showgrid=True,
                     gridcolor=AC["grid"], gridwidth=0.6, zeroline=False,
                     showline=True, linecolor=AC["axis"], linewidth=1.2,
                     ticklen=4, tickfont=dict(family=FONT_MONO, size=FS_TICK,
                                              color=AC["text_muted"]))
    title = dict(font=dict(family=FONT_UI, size=FS_AXIS,
                           color=AC["text_primary"]))
    for row in (1, 2):
        fig.update_yaxes(title=dict(text="score after all 50 tasks", **title),
                         row=row, col=1)
    for col in (1, 2):
        fig.update_xaxes(title=dict(text="score when the task was just learned",
                                    **title), row=2, col=col)

    add_footnote(fig, (
        "One point per task, in units of (score − random) / (1 − random). The "
        "grey line is no change.<br>"
        "<b>Above it a task improved after the model moved on; below it the task "
        "was forgotten.</b><br>"
        "Ours contributes <b>150 points: 50 tasks × 3 complete seeds</b>.<br>"
        "The other three have one complete seed each so far, so 50 points "
        "apiece; their remaining seeds are still running.<br>"
        "This is the backward-transfer column read one task at a time, and it is "
        "where ours and CKA-RL stop being close."), 78)
    fig.update_layout(title=None, showlegend=False, plot_bgcolor=AC["bg"],
                      margin=dict(l=78, r=26, t=56, b=136))
    export_pair(fig, "learned_vs_retained", W_FULL, 656)


# ── Figure 2: retention across the sequence ─────────────────────────────────
def figure_retention_curve() -> None:
    """After finishing task k, how the model is doing on the k-1 tasks behind it.

    The state of the whole system as the sequence grows: a method that forgets
    shows a falling curve, one that consolidates shows a flat or rising one. At
    50 tasks this is the readable form of the forgetting matrix, whose triangle
    is far too dense to see anything in.
    """
    fig = go.Figure()
    ends: list[tuple[str, float, float, str]] = []

    for key, label, color, prefix in METHODS:
        runs = complete_runs(prefix)
        per_seed = np.full((len(runs), N_TASKS), np.nan)
        for index, run in enumerate(runs):
            grid = load_matrix(run)
            for after in range(1, N_TASKS):
                prior = grid[after, :after]
                if np.isfinite(prior).any():
                    per_seed[index, after] = np.nanmean(prior)
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(per_seed, axis=0)
            low, high = np.nanmin(per_seed, axis=0), np.nanmax(per_seed, axis=0)

        # Band only where more than one seed reports. Drawing one over a single
        # run would invent an interval that does not exist.
        if len(runs) > 1:
            band = np.where(np.isfinite(per_seed).sum(axis=0) >= 2)[0]
            if band.size:
                xs = band + 1
                fig.add_trace(go.Scatter(
                    x=np.concatenate([xs, xs[::-1]]),
                    y=np.concatenate([high[band], low[band][::-1]]),
                    fill="toself", fillcolor=hex_to_rgba(color, 0.20),
                    line=dict(width=0), hoverinfo="skip", showlegend=False))

        valid = np.where(np.isfinite(mean))[0]
        fig.add_trace(go.Scatter(
            x=valid + 1, y=mean[valid], mode="lines",
            line=dict(color=color, width=2.6, shape="spline", smoothing=0.5),
            hovertemplate=f"{label}<br>%{{x}} tasks: %{{y:.3f}}<extra></extra>",
            showlegend=False))
        ends.append((label, float(valid[-1] + 1), float(mean[valid[-1]]), color))

    positions = {label: y for label, _x, y, _c in ends}
    for lower, upper in zip(sorted(positions, key=positions.get),
                            sorted(positions, key=positions.get)[1:]):
        if positions[upper] - positions[lower] < 0.055:
            positions[upper] = positions[lower] + 0.055
    for label, x_end, _y, color in ends:
        fig.add_annotation(
            x=x_end, y=positions[label], text=f"<b>{label}</b>", showarrow=False,
            xanchor="left", xshift=10, yanchor="middle",
            font=dict(family=FONT_UI, size=FS_AXIS, color=color))

    fig.update_xaxes(
        title=dict(text="tasks learned so far",
                   font=dict(family=FONT_UI, size=FS_AXIS,
                             color=AC["text_primary"])),
        range=[0, N_TASKS + 1], dtick=10, showgrid=False, zeroline=False,
        showline=True, linecolor=AC["axis"], linewidth=1.2, ticklen=4,
        tickfont=dict(family=FONT_MONO, size=FS_TICK, color=AC["text_muted"]))
    fig.update_yaxes(
        title=dict(text="mean score of the tasks already learned",
                   font=dict(family=FONT_UI, size=FS_AXIS,
                             color=AC["text_primary"])),
        range=[-0.02, 0.80], showgrid=True, gridcolor=AC["grid"], gridwidth=0.6,
        zeroline=True, zerolinecolor=AC["border"], zerolinewidth=1.0,
        showline=True, linecolor=AC["axis"], linewidth=1.2, ticklen=4,
        tickfont=dict(family=FONT_MONO, size=FS_TICK, color=AC["text_muted"]))

    add_footnote(fig, (
        "After finishing task k, the mean score over the k−1 tasks learned "
        "before it, in units of (score − random) / (1 − random).<br>"
        "The just-learned task is excluded, because including it lets a method "
        "that merely learns the newest task well post a flattering curve.<br>"
        "<b>The shaded band is the min-max over ours' 3 complete seeds.</b><br>"
        "The other three have one complete seed each so far, so they carry no "
        "band; their remaining seeds are still running."), 74)
    fig.update_layout(title=None, showlegend=False,
                      margin=dict(l=74, r=150, t=24, b=122))
    export_pair(fig, "retention_curve", W_FULL, 446)


# ── Figure 3: raw against normalised ────────────────────────────────────────
def figure_raw_vs_normalised() -> None:
    """Final per-task score on both scales, and why the choice matters.

    A random policy on this grid already collects most of the raw discounted
    return, so raw scores crowd into a narrow band and the methods look closer
    than they are. The random floor also varies per task, so the same raw number
    is a different achievement on different tasks. Showing both panels makes the
    case that the separation normalisation exposes is real rather than an
    artefact of the normaliser.
    """
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.09,
                        subplot_titles=["Raw discounted return",
                                        "Normalised against each task's random floor"])

    for col, column in enumerate(["raw", "normalized"], start=1):
        for index, (key, label, color, prefix) in enumerate(METHODS):
            values = np.concatenate([learned_and_final(r, column)[1]
                                     for r in complete_runs(prefix)])
            jitter = (np.random.default_rng(index).random(len(values)) - 0.5) * 0.28
            fig.add_trace(go.Scatter(
                x=values, y=index + jitter, mode="markers",
                marker=dict(color=hex_to_rgba(color, 0.30), size=5,
                            line=dict(width=0)),
                hovertemplate="%{x:.3f}<extra></extra>", showlegend=False,
            ), row=1, col=col)
            # The median rule is drawn after the points so it sits on top, and
            # runs taller than the cloud: at this density a short tick vanishes.
            median = float(np.median(values))
            fig.add_trace(go.Scatter(
                x=[median, median], y=[index - 0.33, index + 0.33], mode="lines",
                line=dict(color=color, width=4.5),
                hoverinfo="skip", showlegend=False), row=1, col=col)
            fig.add_annotation(
                x=median, y=index - 0.35, text=f"<b>{median:.2f}</b>",
                showarrow=False, xanchor="center", yanchor="bottom", yshift=3,
                font=dict(family=FONT_MONO, size=FS_VALUE, color=color),
                bgcolor="rgba(255,255,255,0.90)", borderpad=2,
                row=1, col=col)

    fig.update_yaxes(tickmode="array", tickvals=list(range(len(METHODS))),
                     ticktext=[m[1] for m in METHODS],
                     range=[len(METHODS) - 0.35, -0.90], showgrid=False,
                     zeroline=False, showline=False, ticklen=0,
                     tickfont=dict(family=FONT_UI, size=FS_AXIS,
                                   color=AC["text_primary"]))
    fig.update_yaxes(showticklabels=False, row=1, col=2)
    fig.update_xaxes(showgrid=True, gridcolor=AC["grid"], gridwidth=0.6,
                     zeroline=False, showline=True, linecolor=AC["axis"],
                     linewidth=1.2, ticklen=4,
                     tickfont=dict(family=FONT_MONO, size=FS_TICK,
                                   color=AC["text_muted"]))
    axis_title = dict(font=dict(family=FONT_UI, size=FS_AXIS,
                                color=AC["text_muted"]))
    fig.update_xaxes(range=[0.55, 1.02], dtick=0.1, row=1, col=1,
                     title=dict(text="discounted return", **axis_title))
    fig.update_xaxes(range=[-0.55, 1.05], dtick=0.5, zeroline=True,
                     zerolinecolor=AC["border"], zerolinewidth=1.0, row=1, col=2,
                     title=dict(text="0 = random policy, 1 = solved", **axis_title))
    for annotation in fig.layout.annotations[:2]:
        annotation.font = dict(family=FONT_UI, size=FS_PANEL,
                               color=AC["text_primary"])
        annotation.y = 1.09

    add_footnote(fig, (
        "One point per task after all 50 are learned; the heavy rule is the "
        "median. <b>Both panels are the same runs.</b><br>"
        "A random policy already collects most of the raw return here, so raw "
        "scores crowd into [0.59, 0.99] and the methods look closer than they "
        "are.<br>"
        "The random floor also varies per task, from 0.55 to 0.87, so the same "
        "raw number is a different achievement on different tasks.<br>"
        "Normalising against each task's own floor is what makes them "
        "comparable. Ours is 3 complete seeds, the others 1 each."), 118)
    fig.update_layout(title=None, showlegend=False, plot_bgcolor=AC["bg"],
                      margin=dict(l=118, r=26, t=56, b=136))
    export_pair(fig, "raw_vs_normalised", W_FULL, 440)


# ── Figure 4: compute cost ──────────────────────────────────────────────────
def figure_compute_cost() -> None:
    """Wall-clock for one complete 50-task run."""
    fig = go.Figure()
    totals = {key: np.array([
        sum(float(r["wall_s_phase"]) for r in load_phases(run)) / 60.0
        for run in complete_runs(prefix)])
        for key, _label, _color, prefix in METHODS}

    reference = float(np.mean(totals["ours"]))
    top = max(float(v.max()) for v in totals.values())

    for index, (key, label, color, _prefix) in enumerate(METHODS):
        values = totals[key]
        mean = float(np.mean(values))
        fig.add_trace(go.Scatter(
            x=[0, top * 1.32], y=[index, index], mode="lines",
            line=dict(color=AC["grid"], width=0.9),
            hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(
            x=[0, mean], y=[index, index], mode="lines",
            line=dict(color=hex_to_rgba(color, 0.40), width=4),
            hoverinfo="skip", showlegend=False))
        if len(values) > 1:
            fig.add_trace(go.Scatter(
                x=[values.min(), values.max()], y=[index, index], mode="lines",
                line=dict(color=color, width=2.4),
                hoverinfo="skip", showlegend=False))
            for edge in (values.min(), values.max()):
                fig.add_trace(go.Scatter(
                    x=[edge, edge], y=[index - 0.11, index + 0.11], mode="lines",
                    line=dict(color=color, width=2.4),
                    hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(
            x=[mean], y=[index], mode="markers",
            marker=dict(color=color, size=13,
                        line=dict(color=AC["bg"], width=2)),
            hovertemplate="%{x:.0f} min<extra></extra>", showlegend=False))
        tail = "" if key == "ours" else (
            f"<span style=\"font-size:{FS_TICK}px;color:{AC['text_muted']}\">"
            f"  {mean / reference:.2f}×</span>")
        fig.add_annotation(
            x=float(values.max()), y=index, text=f"<b>{mean:,.0f} min</b>{tail}",
            showarrow=False, xanchor="left", yanchor="middle", xshift=11,
            font=dict(family=FONT_MONO, size=FS_VALUE, color=AC["text_primary"]))

    fig.update_xaxes(
        title=dict(text="wall-clock minutes for one complete 50-task run",
                   font=dict(family=FONT_UI, size=FS_AXIS,
                             color=AC["text_primary"])),
        range=[0, top * 1.32], showgrid=True, gridcolor=AC["grid"],
        gridwidth=0.6, zeroline=False, showline=True, linecolor=AC["border"],
        ticklen=0, tickfont=dict(family=FONT_MONO, size=FS_TICK,
                                 color=AC["text_muted"]))
    fig.update_yaxes(tickmode="array", tickvals=list(range(len(METHODS))),
                     ticktext=[m[1] for m in METHODS],
                     range=[len(METHODS) - 0.5, -0.5], showgrid=False,
                     zeroline=False, showline=False, ticklen=0,
                     tickfont=dict(family=FONT_UI, size=FS_AXIS,
                                   color=AC["text_primary"]))

    add_footnote(fig, (
        "× is relative to ours. Complete runs only; the capped segment on ours "
        "is its min-max over 3 seeds, the others are single runs.<br>"
        "<b>Ours is the most expensive because consolidation re-simulates past "
        "environments</b>, spending time on tasks it has already learned.<br>"
        "Measured on a shared, contended cluster, so read it as indicative."), 122)
    fig.update_layout(title=None, showlegend=False, plot_bgcolor=AC["bg"],
                      margin=dict(l=122, r=126, t=20, b=116))
    export_pair(fig, "compute_cost", W_FULL, 300)


# ── Figure 5: headline metrics, as a table ──────────────────────────────────
def figure_headline_table() -> None:
    """The four continual-learning metrics as a booktabs table.

    A table rather than four lollipop panels: there are sixteen numbers and the
    reader wants to compare them exactly, which is what a table is for.
    """
    data = metrics()
    rows = [("perf", "Average performance", "higher is better", +1),
            ("forgetting", "Forgetting", "lower is better", -1),
            ("bwt", "Backward transfer", "higher is better", +1),
            ("fwt", "Forward transfer", "higher is better", +1)]

    def values_for(prefix: str, metric: str) -> list[float]:
        return [data[r][metric] for r in complete_runs(prefix)
                if data[r].get(metric) is not None]

    label_x = 0.0
    col_x = [40.0, 58.0, 76.0, 94.0]
    right_edge = col_x[-1] + 2

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                             hoverinfo="skip", showlegend=False))

    def rule(y: float, width: float, color: str) -> None:
        fig.add_shape(type="line", x0=label_x - 1, x1=right_edge, y0=y, y1=y,
                      line=dict(color=color, width=width), layer="above")

    header_y = 0.60
    rule(header_y + 0.62, 1.4, AC["axis"])
    rule(header_y - 0.52, 1.1, AC["axis"])
    fig.add_annotation(
        x=label_x, y=header_y, text="<b>normalised units</b>", showarrow=False,
        xanchor="left", yanchor="middle",
        font=dict(family=FONT_UI, size=FS_TICK, color=AC["text_muted"]))
    for x, (_key, label, color, _prefix) in zip(col_x, METHODS):
        fig.add_annotation(
            x=x, y=header_y, text=f"<b>{label}</b>", showarrow=False,
            xanchor="right", yanchor="middle",
            font=dict(family=FONT_UI, size=FS_AXIS, color=color))

    for index, (metric, label, sense, better) in enumerate(rows):
        y = -index - 0.55
        fig.add_annotation(
            x=label_x, y=y, text=label, showarrow=False, yshift=6,
            xanchor="left", yanchor="middle",
            font=dict(family=FONT_UI, size=FS_AXIS, color=AC["text_primary"]))
        fig.add_annotation(
            x=label_x, y=y, text=sense, showarrow=False, yshift=-9,
            xanchor="left", yanchor="middle",
            font=dict(family=FONT_UI, size=FS_NOTE, color=AC["text_faint"]))

        present = {key: values_for(prefix, metric)
                   for key, _l, _c, prefix in METHODS}
        # From-scratch is forward transfer's own reference, so it scores 0 by
        # construction there and cannot compete for the best value.
        contenders = {k: float(np.mean(v)) for k, v in present.items() if v}
        best = ((max(contenders.values()) if better > 0
                 else min(contenders.values())) if contenders else None)

        ours_values = present["ours"]
        lent_sd = (float(np.std(ours_values, ddof=1))
                   if len(ours_values) > 1 else 0.0)

        for x, (key, _label, _color, _prefix) in zip(col_x, METHODS):
            values = present[key]
            if not values:
                fig.add_annotation(
                    x=x, y=y, text="0, by construction", showarrow=False,
                    xanchor="right", yanchor="middle",
                    font=dict(family=FONT_UI, size=FS_NOTE,
                              color=AC["text_faint"]))
                continue
            mean = float(np.mean(values))
            measured = len(values) > 1
            sd = float(np.std(values, ddof=1)) if measured else lent_sd
            body = f"{mean:+.2f}" if metric == "bwt" else f"{mean:.2f}"
            if best is not None and abs(mean - best) < 1e-9:
                body = f"<b>{body}</b>"
            if sd > 1e-9:
                dagger = "" if measured else "†"
                body += (f"<span style=\"font-size:{FS_NOTE}px;"
                         f"color:{AC['text_faint']}\"> ±{sd:.2f}{dagger}</span>")
            fig.add_annotation(
                x=x, y=y, text=body, showarrow=False,
                xanchor="right", yanchor="middle",
                font=dict(family=FONT_MONO, size=FS_VALUE,
                          color=AC["text_primary"]))

    bottom = -len(rows) - 0.1
    rule(bottom, 1.4, AC["axis"])
    fig.add_annotation(
        x=label_x - 1, y=bottom - 0.35,
        text=("Complete 50-task runs only; partial seeds are excluded rather "
              "than averaged in. <b>Bold</b> is the best value in each row.<br>"
              "Ours is the mean over 3 seeds and its ±1 s.d. is measured across "
              "them.<br>"
              "<b>† marks a placeholder:</b> the other three have one complete "
              "seed each,<br>"
              "so they carry ours' s.d. on that metric until their own land. A "
              "lent interval is not a measurement.<br>"
              "Forward transfer is measured against the from-scratch run, which "
              "is therefore 0 by construction."),
        showarrow=False, xanchor="left", yanchor="top", align="left",
        font=dict(family=FONT_UI, size=FS_NOTE, color=AC["text_muted"]))

    fig.update_xaxes(visible=False, range=[label_x - 2, right_edge + 1])
    fig.update_yaxes(visible=False, range=[bottom - 2.6, header_y + 1.2])
    fig.update_layout(title=None, showlegend=False, plot_bgcolor=AC["bg"],
                      margin=dict(l=20, r=20, t=14, b=12))
    export_pair(fig, "headline_metrics", W_FULL, 356)


# ── Export ──────────────────────────────────────────────────────────────────
def export_pair(fig: go.Figure, stem: str, width: int, height: int) -> None:
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    SVG_DIR.mkdir(parents=True, exist_ok=True)
    export_figure(fig, PNG_DIR / stem, width, height, label=stem)
    (PNG_DIR / f"{stem}.svg").replace(SVG_DIR / f"{stem}.svg")
    print(f"  png/{stem}.png  svg/{stem}.svg")


def main() -> int:
    install_template()
    print("reports/final/gridworld")
    figure_learned_vs_retained()
    figure_retention_curve()
    figure_raw_vs_normalised()
    figure_compute_cost()
    figure_headline_table()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
