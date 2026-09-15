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
from collections import defaultdict
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
sys.path.insert(0, str(REPO / "report"))

from acviz import (  # noqa: E402
    AC, FONT_MONO, FONT_UI, W_FULL, export_figure, height_for, hex_to_rgba,
    install_template,
)

RUNS = REPO / "reports" / "gridworld_sharedhead"
PNG_DIR, SVG_DIR = HERE / "png", HERE / "svg"
N_TASKS = 50

# Ours is primary; CKA-RL is the method being compared against; fine-tuning is
# the catastrophic-forgetting case; from-scratch is a reference, so it stays
# neutral rather than competing for attention.
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
        xshift=-left_margin + 4, yshift=-(58 if clear_axis_title else 30),
        text=text, showarrow=False, xanchor="left", yanchor="top", align="left",
        font=dict(family=FONT_UI, size=8.5, color=AC["text_muted"]))


# ── Figure 1: learned vs retained ───────────────────────────────────────────
def figure_learned_vs_retained() -> None:
    """Each task's score when learned against its score at the end.

    The separator. Average performance puts ours and CKA-RL within 0.09 of each
    other, which is easy to wave away; this asks a yes/no question of every task
    — did it end better or worse than when it was learned — and the answer
    splits the methods completely. The diagonal is "no change": above it a task
    improved after the model moved on, below it the task was forgotten.
    """
    fig = make_subplots(rows=1, cols=4, horizontal_spacing=0.030,
                        shared_yaxes=True,
                        subplot_titles=[label for _k, label, _c, _p in METHODS])

    for col, (key, label, color, prefix) in enumerate(METHODS, start=1):
        runs = complete_runs(prefix)
        pairs = [learned_and_final(r) for r in runs]
        learned = np.concatenate([p[0] for p in pairs])
        final = np.concatenate([p[1] for p in pairs])
        improved = float(np.mean(final > learned))
        suffix = f"{col if col > 1 else ''}"

        fig.add_trace(go.Scatter(
            x=[-0.6, 1.05], y=[-0.6, 1.05], mode="lines",
            line=dict(color=AC["border"], width=1.2),
            hoverinfo="skip", showlegend=False), row=1, col=col)
        fig.add_trace(go.Scatter(
            x=learned, y=final, mode="markers",
            marker=dict(color=hex_to_rgba(color, 0.55), size=5,
                        line=dict(color=color, width=0.6)),
            hovertemplate="learned %{x:.2f} → final %{y:.2f}<extra></extra>",
            showlegend=False), row=1, col=col)

        fig.add_annotation(
            x=0.04, y=0.97, xref=f"x{suffix} domain", yref=f"y{suffix} domain",
            text=(f"<b>{improved:.0%}</b> of tasks<br>"
                  "<span style=\"font-size:9px\">ended better</span>"),
            showarrow=False, xanchor="left", yanchor="top", align="left",
            font=dict(family=FONT_UI, size=13, color=color),
            # Never text straight over data: the pad is the house rule.
            bgcolor="rgba(255,255,255,0.84)", borderpad=3)
        fig.add_annotation(
            x=0.97, y=0.03, xref=f"x{suffix} domain", yref=f"y{suffix} domain",
            text=f"n = {len(learned)}", showarrow=False,
            xanchor="right", yanchor="bottom",
            font=dict(family=FONT_MONO, size=8.5, color=AC["text_faint"]))

    fig.update_xaxes(range=[-0.6, 1.05], dtick=0.5, showgrid=True,
                     gridcolor=AC["grid"], gridwidth=0.6, zeroline=False,
                     showline=True, linecolor=AC["axis"], linewidth=1.2,
                     ticklen=4, tickfont=dict(family=FONT_MONO, size=9,
                                              color=AC["text_muted"]))
    fig.update_yaxes(range=[-0.6, 1.05], dtick=0.5, showgrid=True,
                     gridcolor=AC["grid"], gridwidth=0.6, zeroline=False,
                     showline=True, linecolor=AC["axis"], linewidth=1.2,
                     ticklen=4, tickfont=dict(family=FONT_MONO, size=9,
                                              color=AC["text_muted"]))
    fig.update_yaxes(title=dict(text="score after all 50 tasks",
                                font=dict(family=FONT_UI, size=11,
                                          color=AC["text_primary"])),
                     row=1, col=1)
    for annotation in fig.layout.annotations[:4]:
        annotation.font = dict(family=FONT_UI, size=11.5, color=AC["text_primary"])
        annotation.y = 1.05

    fig.add_annotation(
        x=0.5, y=-0.14, xref="paper", yref="paper",
        text="score when the task was just learned", showarrow=False,
        xanchor="center", yanchor="top",
        font=dict(family=FONT_UI, size=11, color=AC["text_primary"]))
    add_footnote(fig, (
"One point per task, in units of (score − random) / (1 − random). "
              "The grey line is no change.<br>"
              "<b>Above it, a task improved after the model moved on; below it, "
              "the task was forgotten.</b> Ours pools its 3 complete seeds, the "
              "others their 1 complete seed each.<br>"
              "This is the backward-transfer column read one task at a time, "
              "and it is where ours and CKA-RL stop being close."
    ), 52)
    fig.update_layout(title=None, showlegend=False, plot_bgcolor=AC["bg"],
                      margin=dict(l=60, r=18, t=46, b=116))
    export_pair(fig, "learned_vs_retained", W_FULL, height_for(W_FULL, 2.45))


# ── Figure 2: two views of the same sequence ────────────────────────────────
def figure_task_life() -> None:
    """Two questions that both reduce to "how much survives", side by side.

    They are easy to confuse, so they share a figure with the difference stated
    on each panel. **Left** indexes by position in the sequence: after finishing
    task k, how is the model doing on the k-1 tasks behind it. **Right** indexes
    by time since a task was learned, pooling every task that is n phases old
    whenever it happened. A method can be flat on the left and still decaying on
    the right, if the later tasks happened to be easier.
    """
    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.13, shared_yaxes=True,
        subplot_titles=["Where the model is in the sequence",
                        "What happens to a task after it is learned"])

    ends: list[list] = [[], []]
    for key, label, color, prefix in METHODS:
        runs = complete_runs(prefix)

        per_phase = np.full((len(runs), N_TASKS), np.nan)
        for index, run in enumerate(runs):
            grid = load_matrix(run)
            for after in range(1, N_TASKS):
                prior = grid[after, :after]
                if np.isfinite(prior).any():
                    per_phase[index, after] = np.nanmean(prior)
        with np.errstate(invalid="ignore"):
            mean_phase = np.nanmean(per_phase, axis=0)
        valid = np.where(np.isfinite(mean_phase))[0]
        fig.add_trace(go.Scatter(
            x=valid + 1, y=mean_phase[valid], mode="lines",
            line=dict(color=color, width=2.2, shape="spline", smoothing=0.5),
            hoverinfo="skip", showlegend=False), row=1, col=1)
        ends[0].append((label, float(valid[-1] + 1),
                        float(mean_phase[valid[-1]]), color))

        by_age: dict[int, list[float]] = defaultdict(list)
        for run in runs:
            grid = load_matrix(run)
            end = last_phase(grid)
            for after in range(end + 1):
                for task in range(after + 1):
                    if np.isfinite(grid[after, task]):
                        by_age[after - task].append(grid[after, task])
        ages = sorted(a for a in by_age if len(by_age[a]) >= 5)
        means = [float(np.mean(by_age[a])) for a in ages]
        fig.add_trace(go.Scatter(
            x=ages, y=means, mode="lines",
            line=dict(color=color, width=2.2, shape="spline", smoothing=0.5),
            hoverinfo="skip", showlegend=False), row=1, col=2)
        ends[1].append((label, ages[-1], means[-1], color))

    for panel, items in enumerate(ends):
        positions = {label: y for label, _x, y, _c in items}
        for lower, upper in zip(sorted(positions, key=positions.get),
                                sorted(positions, key=positions.get)[1:]):
            if positions[upper] - positions[lower] < 0.055:
                positions[upper] = positions[lower] + 0.055
        for label, x_end, _y, color in items:
            fig.add_annotation(
                x=x_end, y=positions[label], text=f"<b>{label}</b>",
                showarrow=False, xanchor="left", xshift=8, yanchor="middle",
                font=dict(family=FONT_UI, size=10.5, color=color),
                row=1, col=panel + 1)

    fig.update_xaxes(title=dict(text="tasks learned so far",
                                font=dict(family=FONT_UI, size=10.5,
                                          color=AC["text_primary"])),
                     range=[0, N_TASKS + 16], tickmode="array",
                     tickvals=[0, 10, 20, 30, 40, 50], row=1, col=1)
    fig.update_xaxes(title=dict(text="phases since the task was learned",
                                font=dict(family=FONT_UI, size=10.5,
                                          color=AC["text_primary"])),
                     range=[0, N_TASKS + 16], tickmode="array",
                     tickvals=[0, 10, 20, 30, 40, 50], row=1, col=2)
    fig.update_xaxes(showgrid=False, zeroline=False, showline=True,
                     linecolor=AC["axis"], linewidth=1.2, ticklen=4,
                     tickfont=dict(family=FONT_MONO, size=9.5,
                                   color=AC["text_muted"]))
    fig.update_yaxes(range=[-0.02, 0.80], showgrid=True, gridcolor=AC["grid"],
                     gridwidth=0.6, zeroline=True, zerolinecolor=AC["border"],
                     zerolinewidth=1.0, showline=True, linecolor=AC["axis"],
                     linewidth=1.2, ticklen=4,
                     tickfont=dict(family=FONT_MONO, size=9.5,
                                   color=AC["text_muted"]))
    fig.update_yaxes(title=dict(text="mean score of past tasks",
                                font=dict(family=FONT_UI, size=11,
                                          color=AC["text_primary"])),
                     row=1, col=1)
    for annotation in fig.layout.annotations[:2]:
        annotation.font = dict(family=FONT_UI, size=11.5, color=AC["text_primary"])
        annotation.y = 1.07

    add_footnote(fig, (
"Same runs, two indexes, two different questions.<br>"
              "<b>Left:</b> after finishing task k, how the model does on the "
              "k−1 tasks behind it — the state of the whole system as the "
              "sequence grows.<br>"
              "<b>Right:</b> every task that is n phases old, pooled whenever it "
              "occurred — the life of a single task.<br>"
              "A method can be flat on the left and still decaying on the "
              "right, if the later tasks were easier.<br>"
              "The right panel is where ours differs in kind: it starts "
              "<i>lowest</i> at age 0 and is the only curve that rises.<br>"
              "Complete runs only — ours 3 seeds, the others 1 each."
    ), 58)
    fig.update_layout(title=None, showlegend=False, plot_bgcolor=AC["bg"],
                      margin=dict(l=62, r=20, t=52, b=148))
    export_pair(fig, "task_life", W_FULL, height_for(W_FULL, 1.80))


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
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.10,
                        subplot_titles=["Raw discounted return",
                                        "Normalised against each task's random floor"])

    for col, column in enumerate(["raw", "normalized"], start=1):
        for index, (key, label, color, prefix) in enumerate(METHODS):
            values = np.concatenate([learned_and_final(r, column)[1]
                                     for r in complete_runs(prefix)])
            jitter = (np.random.default_rng(index).random(len(values)) - 0.5) * 0.34
            fig.add_trace(go.Scatter(
                x=values, y=index + jitter, mode="markers",
                marker=dict(color=hex_to_rgba(color, 0.38), size=4.5,
                            line=dict(width=0)),
                hovertemplate="%{x:.3f}<extra></extra>", showlegend=False,
            ), row=1, col=col)
            median = float(np.median(values))
            fig.add_trace(go.Scatter(
                x=[median, median], y=[index - 0.26, index + 0.26], mode="lines",
                line=dict(color=color, width=2.8),
                hoverinfo="skip", showlegend=False), row=1, col=col)
            fig.add_annotation(
                x=median, y=index - 0.40, text=f"<b>{median:.2f}</b>",
                showarrow=False, xanchor="center", yanchor="bottom",
                font=dict(family=FONT_MONO, size=9, color=color),
                # Sits in the gutter between rows, so it needs the pad to stay
                # legible over the neighbouring row's points.
                bgcolor="rgba(255,255,255,0.88)", borderpad=2,
                row=1, col=col)

    fig.update_yaxes(tickmode="array", tickvals=list(range(len(METHODS))),
                     ticktext=[m[1] for m in METHODS],
                     range=[len(METHODS) - 0.4, -0.7], showgrid=False,
                     zeroline=False, showline=False, ticklen=0,
                     tickfont=dict(family=FONT_UI, size=10,
                                   color=AC["text_primary"]))
    fig.update_yaxes(showticklabels=False, row=1, col=2)
    fig.update_xaxes(showgrid=True, gridcolor=AC["grid"], gridwidth=0.6,
                     zeroline=False, showline=True, linecolor=AC["axis"],
                     linewidth=1.2, ticklen=4,
                     tickfont=dict(family=FONT_MONO, size=9.5,
                                   color=AC["text_muted"]))
    fig.update_xaxes(range=[0.55, 1.02], dtick=0.1, row=1, col=1,
                     title=dict(text="discounted return",
                                font=dict(family=FONT_UI, size=10.5,
                                          color=AC["text_muted"])))
    fig.update_xaxes(range=[-0.55, 1.05], dtick=0.5, zeroline=True,
                     zerolinecolor=AC["border"], zerolinewidth=1.0, row=1, col=2,
                     title=dict(text="0 = random policy, 1 = solved",
                                font=dict(family=FONT_UI, size=10.5,
                                          color=AC["text_muted"])))
    for annotation in fig.layout.annotations[:2]:
        annotation.font = dict(family=FONT_UI, size=11.5, color=AC["text_primary"])
        annotation.y = 1.08

    add_footnote(fig, (
"One point per task after all 50 are learned; the heavy tick is "
              "the median. <b>Both panels are the same runs.</b><br>"
              "A random policy already collects most of the raw return here, so "
              "raw scores crowd into [0.59, 0.99] and the methods look "
              "closer than they are.<br>"
              "The random floor also varies per task, from 0.55 to 0.87, so "
              "the same raw number is a different achievement on different "
              "tasks.<br>"
              "Normalising against each task's own floor is what makes them "
              "comparable."
    ), 96)
    fig.update_layout(title=None, showlegend=False, plot_bgcolor=AC["bg"],
                      margin=dict(l=100, r=20, t=52, b=124))
    export_pair(fig, "raw_vs_normalised", W_FULL, height_for(W_FULL, 2.55))


# ── Figure 4: compute cost ──────────────────────────────────────────────────
def figure_compute_cost() -> None:
    """What the retention costs, in wall-clock and in environment frames."""
    panels = [("wall", "Wall-clock", 1 / 60.0, "minutes, one 50-task run"),
              ("frames", "Environment frames", 1 / 1e6,
               "millions of environment frames")]
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.13,
                        shared_yaxes=True,
                        subplot_titles=[t for _k, t, _s, _u in panels])

    totals: dict[str, dict[str, np.ndarray]] = {}
    for key, _label, _color, prefix in METHODS:
        wall, frames = [], []
        for run in complete_runs(prefix):
            rows = load_phases(run)
            wall.append(sum(float(r["wall_s_phase"]) for r in rows))
            frames.append(sum(float(r["frames_phase"]) for r in rows))
        totals[key] = {"wall": np.array(wall), "frames": np.array(frames)}

    for col, (metric, _title, scale, unit) in enumerate(panels, start=1):
        ref = float(np.mean(totals["ours"][metric])) * scale
        top = 0.0
        for index, (key, label, color, _prefix) in enumerate(METHODS):
            values = totals[key][metric] * scale
            mean = float(np.mean(values))
            top = max(top, float(values.max()))
            fig.add_trace(go.Scatter(
                x=[0, mean], y=[index, index], mode="lines",
                line=dict(color=hex_to_rgba(color, 0.40), width=3),
                hoverinfo="skip", showlegend=False), row=1, col=col)
            if len(values) > 1:
                fig.add_trace(go.Scatter(
                    x=[values.min(), values.max()], y=[index, index],
                    mode="lines", line=dict(color=color, width=2),
                    hoverinfo="skip", showlegend=False), row=1, col=col)
            fig.add_trace(go.Scatter(
                x=[mean], y=[index], mode="markers",
                marker=dict(color=color, size=11,
                            line=dict(color=AC["bg"], width=1.8)),
                hovertemplate="%{x:.1f}<extra></extra>", showlegend=False,
            ), row=1, col=col)
            tail = "" if key == "ours" else (
                f"<span style=\"font-size:8.5px;color:{AC['text_muted']}\">"
                f"  {mean / ref:.2f}×</span>")
            body = f"{mean:,.0f}" if metric == "wall" else f"{mean:.1f}"
            fig.add_annotation(
                x=float(values.max()), y=index, text=f"<b>{body}</b>{tail}",
                showarrow=False, xanchor="left", yanchor="middle", xshift=9,
                font=dict(family=FONT_MONO, size=10, color=AC["text_primary"]),
                row=1, col=col)
        fig.update_xaxes(range=[0, top * 1.40], row=1, col=col,
                         title=dict(text=unit,
                                    font=dict(family=FONT_UI, size=10.5,
                                              color=AC["text_muted"])))

    fig.update_xaxes(showgrid=True, gridcolor=AC["grid"], gridwidth=0.6,
                     zeroline=False, showline=True, linecolor=AC["border"],
                     ticklen=0, tickfont=dict(family=FONT_MONO, size=9.5,
                                              color=AC["text_muted"]))
    fig.update_yaxes(tickmode="array", tickvals=list(range(len(METHODS))),
                     ticktext=[m[1] for m in METHODS],
                     range=[len(METHODS) - 0.5, -0.5], showgrid=False,
                     zeroline=False, showline=False, ticklen=0,
                     tickfont=dict(family=FONT_UI, size=10.5,
                                   color=AC["text_primary"]))
    for annotation in fig.layout.annotations[:2]:
        annotation.font = dict(family=FONT_UI, size=11.5, color=AC["text_primary"])
        annotation.y = 1.12

    add_footnote(fig, (
"× is relative to ours. Complete runs only; the segment on ours is "
              "its min-max over 3 seeds, the others are single runs.<br>"
              "<b>Ours is the most expensive, and the frames panel says why:</b><br>"
              "consolidation re-simulates past environments, so it spends frames "
              "on tasks it has already learned.<br>"
              "Wall-clock came off a shared, contended cluster and is the softer "
              "of the two; frames is the number to quote."
    ), 100)
    fig.update_layout(title=None, showlegend=False, plot_bgcolor=AC["bg"],
                      margin=dict(l=104, r=24, t=52, b=120))
    export_pair(fig, "compute_cost", W_FULL, height_for(W_FULL, 2.95))


# ── Figure 5: headline metrics ──────────────────────────────────────────────
def figure_headline_metrics() -> None:
    """The four continual-learning metrics, one panel each."""
    data = metrics()
    panels = [("perf", "Average performance", "higher is better"),
              ("forgetting", "Forgetting", "lower is better"),
              ("bwt", "Backward transfer", "higher is better"),
              ("fwt", "Forward transfer", "higher is better")]

    fig = make_subplots(rows=1, cols=4, horizontal_spacing=0.055,
                        subplot_titles=[t for _m, t, _s in panels])

    def values_for(prefix: str, metric: str) -> list[float]:
        return [data[r][metric] for r in complete_runs(prefix)
                if data[r].get(metric) is not None]

    order = sorted(METHODS, key=lambda m: -float(np.mean(values_for(m[3], "perf"))))
    row_of = {m[0]: i for i, m in enumerate(order)}

    for col, (metric, _title, _sense) in enumerate(panels, start=1):
        ours_values = values_for("biggrid50_sh_ours_", metric)
        lent_sd = float(np.std(ours_values, ddof=1)) if len(ours_values) > 1 else 0.0

        rows = []
        for key, label, color, prefix in order:
            values = values_for(prefix, metric)
            if not values:
                fig.add_annotation(
                    x=0, y=row_of[key], text="0, by construction", showarrow=False,
                    xanchor="left", yanchor="middle", xshift=6,
                    font=dict(family=FONT_UI, size=8.5, color=AC["text_faint"]),
                    row=1, col=col)
                continue
            mean = float(np.mean(values))
            if len(values) > 1:
                sd, measured = float(np.std(values, ddof=1)), True
            else:
                sd, measured = lent_sd, False
            rows.append((row_of[key], color, mean, sd, measured))

        for index, color, mean, sd, measured in rows:
            fig.add_trace(go.Scatter(
                x=[0, mean], y=[index, index], mode="lines",
                line=dict(color=hex_to_rgba(color, 0.40), width=3),
                hoverinfo="skip", showlegend=False), row=1, col=col)
            if sd > 1e-9:
                fig.add_trace(go.Scatter(
                    x=[mean - sd, mean + sd], y=[index, index], mode="lines",
                    line=dict(color=color, width=2 if measured else 1.4,
                              dash=None if measured else "dot"),
                    hoverinfo="skip", showlegend=False), row=1, col=col)
                if measured:
                    for edge in (mean - sd, mean + sd):
                        fig.add_trace(go.Scatter(
                            x=[edge, edge], y=[index - 0.13, index + 0.13],
                            mode="lines", line=dict(color=color, width=2),
                            hoverinfo="skip", showlegend=False), row=1, col=col)
            fig.add_trace(go.Scatter(
                x=[mean], y=[index], mode="markers",
                marker=dict(color=color, size=10,
                            line=dict(color=AC["bg"], width=1.6)),
                hovertemplate="%{x:.3f}<extra></extra>", showlegend=False,
            ), row=1, col=col)
            text = f"{mean:+.2f}" if metric == "bwt" else f"{mean:.2f}"
            dagger = "" if measured else (
                f"<span style=\"font-size:8px;color:{AC['text_muted']}\">†</span>")
            fig.add_annotation(
                x=mean + sd, y=index, text=f"<b>{text}</b>{dagger}",
                showarrow=False, xanchor="left", yanchor="middle", xshift=8,
                font=dict(family=FONT_MONO, size=9.5, color=AC["text_primary"]),
                row=1, col=col)

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
            row=1, col=col)

    for annotation, (_m, _t, _s) in zip(fig.layout.annotations[:4], panels):
        annotation.font = dict(family=FONT_UI, size=11, color=AC["text_primary"])
        annotation.y = 1.10
    for col, (_m, _t, sense) in enumerate(panels, start=1):
        fig.add_annotation(
            x=0.5, y=1.015, xref=f"x{col if col > 1 else ''} domain",
            yref="paper", text=sense, showarrow=False, xanchor="center",
            yanchor="bottom",
            font=dict(family=FONT_UI, size=8.5, color=AC["text_muted"]))

    add_footnote(fig, (
"Complete 50-task runs only; partial seeds are excluded rather "
              "than averaged in. Rows keep one order in every panel, best "
              "average performance first.<br>"
              "Ours is the mean over 3 seeds, its bar ±1 s.d. across them — "
              "measured, solid, capped.<br>"
              "<b>† marks a placeholder:</b> the other three have one complete "
              "seed each, so they carry ours' s.d. until their own land. "
              "A lent interval is not a measurement, which is why it is dotted "
              "and uncapped."
    ), 100)
    fig.update_layout(title=None, showlegend=False, plot_bgcolor=AC["bg"],
                      margin=dict(l=104, r=20, t=54, b=104))
    export_pair(fig, "headline_metrics", W_FULL, height_for(W_FULL, 2.35))


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
    figure_task_life()
    figure_raw_vs_normalised()
    figure_compute_cost()
    figure_headline_metrics()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
