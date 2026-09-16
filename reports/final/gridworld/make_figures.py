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

# The two shaded quadrants are mirror images, so one pair of bounds defines
# both: POOR is "barely off a random policy", STRONG is "most of the way to
# solved". Blue is learned < POOR and ended > STRONG; red swaps the roles.
POOR, STRONG = 0.25, 0.60

RESCUED_FILL = "#DCE7FB"   # learned poorly, ended strong
LOST_FILL = "#FBDCDC"      # learned strong, ended poorly

METHODS = [
    ("ours",     "Min-Max (ours)", AC["blue"],       "biggrid50_sh_ours_"),
    ("cka_rl",   "CKA-RL",         AC["amber"],      "biggrid50_cka_rl_"),
    ("cbp",      "CbpNet",         AC["violet"],     "biggrid50_cbp_"),
    ("crelu",    "CReLUs",         AC["teal"],       "biggrid50_crelu_"),
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


def active_methods() -> list[tuple[str, str, str, str]]:
    """The methods with at least one complete 50-task run.

    A method whose runs are still in flight is absent from every figure rather
    than drawn as an empty row, and rejoins on the next rebuild once a seed
    lands. Method count is therefore data, not a constant, which is why the
    layouts below are all derived from ``len(active_methods())``.
    """
    return [m for m in METHODS if complete_runs(m[3])]


def pending_methods() -> list[str]:
    """Methods declared but with no complete run yet, for the captions.

    A figure that silently omits a baseline reads as a choice. Naming what is
    still running makes it a status instead.
    """
    return [label for _key, label, _color, prefix in METHODS
            if not complete_runs(prefix)]


def pending_note() -> str:
    """" CbpNet and CReLUs are still running." — empty when nothing is."""
    names = pending_methods()
    if not names:
        return ""
    listed = (names[0] if len(names) == 1
              else " and ".join([", ".join(names[:-1]), names[-1]]))
    single = len(names) == 1
    # Always its own line. Appending it to whatever sentence ends the caption is
    # what pushed the last line off the right edge of the table.
    return (f"<br><b>{listed} {'is' if single else 'are'} still running</b> and "
            f"{'joins' if single else 'join'} these figures on the next rebuild.")


def seed_summary() -> str:
    """"Min-Max 3, CKA-RL 2, ..." — the complete-seed count per method.

    Captions state seed counts, and seed counts change every time a run lands.
    Deriving the sentence rather than typing it is what stops a figure from
    describing a state of the world that stopped being true two commits ago.
    """
    return ", ".join(f"{label.split(' (')[0]} {len(complete_runs(prefix))}"
                     for _key, label, _color, prefix in active_methods())


def add_footnote(fig: go.Figure, text: str, left_margin: int,
                 clear_axis_title: bool = True) -> int:
    """Place a footnote below the plot, measured in pixels rather than fractions.

    Anchoring at a paper fraction fails on short figures: -0.2 of a shallow plot
    area is only a few pixels, so the note lands on the x-axis title. Pixels are
    what actually need clearing, so pixels are what this uses.

    Returns the bottom margin the note needs. Captions here are assembled from
    the data and change length as runs land, and a hand-set margin that was
    right for six lines clips the seventh. Asking the note how much room it
    wants is what keeps that from happening again.
    """
    offset = 64 if clear_axis_title else 34
    fig.add_annotation(
        x=0, y=0, xref="paper", yref="paper",
        xshift=-left_margin + 4, yshift=-offset,
        text=text, showarrow=False, xanchor="left", yanchor="top", align="left",
        font=dict(family=FONT_UI, size=FS_NOTE, color=AC["text_muted"]))
    return offset + round(13.5 * (text.count("<br>") + 1)) + 14


# ── Figure 1: learned vs retained ───────────────────────────────────────────
def figure_learned_vs_retained() -> None:
    """Each task's score when learned against its score at the end.

    The separator. Average performance puts ours and CKA-RL within 0.09 of each
    other, which is easy to wave away; asking a yes/no question of every task
    splits them completely.

    The two mean lines cut the plane into quadrants. Each panel shades the one
    that characterises its method: ours the tasks it left below average and
    brought back above it, the baselines the tasks they learned above average
    and then lost.
    """
    methods = active_methods()
    n_rows = (len(methods) + 1) // 2

    # Panel height and the gap above each panel's two-line header are fixed in
    # pixels, and the figure grows to fit however many methods have landed.
    # Deriving the spacing *fraction* from those pixels is what keeps a 6-method
    # render from squeezing the headers into the panel above it.
    panel_h, gap_h, top_m, bottom_m = 200, 100, 112, 152
    plot_h = n_rows * panel_h + (n_rows - 1) * gap_h
    fig = make_subplots(rows=n_rows, cols=2, horizontal_spacing=0.13,
                        vertical_spacing=gap_h / plot_h)
    tally: dict[str, tuple[int, int, int]] = {}

    for index, (key, label, color, prefix) in enumerate(methods):
        row, col = index // 2 + 1, index % 2 + 1
        runs = complete_runs(prefix)
        pairs = [learned_and_final(r) for r in runs]
        learned = np.concatenate([p[0] for p in pairs])
        final = np.concatenate([p[1] for p in pairs])
        improved = float(np.mean(final > learned))
        rescue = key == "ours"
        tally[key] = (
            int(((learned < POOR) & (final > STRONG)).sum()),
            int(((learned > STRONG) & (final < POOR)).sum()),
            len(learned))

        # Only the two bounds of this panel's own quadrant are drawn; all four
        # would be clutter, and each panel is asking a single question.
        box = (dict(x0=-0.6, x1=POOR, y0=STRONG, y1=1.05) if rescue
               else dict(x0=STRONG, x1=1.05, y0=-0.6, y1=POOR))
        fig.add_shape(type="rect", layer="below",
                      fillcolor=RESCUED_FILL if rescue else LOST_FILL,
                      line=dict(width=0), row=row, col=col, **box)
        for value, horizontal in (((POOR, False), (STRONG, True)) if rescue
                                  else ((STRONG, False), (POOR, True))):
            fig.add_shape(
                type="line", layer="below",
                x0=-0.6 if horizontal else value, x1=1.05 if horizontal else value,
                y0=value if horizontal else -0.6, y1=value if horizontal else 1.05,
                line=dict(color=hex_to_rgba(AC["axis"], 0.40), width=1,
                          dash="dot"),
                row=row, col=col)
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

        inside = (((learned < POOR) & (final > STRONG)) if rescue
                  else ((learned > STRONG) & (final < POOR)))
        axis = f"{index + 1 if index else ''}"
        fig.add_annotation(
            x=0, y=1.19, xref=f"x{axis} domain", yref=f"y{axis} domain",
            text=f"<b>{label}</b>", showarrow=False,
            xanchor="left", yanchor="bottom",
            font=dict(family=FONT_UI, size=FS_PANEL, color=AC["text_primary"]))
        fig.add_annotation(
            x=0, y=1.035, xref=f"x{axis} domain", yref=f"y{axis} domain",
            text=(f"<b>{improved:.0%}</b> ended better &nbsp;·&nbsp; "
                  f"<b>{int(inside.sum())}</b> of {len(learned)} in the "
                  "shaded quadrant"),
            showarrow=False, xanchor="left", yanchor="bottom",
            font=dict(family=FONT_UI, size=FS_TICK, color=color))

    # Real legend entries rather than text inside the panels.
    for name, fill in ((f"learned poorly (&lt; {POOR:.2f}), ended strong "
                        f"(&gt; {STRONG:.2f})", RESCUED_FILL),
                       (f"learned strong (&gt; {STRONG:.2f}), ended poorly "
                        f"(&lt; {POOR:.2f})", LOST_FILL)):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", name=name,
            marker=dict(size=13, symbol="square", color=fill,
                        line=dict(color=hex_to_rgba(AC["axis"], 0.40), width=1)),
            showlegend=True, hoverinfo="skip"), row=1, col=1)

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
    for row in range(1, n_rows + 1):
        fig.update_yaxes(title=dict(text="score after all 50 tasks", **title),
                         row=row, col=1)
    # An odd method count leaves the last cell empty, so each column carries its
    # x-axis title on the lowest panel that column actually uses, and the unused
    # cell is blanked rather than left as an empty styled frame.
    for col in (1, 2):
        used = [i for i in range(len(methods)) if i % 2 + 1 == col]
        if not used:
            continue
        fig.update_xaxes(title=dict(text="score when the task was just learned",
                                    **title), row=used[-1] // 2 + 1, col=col)
    for index in range(len(methods), n_rows * 2):
        fig.update_xaxes(visible=False, row=index // 2 + 1, col=index % 2 + 1)
        fig.update_yaxes(visible=False, row=index // 2 + 1, col=index % 2 + 1)

    def phrase(which: int) -> str:
        parts = [f"{label.split(' (')[0]} "
                 f"{tally[key][which] or 'none'}"
                 + (f" of {tally[key][2]}" if tally[key][which] else "")
                 for key, label, _c, _p in methods]
        return "; ".join(parts)

    rescued_line, lost_line = phrase(0), phrase(1)
    bottom_m = add_footnote(fig, (
        "One point per task, in units of (score − random) / (1 − random), so 0 "
        "is a random policy and 1 a solved task; the diagonal is no change.<br>"
        "Each panel shades one quadrant and draws only its two bounds.<br>"
        "The quadrants are mirror images: 0.25 is barely off random, 0.60 is "
        "most of the way to solved.<br>"
        f"<b>Rescued: {rescued_line}.</b><br>"
        f"<b>Lost: {lost_line}.</b><br>"
        "Each complete seed contributes 50 points.<br>"
        f"Complete seeds: {seed_summary()}.{pending_note()}"), 78)
    fig.update_layout(
        title=None, plot_bgcolor=AC["bg"],
        margin=dict(l=78, r=26, t=top_m, b=bottom_m),
        legend=dict(orientation="h", x=0.0, xanchor="left",
                    y=1 + 68 / plot_h,   # a fixed 68 px above the plot area
                    yanchor="bottom", bgcolor="rgba(0,0,0,0)", borderwidth=0,
                    itemsizing="constant",
                    font=dict(family=FONT_UI, size=FS_TICK,
                              color=AC["text_primary"])),
        showlegend=True)
    export_pair(fig, "learned_vs_retained", W_FULL, plot_h + top_m + bottom_m)


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
    extent = [np.inf, -np.inf]   # lowest and highest point any band reaches

    for key, label, color, prefix in active_methods():
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
        extent = [min(extent[0], float(np.nanmin(per_seed))),
                  max(extent[1], float(np.nanmax(per_seed)))]

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
        # Range follows the bands rather than a fixed pair of numbers, so a
        # method whose seeds spread below zero is not silently clipped.
        range=[min(-0.02, extent[0] - 0.03), max(0.80, extent[1] + 0.03)],
        showgrid=True, gridcolor=AC["grid"], gridwidth=0.6,
        zeroline=True, zerolinecolor=AC["border"], zerolinewidth=1.0,
        showline=True, linecolor=AC["axis"], linewidth=1.2, ticklen=4,
        tickfont=dict(family=FONT_MONO, size=FS_TICK, color=AC["text_muted"]))

    bottom = add_footnote(fig, (
        "After finishing task k, the mean score over the k−1 tasks learned "
        "before it, in units of (score − random) / (1 − random).<br>"
        "The just-learned task is excluded, because including it lets a method "
        "that merely learns the newest task well post a flattering curve.<br>"
        "<b>Shaded bands are the min-max across each method's complete "
        "seeds.</b><br>"
        "A method with one seed carries no band: an interval over a single run "
        "is invented rather than measured.<br>"
        f"Complete seeds: {seed_summary()}.{pending_note()}"), 74)
    fig.update_layout(title=None, showlegend=False,
                      margin=dict(l=74, r=150, t=24, b=bottom))
    export_pair(fig, "retention_curve", W_FULL, 300 + 24 + bottom)


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

    methods = active_methods()
    for col, column in enumerate(["raw", "normalized"], start=1):
        for index, (key, label, color, prefix) in enumerate(methods):
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

    fig.update_yaxes(tickmode="array", tickvals=list(range(len(methods))),
                     ticktext=[m[1] for m in methods],
                     range=[len(methods) - 0.35, -0.90], showgrid=False,
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

    # The two ranges the caption quotes are properties of the runs, so they are
    # measured here rather than typed. Typed, they drift the moment a seed lands.
    all_raw = np.concatenate([learned_and_final(r, "raw")[1]
                              for _k, _l, _c, p in methods
                              for r in complete_runs(p)])
    ours_runs = complete_runs("biggrid50_sh_ours_")
    raw_one = learned_and_final(ours_runs[0], "raw")[1]
    norm_one = learned_and_final(ours_runs[0], "normalized")[1]
    floor = (raw_one - norm_one) / (1 - norm_one)   # invert the normaliser

    bottom = add_footnote(fig, (
        "One point per task after all 50 are learned; the heavy rule is the "
        "median. <b>Both panels are the same runs.</b><br>"
        "A random policy already collects most of the raw return here, so raw "
        f"scores crowd into [{all_raw.min():.2f}, {all_raw.max():.2f}] and the "
        "methods look closer than they are.<br>"
        f"The random floor also varies per task, from {floor.min():.2f} to "
        f"{floor.max():.2f}, so the same raw number is a different achievement "
        "on different tasks.<br>"
        "Normalising against each task's own floor is what makes them "
        "comparable.<br>"
        f"Complete seeds: {seed_summary()}.{pending_note()}"), 118)
    fig.update_layout(title=None, showlegend=False, plot_bgcolor=AC["bg"],
                      margin=dict(l=118, r=26, t=56, b=bottom))
    # 62 px per method row keeps the point clouds from overlapping as rows are
    # added, so the figure grows rather than compressing.
    export_pair(fig, "raw_vs_normalised", W_FULL,
                62 * len(methods) + 56 + bottom)


# ── Figure 4: compute cost ──────────────────────────────────────────────────
def figure_compute_cost() -> None:
    """Wall-clock for one complete 50-task run."""
    fig = go.Figure()
    methods = active_methods()
    totals = {key: np.array([
        sum(float(r["wall_s_phase"]) for r in load_phases(run)) / 60.0
        for run in complete_runs(prefix)])
        for key, _label, _color, prefix in methods}

    reference = float(np.mean(totals["ours"]))
    top = max(float(v.max()) for v in totals.values())

    for index, (key, label, color, _prefix) in enumerate(methods):
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
    fig.update_yaxes(tickmode="array", tickvals=list(range(len(methods))),
                     ticktext=[m[1] for m in methods],
                     range=[len(methods) - 0.5, -0.5], showgrid=False,
                     zeroline=False, showline=False, ticklen=0,
                     tickfont=dict(family=FONT_UI, size=FS_AXIS,
                                   color=AC["text_primary"]))

    bottom = add_footnote(fig, (
        "× is relative to ours. Complete runs only; a capped segment is that "
        "method's min-max across its complete seeds.<br>"
        f"Complete seeds: {seed_summary()}.<br>"
        "<b>Ours is the most expensive because consolidation re-simulates past "
        "environments</b>, spending time on tasks it has already learned.<br>"
        "Measured on a shared, contended cluster, so read it as "
        f"indicative.{pending_note()}"), 122)
    fig.update_layout(title=None, showlegend=False, plot_bgcolor=AC["bg"],
                      margin=dict(l=122, r=126, t=20, b=bottom))
    export_pair(fig, "compute_cost", W_FULL, 41 * len(methods) + 20 + bottom)


# ── Figure 5: headline metrics, as a table ──────────────────────────────────
def figure_headline_table() -> None:
    """The four continual-learning metrics as a booktabs table.

    A table rather than lollipop panels: the reader wants to compare the numbers
    exactly, which is what a table is for.

    Methods run down the rows and metrics across the columns, because the method
    count grows as baselines land while the metric count does not. Putting the
    growing axis vertical is what lets a sixth method join without squeezing
    every column past the width of its own header.
    """
    data = metrics()
    methods = active_methods()
    # (metric, header line 1, header line 2, which way is better)
    cols = [("perf", "Average", "performance", "higher is better", +1),
            ("forgetting", "Forgetting", "", "lower is better", -1),
            ("bwt", "Backward", "transfer", "higher is better", +1),
            ("fwt", "Forward", "transfer", "higher is better", +1)]

    def values_for(prefix: str, metric: str) -> list[float]:
        return [data[r][metric] for r in complete_runs(prefix)
                if data[r].get(metric) is not None]

    label_x = 0.0
    col_x = [46.0, 62.0, 78.0, 94.0]
    right_edge = col_x[-1] + 2

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                             hoverinfo="skip", showlegend=False))

    def rule(y: float, width: float, color: str) -> None:
        fig.add_shape(type="line", x0=label_x - 1, x1=right_edge, y0=y, y1=y,
                      line=dict(color=color, width=width), layer="above")

    # Column statistics first: the best value in a column is only knowable once
    # every method in it has been read.
    present = {metric: {key: values_for(prefix, metric)
                        for key, _l, _c, prefix in methods}
               for metric, _h1, _h2, _sense, _better in cols}
    ours_sd = {metric: (float(np.std(v["ours"], ddof=1))
                        if len(v["ours"]) > 1 else 0.0)
               for metric, v in present.items()}
    best = {}
    for metric, _h1, _h2, _sense, better in cols:
        means = [float(np.mean(v)) for v in present[metric].values() if v]
        best[metric] = (max(means) if better > 0 else min(means)) if means else None

    header_y = 0.0
    rule(header_y + 1.05, 1.4, AC["axis"])
    rule(header_y - 0.62, 1.1, AC["axis"])
    fig.add_annotation(
        x=label_x, y=header_y + 0.12, text="<b>normalised units</b>",
        showarrow=False, xanchor="left", yanchor="middle",
        font=dict(family=FONT_UI, size=FS_TICK, color=AC["text_muted"]))
    for x, (_metric, head1, head2, sense, _better) in zip(col_x, cols):
        fig.add_annotation(
            x=x, y=header_y + (0.46 if head2 else 0.28), text=f"<b>{head1}</b>",
            showarrow=False, xanchor="right", yanchor="middle",
            font=dict(family=FONT_UI, size=FS_AXIS, color=AC["text_primary"]))
        if head2:
            fig.add_annotation(
                x=x, y=header_y + 0.12, text=f"<b>{head2}</b>", showarrow=False,
                xanchor="right", yanchor="middle",
                font=dict(family=FONT_UI, size=FS_AXIS,
                          color=AC["text_primary"]))
        fig.add_annotation(
            x=x, y=header_y - 0.28, text=sense, showarrow=False,
            xanchor="right", yanchor="middle",
            font=dict(family=FONT_UI, size=FS_NOTE, color=AC["text_faint"]))

    lent_used = False
    for index, (key, label, color, _prefix) in enumerate(methods):
        y = -index - 0.95
        if key == "ours":
            fig.add_shape(type="rect", layer="below",
                          x0=label_x - 1, x1=right_edge,
                          y0=y - 0.45, y1=y + 0.45,
                          fillcolor=hex_to_rgba(AC["blue"], 0.07),
                          line=dict(width=0))
        fig.add_annotation(
            x=label_x, y=y, text=f"<b>{label}</b>", showarrow=False,
            xanchor="left", yanchor="middle",
            font=dict(family=FONT_UI, size=FS_AXIS, color=color))

        for x, (metric, _h1, _h2, _sense, _better) in zip(col_x, cols):
            values = present[metric][key]
            if not values:
                fig.add_annotation(
                    x=x, y=y, text="0, by construction", showarrow=False,
                    xanchor="right", yanchor="middle",
                    font=dict(family=FONT_UI, size=FS_NOTE,
                              color=AC["text_faint"]))
                continue
            mean = float(np.mean(values))
            measured = len(values) > 1
            sd = float(np.std(values, ddof=1)) if measured else ours_sd[metric]
            lent_used = lent_used or (not measured and sd > 1e-9)
            body = f"{mean:+.2f}" if metric == "bwt" else f"{mean:.2f}"
            if best[metric] is not None and abs(mean - best[metric]) < 1e-9:
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

    bottom = -len(methods) - 0.5
    rule(bottom, 1.4, AC["axis"])
    note_lines = [
        "Complete 50-task runs only; partial seeds are excluded rather than "
        "averaged in. <b>Bold</b> is the best value in each column.",
        "Values are the mean over each method's complete seeds with ±1 s.d. "
        "across them.",
        f"Complete seeds: {seed_summary()}.",
        *(["<b>† marks a placeholder:</b> that method has a single complete "
           "seed, so it carries ours' s.d. until its own land. A lent interval "
           "is not a measurement."] if lent_used else []),
        "Forward transfer is measured against the from-scratch run, which is "
        "therefore 0 by construction.",
        # removeprefix, not lstrip: lstrip("<br>") strips those four characters
        # in any order and would eat the opening <b> of the sentence too.
        *([pending_note().removeprefix("<br>")] if pending_note() else []),
    ]
    fig.add_annotation(
        x=label_x - 1, y=bottom - 0.30, text="<br>".join(note_lines),
        showarrow=False, xanchor="left", yanchor="top", align="left",
        font=dict(family=FONT_UI, size=FS_NOTE, color=AC["text_muted"]))

    top_y = header_y + 1.5
    # The caption block is sized by its own line count, so the table neither
    # clips it nor leaves a band of empty paper under a short one.
    bottom_y = bottom - 0.55 - 0.40 * len(note_lines)
    fig.update_xaxes(visible=False, range=[label_x - 2, right_edge + 1])
    fig.update_yaxes(visible=False, range=[bottom_y, top_y])
    fig.update_layout(title=None, showlegend=False, plot_bgcolor=AC["bg"],
                      margin=dict(l=20, r=20, t=14, b=12))
    # 39 px per unit row keeps the type scale identical however many rows there
    # are, instead of stretching the same table to a fixed height.
    export_pair(fig, "headline_metrics", W_FULL,
                round(39 * (top_y - bottom_y)) + 26)


# ── Export ──────────────────────────────────────────────────────────────────
def export_pair(fig: go.Figure, stem: str, width: int, height: int) -> None:
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    SVG_DIR.mkdir(parents=True, exist_ok=True)
    export_figure(fig, PNG_DIR / stem, width, height, label=stem)
    (PNG_DIR / f"{stem}.svg").replace(SVG_DIR / f"{stem}.svg")
    print(f"  png/{stem}.png  svg/{stem}.svg")


def verify() -> list[str]:
    """Check every export for the two faults this figure set keeps hitting.

    Clipped captions and rasterised SVGs both survive a glance at the PNG, and
    both have shipped here before. Captions are assembled from the data and grow
    as runs land, so the check belongs in the run rather than in someone's
    memory.
    """
    from PIL import Image

    faults = []
    for png in sorted(PNG_DIR.glob("*.png")):
        pixels = np.asarray(Image.open(png).convert("L"))
        ink = pixels < 245
        rows, cols = np.where(ink.any(axis=1))[0], np.where(ink.any(axis=0))[0]
        if not rows.size:
            faults.append(f"{png.name}: blank")
            continue
        edges = (rows[0], pixels.shape[0] - 1 - rows[-1],
                 cols[0], pixels.shape[1] - 1 - cols[-1])
        if min(edges) <= 8:
            faults.append(f"{png.name}: content within {min(edges)}px of an edge "
                          "(top, bottom, left, right = "
                          f"{', '.join(str(e) for e in edges)})")
    for svg in sorted(SVG_DIR.glob("*.svg")):
        embedded = svg.read_text(encoding="utf-8").count("<image")
        if embedded:
            faults.append(f"{svg.name}: {embedded} embedded raster image(s), "
                          "so the figure is not fully vector")
    return faults


def main() -> int:
    install_template()
    print("reports/final/gridworld")
    figure_learned_vs_retained()
    figure_retention_curve()
    figure_raw_vs_normalised()
    figure_compute_cost()
    figure_headline_table()

    faults = verify()
    for fault in faults:
        print(f"  FAULT  {fault}")
    if not faults:
        pending = pending_methods()
        print(f"  checked: {seed_summary()}"
              + (f"; pending {', '.join(pending)}" if pending else ""))
    return 1 if faults else 0


if __name__ == "__main__":
    raise SystemExit(main())
