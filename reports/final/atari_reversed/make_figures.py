#!/usr/bin/env python3
"""Finalised paper figures, line 1: the reversed five-game Atari sequence.

Two figures only, both on the reversed order SpaceInvaders -> Boxing -> Breakout
-> Pong -> Q*bert. Order sensitivity is acknowledged in the paper as an observed
phenomenon and its analysis is out of scope, so the canonical-order variants are
not rebuilt here.

Differences from ``reports/order_sensitivity`` (the exploratory version):

* The method is named **Min-Max** everywhere. The internal "V5" label is gone.
* The retention colour scale is **clamped at the ceiling**. Anything at or above
  100% of the Joint ceiling renders as the same blue, so a well-retained 82% is
  no longer dragged toward red by a 360% outlier at the top of the old scale.
* **No in-figure titles.** Papers carry the caption in LaTeX; a title baked into
  the artwork duplicates it and is dropped at typesetting time anyway.
* Raw scores are redrawn as bars against a **drawn ceiling rule** rather than
  four undifferentiated bars with rotated tick labels.

Data is read from ``reports/order_sensitivity/data.json`` and is not duplicated;
that file remains the single transcription of the cluster eval matrices.

Usage::

    python reports/final/atari_reversed/make_figures.py
"""

from __future__ import annotations

import ast
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
    AC, FONT_MONO, FONT_UI, W_FULL, W_ONE_HALF, export_figure, height_for,
    hex_to_rgba, install_template,
)

DATA = REPO / "reports" / "order_sensitivity" / "data.json"
PNG_DIR = HERE / "png"
SVG_DIR = HERE / "svg"

ORDER_KEY = "reversed"

# ── Series identity ─────────────────────────────────────────────────────────
# Ours is the primary series. CLEAR is the secondary. The two references are
# deliberately quieter: the specialist is neutral grey, the ceiling is drawn as
# a rule rather than a bar so it reads as a threshold and not as a competitor.
COLOR = {
    "MinMax": AC["blue"],
    "CLEAR": AC["amber"],
    "Local": AC["text_faint"],
    "Joint": AC["green"],
}
NAME = {
    "MinMax": "Min-Max (ours)",
    "CLEAR": "CLEAR",
    "Local": "Local specialist",
    "Joint": "Joint ceiling",
}

# ── Retention colour scale ──────────────────────────────────────────────────
# Clamped at the ceiling: z is min(retention, 1.0), so every cell at or above
# 100% lands on the same blue. The old scale ran to 250% to accommodate CLEAR's
# 360% Q*bert cell, which pushed genuinely retained cells (82%) into pink.
#
# The ramp is monotone in "how much survived", which is what the reader is
# actually asked to judge: saturated red is forgotten, pale is partial, blue is
# at or above the ceiling. The pivot sits at PIVOT, so anything below that reads
# as a tint of red however close to the ceiling the top of the scale is.
PIVOT = 0.65
RETENTION_SCALE = [
    [0.00, "#DC2626"],          # forgotten
    [0.22, "#E8736F"],
    [0.45, "#F4B3AE"],
    [PIVOT, "#F2F3F5"],         # neutral
    [0.78, "#B9CDF6"],
    [0.89, "#7CA2F0"],
    [1.00, "#2563EB"],          # at or above the ceiling
]
CEILING_TICKS = [0.0, 0.25, PIVOT, 1.0]
CEILING_TICKTEXT = ["0%", "25%", f"{PIVOT:.0%}", "≥100%"]


# ── Data ────────────────────────────────────────────────────────────────────
def load_data() -> dict:
    """Load the cached eval matrices transcribed from the cluster runs."""
    return json.loads(DATA.read_text(encoding="utf-8"))


def reference_in_order(data: dict, key: str, order: list[str]) -> np.ndarray:
    """Reorder a reference vector from its canonical index into ``order``."""
    index = data["references"]["_index"]
    values = data["references"][key]
    return np.array([values[index.index(game)] for game in order], dtype=float)


def matrix(data: dict, key: str) -> np.ndarray:
    """Load one forgetting matrix, upper triangle as NaN (never evaluated)."""
    rows = data["matrices"][key]
    return np.array(
        [[np.nan if v is None else float(v) for v in row] for row in rows],
        dtype=float,
    )


def retention_matrix(scores: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Divide every evaluation by its task's order-independent ceiling."""
    return scores / reference[None, :]


def random_scores() -> dict[str, float]:
    """Read ``RANDOM_SCORES`` out of ``crl/envs/atari.py`` without importing it.

    Importing the module pulls in ``ale_py``, which a figure-building machine has
    no reason to have. Parsing the literal keeps the one definition authoritative.
    """
    tree = ast.parse((REPO / "crl" / "envs" / "atari.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "RANDOM_SCORES":
            return ast.literal_eval(node.value)
    raise RuntimeError("RANDOM_SCORES not found in crl/envs/atari.py")


def transfer_metrics(data: dict, key: str) -> dict:
    """Backward transfer and the aggregates it travels with, in normalised units.

    Scores are put on a common scale as ``(raw - random) / (ceiling - random)``
    before any averaging: raw backward transfer cannot be averaged across games
    whose scores differ by ~700x (Pong around 20, Q*bert around 4000).

    Backward transfer follows ``analysis/continual_metrics.py``: for each task
    learned before the last one, ``final - just_learned``. Negative is
    forgetting. The final task has nothing trained after it, so it has no
    backward transfer and is excluded from every mean.
    """
    order = data["orders"][ORDER_KEY]
    joint = reference_in_order(data, "joint", order)
    random_by_game = random_scores()
    floor = np.array([random_by_game[game] for game in order], dtype=float)

    scores = matrix(data, key)
    normalised = (scores - floor[None, :]) / (joint - floor)[None, :]

    count = len(order)
    just_learned = np.array([normalised[i, i] for i in range(count)])
    final = normalised[count - 1, :count]
    peak = np.array([np.nanmax(normalised[i:count, i]) for i in range(count)])

    prior = slice(0, count - 1)
    return {
        "games": order,
        "bwt": final - just_learned,          # last entry is 0 by construction
        "bwt_mean": float((final - just_learned)[prior].mean()),
        "forgetting": float((peak - final)[prior].mean()),
        "ap_all": float(final.mean()),
        "ap_prior": float(final[prior].mean()),
    }


def format_score(value: float) -> str:
    """Score label: thousands separated above 100, one decimal below."""
    return f"{value:,.0f}" if abs(value) >= 100 else f"{value:.1f}"


def sample_scale(scale: list, fraction: float) -> str:
    """Sample a Plotly colourscale at ``fraction`` in [0, 1], blending in sRGB.

    Needed because the matrices are drawn as vector rectangles rather than as a
    ``go.Heatmap``: Plotly rasterises heatmap cells into an embedded bitmap, and
    a paper figure has to stay vector all the way down.
    """
    fraction = min(max(fraction, 0.0), 1.0)
    for (low, low_hex), (high, high_hex) in zip(scale, scale[1:]):
        if fraction <= high:
            span = high - low
            weight = 0.0 if span == 0 else (fraction - low) / span
            left = [int(low_hex[i:i + 2], 16) for i in (1, 3, 5)]
            right = [int(high_hex[i:i + 2], 16) for i in (1, 3, 5)]
            blend = [round(a + (b - a) * weight) for a, b in zip(left, right)]
            return "#{:02X}{:02X}{:02X}".format(*blend)
    return scale[-1][1]


def scale_color(fraction: float) -> str:
    """Retention colour for ``fraction`` of the ceiling, clamped to [0, 1]."""
    return sample_scale(RETENTION_SCALE, fraction)


# ── Figure 1: forgetting matrices ───────────────────────────────────────────
def figure_matrices(data: dict) -> None:
    """Min-Max vs CLEAR retention matrices, reversed order, clamped colour."""
    order = data["orders"][ORDER_KEY]
    labels = data["short_labels"]
    joint = reference_in_order(data, "joint", order)
    panels = [
        (NAME["MinMax"], retention_matrix(matrix(data, f"v5_{ORDER_KEY}"), joint)),
        (NAME["CLEAR"], retention_matrix(matrix(data, f"clear_{ORDER_KEY}"), joint)),
    ]
    axis_labels = [labels[game] for game in order]

    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.15,
        subplot_titles=[name for name, _ in panels],
    )

    size = len(axis_labels)
    gap = 0.035  # emulates the heatmap's xgap/ygap without a raster trace

    for column, (_name, values) in enumerate(panels, start=1):
        suffix = "" if column == 1 else str(column)
        for row_index in range(size):
            for column_index in range(row_index + 1):
                value = values[row_index, column_index]
                # Colour is clamped at the ceiling; the printed number is always
                # the true value, however far above 100% it runs.
                shade = min(max(value, 0.0), 1.0)
                fig.add_shape(
                    type="rect", layer="below",
                    x0=column_index + gap, x1=column_index + 1 - gap,
                    y0=row_index + gap, y1=row_index + 1 - gap,
                    xref=f"x{suffix}", yref=f"y{suffix}",
                    fillcolor=scale_color(shade), line=dict(width=0),
                )
                # White only on the two saturated ends of the ramp.
                light = shade < 0.26 or shade > 0.93
                on_diagonal = row_index == column_index
                fig.add_annotation(
                    x=column_index + 0.5, y=row_index + 0.5,
                    xref=f"x{suffix}", yref=f"y{suffix}",
                    text=f"<b>{value:.0%}</b>" if on_diagonal else f"{value:.0%}",
                    showarrow=False,
                    font=dict(
                        family=FONT_MONO, size=10,
                        color=AC["bg"] if light else AC["text_primary"],
                    ),
                )

        # An invisible marker trace carries the colourbar. Plotly renders that as
        # an SVG gradient, unlike a heatmap, which it rasterises.
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", hoverinfo="skip", showlegend=False,
            marker=dict(
                color=[0], colorscale=RETENTION_SCALE, cmin=0.0, cmax=1.0,
                showscale=(column == 2), opacity=0,
                colorbar=dict(
                    title=dict(
                        text="retained vs ceiling",
                        font=dict(family=FONT_UI, size=10, color=AC["text_muted"]),
                        side="right",
                    ),
                    tickmode="array", tickvals=CEILING_TICKS,
                    ticktext=CEILING_TICKTEXT,
                    thickness=9, len=0.70, y=0.5, yanchor="middle", x=1.015,
                    outlinewidth=0, ticklen=3, tickcolor=AC["border"],
                    tickfont=dict(family=FONT_MONO, size=9, color=AC["text_muted"]),
                ),
            ),
        ), row=1, col=column)

    for annotation in fig.layout.annotations[:2]:
        annotation.font = dict(family=FONT_UI, size=12.5, color=AC["text_primary"])
        annotation.y = 1.04

    # Numeric axes with cells on the unit lattice, ticks re-labelled at the cell
    # centres. Category axes cannot address the fractional edges the rectangles
    # need, and the reader sees exactly the same labels either way.
    centres = [i + 0.5 for i in range(size)]
    fig.update_xaxes(
        title=dict(text="evaluated on  (learning order →)",
                   font=dict(family=FONT_UI, size=10.5, color=AC["text_muted"])),
        range=[0, size], tickmode="array", tickvals=centres, ticktext=axis_labels,
        showgrid=False, ticklen=0, side="bottom", showline=False, zeroline=False,
        tickfont=dict(family=FONT_UI, size=9, color=AC["text_muted"]),
    )
    fig.update_yaxes(
        range=[size, 0], tickmode="array", tickvals=centres, ticktext=axis_labels,
        showgrid=False, ticklen=0, showline=False, zeroline=False,
        tickfont=dict(family=FONT_UI, size=9, color=AC["text_muted"]),
    )
    fig.update_yaxes(
        title=dict(text="after consolidating",
                   font=dict(family=FONT_UI, size=10.5, color=AC["text_muted"])),
        row=1, col=1,
    )
    fig.update_layout(title=None, margin=dict(l=76, r=118, t=44, b=58))

    export_pair(fig, "forgetting_matrices", W_FULL, height_for(W_FULL, 1.80))


# ── Figure 2: final raw scores ──────────────────────────────────────────────
def figure_final_scores(data: dict) -> None:
    """Per-game final scores as bars, with the Joint ceiling drawn as a rule."""
    order = data["orders"][ORDER_KEY]
    labels = data["short_labels"]
    joint = reference_in_order(data, "joint", order)
    local = reference_in_order(data, f"local_{ORDER_KEY}", order)
    minmax = matrix(data, f"v5_{ORDER_KEY}")[-1]
    clear = matrix(data, f"clear_{ORDER_KEY}")[-1]

    # Bars are the two methods plus the specialist. The ceiling is a rule, not a
    # bar: it is a threshold to clear, and drawing it as a fourth bar made the
    # panel read as a four-way race with no reference at all.
    series = [("Local", local), ("MinMax", minmax), ("CLEAR", clear)]

    # The ceiling value rides in the panel subtitle rather than beside its rule.
    # Beside the rule it either collided with a bar label (Pong: ceiling 20.7
    # against bars at 21.0) or was struck through by the rule itself.
    titles = [
        f"{labels[game]}<br><span style=\"font-size:9px;color:{COLOR['Joint']}\">"
        f"ceiling {format_score(joint[i])}</span>"
        for i, game in enumerate(order)
    ]
    fig = make_subplots(rows=1, cols=len(order), horizontal_spacing=0.055,
                        subplot_titles=titles)

    for column, _game in enumerate(order, start=1):
        index = column - 1
        for key, values in series:
            value = float(values[index])
            fig.add_trace(go.Bar(
                x=[NAME[key]], y=[value],
                name=NAME[key], legendgroup=key, showlegend=(column == 1),
                marker=dict(color=COLOR[key], line=dict(width=0)),
                width=0.62,
                text=[format_score(value)], textposition="outside", cliponaxis=False,
                textfont=dict(family=FONT_MONO, size=9.5, color=AC["text_primary"]),
                hovertemplate="%{x}: %{y:,.1f}<extra></extra>",
            ), row=1, col=column)

        ceiling = float(joint[index])
        fig.add_hline(
            y=ceiling, line=dict(color=COLOR["Joint"], width=1.3, dash="dash"),
            row=1, col=column,
        )

        # Bars need headroom for their outside labels; the ceiling rule carries
        # its label beside it and needs far less. Giving the rule the same 1.26
        # headroom as a bar left a third of every panel empty.
        bar_top = max(float(v[index]) for _, v in series)
        top = max(bar_top * 1.20, ceiling * 1.04)
        fig.update_yaxes(range=[0, top], row=1, col=column)

    # Q*bert is the one panel where a bar runs away from the ceiling; say by how
    # much rather than leaving the reader to divide two four-digit numbers.
    last = len(order)
    fig.add_annotation(
        x=0.02, y=0.72, xref=f"x{last} domain", yref=f"y{last} domain",
        text=f"<b>{clear[-1] / joint[-1]:.1f}×</b> ceiling", showarrow=False,
        xanchor="left", yanchor="middle",
        font=dict(family=FONT_UI, size=10, color=COLOR["CLEAR"]),
    )

    for annotation in fig.layout.annotations[:len(order)]:
        annotation.font = dict(family=FONT_UI, size=11.5, color=AC["text_primary"])
        annotation.y = 1.04

    # Method names live in the legend once, which is what lets the per-panel
    # x-axis drop its tick labels; four rotated labels per panel were the single
    # worst thing about the exploratory version of this figure.
    fig.update_xaxes(showticklabels=False, ticklen=0, showgrid=False,
                     showline=True, linecolor=AC["border"], linewidth=1.0)
    fig.update_yaxes(showgrid=True, gridcolor=AC["grid"], gridwidth=0.6,
                     nticks=4, zeroline=False, showline=False, ticklen=0,
                     tickangle=0,
                     tickfont=dict(family=FONT_MONO, size=9,
                                   color=AC["text_muted"]))
    fig.update_yaxes(
        title=dict(text="greedy-100 score",
                   font=dict(family=FONT_UI, size=11, color=AC["text_muted"])),
        row=1, col=1,
    )
    fig.update_layout(
        title=None, barmode="group", bargap=0.30,
        margin=dict(l=58, r=14, t=98, b=22),
        legend=dict(
            orientation="h", x=0.0, xanchor="left", y=1.26, yanchor="bottom",
            font=dict(family=FONT_UI, size=10.5, color=AC["text_primary"]),
            bgcolor="rgba(0,0,0,0)", borderwidth=0,
            itemsizing="constant", tracegroupgap=0,
        ),
        showlegend=True,
    )
    # The ceiling has no trace, so it needs its own legend entry to be named.
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines", name=NAME["Joint"],
        line=dict(color=COLOR["Joint"], width=1.3, dash="dash"),
        showlegend=True, hoverinfo="skip",
    ), row=1, col=1)

    export_pair(fig, "final_scores", W_FULL, height_for(W_FULL, 1.52))


# ── Figure: backward-transfer matrix ────────────────────────────────────────
# Signed, so the pivot is 0. The arms are deliberately NOT the same length:
# backward transfer here runs to -1.12 but only to +0.16, so equal arms would
# spend half the palette on a sign that barely occurs and leave every gain as a
# tint indistinguishable from no change. The gain arm therefore saturates at
# BWT_GAIN and the loss arm at BWT_LOSS, and the colourbar shows both extents so
# the asymmetry is visible rather than hidden.
BWT_LOSS, BWT_GAIN = 1.15, 0.25

# Stops are expressed on the symmetric [-BWT_LOSS, +BWT_LOSS] axis the colourbar
# uses, with full blue pulled in to where +BWT_GAIN falls.
_GAIN_STOP = 0.5 + 0.5 * BWT_GAIN / BWT_LOSS
BWT_SCALE = [
    [0.00, "#DC2626"],                        # heaviest loss
    [0.16, "#E8736F"],
    [0.33, "#F4B3AE"],
    [0.50, "#F2F3F5"],                        # no change
    [0.50 + 0.35 * (_GAIN_STOP - 0.5), "#C7D8F8"],
    [0.50 + 0.70 * (_GAIN_STOP - 0.5), "#7CA2F0"],
    [_GAIN_STOP, "#2563EB"],                  # gain at or above BWT_GAIN
    [1.00, "#2563EB"],
]


def bwt_matrix(data: dict, key: str) -> np.ndarray:
    """Backward transfer at every training phase, in normalised units.

    Cell ``(i, j)`` is task ``j`` measured after consolidating task ``i``, minus
    task ``j`` when it was just learned. Negative is forgetting. The diagonal is
    zero by construction and the upper triangle was never evaluated.

    This is the whole lower triangle of what ``transfer_metrics`` reduces to its
    last row, so it shows *when* a task was lost, and whether it came back.
    """
    order = data["orders"][ORDER_KEY]
    joint = reference_in_order(data, "joint", order)
    floor = np.array([random_scores()[game] for game in order], dtype=float)

    normalised = (matrix(data, key) - floor[None, :]) / (joint - floor)[None, :]
    just_learned = np.array([normalised[i, i] for i in range(len(order))])
    return normalised - just_learned[None, :]


def figure_bwt_matrix(data: dict) -> None:
    """Backward transfer at every phase, Min-Max against CLEAR."""
    order = data["orders"][ORDER_KEY]
    labels = data["short_labels"]
    panels = [
        (NAME["MinMax"], bwt_matrix(data, f"v5_{ORDER_KEY}")),
        (NAME["CLEAR"], bwt_matrix(data, f"clear_{ORDER_KEY}")),
    ]
    axis_labels = [labels[game] for game in order]
    size = len(axis_labels)

    limit = BWT_LOSS
    observed = max(abs(np.nanmin(v)) for _, v in panels)
    if observed > limit:
        raise RuntimeError(
            f"BWT_LOSS={limit} is below the observed {observed:.3f}; widen it "
            "rather than letting a cell silently saturate")

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.15,
                        subplot_titles=[name for name, _ in panels])

    gap = 0.035
    for column, (_name, values) in enumerate(panels, start=1):
        suffix = "" if column == 1 else str(column)
        for row_index in range(size):
            for column_index in range(row_index + 1):
                value = values[row_index, column_index]
                on_diagonal = row_index == column_index
                # The diagonal is zero by definition, not a measurement, and is
                # held apart from the scale so it cannot be confused with a
                # measured zero (Min-Max holds Pong at exactly 0.00).
                fraction = 0.5 + 0.5 * value / limit
                fill = AC["surface"] if on_diagonal else sample_scale(
                    BWT_SCALE, fraction)
                fig.add_shape(
                    type="rect", layer="below",
                    x0=column_index + gap, x1=column_index + 1 - gap,
                    y0=row_index + gap, y1=row_index + 1 - gap,
                    xref=f"x{suffix}", yref=f"y{suffix}",
                    fillcolor=fill, line=dict(width=0),
                )
                if on_diagonal:
                    text, tone = "·", AC["text_faint"]
                else:
                    # Contrast is judged on where the cell sits in the ramp, not
                    # on |value|: the arms are different lengths, so a modest
                    # gain can be a saturated blue while a larger loss is not.
                    text = f"{value:+.2f}"
                    light = fraction < 0.18 or fraction > _GAIN_STOP - 0.03
                    tone = AC["bg"] if light else AC["text_primary"]
                fig.add_annotation(
                    x=column_index + 0.5, y=row_index + 0.5,
                    xref=f"x{suffix}", yref=f"y{suffix}",
                    text=text, showarrow=False,
                    font=dict(family=FONT_MONO, size=9.5, color=tone),
                )

        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", hoverinfo="skip", showlegend=False,
            marker=dict(
                color=[0], colorscale=BWT_SCALE, cmin=-limit, cmax=limit,
                showscale=(column == 2), opacity=0,
                colorbar=dict(
                    title=dict(text="backward transfer",
                               font=dict(family=FONT_UI, size=10,
                                         color=AC["text_muted"]),
                               side="right"),
                    tickmode="array",
                    tickvals=[-1.0, -0.5, -0.25, 0.0, BWT_GAIN],
                    ticktext=["−1.00", "−0.50", "−0.25", "0",
                              f"≥ +{BWT_GAIN:.2f}"],
                    thickness=9, len=0.70, y=0.5, yanchor="middle", x=1.015,
                    outlinewidth=0, ticklen=3, tickcolor=AC["border"],
                    tickfont=dict(family=FONT_MONO, size=9,
                                  color=AC["text_muted"]),
                ),
            ),
        ), row=1, col=column)

    for annotation in fig.layout.annotations[:2]:
        annotation.font = dict(family=FONT_UI, size=12.5, color=AC["text_primary"])
        annotation.y = 1.04

    centres = [i + 0.5 for i in range(size)]
    fig.update_xaxes(
        title=dict(text="task  (learning order →)",
                   font=dict(family=FONT_UI, size=10.5, color=AC["text_muted"])),
        range=[0, size], tickmode="array", tickvals=centres, ticktext=axis_labels,
        showgrid=False, ticklen=0, side="bottom", showline=False, zeroline=False,
        tickfont=dict(family=FONT_UI, size=9, color=AC["text_muted"]),
    )
    fig.update_yaxes(
        range=[size, 0], tickmode="array", tickvals=centres, ticktext=axis_labels,
        showgrid=False, ticklen=0, showline=False, zeroline=False,
        tickfont=dict(family=FONT_UI, size=9, color=AC["text_muted"]),
    )
    fig.update_yaxes(
        title=dict(text="after consolidating",
                   font=dict(family=FONT_UI, size=10.5, color=AC["text_muted"])),
        row=1, col=1,
    )
    fig.update_layout(title=None, margin=dict(l=76, r=118, t=44, b=142))

    fig.add_annotation(
        x=0, y=-0.30, xref="paper", yref="paper", xshift=-70,
        text=("Task j after consolidating task i, minus task j when it was just "
              "learned, in units of (score − random) / (ceiling − random). "
              "Negative is forgetting.<br>"
              "The diagonal (·) is zero by construction, not a measurement, and "
              "is held off the scale so it cannot be mistaken for a measured "
              "zero. The upper triangle was<br>"
              "never evaluated. The loss and gain arms of the scale are different lengths, "
              "because backward transfer here runs to −1.12 but only to +0.16; the "
              "colourbar shows both.<br>"
              "Nothing is clamped. Seed 0, so no "
              "error bars are drawn and none are invented."),
        showarrow=False, xanchor="left", yanchor="top", align="left",
        font=dict(family=FONT_UI, size=8.5, color=AC["text_muted"]),
    )

    export_pair(fig, "backward_transfer_matrix", W_FULL, height_for(W_FULL, 1.34))


# ── Figure 3: transfer table ────────────────────────────────────────────────
# Backward transfer is tinted about zero, because its sign is unambiguous:
# below zero the method lost ground on a task it had already learned.
BWT_SCALE = [
    [0.00, "#DC2626"], [0.28, "#F09A96"], [0.50, "#F2F3F5"],
    [0.72, "#9DBAF3"], [1.00, "#2563EB"],
]
BWT_SPAN = 1.0  # tint saturates at +-1.0 normalised units


def bwt_color(value: float) -> str:
    """Tint for one backward-transfer cell, neutral at zero."""
    fraction = 0.5 + 0.5 * max(min(value / BWT_SPAN, 1.0), -1.0)
    for (low, low_hex), (high, high_hex) in zip(BWT_SCALE, BWT_SCALE[1:]):
        if fraction <= high:
            span = high - low
            weight = 0.0 if span == 0 else (fraction - low) / span
            left = [int(low_hex[i:i + 2], 16) for i in (1, 3, 5)]
            right = [int(high_hex[i:i + 2], 16) for i in (1, 3, 5)]
            return "#{:02X}{:02X}{:02X}".format(
                *[round(a + (b - a) * weight) for a, b in zip(left, right)])
    return BWT_SCALE[-1][1]


def figure_transfer_table(data: dict) -> None:
    """Booktabs-style transfer table: per-task backward transfer plus aggregates.

    Forward transfer is reported as not measurable rather than estimated. Two
    independent reasons, either sufficient: the runs set ``eval_all_tasks:
    false`` so a task is never evaluated before it is trained, and the
    multi-head policy gives every task its own head that stays at its random
    initialisation until that task arrives, so a zero-shot number would measure
    an untrained head rather than transfer through the shared trunk.
    """
    minmax = transfer_metrics(data, f"v5_{ORDER_KEY}")
    clear = transfer_metrics(data, f"clear_{ORDER_KEY}")
    labels = data["labels"]
    order = minmax["games"]
    last = len(order) - 1

    # (kind, label, minmax cell, clear cell). "tint" cells carry a background.
    rows: list[tuple[str, str, str, str]] = [("section", "Backward transfer", "", "")]
    for i, game in enumerate(order):
        if i == last:
            rows.append(("muted", f"{i + 1}.  {labels[game]}", "—", "—"))
        else:
            rows.append(("tint", f"{i + 1}.  {labels[game]}",
                         f"{minmax['bwt'][i]:+.2f}", f"{clear['bwt'][i]:+.2f}"))
    rows.append(("rule", "", "", ""))
    rows.append(("total", "Mean", f"{minmax['bwt_mean']:+.2f}",
                 f"{clear['bwt_mean']:+.2f}"))
    rows.append(("section", "Aggregate", "", ""))
    rows.append(("plain", "Forgetting",
                 f"{minmax['forgetting']:.2f}", f"{clear['forgetting']:.2f}"))
    rows.append(("plain", "Average performance, all 5 tasks",
                 f"{minmax['ap_all']:.2f}", f"{clear['ap_all']:.2f}"))
    rows.append(("plain", "Average performance, prior 4 tasks",
                 f"{minmax['ap_prior']:.2f}", f"{clear['ap_prior']:.2f}"))
    rows.append(("section", "Forward transfer", "", ""))
    rows.append(("muted", "Zero-shot, before training on the task", "—", "—"))

    # Geometry in arbitrary units; the axes are hidden and only host the layout.
    # Rows advance a cursor downward by their own height, so a separator costs a
    # third of a row rather than a whole empty one.
    label_x, col_x = 0.0, [66.0, 92.0]
    tint_half = 11.0
    ROW_HEIGHT = {"rule": 0.34, "section": 0.92}

    placed: list[tuple[tuple[str, str, str, str], float]] = []
    cursor = 0.0
    for row in rows:
        height = ROW_HEIGHT.get(row[0], 1.0)
        # A rule is drawn on the cursor; everything else is centred in its band.
        placed.append((row, cursor if row[0] == "rule" else cursor + height / 2))
        cursor += height
    body_bottom = cursor

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                             hoverinfo="skip", showlegend=False))

    # Header: the two method columns, over the top rule.
    header_y = 0.62
    for x, name in zip(col_x, [NAME["MinMax"], NAME["CLEAR"]]):
        fig.add_annotation(
            x=x, y=header_y, text=f"<b>{name}</b>", showarrow=False,
            xanchor="right", yanchor="middle",
            font=dict(family=FONT_UI, size=11, color=AC["text_primary"]),
        )
    fig.add_annotation(
        x=label_x, y=header_y,
        text="<b>normalised units</b>", showarrow=False,
        xanchor="left", yanchor="middle",
        font=dict(family=FONT_UI, size=10, color=AC["text_muted"]),
    )

    def rule(y: float, width: float, color: str) -> None:
        fig.add_shape(type="line", x0=label_x - 1, x1=col_x[-1] + 1, y0=y, y1=y,
                      line=dict(color=color, width=width), layer="above")

    rule(header_y + 0.55, 1.3, AC["axis"])   # top rule
    rule(header_y - 0.48, 1.0, AC["axis"])   # under the header

    for (kind, label, left_cell, right_cell), depth in placed:
        y = -depth
        if kind == "rule":
            rule(y, 0.7, AC["border"])
            continue
        if kind == "section":
            fig.add_annotation(
                x=label_x, y=y, text=label.upper(), showarrow=False,
                xanchor="left", yanchor="middle",
                font=dict(family=FONT_UI, size=9, color=AC["text_muted"]),
            )
            continue

        bold = kind == "total"
        label_color = AC["text_muted"] if kind == "muted" else AC["text_primary"]
        fig.add_annotation(
            x=label_x + 3, y=y,
            text=f"<b>{label}</b>" if bold else label, showarrow=False,
            xanchor="left", yanchor="middle",
            font=dict(family=FONT_UI, size=10.5, color=label_color),
        )
        for x, cell in zip(col_x, [left_cell, right_cell]):
            if kind == "tint":
                fig.add_shape(
                    type="rect", layer="below",
                    x0=x - 2 * tint_half, x1=x + 1.5,
                    y0=y - 0.42, y1=y + 0.42,
                    fillcolor=bwt_color(float(cell)), line=dict(width=0),
                )
            fig.add_annotation(
                x=x, y=y, text=f"<b>{cell}</b>" if bold else cell,
                showarrow=False, xanchor="right", yanchor="middle",
                font=dict(family=FONT_MONO, size=10.5,
                          color=AC["text_faint"] if kind == "muted"
                          else AC["text_primary"]),
            )

    rule(-body_bottom - 0.12, 1.3, AC["axis"])  # bottom rule

    # Table footnotes. Without them the "—" cells look like work left undone
    # rather than a measurement this study's design cannot support. Wrapped by
    # hand: Plotly annotations do not reflow.
    footnote = (
        "Normalised as (score − random) / (Joint ceiling − random).<br>"
        "Backward transfer is final − just-learned; the last task has none.<br>"
        "— Forward transfer is not measurable here: every task has its own head,<br>"
        "&nbsp;&nbsp;&nbsp;untrained until that task arrives, and no task is "
        "evaluated before it is trained."
    )
    fig.add_annotation(
        x=label_x - 1, y=-body_bottom - 0.5, text=footnote,
        showarrow=False, xanchor="left", yanchor="top", align="left",
        font=dict(family=FONT_UI, size=8.5, color=AC["text_muted"]),
    )

    fig.update_xaxes(visible=False, range=[label_x - 2, col_x[-1] + 2])
    fig.update_yaxes(visible=False, range=[-body_bottom - 3.1, header_y + 1.1])
    fig.update_layout(title=None, showlegend=False, plot_bgcolor=AC["bg"],
                      margin=dict(l=16, r=16, t=14, b=10))

    export_pair(fig, "transfer_table", W_ONE_HALF, 330)


# ── Figure 4: compute cost ──────────────────────────────────────────────────
def figure_compute_cost() -> None:
    """Wall-clock cost of one full five-game run, as a lollipop with a reference.

    Lollipop rather than bars: with three values the encoding is identical but
    the ink is a tenth of it, which leaves the panel quiet enough to carry a
    reference rule at ours. The rule is what turns three numbers into a
    comparison, and it is the comparison the reader is here for.

    Zero is kept on the axis. Unlike a threshold comparison, hours have a
    meaningful zero and the stem length is a real magnitude.

    CLEAR spans its replay-buffer configurations, which land within 0.9 h of one
    another. That span is drawn at the end of its stem so the dot is not read as
    a single measurement.
    """
    compute = json.loads((HERE / "compute.json").read_text(encoding="utf-8"))
    hours, labels = compute["hours"], compute["labels"]

    clear_values = hours["clear"]
    clear_mean = sum(clear_values) / len(clear_values)
    series = [
        ("ours", hours["ours"], COLOR["MinMax"], None),
        ("clear", clear_mean, COLOR["CLEAR"], (min(clear_values), max(clear_values))),
        ("joint", hours["joint"], COLOR["Joint"], None),
    ]
    # Cheapest at the top: the reader's question is who costs least.
    series.sort(key=lambda item: item[1])

    names = [labels[key] for key, *_ in series]
    reference = hours["ours"]
    axis_high = max(value for _, value, *_ in series) * 1.16

    fig = go.Figure()

    # Reference rule at ours, behind everything. Unlabelled: our own dot sits on
    # it, which says what it is more economically than a label would.
    fig.add_shape(
        type="line", x0=reference, x1=reference, y0=-0.55, y1=len(series) - 0.45,
        line=dict(color=hex_to_rgba(COLOR["MinMax"], 0.35), width=1.1, dash="dash"),
        layer="below",
    )

    for index, (_key, value, color, spread) in enumerate(series):
        # Faint track, so each row reads as its own lane.
        fig.add_trace(go.Scatter(
            x=[0, axis_high], y=[index, index], mode="lines",
            line=dict(color=AC["grid"], width=0.8),
            hoverinfo="skip", showlegend=False,
        ))

        fig.add_trace(go.Scatter(
            x=[0, value], y=[index, index], mode="lines",
            line=dict(color=hex_to_rgba(color, 0.40), width=3),
            hoverinfo="skip", showlegend=False,
        ))

        if spread is not None:
            # Capped, because the span is under a unit wide and the dot would
            # otherwise swallow it whole, leaving the footnote describing a mark
            # nobody can see.
            low, high = spread
            fig.add_trace(go.Scatter(
                x=[low, high], y=[index, index], mode="lines",
                line=dict(color=color, width=2),
                hoverinfo="skip", showlegend=False,
            ))
            for edge in (low, high):
                fig.add_trace(go.Scatter(
                    x=[edge, edge], y=[index - 0.13, index + 0.13], mode="lines",
                    line=dict(color=color, width=2),
                    hoverinfo="skip", showlegend=False,
                ))

        fig.add_trace(go.Scatter(
            x=[value], y=[index], mode="markers",
            marker=dict(color=color, size=11,
                        line=dict(color=AC["bg"], width=2)),
            hovertemplate="%{x:.1f} h<extra></extra>", showlegend=False,
        ))

        ratio = value / reference
        tail = "" if abs(ratio - 1.0) < 1e-9 else (
            f"<span style=\"font-size:9px;color:{AC['text_muted']}\">"
            f"  {ratio:.2f}×</span>")
        fig.add_annotation(
            x=value, y=index, text=f"<b>{value:.1f} h</b>{tail}",
            showarrow=False, xanchor="left", yanchor="middle", xshift=12,
            font=dict(family=FONT_MONO, size=11, color=AC["text_primary"]),
        )

    fig.update_xaxes(
        title=dict(text="wall-clock hours for one full five-game run",
                   font=dict(family=FONT_UI, size=11, color=AC["text_muted"])),
        range=[0, axis_high],
        showgrid=True, gridcolor=AC["grid"], gridwidth=0.6,
        zeroline=False, showline=True, linecolor=AC["border"], ticklen=0,
        tickfont=dict(family=FONT_MONO, size=9.5, color=AC["text_muted"]),
    )
    fig.update_yaxes(
        tickmode="array", tickvals=list(range(len(series))), ticktext=names,
        showgrid=False, zeroline=False, showline=False, ticklen=0,
        range=[len(series) - 0.45, -0.55],
        tickfont=dict(family=FONT_UI, size=11.5, color=AC["text_primary"]),
    )
    fig.add_annotation(
        x=0, y=-0.40, xref="paper", yref="paper",
        text=("× is relative to Min-Max, marked by the dashed rule. CLEAR's capped span "
              "covers its buffer configurations (36.4–37.3 h).<br>"
              "Single seed, single GPU. Ours sums per-phase training time while "
              "CLEAR and Joint use total elapsed<br>"
              "time, so ours is the series understated by the difference."),
        showarrow=False, xanchor="left", yanchor="top", align="left",
        # Back out of the left margin so the note starts at the figure edge,
        # not at the plot edge, which is held wide by the category labels.
        xshift=-100,
        font=dict(family=FONT_UI, size=8.5, color=AC["text_muted"]),
    )
    fig.update_layout(title=None, showlegend=False,
                      margin=dict(l=108, r=96, t=14, b=104))

    export_pair(fig, "compute_cost", W_ONE_HALF, 250)


# ── Export ──────────────────────────────────────────────────────────────────
def export_pair(fig: go.Figure, stem: str, width: int, height: int) -> None:
    """Write ``png/<stem>.png`` and ``svg/<stem>.svg``, both verified non-empty."""
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    SVG_DIR.mkdir(parents=True, exist_ok=True)

    export_figure(fig, PNG_DIR / stem, width, height, label=stem)
    (PNG_DIR / f"{stem}.svg").replace(SVG_DIR / f"{stem}.svg")
    print(f"  png/{stem}.png  svg/{stem}.svg")


def main() -> int:
    install_template()
    data = load_data()
    print("reports/final/atari_reversed")
    figure_matrices(data)
    figure_final_scores(data)
    figure_bwt_matrix(data)
    figure_transfer_table(data)
    figure_compute_cost()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
