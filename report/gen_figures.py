#!/usr/bin/env python3
"""Generate all research figures for the CRL-Minimax dashboard.

Reads from results/ directories, uses report/acviz.py for template/palette/export,
writes PNG+SVG into report/figures/, and appends entries to report/manifest.json.

Single-seed runs only. No CI bands are drawn; captions note single-seed origin.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Paths ───────────────────────────────────────────────────────────────────
REPO = Path("/work/pnag/CRL-Minimax")
REPORT = REPO / "report"
FIGURES = REPORT / "figures"
MANIFEST = REPORT / "manifest.json"
RESULTS = REPO / "results"
FIGURES.mkdir(parents=True, exist_ok=True)

# ── Import acviz (the canonical template module, do NOT rewrite) ─────────────
sys.path.insert(0, str(REPORT))
from acviz import (
    AC, AC_SERIES, EXPORT_SCALE, W_FULL, W_ONE_HALF, W_SINGLE_COL,
    add_callout, add_end_labels, append_manifest_entry,
    export_figure, hex_to_rgba, install_template, height_for,
)

install_template()  # registers "academic" as default

FONT_UI    = "Inter, Helvetica, Arial, sans-serif"
FONT_TITLE = "Source Serif 4, Georgia, serif"
FONT_MONO  = "JetBrains Mono, Menlo, Consolas, monospace"

# ── Game metadata ────────────────────────────────────────────────────────────
GAMES = ["Qbert", "Pong", "Breakout", "Boxing", "SpaceInvaders"]
GAME_LABELS = ["Q*bert", "Pong", "Breakout", "Boxing", "Space Invaders"]

# From crl/envs/atari.py  RANDOM_SCORES
RANDOM_SCORES = {
    "Qbert": 163.9, "Pong": -20.7, "Breakout": 1.7, "Boxing": 0.1,
    "SpaceInvaders": 148.0,
}

# Thresholds from config.yaml (V5 canonical)
THRESHOLDS = {
    "Qbert": 4000.0, "Pong": 18.0, "Breakout": 80.0, "Boxing": 55.0,
    "SpaceInvaders": 800.0,
}

# ── Raw data ─────────────────────────────────────────────────────────────────
def load_eval_matrix(run: str) -> np.ndarray:
    """Load the triangular eval matrix; pad to 5x5 with NaN."""
    path = RESULTS / run / "eval_matrix.json"
    rows_raw = json.loads(path.read_text())
    T = 5
    M = np.full((T, T), np.nan)
    for i, row in enumerate(rows_raw):
        for j, v in enumerate(row):
            M[i, j] = v
    return M

def load_resource(run: str) -> dict:
    path = RESULTS / run / "resource_usage.json"
    return json.loads(path.read_text())

def load_joint_result(run: str) -> dict:
    path = RESULTS / run / "joint_result.json"
    return json.loads(path.read_text())

def load_logs(run: str) -> list[dict]:
    path = RESULTS / run / "logs.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

# Canonical order: Qbert, Pong, Breakout, Boxing, SpaceInvaders
M_v5 = load_eval_matrix("atari5_v5_seed0")

# Reversed order: SpaceInvaders, Boxing, Breakout, Pong, Qbert
# eval_matrix rows/cols follow the training order; we need to re-index by game
GAMES_REV = ["SpaceInvaders", "Boxing", "Breakout", "Pong", "Qbert"]
M_v5r_raw = load_eval_matrix("atari5_v5_order2_seed0")  # 5x5, in reversed-order indices

# CLEAR matrices (all in canonical order)
M_clearA = load_eval_matrix("atari5_v5_clearA_equal_seed0")
M_clearB = load_eval_matrix("atari5_v5_clearB_gen_seed0")
M_clearC = load_eval_matrix("atari5_v5_clearC_vgen_seed0")

# Joint 6M
joint_6m = load_joint_result("atari5_joint_6m_seed0")
JOINT_6M_SCORES = dict(zip(
    [g.replace("atari-", "") for g in joint_6m["games"]],
    joint_6m["joint"],
))

# Joint stopped (last probe row at ~46M frames / step 4500)
joint_stopped_scores_raw = [4430.25, 21.0, 218.07, 70.06, 1375.2]  # Qbert,Pong,Breakout,Boxing,SI

# V5 canonical final per game (cross-checked)
V5_FINAL = dict(zip(GAMES, [4075.0, 19.75, 51.81, -25.76, 765.85]))
V5_LOCAL  = dict(zip(GAMES, [4467.8, 20.01, 132.68, 93.99, 1132.2]))

# V5 reversed final per game (by game name, not training order)
# Cross-check: Qbert=4270.2, Pong=21.0, Breakout=199.8, Boxing=55.4, SI=711.7
V5R_FINAL = dict(zip(GAMES, [4270.2, 21.0, 199.8, 55.4, 711.7]))
# Local refs for reversed order
V5R_LOCAL  = dict(zip(GAMES, [4341.0, 21.0, 363.9, 98.9, 588.5]))

# CLEAR-A final (Qbert,Pong,Breakout,Boxing,SI)
CLEARA_FINAL = dict(zip(GAMES, [4350.0, 21.0, 0.0, 100.0, 800.0]))

# Helper: normalize score to [0,1] relative to (random, threshold)
def normalize(score: float, game: str) -> float:
    rnd = RANDOM_SCORES[game]
    tgt = THRESHOLDS[game]
    denom = tgt - rnd
    return (score - rnd) / denom if denom != 0 else 0.0

def norm_dict(scores: dict[str, float]) -> dict[str, float]:
    return {g: normalize(v, g) for g, v in scores.items()}

# ── CL metrics from matrix ────────────────────────────────────────────────────
def cl_metrics_from_matrix(M: np.ndarray) -> dict[str, float]:
    T = M.shape[0]
    final = M[T - 1, :T]
    ap = float(np.nanmean(final))
    forgets, bwts = [], []
    for j in range(T - 1):
        col = M[j:T, j]
        col_valid = col[~np.isnan(col)]
        if len(col_valid) == 0:
            continue
        peak = float(np.nanmax(col))
        just = float(M[j, j])
        fin  = float(M[T - 1, j])
        forgets.append(peak - fin)
        bwts.append(fin - just)
    return {
        "avg_performance": ap,
        "forgetting": float(np.mean(forgets)) if forgets else 0.0,
        "bwt": float(np.mean(bwts)) if bwts else 0.0,
    }

def cl_metrics_norm(M: np.ndarray, game_order: list[str]) -> dict[str, float]:
    """CL metrics on normalized scores."""
    T = M.shape[0]
    Mn = np.full_like(M, np.nan)
    for j, g in enumerate(game_order):
        rnd = RANDOM_SCORES[g]
        tgt = THRESHOLDS[g]
        denom = tgt - rnd
        if denom != 0:
            Mn[:, j] = (M[:, j] - rnd) / denom
    return cl_metrics_from_matrix(Mn)

# ─────────────────────────────────────────────────────────────────────────────
# GROUP 1: method_comparison
# Figure 1a: per-game final scores — small multiples bar chart
# ─────────────────────────────────────────────────────────────────────────────
def fig_method_comparison_scores() -> None:
    """Per-game final scores: V5 vs CLEAR-A vs Joint-6M. Small multiples."""
    methods = ["Ours (V5)", "CLEAR (std)", "Joint 6M"]
    colors  = [AC_SERIES[0], AC_SERIES[1], AC_SERIES[2]]

    # Per-game scores in canonical order
    data = {
        "Ours (V5)":    [V5_FINAL[g] for g in GAMES],
        "CLEAR (std)":  [CLEARA_FINAL[g] for g in GAMES],
        "Joint 6M":     [JOINT_6M_SCORES[g] for g in GAMES],
    }

    W = W_FULL
    H = int(W * 0.38)  # wide and short for 5-column multiples
    fig = make_subplots(
        rows=1, cols=5,
        subplot_titles=GAME_LABELS,
        shared_yaxes=False,
        horizontal_spacing=0.055,
    )

    for col_i, game in enumerate(GAMES):
        for m_i, method in enumerate(methods):
            score = data[method][col_i]
            fig.add_trace(
                go.Bar(
                    x=[method],
                    y=[score],
                    marker_color=colors[m_i],
                    name=method,
                    showlegend=(col_i == 0),
                    width=0.55,
                    text=[f"{score:.0f}"],
                    textposition="outside",
                    textfont=dict(family=FONT_MONO, size=9, color=AC["text_muted"]),
                    cliponaxis=False,
                ),
                row=1, col=col_i + 1,
            )

        # Zero line for games that can go negative
        fig.add_hline(y=0, line_width=0.8, line_color=AC["axis"],
                      line_dash="solid", row=1, col=col_i + 1)

        # Threshold line
        thr = THRESHOLDS[game]
        fig.add_hline(y=thr, line_width=1.0, line_color=AC["amber"],
                      line_dash="dot", row=1, col=col_i + 1)

    # Callout: Breakout=0 for CLEAR is notable (col 3)
    fig.add_annotation(
        x="CLEAR (std)", y=0,
        xref="x3", yref="y3",
        text="CLEAR forgot<br>Breakout entirely",
        showarrow=True, arrowhead=0, arrowwidth=1.0, arrowcolor=AC["axis"],
        ax=0, ay=-45, xanchor="center",
        font=dict(family=FONT_UI, size=10, color=AC["red"]),
        bgcolor="rgba(255,255,255,0.85)", borderpad=3,
    )

    # Style
    fig.update_layout(
        title=dict(text="Per-game final greedy-100 scores: Ours vs CLEAR vs Joint",
                   font=dict(family=FONT_TITLE, size=15)),
        paper_bgcolor=AC["bg"], plot_bgcolor=AC["bg"],
        margin=dict(l=48, r=24, t=68, b=56),
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5,
            font=dict(family=FONT_UI, size=11), bgcolor="rgba(0,0,0,0)",
        ),
        barmode="group",
        font=dict(family=FONT_UI, size=11, color=AC["text_primary"]),
    )
    fig.update_annotations(font=dict(family=FONT_TITLE, size=11))
    for i in range(1, 6):
        fig.update_xaxes(showline=True, linecolor=AC["axis"], linewidth=1.0,
                         ticks="", showticklabels=False,
                         showgrid=False, row=1, col=i)
        fig.update_yaxes(showline=True, linecolor=AC["axis"], linewidth=1.0,
                         showgrid=True, gridcolor=AC["grid"], gridwidth=0.6,
                         tickfont=dict(family=FONT_UI, size=10, color=AC["text_muted"]),
                         zeroline=False, row=1, col=i)

    stem = FIGURES / "method_comparison_scores"
    v = export_figure(fig, stem, W, H, label="Small multiples")
    append_manifest_entry(
        MANIFEST,
        id="method_comparison_scores",
        group="method_comparison",
        group_title="Method Comparison",
        title="Per-game final scores: Ours vs CLEAR vs Joint ceiling",
        caption="Final greedy-100 scores per game. Dotted amber line = task threshold. CLEAR-std forgot Breakout (score 0). Single seed.",
        details=(
            "Five-game Atari suite. Ours (V5, min-max consolidation, canonical order), "
            "CLEAR standard buffer (batches=4), Joint 6M ceiling (6M frames/game). "
            "All single seed 0, impala_ac_multihead, greedy-100 evaluation. "
            "Boxing can be negative (V5 = -25.8). "
            "Dotted amber line is the per-game task threshold from config. "
            "Zero line drawn explicitly. CLEAR forgot Breakout entirely (0.0)."
        ),
        variants=[v],
        seed=0, dataset="Atari-5", model="impala_ac_multihead",
        metrics=[
            {"name": "V5 Boxing", "value": round(V5_FINAL["Boxing"], 1), "unit": "score"},
            {"name": "CLEAR Breakout", "value": CLEARA_FINAL["Breakout"], "unit": "score"},
            {"name": "Joint Breakout", "value": round(JOINT_6M_SCORES["Breakout"], 1), "unit": "score"},
        ],
    )
    _verify(stem)
    print("  [ok] method_comparison_scores")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1b-d: forgetting matrices (heatmaps) — V5-canonical, V5-reversed, CLEAR-A
# ─────────────────────────────────────────────────────────────────────────────
def _make_heatmap(M: np.ndarray, game_order: list[str],
                  title: str, label_pfx: str) -> tuple[go.Figure, int, int]:
    """Build a 5x5 forgetting-matrix heatmap figure."""
    GL = [GAME_LABELS[GAMES.index(g)] for g in game_order]

    # Determine color range symmetrically around midpoint, diverging
    vals = M[~np.isnan(M)]
    vmax = max(abs(float(np.nanmax(M))), abs(float(np.nanmin(M))), 1.0)
    # Use blue-white-red diverging scale
    colorscale = [
        [0.0, AC["blue"]], [0.5, AC["bg"]], [1.0, AC["red"]]
    ]

    # Build cell text
    text = [[
        f"{M[i, j]:.0f}" if not np.isnan(M[i, j]) else ""
        for j in range(5)
    ] for i in range(5)]

    fig = go.Figure(go.Heatmap(
        z=M,
        x=[f"Game {j+1}<br>{GL[j]}" for j in range(5)],
        y=[f"After T{i+1}" for i in range(5)],
        colorscale=colorscale,
        zmid=0,
        zmin=-vmax, zmax=vmax,
        text=text,
        texttemplate="%{text}",
        textfont=dict(family=FONT_MONO, size=10, color=AC["text_primary"]),
        showscale=True,
        colorbar=dict(
            title=dict(text="Score", font=dict(family=FONT_UI, size=11)),
            thickness=12, len=0.8,
            tickfont=dict(family=FONT_MONO, size=10),
        ),
        xgap=2, ygap=2,
    ))

    W = W_SINGLE_COL + 60
    H = W
    fig.update_layout(
        title=dict(text=title, font=dict(family=FONT_TITLE, size=13)),
        paper_bgcolor=AC["bg"], plot_bgcolor=AC["bg"],
        margin=dict(l=72, r=80, t=52, b=56),
        font=dict(family=FONT_UI, size=11, color=AC["text_primary"]),
        xaxis=dict(
            showline=False, showgrid=False, ticks="",
            tickfont=dict(family=FONT_UI, size=9, color=AC["text_muted"]),
        ),
        yaxis=dict(
            showline=False, showgrid=False, ticks="", autorange="reversed",
            tickfont=dict(family=FONT_UI, size=9, color=AC["text_muted"]),
        ),
    )
    return fig, W, H


def fig_forgetting_matrices() -> None:
    """Three forgetting-matrix heatmaps: V5-canonical, V5-reversed, CLEAR-A."""
    configs = [
        (M_v5,    GAMES,     "V5 canonical (Ours)",    "v5_can"),
        (M_v5r_raw, GAMES_REV, "V5 reversed order (Ours)", "v5_rev"),
        (M_clearA, GAMES,    "CLEAR-A standard buffer", "clearA"),
    ]
    variants = []
    for M, game_order, title, tag in configs:
        fig, W, H = _make_heatmap(M, game_order, title, tag)
        stem = FIGURES / f"forgetting_matrix_{tag}"
        v = export_figure(fig, stem, W, H, label=title)
        variants.append(v)
        _verify(stem)

    append_manifest_entry(
        MANIFEST,
        id="forgetting_matrices",
        group="method_comparison",
        group_title="Method Comparison",
        title="Forgetting matrices: per-method evaluation over task sequence",
        caption="Rows = after task i; cols = game j. Blue = above zero, red = below zero (forgetting). Single seed.",
        details=(
            "Each cell M[i,j] is the greedy-100 score on game j measured after finishing task i. "
            "Diagonal = score right after learning each game. "
            "V5 canonical order: Qbert, Pong, Breakout, Boxing, SpaceInvaders. "
            "V5 reversed order: SpaceInvaders, Boxing, Breakout, Pong, Qbert. "
            "CLEAR-A uses standard replay buffer (batches=4, canonical order). "
            "Blue=positive score, red=negative (below zero). Diverging scale centred at 0."
        ),
        variants=variants,
        seed=0, dataset="Atari-5", model="impala_ac_multihead",
    )
    print("  [ok] forgetting_matrices")


# ─────────────────────────────────────────────────────────────────────────────
# GROUP 2: retention_forgetting
# Figure 2a: AP / Forgetting / BWT (raw + normalized) grouped bars
# ─────────────────────────────────────────────────────────────────────────────
def fig_cl_metrics_bars() -> None:
    """Grouped bars: Average Performance, Forgetting, BWT — raw and normalized."""
    # Compute for each method
    methods = {
        "Ours (V5)":   (M_v5,    GAMES),
        "CLEAR-A":     (M_clearA, GAMES),
        "CLEAR-B":     (M_clearB, GAMES),
        "CLEAR-C":     (M_clearC, GAMES),
    }

    raw_metrics = {}
    norm_metrics = {}
    for mname, (M, gord) in methods.items():
        raw_metrics[mname]  = cl_metrics_from_matrix(M)
        norm_metrics[mname] = cl_metrics_norm(M, gord)

    # Joint as reference (no forgetting by definition)
    joint_ap = float(np.mean([JOINT_6M_SCORES[g] for g in GAMES]))
    joint_ap_norm = float(np.mean([normalize(JOINT_6M_SCORES[g], g) for g in GAMES]))

    metric_keys   = ["avg_performance", "forgetting", "bwt"]
    metric_labels = ["Avg Performance", "Forgetting", "BWT"]

    method_labels = list(methods.keys())
    colors = AC_SERIES[:len(method_labels)]

    # --- RAW variant ---
    fig_raw = make_subplots(rows=1, cols=3,
                            subplot_titles=metric_labels,
                            horizontal_spacing=0.08)
    for m_i, method in enumerate(method_labels):
        for c_i, mk in enumerate(metric_keys):
            val = raw_metrics[method][mk]
            fig_raw.add_trace(go.Bar(
                x=[method], y=[val],
                marker_color=colors[m_i],
                name=method, showlegend=(c_i == 0),
                width=0.55,
                text=[f"{val:.0f}"], textposition="outside",
                textfont=dict(family=FONT_MONO, size=9),
                cliponaxis=False,
            ), row=1, col=c_i + 1)

    # Joint AP reference line (col 1 only)
    fig_raw.add_hline(y=joint_ap, line_width=1.2, line_color=AC["green"],
                      line_dash="dot", row=1, col=1, annotation_text="Joint 6M",
                      annotation_font=dict(size=10, color=AC["green"]),
                      annotation_position="top right")
    fig_raw.add_hline(y=0, line_width=0.8, line_color=AC["axis"], row=1, col=2)
    fig_raw.add_hline(y=0, line_width=0.8, line_color=AC["axis"], row=1, col=3)

    W = W_FULL; H = height_for(W, 2.2)
    _style_cl_bar_fig(fig_raw, "CL metrics — raw scores")

    # Callout: CLEAR-A has Breakout forgetting = peak - final = big number
    fig_raw.add_annotation(
        x="CLEAR-A", y=raw_metrics["CLEAR-A"]["forgetting"],
        xref="x2", yref="y2",
        text="CLEAR-A: Breakout<br>fully forgotten",
        showarrow=True, arrowhead=0, arrowwidth=1.0, arrowcolor=AC["axis"],
        ax=40, ay=-30, xanchor="left",
        font=dict(family=FONT_UI, size=10, color=AC["red"]),
        bgcolor="rgba(255,255,255,0.85)", borderpad=3,
    )

    stem_raw = FIGURES / "cl_metrics_raw"
    v_raw = export_figure(fig_raw, stem_raw, W, H, label="Raw scores")
    _verify(stem_raw)

    # --- NORMALIZED variant ---
    fig_norm = make_subplots(rows=1, cols=3,
                             subplot_titles=[f"{l} (norm.)" for l in metric_labels],
                             horizontal_spacing=0.08)
    for m_i, method in enumerate(method_labels):
        for c_i, mk in enumerate(metric_keys):
            val = norm_metrics[method][mk]
            fig_norm.add_trace(go.Bar(
                x=[method], y=[val],
                marker_color=colors[m_i],
                name=method, showlegend=(c_i == 0),
                width=0.55,
                text=[f"{val:.2f}"], textposition="outside",
                textfont=dict(family=FONT_MONO, size=9),
                cliponaxis=False,
            ), row=1, col=c_i + 1)

    fig_norm.add_hline(y=joint_ap_norm, line_width=1.2, line_color=AC["green"],
                       line_dash="dot", row=1, col=1, annotation_text="Joint 6M",
                       annotation_font=dict(size=10, color=AC["green"]),
                       annotation_position="top right")
    fig_norm.add_hline(y=0, line_width=0.8, line_color=AC["axis"], row=1, col=2)
    fig_norm.add_hline(y=0, line_width=0.8, line_color=AC["axis"], row=1, col=3)

    _style_cl_bar_fig(fig_norm, "CL metrics — normalized (raw−random)/(threshold−random)")
    stem_norm = FIGURES / "cl_metrics_norm"
    v_norm = export_figure(fig_norm, stem_norm, W, H, label="Normalized")
    _verify(stem_norm)

    append_manifest_entry(
        MANIFEST,
        id="cl_metrics_bars",
        group="retention_forgetting",
        group_title="Retention and Forgetting",
        title="CL metrics: Avg Performance, Forgetting, BWT across methods",
        caption="Raw and normalized CL metrics. Normalized=(score−random)/(threshold−random). Joint 6M ceiling shown. Single seed.",
        details=(
            "Average Performance (AP): mean final-row score over all games. "
            "Forgetting: mean over earlier games of (peak − final score). "
            "Backward Transfer (BWT): mean over earlier games of (final − just-learned). "
            "Normalized scores use RANDOM_SCORES from crl.envs.atari and per-game thresholds "
            "from config. Joint 6M AP line = order-independent ceiling. "
            "Positive BWT = method improved old tasks during later training."
        ),
        variants=[v_raw, v_norm],
        seed=0, dataset="Atari-5", model="impala_ac_multihead",
        metrics=[
            {"name": "V5 AP (raw)",      "value": round(raw_metrics["Ours (V5)"]["avg_performance"], 1),  "unit": "score"},
            {"name": "V5 Forgetting",    "value": round(raw_metrics["Ours (V5)"]["forgetting"], 1),        "unit": "score"},
            {"name": "CLEAR-A Forgetting","value": round(raw_metrics["CLEAR-A"]["forgetting"], 1),          "unit": "score"},
            {"name": "V5 AP (norm)",     "value": round(norm_metrics["Ours (V5)"]["avg_performance"], 3),  "unit": ""},
        ],
    )
    print("  [ok] cl_metrics_bars")


def _style_cl_bar_fig(fig: go.Figure, title: str) -> None:
    fig.update_layout(
        title=dict(text=title, font=dict(family=FONT_TITLE, size=14)),
        paper_bgcolor=AC["bg"], plot_bgcolor=AC["bg"],
        margin=dict(l=56, r=24, t=68, b=72),
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5,
            font=dict(family=FONT_UI, size=11), bgcolor="rgba(0,0,0,0)",
        ),
        barmode="group",
        font=dict(family=FONT_UI, size=11, color=AC["text_primary"]),
    )
    for i in range(1, 4):
        fig.update_xaxes(showline=True, linecolor=AC["axis"], linewidth=1.0,
                         showgrid=False, ticks="", showticklabels=False, row=1, col=i)
        fig.update_yaxes(showline=True, linecolor=AC["axis"], linewidth=1.0,
                         showgrid=True, gridcolor=AC["grid"], gridwidth=0.6,
                         tickfont=dict(family=FONT_UI, size=10, color=AC["text_muted"]),
                         zeroline=False, row=1, col=i)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2b: Retention vs local and vs joint — per game
# ─────────────────────────────────────────────────────────────────────────────
def fig_retention_per_game() -> None:
    """Retention per game: final÷local and final÷joint, for V5 and CLEAR-A."""
    joint = {g: JOINT_6M_SCORES[g] for g in GAMES}

    # V5 canonical
    ret_local_v5  = {g: V5_FINAL[g] / V5_LOCAL[g]  if V5_LOCAL[g]  != 0 else 0 for g in GAMES}
    ret_joint_v5  = {g: V5_FINAL[g] / joint[g]      if joint[g]      != 0 else 0 for g in GAMES}

    # CLEAR-A (no local greedy stored per CLEAR config — only task1 Qbert)
    # CLEAR-A has only one resource_usage entry. For fair comparison use joint as denominator.
    ret_joint_cA  = {g: CLEARA_FINAL[g] / joint[g]  if joint[g] != 0 else 0 for g in GAMES}

    x = GAME_LABELS
    fig = go.Figure()

    bar_configs = [
        (ret_local_v5,  "V5 ÷ local",  AC_SERIES[0], -0.22),
        (ret_joint_v5,  "V5 ÷ joint",  AC_SERIES[4], -0.07),
        (ret_joint_cA,  "CLEAR-A ÷ joint", AC_SERIES[1],  0.08),
    ]
    for data, name, color, offset in bar_configs:
        vals = [data[g] for g in GAMES]
        fig.add_trace(go.Bar(
            name=name,
            x=x,
            y=vals,
            marker_color=color,
            width=0.22,
            offset=offset,
            text=[f"{v:.2f}" for v in vals],
            textposition="outside",
            textfont=dict(family=FONT_MONO, size=9),
            cliponaxis=False,
        ))

    fig.add_hline(y=1.0, line_width=1.0, line_color=AC["axis"], line_dash="dot",
                  annotation_text="1.0 (100%)", annotation_position="right",
                  annotation_font=dict(size=10))
    fig.add_hline(y=0.0, line_width=0.8, line_color=AC["axis"])

    # Callout: Boxing retention < 0 (V5 forgot entirely)
    fig.add_annotation(
        x="Boxing", y=ret_local_v5["Boxing"],
        text="V5 Boxing<br>retention < 0",
        showarrow=True, arrowhead=0, arrowwidth=1.0, arrowcolor=AC["axis"],
        ax=-50, ay=-35, xanchor="right",
        font=dict(family=FONT_UI, size=10, color=AC["red"]),
        bgcolor="rgba(255,255,255,0.85)", borderpad=3,
    )

    W = W_FULL; H = height_for(W, 2.0)
    fig.update_layout(
        title=dict(text="Per-game retention: final ÷ reference score",
                   font=dict(family=FONT_TITLE, size=15)),
        paper_bgcolor=AC["bg"], plot_bgcolor=AC["bg"],
        margin=dict(l=64, r=24, t=60, b=72),
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5,
            font=dict(family=FONT_UI, size=11), bgcolor="rgba(0,0,0,0)",
        ),
        barmode="overlay",
        xaxis=dict(showline=True, linecolor=AC["axis"], showgrid=False,
                   tickfont=dict(family=FONT_UI, size=11, color=AC["text_muted"])),
        yaxis=dict(showline=True, linecolor=AC["axis"],
                   showgrid=True, gridcolor=AC["grid"], gridwidth=0.6,
                   title=dict(text="Retention ratio", font=dict(family=FONT_UI, size=13)),
                   tickfont=dict(family=FONT_UI, size=11, color=AC["text_muted"]),
                   zeroline=False),
        font=dict(family=FONT_UI, size=11, color=AC["text_primary"]),
    )

    stem = FIGURES / "retention_per_game"
    v = export_figure(fig, stem, W, H, label="Retention ratios")
    _verify(stem)

    append_manifest_entry(
        MANIFEST,
        id="retention_per_game",
        group="retention_forgetting",
        group_title="Retention and Forgetting",
        title="Per-game retention relative to local and joint reference",
        caption="Retention = final score ÷ reference. V5 Boxing < 0 (forgotten). Joint 6M is order-independent ceiling. Single seed.",
        details=(
            "Retention ratio = final greedy-100 score ÷ reference score. "
            "'÷ local' uses the single-task local training score from resource_usage.json "
            "(order-dependent). '÷ joint' uses Joint 6M final scores (order-independent). "
            "Ratio > 1 means consolidation improved on the reference (rare). "
            "Ratio < 0 means the method scored below zero (e.g. V5 Boxing = -25.8). "
            "CLEAR-A local scores not stored, so only joint retention shown for CLEAR."
        ),
        variants=[v],
        seed=0, dataset="Atari-5", model="impala_ac_multihead",
        metrics=[
            {"name": "V5 Boxing ret (vs local)", "value": round(ret_local_v5["Boxing"], 3), "unit": ""},
            {"name": "V5 Breakout ret (vs joint)", "value": round(ret_joint_v5["Breakout"], 3), "unit": ""},
        ],
    )
    print("  [ok] retention_per_game")


# ─────────────────────────────────────────────────────────────────────────────
# GROUP 3: clear_buffer_sweep
# Figure 3: CLEAR buffer sweep — final scores + retention vs buffer size
# ─────────────────────────────────────────────────────────────────────────────
def fig_clear_buffer_sweep() -> None:
    """CLEAR buffer sweep: CLEAR-A/B/C final scores and normalized AP vs buffer."""
    buffer_sizes = [4, 16, 48]  # snapshot_batches from config
    labels = ["CLEAR-A (4)", "CLEAR-B (16)", "CLEAR-C (48)"]
    colors = [AC_SERIES[1], AC_SERIES[4], AC_SERIES[2]]

    matrices = [M_clearA, M_clearB, M_clearC]
    finals = [
        dict(zip(GAMES, M.shape and [float(M[4, j]) for j in range(5)]))
        for M in matrices
    ]

    # Normalized AP per method
    norm_aps = [
        float(np.mean([normalize(finals[i][g], g) for g in GAMES]))
        for i in range(3)
    ]

    # --- Per-game scores (small multiples) ---
    W = W_FULL; H = int(W * 0.35)
    fig_scores = make_subplots(rows=1, cols=5, subplot_titles=GAME_LABELS,
                               shared_yaxes=False, horizontal_spacing=0.055)
    for col_i, game in enumerate(GAMES):
        for m_i, label in enumerate(labels):
            score = finals[m_i][game]
            fig_scores.add_trace(go.Bar(
                x=[label], y=[score],
                marker_color=colors[m_i],
                name=label, showlegend=(col_i == 0),
                width=0.55,
                text=[f"{score:.0f}"], textposition="outside",
                textfont=dict(family=FONT_MONO, size=9),
                cliponaxis=False,
            ), row=1, col=col_i + 1)
        fig_scores.add_hline(y=0, line_width=0.8, line_color=AC["axis"],
                             line_dash="solid", row=1, col=col_i + 1)
        fig_scores.add_hline(y=THRESHOLDS[game], line_width=0.8, line_color=AC["amber"],
                             line_dash="dot", row=1, col=col_i + 1)

    # Callout: Breakout is 0 for all CLEAR variants
    fig_scores.add_annotation(
        x="CLEAR-A (4)", y=0,
        xref="x3", yref="y3",
        text="All CLEAR variants<br>forget Breakout",
        showarrow=True, arrowhead=0, arrowwidth=1.0, arrowcolor=AC["axis"],
        ax=0, ay=-50, xanchor="center",
        font=dict(family=FONT_UI, size=10, color=AC["red"]),
        bgcolor="rgba(255,255,255,0.85)", borderpad=3,
    )
    fig_scores.update_layout(
        title=dict(text="CLEAR buffer sweep — per-game final scores",
                   font=dict(family=FONT_TITLE, size=14)),
        paper_bgcolor=AC["bg"], plot_bgcolor=AC["bg"],
        margin=dict(l=48, r=24, t=68, b=64),
        showlegend=True, barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5,
                    font=dict(family=FONT_UI, size=11), bgcolor="rgba(0,0,0,0)"),
        font=dict(family=FONT_UI, size=11, color=AC["text_primary"]),
    )
    for i in range(1, 6):
        fig_scores.update_xaxes(showline=True, linecolor=AC["axis"], showgrid=False,
                                ticks="", showticklabels=False, row=1, col=i)
        fig_scores.update_yaxes(showline=True, linecolor=AC["axis"],
                                showgrid=True, gridcolor=AC["grid"], gridwidth=0.6,
                                tickfont=dict(family=FONT_UI, size=10, color=AC["text_muted"]),
                                zeroline=False, row=1, col=i)

    stem_sc = FIGURES / "clear_sweep_scores"
    v_sc = export_figure(fig_scores, stem_sc, W, H, label="Per-game scores")
    _verify(stem_sc)

    # --- Normalized AP vs buffer size (line + scatter) ---
    fig_ap = go.Figure()
    fig_ap.add_trace(go.Scatter(
        x=buffer_sizes, y=norm_aps,
        mode="lines+markers",
        line=dict(color=AC_SERIES[1], width=1.8),
        marker=dict(size=9, color=AC_SERIES[1]),
        name="Norm. AP",
        showlegend=False,
    ))
    add_end_labels(fig_ap, [("Norm. AP", buffer_sizes[-1], norm_aps[-1], AC_SERIES[1])])

    W2 = W_ONE_HALF; H2 = height_for(W2)
    fig_ap.update_layout(
        title=dict(text="CLEAR buffer sweep — normalized average performance",
                   font=dict(family=FONT_TITLE, size=14)),
        paper_bgcolor=AC["bg"], plot_bgcolor=AC["bg"],
        margin=dict(l=64, r=120, t=52, b=56),
        xaxis=dict(
            title=dict(text="Snapshot batches (buffer size)", font=dict(family=FONT_UI, size=13)),
            tickvals=buffer_sizes, ticktext=[str(b) for b in buffer_sizes],
            showline=True, linecolor=AC["axis"], showgrid=False,
            tickfont=dict(family=FONT_UI, size=11, color=AC["text_muted"]),
        ),
        yaxis=dict(
            title=dict(text="Normalized AP (random=0, threshold=1)", font=dict(family=FONT_UI, size=13)),
            showline=True, linecolor=AC["axis"],
            showgrid=True, gridcolor=AC["grid"], gridwidth=0.6,
            tickfont=dict(family=FONT_UI, size=11, color=AC["text_muted"]),
            zeroline=False,
        ),
        font=dict(family=FONT_UI, size=11, color=AC["text_primary"]),
        showlegend=False,
    )

    # Callout: best is not necessarily biggest buffer
    best_i = int(np.argmax(norm_aps))
    add_callout(fig_ap, buffer_sizes[best_i], norm_aps[best_i],
                f"Best: {labels[best_i]}", ax=28, ay=-30)

    stem_ap = FIGURES / "clear_sweep_norm_ap"
    v_ap = export_figure(fig_ap, stem_ap, W2, H2, label="Normalized AP vs buffer")
    _verify(stem_ap)

    append_manifest_entry(
        MANIFEST,
        id="clear_buffer_sweep",
        group="clear_buffer_sweep",
        group_title="CLEAR Buffer Sweep",
        title="CLEAR buffer sweep: A (4 batches), B (16), C (48) — scores and normalized AP",
        caption="Breakout is forgotten (0.0) by all three CLEAR buffer sizes. Normalized AP varies only slightly with buffer. Single seed.",
        details=(
            "CLEAR replay buffer sweep. Three configurations: "
            "A = 4 snapshot batches (standard/equal), "
            "B = 16 batches (generous), C = 48 batches (very generous). "
            "All use all-past rehearsal at rate replay_task_per_step=8, same frame budget (~30k iters). "
            "Canonical game order. "
            "Normalized AP = mean over games of (score−random)/(threshold−random). "
            "Breakout forgotten by all variants (0.0) — order-sensitive catastrophic forgetting not cured by replay alone."
        ),
        variants=[v_sc, v_ap],
        seed=0, dataset="Atari-5", model="impala_ac_multihead",
        metrics=[
            {"name": "CLEAR-A norm AP", "value": round(norm_aps[0], 3), "unit": ""},
            {"name": "CLEAR-B norm AP", "value": round(norm_aps[1], 3), "unit": ""},
            {"name": "CLEAR-C norm AP", "value": round(norm_aps[2], 3), "unit": ""},
        ],
    )
    print("  [ok] clear_buffer_sweep")


# ─────────────────────────────────────────────────────────────────────────────
# GROUP 4: order_sensitivity
# Figure 4: V5 canonical vs reversed — per-game retention vs joint ceiling
# ─────────────────────────────────────────────────────────────────────────────
def fig_order_sensitivity() -> None:
    """V5 canonical vs reversed: per-game retention vs fixed Joint-6M ceiling."""
    joint = {g: JOINT_6M_SCORES[g] for g in GAMES}

    ret_can = {g: V5_FINAL[g]  / joint[g] if joint[g] != 0 else 0 for g in GAMES}
    ret_rev = {g: V5R_FINAL[g] / joint[g] if joint[g] != 0 else 0 for g in GAMES}

    mean_can = float(np.mean(list(ret_can.values())))
    mean_rev = float(np.mean(list(ret_rev.values())))

    x = GAME_LABELS + ["Mean"]
    y_can = [ret_can[g] for g in GAMES] + [mean_can]
    y_rev = [ret_rev[g] for g in GAMES] + [mean_rev]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="V5 canonical order", x=x, y=y_can,
        marker_color=AC_SERIES[0], width=0.33, offset=-0.18,
        text=[f"{v:.2f}" for v in y_can], textposition="outside",
        textfont=dict(family=FONT_MONO, size=9), cliponaxis=False,
    ))
    fig.add_trace(go.Bar(
        name="V5 reversed order", x=x, y=y_rev,
        marker_color=AC_SERIES[3], width=0.33, offset=0.14,
        text=[f"{v:.2f}" for v in y_rev], textposition="outside",
        textfont=dict(family=FONT_MONO, size=9), cliponaxis=False,
    ))
    fig.add_hline(y=1.0, line_width=1.0, line_color=AC["green"], line_dash="dot",
                  annotation_text="Joint 6M = 1.0", annotation_position="right",
                  annotation_font=dict(size=10, color=AC["green"]))
    fig.add_hline(y=0.0, line_width=0.8, line_color=AC["axis"])

    # Callout: reversed order recovers Breakout substantially
    fig.add_annotation(
        x="Breakout", y=ret_rev["Breakout"],
        text=f"Reversed order: {ret_rev['Breakout']:.2f}<br>vs canonical: {ret_can['Breakout']:.2f}",
        showarrow=True, arrowhead=0, arrowwidth=1.0, arrowcolor=AC["axis"],
        ax=60, ay=-40, xanchor="left",
        font=dict(family=FONT_UI, size=10, color=AC["text_primary"]),
        bgcolor="rgba(255,255,255,0.85)", borderpad=3,
    )

    W = W_FULL; H = height_for(W, 2.0)
    fig.update_layout(
        title=dict(text="Order sensitivity: V5 canonical vs reversed — retention vs Joint 6M",
                   font=dict(family=FONT_TITLE, size=15)),
        paper_bgcolor=AC["bg"], plot_bgcolor=AC["bg"],
        margin=dict(l=64, r=24, t=60, b=80),
        showlegend=True, barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5,
                    font=dict(family=FONT_UI, size=11), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showline=True, linecolor=AC["axis"], showgrid=False,
                   tickfont=dict(family=FONT_UI, size=11, color=AC["text_muted"])),
        yaxis=dict(showline=True, linecolor=AC["axis"],
                   showgrid=True, gridcolor=AC["grid"], gridwidth=0.6,
                   title=dict(text="Retention vs Joint 6M", font=dict(family=FONT_UI, size=13)),
                   tickfont=dict(family=FONT_UI, size=11, color=AC["text_muted"]),
                   zeroline=False),
        font=dict(family=FONT_UI, size=11, color=AC["text_primary"]),
    )

    stem = FIGURES / "order_sensitivity"
    v = export_figure(fig, stem, W, H, label="Retention vs joint")
    _verify(stem)

    append_manifest_entry(
        MANIFEST,
        id="order_sensitivity",
        group="order_sensitivity",
        group_title="Order Sensitivity",
        title="Order sensitivity: per-game retention vs joint ceiling for V5 canonical vs reversed",
        caption="Retention = final ÷ Joint-6M score. Reversed order substantially recovers Breakout. Mean shown as rightmost bar. Single seed.",
        details=(
            "Joint 6M score is the order-independent ceiling (joint training, 6M frames/game). "
            "Canonical order: Qbert, Pong, Breakout, Boxing, SpaceInvaders. "
            "Reversed order: SpaceInvaders, Boxing, Breakout, Pong, Qbert. "
            "Local (single-task) references are order-dependent and therefore not used here as the "
            "common denominator; Joint 6M provides a clean cross-order reference. "
            "Reversed-order Breakout retention = 199.8/285.4 ≈ 0.70 vs canonical 51.8/285.4 ≈ 0.18."
        ),
        variants=[v],
        seed=0, dataset="Atari-5", model="impala_ac_multihead",
        metrics=[
            {"name": "Canonical mean ret (vs joint)", "value": round(mean_can, 3), "unit": ""},
            {"name": "Reversed mean ret (vs joint)",  "value": round(mean_rev, 3), "unit": ""},
            {"name": "Breakout canon ret",  "value": round(ret_can["Breakout"], 3), "unit": ""},
            {"name": "Breakout rev ret",    "value": round(ret_rev["Breakout"], 3), "unit": ""},
        ],
    )
    print("  [ok] order_sensitivity")


# ─────────────────────────────────────────────────────────────────────────────
# GROUP 5: training_dynamics
# Figure 5a: Joint 6M learning curves (joint_probe scores over frames)
# Figure 5b: Joint stopped probe trace
# Figure 5c: V5 eval probe trace (task-level scores at evaluation points)
# ─────────────────────────────────────────────────────────────────────────────
def fig_joint_learning_curves() -> None:
    """Joint 6M: per-game score vs frames (joint_probe rows)."""
    logs = load_logs("atari5_joint_6m_seed0")
    probes = [l for l in logs if l.get("phase") == "joint_probe"]

    frames_M = [p["frames"] / 1e6 for p in probes]  # millions
    game_keys = ["atari-Qbert", "atari-Pong", "atari-Breakout",
                 "atari-Boxing", "atari-SpaceInvaders"]

    W = W_FULL; H = height_for(W, 1.6)
    fig = go.Figure()

    for g_i, (gkey, glabel) in enumerate(zip(game_keys, GAME_LABELS)):
        game_idx = probes[0]["games"].index(gkey)
        scores = [p["scores"][game_idx] for p in probes]
        color = AC_SERIES[g_i]
        fig.add_trace(go.Scatter(
            x=frames_M, y=scores,
            mode="lines+markers",
            line=dict(color=color, width=1.8),
            marker=dict(size=5, color=color),
            name=glabel, showlegend=False,
        ))

    # End labels
    last_f = frames_M[-1]
    end_labels = []
    for g_i, (gkey, glabel) in enumerate(zip(game_keys, GAME_LABELS)):
        game_idx = probes[0]["games"].index(gkey)
        last_score = probes[-1]["scores"][game_idx]
        end_labels.append((glabel, last_f, last_score, AC_SERIES[g_i]))
    span = max(s for _, _, s, _ in end_labels) - min(s for _, _, s, _ in end_labels)
    add_end_labels(fig, end_labels, min_gap=span * 0.06)

    # Callout: all games converge
    fig.add_annotation(
        x=frames_M[-1], y=probes[-1]["scores"][0],  # Qbert
        text=f"Qbert final: {probes[-1]['scores'][0]:.0f}",
        showarrow=True, arrowhead=0, arrowwidth=1.0, arrowcolor=AC["axis"],
        ax=-60, ay=30, xanchor="right",
        font=dict(family=FONT_UI, size=10, color=AC_SERIES[0]),
        bgcolor="rgba(255,255,255,0.85)", borderpad=3,
    )

    fig.update_layout(
        title=dict(text="Joint 6M training: per-game score vs frames (single seed)",
                   font=dict(family=FONT_TITLE, size=15)),
        paper_bgcolor=AC["bg"], plot_bgcolor=AC["bg"],
        margin=dict(l=64, r=120, t=60, b=56),
        xaxis=dict(
            title=dict(text="Env frames (millions)", font=dict(family=FONT_UI, size=13)),
            showline=True, linecolor=AC["axis"], showgrid=False,
            tickfont=dict(family=FONT_UI, size=11, color=AC["text_muted"]),
        ),
        yaxis=dict(
            title=dict(text="Greedy-100 score", font=dict(family=FONT_UI, size=13)),
            showline=True, linecolor=AC["axis"],
            showgrid=True, gridcolor=AC["grid"], gridwidth=0.6,
            tickfont=dict(family=FONT_UI, size=11, color=AC["text_muted"]),
            zeroline=False,
        ),
        font=dict(family=FONT_UI, size=11, color=AC["text_primary"]),
        showlegend=False,
    )

    stem = FIGURES / "joint_6m_learning_curves"
    v = export_figure(fig, stem, W, H, label="Joint 6M curves")
    _verify(stem)

    append_manifest_entry(
        MANIFEST,
        id="joint_6m_learning_curves",
        group="training_dynamics",
        group_title="Training Dynamics",
        title="Joint 6M: per-game learning curves over environment frames",
        caption="Joint training ceiling (6M frames/game). All five games trained simultaneously. Single seed; no CI band.",
        details=(
            "Joint model trains all five games simultaneously with a shared IMPALA-AC-multihead network. "
            "Probe points (joint_probe) at every 500 iters = 5.12M frames. "
            "6M frames/game ≈ 30M total. "
            "This is the order-independent upper bound for comparison. "
            "Single seed; no CI band is drawn."
        ),
        variants=[v],
        seed=0, dataset="Atari-5", model="impala_ac_multihead (joint)",
        metrics=[{
            "name": f"{g} final", "value": round(JOINT_6M_SCORES[g], 1), "unit": "score"
        } for g in GAMES],
    )
    print("  [ok] joint_6m_learning_curves")


def fig_joint_stopped_curves() -> None:
    """Joint 59.8M run (stopped): probe trace + final label."""
    logs = load_logs("atari5_joint_seed0")
    probes = [l for l in logs if l.get("phase") == "joint_probe"]

    frames_M = [p["frames"] / 1e6 for p in probes]
    game_keys = ["atari-Qbert", "atari-Pong", "atari-Breakout",
                 "atari-Boxing", "atari-SpaceInvaders"]

    W = W_FULL; H = height_for(W, 1.6)
    fig = go.Figure()

    for g_i, (gkey, glabel) in enumerate(zip(game_keys, GAME_LABELS)):
        game_idx = probes[0]["games"].index(gkey)
        scores = [p["scores"][game_idx] for p in probes]
        color = AC_SERIES[g_i]
        fig.add_trace(go.Scatter(
            x=frames_M, y=scores,
            mode="lines+markers",
            line=dict(color=color, width=1.8),
            marker=dict(size=5, color=color),
            name=glabel, showlegend=False,
        ))

    # End labels
    last_f = frames_M[-1]
    end_labels = []
    for g_i, (gkey, glabel) in enumerate(zip(game_keys, GAME_LABELS)):
        game_idx = probes[0]["games"].index(gkey)
        last_score = probes[-1]["scores"][game_idx]
        end_labels.append((glabel, last_f, last_score, AC_SERIES[g_i]))
    span = max(s for _, _, s, _ in end_labels) - min(s for _, _, s, _ in end_labels)
    add_end_labels(fig, end_labels, min_gap=span * 0.06)

    # Mark stop point
    fig.add_vline(x=last_f, line_width=1.2, line_color=AC["red"], line_dash="dash")
    fig.add_annotation(
        x=last_f, y=probes[-1]["scores"][4],  # SI — highest at end
        text="Stopped ~46M frames<br>(incomplete run)",
        showarrow=True, arrowhead=0, arrowwidth=1.0, arrowcolor=AC["red"],
        ax=-70, ay=-30, xanchor="right",
        font=dict(family=FONT_UI, size=10, color=AC["red"]),
        bgcolor="rgba(255,255,255,0.85)", borderpad=3,
    )

    fig.update_layout(
        title=dict(text="Joint 59.8M budget (stopped at ~46M frames) — single seed",
                   font=dict(family=FONT_TITLE, size=15)),
        paper_bgcolor=AC["bg"], plot_bgcolor=AC["bg"],
        margin=dict(l=64, r=120, t=60, b=56),
        xaxis=dict(
            title=dict(text="Env frames (millions)", font=dict(family=FONT_UI, size=13)),
            showline=True, linecolor=AC["axis"], showgrid=False,
            tickfont=dict(family=FONT_UI, size=11, color=AC["text_muted"]),
        ),
        yaxis=dict(
            title=dict(text="Greedy-100 score", font=dict(family=FONT_UI, size=13)),
            showline=True, linecolor=AC["axis"],
            showgrid=True, gridcolor=AC["grid"], gridwidth=0.6,
            tickfont=dict(family=FONT_UI, size=11, color=AC["text_muted"]),
            zeroline=False,
        ),
        font=dict(family=FONT_UI, size=11, color=AC["text_primary"]),
        showlegend=False,
    )

    stem = FIGURES / "joint_stopped_curves"
    v = export_figure(fig, stem, W, H, label="Joint stopped")
    _verify(stem)

    append_manifest_entry(
        MANIFEST,
        id="joint_stopped_curves",
        group="training_dynamics",
        group_title="Training Dynamics",
        title="Joint 59.8M run (stopped at ~46M frames): per-game probe scores",
        caption="Run stopped early at ~46M frames (step 4500); no joint_result.json. Last probe used. Single seed; no CI.",
        details=(
            "Joint model with 59.8M frame budget, but run preempted at step 4500 (~46M frames). "
            "No joint_result.json exists; last joint_probe row at frames=46,080,000 is used as final. "
            "SpaceInvaders reaches 1375 at stop, highest across all runs. "
            "Dashed red line marks the stop point. "
            "Single seed; no CI band."
        ),
        variants=[v],
        seed=0, dataset="Atari-5", model="impala_ac_multihead (joint, incomplete)",
        metrics=[
            {"name": "SI at stop",     "value": round(joint_stopped_scores_raw[4], 1), "unit": "score"},
            {"name": "Qbert at stop",  "value": round(joint_stopped_scores_raw[0], 1), "unit": "score"},
            {"name": "Frames at stop", "value": 46.08, "unit": "M"},
        ],
    )
    print("  [ok] joint_stopped_curves")


def fig_v5_eval_probes() -> None:
    """V5 canonical: per-task evaluation scores at end-of-task eval points."""
    # The 'eval' phase rows give the forgetting matrix rows at task boundaries
    logs = load_logs("atari5_v5_seed0")
    eval_rows = [l for l in logs if l.get("phase") == "eval"]

    # Reconstruct which game is which
    # eval task=4 has 4 values (games 1-4), eval task=5 has 5 values (games 1-5)
    task_labels = [f"After T{r['task']}" for r in eval_rows]
    n_tasks = len(eval_rows)

    W = W_FULL; H = height_for(W, 1.8)
    fig = go.Figure()

    for g_i, (game, glabel) in enumerate(zip(GAMES, GAME_LABELS)):
        # Only plot where game j was trained (j <= i-1 in 0-indexed)
        x_pts = []
        y_pts = []
        for r_i, row in enumerate(eval_rows):
            task_idx = row["task"]  # 1-indexed; game j is available from task j+1
            vals = row["values"]
            if g_i < len(vals):
                x_pts.append(row["task"])
                y_pts.append(vals[g_i])

        color = AC_SERIES[g_i]
        fig.add_trace(go.Scatter(
            x=x_pts, y=y_pts,
            mode="lines+markers",
            line=dict(color=color, width=1.8),
            marker=dict(size=7, color=color),
            name=glabel, showlegend=False,
        ))

    # End labels at last available task
    end_lbls = []
    for g_i, (game, glabel) in enumerate(zip(GAMES, GAME_LABELS)):
        y_last = eval_rows[-1]["values"][g_i]
        end_lbls.append((glabel, eval_rows[-1]["task"], y_last, AC_SERIES[g_i]))
    span = max(s for _, _, s, _ in end_lbls) - min(s for _, _, s, _ in end_lbls)
    add_end_labels(fig, end_lbls, min_gap=span * 0.06)

    # Zero line
    fig.add_hline(y=0, line_width=0.8, line_color=AC["axis"])

    # Callout: Boxing drops to negative after T5
    fig.add_annotation(
        x=5, y=eval_rows[-1]["values"][3],  # Boxing at T5
        text=f"Boxing: {eval_rows[-1]['values'][3]:.1f}<br>(forgotten after consolidation)",
        showarrow=True, arrowhead=0, arrowwidth=1.0, arrowcolor=AC["axis"],
        ax=-50, ay=-35, xanchor="right",
        font=dict(family=FONT_UI, size=10, color=AC["red"]),
        bgcolor="rgba(255,255,255,0.85)", borderpad=3,
    )

    fig.update_layout(
        title=dict(text="V5 canonical: per-game scores at end-of-task evaluation points",
                   font=dict(family=FONT_TITLE, size=15)),
        paper_bgcolor=AC["bg"], plot_bgcolor=AC["bg"],
        margin=dict(l=64, r=120, t=60, b=56),
        xaxis=dict(
            title=dict(text="After task", font=dict(family=FONT_UI, size=13)),
            tickvals=list(range(1, n_tasks + 1)),
            ticktext=[f"T{i}" for i in range(1, n_tasks + 1)],
            showline=True, linecolor=AC["axis"], showgrid=False,
            tickfont=dict(family=FONT_UI, size=11, color=AC["text_muted"]),
        ),
        yaxis=dict(
            title=dict(text="Greedy-100 score", font=dict(family=FONT_UI, size=13)),
            showline=True, linecolor=AC["axis"],
            showgrid=True, gridcolor=AC["grid"], gridwidth=0.6,
            tickfont=dict(family=FONT_UI, size=11, color=AC["text_muted"]),
            zeroline=False,
        ),
        font=dict(family=FONT_UI, size=11, color=AC["text_primary"]),
        showlegend=False,
    )

    stem = FIGURES / "v5_eval_probes"
    v = export_figure(fig, stem, W, H, label="V5 eval probes")
    _verify(stem)

    append_manifest_entry(
        MANIFEST,
        id="v5_eval_probes",
        group="training_dynamics",
        group_title="Training Dynamics",
        title="V5 canonical: per-game scores at end-of-each-task evaluation",
        caption="Scores on each game measured after completing task T1–T5. Boxing drops to −25.8 by T5 (forgotten). Single seed.",
        details=(
            "V5 canonical (min-max consolidation), canonical order: Qbert, Pong, Breakout, Boxing, SpaceInvaders. "
            "Each point is the greedy-100 score on game j measured right after finishing task i (from 'eval' phase rows in logs.jsonl). "
            "A game appears only from the task it was learned. "
            "Boxing was at +97 after T4 but dropped to -25.8 after T5 consolidation. "
            "Single seed; no CI band."
        ),
        variants=[v],
        seed=0, dataset="Atari-5", model="impala_ac_multihead",
        metrics=[
            {"name": "Boxing after T4", "value": round(eval_rows[0]["values"][3], 1), "unit": "score"},
            {"name": "Boxing after T5", "value": round(eval_rows[1]["values"][3], 1), "unit": "score"},
        ],
    )
    print("  [ok] v5_eval_probes")


# ─────────────────────────────────────────────────────────────────────────────
# GROUP 6: compute_cost
# Figure 6: Wall-time per method
# ─────────────────────────────────────────────────────────────────────────────
def fig_compute_cost() -> None:
    """Wall-time comparison: V5 vs CLEAR-A/B/C. Note V5 re-collects all past envs each iter."""
    # V5 canonical: sum all local + global wall_s
    res_v5 = load_resource("atari5_v5_seed0")
    res_v5r = load_resource("atari5_v5_order2_seed0")

    def sum_wall(res: dict) -> float:
        total = 0.0
        for game_data in res.values():
            for phase in ["task1", "local", "global"]:
                if phase in game_data:
                    total += game_data[phase].get("wall_s", 0.0)
        return total

    wall_v5  = sum_wall(res_v5)
    wall_v5r = sum_wall(res_v5r)

    # CLEAR: from task1 wall_s for their only stored entry
    # CLEAR-A resource_usage only has Qbert (the task1 run)
    # CLEAR also runs global per task — get from logs t_wall
    def clear_total_wall(run: str) -> float:
        logs = load_logs(run)
        if logs:
            return logs[-1]["t_wall"]
        return 0.0

    wall_cA = clear_total_wall("atari5_v5_clearA_equal_seed0")
    wall_cB = clear_total_wall("atari5_v5_clearB_gen_seed0")
    wall_cC = clear_total_wall("atari5_v5_clearC_vgen_seed0")
    wall_joint6m = load_logs("atari5_joint_6m_seed0")[-1]["t_wall"]

    methods = ["Ours V5\n(canon)", "Ours V5\n(rev)", "CLEAR-A", "CLEAR-B", "CLEAR-C", "Joint 6M"]
    walls_h  = [w / 3600 for w in [wall_v5, wall_v5r, wall_cA, wall_cB, wall_cC, wall_joint6m]]
    colors   = [AC_SERIES[0], AC_SERIES[0], AC_SERIES[1], AC_SERIES[4], AC_SERIES[2], AC_SERIES[2]]
    opacities = [1.0, 0.65, 1.0, 0.85, 0.7, 0.55]

    W = W_FULL; H = height_for(W, 2.2)
    fig = go.Figure()
    for i, (m, w, c) in enumerate(zip(methods, walls_h, colors)):
        fig.add_trace(go.Bar(
            x=[m], y=[w],
            marker_color=c,
            marker_opacity=opacities[i],
            name=m.replace("\n", " "),
            showlegend=False,
            text=[f"{w:.1f}h"], textposition="outside",
            textfont=dict(family=FONT_MONO, size=10),
            cliponaxis=False,
        ))

    # Callout: V5 is much longer (re-collects all past envs each iter)
    fig.add_annotation(
        x="Ours V5\n(canon)", y=walls_h[0],
        text=f"V5: {walls_h[0]:.1f}h<br>(global re-collects all past envs)",
        showarrow=True, arrowhead=0, arrowwidth=1.0, arrowcolor=AC["axis"],
        ax=60, ay=-30, xanchor="left",
        font=dict(family=FONT_UI, size=10, color=AC_SERIES[0]),
        bgcolor="rgba(255,255,255,0.85)", borderpad=3,
    )

    fig.update_layout(
        title=dict(text="Wall-clock time per method (single seed, single GPU)",
                   font=dict(family=FONT_TITLE, size=15)),
        paper_bgcolor=AC["bg"], plot_bgcolor=AC["bg"],
        margin=dict(l=64, r=24, t=60, b=72),
        showlegend=False,
        xaxis=dict(showline=True, linecolor=AC["axis"], showgrid=False,
                   tickfont=dict(family=FONT_UI, size=11, color=AC["text_muted"])),
        yaxis=dict(showline=True, linecolor=AC["axis"],
                   showgrid=True, gridcolor=AC["grid"], gridwidth=0.6,
                   title=dict(text="Wall-clock time (hours)", font=dict(family=FONT_UI, size=13)),
                   tickfont=dict(family=FONT_UI, size=11, color=AC["text_muted"]),
                   zeroline=False),
        font=dict(family=FONT_UI, size=11, color=AC["text_primary"]),
    )

    stem = FIGURES / "compute_cost_wall"
    v = export_figure(fig, stem, W, H, label="Wall-clock hours")
    _verify(stem)

    append_manifest_entry(
        MANIFEST,
        id="compute_cost_wall",
        group="compute_cost",
        group_title="Compute Cost",
        title="Wall-clock time per method: Ours (V5) vs CLEAR vs Joint",
        caption="V5 is substantially slower than CLEAR (global phase re-collects all past envs each iter = O(k)). Single GPU, single seed.",
        details=(
            "Wall-clock time in hours, single GPU, seed 0. "
            "V5 canonical: sum of task1 + local + global wall_s from resource_usage.json. "
            "V5 reversed: same. "
            "CLEAR: t_wall from last row of logs.jsonl (total elapsed). "
            "Joint 6M: t_wall from joint_final row. "
            "V5 global consolidation re-collects all past environments each iteration (O(k) cost), "
            "explaining the much higher wall-time vs CLEAR replay."
        ),
        variants=[v],
        seed=0, dataset="Atari-5", model="impala_ac_multihead",
        metrics=[
            {"name": "V5 canonical", "value": round(walls_h[0], 1), "unit": "h"},
            {"name": "CLEAR-A",      "value": round(walls_h[2], 1), "unit": "h"},
            {"name": "Joint 6M",     "value": round(walls_h[5], 1), "unit": "h"},
        ],
    )
    print("  [ok] compute_cost_wall")


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
def _verify(stem: Path) -> None:
    """Assert PNG and SVG exist and are non-empty."""
    for suffix in (".png", ".svg"):
        p = stem.with_suffix(suffix)
        if not p.exists():
            raise RuntimeError(f"Export missing: {p}")
        if p.stat().st_size == 0:
            raise RuntimeError(f"Export is empty: {p}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating figures…")
    print("Group: method_comparison")
    fig_method_comparison_scores()
    fig_forgetting_matrices()

    print("Group: retention_forgetting")
    fig_cl_metrics_bars()
    fig_retention_per_game()

    print("Group: clear_buffer_sweep")
    fig_clear_buffer_sweep()

    print("Group: order_sensitivity")
    fig_order_sensitivity()

    print("Group: training_dynamics")
    fig_joint_learning_curves()
    fig_joint_stopped_curves()
    fig_v5_eval_probes()

    print("Group: compute_cost")
    fig_compute_cost()

    print("\nAll figures generated successfully.")
