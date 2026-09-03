"""Build the Atari headline figure: min-max consolidation against CLEAR.

Left: the 5-task normalized forgetting matrices, redrawn in the academic palette
(blue = ours, amber = CLEAR) instead of a red-green colormap. Right: what each
game is worth the moment it is learned and what is left of it after the whole
sequence, which is where the two methods differ most. Bottom: the published
continual-learning metrics.

Source. The full-budget v5 runs live on the cluster, not in this repo, so the two
mean matrices below are transcribed from the generated comparison figure at
`reports/atari5_ppo_v4/figures/clear_comparison/` (2 seeds). Recomputing the CL
metrics from them reproduces the published values to within seed-averaging order
(AvgPerf 0.45 / 0.73, Forgetting 0.42 / 0.45, BWT -0.39 / -0.45), which is the
check that the transcription is faithful.
"""

from __future__ import annotations

from pathlib import Path

OUT_SVG = Path(__file__).with_name("atari_comparison.svg")

# ── Academic palette ────────────────────────────────────────────────────────
AC = {
    "blue": "#2563EB",
    "amber": "#D97706",
    "bg": "#FFFFFF",
    "surface": "#F8F9FA",
    "border": "#DEE2E6",
    "axis": "#495057",
    "grid": "#E9ECEF",
    "text": "#212529",
    "muted": "#6C757D",
    "faint": "#ADB5BD",
    "highlight": "#EFF6FF",
    "empty": "#F1F3F5",
}
FONT_UI = "Inter, -apple-system, BlinkMacSystemFont, Helvetica, Arial, sans-serif"
FONT_TITLE = "'Source Serif 4', 'Source Serif Pro', Georgia, serif"
FONT_NUM = "'JetBrains Mono', 'SF Mono', Menlo, Consolas, monospace"

W, H = 1560, 820

GAMES = ["Pong", "Breakout", "Boxing", "Qbert", "SpaceInvaders"]
NAN = None

# Normalized score (random = 0, task threshold = 1) of every game, evaluated
# after each task of the sequence. Row i = state after task i+1.
MATRIX = {
    "ours": [
        [0.85, NAN, NAN, NAN, NAN],
        [0.97, 0.69, NAN, NAN, NAN],
        [0.95, 0.44, 0.59, NAN, NAN],
        [0.80, 0.36, -0.02, 0.96, NAN],
        [0.87, 0.38, 0.05, 0.22, 0.71],
    ],
    "clear": [
        [1.01, NAN, NAN, NAN, NAN],
        [0.78, 0.96, NAN, NAN, NAN],
        [0.27, 0.56, 0.91, NAN, NAN],
        [0.20, 0.68, 0.89, 1.78, NAN],
        [0.61, 0.43, 0.88, 0.93, 0.81],
    ],
}

# Published metrics (mean over the 2 seeds, from cl_metrics_3way).
METRICS = {
    "avg_performance": (0.45, 0.73),
    "forgetting": (0.44, 0.45),
}
# CLEAR stores clear_snapshot_batches (2) x n_envs (8) x n_steps (128) frames per
# task as cloning targets; four past tasks are held while the fifth trains.
STORED_FRAMES_CLEAR = 2 * 8 * 128 * 4

INK_MAX = 1.1  # score at which a cell reaches full colour saturation


# ── SVG primitives ──────────────────────────────────────────────────────────

def esc(text: str) -> str:
    """Escape the XML-significant characters that appear in labels."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def label(
    x: float,
    y: float,
    text: str,
    size: float = 13,
    fill: str = AC["text"],
    weight: int = 400,
    anchor: str = "start",
    font: str = FONT_UI,
    spacing: str = "0",
    extra: str = "",
) -> str:
    """One line of SVG text with the project's typographic defaults."""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{font}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" '
        f'letter-spacing="{spacing}"{extra}>{esc(text)}</text>'
    )


def draw_matrix(x0: float, y0: float, pitch: float, cell: float,
                matrix: list[list[float | None]], color: str) -> list[str]:
    """One lower-triangular forgetting matrix as tiles inked by score."""
    out: list[str] = []
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            x, y = x0 + j * pitch, y0 + i * pitch
            if value is None:
                out.append(
                    f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell:.1f}" height="{cell:.1f}" '
                    f'rx="4" fill="{AC["empty"]}"/>'
                )
                continue

            ink = max(0.05, min(1.0, value / INK_MAX))
            out.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell:.1f}" height="{cell:.1f}" '
                f'rx="4" fill="{color}" fill-opacity="{ink:.3f}"/>'
            )
            if i == j:  # the score the moment that game finished training
                out.append(
                    f'<rect x="{x + 1:.1f}" y="{y + 1:.1f}" width="{cell - 2:.1f}" '
                    f'height="{cell - 2:.1f}" rx="3.5" fill="none" stroke="{color}" '
                    f'stroke-width="2.2"/>'
                )
            out.append(label(
                x + cell / 2, y + cell / 2 + 4.5, f"{value:.2f}", 13,
                AC["bg"] if ink > 0.62 else AC["text"], 600 if i == j else 400,
                "middle", FONT_NUM,
            ))
    return out


def draw_drops(x0: float, y0: float, width: float, height: float) -> list[str]:
    """Per game, the fall from its score at learning to its score at the end.

    Two glyphs per game, ours beside CLEAR: a hollow marker at the score the game
    reached when it was trained, a solid marker at what survived the rest of the
    sequence, and the connector between them. Glyph length is the forgetting.
    """
    out: list[str] = []
    y_max = 1.9
    slot = width / len(GAMES)

    def py(score: float) -> float:
        return y0 + height - height * score / y_max

    for tick in (0.0, 0.5, 1.0, 1.5):
        y = py(tick)
        dashed = ' stroke-dasharray="5 5"' if tick == 1.0 else ""
        stroke = AC["faint"] if tick == 1.0 else AC["grid"]
        out.append(
            f'<line x1="{x0:.1f}" y1="{y:.1f}" x2="{x0 + width:.1f}" y2="{y:.1f}" '
            f'stroke="{stroke}" stroke-width="1"{dashed}/>'
        )
        out.append(label(x0 - 10, y + 4, f"{tick:.1f}", 12, AC["muted"], 400, "end",
                         FONT_NUM))
    out.append(label(x0 + width, py(1.0) - 8, "threshold", 11.5, AC["muted"], 500, "end"))

    out.append(
        f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x0:.1f}" y2="{y0 + height:.1f}" '
        f'stroke="{AC["axis"]}" stroke-width="1.2"/>'
    )
    out.append(
        f'<line x1="{x0:.1f}" y1="{y0 + height:.1f}" x2="{x0 + width:.1f}" '
        f'y2="{y0 + height:.1f}" stroke="{AC["axis"]}" stroke-width="1.2"/>'
    )

    for g, game in enumerate(GAMES):
        center = x0 + (g + 0.5) * slot
        out.append(label(center, y0 + height + 22, game, 12, AC["text"], 500, "middle"))

        for k, (key, color) in enumerate((("ours", AC["blue"]), ("clear", AC["amber"]))):
            x = center + (k - 0.5) * 22
            start, end = MATRIX[key][g][g], MATRIX[key][4][g]
            out.append(
                f'<line x1="{x:.1f}" y1="{py(start):.1f}" x2="{x:.1f}" y2="{py(end):.1f}" '
                f'stroke="{color}" stroke-width="4" stroke-opacity="0.35" '
                f'stroke-linecap="round"/>'
            )
            out.append(
                f'<circle cx="{x:.1f}" cy="{py(start):.1f}" r="6" fill="{AC["bg"]}" '
                f'stroke="{color}" stroke-width="2.2"/>'
            )
            out.append(f'<circle cx="{x:.1f}" cy="{py(end):.1f}" r="6.5" fill="{color}"/>')

    # The final game is never trained over, so it has no fall to show.
    out.append(label(x0 + width - slot / 2, y0 + height + 38, "trained last",
                     10.5, AC["faint"], 400, "middle"))
    return out


def stat_tile(x: float, y: float, width: float, height: float, caption: str,
              ours: str, base: str, note: str) -> list[str]:
    """A single headline comparison tile."""
    return [
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" rx="10" '
        f'fill="{AC["surface"]}" stroke="{AC["border"]}" stroke-width="1"/>',
        label(x + 18, y + 26, caption.upper(), 10.5, AC["muted"], 600, "start",
              FONT_UI, "0.09em"),
        label(x + 18, y + 66, ours, 30, AC["blue"], 400, "start", FONT_NUM),
        label(x + 18, y + 92, f"vs {base} CLEAR", 12, AC["amber"], 500),
        label(x + 18, y + 112, note, 11, AC["faint"], 400),
    ]


def build() -> str:
    """Assemble the complete figure."""
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
        f'height="{H}" role="img" aria-label="Atari continual learning comparison">',
        "<title>Atari 5-task continual RL: min-max consolidation vs CLEAR</title>",
        f'<rect x="0" y="0" width="{W}" height="{H}" fill="{AC["bg"]}"/>',
    ]

    parts.append(label(64, 62, "CLEAR's retention, without CLEAR's replay buffer",
                       31, AC["text"], 600, "start", FONT_TITLE))
    parts.append(label(64, 92,
                       "5 Atari games in sequence  ·  normalized score, random = 0, "
                       "threshold = 1  ·  2 seeds",
                       14, AC["muted"]))
    parts.append(f'<line x1="64" y1="116" x2="{W - 64}" y2="116" '
                 f'stroke="{AC["border"]}" stroke-width="1"/>')

    # ── Matrices ────────────────────────────────────────────────────────────
    pitch, cell = 60.0, 56.0
    mat_y = 214.0
    mat_a_x, mat_b_x = 154.0, 520.0

    parts.append(label(mat_a_x, 178, "Min-max (ours)", 17, AC["blue"], 600))
    parts.append(label(mat_b_x, 178, "CLEAR (Rolnick '19)", 17, AC["amber"], 600))

    # Shared row labels: both matrices use the same sequence positions.
    for i in range(5):
        parts.append(label(mat_a_x - 14, mat_y + i * pitch + cell / 2 + 4.5,
                           f"after T{i + 1}", 12, AC["muted"], 400, "end"))

    parts += draw_matrix(mat_a_x, mat_y, pitch, cell, MATRIX["ours"], AC["blue"])
    parts += draw_matrix(mat_b_x, mat_y, pitch, cell, MATRIX["clear"], AC["amber"])

    for x0 in (mat_a_x, mat_b_x):
        for j, game in enumerate(GAMES):
            cx = x0 + j * pitch + cell / 2
            cy = mat_y + 5 * pitch + 18
            parts.append(
                f'<g transform="translate({cx:.1f},{cy:.1f}) rotate(-32)">'
                + label(0, 0, game, 11.5, AC["muted"], 400, "end")
                + "</g>"
            )

    parts.append(label(mat_a_x, mat_y + 5 * pitch + 86,
                       "Read down a column. Outlined cell = the moment that game was learned.",
                       12, AC["faint"]))

    # ── Drop panel ──────────────────────────────────────────────────────────
    parts.append(label(940, 178, "How far each game falls", 17, AC["text"], 600))
    parts.append(
        f'<circle cx="1200" cy="173" r="5.5" fill="{AC["bg"]}" '
        f'stroke="{AC["muted"]}" stroke-width="2"/>'
    )
    parts.append(label(1212, 178, "at learning", 12, AC["muted"]))
    parts.append(f'<circle cx="1320" cy="173" r="6" fill="{AC["muted"]}"/>')
    parts.append(label(1332, 178, "after all 5", 12, AC["muted"]))

    parts += draw_drops(990.0, 214.0, 480.0, 300.0)

    # ── Bottom band ─────────────────────────────────────────────────────────
    band_y = 620.0
    parts.append(
        f'<line x1="64" y1="{band_y:.1f}" x2="{W - 64}" y2="{band_y:.1f}" '
        f'stroke="{AC["border"]}" stroke-width="1"/>'
    )

    tile_w, tile_h, gap = 196.0, 132.0, 16.0
    tile_x, tile_y = 64.0, band_y + 30
    ours_f, clear_f = METRICS["forgetting"]
    ours_p, clear_p = METRICS["avg_performance"]
    parts += stat_tile(tile_x, tile_y, tile_w, tile_h, "Forgetting  ↓",
                       f"{ours_f:.2f}", f"{clear_f:.2f}", "same within noise, n = 2")
    parts += stat_tile(tile_x + tile_w + gap, tile_y, tile_w, tile_h, "Stored frames",
                       "0", f"{STORED_FRAMES_CLEAR:,}", "replay-free")
    parts += stat_tile(tile_x + 2 * (tile_w + gap), tile_y, tile_w, tile_h,
                       "Avg performance  ↑", f"{ours_p:.2f}", f"{clear_p:.2f}",
                       "CLEAR still ahead")

    callout_x = tile_x + 3 * (tile_w + gap)
    callout_w = W - 64 - callout_x
    parts.append(
        f'<rect x="{callout_x:.1f}" y="{tile_y:.1f}" width="{callout_w:.1f}" '
        f'height="{tile_h:.1f}" rx="10" fill="{AC["highlight"]}"/>'
    )
    parts.append(
        f'<rect x="{callout_x:.1f}" y="{tile_y:.1f}" width="4" height="{tile_h:.1f}" '
        f'rx="2" fill="{AC["blue"]}"/>'
    )
    parts.append(label(callout_x + 24, tile_y + 46,
                       "A constraint retains as well as a replay buffer.",
                       17, AC["text"], 600))
    parts.append(label(callout_x + 24, tile_y + 78,
                       "What is left to close is raw score, not memory.",
                       14, AC["muted"]))

    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    OUT_SVG.write_text(build(), encoding="utf-8")
    print(f"wrote {OUT_SVG} ({OUT_SVG.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
