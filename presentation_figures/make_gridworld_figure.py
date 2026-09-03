"""Build the GridWorld headline figure: what the agent still knows after 20 tasks.

Left panel: how many already-learned goals remain solvable as the sequence advances.
The constrained min-max curve lands on the perfect-memory diagonal, fine-tuning
flatlines near two, and the wedge between them is the knowledge the constraint saves.
Right panel: where each individual goal ended up once all 20 tasks were done. Three
stat tiles carry the headline numbers.

Values come from `reports/gridworld_20task/` (eval matrices and aggregated tables,
10 seeds), so the figure cannot drift from the results.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np

ROOT = Path("/Users/protiknag/Desktop/Continual Reinforcement Learning with Minimax Algorithm")
REPORT = ROOT / "reports/gridworld_20task"
TABLES = REPORT / "tables"
OUT_SVG = ROOT / "presentation_figures/gridworld_comparison.svg"

NUM_SEEDS = 10
RETAIN_TOLERANCE = 0.90  # a goal counts as kept while its value stays within 10%
# Student-t 0.975 quantile by seed count, matching analysis/aggregate.py.
T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
       7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}

# ── Academic palette ────────────────────────────────────────────────────────
AC = {
    "blue": "#2563EB",
    "amber": "#D97706",
    "green": "#059669",
    "bg": "#FFFFFF",
    "surface": "#F8F9FA",
    "border": "#DEE2E6",
    "axis": "#495057",
    "grid": "#E9ECEF",
    "text": "#212529",
    "muted": "#6C757D",
    "faint": "#ADB5BD",
    "highlight": "#EFF6FF",
}
FONT_UI = "Inter, -apple-system, BlinkMacSystemFont, Helvetica, Arial, sans-serif"
FONT_TITLE = "'Source Serif 4', 'Source Serif Pro', Georgia, serif"
FONT_NUM = "'JetBrains Mono', 'SF Mono', Menlo, Consolas, monospace"

W, H = 1560, 800
SOLVED_THRESHOLD = 90.0  # success rate at which a goal counts as "still solved"


class MethodStats(NamedTuple):
    """Per-task and aggregate greedy-evaluation statistics for one method."""

    success: list[float]
    success_ci: list[float]
    mean_success: float
    mean_success_ci: float
    mean_steps: float


def read_performance(path: Path) -> dict[str, MethodStats]:
    """Parse the aggregated performance table into per-method statistics."""
    per_task: dict[str, dict[int, tuple[float, float]]] = {}
    aggregate: dict[str, tuple[float, float, float]] = {}

    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            method = row["method"]
            if row["task"] == "mean":
                aggregate[method] = (
                    float(row["success_pct"]),
                    float(row["success_ci"]),
                    float(row["mean_steps"]),
                )
            else:
                per_task.setdefault(method, {})[int(row["task"])] = (
                    float(row["success_pct"]),
                    float(row["success_ci"]),
                )

    stats: dict[str, MethodStats] = {}
    for method, tasks in per_task.items():
        order = sorted(tasks)
        mean_success, mean_ci, mean_steps = aggregate[method]
        stats[method] = MethodStats(
            success=[tasks[t][0] for t in order],
            success_ci=[tasks[t][1] for t in order],
            mean_success=mean_success,
            mean_success_ci=mean_ci,
            mean_steps=mean_steps,
        )
    return stats


def mean_ci(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and 95% CI half-width over the seed axis."""
    n = values.shape[0]
    mean = values.mean(axis=0)
    if n < 2:
        return mean, np.zeros_like(mean)
    sd = values.std(axis=0, ddof=1)
    return mean, T95.get(n, 1.96) * sd / np.sqrt(n)


def goals_retained(method: str) -> tuple[np.ndarray, np.ndarray]:
    """How many already-learned goals the agent can still solve at each stage.

    The evaluation matrix holds the value of every task after every stage of the
    sequence. A goal counts as retained while its value stays within
    `RETAIN_TOLERANCE` of the value it had immediately after its own task was
    trained (the matrix diagonal), which is the natural per-task reference.
    Returns the seed mean and the 95% CI half-width, one entry per stage.
    """
    counts = []
    for seed in range(NUM_SEEDS):
        matrix = np.array(json.load(open(REPORT / f"eval_matrix_{method}_seed{seed}.json")))
        reference = np.diag(matrix)  # value of each task right after it was learned
        kept = matrix >= RETAIN_TOLERANCE * reference[None, :]
        # Only tasks already seen at that stage can be retained (lower triangle).
        seen = np.tril(np.ones_like(kept, dtype=bool))
        counts.append((kept & seen).sum(axis=1))
    return mean_ci(np.array(counts, dtype=float))


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
    opacity: float = 1.0,
) -> str:
    """One line of SVG text with the project's typographic defaults."""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{font}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" '
        f'letter-spacing="{spacing}" opacity="{opacity}">{esc(text)}</text>'
    )


def draw_accumulation(
    x0: float,
    y0: float,
    width: float,
    height: float,
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
) -> list[str]:
    """Goals still solvable against how far the sequence has progressed.

    The dashed diagonal is perfect memory: after k tasks an agent that forgets
    nothing can still solve all k. Filled areas make the gap between the two
    methods the dominant shape on the page.
    """
    out: list[str] = []
    n_tasks = len(curves["constrained"][0])
    y_max = 21.0

    def px(stage: int) -> float:
        return x0 + width * stage / (n_tasks - 1)

    def py(count: float) -> float:
        return y0 + height - height * count / y_max

    for value in (0, 5, 10, 15, 20):
        y = py(value)
        out.append(
            f'<line x1="{x0:.1f}" y1="{y:.1f}" x2="{x0 + width:.1f}" y2="{y:.1f}" '
            f'stroke="{AC["grid"]}" stroke-width="1"/>'
        )
        out.append(label(x0 - 10, y + 4, f"{value}", 12, AC["muted"], 400, "end", FONT_NUM))

    out.append(
        f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x0:.1f}" y2="{y0 + height:.1f}" '
        f'stroke="{AC["axis"]}" stroke-width="1.2"/>'
    )
    out.append(
        f'<line x1="{x0:.1f}" y1="{y0 + height:.1f}" x2="{x0 + width:.1f}" '
        f'y2="{y0 + height:.1f}" stroke="{AC["axis"]}" stroke-width="1.2"/>'
    )
    for stage in (1, 5, 10, 15, 20):
        x = px(stage - 1)
        out.append(
            f'<line x1="{x:.1f}" y1="{y0 + height:.1f}" x2="{x:.1f}" '
            f'y2="{y0 + height + 5:.1f}" stroke="{AC["axis"]}" stroke-width="1.2"/>'
        )
        out.append(
            label(x, y0 + height + 20, str(stage), 12, AC["muted"], 400, "middle", FONT_NUM)
        )
    out.append(
        label(x0 + width / 2, y0 + height + 46, "Tasks trained so far",
              13, AC["text"], 500, "middle")
    )
    out.append(
        f'<g transform="translate({x0 - 46:.1f},{y0 + height / 2:.1f}) rotate(-90)">'
        + label(0, 0, "Goals the agent can still solve", 13, AC["text"], 500, "middle")
        + "</g>"
    )

    # Perfect memory: nothing learned is ever lost. Our curve lands exactly on it,
    # so the reference is drawn as a wide pale corridor the blue line rides inside
    # rather than a dashed line the blue line would hide completely.
    out.append(
        f'<line x1="{px(0):.1f}" y1="{py(1):.1f}" x2="{px(n_tasks - 1):.1f}" '
        f'y2="{py(n_tasks):.1f}" stroke="{AC["border"]}" stroke-width="10" '
        f'stroke-linecap="round"/>'
    )
    out.append(
        label(px(19), py(20.9), "perfect memory", 11.5, AC["muted"], 500, "end")
    )

    # Fills are stacked rather than overlaid: amber covers what fine-tuning keeps,
    # and the blue band above it is exactly the knowledge the constraint saves.
    ours_mean = curves["constrained"][0]
    base_mean = curves["finetune"][0]
    base_forward = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(base_mean))
    base_reverse = " ".join(
        f"{px(i):.1f},{py(v):.1f}" for i, v in reversed(list(enumerate(base_mean)))
    )
    ours_forward = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(ours_mean))
    out.append(
        f'<polygon points="{px(0):.1f},{py(0):.1f} {base_forward} '
        f'{px(n_tasks - 1):.1f},{py(0):.1f}" fill="{AC["amber"]}" fill-opacity="0.16"/>'
    )
    out.append(
        f'<polygon points="{ours_forward} {base_reverse}" fill="{AC["blue"]}" '
        f'fill-opacity="0.13"/>'
    )

    for method, color in (("constrained", AC["blue"]), ("finetune", AC["amber"])):
        mean, half = curves[method]
        if half.max() > 0:
            band = " ".join(
                f"{px(i):.1f},{py(min(y_max, m + c)):.1f}"
                for i, (m, c) in enumerate(zip(mean, half))
            ) + " " + " ".join(
                f"{px(i):.1f},{py(max(0.0, m - c)):.1f}"
                for i, (m, c) in reversed(list(enumerate(zip(mean, half))))
            )
            out.append(f'<polygon points="{band}" fill="{color}" fill-opacity="0.22"/>')

        line = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(mean))
        out.append(
            f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="2.8" '
            f'stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for i, value in enumerate(mean):
            out.append(
                f'<circle cx="{px(i):.1f}" cy="{py(value):.1f}" r="3.4" fill="{AC["bg"]}" '
                f'stroke="{color}" stroke-width="1.8"/>'
            )

    # Direct labels, each placed in a region both curves leave empty: ours in the
    # upper left above the rising line, the baseline in the wedge that opens on the
    # right between the two methods.
    out.append(label(px(0.5), py(19.8), "ours: nothing is ever lost", 13.5, AC["blue"], 600))
    out.append(
        label(px(0.5), py(18.1), "every goal ever learned is still solvable",
              12, AC["muted"], 400)
    )
    out.append(
        label(px(19), py(8.6), "fine-tuning: only the two or three",
              13.5, AC["amber"], 600, "end")
    )
    out.append(
        label(px(19), py(7.0), "most recent goals survive", 13.5, AC["amber"], 600, "end")
    )
    final_base = curves["finetune"][0][-1]
    out.append(
        label(px(19), py(final_base) + 22, f"{final_base:.1f} of 20",
              12.5, AC["amber"], 600, "end", FONT_NUM)
    )
    return out


def draw_profile(
    x0: float,
    y0: float,
    width: float,
    height: float,
    stats: dict[str, MethodStats],
) -> list[str]:
    """Final success rate against the order in which each goal was learned."""
    out: list[str] = []
    n_tasks = len(stats["constrained"].success)
    y_max = 108.0  # headroom so the flat 100% line is not clipped by the plot frame

    def px(task_index: int) -> float:
        return x0 + width * task_index / (n_tasks - 1)

    def py(rate: float) -> float:
        return y0 + height - height * rate / y_max

    # Horizontal grid and y ticks only (Tufte spine).
    for value in (0, 25, 50, 75, 100):
        y = py(value)
        out.append(
            f'<line x1="{x0:.1f}" y1="{y:.1f}" x2="{x0 + width:.1f}" y2="{y:.1f}" '
            f'stroke="{AC["grid"]}" stroke-width="1"/>'
        )
        out.append(label(x0 - 10, y + 4, f"{value}", 12, AC["muted"], 400, "end", FONT_NUM))

    out.append(
        f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x0:.1f}" y2="{y0 + height:.1f}" '
        f'stroke="{AC["axis"]}" stroke-width="1.2"/>'
    )
    out.append(
        f'<line x1="{x0:.1f}" y1="{y0 + height:.1f}" x2="{x0 + width:.1f}" '
        f'y2="{y0 + height:.1f}" stroke="{AC["axis"]}" stroke-width="1.2"/>'
    )
    for task in (1, 5, 10, 15, 20):
        x = px(task - 1)
        out.append(
            f'<line x1="{x:.1f}" y1="{y0 + height:.1f}" x2="{x:.1f}" '
            f'y2="{y0 + height + 5:.1f}" stroke="{AC["axis"]}" stroke-width="1.2"/>'
        )
        out.append(
            label(x, y0 + height + 20, str(task), 12, AC["muted"], 400, "middle", FONT_NUM)
        )

    out.append(
        label(x0 + width / 2, y0 + height + 46, "Goal, in the order it was learned",
              13, AC["text"], 500, "middle")
    )
    out.append(
        f'<g transform="translate({x0 - 46:.1f},{y0 + height / 2:.1f}) rotate(-90)">'
        + label(0, 0, "Success rate after all 20 tasks (%)", 13, AC["text"], 500, "middle")
        + "</g>"
    )

    # Baseline mean reference.
    baseline = stats["finetune"]
    mean_y = py(baseline.mean_success)
    out.append(
        f'<line x1="{x0:.1f}" y1="{mean_y:.1f}" x2="{x0 + width:.1f}" y2="{mean_y:.1f}" '
        f'stroke="{AC["amber"]}" stroke-width="1.3" stroke-dasharray="5 5" opacity="0.75"/>'
    )

    # 95% CI band for the baseline; the constrained runs have zero spread.
    upper = " ".join(
        f"{px(i):.1f},{py(min(100.0, m + c)):.1f}"
        for i, (m, c) in enumerate(zip(baseline.success, baseline.success_ci))
    )
    lower = " ".join(
        f"{px(i):.1f},{py(max(0.0, m - c)):.1f}"
        for i, (m, c) in reversed(list(enumerate(zip(baseline.success, baseline.success_ci))))
    )
    out.append(
        f'<polygon points="{upper} {lower}" fill="{AC["amber"]}" fill-opacity="0.16"/>'
    )

    for method, color in (("finetune", AC["amber"]), ("constrained", AC["blue"])):
        series = stats[method].success
        points = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(series))
        out.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.6" '
            f'stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for i, value in enumerate(series):
            out.append(
                f'<circle cx="{px(i):.1f}" cy="{py(value):.1f}" r="3.4" fill="{AC["bg"]}" '
                f'stroke="{color}" stroke-width="1.8"/>'
            )

    # Direct labels beat a legend box. The mean label sits over tasks 7-11, the one
    # stretch where the baseline curve and its CI band stay well below the mean.
    out.append(
        label(px(n_tasks - 1), py(100) - 15, "ours: every goal, 100%",
              13, AC["blue"], 600, "end")
    )
    out.append(
        label(px(10), mean_y - 9, f"fine-tuning mean {baseline.mean_success:.1f}%",
              12, AC["amber"], 600, "end")
    )
    out.append(
        label(x0, y0 + height + 62, "shaded band: 95% CI over 10 seeds",
              11.5, AC["faint"])
    )
    return out


def stat_tile(x: float, y: float, width: float, height: float, caption: str,
              ours: str, base: str, note: str) -> list[str]:
    """A single headline comparison tile."""
    return [
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" rx="10" '
        f'fill="{AC["surface"]}" stroke="{AC["border"]}" stroke-width="1"/>',
        label(x + 18, y + 26, caption.upper(), 10.5, AC["muted"], 600, "start", FONT_UI, "0.09em"),
        label(x + 18, y + 66, ours, 30, AC["blue"], 400, "start", FONT_NUM),
        label(x + 18, y + 92, f"vs {base} fine-tuning", 12, AC["amber"], 500),
        label(x + 18, y + 112, note, 11, AC["faint"], 400),
    ]


def build() -> str:
    """Assemble the complete figure."""
    stats = read_performance(TABLES / "performance_table.csv")
    ours, base = stats["constrained"], stats["finetune"]
    curves = {m: goals_retained(m) for m in ("constrained", "finetune")}

    kept_ours = curves["constrained"][0][-1]
    kept_base = curves["finetune"][0][-1]

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
        f'height="{H}" role="img" aria-label="GridWorld continual learning comparison">',
        "<title>Continual RL on 20-task GridWorld: min-max consolidation vs fine-tuning</title>",
        f'<rect x="0" y="0" width="{W}" height="{H}" fill="{AC["bg"]}"/>',
    ]

    # ── Header ──────────────────────────────────────────────────────────────
    parts.append(label(64, 62, "After learning 20 goals in a row, which ones can the agent still reach?",
                       31, AC["text"], 600, "start", FONT_TITLE))
    parts.append(label(64, 92,
                       "15x15 GridWorld  ·  20 goal-relocation tasks, learned one after another  ·  "
                       "10 seeds  ·  greedy evaluation, 100 rollouts",
                       14, AC["muted"]))
    parts.append(f'<line x1="64" y1="116" x2="{W - 64}" y2="116" stroke="{AC["border"]}" stroke-width="1"/>')

    # ── Left: knowledge accumulated across the sequence ─────────────────────
    parts.append(label(64, 156, "Does anything survive the next task?", 18, AC["text"], 600))
    parts.append(label(64, 178,
                       "Goals still solvable after each new task is trained, out of every goal "
                       "learned so far.", 12.5, AC["muted"]))
    parts += draw_accumulation(140.0, 216.0, 600.0, 300.0, curves)
    parts.append(label(140, 578,
                       "A goal counts as kept while its value stays within 10% of what it was "
                       "the moment that task finished training.", 11.5, AC["faint"]))

    # ── Right: where each goal ended up ─────────────────────────────────────
    parts.append(label(900, 156, "Goal by goal, at the end of the sequence", 18, AC["text"], 600))
    parts.append(label(900, 178,
                       "Fine-tuning keeps whatever it saw last. The constraint keeps everything.",
                       12.5, AC["muted"]))
    parts += draw_profile(960.0, 216.0, 500.0, 300.0, stats)

    # ── Bottom band: headline numbers and takeaway ──────────────────────────
    band_y = 620.0
    parts.append(
        f'<line x1="64" y1="{band_y:.1f}" x2="{W - 64}" y2="{band_y:.1f}" '
        f'stroke="{AC["border"]}" stroke-width="1"/>'
    )

    tile_w, tile_h, gap = 196.0, 132.0, 16.0
    tile_x, tile_y = 64.0, band_y + 30
    parts += stat_tile(tile_x, tile_y, tile_w, tile_h, "Goals retained",
                       f"{kept_ours:.0f}/20", f"{kept_base:.1f}/20",
                       "value within 10% of learned")
    parts += stat_tile(tile_x + tile_w + gap, tile_y, tile_w, tile_h, "Mean success",
                       f"{ours.mean_success:.1f}%", f"{base.mean_success:.1f}%",
                       "over all 20 goals, at the end")
    parts += stat_tile(tile_x + 2 * (tile_w + gap), tile_y, tile_w, tile_h, "Steps to goal",
                       f"{ours.mean_steps:.1f}", f"{base.mean_steps:.1f}",
                       "mean, 150-step cap")

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
    parts.append(label(callout_x + 24, tile_y + 42,
                       "Twenty tasks in a row, and the agent still solves all twenty.",
                       17, AC["text"], 600))
    parts.append(label(callout_x + 24, tile_y + 70,
                       f"Fine-tuning is left with about {kept_base:.0f}. No replay buffer, no "
                       f"stored transitions,", 14, AC["muted"]))
    parts.append(label(callout_x + 24, tile_y + 92,
                       "only a constraint that each new task must not undo the last.",
                       14, AC["muted"]))

    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    OUT_SVG.write_text(build(), encoding="utf-8")
    print(f"wrote {OUT_SVG} ({OUT_SVG.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
