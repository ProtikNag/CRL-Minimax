---
name: algorithmic-art-academic
description: >
  Publication-quality research figures and academic algorithmic art. Use this
  skill for ANY figure, plot, chart, or visual result destined for a paper,
  poster, talk, or results report (NeurIPS, ICML, ICLR, journals), and also for
  generative art, interactive figures, flow fields, and particle systems. Track F
  (research figures) renders with Plotly plus Kaleido and exports PNG and SVG at
  300 dpi, appends an entry to report/manifest.json, and hands off to the
  visualization-dashboard skill. Track A (generative art) renders with p5.js in a
  self-contained HTML artifact. Both tracks enforce a fixed academic palette,
  a clean typographic system, white backgrounds, and minimal chrome.
---

# Academic Figures and Algorithmic Art

## Step 0: Read This File Before Writing Any Code

Before generating any figure, script, or HTML, re-read this skill in full. The
color palette, typography rules, export contract, and manifest schema below are
**mandatory**. Do not substitute Anthropic brand colors, do not use dark themes
for figures, do not invent new fonts, do not skip the manifest entry.

---

## Track Selection

This skill covers two distinct products. Pick one before writing code.

| Track | Product | Renderer | Output |
|-------|---------|----------|--------|
| **F — Research figure** | A figure that encodes data and could appear in a paper | Plotly + Kaleido (matplotlib fallback) | `figure.png`, `figure.svg`, optional `figure.gif` + `figure.html`, plus a `report/manifest.json` entry |
| **A — Generative art** | An exploratory, seeded, interactive generative artifact | p5.js in a self-contained HTML file | `<movement>.html` + `<movement>.md` philosophy |

If the request mentions results, metrics, curves, ablations, benchmarks, seeds,
or "make a figure for the paper", it is **Track F**. If it mentions generative,
flow field, particle, emergent, seeded art, or "make something beautiful", it is
**Track A**. When the request is a results batch, Track F is followed by the
`visualization-dashboard` skill — always.

---

# Part 0 — Shared Design System

Both tracks use the same palette and typography. Nothing in this section is
negotiable.

## Color Palette

This palette supports up to 8 distinguishable categorical series, heatmaps,
diverging scales, and neutral UI chrome simultaneously. It is calibrated for
readability on white backgrounds under typical screen and projector conditions.

### Categorical / Series Colors

Use in this order for multi-series plots:

| Token       | Hex       | Role                              |
|-------------|-----------|-----------------------------------|
| `--ac-blue` | `#2563EB` | Primary series (e.g., "train")    |
| `--ac-amber`| `#D97706` | Secondary series (e.g., "test")   |
| `--ac-green`| `#059669` | Tertiary series (e.g., "random")  |
| `--ac-red`  | `#DC2626` | Highlight / error / adversarial   |
| `--ac-violet`|`#7C3AED` | 5th series                        |
| `--ac-teal` | `#0891B2` | 6th series                        |
| `--ac-rose` | `#BE185D` | 7th series                        |
| `--ac-sienna`|`#92400E` | 8th series / secondary highlight  |

### Neutral / Structural Colors

| Token              | Hex       | Role                          |
|--------------------|-----------|-------------------------------|
| `--ac-bg`          | `#FFFFFF` | Canvas / main background      |
| `--ac-surface`     | `#F8F9FA` | Sidebar / panel background    |
| `--ac-border`      | `#DEE2E6` | Panel borders, dividers       |
| `--ac-axis`        | `#495057` | Axis lines, tick marks        |
| `--ac-grid`        | `#E9ECEF` | Grid lines (use sparingly)    |
| `--ac-text-primary`| `#212529` | Primary labels, headings      |
| `--ac-text-muted`  | `#6C757D` | Axis tick labels, captions    |
| `--ac-text-faint`  | `#ADB5BD` | Placeholder, disabled states  |

### Semantic / Functional Colors

| Token              | Hex       | Role                          |
|--------------------|-----------|-------------------------------|
| `--ac-highlight`   | `#EFF6FF` | Selected region, hover tint   |
| `--ac-warn`        | `#FEF3C7` | Warning indicators            |
| `--ac-ok`          | `#D1FAE5` | Confirmation, success states  |

### CSS Variables Block (copy into every `<style>` tag)

```css
:root {
  /* Categorical series */
  --ac-blue:    #2563EB;
  --ac-amber:   #D97706;
  --ac-green:   #059669;
  --ac-red:     #DC2626;
  --ac-violet:  #7C3AED;
  --ac-teal:    #0891B2;
  --ac-rose:    #BE185D;
  --ac-sienna:  #92400E;

  /* Neutrals */
  --ac-bg:           #FFFFFF;
  --ac-surface:      #F8F9FA;
  --ac-border:       #DEE2E6;
  --ac-axis:         #495057;
  --ac-grid:         #E9ECEF;
  --ac-text-primary: #212529;
  --ac-text-muted:   #6C757D;
  --ac-text-faint:   #ADB5BD;

  /* Semantic */
  --ac-highlight: #EFF6FF;
  --ac-warn:      #FEF3C7;
  --ac-ok:        #D1FAE5;
}
```

### Python Color Constants (Track F)

```python
AC = {
    "blue": "#2563EB", "amber": "#D97706", "green": "#059669", "red": "#DC2626",
    "violet": "#7C3AED", "teal": "#0891B2", "rose": "#BE185D", "sienna": "#92400E",
    "bg": "#FFFFFF", "surface": "#F8F9FA", "border": "#DEE2E6",
    "axis": "#495057", "grid": "#E9ECEF",
    "text_primary": "#212529", "text_muted": "#6C757D", "text_faint": "#ADB5BD",
}

AC_SERIES = [AC["blue"], AC["amber"], AC["green"], AC["red"],
             AC["violet"], AC["teal"], AC["rose"], AC["sienna"]]
```

### p5.js Color Constants (Track A)

```javascript
// ── Academic palette ────────────────────────────────────────────
const AC = {
  blue:    '#2563EB',
  amber:   '#D97706',
  green:   '#059669',
  red:     '#DC2626',
  violet:  '#7C3AED',
  teal:    '#0891B2',
  rose:    '#BE185D',
  sienna:  '#92400E',

  bg:           '#FFFFFF',
  surface:      '#F8F9FA',
  border:       '#DEE2E6',
  axis:         '#495057',
  grid:         '#E9ECEF',
  textPrimary:  '#212529',
  textMuted:    '#6C757D',
  textFaint:    '#ADB5BD',
};

// Series array for cycling through colors programmatically
const AC_SERIES = [
  AC.blue, AC.amber, AC.green, AC.red,
  AC.violet, AC.teal, AC.rose, AC.sienna
];
```

### Alpha / Transparency Conventions

- Particle trails and overlapping geometry: use `color + 'AA'` (67% alpha) or
  call `setAlpha()` on a p5.Color object.
- Grid lines: 60% alpha (`99` suffix in hex).
- Fill regions / confidence bands: 20–30% alpha (`33`–`4D` suffix).
- Never use alpha to "soften" a color that should be fully saturated — reserve
  transparency for layering and density effects only.

---

## Typography System

### Fonts

Load via Google Fonts CDN in every HTML artifact:

```html
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Source+Serif+4:wght@400;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
```

| Role                     | Font              | Weight | Size    |
|--------------------------|-------------------|--------|---------|
| Page / artifact title    | Source Serif 4    | 600    | 22–24px |
| Figure title (Track F)   | Source Serif 4    | 600    | 15–16px |
| Section headings (UI)    | Inter             | 600    | 13–14px |
| Body / labels (UI)       | Inter             | 400    | 13px    |
| Axis tick labels         | Inter             | 400    | 11–12px |
| Axis titles              | Inter             | 500    | 13px    |
| Captions / metadata      | Inter             | 400    | 12px    |
| Numeric annotation / seed| JetBrains Mono    | 400    | 11–13px |
| Code blocks              | JetBrains Mono    | 400    | 12px    |

For static export (Track F), Kaleido renders with whatever fonts the export
engine can resolve. Always specify the family as a **stack with fallbacks**,
e.g. `"Inter, Helvetica, Arial, sans-serif"`, so a missing webfont degrades to a
neutral grotesque rather than a serif default.

For p5.js canvas text (Track A), p5 cannot load Google Fonts via `loadFont()`
without a `.otf`/`.ttf` URL. Use system-safe stacks instead:

```javascript
textFont('Inter, -apple-system, Arial, sans-serif');
// For monospaced labels:
// textFont('JetBrains Mono, Consolas, monospace');
```

---

# Part I — Track F: Research Figures

## F.1 Setup and Dependencies

Required Python packages, installed with `pip` into the project virtualenv (or
`pip install --user` when no venv exists):

```bash
pip install "plotly>=5.24" "kaleido>=0.2.1" "imageio>=2.34" "numpy>=1.24" "pandas>=2.0"
```

Install into the project virtualenv when one exists (`.venv/bin/python -m pip
install ...`), otherwise `pip install --user`. Pin them in the project's
`requirements-viz.txt` once installed. **Do not require Node or npm for figure
generation.** Never introduce a JavaScript build step.

**Ask before installing anything outside pip and the Playwright browser
download.** Kaleido 1.x drives a headless Chrome; if it cannot find one, say so
rather than pulling in a system package unasked.

Verify the stack once before the first figure of a session:

```python
import plotly.graph_objects as go
go.Figure(go.Scatter(x=[0, 1], y=[0, 1])).write_image("/tmp/_kaleido_probe.png")
```

If that raises, fall back to matplotlib (Section F.9) and say so in the report.

## F.2 Rendering Stack Rules

- **Primary renderer is Plotly** (`plotly.py`) with **Kaleido** for static export.
- Each figure is **defined once** as a Plotly figure object, then exported to
  **both** `figure.png` and `figure.svg` from that same object. Never build the
  PNG and the SVG from two different code paths — they must be the same figure.
- **Fallback renderer is matplotlib**, used only when Kaleido cannot produce a
  faithful export: dense LaTeX math, ridge / joy plots, custom geometry, complex
  hatching, or anything Plotly renders incorrectly in static mode. When
  matplotlib is used, apply the rcParams block in Section F.9 verbatim.
- **Verify every export opens and is non-empty before reporting success.** Use
  the helper in F.7. A zero-byte or truncated file is a failed figure, not a
  warning.

## F.3 The Academic Plotly Template

Apply this template to every Track F figure. The canonical implementation lives
in `report/acviz.py` when the project has one; otherwise inline it.

```python
"""Academic Plotly template: white ground, Tufte spine, faint horizontal grid."""
import plotly.graph_objects as go
import plotly.io as pio

FONT_UI = "Inter, Helvetica, Arial, sans-serif"
FONT_TITLE = "Source Serif 4, Georgia, serif"
FONT_MONO = "JetBrains Mono, Menlo, Consolas, monospace"

ac_template = go.layout.Template(
    layout=dict(
        paper_bgcolor=AC["bg"],
        plot_bgcolor=AC["bg"],
        colorway=AC_SERIES,
        font=dict(family=FONT_UI, size=12, color=AC["text_primary"]),
        title=dict(font=dict(family=FONT_TITLE, size=16,
                             color=AC["text_primary"]), x=0.0, xanchor="left"),
        margin=dict(l=64, r=110, t=52, b=56),  # right margin holds end-labels
        showlegend=False,                      # direct labeling is the default
        xaxis=dict(
            showline=True, linecolor=AC["axis"], linewidth=1.2, mirror=False,
            ticks="outside", tickcolor=AC["axis"], ticklen=4, tickwidth=1.0,
            tickfont=dict(family=FONT_UI, size=11, color=AC["text_muted"]),
            title=dict(font=dict(family=FONT_UI, size=13,
                                 color=AC["text_primary"])),
            showgrid=False, zeroline=False,    # no vertical grid
        ),
        yaxis=dict(
            showline=True, linecolor=AC["axis"], linewidth=1.2, mirror=False,
            ticks="outside", tickcolor=AC["axis"], ticklen=4, tickwidth=1.0,
            tickfont=dict(family=FONT_UI, size=11, color=AC["text_muted"]),
            title=dict(font=dict(family=FONT_UI, size=13,
                                 color=AC["text_primary"])),
            showgrid=True, gridcolor=AC["grid"], gridwidth=0.6,  # horizontal only
            zeroline=False,
        ),
        hoverlabel=dict(font=dict(family=FONT_UI, size=12)),
    )
)
pio.templates["academic"] = ac_template
pio.templates.default = "academic"
```

Heatmaps and matrix plots are the one exception to "horizontal grid only": they
may show both axes' grid, or neither.

## F.4 Visual Standard — Unchanged Constraints

These carry over unchanged and are non-negotiable.

- **White background always.** Reserve dark backgrounds only for an explicit
  request (e.g. "dark mode poster").
- **Left and bottom axes only** — no top/right border (the "Tufte spine").
- **Horizontal grid lines only**, in `#E9ECEF`, very light. No vertical grid
  unless the figure is a heatmap or matrix.
- Palette in order `#2563EB #D97706 #059669 #DC2626 #7C3AED #0891B2 #BE185D #92400E`.
- Fonts: Inter for labels, Source Serif 4 for titles, JetBrains Mono for numeric
  annotations.
- **Axis titles** in `#212529`; **tick labels** in `#6C757D`.
- Line width 1.8 px for primary series, 1.2 px for secondary, never above 3.0
  except for deliberate emphasis. Rounded caps and joins.
- **No rainbow or jet colormaps.** If a sequential colormap is needed, map
  lightness from `#F8F9FA` to the relevant series color. For diverging data use
  blue → `#F8F9FA` → red.
- **No Anthropic brand colors** (`#d97757`, `#6a9bcc`, `#788c5d`, `#faf9f5`,
  `#141413`).
- No font size below 11 px on any visible label.
- Never place text directly over data without a semi-opaque background pad.
- Never use pure black (`#000000`) or an explicit pure-white fill; use `#495057`
  and `#FFFFFF` / `#F8F9FA` from the palette.

## F.5 Visual Standard — New Requirements

The point of these is to make figures **less plain without making them
decorative**. Modern means clean hierarchy, not ornamentation.

### Direct labeling

With **five or fewer series**, drop the legend box entirely and label each series
at the right end of its line, in the series color:

```python
fig.update_layout(showlegend=False)
for name, color, y_last, x_last in end_points:
    fig.add_annotation(
        x=x_last, y=y_last, text=f"<b>{name}</b>", showarrow=False,
        xanchor="left", xshift=8, yanchor="middle",
        font=dict(family=FONT_UI, size=12, color=color),
    )
```

Reserve `margin.r >= 100` so the labels are not clipped. With six or more series,
switch to small multiples (below) rather than reinstating a crowded legend.

### The one callout

Every figure annotates **the single most important result**: the best value, the
crossover point, the convergence step, the point where the gap opens. One short
callout, one thin leader line, no box shadow.

```python
fig.add_annotation(
    x=x_star, y=y_star, text="crossover, step 4.2k",
    showarrow=True, arrowhead=0, arrowwidth=1.0, arrowcolor=AC["axis"],
    ax=28, ay=-30, xanchor="left",
    font=dict(family=FONT_UI, size=11, color=AC["text_primary"]),
    bgcolor="rgba(255,255,255,0.82)", borderpad=3,
)
```

Two callouts maximum. Three is clutter.

### Uncertainty is never omitted

When the data has repeated seeds or runs, uncertainty is **always shown** — a
shaded band for mean ± std (or a bootstrap CI), or error bars for categorical
data. Band fill is the series color at 20% alpha, no border:

```python
fig.add_trace(go.Scatter(
    x=list(x) + list(x)[::-1],
    y=list(mean + std) + list(mean - std)[::-1],
    fill="toself", fillcolor=hex_to_rgba(color, 0.20),
    line=dict(width=0), hoverinfo="skip", showlegend=False,
))
```

If a metric has only one seed, say so in the caption rather than drawing a bare
line as if it were a mean.

### Ticks and units

- Consistent tick density: aim for 4–7 ticks per axis, never more than 8.
- Human-readable formatting: `4.2k` not `4200`, `12%` not `0.12`, `1.4e-3` only
  when the exponent genuinely varies. Use `tickformat="~s"` for SI, `".0%"` for
  percentages.
- **Axis titles carry units**: `Environment steps (millions)`, `Return
  (undiscounted)`, `Wall-clock (s)`, `Forgetting (Δ return)`.

### Small multiples over spaghetti

More than five overlapping lines, or more than one dependent variable, becomes a
faceted grid. Shared axes, one row of tick labels on the bottom, one column on
the left, a small bold panel title per facet. Use
`plotly.subplots.make_subplots(shared_xaxes=True, shared_yaxes=True)` and keep
per-panel margins tight.

**Exception — panels whose scales differ by more than ~10×.** A shared axis is
wrong when one panel tops out at 20 and another at 4000: four panels flatten to
nothing. Give each panel its own axis, always including zero, and say so in the
caption. No log scale, no clipping, no silent normalisation as a substitute.

### No decorative effects

No gradient fills, no glow, no drop shadows, no rounded plot frames, no 3-D
extrusion of 2-D data, no textured backgrounds on a paper figure. Shadows are
permitted only on dashboard UI chrome, never inside the plotting area.

## F.5b Publication Figures

Everything above assumes a figure that stands alone (a dashboard card, a slide).
A figure **destined for a paper** is set inside a LaTeX caption and carries extra
obligations. These override F.3–F.5 where they conflict.

### No title inside the artwork

The caption carries the title. A baked-in one duplicates it and is cropped at
typesetting anyway. Set `title=None` and reclaim the top margin. Panel titles,
axis titles and legends stay. Dashboard figures are the exception and keep a
title.

### Fixed palette roles across a figure set

Assign roles once and hold them, so a colour means one thing across every figure
in the paper:

| Role | Token |
|------|-------|
| Ours / primary method | `blue` |
| The baseline being compared against | `amber` |
| Ceiling, upper bound, reference model | `green` |
| Failure, below threshold, regression | `red` |
| Neutral reference (a specialist, a prior) | `text_faint` |

Band the two rows that matter in a table: ours on `#EFF6FF`, the rival on
`#FEF3C7`, with matching text colour. Everything else stays neutral.

**Every numeral is mono** (`FONT_MONO`), in tables, value labels and tick labels
alike. It is what makes columns align and read as data.

### Pick the mark by what the reader must judge

| The question | The mark |
|---|---|
| How large is each value, across many? | Bar from zero |
| How large is each value, across **3–6**? | Lollipop (below) |
| How far past a **per-row** target? | Dumbbell (below) |
| Did it clear one shared bar? | Bars or lollipops plus a dashed rule |
| Matrix of ratios | Heatmap, clamped and pivoted scale (below) |

**Lollipop.** Below about six categories, thick bars are mostly ink. A thin stem
(3 px at 40% alpha) from zero with a white-ringed dot at the value encodes the
same thing and leaves the panel quiet enough to carry a **dashed reference rule**
at the subject's value — and that rule is what turns a list of numbers into a
comparison. Keep zero on the axis: stem length is a real magnitude here, unlike
the dumbbell case.

**Dumbbell.** When every row has its *own* reference, a bar from zero is wrong:
the stretch from 0 to the reference compares nothing, and it forces a shared axis
that flattens the margins you care about. Run the bar from reference to achieved
so its **length and direction are the margin**, put a white-ringed dot at the
achieved end, and mark the reference with a short vertical gate tick. Dropping
the zero baseline is legitimate here because length now encodes a *difference*,
not a magnitude — state that in the footnote. Draw the ±1 s.d. spread *under* the
margin bar in a 60%-lightened tint, so only the overhang past the mean shows.

**A reference is a rule, not a bar.** A ceiling or threshold drawn as one more
bar reads as one more competitor. Draw it as a dashed line across the panel and
put its value in the panel subtitle.

**One shared legend at the top, then delete per-panel tick labels.** Rotated
category labels repeated across every panel are the worst thing you can do to a
small-multiple figure.

**Headroom** is `max(bar_max * 1.20, reference * 1.04)`. Giving a reference rule
a bar's headroom leaves a third of the panel empty.

### Clamped, pivoted scales for ratio data

Clamp at the semantically meaningful ceiling; pivot at the meaningful threshold,
not at the midpoint of the data. A scale stretched to fit one 360% outlier drags
a genuinely good 82% into alarming red.

```python
PIVOT = 0.65  # the "acceptable" line
SCALE = [
    [0.00, "#DC2626"], [0.22, "#E8736F"], [0.45, "#F4B3AE"],
    [PIVOT, "#F2F3F5"],
    [0.78, "#B9CDF6"], [0.89, "#7CA2F0"], [1.00, "#2563EB"],
]
# z = min(value, 1.0); the PRINTED number is always the true value.
```

Cell text white below 0.26 or above 0.93, `text_primary` between; bold the
diagonal; label the clamp on the colourbar as `≥100%`.

### Tables as figures

Booktabs, drawn with shapes and annotations on hidden axes. Three rules only:
top (1.3 px `axis`), under the header (1.0 px), bottom (1.3 px). **No vertical
rules and no row shading** beyond the two banded rows. Section labels in 9 px
uppercase `text_muted`; values right-aligned mono; labels left-aligned Inter;
`±std` as a 6.5 px `text_faint` span appended to the value rather than its own
column. Advance a cursor by each row's own height so a separator costs ~0.34 of a
row instead of a whole empty one.

For per-row results, pair the graph with a **narrow second subplot of mono
numbers** (value, reference, ratio) and a faint rail across each row, so the eye
tracks from mark to exact value. The graph carries shape; the column carries
precision.

### Footnotes inside the figure

A figure needing a caveat to avoid misleading gets one, 7.5–8.5 px `text_muted`
under the bottom rule: what the normaliser is, what the error bars mean, why a
cell is empty, any measurement asymmetry, the seed count.

Plotly does not reflow — hand-wrap with `<br>`, indent continuations with
`&nbsp;&nbsp;&nbsp;`, budget ~108 characters per line at 8.5 px in a 648 px
figure. Anchor at `xref="paper", x=0` with `xshift=-<left_margin>` so it starts
at the figure edge, not the plot edge.

### Data hygiene

Keep the numbers in a `data.json` beside the script, never inline in plotting
code, with a `provenance` block naming the source file and any caveat. Read
shared constants out of the codebase rather than retyping them — parse with
`ast.literal_eval` when importing would drag in heavy dependencies.

**Recompute headline statistics from the raw data** instead of copying them out
of an older README. A caption that contradicts its own figure is the most common
defect in a results folder, and it survives because nobody re-derives the number.

### Honest numbers

- **Never pad precision.** Print the decimals the run reports, even beside
  someone else's extra digit. Padding implies precision you do not have.
- **Never combine standard deviations** without their seed-level covariance.
- **Mark any cell that is not the same quantity as its column** with `†`, explain
  it in the footnote, and exclude it from the "best value" bolding.
- A value that is a definitional constant (a reference method scoring exactly 0
  on a metric defined against itself) is **not** a competitor for bold.

## F.6 Animation

When the data has a **temporal or iterative axis** — training curves over epochs,
a loss landscape evolving over consolidation steps, parameter trajectories,
forgetting matrices filling in task by task — additionally produce an animated
version using Plotly frames.

**Only animate when the animation reveals something the static figure cannot.**
A curve that is fully legible as a static line does not need to be redrawn frame
by frame. Good candidates: the *order* in which structure appears, a trajectory
whose path matters, a matrix that fills in.

Exports:

- `figures/<id>.gif` via `imageio`, from per-frame PNGs rendered by Kaleido.
- `figures/<id>.html` via `fig.write_html(path, include_plotlyjs="cdn",
  full_html=True, auto_play=False)` — an interactive Plotly export. Pass
  `auto_play=False`: an animated figure that starts playing the instant a modal
  opens is disorienting, and the reader should press Play deliberately.
  Then inject `<style>html,body{margin:0;padding:0;height:100%}</style>` before
  `</head>`. Plotly sizes its div to 100% of the body, and the browser's default
  8 px body margin pushes the x-axis title out of an embedding iframe.

The static `PNG`/`SVG` for an animated figure shows either the **final frame** or
a **small multiple of selected frames** (e.g. frames at 0%, 33%, 66%, 100%),
never a blank first frame.

**Non-animatable trace types.** plotly.js animates only a subset of trace types.
A `Contour`, `Heatmap`, or similar trace is **cleared for the duration of a
transition** and redrawn only when it ends — so an animated contour renders as a
blank panel while it plays, whatever the frame's `traces` subset says. Kaleido is
unaffected, because it rasterises each frame as an independent figure.

So when a figure's background is a non-animatable trace:

- the **GIF** carries the motion (build complete frames and let Kaleido render
  each one);
- the **interactive HTML** is *not* a frame animation. Export the full path or
  final state with hover inspection instead — per-point `text` plus a
  `hovertemplate` naming the step — which is the interactivity a reader actually
  wants from that figure.

Check this before promising an animated interactive export: load the HTML and
confirm the background survives playback.

```python
import imageio.v2 as imageio

def write_gif(fig: go.Figure, frames: list, path: str, fps: int = 8) -> None:
    """Render each Plotly frame to PNG via Kaleido and stitch into a GIF."""
    images = []
    for frame in frames:
        snapshot = go.Figure(data=frame.data, layout=fig.layout)
        images.append(imageio.imread(snapshot.to_image(format="png", scale=2)))
    imageio.mimsave(path, images, fps=fps, loop=0)
```

## F.7 Export Contract

**Print resolution.** CSS pixels are 96 per inch; the target is 300 dpi at the
intended print width. Therefore:

```python
CSS_DPI, PRINT_DPI = 96, 300
EXPORT_SCALE = PRINT_DPI / CSS_DPI          # 3.125

def width_px(print_inches: float) -> int:
    """Plotly figure width in CSS px for a given intended print width."""
    return int(round(print_inches * CSS_DPI))

# Standard widths
W_SINGLE_COL = width_px(3.25)   # 312 px  -> 975 px PNG
W_ONE_HALF   = width_px(5.00)   # 480 px  -> 1500 px PNG
W_FULL       = width_px(6.75)   # 648 px  -> 2025 px PNG
```

Export both formats from the same figure object, then verify:

```python
from pathlib import Path

def export_figure(fig: go.Figure, stem: Path, width: int, height: int) -> dict:
    """Write PNG at 300 dpi-equivalent and SVG; verify both are non-empty.

    Returns a dict of {"png": relpath, "svg": relpath} for the manifest.
    Raises RuntimeError if either export is missing or empty.
    """
    png, svg = stem.with_suffix(".png"), stem.with_suffix(".svg")
    fig.write_image(png, width=width, height=height, scale=EXPORT_SCALE)
    fig.write_image(svg, width=width, height=height)   # vector: scale is moot
    for path in (png, svg):
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"export failed or empty: {path}")
    return {"png": str(png), "svg": str(svg)}
```

Aspect ratio: default to the golden-ish 1.55:1 (`height = round(width / 1.55)`)
for line plots; square for matrices and heatmaps.

**Every figure produces both PNG and SVG. There is no single-format figure.**

### The SVG must actually be vector — verify it

Non-empty is not the same as vector. Check:

```bash
grep -c '<image' fig.svg    # must be 0
```

**`go.Heatmap` rasterises its cells into an embedded bitmap** and fails this,
which silently ships a blurry matrix into a paper. For any matrix small enough to
label (say up to 15×15), draw the cells yourself:

- numeric axes, `range=[0, n]`, ticks re-labelled at cell centres `i + 0.5`;
- one `add_shape(type="rect")` per cell, inset ~0.035 to emulate `xgap`/`ygap`,
  `fillcolor` sampled from the colourscale by interpolating in sRGB;
- the colourbar carried on an invisible marker trace, which Plotly renders as an
  SVG gradient:

```python
go.Scatter(x=[None], y=[None], mode="markers", showlegend=False,
           marker=dict(color=[0], colorscale=SCALE, cmin=0, cmax=1,
                       showscale=True, opacity=0, colorbar=dict(...)))
```

Annotations, shapes and `go.Scatter` are all vector and safe.

## F.8 Manifest Contract

Every figure generation appends **one entry** to `report/manifest.json`. Create
the file if absent (as a JSON array, or as `{"figures": [...]}` with optional
top-level `project` and `run_date`). Re-running a generator **replaces** the
entry with the same `id` rather than appending a duplicate.

```json
{
  "id": "loss_curves_seed_sweep",
  "group": "training_dynamics",
  "group_title": "Training dynamics",
  "title": "Validation loss across seeds",
  "caption": "One sentence, at most 160 characters.",
  "details": "Full description. Method, dataset, model, seeds, hyperparameters, what to look for. Markdown allowed.",
  "seed": 0,
  "dataset": "Atari Breakout",
  "model": "DQN",
  "variants": [
    {"label": "Line", "png": "figures/loss_curves_seed_sweep_line.png", "svg": "figures/loss_curves_seed_sweep_line.svg", "html": null},
    {"label": "Small multiples", "png": "figures/loss_curves_seed_sweep_grid.png", "svg": "figures/loss_curves_seed_sweep_grid.svg", "html": null}
  ],
  "animation": {"gif": "figures/loss_curves_seed_sweep.gif", "html": "figures/loss_curves_seed_sweep.html"},
  "metrics": [{"name": "final val loss", "value": 0.412, "unit": ""}],
  "created": "2026-09-09T14:00:00"
}
```

**Field rules.**

| Field | Required | Notes |
|-------|----------|-------|
| `id` | yes | snake_case, unique, stable across reruns |
| `group` | yes | snake_case key; figures with the same `group` render in one section |
| `group_title` | yes | Human-readable heading, consistent across a group |
| `title` | yes | Sentence case, no trailing period |
| `caption` | yes | One sentence, **≤ 160 characters** |
| `details` | yes | Full description, Markdown allowed. Method, dataset, model, seeds, hyperparameters, what to look for |
| `seed` | no | `int`, or `null` when aggregated over seeds |
| `dataset` | no | string |
| `model` | no | string |
| `variants` | yes | non-empty list; `label` plus `png`, `svg`, and `html` (nullable) |
| `animation` | no | `null`, or an object with `gif` and/or `html` |
| `metrics` | no | list of `{name, value, unit}` |
| `created` | yes | ISO 8601 timestamp |

**Rules for `variants`.** Add a second or third variant **only when a different
chart type gives a different perspective on the same data** — distribution versus
trend, absolute versus relative, bar versus pie for composition, line versus
small multiples. Do not produce redundant variants (the same chart at two sizes
is not a variant). **The first variant is the paper figure.**

**All paths in the manifest are relative to `report/`.** Figures are written
under `report/figures/`.

After appending, always regenerate the dashboard:

```bash
python report/build_dashboard.py && python report/verify_dashboard.py
```

## F.9 matplotlib Fallback

Only when Kaleido cannot render the figure faithfully. Inject these rcParams
verbatim:

```python
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Inter', 'Helvetica', 'Arial'],
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.color':        '#E9ECEF',
    'grid.linewidth':    0.6,
    'axes.edgecolor':    '#495057',
    'axes.labelcolor':   '#212529',
    'xtick.color':       '#6C757D',
    'ytick.color':       '#6C757D',
    'figure.dpi':        150,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
})

AC_SERIES = ['#2563EB', '#D97706', '#059669', '#DC2626',
             '#7C3AED', '#0891B2', '#BE185D', '#92400E']
```

Grid on the y-axis only (`ax.grid(axis="y")`), and still export both formats:

```python
fig.savefig(stem.with_suffix(".png"), dpi=300)
fig.savefig(stem.with_suffix(".svg"))
```

Then verify non-empty and append the manifest entry exactly as in F.8. A
matplotlib figure is a first-class manifest entry, not a second-class one.

## F.10 Track F Checklist

Before reporting a figure complete:

- [ ] Plotly used (or matplotlib with a stated reason)
- [ ] `academic` template applied: white ground, Tufte spine, horizontal grid only
- [ ] Palette used in order, no rainbow, no Anthropic colors
- [ ] Direct end-of-line labels when ≤ 5 series; small multiples when > 5
- [ ] Exactly one (at most two) callout annotating the key result
- [ ] Uncertainty shown wherever repeated runs exist
- [ ] 4–7 ticks per axis, human-readable formatting, units in the axis title
- [ ] No decorative effects
- [ ] PNG exported at `scale = 3.125` (300 dpi at print width) **and** SVG
- [ ] Both files verified to exist and be non-empty
- [ ] `grep -c '<image' *.svg` returns 0 — no rasterised `go.Heatmap`
- [ ] Animation produced if and only if it reveals something static cannot
- [ ] One manifest entry appended (or replaced by `id`), paths relative to `report/`
- [ ] `report/build_dashboard.py` re-run afterwards

Paper-destined figures additionally (F.5b):

- [ ] No title inside the artwork
- [ ] Palette roles fixed across the whole figure set; every numeral in mono
- [ ] Mark chosen by what must be judged; per-row references as dumbbell anchors
      or rules, never extra bars
- [ ] One shared legend; no rotated per-panel tick labels
- [ ] Ratio scales clamped and pivoted; printed numbers un-clamped
- [ ] Precision not padded; non-comparable cells daggered and excluded from bolding
- [ ] Footnote covers normaliser, error-bar meaning, empty cells, seed count

---

# Part II — Track A: Generative Art Artifacts

## A.1 Algorithmic Philosophy

Every Track A artifact begins with an algorithmic philosophy — a 4–6 paragraph
manifesto describing the computational aesthetic movement being expressed. The
philosophy must:

1. **Name the movement** (1–2 words, e.g., "Diffusion Lattice", "Phase Boundary").
2. **Articulate the computational essence**: what mathematical processes, noise
   fields, particle behaviors, or emergent dynamics drive the work.
3. **Emphasize academic rigor**: the algorithm should feel like the product of
   careful parameter tuning, mathematical derivation, and iterative refinement —
   not random noise. Repeat phrases like "meticulously tuned", "analytically
   motivated", "convergent under the right conditions."
4. **Leave room for interpretation**: the philosophy guides but does not
   over-specify. The implementation should make creative choices.

Save the philosophy as a `.md` file alongside the HTML artifact.

### Philosophy Examples Tuned to Academic Aesthetic

**"Gradient Descent Topology"**
Loss landscapes as generative art. Particles initialized at random positions in a
2D parameter space flow along gradient fields derived from a mixture of Gaussian
bowls. The gradient is intentionally perturbed by Perlin noise — a proxy for
stochastic gradient noise. Convergence basins form slowly, with each run
deterministically seeded, producing unique but analytically interpretable phase
diagrams. The final image looks like a hand-drawn contour plot from an
optimization theory textbook, yet no two seeds produce the same topology.

**"Region Migration"**
Inspired by the geometry of decision boundaries in piecewise-linear networks.
Line segments (non-linearities) are born near data point clusters and gradually
migrate toward the boundary between two slowly-separating attractor regions.
Density accumulates at the boundary; the interior becomes sparse and smooth. The
result is a formal geometric diagram that could accompany a theorem — clean,
structured, quietly animate.

## A.2 p5.js Implementation Requirements

### Canvas and Sketch Structure

```javascript
let seed = 42;
const W = 900, H = 900;   // Square canvas; adjust for non-square as needed

function setup() {
  let cnv = createCanvas(W, H);
  cnv.parent('canvas-container');
  randomSeed(seed);
  noiseSeed(seed);
  background(AC.bg);
  pixelDensity(2);   // Retina / high-DPI output
}
```

Always call `pixelDensity(2)` for crisp PNG exports. This is the single biggest
quality improvement for print/poster output.

### Axes and Grid

When the visualization encodes data (as opposed to pure generative art), draw
minimal axes following these rules:

```javascript
function drawAxes(x0, y0, w, h, xTicks, yTicks, xLabel, yLabel) {
  stroke(AC.axis); strokeWeight(1.2); noFill();

  // Axis lines (left and bottom only — no top/right border)
  line(x0, y0, x0, y0 - h);      // y-axis
  line(x0, y0, x0 + w, y0);      // x-axis

  // Grid lines (horizontal only, very faint)
  stroke(AC.grid); strokeWeight(0.6);
  for (let t of yTicks) {
    let py = map(t.val, t.min, t.max, y0, y0 - h);
    line(x0, py, x0 + w, py);
  }

  // Tick labels
  noStroke(); fill(AC.textMuted);
  textSize(11); textFont('Inter, Arial, sans-serif');
  textAlign(RIGHT, CENTER);
  for (let t of yTicks) {
    let py = map(t.val, t.min, t.max, y0, y0 - h);
    text(t.label, x0 - 6, py);
  }
  textAlign(CENTER, TOP);
  for (let t of xTicks) {
    let px = map(t.val, t.min, t.max, x0, x0 + w);
    text(t.label, px, y0 + 6);
  }

  // Axis labels
  fill(AC.textPrimary); textSize(13); textStyle(NORMAL);
  textAlign(CENTER, BOTTOM);
  text(xLabel, x0 + w / 2, y0 + 36);

  push();
  translate(x0 - 44, y0 - h / 2);
  rotate(-HALF_PI);
  text(yLabel, 0, 0);
  pop();
}
```

### Line Strokes for Series

```javascript
function seriesStroke(idx, weight = 1.8) {
  stroke(AC_SERIES[idx % AC_SERIES.length]);
  strokeWeight(weight);
}
```

Use `weight = 1.8` as default for line plots. Increase to `2.2` for the primary
series when overlapping many lines. Never exceed `3.0` except for emphasis.

### Seeded Randomness (Mandatory)

```javascript
// Always expose seed as a parameter
let params = {
  seed: 42,
  // ... other parameters
};

function applyParams() {
  randomSeed(params.seed);
  noiseSeed(params.seed);
  // re-initialize all derived state
}
```

Same seed must produce pixel-identical output across reloads. Test this.

### Export: PNG and SVG

The artifact must include two export buttons.

**PNG export** (via canvas `toDataURL`):
```javascript
function exportPNG() {
  saveCanvas('figure', 'png');
}
```

**SVG export** requires the p5.svg library:

```html
<script src="https://cdn.jsdelivr.net/npm/p5.js-svg@1.5.1/dist/p5.svg.min.js"></script>
```

```javascript
// In setup(), switch renderer to SVG for export:
function exportSVG() {
  // Re-render once with SVG renderer
  let svgGraphics = createGraphics(width, height, SVG);
  drawAll(svgGraphics);   // your draw function accepts a graphics context
  save(svgGraphics, 'figure.svg');
}
```

If SVG export is complex to implement, at minimum provide PNG export and add a
comment noting the SVG CDN link for the user to enable. Always prefer providing
both.

## A.3 HTML Viewer Template

Every Track A artifact is a single self-contained `.html` file. Use this
structure exactly. Replace only the sections marked `<!-- VARIABLE -->`.

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <!-- VARIABLE: set title to the movement name -->
  <title><!-- movement name --> | Academic Generative Art</title>

  <script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.7.0/p5.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/p5.js-svg@1.5.1/dist/p5.svg.min.js"></script>

  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Source+Serif+4:wght@400;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

  <style>
    /* ── Academic palette ───────────────────── */
    :root {
      --ac-blue:    #2563EB; --ac-amber:  #D97706;
      --ac-green:   #059669; --ac-red:    #DC2626;
      --ac-violet:  #7C3AED; --ac-teal:   #0891B2;
      --ac-rose:    #BE185D; --ac-sienna: #92400E;

      --ac-bg:           #FFFFFF; --ac-surface:  #F8F9FA;
      --ac-border:       #DEE2E6; --ac-axis:     #495057;
      --ac-grid:         #E9ECEF;
      --ac-text-primary: #212529; --ac-text-muted: #6C757D;
      --ac-text-faint:   #ADB5BD; --ac-highlight: #EFF6FF;
    }

    /* ── Layout ─────────────────────────────── */
    * { margin: 0; padding: 0; box-sizing: border-box; }

    body {
      font-family: 'Inter', -apple-system, Arial, sans-serif;
      background: var(--ac-bg);
      color: var(--ac-text-primary);
      min-height: 100vh;
    }

    .app {
      display: flex;
      min-height: 100vh;
    }

    /* ── Sidebar ─────────────────────────────── */
    .sidebar {
      width: 300px;
      flex-shrink: 0;
      background: var(--ac-surface);
      border-right: 1px solid var(--ac-border);
      padding: 28px 20px;
      overflow-y: auto;
    }

    .sidebar-title {
      font-family: 'Source Serif 4', Georgia, serif;
      font-size: 20px;
      font-weight: 600;
      color: var(--ac-text-primary);
      margin-bottom: 4px;
      line-height: 1.3;
    }

    .sidebar-subtitle {
      font-size: 12px;
      color: var(--ac-text-muted);
      margin-bottom: 28px;
      line-height: 1.5;
    }

    .section {
      margin-bottom: 28px;
    }

    .section-label {
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--ac-text-muted);
      margin-bottom: 12px;
    }

    /* ── Seed controls (FIXED — always include) ── */
    .seed-display {
      font-family: 'JetBrains Mono', Consolas, monospace;
      font-size: 22px;
      font-weight: 400;
      color: var(--ac-text-primary);
      letter-spacing: 0.03em;
      margin-bottom: 10px;
    }

    .btn-row {
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
      margin-bottom: 8px;
    }

    .btn {
      font-family: 'Inter', sans-serif;
      font-size: 12px;
      font-weight: 500;
      padding: 6px 12px;
      border: 1px solid var(--ac-border);
      border-radius: 4px;
      background: var(--ac-bg);
      color: var(--ac-text-primary);
      cursor: pointer;
      transition: background 80ms, border-color 80ms;
    }
    .btn:hover { background: var(--ac-highlight); border-color: var(--ac-blue); }
    .btn.primary {
      background: var(--ac-blue);
      color: #fff;
      border-color: var(--ac-blue);
    }
    .btn.primary:hover { background: #1d4ed8; }

    .seed-jump {
      display: flex;
      gap: 6px;
      margin-top: 8px;
    }
    .seed-jump input {
      font-family: 'JetBrains Mono', monospace;
      font-size: 13px;
      flex: 1;
      padding: 5px 8px;
      border: 1px solid var(--ac-border);
      border-radius: 4px;
      background: var(--ac-bg);
      color: var(--ac-text-primary);
      outline: none;
    }
    .seed-jump input:focus { border-color: var(--ac-blue); }

    /* ── Parameter controls (VARIABLE) ────────── */
    .control-group {
      margin-bottom: 14px;
    }
    .control-group label {
      display: flex;
      justify-content: space-between;
      font-size: 12px;
      color: var(--ac-text-primary);
      margin-bottom: 5px;
    }
    .control-group label span {
      font-family: 'JetBrains Mono', monospace;
      font-size: 11px;
      color: var(--ac-text-muted);
    }
    .control-group input[type=range] {
      width: 100%;
      accent-color: var(--ac-blue);
      height: 4px;
      cursor: pointer;
    }

    /* ── Legend (optional, for multi-series art) ── */
    .legend-item {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 12px;
      color: var(--ac-text-muted);
      margin-bottom: 6px;
    }
    .legend-swatch {
      width: 24px;
      height: 3px;
      border-radius: 2px;
      flex-shrink: 0;
    }

    /* ── Canvas area ──────────────────────────── */
    .main {
      flex: 1;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 32px;
      background: var(--ac-bg);
    }

    #canvas-container canvas {
      display: block;
      box-shadow: 0 2px 12px rgba(0,0,0,0.08);
      border: 1px solid var(--ac-border);
    }
  </style>
</head>
<body>
<div class="app">

  <!-- ── Sidebar ──────────────────────────── -->
  <aside class="sidebar">

    <!-- VARIABLE: movement name and one-line description -->
    <div class="sidebar-title">Movement Name</div>
    <div class="sidebar-subtitle">One-sentence description of the algorithmic philosophy.</div>

    <!-- FIXED: Seed navigation -->
    <div class="section">
      <div class="section-label">Seed</div>
      <div class="seed-display" id="seed-display">42</div>
      <div class="btn-row">
        <button class="btn" onclick="prevSeed()">← Prev</button>
        <button class="btn" onclick="nextSeed()">Next →</button>
        <button class="btn" onclick="randomSeedBtn()">Random</button>
      </div>
      <div class="seed-jump">
        <input type="number" id="seed-input" placeholder="Jump to seed…">
        <button class="btn primary" onclick="jumpSeed()">Go</button>
      </div>
    </div>

    <!-- VARIABLE: Parameters — adapt to your algorithm -->
    <div class="section">
      <div class="section-label">Parameters</div>

      <div class="control-group">
        <label>Particle Count <span id="count-val">800</span></label>
        <input type="range" id="count" min="100" max="3000" step="100" value="800"
               oninput="updateParam('count', +this.value, 'count-val')">
      </div>

      <div class="control-group">
        <label>Noise Scale <span id="noiseScale-val">0.003</span></label>
        <input type="range" id="noiseScale" min="0.001" max="0.02" step="0.001" value="0.003"
               oninput="updateParam('noiseScale', +this.value, 'noiseScale-val')">
      </div>

      <!-- Add more control-group divs as needed for your algorithm -->
    </div>

    <!-- OPTIONAL: Series legend for multi-series visualizations -->
    <!--
    <div class="section">
      <div class="section-label">Legend</div>
      <div class="legend-item">
        <div class="legend-swatch" style="background:#2563EB"></div> Train
      </div>
      <div class="legend-item">
        <div class="legend-swatch" style="background:#D97706"></div> Test
      </div>
      <div class="legend-item">
        <div class="legend-swatch" style="background:#059669"></div> Random
      </div>
    </div>
    -->

    <!-- FIXED: Actions -->
    <div class="section">
      <div class="section-label">Export</div>
      <div class="btn-row">
        <button class="btn primary" onclick="regenerate()">Regenerate</button>
        <button class="btn" onclick="resetParams()">Reset</button>
      </div>
      <div class="btn-row" style="margin-top:6px;">
        <button class="btn" onclick="exportPNG()">↓ PNG</button>
        <button class="btn" onclick="exportSVG()">↓ SVG</button>
      </div>
    </div>

  </aside>

  <!-- ── Canvas ───────────────────────────── -->
  <main class="main">
    <div id="canvas-container"></div>
  </main>

</div>

<script>
// ── Academic palette constants ──────────────────────────────────────
const AC = {
  blue:    '#2563EB', amber:   '#D97706',
  green:   '#059669', red:     '#DC2626',
  violet:  '#7C3AED', teal:    '#0891B2',
  rose:    '#BE185D', sienna:  '#92400E',
  bg:           '#FFFFFF', surface:  '#F8F9FA',
  border:       '#DEE2E6', axis:     '#495057',
  grid:         '#E9ECEF',
  textPrimary:  '#212529', textMuted: '#6C757D',
};
const AC_SERIES = [
  AC.blue, AC.amber, AC.green, AC.red,
  AC.violet, AC.teal, AC.rose, AC.sienna
];

// ── Parameters (VARIABLE — define what your algorithm needs) ────────
const DEFAULTS = {
  seed:       42,
  count:      800,
  noiseScale: 0.003,
  // ... add your parameters here
};
let params = { ...DEFAULTS };

// ── Seed management (FIXED) ─────────────────────────────────────────
function updateSeedDisplay() {
  document.getElementById('seed-display').textContent = params.seed;
}
function prevSeed()       { params.seed = Math.max(0, params.seed - 1); regenerate(); }
function nextSeed()       { params.seed += 1; regenerate(); }
function randomSeedBtn()  { params.seed = Math.floor(Math.random() * 99999); regenerate(); }
function jumpSeed() {
  const v = parseInt(document.getElementById('seed-input').value);
  if (!isNaN(v) && v >= 0) { params.seed = v; regenerate(); }
}
function regenerate() {
  updateSeedDisplay();
  redraw();
}
function resetParams() {
  params = { ...DEFAULTS };
  // sync all slider values
  for (const [k, v] of Object.entries(DEFAULTS)) {
    const el = document.getElementById(k);
    if (el) { el.value = v; }
    const lbl = document.getElementById(k + '-val');
    if (lbl) { lbl.textContent = v; }
  }
  regenerate();
}
function updateParam(key, val, labelId) {
  params[key] = val;
  if (labelId) document.getElementById(labelId).textContent = val;
  regenerate();
}

// ── Export (FIXED) ─────────────────────────────────────────────────
function exportPNG() {
  saveCanvas('figure_seed' + params.seed, 'png');
}
function exportSVG() {
  // p5.svg: re-render once with SVG renderer then save
  // If p5.svg is loaded, this works automatically via saveCanvas with .svg
  saveCanvas('figure_seed' + params.seed, 'svg');
}

// ── p5.js sketch (VARIABLE — implement your algorithm here) ─────────
function setup() {
  let cnv = createCanvas(900, 900);
  cnv.parent('canvas-container');
  pixelDensity(2);    // high-DPI — do not remove
  noLoop();
  updateSeedDisplay();
}

function draw() {
  randomSeed(params.seed);
  noiseSeed(params.seed);
  background(AC.bg);

  // ── YOUR ALGORITHM GOES HERE ──────────────────────────────────────
  // Use AC_SERIES[0], AC_SERIES[1], ... for series colors
  // Use AC.axis for structural lines
  // Use AC.textPrimary / AC.textMuted for labels
  // Enforce pixelDensity(2) for crisp output
  // ─────────────────────────────────────────────────────────────────
}
</script>
</body>
</html>
```

## A.4 Track A Development Process

1. **Receive the user's request.** Identify the conceptual seed: what
   mathematical or computational idea is latent in the request?
2. **Write the algorithmic philosophy** (4–6 paragraphs, save as `.md`).
   Emphasize the analytical motivation, the emergence behavior, and the
   parameter sensitivities. Use academic language — this is a short methods
   section for a generative system.
3. **Design the parameter space.** Prefer parameters with clear geometric or
   mathematical interpretations (noise scale, diffusion coefficient, step size,
   temperature) over vague sliders.
4. **Implement the algorithm** in `draw()`. Start from structure and add
   complexity — do not start from complexity.
5. **Validate reproducibility.** Same seed must produce identical output on every
   call to `draw()`. Reset `randomSeed` and `noiseSeed` at the top of `draw()`.
6. **Export both PNG and SVG.**

## A.5 File Output Convention

| File                          | Content                                       |
|-------------------------------|-----------------------------------------------|
| `<movement_name>.md`          | Algorithmic philosophy (4–6 paragraphs)       |
| `<movement_name>.html`        | Self-contained interactive p5.js artifact     |

Name both files using snake_case of the movement name, e.g.
`gradient_descent_topology.md` and `gradient_descent_topology.html`.

If the workflow requires saving canvas frames to disk (e.g. for animation), also
provide a headless Node.js script using `canvas` + `jsdom` to render and save
PNGs programmatically. **Ask before generating this** unless explicitly
requested — and note that this is the only place Node is permitted; figure
generation (Track F) must never depend on it.

---

# Quick Reference Checklist

## Track F (research figures)

- [ ] `plotly` + `kaleido` installed and probed
- [ ] `academic` Plotly template registered and set as default
- [ ] White background, Tufte spine, horizontal grid only
- [ ] Palette in order; no rainbow/jet; no Anthropic brand colors
- [ ] Direct end-labels (≤ 5 series) or small multiples (> 5)
- [ ] One key-result callout
- [ ] Uncertainty band or error bars wherever repeats exist
- [ ] Units in axis titles, 4–7 readable ticks
- [ ] PNG at `scale = 3.125` **and** SVG, both verified non-empty
- [ ] SVG verified genuinely vector: `grep -c '<image'` returns 0
- [ ] Paper figures: F.5b applied (no title, fixed roles, honest precision, footnote)
- [ ] Animation only when it adds information; `gif` + `html` if so
- [ ] Manifest entry appended/replaced, paths relative to `report/`
- [ ] `report/build_dashboard.py` and `report/verify_dashboard.py` re-run

## Track A (generative art)

- [ ] CSS variables block (`--ac-*`) present in `<style>`
- [ ] `AC` and `AC_SERIES` constants declared in sketch
- [ ] `pixelDensity(2)` called in `setup()`
- [ ] `randomSeed(params.seed)` and `noiseSeed(params.seed)` at top of `draw()`
- [ ] Seed navigation (prev / next / random / jump) functional
- [ ] PNG export button functional
- [ ] SVG export button functional (or note CDN for user to enable)
- [ ] No Anthropic brand colors, no dark background unless requested
- [ ] Axis labels `#212529`; tick labels `#6C757D`
- [ ] Series colors drawn from `AC_SERIES` in order
- [ ] Philosophy `.md` file generated alongside the HTML
