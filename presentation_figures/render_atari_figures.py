"""Render authentic Atari game frames and vectorize them into lossless SVG figures.

Each ALE frame is a 210x160 array over a small quantized palette, so it converts
exactly into a set of axis-aligned rectangles: horizontal run-length encoding
followed by a vertical merge of identical runs. The resulting SVG is true vector
(no embedded raster), scales to any projector size, and stays pixel-faithful.

Also writes a high-resolution nearest-neighbour PNG of the same frame.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Optional

import gymnasium as gym
import numpy as np
from PIL import Image

import ale_py

gym.register_envs(ale_py)

OUT_DIR = Path(
    "/Users/protiknag/Desktop/Continual Reinforcement Learning with Minimax Algorithm/presentation_figures"
)
PNG_SCALE = 8  # 160x210 -> 1280x1680


def set_seed(seed: int) -> None:
    """Seed every source of randomness used during frame collection."""
    random.seed(seed)
    np.random.seed(seed)


def runs_to_rects(frame: np.ndarray) -> list[tuple[int, int, int, int, tuple[int, int, int]]]:
    """Convert an RGB frame into merged (x, y, w, h, color) rectangles.

    Rows are run-length encoded, then a run is extended downward whenever the row
    below carries an identical (x, width, color) run. Complexity is O(H*W) time and
    O(W) space for the open-run table.
    """
    height, width, _ = frame.shape
    packed = (
        frame[:, :, 0].astype(np.int32) << 16
    ) | (frame[:, :, 1].astype(np.int32) << 8) | frame[:, :, 2].astype(np.int32)

    rects: list[tuple[int, int, int, int, tuple[int, int, int]]] = []
    open_runs: dict[tuple[int, int, int], int] = {}  # (x, w, color) -> y_start

    for y in range(height):
        row = packed[y]
        # Boundaries where the colour changes along the row.
        change = np.flatnonzero(row[1:] != row[:-1]) + 1
        starts = np.concatenate(([0], change))
        ends = np.concatenate((change, [width]))

        current: dict[tuple[int, int, int], int] = {}
        for x_start, x_end in zip(starts.tolist(), ends.tolist()):
            key = (x_start, x_end - x_start, int(row[x_start]))
            current[key] = open_runs.get(key, y)

        # Any run that did not continue into this row is now a finished rectangle.
        for key, y_start in open_runs.items():
            if key not in current:
                x_start, run_width, color = key
                rects.append(
                    (
                        x_start,
                        y_start,
                        run_width,
                        y - y_start,
                        ((color >> 16) & 255, (color >> 8) & 255, color & 255),
                    )
                )
        open_runs = current

    for (x_start, run_width, color), y_start in open_runs.items():
        rects.append(
            (
                x_start,
                y_start,
                run_width,
                height - y_start,
                ((color >> 16) & 255, (color >> 8) & 255, color & 255),
            )
        )
    return rects


def frame_to_svg(frame: np.ndarray, title: str, scale: int = PNG_SCALE) -> str:
    """Emit a self-contained SVG string that reproduces the frame exactly."""
    height, width, _ = frame.shape
    rects = runs_to_rects(frame)

    # Largest-area colour becomes the background rect so it need not be emitted per-run.
    areas: dict[tuple[int, int, int], int] = {}
    for _, _, w, h, color in rects:
        areas[color] = areas.get(color, 0) + w * h
    background = max(areas, key=areas.get)

    def hexcolor(rgb: tuple[int, int, int]) -> str:
        return "#{:02X}{:02X}{:02X}".format(*rgb)

    body = [
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="{hexcolor(background)}"/>'
    ]
    # Group by colour so the fill attribute is written once per colour, not per rect.
    by_color: dict[tuple[int, int, int], list[tuple[int, int, int, int]]] = {}
    for x, y, w, h, color in rects:
        if color == background:
            continue
        by_color.setdefault(color, []).append((x, y, w, h))

    for color, items in sorted(by_color.items(), key=lambda kv: -len(kv[1])):
        parts = "".join(f'<rect x="{x}" y="{y}" width="{w}" height="{h}"/>' for x, y, w, h in items)
        body.append(f'<g fill="{hexcolor(color)}">{parts}</g>')

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width * scale}" height="{height * scale}" '
        f'shape-rendering="crispEdges" role="img" aria-label="{title}">\n'
        f"<title>{title}</title>\n" + "\n".join(body) + "\n</svg>\n"
    )


def collect_frame(
    env_id: str,
    seed: int,
    warmup_steps: int,
    accept: Optional[Callable[[np.ndarray], bool]] = None,
    max_extra_steps: int = 400,
    fire_every: int = 0,
) -> np.ndarray:
    """Roll out a seeded random policy and return a mid-game RGB frame.

    `accept` gates which frames qualify (e.g. ball currently on screen); if no frame
    qualifies within `max_extra_steps` past the warm-up, the last frame is returned.
    `fire_every` periodically issues the FIRE action, needed to serve in Breakout.
    """
    env = gym.make(env_id, render_mode="rgb_array", full_action_space=False)
    env.reset(seed=seed)
    env.action_space.seed(seed)

    frame = np.zeros((210, 160, 3), dtype=np.uint8)
    for step in range(warmup_steps + max_extra_steps):
        action = 1 if (fire_every and step % fire_every == 0) else env.action_space.sample()
        frame, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            env.reset(seed=seed + step)
        if step >= warmup_steps and (accept is None or accept(frame)):
            break
    env.close()
    return np.asarray(frame, dtype=np.uint8)


def ball_visible(region: tuple[int, int, int, int], color: tuple[int, int, int]) -> Callable:
    """Build a predicate testing for a bright object inside a bounding box."""
    y0, y1, x0, x1 = region

    def predicate(frame: np.ndarray) -> bool:
        patch = frame[y0:y1, x0:x1]
        hit = np.all(patch == np.array(color, dtype=np.uint8), axis=-1)
        return bool(hit.sum() >= 2)

    return predicate


def pong_rally_state(frame: np.ndarray) -> bool:
    """True when both paddles sit fully inside the court and the ball is mid-flight."""
    court = frame[34:194]
    ball = np.all(court[:, 20:140] == np.array([236, 236, 236], dtype=np.uint8), axis=-1)
    if ball.sum() < 2:
        return False
    ball_x = np.flatnonzero(ball.any(axis=0))
    if ball_x.min() < 25 or ball_x.max() > 115:  # keep the ball away from either paddle
        return False

    player = np.all(court == np.array([92, 186, 92], dtype=np.uint8), axis=-1)
    opponent = np.all(court == np.array([213, 130, 74], dtype=np.uint8), axis=-1)
    if player.sum() < 40 or opponent.sum() < 40:
        return False
    # Both paddles clear of the top and bottom walls, so neither is clipped.
    player_rows = np.flatnonzero(player.any(axis=1))
    opponent_rows = np.flatnonzero(opponent.any(axis=1))
    return bool(
        player_rows.min() > 6
        and player_rows.max() < court.shape[0] - 8
        and opponent_rows.min() > 6
        and opponent_rows.max() < court.shape[0] - 8
    )


def breakout_mid_game(min_broken_fraction: float = 0.06) -> Callable[[np.ndarray], bool]:
    """Require a partially demolished brick wall plus the ball in open play."""

    def predicate(frame: np.ndarray) -> bool:
        wall = frame[57:93, 8:152]
        intact = np.any(wall != 0, axis=-1).mean()
        # Rows of the wall are solid at reset, so intact ~1.0 until bricks are cleared.
        if intact > 1.0 - min_broken_fraction:
            return False
        play = frame[93:188, 8:152]
        ball = np.all(play == np.array([200, 72, 72], dtype=np.uint8), axis=-1)
        return bool(ball.sum() >= 2)

    return predicate


BREAKOUT_OBJECT_COLOR = np.array([200, 72, 72], dtype=np.uint8)


def _breakout_x_center(region: np.ndarray) -> Optional[float]:
    """Horizontal centre of the ball/paddle blob inside a cropped region, if present."""
    mask = np.all(region == BREAKOUT_OBJECT_COLOR, axis=-1)
    if mask.sum() < 2:
        return None
    return float(np.flatnonzero(mask.any(axis=0)).mean())


def collect_breakout_frame(seed: int, min_broken_fraction: float, max_steps: int = 4000) -> np.ndarray:
    """Play Breakout with a ball-tracking heuristic until the wall is visibly chipped.

    A random policy almost never returns the ball, so the wall stays intact. Tracking
    the ball horizontally with the paddle clears enough bricks to give a screen that
    actually reads as mid-game.
    """
    env = gym.make("ALE/Breakout-v5", render_mode="rgb_array", full_action_space=False)
    frame, _ = env.reset(seed=seed)
    best = np.asarray(frame, dtype=np.uint8)
    best_broken = 0.0

    for _ in range(max_steps):
        ball_x = _breakout_x_center(frame[93:188, 8:152])
        paddle_x = _breakout_x_center(frame[189:194, 8:152])

        if ball_x is None:
            action = 1  # serve
        elif paddle_x is None or abs(ball_x - paddle_x) < 2:
            action = 0
        else:
            action = 2 if ball_x > paddle_x else 3

        frame, _, terminated, truncated, _ = env.step(action)
        frame = np.asarray(frame, dtype=np.uint8)
        if terminated or truncated:
            frame, _ = env.reset(seed=seed)
            frame = np.asarray(frame, dtype=np.uint8)
            continue

        wall = frame[57:93, 8:152]
        broken = 1.0 - float(np.any(wall != 0, axis=-1).mean())
        play = frame[93:188, 8:152]
        ball_in_play = np.all(play == BREAKOUT_OBJECT_COLOR, axis=-1).sum() >= 2

        if ball_in_play and broken > best_broken:
            best_broken, best = broken, frame.copy()
            if broken >= min_broken_fraction:
                break

    env.close()
    print(f"  breakout wall cleared: {best_broken * 100:.1f}%")
    return best


def save_figure(frame: np.ndarray, stem: str, title: str) -> None:
    """Write the SVG (vector) and a high-resolution PNG for one frame."""
    svg_path = OUT_DIR / f"{stem}.svg"
    png_path = OUT_DIR / f"{stem}.png"
    svg_path.write_text(frame_to_svg(frame, title), encoding="utf-8")

    image = Image.fromarray(frame, mode="RGB")
    image = image.resize(
        (frame.shape[1] * PNG_SCALE, frame.shape[0] * PNG_SCALE), Image.NEAREST
    )
    image.save(png_path)
    n_colors = len(np.unique(frame.reshape(-1, 3), axis=0))
    print(
        f"{stem:24s} svg={svg_path.stat().st_size/1024:6.1f} KB  "
        f"png={image.size[0]}x{image.size[1]}  colors={n_colors}"
    )


def main() -> None:
    set_seed(0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Pong: white ball mid-court with both paddles fully inside the playing field.
    pong = collect_frame(
        "ALE/Pong-v5",
        seed=7,
        warmup_steps=260,
        accept=pong_rally_state,
        max_extra_steps=1500,
    )
    save_figure(pong, "pong_state", "Atari 2600 Pong - mid-rally state")

    # Breakout: heuristic rally play until roughly a fifth of the wall is gone.
    breakout = collect_breakout_frame(seed=3, min_broken_fraction=0.20)
    save_figure(breakout, "breakout_state", "Atari 2600 Breakout - mid-game state")

    # Space Invaders: warm up long enough for the formation to descend and shots to fly.
    invaders = collect_frame("ALE/SpaceInvaders-v5", seed=11, warmup_steps=210)
    save_figure(invaders, "space_invaders_state", "Atari 2600 Space Invaders - mid-game state")


if __name__ == "__main__":
    main()
