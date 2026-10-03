#!/usr/bin/env python3
"""Render a short, original motion-graphics trailer for this project.

Install the optional rendering dependencies with:
    python -m pip install -r requirements-video.txt

Then run:
    python scripts/create_trailer.py

The renderer streams full-HD frames directly to FFmpeg (no frame dump is kept)
and synthesizes its own quiet ambient soundtrack. It does not use external media.
"""
from __future__ import annotations

import argparse
import math
import random
import struct
import subprocess
import tempfile
import wave
from array import array
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont
import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT = 1920, 1080
FPS = 24
SCENE_LENGTH = 4.5
TRANSITION = 0.34
SCENE_NAMES = ("opening", "dataset", "workflow", "selection", "demo", "evaluation", "outro")
DURATION = SCENE_LENGTH * len(SCENE_NAMES)

# A restrained night-blue palette, with the app's blue/white identity carried through.
INK = (7, 14, 27)
PANEL = (13, 24, 43)
PANEL_RAISED = (18, 33, 57)
WHITE = (242, 247, 253)
MUTED = (145, 165, 193)
FAINT = (87, 109, 141)
BLUE = (75, 157, 255)
CYAN = (102, 226, 224)
MINT = (120, 235, 206)
AMBER = (255, 194, 111)
GRID = (35, 55, 81)

FONT_ROOT = Path("/usr/share/fonts/truetype/dejavu")
FONT_FILES = {
    "regular": FONT_ROOT / "DejaVuSans.ttf",
    "bold": FONT_ROOT / "DejaVuSans-Bold.ttf",
    "mono": FONT_ROOT / "DejaVuSansMono.ttf",
    "mono_bold": FONT_ROOT / "DejaVuSansMono-Bold.ttf",
}

COHORTS = (
    ("CLEVELAND", 303),
    ("HUNGARY", 294),
    ("VA LONG BEACH", 200),
    ("SWITZERLAND", 123),
)
CANDIDATES = (
    ("LOGISTIC REGRESSION  /  11 FIELDS", 0.816, 0.053, True),
    ("LOGISTIC REGRESSION  /  13 FIELDS", 0.815, 0.056, False),
    ("XGBOOST  /  11 FIELDS", 0.812, 0.062, False),
    ("RANDOM FOREST  /  11 FIELDS", 0.811, 0.059, False),
    ("RANDOM FOREST  /  13 FIELDS", 0.809, 0.060, False),
    ("XGBOOST  /  13 FIELDS", 0.802, 0.057, False),
)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def smooth(value: float) -> float:
    value = clamp(value)
    return value * value * (3.0 - 2.0 * value)


def ease_out(value: float) -> float:
    value = clamp(value)
    return 1.0 - (1.0 - value) ** 3


def mix(a: float, b: float, amount: float) -> float:
    return a + (b - a) * amount


def with_alpha(color: tuple[int, int, int], alpha: float) -> tuple[int, int, int, int]:
    return color[0], color[1], color[2], int(clamp(alpha, 0, 255))


@lru_cache(maxsize=64)
def font(size: int, face: str = "regular") -> ImageFont.FreeTypeFont:
    path = FONT_FILES.get(face, FONT_FILES["regular"])
    try:
        return ImageFont.truetype(str(path), size=size)
    except OSError:
        return ImageFont.load_default()


def text(draw: ImageDraw.ImageDraw, x: float, y: float, content: str,
         size: int, color: tuple[int, int, int] = WHITE, face: str = "regular",
         alpha: int = 255, anchor: str = "lt", tracking: float = 0.0) -> None:
    f = font(size, face)
    fill = with_alpha(color, alpha)
    if tracking:
        cursor = float(x)
        for character in content:
            draw.text((round(cursor), round(y)), character, font=f, fill=fill, anchor=anchor)
            cursor += draw.textlength(character, font=f) + tracking
    else:
        draw.text((round(x), round(y)), content, font=f, fill=fill, anchor=anchor)


def rounded(draw: ImageDraw.ImageDraw, box: tuple[float, float, float, float],
            radius: int, fill: tuple[int, int, int] | None,
            outline: tuple[int, int, int] | None = None, width: int = 1,
            alpha: int = 255) -> None:
    draw.rounded_rectangle(
        tuple(round(v) for v in box), radius=radius,
        fill=with_alpha(fill, alpha) if fill else None,
        outline=with_alpha(outline, alpha) if outline else None,
        width=width,
    )


def line(draw: ImageDraw.ImageDraw, points: list[tuple[float, float]],
         color: tuple[int, int, int], width: int = 1, alpha: int = 255,
         joint: str | None = "curve") -> None:
    draw.line([(round(x), round(y)) for x, y in points], fill=with_alpha(color, alpha),
              width=width, joint=joint)


def pill(draw: ImageDraw.ImageDraw, x: int, y: int, label: str,
         fill: tuple[int, int, int], ink: tuple[int, int, int] = WHITE,
         size: int = 17, pad_x: int = 13, pad_y: int = 8,
         outline: tuple[int, int, int] | None = None) -> tuple[int, int]:
    f = font(size, "mono_bold")
    width = round(draw.textlength(label, font=f) + pad_x * 2)
    height = size + pad_y * 2 + 2
    rounded(draw, (x, y, x + width, y + height), height // 2, fill, outline, alpha=238)
    text(draw, x + pad_x, y + pad_y - 1, label, size, ink, "mono_bold")
    return width, height


def tracked_label(draw: ImageDraw.ImageDraw, x: int, y: int, label: str,
                  color: tuple[int, int, int] = CYAN, size: int = 16,
                  alpha: int = 255, tracking: float = 2.2) -> None:
    text(draw, x, y, label, size, color, "mono_bold", alpha, tracking=tracking)


def make_glow(cx: int, cy: int, radius: int, color: tuple[int, int, int], strength: int) -> Image.Image:
    layer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    # Soft nested discs give a luminous falloff without a hard spotlight edge.
    for i in range(32, 0, -1):
        r = round(radius * i / 32)
        a = round(strength * ((33 - i) / 32) ** 1.7)
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(*color, a))
    return layer.filter(ImageFilter.GaussianBlur(radius=74))


def build_backgrounds() -> tuple[Image.Image, ...]:
    base = Image.new("RGB", (WIDTH, HEIGHT))
    d = ImageDraw.Draw(base)
    top = (6, 13, 25)
    bottom = (11, 22, 41)
    for y in range(HEIGHT):
        p = y / (HEIGHT - 1)
        c = tuple(round(mix(top[i], bottom[i], p)) for i in range(3))
        d.line((0, y, WIDTH, y), fill=c)

    # A very fine editorial grid, plus sparse registration marks.
    grid_layer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    gd = ImageDraw.Draw(grid_layer)
    for x in range(96, WIDTH, 96):
        gd.line((x, 0, x, HEIGHT), fill=(*GRID, 25), width=1)
    for y in range(94, HEIGHT, 94):
        gd.line((0, y, WIDTH, y), fill=(*GRID, 22), width=1)
    rng = random.Random(91)
    for _ in range(900):
        x = rng.randrange(80, WIDTH - 80)
        y = rng.randrange(145, HEIGHT - 80)
        r = rng.choice((1, 1, 1, 2))
        gd.ellipse((x - r, y - r, x + r, y + r), fill=(104, 157, 221, rng.randrange(12, 31)))
    # Corner calibration brackets add a little film/measurement language.
    for x, y, sx, sy in ((76, 150, 1, 1), (WIDTH - 76, 150, -1, 1),
                         (76, HEIGHT - 78, 1, -1), (WIDTH - 76, HEIGHT - 78, -1, -1)):
        gd.line((x, y, x + sx * 22, y), fill=(85, 130, 186, 65), width=2)
        gd.line((x, y, x, y + sy * 22), fill=(85, 130, 186, 65), width=2)
    base = Image.alpha_composite(base.convert("RGBA"), grid_layer)

    # Per-scene ambient lights, kept deliberately subtle.
    glow_specs = (
        ((1460, 510, 530, BLUE, 28), (1660, 220, 370, CYAN, 18)),
        ((1490, 570, 560, BLUE, 25), (490, 780, 440, CYAN, 13)),
        ((1050, 560, 620, BLUE, 20), (1640, 430, 430, CYAN, 18)),
        ((1470, 500, 540, BLUE, 25), (540, 760, 400, CYAN, 14)),
        ((1420, 570, 580, BLUE, 22), (330, 330, 440, CYAN, 13)),
        ((720, 500, 560, BLUE, 19), (1580, 580, 470, CYAN, 13)),
        ((950, 530, 720, BLUE, 24), (1560, 450, 500, CYAN, 20)),
    )
    backgrounds: list[Image.Image] = []
    for spec in glow_specs:
        layer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
        for cx, cy, radius, color, strength in spec:
            layer = Image.alpha_composite(layer, make_glow(cx, cy, radius, color, strength))
        backgrounds.append(Image.alpha_composite(base, layer).convert("RGB"))
    return tuple(backgrounds)


BACKGROUNDS = build_backgrounds()


def draw_particles(draw: ImageDraw.ImageDraw, t: float, scene: int) -> None:
    rng = random.Random(680 + scene)
    for i in range(42):
        x0 = rng.randrange(110, WIDTH - 100)
        y0 = rng.randrange(160, HEIGHT - 100)
        speed = 7 + (i % 5) * 3
        x = 105 + ((x0 - 105 + t * speed) % (WIDTH - 210))
        y = y0 + math.sin(t * 0.55 + i * 1.7) * (4 + i % 5)
        radius = 1 + (i % 4 == 0)
        alpha = 16 + int((math.sin(t * 1.4 + i) + 1) * 9)
        draw.ellipse((round(x - radius), round(y - radius), round(x + radius), round(y + radius)),
                     fill=with_alpha(CYAN if i % 3 == 0 else BLUE, alpha))


def draw_chrome(image: Image.Image, scene: int, local_time: float, global_progress: float) -> None:
    d = ImageDraw.Draw(image, "RGBA")
    # Compact project mark: an abstract pulse, not a medical or institutional logo.
    rounded(d, (82, 43, 132, 93), 14, BLUE, (134, 190, 255), 1, 242)
    points = [(92, 69), (101, 69), (106, 58), (113, 80), (119, 65), (125, 65)]
    line(d, points, WHITE, 3, 245)
    tracked_label(d, 151, 56, "HEART DISEASE  /  MODEL DEMO", WHITE, 17, 236, 1.2)
    text(d, 1817, 58, "INDEPENDENT EDUCATIONAL PROJECT", 15, MUTED, "mono", 225, anchor="rt")
    d.line((82, 115, 1838, 115), fill=(86, 115, 153, 66), width=1)
    d.rounded_rectangle((82, 113, 1838, 117), radius=2, fill=(29, 47, 72, 150))
    bar_end = 82 + round(1756 * clamp(global_progress))
    if bar_end > 82:
        d.rounded_rectangle((82, 113, bar_end, 117), radius=2, fill=with_alpha(CYAN, 216))
    text(d, 1837, 135, f"{scene + 1:02d}  /  07", 15, FAINT, "mono", 210, anchor="rt")


def enter(local_time: float, delay: float = 0.0, duration: float = 0.72) -> float:
    return ease_out((local_time - delay) / duration)


def section_label(draw: ImageDraw.ImageDraw, number: str, label: str,
                   local_time: float, x: int = 138, y: int = 169) -> None:
    p = enter(local_time, 0.06, 0.5)
    tracked_label(draw, x, y + round((1 - p) * 10), f"{number}  /  {label}", CYAN, 17, round(245 * p), 2.0)


def heading(draw: ImageDraw.ImageDraw, x: int, y: int, content: str,
            size: int, local_time: float, color: tuple[int, int, int] = WHITE,
            face: str = "bold", delay: float = 0.13, alpha: int = 255) -> None:
    p = enter(local_time, delay, 0.72)
    text(draw, x, y + round((1 - p) * 24), content, size, color, face,
         round(alpha * p))


def orbit_points(cx: float, cy: float, rx: float, ry: float,
                 phase: float, count: int = 180, tilt: float = 0.0) -> list[tuple[float, float]]:
    out = []
    ca, sa = math.cos(tilt), math.sin(tilt)
    for i in range(count + 1):
        a = 2 * math.pi * i / count + phase
        x, y = rx * math.cos(a), ry * math.sin(a)
        out.append((cx + x * ca - y * sa, cy + x * sa + y * ca))
    return out


def ecg_wave(x0: float, y0: float, width: float, scale: float,
             travel: float = 0.0, cycles: int = 3) -> list[tuple[float, float]]:
    pulse = ((0, 0), (22, 0), (31, -5), (38, 11), (45, -15), (52, 33),
             (60, -58), (69, 91), (77, 0), (106, 0), (118, -7),
             (126, -14), (136, 0), (178, 0))
    cycle_width = width / cycles
    pts: list[tuple[float, float]] = []
    for cycle in range(cycles):
        start = x0 + cycle * cycle_width - travel
        factor = cycle_width / 178
        for px, py in pulse:
            x = start + px * factor
            if x0 - 4 <= x <= x0 + width + 4:
                pts.append((x, y0 + py * scale))
    return pts


def draw_opening(image: Image.Image, t: float, progress: float) -> None:
    d = ImageDraw.Draw(image, "RGBA")
    draw_chrome(image, 0, t, progress)
    draw_particles(d, t, 0)
    p = enter(t, 0.18, 1.0)
    section_label(d, "01", "THE QUESTION", t, 138, 184)

    # Orbital data mark on the right: a signal moving through a small measured system.
    cx, cy = 1445, 566
    for k, (rx, ry, tilt, color, alpha) in enumerate(((285, 190, 0.18, BLUE, 80),
                                                       (235, 305, -0.38, CYAN, 66),
                                                       (328, 225, 0.73, BLUE, 42))):
        pts = orbit_points(cx, cy, rx, ry, t * (0.08 + k * 0.025), 180, tilt)
        line(d, pts, color, 1 if k == 2 else 2, alpha)
        theta = (t * (0.55 + k * 0.13) + k * 1.9) % (2 * math.pi)
        px = cx + rx * math.cos(theta) * math.cos(tilt) - ry * math.sin(theta) * math.sin(tilt)
        py = cy + rx * math.cos(theta) * math.sin(tilt) + ry * math.sin(theta) * math.cos(tilt)
        d.ellipse((px - 6, py - 6, px + 6, py + 6), fill=with_alpha(CYAN if k == 1 else BLUE, 235))
    for r, alpha in ((118, 20), (86, 30), (53, 40)):
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(32, 105, 170, alpha),
                  outline=with_alpha(BLUE, alpha + 55), width=1)
    text(d, cx, cy - 47, "920", 46, WHITE, "bold", 240, anchor="mt")
    tracked_label(d, cx, cy + 17, "RECORDS", CYAN, 14, 232, 1.6)
    # Clean ECG motif layered over the orbital geometry.
    waveform = ecg_wave(1110, 570, 670, 0.82, (t * 74) % 220, 4)
    if len(waveform) > 1:
        line(d, waveform, BLUE, 13, 34)
        line(d, waveform, CYAN, 3, 220)

    # Kinetic title block.
    x = 142 + round((1 - p) * 32)
    text(d, x, 320, "HEART", 130, WHITE, "bold", round(255 * p))
    text(d, x, 454, "DISEASE", 130, WHITE, "bold", round(255 * p))
    rounded(d, (x + 5, 608, x + 215, 614), 3, CYAN, None, alpha=round(230 * p))
    tracked_label(d, x + 5, 642, "PREDICTION DEMO", CYAN, 22, round(250 * p), 2.2)
    text(d, x + 5, 704, "A historical dataset. A transparent model.", 30,
         (208, 220, 237), "regular", round(246 * p))
    text(d, x + 5, 754, "Built to explore patterns—not to assess a person.", 23,
         MUTED, "regular", round(230 * p))

    tag_y = 847
    tag_specs = (("920 RECORDS", 225), ("4 COHORTS", 210), ("11 CORE FIELDS", 260))
    tag_x = x + 5
    for i, (label, width_hint) in enumerate(tag_specs):
        q = enter(t, 0.56 + i * 0.09, 0.6)
        bg = PANEL_RAISED
        width, _ = pill(d, tag_x, tag_y + round((1 - q) * 12), label, bg, WHITE,
                        16, 14, 9, (54, 91, 137))
        tag_x += max(width, width_hint) + 12
    text(d, 138, 967, "UCI HEART DISEASE  ·  HISTORICAL COHORTS", 15, FAINT, "mono", 190)
    pill(d, 1497, 950, "EDUCATIONAL ONLY", (50, 40, 29), AMBER, 15, 14, 8,
         (126, 91, 48))


def draw_dataset(image: Image.Image, t: float, progress: float) -> None:
    d = ImageDraw.Draw(image, "RGBA")
    draw_chrome(image, 1, t, progress)
    draw_particles(d, t, 1)
    section_label(d, "02", "THE DATA", t)
    heading(d, 138, 218, "A 920-record view across four cohorts.", 63, t)
    text(d, 141, 306,
         "Historical UCI records, kept with their source-cohort labels for evaluation.",
         26, (184, 201, 223), "regular", round(245 * enter(t, 0.28, 0.7)))

    # Oversized record count and a slim reference rule.
    count_p = ease_out((t - 0.45) / 1.55)
    rounded(d, (137, 421, 626, 822), 28, PANEL, (50, 77, 111), 1, 225)
    text(d, 177, 445, f"{round(920 * count_p):,}", 174, WHITE, "bold", 255)
    tracked_label(d, 182, 641, "SOURCE ROWS", CYAN, 17, 235, 2.1)
    text(d, 182, 688, "4 preserved cohorts", 24, MUTED, "regular", 238)
    # Index-style ticks connect the total to the cohort breakdown.
    for i in range(13):
        h = 12 if i % 3 else 22
        d.line((182 + i * 31, 768, 182 + i * 31, 768 - h),
               fill=with_alpha(BLUE if i < round(13 * count_p) else GRID, 200), width=3)
    text(d, 182, 783, "PROCESSED UCI TABLES", 13, FAINT, "mono", 220)

    # Cohort distribution as four animated rails; widths match the row counts.
    x0, x1 = 718, 1757
    max_width = x1 - x0 - 110
    bar_p = ease_out((t - 0.48) / 1.12)
    for i, (name, count) in enumerate(COHORTS):
        y = 423 + i * 105
        row_p = enter(t, 0.28 + i * 0.11, 0.68)
        text(d, x0 + round((1 - row_p) * 26), y, name, 21, WHITE, "mono_bold", round(246 * row_p))
        val = round(count * ease_out((t - 0.5 - i * 0.08) / 1.3))
        text(d, x1, y - 2, f"{val:03d}", 23, CYAN, "mono_bold", round(250 * row_p), anchor="rt")
        rounded(d, (x0, y + 38, x1, y + 49), 5, (23, 37, 59), (41, 62, 91), 1, 220)
        barw = max_width * (count / 303) * bar_p
        if barw > 2:
            rounded(d, (x0 + 1, y + 39, x0 + barw, y + 48), 4,
                    CYAN if i == 0 else BLUE, None, alpha=round(220 * row_p))
        # Tiny segmented highlights echo individual rows without showing false granularity.
        for j in range(1, 13):
            xx = x0 + barw * j / 13
            if xx < x0 + max_width:
                d.line((xx, y + 37, xx, y + 50), fill=(5, 14, 27, 85), width=1)
    text(d, 720, 868,
         "Cohort membership is used for grouped validation—not as a model input.",
         22, (184, 201, 223), "regular", 240)
    text(d, 138, 966, "CLEVELAND  ·  HUNGARY  ·  SWITZERLAND  ·  VA LONG BEACH", 15,
         FAINT, "mono", 205)
    tracked_label(d, 1607, 966, "UCI  /  920 ROWS", CYAN, 13, 205, 1.1)


def draw_workflow(image: Image.Image, t: float, progress: float) -> None:
    d = ImageDraw.Draw(image, "RGBA")
    draw_chrome(image, 2, t, progress)
    draw_particles(d, t, 2)
    section_label(d, "03", "THE WORKFLOW", t)
    heading(d, 138, 218, "Validation begins before the model.", 63, t)
    text(d, 141, 306, "Each step is visible, repeatable, and designed to keep the test split in its place.",
         25, (184, 201, 223), "regular", round(244 * enter(t, 0.28, 0.7)))

    boxes = ((138, 405, 570, 765), (744, 405, 1176, 765), (1350, 405, 1782, 765))
    for i, box in enumerate(boxes):
        p = enter(t, 0.22 + i * 0.14, 0.78)
        x0, y0, x1, y1 = box
        rounded(d, (x0, y0 + round((1 - p) * 20), x1, y1 + round((1 - p) * 20)),
                24, PANEL, (54, 79, 112), 1, round(232 * p))
        # A small leading step marker.
        d.ellipse((x0 + 27, y0 + 27, x0 + 48, y0 + 48),
                  fill=with_alpha(CYAN if i == 2 else BLUE, round(220 * p)))
        text(d, x0 + 39, y0 + 36, f"{i + 1:02d}", 11, INK, "mono_bold", round(255 * p), anchor="mm")

    # First card: source data.
    x0, y0, _, _ = boxes[0]
    tracked_label(d, x0 + 75, y0 + 27, "SOURCE DATA", CYAN, 15, 234, 1.8)
    text(d, x0 + 31, y0 + 100, "920", 96, WHITE, "bold", round(255 * enter(t, 0.35, 0.9)))
    text(d, x0 + 35, y0 + 207, "historical records", 21, MUTED, "regular", 240)
    # A compact table glyph.
    gx, gy = x0 + 33, y0 + 260
    for r in range(3):
        for c in range(7):
            color = CYAN if (r == 0 and c < 3) else BLUE
            rounded(d, (gx + c * 44, gy + r * 17, gx + c * 44 + 28, gy + r * 17 + 5),
                    2, color, None, alpha=115 + (c + r) % 3 * 22)
    pill(d, x0 + 30, y0 + 315, "4 COHORTS", PANEL_RAISED, WHITE, 14, 12, 7, (57, 82, 116))

    # Second card: pipeline, with the three operations laid out as a sequence.
    x0, y0, _, _ = boxes[1]
    tracked_label(d, x0 + 75, y0 + 27, "PREPROCESSING", CYAN, 15, 234, 1.8)
    text(d, x0 + 32, y0 + 96, "Inside every fold", 31, WHITE, "bold", 249)
    steps = (("01", "IMPUTE"), ("02", "ENCODE"), ("03", "SCALE"))
    step_y = y0 + 168
    for i, (num, label) in enumerate(steps):
        q = enter(t, 0.58 + i * 0.13, 0.65)
        rounded(d, (x0 + 33, step_y + i * 47, x0 + 400, step_y + i * 47 + 34),
                9, PANEL_RAISED, (48, 71, 102), 1, round(238 * q))
        text(d, x0 + 48, step_y + i * 47 + 8, num, 14, CYAN, "mono", round(245 * q))
        text(d, x0 + 101, step_y + i * 47 + 6, label, 18, WHITE, "mono_bold", round(245 * q))
        text(d, x0 + 365, step_y + i * 47 + 6, "✓", 18, MINT, "bold", round(225 * q), anchor="rt")
    text(d, x0 + 35, y0 + 328, "Fit on training folds only", 18, MUTED, "regular", 235)

    # Third card: grouped CV and separate patient-level test set.
    x0, y0, _, _ = boxes[2]
    tracked_label(d, x0 + 75, y0 + 27, "EVALUATION", CYAN, 15, 234, 1.8)
    text(d, x0 + 32, y0 + 103, "GroupKFold", 42, WHITE, "bold", 255)
    text(d, x0 + 35, y0 + 165, "by source cohort", 22, CYAN, "mono", 241)
    # Four cohort columns, one is held aside per validation fold.
    gx, gy = x0 + 38, y0 + 229
    for c in range(4):
        held = (round(t * 0.75) % 4 == c)
        color = AMBER if held else BLUE
        rounded(d, (gx + c * 81, gy, gx + c * 81 + 62, gy + 30), 8,
                color, None, alpha=195 if held else 130)
    text(d, x0 + 35, y0 + 282, "A separate patient-level test split", 18, MUTED, "regular", 236)
    text(d, x0 + 35, y0 + 315, "stays out of model selection.", 18, MUTED, "regular", 236)

    # Animated flow pulses between the stages.
    flow = (t * 210) % 1.0
    for start_x, end_x in ((570, 744), (1176, 1350)):
        y = 585
        d.line((start_x + 12, y, end_x - 12, y), fill=(62, 99, 143, 86), width=2)
        for k in range(3):
            q = (flow + k / 3) % 1
            px = mix(start_x + 15, end_x - 15, q)
            d.ellipse((px - 4, y - 4, px + 4, y + 4), fill=with_alpha(CYAN, 235 - k * 30))

    rounded(d, (138, 833, 1782, 910), 17, (11, 22, 39), (48, 72, 104), 1, 218)
    text(d, 172, 857,
         "Source-cohort labels are validation groups—not model features.",
         23, WHITE, "regular", 245)
    pill(d, 1433, 849, "LEAKAGE-AWARE", (21, 51, 66), CYAN, 14, 12, 8, (52, 111, 119))
    text(d, 138, 966, "REPRODUCIBLE PIPELINES  ·  COHORT-AWARE CROSS-VALIDATION", 15,
         FAINT, "mono", 205)


def draw_selection(image: Image.Image, t: float, progress: float) -> None:
    d = ImageDraw.Draw(image, "RGBA")
    draw_chrome(image, 3, t, progress)
    draw_particles(d, t, 3)
    section_label(d, "04", "MODEL SELECTION", t)
    heading(d, 138, 212, "Six candidates. One restrained choice.", 61, t)
    text(d, 141, 294, "Grouped cross-validation by cohort; the patient test split did not choose the model.",
         24, (184, 201, 223), "regular", round(243 * enter(t, 0.27, 0.7)))

    # Ranking panel.
    left = (137, 365, 1240, 843)
    p = enter(t, 0.22, 0.72)
    rounded(d, left, 23, PANEL, (48, 73, 105), 1, round(234 * p))
    tracked_label(d, 171, 393, "CANDIDATE  /  FEATURE SET", MUTED, 13, 218, 1.3)
    tracked_label(d, 715, 393, "GROUPED-CV ROC-AUC", MUTED, 13, 218, 1.3)
    chart_x, chart_w = 710, 360
    value_x = 1174
    y0, row_h = 448, 57
    for i, (name, score, sd, selected) in enumerate(CANDIDATES):
        row_y = y0 + i * row_h
        row_progress = enter(t, 0.38 + i * 0.07, 0.72)
        if selected:
            rounded(d, (157, row_y - 8, 1220, row_y + 40), 10,
                    (18, 51, 66), (67, 167, 166), 1, round(232 * row_progress))
        text(d, 173 + round((1 - row_progress) * 10), row_y, name, 16,
             WHITE if selected else (190, 204, 224), "mono_bold" if selected else "mono",
             round(242 * row_progress))
        rounded(d, (chart_x, row_y + 12, chart_x + chart_w, row_y + 25), 6,
                (28, 43, 64), None, alpha=205)
        normalized = clamp((score - 0.78) / 0.04)
        bar_width = chart_w * normalized * row_progress
        if bar_width > 1:
            rounded(d, (chart_x, row_y + 12, chart_x + bar_width, row_y + 25), 6,
                    CYAN if selected else BLUE, None, alpha=round((238 if selected else 166) * row_progress))
        color = CYAN if selected else (184, 201, 222)
        text(d, value_x, row_y - 2, f"{score:.3f} ± {sd:.3f}", 16, color,
             "mono_bold" if selected else "mono", round(245 * row_progress), anchor="rt")
    # Honest axis labels, so the tight score differences are clear.
    axis_y = 803
    for fraction, label in ((0, "0.78"), (0.5, "0.80"), (1, "0.82")):
        x = chart_x + chart_w * fraction
        d.line((x, axis_y - 7, x, axis_y), fill=(101, 127, 160, 140), width=1)
        text(d, x, axis_y + 3, label, 13, FAINT, "mono", 215, anchor="mt")

    # Selection rationale panel.
    right = (1282, 365, 1782, 843)
    rounded(d, right, 23, (16, 30, 52), (57, 84, 119), 1, round(237 * p))
    tracked_label(d, 1323, 399, "SELECTED PIPELINE", CYAN, 14, 239, 1.9)
    text(d, 1323, 450, "LOGISTIC", 41, WHITE, "bold", round(255 * enter(t, 0.34, 0.7)))
    text(d, 1323, 500, "REGRESSION", 41, WHITE, "bold", round(255 * enter(t, 0.4, 0.7)))
    pill(d, 1324, 568, "11 CORE FIELDS", (22, 54, 74), CYAN, 15, 13, 8, (56, 111, 137))
    d.line((1322, 626, 1740, 626), fill=(62, 86, 116, 165), width=1)
    text(d, 1324, 647, "0.816", 61, CYAN, "bold", round(255 * enter(t, 0.48, 0.8)))
    text(d, 1518, 674, "± 0.053", 22, WHITE, "mono", 239)
    text(d, 1327, 716, "Grouped-CV ROC-AUC", 18, MUTED, "regular", 239)
    text(d, 1327, 749, "One-standard-error rule; then prefer", 16, (192, 207, 226), "regular", 232)
    text(d, 1327, 774, "fewer fields and a simpler model.", 16, (192, 207, 226), "regular", 232)

    text(d, 138, 885,
         "The held-out patient split is reserved for reporting—not selection.",
         21, (207, 219, 235), "regular", 239)
    pill(d, 1516, 867, "NO TEST-SET TUNING", (43, 37, 30), AMBER, 14, 12, 8, (116, 87, 51))
    text(d, 138, 966, "SIX PIPELINES  ·  ONE-STANDARD-ERROR SELECTION  ·  SEED 42", 15,
         FAINT, "mono", 205)


def draw_demo(image: Image.Image, t: float, progress: float) -> None:
    d = ImageDraw.Draw(image, "RGBA")
    draw_chrome(image, 4, t, progress)
    draw_particles(d, t, 4)
    section_label(d, "05", "THE DEMO", t)
    heading(d, 138, 207, "Explore a historical model.", 61, t)
    text(d, 141, 287, "Not a personal risk score. Fictional values only.", 26,
         (184, 201, 223), "regular", round(244 * enter(t, 0.28, 0.7)))

    # A carefully composed miniature of the real blue-and-white browser UI.
    px0, py0, px1, py1 = 138, 359, 1782, 911
    p = enter(t, 0.24, 0.78)
    rounded(d, (px0 + round((1 - p) * 22), py0, px1 + round((1 - p) * 22), py1),
            22, (244, 248, 253), (159, 181, 207), 1, round(250 * p))
    rounded(d, (px0 + round((1 - p) * 22), py0, px1 + round((1 - p) * 22), py0 + 48),
            22, (231, 238, 248), None, alpha=round(250 * p))
    # Mask the lower half of the browser bar's rounded corners for a square divider.
    d.rectangle((px0 + round((1 - p) * 22), py0 + 25, px1 + round((1 - p) * 22), py0 + 49),
                fill=with_alpha((231, 238, 248), round(250 * p)))
    for i, color in enumerate(((240, 112, 107), (238, 190, 96), (92, 190, 145))):
        d.ellipse((px0 + 24 + i * 18, py0 + 17, px0 + 33 + i * 18, py0 + 26), fill=with_alpha(color, 230))
    rounded(d, (px0 + 104, py0 + 10, px0 + 648, py0 + 38), 9,
            (247, 250, 254), (207, 219, 233), 1, 255)
    text(d, px0 + 126, py0 + 16, "heart-disease-model-demo  /  fictional example", 13,
         (103, 122, 149), "mono", 255)

    # Main app content.
    text(d, 198, 435, "HEART DISEASE MODEL DEMO", 15, (37, 99, 235), "mono_bold", 255)
    text(d, 198, 473, "Explore a historical model,", 36, (15, 23, 42), "bold", 255)
    text(d, 198, 516, "not a personal risk score", 36, (37, 99, 235), "bold", 255)
    text(d, 198, 569, "Adjust a fictional example; see the source-data label change.",
         17, (71, 85, 105), "regular", 246)

    field_specs = (
        ("AGE", "45 years"), ("RESTING PRESSURE", "120 mm Hg"),
        ("MAX. HEART RATE", "170 bpm"), ("ST CHANGE", "0.2"),
    )
    field_xs = (198, 625)
    field_ys = (626, 725)
    for i, (label, value) in enumerate(field_specs):
        col, row = i % 2, i // 2
        fx, fy = field_xs[col], field_ys[row]
        rounded(d, (fx, fy, fx + 385, fy + 76), 11, (255, 255, 255), (218, 226, 237), 1, 255)
        text(d, fx + 17, fy + 9, label, 11, (100, 116, 139), "mono_bold", 255)
        text(d, fx + 17, fy + 33, value, 19, (15, 23, 42), "regular", 255)
        d.line((fx + 354, fy + 31, fx + 360, fy + 37), fill=(133, 152, 178, 220), width=2)

    # Output card uses the verified fictional preset-A score from web/src/predict.js.
    ox0, oy0, ox1, oy1 = 1060, 452, 1718, 847
    rounded(d, (ox0, oy0, ox1, oy1), 17, (235, 243, 255), (203, 220, 245), 1, 255)
    tracked_label(d, ox0 + 29, oy0 + 28, "MODEL SCORE  /  0–1", (64, 91, 132), 13, 255, 1.25)
    pill(d, ox1 - 204, oy0 + 18, "FICTIONAL A", (222, 235, 255), (37, 99, 235), 12, 10, 6, (193, 213, 245))
    score_p = ease_out((t - 0.88) / 1.0)
    text(d, ox0 + 29, oy0 + 74, f"{0.027 * score_p:0.3f}", 86, (29, 78, 216), "bold", 255)
    text(d, ox0 + 34, oy0 + 167, "UNCALIBRATED OUTPUT", 13, (71, 85, 105), "mono_bold", 243)
    rounded(d, (ox0 + 30, oy0 + 207, ox1 - 30, oy0 + 220), 7, (255, 255, 255), (210, 224, 246), 1, 255)
    bw = (ox1 - ox0 - 62) * 0.027 * score_p
    if bw > 1:
        rounded(d, (ox0 + 31, oy0 + 208, ox0 + 31 + bw, oy0 + 219), 6, BLUE, None, alpha=245)
    pill(d, ox0 + 29, oy0 + 245, "NEGATIVE SOURCE-DATA LABEL", (37, 99, 235), WHITE, 12, 12, 8)
    text(d, ox0 + 30, oy0 + 304, "Not a percentage chance or a diagnosis.", 16,
         (71, 85, 105), "regular", 248)
    text(d, ox0 + 30, oy0 + 332, "Class at the demo's 0.50 software cutoff.", 14,
         (100, 116, 139), "regular", 242)

    text(d, 140, 949, "The browser demo runs on fictional inputs; the score is not calibrated for a person.",
         18, (192, 207, 226), "regular", 241)
    tracked_label(d, 1392, 966, "11 INPUT FIELDS  ·  FICTIONAL VALUES ONLY", MUTED, 12, 202, 1.1)


def draw_evaluation(image: Image.Image, t: float, progress: float) -> None:
    d = ImageDraw.Draw(image, "RGBA")
    draw_chrome(image, 5, t, progress)
    draw_particles(d, t, 5)
    section_label(d, "06", "THE EVALUATION", t)
    heading(d, 138, 210, "Every metric needs context.", 64, t)
    text(d, 141, 294, "A held-out patient-level software check—not an external validation study.",
         24, (184, 201, 223), "regular", round(244 * enter(t, 0.28, 0.7)))

    # Left hero metric.
    left = (138, 366, 741, 816)
    p = enter(t, 0.22, 0.72)
    rounded(d, left, 25, PANEL, (60, 87, 121), 1, round(236 * p))
    tracked_label(d, 179, 400, "HELD-OUT PATIENT SPLIT", CYAN, 14, 238, 1.65)
    metric_p = ease_out((t - 0.54) / 1.25)
    text(d, 177, 457, f"{0.905 * metric_p:0.3f}", 142, WHITE, "bold", round(255 * p))
    text(d, 183, 613, "ROC-AUC", 24, CYAN, "mono_bold", 248)
    pill(d, 180, 673, "184 TEST ROWS", PANEL_RAISED, WHITE, 14, 14, 8, (60, 88, 124))
    text(d, 183, 752, "Same four historical source cohorts", 17, MUTED, "regular", 232)

    # Supporting metrics, not over-sold as clinical performance.
    metrics = (
        ("ACCURACY", 0.826, "overall match"),
        ("RECALL", 0.882, "positive labels caught"),
        ("SPECIFICITY", 0.756, "negative labels matched"),
    )
    card_xs = (797, 1111, 1425)
    for i, (label, value, descriptor) in enumerate(metrics):
        x = card_xs[i]
        q = enter(t, 0.34 + i * 0.12, 0.75)
        rounded(d, (x, 390 + round((1 - q) * 12), x + 282, 626 + round((1 - q) * 12)),
                18, PANEL, (52, 78, 110), 1, round(235 * q))
        tracked_label(d, x + 23, 418 + round((1 - q) * 12), label, MUTED, 13, round(230 * q), 1.1)
        value_p = ease_out((t - 0.55 - i * 0.08) / 1.15)
        text(d, x + 23, 465 + round((1 - q) * 12), f"{value * value_p:.3f}", 63,
             CYAN if i == 1 else WHITE, "bold", round(250 * q))
        text(d, x + 23, 554 + round((1 - q) * 12), descriptor, 15,
             MUTED, "regular", round(235 * q))
        rounded(d, (x + 23, 592 + round((1 - q) * 12), x + 257, 597 + round((1 - q) * 12)),
                3, (29, 44, 65), None, alpha=220)
        rounded(d, (x + 23, 592 + round((1 - q) * 12),
                    x + 23 + 234 * value * value_p, 597 + round((1 - q) * 12)),
                3, BLUE if i != 1 else CYAN, None, alpha=230)

    # Error counts reinforce the threshold trade-off instead of hiding it.
    rounded(d, (797, 666, 1707, 817), 18, (16, 29, 48), (53, 77, 108), 1, 228)
    tracked_label(d, 827, 689, "AT THE 0.50 DEMO CUTOFF", AMBER, 13, 234, 1.25)
    text(d, 827, 727, "12", 36, WHITE, "bold", 255)
    text(d, 880, 741, "false negatives", 16, MUTED, "regular", 239)
    d.line((1075, 727, 1075, 785), fill=(65, 86, 113, 170), width=1)
    text(d, 1120, 727, "20", 36, WHITE, "bold", 255)
    text(d, 1173, 741, "false positives", 16, MUTED, "regular", 239)
    text(d, 1470, 739, "Software default,", 15, MUTED, "regular", 229)
    text(d, 1470, 762, "not a clinical threshold.", 15, MUTED, "regular", 229)

    rounded(d, (138, 853, 1782, 921), 16, (47, 37, 26), (125, 92, 50), 1, 225)
    text(d, 170, 875,
         "Not externally validated. Not clinically validated. Dataset-specific results only.",
         21, (255, 216, 160), "regular", 246)
    text(d, 138, 966, "184-ROW PATIENT-LEVEL HOLDOUT  ·  SAME HISTORICAL COHORTS  ·  NO CLINICAL CLAIMS", 14,
         FAINT, "mono", 204)


def draw_outro(image: Image.Image, t: float, progress: float) -> None:
    d = ImageDraw.Draw(image, "RGBA")
    draw_chrome(image, 6, t, progress)
    draw_particles(d, t, 6)
    p = enter(t, 0.16, 0.95)
    section_label(d, "07", "THE TAKEAWAY", t)

    cx, cy = 1440, 558
    for i, radius in enumerate((304, 228, 142)):
        phase = t * (0.07 + i * 0.04)
        pts = orbit_points(cx, cy, radius, radius * (0.65 + 0.08 * i), phase,
                           170, -0.25 + i * 0.38)
        line(d, pts, CYAN if i == 1 else BLUE, 2 if i < 2 else 1, 70 - i * 13)
    for i in range(30):
        angle = (i / 30) * math.tau + t * 0.09
        r = 258 + 27 * math.sin(i * 4.3)
        x, y = cx + math.cos(angle) * r, cy + math.sin(angle) * r * 0.77
        radius = 3 if i % 4 == 0 else 2
        d.ellipse((x - radius, y - radius, x + radius, y + radius),
                  fill=with_alpha(CYAN if i % 4 == 0 else BLUE, 110 + (i % 3) * 25))
    # An open ring rather than a heart icon: the work is about the data and its limits.
    for r, alpha in ((104, 19), (74, 23)):
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(30, 113, 174, alpha),
                  outline=with_alpha(CYAN, 105), width=1)

    x = 142 + round((1 - p) * 34)
    tracked_label(d, x + 4, 292, "A PROJECT IN MODEL TRANSPARENCY", CYAN, 16, round(230 * p), 2.0)
    text(d, x, 362, "PATTERNS IN", 86, WHITE, "bold", round(255 * p))
    text(d, x, 454, "HISTORY.", 86, WHITE, "bold", round(255 * p))
    text(d, x, 568, "NOT A", 72, MUTED, "bold", round(247 * p))
    text(d, x, 648, "PROGNOSIS.", 86, CYAN, "bold", round(255 * p))

    # Repo CTA, staged after the headline.
    q = enter(t, 0.58, 0.7)
    rounded(d, (x + 5, 803 + round((1 - q) * 13), x + 371, 862 + round((1 - q) * 13)),
            15, (32, 92, 152), (97, 169, 229), 1, round(246 * q))
    text(d, x + 29, 817 + round((1 - q) * 13), "EXPLORE THE PROJECT", 17, WHITE,
         "mono_bold", round(255 * q))
    text(d, x + 330, 813 + round((1 - q) * 13), "↗", 29, WHITE, "bold", round(255 * q), anchor="mt")
    text(d, x + 7, 884, "github.com/KeshavCracks/Heart-Disease-Predictor", 18,
         (184, 201, 223), "mono", round(235 * q))

    rounded(d, (1266, 898, 1782, 964), 15, (45, 36, 27), (127, 90, 47), 1, 226)
    tracked_label(d, 1291, 911, "EDUCATIONAL DEMO", AMBER, 13, 240, 1.25)
    text(d, 1291, 937, "Not clinically validated · Use fictional values", 14,
         (238, 218, 187), "regular", 238)
    text(d, 138, 983, "INDEPENDENT PROJECT  ·  HISTORICAL DATA  ·  SOURCE LABELS, NOT DIAGNOSES", 14,
         FAINT, "mono", 198)


SCENE_RENDERERS = (draw_opening, draw_dataset, draw_workflow, draw_selection,
                   draw_demo, draw_evaluation, draw_outro)


def render_frame(time_s: float) -> Image.Image:
    time_s = clamp(time_s, 0, DURATION)
    scene_index = min(int(time_s // SCENE_LENGTH), len(SCENE_NAMES) - 1)
    scene_start = scene_index * SCENE_LENGTH
    local_time = time_s - scene_start
    progress = time_s / DURATION
    current = BACKGROUNDS[scene_index].copy()
    SCENE_RENDERERS[scene_index](current, local_time, progress)

    # Cross-dissolve at the scene boundaries, preserving each scene's own motion.
    if scene_index > 0 and local_time < TRANSITION:
        previous_index = scene_index - 1
        previous_local = SCENE_LENGTH - TRANSITION + local_time
        previous = BACKGROUNDS[previous_index].copy()
        SCENE_RENDERERS[previous_index](previous, previous_local, progress)
        current = Image.blend(previous, current, smooth(local_time / TRANSITION))

    # A short, quiet end fade makes the cut intentional.
    if time_s > DURATION - 0.38:
        current = Image.blend(current, Image.new("RGB", (WIDTH, HEIGHT), (3, 7, 13)),
                              smooth((time_s - (DURATION - 0.38)) / 0.38))
    return current


def synthesize_soundtrack(path: Path, duration: float) -> None:
    """Write an original, gentle synth bed; no third-party music or samples."""
    sample_rate = 48_000
    frames = round(duration * sample_rate)
    samples = array("h")
    # A slow D-minor / Bb / F / Csus progression, four bars through the trailer.
    chords = (
        (146.83, 174.61, 220.00, 261.63),  # Dm7
        (116.54, 146.83, 174.61, 220.00),  # Bbmaj7
        (130.81, 174.61, 220.00, 261.63),  # Fadd9
        (130.81, 164.81, 196.00, 233.08),  # Csus
    )
    bar_duration = 60.0 / 78.0 * 4.0
    pulse_duration = 60.0 / 78.0
    two_pi = math.tau
    for n in range(frames):
        t = n / sample_rate
        bar_pos = t / bar_duration
        chord_idx = int(bar_pos) % len(chords)
        next_idx = (chord_idx + 1) % len(chords)
        blend = smooth((bar_pos % 1.0 - 0.72) / 0.28)
        tremolo = 0.78 + 0.22 * math.sin(two_pi * 0.11 * t)
        pad_l = pad_r = 0.0
        for voice, (fa, fb) in enumerate(zip(chords[chord_idx], chords[next_idx])):
            f = mix(fa, fb, blend)
            phase = voice * 0.41
            pad_l += math.sin(two_pi * f * t + phase) * (0.54 if voice == 0 else 0.29)
            pad_r += math.sin(two_pi * (f * 1.001) * t + phase + 0.24) * (0.54 if voice == 0 else 0.29)
        pad_l *= 0.032 * tremolo
        pad_r *= 0.032 * tremolo

        beat_phase = t % pulse_duration
        kick_env = math.exp(-beat_phase * 18.0)
        kick_freq = 50.0 - 12.0 * min(beat_phase, 0.28)
        kick = math.sin(two_pi * kick_freq * beat_phase) * kick_env * 0.12
        # A restrained glassy tick marks the beat without masking text or narration.
        tick = math.sin(two_pi * 1320 * beat_phase) * math.exp(-beat_phase * 42.0) * 0.012
        shimmer = math.sin(two_pi * 587.33 * t) * math.sin(two_pi * 0.075 * t) * 0.006

        fade_in = smooth(t / 1.25)
        fade_out = smooth((duration - t) / 2.5)
        envelope = min(fade_in, fade_out)
        left = (pad_l + kick + tick + shimmer) * envelope
        right = (pad_r + kick + tick * 0.86 + shimmer * 0.94) * envelope
        samples.append(max(-32767, min(32767, round(left * 32767))))
        samples.append(max(-32767, min(32767, round(right * 32767))))

    # array('h') is native-endian; the target sandbox is little-endian. Keep WAV portable.
    if samples.itemsize != 2:
        raise RuntimeError("This system does not provide 16-bit signed audio samples.")
    payload = samples.tobytes()
    if struct.pack("=I", 1) != struct.pack("<I", 1):
        swapped = array("h", samples)
        swapped.byteswap()
        payload = swapped.tobytes()
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(payload)


def encode_video(output: Path, audio_path: Path, fps: int = FPS) -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s:v", f"{WIDTH}x{HEIGHT}",
        "-r", str(fps), "-i", "pipe:0",
        "-i", str(audio_path),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-preset", "medium", "-crf", "19",
        "-profile:v", "high", "-level:v", "4.1", "-pix_fmt", "yuv420p",
        "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
        "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2",
        "-af", "volume=2.5", "-shortest", "-movflags", "+faststart",
        "-metadata", "title=Heart Disease Predictor — an educational ML demo",
        "-metadata", "comment=Not clinically validated. Use fictional values only.",
        str(output),
    ]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None
    total_frames = round(DURATION * fps)
    try:
        for index in range(total_frames):
            timestamp = index / fps
            frame = render_frame(timestamp).convert("RGB")
            process.stdin.write(frame.tobytes())
            if index and index % (fps * 5) == 0:
                print(f"Rendered {timestamp:05.1f}s / {DURATION:.1f}s", flush=True)
        process.stdin.close()
        error = process.stderr.read() if process.stderr else b""
        code = process.wait()
    except (BrokenPipeError, OSError):
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
        error = process.stderr.read() if process.stderr else b""
        process.wait()
        raise RuntimeError(f"FFmpeg could not finish encoding: {error.decode(errors='replace')}")
    if code:
        raise RuntimeError(f"FFmpeg exited with code {code}: {error.decode(errors='replace')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the Heart Disease Predictor trailer.")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "assets" / "heart-disease-predictor-trailer.mp4")
    parser.add_argument("--poster", type=Path,
                        default=ROOT / "assets" / "heart-disease-predictor-trailer-poster.jpg")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.poster.parent.mkdir(parents=True, exist_ok=True)

    # Use the title reveal at its most legible moment as the README preview image.
    render_frame(2.05).save(args.poster, "JPEG", quality=93, optimize=True, progressive=True)
    with tempfile.TemporaryDirectory(prefix="heart-disease-trailer-") as temp_dir:
        audio_path = Path(temp_dir) / "original-ambient-bed.wav"
        print(f"Synthesizing original ambient soundtrack ({DURATION:.1f}s)…", flush=True)
        synthesize_soundtrack(audio_path, DURATION)
        print(f"Rendering {WIDTH}×{HEIGHT} at {FPS} fps…", flush=True)
        encode_video(args.output, audio_path)
    print(f"Trailer: {args.output} ({args.output.stat().st_size / 1_000_000:.2f} MB)")
    print(f"Poster:  {args.poster}")


if __name__ == "__main__":
    main()
