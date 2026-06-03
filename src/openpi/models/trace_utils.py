"""Utilities for the TraceVLA pipeline.

Contains:
  - ``arc_length_resample``       : resample a (T, 2) polyline to (N, 2) by arc length.
  - ``time_uniform_resample``     : same, but by time index (subsampling).
  - ``resample_trace``            : dispatcher that picks an algorithm.
  - ``draw_polyline_overlay``     : draw an antialiased polyline onto an HxWx3 image array.
  - ``SKILL_TO_EXPERT``           : the canonical mapping for our 5-expert MoE.
  - ``skill_to_expert_id``        : str(skill) -> int expert id.

All NumPy-only / no JAX dependence so these can run inside torch DataLoader workers.
"""
from __future__ import annotations

from typing import Literal

import numpy as np


# Canonical skill name to expert id mapping (matches user spec):
#   PICKUP_FROM         -> 0
#   PLACE_ON / PLACE_IN -> 1
#   OPEN                -> 2
#   CLOSE               -> 3
#   TURN_ON / TURN_OFF  -> 4
SKILL_TO_EXPERT: dict[str, int] = {
    "PICKUP_FROM": 0,
    "PLACE_ON": 1,
    "PLACE_IN": 1,
    "OPEN": 2,
    "CLOSE": 3,
    "TURN_ON": 4,
    "TURN_OFF": 4,
}


def skill_to_expert_id(skill: str | None) -> int:
    """Map a skill name (possibly with parameters) to an expert id in [0, K)."""
    if skill is None:
        return 0
    s = skill.strip().upper()
    if "(" in s:
        s = s.split("(", 1)[0].strip()
    return SKILL_TO_EXPERT.get(s, 0)


# ---------------------------------------------------------------------------
# Trace resampling
# ---------------------------------------------------------------------------

def time_uniform_resample(trace_pixels: np.ndarray, n_out: int) -> np.ndarray:
    """Pick `n_out` evenly-spaced indices along the time axis."""
    trace_pixels = np.asarray(trace_pixels, dtype=np.float32)
    t = trace_pixels.shape[0]
    if t == 0:
        raise ValueError("Empty trace.")
    if t == 1:
        return np.tile(trace_pixels[0:1], (n_out, 1))
    indices = np.linspace(0.0, t - 1.0, n_out)
    floor = np.floor(indices).astype(np.int64)
    ceil = np.minimum(floor + 1, t - 1)
    frac = (indices - floor)[:, None]
    return (1.0 - frac) * trace_pixels[floor] + frac * trace_pixels[ceil]


def arc_length_resample(trace_pixels: np.ndarray, n_out: int) -> np.ndarray:
    """Resample a polyline to ``n_out`` arc-length-uniform waypoints."""
    trace_pixels = np.asarray(trace_pixels, dtype=np.float32)
    t = trace_pixels.shape[0]
    if t == 0:
        raise ValueError("Empty trace.")
    if t == 1:
        return np.tile(trace_pixels[0:1], (n_out, 1))

    diffs = np.diff(trace_pixels, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cumlen = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = cumlen[-1]

    if total == 0.0:
        return np.tile(trace_pixels[0:1], (n_out, 1))

    target_lens = np.linspace(0.0, total, n_out)
    out = np.empty((n_out, 2), dtype=np.float32)
    out[:, 0] = np.interp(target_lens, cumlen, trace_pixels[:, 0])
    out[:, 1] = np.interp(target_lens, cumlen, trace_pixels[:, 1])
    return out


def resample_trace(
    trace_pixels: np.ndarray,
    n_out: int,
    method: Literal["arc_length", "time_uniform"] = "arc_length",
) -> np.ndarray:
    if method == "arc_length":
        return arc_length_resample(trace_pixels, n_out)
    if method == "time_uniform":
        return time_uniform_resample(trace_pixels, n_out)
    raise ValueError(f"Unknown resample method: {method!r}")


# ---------------------------------------------------------------------------
# Polyline overlay rendering
# ---------------------------------------------------------------------------

def _xiaolin_wu_line(
    image: np.ndarray, x0: float, y0: float, x1: float, y1: float, rgb: tuple[int, int, int]
) -> None:
    """Anti-aliased line drawing (Xiaolin Wu's algorithm).

    Modifies `image` in place. `image` must be HxWx3 uint8.
    """
    h, w = image.shape[0], image.shape[1]
    steep = abs(y1 - y0) > abs(x1 - x0)
    if steep:
        x0, y0 = y0, x0
        x1, y1 = y1, x1
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    dx = x1 - x0
    dy = y1 - y0
    if dx == 0:
        gradient = 1.0
    else:
        gradient = dy / dx

    def _plot(x: int, y: int, c: float):
        if 0 <= x < w and 0 <= y < h:
            for k in range(3):
                v = float(image[y, x, k])
                v = (1.0 - c) * v + c * float(rgb[k])
                image[y, x, k] = np.uint8(min(255.0, max(0.0, v)))

    # First endpoint
    xend = round(x0)
    yend = y0 + gradient * (xend - x0)
    xgap = 1.0 - ((x0 + 0.5) - int(x0 + 0.5))
    xpxl1 = int(xend)
    ypxl1 = int(np.floor(yend))
    if steep:
        _plot(ypxl1, xpxl1, (1.0 - (yend - ypxl1)) * xgap)
        _plot(ypxl1 + 1, xpxl1, (yend - ypxl1) * xgap)
    else:
        _plot(xpxl1, ypxl1, (1.0 - (yend - ypxl1)) * xgap)
        _plot(xpxl1, ypxl1 + 1, (yend - ypxl1) * xgap)
    intery = yend + gradient

    # Second endpoint
    xend = round(x1)
    yend = y1 + gradient * (xend - x1)
    xgap = (x1 + 0.5) - int(x1 + 0.5)
    xpxl2 = int(xend)
    ypxl2 = int(np.floor(yend))
    if steep:
        _plot(ypxl2, xpxl2, (1.0 - (yend - ypxl2)) * xgap)
        _plot(ypxl2 + 1, xpxl2, (yend - ypxl2) * xgap)
    else:
        _plot(xpxl2, ypxl2, (1.0 - (yend - ypxl2)) * xgap)
        _plot(xpxl2, ypxl2 + 1, (yend - ypxl2) * xgap)

    # Main loop
    for x in range(xpxl1 + 1, xpxl2):
        ix = int(np.floor(intery))
        if steep:
            _plot(ix, x, 1.0 - (intery - ix))
            _plot(ix + 1, x, intery - ix)
        else:
            _plot(x, ix, 1.0 - (intery - ix))
            _plot(x, ix + 1, intery - ix)
        intery += gradient


def _filled_disk(image: np.ndarray, cx: float, cy: float, radius: float, rgb: tuple[int, int, int]) -> None:
    """Draw a small filled disk at (cx, cy)."""
    h, w = image.shape[0], image.shape[1]
    r = int(np.ceil(radius))
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            x = int(round(cx)) + dx
            y = int(round(cy)) + dy
            if 0 <= x < w and 0 <= y < h:
                d = (dx ** 2 + dy ** 2) ** 0.5
                if d <= radius:
                    a = 1.0 if d <= radius - 1.0 else max(0.0, radius - d)
                    for k in range(3):
                        v = float(image[y, x, k])
                        v = (1.0 - a) * v + a * float(rgb[k])
                        image[y, x, k] = np.uint8(min(255.0, max(0.0, v)))


def draw_polyline_overlay(
    image: np.ndarray,
    trace_xy_normalized: np.ndarray,
    *,
    color: tuple[int, int, int] = (0, 255, 255),  # cyan
    line_thickness: int = 2,
    endpoint_radius: float = 2.5,
) -> np.ndarray:
    """Draw a polyline overlay on a copy of ``image``.

    Args:
        image: HxWx3 uint8 array (RGB) — base camera image, will not be mutated.
        trace_xy_normalized: (N, 2) normalized [0, 1] pixel coords.
            x is column, y is row, matching the annotation pipeline.
        color: RGB tuple to draw the polyline.
        line_thickness: integer pixel thickness (>=1). Implemented by drawing parallel
            offset lines for thickness > 1.
        endpoint_radius: radius of small disks at the start and end of the polyline.
    """
    image = np.array(image, dtype=np.uint8, copy=True)
    h, w = image.shape[0], image.shape[1]
    if trace_xy_normalized.size == 0:
        return image

    pts = np.asarray(trace_xy_normalized, dtype=np.float32)
    pts = np.stack(
        [
            np.clip(pts[:, 0] * (w - 1), 0.0, w - 1.0),
            np.clip(pts[:, 1] * (h - 1), 0.0, h - 1.0),
        ],
        axis=1,
    )

    # Draw segments. For thickness > 1, draw additional lines offset perpendicular to the segment direction.
    n = pts.shape[0]
    half_t = max(0, line_thickness - 1)
    for i in range(n - 1):
        x0, y0 = float(pts[i, 0]), float(pts[i, 1])
        x1, y1 = float(pts[i + 1, 0]), float(pts[i + 1, 1])
        _xiaolin_wu_line(image, x0, y0, x1, y1, color)
        if half_t > 0:
            # Compute perpendicular offset
            dx, dy = x1 - x0, y1 - y0
            length = (dx * dx + dy * dy) ** 0.5
            if length > 1e-6:
                nx, ny = -dy / length, dx / length
                for off in range(1, half_t + 1):
                    _xiaolin_wu_line(
                        image,
                        x0 + nx * off,
                        y0 + ny * off,
                        x1 + nx * off,
                        y1 + ny * off,
                        color,
                    )
                    _xiaolin_wu_line(
                        image,
                        x0 - nx * off,
                        y0 - ny * off,
                        x1 - nx * off,
                        y1 - ny * off,
                        color,
                    )

    # Endpoint markers
    if endpoint_radius > 0:
        _filled_disk(image, float(pts[0, 0]), float(pts[0, 1]), endpoint_radius, color)
        _filled_disk(image, float(pts[-1, 0]), float(pts[-1, 1]), endpoint_radius, color)

    return image


# ---------------------------------------------------------------------------
# Smooth low-frequency trace perturbation
# ---------------------------------------------------------------------------

def smooth_low_freq_perturb(
    trace_xy_norm: np.ndarray,
    rng: np.random.RandomState,
    *,
    max_sigma: float = 0.03,
    num_freqs: int = 3,
) -> np.ndarray:
    """Add a smooth low-frequency 2-D offset field to a trace polyline.

    Used to perturb the overlay-rendered trace seen by the action head during
    training so that it learns to handle imperfect predicted traces (the
    inference-time failure mode), without exposing it to high-frequency
    artefacts that don't occur at inference. The offset is built as a sum of a
    few low-frequency sinusoids in the trace's arc-length parameter, so the
    perturbed trace remains smooth and "plausible" — just bent.

    Args:
        trace_xy_norm: ``(N, 2)`` polyline in normalized [0, 1]^2 coords.
        rng: NumPy RandomState for reproducibility (shared with the dataset).
        max_sigma: upper bound on the per-sample perturbation magnitude (in
            normalized units). The actual sigma is drawn uniformly from
            ``[0, max_sigma]`` per call so the training distribution mixes
            no-op and stronger perturbations.
        num_freqs: number of sinusoidal components per axis. Amplitudes decay
            as ``1 / f`` with frequency so low-frequency bending dominates.

    Returns:
        ``(N, 2)`` perturbed polyline (same dtype as input). May fall outside
        [0, 1] — the polyline overlay rasterizer clips out-of-bounds pixels.
    """
    trace_xy_norm = np.asarray(trace_xy_norm, dtype=np.float32)
    if trace_xy_norm.ndim != 2 or trace_xy_norm.shape[-1] != 2:
        raise ValueError(f"Expected (N, 2) input, got shape {trace_xy_norm.shape}")
    N = trace_xy_norm.shape[0]
    if N < 1 or max_sigma <= 0.0 or num_freqs <= 0:
        return trace_xy_norm.copy()

    sigma = float(rng.uniform(0.0, max_sigma))
    if sigma <= 0.0:
        return trace_xy_norm.copy()

    # Normalized arc-position. With N == 1 we degenerate to a single offset.
    if N == 1:
        s = np.zeros((1,), dtype=np.float32)
    else:
        s = np.linspace(0.0, 1.0, N, dtype=np.float32)

    perturb = np.zeros((N, 2), dtype=np.float32)
    for f in range(1, int(num_freqs) + 1):
        # Random amplitude per axis, decaying with frequency.
        amp_x = float(rng.normal()) * sigma / float(f)
        amp_y = float(rng.normal()) * sigma / float(f)
        phase_x = float(rng.uniform(0.0, 2.0 * np.pi))
        phase_y = float(rng.uniform(0.0, 2.0 * np.pi))
        perturb[:, 0] += amp_x * np.sin(2.0 * np.pi * float(f) * s + phase_x)
        perturb[:, 1] += amp_y * np.sin(2.0 * np.pi * float(f) * s + phase_y)

    return (trace_xy_norm + perturb).astype(trace_xy_norm.dtype)
