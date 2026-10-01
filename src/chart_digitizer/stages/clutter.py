"""Stage 5: remove clutter from the ink mask.

After rectification, gridlines, axes and reference lines are exactly horizontal or vertical, so
they are found with a morphological opening by a long line kernel (after a short closing along
the same direction bridges dashes and moire breaks). A curve is only removed where it runs
straight along an axis for longer than the kernel.

Where the curve crosses a removed line it would be cut, so removed pixels are put back where
remaining ink lies close by on both sides across the line, or where a small closing of the
remaining ink bridges the cut (a steep crossing). A cut that is left, typically a sloped curve
crossing a thick vertical line, is a short run of empty columns that the trace bridges. Finally,
components too small to be part of the curve (specks, text, stray dash pieces) are dropped.

A plateau of the curve longer than the kernel looks exactly like a horizontal reference line,
so the removed horizontal line pixels are passed on: the trace may cross a gap in the curve
through them. Vertical lines are not: a curve y(x) never runs along one.
"""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

from chart_digitizer.config import ClutterConfig
from chart_digitizer.debug import DebugSink
from chart_digitizer.models import CurveMask, PlotBox

Orientation = Literal["horizontal", "vertical"]


def _kernel(length: int, orientation: Orientation) -> np.ndarray:
    shape = (max(length, 1), 1) if orientation == "horizontal" else (1, max(length, 1))
    # getStructuringElement takes (width, height).
    return cv2.getStructuringElement(cv2.MORPH_RECT, shape)


def straight_lines(mask: np.ndarray, length: int, bridge: int, orientation: Orientation) -> np.ndarray:
    """Pixels of ``mask`` on straight horizontal or vertical ink at least ``length`` long."""
    ink = mask.astype(np.uint8)
    bridged = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, _kernel(bridge, orientation)) if bridge > 1 else ink
    lines = cv2.morphologyEx(bridged, cv2.MORPH_OPEN, _kernel(length, orientation))
    # Include the anti-aliased fringe: one pixel either side across the line.
    across: Orientation = "vertical" if orientation == "horizontal" else "horizontal"
    lines = cv2.dilate(lines, _kernel(3, across))
    return lines.astype(bool) & mask


def ink_within(mask: np.ndarray, reach: int, axis: int) -> tuple[np.ndarray, np.ndarray]:
    """For each pixel: is there ink within ``reach`` pixels before it / after it along ``axis``?"""
    n = mask.shape[axis]
    cum = np.concatenate(
        [np.zeros_like(np.take(mask, [0], axis=axis), dtype=np.int32), np.cumsum(mask, axis=axis, dtype=np.int32)],
        axis=axis,
    )  # cum[i] = number of ink pixels before index i
    idx = np.arange(n)
    before = np.take(cum, idx, axis=axis) - np.take(cum, np.maximum(idx - reach, 0), axis=axis)
    after = np.take(cum, np.minimum(idx + 1 + reach, n), axis=axis) - np.take(cum, idx + 1, axis=axis)
    return before > 0, after > 0


def crossings(removed: np.ndarray, remaining: np.ndarray, reach: int, axis: int) -> np.ndarray:
    """Removed pixels with remaining ink within ``reach`` on both sides along ``axis``."""
    before, after = ink_within(remaining, reach, axis)
    return removed & before & after


def bridged_cuts(removed: np.ndarray, remaining: np.ndarray, reach: int) -> np.ndarray:
    """Removed pixels inside a closing of the remaining ink: cuts where ink continues across.

    This catches a curve crossing a line at a steep angle, where no single row (or column) has
    ink on both sides of the line.
    """
    size = 2 * reach + 1
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    closed = cv2.morphologyEx(remaining.astype(np.uint8), cv2.MORPH_CLOSE, disc).astype(bool)
    return removed & closed


def small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Pixels of 8-connected components whose bounding box is under ``min_size`` in both directions."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    small = (stats[:, cv2.CC_STAT_WIDTH] < min_size) & (stats[:, cv2.CC_STAT_HEIGHT] < min_size)
    small[0] = False  # background
    return small[labels] if count > 1 else np.zeros_like(mask)


def split_clutter(mask: np.ndarray, box: PlotBox, cfg: ClutterConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pixels of ``mask`` judged to be clutter: (horizontal lines, vertical lines, small specks)."""
    size = max(box.width, box.height)
    length = int(round(cfg.line_min_length_frac * size))
    bridge = int(round(cfg.line_bridge_frac * size))
    reach = max(int(round(cfg.restore_reach_frac * size)), 1)

    horizontal = straight_lines(mask, length, bridge, "horizontal")
    vertical = straight_lines(mask, length, bridge, "vertical")
    remaining = mask & ~horizontal & ~vertical
    # A horizontal line is crossed where ink continues above and below it, and vice versa. Where
    # two lines meet, either test will do: a bare grid intersection has no remaining ink nearby.
    removed = horizontal | vertical
    restored = (
        crossings(horizontal, remaining, reach, axis=0)
        | crossings(vertical, remaining, reach, axis=1)
        | bridged_cuts(removed, remaining, reach)
    )
    kept = mask & ~(removed & ~restored)
    specks = small_components(kept, int(round(cfg.min_component_frac * size)))
    return horizontal & ~restored, vertical & ~restored, specks


def clutter_mask(mask: np.ndarray, box: PlotBox, cfg: ClutterConfig) -> np.ndarray:
    horizontal, vertical, specks = split_clutter(mask, box, cfg)
    return horizontal | vertical | specks


def remove_clutter(ink: CurveMask, box: PlotBox, cfg: ClutterConfig, debug: DebugSink) -> CurveMask:
    if not cfg.enabled:
        return ink
    horizontal, vertical, specks = split_clutter(ink.mask, box, cfg)
    clutter = horizontal | vertical | specks
    mask = ink.mask & ~clutter
    if debug.enabled:
        view = np.full((*mask.shape, 3), 255, np.uint8)
        view[clutter] = (160, 160, 255)  # removed: light red
        view[mask] = (0, 0, 0)
        debug.save("06_clutter", view)
    weight = np.where(mask, ink.weight, 0.0).astype(np.float32)
    return CurveMask(mask=mask, weight=weight, label=ink.label, removed=horizontal)
