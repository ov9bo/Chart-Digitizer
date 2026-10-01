"""Stage 2: find the plot area automatically.

Classical line geometry, no learned models:

1. At a reduced working size, thin dark features are picked out with a black-hat filter (a
   grayscale closing minus the image), divided by the closing so the threshold is a contrast
   and doesn't depend on lighting. A step edge, such as the border of a photographed page, has
   no black-hat response, and nor does a dark region reaching the image edge. The threshold also
   rises with the contrast of the surrounding paper, which noise and texture raise in shadows.
2. A Hough transform over a limited tilt range finds long nearly-horizontal and nearly-vertical
   lines: gridlines, axis lines, reference lines, frames.
3. Each line's extent is its longest well-filled run of ink.
4. The plot box is the largest quadrilateral of two horizontal and two vertical lines that meet
   at all four corners, meaning each corner lies on the ink runs of both its lines. Axis lines
   drawn as a frame meet like this, and so do the outermost gridlines of a chart without a
   frame, because gridlines end at the axes. Tick labels, legends and text never close a box.
5. Each side is refit by weighted least squares over its ink, and the corners are the
   intersections of the refit sides.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import cv2
import numpy as np
from scipy import ndimage

from chart_digitizer.config import PlotAreaConfig
from chart_digitizer.debug import DebugSink
from chart_digitizer.errors import CornerError
from chart_digitizer.models import Corners

Family = Literal["horizontal", "vertical"]

_ANGLE_STEP_DEG = 0.1
# Peaks closer than this in the Hough space are one line: (degrees, working pixels).
_PEAK_WINDOW = (1.0, 5)
# Half-width of the strip around a line whose ink belongs to it, working pixels.
_BAND_PX = 2.0
_REFIT_BAND_PX = 4.0
# A refit needs this many positions along the line; positions further from the first fit than
# this many (scaled) median residuals, and at least the minimum, are dropped as outliers.
_MIN_FIT_SAMPLES = 10
_OUTLIER_MADS = 3.0
_OUTLIER_MIN_PX = 0.5
# Images noisier than this (gray levels) are smoothed with this sigma (working pixels) first.
_NOISY_SIGMA = 1.5
_DENOISE_SIGMA_PX = 1.0
# Brightness percentile taken as the paper white.
_PAPER_PERCENTILE = 90

_HINT = (
    "Pass the plot corners with --corners x1,y1,x2,y2,x3,y3,x4,y4 in the order (xmin,ymin), "
    "(xmax,ymin), (xmax,ymax), (xmin,ymax), or click them with --interactive. Automatic detection "
    "needs a closed box: a frame around the plot, or gridlines at both ends of each axis. "
    "--debug shows the lines it found in debug/02_plot_area.png."
)


@dataclass(frozen=True, eq=False)
class Line:
    """The line ``normal · p = rho`` for p = (u, v), with ink from ``t0`` to ``t1`` along it (t = direction · p)."""

    direction: np.ndarray  # unit vector along the line
    normal: np.ndarray  # unit normal
    rho: float
    t0: float
    t1: float

    @property
    def length(self) -> float:
        return self.t1 - self.t0


def frame(family: Family, angle: float) -> tuple[np.ndarray, np.ndarray]:
    """Unit direction and normal of a line tilted by ``angle`` radians from its family's axis."""
    c, s = np.cos(angle), np.sin(angle)
    if family == "horizontal":
        return np.array([c, s]), np.array([-s, c])  # rho ~ row, t ~ column
    return np.array([-s, c]), np.array([c, s])  # rho ~ column, t ~ row


def noise_level(gray: np.ndarray) -> float:
    """Robust estimate of the pixel noise standard deviation, in gray levels.

    From the median absolute difference of horizontal neighbours, which flat paper dominates;
    edges and lines are too few to move the median.
    """
    diff = np.abs(np.diff(gray.astype(np.float32), axis=1))
    return float(1.4826 * np.median(diff) / np.sqrt(2.0))


def line_contrast(gray: np.ndarray, kernel_px: int) -> np.ndarray:
    """How much darker each pixel is than its surroundings once thin dark features are closed, 0-1.

    The difference is divided by the local background, so shading doesn't matter, but never by
    less than half the paper white: in dark regions (a bezel, the table under a page) noise would
    otherwise be magnified into contrast. A closing takes the local maximum, so on noise alone it
    reports several standard deviations of contrast; noisy images are smoothed first. Clean ones
    are not, since smoothing would fade faint, thin lines.
    """
    k = max(3, kernel_px | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    smooth = gray.astype(np.float32)
    if noise_level(gray) > _NOISY_SIGMA:
        smooth = cv2.GaussianBlur(smooth, (0, 0), _DENOISE_SIGMA_PX)
    # Outside the image counts as dark, so dark regions that reach the image edge (the desk, or the
    # thin wedge between a tilted page and the image border) are not closed over and read as ink.
    # A line with paper between it and the edge is still enclosed by brighter pixels.
    padded = cv2.copyMakeBorder(smooth, k, k, k, k, cv2.BORDER_CONSTANT, value=0)
    closed = cv2.morphologyEx(padded, cv2.MORPH_CLOSE, kernel)[k:-k, k:-k]
    floor = 0.5 * float(np.percentile(closed, _PAPER_PERCENTILE))
    return (closed - smooth) / np.maximum(closed, max(floor, 1.0))


def block_median(values: np.ndarray, block_px: int) -> np.ndarray:
    """Median of ``values`` over square blocks, interpolated smoothly back to full size."""
    h, w = values.shape
    b = max(1, min(block_px, h, w))
    rows, cols = h // b, w // b
    blocks = values[: rows * b, : cols * b].reshape(rows, b, cols, b).transpose(0, 2, 1, 3)
    medians = np.median(blocks.reshape(rows, cols, -1), axis=2).astype(np.float32)
    return cv2.resize(medians, (w, h), interpolation=cv2.INTER_LINEAR)


def ink_threshold(contrast: np.ndarray, cfg: PlotAreaConfig, size: int) -> np.ndarray:
    """Per-pixel contrast a pixel needs to count as line ink.

    At least ``min_contrast``, and several times the median contrast around it. Most of any block
    is paper, so the median measures the paper's own contrast: zero on clean paper, but a few
    percent where shading lowers the paper's brightness and noise then counts for more.
    """
    local = block_median(contrast, int(round(cfg.noise_block_frac * size)))
    return np.maximum(cfg.min_contrast, cfg.noise_contrast_mult * local)


def hough_peaks(
    points: np.ndarray, family: Family, max_tilt_deg: float, min_votes: float, max_lines: int
) -> list[tuple[float, float]]:
    """(angle, rho) of the strongest lines through ``points`` (N, 2), strongest first."""
    angles = np.deg2rad(np.arange(-max_tilt_deg, max_tilt_deg + _ANGLE_STEP_DEG / 2, _ANGLE_STEP_DEG))
    bound = int(np.ceil(np.abs(points).max(initial=0.0) * 2)) + 1
    acc = np.empty((len(angles), 2 * bound + 1), np.float32)
    for i, angle in enumerate(angles):
        rho = np.rint(points @ frame(family, angle)[1]).astype(np.int64) + bound
        acc[i] = np.bincount(rho, minlength=acc.shape[1])
    # A line's votes spread over neighbouring rho bins when its angle falls between steps.
    acc = ndimage.uniform_filter1d(acc, 3, axis=1, mode="constant") * 3
    window = (2 * int(round(_PEAK_WINDOW[0] / _ANGLE_STEP_DEG)) + 1, 2 * _PEAK_WINDOW[1] + 1)
    peaks = (acc == ndimage.maximum_filter(acc, size=window, mode="constant")) & (acc >= min_votes)
    ai, ri = np.nonzero(peaks)
    order = np.argsort(-acc[ai, ri], kind="stable")
    chosen: list[tuple[int, int]] = []
    for a, r in zip(ai[order].tolist(), ri[order].tolist(), strict=True):
        # Plateaus give several equal maxima; keep one.
        if all(abs(a - ca) > window[0] // 2 or abs(r - cr) > window[1] // 2 for ca, cr in chosen):
            chosen.append((a, r))
        if len(chosen) == max_lines:
            break
    return [(float(angles[a]), float(r - bound)) for a, r in chosen]


def ink_run(t: np.ndarray, window: int, min_fill: float) -> tuple[float, float] | None:
    """Longest stretch of positions ``t`` along a line where at least ``min_fill`` of a sliding window holds ink."""
    if t.size == 0:
        return None
    lo = int(np.floor(t.min()))
    occupied = np.zeros(int(np.ceil(t.max())) - lo + 1, np.float32)
    occupied[np.rint(t).astype(np.int64) - lo] = 1.0
    filled = ndimage.uniform_filter1d(occupied, max(window, 1) | 1, mode="constant") >= min_fill - 1e-6
    edges = np.diff(np.concatenate([[0], filled.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
    if starts.size == 0:
        return None
    best = int(np.argmax(ends - starts))
    return float(lo + starts[best]), float(lo + ends[best])


def find_lines(points: np.ndarray, family: Family, cfg: PlotAreaConfig, size: int) -> list[Line]:
    """Long lines of one family through the ink ``points``, each with its ink run."""
    min_side = cfg.min_side_frac * size
    window = int(round(cfg.gap_frac * size))
    lines = []
    for angle, rho in hough_peaks(points, family, cfg.max_tilt_deg, min_side, cfg.max_lines):
        direction, normal = frame(family, angle)
        near = np.abs(points @ normal - rho) <= _BAND_PX
        run = ink_run(points[near] @ direction, window, cfg.min_fill)
        if run is not None and run[1] - run[0] >= min_side:
            lines.append(Line(direction=direction, normal=normal, rho=rho, t0=run[0], t1=run[1]))
    return lines


def intersect(a: Line, b: Line) -> np.ndarray:
    return np.linalg.solve(np.array([a.normal, b.normal]), np.array([a.rho, b.rho]))


def meets(a: Line, b: Line, tol: float) -> bool:
    """True if the lines' intersection lies on both lines' ink runs, within ``tol``."""
    p = intersect(a, b)
    return all(line.t0 - tol <= float(line.direction @ p) <= line.t1 + tol for line in (a, b))


def box_corners(top: Line, bottom: Line, left: Line, right: Line) -> np.ndarray:
    """Corners in the order (xmin,ymin), (xmax,ymin), (xmax,ymax), (xmin,ymax) of an upright chart."""
    return np.array([intersect(bottom, left), intersect(bottom, right), intersect(top, right), intersect(top, left)])


def quad_area(corners: np.ndarray) -> float:
    x, y = corners[:, 0], corners[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)))


def find_box(
    horizontal: list[Line], vertical: list[Line], tol: float, min_side: float
) -> tuple[Line, Line, Line, Line] | None:
    """(top, bottom, left, right): the largest box of lines that meet at all four corners."""
    horizontal = sorted(horizontal, key=lambda line: line.rho)  # top first
    vertical = sorted(vertical, key=lambda line: line.rho)  # left first
    meet = np.array([[meets(h, v, tol) for v in vertical] for h in horizontal], dtype=bool)
    best: tuple[Line, Line, Line, Line] | None = None
    best_area = 0.0
    for i, j in combinations(range(len(horizontal)), 2):
        both = np.flatnonzero(meet[i] & meet[j])
        if both.size < 2:
            continue
        sides = (horizontal[i], horizontal[j], vertical[both[0]], vertical[both[-1]])
        corners = box_corners(*sides)
        lengths = np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1)
        area = quad_area(corners)
        if lengths.min() >= min_side and area > best_area:
            best, best_area = sides, area
    return best


def robust_line_fit(t: np.ndarray, offset: np.ndarray) -> tuple[float, float] | None:
    """(slope, intercept) of ``offset = intercept + slope * t``, refit once without outliers."""
    if t.size < _MIN_FIT_SAMPLES:
        return None
    slope, intercept = np.polyfit(t, offset, 1)
    residual = np.abs(offset - (intercept + slope * t))
    keep = residual <= max(_OUTLIER_MADS * 1.4826 * float(np.median(residual)), _OUTLIER_MIN_PX)
    if keep.sum() >= _MIN_FIT_SAMPLES:
        slope, intercept = np.polyfit(t[keep], offset[keep], 1)
    return float(slope), float(intercept)


def refit(line: Line, points: np.ndarray, weights: np.ndarray, trim: float) -> Line:
    """Least-squares fit of the line to the ink near it, away from its ends.

    Each position along the line gives one sample, the contrast-weighted centre of the ink across
    it, so a dark curve crossing a faint gridline counts no more than any other position, and
    positions far off the fit (crossings, labels touching the line) are dropped.
    """
    for _ in range(2):
        t = points @ line.direction
        offset = points @ line.normal - line.rho
        near = (np.abs(offset) <= _REFIT_BAND_PX) & (t >= line.t0 + trim) & (t <= line.t1 - trim)
        if not near.any():
            return line
        bins = np.rint(t[near] - t[near].min()).astype(np.int64)
        mass = np.bincount(bins, weights=weights[near])
        centre = np.bincount(bins, weights=weights[near] * offset[near])
        filled = mass > 0
        fit = robust_line_fit(np.flatnonzero(filled) + t[near].min(), centre[filled] / mass[filled])
        if fit is None:
            return line
        slope, intercept = fit
        norm = float(np.hypot(1.0, slope))
        line = Line(
            direction=(line.direction + slope * line.normal) / norm,
            normal=(line.normal - slope * line.direction) / norm,
            rho=(line.rho + intercept) / norm,
            t0=line.t0,
            t1=line.t1,
        )
    return line


def _draw_debug(
    image: np.ndarray, lines: list[Line], box: tuple[Line, Line, Line, Line] | None, corners: np.ndarray | None
) -> np.ndarray:
    view = image.copy()
    for line in lines:
        a = line.t0 * line.direction + line.rho * line.normal
        b = line.t1 * line.direction + line.rho * line.normal
        cv2.line(view, tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int)), (255, 160, 0), 1, cv2.LINE_AA)
    if corners is not None:
        pts = np.rint(corners).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(view, [pts], True, (0, 0, 255), 2, cv2.LINE_AA)
        for p in pts[:, 0]:
            cv2.circle(view, tuple(p), 6, (0, 0, 255), 2, cv2.LINE_AA)
    return view


def detect_plot_area(image: np.ndarray, cfg: PlotAreaConfig, debug: DebugSink) -> Corners:
    """Find the axes box in ``image`` (BGR); raises CornerError if no closed box is found."""
    h, w = image.shape[:2]
    scale = min(1.0, cfg.work_size / max(h, w))
    small = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else image
    size = max(small.shape[:2])
    contrast = line_contrast(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), int(round(cfg.line_kernel_frac * size)))
    rows, cols = np.nonzero(contrast >= ink_threshold(contrast, cfg, size))
    points = np.column_stack([cols, rows]).astype(np.float64)
    weights = contrast[rows, cols].astype(np.float64)

    horizontal = find_lines(points, "horizontal", cfg, size)
    vertical = find_lines(points, "vertical", cfg, size)
    box = find_box(horizontal, vertical, cfg.corner_tol_frac * size, cfg.min_side_frac * size)
    corners = None
    if box is not None:
        trim = cfg.gap_frac * size
        corners = box_corners(*(refit(line, points, weights, trim) for line in box))
    if debug.enabled:
        debug.save("02_plot_area", _draw_debug(small, horizontal + vertical, box, corners))
    if corners is None:
        raise CornerError(
            "Could not find the plot area automatically: no two horizontal and two vertical lines "
            f"close a box (found {len(horizontal)} long horizontal and {len(vertical)} long vertical lines)",
            hint=_HINT,
        )
    # Working pixels -> source pixels, with pixel centres on integers in both.
    return Corners(points=(corners + 0.5) / scale - 0.5, source="auto")
