"""overlay.png (trace drawn back onto the input) and replot.png (clean matplotlib chart)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator

from chart_digitizer.config import AxesConfig
from chart_digitizer.models import Calibration, Corners, Points, Rectified, Trace

# BGR colours and relative line widths for drawn overlays.
_TRACE_BGR = (255, 0, 255)
_FILLED_BGR = (0, 200, 255)
_CORNER_BGR = (255, 200, 0)
_POINT_BGR = (0, 160, 0)
_STROKE_FRAC = 0.002  # line width as a fraction of the image diagonal
_REPLOT_SIZE_IN = (8.0, 6.0)
_REPLOT_DPI = 150


def _stroke(image: np.ndarray) -> int:
    return max(1, round(_STROKE_FRAC * float(np.hypot(*image.shape[:2]))))


def to_source(points_rect: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """Map (N, 2) rectified pixel coordinates back into the source image."""
    pts = points_rect.reshape(-1, 1, 2).astype(np.float64)
    return cv2.perspectiveTransform(pts, np.linalg.inv(homography)).reshape(-1, 2)


def _draw_segments(
    canvas: np.ndarray, xy: np.ndarray, filled: np.ndarray, width: int
) -> None:
    """Polyline in the trace colour, with gap-filled stretches in a second colour."""
    breaks = np.flatnonzero(np.diff(filled.astype(np.int8))) + 1
    for chunk in np.split(np.arange(len(xy)), breaks):
        if len(chunk) == 0:
            continue
        # Extend by one so consecutive chunks join up.
        idx = np.append(chunk, chunk[-1] + 1) if chunk[-1] + 1 < len(xy) else chunk
        colour = _FILLED_BGR if filled[chunk[0]] else _TRACE_BGR
        poly = np.round(xy[idx]).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [poly], isClosed=False, color=colour, thickness=width, lineType=cv2.LINE_AA)


def _trace_xy(trace: Trace) -> np.ndarray:
    return np.column_stack([trace.cols.astype(np.float64), trace.rows])


def draw_rectified(rectified: Rectified, trace: Trace) -> np.ndarray:
    """Debug view: the trace on the rectified image, with the axes box outlined."""
    canvas = rectified.image.copy()
    box = rectified.box
    width = _stroke(canvas)
    cv2.rectangle(canvas, (box.left, box.top), (box.right, box.bottom), _CORNER_BGR, width)
    _draw_segments(canvas, _trace_xy(trace), trace.filled, width)
    return canvas


def draw_overlay(
    image: np.ndarray,
    corners: Corners,
    rectified: Rectified,
    calibration: Calibration,
    trace: Trace,
    points: Points,
) -> np.ndarray:
    """The trace, the plot corners and the resampled points, drawn on the source image."""
    canvas = image.copy()
    width = _stroke(canvas)
    quad = np.round(corners.points).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas, [quad], isClosed=True, color=_CORNER_BGR, thickness=width, lineType=cv2.LINE_AA)
    _draw_segments(canvas, to_source(_trace_xy(trace), rectified.homography), trace.filled, width)

    ok = points.in_range
    if ok.any():
        rect = np.column_stack([calibration.x.to_px(points.x[ok]), calibration.y.to_px(points.y[ok])])
        for u, v in to_source(rect, rectified.homography):
            cv2.circle(canvas, (round(u), round(v)), 2 * width, _POINT_BGR, width, cv2.LINE_AA)
    return canvas


def render_replot(
    path: Path, axes: AxesConfig, calibration: Calibration, trace: Trace, points: Points, title: str
) -> None:
    """A clean chart of the digitized curve over the user's axis ranges."""
    fig = Figure(figsize=_REPLOT_SIZE_IN, dpi=_REPLOT_DPI)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    ax.plot(calibration.x.to_data(trace.cols), calibration.y.to_data(trace.rows), color="0.35", lw=1, label="trace")

    ok = points.in_range
    measured = ok & ~points.interpolated
    interpolated = ok & points.interpolated
    ax.plot(points.x[measured], points.y[measured], "o", color="C0", ms=4, clip_on=False, label="points")
    if interpolated.any():
        ax.plot(
            points.x[interpolated], points.y[interpolated], "o", mfc="none", color="C1", ms=5, clip_on=False, label="interpolated"
        )

    ax.set_xlim(*axes.x_range)
    ax.set_ylim(*axes.y_range)
    ax.set_xscale("log" if axes.x_log else "linear")
    ax.set_yscale("log" if axes.y_log else "linear")
    if axes.grid_step is not None:
        if not axes.x_log:
            ax.xaxis.set_major_locator(MultipleLocator(axes.grid_step[0]))
        if not axes.y_log:
            ax.yaxis.set_major_locator(MultipleLocator(axes.grid_step[1]))
    ax.grid(True, color="0.85", lw=0.6)
    ax.set_title(title)
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
