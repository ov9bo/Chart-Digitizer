"""Immutable data passed between pipeline stages.

Pixel coordinates follow OpenCV: column ``u`` grows to the right, row ``v`` grows downward, and
pixel centres sit on integer coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

CornerSource = Literal["manual", "auto", "interactive"]


@dataclass(frozen=True, eq=False)
class Corners:
    """Plot corners in source-image pixels: (xmin,ymin), (xmax,ymin), (xmax,ymax), (xmin,ymax)."""

    points: np.ndarray  # shape (4, 2), float64, columns (u, v)
    source: CornerSource


@dataclass(frozen=True)
class PlotBox:
    """Axes box inside the rectified image, in pixel coordinates (inclusive)."""

    left: int  # column of xmin
    top: int  # row of ymax
    right: int  # column of xmax
    bottom: int  # row of ymin

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


@dataclass(frozen=True, eq=False)
class Rectified:
    image: np.ndarray  # BGR uint8, axes box plus padding
    homography: np.ndarray  # 3x3, source pixels -> rectified pixels
    box: PlotBox


@dataclass(frozen=True)
class AxisMap:
    """Linear map between a pixel coordinate and a data value, in log10 space for log axes."""

    lo: float  # data value at px_lo
    hi: float  # data value at px_hi
    px_lo: float
    px_hi: float
    log: bool = False

    def _forward(self, values: np.ndarray) -> np.ndarray:
        return np.log10(values) if self.log else values

    def _inverse(self, values: np.ndarray) -> np.ndarray:
        return np.power(10.0, values) if self.log else values

    def to_px(self, values: np.ndarray | float) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        a, b = self._forward(np.array([self.lo, self.hi]))
        return self.px_lo + (self._forward(values) - a) * (self.px_hi - self.px_lo) / (b - a)

    def to_data(self, px: np.ndarray | float) -> np.ndarray:
        px = np.asarray(px, dtype=np.float64)
        a, b = self._forward(np.array([self.lo, self.hi]))
        return self._inverse(a + (px - self.px_lo) * (b - a) / (self.px_hi - self.px_lo))


@dataclass(frozen=True)
class Calibration:
    x: AxisMap  # column -> x
    y: AxisMap  # row -> y


@dataclass(frozen=True, eq=False)
class CurveMask:
    """Pixels that belong to one candidate curve, in rectified space."""

    mask: np.ndarray  # bool, same shape as the rectified image
    weight: np.ndarray  # float32 ink strength in [0, 1], zero outside the mask
    label: str
    # bool, ink removed as horizontal lines; the trace may cross gaps in the mask through it
    removed: np.ndarray | None = None


@dataclass(frozen=True, eq=False)
class Trace:
    """The traced curve: one row per column over a contiguous column span."""

    cols: np.ndarray  # int, contiguous from first to last traced column
    rows: np.ndarray  # float, sub-pixel row of the curve centre
    filled: np.ndarray  # bool, True where the row was interpolated across a gap
    warnings: tuple[str, ...] = ()

    @property
    def filled_fraction(self) -> float:
        return float(self.filled.mean()) if self.filled.size else 0.0


@dataclass(frozen=True, eq=False)
class Points:
    """Resampled output points, sorted by x. ``y`` is NaN where x fell outside the trace."""

    x: np.ndarray
    y: np.ndarray
    interpolated: np.ndarray  # bool
    in_range: np.ndarray  # bool
    warnings: tuple[str, ...] = ()
