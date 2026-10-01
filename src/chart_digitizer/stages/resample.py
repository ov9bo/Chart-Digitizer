"""Stage 8: resample the traced curve at the requested x values.

Nothing is extrapolated. A requested x outside the traced extent (beyond ``edge_tol_px``) gets
``y = NaN`` and ``in_range = False``. A point counts as interpolated when either traced column
next to it was gap-filled.
"""

from __future__ import annotations

import numpy as np

from chart_digitizer.config import AxesConfig, SamplingConfig
from chart_digitizer.models import Calibration, Points, Trace


def requested_x(axes: AxesConfig, cfg: SamplingConfig) -> np.ndarray:
    """The x values to sample, sorted ascending. Log axes space ``points`` evenly in log10."""
    lo, hi = sorted(axes.x_range)
    if cfg.x_values is not None:
        return np.unique(np.asarray(cfg.x_values, dtype=np.float64))
    if cfg.step is not None:
        count = int(np.floor((hi - lo) / cfg.step + 1e-9)) + 1
        return lo + cfg.step * np.arange(count)
    assert cfg.points is not None  # guaranteed by SamplingConfig
    if axes.x_log:
        return np.geomspace(lo, hi, cfg.points)
    return np.linspace(lo, hi, cfg.points)


def resample(trace: Trace, calibration: Calibration, x_values: np.ndarray, edge_tol_px: float) -> Points:
    cols = calibration.x.to_px(x_values)
    first, last = float(trace.cols[0]), float(trace.cols[-1])
    in_range = (cols >= first - edge_tol_px) & (cols <= last + edge_tol_px)
    snapped = np.clip(cols, first, last)

    rows = np.interp(snapped, trace.cols, trace.rows)
    y = np.where(in_range, calibration.y.to_data(rows), np.nan)

    left = np.floor(snapped).astype(np.int64) - trace.cols[0]
    right = np.ceil(snapped).astype(np.int64) - trace.cols[0]
    interpolated = in_range & (trace.filled[left] | trace.filled[right])

    warnings = list(trace.warnings)
    missing = int((~in_range).sum())
    if missing:
        x_first, x_last = sorted(calibration.x.to_data(np.array([first, last])))
        warnings.append(
            f"{missing} of {len(x_values)} requested x values lie outside the traced range "
            f"[{x_first:.4g}, {x_last:.4g}] and have no y value."
        )
    return Points(
        x=np.asarray(x_values, dtype=np.float64),
        y=y,
        interpolated=interpolated,
        in_range=in_range,
        warnings=tuple(warnings),
    )
