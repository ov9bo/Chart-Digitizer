"""Stage 4: map rectified pixels to data values using the user-supplied axis ranges."""

from __future__ import annotations

from chart_digitizer.config import AxesConfig
from chart_digitizer.models import AxisMap, Calibration, PlotBox


def calibrate(box: PlotBox, axes: AxesConfig) -> Calibration:
    x_lo, x_hi = axes.x_range
    y_lo, y_hi = axes.y_range
    return Calibration(
        x=AxisMap(lo=x_lo, hi=x_hi, px_lo=box.left, px_hi=box.right, log=axes.x_log),
        y=AxisMap(lo=y_lo, hi=y_hi, px_lo=box.bottom, px_hi=box.top, log=axes.y_log),
    )
