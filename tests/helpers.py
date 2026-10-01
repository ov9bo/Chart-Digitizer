"""Shared helpers for running the pipeline on synthetic charts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from chart_digitizer.config import Config, build_config
from chart_digitizer.imgio import write_image
from chart_digitizer.pipeline import Result, digitize


def synthetic_config(corners: np.ndarray | None, x_range: tuple[float, float], y_range: tuple[float, float], out: Path, **extra: Any) -> Config:
    """Pipeline config for a synthetic chart; ``corners=None`` finds the plot area automatically."""
    data: dict[str, Any] = {
        "axes": {"x_range": x_range, "y_range": y_range},
        "corners": {"mode": "auto"} if corners is None else {"points": np.asarray(corners).tolist()},
        "output": {"dir": str(out)},
    }
    for section, values in extra.items():
        data[section] = {**data.get(section, {}), **values}
    return build_config(data)


def run_on_image(image: np.ndarray, corners: np.ndarray | None, x_range: tuple[float, float], y_range: tuple[float, float], tmp_path: Path, **extra: Any) -> Result:
    path = tmp_path / "chart.png"
    write_image(path, image)
    return digitize(path, synthetic_config(corners, x_range, y_range, tmp_path / "out", **extra))


def rmse_fraction(
    result: Result, truth_x: np.ndarray, truth_y: np.ndarray, y_range: tuple[float, float], *, x_log: bool = False, y_log: bool = False
) -> float:
    """RMSE over in-range points, as a fraction of the y-axis height.

    On a log y axis the error is measured in log10(y), so it is a fraction of the axis on the page,
    as on a linear axis. On a log x axis the truth is interpolated in log10(x), where it is smooth.
    """
    ok = result.points.in_range
    fx = np.log10 if x_log else np.asarray
    fy = np.log10 if y_log else np.asarray
    expected = np.interp(fx(result.points.x[ok]), fx(truth_x), fy(truth_y))
    error = fy(result.points.y[ok]) - expected
    lo, hi = fy(np.asarray(y_range, dtype=np.float64))
    return float(np.sqrt(np.mean(error**2)) / abs(hi - lo))
