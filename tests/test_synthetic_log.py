"""End-to-end accuracy on synthetic charts with logarithmic axes.

Log charts carry minor gridlines between the decades, so the clean and cluttered charts are both
busier than their linear counterparts. Error is measured along the axis as drawn (in log10 of the
value on a log axis), with the same bounds as linear charts: RMSE < 1% of the axis height clean or
cluttered, < 2% degraded.
"""

from pathlib import Path

import numpy as np
import pytest

from chart_digitizer.synth.degrade import DegradeSpec, degrade
from chart_digitizer.synth.generate import SynthSpec, SyntheticChart, render
from helpers import rmse_fraction, run_on_image

pytestmark = pytest.mark.slow

MIN_IN_RANGE_FRAC = 0.95
MAX_FILLED_FRAC = 0.05

AXES = {
    "log-x": {"x_range": (1.0, 1000.0), "x_log": True},
    "log-y": {"y_range": (0.1, 1000.0), "y_log": True},
    "log-log": {"x_range": (10.0, 1e5), "y_range": (1.0, 1e4), "x_log": True, "y_log": True},
}


def run(chart: SyntheticChart, image: np.ndarray, corners: np.ndarray | None, tmp_path: Path) -> float:
    spec = chart.spec
    result = run_on_image(image, corners, spec.x_range, spec.y_range, tmp_path, axes={"x_log": spec.x_log, "y_log": spec.y_log})
    assert result.points.in_range.mean() >= MIN_IN_RANGE_FRAC
    assert result.trace.filled_fraction <= MAX_FILLED_FRAC
    return rmse_fraction(result, chart.truth_x, chart.truth_y, spec.y_range, x_log=spec.x_log, y_log=spec.y_log)


@pytest.mark.parametrize("axes", AXES)
@pytest.mark.parametrize("kind", ["monotonic", "wavy"])
@pytest.mark.parametrize("clutter", [False, True], ids=["clean", "clutter"])
def test_log_axes(axes: str, kind: str, clutter: bool, tmp_path: Path) -> None:
    chart = render(SynthSpec(seed=5, kind=kind, clutter=clutter, **AXES[axes]))
    assert run(chart, chart.image, chart.corners, tmp_path) < 0.01


@pytest.mark.parametrize("axes", AXES)
@pytest.mark.parametrize("seed", range(3))
def test_log_axes_degraded(axes: str, seed: int, tmp_path: Path) -> None:
    chart = render(SynthSpec(seed=seed, kind="wavy", clutter=True, **AXES[axes]))
    image, corners = degrade(chart.image, chart.corners, DegradeSpec(), np.random.default_rng(seed))
    assert run(chart, image, corners, tmp_path) < 0.02


@pytest.mark.parametrize("axes", AXES)
def test_log_axes_auto_corners(axes: str, tmp_path: Path) -> None:
    chart = render(SynthSpec(seed=7, kind="wavy", clutter=True, **AXES[axes]))
    image, _ = degrade(chart.image, chart.corners, DegradeSpec(), np.random.default_rng(7))
    assert run(chart, image, None, tmp_path) < 0.02
