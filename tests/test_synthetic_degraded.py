"""End-to-end accuracy on cluttered and photo-degraded synthetic charts with exact corners.

Cluttered charts carry dark gridlines, spines, reference lines crossing the curve, a dashed
colored series, a legend, text and specks. Clutter alone must stay under the clean-chart bound
(RMSE < 1% of the y-range); clutter plus photo degradations under 2%.
"""

from pathlib import Path

import numpy as np
import pytest

from chart_digitizer.synth.degrade import DegradeSpec, degrade
from chart_digitizer.synth.generate import SynthSpec, render
from helpers import rmse_fraction, run_on_image

pytestmark = pytest.mark.slow

MIN_IN_RANGE_FRAC = 0.95
MAX_FILLED_FRAC = 0.05


def check(result, chart, y_range, max_rmse: float) -> None:  # noqa: ANN001
    assert result.points.in_range.mean() >= MIN_IN_RANGE_FRAC
    assert rmse_fraction(result, chart.truth_x, chart.truth_y, y_range) < max_rmse
    assert result.trace.filled_fraction <= MAX_FILLED_FRAC


@pytest.mark.parametrize("kind", ["monotonic", "wavy"])
@pytest.mark.parametrize("seed", range(4))
def test_clutter(kind: str, seed: int, tmp_path: Path) -> None:
    spec = SynthSpec(seed=seed, kind=kind, clutter=True)
    chart = render(spec)
    result = run_on_image(chart.image, chart.corners, spec.x_range, spec.y_range, tmp_path)
    check(result, chart, spec.y_range, max_rmse=0.01)


@pytest.mark.parametrize("kind", ["monotonic", "wavy"])
@pytest.mark.parametrize("seed", range(6))
def test_degraded(kind: str, seed: int, tmp_path: Path) -> None:
    spec = SynthSpec(seed=seed, kind=kind, clutter=True)
    chart = render(spec)
    image, corners = degrade(chart.image, chart.corners, DegradeSpec(), np.random.default_rng(seed))
    result = run_on_image(image, corners, spec.x_range, spec.y_range, tmp_path)
    check(result, chart, spec.y_range, max_rmse=0.02)


def test_degraded_colored_curve(tmp_path: Path) -> None:
    spec = SynthSpec(seed=41, kind="wavy", color="#1f77b4", clutter=True)
    chart = render(spec)
    image, corners = degrade(chart.image, chart.corners, DegradeSpec(), np.random.default_rng(41))
    result = run_on_image(image, corners, spec.x_range, spec.y_range, tmp_path, curve={"color": "#1f77b4"})
    check(result, chart, spec.y_range, max_rmse=0.02)
