"""End-to-end accuracy on clean synthetic charts with exact corners: RMSE < 1% of the y-range."""

from pathlib import Path

import numpy as np
import pytest

from chart_digitizer.synth.degrade import perspective
from chart_digitizer.synth.generate import SynthSpec, render
from helpers import rmse_fraction, run_on_image

pytestmark = pytest.mark.slow

MAX_RMSE_FRAC = 0.01
MIN_IN_RANGE_FRAC = 0.95


def check(result, chart, y_range) -> None:  # noqa: ANN001
    assert result.points.in_range.mean() >= MIN_IN_RANGE_FRAC
    assert rmse_fraction(result, chart.truth_x, chart.truth_y, y_range) < MAX_RMSE_FRAC
    assert result.trace.filled_fraction == 0


@pytest.mark.parametrize("kind", ["monotonic", "wavy"])
@pytest.mark.parametrize("seed", range(6))
def test_clean(kind: str, seed: int, tmp_path: Path) -> None:
    spec = SynthSpec(seed=seed, kind=kind)
    chart = render(spec)
    result = run_on_image(chart.image, chart.corners, spec.x_range, spec.y_range, tmp_path)
    check(result, chart, spec.y_range)


@pytest.mark.parametrize(
    ("x_range", "y_range"),
    [((-5.0, 5.0), (1000.0, 5000.0)), ((0.0, 1.0), (-0.2, 0.2)), ((1990.0, 2020.0), (0.0, 7.0))],
)
def test_other_ranges(x_range: tuple[float, float], y_range: tuple[float, float], tmp_path: Path) -> None:
    spec = SynthSpec(seed=11, kind="wavy", x_range=x_range, y_range=y_range)
    chart = render(spec)
    result = run_on_image(chart.image, chart.corners, x_range, y_range, tmp_path)
    check(result, chart, y_range)


@pytest.mark.parametrize("seed", range(3))
def test_perspective_only(seed: int, tmp_path: Path) -> None:
    spec = SynthSpec(seed=100 + seed, kind="wavy" if seed % 2 else "monotonic")
    chart = render(spec)
    image, corners = perspective(chart.image, chart.corners, 0.08, np.random.default_rng(seed))
    result = run_on_image(image, corners, spec.x_range, spec.y_range, tmp_path)
    check(result, chart, spec.y_range)


@pytest.mark.parametrize("color", ["#1f77b4", "#d62728"])
def test_colored_curve(color: str, tmp_path: Path) -> None:
    spec = SynthSpec(seed=21, kind="wavy", color=color)
    chart = render(spec)
    result = run_on_image(chart.image, chart.corners, spec.x_range, spec.y_range, tmp_path, curve={"color": color})
    check(result, chart, spec.y_range)


def test_thin_and_thick_lines(tmp_path: Path) -> None:
    for width in (0.8, 4.0):
        spec = SynthSpec(seed=31, kind="wavy", line_width_pt=width)
        chart = render(spec)
        result = run_on_image(chart.image, chart.corners, spec.x_range, spec.y_range, tmp_path)
        check(result, chart, spec.y_range)
