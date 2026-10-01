"""End-to-end accuracy with the plot area found automatically instead of given.

The same bounds as with exact corners: RMSE < 1% of the y-range on clean and cluttered charts,
< 2% on photo-degraded ones, including a hard shadow and a colour cast.
"""

from pathlib import Path

import numpy as np
import pytest

from chart_digitizer.synth.degrade import DegradeSpec, degrade
from chart_digitizer.synth.generate import SynthSpec, render
from helpers import rmse_fraction, run_on_image

pytestmark = pytest.mark.slow

MIN_IN_RANGE_FRAC = 0.95
# Detected corners must land this close to the true ones, in source pixels.
MAX_CORNER_ERROR_PX = 2.0
LIGHTING = {
    "shadow": DegradeSpec(shadow=0.5),
    "shadow_hard": DegradeSpec(shadow=0.6, shadow_penumbra_px=2),
    "tint": DegradeSpec(tint=(0.7, 0.88, 1.0)),
    "shadow_tint": DegradeSpec(shadow=0.5, tint=(0.7, 0.88, 1.0)),
}


def check(image: np.ndarray, corners: np.ndarray, spec: SynthSpec, tmp_path: Path, max_rmse: float) -> None:
    chart = render(spec)
    result = run_on_image(image, None, spec.x_range, spec.y_range, tmp_path)
    assert result.corners.source == "auto"
    assert np.abs(result.corners.points - corners).max() <= MAX_CORNER_ERROR_PX
    assert result.points.in_range.mean() >= MIN_IN_RANGE_FRAC
    assert rmse_fraction(result, chart.truth_x, chart.truth_y, spec.y_range) < max_rmse


@pytest.mark.parametrize("clutter", [False, True], ids=["clean", "clutter"])
@pytest.mark.parametrize("kind", ["monotonic", "wavy"])
@pytest.mark.parametrize("seed", range(2))
def test_undegraded(clutter: bool, kind: str, seed: int, tmp_path: Path) -> None:
    spec = SynthSpec(seed=seed, kind=kind, clutter=clutter)
    chart = render(spec)
    check(chart.image, chart.corners, spec, tmp_path, max_rmse=0.01)


@pytest.mark.parametrize("kind", ["monotonic", "wavy"])
@pytest.mark.parametrize("seed", range(4))
def test_degraded(kind: str, seed: int, tmp_path: Path) -> None:
    spec = SynthSpec(seed=seed, kind=kind, clutter=True)
    chart = render(spec)
    image, corners = degrade(chart.image, chart.corners, DegradeSpec(), np.random.default_rng(seed))
    check(image, corners, spec, tmp_path, max_rmse=0.02)


@pytest.mark.parametrize("lighting", LIGHTING)
@pytest.mark.parametrize("kind", ["monotonic", "wavy"])
@pytest.mark.parametrize("seed", range(4))
def test_uneven_lighting(lighting: str, kind: str, seed: int, tmp_path: Path) -> None:
    spec = SynthSpec(seed=seed, kind=kind, clutter=True)
    chart = render(spec)
    image, corners = degrade(chart.image, chart.corners, LIGHTING[lighting], np.random.default_rng(seed))
    check(image, corners, spec, tmp_path, max_rmse=0.02)
