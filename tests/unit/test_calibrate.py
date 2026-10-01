import numpy as np
import pytest

from chart_digitizer.config import AxesConfig
from chart_digitizer.models import AxisMap, PlotBox
from chart_digitizer.stages.calibrate import calibrate

BOX = PlotBox(left=40, top=40, right=2039, bottom=2039)


def test_linear_corners_map_to_ranges() -> None:
    cal = calibrate(BOX, AxesConfig(x_range=(0, 100), y_range=(-5, 5)))
    assert cal.x.to_data(BOX.left) == pytest.approx(0)
    assert cal.x.to_data(BOX.right) == pytest.approx(100)
    assert cal.y.to_data(BOX.bottom) == pytest.approx(-5)
    assert cal.y.to_data(BOX.top) == pytest.approx(5)
    assert cal.y.to_data((BOX.top + BOX.bottom) / 2) == pytest.approx(0)


def test_log_axis_is_linear_in_log_space() -> None:
    axis = AxisMap(lo=1, hi=1000, px_lo=0, px_hi=300, log=True)
    assert axis.to_data(np.array([0, 100, 200, 300])) == pytest.approx([1, 10, 100, 1000])
    assert axis.to_px(10) == pytest.approx(100)


@pytest.mark.parametrize("log", [False, True])
def test_round_trip(log: bool) -> None:
    axis = AxisMap(lo=2, hi=50, px_lo=1900, px_hi=12, log=log)
    values = np.linspace(2, 50, 17)
    assert axis.to_data(axis.to_px(values)) == pytest.approx(values)


def test_reversed_range() -> None:
    cal = calibrate(BOX, AxesConfig(x_range=(100, 0), y_range=(0, 1)))
    assert cal.x.to_data(BOX.left) == pytest.approx(100)
    assert cal.x.to_data(BOX.right) == pytest.approx(0)
