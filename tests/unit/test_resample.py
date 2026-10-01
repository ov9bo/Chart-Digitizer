import numpy as np
import pytest

from chart_digitizer.config import AxesConfig, SamplingConfig
from chart_digitizer.models import AxisMap, Calibration, Trace
from chart_digitizer.stages.resample import requested_x, resample

AXES = AxesConfig(x_range=(0, 100), y_range=(0, 100))
# Columns 0..1000 map to x 0..100; rows 1000..0 map to y 0..100.
CAL = Calibration(x=AxisMap(0, 100, 0, 1000), y=AxisMap(0, 100, 1000, 0))


def make_trace(first: int, last: int, filled: slice | None = None) -> Trace:
    cols = np.arange(first, last + 1)
    rows = 1000.0 - cols  # y == x
    flags = np.zeros(cols.size, bool)
    if filled is not None:
        flags[filled] = True
    return Trace(cols=cols, rows=rows, filled=flags)


def test_requested_points() -> None:
    assert requested_x(AXES, SamplingConfig(points=5)).tolist() == [0, 25, 50, 75, 100]


def test_requested_step_includes_end() -> None:
    assert requested_x(AXES, SamplingConfig(step=12.5)).tolist() == [0, 12.5, 25, 37.5, 50, 62.5, 75, 87.5, 100]


def test_requested_values_sorted_unique() -> None:
    assert requested_x(AXES, SamplingConfig(x_values=(30, 10, 30))).tolist() == [10, 30]


def test_requested_log_spacing() -> None:
    axes = AxesConfig(x_range=(1, 1000), y_range=(0, 1), x_log=True)
    assert requested_x(axes, SamplingConfig(points=4)) == pytest.approx([1, 10, 100, 1000])


def test_reversed_range_is_sorted() -> None:
    axes = AxesConfig(x_range=(100, 0), y_range=(0, 1))
    assert requested_x(axes, SamplingConfig(points=3)).tolist() == [0, 50, 100]


def test_values_follow_trace() -> None:
    points = resample(make_trace(0, 1000), CAL, np.array([0, 12.34, 100]), edge_tol_px=1.5)
    assert points.y == pytest.approx([0, 12.34, 100])
    assert points.in_range.all() and not points.interpolated.any()
    assert points.warnings == ()


def test_no_extrapolation() -> None:
    points = resample(make_trace(200, 800), CAL, np.array([10, 20, 50, 80, 90]), edge_tol_px=1.5)
    assert points.in_range.tolist() == [False, True, True, True, False]
    assert np.isnan(points.y[[0, 4]]).all()
    assert points.y[1:4] == pytest.approx([20, 50, 80])
    assert any("outside the traced range" in w for w in points.warnings)


def test_edge_tolerance_snaps() -> None:
    points = resample(make_trace(1, 999), CAL, np.array([0, 100]), edge_tol_px=1.5)
    assert points.in_range.all()
    assert points.y == pytest.approx([0.1, 99.9])


def test_interpolated_flag() -> None:
    points = resample(make_trace(0, 1000, filled=slice(400, 500)), CAL, np.array([30, 45, 49.95, 60]), 1.5)
    assert points.interpolated.tolist() == [False, True, True, False]


def test_trace_warnings_are_kept() -> None:
    trace = Trace(cols=np.arange(0, 1001), rows=np.zeros(1001), filled=np.zeros(1001, bool), warnings=("w",))
    assert resample(trace, CAL, np.array([50.0]), 1.5).warnings == ("w",)
