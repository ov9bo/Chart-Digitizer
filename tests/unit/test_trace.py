import numpy as np
import pytest

from chart_digitizer.config import TraceConfig
from chart_digitizer.errors import TraceError
from chart_digitizer.models import CurveMask, PlotBox
from chart_digitizer.stages.trace import find_runs, trace

BOX = PlotBox(left=10, top=10, right=409, bottom=409)
SHAPE = (420, 420)
CFG = TraceConfig()


def curve_mask(rows_of: np.ndarray, thickness: int = 5) -> CurveMask:
    """A mask with a ``thickness``-pixel band centred on rows_of[col - BOX.left]."""
    mask = np.zeros(SHAPE, bool)
    half = thickness // 2
    for i, centre in enumerate(rows_of):
        c = BOX.left + i
        mask[int(round(centre)) - half : int(round(centre)) + half + 1, c] = True
    return CurveMask(mask=mask, weight=mask.astype(np.float32), label="test")


def test_find_runs() -> None:
    mask = np.array([[1, 0], [1, 0], [0, 1], [1, 1]], bool)
    runs = find_runs(mask, mask.astype(np.float32), row_offset=5)
    assert runs.col.tolist() == [0, 0, 1]
    assert runs.start.tolist() == [5, 8, 7]
    assert runs.end.tolist() == [7, 9, 9]
    assert runs.center.tolist() == [5.5, 8.0, 7.5]
    assert runs.offsets.tolist() == [0, 2, 3]


def test_straight_line() -> None:
    rows = np.linspace(380, 40, BOX.width + 1)
    result = trace(curve_mask(rows), BOX, CFG)
    assert result.cols[0] == BOX.left and result.cols[-1] == BOX.right
    assert np.abs(result.rows - np.round(rows)).max() <= 0.5
    assert not result.filled.any()
    assert result.warnings == ()


def test_gap_is_filled_and_flagged() -> None:
    rows = 200 + 50 * np.sin(np.linspace(0, 3, BOX.width + 1))
    curve = curve_mask(rows)
    curve.mask[:, 200:215] = False
    result = trace(curve, BOX, CFG)
    gap = (result.cols >= 200) & (result.cols < 215)
    assert result.filled[gap].all() and result.filled.sum() == 15
    assert np.abs(result.rows[gap] - rows[gap.nonzero()[0]]).max() < 2


def test_ignores_distant_speck() -> None:
    rows = np.full(BOX.width + 1, 300.0)
    curve = curve_mask(rows)
    curve.mask[50:56, 150:153] = True
    result = trace(curve, BOX, CFG)
    assert np.abs(result.rows - 300).max() < 1e-9


def test_steep_step_is_followed() -> None:
    # A near-vertical rise: long runs stacked end to end must connect.
    rows = np.where(np.arange(BOX.width + 1) < 200, 380.0, 40.0)
    curve = curve_mask(rows)
    curve.mask[40:381, BOX.left + 200] = True
    result = trace(curve, BOX, CFG)
    assert result.cols[0] == BOX.left and result.cols[-1] == BOX.right
    assert result.rows[0] == pytest.approx(380) and result.rows[-1] == pytest.approx(40)


def test_short_trace_warns() -> None:
    rows = np.full(BOX.width + 1, 200.0)
    curve = curve_mask(rows)
    curve.mask[:, BOX.left + 200 :] = False
    result = trace(curve, BOX, CFG)
    assert any("spans only" in w for w in result.warnings)


def test_empty_mask_fails() -> None:
    empty = CurveMask(mask=np.zeros(SHAPE, bool), weight=np.zeros(SHAPE, np.float32), label="none")
    with pytest.raises(TraceError, match="No curve pixels"):
        trace(empty, BOX, CFG)


def test_tiny_trace_fails() -> None:
    mask = np.zeros(SHAPE, bool)
    mask[200:203, 100:105] = True
    with pytest.raises(TraceError, match="spans only"):
        trace(CurveMask(mask=mask, weight=mask.astype(np.float32), label="t"), BOX, CFG)


def with_removed(curve: CurveMask, removed: np.ndarray) -> CurveMask:
    return CurveMask(mask=curve.mask, weight=curve.weight, label=curve.label, removed=removed)


def test_plateau_removed_as_line_is_traced() -> None:
    # A curve that rises, runs flat for 150 columns, then rises again. Clutter removal took the
    # flat stretch as a horizontal line; the trace follows it instead of bridging a straight chord.
    rows = np.concatenate([np.linspace(380, 250, 120), np.full(150, 250.0), np.linspace(250, 40, 130)])
    full = curve_mask(rows)
    removed = np.zeros(SHAPE, bool)
    flat = np.s_[:, BOX.left + 120 : BOX.left + 270]
    removed[flat] = full.mask[flat]
    full.mask[flat] = False
    result = trace(with_removed(full, removed), BOX, CFG)
    assert result.cols[0] == BOX.left and result.cols[-1] == BOX.right
    assert not result.filled.any()
    assert np.abs(result.rows - np.round(rows)).max() <= 0.5
    assert any("straight lines" in w for w in result.warnings)


def test_gridline_does_not_extend_the_curve() -> None:
    rows = np.full(BOX.width + 1, 300.0)
    curve = curve_mask(rows)
    curve.mask[:, BOX.left + 250 :] = False
    removed = np.zeros(SHAPE, bool)
    removed[299:302, :] = True  # a gridline the curve ends on
    result = trace(with_removed(curve, removed), BOX, CFG)
    assert result.cols[0] == BOX.left and result.cols[-1] == BOX.left + 249


def test_gridline_does_not_cross_a_gap() -> None:
    # Two pieces of curve at different heights; a gridline joins neither and must not be used.
    rows = np.where(np.arange(BOX.width + 1) < 200, 300.0, 280.0)
    curve = curve_mask(rows)
    curve.mask[:, BOX.left + 195 : BOX.left + 205] = False
    removed = np.zeros(SHAPE, bool)
    removed[100:103, :] = True
    result = trace(with_removed(curve, removed), BOX, CFG)
    assert result.filled.sum() == 10
    assert result.rows.max() <= 300 and result.rows.min() >= 280


def test_line_cost_must_be_below_skip_cost() -> None:
    with pytest.raises(ValueError, match="line_cost"):
        TraceConfig(line_cost=0.5, skip_cost=0.5)
