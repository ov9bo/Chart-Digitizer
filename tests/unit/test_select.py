import numpy as np
import pytest

from chart_digitizer.config import CurveConfig
from chart_digitizer.debug import DebugSink
from chart_digitizer.errors import SelectionError
from chart_digitizer.models import CurveMask
from chart_digitizer.stages.select import group_components, select_curves


def ink_of(mask: np.ndarray) -> CurveMask:
    return CurveMask(mask=mask, weight=mask.astype(np.float32), label="dark")


def test_pieces_of_a_split_curve_are_grouped_and_clutter_is_left_out() -> None:
    mask = np.zeros((100, 200), bool)
    mask[50:53, 0:90] = True  # curve, left of a removed crossing
    mask[50:53, 95:200] = True  # curve, right of it
    mask[10:40, 20:60] = True  # leftover blob stacked above the curve
    first, second = select_curves(ink_of(mask), CurveConfig(), DebugSink())
    assert first.mask[51].sum() == 195
    assert not first.mask[10:40].any()
    assert second.mask[10:40, 20:60].all()
    assert second.label == "dark#2"
    assert first.weight[0, 0] == 0


def test_group_components_orders_by_width() -> None:
    columns = [np.arange(10), np.arange(100), np.arange(100, 150), np.arange(40, 60)]
    assert group_components(columns, 0.3) == [[1, 2], [3, 0]]


def test_overlap_threshold() -> None:
    columns = [np.arange(100), np.arange(80, 180)]  # 20% of the second overlaps the first
    assert group_components(columns, 0.3) == [[0, 1]]
    assert group_components(columns, 0.1) == [[0], [1]]


def test_empty_mask_fails_with_hint() -> None:
    with pytest.raises(SelectionError) as info:
        select_curves(ink_of(np.zeros((20, 20), bool)), CurveConfig(), DebugSink())
    assert "--no-clutter" in (info.value.hint or "")
