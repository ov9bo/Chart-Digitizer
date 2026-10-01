import numpy as np
import pytest

from chart_digitizer.config import CurveConfig
from chart_digitizer.debug import DebugSink
from chart_digitizer.errors import SelectionError
from chart_digitizer.models import PlotBox
from chart_digitizer.stages.ink import find_ink, parse_color_bgr

FULL = PlotBox(left=0, top=0, right=59, bottom=59)


def image_with_lines() -> np.ndarray:
    image = np.full((60, 60, 3), 255, np.uint8)
    image[10:13, :] = (0, 0, 0)  # black
    image[30:33, :] = (40, 40, 200)  # dark red (BGR)
    image[50:53, :] = (200, 200, 200)  # light gray grid
    return image


def rows_of(mask: np.ndarray) -> list[int]:
    return np.flatnonzero(mask.any(axis=1)).tolist()


def test_parse_color() -> None:
    assert parse_color_bgr("#ff8000").tolist() == [0, 128, 255]
    assert parse_color_bgr("hsv:0,255,255").tolist() == [0, 0, 255]


def test_dark_mode_ignores_colored_and_gray_lines() -> None:
    ink = find_ink(image_with_lines(), FULL, CurveConfig(), 0, DebugSink())
    assert rows_of(ink.mask) == [10, 11, 12]
    assert ink.weight[11, 5] == pytest.approx(1.0)
    assert ink.weight[0, 0] == 0


def test_color_mode_picks_the_color() -> None:
    ink = find_ink(image_with_lines(), FULL, CurveConfig(color="#c82828"), 0, DebugSink())
    assert rows_of(ink.mask) == [30, 31, 32]


def test_ink_outside_the_box_and_margin_is_ignored() -> None:
    box = PlotBox(left=0, top=20, right=59, bottom=59)
    ink = find_ink(image_with_lines(), box, CurveConfig(color="#c82828"), 0, DebugSink())
    assert rows_of(ink.mask) == [30, 31, 32]
    with pytest.raises(SelectionError):
        find_ink(image_with_lines(), box, CurveConfig(), 5, DebugSink())
    assert rows_of(find_ink(image_with_lines(), box, CurveConfig(), 8, DebugSink()).mask) == [12]


def test_no_ink_fails_with_hint() -> None:
    blank = np.full((20, 20, 3), 255, np.uint8)
    with pytest.raises(SelectionError) as info:
        find_ink(blank, PlotBox(0, 0, 19, 19), CurveConfig(), 0, DebugSink())
    assert "--curve-color" in (info.value.hint or "")
