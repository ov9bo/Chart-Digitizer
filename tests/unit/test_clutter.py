import numpy as np

from chart_digitizer.config import ClutterConfig
from chart_digitizer.debug import DebugSink
from chart_digitizer.models import CurveMask, PlotBox
from chart_digitizer.stages.clutter import clutter_mask, ink_within, remove_clutter, split_clutter

# With a 1000 px box: lines >= 100 px are removed, breaks <= 15 px bridged, reach 6 px, specks < 20 px.
SIZE = 1000
BOX = PlotBox(left=0, top=0, right=SIZE, bottom=SIZE)
CFG = ClutterConfig()


def curve_mask(slope: float = 0.3, thickness: int = 5, through: tuple[int, int] = (0, 880)) -> np.ndarray:
    """A thick straight 'curve' rising left to right through the (col, row) point ``through``."""
    rows, cols = np.mgrid[0 : SIZE + 1, 0 : SIZE + 1]
    centre = through[1] - slope * (cols - through[0])
    return np.abs(rows - centre) / np.hypot(1, slope) <= thickness / 2  # perpendicular distance


def with_grid(curve: np.ndarray, width: int = 3) -> np.ndarray:
    mask = curve.copy()
    for k in range(100, SIZE, 200):
        mask[k : k + width, :] = True
        mask[:, k : k + width] = True
    return mask


def test_ink_within() -> None:
    mask = np.zeros((1, 10), bool)
    mask[0, 2] = True
    before, after = ink_within(mask, 3, axis=1)
    assert before[0].tolist() == [0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
    assert after[0].tolist() == [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]


def test_gridlines_removed_curve_kept_through_crossings() -> None:
    curve = curve_mask()  # crosses five vertical gridlines and, at a shallow angle, one horizontal
    cleaned = with_grid(curve) & ~clutter_mask(with_grid(curve), BOX, CFG)
    assert not (cleaned & ~curve).any()  # nothing but curve is left
    # Where the curve runs inside a gridline at a shallow angle, a sliver of it goes too.
    kept = (cleaned & curve).sum() / curve.sum()
    assert kept > 0.98
    assert cleaned.any(axis=0).all()  # every column still has curve ink


def test_steep_crossing_is_restored() -> None:
    curve = curve_mask(slope=4.0, thickness=8, through=(505, 500))
    mask = curve.copy()
    mask[:, 500:505] = True  # vertical reference line
    cleaned = mask & ~clutter_mask(mask, BOX, CFG)
    assert cleaned[:, 495:515].any(axis=0).all()
    assert not (cleaned & ~curve).any()


def test_uncrossable_cut_leaves_empty_columns_and_no_line_runs() -> None:
    # A thin sloped curve through a thick line: no row or small closing reconnects it, so the cut
    # is left for the trace to bridge, and the vertical line is not offered as a plateau.
    curve = curve_mask(slope=1.0, thickness=5, through=(505, 500))
    mask = curve.copy()
    mask[:, 500:511] = True
    horizontal, vertical, _ = split_clutter(mask, BOX, CFG)
    cleaned = mask & ~(horizontal | vertical)
    assert not (cleaned & ~curve).any()
    assert not horizontal.any()
    assert vertical[:, 500:511].all()


def test_dashed_line_and_specks_removed() -> None:
    curve = curve_mask()
    mask = curve.copy()
    for start in range(0, SIZE, 18):  # dashes 10 px long, 8 px gaps
        mask[950:953, start : start + 10] = True
    for r, c in ((100, 100), (300, 700), (700, 200)):
        mask[r : r + 6, c : c + 6] = True  # specks
    cleaned = mask & ~clutter_mask(mask, BOX, CFG)
    assert not (cleaned & ~curve).any()


def test_disabled_returns_input_unchanged() -> None:
    mask = with_grid(curve_mask())
    ink = CurveMask(mask=mask, weight=mask.astype(np.float32), label="dark")
    assert remove_clutter(ink, BOX, ClutterConfig(enabled=False), DebugSink()) is ink
    cleaned = remove_clutter(ink, BOX, CFG, DebugSink())
    assert cleaned.mask.sum() < mask.sum()
    assert not cleaned.weight[~cleaned.mask].any()
