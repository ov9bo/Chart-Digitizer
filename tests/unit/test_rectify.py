import cv2
import numpy as np
import pytest

from chart_digitizer.config import RectifyConfig
from chart_digitizer.debug import DebugSink
from chart_digitizer.errors import CornerError
from chart_digitizer.models import Corners
from chart_digitizer.stages.rectify import box_corners, rectify, target_box, validate_corners

SHAPE = (600, 800, 3)
UPRIGHT = np.array([[100.0, 500.0], [700.0, 500.0], [700.0, 100.0], [100.0, 100.0]])
CFG = RectifyConfig(size=(400, 300), pad_frac=0.05)


def test_target_box_has_padding() -> None:
    box, (w, h) = target_box(CFG)
    assert (box.left, box.top) == (20, 15)
    assert (box.width, box.height) == (399, 299)
    assert (w, h) == (440, 330)


def test_accepts_expected_order() -> None:
    validate_corners(Corners(UPRIGHT, "manual"), SHAPE)


def test_rejects_mirrored_order() -> None:
    with pytest.raises(CornerError, match="mirrored") as info:
        validate_corners(Corners(UPRIGHT[::-1].copy(), "manual"), SHAPE)
    assert "--interactive" in (info.value.hint or "")


def test_rejects_self_intersecting() -> None:
    bowtie = UPRIGHT[[0, 2, 1, 3]]
    with pytest.raises(CornerError, match="convex"):
        validate_corners(Corners(bowtie, "manual"), SHAPE)


def test_rejects_tiny_area() -> None:
    tiny = np.array([[10.0, 12.0], [12.0, 12.0], [12.0, 10.0], [10.0, 10.0]])
    with pytest.raises(CornerError, match="area"):
        validate_corners(Corners(tiny, "manual"), SHAPE)


def test_homography_maps_corners_onto_box() -> None:
    skewed = np.array([[120.0, 520.0], [690.0, 470.0], [650.0, 90.0], [80.0, 130.0]])
    image = np.full(SHAPE, 255, np.uint8)
    rect = rectify(image, Corners(skewed, "manual"), CFG, DebugSink())
    mapped = cv2.perspectiveTransform(skewed.reshape(-1, 1, 2), rect.homography).reshape(-1, 2)
    assert mapped == pytest.approx(box_corners(rect.box), abs=1e-6)
    assert rect.image.shape[:2] == (330, 440)


def test_marked_pixel_lands_where_expected() -> None:
    image = np.full(SHAPE, 255, np.uint8)
    # The source quad centre maps to the box centre.
    cv2.circle(image, (400, 300), 3, (0, 0, 0), -1)
    rect = rectify(image, Corners(UPRIGHT, "manual"), CFG, DebugSink())
    dark = np.argwhere(rect.image[:, :, 0] < 128)
    row, col = dark.mean(axis=0)
    box = rect.box
    assert col == pytest.approx((box.left + box.right) / 2, abs=0.5)
    assert row == pytest.approx((box.top + box.bottom) / 2, abs=0.5)
