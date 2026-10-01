"""Stage 3: warp the plot quadrilateral to an axis-aligned rectangle with a padded margin."""

from __future__ import annotations

import cv2
import numpy as np

from chart_digitizer.config import RectifyConfig
from chart_digitizer.debug import DebugSink
from chart_digitizer.errors import CornerError
from chart_digitizer.models import Corners, PlotBox, Rectified

_ORDER_HINT = (
    "Give corners in the order (xmin,ymin), (xmax,ymin), (xmax,ymax), (xmin,ymax); "
    "for an upright chart that is bottom-left, bottom-right, top-right, top-left. "
    "Or use --interactive to click them."
)
# Smallest accepted quadrilateral, as a fraction of the image area.
_MIN_AREA_FRAC = 1e-3


def signed_area(points: np.ndarray) -> float:
    """Shoelace area in image coordinates. Negative for the expected corner order."""
    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def is_convex(points: np.ndarray) -> bool:
    edges = np.roll(points, -1, axis=0) - points
    cross = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)
    return bool(np.all(cross > 0) or np.all(cross < 0))


def validate_corners(corners: Corners, image_shape: tuple[int, ...]) -> None:
    pts = corners.points
    if pts.shape != (4, 2) or not np.all(np.isfinite(pts)):
        raise CornerError(f"Expected 4 finite (x, y) corner points, got shape {pts.shape}", hint=_ORDER_HINT)
    if not is_convex(pts):
        raise CornerError("Corners don't form a convex quadrilateral", hint=_ORDER_HINT)
    area = signed_area(pts)
    if abs(area) < _MIN_AREA_FRAC * image_shape[0] * image_shape[1]:
        raise CornerError(f"Corners enclose almost no area ({abs(area):.0f} px²)", hint=_ORDER_HINT)
    if area > 0:
        raise CornerError("Corners are in mirrored order", hint=_ORDER_HINT)


def target_box(cfg: RectifyConfig) -> tuple[PlotBox, tuple[int, int]]:
    """Axes box in rectified pixels, and the full rectified image size (width, height)."""
    width, height = cfg.size
    pad_x = round(cfg.pad_frac * width)
    pad_y = round(cfg.pad_frac * height)
    box = PlotBox(left=pad_x, top=pad_y, right=pad_x + width - 1, bottom=pad_y + height - 1)
    return box, (width + 2 * pad_x, height + 2 * pad_y)


def box_corners(box: PlotBox) -> np.ndarray:
    """Rectified positions of (xmin,ymin), (xmax,ymin), (xmax,ymax), (xmin,ymax)."""
    return np.array(
        [[box.left, box.bottom], [box.right, box.bottom], [box.right, box.top], [box.left, box.top]],
        dtype=np.float64,
    )


def rectify(image: np.ndarray, corners: Corners, cfg: RectifyConfig, debug: DebugSink) -> Rectified:
    validate_corners(corners, image.shape)
    box, size = target_box(cfg)
    homography = cv2.getPerspectiveTransform(
        corners.points.astype(np.float32), box_corners(box).astype(np.float32)
    )
    warped = cv2.warpPerspective(
        image, homography, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    debug.save("04_rectified", warped)
    return Rectified(image=warped, homography=homography.astype(np.float64), box=box)
