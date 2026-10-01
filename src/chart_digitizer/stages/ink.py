"""Find candidate curve ink in the rectified, illumination-flattened image.

Dark mode keeps dark pixels with low CIELAB chroma, so a red reference line is not mistaken for a
black curve. Color mode keeps pixels close to ``--curve-color``. Everything outside the axes box
(plus a small margin) is dropped: tick labels and the photo's surroundings can't be the curve.
"""

from __future__ import annotations

import cv2
import numpy as np

from chart_digitizer.config import CurveConfig
from chart_digitizer.debug import DebugSink
from chart_digitizer.errors import SelectionError
from chart_digitizer.models import CurveMask, PlotBox


def parse_color_bgr(spec: str) -> np.ndarray:
    """Convert '#rrggbb' or 'hsv:h,s,v' (already validated by the config) to a BGR uint8 triple."""
    if spec.startswith("#"):
        r, g, b = (int(spec[i : i + 2], 16) for i in (1, 3, 5))
        return np.array([b, g, r], dtype=np.uint8)
    h, s, v = (int(part) for part in spec.split(":", 1)[1].split(","))
    hsv = np.array([[[h, s, v]]], dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]


def to_lab(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)


def chroma(image: np.ndarray, blur_sigma: float) -> np.ndarray:
    """CIELAB chroma per pixel, with a and b blurred first when ``blur_sigma`` > 0."""
    lab = to_lab(image)
    a, b = lab[:, :, 1], lab[:, :, 2]
    if blur_sigma > 0:
        a = cv2.GaussianBlur(a, (0, 0), blur_sigma)
        b = cv2.GaussianBlur(b, (0, 0), blur_sigma)
    return np.hypot(a, b)


def dark_ink(image: np.ndarray, cfg: CurveConfig, blur_sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Mask and weight for dark, colorless ink. Weight is darkness scaled to [0, 1]."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = (gray <= cfg.ink_max_value) & (chroma(image, blur_sigma) <= cfg.max_chroma)
    weight = (255.0 - gray.astype(np.float32)) / 255.0
    return mask, weight


def colored_ink(image: np.ndarray, color_bgr: np.ndarray, tol: float) -> tuple[np.ndarray, np.ndarray]:
    """Mask and weight for pixels within ``tol`` CIELAB distance of ``color_bgr``."""
    lab = to_lab(image)
    target = to_lab(color_bgr.reshape(1, 1, 3))[0, 0]
    distance = np.linalg.norm(lab - target, axis=2)
    mask = distance <= tol
    weight = np.clip(1.0 - distance / tol, 0.0, 1.0).astype(np.float32)
    return mask, weight


def box_region(shape: tuple[int, ...], box: PlotBox, margin_px: int) -> np.ndarray:
    """Bool mask of the axes box grown by ``margin_px`` on every side."""
    region = np.zeros(shape[:2], dtype=bool)
    region[
        max(box.top - margin_px, 0) : box.bottom + margin_px + 1,
        max(box.left - margin_px, 0) : box.right + margin_px + 1,
    ] = True
    return region


def find_ink(image: np.ndarray, box: PlotBox, cfg: CurveConfig, margin_px: int, debug: DebugSink) -> CurveMask:
    """All pixels that could belong to the curve, before clutter removal."""
    if cfg.color is None:
        blur_sigma = cfg.chroma_blur_frac * max(box.width, box.height)
        mask, weight = dark_ink(image, cfg, blur_sigma)
        label = "dark"
    else:
        mask, weight = colored_ink(image, parse_color_bgr(cfg.color), cfg.color_tol)
        label = cfg.color
    mask &= box_region(image.shape, box, margin_px)
    if not mask.any():
        raise SelectionError(
            f"No {'dark' if cfg.color is None else cfg.color} curve pixels found in the plot area",
            hint="Check the corners, or set --curve-color for a colored curve "
            "(curve.ink_max_value / curve.max_chroma / curve.color_tol in --config loosen the match).",
        )
    debug.save("05_ink", mask)
    return CurveMask(mask=mask, weight=np.where(mask, weight, 0.0).astype(np.float32), label=label)
