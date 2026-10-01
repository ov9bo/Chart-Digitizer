"""Stage 1: illumination correction.

Photos of screens and paper have uneven lighting and vignetting, so a fixed "dark ink" threshold
fails. The background is estimated with a grayscale morphological closing (which erases dark
features narrower than the kernel: lines, text, the curve), smoothed, and divided out per channel.
The result has a near-white background everywhere and keeps the ink's relative darkness and hue.
"""

from __future__ import annotations

import cv2
import numpy as np

from chart_digitizer.config import IlluminationConfig

# The background is estimated at reduced resolution; it varies slowly, and this keeps it fast.
_WORK_SIZE = 512


def estimate_background(image: np.ndarray, kernel_px: int) -> np.ndarray:
    """Per-channel background brightness, float32, same shape as ``image``."""
    h, w = image.shape[:2]
    scale = min(1.0, _WORK_SIZE / max(h, w))
    small = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    k = max(3, int(round(kernel_px * scale)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(small, cv2.MORPH_CLOSE, kernel)
    smooth = cv2.GaussianBlur(closed, (0, 0), k / 2)
    background = cv2.resize(smooth, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return background.reshape(image.shape)


def flatten_illumination(image: np.ndarray, cfg: IlluminationConfig, reference_px: int) -> np.ndarray:
    """Divide out the background so it becomes white. ``reference_px`` scales ``cfg.kernel_frac``."""
    if not cfg.enabled:
        return image
    background = estimate_background(image, int(round(cfg.kernel_frac * reference_px)))
    flat = image.astype(np.float32) * 255.0 / np.maximum(background, 1.0)
    return np.clip(flat, 0, 255).astype(np.uint8)
