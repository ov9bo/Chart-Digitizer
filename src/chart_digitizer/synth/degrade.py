"""Photo-like degradations for synthetic charts. Each returns a new image (and corners if moved)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class DegradeSpec:
    perspective: float = 0.08  # max corner displacement, fraction of image size
    blur_sigma: float = 1.2  # Gaussian blur, pixels
    noise_sigma: float = 6.0  # Gaussian noise, gray levels
    vignette: float = 0.35  # brightness loss at the corners, fraction
    moire_amplitude: float = 8.0  # gray levels
    moire_period_px: float = 5.0
    jpeg_quality: int = 70
    shadow: float = 0.0  # brightness loss inside a hard-edged shadow, fraction (0 = none)
    shadow_penumbra_px: float = 6.0  # width of the shadow's soft edge
    tint: tuple[float, float, float] = (1.0, 1.0, 1.0)  # per-channel gain (B, G, R): a colour cast


def perspective(
    image: np.ndarray, corners: np.ndarray, strength: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Warp as if photographed at an angle; returns the warped image and moved plot corners.

    The canvas grows to hold the whole warped page, so the plot is never cropped.
    """
    h, w = image.shape[:2]
    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    dst = src + rng.uniform(-strength, strength, size=(4, 2)).astype(np.float32) * np.array([w, h], np.float32)
    lo, hi = np.floor(dst.min(axis=0)), np.ceil(dst.max(axis=0))
    homography = cv2.getPerspectiveTransform(src, (dst - lo).astype(np.float32))
    size = (int(hi[0] - lo[0]) + 1, int(hi[1] - lo[1]) + 1)
    warped = cv2.warpPerspective(
        image, homography, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(90, 90, 90)
    )
    moved = cv2.perspectiveTransform(corners.reshape(-1, 1, 2).astype(np.float64), homography).reshape(-1, 2)
    return warped, moved


def blur(image: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(image, (0, 0), sigma) if sigma > 0 else image


def vignette(image: np.ndarray, strength: float) -> np.ndarray:
    h, w = image.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
    gain = 1.0 - strength * np.clip(r2 / 2.0, 0.0, 1.0)
    return np.clip(image.astype(np.float32) * gain[:, :, None], 0, 255).astype(np.uint8)


def shadow(image: np.ndarray, strength: float, penumbra_px: float, rng: np.random.Generator) -> np.ndarray:
    """Darken one side of a random straight edge through the image, as a hand or phone would."""
    h, w = image.shape[:2]
    angle = rng.uniform(0, 2 * np.pi)
    through = rng.uniform(0.3, 0.7, size=2) * np.array([w, h])
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    distance = (xx - through[0]) * np.cos(angle) + (yy - through[1]) * np.sin(angle)
    inside = np.clip(0.5 + distance / max(penumbra_px, 1e-6), 0.0, 1.0)
    gain = 1.0 - strength * inside * inside * (3 - 2 * inside)  # smoothstep across the penumbra
    return np.clip(image.astype(np.float32) * gain[:, :, None], 0, 255).astype(np.uint8)


def tint(image: np.ndarray, gains: tuple[float, float, float]) -> np.ndarray:
    return np.clip(image.astype(np.float32) * np.asarray(gains, np.float32), 0, 255).astype(np.uint8)


def moire(image: np.ndarray, amplitude: float, period_px: float, rng: np.random.Generator) -> np.ndarray:
    h, w = image.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    angle = rng.uniform(0, np.pi)
    phase = (xx * np.cos(angle) + yy * np.sin(angle)) * (2 * np.pi / period_px)
    pattern = amplitude * np.sin(phase) * np.sin(phase * 0.07)
    return np.clip(image.astype(np.float32) + pattern[:, :, None], 0, 255).astype(np.uint8)


def noise(image: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return np.clip(image.astype(np.float32) + rng.normal(0, sigma, image.shape), 0, 255).astype(np.uint8)


def jpeg(image: np.ndarray, quality: int) -> np.ndarray:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise OSError("JPEG encoding failed")
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def degrade(
    image: np.ndarray, corners: np.ndarray, spec: DegradeSpec, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Apply every degradation in ``spec`` in a photo-like order."""
    if spec.perspective > 0:
        image, corners = perspective(image, corners, spec.perspective, rng)
    image = blur(image, spec.blur_sigma)
    image = vignette(image, spec.vignette)
    if spec.shadow > 0:
        image = shadow(image, spec.shadow, spec.shadow_penumbra_px, rng)
    if spec.tint != (1.0, 1.0, 1.0):
        image = tint(image, spec.tint)
    image = moire(image, spec.moire_amplitude, spec.moire_period_px, rng)
    image = noise(image, spec.noise_sigma, rng)
    return jpeg(image, spec.jpeg_quality), corners
