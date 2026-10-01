"""Image file I/O that works with non-ASCII paths on Windows (cv2.imread/imwrite do not)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from chart_digitizer.errors import LoadError

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"})


def read_image(path: Path) -> np.ndarray:
    """Read an image as BGR uint8, applying EXIF orientation."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError as exc:
        raise LoadError(f"Could not read {path}: {exc}") from exc
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise LoadError(
            f"Could not decode {path} as an image",
            hint=f"Supported formats: {', '.join(sorted(IMAGE_SUFFIXES))}.",
        )
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise OSError(f"Could not encode image for {path}")
    encoded.tofile(path)
