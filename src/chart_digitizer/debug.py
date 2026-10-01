"""Collects intermediate images from each stage when --debug is set."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from chart_digitizer.imgio import write_image


@dataclass(frozen=True)
class DebugSink:
    directory: Path | None = None

    @property
    def enabled(self) -> bool:
        return self.directory is not None

    def save(self, name: str, image: np.ndarray) -> None:
        """Write ``image`` as ``<directory>/<name>.png``. Bool masks are saved black on white."""
        if self.directory is None:
            return
        if image.dtype == bool:
            image = np.where(image, 0, 255).astype(np.uint8)
        write_image(self.directory / f"{name}.png", image)
