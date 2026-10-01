"""Stage 1: load the image."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from chart_digitizer.debug import DebugSink
from chart_digitizer.imgio import read_image


def load(path: Path, debug: DebugSink) -> np.ndarray:
    image = read_image(path)
    debug.save("01_input", image)
    return image
