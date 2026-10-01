"""Regression test on the photographed sample charts listed in fixtures/samples_truth.yaml.

Each sample runs with its hand-picked corners and with automatically detected ones.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from chart_digitizer.config import build_config
from chart_digitizer.pipeline import digitize

pytestmark = pytest.mark.slow

ROOT = Path(__file__).parent.parent
SAMPLES: list[dict[str, Any]] = yaml.safe_load((Path(__file__).parent / "fixtures" / "samples_truth.yaml").read_text())[
    "samples"
]


@pytest.mark.parametrize("corners", ["manual", "auto"])
@pytest.mark.parametrize("sample", SAMPLES, ids=[Path(s["image"]).stem for s in SAMPLES])
def test_sample(sample: dict[str, Any], corners: str, tmp_path: Path) -> None:
    image = ROOT / sample["image"]
    if not image.exists():
        pytest.skip(f"{image} is not present")
    xs, expected = np.array(sample["points"], dtype=float).T
    cfg = build_config(
        {
            "axes": {"x_range": sample["x_range"], "y_range": sample["y_range"]},
            "corners": {"points": sample["corners"]} if corners == "manual" else {"mode": "auto"},
            "sampling": {"x_values": xs.tolist()},
            "output": {"dir": str(tmp_path)},
        }
    )
    result = digitize(image, cfg)
    assert np.allclose(result.points.x, xs)
    error = np.abs(result.points.y - expected)
    assert error.max() <= sample["tol"], dict(zip(xs.tolist(), result.points.y.round(2).tolist(), strict=True))
