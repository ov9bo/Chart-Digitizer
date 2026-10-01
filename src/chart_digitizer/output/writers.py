"""points.csv and points.json."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from chart_digitizer.models import Points

OUT_OF_RANGE = "out_of_range"


def point_flags(points: Points) -> list[str]:
    """'1' for interpolated, '0' for measured, 'out_of_range' where y is missing."""
    return [
        OUT_OF_RANGE if not ok else ("1" if interp else "0")
        for ok, interp in zip(points.in_range, points.interpolated, strict=True)
    ]


def _number(value: float) -> float | None:
    return None if not np.isfinite(value) else float(value)


def write_csv(path: Path, points: Points) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "y", "interpolated_flag"])
        for x, y, flag in zip(points.x, points.y, point_flags(points), strict=True):
            writer.writerow([f"{x:.6g}", "" if not np.isfinite(y) else f"{y:.6g}", flag])


def points_records(points: Points) -> list[dict[str, Any]]:
    return [
        {"x": _number(x), "y": _number(y), "interpolated_flag": flag}
        for x, y, flag in zip(points.x, points.y, point_flags(points), strict=True)
    ]


def write_json(path: Path, metadata: dict[str, Any], points: Points) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {**metadata, "warnings": list(points.warnings), "points": points_records(points)}
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
