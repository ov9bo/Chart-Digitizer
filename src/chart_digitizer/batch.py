"""Batch mode: digitize every image in a folder, keep going past failures, write summary.csv.

Each image can have its own settings (typically its axis ranges) in a YAML file with the same
stem next to it: ``chart1.png`` + ``chart1.yaml``. See ``image_config_data`` for the merge order.
"""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from chart_digitizer.config import Config, merge, read_yaml
from chart_digitizer.errors import DigitizeError
from chart_digitizer.pipeline import digitize, write_outputs

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
SIDECAR_SUFFIXES = (".yaml", ".yml")
SUMMARY_NAME = "summary.csv"

Status = Literal["ok", "failed", "error"]


@dataclass(frozen=True)
class Job:
    image: Path
    name: str  # output folder name under the output root
    config: Config
    sidecar: Path | None


@dataclass(frozen=True)
class Outcome:
    image: Path
    status: Status  # ok; failed: a stage gave up (DigitizeError); error: an unexpected exception
    out_dir: Path | None = None
    points_kept: int | None = None
    points_total: int | None = None
    corners: str | None = None
    filled_fraction: float | None = None
    warnings: tuple[str, ...] = ()
    error: str | None = None


def find_images(folder: Path) -> list[Path]:
    """Image files directly inside ``folder`` (not in subfolders), sorted by name."""
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def output_names(images: Iterable[Path]) -> dict[Path, str]:
    """Output folder name per image: its stem, or stem_ext where two images share a stem."""
    images = list(images)
    counts = Counter(p.stem.lower() for p in images)  # Windows paths are case-insensitive
    return {p: p.stem if counts[p.stem.lower()] == 1 else f"{p.stem}_{p.suffix.lstrip('.').lower()}" for p in images}


def sidecar_path(image: Path) -> Path | None:
    """``<stem>.yaml`` or ``<stem>.yml`` next to the image, if present."""
    for suffix in SIDECAR_SUFFIXES:
        candidate = image.with_suffix(suffix)
        if candidate.is_file():
            return candidate
    return None


def image_config_data(base: dict[str, Any], sidecar: Path | None, overrides: dict[str, Any]) -> dict[str, Any]:
    """Settings for one image: --config file, then the image's own YAML, then command-line flags."""
    data = merge(base, read_yaml(sidecar)) if sidecar is not None else base
    return merge(data, overrides)


def process(job: Job) -> Outcome:
    """Digitize one image and write its outputs, turning any failure into an ``Outcome``."""
    try:
        result = digitize(job.image, job.config, job.name)
        out = write_outputs(result, job.config)
    except DigitizeError as exc:
        return Outcome(job.image, "failed", error=str(exc))
    except Exception as exc:  # noqa: BLE001 - one bad image must not stop the batch; reported, not hidden
        return Outcome(job.image, "error", error=f"internal error ({type(exc).__name__}): {exc}")
    return Outcome(
        job.image,
        "ok",
        out_dir=out,
        points_kept=int(result.points.in_range.sum()),
        points_total=len(result.points.x),
        corners=result.corners.source,
        filled_fraction=result.trace.filled_fraction,
        warnings=result.warnings,
    )


def run_batch(jobs: Iterable[Job], report: Callable[[int, Outcome], None]) -> list[Outcome]:
    """Process every job in order, calling ``report(number, outcome)`` (from 1) as each one finishes."""
    outcomes = []
    for number, job in enumerate(jobs, start=1):
        outcome = process(job)
        report(number, outcome)
        outcomes.append(outcome)
    return outcomes


def summary_row(outcome: Outcome) -> dict[str, str]:
    return {
        "image": outcome.image.name,
        "status": outcome.status,
        "output": str(outcome.out_dir) if outcome.out_dir is not None else "",
        "points_kept": "" if outcome.points_kept is None else str(outcome.points_kept),
        "points_total": "" if outcome.points_total is None else str(outcome.points_total),
        "corners": outcome.corners or "",
        "filled_fraction": "" if outcome.filled_fraction is None else f"{outcome.filled_fraction:.4f}",
        "warnings": " | ".join(outcome.warnings),
        "error": (outcome.error or "").replace("\n", " "),
    }


def write_summary(path: Path, outcomes: list[Outcome]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [summary_row(o) for o in outcomes]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_row(Outcome(Path(), "ok"))))
        writer.writeheader()
        writer.writerows(rows)
