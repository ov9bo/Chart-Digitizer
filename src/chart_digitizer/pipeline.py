"""Run the stages in order and write the outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from chart_digitizer import __version__
from chart_digitizer.config import Config
from chart_digitizer.debug import DebugSink
from chart_digitizer.errors import CornerError
from chart_digitizer.imgio import write_image
from chart_digitizer.models import Calibration, Corners, CurveMask, Points, Rectified, Trace
from chart_digitizer.output.render import draw_overlay, draw_rectified, render_replot
from chart_digitizer.output.writers import write_csv, write_json
from chart_digitizer.stages.calibrate import calibrate
from chart_digitizer.stages.clutter import remove_clutter
from chart_digitizer.stages.illumination import flatten_illumination
from chart_digitizer.stages.ink import find_ink
from chart_digitizer.stages.interactive import pick_corners
from chart_digitizer.stages.load import load
from chart_digitizer.stages.plotarea import detect_plot_area
from chart_digitizer.stages.rectify import rectify
from chart_digitizer.stages.resample import requested_x, resample
from chart_digitizer.stages.select import select_curves
from chart_digitizer.stages.trace import trace


@dataclass(frozen=True, eq=False)
class Result:
    image_path: Path
    out_dir: Path
    image: np.ndarray
    corners: Corners
    rectified: Rectified
    calibration: Calibration
    curve: CurveMask
    trace: Trace
    points: Points

    @property
    def warnings(self) -> tuple[str, ...]:
        return self.points.warnings


def output_dir(image_path: Path, cfg: Config, name: str | None = None) -> Path:
    """OUT/<name>, where the name defaults to the image's file name without its extension."""
    return cfg.output.dir / (name or image_path.stem)


def resolve_corners(image: np.ndarray, cfg: Config, debug: DebugSink) -> Corners:
    """Corners from --corners, from automatic detection, or clicked by the user.

    The interactive picker starts from the detected corners when detection succeeds.
    """
    mode = cfg.corners.mode
    if mode == "manual":
        assert cfg.corners.points is not None  # guaranteed by CornersConfig
        return Corners(points=np.asarray(cfg.corners.points, dtype=np.float64), source="manual")
    if mode == "auto":
        return detect_plot_area(image, cfg.plot_area, debug)
    try:
        initial: Corners | None = detect_plot_area(image, cfg.plot_area, debug)
        note = "Detected corners shown: drag to adjust, or press Enter to accept"
    except CornerError:
        initial, note = None, "The plot area was not found automatically: click its 4 corners"
    return pick_corners(image, initial, note, cfg.interactive)


def plot_size_px(corners: Corners) -> int:
    """Longest side of the plot's quadrilateral, in source pixels."""
    pts = corners.points
    return int(round(max(np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4))))


def digitize(image_path: Path, cfg: Config, name: str | None = None) -> Result:
    """Run every stage on one image. ``name`` overrides the output folder name (see ``output_dir``)."""
    out_dir = output_dir(image_path, cfg, name)
    debug = DebugSink(out_dir / "debug" if cfg.output.debug else None)
    image = load(image_path, debug)
    corners = resolve_corners(image, cfg, debug)
    flat = flatten_illumination(image, cfg.illumination, plot_size_px(corners))
    if cfg.illumination.enabled:
        debug.save("03_flat", flat)
    rectified = rectify(flat, corners, cfg.rectify, debug)
    calibration = calibrate(rectified.box, cfg.axes)
    box = rectified.box
    margin_px = int(round(cfg.clutter.box_margin_frac * max(box.width, box.height)))
    ink = find_ink(rectified.image, box, cfg.curve, margin_px, debug)
    cleaned = remove_clutter(ink, box, cfg.clutter, debug)
    curve = select_curves(cleaned, cfg.curve, debug)[0]
    traced = trace(curve, rectified.box, cfg.trace)
    if debug.enabled:
        debug.save("08_trace", draw_rectified(rectified, traced))
    points = resample(traced, calibration, requested_x(cfg.axes, cfg.sampling), cfg.sampling.edge_tol_px)
    return Result(
        image_path=image_path,
        out_dir=out_dir,
        image=image,
        corners=corners,
        rectified=rectified,
        calibration=calibration,
        curve=curve,
        trace=traced,
        points=points,
    )


def metadata(result: Result, cfg: Config) -> dict[str, Any]:
    first, last = result.calibration.x.to_data(result.trace.cols[[0, -1]].astype(np.float64))
    return {
        "tool": "chart-digitizer",
        "version": __version__,
        "image": str(result.image_path),
        "axes": cfg.axes.model_dump(mode="json"),
        "corners": {"source": result.corners.source, "points": result.corners.points.tolist()},
        "curve": result.curve.label,
        "trace": {
            "x_extent": sorted([float(first), float(last)]),
            "filled_fraction": result.trace.filled_fraction,
        },
        "sampling": cfg.sampling.model_dump(mode="json", exclude_none=True),
    }


def write_outputs(result: Result, cfg: Config) -> Path:
    """Write points.csv, points.json, overlay.png and replot.png; return the output directory."""
    out = result.out_dir
    write_csv(out / "points.csv", result.points)
    write_json(out / "points.json", metadata(result, cfg), result.points)
    overlay = draw_overlay(
        result.image, result.corners, result.rectified, result.calibration, result.trace, result.points
    )
    write_image(out / "overlay.png", overlay)
    render_replot(
        out / "replot.png", cfg.axes, result.calibration, result.trace, result.points, result.image_path.name
    )
    return out
