"""Configuration schema and loading.

Values are merged in this order, later winning: built-in defaults, the --config YAML file, the
image's own YAML file (batch.sidecar_path), CLI flags.
The ``sampling`` and ``corners`` sections are replaced as a whole when the CLI sets any of their
keys, so ``--step`` on the command line doesn't collide with ``points`` from a YAML file.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from chart_digitizer.errors import ConfigError

Pair = tuple[float, float]

DEFAULT_POINTS = 50
_HEX_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")
_HSV_COLOR = re.compile(r"^hsv:\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)$")
_REPLACED_WHOLE = frozenset({"sampling", "corners"})


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AxesConfig(_Section):
    x_range: Pair
    y_range: Pair
    x_log: bool = False
    y_log: bool = False
    grid_step: Pair | None = None

    @model_validator(mode="after")
    def _check_ranges(self) -> AxesConfig:
        for name, (lo, hi), log in (("x", self.x_range, self.x_log), ("y", self.y_range, self.y_log)):
            if lo == hi:
                raise ValueError(f"{name}_range needs two different values, got {lo},{hi}")
            if log and (lo <= 0 or hi <= 0):
                raise ValueError(f"{name}_range must be positive on a log axis, got {lo},{hi}")
        if self.grid_step is not None and min(self.grid_step) <= 0:
            raise ValueError(f"grid_step must be positive, got {self.grid_step}")
        return self


class CornersConfig(_Section):
    mode: Literal["auto", "manual", "interactive"] = "auto"
    # Source-image pixels, ordered (xmin,ymin), (xmax,ymin), (xmax,ymax), (xmin,ymax).
    points: tuple[Pair, Pair, Pair, Pair] | None = None

    @model_validator(mode="before")
    @classmethod
    def _points_imply_manual(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("points") is not None and "mode" not in data:
            return {**data, "mode": "manual"}
        return data

    @model_validator(mode="after")
    def _check_mode(self) -> CornersConfig:
        if self.mode == "manual" and self.points is None:
            raise ValueError("corners.mode is 'manual' but no corner points were given")
        if self.mode != "manual" and self.points is not None:
            raise ValueError(f"corner points were given but corners.mode is '{self.mode}'")
        return self


class SamplingConfig(_Section):
    points: int | None = Field(default=None, ge=2)
    x_values: tuple[float, ...] | None = None
    step: float | None = Field(default=None, gt=0)
    # Requested x within this many rectified pixels of the traced extent is snapped onto it.
    edge_tol_px: float = Field(default=1.5, ge=0)

    @model_validator(mode="before")
    @classmethod
    def _default_points(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        chosen = [k for k in ("points", "x_values", "step") if data.get(k) is not None]
        if len(chosen) > 1:
            raise ValueError(f"use only one of points, x_values, step (got {', '.join(chosen)})")
        if not chosen:
            return {**data, "points": DEFAULT_POINTS}
        return data

    @model_validator(mode="after")
    def _check_values(self) -> SamplingConfig:
        if self.x_values is not None and len(self.x_values) == 0:
            raise ValueError("x_values is empty")
        return self


class PlotAreaConfig(_Section):
    """Automatic plot-area detection. Lengths are fractions of the image's longer side."""

    # The image is shrunk so its longer side is at most this many pixels before detection.
    work_size: int = Field(default=1600, ge=200)
    # Dark features thinner than this are candidate line ink (black-hat closing kernel).
    line_kernel_frac: float = Field(default=0.015, gt=0, le=0.2)
    # Line ink must be at least this much darker than its surroundings, as a fraction.
    min_contrast: float = Field(default=0.06, gt=0, lt=1)
    # ...and this many times the typical contrast of the paper around it (the median over square
    # blocks of noise_block_frac), so paper texture and JPEG noise in shadows are not ink.
    noise_contrast_mult: float = Field(default=3.0, ge=0)
    noise_block_frac: float = Field(default=0.025, gt=0, le=0.5)
    # Largest tilt from horizontal or vertical accepted for a plot side, in degrees.
    max_tilt_deg: float = Field(default=12.0, gt=0, le=30)
    # A plot side must be at least this long.
    min_side_frac: float = Field(default=0.15, gt=0, le=1)
    # Ink runs along a line are found after smoothing with a window this long, so dashes and
    # breaks shorter than about half of it are bridged.
    gap_frac: float = Field(default=0.02, gt=0)
    # Fraction of that window that must hold ink for the line to continue.
    min_fill: float = Field(default=0.5, gt=0, le=1)
    # Two lines meet at a corner if each one's ink run reaches within this distance of it.
    corner_tol_frac: float = Field(default=0.01, ge=0)
    # At most this many of the strongest lines of each orientation are considered.
    max_lines: int = Field(default=40, ge=2)


class InteractiveConfig(_Section):
    """The --interactive corner-picking window."""

    # The image is shown scaled down to fit in this many screen pixels (width, height).
    max_window: tuple[int, int] = (1400, 900)
    # The magnifier shows the source image around the cursor at this zoom, in a square this big.
    magnifier_zoom: float = Field(default=4.0, ge=1)
    magnifier_px: int = Field(default=220, ge=50)
    # A click within this many screen pixels of a placed corner grabs it for dragging.
    grab_radius_px: float = Field(default=12.0, gt=0)

    @model_validator(mode="after")
    def _check_window(self) -> InteractiveConfig:
        if min(self.max_window) < 200:
            raise ValueError(f"interactive.max_window must be at least 200x200, got {self.max_window}")
        return self


class OutputConfig(_Section):
    dir: Path = Path("out")
    debug: bool = False


class RectifyConfig(_Section):
    # Size of the axes box in the rectified image, in pixels (width, height).
    size: tuple[int, int] = (2000, 2000)
    # Margin kept around the axes box, as a fraction of the box size.
    pad_frac: float = Field(default=0.02, ge=0.0, le=0.25)

    @model_validator(mode="after")
    def _check_size(self) -> RectifyConfig:
        if min(self.size) < 100:
            raise ValueError(f"rectify.size must be at least 100x100, got {self.size}")
        return self


class IlluminationConfig(_Section):
    enabled: bool = True
    # Background-estimation kernel, as a fraction of the plot size. Dark features narrower than
    # this (lines, text) are treated as ink; wider ones are treated as shading.
    kernel_frac: float = Field(default=0.04, gt=0, le=0.5)


class CurveConfig(_Section):
    # None selects the dark curve; otherwise "#rrggbb" or "hsv:h,s,v" (OpenCV ranges: h 0-179).
    color: str | None = None
    # Maximum CIELAB distance from `color` still counted as curve ink.
    color_tol: float = Field(default=40.0, gt=0)
    # Dark-curve mode: brightest gray value (0-255) still counted as ink.
    ink_max_value: int = Field(default=110, ge=1, le=254)
    # Dark-curve mode: largest CIELAB chroma still counted as ink; rejects colored lines.
    max_chroma: float = Field(default=30.0, gt=0)
    # Chroma is measured after a Gaussian blur of this sigma (fraction of plot size), which averages
    # out the colored fringes of moire and demosaicing.
    chroma_blur_frac: float = Field(default=0.001, ge=0)
    # A component is kept alongside the dominant one only if at most this fraction of its columns
    # are already covered; pieces of the same curve split at crossings barely overlap.
    max_overlap_frac: float = Field(default=0.3, ge=0, le=1)

    @model_validator(mode="after")
    def _check_color(self) -> CurveConfig:
        if self.color is None:
            return self
        if _HEX_COLOR.match(self.color):
            return self
        match = _HSV_COLOR.match(self.color)
        if match is None:
            raise ValueError(f"curve color must be '#rrggbb' or 'hsv:h,s,v', got '{self.color}'")
        h, s, v = (int(g) for g in match.groups())
        if h > 179 or s > 255 or v > 255:
            raise ValueError(f"hsv color out of range (h 0-179, s/v 0-255): '{self.color}'")
        return self


class ClutterConfig(_Section):
    enabled: bool = True
    # Straight horizontal/vertical ink at least this long (fraction of plot size) is removed as
    # a gridline, axis or reference line.
    line_min_length_frac: float = Field(default=0.1, gt=0, le=1)
    # Breaks up to this long (fraction of plot size) are bridged when finding lines, so dashed
    # and moire-broken lines still count as long.
    line_bridge_frac: float = Field(default=0.015, ge=0)
    # Removed line pixels with curve ink this close on both sides (fraction of plot size) are
    # put back: that is where the curve crosses the line.
    restore_reach_frac: float = Field(default=0.006, ge=0)
    # Components whose bounding box is smaller than this in both directions (fraction of plot
    # size) are removed as specks, text and short dashes.
    min_component_frac: float = Field(default=0.02, ge=0)
    # Ink this far outside the axes box (fraction of plot size) is kept; the rest is ignored.
    box_margin_frac: float = Field(default=0.01, ge=0)


class TraceConfig(_Section):
    # Largest vertical gap between ink in neighbouring columns, as a fraction of plot height.
    max_jump_frac: float = Field(default=0.03, gt=0)
    # Widest run of empty columns the trace may bridge, as a fraction of plot width.
    max_gap_frac: float = Field(default=0.08, ge=0)
    # Cost of each bridged column; each traced column earns a reward of 1.
    skip_cost: float = Field(default=0.5, ge=0)
    # Cost of each column traced through ink removed as a straight line (a plateau of the curve
    # looks like one). Below skip_cost, so a removed plateau is followed rather than bridged.
    line_cost: float = Field(default=0.2, ge=0)
    # Cost per 0.1% of plot height of vertical gap between consecutive ink runs.
    jump_weight: float = Field(default=0.1, ge=0)
    # Cost per 0.1% of plot height of change in run centre; breaks ties toward smooth paths.
    smooth_weight: float = Field(default=0.01, ge=0)
    # Rows this far outside the axes box (fraction of height) are still searched.
    row_margin_frac: float = Field(default=0.01, ge=0)
    # Fail if the trace spans less than this fraction of the axes width.
    min_coverage_frac: float = Field(default=0.05, gt=0, le=1)
    # Warn if the trace spans less than this fraction of the axes width.
    warn_coverage_frac: float = Field(default=0.9, gt=0, le=1)
    # Warn if more than this fraction of the traced columns were gap-filled.
    warn_filled_frac: float = Field(default=0.1, ge=0, le=1)

    @model_validator(mode="after")
    def _check_costs(self) -> TraceConfig:
        if self.line_cost >= self.skip_cost:
            raise ValueError(f"trace.line_cost ({self.line_cost}) must be below trace.skip_cost ({self.skip_cost})")
        return self


class Config(_Section):
    axes: AxesConfig
    corners: CornersConfig = CornersConfig()
    plot_area: PlotAreaConfig = PlotAreaConfig()
    interactive: InteractiveConfig = InteractiveConfig()
    sampling: SamplingConfig = SamplingConfig()
    output: OutputConfig = OutputConfig()
    rectify: RectifyConfig = RectifyConfig()
    illumination: IlluminationConfig = IlluminationConfig()
    curve: CurveConfig = CurveConfig()
    clutter: ClutterConfig = ClutterConfig()
    trace: TraceConfig = TraceConfig()

    @model_validator(mode="after")
    def _check_log_sampling(self) -> Config:
        values = self.sampling.x_values
        if self.axes.x_log and values is not None and min(values) <= 0:
            raise ValueError(f"sampling.x_values must be positive on a log x axis, got {min(values):g}")
        return self


def merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge ``override`` into ``base``. Sections in ``_REPLACED_WHOLE`` are replaced."""
    merged = dict(base)
    for key, value in override.items():
        if key in _REPLACED_WHOLE or not isinstance(value, dict) or not isinstance(merged.get(key), dict):
            merged[key] = value
        else:
            merged[key] = merge(merged[key], value)
    return merged


def read_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"Could not read config file {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Config file {path} is not valid YAML: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config file {path} must contain a mapping at the top level")
    return data


def build_config(data: dict[str, Any]) -> Config:
    try:
        return Config.model_validate(data)
    except ValidationError as exc:
        lines = []
        for err in exc.errors():
            where = ".".join(str(part) for part in err["loc"]) or "config"
            lines.append(f"  {where}: {err['msg']}")
        raise ConfigError(
            "Invalid configuration:\n" + "\n".join(lines),
            hint="Axis ranges are required: pass --x-range and --y-range, or set them under 'axes:' in --config.",
        ) from exc


def load_config(path: Path | None, cli_overrides: dict[str, Any]) -> Config:
    base = read_yaml(path) if path is not None else {}
    return build_config(merge(base, cli_overrides))
