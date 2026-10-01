"""Render synthetic line charts with exact ground truth and exact plot corners.

Clean charts have no spines and light-gray gridlines, so the only dark ink inside the plot area
is the curve. Cluttered charts add what a real chart has: dark gridlines and spines, black
reference lines crossing the curve, a dashed colored series, a legend, text and specks.

Either axis can be logarithmic. The curve is then generated evenly in log10 space, which is where
it is smooth on the page, and log charts carry minor gridlines as real ones do.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

CurveKind = Literal["monotonic", "wavy"]

_TRUTH_SAMPLES = 4001
# Fraction of the y-range kept free at the top and bottom of the plot.
_Y_MARGIN = 0.08


@dataclass(frozen=True)
class SynthSpec:
    seed: int
    kind: CurveKind = "monotonic"
    x_range: tuple[float, float] = (0.0, 100.0)
    y_range: tuple[float, float] = (0.0, 100.0)
    grid_step: tuple[float, float] | None = None  # None: 10 divisions per axis
    size_in: tuple[float, float] = (8.0, 6.0)
    dpi: int = 150
    line_width_pt: float = 1.8
    color: str = "#000000"
    grid_color: str = "#d9d9d9"
    clutter: bool = False
    x_log: bool = False
    y_log: bool = False


@dataclass(frozen=True, eq=False)
class SyntheticChart:
    spec: SynthSpec
    image: np.ndarray  # BGR uint8
    corners: np.ndarray  # (4, 2): (xmin,ymin), (xmax,ymin), (xmax,ymax), (xmin,ymax) in pixels
    truth_x: np.ndarray
    truth_y: np.ndarray

    def truth_at(self, x: np.ndarray) -> np.ndarray:
        return np.interp(x, self.truth_x, self.truth_y)


def _unit_monotonic(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A random increasing function on [0, 1] with g(0)=0, g(1)=1."""
    family = rng.integers(3)
    if family == 0:
        return t ** rng.uniform(0.4, 3.0)
    if family == 1:
        k = rng.uniform(1.5, 5.0) * rng.choice([-1.0, 1.0])
        return np.expm1(k * t) / np.expm1(k)
    centre, width = rng.uniform(0.3, 0.7), rng.uniform(0.08, 0.25)
    raw = 1.0 / (1.0 + np.exp(-(t - centre) / width))
    return (raw - raw[0]) / (raw[-1] - raw[0])


def _unit_wavy(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A random smooth non-monotonic function on [0, 1], normalised to [0, 1]."""
    y = rng.uniform(-1.0, 1.0) * t
    for _ in range(rng.integers(2, 4)):
        cycles = rng.uniform(0.5, 2.5)
        y = y + rng.uniform(0.3, 1.0) * np.sin(2 * np.pi * cycles * t + rng.uniform(0, 2 * np.pi))
    return (y - y.min()) / (y.max() - y.min())


def to_unit(values: np.ndarray, value_range: tuple[float, float], log: bool) -> np.ndarray:
    """Data values -> position along the axis, 0 at the first range value and 1 at the second."""
    lo, hi = np.log10(value_range) if log else value_range
    return ((np.log10(values) if log else values) - lo) / (hi - lo)


def from_unit(unit: np.ndarray, value_range: tuple[float, float], log: bool) -> np.ndarray:
    """Position along the axis (0..1) -> data values; the inverse of ``to_unit``."""
    lo, hi = np.log10(value_range) if log else value_range
    values = lo + unit * (hi - lo)
    return np.power(10.0, values) if log else values


def truth_curve(spec: SynthSpec) -> tuple[np.ndarray, np.ndarray]:
    """Dense ground-truth samples of the curve described by ``spec``."""
    rng = np.random.default_rng(spec.seed)
    t = np.linspace(0.0, 1.0, _TRUTH_SAMPLES)
    unit = _unit_monotonic(t, rng) if spec.kind == "monotonic" else _unit_wavy(t, rng)
    if spec.kind == "monotonic" and rng.random() < 0.3:
        unit = 1.0 - unit  # decreasing
    x = from_unit(t, spec.x_range, spec.x_log)
    y = from_unit(_Y_MARGIN + (1 - 2 * _Y_MARGIN) * unit, spec.y_range, spec.y_log)
    return x, y


def _figure_to_bgr(canvas: FigureCanvasAgg) -> np.ndarray:
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    return np.ascontiguousarray(rgba[:, :, 2::-1])


def plot_corners(
    ax: Axes, image_height: int, x_range: tuple[float, float], y_range: tuple[float, float]
) -> np.ndarray:
    """Axes corners in OpenCV pixel coordinates (pixel centres on integers, rows downward)."""
    (x_lo, x_hi), (y_lo, y_hi) = x_range, y_range
    data = np.array([[x_lo, y_lo], [x_hi, y_lo], [x_hi, y_hi], [x_lo, y_hi]])
    display = ax.transData.transform(data)  # origin at the bottom-left corner of the image
    return np.column_stack([display[:, 0] - 0.5, image_height - display[:, 1] - 0.5])


def _clearance(x: np.ndarray, y: np.ndarray, px: float, py: float, half_width: float) -> float:
    """Smallest |py - y| over the curve within ``half_width`` of ``px`` in x."""
    near = np.abs(x - px) <= half_width
    return float(np.abs(py - y[near]).min())


def _add_clutter(ax: Axes, spec: SynthSpec, x: np.ndarray, y: np.ndarray) -> None:
    """Draw clutter that doesn't touch the curve except where lines cross it.

    Positions are chosen along the axes (0..1), so log charts get the same layout as linear ones.
    """
    rng = np.random.default_rng([spec.seed, 1])  # independent of the curve's random stream
    u, v = to_unit(x, spec.x_range, spec.x_log), to_unit(y, spec.y_range, spec.y_log)

    def data_x(values: np.ndarray | float) -> np.ndarray:
        return from_unit(np.asarray(values, dtype=np.float64), spec.x_range, spec.x_log)

    def data_y(values: np.ndarray | float) -> np.ndarray:
        return from_unit(np.asarray(values, dtype=np.float64), spec.y_range, spec.y_log)

    # A horizontal reference line through the curve, and a dashed vertical one.
    ax.axhline(float(data_y(np.interp(rng.uniform(0.3, 0.7), u, v))), color="black", lw=1.0)
    ax.axvline(float(data_x(rng.uniform(0.2, 0.8))), color="black", lw=1.0, ls="--")
    # A dashed colored series crossing the curve, drawn on top of it (straight on the page).
    ends = rng.uniform(0.1, 0.9, size=2)
    along = np.linspace(0.0, 1.0, 50)
    ax.plot(data_x(along), data_y(ends[0] + along * (ends[1] - ends[0])), color="tab:orange", lw=1.5, ls="--", label="reference", zorder=3)
    # Text where it has the most room.
    spots = [(fx, fy) for fx in (0.05, 0.35, 0.65) for fy in (0.1, 0.9)]
    tx, ty = max(spots, key=lambda p: _clearance(u, v, p[0] + 0.15, p[1], 0.17))
    ax.text(float(data_x(tx)), float(data_y(ty)), "n = 128, p < 0.05", fontsize=10, va="center")
    # Specks well away from the curve.
    px, py = rng.uniform(0.02, 0.98, 60), rng.uniform(0.02, 0.98, 60)
    clear = np.array([_clearance(u, v, a, b, 0.02) for a, b in zip(px, py, strict=True)]) > 0.05
    ax.plot(data_x(px[clear][:25]), data_y(py[clear][:25]), "o", color="black", ms=rng.uniform(1.5, 3.0), ls="none")
    ax.legend(loc="best", fontsize=8)


def _set_ticks(ax: Axes, spec: SynthSpec) -> None:
    """Ten divisions (or ``grid_step``) on linear axes; matplotlib's decades on log axes."""
    ax.set_xscale("log" if spec.x_log else "linear")
    ax.set_yscale("log" if spec.y_log else "linear")
    ax.set_xlim(*spec.x_range)
    ax.set_ylim(*spec.y_range)
    default_step = ((spec.x_range[1] - spec.x_range[0]) / 10, (spec.y_range[1] - spec.y_range[0]) / 10)
    step_x, step_y = spec.grid_step or default_step
    if not spec.x_log:
        ax.set_xticks(np.arange(spec.x_range[0], spec.x_range[1] + step_x / 2, step_x))
    if not spec.y_log:
        ax.set_yticks(np.arange(spec.y_range[0], spec.y_range[1] + step_y / 2, step_y))


def render(spec: SynthSpec) -> SyntheticChart:
    x, y = truth_curve(spec)
    fig = Figure(figsize=spec.size_in, dpi=spec.dpi)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    ax.plot(x, y, color=spec.color, lw=spec.line_width_pt, solid_capstyle="butt", label="data")
    _set_ticks(ax, spec)
    log = spec.x_log or spec.y_log
    if spec.clutter:
        ax.grid(True, color="#303030", lw=1.5)
        if log:
            ax.grid(True, which="minor", color="#606060", lw=0.8)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
        ax.tick_params(direction="out", length=4)
        _add_clutter(ax, spec, x, y)
    else:
        ax.grid(True, which="both" if log else "major", color=spec.grid_color, lw=0.8)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
        # No tick marks, and labels kept well clear of the plot area.
        ax.tick_params(length=0, pad=10)
    fig.tight_layout()

    image = _figure_to_bgr(canvas)
    corners = plot_corners(ax, image.shape[0], spec.x_range, spec.y_range)
    return SyntheticChart(spec=spec, image=image, corners=corners, truth_x=x, truth_y=y)
