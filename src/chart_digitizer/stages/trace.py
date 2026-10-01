"""Stage 7: trace the curve column by column with a continuity constraint.

Each column of the curve mask is split into vertical runs of ink. Dynamic programming then picks
at most one run per column so that the path covers as many columns as possible while paying for
vertical jumps between consecutive runs and for columns it skips. Runs that overlap or touch
cost nothing to connect, so steep sections (long runs stacked end to end) trace cleanly. Skipped
columns are filled with PCHIP interpolation and flagged.

Ink that clutter removal took out as straight lines is offered too, as runs that cost a little
instead of earning a reward. The path may only step through them from ink they touch, column by
column, so they carry the trace along a plateau that looked like a reference line but can never
extend the curve or pull it across to a nearby gridline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator

from chart_digitizer.config import TraceConfig
from chart_digitizer.errors import TraceError
from chart_digitizer.models import CurveMask, PlotBox, Trace

_HINT = (
    "Check the corners and --curve-color, or try --no-clutter if the curve was removed with the gridlines; "
    "--debug writes the curve mask to debug/07_curve_mask.png."
)


@dataclass(frozen=True, eq=False)
class Runs:
    """Vertical ink runs, sorted by column then row. ``end`` is exclusive."""

    col: np.ndarray  # column index relative to the first searched column
    start: np.ndarray
    end: np.ndarray
    center: np.ndarray  # weighted centroid row
    offsets: np.ndarray  # runs of column c are [offsets[c], offsets[c + 1])
    on_line: np.ndarray  # bool, the run is ink removed as a straight line


def find_runs(mask: np.ndarray, weight: np.ndarray, row_offset: int = 0, on_line: bool = False) -> Runs:
    """Find vertical runs of True in every column of ``mask``. Rows are shifted by ``row_offset``."""
    height, n_cols = mask.shape
    padded = np.zeros((height + 2, n_cols), dtype=np.int8)
    padded[1:-1] = mask
    step = np.diff(padded, axis=0).T  # (n_cols, height + 1), column-major order
    run_col, start = np.nonzero(step == 1)
    _, end = np.nonzero(step == -1)

    w = np.where(mask, weight, 0.0).astype(np.float64)
    rows = np.arange(height, dtype=np.float64)[:, None]
    cum_w = np.vstack([np.zeros((1, n_cols)), np.cumsum(w, axis=0)])
    cum_wr = np.vstack([np.zeros((1, n_cols)), np.cumsum(w * rows, axis=0)])
    total = cum_w[end, run_col] - cum_w[start, run_col]
    moment = cum_wr[end, run_col] - cum_wr[start, run_col]
    midpoint = (start + end - 1) / 2.0
    center = np.where(total > 0, moment / np.where(total > 0, total, 1.0), midpoint)

    offsets = np.searchsorted(run_col, np.arange(n_cols + 1))
    return Runs(
        col=run_col,
        start=start + row_offset,
        end=end + row_offset,
        center=center + row_offset,
        offsets=offsets,
        on_line=np.full(len(run_col), on_line),
    )


def merge_runs(a: Runs, b: Runs) -> Runs:
    """Combine two sets of runs over the same columns, keeping them sorted by column then row."""
    col = np.concatenate([a.col, b.col])
    start = np.concatenate([a.start, b.start])
    order = np.lexsort((start, col))
    return Runs(
        col=col[order],
        start=start[order],
        end=np.concatenate([a.end, b.end])[order],
        center=np.concatenate([a.center, b.center])[order],
        offsets=np.searchsorted(col[order], np.arange(len(a.offsets))),
        on_line=np.concatenate([a.on_line, b.on_line])[order],
    )


def best_path(runs: Runs, cfg: TraceConfig, box: PlotBox) -> np.ndarray:
    """Indices into ``runs`` of the highest-scoring path, ordered by column."""
    n_cols = len(runs.offsets) - 1
    unit = box.height / 1000.0  # 0.1% of plot height, in pixels
    max_jump = cfg.max_jump_frac * box.height
    max_skip = int(round(cfg.max_gap_frac * box.width))

    reward = np.where(runs.on_line, -cfg.line_cost, 1.0)
    score = np.full(len(runs.col), -np.inf)
    prev = np.full(len(runs.col), -1, dtype=np.int64)
    for c in range(n_cols):
        a, b = runs.offsets[c], runs.offsets[c + 1]
        if a == b:
            continue
        score[a:b] = np.where(runs.on_line[a:b], -np.inf, 1.0)  # a path may start on curve ink
        wa = runs.offsets[max(c - max_skip - 1, 0)]
        if wa == a:
            continue
        dc = c - runs.col[wa:a]  # >= 1
        gap = np.maximum(
            0,
            np.maximum(
                runs.start[a:b, None] - runs.end[None, wa:a],
                runs.start[None, wa:a] - runs.end[a:b, None],
            ),
        )
        shift = np.abs(runs.center[a:b, None] - runs.center[None, wa:a])
        candidate = (
            score[None, wa:a]
            + reward[a:b, None]
            - cfg.skip_cost * (dc - 1)
            - cfg.jump_weight * gap / unit
            - cfg.smooth_weight * shift / unit
        )
        candidate[gap > max_jump * dc] = -np.inf
        via_line = runs.on_line[a:b, None] | runs.on_line[None, wa:a]
        candidate[via_line & ((gap > 0) | (dc > 1))] = -np.inf  # line ink only continues what it touches
        best = np.argmax(candidate, axis=1)
        value = candidate[np.arange(b - a), best]
        better = value > score[a:b]
        score[a:b] = np.where(better, value, score[a:b])
        prev[a:b] = np.where(better, wa + best, -1)

    path = [int(np.argmax(np.where(runs.on_line, -np.inf, score)))]  # ends on curve ink
    while prev[path[-1]] >= 0:
        path.append(int(prev[path[-1]]))
    return np.array(path[::-1], dtype=np.int64)


def fill_gaps(cols: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand to every column between the first and last, PCHIP-filling and flagging the gaps."""
    all_cols = np.arange(cols[0], cols[-1] + 1)
    filled = ~np.isin(all_cols, cols)
    all_rows = PchipInterpolator(cols, rows)(all_cols) if filled.any() else rows.astype(np.float64)
    return all_cols, all_rows, filled


def coverage_warnings(
    trace_cols: np.ndarray, filled: np.ndarray, line_frac: float, box: PlotBox, cfg: TraceConfig
) -> list[str]:
    warnings = []
    coverage = (trace_cols[-1] - trace_cols[0]) / box.width
    if coverage < cfg.warn_coverage_frac:
        warnings.append(
            f"The traced curve spans only {coverage:.0%} of the x-axis; part of it may be missing."
        )
    if filled.mean() > cfg.warn_filled_frac:
        warnings.append(f"{filled.mean():.0%} of the traced curve was interpolated across gaps.")
    if line_frac > cfg.warn_filled_frac:
        warnings.append(
            f"{line_frac:.0%} of the traced curve follows ink removed as straight lines; "
            "check the overlay where the curve is flat or runs along a gridline."
        )
    return warnings


def trace(curve: CurveMask, box: PlotBox, cfg: TraceConfig) -> Trace:
    margin = int(round(cfg.row_margin_frac * box.height))
    top = max(box.top - margin, 0)
    bottom = min(box.bottom + margin, curve.mask.shape[0] - 1)
    window = (slice(top, bottom + 1), slice(box.left, box.right + 1))

    runs = find_runs(curve.mask[window], curve.weight[window], row_offset=top)
    if len(runs.col) == 0:
        raise TraceError("No curve pixels inside the plot area", hint=_HINT)
    if curve.removed is not None:
        lines = curve.removed[window]
        runs = merge_runs(runs, find_runs(lines, lines.astype(np.float32), row_offset=top, on_line=True))

    path = best_path(runs, cfg, box)
    if len(path) < 2:
        raise TraceError("Could not follow the curve for more than one column", hint=_HINT)

    cols, rows, filled = fill_gaps(runs.col[path] + box.left, runs.center[path])
    coverage = (cols[-1] - cols[0]) / box.width
    if coverage < cfg.min_coverage_frac:
        raise TraceError(
            f"The traced curve spans only {coverage:.1%} of the x-axis (need {cfg.min_coverage_frac:.0%})",
            hint=_HINT,
        )
    return Trace(
        cols=cols,
        rows=rows,
        filled=filled,
        warnings=tuple(coverage_warnings(cols, filled, float(runs.on_line[path].sum()) / len(cols), box, cfg)),
    )
