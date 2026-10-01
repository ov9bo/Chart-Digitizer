"""Stage 6: group the cleaned ink into curve candidates and rank them.

A line chart's curve is a function of x, so it covers many columns, while leftover clutter covers
few. Connected components are ranked by the number of columns they cover. The widest one seeds a
candidate; each next component joins it if it mostly covers new columns (pieces of the same curve
split where it crossed a removed line), and otherwise is left for the next candidate. Candidates
come out best first; the tracer uses the first.
"""

from __future__ import annotations

import cv2
import numpy as np

from chart_digitizer.config import CurveConfig
from chart_digitizer.debug import DebugSink
from chart_digitizer.errors import SelectionError
from chart_digitizer.models import CurveMask

# Colors for the debug view of up to this many candidates, BGR.
_PALETTE = ((0, 0, 0), (40, 40, 220), (220, 120, 30), (40, 160, 40), (160, 40, 160))


def component_columns(mask: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Label 8-connected components; return the labels and each component's covered columns."""
    count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    rows, cols = np.nonzero(labels)
    ids = labels[rows, cols]
    pairs = np.unique(ids.astype(np.int64) * mask.shape[1] + cols)
    comp, col = np.divmod(pairs, mask.shape[1])
    bounds = np.searchsorted(comp, np.arange(1, count + 1))
    return labels, [col[bounds[i] : bounds[i + 1]] for i in range(count - 1)]


def group_components(columns: list[np.ndarray], max_overlap_frac: float) -> list[list[int]]:
    """Greedily group component indices into candidates, widest component first."""
    order = sorted(range(len(columns)), key=lambda i: len(columns[i]), reverse=True)
    groups: list[list[int]] = []
    while order:
        seed, rest = order[0], order[1:]
        covered = set(columns[seed].tolist())
        group, leftover = [seed], []
        for i in rest:
            overlap = sum(1 for c in columns[i].tolist() if c in covered)
            if overlap <= max_overlap_frac * len(columns[i]):
                group.append(i)
                covered.update(columns[i].tolist())
            else:
                leftover.append(i)
        groups.append(group)
        order = leftover
    return groups


def select_curves(ink: CurveMask, cfg: CurveConfig, debug: DebugSink) -> list[CurveMask]:
    """Return curve candidates, best first."""
    if not ink.mask.any():
        raise SelectionError(
            "Nothing is left of the curve after removing gridlines, axes, text and specks",
            hint="Try --no-clutter to keep all ink, or loosen the clutter.* settings in --config; "
            "--debug shows what was removed in debug/06_clutter.png.",
        )
    labels, columns = component_columns(ink.mask)
    groups = group_components(columns, cfg.max_overlap_frac)
    candidates = []
    for rank, group in enumerate(groups):
        mask = np.isin(labels, np.asarray(group) + 1)
        label = ink.label if rank == 0 else f"{ink.label}#{rank + 1}"
        weight = np.where(mask, ink.weight, 0.0).astype(np.float32)
        candidates.append(CurveMask(mask=mask, weight=weight, label=label, removed=ink.removed))
    if debug.enabled:
        view = np.full((*ink.mask.shape, 3), 255, np.uint8)
        for rank in reversed(range(min(len(candidates), len(_PALETTE)))):
            view[candidates[rank].mask] = _PALETTE[rank]
        debug.save("07_curve_mask", view)
    return candidates
