"""``digitize-synth OUT_DIR``: write synthetic charts, their ground truth, and the matching command."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from chart_digitizer.imgio import write_image
from chart_digitizer.synth.degrade import DegradeSpec, degrade
from chart_digitizer.synth.generate import CurveKind, SyntheticChart, SynthSpec, render

app = typer.Typer(add_completion=False, help="Generate synthetic line charts with exact ground truth.")


def digitize_command(image: Path, spec: SynthSpec, corners: np.ndarray) -> str:
    flat = ",".join(f"{v:.2f}" for v in corners.ravel())
    x_lo, x_hi = spec.x_range
    y_lo, y_hi = spec.y_range
    color = "" if spec.color == "#000000" else f' --curve-color "{spec.color}"'
    log = (" --x-log" if spec.x_log else "") + (" --y-log" if spec.y_log else "")
    return f'digitize "{image}" --x-range {x_lo:g},{x_hi:g} --y-range {y_lo:g},{y_hi:g}{log} --corners {flat}{color}'


def parse_range(text: str, option: str, log: bool) -> tuple[float, float]:
    try:
        lo, hi = (float(v) for v in text.split(","))
    except ValueError:
        raise typer.BadParameter(f"expected two numbers 'lo,hi', got {text!r}", param_hint=option) from None
    if lo >= hi:
        raise typer.BadParameter(f"lo must be below hi, got {text!r}", param_hint=option)
    if log and lo <= 0:
        raise typer.BadParameter(f"must be positive on a log axis, got {text!r}", param_hint=option)
    return lo, hi


def truth_document(chart: SyntheticChart, corners: np.ndarray, degraded: bool) -> dict[str, object]:
    spec = chart.spec
    return {
        "seed": spec.seed,
        "kind": spec.kind,
        "x_range": list(spec.x_range),
        "y_range": list(spec.y_range),
        "x_log": spec.x_log,
        "y_log": spec.y_log,
        "color": spec.color,
        "clutter": spec.clutter,
        "degraded": degraded,
        "corners": corners.tolist(),
        "truth": {"x": chart.truth_x[::10].tolist(), "y": chart.truth_y[::10].tolist()},
    }


@app.command()
def main(
    out_dir: Annotated[Path, typer.Argument(help="Directory for <name>.png and <name>.json files.")],
    count: Annotated[int, typer.Option(min=1, help="Number of charts.")] = 4,
    seed: Annotated[int, typer.Option(help="Seed of the first chart; later charts use seed+1, seed+2, ...")] = 0,
    kind: Annotated[str, typer.Option(help="'monotonic' or 'wavy'.")] = "monotonic",
    color: Annotated[str, typer.Option(help="Curve colour '#rrggbb'.")] = "#000000",
    degraded: Annotated[
        bool, typer.Option("--degraded", help="Add clutter (gridlines, reference lines, text, specks) and photo-like degradations.")
    ] = False,
    clutter: Annotated[bool, typer.Option("--clutter", help="Add clutter only, without photo degradations.")] = False,
    x_log: Annotated[bool, typer.Option("--x-log", help="Logarithmic x axis.")] = False,
    y_log: Annotated[bool, typer.Option("--y-log", help="Logarithmic y axis.")] = False,
    x_range: Annotated[str | None, typer.Option(help="'lo,hi'. Default 0,100, or 1,1000 with --x-log.")] = None,
    y_range: Annotated[str | None, typer.Option(help="'lo,hi'. Default 0,100, or 0.1,1000 with --y-log.")] = None,
) -> None:
    if kind not in ("monotonic", "wavy"):
        raise typer.BadParameter("must be 'monotonic' or 'wavy'", param_hint="--kind")
    curve_kind: CurveKind = "monotonic" if kind == "monotonic" else "wavy"
    xr = parse_range(x_range or ("1,1000" if x_log else "0,100"), "--x-range", x_log)
    yr = parse_range(y_range or ("0.1,1000" if y_log else "0,100"), "--y-range", y_log)
    for i in range(count):
        spec = SynthSpec(
            seed=seed + i, kind=curve_kind, color=color, clutter=clutter or degraded, x_range=xr, y_range=yr, x_log=x_log, y_log=y_log
        )
        chart = render(spec)
        image, corners = chart.image, chart.corners
        if degraded:
            image, corners = degrade(image, corners, DegradeSpec(), np.random.default_rng(spec.seed))
        suffix = ("_logx" if x_log else "") + ("_logy" if y_log else "")
        suffix += "_degraded" if degraded else "_clutter" if clutter else ""
        name = f"synth_{kind}_{spec.seed:03d}{suffix}"
        write_image(out_dir / f"{name}.png", image)
        (out_dir / f"{name}.json").write_text(
            json.dumps(truth_document(chart, corners, degraded), indent=2), encoding="utf-8"
        )
        typer.echo(digitize_command(out_dir / f"{name}.png", spec, corners))
