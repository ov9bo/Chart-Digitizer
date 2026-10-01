"""Command-line entry point: ``digitize IMAGE --x-range 0,100 --y-range 0,100 ...``.

IMAGE may be a folder, in which case every image directly inside it is digitized (batch mode).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

from chart_digitizer.batch import (
    IMAGE_SUFFIXES,
    SUMMARY_NAME,
    Job,
    Outcome,
    find_images,
    image_config_data,
    output_names,
    run_batch,
    sidecar_path,
    write_summary,
)
from chart_digitizer.config import OutputConfig, build_config, merge, read_yaml
from chart_digitizer.errors import ConfigError, DigitizeError
from chart_digitizer.pipeline import digitize, write_outputs

EXIT_STAGE_FAILED = 1
EXIT_USAGE = 2

app = typer.Typer(add_completion=False, help="Digitize a line chart from a photo or screenshot.")


def parse_floats(text: str, option: str, count: int | None = None) -> tuple[float, ...]:
    """Parse 'a,b,c' into floats, optionally requiring exactly ``count`` values."""
    try:
        values = tuple(float(part) for part in text.split(","))
    except ValueError:
        raise typer.BadParameter(f"expected comma-separated numbers, got '{text}'", param_hint=option) from None
    if count is not None and len(values) != count:
        raise typer.BadParameter(f"expected {count} comma-separated numbers, got {len(values)}", param_hint=option)
    return values


def cli_overrides(
    x_range: str | None,
    y_range: str | None,
    x_log: bool | None,
    y_log: bool | None,
    grid_step: str | None,
    points: int | None,
    x_values: str | None,
    step: float | None,
    corners: str | None,
    interactive: bool,
    curve_color: str | None,
    out: Path | None,
    debug: bool | None,
    clutter: bool | None = None,
) -> dict[str, Any]:
    """Nested config overrides for the flags the user actually passed."""
    if corners is not None and interactive:
        raise typer.BadParameter("use either --corners or --interactive, not both", param_hint="--corners")
    axes: dict[str, Any] = {}
    if x_range is not None:
        axes["x_range"] = parse_floats(x_range, "--x-range", 2)
    if y_range is not None:
        axes["y_range"] = parse_floats(y_range, "--y-range", 2)
    if x_log is not None:
        axes["x_log"] = x_log
    if y_log is not None:
        axes["y_log"] = y_log
    if grid_step is not None:
        axes["grid_step"] = parse_floats(grid_step, "--grid-step", 2)

    sampling: dict[str, Any] = {}
    if points is not None:
        sampling["points"] = points
    if x_values is not None:
        sampling["x_values"] = parse_floats(x_values, "--x-values")
    if step is not None:
        sampling["step"] = step

    overrides: dict[str, Any] = {}
    if axes:
        overrides["axes"] = axes
    if sampling:
        overrides["sampling"] = sampling
    if corners is not None:
        flat = parse_floats(corners, "--corners", 8)
        overrides["corners"] = {"mode": "manual", "points": [flat[i : i + 2] for i in range(0, 8, 2)]}
    elif interactive:
        overrides["corners"] = {"mode": "interactive"}
    if curve_color is not None:
        overrides["curve"] = {"color": curve_color}
    if clutter is not None:
        overrides["clutter"] = {"enabled": clutter}
    output: dict[str, Any] = {}
    if out is not None:
        output["dir"] = out
    if debug is not None:
        output["debug"] = debug
    if output:
        overrides["output"] = output
    return overrides


def build_job(image: Path, name: str, base: dict[str, Any], overrides: dict[str, Any]) -> Job:
    """The job for one image, with its settings validated; names the image and its YAML on failure."""
    sidecar = sidecar_path(image)
    try:
        cfg = build_config(image_config_data(base, sidecar, overrides))
    except ConfigError as exc:
        where = f"{image.name} (with {sidecar.name})" if sidecar is not None else image.name
        raise ConfigError(f"{where}: {exc.message}", hint=exc.hint) from exc
    return Job(image=image, name=name, config=cfg, sidecar=sidecar)


def warn(warnings: tuple[str, ...], indent: str = "") -> None:
    for warning in warnings:
        typer.secho(f"{indent}Warning: {warning}", fg=typer.colors.YELLOW, err=True)


def run_single(image: Path, base: dict[str, Any], overrides: dict[str, Any]) -> Path:
    job = build_job(image, image.stem, base, overrides)
    if job.sidecar is not None:
        typer.echo(f"Using settings from {job.sidecar}")
    result = digitize(job.image, job.config, job.name)
    out = write_outputs(result, job.config)
    warn(result.warnings)
    kept = int(result.points.in_range.sum())
    typer.echo(f"{image.name}: {kept}/{len(result.points.x)} points -> {out}")
    return out


def report_outcome(number: int, total: int, outcome: Outcome) -> None:
    prefix = f"[{number}/{total}] {outcome.image.name}:"
    if outcome.status == "ok":
        typer.echo(f"{prefix} {outcome.points_kept}/{outcome.points_total} points -> {outcome.out_dir}")
        warn(outcome.warnings, indent="    ")
    else:
        error = (outcome.error or "").replace("\n", "\n    ")
        typer.secho(f"{prefix} FAILED {error}", fg=typer.colors.RED, err=True)


def run_folder(folder: Path, base: dict[str, Any], overrides: dict[str, Any]) -> int:
    """Digitize every image in ``folder``; returns the exit code."""
    images = find_images(folder)
    if not images:
        raise ConfigError(
            f"No images found in {folder}",
            hint=f"Batch mode reads files ending in {', '.join(IMAGE_SUFFIXES)} directly inside the folder, not in subfolders.",
        )
    names = output_names(images)
    jobs, problems, hint = [], [], None
    for image in images:
        try:
            jobs.append(build_job(image, names[image], base, overrides))
        except ConfigError as exc:
            problems.append(exc.message)
            hint = hint or exc.hint
    if problems:
        raise ConfigError(
            f"Settings are invalid for {len(problems)} of {len(images)} images, so none were processed:\n  "
            + "\n  ".join(problems),
            hint=(hint or "") + " Per-image settings go in a YAML file named like the image (chart1.png -> chart1.yaml).",
        )
    for job in jobs:
        if job.sidecar is not None:
            typer.echo(f"{job.image.name}: using settings from {job.sidecar.name}")
    outcomes = run_batch(jobs, lambda number, outcome: report_outcome(number, len(jobs), outcome))
    root = OutputConfig.model_validate(merge(base, overrides).get("output", {})).dir
    write_summary(root / SUMMARY_NAME, outcomes)
    ok = sum(o.status == "ok" for o in outcomes)
    failed = len(outcomes) - ok
    typer.secho(
        f"Done: {ok} of {len(outcomes)} images digitized" + (f", {failed} failed" if failed else "") + f". Summary: {root / SUMMARY_NAME}",
        fg=typer.colors.RED if failed else None,
    )
    return EXIT_STAGE_FAILED if failed else 0


@app.command()
def main(
    image: Annotated[
        Path, typer.Argument(help="Chart image, or a folder: every image in it is digitized and summarised in OUT/summary.csv.")
    ],
    x_range: Annotated[str | None, typer.Option(help="Data values at the left and right plot edges: 'lo,hi'.")] = None,
    y_range: Annotated[str | None, typer.Option(help="Data values at the bottom and top plot edges: 'lo,hi'.")] = None,
    points: Annotated[int | None, typer.Option(help="Number of evenly spaced output points (default 50).")] = None,
    x_values: Annotated[str | None, typer.Option(help="Explicit x values to sample: 'a,b,c,...'.")] = None,
    step: Annotated[float | None, typer.Option(help="Sample every STEP along x.")] = None,
    corners: Annotated[
        str | None,
        typer.Option(help="Plot corners (default: found automatically) in image pixels 'x1,y1,...,x4,y4': (xmin,ymin),(xmax,ymin),(xmax,ymax),(xmin,ymax)."),
    ] = None,
    interactive: Annotated[bool, typer.Option("--interactive", help="Click or adjust the plot corners in a window.")] = False,
    curve_color: Annotated[str | None, typer.Option(help="Curve colour '#rrggbb' or 'hsv:h,s,v' (default: dark curve).")] = None,
    grid_step: Annotated[str | None, typer.Option(help="Gridline spacing 'dx,dy' (used for the re-plot).")] = None,
    clutter: Annotated[
        bool | None,
        typer.Option("--clutter/--no-clutter", help="Remove gridlines, axes, text and specks before tracing (default on)."),
    ] = None,
    x_log: Annotated[bool | None, typer.Option("--x-log/--no-x-log", help="Logarithmic x axis.")] = None,
    y_log: Annotated[bool | None, typer.Option("--y-log/--no-y-log", help="Logarithmic y axis.")] = None,
    out: Annotated[Path | None, typer.Option(help="Output root; results go to OUT/<image-stem>/, and a folder's summary to OUT/summary.csv (default 'out').")] = None,
    debug: Annotated[bool | None, typer.Option("--debug/--no-debug", help="Write an image from every stage to debug/.")] = None,
    config: Annotated[Path | None, typer.Option(help="YAML config file; a YAML named like the image, then CLI flags, override it.")] = None,
) -> None:
    """Digitize the curve in IMAGE into points.csv, points.json, overlay.png and replot.png.

    Settings come from, in increasing priority: built-in defaults, --config, a YAML file named like
    the image (chart1.png -> chart1.yaml) if present, and the flags given here.
    """
    overrides = cli_overrides(
        x_range, y_range, x_log, y_log, grid_step, points, x_values, step,
        corners, interactive, curve_color, out, debug, clutter,
    )  # fmt: skip
    try:
        base = read_yaml(config) if config is not None else {}
        if image.is_dir():
            code = run_folder(image, base, overrides)
            if code:
                raise typer.Exit(code)
        else:
            run_single(image, base, overrides)
    except ConfigError as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(EXIT_USAGE) from None
    except DigitizeError as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(EXIT_STAGE_FAILED) from None
