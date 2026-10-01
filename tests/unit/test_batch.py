import csv
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from chart_digitizer import batch
from chart_digitizer.batch import (
    Job,
    Outcome,
    find_images,
    image_config_data,
    output_names,
    process,
    sidecar_path,
    write_summary,
)
from chart_digitizer.cli import app
from chart_digitizer.config import build_config
from chart_digitizer.imgio import write_image
from chart_digitizer.synth.generate import SynthSpec, render

runner = CliRunner()


def read_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def add_chart(folder: Path, name: str, seed: int, x_range: tuple[float, float]) -> None:
    """A synthetic chart plus a sidecar YAML holding its x-range and exact corners."""
    chart = render(SynthSpec(seed=seed, x_range=x_range))
    write_image(folder / f"{name}.png", chart.image)
    sidecar = {"axes": {"x_range": list(x_range)}, "corners": {"points": chart.corners.tolist()}}
    (folder / f"{name}.yaml").write_text(yaml.safe_dump(sidecar))


def test_find_images_is_flat_and_filters_by_suffix(tmp_path: Path) -> None:
    for name in ("b.PNG", "a.jpg", "notes.txt", "a.yaml"):
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.png").write_bytes(b"")
    assert [p.name for p in find_images(tmp_path)] == ["a.jpg", "b.PNG"]


def test_output_names_disambiguate_shared_stems() -> None:
    images = [Path("chart.png"), Path("Chart.JPG"), Path("other.png")]
    assert output_names(images) == {images[0]: "chart_png", images[1]: "Chart_jpg", images[2]: "other"}


def test_sidecar_prefers_yaml(tmp_path: Path) -> None:
    image = tmp_path / "c.png"
    assert sidecar_path(image) is None
    (tmp_path / "c.yml").write_text("{}")
    assert sidecar_path(image) == tmp_path / "c.yml"
    (tmp_path / "c.yaml").write_text("{}")
    assert sidecar_path(image) == tmp_path / "c.yaml"


def test_sidecar_sits_between_config_file_and_flags(tmp_path: Path) -> None:
    sidecar = tmp_path / "c.yaml"
    sidecar.write_text("axes:\n  x_range: [0, 10]\n  y_range: [0, 10]\nsampling:\n  points: 7\n")
    base = {"axes": {"x_range": (0, 1), "y_range": (0, 1)}, "trace": {"skip_cost": 0.9}}
    cfg = build_config(image_config_data(base, sidecar, {"axes": {"y_range": (5, 50)}}))
    assert cfg.axes.x_range == (0, 10)  # the sidecar beats --config
    assert cfg.axes.y_range == (5, 50)  # flags beat the sidecar
    assert cfg.sampling.points == 7 and cfg.trace.skip_cost == 0.9


def job_for(image: Path) -> Job:
    cfg = build_config({"axes": {"x_range": (0, 1), "y_range": (0, 1)}, "output": {"dir": image.parent / "out"}})
    return Job(image=image, name=image.stem, config=cfg, sidecar=None)


def test_stage_failure_becomes_outcome(tmp_path: Path) -> None:
    bogus = tmp_path / "bogus.png"
    bogus.write_bytes(b"not an image")
    outcome = process(job_for(bogus))
    assert outcome.status == "failed"
    assert outcome.error is not None and outcome.error.startswith("[load]")


def test_unexpected_exception_becomes_outcome(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args: object) -> None:
        raise RuntimeError("kaput")

    monkeypatch.setattr(batch, "digitize", boom)
    outcome = process(job_for(tmp_path / "x.png"))
    assert outcome.status == "error"
    assert outcome.error == "internal error (RuntimeError): kaput"


def test_summary_csv(tmp_path: Path) -> None:
    outcomes = [
        Outcome(Path("a.png"), "ok", out_dir=Path("out/a"), points_kept=48, points_total=50, corners="auto",
                filled_fraction=0.01234, warnings=("w1", "w2")),
        Outcome(Path("b.png"), "failed", error="[load] bad\nHint: fix it"),
    ]  # fmt: skip
    path = tmp_path / "deep" / "summary.csv"
    write_summary(path, outcomes)
    rows = read_summary(path)
    assert rows[0]["status"] == "ok" and rows[0]["points_kept"] == "48" and rows[0]["filled_fraction"] == "0.0123"
    assert rows[0]["warnings"] == "w1 | w2" and rows[0]["error"] == ""
    assert rows[1]["error"] == "[load] bad Hint: fix it" and rows[1]["output"] == ""


# --- the CLI in batch mode ---------------------------------------------------------------


def invoke(*args: str | Path) -> tuple[int, str]:
    result = runner.invoke(app, [str(a) for a in args])
    return result.exit_code, result.output


def test_batch_all_ok(tmp_path: Path) -> None:
    charts, out = tmp_path / "charts", tmp_path / "out"
    charts.mkdir()
    add_chart(charts, "first", seed=3, x_range=(0, 100))
    add_chart(charts, "second", seed=4, x_range=(0, 10))
    code, output = invoke(charts, "--y-range", "0,100", "--points", "11", "--out", out)
    assert code == 0, output
    assert "[2/2] second.png: 11/11 points" in output
    assert "first.png: using settings from first.yaml" in output
    rows = read_summary(out / "summary.csv")
    assert [(r["image"], r["status"], r["corners"]) for r in rows] == [
        ("first.png", "ok", "manual"),
        ("second.png", "ok", "manual"),
    ]
    x_last = (out / "second" / "points.csv").read_text().splitlines()[-1].split(",")[0]
    assert float(x_last) == 10  # the sidecar's x-range was used
    assert (out / "first" / "replot.png").exists()


def test_batch_keeps_going_past_a_failure(tmp_path: Path) -> None:
    charts, out = tmp_path / "charts", tmp_path / "out"
    charts.mkdir()
    add_chart(charts, "good", seed=3, x_range=(0, 100))
    (charts / "bad.png").write_bytes(b"not an image")
    code, output = invoke(charts, "--x-range", "0,100", "--y-range", "0,100", "--out", out)
    assert code == 1, output
    assert "bad.png: FAILED [load]" in output and "1 of 2 images digitized, 1 failed" in output
    rows = {r["image"]: r for r in read_summary(out / "summary.csv")}
    assert rows["bad.png"]["status"] == "failed" and "decode" in rows["bad.png"]["error"]
    assert rows["good.png"]["status"] == "ok" and (out / "good" / "points.csv").exists()


def test_batch_shared_stems_get_separate_folders(tmp_path: Path) -> None:
    charts, out = tmp_path / "charts", tmp_path / "out"
    charts.mkdir()
    add_chart(charts, "chart", seed=3, x_range=(0, 100))
    (charts / "chart.jpg").write_bytes((charts / "chart.png").read_bytes())  # decoding goes by content
    code, output = invoke(charts, "--y-range", "0,100", "--out", out)
    assert code == 0, output
    assert (out / "chart_png" / "points.csv").exists() and (out / "chart_jpg" / "points.csv").exists()


def test_batch_invalid_settings_stop_before_processing(tmp_path: Path) -> None:
    charts, out = tmp_path / "charts", tmp_path / "out"
    charts.mkdir()
    add_chart(charts, "good", seed=3, x_range=(0, 100))
    add_chart(charts, "typo", seed=4, x_range=(0, 100))
    (charts / "typo.yaml").write_text("axes:\n  x_range: [0, 100]\ntracee: {}\n")
    code, output = invoke(charts, "--y-range", "0,100", "--out", out)
    assert code == 2
    assert "invalid for 1 of 2 images" in output and "typo.png (with typo.yaml)" in output and "tracee" in output
    assert not out.exists()


def test_batch_empty_folder(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("")
    code, output = invoke(tmp_path, "--x-range", "0,1", "--y-range", "0,1")
    assert code == 2
    assert "No images found" in output and ".png" in output


def test_single_image_uses_its_sidecar(tmp_path: Path) -> None:
    add_chart(tmp_path, "chart", seed=3, x_range=(0, 100))
    code, output = invoke(tmp_path / "chart.png", "--y-range", "0,100", "--out", tmp_path / "out")
    assert code == 0, output
    assert "Using settings from" in output and "chart.yaml" in output
    assert not (tmp_path / "out" / "summary.csv").exists()
