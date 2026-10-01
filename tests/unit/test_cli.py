import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from chart_digitizer.cli import app
from chart_digitizer.imgio import write_image
from chart_digitizer.synth.generate import SynthSpec, render

runner = CliRunner()


@pytest.fixture(scope="module")
def chart_file(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    chart = render(SynthSpec(seed=3))
    path = tmp_path_factory.mktemp("cli") / "chart.png"
    write_image(path, chart.image)
    return path, ",".join(f"{v:.3f}" for v in chart.corners.ravel())


def invoke(*args: str) -> tuple[int, str]:
    result = runner.invoke(app, list(args))
    return result.exit_code, result.output


def test_success_writes_outputs(chart_file: tuple[Path, str], tmp_path: Path) -> None:
    path, corners = chart_file
    code, output = invoke(
        str(path), "--x-range", "0,100", "--y-range", "0,100", "--corners", corners,
        "--points", "11", "--out", str(tmp_path), "--debug",
    )  # fmt: skip
    assert code == 0, output
    out = tmp_path / "chart"
    for name in ("points.csv", "points.json", "overlay.png", "replot.png", "debug/04_rectified.png"):
        assert (out / name).exists(), name
    lines = (out / "points.csv").read_text().splitlines()
    assert lines[0] == "x,y,interpolated_flag" and len(lines) == 12
    document = json.loads((out / "points.json").read_text())
    assert len(document["points"]) == 11
    assert document["corners"]["source"] == "manual"


def test_out_of_range_points_are_flagged(chart_file: tuple[Path, str], tmp_path: Path) -> None:
    path, corners = chart_file
    code, output = invoke(
        str(path), "--x-range", "0,100", "--y-range", "0,100", "--corners", corners,
        "--x-values", "50,150", "--out", str(tmp_path),
    )  # fmt: skip
    assert code == 0, output
    assert "Warning:" in output
    rows = (tmp_path / "chart" / "points.csv").read_text().splitlines()
    assert rows[2] == "150,,out_of_range"
    document = json.loads((tmp_path / "chart" / "points.json").read_text())
    assert document["points"][1]["y"] is None


def test_missing_range_is_usage_error(chart_file: tuple[Path, str]) -> None:
    code, output = invoke(str(chart_file[0]), "--y-range", "0,100")
    assert code == 2
    assert "--x-range" in output


def test_malformed_corners_is_usage_error(chart_file: tuple[Path, str]) -> None:
    code, _ = invoke(str(chart_file[0]), "--x-range", "0,100", "--y-range", "0,100", "--corners", "1,2,3")
    assert code == 2


def test_auto_corners_found_without_corners_option(chart_file: tuple[Path, str], tmp_path: Path) -> None:
    code, output = invoke(str(chart_file[0]), "--x-range", "0,100", "--y-range", "0,100", "--out", str(tmp_path))
    assert code == 0, output
    document = json.loads((tmp_path / "chart" / "points.json").read_text())
    assert document["corners"]["source"] == "auto"


def test_auto_corners_failure_suggests_fallback(tmp_path: Path) -> None:
    blank = tmp_path / "blank.png"
    write_image(blank, np.full((400, 600, 3), 240, np.uint8))
    code, output = invoke(str(blank), "--x-range", "0,100", "--y-range", "0,100", "--out", str(tmp_path))
    assert code == 1
    assert "--corners" in output and "--interactive" in output


def test_mirrored_corners_fail_with_hint(chart_file: tuple[Path, str], tmp_path: Path) -> None:
    path, corners = chart_file
    pts = np.array([float(v) for v in corners.split(",")]).reshape(4, 2)[::-1]
    code, output = invoke(
        str(path), "--x-range", "0,100", "--y-range", "0,100",
        "--corners", ",".join(map(str, pts.ravel())), "--out", str(tmp_path),
    )  # fmt: skip
    assert code == 1
    assert "mirrored" in output and "--interactive" in output


def test_unreadable_image(tmp_path: Path) -> None:
    bogus = tmp_path / "bogus.png"
    bogus.write_bytes(b"not an image")
    code, output = invoke(str(bogus), "--x-range", "0,1", "--y-range", "0,1", "--corners", "0,9,9,9,9,0,0,0")
    assert code == 1
    assert "decode" in output
