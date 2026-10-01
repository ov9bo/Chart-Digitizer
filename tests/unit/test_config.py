from pathlib import Path

import pytest

from chart_digitizer.config import DEFAULT_POINTS, build_config, load_config, merge
from chart_digitizer.errors import ConfigError

AXES = {"axes": {"x_range": (0, 100), "y_range": (0, 100)}}


def test_defaults() -> None:
    cfg = build_config(AXES)
    assert cfg.sampling.points == DEFAULT_POINTS
    assert cfg.corners.mode == "auto"
    assert cfg.rectify.size == (2000, 2000)
    assert cfg.curve.color is None


def test_axis_ranges_are_required() -> None:
    with pytest.raises(ConfigError, match="axes") as info:
        build_config({})
    assert "--x-range" in (info.value.hint or "")


@pytest.mark.parametrize(
    "axes",
    [
        {"x_range": (5, 5), "y_range": (0, 1)},
        {"x_range": (0, 100), "y_range": (0, 1), "x_log": True},
        {"x_range": (0, 1), "y_range": (0, 1), "grid_step": (0, 1)},
    ],
)
def test_invalid_axes(axes: dict) -> None:
    with pytest.raises(ConfigError):
        build_config({"axes": axes})


def test_log_x_needs_positive_x_values() -> None:
    log_axes = {"axes": {"x_range": (1, 1000), "y_range": (0, 1), "x_log": True}}
    with pytest.raises(ConfigError, match="positive on a log x axis"):
        build_config({**log_axes, "sampling": {"x_values": [0, 10, 100]}})
    assert build_config({**log_axes, "sampling": {"x_values": [1, 10, 100]}}).sampling.x_values == (1, 10, 100)


def test_unknown_keys_are_rejected() -> None:
    with pytest.raises(ConfigError, match="tracee"):
        build_config({**AXES, "tracee": {}})


def test_only_one_sampling_mode() -> None:
    with pytest.raises(ConfigError, match="only one"):
        build_config({**AXES, "sampling": {"points": 10, "step": 5}})


def test_corner_points_imply_manual() -> None:
    cfg = build_config({**AXES, "corners": {"points": [(0, 10), (10, 10), (10, 0), (0, 0)]}})
    assert cfg.corners.mode == "manual"


def test_manual_mode_needs_points() -> None:
    with pytest.raises(ConfigError, match="no corner points"):
        build_config({**AXES, "corners": {"mode": "manual"}})


@pytest.mark.parametrize("color", ["#12ab3F", "hsv:0,255,255", "hsv: 179, 10, 20"])
def test_valid_colors(color: str) -> None:
    assert build_config({**AXES, "curve": {"color": color}}).curve.color == color


@pytest.mark.parametrize("color", ["red", "#12ab3", "hsv:180,0,0", "hsv:1,2"])
def test_invalid_colors(color: str) -> None:
    with pytest.raises(ConfigError, match="color"):
        build_config({**AXES, "curve": {"color": color}})


def test_merge_is_deep_except_replaced_sections() -> None:
    base = {"trace": {"skip_cost": 1, "jump_weight": 2}, "sampling": {"points": 10}}
    merged = merge(base, {"trace": {"skip_cost": 3}, "sampling": {"step": 5}})
    assert merged == {"trace": {"skip_cost": 3, "jump_weight": 2}, "sampling": {"step": 5}}


def test_cli_overrides_yaml(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text("axes:\n  x_range: [0, 10]\n  y_range: [0, 10]\nsampling:\n  points: 7\ntrace:\n  skip_cost: 0.9\n")
    cfg = load_config(path, {"axes": {"x_range": (0, 50)}, "sampling": {"step": 5}})
    assert cfg.axes.x_range == (0, 50)
    assert cfg.axes.y_range == (0, 10)
    assert cfg.sampling.step == 5 and cfg.sampling.points is None
    assert cfg.trace.skip_cost == 0.9


EXAMPLES = Path(__file__).parents[2] / "examples"


def test_example_config_is_valid() -> None:
    cfg = load_config(EXAMPLES / "sample.yaml", {})
    assert cfg.axes.x_range == (0, 100) and cfg.sampling.x_values == tuple(range(10, 100, 10))


@pytest.mark.parametrize("stem", ["skewed-photo", "wavy-orange", "log-log"])
def test_example_sidecars_are_valid(stem: str) -> None:
    assert (EXAMPLES / f"{stem}.jpg").is_file()
    load_config(EXAMPLES / f"{stem}.yaml", {})


def test_bad_yaml(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text("- just\n- a list\n")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(path, {})


def test_missing_yaml(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Could not read"):
        load_config(tmp_path / "nope.yaml", {})
