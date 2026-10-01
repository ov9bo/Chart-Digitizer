"""Web helpers: names, displayed paths and the request checks that keep the server local."""

from __future__ import annotations

from pathlib import Path

import pytest

from chart_digitizer.web.api import display_path, host_name, request_allowed, safe_name, sidecar_settings


@pytest.mark.parametrize(
    ("text", "expected"),
    [("chart 1.png", "chart_1.png"), ("../../etc/passwd", "etc_passwd"), ("", "chart"), ("...", "chart")],
)
def test_safe_name_keeps_names_inside_one_folder(text: str, expected: str) -> None:
    name = safe_name(text)
    assert "/" not in name and "\\" not in name and not name.startswith(".")
    assert name == expected


def test_display_path_is_relative_inside_cwd(tmp_path: Path) -> None:
    assert display_path(tmp_path / "out" / "a", cwd=tmp_path) == str(Path("out") / "a")


def test_display_path_is_absolute_outside_cwd(tmp_path: Path) -> None:
    other = tmp_path / "elsewhere"
    assert display_path(other, cwd=tmp_path / "cwd") == str(other.resolve())


@pytest.mark.parametrize(
    ("netloc", "host"),
    [("127.0.0.1:8765", "127.0.0.1"), ("LOCALHOST", "localhost"), ("[::1]:8765", "::1"), ("[::1]", "::1")],
)
def test_host_name_strips_port_and_brackets(netloc: str, host: str) -> None:
    assert host_name(netloc) == host


@pytest.mark.parametrize("host", ["127.0.0.1:8765", "localhost:8765", "[::1]:8765"])
def test_loopback_requests_are_allowed(host: str) -> None:
    assert request_allowed(host, None, "127.0.0.1")
    assert request_allowed(host, f"http://{host}", "127.0.0.1")


def test_rebound_hostname_is_refused_on_loopback() -> None:
    assert not request_allowed("evil.example:8765", None, "127.0.0.1")
    assert not request_allowed(None, None, "127.0.0.1")


def test_cross_origin_post_is_refused() -> None:
    assert not request_allowed("127.0.0.1:8765", "https://evil.example", "127.0.0.1")


def test_any_host_is_allowed_when_bound_to_the_network() -> None:
    assert request_allowed("myhost.lan:8765", "http://myhost.lan:8765", "0.0.0.0")


def test_sidecar_settings_keep_only_browser_sections(tmp_path: Path) -> None:
    image = tmp_path / "chart.png"
    image.write_bytes(b"")
    (tmp_path / "chart.yaml").write_text("axes: {x_range: [0, 5], y_range: [1, 2]}\noutput: {dir: /elsewhere}\n")
    settings, error = sidecar_settings(image)
    assert error is None and settings == {"axes": {"x_range": [0, 5], "y_range": [1, 2]}}


def test_sidecar_settings_report_a_broken_file(tmp_path: Path) -> None:
    image = tmp_path / "chart.png"
    (tmp_path / "chart.yaml").write_text("axes: [unclosed\n")
    settings, error = sidecar_settings(image)
    assert settings is None and error is not None and "not valid YAML" in error


def test_no_sidecar_means_no_settings(tmp_path: Path) -> None:
    assert sidecar_settings(tmp_path / "chart.png") == (None, None)
