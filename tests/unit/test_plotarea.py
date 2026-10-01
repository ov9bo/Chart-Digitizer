from pathlib import Path

import cv2
import numpy as np
import pytest

from chart_digitizer.config import PlotAreaConfig
from chart_digitizer.debug import DebugSink
from chart_digitizer.errors import CornerError
from chart_digitizer.stages.plotarea import (
    Line,
    block_median,
    detect_plot_area,
    find_box,
    frame,
    hough_peaks,
    ink_run,
    ink_threshold,
    line_contrast,
    meets,
    noise_level,
    refit,
    robust_line_fit,
)
from chart_digitizer.synth.generate import SynthSpec, render

NO_DEBUG = DebugSink(None)


def line(family: str, rho: float, t0: float, t1: float, angle: float = 0.0) -> Line:
    direction, normal = frame(family, angle)  # type: ignore[arg-type]
    return Line(direction=direction, normal=normal, rho=rho, t0=t0, t1=t1)


def framed_page(size: tuple[int, int] = (600, 800), box: tuple[int, int, int, int] = (100, 80, 700, 500)) -> np.ndarray:
    """White page with a black rectangular frame (left, top, right, bottom) and a curve inside."""
    image = np.full((*size, 3), 245, np.uint8)
    left, top, right, bottom = box
    cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 0), 2)
    xs = np.arange(left + 10, right - 10)
    ys = bottom - 10 - (xs - left) ** 2 / (right - left) ** 2 * (bottom - top - 40)
    cv2.polylines(image, [np.column_stack([xs, ys]).astype(np.int32).reshape(-1, 1, 2)], False, (0, 0, 0), 2)
    return image


def test_noise_level_measures_gaussian_noise() -> None:
    rng = np.random.default_rng(0)
    gray = np.clip(128 + rng.normal(0, 5, (300, 300)), 0, 255).astype(np.uint8)
    assert noise_level(gray) == pytest.approx(5, rel=0.2)
    assert noise_level(np.full((50, 50), 200, np.uint8)) == 0


def test_line_contrast_picks_thin_lines_not_edges() -> None:
    gray = np.full((200, 300), 240, np.uint8)
    gray[100:102, 20:200] = 40  # thin dark line
    gray[:, 250:] = 60  # dark region reaching the image edge (a desk)
    contrast = line_contrast(gray, 15)
    assert contrast[100, 150] > 0.7
    assert contrast[50, 150] == pytest.approx(0, abs=1e-6)
    # The dark region and its step edge are not line ink.
    assert contrast[:, 240:].max() < 0.06


def test_line_contrast_ignores_wedge_at_image_border() -> None:
    gray = np.full((200, 300), 240, np.uint8)
    for r in range(200):
        gray[r, : max(0, 6 - r // 30)] = 20  # thin tapering wedge along the left edge
    assert line_contrast(gray, 15)[:, :8].max() < 0.06


def test_block_median_is_smooth_and_full_size() -> None:
    values = np.zeros((100, 160), np.float32)
    values[:, 80:] = 1.0
    medians = block_median(values, 20)
    assert medians.shape == values.shape
    assert medians[50, 10] == 0 and medians[50, 150] == 1
    assert 0 < medians[50, 80] < 1  # interpolated between blocks


def test_ink_threshold_rises_with_local_paper_contrast() -> None:
    cfg = PlotAreaConfig()
    contrast = np.zeros((400, 400), np.float32)
    contrast[:, 200:] = 0.05  # noisy shadowed paper on the right
    threshold = ink_threshold(contrast, cfg, 400)
    assert threshold[200, 20] == pytest.approx(cfg.min_contrast)
    assert threshold[200, 380] == pytest.approx(cfg.noise_contrast_mult * 0.05)


def test_hough_peaks_finds_tilted_line() -> None:
    t = np.arange(0, 400, dtype=np.float64)
    angle = np.deg2rad(3.0)
    points = np.column_stack([t * np.cos(angle), 100 + t * np.sin(angle)])
    peaks = hough_peaks(points, "horizontal", max_tilt_deg=10, min_votes=200, max_lines=5)
    assert len(peaks) == 1
    # Coarse: rho votes are pooled over 3 bins, so over 400 px the angle is found to about 0.3 degrees.
    # The least-squares refit makes it exact.
    assert peaks[0][0] == pytest.approx(angle, abs=np.deg2rad(0.4))
    assert peaks[0][1] == pytest.approx(100, abs=1.5)


def test_ink_run_bridges_short_gaps_only() -> None:
    t = np.concatenate([np.arange(0, 100), np.arange(104, 200), np.arange(300, 330)]).astype(float)
    assert ink_run(t, window=20, min_fill=0.5) == pytest.approx((0, 199), abs=10)
    assert ink_run(np.array([]), 20, 0.5) is None


def test_meets_needs_corner_on_both_runs() -> None:
    top = line("horizontal", 50, 100, 500)
    left = line("vertical", 100, 50, 400)
    assert meets(top, left, tol=3)
    short_left = line("vertical", 100, 80, 400)  # stops 30 px below the top line
    assert not meets(top, short_left, tol=3)


def test_find_box_picks_largest_closed_box() -> None:
    horizontal = [line("horizontal", 50, 100, 500), line("horizontal", 300, 100, 500), line("horizontal", 400, 100, 500)]
    vertical = [line("vertical", 100, 50, 400), line("vertical", 500, 50, 400), line("vertical", 700, 0, 600)]
    top, bottom, left, right = find_box(horizontal, vertical, tol=3, min_side=50)  # type: ignore[misc]
    assert (top.rho, bottom.rho, left.rho, right.rho) == (50, 400, 100, 500)
    assert find_box(horizontal[:1], vertical, tol=3, min_side=50) is None


def test_robust_line_fit_drops_outliers() -> None:
    t = np.arange(50, dtype=np.float64)
    offset = 0.02 * t + 1.0
    offset[[5, 20, 33]] += 6  # a curve crossing, a label
    slope, intercept = robust_line_fit(t, offset)  # type: ignore[misc]
    assert slope == pytest.approx(0.02, abs=1e-6)
    assert intercept == pytest.approx(1.0, abs=1e-6)
    assert robust_line_fit(t[:5], offset[:5]) is None


def test_refit_moves_line_onto_ink() -> None:
    cols = np.arange(0, 400, dtype=np.float64)
    points = np.column_stack([cols, 101.6 + 0.01 * cols])
    fitted = refit(line("horizontal", 100, 0, 400), points, np.ones(len(points)), trim=10)
    for u in (50.0, 350.0):
        v = (fitted.rho - fitted.normal[0] * u) / fitted.normal[1]
        assert v == pytest.approx(101.6 + 0.01 * u, abs=0.05)


def test_detects_frame_on_drawn_page() -> None:
    corners = detect_plot_area(framed_page(), PlotAreaConfig(), NO_DEBUG)
    assert corners.source == "auto"
    expected = np.array([[100, 500], [700, 500], [700, 80], [100, 80]])
    assert np.abs(corners.points - expected).max() <= 1.5


def test_detects_synthetic_chart_to_subpixel() -> None:
    chart = render(SynthSpec(seed=0))
    corners = detect_plot_area(chart.image, PlotAreaConfig(), NO_DEBUG)
    assert np.abs(corners.points - chart.corners).max() <= 1.0


def test_scales_back_to_source_pixels() -> None:
    image = cv2.resize(framed_page(), None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    corners = detect_plot_area(image, PlotAreaConfig(work_size=800), NO_DEBUG)
    expected = np.array([[100, 500], [700, 500], [700, 80], [100, 80]]) * 3 + 1
    assert np.abs(corners.points - expected).max() <= 4


def test_blank_image_fails_with_hint(tmp_path: Path) -> None:
    debug = DebugSink(tmp_path)
    with pytest.raises(CornerError) as info:
        detect_plot_area(np.full((400, 600, 3), 240, np.uint8), PlotAreaConfig(), debug)
    assert "--corners" in info.value.hint and "--interactive" in info.value.hint
    assert (tmp_path / "02_plot_area.png").exists()
