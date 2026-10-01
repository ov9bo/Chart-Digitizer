from itertools import permutations

import cv2
import numpy as np
import pytest

from chart_digitizer.config import InteractiveConfig
from chart_digitizer.errors import CornerError
from chart_digitizer.stages.interactive import (
    PickerState,
    confirm,
    display_scale,
    handle_key,
    handle_mouse,
    nearest,
    order_corners,
    render_view,
    to_screen,
    to_source,
)

SHAPE = (600, 800, 3)
# (xmin,ymin), (xmax,ymin), (xmax,ymax), (xmin,ymax): bottom-left, bottom-right, top-right, top-left.
BOX = ((100.0, 500.0), (700.0, 500.0), (700.0, 80.0), (100.0, 80.0))


@pytest.mark.parametrize("order", list(permutations(range(4))))
def test_order_corners_from_any_click_order(order: tuple[int, ...]) -> None:
    clicked = np.array([BOX[i] for i in order])
    assert np.array_equal(order_corners(clicked), np.array(BOX))


def test_order_corners_on_tilted_quad() -> None:
    tilted = np.array([[120, 520], [690, 470], [720, 60], [90, 110]], dtype=float)  # about 5 degrees
    rng = np.random.default_rng(0)
    assert np.array_equal(order_corners(tilted[rng.permutation(4)]), tilted)


def test_nearest_within_radius() -> None:
    assert nearest(BOX, (104, 497), radius=10) == 0
    assert nearest(BOX, (400, 300), radius=10) is None
    assert nearest((), (0, 0), radius=10) is None


def click(state: PickerState, p: tuple[float, float]) -> PickerState:
    state = handle_mouse(state, cv2.EVENT_LBUTTONDOWN, p, 10)
    return handle_mouse(state, cv2.EVENT_LBUTTONUP, p, 10)


def test_click_places_up_to_four_points() -> None:
    state = PickerState()
    for p in BOX:
        state = click(state, p)
    state = click(state, (400, 300))  # a fifth click away from the points does nothing
    assert state.points == BOX and state.dragging is None


def test_drag_moves_grabbed_point() -> None:
    state = PickerState(points=BOX)
    state = handle_mouse(state, cv2.EVENT_LBUTTONDOWN, (702, 82), 10)
    assert state.dragging == 2
    state = handle_mouse(state, cv2.EVENT_MOUSEMOVE, (690, 90), 10)
    state = handle_mouse(state, cv2.EVENT_LBUTTONUP, (690, 90), 10)
    state = handle_mouse(state, cv2.EVENT_MOUSEMOVE, (300, 300), 10)  # no longer dragging
    assert state.points[2] == (690, 90) and state.points[:2] == BOX[:2]
    assert state.cursor == (300, 300)


def test_undo_and_reset() -> None:
    state = PickerState(points=BOX)
    assert handle_mouse(state, cv2.EVENT_RBUTTONDOWN, (0, 0), 10).points == BOX[:3]
    assert handle_key(state, ord("u"), SHAPE)[0].points == BOX[:3]
    assert handle_key(state, ord("r"), SHAPE)[0].points == ()


def test_confirm_needs_four_points() -> None:
    state, corners = handle_key(PickerState(points=BOX[:3]), 13, SHAPE)
    assert corners is None and "3/4" in state.message


def test_confirm_rejects_degenerate_corners() -> None:
    flat = ((100.0, 300.0), (400.0, 301.0), (700.0, 302.0), (400.0, 299.0))
    state, corners = confirm(PickerState(points=flat), SHAPE)
    assert corners is None and "adjust the corners" in state.message


def test_confirm_orders_valid_corners() -> None:
    state = PickerState(points=(BOX[2], BOX[0], BOX[3], BOX[1]))
    _, corners = handle_key(state, 13, SHAPE)
    assert corners is not None and corners.source == "interactive"
    assert np.array_equal(corners.points, np.array(BOX))


@pytest.mark.parametrize("key", [27, ord("q")])
def test_escape_cancels_with_hint(key: int) -> None:
    with pytest.raises(CornerError) as info:
        handle_key(PickerState(points=BOX), key, SHAPE)
    assert "--corners" in info.value.hint


def test_display_scale_fits_window_and_never_enlarges() -> None:
    assert display_scale((600, 800, 3), (1400, 900)) == 1.0
    scale = display_scale((3000, 4000, 3), (1400, 900))
    assert 4000 * scale <= 1400 and 3000 * scale + 52 <= 900


def test_screen_source_round_trip() -> None:
    scale = 0.35
    for p in BOX:
        back = to_source(to_screen(p, scale), scale)
        assert np.hypot(back[0] - p[0], back[1] - p[1]) <= 0.5 / scale + 1e-9


def test_render_view_draws_banner_box_and_magnifier() -> None:
    cfg = InteractiveConfig()
    source = np.full(SHAPE, 255, np.uint8)
    scale = 0.5
    base = cv2.resize(source, None, fx=scale, fy=scale)
    state = PickerState(points=BOX, cursor=(650.0, 450.0))
    view = render_view(base, source, state, scale, cfg)
    assert view.shape == (300 + 52, 400, 3)
    assert view[:52].mean() < 100  # dark banner with text
    # The magnifier sits in the top-left corner, away from the cursor on the right.
    centre = 60 + cfg.magnifier_px // 2
    assert tuple(view[centre, 8 + 20]) == (0, 0, 255)  # its crosshair
    assert tuple(view[centre, 400 - 20]) != (0, 0, 255)
