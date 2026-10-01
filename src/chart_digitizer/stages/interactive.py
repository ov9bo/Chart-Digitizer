"""Stage 2 fallback: the user clicks the plot corners in a window.

The corners can be clicked in any order; they are sorted into (xmin,ymin), (xmax,ymin),
(xmax,ymax), (xmin,ymax) by position, as for an upright chart. A magnifier around the cursor shows
the source image at full resolution, so corners can be placed to the pixel on a downscaled view.

Event handling is a small state machine of pure functions over ``PickerState``, so it is tested
without a display; ``pick_corners`` only connects it to an OpenCV window.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

from chart_digitizer.config import InteractiveConfig
from chart_digitizer.errors import CornerError
from chart_digitizer.models import Corners
from chart_digitizer.stages.rectify import validate_corners

Point = tuple[float, float]

CORNER_LABELS = ("xmin,ymin", "xmax,ymin", "xmax,ymax", "xmin,ymax")
_WINDOW = "chart-digitizer: plot corners"
_KEYS = "drag: adjust   u / right-click: undo   r: reset   Enter: accept   Esc: cancel"
_ENTER, _SPACE, _BACKSPACE, _ESC = (13, 10), 32, 8, 27
_WAIT_MS = 20
_BANNER_PX = 52
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_COLOR_BOX = (0, 0, 255)
_COLOR_POINT = (0, 200, 255)
_COLOR_MESSAGE = (80, 220, 255)

_NO_DISPLAY_HINT = (
    "--interactive needs a desktop session that can open a window. Otherwise pass the corners with "
    "--corners x1,y1,x2,y2,x3,y3,x4,y4 in the order (xmin,ymin), (xmax,ymin), (xmax,ymax), (xmin,ymax)."
)


@dataclass(frozen=True)
class PickerState:
    points: tuple[Point, ...] = ()  # source pixels, in the order placed
    dragging: int | None = None  # index of the point following the mouse
    cursor: Point | None = None  # source pixels
    message: str = ""  # status or error shown above the key help


def order_corners(points: np.ndarray) -> np.ndarray:
    """Sort four points into (xmin,ymin), (xmax,ymin), (xmax,ymax), (xmin,ymax) of an upright chart.

    That is bottom-left, bottom-right, top-right, top-left: counter-clockwise on screen, starting
    from the point furthest down and to the left.
    """
    centre = points.mean(axis=0)
    angle = np.arctan2(centre[1] - points[:, 1], points[:, 0] - centre[0])  # y up
    ordered = points[np.argsort(angle, kind="stable")]
    start = int(np.argmax(ordered[:, 1] - ordered[:, 0]))
    return np.roll(ordered, -start, axis=0)


def nearest(points: tuple[Point, ...], p: Point, radius: float) -> int | None:
    """Index of the placed point closest to ``p`` within ``radius``, if any."""
    if not points:
        return None
    dist = np.hypot(*(np.asarray(points) - np.asarray(p)).T)
    i = int(np.argmin(dist))
    return i if dist[i] <= radius else None


def press(state: PickerState, p: Point, radius: float) -> PickerState:
    """Left button down: grab a nearby point, or place a new one while fewer than four exist."""
    grabbed = nearest(state.points, p, radius)
    if grabbed is not None:
        return replace(state, dragging=grabbed, cursor=p, message="")
    if len(state.points) < 4:
        return replace(state, points=(*state.points, p), dragging=len(state.points), cursor=p, message="")
    return replace(state, cursor=p)


def move(state: PickerState, p: Point) -> PickerState:
    i = state.dragging
    points = state.points if i is None else (*state.points[:i], p, *state.points[i + 1 :])
    return replace(state, points=points, cursor=p)


def release(state: PickerState) -> PickerState:
    return replace(state, dragging=None)


def undo(state: PickerState) -> PickerState:
    return replace(state, points=state.points[:-1], dragging=None, message="")


def reset(state: PickerState) -> PickerState:
    return PickerState(cursor=state.cursor)


def confirm(state: PickerState, image_shape: tuple[int, ...]) -> tuple[PickerState, Corners | None]:
    """Accept the four corners if they form a valid plot box; otherwise say why not."""
    if len(state.points) < 4:
        return replace(state, message=f"Place all four corners first ({len(state.points)}/4 placed)"), None
    corners = Corners(points=order_corners(np.asarray(state.points, dtype=np.float64)), source="interactive")
    try:
        validate_corners(corners, image_shape)
    except CornerError as exc:
        return replace(state, message=f"{exc.message}; adjust the corners"), None
    return state, corners


def handle_mouse(state: PickerState, event: int, p: Point, radius: float) -> PickerState:
    if event == cv2.EVENT_LBUTTONDOWN:
        return press(state, p, radius)
    if event == cv2.EVENT_MOUSEMOVE:
        return move(state, p)
    if event == cv2.EVENT_LBUTTONUP:
        return release(state)
    if event == cv2.EVENT_RBUTTONDOWN:
        return undo(state)
    return state


def handle_key(state: PickerState, key: int, image_shape: tuple[int, ...]) -> tuple[PickerState, Corners | None]:
    """Apply a key press; returns the accepted corners on Enter. Raises CornerError on Esc or q."""
    if key in (_ESC, ord("q")):
        raise CornerError("Corner picking was cancelled", hint=_NO_DISPLAY_HINT)
    if key in (_BACKSPACE, ord("u")):
        return undo(state), None
    if key == ord("r"):
        return reset(state), None
    if key in (*_ENTER, _SPACE):
        return confirm(state, image_shape)
    return state, None


def display_scale(image_shape: tuple[int, ...], max_window: tuple[int, int]) -> float:
    """Factor that shrinks the image to fit ``max_window`` (width, height) below the banner; never enlarges it."""
    h, w = image_shape[:2]
    return min(1.0, max_window[0] / w, (max_window[1] - _BANNER_PX) / h)


def to_source(p: tuple[float, float], scale: float) -> Point:
    """Window pixels -> source pixels (pixel centres on integers in both)."""
    return ((p[0] + 0.5) / scale - 0.5, (p[1] - _BANNER_PX + 0.5) / scale - 0.5)


def to_screen(p: Point, scale: float) -> tuple[int, int]:
    return round((p[0] + 0.5) * scale - 0.5), round((p[1] + 0.5) * scale - 0.5) + _BANNER_PX


def prompt(state: PickerState) -> str:
    if state.message:
        return state.message
    if len(state.points) < 4:
        return f"Click the 4 corners of the plot area, in any order ({len(state.points)}/4 placed)"
    return "Check the corners, then press Enter to accept"


def _text(view: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int], scale: float = 0.5) -> None:
    cv2.putText(view, text, origin, _FONT, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(view, text, origin, _FONT, scale, color, 1, cv2.LINE_AA)


def magnifier(source: np.ndarray, state: PickerState, cfg: InteractiveConfig) -> np.ndarray:
    """Zoomed view of the source image around the cursor, with a crosshair and the placed points."""
    assert state.cursor is not None
    n = cfg.magnifier_px
    crop = max(3, int(np.ceil(n / cfg.magnifier_zoom)))
    patch = cv2.getRectSubPix(source, (crop, crop), (float(state.cursor[0]), float(state.cursor[1])))
    zoom = n / crop
    view = cv2.resize(patch, (n, n), interpolation=cv2.INTER_NEAREST)
    for p in state.points:
        u = (p[0] - state.cursor[0]) * zoom + n / 2
        v = (p[1] - state.cursor[1]) * zoom + n / 2
        if 0 <= u < n and 0 <= v < n:
            cv2.circle(view, (round(u), round(v)), 5, _COLOR_POINT, 1, cv2.LINE_AA)
    c = n // 2
    cv2.line(view, (c, 0), (c, n - 1), _COLOR_BOX, 1)
    cv2.line(view, (0, c), (n - 1, c), _COLOR_BOX, 1)
    cv2.rectangle(view, (0, 0), (n - 1, n - 1), (255, 255, 255), 2)
    return view


def render_view(base: np.ndarray, source: np.ndarray, state: PickerState, scale: float, cfg: InteractiveConfig) -> np.ndarray:
    """The window contents: key help and prompt above the scaled image, points, box, magnifier."""
    h, w = base.shape[:2]
    view = np.zeros((h + _BANNER_PX, w, 3), np.uint8)
    view[_BANNER_PX:] = base
    _text(view, prompt(state), (8, 20), _COLOR_MESSAGE if state.message else (255, 255, 255))
    _text(view, _KEYS, (8, 42), (200, 200, 200), 0.45)
    if len(state.points) == 4:
        ordered = order_corners(np.asarray(state.points, dtype=np.float64))
        screen = np.array([to_screen(tuple(p), scale) for p in ordered], np.int32)
        cv2.polylines(view, [screen.reshape(-1, 1, 2)], True, _COLOR_BOX, 1, cv2.LINE_AA)
        for p, label in zip(screen, CORNER_LABELS, strict=True):
            _text(view, label, (int(p[0]) + 8, int(p[1]) - 8), _COLOR_POINT)
    for i, p in enumerate(state.points):
        s = to_screen(p, scale)
        cv2.circle(view, s, 6, _COLOR_POINT, 2, cv2.LINE_AA)
        if len(state.points) < 4:
            _text(view, str(i + 1), (s[0] + 8, s[1] - 8), _COLOR_POINT)
    if state.cursor is not None:
        inset = magnifier(source, state, cfg)
        n = inset.shape[0]
        # Put the magnifier in the top corner away from the cursor.
        left = to_screen(state.cursor, scale)[0] > w / 2
        x0 = 8 if left else max(0, w - n - 8)
        y0 = _BANNER_PX + 8
        region = view[y0 : y0 + n, x0 : x0 + n]
        region[:] = inset[: region.shape[0], : region.shape[1]]
    return view


def pick_corners(image: np.ndarray, initial: Corners | None, note: str, cfg: InteractiveConfig) -> Corners:
    """Show ``image`` in a window and let the user place or adjust the four plot corners."""
    scale = display_scale(image.shape, cfg.max_window)
    h, w = image.shape[:2]
    base = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else image
    points = tuple((float(u), float(v)) for u, v in initial.points) if initial is not None else ()
    state = PickerState(points=points, message=note)
    radius = cfg.grab_radius_px / scale
    events: list[tuple[int, int, int]] = []
    try:
        cv2.namedWindow(_WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(_WINDOW, lambda event, x, y, _flags, _param: events.append((event, x, y)))
    except cv2.error as exc:
        raise CornerError(f"Could not open a window to pick the corners: {exc.err or exc}", hint=_NO_DISPLAY_HINT) from exc
    try:
        while True:
            pending, events[:] = list(events), []
            for event, x, y in pending:
                state = handle_mouse(state, event, to_source((x, y), scale), radius)
            cv2.imshow(_WINDOW, render_view(base, image, state, scale, cfg))
            key = cv2.waitKey(_WAIT_MS)
            if cv2.getWindowProperty(_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                raise CornerError("The corner window was closed before the corners were accepted", hint=_NO_DISPLAY_HINT)
            if key == -1:
                continue
            state, corners = handle_key(state, key & 0xFF, image.shape)
            if corners is not None:
                return corners
    finally:
        try:
            cv2.destroyWindow(_WINDOW)
        except cv2.error:
            pass
