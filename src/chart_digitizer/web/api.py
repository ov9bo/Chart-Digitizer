"""Pure helpers behind the web UI: request settings to config data, and results to JSON."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import quote

import cv2
import numpy as np

from chart_digitizer.batch import sidecar_path
from chart_digitizer.config import Config, PlotAreaConfig, read_yaml
from chart_digitizer.debug import DebugSink
from chart_digitizer.errors import ConfigError, CornerError, DigitizeError
from chart_digitizer.output.writers import points_records
from chart_digitizer.pipeline import Result, metadata
from chart_digitizer.stages.plotarea import detect_plot_area

# Config sections the browser may set. Everything else keeps its default, and the output folder is
# always the server's, so a request cannot write outside it.
SETTINGS_SECTIONS = ("axes", "corners", "sampling", "curve", "clutter")
SETTINGS_KEYS = frozenset({*SETTINGS_SECTIONS, "debug", "name"})
PREVIEW_MAX_SIDE = 2000
PREVIEW_QUALITY = 88
_UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def display_path(path: Path, cwd: Path | None = None) -> str:
    """``path`` relative to the working directory when it lies inside it, else absolute."""
    path = path.resolve()
    try:
        return str(path.relative_to((cwd or Path.cwd()).resolve()))
    except ValueError:
        return str(path)


def host_name(netloc: str) -> str:
    """The host part of a Host header or URL netloc, lower-cased, without port or IPv6 brackets."""
    netloc = netloc.strip().lower()
    if netloc.startswith("["):
        return netloc[1 : netloc.find("]")] if "]" in netloc else netloc[1:]
    return netloc.rsplit(":", 1)[0] if netloc.count(":") == 1 else netloc


def request_allowed(host_header: str | None, origin_header: str | None, bind_host: str) -> bool:
    """Whether a request may reach a server bound to ``bind_host``.

    On a loopback address, the Host header must name a loopback host. This stops DNS-rebinding
    pages from reaching the server through a hostname of their own. A request that carries an
    Origin must come from the same host, so other sites cannot post to the server.
    """
    host = host_name(host_header or "")
    if host_name(bind_host) in LOOPBACK_HOSTS and host not in LOOPBACK_HOSTS:
        return False
    if origin_header and origin_header != "null":
        return host_name(origin_header.split("://", 1)[-1]) == host
    return True


def safe_name(text: str, fallback: str = "chart") -> str:
    """One path component made only of letters, digits, '.', '_' and '-'."""
    name = _UNSAFE_NAME_CHARS.sub("_", text.strip()).strip("._")
    return name[:80] or fallback


def sidecar_settings(image: Path) -> tuple[dict[str, Any] | None, str | None]:
    """The browser-settable sections of the image's own YAML file, and an error message if it is unreadable."""
    sidecar = sidecar_path(image)
    if sidecar is None:
        return None, None
    try:
        data = read_yaml(sidecar)
    except ConfigError as exc:
        return None, str(exc)
    return {key: data[key] for key in SETTINGS_SECTIONS if isinstance(data.get(key), dict)}, None


def config_data(settings: dict[str, Any], out_root: Path) -> dict[str, Any]:
    """Config data for one run from the settings the browser sent."""
    unknown = sorted(set(settings) - SETTINGS_KEYS)
    if unknown:
        raise ConfigError(f"Unknown settings: {', '.join(unknown)}", hint=f"Allowed: {', '.join(sorted(SETTINGS_KEYS))}.")
    corners = settings.get("corners")
    if isinstance(corners, dict) and corners.get("mode") == "interactive":
        raise ConfigError(
            "The interactive corner window is not available from the browser",
            hint="Drag the corner handles on the image instead; they are sent as manual corners.",
        )
    data = {key: settings[key] for key in SETTINGS_SECTIONS if settings.get(key) is not None}
    data["output"] = {"dir": out_root, "debug": bool(settings.get("debug", False))}
    return data


def error_payload(exc: DigitizeError) -> dict[str, Any]:
    return {"stage": exc.stage, "message": exc.message, "hint": exc.hint}


def detect_corners(image: np.ndarray) -> tuple[list[list[float]] | None, dict[str, Any] | None]:
    """Automatically detected corners in source pixels, or the detection error."""
    try:
        corners = detect_plot_area(image, PlotAreaConfig(), DebugSink())
    except CornerError as exc:
        return None, error_payload(exc)
    return corners.points.tolist(), None


def preview_jpeg(image: np.ndarray, max_side: int = PREVIEW_MAX_SIDE) -> bytes:
    """The image shrunk to fit ``max_side``, as JPEG bytes, for display in the browser."""
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        image = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, PREVIEW_QUALITY])
    if not ok:
        raise OSError("Could not encode the preview image")
    return encoded.tobytes()


def file_url(path: Path, out_root: Path, version: int) -> str:
    """URL under /files/ for a file inside ``out_root``; ``version`` defeats browser caching."""
    relative = path.resolve().relative_to(out_root.resolve()).as_posix()
    return f"/files/{quote(relative)}?v={version}"


def debug_images(out_dir: Path, since: float) -> list[Path]:
    """Stage images written by this run (older files from earlier runs are skipped)."""
    folder = out_dir / "debug"
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.glob("*.png") if p.stat().st_mtime >= since - 1.0)


def result_payload(result: Result, cfg: Config, out_root: Path, started: float, elapsed: float) -> dict[str, Any]:
    """Everything the browser shows after a successful run."""
    version = int(started * 1000)
    out_dir = result.out_dir
    points = result.points
    debug = debug_images(out_dir, started) if cfg.output.debug else []
    return {
        "name": out_dir.name,
        "out_dir": display_path(out_dir),
        "elapsed_s": round(elapsed, 2),
        "meta": metadata(result, cfg),
        "warnings": list(result.warnings),
        "points": points_records(points),
        "stats": {
            "kept": int(points.in_range.sum()),
            "total": int(points.x.size),
            "interpolated": int((points.interpolated & points.in_range).sum()),
            "filled_fraction": result.trace.filled_fraction,
            "corners_source": result.corners.source,
        },
        "files": {
            name: file_url(out_dir / name, out_root, version)
            for name in ("overlay.png", "replot.png", "points.csv", "points.json")
        },
        "debug": [{"name": p.stem, "url": file_url(p, out_root, version)} for p in debug],
    }
