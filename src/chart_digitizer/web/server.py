"""``digitize-ui``: a local web page for digitizing charts, served on 127.0.0.1 only.

Uses only the standard library's HTTP server. Uploaded images live in a temporary folder that is
removed on exit; results are written to the output folder exactly as the CLI writes them.
"""

from __future__ import annotations

import json
import mimetypes
import shutil
import tempfile
import threading
import time
import traceback
import uuid
import webbrowser
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import unquote, urlsplit

import typer

from chart_digitizer.config import build_config
from chart_digitizer.errors import DigitizeError, LoadError
from chart_digitizer.imgio import IMAGE_SUFFIXES, read_image
from chart_digitizer.pipeline import digitize, write_outputs
from chart_digitizer.web.api import (
    config_data,
    detect_corners,
    display_path,
    error_payload,
    preview_jpeg,
    request_allowed,
    result_payload,
    sidecar_settings,
    safe_name,
)

STATIC_DIR = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 80 * 1024 * 1024
MAX_JSON_BYTES = 1024 * 1024


@dataclass(frozen=True)
class Upload:
    id: str
    path: Path
    width: int
    height: int
    preview: bytes  # JPEG
    corners: list[list[float]] | None
    corners_error: dict[str, Any] | None
    settings: dict[str, Any] | None = None  # from the image's own YAML file, if it has one
    settings_error: str | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.path.name,
            "stem": safe_name(self.path.stem),
            "width": self.width,
            "height": self.height,
            "preview": f"/api/upload/{self.id}/preview.jpg",
            "corners": self.corners,
            "corners_error": self.corners_error,
            "settings": self.settings,
            "settings_error": self.settings_error,
        }


@dataclass
class AppState:
    out_root: Path
    upload_root: Path
    samples_dir: Path | None
    uploads: dict[str, Upload] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def samples(self) -> list[Path]:
        if self.samples_dir is None or not self.samples_dir.is_dir():
            return []
        return sorted(p for p in self.samples_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)

    def register(self, path: Path) -> Upload:
        """Read the image, detect its corners and remember it; raises LoadError if unreadable."""
        image = read_image(path)
        corners, corners_error = detect_corners(image)
        h, w = image.shape[:2]
        settings, settings_error = sidecar_settings(path)
        upload = Upload(
            uuid.uuid4().hex[:12], path, w, h, preview_jpeg(image), corners, corners_error, settings, settings_error
        )
        with self.lock:
            self.uploads[upload.id] = upload
        return upload

    def upload(self, upload_id: Any) -> Upload:
        with self.lock:
            found = self.uploads.get(str(upload_id))
        if found is None:
            raise LoadError("That image is no longer on the server", hint="Drop the image onto the page again.")
        return found


class UIServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], state: AppState) -> None:
        super().__init__(address, Handler)
        self.state = state
        self.bind_host = address[0]


class Handler(BaseHTTPRequestHandler):
    server: UIServer
    protocol_version = "HTTP/1.1"

    # ---- plumbing -------------------------------------------------------------------------------

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - signature from the base class
        if not self.path.startswith(("/static/", "/files/", "/api/upload/")):
            super().log_message(format, *args)

    def send_bytes(self, body: bytes, content_type: str, status: int = HTTPStatus.OK, cache: bool = False) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
        self.send_bytes(json.dumps(payload).encode("utf-8"), "application/json", status)

    def send_error_json(self, status: int, stage: str, message: str, hint: str | None = None) -> None:
        self.send_json({"error": {"stage": stage, "message": message, "hint": hint}}, status)

    def send_file(self, path: Path, cache: bool = False) -> None:
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type == "application/javascript":
            content_type += "; charset=utf-8"
        self.send_bytes(path.read_bytes(), content_type, cache=cache)

    def read_body(self, limit: int) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            raise LoadError("The request was empty", hint="Choose an image file and try again.")
        if length > limit:
            raise LoadError(f"The upload is too large ({length / 1e6:.0f} MB; the limit is {limit / 1e6:.0f} MB)")
        return self.rfile.read(length)

    def read_json(self) -> dict[str, Any]:
        try:
            data = json.loads(self.read_body(MAX_JSON_BYTES))
        except json.JSONDecodeError as exc:
            raise LoadError(f"The request was not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise LoadError("The request must be a JSON object")
        return data

    # ---- routing --------------------------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - name from the base class
        self.dispatch(self.route_get)

    def do_POST(self) -> None:  # noqa: N802 - name from the base class
        self.dispatch(self.route_post)

    def dispatch(self, route: Any) -> None:
        """Run a route, turning every failure into a JSON error; nothing fails silently."""
        path = unquote(urlsplit(self.path).path)
        if not request_allowed(self.headers.get("Host"), self.headers.get("Origin"), self.server.bind_host):
            self.send_error_json(
                HTTPStatus.FORBIDDEN,
                "http",
                "Refused a request from another site or hostname",
                "Open the page at http://127.0.0.1 or http://localhost on the port the server printed.",
            )
            return
        try:
            route(path)
        except DigitizeError as exc:
            self.send_json({"error": error_payload(exc)}, HTTPStatus.UNPROCESSABLE_ENTITY)
        except (BrokenPipeError, ConnectionResetError):
            pass  # the browser went away; there is no one left to tell
        except Exception as exc:  # noqa: BLE001 - reported to the browser and printed in full
            traceback.print_exc()
            self.send_error_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "internal",
                f"{type(exc).__name__}: {exc}",
                "This is a bug in chart-digitizer; the full traceback is in the server console.",
            )

    def route_get(self, path: str) -> None:
        state = self.server.state
        if path in ("/", "/index.html"):
            self.send_file(STATIC_DIR / "index.html")
        elif path.startswith("/static/"):
            self.send_file(self.contained(STATIC_DIR, path.removeprefix("/static/")))
        elif path == "/api/samples":
            samples_dir = display_path(state.samples_dir) if state.samples_dir else ""
            self.send_json(
                {
                    "samples": [p.name for p in state.samples()],
                    "samples_dir": samples_dir.replace("\\", "/"),
                    "out_root": display_path(state.out_root),
                }
            )
        elif path.startswith("/api/upload/") and path.endswith("/preview.jpg"):
            upload = state.upload(path.split("/")[3])
            self.send_bytes(upload.preview, "image/jpeg", cache=True)
        elif path.startswith("/files/"):
            self.send_file(self.contained(state.out_root, path.removeprefix("/files/")))
        else:
            self.send_error_json(HTTPStatus.NOT_FOUND, "http", f"Nothing at {path}")

    def route_post(self, path: str) -> None:
        state = self.server.state
        if path == "/api/upload":
            name = safe_name(Path(unquote(self.headers.get("X-Filename", "chart.png"))).name, "chart.png")
            body = self.read_body(MAX_UPLOAD_BYTES)
            folder = state.upload_root / uuid.uuid4().hex[:12]
            folder.mkdir(parents=True)
            (folder / name).write_bytes(body)
            self.send_json(state.register(folder / name).describe())
        elif path == "/api/sample":
            wanted = str(self.read_json().get("name", ""))
            match = next((p for p in state.samples() if p.name == wanted), None)
            if match is None:
                raise LoadError(f"No sample named '{wanted}'")
            self.send_json(state.register(match).describe())
        elif path == "/api/digitize":
            request = self.read_json()
            upload = state.upload(request.get("id"))
            settings = request.get("settings") or {}
            if not isinstance(settings, dict):
                raise LoadError("'settings' must be a JSON object")
            cfg = build_config(config_data(settings, state.out_root))
            name = safe_name(str(settings.get("name") or ""), upload.path.stem)
            started = time.time()
            result = digitize(upload.path, cfg, name)
            write_outputs(result, cfg)
            self.send_json(result_payload(result, cfg, state.out_root, started, time.time() - started))
        else:
            self.send_error_json(HTTPStatus.NOT_FOUND, "http", f"Nothing at {path}")

    def contained(self, root: Path, relative: str) -> Path:
        """``root/relative`` if it is an existing file inside ``root``; refuses anything else."""
        root = root.resolve()
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise LoadError(f"No such file: {relative}")
        return path


app = typer.Typer(add_completion=False, help="Open the chart digitizer in your web browser.")


@app.command()
def main(
    port: Annotated[int, typer.Option(help="Port to serve on.")] = 8765,
    host: Annotated[str, typer.Option(help="Address to bind; keep 127.0.0.1 so only this computer can reach it.")] = "127.0.0.1",
    out: Annotated[Path, typer.Option(help="Output root; results go to OUT/<name>/, as with the CLI.")] = Path("out"),
    samples: Annotated[Path, typer.Option(help="Folder whose images are offered as one-click samples.")] = Path("examples"),
    open_browser: Annotated[bool, typer.Option("--open/--no-open", help="Open the page in the default browser.")] = True,
) -> None:
    upload_root = Path(tempfile.mkdtemp(prefix="chart-digitizer-ui-"))
    state = AppState(out_root=out.resolve(), upload_root=upload_root, samples_dir=samples if samples.is_dir() else None)
    try:
        server = UIServer((host, port), state)
    except OSError as exc:
        shutil.rmtree(upload_root, ignore_errors=True)
        typer.secho(f"Could not listen on {host}:{port}: {exc}", fg=typer.colors.RED, err=True)
        typer.echo("Hint: another program may be using the port; try --port 8766.", err=True)
        raise typer.Exit(1) from exc
    url = f"http://{host}:{port}/"
    typer.echo(f"Chart digitizer UI on {url}  (results in {state.out_root}; Ctrl+C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        typer.echo("Stopping.")
    finally:
        server.server_close()
        shutil.rmtree(upload_root, ignore_errors=True)
