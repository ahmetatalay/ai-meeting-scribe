"""Local web UI for scribe.

Serves a single-page interface on loopback and forwards the recording engine's
event stream to the browser over SSE. The engine is unchanged: this is just
another consumer of :class:`RecordingSession`'s event bus.

Security posture — this server holds verbatim meeting transcripts, so the
localhost-specific threats are treated as real rather than theoretical:

* **Loopback only.** The bind address is hardcoded to ``127.0.0.1`` and there is
  deliberately no host option, so the server cannot be exposed by a typo or a
  stray flag. To reach it from another machine, use an SSH local port-forward
  rather than widening the bind.
* **Host header validation.** A page you visit can point a hostname it controls
  at ``127.0.0.1`` (DNS rebinding) and then talk to this server as same-origin.
  Requests whose ``Host`` is not localhost are refused.
* **CSRF token.** The browser's same-origin policy stops another site *reading*
  responses, but not *sending* a ``POST``. Without a token, any page you visited
  could start or stop a recording. The token is minted at startup, embedded in
  the page itself, and required on every mutating route. ``/api/events`` takes it
  as a query parameter because ``EventSource`` cannot set headers.
* **No CORS middleware.** Same-origin only; nothing else is granted access.

The token authorises *this browser tab*, nothing more. It is not a user
authentication mechanism and is not a substitute for one — there is exactly one
local user, and the loopback bind is what keeps the audience to that user.
"""

from __future__ import annotations

import asyncio
import json
import queue
import secrets
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from ..capture import find_device, get_system_audio, list_devices
from ..session import RecordingSession, SessionConfig

HOST = "127.0.0.1"  # never configurable; see the module docstring
DEFAULT_PORT = 8765

STATIC_DIR = Path(__file__).parent / "static"

# Host header values that mean "this machine". Anything else is refused.
ALLOWED_HOSTNAMES = {"localhost", "127.0.0.1", "[::1]", "::1"}

MUTATING_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


class AppState:
    """Process-wide state: the token, the current session, and defaults."""

    def __init__(self, raw_dir: str | Path, default_model: str = "small"):
        self.token = secrets.token_urlsafe(32)
        self.raw_dir = Path(raw_dir).expanduser()
        self.default_model = default_model
        self.session: RecordingSession | None = None
        self.lock = threading.Lock()

    def active_session(self) -> RecordingSession | None:
        with self.lock:
            session = self.session
        if session is not None and session.is_running:
            return session
        return None


def _hostname_of(host_header: str) -> str:
    """Strip the port from a Host header, keeping bracketed IPv6 intact."""
    if not host_header:
        return ""
    if host_header.startswith("["):
        return host_header.split("]")[0] + "]"
    return host_header.split(":")[0]


def create_app(state: AppState) -> FastAPI:
    app = FastAPI(title="scribe", docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def guard(request: Request, call_next):
        # DNS rebinding: a hostname the attacker controls resolving to 127.0.0.1
        # would otherwise be treated as same-origin by the browser.
        hostname = _hostname_of(request.headers.get("host", ""))
        if hostname not in ALLOWED_HOSTNAMES:
            return JSONResponse(
                {"error": "unexpected Host header"}, status_code=421
            )

        # CSRF: another site cannot read this server's responses, but it can
        # send a POST. The token is only obtainable by loading the page itself.
        if request.method in MUTATING_METHODS:
            supplied = request.headers.get("x-scribe-token", "")
            if not secrets.compare_digest(supplied, state.token):
                return JSONResponse({"error": "invalid or missing token"},
                                    status_code=403)

        return await call_next(request)

    def require_token(token: str) -> None:
        """Token check for GET routes, which pass it as a query parameter."""
        if not secrets.compare_digest(token or "", state.token):
            raise HTTPException(status_code=403, detail="invalid or missing token")

    # ------------------------------------------------------------------ page

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        # Embedded rather than served from an endpoint: a rebinding attack that
        # could fetch the token from an API would defeat the point.
        html = html.replace("__SCRIBE_TOKEN__", state.token)
        return HTMLResponse(
            html,
            headers={
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
                # The page is self-contained; no external loads of any kind.
                "Content-Security-Policy": (
                    "default-src 'none'; style-src 'unsafe-inline'; "
                    "script-src 'unsafe-inline'; connect-src 'self'"
                ),
            },
        )

    # ------------------------------------------------------------------- api

    @app.get("/api/devices")
    async def devices(token: str = Query("")) -> dict:
        require_token(token)
        system = get_system_audio()
        return {
            "inputs": [
                {"id": d.id, "name": d.name, "channels": d.max_input_channels,
                 "sample_rate": int(d.default_samplerate)}
                for d in list_devices("input")
            ],
            "outputs": [
                {"id": d.id, "name": d.name, "channels": d.max_output_channels,
                 "sample_rate": int(d.default_samplerate)}
                for d in list_devices("output")
            ],
            "default_input": system["input"].id if system["input"] else None,
            "default_output_name": (
                system["output"].name if system["output"] else None
            ),
            "loopback_id": (
                find_device("BlackHole", "input").id
                if find_device("BlackHole", "input") else None
            ),
        }

    @app.get("/api/preflight")
    async def preflight(token: str = Query("")) -> dict:
        require_token(token)
        loopback = find_device("BlackHole", "input")
        system = get_system_audio()
        out_name = system["output"].name if system["output"] else None
        routed = bool(out_name) and any(
            hint in out_name.lower()
            for hint in ("blackhole", "meeting capture", "multi-output")
        )
        checks = [
            {
                "id": "loopback",
                "label": "BlackHole loopback device",
                "ok": loopback is not None,
                "detail": loopback.name if loopback else
                          "Not found — brew install blackhole-2ch",
            },
            {
                "id": "routing",
                "label": "System output reaches the loopback",
                "ok": routed,
                "detail": out_name or "No output device",
                "hint": None if routed else
                        "Set Sound Output to a Multi-Output Device that "
                        "includes BlackHole.",
            },
            {
                "id": "microphone",
                "label": "Microphone available",
                "ok": system["input"] is not None,
                "detail": system["input"].name if system["input"] else
                          "No default input",
            },
        ]
        return {
            "checks": checks,
            "ready": all(c["ok"] for c in checks if c["id"] != "routing"),
            "note": (
                "In Zoom/Teams set Speaker to 'Same as System', not your "
                "headset — choosing the headset there bypasses the loopback "
                "and records only your own voice."
            ),
        }

    @app.get("/api/status")
    async def status(token: str = Query("")) -> dict:
        require_token(token)
        session = state.active_session() or state.session
        if session is None:
            return {"state": "idle", "stats": None}
        return {
            "state": session.state,
            "stats": session.stats_dict(),
            "title": session.writer.title if session.writer else None,
            "markdown": str(session.writer.path)
            if session.writer and session.writer.path else None,
            "levels": session.capture.levels() if session.capture else None,
        }

    @app.post("/api/start")
    async def start(request: Request) -> dict:
        if state.active_session() is not None:
            raise HTTPException(
                status_code=409,
                detail="a recording is already running",
            )

        body = await request.json() if await request.body() else {}
        try:
            config = SessionConfig(
                mic_device=_as_device(body.get("mic_device")),
                system_device=_as_device(body.get("system_device")),
                model=body.get("model") or state.default_model,
                language=_as_language(body.get("language")),
                raw_dir=state.raw_dir,
                title=body.get("title") or None,
                use_speakers=bool(body.get("use_speakers", True)),
            )
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        session = RecordingSession(config)
        with state.lock:
            state.session = session
        session.start()
        return {"state": session.state}

    @app.post("/api/stop")
    async def stop() -> dict:
        session = state.active_session()
        if session is None:
            raise HTTPException(status_code=409, detail="nothing is recording")
        # stop() blocks while the tail is flushed and transcribed; keep the
        # event loop free so the browser keeps receiving events meanwhile.
        await asyncio.to_thread(session.stop)
        return {"state": session.state, "stats": session.stats_dict()}

    @app.post("/api/title")
    async def retitle(request: Request) -> dict:
        body = await request.json()
        title = (body.get("title") or "").strip()
        if not title:
            raise HTTPException(status_code=400, detail="title is required")
        with state.lock:
            session = state.session
        if session is None:
            raise HTTPException(status_code=409, detail="no session")
        session.set_title(title)
        return {"title": title}

    @app.get("/api/events")
    async def events(token: str = Query("")) -> StreamingResponse:
        # EventSource cannot set headers, so the token arrives as a query
        # parameter. The URL never leaves this machine.
        require_token(token)
        return StreamingResponse(
            _event_stream(state),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-store",
                "X-Accel-Buffering": "no",
            },
        )

    return app


def _as_device(value) -> int | None:
    if value in (None, "", "none"):
        return None
    return int(value)


def _as_language(value) -> str | None:
    if value in (None, "", "auto"):
        return None
    return str(value)


async def _event_stream(state: AppState):
    """Forward the active session's events to the browser as SSE.

    Resubscribes when a new session starts, so the page does not need to
    reconnect between recordings.
    """
    subscribed: RecordingSession | None = None
    q: queue.Queue | None = None

    yield _sse({"type": "hello"})
    try:
        while True:
            session = state.session
            if session is not subscribed:
                if subscribed is not None and q is not None:
                    subscribed.bus.unsubscribe(q)
                subscribed, q = session, None
                if session is not None:
                    q = session.bus.subscribe(replay=True)

            if q is None:
                await asyncio.sleep(0.2)
                yield ": keep-alive\n\n"
                continue

            try:
                event = await asyncio.to_thread(q.get, True, 0.5)
            except queue.Empty:
                yield ": keep-alive\n\n"
                continue

            yield _sse(event)
    finally:
        if subscribed is not None and q is not None:
            subscribed.bus.unsubscribe(q)


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def serve(raw_dir: str | Path, port: int = DEFAULT_PORT,
          model: str = "small", open_browser: bool = True) -> None:
    """Run the local web UI. Binds loopback only."""
    import uvicorn

    state = AppState(raw_dir=raw_dir, default_model=model)
    app = create_app(state)
    url = f"http://{HOST}:{port}/"

    print(f"[web] scribe UI on {url}")
    print(f"[web] Transcripts: {state.raw_dir}")
    print("[web] Loopback only — not reachable from other machines.")
    print("[web] Ctrl+C to stop.\n")

    if open_browser:
        import webbrowser
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    uvicorn.run(app, host=HOST, port=port, log_level="warning")
