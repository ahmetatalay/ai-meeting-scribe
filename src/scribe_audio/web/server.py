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
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from .. import library, related
from ..capture import find_device, get_system_audio, list_devices
from ..session import RecordingSession, SessionConfig
from ..summarizer import (
    DEFAULT_MODEL,
    OllamaSummarizer,
    SummarizerError,
    transcript_for_model,
)

HOST = "127.0.0.1"  # never configurable; see the module docstring
DEFAULT_PORT = 8765

STATIC_DIR = Path(__file__).parent / "static"

# Host header values that mean "this machine". Anything else is refused.
ALLOWED_HOSTNAMES = {"localhost", "127.0.0.1", "[::1]", "::1"}

MUTATING_METHODS = {"POST", "PUT", "PATCH", "DELETE"}

# How long an idle event stream waits between keep-alive comments.
IDLE_KEEPALIVE_SECONDS = 15

# How long a summariser availability answer is reused. Every check reaches the
# model server, and the browser asks on each state change and view switch; with
# two tabs open that was a steady trickle of requests to answer a question
# whose answer is "is a local model server installed", which does not change
# second to second.
SUMMARIZER_STATUS_TTL = 30.0


class AppState:
    """Process-wide state: the token, the current session, and defaults."""

    def __init__(self, raw_dir: str | Path, default_model: str = "small",
                 summary_model: str = DEFAULT_MODEL,
                 notes_model: str | None = None, auto_notes: float = 0.0):
        self.token = secrets.token_urlsafe(32)
        self.raw_dir = Path(raw_dir).expanduser()
        self.default_model = default_model
        self.summary_model = summary_model
        # The live panel and the final write-up want different things: the
        # panel needs to keep up with the meeting, the write-up needs quality.
        # Measured on qwen2.5:14b a refine pass runs 13-19s, which is marginal
        # for a 12s cadence, so the two are separately selectable.
        self.notes_model = notes_model or summary_model
        self.auto_notes = auto_notes
        self.summarizer_factory = lambda: OllamaSummarizer(model=self.summary_model)
        self.notes_summarizer_factory = lambda: OllamaSummarizer(
            model=self.notes_model
        )
        self.session: RecordingSession | None = None
        self.lock = threading.Lock()
        self._status_cache: dict | None = None
        self._status_checked_at = 0.0

    def summarizer_status(self, now: float | None = None) -> dict:
        """Whether summarisation is available, reusing a recent answer."""
        moment = time.monotonic() if now is None else now
        if (self._status_cache is not None
                and moment - self._status_checked_at < SUMMARIZER_STATUS_TTL):
            return self._status_cache
        status = self.summarizer_factory().status()
        self._status_cache = status
        self._status_checked_at = moment
        return status

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
    # An SSE response never finishes on its own, and uvicorn's graceful
    # shutdown waits for in-flight requests — so without this the server could
    # not be stopped while a browser tab was connected.
    shutting_down = asyncio.Event()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        shutting_down.set()

    app = FastAPI(title="scribe", docs_url=None, redoc_url=None,
                  openapi_url=None, lifespan=lifespan)

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
                notes_interval=state.auto_notes,
                keep_audio=bool(body.get("keep_audio", False)),
            )
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        summarizer = state.notes_summarizer_factory()
        session = RecordingSession(
            config,
            # Only attach it if it can actually run, so a missing Ollama does
            # not spawn a worker that fails every cycle.
            summarizer=summarizer if summarizer.status().get("available") else None,
        )
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

    # ------------------------------------------------------------- library

    @app.get("/api/transcripts")
    async def transcripts(token: str = Query(""), q: str = Query("")) -> dict:
        require_token(token)
        result = library.scan_detailed(state.raw_dir, q)
        return {**result, "directory": str(state.raw_dir)}

    @app.get("/api/transcripts/{name}")
    async def transcript(name: str, token: str = Query("")) -> dict:
        require_token(token)
        try:
            return library.read(state.raw_dir, name)
        except library.LibraryError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="no such transcript") from exc

    @app.post("/api/transcripts/{name}/title")
    async def rename_transcript(name: str, request: Request) -> dict:
        body = await request.json()
        try:
            result = library.set_title(state.raw_dir, name, body.get("title") or "")
        except library.LibraryError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="no such transcript") from exc

        # Keep an in-progress recording's own title in step.
        session = state.active_session()
        if session and session.writer and session.writer.path \
                and session.writer.path.name == name:
            session.writer.title = result["title"]
        return result

    @app.delete("/api/transcripts/{name}")
    async def remove_transcript(name: str) -> dict:
        # Never delete the file a recording is currently appending to.
        session = state.active_session()
        if session and session.writer and session.writer.path \
                and session.writer.path.name == name:
            raise HTTPException(
                status_code=409,
                detail="that recording is still in progress",
            )
        try:
            return library.delete(state.raw_dir, name)
        except library.LibraryError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="no such transcript") from exc

    # ----------------------------------------------------------- summaries

    @app.get("/api/summarizer")
    async def summarizer_status(token: str = Query("")) -> dict:
        require_token(token)
        # Reaches the model server over the network, so off the event loop.
        return await asyncio.to_thread(state.summarizer_status)

    @app.post("/api/summarize")
    async def summarize_live(request: Request) -> dict:
        """Summarise the recording in progress.

        Only decisions, action items and open questions: a prose paragraph is
        not what anyone reads mid-call.
        """
        session = state.active_session() or state.session
        if session is None or session.writer is None:
            raise HTTPException(status_code=409, detail="nothing has been recorded yet")
        if session.summarizer is None:
            raise HTTPException(
                status_code=503,
                detail=state.summarizer_factory().status().get(
                    "detail", "summarisation is unavailable"
                ),
            )
        try:
            return await asyncio.to_thread(session.refresh_notes_now)
        except SummarizerError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/api/transcripts/{name}/summary")
    async def summarize_transcript(name: str) -> dict:
        try:
            entry = library.read(state.raw_dir, name)
        except library.LibraryError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="no such transcript") from exc

        text = transcript_for_model(entry["turns"])
        try:
            summary = await asyncio.to_thread(
                state.summarizer_factory().summarize, text, "full"
            )
        except SummarizerError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        return library.set_summary(state.raw_dir, name, summary.as_dict())

    @app.get("/api/transcripts/{name}/related")
    async def related_transcripts(name: str, token: str = Query("")) -> dict:
        require_token(token)
        try:
            found = related.find(state.raw_dir, name)
        except library.LibraryError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="no such transcript") from exc
        # Links for the reader to judge, never context fed to a model on its
        # own: see scribe_audio.related.
        return {"related": found}

    @app.get("/api/events")
    async def events(token: str = Query("")) -> StreamingResponse:
        # EventSource cannot set headers, so the token arrives as a query
        # parameter. The URL never leaves this machine.
        require_token(token)
        return StreamingResponse(
            _event_stream(state, shutting_down),
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


async def _event_stream(state: AppState, shutting_down: "asyncio.Event | None" = None):
    """Forward the active session's events to the browser as SSE.

    Resubscribes when a new session starts, so the page does not need to
    reconnect between recordings.
    """
    subscribed: RecordingSession | None = None
    q: queue.Queue | None = None

    yield _sse({"type": "hello"})
    try:
        while True:
            if shutting_down is not None and shutting_down.is_set():
                return
            session = state.session
            if session is not subscribed:
                if subscribed is not None and q is not None:
                    subscribed.bus.unsubscribe(q)
                subscribed, q = session, None
                if session is not None:
                    q = session.bus.subscribe(replay=True)

            if q is None:
                # Nothing to report; keep the connection warm without
                # chattering at it several times a second.
                for _ in range(IDLE_KEEPALIVE_SECONDS * 5):
                    if shutting_down is not None and shutting_down.is_set():
                        return
                    if state.session is not None:
                        break
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


def port_is_free(port: int) -> bool:
    """Whether the UI's port can be bound right now.

    Checked before starting, because a second instance that merely fails to
    bind can linger — several accumulated during development, holding nothing
    and burning CPU — and because "already running" deserves a better message
    than a PortAudio-adjacent traceback.
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((HOST, port))
        except OSError:
            return False
    return True


def serve(raw_dir: str | Path, port: int = DEFAULT_PORT,
          model: str = "small", summary_model: str = DEFAULT_MODEL,
          notes_model: str | None = None, auto_notes: float = 0.0,
          open_browser: bool = True) -> None:
    """Run the local web UI. Binds loopback only."""
    import uvicorn

    if not port_is_free(port):
        print(f"[web] Port {port} is already in use.")
        print(f"       scribe may already be running — try http://{HOST}:{port}/")
        print(f"       Otherwise pick another port with --port.")
        raise SystemExit(1)

    state = AppState(raw_dir=raw_dir, default_model=model,
                     summary_model=summary_model, notes_model=notes_model,
                     auto_notes=auto_notes)
    app = create_app(state)
    url = f"http://{HOST}:{port}/"

    print(f"[web] scribe UI on {url}")
    print(f"[web] Transcripts: {state.raw_dir}")
    print(f"[web] Summaries: {state.summary_model}"
          + (f" (live: {state.notes_model})"
             if state.notes_model != state.summary_model else ""))
    print("[web] Loopback only — not reachable from other machines.")
    print("[web] Ctrl+C to stop.\n")

    if open_browser:
        import webbrowser
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    uvicorn.run(app, host=HOST, port=port, log_level="warning")
