"""Recording engine: the headless core that both the CLI and the web UI drive.

A :class:`RecordingSession` owns the capture, transcriber, speaker detector and
writer, runs the capture loop on its own thread, and publishes everything that
happens as events. Nothing in here prints or reads the terminal, so the same
engine serves a CLI that renders events as text and a web server that forwards
them over SSE.

Why a thread rather than the previous inline ``while`` loop: a server cannot
block on the capture loop, model loading takes seconds and needs to be
observable, and ``stop()`` has to be a lifecycle step that flushes the tail
rather than a signal handler flipping a flag.

Segments carry monotonically increasing ids. Nothing depends on that yet, but it
is what lets a live view replace already-shown text when a later window
re-transcribes the same audio more accurately.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .capture import DualStreamCapture, Window
from .dedup import (
    align_tail,
    find_new_index,
    head_segments,
    segment_words,
    slice_segments,
)
from .speakers import SpeakerDetector, label_segments
from .transcriber import Transcriber
from .writer import TranscriptWriter

# States a session moves through. "error" is terminal, like "stopped".
STATE_IDLE = "idle"
STATE_LOADING = "loading_model"
STATE_RECORDING = "recording"
STATE_STOPPING = "stopping"
STATE_STOPPED = "stopped"
STATE_ERROR = "error"

# Consecutive windows of transcribed speech with a silent loopback before
# reporting a routing problem.
SILENT_LOOPBACK_WINDOWS = 2

# Per-subscriber queue depth. A slow consumer loses the oldest events rather
# than stalling the capture loop.
QUEUE_DEPTH = 512


@dataclass
class SessionConfig:
    """Everything needed to start a recording."""

    mic_device: int | None = None
    system_device: int | None = None
    model: str = "small"
    language: str | None = "en"
    window: float = 30.0
    step: float = 15.0
    raw_dir: str | Path = "./raw"
    title: str | None = None
    mic_margin: float = 8.0
    use_speakers: bool = True
    keep_filler: bool = False
    level_interval: float = 0.5

    def __post_init__(self):
        if self.mic_device is None and self.system_device is None:
            raise ValueError("at least one of mic_device or system_device is required")


@dataclass
class SessionStats:
    windows: int = 0
    segments: int = 0
    words: int = 0
    dropped_seconds: float = 0.0
    rejected_segments: int = 0
    collapsed_segments: int = 0
    speakers: dict[str, int] = field(default_factory=dict)


class EventBus:
    """Fan-out of events to any number of subscribers.

    Each subscriber owns a bounded queue. If one stops draining, its oldest
    events are discarded; the publisher is never blocked by a slow reader.
    """

    def __init__(self, depth: int = QUEUE_DEPTH):
        self._depth = depth
        self._lock = threading.Lock()
        self._queues: list[queue.Queue] = []
        self._history: list[dict] = []
        self._history_limit = 1000

    def subscribe(self, replay: bool = True) -> queue.Queue:
        """Register a subscriber. With ``replay``, it receives the backlog first."""
        q: queue.Queue = queue.Queue(maxsize=self._depth)
        with self._lock:
            if replay:
                for event in self._history[-self._depth:]:
                    try:
                        q.put_nowait(event)
                    except queue.Full:
                        break
            self._queues.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._queues:
                self._queues.remove(q)

    def publish(self, event: dict) -> None:
        event = {**event, "at": time.time()}
        with self._lock:
            # Level updates are high-frequency and worthless after the fact.
            if event.get("type") != "levels":
                self._history.append(event)
                if len(self._history) > self._history_limit:
                    del self._history[: len(self._history) - self._history_limit]
            targets = list(self._queues)

        for q in targets:
            try:
                q.put_nowait(event)
            except queue.Full:
                try:
                    q.get_nowait()       # drop the oldest
                    q.put_nowait(event)
                except (queue.Empty, queue.Full):
                    pass

    @property
    def history(self) -> list[dict]:
        with self._lock:
            return list(self._history)

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._queues)


class RecordingSession:
    """Runs one recording from start to stop, publishing events as it goes.

    ``capture_factory``, ``transcriber_factory`` and ``writer_factory`` exist so
    tests can substitute fakes for PortAudio and Whisper.
    """

    def __init__(
        self,
        config: SessionConfig,
        capture_factory: Callable[[SessionConfig], DualStreamCapture] | None = None,
        transcriber_factory: Callable[[SessionConfig], Any] | None = None,
        writer_factory: Callable[[SessionConfig], TranscriptWriter] | None = None,
    ):
        self.config = config
        self.bus = EventBus()
        self.stats = SessionStats()

        self._capture_factory = capture_factory or self._default_capture
        self._transcriber_factory = transcriber_factory or self._default_transcriber
        self._writer_factory = writer_factory or self._default_writer

        self._state = STATE_IDLE
        self._error: str | None = None
        self._state_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_requested = threading.Event()
        self._finished = threading.Event()

        self.capture: DualStreamCapture | None = None
        self.transcriber: Any = None
        self.writer: TranscriptWriter | None = None
        self.detector = SpeakerDetector(margin_db=config.mic_margin)

        self._next_segment_id = 1
        self._prev_words: list[str] = []
        # First id of the batch that may still be revised, so a consumer knows
        # which rendered segments to replace.
        self._revisable_from_id: int | None = None
        # Words of the last written batch, aligned against each new window to
        # decide whether that batch can be replaced wholesale.
        self._batch_words: list[str] = []
        self._reported_stalls: set[str] = set()
        self._silent_windows = 0
        self._warned_silent = False
        self._announced_path = False
        self._announced_language = False

    # --------------------------------------------------------------- factories

    def _default_capture(self, cfg: SessionConfig) -> DualStreamCapture:
        return DualStreamCapture(
            mic_device=cfg.mic_device,
            system_device=cfg.system_device,
            window_duration=cfg.window,
            step_duration=cfg.step,
        )

    def _default_transcriber(self, cfg: SessionConfig) -> Transcriber:
        return Transcriber(
            model_size=cfg.model,
            language=cfg.language,
            drop_filler=not cfg.keep_filler,
        )

    def _default_writer(self, cfg: SessionConfig) -> TranscriptWriter:
        return TranscriptWriter(
            raw_dir=cfg.raw_dir,
            title=cfg.title,
            model=cfg.model,
            language=cfg.language or "auto",
        )

    # ------------------------------------------------------------------- state

    @property
    def state(self) -> str:
        with self._state_lock:
            return self._state

    @property
    def is_running(self) -> bool:
        return self.state in (STATE_LOADING, STATE_RECORDING, STATE_STOPPING)

    def _set_state(self, state: str, **extra) -> None:
        with self._state_lock:
            self._state = state
        self.bus.publish({"type": "state", "value": state, **extra})

    @property
    def use_speakers(self) -> bool:
        cfg = self.config
        return (
            cfg.use_speakers
            and cfg.mic_device is not None
            and cfg.system_device is not None
        )

    # --------------------------------------------------------------- lifecycle

    def start(self) -> None:
        """Begin recording. Returns immediately; progress arrives as events."""
        if self.is_running:
            raise RuntimeError("session already running")
        self._stop_requested.clear()
        self._finished.clear()
        self._thread = threading.Thread(
            target=self._run, name="scribe-session", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 30.0) -> None:
        """Request a stop and wait for the tail to be flushed and written."""
        self._stop_requested.set()
        if self.capture is not None:
            self.capture.stop()  # unblocks a waiting get_window at once
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the session finishes. True if it did."""
        return self._finished.wait(timeout)

    # -------------------------------------------------------------- main loop

    def _run(self) -> None:
        try:
            self._set_state(STATE_LOADING, model=self.config.model)
            self.transcriber = self._transcriber_factory(self.config)
            if hasattr(self.transcriber, "load"):
                self.transcriber.load()
            self.writer = self._writer_factory(self.config)
            self.capture = self._capture_factory(self.config)

            if self._stop_requested.is_set():
                self._set_state(STATE_STOPPED)
                return

            self.capture.start()
            self._set_state(
                STATE_RECORDING,
                speakers=self.use_speakers,
                mic_device=self.config.mic_device,
                system_device=self.config.system_device,
            )
            self._loop()
        except Exception as exc:  # the thread must not die silently
            self._error = f"{type(exc).__name__}: {exc}"
            self.bus.publish({
                "type": "notice",
                "level": "error",
                "code": "session_failed",
                "message": self._error,
            })
        finally:
            self._shutdown()
            self._finished.set()

    def _loop(self) -> None:
        assert self.capture is not None
        next_levels = 0.0

        while not self._stop_requested.is_set():
            window = self.capture.get_window(timeout=self.config.level_interval)

            now = time.monotonic()
            if now >= next_levels:
                self.bus.publish({"type": "levels", **self.capture.levels()})
                next_levels = now + self.config.level_interval

            if window is None:
                if self.capture.is_stopped:
                    break
                continue

            self._handle_window(window)
            self._check_health(window)

    def _shutdown(self) -> None:
        if self._error is None:
            self._set_state(STATE_STOPPING)
        try:
            if self.capture is not None:
                tail = self.capture.flush()
                if tail is not None:
                    # The last partial window; without this every transcript
                    # ends mid-sentence.
                    self._handle_window(tail, final=True)
                self.capture.stop()
        except Exception as exc:
            self.bus.publish({
                "type": "notice",
                "level": "error",
                "code": "flush_failed",
                "message": f"{type(exc).__name__}: {exc}",
            })

        if self.transcriber is not None:
            self.stats.rejected_segments = getattr(self.transcriber, "rejected_count", 0)
            self.stats.collapsed_segments = getattr(
                self.transcriber, "collapsed_count", 0
            )

        self.bus.publish({
            "type": "summary",
            "stats": self.stats_dict(),
            "markdown": str(self.writer.path) if self.writer and self.writer.path else None,
            "sidecar": (
                str(self.writer.sidecar_path)
                if self.writer and self.writer.sidecar_path else None
            ),
        })

        # A failed session must not report itself as a clean finish.
        if self._error is None:
            self._set_state(STATE_STOPPED)
        else:
            self._set_state(STATE_ERROR, error=self._error)

    # ---------------------------------------------------------- window handling

    def _handle_window(self, window: Window, final: bool = False) -> None:
        assert self.transcriber is not None and self.writer is not None
        self.stats.windows += 1

        segments = self.transcriber.transcribe_segments(window.mono)
        self._announce_language()
        if not segments:
            # Silence: previous text is older than this window, so it cannot
            # overlap the next one. Keeping it would let stale text swallow the
            # start of the next utterance, and nothing earlier is revisable.
            self._prev_words = []
            self._batch_words = []
            if self.writer is not None:
                self.writer.seal()
            self._revisable_from_id = None
            return

        if self.use_speakers:
            self.detector.observe(window.mic, window.meeting, window.sample_rate)
            labelled = label_segments(
                segments, window.mic, window.meeting, window.sample_rate,
                self.detector,
            )
        else:
            labelled = [
                {"speaker": "Transcript", "text": s["text"],
                 "start": s["start"], "end": s["end"]}
                for s in segments
            ]
        if not labelled:
            self._prev_words = []
            return

        curr_words = segment_words(labelled)

        # Can this window replace the last batch outright? Only if it
        # re-transcribes that batch in full — see align_tail. When it does, the
        # second reading wins: it had the following context the first pass
        # lacked ("the same purpose" becoming "the same problem").
        covered = (
            align_tail(self._batch_words, curr_words)
            if self.writer.has_revisable_batch else 0
        )

        if covered:
            revision = head_segments(labelled, covered)
            fresh = slice_segments(labelled, covered)
        else:
            # Fall back to appending only what is new relative to the previous
            # window, leaving everything already written untouched.
            revision = []
            fresh = slice_segments(
                labelled, find_new_index(self._prev_words, curr_words)
            )
            self.writer.seal()

        self._prev_words = curr_words

        revised_from = None
        if revision:
            revised_from = self._revisable_from_id
            self._assign_ids(revision)
            if self.writer.revise_last_batch(revision):
                self._recount()
            else:
                revised_from = None

        if fresh:
            self._assign_ids(fresh)
            self.writer.append_segments(fresh)
            self._recount()

        if not revision and not fresh:
            return

        self._announce_path()
        # The batch just written is what the next window may revise.
        last_batch = fresh if fresh else revision
        self._revisable_from_id = last_batch[0]["id"] if last_batch else None
        self._batch_words = segment_words(last_batch)
        if final:
            self.writer.seal()
            self._revisable_from_id = None
            self._batch_words = []

        if revised_from is not None:
            self.bus.publish({
                "type": "revise",
                "from_id": revised_from,
                "segments": [self._wire(s) for s in revision + fresh],
            })
        else:
            self.bus.publish({
                "type": "segments",
                "final": final,
                "segments": [self._wire(s) for s in revision + fresh],
            })

    def _assign_ids(self, segments: list[dict]) -> None:
        for seg in segments:
            seg["id"] = self._next_segment_id
            self._next_segment_id += 1

    def _recount(self) -> None:
        """Recompute totals from what is actually on disk after a revision."""
        written = self.writer.segments if self.writer else []
        self.stats.segments = len(written)
        self.stats.words = sum(len(s["text"].split()) for s in written)
        counts: dict[str, int] = {}
        for seg in written:
            counts[seg["speaker"]] = counts.get(seg["speaker"], 0) + 1
        self.stats.speakers = counts

    @staticmethod
    def _wire(seg: dict) -> dict:
        return {
            "id": seg["id"],
            "speaker": seg["speaker"],
            "text": seg["text"],
            "start": seg.get("start"),
            "end": seg.get("end"),
            # Why this speaker was chosen — surfaced in the UI so an unexpected
            # label can be diagnosed from the transcript itself.
            "reason": seg.get("reason"),
            "mic_db": seg.get("mic_db"),
            "meeting_db": seg.get("meeting_db"),
        }

    def _announce_language(self) -> None:
        """Report the detected language once, when running with language=auto."""
        if self._announced_language or self.config.language is not None:
            return
        detected = getattr(self.transcriber, "language", None)
        if not detected:
            return
        self._announced_language = True
        probability = getattr(self.transcriber, "language_probability", None)
        self.bus.publish({
            "type": "notice",
            "level": "info",
            "code": "language_detected",
            "language": detected,
            "probability": probability,
            "message": (
                f"Language detected: {detected}"
                + (f" ({probability:.0%})" if probability else "")
                + " — pinned for this recording"
            ),
        })

    def _announce_path(self) -> None:
        if self._announced_path or self.writer is None or self.writer.path is None:
            return
        self._announced_path = True
        self.bus.publish({
            "type": "session",
            "title": self.writer.title,
            "markdown": str(self.writer.path),
            "sidecar": str(self.writer.sidecar_path),
        })

    def _check_health(self, window: Window) -> None:
        assert self.capture is not None

        if window.dropped_seconds > 0:
            self.stats.dropped_seconds += window.dropped_seconds
            self.bus.publish({
                "type": "notice",
                "level": "warning",
                "code": "dropped_audio",
                "seconds": window.dropped_seconds,
                "message": (
                    f"Transcription fell behind: {window.dropped_seconds:.0f}s of "
                    f"audio was not transcribed. Try a smaller model."
                ),
            })

        # The mis-routing signature: speech is being transcribed while the
        # loopback carries nothing, so only one side is reaching us. Requiring
        # transcribed speech avoids firing during a quiet start.
        if self.config.system_device is not None and not self._warned_silent:
            silent = float(np.max(np.abs(window.meeting))) < 1e-4
            if silent and self._prev_words:
                self._silent_windows += 1
                if self._silent_windows >= SILENT_LOOPBACK_WINDOWS:
                    self._warned_silent = True
                    self.bus.publish({
                        "type": "notice",
                        "level": "warning",
                        "code": "loopback_silent",
                        "message": (
                            "No meeting audio is arriving. In Zoom/Teams set "
                            "Speaker to 'Same as System' rather than your "
                            "headset, with system output on Meeting Capture."
                        ),
                    })
            elif not silent:
                self._silent_windows = 0

        for stalled in self.capture.stalled_streams:
            if stalled not in self._reported_stalls:
                self._reported_stalls.add(stalled)
                self.bus.publish({
                    "type": "notice",
                    "level": "warning",
                    "code": "stream_stalled",
                    "stream": stalled,
                    "message": (
                        f"The {stalled} stream stopped delivering audio "
                        f"(device disconnected?). Still recording the other side."
                    ),
                })

    # ------------------------------------------------------------------ output

    def stats_dict(self) -> dict:
        return {
            "windows": self.stats.windows,
            "segments": self.stats.segments,
            "words": self.stats.words,
            "dropped_seconds": round(self.stats.dropped_seconds, 1),
            "rejected_segments": self.stats.rejected_segments,
            "collapsed_segments": self.stats.collapsed_segments,
            "speakers": dict(self.stats.speakers),
        }

    def set_title(self, title: str) -> None:
        if self.writer is not None:
            self.writer.set_title(title)
            self.bus.publish({"type": "session", "title": title})
