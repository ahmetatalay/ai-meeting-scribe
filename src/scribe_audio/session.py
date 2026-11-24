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
from datetime import datetime
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

# Shortfall below which falling behind is not worth reporting. A window is
# emitted on wall-clock time, so a cycle finishing a few hundred milliseconds
# late is normal scheduling jitter and loses no audio worth naming.
DROPPED_AUDIO_FLOOR = 1.0

# How far transcription may fall behind the meeting before saying so. One
# window plus a step: less than that is a single slow cycle, not a trend.
BACKLOG_WARN_SECONDS = 45.0
# Hysteresis, so a backlog hovering at the threshold does not flicker.
BACKLOG_CLEAR_SECONDS = 15.0
# How much the step grows, or shrinks back, in one adjustment.
STEP_WIDEN_FACTOR = 1.5

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
    # 'cpu' (ctranslate2, default) or 'metal' (whisper.cpp on the GPU).
    backend: str = "cpu"
    level_interval: float = 0.5
    # Automatic refresh is off by default. A 14B model costs around 20GB
    # resident and seconds of CPU per pass; refreshing every few seconds kept
    # it loaded permanently and drove a 48GB machine 23GB into swap. The
    # summary is refreshed when asked for instead. Set a positive interval to
    # turn automatic refresh back on.
    notes_interval: float = 0.0
    notes_min_new_words: int = 40
    # Captured audio is spooled to disk so transcription can run behind the
    # meeting without losing anything. Deleting it afterwards is the default:
    # a recording is more sensitive than a transcript of it. Keeping it lets
    # you re-run a better model later instead of losing the meeting.
    keep_audio: bool = False
    # Stopping normally finishes the spooled backlog first. This abandons it
    # instead, for a caller that needs the session to end promptly.
    abandon_backlog: bool = False
    # Widen the step when transcription cannot keep up: less overlap means
    # less work per second of audio, so quality degrades before latency does.
    adaptive_step: bool = True

    def __post_init__(self):
        if self.mic_device is None and self.system_device is None:
            raise ValueError("at least one of mic_device or system_device is required")


@dataclass
class SessionStats:
    windows: int = 0
    segments: int = 0
    words: int = 0
    dropped_seconds: float = 0.0
    backlog_seconds: float = 0.0
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
        summarizer: Any = None,
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

        # Running summary, maintained on a background thread so it survives a
        # browser reload and is part of the recording rather than of a tab.
        self.summarizer = summarizer
        self.notes: str | None = None
        self.notes_model: str | None = None
        self._notes_thread: threading.Thread | None = None
        self._notes_covered_words = 0
        self._notes_error: str | None = None
        self.notes_latency = 0.0
        self._reported_stalls: set[str] = set()
        self._silent_windows = 0
        self._warned_silent = False
        self._warned_backlog = False
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
            backend=cfg.backend,
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
            self._start_notes_worker()
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

    # ------------------------------------------------------- running summary

    def _start_notes_worker(self) -> None:
        if self.summarizer is None or self.config.notes_interval <= 0:
            return
        self._notes_thread = threading.Thread(
            target=self._notes_loop, name="scribe-notes", daemon=True
        )
        self._notes_thread.start()

    def _transcript_words(self) -> list[str]:
        if self.writer is None:
            return []
        return [
            word
            for seg in self.writer.segments
            for word in f"{seg['speaker']}: {seg['text']}".split()
        ]

    def _notes_loop(self) -> None:
        """Refresh the running summary while there is new speech to fold in.

        The wait adapts to how long a pass actually took. A refine call on a
        busy machine can exceed the nominal interval, and firing again
        immediately would queue passes behind each other and fall further
        behind for the rest of the meeting.
        """
        delay = self.config.notes_interval
        while not self._stop_requested.wait(delay):
            started = time.monotonic()
            try:
                self._refresh_notes()
            except Exception as exc:  # never let this kill the recording
                self._report_notes_error(exc)
            elapsed = time.monotonic() - started
            delay = max(self.config.notes_interval, elapsed * 1.5)
            self.notes_latency = elapsed

    def _refresh_notes(self) -> bool:
        words = self._transcript_words()

        # A tail revision can shorten the transcript; re-base rather than
        # treating the shrinkage as new material.
        if len(words) < self._notes_covered_words:
            self._notes_covered_words = len(words)
            return False

        new_words = words[self._notes_covered_words:]
        if len(new_words) < self.config.notes_min_new_words:
            return False

        summary = self.summarizer.refine(self.notes or "", " ".join(new_words))
        self._notes_covered_words = len(words)
        self.notes = summary.text
        self.notes_model = summary.model
        self._notes_error = None

        self.bus.publish({
            "type": "notes",
            "text": summary.text,
            "model": summary.model,
            "covered_words": self._notes_covered_words,
            "latency": round(self.notes_latency, 1),
        })
        return True

    def refresh_notes_now(self) -> dict:
        """Force a refresh, for the UI's Refresh button."""
        if self.summarizer is None:
            raise RuntimeError("no summarizer is configured")
        # Ignore the usual minimum so an explicit request always does something
        # when there is anything at all to fold in.
        words = self._transcript_words()
        if len(words) > self._notes_covered_words:
            original = self.config.notes_min_new_words
            self.config.notes_min_new_words = 1
            try:
                self._refresh_notes()
            finally:
                self.config.notes_min_new_words = original
        return {"text": self.notes or "", "model": self.notes_model,
                "covered_words": self._notes_covered_words}

    def _report_notes_error(self, exc: Exception) -> None:
        message = f"{type(exc).__name__}: {exc}"
        if message == self._notes_error:
            return  # report a persistent failure once, not every cycle
        self._notes_error = message
        self.bus.publish({
            "type": "notice",
            "level": "warning",
            "code": "notes_failed",
            "message": f"Live summary unavailable: {exc}",
        })

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
                self._drain_backlog()
                tail = self.capture.flush()
                if tail is not None:
                    # The last partial window; without this every transcript
                    # ends mid-sentence.
                    self._handle_window(tail, final=True)
                self.capture.stop()
                self._settle_spool()
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

        self._persist_notes()
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

    def _report_backlog(self, behind: float) -> None:
        """Say how far behind transcription is, and withdraw it once caught up.

        This replaces a warning that announced destroyed audio. Nothing is
        destroyed now, so the honest message is a delay — but it still has to
        be said, because a live transcript silently drifting minutes behind the
        conversation would otherwise look like the meeting had gone quiet.
        """
        if behind >= BACKLOG_WARN_SECONDS:
            if not self._warned_backlog:
                self._warned_backlog = True
            self.bus.publish({
                "type": "notice",
                "level": "warning",
                "code": "backlog",
                "seconds": round(behind, 1),
                "message": (
                    f"Transcription is {behind:.0f}s behind the meeting. "
                    f"Nothing is lost — it will finish after you stop. "
                    f"A smaller model would keep up live."
                ),
            })
        elif self._warned_backlog and behind < BACKLOG_CLEAR_SECONDS:
            self._warned_backlog = False
            self.bus.publish({
                "type": "notice",
                "level": "info",
                "code": "backlog",
                "resolved": True,
                "message": "Transcription has caught up.",
            })

    def _adapt_step(self, behind: float) -> None:
        """Trade boundary correction for throughput when the machine lags.

        Each second of audio is transcribed window/step times. At the default
        30s window and 15s step that is exactly twice, and the duplicate pass
        is what pays for correcting the previous window's boundary guess —
        align_tail() only replaces a batch it re-transcribes *in full*, which
        requires window >= 2 * step. So the factor-of-two is the price of the
        correction rather than slack to be tuned away: widening the step past
        half the window does not reduce correction, it ends it.

        That makes it the right thing to give up under pressure. Losing
        boundary fixes costs a few words at window edges; falling minutes
        behind costs the point of a live transcript.
        """
        if not self.config.adaptive_step or self.capture is None:
            return

        step = self.capture.step_duration
        if behind >= BACKLOG_WARN_SECONDS and step < self.config.window:
            widened = min(self.config.window, step * STEP_WIDEN_FACTOR)
            if widened > step + 0.01:
                self.capture.set_step(widened)
                revision_lost = (
                    self._revision_possible(step)
                    and not self._revision_possible(widened)
                )
                detail = (
                    " Window-boundary corrections are off while catching up."
                    if revision_lost else ""
                )
                self.bus.publish({
                    "type": "notice",
                    "level": "info",
                    "code": "step_widened",
                    "step": round(widened, 1),
                    "revision_disabled": revision_lost,
                    "message": (
                        f"Transcribing every {widened:.0f}s instead of "
                        f"{step:.0f}s to catch up.{detail}"
                    ),
                })
        elif behind <= BACKLOG_CLEAR_SECONDS and step > self.config.step:
            restored = max(self.config.step, step / STEP_WIDEN_FACTOR)
            if restored < step - 0.01:
                self.capture.set_step(restored)

    def _revision_possible(self, step: float) -> bool:
        """Whether a window at this step can replace the previous batch.

        align_tail() requires the new window to cover the whole previous
        batch, so the overlap must be at least a full step.
        """
        return self.config.window >= 2 * step - 1e-9

    def _drain_backlog(self) -> None:
        """Transcribe whatever is still spooled before finishing.

        On a machine that cannot keep up this is where the recording catches
        up: the audio was never lost, so stopping must mean "finish the rest",
        not "discard it". A meeting that ended ten minutes behind takes ten
        more minutes of transcription and loses nothing.
        """
        assert self.capture is not None
        behind = self.capture.backlog_seconds
        if behind < self.config.step:
            return

        self.bus.publish({
            "type": "notice",
            "level": "info",
            "code": "draining",
            "seconds": round(behind, 1),
            "message": (
                f"Finishing transcription: {behind:.0f}s of recorded audio "
                f"still to process."
            ),
        })

        while True:
            window = self.capture.flush() if self.capture.is_stopped else \
                self.capture.get_window(timeout=0)
            if window is None:
                break
            self._handle_window(window)
            if self._stop_requested.is_set() and self.config.abandon_backlog:
                break

    def _settle_spool(self) -> None:
        """Keep or delete the captured audio, as configured.

        Deleting is the default: a recording of a meeting is more sensitive
        than a transcript of it, and keeping one unintentionally is the worse
        failure. Keeping it is useful when transcription quality disappointed
        and you would rather re-run a better model than lose the meeting.
        """
        assert self.capture is not None
        if not self.config.keep_audio:
            self.capture.discard_spool()
            return

        if self.writer is None or self.writer.path is None:
            self.capture.discard_spool()
            return
        try:
            kept = self.capture.keep_spool(self.writer.path.with_suffix(".wav"))
            self.bus.publish({
                "type": "notice",
                "level": "info",
                "code": "audio_kept",
                "path": str(kept),
                "message": f"Audio kept at {kept.name}",
            })
        except OSError as exc:
            self.bus.publish({
                "type": "notice",
                "level": "warning",
                "code": "audio_keep_failed",
                "message": f"Could not keep the audio: {exc}",
            })
            self.capture.discard_spool()

    def _persist_notes(self) -> None:
        """Keep the running summary with the recording once it stops."""
        if not self.notes or self.writer is None or self.writer.path is None:
            return
        try:
            from . import library

            if library.get_summary(self.writer.path.parent, self.writer.path.name):
                return  # a full summary already exists; do not overwrite it
            library.set_summary(
                self.writer.path.parent, self.writer.path.name,
                {"text": self.notes, "model": self.notes_model or "unknown",
                 "kind": "live", "passes": 1,
                 "generated": datetime.now().isoformat(timespec="seconds")},
            )
        except Exception:
            # Losing the running summary must never fail the recording.
            pass

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

        # Audio is spooled to disk, so falling behind is a delay rather than a
        # loss and dropped_seconds should now always be zero. It is still
        # reported if it ever is not, because that would mean a real defect.
        if window.dropped_seconds > 0:
            self.stats.dropped_seconds += window.dropped_seconds
        if window.dropped_seconds >= DROPPED_AUDIO_FLOOR:
            lost = window.dropped_seconds
            amount = f"{lost:.0f}s" if lost >= 10 else f"{lost:.1f}s"
            self.bus.publish({
                "type": "notice",
                "level": "warning",
                "code": "dropped_audio",
                "seconds": lost,
                "message": (
                    f"Audio was lost before it could be transcribed: {amount}. "
                    f"This should not happen — please report it."
                ),
            })

        self.stats.backlog_seconds = window.backlog_seconds
        self._report_backlog(window.backlog_seconds)
        self._adapt_step(window.backlog_seconds)

        # The mis-routing signature: speech is being transcribed while the
        # loopback carries nothing, so only one side is reaching us. Requiring
        # transcribed speech avoids firing during a quiet start.
        if self.config.system_device is not None:
            silent = float(np.max(np.abs(window.meeting))) < 1e-4
            if silent and self._prev_words:
                self._silent_windows += 1
                if (self._silent_windows >= SILENT_LOOPBACK_WINDOWS
                        and not self._warned_silent):
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
                # Meeting audio started arriving, so the warning is now false.
                # Leaving it on screen sent people to re-check routing that was
                # already working — withdraw it instead.
                if self._warned_silent:
                    self._warned_silent = False
                    self.bus.publish({
                        "type": "notice",
                        "code": "loopback_silent",
                        "resolved": True,
                        "message": "Meeting audio is arriving.",
                        "level": "info",
                    })

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
