"""Shared test doubles for PortAudio and Whisper.

Both the engine tests and the end-to-end transcript tests drive a real
``RecordingSession`` with these substituted, so the production code path is
exercised without audio hardware or a model.
"""

from __future__ import annotations

import queue
import threading

import numpy as np

from scribe_audio.capture import Window
from scribe_audio.session import RecordingSession, SessionConfig

SR = 1000
WINDOW_SECONDS = 10.0


def noise_at(n: int, rms: float, seed: int = 0) -> np.ndarray:
    """Gaussian noise scaled to an exact RMS."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n).astype(np.float32)
    if rms <= 0:
        return np.zeros(n, dtype=np.float32)
    x *= rms / float(np.sqrt(np.mean(x**2)))
    return x.astype(np.float32)


def make_window(
    speaker: str = "Meeting",
    seed: int = 0,
    dropped: float = 0.0,
    meeting_silent: bool = False,
) -> Window:
    """A window in which ``speaker`` talks between 1s and 8s.

    Levels model a headset: a mic with a real noise floor well above the old
    hardcoded threshold, and a quiet loopback.
    """
    n = int(SR * WINDOW_SECONDS)
    mic = noise_at(n, 0.003, seed=seed)
    meeting = (
        np.zeros(n, dtype=np.float32)
        if meeting_silent
        else noise_at(n, 0.0001, seed=seed + 50)
    )

    a, b = int(1.0 * SR), int(8.0 * SR)
    if speaker == "You":
        mic[a:b] = noise_at(b - a, 0.08, seed=seed + 1)
    elif not meeting_silent:
        meeting[a:b] = noise_at(b - a, 0.05, seed=seed + 2)

    return Window(
        mono=np.clip(mic + meeting, -1.0, 1.0),
        mic=mic,
        meeting=meeting,
        end_time=float(seed),
        sample_rate=SR,
        dropped_seconds=dropped,
    )


class FakeCapture:
    """Serves a scripted list of windows, then reports itself stopped."""

    def __init__(self, windows, tail=None, stalled=None):
        self._windows = list(windows)
        self._tail = tail
        self._stalled = stalled or []
        self.is_stopped = False
        self.started = False
        self.flushed = False
        self._exhausted = threading.Event()

    def start(self):
        self.started = True

    def get_window(self, timeout=None):
        if self._windows:
            return self._windows.pop(0)
        self._exhausted.set()
        self.is_stopped = True
        return None

    def flush(self):
        self.flushed = True
        tail, self._tail = self._tail, None
        return tail

    def stop(self):
        self.is_stopped = True
        self._exhausted.set()

    def levels(self):
        return {
            "mic": 0.01, "meeting": 0.02,
            "mic_active": True, "meeting_active": True,
            "stalled": self._stalled,
        }

    @property
    def stalled_streams(self):
        return self._stalled


class ScriptedTranscriber:
    """Returns a predetermined transcription per window, as Whisper would.

    Each entry is one window's text, including its re-worded overlap with the
    previous window.
    """

    def __init__(self, script, rejected=0, collapsed=0):
        self.script = list(script)
        self.calls = 0
        self.loaded = False
        self.rejected_count = rejected
        self.collapsed_count = collapsed

    def load(self):
        self.loaded = True

    def transcribe_segments(self, audio):
        if self.calls >= len(self.script):
            return []
        text = self.script[self.calls]
        self.calls += 1
        if not text:
            return []
        return [{
            "start": 1.0, "end": 8.0, "text": text,
            "avg_logprob": -0.3, "no_speech_prob": 0.01,
        }]


def build_session(
    tmp_path,
    script,
    windows,
    tail=None,
    stalled=None,
    rejected=0,
    collapsed=0,
    **cfg_kwargs,
):
    """A RecordingSession wired to fakes. Returns (session, capture, transcriber)."""
    cfg_kwargs.setdefault("mic_device", 1)
    cfg_kwargs.setdefault("system_device", 2)
    cfg_kwargs.setdefault("window", WINDOW_SECONDS)
    cfg_kwargs.setdefault("step", 5.0)
    cfg_kwargs.setdefault("level_interval", 0.01)
    config = SessionConfig(raw_dir=tmp_path, **cfg_kwargs)

    capture = FakeCapture(windows, tail=tail, stalled=stalled)
    transcriber = ScriptedTranscriber(script, rejected=rejected, collapsed=collapsed)
    session = RecordingSession(
        config,
        capture_factory=lambda c: capture,
        transcriber_factory=lambda c: transcriber,
    )
    return session, capture, transcriber


def drain(q: queue.Queue) -> list[dict]:
    """Everything currently queued, without blocking."""
    out = []
    while True:
        try:
            out.append(q.get_nowait())
        except queue.Empty:
            return out


def of_type(events, kind: str) -> list[dict]:
    return [e for e in events if e.get("type") == kind]
