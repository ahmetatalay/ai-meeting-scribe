"""Two-stream audio capture with wall-clock alignment.

The previous design relied on a macOS Aggregate Device that bundled the
microphone and BlackHole into one multi-channel input. That is fragile in ways
that corrupt recordings silently:

* When the headset is unplugged its slot becomes "Offline Device" and the
  aggregate's channel map shifts, so ``--mic-channel 0`` starts pointing at a
  different signal.
* It needs one aggregate per physical headset.
* It couples "can I capture meeting audio" to "is my headset connected".

Instead, two independent input streams are opened — one for the microphone, one
for the system-audio loopback (BlackHole) — and aligned on the wall clock.
Each block is timestamped as it arrives, so the two streams are placed on a
shared timeline rather than assumed to be sample-aligned. That removes clock
drift as an accumulating error: drift shows up as at most one block of jitter
instead of growing without bound over an hour-long meeting.

Either stream may be absent or may die mid-recording; the other keeps working.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

# Capture blocks are kept short: it bounds the alignment error between the two
# streams, and gives the level meters a responsive update rate.
BLOCK_SECONDS = 0.1

SAMPLE_RATE = 16000  # what Whisper expects

# A stream that has delivered nothing for this long is treated as stalled
# rather than merely slow — the unplugged-headset case.
STALE_AFTER_SECONDS = 2.0


@dataclass(frozen=True)
class Device:
    """An audio device as reported by the host API."""

    id: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float

    @property
    def is_input(self) -> bool:
        return self.max_input_channels > 0

    @property
    def is_output(self) -> bool:
        return self.max_output_channels > 0


def _sd():
    """Import sounddevice lazily so the pure logic stays importable without PortAudio."""
    import sounddevice as sd

    return sd


def _to_device(index: int, raw: dict) -> Device:
    return Device(
        id=index,
        name=raw["name"],
        max_input_channels=raw["max_input_channels"],
        max_output_channels=raw["max_output_channels"],
        default_samplerate=raw["default_samplerate"],
    )


def list_devices(kind: str = "input") -> list[Device]:
    """List input or output devices. ``kind`` is "input", "output" or "all"."""
    if kind not in ("input", "output", "all"):
        raise ValueError(f"kind must be input, output or all, got {kind!r}")
    sd = _sd()
    out = []
    for i, raw in enumerate(sd.query_devices()):
        dev = _to_device(i, raw)
        if kind == "input" and not dev.is_input:
            continue
        if kind == "output" and not dev.is_output:
            continue
        out.append(dev)
    return out


def find_device(name: str, kind: str = "input") -> Device | None:
    """Find a device by partial, case-insensitive name match."""
    needle = name.lower()
    for dev in list_devices(kind):
        if needle in dev.name.lower():
            return dev
    return None


def get_system_audio() -> dict:
    """Current default input and output devices, or None for each if unset."""
    sd = _sd()
    raw_devices = sd.query_devices()
    default_in, default_out = sd.default.device
    result: dict[str, Device | None] = {"input": None, "output": None}
    if default_in is not None and default_in >= 0:
        result["input"] = _to_device(default_in, raw_devices[default_in])
    if default_out is not None and default_out >= 0:
        result["output"] = _to_device(default_out, raw_devices[default_out])
    return result


class TimedRingBuffer:
    """Mono ring buffer whose contents are addressable by wall-clock time.

    Each write records the time of the block's final sample, so a reader can ask
    for "the 30 seconds ending at time T" and get back samples that actually
    correspond to that interval, zero-padded where no audio was captured.
    """

    def __init__(self, sample_rate: int, duration: float):
        self.sample_rate = sample_rate
        self.duration = duration
        self._buf = np.zeros(int(sample_rate * duration), dtype=np.float32)
        self._end_time: float | None = None
        self._samples_written = 0

    @property
    def end_time(self) -> float | None:
        """Wall-clock time of the most recently written sample, or None if empty."""
        return self._end_time

    @property
    def samples_written(self) -> int:
        return self._samples_written

    def write(self, block: np.ndarray, end_time: float) -> None:
        """Append a mono block whose last sample was captured at ``end_time``."""
        block = np.asarray(block, dtype=np.float32).reshape(-1)
        n = block.size
        if n == 0:
            return
        size = self._buf.size
        if n >= size:
            self._buf[:] = block[-size:]
        else:
            self._buf[:-n] = self._buf[n:]
            self._buf[-n:] = block
        self._end_time = end_time
        self._samples_written += n

    def read_window(self, end_time: float, duration: float) -> np.ndarray:
        """Return ``duration`` seconds of audio ending at ``end_time``.

        Regions outside what the buffer holds come back as zeros, so a caller
        never has to special-case start-up or a stream that stalled.
        """
        n = int(self.sample_rate * duration)
        out = np.zeros(n, dtype=np.float32)
        if self._end_time is None:
            return out

        # Samples between the requested end and the newest sample we hold.
        lag = int(round((self._end_time - end_time) * self.sample_rate))
        stop = self._buf.size - lag  # exclusive index of the window's end
        start = stop - n

        src_start = max(0, start)
        src_stop = min(self._buf.size, stop)
        if src_stop <= src_start:
            return out

        dst_start = src_start - start
        out[dst_start:dst_start + (src_stop - src_start)] = self._buf[src_start:src_stop]
        return out

    def recent_rms(self, duration: float = 0.5) -> float:
        """RMS of the most recent ``duration`` seconds — for level meters."""
        if self._end_time is None:
            return 0.0
        n = min(self._buf.size, max(1, int(self.sample_rate * duration)))
        return float(np.sqrt(np.mean(self._buf[-n:] ** 2)))


@dataclass
class Window:
    """One aligned analysis window."""

    mono: np.ndarray       # mic + system, for transcription
    mic: np.ndarray        # microphone only, for speaker attribution
    meeting: np.ndarray    # system audio only
    end_time: float        # wall clock of the last sample
    sample_rate: int
    dropped_seconds: float = 0.0  # audio never analysed because we fell behind

    @property
    def duration(self) -> float:
        return len(self.mono) / self.sample_rate


class DualStreamCapture:
    """Captures microphone and system audio as two independent streams.

    ``stream_factory`` and ``time_fn`` exist so tests can drive the whole
    alignment path with synthetic blocks and a controlled clock, without
    PortAudio or real devices.
    """

    def __init__(
        self,
        mic_device: int | None,
        system_device: int | None,
        sample_rate: int = SAMPLE_RATE,
        window_duration: float = 30.0,
        step_duration: float = 15.0,
        stream_factory=None,
        time_fn=time.monotonic,
    ):
        if window_duration <= 0 or step_duration <= 0:
            raise ValueError("window and step durations must be positive")
        if step_duration > window_duration:
            raise ValueError("step must not exceed window")
        if mic_device is None and system_device is None:
            raise ValueError("at least one of mic_device or system_device is required")

        self.mic_device = mic_device
        self.system_device = system_device
        self.sample_rate = sample_rate
        self.window_duration = window_duration
        self.step_duration = step_duration
        self._time = time_fn
        self._stream_factory = stream_factory or self._open_real_stream

        self.mic_buffer = TimedRingBuffer(sample_rate, window_duration)
        self.system_buffer = TimedRingBuffer(sample_rate, window_duration)

        self._lock = threading.Lock()
        self._ready = threading.Condition(self._lock)
        self._last_emit_time: float | None = None
        self._latest_time: float | None = None
        self._first_time: float | None = None
        self._streams: list = []
        self._stopped = False
        self._errors: list[str] = []
        self._start_time: float | None = None

    # ------------------------------------------------------------------ feeds

    def feed_mic(self, block: np.ndarray, now: float | None = None) -> None:
        """Submit a microphone block. Public because it is also the test seam."""
        self._feed(self.mic_buffer, block, now)

    def feed_system(self, block: np.ndarray, now: float | None = None) -> None:
        """Submit a system-audio block. Public because it is also the test seam."""
        self._feed(self.system_buffer, block, now)

    def _feed(self, buffer: TimedRingBuffer, block: np.ndarray, now: float | None) -> None:
        block = np.asarray(block, dtype=np.float32)
        if block.ndim == 2:  # (frames, channels) -> mono
            block = block.mean(axis=1)
        if now is None:
            now = self._time()

        with self._lock:
            buffer.write(block, now)
            if self._first_time is None:
                # Time of the first *sample*, not of the first block's arrival,
                # so the opening window is measured from when audio began.
                self._first_time = now - block.size / self.sample_rate
            if self._latest_time is None or now > self._latest_time:
                self._latest_time = now
            if self._window_ready_locked():
                self._ready.notify_all()

    # ------------------------------------------------------------- alignment

    def _live_buffers_locked(self) -> list[tuple[str, TimedRingBuffer]]:
        return [
            (name, buf)
            for name, buf, dev in (
                ("mic", self.mic_buffer, self.mic_device),
                ("meeting", self.system_buffer, self.system_device),
            )
            if dev is not None and buf.end_time is not None
        ]

    def _stale_locked(self) -> list[str]:
        """Streams that have stopped delivering audio.

        This is the unplugged-headset case: the mic stream simply stops firing.
        Such a stream must be excluded from alignment, or the minimum below
        would freeze and no further windows would ever be produced.
        """
        if self._latest_time is None:
            return []
        return [
            name
            for name, buf in self._live_buffers_locked()
            if self._latest_time - buf.end_time > STALE_AFTER_SECONDS
        ]

    def _aligned_end_locked(self) -> float | None:
        """Newest time for which every responsive stream has data.

        Using the minimum means neither stream is ever read ahead of what it has
        actually captured, which is what keeps the two on one timeline.
        """
        stale = set(self._stale_locked())
        ends = [
            buf.end_time
            for name, buf in self._live_buffers_locked()
            if name not in stale
        ]
        if ends:
            return min(ends)
        return self._latest_time

    def _window_ready_locked(self) -> bool:
        end = self._aligned_end_locked()
        if end is None:
            return False
        # Before the first window, measure from when audio started arriving, so
        # the opening window is not mostly zero padding.
        reference = self._last_emit_time if self._last_emit_time is not None else self._first_time
        if reference is None:
            return False
        return end - reference >= self.step_duration

    def get_window(self, timeout: float | None = None) -> Window | None:
        """Block until a step of new audio is available, then return it aligned.

        Returns None on timeout or after :meth:`stop`.
        """
        deadline = None if timeout is None else self._time() + timeout
        with self._ready:
            while not self._stopped and not self._window_ready_locked():
                remaining = None if deadline is None else deadline - self._time()
                if remaining is not None and remaining <= 0:
                    return None
                self._ready.wait(remaining if remaining is not None else 0.25)
            if self._stopped and not self._window_ready_locked():
                return None

            end = self._aligned_end_locked()
            assert end is not None  # guaranteed by _window_ready_locked
            return self._build_window_locked(end)

    def flush(self) -> Window | None:
        """Return whatever audio has arrived since the last window.

        Called on shutdown so the tail of the meeting is transcribed instead of
        being discarded, which is why recordings used to end mid-sentence.
        """
        with self._lock:
            end = self._aligned_end_locked()
            if end is None:
                return None
            if self._last_emit_time is not None and end <= self._last_emit_time:
                return None
            return self._build_window_locked(end)

    def _build_window_locked(self, end: float) -> Window:
        mic = self.mic_buffer.read_window(end, self.window_duration)
        meeting = self.system_buffer.read_window(end, self.window_duration)
        # Sum rather than average: averaging halves a lone speaker's level and
        # can push a quiet talker under the downstream silence gate.
        mono = np.clip(mic + meeting, -1.0, 1.0)

        dropped = 0.0
        if self._last_emit_time is not None:
            advanced = end - self._last_emit_time
            if advanced > self.window_duration:
                # More time passed than the buffer holds, so some audio was
                # overwritten before it was ever analysed.
                dropped = advanced - self.window_duration
        self._last_emit_time = end

        return Window(
            mono=mono,
            mic=mic,
            meeting=meeting,
            end_time=end,
            sample_rate=self.sample_rate,
            dropped_seconds=dropped,
        )

    # ------------------------------------------------------------------ meters

    def levels(self) -> dict:
        """Recent RMS per stream, for the UI level meters and routing checks."""
        with self._lock:
            return {
                "mic": self.mic_buffer.recent_rms(),
                "meeting": self.system_buffer.recent_rms(),
                "mic_active": self.mic_device is not None,
                "meeting_active": self.system_device is not None,
                "stalled": self._stale_locked(),
            }

    @property
    def is_stopped(self) -> bool:
        return self._stopped

    @property
    def stalled_streams(self) -> list[str]:
        """Names of streams that have stopped delivering audio."""
        with self._lock:
            return self._stale_locked()

    @property
    def errors(self) -> list[str]:
        with self._lock:
            return list(self._errors)

    # ----------------------------------------------------------- lifecycle

    def _open_real_stream(self, device: int, channels: int, callback):
        sd = _sd()
        stream = sd.InputStream(
            device=device,
            samplerate=self.sample_rate,
            channels=channels,
            dtype="float32",
            blocksize=int(self.sample_rate * BLOCK_SECONDS),
            callback=callback,
        )
        stream.start()
        return stream

    def _channels_for(self, device: int) -> int:
        try:
            sd = _sd()
            info = sd.query_devices(device)
            return max(1, min(2, info["max_input_channels"]))
        except Exception:
            return 1

    def start(self) -> None:
        self._start_time = self._time()
        self._stopped = False

        if self.mic_device is not None:
            self._streams.append(
                self._stream_factory(
                    self.mic_device,
                    self._channels_for(self.mic_device),
                    self._make_callback(self.feed_mic, "mic"),
                )
            )
        if self.system_device is not None:
            self._streams.append(
                self._stream_factory(
                    self.system_device,
                    self._channels_for(self.system_device),
                    self._make_callback(self.feed_system, "system"),
                )
            )

    def _make_callback(self, sink, label: str):
        def callback(indata, frames, time_info, status):
            if status:
                with self._lock:
                    self._errors.append(f"{label}: {status}")
            sink(np.asarray(indata, dtype=np.float32))

        return callback

    def stop(self) -> None:
        with self._ready:
            self._stopped = True
            self._ready.notify_all()
        for stream in self._streams:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
        self._streams.clear()
