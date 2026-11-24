"""Durable audio spool: the buffer that makes falling behind survivable.

The capture path used to hand transcription a fixed 30-second ring buffer
addressed by wall clock. When transcription could not keep up, the newest
window simply skipped ahead and whatever sat in the middle was overwritten
before anyone read it. Measured on a passively cooled MacBook Air running
large-v3: a single cycle reported 117 seconds of audio gone, twenty-five times
over. Not delayed — gone, with matching holes in the transcript.

A bounded queue that drops is the wrong shape for this. Audio is small: one
second of 16kHz stereo int16 is 64KB, so an hour of meeting is 230MB and any
disk absorbs any backlog a laptop can produce. So capture appends here, and
transcription reads at a cursor that advances independently of the clock. A
machine that manages 0.8x realtime now finishes late rather than destroying
the parts it could not reach in time.

The file is a plain WAV so it can be played, re-transcribed with a better
model later, or recovered after a crash. Left=microphone, right=meeting audio,
the two already aligned by the caller.

Raw meeting audio is more sensitive than a transcript of it, so the default is
to delete the spool once the recording has been written. Keeping it is opt-in.
"""

from __future__ import annotations

import struct
import threading
from pathlib import Path

import numpy as np

CHANNELS = 2
SAMPLE_WIDTH = 2          # int16: 96dB of range, far more than speech needs
HEADER_BYTES = 44
BYTES_PER_FRAME = CHANNELS * SAMPLE_WIDTH
_INT16_PEAK = 32767.0


def _wav_header(sample_rate: int, data_bytes: int) -> bytes:
    """A 44-byte canonical WAV header for ``data_bytes`` of PCM."""
    byte_rate = sample_rate * BYTES_PER_FRAME
    return b"".join([
        b"RIFF",
        struct.pack("<I", 36 + data_bytes),
        b"WAVEfmt ",
        struct.pack("<IHHIIHH", 16, 1, CHANNELS, sample_rate,
                    byte_rate, BYTES_PER_FRAME, SAMPLE_WIDTH * 8),
        b"data",
        struct.pack("<I", data_bytes),
    ])


class AudioSpool:
    """Append-only stereo PCM on disk, readable by frame position.

    Appends and reads are both guarded, so the capture thread can keep writing
    while the transcription thread reads behind it.
    """

    def __init__(self, path: Path | str, sample_rate: int):
        self.path = Path(path)
        self.sample_rate = sample_rate
        self._frames = 0
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Opened for update so the header can be patched in place on close.
        self._fh = open(self.path, "wb+")
        self._fh.write(_wav_header(sample_rate, 0))
        self._fh.flush()
        self._closed = False

    # ---------------------------------------------------------------- writing

    def append(self, mic: np.ndarray, meeting: np.ndarray) -> int:
        """Append one aligned chunk. Returns the new total frame count.

        The two arrays must already cover the same interval; alignment is the
        caller's job because only it knows the two streams' timing.
        """
        mic = np.asarray(mic, dtype=np.float32).reshape(-1)
        meeting = np.asarray(meeting, dtype=np.float32).reshape(-1)
        if mic.size != meeting.size:
            raise ValueError(
                f"channels must be the same length: {mic.size} vs {meeting.size}"
            )
        if mic.size == 0:
            return self._frames

        stereo = np.empty((mic.size, CHANNELS), dtype=np.int16)
        # Clip before scaling: a sum of two channels can exceed unity, and
        # wrapping would turn a loud passage into noise.
        stereo[:, 0] = (np.clip(mic, -1.0, 1.0) * _INT16_PEAK).astype(np.int16)
        stereo[:, 1] = (np.clip(meeting, -1.0, 1.0) * _INT16_PEAK).astype(np.int16)

        with self._lock:
            if self._closed:
                raise ValueError("spool is closed")
            self._fh.seek(HEADER_BYTES + self._frames * BYTES_PER_FRAME)
            self._fh.write(stereo.tobytes())
            self._fh.flush()
            self._frames += mic.size
            return self._frames

    @property
    def frames_written(self) -> int:
        with self._lock:
            return self._frames

    @property
    def seconds_written(self) -> float:
        return self.frames_written / self.sample_rate

    # ---------------------------------------------------------------- reading

    def read(self, start: int, frames: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``frames`` frames from ``start`` as (mic, meeting).

        A request reaching past either end is zero-padded rather than refused,
        so the first window of a recording and the last partial one need no
        special handling from the caller.
        """
        if frames <= 0:
            empty = np.zeros(0, dtype=np.float32)
            return empty, empty.copy()

        mic = np.zeros(frames, dtype=np.float32)
        meeting = np.zeros(frames, dtype=np.float32)

        with self._lock:
            available = self._frames
            first = max(0, start)
            last = min(available, start + frames)
            if last <= first:
                return mic, meeting
            self._fh.seek(HEADER_BYTES + first * BYTES_PER_FRAME)
            raw = self._fh.read((last - first) * BYTES_PER_FRAME)

        stereo = np.frombuffer(raw, dtype=np.int16).reshape(-1, CHANNELS)
        offset = first - start
        scaled = stereo.astype(np.float32) / _INT16_PEAK
        mic[offset:offset + len(stereo)] = scaled[:, 0]
        meeting[offset:offset + len(stereo)] = scaled[:, 1]
        return mic, meeting

    # --------------------------------------------------------------- lifecycle

    def close(self) -> None:
        """Patch the header with the real length and close the file."""
        with self._lock:
            if self._closed:
                return
            data_bytes = self._frames * BYTES_PER_FRAME
            self._fh.seek(0)
            self._fh.write(_wav_header(self.sample_rate, data_bytes))
            self._fh.flush()
            self._fh.close()
            self._closed = True

    @property
    def is_closed(self) -> bool:
        return self._closed

    def delete(self) -> None:
        """Close and remove the file. The default once a transcript is written."""
        self.close()
        self.path.unlink(missing_ok=True)

    def __enter__(self) -> AudioSpool:
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False
