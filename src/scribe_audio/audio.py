"""Audio capture from a specific device using sounddevice with rolling buffer."""

import threading
import numpy as np
import sounddevice as sd


def list_devices() -> list[dict]:
    """List all available audio input devices."""
    devices = sd.query_devices()
    inputs = []
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            inputs.append({"id": i, "name": d["name"], "channels": d["max_input_channels"],
                           "sample_rate": d["default_samplerate"]})
    return inputs


def find_device(name: str) -> int | None:
    """Find device ID by partial name match (case-insensitive)."""
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if name.lower() in d["name"].lower() and d["max_input_channels"] > 0:
            return i
    return None


class AudioCapture:
    """Captures audio into rolling buffers with overlapping window support.

    Keeps separate buffers for mic (you) and meeting (BlackHole) channels,
    plus a mono mix for Whisper transcription.
    """

    def __init__(self, device_id: int, sample_rate: int = 16000, channels: int | None = None,
                 window_duration: float = 30.0, step_duration: float = 15.0,
                 mic_channel: int = 0):
        """
        Args:
            device_id: sounddevice device index
            sample_rate: Hz — Whisper expects 16000
            channels: number of input channels (None = auto-detect from device)
            window_duration: seconds of audio to transcribe at a time
            step_duration: seconds between each transcription (overlap = window - step)
            mic_channel: which channel index is your microphone (default: 0)
        """
        self.device_id = device_id
        self.sample_rate = sample_rate
        if channels is None:
            device_info = sd.query_devices(device_id)
            self.channels = device_info["max_input_channels"]
        else:
            self.channels = channels
        self.window_duration = window_duration
        self.step_duration = step_duration
        self.mic_channel = mic_channel

        buf_size = int(sample_rate * window_duration)
        # Three rolling buffers: mono mix, mic only, meeting only
        self._buf_mono = np.zeros(buf_size, dtype=np.float32)
        self._buf_mic = np.zeros(buf_size, dtype=np.float32)
        self._buf_meeting = np.zeros(buf_size, dtype=np.float32)
        self._buf_lock = threading.Lock()
        self._samples_written = 0

        self._step_samples = int(sample_rate * step_duration)
        self._samples_since_last_step = 0
        self._step_ready = threading.Event()

        self._stream: sd.InputStream | None = None

    def _shift_and_append(self, buf: np.ndarray, data: np.ndarray):
        """Shift buffer left and append new data."""
        n = len(data)
        buf_size = len(buf)
        if n >= buf_size:
            buf[:] = data[-buf_size:]
        else:
            buf[:-n] = buf[n:]
            buf[-n:] = data

    def _callback(self, indata: np.ndarray, frames: int, time_info, status):
        """Called by sounddevice for each audio block."""
        if status:
            print(f"[audio] {status}")

        # Split channels
        mic = indata[:, self.mic_channel].astype(np.float32)

        # Meeting = all other channels averaged
        other_channels = [i for i in range(self.channels) if i != self.mic_channel]
        if other_channels:
            meeting = np.mean(indata[:, other_channels], axis=1).astype(np.float32)
        else:
            meeting = mic  # fallback: single channel device

        # Mono mix of everything for Whisper
        mono = np.mean(indata, axis=1).astype(np.float32)

        with self._buf_lock:
            n = len(mono)
            self._shift_and_append(self._buf_mono, mono)
            self._shift_and_append(self._buf_mic, mic)
            self._shift_and_append(self._buf_meeting, meeting)
            self._samples_written += n
            self._samples_since_last_step += n
            if self._samples_since_last_step >= self._step_samples:
                self._samples_since_last_step = 0
                self._step_ready.set()

    def start(self):
        """Start capturing audio."""
        blocksize = int(self.sample_rate * 0.5)
        self._stream = sd.InputStream(
            device=self.device_id,
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=blocksize,
            callback=self._callback,
        )
        self._stream.start()
        overlap = self.window_duration - self.step_duration
        print(f"[audio] Capturing {self.channels} channels from device {self.device_id} "
              f"@ {self.sample_rate}Hz | window={self.window_duration}s, "
              f"step={self.step_duration}s, overlap={overlap}s")
        print(f"[audio] Mic = channel {self.mic_channel}, "
              f"Meeting = channels {[i for i in range(self.channels) if i != self.mic_channel]}")

    def get_window(self, timeout: float = 30.0) -> dict | None:
        """Wait for a step's worth of new audio, then return all buffers.

        Returns:
            {"mono": ndarray, "mic": ndarray, "meeting": ndarray} or None on timeout
        """
        if not self._step_ready.wait(timeout=timeout):
            return None
        self._step_ready.clear()
        with self._buf_lock:
            return {
                "mono": self._buf_mono.copy(),
                "mic": self._buf_mic.copy(),
                "meeting": self._buf_meeting.copy(),
            }

    def stop(self):
        """Stop capturing."""
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            print("[audio] Stopped")
