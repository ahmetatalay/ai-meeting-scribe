"""Local speech-to-text using faster-whisper."""

import numpy as np
from faster_whisper import WhisperModel


# Model sizes: tiny, base, small, medium, large-v3
# Recommended: "base" for speed, "small" for quality, "medium" for best local quality
DEFAULT_MODEL = "base"


class Transcriber:
    """Transcribes audio chunks using faster-whisper (local Whisper)."""

    def __init__(self, model_size: str = DEFAULT_MODEL, language: str = "en"):
        """
        Args:
            model_size: Whisper model size (tiny/base/small/medium/large-v3)
            language: Language code — set to avoid auto-detection overhead
        """
        self.language = language
        self._model: WhisperModel | None = None
        self._model_size = model_size

    def load(self):
        """Load the Whisper model. Downloads on first use (~150MB for base)."""
        print(f"[transcriber] Loading whisper model '{self._model_size}'...")
        self._model = WhisperModel(
            self._model_size,
            device="cpu",         # cpu works well on Apple Silicon
            compute_type="int8",  # fast + low memory
        )
        print(f"[transcriber] Model loaded")

    def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe an audio chunk to text.

        Args:
            audio: numpy array of float32 audio samples (16kHz mono)

        Returns:
            Transcribed text string (may be empty if silence)
        """
        segments = self.transcribe_segments(audio)
        return " ".join(s["text"] for s in segments)

    def transcribe_segments(self, audio: np.ndarray) -> list[dict]:
        """
        Transcribe audio and return segments with timestamps.

        Args:
            audio: numpy array of float32 audio samples (16kHz mono)

        Returns:
            List of segments: [{"start": float, "end": float, "text": str}, ...]
        """
        if self._model is None:
            self.load()

        audio_flat = audio.flatten().astype(np.float32)

        # Skip silence
        if np.max(np.abs(audio_flat)) < 0.01:
            return []

        segments, info = self._model.transcribe(
            audio_flat,
            language=self.language,
            beam_size=5,
            word_timestamps=True,   # enable per-word timing
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
            ),
        )

        result = []
        for segment in segments:
            if not segment.words:
                # Fallback: no word timestamps
                text = segment.text.strip()
                if text:
                    result.append({
                        "start": segment.start,
                        "end": segment.end,
                        "text": text,
                    })
                continue

            # Return individual words with timestamps
            for word in segment.words:
                text = word.word.strip()
                if text:
                    result.append({
                        "start": word.start,
                        "end": word.end,
                        "text": text,
                    })

        return result
