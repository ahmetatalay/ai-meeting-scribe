"""Local speech-to-text using faster-whisper.

Two things beyond calling the model:

**Confidence filtering.** Whisper never says "I didn't understand that" — it
emits fluent, confident text regardless. Forced into the wrong language it
produces pure invention: a Turkish football broadcast decoded as English came
back as "Saabekte, Sadayi. Stopperde, Davil Sonsanchez". It also hallucinates
stock phrases over silence ("You", "Thank you.", "Thanks for watching"), which
is where the stray one-word chunks in real transcripts came from. The model
reports ``avg_logprob`` and ``no_speech_prob`` per segment and the previous
implementation discarded both; they are applied here.

**Segment-level output.** Speaker attribution now works per segment rather than
per word, so word timestamps are no longer requested. That is faster and stops
transcripts from flipping speaker mid-sentence.
"""

from __future__ import annotations

import numpy as np

DEFAULT_MODEL = "small"

# Below this average log-probability a segment is more likely invention than
# transcription. faster-whisper uses the same value internally as its own
# fallback trigger.
DEFAULT_LOGPROB_THRESHOLD = -1.0

# Above this probability the model itself believes the audio is silence.
DEFAULT_NO_SPEECH_THRESHOLD = 0.6

# Peak amplitude below which a window is not worth sending to the model.
SILENCE_PEAK = 0.01

# Phrases Whisper emits over silence or noise. Only ever dropped when the
# segment's own confidence is already poor, so a genuine "Thank you." in a real
# meeting survives.
FILLER_ARTEFACTS = frozenset({
    "you", "thankyou", "thankyouverymuch", "thanksforwatching",
    "pleasesubscribe", "bye", "byebye", "mmhmm", "uh", "um",
})
FILLER_LOGPROB_THRESHOLD = -0.7


# Whisper gets stuck repeating a phrase on poor audio — "Permission? Permission?
# Permission?" or "I'm happy with you. I'm happy with you. I'm happy with you."
# These score well on avg_logprob, so confidence filtering does not catch them.
# Runs of this many or more identical consecutive phrases are collapsed to one;
# two in a row is left alone, because people really do say "Yes. Yes."
REPEAT_RUN_THRESHOLD = 3
MAX_REPEAT_PHRASE_WORDS = 8


def _normalise(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def collapse_repetitions(
    text: str,
    threshold: int = REPEAT_RUN_THRESHOLD,
    max_phrase: int = MAX_REPEAT_PHRASE_WORDS,
) -> str:
    """Collapse runs of an immediately repeated phrase down to one occurrence.

    Longer phrases are collapsed first so that a repeated sentence is caught as
    a sentence rather than being picked apart word by word.
    """
    words = text.split()
    if len(words) < threshold:
        return text

    for length in range(min(max_phrase, len(words) // threshold), 0, -1):
        out: list[str] = []
        i = 0
        while i < len(words):
            phrase = [_normalise(w) for w in words[i:i + length]]
            if not any(phrase):
                out.append(words[i])
                i += 1
                continue

            repeats = 1
            j = i + length
            while j + length <= len(words):
                if [_normalise(w) for w in words[j:j + length]] != phrase:
                    break
                repeats += 1
                j += length

            if repeats >= threshold:
                out.extend(words[i:i + length])  # keep the first occurrence
                i = j
            else:
                out.append(words[i])
                i += 1
        words = out

    return " ".join(words)


class Transcriber:
    """Transcribes audio windows using faster-whisper (local Whisper)."""

    def __init__(
        self,
        model_size: str = DEFAULT_MODEL,
        language: str | None = "en",
        logprob_threshold: float = DEFAULT_LOGPROB_THRESHOLD,
        no_speech_threshold: float = DEFAULT_NO_SPEECH_THRESHOLD,
        drop_filler: bool = True,
        collapse_loops: bool = True,
        pin_language: bool = True,
        language_confidence: float = 0.6,
    ):
        """
        Args:
            model_size: tiny, base, small, medium or large-v3.
            language: ISO code, or None to let Whisper detect it per window.
            logprob_threshold: drop segments below this average log-probability.
            no_speech_threshold: drop segments the model thinks are silence.
            drop_filler: drop low-confidence stock phrases ("You", "Thank you.").
            collapse_loops: collapse Whisper's repeated-phrase loops.
            pin_language: with language=None, keep the first confident detection
                instead of re-detecting every window.
            language_confidence: detection probability required to pin.
        """
        self.model_size = model_size
        self.requested_language = language
        self.language = language
        self.pin_language = pin_language
        self.language_confidence = language_confidence
        self.logprob_threshold = logprob_threshold
        self.no_speech_threshold = no_speech_threshold
        self.drop_filler = drop_filler
        self.collapse_loops = collapse_loops
        self._model = None
        self.last_detected_language: str | None = None
        self.rejected_count = 0
        self.collapsed_count = 0
        self.language_probability: float | None = None

    def load(self) -> None:
        """Load the model, downloading it on first use."""
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            self.model_size,
            device="cpu",         # CPU works well on Apple Silicon
            compute_type="int8",  # fast, low memory
        )

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def transcribe_segments(self, audio: np.ndarray) -> list[dict]:
        """Transcribe a window into segments.

        Returns:
            ``[{"start", "end", "text", "avg_logprob", "no_speech_prob"}, ...]``,
            empty for silence or when everything was rejected as low confidence.
        """
        if self._model is None:
            self.load()

        flat = np.asarray(audio, dtype=np.float32).flatten()
        if flat.size == 0 or float(np.max(np.abs(flat))) < SILENCE_PEAK:
            return []

        segments, info = self._model.transcribe(
            flat,
            language=self.language,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        self.last_detected_language = getattr(info, "language", None)

        kept = []
        for segment in segments:
            text = (segment.text or "").strip()
            if not text:
                continue
            if self._reject(segment, text):
                self.rejected_count += 1
                continue
            if self.collapse_loops:
                collapsed = collapse_repetitions(text)
                if collapsed != text:
                    self.collapsed_count += 1
                    text = collapsed
            kept.append({
                "start": float(segment.start),
                "end": float(segment.end),
                "text": text,
                "avg_logprob": float(getattr(segment, "avg_logprob", 0.0) or 0.0),
                "no_speech_prob": float(getattr(segment, "no_speech_prob", 0.0) or 0.0),
            })

        if kept:
            self._maybe_pin_language(info)
        return kept

    def _maybe_pin_language(self, info) -> None:
        """Lock in the first confident detection.

        Re-detecting every window lets a short or noisy one misfire, which
        produces a stray foreign-language line in an otherwise English
        transcript. Only called when the window produced real speech.
        """
        if self.requested_language is not None or not self.pin_language:
            return
        if self.language is not None:
            return
        detected = getattr(info, "language", None)
        probability = float(getattr(info, "language_probability", 0.0) or 0.0)
        if detected and probability >= self.language_confidence:
            self.language = detected
            self.language_probability = probability

    def _reject(self, segment, text: str) -> bool:
        logprob = getattr(segment, "avg_logprob", None)
        no_speech = getattr(segment, "no_speech_prob", None)

        if logprob is not None and logprob < self.logprob_threshold:
            return True
        if no_speech is not None and no_speech > self.no_speech_threshold:
            return True
        if (
            self.drop_filler
            and _normalise(text) in FILLER_ARTEFACTS
            and logprob is not None
            and logprob < FILLER_LOGPROB_THRESHOLD
        ):
            return True
        return False

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe a window to a single string (speaker-less mode)."""
        return " ".join(s["text"] for s in self.transcribe_segments(audio))
