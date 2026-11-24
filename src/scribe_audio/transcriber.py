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

import logging
import math
from dataclasses import dataclass

import numpy as np


def _supported_devices() -> set[str]:
    """Which ctranslate2 backends this build can actually use."""
    import ctranslate2

    found = set()
    for device in ("cpu", "cuda", "mps"):
        try:
            ctranslate2.get_supported_compute_types(device)
        except Exception:
            continue
        found.add(device)
    return found

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


BACKEND_CPU = "cpu"       # ctranslate2, int8 on the CPU
BACKEND_METAL = "metal"   # whisper.cpp, Apple GPU via Metal
BACKENDS = (BACKEND_CPU, BACKEND_METAL)

# whisper.cpp reports segment boundaries in hundredths of a second.
_CENTISECONDS = 100.0


@dataclass
class _Segment:
    """One segment, in the shape the filtering below expects.

    The two backends report different things, so each normalises to this.
    ``no_speech_prob`` is None where a backend does not expose it; _reject()
    skips that check rather than inventing a value.
    """

    start: float
    end: float
    text: str
    avg_logprob: float | None = None
    no_speech_prob: float | None = None


@dataclass
class _Info:
    language: str | None = None
    language_probability: float = 0.0


class Transcriber:
    """Transcribes audio windows using local Whisper.

    Two backends. ``cpu`` runs ctranslate2 with int8 weights and is the
    default: it needs no extra dependency and reports both confidence signals.
    ``metal`` runs whisper.cpp on the GPU: measured 8-11x faster through this
    pipeline on an M5 Pro (`small` 4.71s -> 0.50s, `large-v3` 16.85s -> 2.07s
    on a 30-second window), at the cost of an extra dependency and one of the
    two confidence signals.
    """

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
        backend: str = BACKEND_CPU,
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
        if backend not in BACKENDS:
            raise ValueError(
                f"unknown backend {backend!r}; expected one of {BACKENDS}"
            )
        self.backend = backend
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
        if self.backend == BACKEND_METAL:
            self._load_metal()
            return

        import ctranslate2
        from faster_whisper import WhisperModel

        # ctranslate2 prints a tip on every load suggesting device="mps" as
        # "often 2x+ faster". On this build MPS is not compiled in at all —
        # get_supported_compute_types("mps") raises "unsupported device mps" —
        # so the tip advertises a backend that cannot be selected, once per
        # model load, into the middle of the transcript stream. Errors still
        # come through.
        if "mps" not in _supported_devices():
            ctranslate2.set_log_level(logging.ERROR)

        self._model = WhisperModel(
            self.model_size,
            device="cpu",         # CPU works well on Apple Silicon
            compute_type="int8",  # fast, low memory
        )

    def _load_metal(self) -> None:
        """Load whisper.cpp, which uses the GPU through Metal."""
        import os

        try:
            from pywhispercpp.model import Model
        except ImportError as exc:  # pragma: no cover - depends on install
            raise RuntimeError(
                "The metal backend needs pywhispercpp: "
                "uv tool install --force '.[metal]'"
            ) from exc

        # whisper.cpp writes its own init and per-inference log to stderr,
        # which lands in the middle of the transcript stream.
        self._model = Model(
            self.model_size,
            print_progress=False,
            print_realtime=False,
            redirect_whispercpp_logs_to=os.devnull,
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

        if self.backend == BACKEND_METAL:
            segments, info = self._run_metal(flat)
        else:
            segments, info = self._run_cpu(flat)
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
            logprob = getattr(segment, "avg_logprob", None)
            no_speech = getattr(segment, "no_speech_prob", None)
            kept.append({
                "start": float(segment.start),
                "end": float(segment.end),
                "text": text,
                "avg_logprob": 0.0 if logprob is None else float(logprob),
                # None, not 0.0, where the backend does not report it: zero
                # would read as "certainly speech" rather than "unknown".
                "no_speech_prob": None if no_speech is None else float(no_speech),
            })

        if kept:
            self._maybe_pin_language(info)
        return kept

    def _run_cpu(self, flat: np.ndarray):
        segments, info = self._model.transcribe(
            flat,
            language=self.language,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        return segments, info

    def _run_metal(self, flat: np.ndarray):
        """Run whisper.cpp and normalise what it reports.

        It gives one probability per segment rather than an average log
        probability, and no no-speech probability at all. The probability maps
        onto avg_logprob directly, so the confidence threshold keeps working;
        the missing signal is left as None so _reject() skips that check
        instead of treating absence as certainty.

        extract_probability must be asked for — without it the field comes
        back NaN, which would silently disable confidence filtering. Measured
        cost of asking: 0.37s against 0.35s.
        """
        raw = self._model.transcribe(
            flat,
            language=self.language or "auto",
            extract_probability=True,
        )

        segments = []
        for item in raw:
            probability = float(getattr(item, "probability", float("nan")))
            if probability != probability or probability <= 0:   # NaN or zero
                avg_logprob = None
            else:
                avg_logprob = math.log(min(1.0, probability))
            segments.append(_Segment(
                start=float(item.t0) / _CENTISECONDS,
                end=float(item.t1) / _CENTISECONDS,
                text=item.text or "",
                avg_logprob=avg_logprob,
                no_speech_prob=None,
            ))

        detected = self.language
        probability = 1.0 if self.language else 0.0
        return segments, _Info(language=detected, language_probability=probability)

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
