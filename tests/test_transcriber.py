"""Transcription filtering tests.

A fake model stands in for faster-whisper: the logic worth testing is which
segments are kept, not the decoder itself.
"""

from dataclasses import dataclass

import numpy as np
import pytest

from scribe_audio.transcriber import (
    SILENCE_PEAK,
    Transcriber,
    collapse_repetitions,
)


@dataclass
class FakeSegment:
    start: float
    end: float
    text: str
    avg_logprob: float = -0.3
    no_speech_prob: float = 0.05


@dataclass
class FakeInfo:
    language: str = "en"
    language_probability: float = 0.95


class FakeModel:
    def __init__(self, segments, infos=None):
        self.segments = segments
        self.infos = infos
        self.calls = []

    def transcribe(self, audio, **kwargs):
        info = FakeInfo()
        if self.infos:
            info = self.infos[min(len(self.calls), len(self.infos) - 1)]
        self.calls.append(kwargs)
        return iter(self.segments), info


def transcriber_with(segments, infos=None, **kwargs) -> tuple[Transcriber, FakeModel]:
    t = Transcriber(**kwargs)
    model = FakeModel(segments, infos=infos)
    t._model = model
    return t, model


def audible(duration: float = 1.0, sr: int = 16000) -> np.ndarray:
    return np.full(int(sr * duration), 0.3, dtype=np.float32)


class TestConfidenceFiltering:
    def test_confident_segment_is_kept(self):
        t, _ = transcriber_with([FakeSegment(0.0, 2.0, "a real sentence")])
        out = t.transcribe_segments(audible())
        assert len(out) == 1
        assert out[0]["text"] == "a real sentence"
        assert out[0]["avg_logprob"] == pytest.approx(-0.3)

    def test_low_logprob_is_rejected(self):
        """The forced-wrong-language case: fluent but invented text."""
        t, _ = transcriber_with([
            FakeSegment(0.0, 2.0, "Saabekte Sadayi Stopperde", avg_logprob=-1.8),
        ])
        assert t.transcribe_segments(audible()) == []
        assert t.rejected_count == 1

    def test_high_no_speech_probability_is_rejected(self):
        t, _ = transcriber_with([
            FakeSegment(0.0, 2.0, "invented over silence", no_speech_prob=0.95),
        ])
        assert t.transcribe_segments(audible()) == []

    def test_mixed_batch_keeps_only_the_good_segments(self):
        t, _ = transcriber_with([
            FakeSegment(0.0, 1.0, "keep this"),
            FakeSegment(1.0, 2.0, "drop this", avg_logprob=-2.0),
            FakeSegment(2.0, 3.0, "keep this too"),
        ])
        out = t.transcribe_segments(audible())
        assert [s["text"] for s in out] == ["keep this", "keep this too"]
        assert t.rejected_count == 1

    def test_thresholds_are_configurable(self):
        seg = [FakeSegment(0.0, 2.0, "borderline", avg_logprob=-1.5)]
        strict, _ = transcriber_with(seg, logprob_threshold=-1.0)
        assert strict.transcribe_segments(audible()) == []

        lenient, _ = transcriber_with(seg, logprob_threshold=-3.0)
        assert len(lenient.transcribe_segments(audible())) == 1


class TestFillerArtefacts:
    def test_low_confidence_filler_is_dropped(self):
        """Stray one-word chunks like a bare 'You' over silence."""
        t, _ = transcriber_with([FakeSegment(0.0, 0.5, "You", avg_logprob=-0.9)])
        assert t.transcribe_segments(audible()) == []

    def test_confident_thank_you_survives(self):
        """A real 'Thank you.' in a real meeting must not be censored."""
        t, _ = transcriber_with([
            FakeSegment(0.0, 1.0, "Thank you.", avg_logprob=-0.2),
        ])
        out = t.transcribe_segments(audible())
        assert len(out) == 1
        assert out[0]["text"] == "Thank you."

    def test_filler_dropping_can_be_disabled(self):
        t, _ = transcriber_with(
            [FakeSegment(0.0, 0.5, "You", avg_logprob=-0.9)], drop_filler=False
        )
        assert len(t.transcribe_segments(audible())) == 1

    def test_real_content_is_never_treated_as_filler(self):
        t, _ = transcriber_with([
            FakeSegment(0.0, 2.0, "you should check the dashboard", avg_logprob=-0.9),
        ])
        assert len(t.transcribe_segments(audible())) == 1


class TestRepetitionLoops:
    """Whisper loops on poor audio. These score well, so confidence filtering
    does not catch them — the examples are from a real capture."""

    def test_single_word_loop_collapses(self):
        got = collapse_repetitions("Permission? Permission? Permission?")
        assert got == "Permission?"

    def test_sentence_loop_collapses(self):
        text = ("I'm happy with you. I'm happy with you. I'm happy with you.")
        assert collapse_repetitions(text) == "I'm happy with you."

    def test_loop_inside_a_longer_segment(self):
        text = (
            "And I've been in the middle of the day. I've been in the middle of "
            "the day. I've been in the middle of the day. But I'm very happy."
        )
        got = collapse_repetitions(text)
        assert got.count("in the middle of the day") == 1
        assert got.startswith("And")
        assert got.endswith("But I'm very happy.")

    def test_two_repeats_are_left_alone(self):
        """People really do say things twice."""
        assert collapse_repetitions("Yes. Yes.") == "Yes. Yes."
        assert collapse_repetitions("No no") == "No no"

    def test_normal_text_is_untouched(self):
        text = "the cache layer is returning stale keys after the deploy"
        assert collapse_repetitions(text) == text

    def test_repeated_word_in_normal_speech_survives(self):
        text = "it is very very good"
        assert collapse_repetitions(text) == text

    def test_short_text_is_untouched(self):
        assert collapse_repetitions("hello") == "hello"
        assert collapse_repetitions("") == ""

    def test_longest_phrase_wins_over_word_level(self):
        """'a b a b a b' is a repeated pair, not three repeated words."""
        assert collapse_repetitions("a b a b a b") == "a b"

    def test_applied_during_transcription(self):
        t, _ = transcriber_with([
            FakeSegment(0.0, 3.0, "Permission? Permission? Permission?"),
        ])
        out = t.transcribe_segments(audible())
        assert out[0]["text"] == "Permission?"
        assert t.collapsed_count == 1

    def test_can_be_disabled(self):
        t, _ = transcriber_with(
            [FakeSegment(0.0, 3.0, "Permission? Permission? Permission?")],
            collapse_loops=False,
        )
        out = t.transcribe_segments(audible())
        assert out[0]["text"] == "Permission? Permission? Permission?"


class TestSilenceGate:
    def test_quiet_window_never_reaches_the_model(self):
        t, model = transcriber_with([FakeSegment(0.0, 1.0, "should not appear")])
        quiet = np.full(16000, SILENCE_PEAK / 2, dtype=np.float32)
        assert t.transcribe_segments(quiet) == []
        assert model.calls == []

    def test_empty_window(self):
        t, model = transcriber_with([FakeSegment(0.0, 1.0, "x")])
        assert t.transcribe_segments(np.array([], dtype=np.float32)) == []
        assert model.calls == []

    def test_blank_text_is_skipped(self):
        t, _ = transcriber_with([FakeSegment(0.0, 1.0, "   ")])
        assert t.transcribe_segments(audible()) == []


class TestLanguage:
    def test_explicit_language_is_passed_through(self):
        t, model = transcriber_with([FakeSegment(0.0, 1.0, "merhaba")], language="tr")
        t.transcribe_segments(audible())
        assert model.calls[0]["language"] == "tr"

    def test_auto_detection_passes_none_and_records_the_result(self):
        t, model = transcriber_with([FakeSegment(0.0, 1.0, "hello")], language=None)
        t.transcribe_segments(audible())
        assert model.calls[0]["language"] is None
        assert t.last_detected_language == "en"

    def test_confident_detection_is_pinned_for_later_windows(self):
        """A stray 'Willkommen.' in an English transcript comes from
        re-detecting the language on every window."""
        t, model = transcriber_with(
            [FakeSegment(0.0, 1.0, "hello")],
            infos=[FakeInfo("en", 0.95), FakeInfo("de", 0.55)],
            language=None,
        )
        t.transcribe_segments(audible())
        t.transcribe_segments(audible())

        assert model.calls[0]["language"] is None   # first window detects
        assert model.calls[1]["language"] == "en"   # later windows reuse it

    def test_low_confidence_detection_is_not_pinned(self):
        t, model = transcriber_with(
            [FakeSegment(0.0, 1.0, "hello")],
            infos=[FakeInfo("de", 0.30)],
            language=None,
        )
        t.transcribe_segments(audible())
        t.transcribe_segments(audible())
        assert model.calls[1]["language"] is None

    def test_silent_window_does_not_pin(self):
        """Pinning off a window with no speech would lock in a guess."""
        t, model = transcriber_with([], infos=[FakeInfo("de", 0.99)], language=None)
        t.transcribe_segments(audible())
        assert t.language is None

    def test_pinning_can_be_disabled(self):
        t, model = transcriber_with(
            [FakeSegment(0.0, 1.0, "hello")], language=None, pin_language=False
        )
        t.transcribe_segments(audible())
        t.transcribe_segments(audible())
        assert model.calls[1]["language"] is None

    def test_explicit_language_is_never_overridden(self):
        t, model = transcriber_with(
            [FakeSegment(0.0, 1.0, "merhaba")],
            infos=[FakeInfo("en", 0.99)],
            language="tr",
        )
        t.transcribe_segments(audible())
        t.transcribe_segments(audible())
        assert model.calls[1]["language"] == "tr"


class TestPlainMode:
    def test_transcribe_joins_segments(self):
        t, _ = transcriber_with([
            FakeSegment(0.0, 1.0, "first part"),
            FakeSegment(1.0, 2.0, "second part"),
        ])
        assert t.transcribe(audible()) == "first part second part"

    def test_transcribe_returns_empty_for_silence(self):
        t, _ = transcriber_with([FakeSegment(0.0, 1.0, "x")])
        assert t.transcribe(np.zeros(16000, dtype=np.float32)) == ""
