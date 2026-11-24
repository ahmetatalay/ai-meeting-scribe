"""Backend tests.

The metal backend reports different things from the default one, and the
interesting cases are the ones where that difference could quietly change
behaviour: a missing confidence signal, a probability that needs converting,
and timestamps in different units. A fake stands in for whisper.cpp so none of
this needs the optional dependency installed.
"""

import math

import numpy as np
import pytest

from scribe_audio.transcriber import (
    BACKEND_CPU,
    BACKEND_METAL,
    BACKENDS,
    DEFAULT_LOGPROB_THRESHOLD,
    Transcriber,
)


class FakeCppSegment:
    """What pywhispercpp returns: centisecond bounds, one probability."""

    def __init__(self, t0, t1, text, probability=0.9):
        self.t0 = t0
        self.t1 = t1
        self.text = text
        self.probability = probability


class FakeCppModel:
    def __init__(self, segments, record=None):
        self._segments = segments
        self._record = record if record is not None else []

    def transcribe(self, audio, **kwargs):
        self._record.append(kwargs)
        return self._segments


def metal_transcriber(segments, **kwargs):
    """A Transcriber on the metal backend with whisper.cpp faked out."""
    calls = []
    t = Transcriber(backend=BACKEND_METAL, language="en", **kwargs)
    t._model = FakeCppModel(segments, record=calls)
    t._calls = calls
    return t


def speech(seconds=1.0, sample_rate=16000):
    """Loud enough to clear the silence gate."""
    return np.full(int(seconds * sample_rate), 0.3, dtype=np.float32)


class TestBackendSelection:
    def test_cpu_is_the_default(self):
        assert Transcriber().backend == BACKEND_CPU

    def test_an_unknown_backend_is_refused_at_construction(self):
        """Better than failing when the model loads, mid-meeting."""
        with pytest.raises(ValueError, match="unknown backend"):
            Transcriber(backend="cuda")

    def test_both_backends_are_listed(self):
        assert set(BACKENDS) == {BACKEND_CPU, BACKEND_METAL}


class TestMetalNormalisation:
    def test_centisecond_bounds_become_seconds(self):
        """whisper.cpp reports hundredths of a second; 2300 is 23s, not 2300s.
        Getting this wrong would put every segment hours into the recording."""
        t = metal_transcriber([FakeCppSegment(0, 2300, "a full window")])
        segments = t.transcribe_segments(speech())
        assert segments[0]["start"] == pytest.approx(0.0)
        assert segments[0]["end"] == pytest.approx(23.0)

    def test_probability_becomes_a_log_probability(self):
        """The confidence threshold is expressed in log space, so a raw
        probability compared against it would reject nearly everything."""
        t = metal_transcriber([FakeCppSegment(0, 100, "hello", probability=0.8)])
        segments = t.transcribe_segments(speech())
        assert segments[0]["avg_logprob"] == pytest.approx(math.log(0.8))

    def test_a_confident_segment_survives_the_threshold(self):
        """Measured on real speech: probabilities around 0.78-0.90, which is
        -0.25 to -0.10 in log space against a -1.0 threshold."""
        t = metal_transcriber([FakeCppSegment(0, 100, "clear speech",
                                             probability=0.78)])
        assert t.transcribe_segments(speech())
        assert math.log(0.78) > DEFAULT_LOGPROB_THRESHOLD

    def test_an_unconfident_segment_is_still_rejected(self):
        """The filter has to keep working on this backend, or wrong-language
        audio comes back as fluent invented text."""
        t = metal_transcriber([FakeCppSegment(0, 100, "garbage",
                                             probability=0.01)])
        assert t.transcribe_segments(speech()) == []
        assert t.rejected_count == 1

    def test_the_missing_signal_is_none_not_zero(self):
        """whisper.cpp does not report no_speech_prob. Zero would read as
        "certainly speech"; None reads as "unknown", which is the truth."""
        t = metal_transcriber([FakeCppSegment(0, 100, "hello")])
        assert t.transcribe_segments(speech())[0]["no_speech_prob"] is None

    def test_a_missing_no_speech_probability_does_not_reject(self):
        t = metal_transcriber([FakeCppSegment(0, 100, "hello")])
        assert len(t.transcribe_segments(speech())) == 1


class TestMetalRequestsWhatItNeeds:
    def test_probability_extraction_is_asked_for(self):
        """Without extract_probability the field comes back NaN, which would
        silently disable confidence filtering altogether. Measured cost of
        asking: 0.37s against 0.35s."""
        t = metal_transcriber([FakeCppSegment(0, 100, "hello")])
        t.transcribe_segments(speech())
        assert t._calls[0]["extract_probability"] is True

    def test_a_nan_probability_is_treated_as_unknown(self):
        """Defensively: if it ever comes back NaN anyway, the segment must not
        be rejected by a comparison against NaN, nor pass as confident."""
        t = metal_transcriber([
            FakeCppSegment(0, 100, "hello", probability=float("nan"))
        ])
        segments = t.transcribe_segments(speech())
        assert len(segments) == 1
        assert segments[0]["avg_logprob"] == 0.0

    def test_a_zero_probability_does_not_raise(self):
        """log(0) is -inf and log of a negative is a domain error."""
        t = metal_transcriber([FakeCppSegment(0, 100, "hello", probability=0.0)])
        assert len(t.transcribe_segments(speech())) == 1

    def test_the_language_is_passed_through(self):
        t = metal_transcriber([FakeCppSegment(0, 100, "hello")])
        t.transcribe_segments(speech())
        assert t._calls[0]["language"] == "en"

    def test_automatic_detection_is_requested_as_auto(self):
        """whisper.cpp wants the string "auto", not None."""
        t = Transcriber(backend=BACKEND_METAL, language=None)
        calls = []
        t._model = FakeCppModel([FakeCppSegment(0, 100, "hello")], record=calls)
        t.transcribe_segments(speech())
        assert calls[0]["language"] == "auto"


class TestBothBackendsShareTheFiltering:
    """The confidence filter, loop collapsing and silence gate live above the
    backend, so a quality fix cannot land on only one of them."""

    def test_silence_short_circuits_before_the_model(self):
        t = metal_transcriber([FakeCppSegment(0, 100, "should not be reached")])
        assert t.transcribe_segments(np.zeros(16000, dtype=np.float32)) == []
        assert t._calls == [], "the model should not have been called"

    def test_empty_text_is_dropped(self):
        t = metal_transcriber([FakeCppSegment(0, 100, "   ")])
        assert t.transcribe_segments(speech()) == []

    def test_repetition_loops_are_collapsed(self):
        t = metal_transcriber([
            FakeCppSegment(0, 100, "the same thing " * 6 + "and then this")
        ])
        segments = t.transcribe_segments(speech())
        assert segments
        assert t.collapsed_count == 1

    def test_the_metal_backend_still_counts_rejections(self):
        t = metal_transcriber([
            FakeCppSegment(0, 100, "good one", probability=0.9),
            FakeCppSegment(100, 200, "bad one", probability=0.001),
        ])
        segments = t.transcribe_segments(speech())
        assert len(segments) == 1
        assert t.rejected_count == 1


class TestTheDependencyIsOptional:
    def test_the_default_backend_needs_nothing_extra(self):
        """A new user installs the project and it works; the GPU backend is an
        opt-in with its own extra."""
        from pathlib import Path

        import scribe_audio

        pyproject = (Path(scribe_audio.__file__).parent.parent.parent
                     / "pyproject.toml").read_text()
        # pywhispercpp must appear only under an optional extra.
        assert "pywhispercpp" in pyproject
        required = pyproject.split("[project.optional-dependencies]")[0]
        assert "pywhispercpp" not in required

    def test_a_missing_dependency_explains_the_fix(self):
        t = Transcriber(backend=BACKEND_METAL)
        import builtins

        real_import = builtins.__import__

        def no_pywhispercpp(name, *args, **kwargs):
            if name.startswith("pywhispercpp"):
                raise ImportError("No module named 'pywhispercpp'")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = no_pywhispercpp
        try:
            with pytest.raises(RuntimeError, match=r"\[metal\]"):
                t.load()
        finally:
            builtins.__import__ = real_import
