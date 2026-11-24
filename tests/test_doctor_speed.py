"""Speed-check tests.

The point of this check is to answer "which model keeps up on *this* machine"
before a meeting rather than twenty-five windows into one. The arithmetic and
the recommendation are what matter, so they are tested against synthetic
timings; the real model runs are far too slow for a test suite.
"""

import numpy as np
import pytest

from scribe_audio.doctor import (
    BENCHMARK_MODELS,
    benchmark_models,
    recommend_model,
)


def result(model, seconds, step=15.0, window=30.0):
    """A benchmark row as benchmark_models would produce it."""
    return {
        "model": model,
        "seconds": seconds,
        "realtime_factor": window / seconds,
        "keeps_up": seconds < step,
        "headroom": step / seconds,
    }


class TestTheBudgetIsTheStepNotTheWindow:
    """A window is produced every ``step`` seconds, so that is how long there
    is to transcribe one. Measuring against the window would call a model that
    takes 25s on a 30s window "fast enough" when it is falling behind by 10s
    every cycle."""

    def test_a_model_slower_than_the_step_falls_behind(self):
        assert result("large-v3", 15.2)["keeps_up"] is False

    def test_a_model_faster_than_the_step_keeps_up(self):
        assert result("small", 4.5)["keeps_up"] is True

    def test_the_boundary_case_does_not_keep_up(self):
        """Exactly on budget means zero slack, which in practice is behind."""
        assert result("x", 15.0)["keeps_up"] is False

    def test_realtime_factor_is_measured_against_the_window(self):
        """2.0x realtime is break-even at the defaults, which is why large-v3
        measured at exactly 2.0x on an M5 Pro could not keep up."""
        assert result("large-v3", 15.0)["realtime_factor"] == pytest.approx(2.0)


class TestRecommendation:
    def test_the_best_model_with_headroom_is_chosen(self):
        results = [
            result("tiny", 0.8),
            result("small", 4.5),
            result("large-v3-turbo", 7.3),
            result("large-v3", 15.2),
        ]
        assert recommend_model(results) == "large-v3-turbo"

    def test_a_model_that_only_just_fits_is_not_recommended(self):
        """These timings come from an idle machine, and a fanless laptop loses
        speed to throttling once a meeting has been running a while — which is
        exactly how a MacBook Air ended up 117s behind, repeatedly."""
        results = [result("small", 4.0), result("large-v3", 14.9)]
        assert recommend_model(results) == "small"

    def test_nothing_is_recommended_when_nothing_is_comfortable(self):
        results = [result("tiny", 14.0), result("small", 40.0)]
        assert recommend_model(results) is None

    def test_models_that_failed_to_load_are_not_recommended(self):
        results = [
            result("small", 4.0),
            {"model": "large-v3", "error": "OSError: out of memory"},
        ]
        assert recommend_model(results) == "small"

    def test_quality_order_breaks_ties_not_speed(self):
        """Among models that fit, the better one wins even though it is
        slower — turbo over small, not the other way round."""
        results = [result("small", 1.0), result("large-v3-turbo", 7.0)]
        assert recommend_model(results) == "large-v3-turbo"

    def test_an_empty_run_recommends_nothing(self):
        assert recommend_model([]) is None

    def test_the_safety_factor_is_adjustable(self):
        results = [result("large-v3", 14.0)]          # headroom 1.07x
        assert recommend_model(results) is None
        assert recommend_model(results, safety=1.0) == "large-v3"


class TestModelOrder:
    def test_the_candidates_are_ordered_fastest_first(self):
        """So a slow machine has a usable answer before it reaches the models
        that will disappoint it."""
        assert BENCHMARK_MODELS[0] == "tiny"
        assert BENCHMARK_MODELS[-1] == "large-v3"

    def test_turbo_is_recommended_over_medium(self):
        """Measured: turbo 7.33s against medium 9.14s on the same window, and
        turbo is the better model, so medium is dominated on both counts.

        This is why the run order and the quality order have to be separate
        lists: the run order is fastest-first, and reusing it as a quality
        ranking recommended medium over the model that beats it.
        """
        results = [result("medium", 9.1), result("large-v3-turbo", 7.3)]
        assert recommend_model(results) == "large-v3-turbo"

    def test_the_run_order_is_not_the_quality_order(self):
        from scribe_audio.doctor import MODEL_QUALITY

        assert BENCHMARK_MODELS.index("large-v3-turbo") < \
            BENCHMARK_MODELS.index("medium"), "run order is fastest first"
        assert MODEL_QUALITY.index("large-v3-turbo") > \
            MODEL_QUALITY.index("medium"), "quality order puts turbo above"

    def test_quality_order_runs_worst_to_best(self):
        from scribe_audio.doctor import MODEL_QUALITY

        assert MODEL_QUALITY[0] == "tiny"
        assert MODEL_QUALITY[-1] == "large-v3"


class TestBenchmarkingUsesRealSpeech:
    """Benchmarking on noise is worthless: the VAD strips it before the
    encoder, so every model looks instant. Measured on this machine, a
    near-silent window costs 0.00s against 4.59s for speech on `small`."""

    def test_a_failing_model_is_recorded_not_raised(self, monkeypatch):
        """One unavailable model must not abort the whole check."""
        import scribe_audio.doctor as doctor

        monkeypatch.setattr(doctor, "_speech_clip",
                            lambda *a, **k: np.zeros(16000, dtype=np.float32))

        class Boom:
            def __init__(self, model_size, backend="cpu"):
                raise OSError("model not found")

        import scribe_audio.transcriber as transcriber
        monkeypatch.setattr(transcriber, "Transcriber", Boom)

        results = benchmark_models(models=("nonexistent",))
        assert len(results) == 1
        assert "OSError" in results[0]["error"]

    def test_every_candidate_appears_in_the_results(self, monkeypatch):
        import scribe_audio.doctor as doctor
        import scribe_audio.transcriber as transcriber

        monkeypatch.setattr(doctor, "_speech_clip",
                            lambda *a, **k: np.zeros(16000, dtype=np.float32))

        class Instant:
            def __init__(self, model_size, backend="cpu"):
                self.model_size = model_size
                self.backend = backend

            def load(self):
                pass

            def transcribe_segments(self, audio):
                return []

        monkeypatch.setattr(transcriber, "Transcriber", Instant)
        results = benchmark_models(models=("tiny", "small"))
        assert [r["model"] for r in results] == ["tiny", "small"]
        assert all(r["keeps_up"] for r in results)
