"""Speaker-attribution tests.

The scenarios mirror measured failures. In a real 36-minute capture, 97 of 97
chunks were attributed to "You" — including the whole other side of the
conversation — so "a noisy microphone must not swallow the meeting" is the
primary regression test here.
"""

import numpy as np
import pytest

from scribe_audio.speakers import (
    SPEAKER_MEETING,
    SPEAKER_YOU,
    SpeakerDetector,
    label_segments,
    merge_runs,
)

SR = 1000


def noise_at(n: int, rms: float, seed: int = 0) -> np.ndarray:
    """Gaussian noise scaled to an exact RMS."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n).astype(np.float32)
    if rms <= 0:
        return np.zeros(n, dtype=np.float32)
    x *= rms / float(np.sqrt(np.mean(x**2)))
    return x.astype(np.float32)


def channel(
    duration: float,
    base_rms: float,
    loud: tuple[float, float, float] | None = None,
    seed: int = 0,
) -> np.ndarray:
    """A channel of baseline noise, optionally with a louder region.

    ``loud`` is ``(start_sec, end_sec, rms)``.
    """
    n = int(SR * duration)
    audio = noise_at(n, base_rms, seed=seed)
    if loud is not None:
        start, end, rms = loud
        a, b = int(start * SR), int(end * SR)
        audio[a:b] = noise_at(b - a, rms, seed=seed + 1)
    return audio


def calibrated(mic, meeting, **kwargs) -> SpeakerDetector:
    det = SpeakerDetector(**kwargs)
    det.observe(mic, meeting, SR)
    return det


class TestNoisyMicrophoneRegression:
    """The 97-of-97 bug: an idle mic must not win every decision."""

    def test_idle_but_noisy_mic_does_not_claim_meeting_speech(self):
        # Mic floor at 0.003 RMS — three times the old hardcoded 0.001
        # threshold, but it is just room noise, not speech.
        mic = channel(10.0, base_rms=0.003, seed=1)
        meeting = channel(10.0, base_rms=0.0001, loud=(2.0, 4.0, 0.05), seed=2)

        det = calibrated(mic, meeting)
        decision = det.classify(mic, meeting, 2.0, 4.0, SR)

        assert decision.speaker == SPEAKER_MEETING

    def test_old_absolute_threshold_would_have_been_fooled(self):
        """Documents precisely why the previous rule failed."""
        mic = channel(10.0, base_rms=0.003, seed=1)
        # The retired logic: mic_energy > 0.001 and mic_energy > meeting_energy
        mic_rms = float(np.sqrt(np.mean(mic[2000:4000] ** 2)))
        assert mic_rms > 0.001  # "mic is active" — but it is only room noise

    def test_your_speech_through_the_headset_is_yours(self):
        mic = channel(10.0, base_rms=0.003, loud=(2.0, 4.0, 0.08), seed=1)
        meeting = channel(10.0, base_rms=0.0001, seed=2)

        det = calibrated(mic, meeting)
        assert det.classify(mic, meeting, 2.0, 4.0, SR).speaker == SPEAKER_YOU


class TestSpeakerBleed:
    def test_laptop_speaker_bleed_into_the_mic_stays_with_the_meeting(self):
        """The inverse failure: built-in mic hears the laptop speakers.

        The bleed is attenuated, so its prominence above the mic's own floor
        stays well below the loopback's.
        """
        mic = channel(10.0, base_rms=0.003, loud=(2.0, 4.0, 0.012), seed=1)
        meeting = channel(10.0, base_rms=0.0001, loud=(2.0, 4.0, 0.05), seed=2)

        det = calibrated(mic, meeting)
        decision = det.classify(mic, meeting, 2.0, 4.0, SR)

        assert decision.speaker == SPEAKER_MEETING

    def test_crosstalk_picks_the_more_prominent_channel(self):
        # Matched floors, so prominence is directly comparable.
        mic = channel(10.0, base_rms=0.001, loud=(2.0, 4.0, 0.10), seed=1)
        meeting = channel(10.0, base_rms=0.001, loud=(2.0, 4.0, 0.02), seed=2)

        det = calibrated(mic, meeting)
        assert det.classify(mic, meeting, 2.0, 4.0, SR).speaker == SPEAKER_YOU


class TestUnroutedLoopback:
    def test_silent_meeting_channel_is_never_credited(self):
        """BlackHole not routed: the channel is digital silence."""
        mic = channel(10.0, base_rms=0.003, loud=(2.0, 4.0, 0.08), seed=1)
        meeting = np.zeros(int(SR * 10), dtype=np.float32)

        det = calibrated(mic, meeting)
        decision = det.classify(mic, meeting, 2.0, 4.0, SR)

        assert decision.speaker == SPEAKER_YOU
        assert decision.meeting_rms == 0.0

    def test_both_channels_silent_holds_rather_than_flipping(self):
        mic = np.zeros(int(SR * 10), dtype=np.float32)
        meeting = np.zeros(int(SR * 10), dtype=np.float32)

        det = calibrated(mic, meeting)
        det._last_speaker = SPEAKER_YOU
        decision = det.classify(mic, meeting, 2.0, 4.0, SR)

        assert decision.speaker == SPEAKER_YOU
        assert "held" in decision.reason


class TestCalibration:
    def test_floor_tracks_the_quiet_baseline_not_the_speech(self):
        mic = channel(10.0, base_rms=0.002, loud=(1.0, 9.0, 0.2), seed=3)
        meeting = channel(10.0, base_rms=0.002, seed=4)

        det = calibrated(mic, meeting)
        # 80% of the window is loud speech, yet the floor must stay near the
        # quiet baseline or every later decision is skewed.
        assert det.mic_floor == pytest.approx(0.002, rel=0.6)

    def test_floor_adapts_across_windows(self):
        quiet = channel(10.0, base_rms=0.0005, seed=5)
        loud_floor = channel(10.0, base_rms=0.02, seed=6)

        det = SpeakerDetector()
        det.observe(quiet, quiet, SR)
        first = det.mic_floor
        for _ in range(20):
            det.observe(loud_floor, loud_floor, SR)

        assert det.mic_floor > first * 5

    def test_uncalibrated_detector_still_decides(self):
        """classify() before any observe() must not raise."""
        mic = channel(10.0, base_rms=0.003, loud=(2.0, 4.0, 0.08), seed=1)
        meeting = channel(10.0, base_rms=0.0001, seed=2)
        det = SpeakerDetector()
        assert det.classify(mic, meeting, 2.0, 4.0, SR).speaker in (
            SPEAKER_YOU, SPEAKER_MEETING,
        )


class TestSegmentLabelling:
    def test_alternating_speakers_are_attributed_separately(self):
        mic = channel(10.0, base_rms=0.002, loud=(1.0, 3.0, 0.08), seed=1)
        meeting = channel(10.0, base_rms=0.0002, loud=(4.0, 6.0, 0.05), seed=2)

        det = calibrated(mic, meeting)
        segments = [
            {"start": 1.0, "end": 3.0, "text": "my question"},
            {"start": 4.0, "end": 6.0, "text": "their answer"},
        ]
        out = label_segments(segments, mic, meeting, SR, det)

        assert [s["speaker"] for s in out] == [SPEAKER_YOU, SPEAKER_MEETING]
        assert out[0]["text"] == "my question"
        assert out[1]["text"] == "their answer"

    def test_one_decision_per_segment_so_sentences_do_not_split(self):
        """Transcripts used to flip speaker mid-sentence; they cannot now."""
        mic = channel(10.0, base_rms=0.002, loud=(1.0, 5.0, 0.08), seed=1)
        meeting = channel(10.0, base_rms=0.0002, seed=2)

        det = calibrated(mic, meeting)
        segments = [{"start": 1.0, "end": 5.0, "text": "a long single sentence"}]
        out = label_segments(segments, mic, meeting, SR, det)

        assert len(out) == 1
        assert out[0]["text"] == "a long single sentence"

    def test_blank_segments_are_dropped(self):
        mic = channel(5.0, base_rms=0.002, seed=1)
        meeting = channel(5.0, base_rms=0.002, seed=2)
        det = calibrated(mic, meeting)
        out = label_segments(
            [{"start": 0.0, "end": 1.0, "text": "   "}], mic, meeting, SR, det
        )
        assert out == []

    def test_empty_input(self):
        mic = channel(5.0, base_rms=0.002, seed=1)
        det = calibrated(mic, mic)
        assert label_segments([], mic, mic, SR, det) == []

    def test_segment_beyond_the_buffer_is_clamped(self):
        mic = channel(2.0, base_rms=0.002, seed=1)
        meeting = channel(2.0, base_rms=0.002, seed=2)
        det = calibrated(mic, meeting)
        # Whisper occasionally reports a timestamp past the window end.
        decision = det.classify(mic, meeting, 1.9, 99.0, SR)
        assert decision.speaker in (SPEAKER_YOU, SPEAKER_MEETING)


class TestSimulatedConversation:
    """End-to-end attribution over many windows — the 97-of-97 shape.

    A two-sided conversation on a headset: the mic carries a steady noise floor
    plus your speech, the loopback carries theirs. Both sides must be
    recognised across a long run with the floors adapting as they go.
    """

    def _run(self, turns: list[str], margin_db: float = 8.0) -> list[str]:
        det = SpeakerDetector(margin_db=margin_db)
        got = []
        for i, who in enumerate(turns):
            mic_loud = (2.0, 8.0, 0.08) if who == SPEAKER_YOU else None
            meeting_loud = (2.0, 8.0, 0.05) if who == SPEAKER_MEETING else None
            mic = channel(10.0, base_rms=0.003, loud=mic_loud, seed=100 + i)
            meeting = channel(10.0, base_rms=0.0001, loud=meeting_loud, seed=200 + i)

            det.observe(mic, meeting, SR)
            got.append(det.classify(mic, meeting, 2.0, 8.0, SR).speaker)
        return got

    def test_both_sides_are_recognised_across_a_long_call(self):
        turns = [SPEAKER_YOU, SPEAKER_MEETING] * 15
        got = self._run(turns)

        correct = sum(1 for want, have in zip(turns, got) if want == have)
        assert correct / len(turns) >= 0.9, f"only {correct}/{len(turns)} correct"

    def test_a_mostly_one_sided_call_still_catches_the_other_side(self):
        """Nine of your turns to one of theirs — theirs must not be swallowed."""
        turns = [SPEAKER_YOU] * 9 + [SPEAKER_MEETING]
        got = self._run(turns)
        assert got[-1] == SPEAKER_MEETING
        assert got.count(SPEAKER_YOU) == 9

    def test_listening_for_a_long_stretch(self):
        """The failure case: you stay quiet while the other side talks."""
        turns = [SPEAKER_MEETING] * 20
        got = self._run(turns)
        assert got.count(SPEAKER_MEETING) == 20


class TestMergeRuns:
    def test_consecutive_same_speaker_merges(self):
        runs = merge_runs([
            {"speaker": "You", "text": "hello", "start": 0.0, "end": 1.0},
            {"speaker": "You", "text": "there", "start": 1.0, "end": 2.0},
            {"speaker": "Meeting", "text": "hi", "start": 2.0, "end": 3.0},
        ])
        assert len(runs) == 2
        assert runs[0]["text"] == "hello there"
        assert runs[0]["end"] == 2.0
        assert runs[1]["text"] == "hi"

    def test_alternating_speakers_do_not_merge(self):
        runs = merge_runs([
            {"speaker": "You", "text": "a", "start": 0.0, "end": 1.0},
            {"speaker": "Meeting", "text": "b", "start": 1.0, "end": 2.0},
            {"speaker": "You", "text": "c", "start": 2.0, "end": 3.0},
        ])
        assert len(runs) == 3

    def test_does_not_mutate_its_input(self):
        original = [
            {"speaker": "You", "text": "a", "start": 0.0, "end": 1.0},
            {"speaker": "You", "text": "b", "start": 1.0, "end": 2.0},
        ]
        merge_runs(original)
        assert original[0]["text"] == "a"

    def test_empty(self):
        assert merge_runs([]) == []
