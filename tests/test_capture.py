"""Capture and alignment tests.

These drive ``DualStreamCapture`` through its ``feed_*`` seam with synthetic
blocks and a controlled clock, so no PortAudio or real device is involved.
"""

import numpy as np
import pytest

from scribe_audio.capture import (
    STALE_AFTER_SECONDS,
    DualStreamCapture,
    TimedRingBuffer,
)

SR = 1000  # low rate keeps the arrays small and the arithmetic readable


class FakeClock:
    def __init__(self, start: float = 1000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        self.now += seconds
        return self.now


def tone(n: int, value: float = 1.0) -> np.ndarray:
    return np.full(n, value, dtype=np.float32)


class TestTimedRingBuffer:
    def test_empty_buffer_reads_as_silence(self):
        buf = TimedRingBuffer(SR, duration=2.0)
        out = buf.read_window(end_time=100.0, duration=2.0)
        assert out.shape == (2 * SR,)
        assert np.all(out == 0)

    def test_write_then_read_returns_the_samples(self):
        buf = TimedRingBuffer(SR, duration=2.0)
        buf.write(tone(SR, 0.5), end_time=10.0)
        out = buf.read_window(end_time=10.0, duration=1.0)
        assert np.allclose(out, 0.5)

    def test_older_region_is_zero_padded(self):
        """A 2s window when only 1s has been captured: first half is silence."""
        buf = TimedRingBuffer(SR, duration=2.0)
        buf.write(tone(SR, 0.5), end_time=10.0)
        out = buf.read_window(end_time=10.0, duration=2.0)
        assert np.all(out[:SR] == 0)
        assert np.allclose(out[SR:], 0.5)

    def test_reading_in_the_past_shifts_the_window(self):
        buf = TimedRingBuffer(SR, duration=4.0)
        buf.write(tone(SR, 1.0), end_time=10.0)   # covers t=9..10
        buf.write(tone(SR, 2.0), end_time=11.0)   # covers t=10..11
        # Ask for the second ending at t=10 — should be all 1.0, not 2.0.
        out = buf.read_window(end_time=10.0, duration=1.0)
        assert np.allclose(out, 1.0)

    def test_reading_the_future_returns_silence(self):
        buf = TimedRingBuffer(SR, duration=2.0)
        buf.write(tone(SR, 1.0), end_time=10.0)
        out = buf.read_window(end_time=20.0, duration=1.0)
        assert np.all(out == 0)

    def test_block_larger_than_buffer_keeps_the_newest_samples(self):
        buf = TimedRingBuffer(SR, duration=1.0)
        block = np.concatenate([tone(SR, 1.0), tone(SR, 2.0)])
        buf.write(block, end_time=10.0)
        out = buf.read_window(end_time=10.0, duration=1.0)
        assert np.allclose(out, 2.0)

    def test_recent_rms(self):
        buf = TimedRingBuffer(SR, duration=2.0)
        assert buf.recent_rms() == 0.0
        buf.write(tone(SR, 0.5), end_time=10.0)
        assert buf.recent_rms(0.5) == pytest.approx(0.5)


def make_capture(clock, mic=1, system=2, window=2.0, step=1.0):
    return DualStreamCapture(
        mic_device=mic,
        system_device=system,
        sample_rate=SR,
        window_duration=window,
        step_duration=step,
        stream_factory=lambda device, channels, callback: None,
        time_fn=clock,
    )


def feed_both(cap, clock, seconds, mic_value=0.0, meeting_value=0.0, block=0.1):
    """Feed both streams in lockstep for ``seconds`` of audio."""
    n = int(SR * block)
    for _ in range(int(seconds / block)):
        now = clock.advance(block)
        cap.feed_mic(tone(n, mic_value), now)
        cap.feed_system(tone(n, meeting_value), now)


class TestWindowing:
    def test_no_window_before_a_full_step(self):
        clock = FakeClock()
        cap = make_capture(clock)
        feed_both(cap, clock, seconds=0.5)
        assert cap.get_window(timeout=0) is None

    def test_window_after_a_step(self):
        clock = FakeClock()
        cap = make_capture(clock)
        feed_both(cap, clock, seconds=1.0, mic_value=0.3, meeting_value=0.2)
        win = cap.get_window(timeout=0)
        assert win is not None
        assert win.sample_rate == SR
        assert win.duration == pytest.approx(2.0)

    def test_windows_arrive_once_per_step(self):
        clock = FakeClock()
        cap = make_capture(clock)
        windows = 0
        for _ in range(6):
            feed_both(cap, clock, seconds=0.5)
            if cap.get_window(timeout=0) is not None:
                windows += 1
        # 3s of audio at a 1s step
        assert windows == 3

    def test_mono_sums_both_streams(self):
        clock = FakeClock()
        cap = make_capture(clock)
        feed_both(cap, clock, seconds=2.0, mic_value=0.3, meeting_value=0.2)
        win = cap.get_window(timeout=0)
        assert np.allclose(win.mono[-SR:], 0.5)

    def test_mono_clips_rather_than_wrapping(self):
        clock = FakeClock()
        cap = make_capture(clock)
        feed_both(cap, clock, seconds=2.0, mic_value=0.9, meeting_value=0.9)
        win = cap.get_window(timeout=0)
        assert win.mono.max() <= 1.0

    def test_streams_stay_separated_for_attribution(self):
        clock = FakeClock()
        cap = make_capture(clock)
        feed_both(cap, clock, seconds=2.0, mic_value=0.4, meeting_value=0.1)
        win = cap.get_window(timeout=0)
        assert np.allclose(win.mic[-SR:], 0.4)
        assert np.allclose(win.meeting[-SR:], 0.1)


class TestAlignment:
    def test_streams_are_aligned_by_time_not_sample_count(self):
        """The core of the two-stream design.

        The system stream delivers blocks at twice the sample count per unit of
        wall-clock time — the shape clock drift takes. A sample-counting design
        would progressively misplace one stream against the other; aligning on
        timestamps must keep a transient at the same position in both windows.
        """
        clock = FakeClock()
        cap = make_capture(clock, window=2.0, step=1.0)

        # 2 seconds of silence in lockstep, but the system stream runs "fast".
        for _ in range(20):
            now = clock.advance(0.1)
            cap.feed_mic(tone(int(SR * 0.1), 0.0), now)
            cap.feed_system(tone(int(SR * 0.2), 0.0), now)

        # A transient in both streams at the same instant.
        now = clock.advance(0.1)
        cap.feed_mic(tone(int(SR * 0.1), 1.0), now)
        cap.feed_system(tone(int(SR * 0.2), 1.0), now)

        feed_both(cap, clock, seconds=0.9)
        win = cap.get_window(timeout=0)
        assert win is not None

        mic_peak = int(np.argmax(win.mic))
        meeting_peak = int(np.argmax(win.meeting))
        # Within one block of each other despite the differing sample rates.
        assert abs(mic_peak - meeting_peak) <= int(SR * 0.1)

    def test_window_is_not_read_ahead_of_the_slower_stream(self):
        clock = FakeClock()
        cap = make_capture(clock)
        # Mic runs ahead in time; system lags behind by half a second.
        for _ in range(20):
            now = clock.advance(0.1)
            cap.feed_mic(tone(int(SR * 0.1)), now)
            cap.feed_system(tone(int(SR * 0.1)), now - 0.5)
        win = cap.get_window(timeout=0)
        assert win is not None
        # Alignment end follows the lagging stream, not the leading one.
        assert win.end_time == pytest.approx(clock.now - 0.5)


class TestStreamLoss:
    def test_unplugged_mic_does_not_stop_capture(self):
        """The headset-unplug failure, as a test.

        The mic stream stops firing. Windows must keep being produced from
        system audio alone rather than the pipeline freezing.
        """
        clock = FakeClock()
        cap = make_capture(clock)
        feed_both(cap, clock, seconds=2.0, mic_value=0.3, meeting_value=0.2)
        assert cap.get_window(timeout=0) is not None

        # Mic goes away; only system audio continues.
        n = int(SR * 0.1)
        for _ in range(int((STALE_AFTER_SECONDS + 2.0) / 0.1)):
            now = clock.advance(0.1)
            cap.feed_system(tone(n, 0.2), now)

        assert "mic" in cap.stalled_streams
        win = cap.get_window(timeout=0)
        assert win is not None
        assert np.allclose(win.meeting[-SR:], 0.2)

    def test_system_only_capture_is_valid(self):
        clock = FakeClock()
        cap = DualStreamCapture(
            mic_device=None,
            system_device=2,
            sample_rate=SR,
            window_duration=2.0,
            step_duration=1.0,
            stream_factory=lambda d, c, cb: None,
            time_fn=clock,
        )
        n = int(SR * 0.1)
        for _ in range(10):
            cap.feed_system(tone(n, 0.5), clock.advance(0.1))
        win = cap.get_window(timeout=0)
        assert win is not None
        assert np.all(win.mic == 0)
        assert np.allclose(win.meeting[-SR:], 0.5)

    def test_requires_at_least_one_device(self):
        with pytest.raises(ValueError):
            DualStreamCapture(mic_device=None, system_device=None)

    def test_rejects_step_larger_than_window(self):
        with pytest.raises(ValueError):
            DualStreamCapture(mic_device=1, system_device=2,
                              window_duration=10.0, step_duration=20.0)


class TestDropDetection:
    def test_falling_behind_is_reported(self):
        """If more than a window of audio passes between calls, some was lost."""
        clock = FakeClock()
        cap = make_capture(clock, window=2.0, step=1.0)
        feed_both(cap, clock, seconds=1.0)
        first = cap.get_window(timeout=0)
        assert first is not None
        assert first.dropped_seconds == 0.0

        # Simulate transcription taking far longer than the window.
        feed_both(cap, clock, seconds=5.0)
        second = cap.get_window(timeout=0)
        assert second is not None
        assert second.dropped_seconds == pytest.approx(3.0)

    def test_keeping_up_reports_no_drops(self):
        clock = FakeClock()
        cap = make_capture(clock, window=2.0, step=1.0)
        for _ in range(5):
            feed_both(cap, clock, seconds=1.0)
            win = cap.get_window(timeout=0)
            assert win is not None
            assert win.dropped_seconds == 0.0


class TestFlush:
    def test_flush_returns_the_tail(self):
        """Why recordings used to end mid-sentence."""
        clock = FakeClock()
        cap = make_capture(clock)
        feed_both(cap, clock, seconds=1.0)
        cap.get_window(timeout=0)

        # Half a step of audio arrives, then the user hits stop.
        feed_both(cap, clock, seconds=0.5, mic_value=0.7)
        tail = cap.flush()
        assert tail is not None
        assert tail.mic.max() == pytest.approx(0.7)

    def test_flush_is_empty_when_nothing_new_arrived(self):
        clock = FakeClock()
        cap = make_capture(clock)
        feed_both(cap, clock, seconds=1.0)
        cap.get_window(timeout=0)
        assert cap.flush() is None

    def test_flush_before_any_audio(self):
        clock = FakeClock()
        cap = make_capture(clock)
        assert cap.flush() is None


class TestStopUnblocks:
    def test_stop_releases_a_waiting_reader(self):
        """Ctrl+C must not wait out the full timeout."""
        import threading

        clock = FakeClock()
        cap = make_capture(clock)
        result = {}

        def reader():
            result["window"] = cap.get_window(timeout=30)

        thread = threading.Thread(target=reader)
        thread.start()
        cap.stop()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert result["window"] is None
