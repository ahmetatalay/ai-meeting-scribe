"""Audio spool tests.

The spool exists to make falling behind survivable, so the tests that matter
are the ones about reading behind the write head, zero-padding at both ends,
and the file still being a valid WAV afterwards.
"""

import struct
import wave

import numpy as np
import pytest

from scribe_audio.spool import AudioSpool, BYTES_PER_FRAME, HEADER_BYTES

SR = 16000


def ramp(n: int, start: float = 0.0) -> np.ndarray:
    """Distinguishable samples, so a misaligned read is visible."""
    return (np.arange(n, dtype=np.float32) + start) / 100000.0


@pytest.fixture
def spool(tmp_path):
    s = AudioSpool(tmp_path / "audio.wav", SR)
    yield s
    if not s.is_closed:
        s.close()


class TestAppendAndRead:
    def test_what_goes_in_comes_back(self, spool):
        mic, meeting = ramp(1000), ramp(1000, 5000)
        spool.append(mic, meeting)

        got_mic, got_meeting = spool.read(0, 1000)
        assert np.allclose(got_mic, mic, atol=1e-4)
        assert np.allclose(got_meeting, meeting, atol=1e-4)

    def test_the_channels_do_not_bleed_into_each_other(self, spool):
        spool.append(np.full(500, 0.5, dtype=np.float32),
                     np.zeros(500, dtype=np.float32))
        mic, meeting = spool.read(0, 500)
        assert np.allclose(mic, 0.5, atol=1e-4)
        assert np.all(meeting == 0.0)

    def test_appends_are_contiguous(self, spool):
        spool.append(ramp(100), ramp(100))
        spool.append(ramp(100, 100), ramp(100, 100))
        mic, _ = spool.read(0, 200)
        assert np.allclose(mic, ramp(200), atol=1e-4)

    def test_frame_count_tracks_appends(self, spool):
        assert spool.frames_written == 0
        spool.append(ramp(160), ramp(160))
        assert spool.frames_written == 160
        spool.append(ramp(40), ramp(40))
        assert spool.frames_written == 200
        assert spool.seconds_written == pytest.approx(200 / SR)

    def test_mismatched_channel_lengths_are_refused(self, spool):
        with pytest.raises(ValueError, match="same length"):
            spool.append(ramp(100), ramp(99))

    def test_an_empty_append_is_harmless(self, spool):
        assert spool.append(np.zeros(0), np.zeros(0)) == 0

    def test_values_outside_unity_are_clipped_not_wrapped(self, spool):
        """A sum of two channels can exceed 1.0; wrapping would turn a loud
        passage into noise."""
        spool.append(np.full(10, 3.0, dtype=np.float32),
                     np.full(10, -3.0, dtype=np.float32))
        mic, meeting = spool.read(0, 10)
        assert np.all(mic > 0.99)
        assert np.all(meeting < -0.99)


class TestReadingBehindTheWriteHead:
    """The whole point: a reader that is far behind still gets its audio."""

    def test_a_reader_far_behind_still_gets_the_old_audio(self, spool):
        # Ten minutes, well past any ring buffer that would have held it.
        for i in range(600):
            spool.append(ramp(SR, i * SR), ramp(SR, i * SR))

        mic, _ = spool.read(0, 100)
        assert np.allclose(mic, ramp(100), atol=1e-4), \
            "the opening second must survive ten minutes of later audio"

    def test_reading_while_writing_continues(self, spool):
        spool.append(ramp(SR), ramp(SR))
        mic_first, _ = spool.read(0, SR)
        spool.append(ramp(SR, SR), ramp(SR, SR))
        mic_again, _ = spool.read(0, SR)
        assert np.array_equal(mic_first, mic_again)

    def test_nothing_is_lost_however_far_behind(self, spool):
        """Every frame written must be readable, in order."""
        total = 0
        for i in range(50):
            spool.append(ramp(1000, i * 1000), ramp(1000, i * 1000))
            total += 1000
        mic, _ = spool.read(0, total)
        assert np.allclose(mic, ramp(total), atol=1e-4)


class TestPadding:
    def test_reading_past_the_end_pads_with_zeros(self, spool):
        spool.append(np.ones(100, dtype=np.float32), np.ones(100, dtype=np.float32))
        mic, _ = spool.read(0, 300)
        assert np.allclose(mic[:100], 1.0, atol=1e-4)
        assert np.all(mic[100:] == 0.0)

    def test_reading_before_the_start_pads_with_zeros(self, spool):
        """The first window of a recording asks for audio from before time
        zero; it must not shift the real samples."""
        spool.append(np.ones(100, dtype=np.float32), np.ones(100, dtype=np.float32))
        mic, _ = spool.read(-50, 100)
        assert np.all(mic[:50] == 0.0)
        assert np.allclose(mic[50:], 1.0, atol=1e-4)

    def test_reading_an_empty_spool_gives_silence(self, spool):
        mic, meeting = spool.read(0, 1000)
        assert mic.shape == (1000,) and np.all(mic == 0.0)
        assert meeting.shape == (1000,) and np.all(meeting == 0.0)

    def test_a_zero_length_read_is_empty(self, spool):
        mic, meeting = spool.read(0, 0)
        assert mic.size == 0 and meeting.size == 0

    def test_a_read_entirely_past_the_end_is_silence(self, spool):
        spool.append(np.ones(100, dtype=np.float32), np.ones(100, dtype=np.float32))
        mic, _ = spool.read(5000, 100)
        assert np.all(mic == 0.0)


class TestTheFileIsPlayable:
    """Keeping the audio is only useful if something can open it."""

    def test_the_closed_file_is_a_valid_wav(self, tmp_path):
        s = AudioSpool(tmp_path / "a.wav", SR)
        s.append(ramp(SR), ramp(SR))
        s.close()

        with wave.open(str(tmp_path / "a.wav")) as w:
            assert w.getnchannels() == 2
            assert w.getframerate() == SR
            assert w.getsampwidth() == 2
            assert w.getnframes() == SR

    def test_the_header_length_matches_the_data(self, tmp_path):
        s = AudioSpool(tmp_path / "a.wav", SR)
        s.append(ramp(1234), ramp(1234))
        s.close()

        raw = (tmp_path / "a.wav").read_bytes()
        riff = struct.unpack("<I", raw[4:8])[0]
        data = struct.unpack("<I", raw[40:44])[0]
        assert data == 1234 * BYTES_PER_FRAME
        assert riff == 36 + data
        assert len(raw) == HEADER_BYTES + data

    def test_a_spool_with_no_audio_is_still_a_valid_wav(self, tmp_path):
        s = AudioSpool(tmp_path / "a.wav", SR)
        s.close()
        with wave.open(str(tmp_path / "a.wav")) as w:
            assert w.getnframes() == 0


class TestLifecycle:
    def test_close_is_idempotent(self, tmp_path):
        s = AudioSpool(tmp_path / "a.wav", SR)
        s.close()
        s.close()
        assert s.is_closed

    def test_appending_after_close_is_refused(self, tmp_path):
        s = AudioSpool(tmp_path / "a.wav", SR)
        s.close()
        with pytest.raises(ValueError, match="closed"):
            s.append(ramp(10), ramp(10))

    def test_delete_removes_the_file(self, tmp_path):
        s = AudioSpool(tmp_path / "a.wav", SR)
        s.append(ramp(100), ramp(100))
        s.delete()
        assert not (tmp_path / "a.wav").exists()

    def test_delete_on_an_already_deleted_spool_is_harmless(self, tmp_path):
        s = AudioSpool(tmp_path / "a.wav", SR)
        s.delete()
        s.delete()

    def test_it_works_as_a_context_manager(self, tmp_path):
        with AudioSpool(tmp_path / "a.wav", SR) as s:
            s.append(ramp(10), ramp(10))
        assert s.is_closed

    def test_a_missing_parent_directory_is_created(self, tmp_path):
        s = AudioSpool(tmp_path / "deep" / "nested" / "a.wav", SR)
        s.close()
        assert (tmp_path / "deep" / "nested" / "a.wav").is_file()


class TestSize:
    def test_an_hour_of_audio_is_a_manageable_file(self):
        """The argument for spooling at all: audio is small enough that disk
        can absorb any backlog a laptop produces."""
        per_hour_mb = SR * 3600 * BYTES_PER_FRAME / 1024 / 1024
        assert per_hour_mb < 300, f"{per_hour_mb:.0f}MB per hour"


class TestConcurrency:
    def test_concurrent_appends_and_reads_do_not_corrupt(self, spool):
        """Capture appends on one thread while transcription reads behind it."""
        import threading

        errors = []

        def write():
            try:
                for i in range(200):
                    spool.append(ramp(500, i * 500), ramp(500, i * 500))
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        def read():
            try:
                for _ in range(200):
                    n = spool.frames_written
                    if n >= 500:
                        mic, _ = spool.read(0, 500)
                        assert np.allclose(mic, ramp(500), atol=1e-4)
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=write), threading.Thread(target=read)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, errors
        assert spool.frames_written == 100000


class TestSpoolReader:
    """Re-transcribing a recording runs the same pipeline a live session does,
    which is what makes "record with a fast model, re-transcribe with a good
    one" possible — a better answer than choosing one model for both jobs."""

    def write_stereo(self, path, mic, meeting, sample_rate=SR):
        s = AudioSpool(path, sample_rate)
        s.append(mic, meeting)
        s.close()
        return path

    def test_it_reads_back_both_channels(self, tmp_path):
        from scribe_audio.spool import SpoolReader

        path = self.write_stereo(
            tmp_path / "a.wav",
            np.full(SR * 4, 0.4, dtype=np.float32),
            np.full(SR * 4, 0.1, dtype=np.float32),
        )
        reader = SpoolReader(path, window_duration=2.0, step_duration=2.0)
        win = reader.get_window()
        assert np.allclose(win.mic, 0.4, atol=1e-4)
        assert np.allclose(win.meeting, 0.1, atol=1e-4)

    def test_every_second_of_the_file_is_covered(self, tmp_path):
        from scribe_audio.spool import SpoolReader

        marks = np.concatenate([
            np.full(SR, (i + 1) / 100.0, dtype=np.float32) for i in range(8)
        ])
        path = self.write_stereo(tmp_path / "a.wav", marks,
                                 np.zeros_like(marks))
        reader = SpoolReader(path, window_duration=2.0, step_duration=1.0)

        seen = set()
        while True:
            win = reader.get_window()
            if win is None:
                break
            for v in np.unique(np.round(win.mic * 100).astype(int)):
                if v > 0:
                    seen.add(int(v))
        assert seen == set(range(1, 9))

    def test_it_reports_exhaustion_as_stopped(self, tmp_path):
        from scribe_audio.spool import SpoolReader

        path = self.write_stereo(tmp_path / "a.wav",
                                 np.full(SR, 0.2, dtype=np.float32),
                                 np.zeros(SR, dtype=np.float32))
        reader = SpoolReader(path, window_duration=1.0, step_duration=1.0)
        assert not reader.is_stopped
        reader.get_window()
        assert reader.is_stopped, "the session loop ends on this"
        assert reader.get_window() is None

    def test_duration_is_reported(self, tmp_path):
        from scribe_audio.spool import SpoolReader

        path = self.write_stereo(tmp_path / "a.wav",
                                 np.zeros(SR * 5, dtype=np.float32),
                                 np.zeros(SR * 5, dtype=np.float32))
        assert SpoolReader(path).duration == pytest.approx(5.0)

    def test_a_mono_file_is_read_as_meeting_audio(self, tmp_path):
        """There is no channel separation to attribute from, so labelling
        everything 'Meeting' is honest; splitting it would invent one."""
        import wave

        from scribe_audio.spool import SpoolReader

        path = tmp_path / "mono.wav"
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes((np.full(SR, 0.3) * 32767).astype(np.int16).tobytes())

        reader = SpoolReader(path, window_duration=1.0, step_duration=1.0)
        assert reader.is_mono
        win = reader.get_window()
        assert np.allclose(win.meeting, 0.3, atol=1e-4)
        assert np.all(win.mic == 0.0)

    def test_an_eight_bit_file_is_refused_with_a_reason(self, tmp_path):
        import wave

        from scribe_audio.spool import SpoolReader

        path = tmp_path / "eight.wav"
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(1)
            w.setframerate(SR)
            w.writeframes(b"\x80" * SR)
        with pytest.raises(ValueError, match="16-bit"):
            SpoolReader(path)

    def test_the_source_file_is_never_deleted(self, tmp_path):
        """discard_spool() deletes a live capture's scratch file; a file the
        user pointed us at is theirs."""
        from scribe_audio.spool import SpoolReader

        path = self.write_stereo(tmp_path / "a.wav",
                                 np.zeros(SR, dtype=np.float32),
                                 np.zeros(SR, dtype=np.float32))
        reader = SpoolReader(path)
        reader.discard_spool()
        reader.keep_spool(tmp_path / "elsewhere.wav")
        assert path.exists()
        assert not (tmp_path / "elsewhere.wav").exists()

    def test_widening_the_step_is_ignored(self, tmp_path):
        """A file has no deadline, so trading boundary corrections for
        throughput would give up quality for nothing."""
        from scribe_audio.spool import SpoolReader

        path = self.write_stereo(tmp_path / "a.wav",
                                 np.zeros(SR * 4, dtype=np.float32),
                                 np.zeros(SR * 4, dtype=np.float32))
        reader = SpoolReader(path, window_duration=2.0, step_duration=1.0)
        reader.set_step(2.0)
        assert reader.step_duration == 1.0

    def test_a_backlog_is_reported_as_what_is_left(self, tmp_path):
        from scribe_audio.spool import SpoolReader

        path = self.write_stereo(tmp_path / "a.wav",
                                 np.zeros(SR * 6, dtype=np.float32),
                                 np.zeros(SR * 6, dtype=np.float32))
        reader = SpoolReader(path, window_duration=2.0, step_duration=2.0)
        assert reader.backlog_seconds == pytest.approx(6.0)
        reader.get_window()
        assert reader.backlog_seconds == pytest.approx(4.0)
