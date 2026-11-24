"""Recording engine tests.

Fakes stand in for PortAudio and Whisper, so the whole lifecycle — model
loading, the capture loop, health notices, the tail flush and the event stream —
runs deterministically in-process.
"""

import json
import queue
import re

import pytest

from fakes import (
    SR,
    FakeCapture,
    ScriptedTranscriber,
    build_session as build,
    drain,
    make_window,
    of_type,
)
from scribe_audio.session import (
    STATE_ERROR,
    STATE_IDLE,
    STATE_RECORDING,
    STATE_STOPPED,
    EventBus,
    RecordingSession,
    SessionConfig,
)


class TestEventBus:
    def test_subscribers_receive_published_events(self):
        bus = EventBus()
        a, b = bus.subscribe(), bus.subscribe()
        bus.publish({"type": "test", "n": 1})
        assert drain(a)[0]["n"] == 1
        assert drain(b)[0]["n"] == 1

    def test_events_are_timestamped(self):
        bus = EventBus()
        q = bus.subscribe()
        bus.publish({"type": "test"})
        assert "at" in drain(q)[0]

    def test_late_subscriber_gets_the_backlog(self):
        bus = EventBus()
        bus.publish({"type": "state", "value": "recording"})
        bus.publish({"type": "segments", "segments": []})
        events = drain(bus.subscribe())
        assert [e["type"] for e in events] == ["state", "segments"]

    def test_replay_can_be_declined(self):
        bus = EventBus()
        bus.publish({"type": "state", "value": "recording"})
        assert drain(bus.subscribe(replay=False)) == []

    def test_level_events_are_not_replayed(self):
        """High-frequency meter updates are worthless after the fact."""
        bus = EventBus()
        bus.publish({"type": "levels", "mic": 0.1})
        bus.publish({"type": "state", "value": "recording"})
        events = drain(bus.subscribe())
        assert [e["type"] for e in events] == ["state"]

    def test_a_slow_subscriber_does_not_block_the_publisher(self):
        bus = EventBus(depth=4)
        q = bus.subscribe()
        for i in range(50):
            bus.publish({"type": "test", "n": i})
        got = drain(q)
        assert len(got) <= 4
        assert got[-1]["n"] == 49  # newest survives, oldest dropped

    def test_unsubscribe_stops_delivery(self):
        bus = EventBus()
        q = bus.subscribe()
        bus.unsubscribe(q)
        bus.publish({"type": "test"})
        assert drain(q) == []
        assert bus.subscriber_count == 0


class TestLifecycle:
    def test_full_run_reaches_stopped(self, tmp_path):
        session, capture, transcriber = build(
            tmp_path, ["hello there"], [make_window()]
        )
        q = session.bus.subscribe()
        session.start()
        assert session.wait(timeout=5)

        assert session.state == STATE_STOPPED
        assert capture.started and capture.flushed
        assert transcriber.loaded

        states = [e["value"] for e in of_type(drain(q), "state")]
        assert states == ["loading_model", "recording", "stopping", "stopped"]

    def test_start_twice_is_rejected(self, tmp_path):
        session, capture, _ = build(tmp_path, [], [make_window()] * 50)
        session.start()
        with pytest.raises(RuntimeError):
            session.start()
        session.stop()

    def test_stop_before_any_window(self, tmp_path):
        session, _, _ = build(tmp_path, [], [])
        session.start()
        session.stop()
        assert session.state == STATE_STOPPED

    def test_stop_is_idempotent(self, tmp_path):
        session, _, _ = build(tmp_path, ["a"], [make_window()])
        session.start()
        session.stop()
        session.stop()
        assert session.state == STATE_STOPPED

    def test_a_failing_transcriber_reports_an_error_state(self, tmp_path):
        class Exploding:
            def load(self):
                raise RuntimeError("model file corrupt")

        cfg = SessionConfig(mic_device=1, system_device=2, raw_dir=tmp_path)
        session = RecordingSession(
            cfg,
            capture_factory=lambda c: FakeCapture([]),
            transcriber_factory=lambda c: Exploding(),
        )
        q = session.bus.subscribe()
        session.start()
        assert session.wait(timeout=5)

        assert session.state == STATE_ERROR
        errors = [e for e in of_type(drain(q), "notice") if e["level"] == "error"]
        assert errors and "model file corrupt" in errors[0]["message"]

    def test_config_requires_a_device(self):
        with pytest.raises(ValueError):
            SessionConfig(mic_device=None, system_device=None)


class TestSegmentEvents:
    def test_segments_are_published_with_ids(self, tmp_path):
        session, _, _ = build(
            tmp_path, ["first thing", "second thing entirely"],
            [make_window(seed=0), make_window(seed=10)],
        )
        q = session.bus.subscribe()
        session.start()
        session.wait(timeout=5)

        published = [s for e in of_type(drain(q), "segments") for s in e["segments"]]
        assert [s["id"] for s in published] == [1, 2]
        assert published[0]["text"] == "first thing"

    def test_ids_increase_monotonically(self, tmp_path):
        script = [f"sentence number {i}" for i in range(5)]
        session, _, _ = build(
            tmp_path, script, [make_window(seed=i * 10) for i in range(5)]
        )
        q = session.bus.subscribe()
        session.start()
        session.wait(timeout=5)

        ids = [s["id"] for e in of_type(drain(q), "segments") for s in e["segments"]]
        assert ids == sorted(ids) and len(set(ids)) == len(ids)

    def test_overlap_is_removed_before_publishing(self, tmp_path):
        session, _, _ = build(
            tmp_path,
            ["we should ship the build", "ship the build today"],
            [make_window(seed=0), make_window(seed=10)],
        )
        q = session.bus.subscribe()
        session.start()
        session.wait(timeout=5)

        texts = [s["text"] for e in of_type(drain(q), "segments") for s in e["segments"]]
        assert " ".join(texts).count("ship the build") == 1

    def test_session_event_announces_the_path_once(self, tmp_path):
        session, _, _ = build(
            tmp_path, ["one", "two separate things"],
            [make_window(seed=0), make_window(seed=10)],
        )
        q = session.bus.subscribe()
        session.start()
        session.wait(timeout=5)

        announcements = of_type(drain(q), "session")
        assert len(announcements) == 1
        assert announcements[0]["markdown"].endswith(".md")

    def test_speaker_attribution_flows_through(self, tmp_path):
        session, _, _ = build(
            tmp_path, ["my question here", "their different answer"],
            [make_window("You", seed=0), make_window("Meeting", seed=10)],
        )
        q = session.bus.subscribe()
        session.start()
        session.wait(timeout=5)

        speakers = [s["speaker"] for e in of_type(drain(q), "segments")
                    for s in e["segments"]]
        assert speakers == ["You", "Meeting"]


class TestTailFlush:
    def test_the_tail_window_is_transcribed(self, tmp_path):
        session, _, _ = build(
            tmp_path,
            ["the main part", "the final words before stopping"],
            [make_window(seed=0)],
            tail=make_window(seed=10),
        )
        q = session.bus.subscribe()
        session.start()
        session.wait(timeout=5)

        events = of_type(drain(q), "segments")
        texts = [s["text"] for e in events for s in e["segments"]]
        assert "the final words before stopping" in " ".join(texts)
        assert any(e["final"] for e in events)

    def test_no_tail_is_fine(self, tmp_path):
        session, _, _ = build(tmp_path, ["only part"], [make_window()], tail=None)
        session.start()
        assert session.wait(timeout=5)
        assert session.state == STATE_STOPPED


class TestHealthNotices:
    def test_dropped_audio_is_reported(self, tmp_path):
        session, _, _ = build(
            tmp_path, ["some speech"], [make_window(dropped=12.0)]
        )
        q = session.bus.subscribe()
        session.start()
        session.wait(timeout=5)

        notices = [e for e in of_type(drain(q), "notice")
                   if e["code"] == "dropped_audio"]
        assert notices and notices[0]["seconds"] == 12.0
        assert session.stats.dropped_seconds == 12.0

    def test_sub_second_overruns_are_not_reported(self, tmp_path):
        """Observed in a real meeting: a stream of warnings reading
        "Transcription fell behind: 0s of audio was not transcribed. Try a
        smaller model." A window is emitted on wall-clock time, so a cycle
        finishing 0.4s late is scheduling jitter, not lost audio — and the
        advice to change model was wrong as well as alarming."""
        session, _, _ = build(
            tmp_path, ["some speech"], [make_window(dropped=0.4)]
        )
        q = session.bus.subscribe()
        session.start()
        session.wait(timeout=5)

        notices = [e for e in of_type(drain(q), "notice")
                   if e["code"] == "dropped_audio"]
        assert notices == []

    def test_a_real_shortfall_is_still_reported(self, tmp_path):
        session, _, _ = build(
            tmp_path, ["some speech"], [make_window(dropped=4.0)]
        )
        q = session.bus.subscribe()
        session.start()
        session.wait(timeout=5)

        notices = [e for e in of_type(drain(q), "notice")
                   if e["code"] == "dropped_audio"]
        assert len(notices) == 1

    def test_a_reported_shortfall_never_reads_as_zero(self, tmp_path):
        """Whatever the threshold, the number shown must not round to 0."""
        from scribe_audio.session import DROPPED_AUDIO_FLOOR

        session, _, _ = build(
            tmp_path, ["some speech"],
            [make_window(dropped=DROPPED_AUDIO_FLOOR)],
        )
        q = session.bus.subscribe()
        session.start()
        session.wait(timeout=5)

        notices = [e for e in of_type(drain(q), "notice")
                   if e["code"] == "dropped_audio"]
        assert notices, "the floor itself must still report"

        amount = re.search(r"(\d+(?:\.\d+)?)s", notices[0]["message"])
        assert amount, f"no duration in {notices[0]['message']!r}"
        assert float(amount.group(1)) > 0, f"rendered as {amount.group(0)}"

    def test_short_overruns_still_count_towards_the_total(self, tmp_path):
        """Not worth a banner, but the statistic should stay honest."""
        session, _, _ = build(
            tmp_path, ["some speech"], [make_window(dropped=0.4)]
        )
        session.start()
        session.wait(timeout=5)
        assert session.stats.dropped_seconds == pytest.approx(0.4)

    def test_silent_loopback_with_speech_is_reported(self, tmp_path):
        """The Zoom-bypass signature."""
        session, _, _ = build(
            tmp_path,
            ["I am talking", "still talking about other things"],
            [make_window("You", seed=0, meeting_silent=True),
             make_window("You", seed=10, meeting_silent=True)],
        )
        q = session.bus.subscribe()
        session.start()
        session.wait(timeout=5)

        codes = [e["code"] for e in of_type(drain(q), "notice")]
        assert "loopback_silent" in codes

    def test_silent_loopback_without_speech_is_not_reported(self, tmp_path):
        """A quiet start must not trigger the warning."""
        session, _, _ = build(
            tmp_path, ["", ""],
            [make_window(seed=0, meeting_silent=True),
             make_window(seed=10, meeting_silent=True)],
        )
        q = session.bus.subscribe()
        session.start()
        session.wait(timeout=5)

        codes = [e["code"] for e in of_type(drain(q), "notice")]
        assert "loopback_silent" not in codes

    def test_the_silent_loopback_warning_is_withdrawn_when_audio_arrives(
        self, tmp_path
    ):
        """Seen on screen: two "No meeting audio is arriving" banners sitting
        above a transcript that plainly contained meeting speech, with every
        setup check green. The warning was true when raised and never retracted,
        so it sent the reader to re-check routing that already worked."""
        session, _, _ = build(
            tmp_path,
            ["I am talking", "still talking about other things",
             "now they reply to me at some length"],
            [make_window("You", seed=0, meeting_silent=True),
             make_window("You", seed=10, meeting_silent=True),
             make_window("Meeting", seed=20)],
        )
        q = session.bus.subscribe()
        session.start()
        session.wait(timeout=5)

        silent = [e for e in of_type(drain(q), "notice")
                  if e["code"] == "loopback_silent"]
        assert len(silent) == 2, "expected the warning and its retraction"
        assert silent[0].get("resolved") is not True
        assert silent[1]["resolved"] is True

    def test_the_silent_warning_can_fire_again_after_recovering(self, tmp_path):
        """Routing that breaks a second time must warn a second time."""
        session, _, _ = build(
            tmp_path,
            ["I am talking", "still talking about other things",
             "now they reply to me at some length",
             "and I am speaking once more here",
             "continuing to speak with nothing coming back"],
            [make_window("You", seed=0, meeting_silent=True),
             make_window("You", seed=10, meeting_silent=True),
             make_window("Meeting", seed=20),
             make_window("You", seed=30, meeting_silent=True),
             make_window("You", seed=40, meeting_silent=True)],
        )
        q = session.bus.subscribe()
        session.start()
        session.wait(timeout=5)

        silent = [e for e in of_type(drain(q), "notice")
                  if e["code"] == "loopback_silent"]
        warnings = [e for e in silent if not e.get("resolved")]
        assert len(warnings) == 2, f"expected two warnings, got {len(silent)}"

    def test_being_behind_is_reported_as_delay_not_loss(self, tmp_path):
        """The old warning announced destroyed audio and told you to change
        model. Audio is spooled now, so the honest message is a delay."""
        from scribe_audio.session import BACKLOG_WARN_SECONDS

        session, _, _ = build(
            tmp_path, ["some speech"],
            [make_window(backlog=BACKLOG_WARN_SECONDS + 10)],
        )
        q = session.bus.subscribe()
        session.start()
        session.wait(timeout=5)

        notices = [e for e in of_type(drain(q), "notice")
                   if e["code"] == "backlog"]
        assert notices, "falling behind must still be surfaced"
        assert "Nothing is lost" in notices[0]["message"]
        assert "behind the meeting" in notices[0]["message"]

    def test_a_small_backlog_is_not_reported(self, tmp_path):
        """One slow cycle is not a trend and must not raise a banner."""
        session, _, _ = build(
            tmp_path, ["some speech"], [make_window(backlog=3.0)]
        )
        q = session.bus.subscribe()
        session.start()
        session.wait(timeout=5)

        assert [e for e in of_type(drain(q), "notice")
                if e["code"] == "backlog"] == []

    def test_catching_up_withdraws_the_backlog_warning(self, tmp_path):
        from scribe_audio.session import BACKLOG_WARN_SECONDS

        session, _, _ = build(
            tmp_path,
            ["first stretch of speech", "second stretch entirely different"],
            [make_window(seed=0, backlog=BACKLOG_WARN_SECONDS + 10),
             make_window(seed=10, backlog=0.0)],
        )
        q = session.bus.subscribe()
        session.start()
        session.wait(timeout=5)

        notices = [e for e in of_type(drain(q), "notice")
                   if e["code"] == "backlog"]
        assert len(notices) == 2
        assert notices[1]["resolved"] is True

    def test_backlog_is_recorded_in_the_statistics(self, tmp_path):
        session, _, _ = build(
            tmp_path, ["some speech"], [make_window(backlog=22.0)]
        )
        session.start()
        session.wait(timeout=5)
        assert session.stats.backlog_seconds == pytest.approx(22.0)

    def test_a_stalled_stream_is_reported_once(self, tmp_path):
        session, _, _ = build(
            tmp_path, ["a", "b different", "c different again"],
            [make_window(seed=i * 10) for i in range(3)],
            stalled=["mic"],
        )
        q = session.bus.subscribe()
        session.start()
        session.wait(timeout=5)

        stalls = [e for e in of_type(drain(q), "notice")
                  if e["code"] == "stream_stalled"]
        assert len(stalls) == 1
        assert stalls[0]["stream"] == "mic"

    def test_levels_are_published(self, tmp_path):
        session, _, _ = build(tmp_path, ["x"], [make_window()])
        q = session.bus.subscribe(replay=False)
        session.start()
        session.wait(timeout=5)
        assert of_type(drain(q), "levels")


class TestLanguageAnnouncement:
    """The engine must report detection as an event; it no longer prints."""

    def test_detected_language_is_announced_once(self, tmp_path):
        session, _, transcriber = build(
            tmp_path, ["a", "b different", "c different again"],
            [make_window(seed=i * 10) for i in range(3)],
            language=None,
        )
        # The real transcriber pins the language on its first confident window.
        original = transcriber.transcribe_segments

        def pinning(audio):
            result = original(audio)
            transcriber.language = "en"
            transcriber.language_probability = 0.99
            return result

        transcriber.transcribe_segments = pinning
        q = session.bus.subscribe()
        session.start()
        session.wait(timeout=5)

        notices = [e for e in of_type(drain(q), "notice")
                   if e["code"] == "language_detected"]
        assert len(notices) == 1
        assert notices[0]["language"] == "en"
        assert notices[0]["level"] == "info"

    def test_nothing_announced_when_the_language_was_given(self, tmp_path):
        session, _, _ = build(
            tmp_path, ["hello"], [make_window()], language="en"
        )
        q = session.bus.subscribe()
        session.start()
        session.wait(timeout=5)

        codes = [e["code"] for e in of_type(drain(q), "notice")]
        assert "language_detected" not in codes


class TestSummary:
    def test_summary_reports_counts_and_paths(self, tmp_path):
        session, _, _ = build(
            tmp_path, ["one two three", "four five six seven"],
            [make_window(seed=0), make_window(seed=10)],
            rejected=2, collapsed=1,
        )
        q = session.bus.subscribe()
        session.start()
        session.wait(timeout=5)

        summary = of_type(drain(q), "summary")[-1]
        assert summary["stats"]["segments"] == 2
        assert summary["stats"]["words"] == 7
        assert summary["stats"]["rejected_segments"] == 2
        assert summary["stats"]["collapsed_segments"] == 1
        assert summary["markdown"].endswith(".md")

    def test_speaker_counts_are_tracked(self, tmp_path):
        session, _, _ = build(
            tmp_path, ["mine", "theirs entirely different"],
            [make_window("You", seed=0), make_window("Meeting", seed=10)],
        )
        session.start()
        session.wait(timeout=5)
        assert session.stats_dict()["speakers"] == {"You": 1, "Meeting": 1}

    def test_transcript_is_written_to_disk(self, tmp_path):
        session, _, _ = build(
            tmp_path, ["written to the file"], [make_window()]
        )
        session.start()
        session.wait(timeout=5)

        assert session.writer.path.exists()
        assert "written to the file" in session.writer.path.read_text(encoding="utf-8")
        data = json.loads(session.writer.sidecar_path.read_text(encoding="utf-8"))
        assert data["segments"][0]["text"] == "written to the file"

    def test_renaming_publishes_and_persists(self, tmp_path):
        session, _, _ = build(tmp_path, ["x"], [make_window()])
        q = session.bus.subscribe()
        session.start()
        session.wait(timeout=5)

        session.set_title("Renamed meeting")
        titles = [e.get("title") for e in of_type(drain(q), "session")]
        assert "Renamed meeting" in titles
        data = json.loads(session.writer.sidecar_path.read_text(encoding="utf-8"))
        assert data["title"] == "Renamed meeting"


class TestSpeakerlessMode:
    def test_single_device_disables_attribution(self, tmp_path):
        cfg = SessionConfig(mic_device=None, system_device=2, raw_dir=tmp_path)
        session = RecordingSession(
            cfg,
            capture_factory=lambda c: FakeCapture([make_window()]),
            transcriber_factory=lambda c: ScriptedTranscriber(["unattributed text"]),
        )
        assert session.use_speakers is False
        q = session.bus.subscribe()
        session.start()
        session.wait(timeout=5)

        speakers = [s["speaker"] for e in of_type(drain(q), "segments")
                    for s in e["segments"]]
        assert speakers == ["Transcript"]

    def test_explicitly_disabled(self, tmp_path):
        session, _, _ = build(
            tmp_path, ["text"], [make_window()], use_speakers=False
        )
        assert session.use_speakers is False


class TestTailRevision:
    """When a window re-transcribes the last batch, consumers are told to
    replace what they have rather than being handed overlapping text."""

    def test_a_revise_event_is_published(self, tmp_path):
        session, _, _ = build(
            tmp_path,
            ["Airbnb has the same purpose",
             "Airbnb has the same problem when two guests book"],
            [make_window("Meeting", seed=0), make_window("Meeting", seed=10)],
        )
        q = session.bus.subscribe()
        session.start()
        session.wait(timeout=5)

        events = drain(q)
        revisions = of_type(events, "revise")
        assert revisions, [e["type"] for e in events]

        first_ids = [s["id"] for e in of_type(events, "segments") for s in e["segments"]]
        assert revisions[0]["from_id"] == min(first_ids)

    def test_the_revision_carries_the_corrected_text(self, tmp_path):
        session, _, _ = build(
            tmp_path,
            ["we will discuss the migra", "we will discuss the migration next week"],
            [make_window("You", seed=0), make_window("You", seed=10)],
        )
        q = session.bus.subscribe()
        session.start()
        session.wait(timeout=5)

        revision = of_type(drain(q), "revise")[0]
        text = " ".join(s["text"] for s in revision["segments"])
        assert "migration next week" in text

    def test_ids_in_a_revision_are_new(self, tmp_path):
        """Replaced text gets fresh ids, so nothing stale lingers on screen."""
        session, _, _ = build(
            tmp_path,
            ["alpha beta", "alpha beta gamma delta"],
            [make_window("You", seed=0), make_window("You", seed=10)],
        )
        q = session.bus.subscribe()
        session.start()
        session.wait(timeout=5)

        events = drain(q)
        revisions = of_type(events, "revise")
        if revisions:
            revision = revisions[0]
            assert all(s["id"] >= revision["from_id"] for s in revision["segments"])

    def test_a_silence_gap_publishes_no_revision(self, tmp_path):
        session, _, _ = build(
            tmp_path,
            ["first utterance", "", "a separate later remark"],
            [make_window("You", seed=i * 10) for i in range(3)],
        )
        q = session.bus.subscribe()
        session.start()
        session.wait(timeout=5)
        assert of_type(drain(q), "revise") == []

    def test_stats_follow_the_file_after_a_revision(self, tmp_path):
        """Counts are recomputed from disk, not accumulated."""
        session, _, _ = build(
            tmp_path,
            ["one two three", "one two three four five"],
            [make_window("You", seed=0), make_window("You", seed=10)],
        )
        session.start()
        session.wait(timeout=5)

        written = session.writer.segments
        assert session.stats.segments == len(written)
        assert session.stats.words == sum(len(s["text"].split()) for s in written)


class FakeSummarizer:
    """Records refine calls and returns a growing canned summary."""

    def __init__(self, fail_with=None, delay=0.0):
        self.calls = []
        self.fail_with = fail_with
        self.delay = delay

    def refine(self, existing, new_text):
        import time as _time

        from scribe_audio.summarizer import Summary

        if self.delay:
            _time.sleep(self.delay)
        self.calls.append((existing, new_text))
        if self.fail_with:
            raise self.fail_with
        return Summary(text=f"## Summary\nfolded {len(self.calls)} parts",
                       model="fake-model", kind="live")


class TestRunningSummary:
    """The live summary is maintained on the server, so it survives a browser
    reload and is part of the recording rather than of a tab.

    The refresh logic is driven directly rather than through the worker thread:
    the fake capture finishes in microseconds, so any timing-based test here
    would be measuring the test harness.
    """

    def _recorded(self, tmp_path, script, summarizer, **cfg):
        """Run a recording to completion, then hand back the session."""
        from fakes import FakeCapture, ScriptedTranscriber

        windows = [make_window("Meeting", seed=i * 10) for i in range(len(script))]
        cfg.setdefault("notes_min_new_words", 1)
        config = SessionConfig(
            mic_device=1, system_device=2, raw_dir=tmp_path,
            window=10.0, step=5.0, level_interval=0.01,
            notes_interval=3600, **cfg,
        )
        session = RecordingSession(
            config,
            capture_factory=lambda c: FakeCapture(windows),
            transcriber_factory=lambda c: ScriptedTranscriber(script),
            summarizer=summarizer,
        )
        session.start()
        assert session.wait(timeout=10)
        return session

    def test_a_refresh_publishes_the_summary(self, tmp_path):
        fake = FakeSummarizer()
        session = self._recorded(
            tmp_path, ["something was said about the migration project"], fake,
        )
        q = session.bus.subscribe(replay=False)
        assert session._refresh_notes() is True

        notes = of_type(drain(q), "notes")
        assert notes and "folded" in notes[0]["text"]
        assert notes[0]["model"] == "fake-model"
        assert notes[0]["covered_words"] > 0

    def test_only_the_new_part_is_sent(self, tmp_path):
        """Re-reading the whole transcript every  cycle is what makes the latency
        grow until it cannot keep up with the meeting."""
        fake = FakeSummarizer()
        session = self._recorded(
            tmp_path, ["the first thing that was said here"], fake,
        )
        session._refresh_notes()
        first_sent = fake.calls[0][1]
        assert "first thing" in first_sent

        session.writer.append_segments(
            [{"speaker": "You", "text": "a brand new sentence arrives later"}]
        )
        assert session._refresh_notes() is True
        second_sent = fake.calls[1][1]
        assert "brand new sentence" in second_sent
        assert "first thing" not in second_sent

    def test_the_previous_summary_is_passed_back_in(self, tmp_path):
        """Continuity: the model must see what it already wrote."""
        fake = FakeSummarizer()
        session = self._recorded(tmp_path, ["an opening statement was made"], fake)
        session._refresh_notes()

        session.writer.append_segments(
            [{"speaker": "You", "text": "and then something else entirely"}]
        )
        session._refresh_notes()
        assert "folded" in fake.calls[1][0]

    def test_nothing_new_means_no_call(self, tmp_path):
        fake = FakeSummarizer()
        session = self._recorded(tmp_path, ["something was said here"], fake)
        assert session._refresh_notes() is True
        assert session._refresh_notes() is False
        assert len(fake.calls) == 1

    def test_too_little_new_speech_is_skipped(self, tmp_path):
        fake = FakeSummarizer()
        session = self._recorded(
            tmp_path, ["three words only"], fake, notes_min_new_words=500,
        )
        assert session._refresh_notes() is False
        assert fake.calls == []

    def test_a_shortened_transcript_is_rebased_not_resent(self, tmp_path):
        """A tail revision can shorten the transcript; that is not new material."""
        fake = FakeSummarizer()
        session = self._recorded(tmp_path, ["something said"], fake)
        session._notes_covered_words = 10_000

        assert session._refresh_notes() is False
        assert session._notes_covered_words < 10_000
        assert fake.calls == []

    def test_a_failing_summarizer_is_reported_once(self, tmp_path):
        from scribe_audio.summarizer import SummarizerError

        fake = FakeSummarizer(fail_with=SummarizerError("model unloaded"))
        session = self._recorded(tmp_path, ["something was said here"], fake)
        q = session.bus.subscribe(replay=False)

        for _ in range(3):
            try:
                session._refresh_notes()
            except SummarizerError as exc:
                session._report_notes_error(exc)

        failures = [e for e in of_type(drain(q), "notice")
                    if e["code"] == "notes_failed"]
        assert len(failures) == 1, "a broken summariser must not spam every cycle"

    def test_a_failing_summarizer_does_not_harm_the_recording(self, tmp_path):
        from scribe_audio.summarizer import SummarizerError

        fake = FakeSummarizer(fail_with=SummarizerError("model unloaded"))
        session = self._recorded(tmp_path, ["something was said here"], fake)
        assert session.state == STATE_STOPPED
        assert session.writer.path.exists()
        assert "something was said" in session.writer.path.read_text(encoding="utf-8")

    def test_no_summarizer_means_no_worker_and_no_notes(self, tmp_path):
        session = self._recorded(tmp_path, ["something said"], None)
        assert session.notes is None
        assert session._notes_thread is None

    def test_a_worker_thread_is_started_when_a_summarizer_is_present(self, tmp_path):
        fake = FakeSummarizer()
        session = self._recorded(tmp_path, ["something said"], fake)
        assert session._notes_thread is not None

    def test_the_running_summary_is_saved_with_the_recording(self, tmp_path):
        import json

        fake = FakeSummarizer()
        session = self._recorded(
            tmp_path, ["something was said about the project"], fake,
        )
        session._refresh_notes()
        session._persist_notes()

        data = json.loads(session.writer.sidecar_path.read_text(encoding="utf-8"))
        assert "folded" in data["summary"]["text"]
        assert data["summary"]["kind"] == "live"

    def test_an_existing_full_summary_is_not_overwritten(self, tmp_path):
        from scribe_audio import library

        fake = FakeSummarizer()
        session = self._recorded(tmp_path, ["something was said here"], fake)
        library.set_summary(
            tmp_path, session.writer.path.name,
            {"text": "a careful full summary", "model": "m", "kind": "full"},
        )
        session._refresh_notes()
        session._persist_notes()

        stored = library.get_summary(tmp_path, session.writer.path.name)
        assert stored["text"] == "a careful full summary"


class TestAutomaticNotesAreOptIn:
    """Refreshing every few seconds kept a 20GB model permanently resident."""

    def test_no_worker_without_an_interval(self, tmp_path):
        config = SessionConfig(mic_device=1, system_device=2, raw_dir=tmp_path)
        assert config.notes_interval == 0.0

        session = RecordingSession(config, summarizer=object())
        session._start_notes_worker()
        assert session._notes_thread is None

    def test_a_positive_interval_turns_it_on(self, tmp_path):
        config = SessionConfig(mic_device=1, system_device=2, raw_dir=tmp_path,
                               notes_interval=3600)
        session = RecordingSession(config, summarizer=object())
        session._start_notes_worker()
        assert session._notes_thread is not None
        session._stop_requested.set()


class TestAdaptiveStep:
    """Overlap is duplicated work: at a 30s window and 15s step every second
    of audio is transcribed twice. Giving some of it up is far cheaper than
    letting the transcript fall minutes behind the conversation."""

    def test_falling_behind_widens_the_step(self, tmp_path):
        from scribe_audio.session import BACKLOG_WARN_SECONDS

        session, capture, _ = build(
            tmp_path, ["some speech"],
            [make_window(backlog=BACKLOG_WARN_SECONDS + 10)],
            step=5.0, window=10.0,
        )
        session.start()
        session.wait(timeout=5)

        assert capture.step_changes, "the step should have been widened"
        assert capture.step_changes[0] > 5.0

    def test_the_step_never_exceeds_the_window(self, tmp_path):
        """A step wider than the window would skip audio outright."""
        from scribe_audio.session import BACKLOG_WARN_SECONDS

        session, capture, _ = build(
            tmp_path, ["a", "b different", "c different again", "d more still"],
            [make_window(seed=i * 10, backlog=BACKLOG_WARN_SECONDS + 100)
             for i in range(4)],
            step=8.0, window=10.0,
        )
        session.start()
        session.wait(timeout=5)

        assert all(s <= 10.0 for s in capture.step_changes), capture.step_changes

    def test_keeping_up_leaves_the_step_alone(self, tmp_path):
        session, capture, _ = build(
            tmp_path, ["some speech"], [make_window(backlog=0.0)],
            step=5.0, window=10.0,
        )
        session.start()
        session.wait(timeout=5)
        assert capture.step_changes == []

    def test_catching_up_restores_the_step(self, tmp_path):
        from scribe_audio.session import BACKLOG_WARN_SECONDS

        session, capture, _ = build(
            tmp_path,
            ["first stretch", "second entirely different stretch"],
            [make_window(seed=0, backlog=BACKLOG_WARN_SECONDS + 10),
             make_window(seed=10, backlog=0.0)],
            step=5.0, window=10.0,
        )
        session.start()
        session.wait(timeout=5)

        assert len(capture.step_changes) >= 2
        assert capture.step_changes[-1] == pytest.approx(5.0)

    def test_adaptation_can_be_turned_off(self, tmp_path):
        from scribe_audio.session import BACKLOG_WARN_SECONDS

        session, capture, _ = build(
            tmp_path, ["some speech"],
            [make_window(backlog=BACKLOG_WARN_SECONDS + 10)],
            step=5.0, window=10.0, adaptive_step=False,
        )
        session.start()
        session.wait(timeout=5)
        assert capture.step_changes == []


class TestCapturedAudioRetention:
    """A recording of a meeting is more sensitive than a transcript of it, so
    the audio goes away unless it is asked for."""

    def test_the_audio_is_deleted_by_default(self, tmp_path):
        session, capture, _ = build(tmp_path, ["some speech"], [make_window()])
        session.start()
        session.wait(timeout=5)

        assert capture.spool_discarded is True
        assert capture.spool_kept_to is None

    def test_the_audio_can_be_kept_beside_the_transcript(self, tmp_path):
        session, capture, _ = build(
            tmp_path, ["some speech"], [make_window()], keep_audio=True
        )
        session.start()
        session.wait(timeout=5)

        assert capture.spool_kept_to is not None
        assert capture.spool_kept_to.suffix == ".wav"
        assert capture.spool_kept_to.stem == session.writer.path.stem, \
            "the audio should be findable from the transcript name"
        assert capture.spool_discarded is False

    def test_a_failed_keep_falls_back_to_deleting(self, tmp_path):
        """Never leave a recording lying around because a move failed."""
        session, capture, _ = build(
            tmp_path, ["some speech"], [make_window()], keep_audio=True
        )

        def boom(destination):
            raise OSError("no space left on device")

        capture.keep_spool = boom
        q = session.bus.subscribe()
        session.start()
        session.wait(timeout=5)

        assert capture.spool_discarded is True
        codes = [e["code"] for e in of_type(drain(q), "notice")]
        assert "audio_keep_failed" in codes


class TestBacklogIsDrainedBeforeFinishing:
    def test_stopping_announces_the_remaining_work(self, tmp_path):
        """A meeting that ends ten minutes behind takes ten more minutes and
        loses nothing; saying so beats looking frozen."""
        from scribe_audio.session import BACKLOG_WARN_SECONDS

        session, capture, _ = build(
            tmp_path, ["some speech"], [make_window()]
        )
        capture.backlog_seconds = BACKLOG_WARN_SECONDS + 60
        q = session.bus.subscribe()
        session.start()
        session.wait(timeout=5)

        drains = [e for e in of_type(drain(q), "notice")
                  if e["code"] == "draining"]
        assert drains, "the remaining backlog should be announced"
        assert "still to process" in drains[0]["message"]

    def test_no_announcement_when_there_is_nothing_left(self, tmp_path):
        session, _, _ = build(tmp_path, ["some speech"], [make_window()])
        q = session.bus.subscribe()
        session.start()
        session.wait(timeout=5)

        assert [e for e in of_type(drain(q), "notice")
                if e["code"] == "draining"] == []


class TestTheOverlapTaxIsStructural:
    """Each second of audio is transcribed window/step times, and the extra
    pass is what pays for boundary correction: align_tail() only replaces a
    batch it re-transcribes in full, which needs window >= 2 * step. So the
    factor of two is the correction's price, not slack to tune away — a point
    worth pinning, because tuning it away looks like free throughput.
    """

    def test_the_default_configuration_can_revise(self, tmp_path):
        session, _, _ = build(tmp_path, [], [], window=30.0, step=15.0)
        assert session._revision_possible(15.0)

    def test_widening_past_half_the_window_ends_revision(self, tmp_path):
        session, _, _ = build(tmp_path, [], [], window=30.0, step=15.0)
        assert not session._revision_possible(16.0)
        assert not session._revision_possible(30.0)

    def test_the_cost_ratio_is_window_over_step(self, tmp_path):
        """2.0x realtime at the defaults, which is why large-v3 — measured at
        exactly 2.0x on an M5 Pro — cannot keep up even on fast hardware."""
        session, _, _ = build(tmp_path, [], [], window=30.0, step=15.0)
        assert session.config.window / session.config.step == pytest.approx(2.0)

    def test_losing_revision_is_stated_when_it_happens(self, tmp_path):
        from scribe_audio.session import BACKLOG_WARN_SECONDS

        session, _, _ = build(
            tmp_path, ["some speech"],
            [make_window(backlog=BACKLOG_WARN_SECONDS + 10)],
            window=10.0, step=5.0,
        )
        q = session.bus.subscribe()
        session.start()
        session.wait(timeout=5)

        widened = [e for e in of_type(drain(q), "notice")
                   if e["code"] == "step_widened"]
        assert widened
        assert widened[0]["revision_disabled"] is True
        assert "corrections are off" in widened[0]["message"]
