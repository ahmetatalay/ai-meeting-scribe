"""Recording engine tests.

Fakes stand in for PortAudio and Whisper, so the whole lifecycle — model
loading, the capture loop, health notices, the tail flush and the event stream —
runs deterministically in-process.
"""

import json
import queue

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
