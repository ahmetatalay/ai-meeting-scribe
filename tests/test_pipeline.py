"""End-to-end transcript tests: window -> transcribe -> attribute -> dedup -> file.

These drive a real ``RecordingSession`` with fake audio and a scripted
transcriber, and assert on the finished markdown and sidecar — the only things
a user ever sees. This is the integration that was broken in production:
overlapping windows that Whisper re-words slightly.
"""

import json
import re

from fakes import build_session, make_window

YOU = "You"
MEETING = "Meeting"


def spoken(body: str) -> str:
    """The transcript's words, with markdown layout normalised away.

    A revision and the text appended after it land on separate lines, which
    markdown renders as one paragraph — so assertions are about content, not
    line breaks.
    """
    text = re.sub(r"\A---\n.*?\n---\n", "", body, flags=re.S)
    text = re.sub(r"^#.*$", "", text, flags=re.M)
    text = re.sub(r"\*\*[^*]+\*\*:", "", text)
    return " ".join(text.split())


def run(tmp_path, script, speakers, **cfg):
    """Drive a session over one window per speaker turn; return (body, session)."""
    windows = [make_window(s, seed=i * 10) for i, s in enumerate(speakers)]
    session, _, _ = build_session(tmp_path, script, windows, **cfg)
    session.start()
    assert session.wait(timeout=10), "session did not finish"

    if session.writer is None or session.writer.path is None:
        return "", session
    return session.writer.path.read_text(encoding="utf-8"), session


class TestOverlapRemoval:
    def test_each_phrase_is_written_exactly_once(self, tmp_path):
        script = [
            "we should ship the build today",
            "ship the build today and tell the team",
            "and tell the team about the new release",
            "about the new release tomorrow morning",
        ]
        body, _ = run(tmp_path, script, [YOU] * 4)

        for phrase in ("ship the build today", "tell the team",
                       "about the new release", "tomorrow morning"):
            assert body.count(phrase) == 1, f"{phrase!r} appears {body.count(phrase)}x"

    def test_reworded_overlap_is_still_removed(self, tmp_path):
        """Whisper renders the overlap differently on the second pass."""
        script = [
            "the cache layer is returning stale keys",
            "cache layer is returning stale keys again after the deploy",
            "stale keys again after the deployment so we rolled back",
        ]
        body, _ = run(tmp_path, script, [YOU] * 3)

        assert body.count("returning stale keys") == 1
        assert "rolled back" in body

    def test_no_duplication_across_a_speaker_change(self, tmp_path):
        """The precise case the retired fallback duplicated a whole turn on."""
        script = [
            "does the dashboard show the new metric",
            "show the new metric yes it went live this morning",
        ]
        body, _ = run(tmp_path, script, [YOU, MEETING])

        assert body.count("show the new metric") == 1
        assert "it went live this morning" in body

    def test_a_window_with_no_new_content_writes_nothing(self, tmp_path):
        script = ["we are waiting on the migration"] * 2
        body, _ = run(tmp_path, script, [YOU] * 2)
        assert body.count("waiting on the migration") == 1

    def test_silent_windows_are_skipped(self, tmp_path):
        script = ["something was said", "", "", "a different remark entirely"]
        body, _ = run(tmp_path, script, [YOU] * 4)
        assert "something was said" in body
        assert "a different remark entirely" in body

    def test_speech_after_silence_is_not_eaten_by_stale_text(self, tmp_path):
        """After a pause the previous text is older than the current window.

        Keeping it would let a new utterance that merely resembles the old one
        be mistaken for an overlap and dropped.
        """
        script = ["something was said", "", "something else was said"]
        body, _ = run(tmp_path, script, [YOU] * 3)
        assert "something was said" in body
        assert "something else was said" in body


class TestTailRevision:
    """A window's last words are guessed without following context; the next
    window has it. Observed live: "Airbnb has the same purpose." became
    "...the same problem when two guests book..." on the second pass."""

    def test_a_wrong_guess_at_the_boundary_is_replaced(self, tmp_path):
        script = [
            "Airbnb has the same purpose",
            "Airbnb has the same problem when two guests book",
        ]
        body, _ = run(tmp_path, script, [MEETING] * 2)

        assert "purpose" not in body
        assert "Airbnb has the same problem when two guests book" in spoken(body)

    def test_the_corrected_text_appears_exactly_once(self, tmp_path):
        script = [
            "the cache layer is returning stale keys",
            "the cache layer is returning stale keys again after the deploy",
        ]
        body, _ = run(tmp_path, script, [YOU] * 2)
        assert spoken(body).count("returning stale keys") == 1
        assert "again after the deploy" in spoken(body)

    def test_a_truncated_final_word_is_completed(self, tmp_path):
        """The window boundary cut a word in half on the first pass."""
        script = [
            "we will discuss the migra",
            "we will discuss the migration next week",
        ]
        body, _ = run(tmp_path, script, [YOU] * 2)

        assert "migra\n" not in body
        assert "we will discuss the migration next week" in spoken(body)

    def test_revision_may_change_the_speaker(self, tmp_path):
        script = ["who said this", "who said this was fine"]
        body, _ = run(tmp_path, script, [YOU, MEETING])
        # Whoever it ends up attributed to, it must appear once under one label.
        assert spoken(body).count("who said this") == 1

    def test_a_silence_gap_prevents_revision(self, tmp_path):
        """Nothing overlaps across a pause, so the earlier text is final."""
        script = ["first utterance stands", "", "a separate later remark"]
        body, _ = run(tmp_path, script, [YOU] * 3)

        assert "first utterance stands" in body
        assert "a separate later remark" in body

    def test_three_windows_keep_converging(self, tmp_path):
        script = [
            "the deploy failed because of",
            "the deploy failed because of the cache",
            "the deploy failed because of the cache invalidation bug",
        ]
        body, _ = run(tmp_path, script, [MEETING] * 3)

        assert spoken(body).count("the deploy failed because of") == 1
        assert "cache invalidation bug" in spoken(body)

    def test_markdown_stays_well_formed(self, tmp_path):
        script = ["alpha beta", "alpha beta gamma", "alpha beta gamma delta"]
        body, _ = run(tmp_path, script, [YOU] * 3)

        assert body.startswith("---\n")
        assert body.count("**You**:") == 1
        assert spoken(body) == "alpha beta gamma delta"


class TestAttribution:
    def test_both_sides_are_labelled(self, tmp_path):
        script = [
            "how is the migration going",
            "it is going well we finished the first phase",
            "that is good news",
        ]
        body, _ = run(tmp_path, script, [YOU, MEETING, YOU])

        assert "**You**: how is the migration going" in body
        assert "**Meeting**:" in body
        assert "finished the first phase" in body

    def test_a_listening_stretch_is_not_credited_to_you(self, tmp_path):
        """The 97-of-97 failure, end to end."""
        script = [f"their point number {i}" for i in range(6)]
        body, _ = run(tmp_path, script, [MEETING] * 6)

        assert "**You**:" not in body
        assert body.count("**Meeting**:") == 1  # one continuous turn

    def test_same_speaker_continues_one_paragraph(self, tmp_path):
        script = ["I was saying that", "we should ship it"]
        body, _ = run(tmp_path, script, [YOU, YOU])
        assert body.count("**You**:") == 1
        assert "we should ship it" in body

    def test_speakerless_mode_uses_a_neutral_label(self, tmp_path):
        script = ["some audio with no attribution"]
        body, _ = run(tmp_path, script, [YOU], use_speakers=False)
        assert "**Transcript**: some audio with no attribution" in body
        assert "**You**:" not in body


class TestFileOutput:
    def test_frontmatter_records_the_settings(self, tmp_path):
        body, _ = run(tmp_path, ["hello"], [YOU], model="small", language="en",
                      title="Sprint planning")
        assert "title: Sprint planning" in body
        assert "model: small" in body
        assert "language: en" in body

    def test_sidecar_matches_the_markdown(self, tmp_path):
        script = ["first thing said", "first thing said and then another"]
        _, session = run(tmp_path, script, [YOU] * 2)

        data = json.loads(session.writer.sidecar_path.read_text(encoding="utf-8"))
        combined = " ".join(s["text"] for s in data["segments"])
        assert combined.count("first thing said") == 1
        assert "and then another" in combined

    def test_non_english_text_round_trips(self, tmp_path):
        body, _ = run(tmp_path, ["Günaydın, nasılsınız?"], [YOU], language="tr")
        assert "Günaydın, nasılsınız?" in body

    def test_nothing_transcribed_leaves_no_file(self, tmp_path):
        body, session = run(tmp_path, ["", ""], [YOU] * 2)
        assert body == ""
        assert session.writer.path is None
