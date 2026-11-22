"""Transcript writing tests."""

import json

import pytest

from scribe_audio.writer import TranscriptWriter


@pytest.fixture
def raw_dir(tmp_path):
    return tmp_path / "raw"


class TestFileCreation:
    def test_no_file_until_something_is_written(self, raw_dir):
        writer = TranscriptWriter(raw_dir=raw_dir)
        assert writer.path is None
        assert list(raw_dir.iterdir()) == []

    def test_directory_is_created(self, tmp_path):
        nested = tmp_path / "a" / "b" / "raw"
        TranscriptWriter(raw_dir=nested)
        assert nested.is_dir()

    def test_frontmatter_and_heading(self, raw_dir):
        writer = TranscriptWriter(raw_dir=raw_dir, model="small", language="en")
        writer.append_segments([{"speaker": "You", "text": "hello"}])
        body = writer.path.read_text(encoding="utf-8")

        assert body.startswith("---\n")
        assert "title: Meeting " in body
        assert "model: small" in body
        assert "language: en" in body
        assert "# Meeting " in body

    def test_custom_title(self, raw_dir):
        writer = TranscriptWriter(raw_dir=raw_dir, title="Sprint planning")
        writer.append_segments([{"speaker": "You", "text": "hello"}])
        assert "title: Sprint planning" in writer.path.read_text(encoding="utf-8")


class TestCollisionSafety:
    def test_two_recordings_in_the_same_minute_do_not_overwrite(self, raw_dir):
        """The previous version opened a minute-granular name with "w"."""
        first = TranscriptWriter(raw_dir=raw_dir)
        first.append_segments([{"speaker": "You", "text": "first meeting"}])

        second = TranscriptWriter(raw_dir=raw_dir)
        second.started = first.started  # same minute
        second.append_segments([{"speaker": "You", "text": "second meeting"}])

        assert first.path != second.path
        assert "first meeting" in first.path.read_text(encoding="utf-8")
        assert "second meeting" in second.path.read_text(encoding="utf-8")

    def test_an_existing_unrelated_file_is_never_clobbered(self, raw_dir):
        raw_dir.mkdir(parents=True)
        writer = TranscriptWriter(raw_dir=raw_dir)
        stem = f"transcript_{writer.started.strftime('%Y-%m-%d_%H%M')}.md"
        squatter = raw_dir / stem
        squatter.write_text("precious", encoding="utf-8")

        writer.append_segments([{"speaker": "You", "text": "new"}])
        assert squatter.read_text(encoding="utf-8") == "precious"
        assert writer.path != squatter


class TestSpeakerFormatting:
    def test_speaker_prefix_per_turn(self, raw_dir):
        writer = TranscriptWriter(raw_dir=raw_dir)
        writer.append_segments([
            {"speaker": "You", "text": "my question"},
            {"speaker": "Meeting", "text": "their answer"},
        ])
        body = writer.path.read_text(encoding="utf-8")
        assert "**You**: my question" in body
        assert "**Meeting**: their answer" in body

    def test_same_speaker_across_writes_continues_the_paragraph(self, raw_dir):
        """A sentence spanning two windows must read as one turn."""
        writer = TranscriptWriter(raw_dir=raw_dir)
        writer.append_segments([{"speaker": "You", "text": "I was saying that"}])
        writer.append_segments([{"speaker": "You", "text": "we should ship it"}])

        body = writer.path.read_text(encoding="utf-8")
        assert body.count("**You**:") == 1
        assert "we should ship it" in body

    def test_speaker_change_starts_a_new_turn(self, raw_dir):
        writer = TranscriptWriter(raw_dir=raw_dir)
        writer.append_segments([{"speaker": "You", "text": "a"}])
        writer.append_segments([{"speaker": "Meeting", "text": "b"}])
        writer.append_segments([{"speaker": "You", "text": "c"}])

        body = writer.path.read_text(encoding="utf-8")
        assert body.count("**You**:") == 2
        assert body.count("**Meeting**:") == 1

    def test_blank_segments_write_nothing(self, raw_dir):
        writer = TranscriptWriter(raw_dir=raw_dir)
        writer.append_segments([{"speaker": "You", "text": "   "}])
        assert writer.path is None

    def test_empty_batch_writes_nothing(self, raw_dir):
        writer = TranscriptWriter(raw_dir=raw_dir)
        writer.append_segments([])
        assert writer.path is None


class TestEncoding:
    def test_non_english_text_round_trips(self, raw_dir):
        writer = TranscriptWriter(raw_dir=raw_dir, language="tr")
        writer.append_segments([
            {"speaker": "You", "text": "Günaydın, nasılsınız?"},
            {"speaker": "Meeting", "text": "很好，谢谢"},
        ])
        body = writer.path.read_text(encoding="utf-8")
        assert "Günaydın, nasılsınız?" in body
        assert "很好，谢谢" in body


class TestSidecar:
    def test_sidecar_holds_structured_segments(self, raw_dir):
        writer = TranscriptWriter(raw_dir=raw_dir, model="small")
        writer.append_segments([
            {"speaker": "You", "text": "hello", "start": 1.0, "end": 2.0},
            {"speaker": "Meeting", "text": "hi", "start": 2.0, "end": 3.0},
        ])

        data = json.loads(writer.sidecar_path.read_text(encoding="utf-8"))
        assert data["model"] == "small"
        assert data["markdown"] == writer.path.name
        assert len(data["segments"]) == 2
        assert data["segments"][0]["speaker"] == "You"
        assert data["segments"][0]["start"] == 1.0
        assert data["turns"] == 2

    def test_sidecar_sits_next_to_the_markdown(self, raw_dir):
        writer = TranscriptWriter(raw_dir=raw_dir)
        writer.append_segments([{"speaker": "You", "text": "x"}])
        assert writer.sidecar_path.parent == writer.path.parent
        assert writer.sidecar_path.stem == writer.path.stem

    def test_sidecar_is_valid_json_after_every_append(self, raw_dir):
        writer = TranscriptWriter(raw_dir=raw_dir)
        for i in range(5):
            writer.append_segments([{"speaker": "You", "text": f"turn {i}"}])
            json.loads(writer.sidecar_path.read_text(encoding="utf-8"))

    def test_no_temp_file_is_left_behind(self, raw_dir):
        writer = TranscriptWriter(raw_dir=raw_dir)
        writer.append_segments([{"speaker": "You", "text": "x"}])
        assert not list(raw_dir.glob("*.tmp"))

    def test_rename_updates_the_sidecar(self, raw_dir):
        writer = TranscriptWriter(raw_dir=raw_dir)
        writer.append_segments([{"speaker": "You", "text": "x"}])
        writer.set_title("Quarterly review")

        data = json.loads(writer.sidecar_path.read_text(encoding="utf-8"))
        assert data["title"] == "Quarterly review"


class TestPlainMode:
    def test_append_text_uses_a_neutral_label(self, raw_dir):
        writer = TranscriptWriter(raw_dir=raw_dir)
        writer.append_text("some unattributed speech")
        body = writer.path.read_text(encoding="utf-8")
        assert "**Transcript**: some unattributed speech" in body
