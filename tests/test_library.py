"""Transcript library tests.

The traversal tests carry the most weight: these functions take a name that
arrives from an HTTP request and then read and *delete* files with it.
"""

import json

import pytest

from scribe_audio.library import (
    LibraryError,
    delete,
    parse_turns,
    read,
    resolve,
    scan,
    scan_detailed,
    set_title,
)


def write_transcript(directory, name, title="A meeting", started="2026-11-23T10:00:00",
                     turns=(("You", "hello there"), ("Meeting", "hi back")),
                     sidecar=True):
    directory.mkdir(parents=True, exist_ok=True)
    body = [f"---\ntitle: {title}\nstarted: {started}\nmodel: small\n---\n", f"# {title}\n"]
    for who, text in turns:
        body.append(f"\n**{who}**: {text}")
    path = directory / name
    path.write_text("\n".join(body) + "\n", encoding="utf-8")

    if sidecar:
        path.with_suffix(".json").write_text(json.dumps({
            "title": title,
            "started": started,
            "model": "small",
            "markdown": name,
            "turns": len(turns),
            "segments": [
                {"speaker": w, "text": t, "start": 1.0, "end": 2.0,
                 "wall": "2026-11-23T10:05:00"}
                for w, t in turns
            ],
        }), encoding="utf-8")
    return path


@pytest.fixture
def library(tmp_path):
    raw = tmp_path / "raw"
    write_transcript(raw, "transcript_2026-11-23_1000.md", title="Sprint planning",
                     started="2026-11-23T10:00:00")
    write_transcript(raw, "transcript_2026-11-22_0900.md", title="One to one",
                     started="2026-11-22T09:00:00",
                     turns=(("You", "about the migration"),))
    return raw


class TestPathSafety:
    """A name from an HTTP request must never reach outside the library."""

    @pytest.mark.parametrize("name", [
        "../../../etc/passwd",
        "../secrets.md",
        "..%2Fescape.md",
        "/etc/passwd",
        "/absolute/path.md",
        "sub/dir/file.md",
        "nested\\file.md",
        ".hidden.md",
        "",
        "noextension",
        "transcript.md.txt",
        "transcript.json",
    ])
    def test_unsafe_names_are_refused(self, library, name):
        with pytest.raises(LibraryError):
            resolve(library, name)

    def test_a_plain_filename_is_accepted(self, library):
        path = resolve(library, "transcript_2026-11-23_1000.md")
        assert path.parent == library.resolve()

    def test_a_symlink_out_of_the_library_is_refused(self, library, tmp_path):
        secret = tmp_path / "outside.md"
        secret.write_text("private", encoding="utf-8")
        (library / "link.md").symlink_to(secret)

        with pytest.raises(LibraryError):
            resolve(library, "link.md")

    def test_delete_refuses_traversal(self, library, tmp_path):
        victim = tmp_path / "victim.md"
        victim.write_text("do not delete", encoding="utf-8")
        with pytest.raises(LibraryError):
            delete(library, "../victim.md")
        assert victim.exists()

    def test_read_refuses_traversal(self, library):
        with pytest.raises(LibraryError):
            read(library, "../../etc/hosts")

    def test_set_title_refuses_traversal(self, library):
        with pytest.raises(LibraryError):
            set_title(library, "../elsewhere.md", "new")


class TestScan:
    def test_lists_every_transcript(self, library):
        entries = scan(library)
        assert len(entries) == 2
        assert {e["title"] for e in entries} == {"Sprint planning", "One to one"}

    def test_newest_first(self, library):
        assert [e["title"] for e in scan(library)] == ["Sprint planning", "One to one"]

    def test_metadata_is_populated(self, library):
        entry = next(e for e in scan(library) if e["title"] == "Sprint planning")
        assert entry["model"] == "small"
        assert entry["words"] > 0
        assert entry["speakers"] == {"You": 1, "Meeting": 1}
        assert entry["has_sidecar"] is True
        assert entry["relative_path"] == "transcript_2026-11-23_1000.md"
        assert entry["duration_seconds"] == pytest.approx(300.0)

    def test_missing_directory_is_empty_not_an_error(self, tmp_path):
        assert scan(tmp_path / "nope") == []

    def test_json_files_are_not_listed_as_transcripts(self, library):
        assert all(e["name"].endswith(".md") for e in scan(library))

    def test_search_matches_the_title(self, library):
        assert [e["title"] for e in scan(library, "sprint")] == ["Sprint planning"]

    def test_search_matches_the_body(self, library):
        found = scan(library, "migration")
        assert [e["title"] for e in found] == ["One to one"]

    def test_search_is_case_insensitive(self, library):
        assert len(scan(library, "SPRINT")) == 1

    def test_search_with_no_match(self, library):
        assert scan(library, "zzzznothing") == []


class TestLegacyTranscripts:
    """Recordings predating sidecars, and markdown dropped in by hand."""

    def test_listed_without_a_sidecar(self, tmp_path):
        raw = tmp_path / "raw"
        write_transcript(raw, "transcript_2026-10-01_1200.md", title="Old one",
                         sidecar=False)
        entry = scan(raw)[0]
        assert entry["title"] == "Old one"
        assert entry["has_sidecar"] is False
        assert entry["speakers"] == {"You": 1, "Meeting": 1}

    def test_old_chunk_marker_format_is_readable(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "legacy.md").write_text(
            "# Transcript — 2026-10-21 12:29\n\n"
            "<!-- chunk 1 @ 12:29:08 -->\n\n"
            "**You**: This is the first thing.\n\n"
            "<!-- chunk 2 @ 12:29:39 -->\n\n"
            "**Meeting**: And the reply.\n",
            encoding="utf-8",
        )
        entry = read(raw, "legacy.md")
        assert [t["speaker"] for t in entry["turns"]] == ["You", "Meeting"]
        assert entry["turns"][0]["text"] == "This is the first thing."

    def test_markdown_with_no_frontmatter_falls_back_to_the_filename(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        # Qualifies by filename pattern, but carries no title anywhere.
        (raw / "transcript_2026-09-09_0808.md").write_text(
            "**You**: just a line\n", encoding="utf-8"
        )
        assert scan(raw)[0]["title"] == "transcript_2026-09-09_0808"

    def test_the_document_heading_is_used_as_a_title(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "transcript_2026-10-21_1229.md").write_text(
            "# Transcript — 2026-10-21 12:29\n\n**You**: hello\n", encoding="utf-8"
        )
        assert scan(raw)[0]["title"] == "Transcript — 2026-10-21 12:29"

    def test_start_time_comes_from_the_filename_when_absent(self, tmp_path):
        """mtime is unreliable — copying a directory rewrites it, which sorted
        old transcripts above new ones."""
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "transcript_2026-10-21_1229.md").write_text(
            "**You**: hello\n", encoding="utf-8"
        )
        assert scan(raw)[0]["started"] == "2026-10-21T12:29:00"

    def test_legacy_and_current_recordings_sort_together_correctly(self, tmp_path):
        raw = tmp_path / "raw"
        # Legacy file, older, but written to disk *last* so its mtime is newest.
        write_transcript(raw, "transcript_2026-11-23_1000.md", title="New one",
                         started="2026-11-23T10:00:00")
        (raw / "transcript_2026-10-01_0900.md").write_text(
            "# Old legacy one\n\n**You**: hello\n", encoding="utf-8"
        )
        assert [e["title"] for e in scan(raw)] == ["New one", "Old legacy one"]

    def test_a_recording_with_an_unparseable_filename_falls_back_to_mtime(self, tmp_path):
        """Qualifies via its sidecar, but no start time can be recovered."""
        import json as _json

        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "random-notes.md").write_text("**You**: hi\n", encoding="utf-8")
        (raw / "random-notes.json").write_text(
            _json.dumps({"markdown": "random-notes.md"}), encoding="utf-8"
        )

        entry = scan(raw)[0]
        assert entry["started"] is None
        assert entry["modified"]


class TestOnlyScribeRecordings:
    """The library offers a Delete button, so it must not list arbitrary
    markdown. Pointing `-r .` at a checkout once listed README.md and
    CHANGELOG.md as recordings."""

    README = (
        "# Scribe — AI Meeting Transcriber\n\n"
        "Scribe transcribes meetings locally.\n\n"
        "Output:\n\n"
        "**You**: How is the migration going?\n"
        "**Meeting**: It is going well.\n"
    )

    def test_a_readme_containing_an_example_transcript_is_skipped(self, tmp_path):
        """Content heuristics fail here: the README *shows* a transcript."""
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "README.md").write_text(self.README, encoding="utf-8")

        result = scan_detailed(raw)
        assert result["transcripts"] == []
        assert result["skipped"] == 1

    def test_unrelated_notes_are_skipped(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "CHANGELOG.md").write_text("# Changelog\n\n## 1.0\n", encoding="utf-8")
        (raw / "shopping-list.md").write_text("- milk\n- bread\n", encoding="utf-8")
        assert scan_detailed(raw)["skipped"] == 2

    def test_recordings_alongside_notes_are_still_listed(self, tmp_path):
        raw = tmp_path / "raw"
        write_transcript(raw, "transcript_2026-11-23_1000.md", title="Real one")
        (raw / "README.md").write_text(self.README, encoding="utf-8")

        result = scan_detailed(raw)
        assert [t["title"] for t in result["transcripts"]] == ["Real one"]
        assert result["skipped"] == 1

    def test_a_sidecar_is_sufficient_evidence(self, tmp_path):
        raw = tmp_path / "raw"
        write_transcript(raw, "renamed-by-hand.md", title="Standup")
        assert [t["title"] for t in scan(raw)] == ["Standup"]

    def test_the_transcript_filename_pattern_is_sufficient(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "transcript_2026-10-21_1229.md").write_text(
            "**You**: hello\n", encoding="utf-8"
        )
        assert len(scan(raw)) == 1

    def test_a_legacy_transcript_heading_is_sufficient(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "notes-from-call.md").write_text(
            "# Transcript — 2026-10-21 12:29\n\n**You**: hello\n", encoding="utf-8"
        )
        assert len(scan(raw)) == 1

    def test_reading_a_non_recording_is_refused(self, tmp_path):
        """A direct URL must not bypass the listing filter."""
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "README.md").write_text(self.README, encoding="utf-8")
        with pytest.raises(LibraryError):
            read(raw, "README.md")

    def test_deleting_a_non_recording_is_refused(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        readme = raw / "README.md"
        readme.write_text(self.README, encoding="utf-8")

        with pytest.raises(LibraryError):
            delete(raw, "README.md")
        assert readme.exists(), "the README must survive"

    def test_renaming_a_non_recording_adopts_it(self, tmp_path):
        """Renaming writes a sidecar, which then makes it a recording.

        This is the deliberate escape hatch for markdown you *want* managed.
        """
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "handwritten.md").write_text("**You**: notes\n", encoding="utf-8")
        set_title(raw, "handwritten.md", "Adopted")
        assert [t["title"] for t in scan(raw)] == ["Adopted"]


class TestRead:
    def test_returns_turns(self, library):
        entry = read(library, "transcript_2026-11-23_1000.md")
        assert [t["speaker"] for t in entry["turns"]] == ["You", "Meeting"]
        assert entry["turns"][0]["text"] == "hello there"

    def test_missing_file_raises(self, library):
        with pytest.raises(FileNotFoundError):
            read(library, "transcript_1999-01-01_0000.md")

    def test_continuation_lines_join_the_previous_turn(self):
        turns = parse_turns(
            "**You**: first part\nsecond part\n\n**Meeting**: their reply\n"
        )
        assert turns[0]["text"] == "first part second part"
        assert turns[1]["text"] == "their reply"

    def test_headings_are_skipped(self):
        assert parse_turns("# Title\n\n**You**: hello\n")[0]["text"] == "hello"

    def test_empty_body(self):
        assert parse_turns("") == []


class TestSetTitle:
    def test_updates_the_sidecar_only(self, library):
        name = "transcript_2026-11-23_1000.md"
        before = (library / name).read_text(encoding="utf-8")

        set_title(library, name, "Quarterly review")

        data = json.loads((library / name).with_suffix(".json").read_text(encoding="utf-8"))
        assert data["title"] == "Quarterly review"
        # The transcript itself is untouched.
        assert (library / name).read_text(encoding="utf-8") == before

    def test_the_new_title_shows_in_the_listing(self, library):
        set_title(library, "transcript_2026-11-23_1000.md", "Renamed")
        assert "Renamed" in {e["title"] for e in scan(library)}

    def test_creates_a_sidecar_when_there_is_none(self, tmp_path):
        raw = tmp_path / "raw"
        write_transcript(raw, "plain.md", title="Plain", sidecar=False)
        set_title(raw, "plain.md", "Now named")

        data = json.loads((raw / "plain.json").read_text(encoding="utf-8"))
        assert data["title"] == "Now named"
        assert data["markdown"] == "plain.md"

    def test_empty_title_is_refused(self, library):
        with pytest.raises(LibraryError):
            set_title(library, "transcript_2026-11-23_1000.md", "   ")

    def test_no_temp_file_is_left_behind(self, library):
        set_title(library, "transcript_2026-11-23_1000.md", "X")
        assert not list(library.glob("*.tmp"))


class TestDelete:
    def test_removes_the_transcript_and_its_sidecar(self, library):
        name = "transcript_2026-11-23_1000.md"
        result = delete(library, name)

        assert not (library / name).exists()
        assert not (library / name).with_suffix(".json").exists()
        assert sorted(result["deleted"]) == sorted(
            [name, name.replace(".md", ".json")]
        )

    def test_leaves_other_transcripts_alone(self, library):
        delete(library, "transcript_2026-11-23_1000.md")
        remaining = scan(library)
        assert [e["title"] for e in remaining] == ["One to one"]

    def test_works_without_a_sidecar(self, tmp_path):
        raw = tmp_path / "raw"
        write_transcript(raw, "plain.md", sidecar=False)
        assert delete(raw, "plain.md")["deleted"] == ["plain.md"]

    def test_missing_file_raises(self, library):
        with pytest.raises(FileNotFoundError):
            delete(library, "transcript_1999-01-01_0000.md")
