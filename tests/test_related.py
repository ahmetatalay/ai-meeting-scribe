"""Related-recording tests.

This is the deliberate alternative to feeding several transcripts to a model:
the overlap is computed here, deterministically, and presented as links with
the shared terms visible so the reader can judge them.
"""

import json

import pytest

from scribe_audio.related import find, salient, terms
from collections import Counter


def write(directory, name, body, title=None, started="2026-11-01T10:00:00"):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(
        f"---\ntitle: {title or name}\nstarted: {started}\n---\n\n"
        f"# {title or name}\n\n**You**: {body}\n",
        encoding="utf-8",
    )
    path.with_suffix(".json").write_text(
        json.dumps({"title": title or name, "started": started,
                    "markdown": name, "segments": []}),
        encoding="utf-8",
    )
    return path


class TestTerms:
    def test_stopwords_are_dropped(self):
        counted = terms("the and that with migration migration cluster")
        assert "the" not in counted
        assert counted["migration"] == 2
        assert counted["cluster"] == 1

    def test_short_words_are_dropped(self):
        assert "ab" not in terms("ab migration")

    def test_case_is_folded(self):
        counted = terms("Migration MIGRATION migration")
        assert counted["migration"] == 3

    def test_turkish_characters_survive(self):
        counted = terms("görüşme planlama görüşme")
        assert counted["görüşme"] == 2
        assert counted["planlama"] == 1

    def test_turkish_stopwords_are_dropped(self):
        assert "için" not in terms("için migration")


class TestSalient:
    def test_terms_common_to_every_transcript_are_suppressed(self):
        """Plain frequency would surface words every meeting contains."""
        counts = Counter({"migration": 5, "everywhere": 50})
        document_frequency = Counter({"migration": 1, "everywhere": 10})
        ranked = dict(salient(counts, document_frequency, documents=10))

        assert "migration" in ranked
        assert ranked.get("everywhere", 0) < ranked["migration"]

    def test_a_term_in_all_documents_scores_nothing(self):
        counts = Counter({"ubiquitous": 100})
        df = Counter({"ubiquitous": 10})
        assert salient(counts, df, documents=9) == []


class TestFind:
    def test_two_meetings_about_the_same_thing_are_linked(self, tmp_path):
        raw = tmp_path / "raw"
        write(raw, "transcript_2026-11-01_1000.md",
              "we discussed the kubernetes cluster migration and the rollout plan "
              "for the kubernetes migration across regions",
              title="Migration planning")
        write(raw, "transcript_2026-11-02_1000.md",
              "following up on the kubernetes cluster migration rollout and the "
              "regions we still have to migrate",
              title="Migration follow-up")
        write(raw, "transcript_2026-11-03_1000.md",
              "performance review conversation about career progression and "
              "promotion expectations for next year",
              title="Career review")

        found = find(raw, "transcript_2026-11-01_1000.md")
        assert found, "the follow-up should be linked"
        assert found[0]["title"] == "Migration follow-up"
        assert "kubernetes" in found[0]["shared_terms"] or \
               "migration" in found[0]["shared_terms"]

    def test_an_unrelated_meeting_is_not_linked(self, tmp_path):
        raw = tmp_path / "raw"
        write(raw, "transcript_2026-11-01_1000.md",
              "kubernetes cluster migration rollout regions deployment",
              title="Migration")
        write(raw, "transcript_2026-11-02_1000.md",
              "bakery supplies flour yeast sourdough oven temperature",
              title="Baking")

        assert find(raw, "transcript_2026-11-01_1000.md") == []

    def test_the_shared_terms_are_reported(self, tmp_path):
        """The reader has to be able to judge the link, so show the evidence."""
        raw = tmp_path / "raw"
        write(raw, "transcript_2026-11-01_1000.md",
              "kubernetes migration rollout kubernetes migration cluster")
        write(raw, "transcript_2026-11-02_1000.md",
              "kubernetes migration rollout again kubernetes migration")
        write(raw, "transcript_2026-11-03_1000.md",
              "unrelated bakery flour yeast")

        found = find(raw, "transcript_2026-11-01_1000.md")
        assert found[0]["shared_terms"]
        assert all(isinstance(t, str) for t in found[0]["shared_terms"])

    def test_the_target_is_never_in_its_own_results(self, tmp_path):
        raw = tmp_path / "raw"
        write(raw, "transcript_2026-11-01_1000.md", "kubernetes migration cluster")
        write(raw, "transcript_2026-11-02_1000.md", "kubernetes migration cluster")
        found = find(raw, "transcript_2026-11-01_1000.md")
        assert all(r["name"] != "transcript_2026-11-01_1000.md" for r in found)

    def test_a_lone_recording_has_nothing_to_relate_to(self, tmp_path):
        raw = tmp_path / "raw"
        write(raw, "transcript_2026-11-01_1000.md", "kubernetes migration")
        assert find(raw, "transcript_2026-11-01_1000.md") == []

    def test_results_are_limited(self, tmp_path):
        raw = tmp_path / "raw"
        for i in range(9):
            write(raw, f"transcript_2026-11-0{i+1}_1000.md",
                  "kubernetes migration cluster rollout regions deployment")
        found = find(raw, "transcript_2026-11-01_1000.md", limit=3)
        assert len(found) == 3

    def test_results_are_ordered_by_strength(self, tmp_path):
        raw = tmp_path / "raw"
        write(raw, "transcript_2026-11-01_1000.md",
              "kubernetes migration cluster rollout regions")
        write(raw, "transcript_2026-11-02_1000.md",
              "kubernetes migration cluster rollout regions deployment",
              title="Strong")
        write(raw, "transcript_2026-11-03_1000.md",
              "kubernetes migration bakery flour yeast sourdough", title="Weak")

        found = find(raw, "transcript_2026-11-01_1000.md")
        scores = [r["score"] for r in found]
        assert scores == sorted(scores, reverse=True)

    def test_a_missing_target_raises(self, tmp_path):
        raw = tmp_path / "raw"
        write(raw, "transcript_2026-11-01_1000.md", "kubernetes")
        with pytest.raises(FileNotFoundError):
            find(raw, "transcript_1999-01-01_0000.md")

    def test_traversal_is_refused(self, tmp_path):
        from scribe_audio.library import LibraryError

        raw = tmp_path / "raw"
        write(raw, "transcript_2026-11-01_1000.md", "kubernetes")
        with pytest.raises(LibraryError):
            find(raw, "../../etc/passwd")

    def test_non_recordings_are_not_candidates(self, tmp_path):
        """A README in the folder must not be offered as a related meeting."""
        raw = tmp_path / "raw"
        write(raw, "transcript_2026-11-01_1000.md",
              "kubernetes migration cluster rollout")
        (raw / "README.md").write_text(
            "# Project\n\nkubernetes migration cluster rollout notes\n",
            encoding="utf-8",
        )
        assert all(r["name"] != "README.md"
                   for r in find(raw, "transcript_2026-11-01_1000.md"))
