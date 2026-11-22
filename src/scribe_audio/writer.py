"""Transcript output: markdown for reading, JSON sidecar for tooling.

The markdown file stays the artifact of record — it is what an Obsidian vault
indexes and what a human opens. Alongside it sits a ``.json`` sidecar holding
the same content as structured segments plus mutable metadata (title, duration,
counts). The web UI reads the sidecar; renaming a meeting touches only that, so
the markdown never has to be rewritten.

Two bugs from the previous version are fixed here:

* Filenames had minute granularity and were opened with ``"w"``, so two
  recordings started in the same minute silently truncated the first.
* Files were opened without an explicit encoding, which matters as soon as a
  transcript is not in English.

Consecutive turns by the same speaker continue the same markdown paragraph
rather than restarting the speaker prefix, so a sentence spanning two windows
reads as one utterance.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def _unique_path(directory: Path, stem: str, suffix: str = ".md") -> Path:
    """A path that does not exist yet, so an existing transcript is never lost."""
    candidate = directory / f"{stem}{suffix}"
    if not candidate.exists():
        return candidate
    for n in range(2, 1000):
        candidate = directory / f"{stem}-{n}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"cannot find an unused filename for {stem!r} in {directory}")


class TranscriptWriter:
    """Appends speaker-labelled turns to a markdown file and a JSON sidecar."""

    def __init__(
        self,
        raw_dir: str | Path = "./raw",
        title: str | None = None,
        model: str | None = None,
        language: str | None = None,
    ):
        self.raw_dir = Path(raw_dir).expanduser()
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.started = datetime.now()
        self.title = title or f"Meeting {self.started.strftime('%Y-%m-%d %H:%M')}"
        self.model = model
        self.language = language

        self._path: Path | None = None
        self._sidecar_path: Path | None = None
        self._last_speaker: str | None = None
        self._segments: list[dict] = []
        self._turn_count = 0

    # ------------------------------------------------------------------ paths

    @property
    def path(self) -> Path | None:
        """The markdown file, or None if nothing has been written yet."""
        return self._path

    @property
    def sidecar_path(self) -> Path | None:
        return self._sidecar_path

    @property
    def segments(self) -> list[dict]:
        return list(self._segments)

    def _ensure_file(self) -> Path:
        if self._path is not None:
            return self._path

        stem = f"transcript_{self.started.strftime('%Y-%m-%d_%H%M')}"
        self._path = _unique_path(self.raw_dir, stem, ".md")
        self._sidecar_path = self._path.with_suffix(".json")

        front = [
            "---",
            f"title: {self.title}",
            f"started: {self.started.isoformat(timespec='seconds')}",
        ]
        if self.model:
            front.append(f"model: {self.model}")
        if self.language:
            front.append(f"language: {self.language}")
        front += ["---", "", f"# {self.title}", ""]

        self._path.write_text("\n".join(front), encoding="utf-8")
        return self._path

    # ----------------------------------------------------------------- output

    def append_segments(self, segments: list[dict]) -> None:
        """Append speaker-labelled turns.

        ``segments`` are ``{"speaker", "text", ...}``; consecutive turns by the
        same speaker continue the previous paragraph.
        """
        usable = [s for s in segments if s.get("text", "").strip()]
        if not usable:
            return

        path = self._ensure_file()
        lines = []
        for seg in usable:
            speaker = seg["speaker"]
            text = seg["text"].strip()
            if speaker == self._last_speaker:
                # Same speaker continuing: markdown joins adjacent lines into
                # one paragraph, so this reads as a single turn.
                lines.append(text)
            else:
                lines.append("")
                lines.append(f"**{speaker}**: {text}")
                self._turn_count += 1
                self._last_speaker = speaker

            self._segments.append({
                "speaker": speaker,
                "text": text,
                "start": seg.get("start"),
                "end": seg.get("end"),
                "wall": datetime.now().isoformat(timespec="seconds"),
            })

        with open(path, "a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

        self._write_sidecar()

    def append_text(self, text: str) -> None:
        """Append unattributed text (speaker detection disabled)."""
        if not text.strip():
            return
        self.append_segments([{"speaker": "Transcript", "text": text}])

    def _write_sidecar(self) -> None:
        if self._sidecar_path is None:
            return
        payload = {
            "title": self.title,
            "started": self.started.isoformat(timespec="seconds"),
            "model": self.model,
            "language": self.language,
            "markdown": self._path.name if self._path else None,
            "turns": self._turn_count,
            "segments": self._segments,
        }
        # Written whole each time: the sidecar is small and this keeps it valid
        # JSON even if the process is killed mid-meeting.
        tmp = self._sidecar_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._sidecar_path)

    def set_title(self, title: str) -> None:
        """Rename the meeting. Takes effect in the sidecar immediately."""
        self.title = title
        self._write_sidecar()
