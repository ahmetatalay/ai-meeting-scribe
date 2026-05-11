"""Write transcribed text to a raw transcript markdown file."""

import os
from datetime import datetime
from pathlib import Path


class TranscriptWriter:
    """Appends transcribed text chunks to a markdown file."""

    def __init__(self, raw_dir: str = "./raw"):
        self.raw_dir = Path(raw_dir).expanduser()
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self._filepath: Path | None = None
        self._chunk_count = 0

    def _ensure_file(self) -> Path:
        """Create transcript file if not yet created."""
        if self._filepath is None:
            ts = datetime.now().strftime("%Y-%m-%d_%H%M")
            self._filepath = self.raw_dir / f"transcript_{ts}.md"
            with open(self._filepath, "w") as f:
                f.write(f"# Transcript — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            print(f"[writer] Transcript: {self._filepath}")
        return self._filepath

    def write(self, text: str):
        """Append a transcribed chunk to the transcript file (plain mode)."""
        if not text.strip():
            return

        self._chunk_count += 1
        filepath = self._ensure_file()
        ts = datetime.now().strftime("%H:%M:%S")

        with open(filepath, "a") as f:
            f.write(f"<!-- chunk {self._chunk_count} @ {ts} -->\n")
            f.write(f"{text}\n\n")

        print(f"[writer] Chunk {self._chunk_count} written ({len(text)} chars)")

    def write_speaker_segments(self, segments: list[dict]):
        """Append speaker-labeled segments to the transcript file.

        Args:
            segments: [{"speaker": "SPEAKER_00", "text": "hello", "start": 0.5}, ...]
        """
        if not segments:
            return

        self._chunk_count += 1
        filepath = self._ensure_file()
        ts = datetime.now().strftime("%H:%M:%S")

        with open(filepath, "a") as f:
            f.write(f"<!-- chunk {self._chunk_count} @ {ts} -->\n")
            current_speaker = None
            for seg in segments:
                speaker = seg["speaker"]
                text = seg["text"]
                if speaker != current_speaker:
                    current_speaker = speaker
                    f.write(f"\n**{speaker}**: {text}\n")
                else:
                    # Same speaker continues — append to previous line
                    f.write(f"{text}\n")
            f.write("\n")

        total_chars = sum(len(s["text"]) for s in segments)
        speakers = set(s["speaker"] for s in segments)
        print(f"[writer] Chunk {self._chunk_count}: {len(segments)} segments, "
              f"{len(speakers)} speakers, {total_chars} chars")

    @property
    def transcript_path(self) -> Path | None:
        return self._filepath
