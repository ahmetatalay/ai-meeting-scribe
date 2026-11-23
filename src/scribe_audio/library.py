"""Transcript library: listing, reading, renaming and deleting recordings.

Every recording writes a markdown file plus a JSON sidecar (see
:mod:`scribe_audio.writer`). The sidecar is authoritative for mutable metadata —
the title especially — so renaming never has to rewrite the transcript itself.

Recordings made before sidecars existed, or markdown dropped into the directory
by hand, are still listed: whatever can be derived from the frontmatter and the
body is used, and the rest is left unknown.

Path safety is the main concern here. Callers pass a *name* that arrives from an
HTTP request, and these functions read and delete files with it. Names are
therefore restricted to a bare filename pattern and the resolved path is checked
to sit directly inside the library directory, so neither ``../`` nor an absolute
path nor a symlink can reach outside it.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

# A bare transcript filename: no directory separators, no leading dot.
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.md$")

FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.S)
SPEAKER_LINE = re.compile(r"^\*\*(.+?)\*\*:", re.M)
CHUNK_MARKER = re.compile(r"<!-- chunk \d+ @ [\d:]+ -->\n?")
HEADING = re.compile(r"^#\s+(.+?)\s*$", re.M)

# transcript_2026-11-19_1331.md — the naming scheme every recording uses.
NAME_DATE = re.compile(r"(\d{4})-(\d{2})-(\d{2})[_-](\d{2})(\d{2})")


def looks_like_recording(path: Path, body: str, front: dict, has_sidecar: bool) -> bool:
    """Is this markdown something scribe produced?

    The library lists a directory of ``*.md`` and offers a Delete button, so
    listing arbitrary markdown is dangerous: pointed at a project folder or an
    Obsidian vault it would present your notes as recordings, one click from
    permanent deletion. (A plain ``-r .`` in a checkout listed README.md and
    CHANGELOG.md exactly this way.)

    So this asks for positive evidence of scribe's own output rather than
    guessing from content — content heuristics fail on documents that merely
    *contain* a transcript, such as this project's README.
    """
    if has_sidecar:
        return True
    if front.get("started"):
        return True
    if NAME_DATE.search(path.stem) and path.stem.lower().startswith("transcript"):
        return True
    heading = HEADING.search(_strip_frontmatter(body))
    return bool(heading and heading.group(1).lower().startswith("transcript"))


def _started_from_name(stem: str) -> str | None:
    """Recover the recording time from the filename.

    Recordings predating sidecars carry no start time in the file, and file
    mtime is unreliable — copying a directory rewrites it, which sorted old
    transcripts above new ones.
    """
    match = NAME_DATE.search(stem)
    if not match:
        return None
    year, month, day, hour, minute = (int(g) for g in match.groups())
    try:
        return datetime(year, month, day, hour, minute).isoformat(timespec="seconds")
    except ValueError:
        return None


class LibraryError(ValueError):
    """A name that is not an acceptable transcript filename."""


def resolve(raw_dir: Path | str, name: str) -> Path:
    """Validate ``name`` and return its path inside ``raw_dir``.

    Raises :class:`LibraryError` for anything that is not a plain filename
    sitting directly in the library directory.
    """
    if not name or not SAFE_NAME.match(name):
        raise LibraryError(f"not a transcript filename: {name!r}")

    directory = Path(raw_dir).expanduser().resolve()
    candidate = (directory / name).resolve()

    # resolve() has followed any symlink, so this also rejects a link planted
    # in the library that points somewhere else.
    if candidate.parent != directory:
        raise LibraryError(f"outside the library directory: {name!r}")
    return candidate


def _parse_frontmatter(body: str) -> dict:
    match = FRONTMATTER.match(body)
    if not match:
        return {}
    data = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            data[key.strip()] = value.strip()
    return data


def _strip_frontmatter(body: str) -> str:
    return FRONTMATTER.sub("", body, count=1)


def _word_count(body: str) -> int:
    text = _strip_frontmatter(body)
    text = CHUNK_MARKER.sub("", text)
    text = SPEAKER_LINE.sub("", text)
    text = re.sub(r"^#.*$", "", text, flags=re.M)
    return len(text.split())


def _duration_seconds(sidecar: dict) -> float | None:
    """Wall-clock span from the start of the recording to its last segment."""
    started, segments = sidecar.get("started"), sidecar.get("segments") or []
    if not started or not segments:
        return None
    last = next(
        (s.get("wall") for s in reversed(segments) if s.get("wall")), None
    )
    if not last:
        return None
    try:
        return max(
            0.0,
            (datetime.fromisoformat(last) - datetime.fromisoformat(started))
            .total_seconds(),
        )
    except ValueError:
        return None


def describe(path: Path, raw_dir: Path) -> dict:
    """Summarise one transcript for the library list."""
    body = path.read_text(encoding="utf-8", errors="replace")
    front = _parse_frontmatter(body)

    sidecar_path = path.with_suffix(".json")
    sidecar: dict = {}
    if sidecar_path.is_file():
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            sidecar = {}

    speakers = sidecar.get("segments")
    if speakers:
        counts: dict[str, int] = {}
        for seg in speakers:
            who = seg.get("speaker")
            if who:
                counts[who] = counts.get(who, 0) + 1
    else:
        # No sidecar: count speaker prefixes in the markdown instead.
        counts = {}
        for who in SPEAKER_LINE.findall(body):
            counts[who] = counts.get(who, 0) + 1

    heading = HEADING.search(_strip_frontmatter(body))
    is_recording = looks_like_recording(path, body, front, bool(sidecar))

    stat = path.stat()
    return {
        "name": path.name,
        # The sidecar wins: renaming updates it and leaves the markdown alone.
        # Then the frontmatter, then the document's own heading, then the stem.
        "title": (sidecar.get("title") or front.get("title")
                  or (heading.group(1) if heading else None) or path.stem),
        "started": (sidecar.get("started") or front.get("started")
                    or _started_from_name(path.stem)),
        "model": sidecar.get("model") or front.get("model"),
        "language": sidecar.get("language") or front.get("language"),
        "duration_seconds": _duration_seconds(sidecar),
        "words": _word_count(body),
        "speakers": counts,
        "has_sidecar": bool(sidecar),
        "has_summary": bool((sidecar.get("summary") or {}).get("text")),
        "is_recording": is_recording,
        "relative_path": str(path.relative_to(raw_dir)) if path.is_relative_to(raw_dir)
                         else path.name,
        "size_bytes": stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
    }


def scan(raw_dir: Path | str, query: str = "") -> list[dict]:
    """List recordings, newest first.

    Markdown that scribe did not produce is skipped — see
    :func:`looks_like_recording`. ``query`` matches the title and the body,
    case-insensitively.
    """
    return scan_detailed(raw_dir, query)["transcripts"]


def scan_detailed(raw_dir: Path | str, query: str = "") -> dict:
    """:func:`scan`, plus a count of the markdown files that were skipped."""
    directory = Path(raw_dir).expanduser()
    if not directory.is_dir():
        return {"transcripts": [], "skipped": 0}

    needle = query.strip().lower()
    out = []
    skipped = 0
    for path in directory.glob("*.md"):
        if not path.is_file():
            continue
        try:
            entry = describe(path, directory)
        except OSError:
            continue
        if not entry["is_recording"]:
            skipped += 1
            continue
        if needle:
            haystack = entry["title"].lower()
            if needle not in haystack:
                try:
                    body = path.read_text(encoding="utf-8", errors="replace").lower()
                except OSError:
                    continue
                if needle not in body:
                    continue
        out.append(entry)

    out.sort(key=lambda e: e["started"] or e["modified"], reverse=True)
    return {"transcripts": out, "skipped": skipped}


def read(raw_dir: Path | str, name: str) -> dict:
    """Return one transcript's metadata plus its turns."""
    path = resolve(raw_dir, name)
    if not path.is_file():
        raise FileNotFoundError(name)

    directory = Path(raw_dir).expanduser().resolve()
    entry = describe(path, directory)
    if not entry["is_recording"]:
        raise LibraryError(f"not a scribe recording: {name!r}")
    body = path.read_text(encoding="utf-8", errors="replace")
    entry["turns"] = parse_turns(body)
    entry["summary"] = (_load_sidecar(path).get("summary") or None)
    return entry


def parse_turns(body: str) -> list[dict]:
    """Split a transcript body into speaker turns.

    Handles both the current format and the older one that interleaved
    ``<!-- chunk -->`` markers.
    """
    text = CHUNK_MARKER.sub("", _strip_frontmatter(body))
    turns: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^\*\*(.+?)\*\*:\s*(.*)$", line)
        if match:
            turns.append({"speaker": match.group(1), "text": match.group(2).strip()})
        elif turns:
            # Continuation of the previous speaker's paragraph.
            turns[-1]["text"] = f"{turns[-1]['text']} {line}".strip()
        else:
            turns.append({"speaker": None, "text": line})
    return [t for t in turns if t["text"]]


def _load_sidecar(path: Path) -> dict:
    sidecar_path = path.with_suffix(".json")
    if not sidecar_path.is_file():
        return {}
    try:
        return json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_sidecar(path: Path, data: dict) -> None:
    sidecar_path = path.with_suffix(".json")
    tmp = sidecar_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(sidecar_path)


def get_summary(raw_dir: Path | str, name: str) -> dict | None:
    """The stored summary for a recording, if one has been generated."""
    path = resolve(raw_dir, name)
    if not path.is_file():
        raise FileNotFoundError(name)
    summary = _load_sidecar(path).get("summary") or None
    return summary if summary and summary.get("text") else None


def set_summary(raw_dir: Path | str, name: str, summary: dict) -> dict:
    """Store a summary in the sidecar.

    The sidecar rather than the markdown: the transcript is append-and-revise
    during recording, so inserting a section into it would collide with the
    tail-revision offsets.
    """
    if not (summary.get("text") or "").strip():
        raise LibraryError("summary text must not be empty")

    path = resolve(raw_dir, name)
    if not path.is_file():
        raise FileNotFoundError(name)
    directory = Path(raw_dir).expanduser().resolve()
    if not describe(path, directory)["is_recording"]:
        raise LibraryError(f"not a scribe recording: {name!r}")

    data = _load_sidecar(path)
    data.setdefault("markdown", path.name)
    data["summary"] = summary
    _save_sidecar(path, data)
    return summary


def set_title(raw_dir: Path | str, name: str, title: str) -> dict:
    """Rename a recording. Writes the sidecar, never the transcript."""
    title = title.strip()
    if not title:
        raise LibraryError("title must not be empty")

    path = resolve(raw_dir, name)
    if not path.is_file():
        raise FileNotFoundError(name)

    data = _load_sidecar(path)
    data.setdefault("markdown", path.name)
    data["title"] = title
    _save_sidecar(path, data)
    return {"name": path.name, "title": title}


def delete(raw_dir: Path | str, name: str) -> dict:
    """Delete a transcript and its sidecar.

    Irreversible: there is no trash. The caller is responsible for confirming
    with the user first.
    """
    path = resolve(raw_dir, name)
    if not path.is_file():
        raise FileNotFoundError(name)

    directory = Path(raw_dir).expanduser().resolve()
    if not describe(path, directory)["is_recording"]:
        # Refusing here is what stops a stray README or vault note being
        # deleted through an API the library would never have offered it in.
        raise LibraryError(f"not a scribe recording: {name!r}")

    removed = [path.name]
    sidecar_path = path.with_suffix(".json")
    path.unlink()
    if sidecar_path.is_file():
        sidecar_path.unlink()
        removed.append(sidecar_path.name)
    return {"deleted": removed}
