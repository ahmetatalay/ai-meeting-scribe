"""Overlap removal for consecutive transcription windows.

Each window overlaps the previous one by ``window - step`` seconds, so Whisper
re-transcribes the same audio two or more times — and *not* identically. Real
captures show the same speech coming back as "AWS product" / "AWS production"
or "AIOps and HGTNKI" / "AI Ops and Aged in AI". Any method that relies on
exact string matching therefore fails on most chunks.

Strategy:

1. Normalise aggressively (lowercase, drop every non-alphanumeric character) so
   re-wordings of the same token still compare equal.
2. The overlap is by construction a *prefix* of the current window, so only an
   alignment anchored at the head of the current window counts. This is what
   stops a phrase the speaker merely repeated later ("during this time", "yeah
   so I mean") from being mistaken for the overlap — the failure that made the
   previous implementation re-emit whole segments.
3. Walk the matching blocks forward from that head anchor, absorbing small gaps
   so that a re-wording *inside* the overlap does not end the alignment early.
4. Return an **index** into the current word list, not a string. Callers slice
   their own structures with it, so nothing has to re-find the split point by
   searching text again.
"""

from difflib import SequenceMatcher

# How far into the current window the overlap may start. Whisper sometimes
# re-words the first token or two of a window, so a strict 0 is too brittle.
_HEAD_TOLERANCE = 2

# A gap in the current window between two matching blocks is treated as a
# re-wording inside the overlap (rather than the end of it) when it is no
# larger than this many words, or half of what has matched so far.
_MIN_GAP_TOLERANCE = 3


def normalise_word(word: str) -> str:
    """Lowercase and strip every non-alphanumeric character.

    "AWS production," -> "awsproduction"; "Mm-hmm." -> "mmhmm". This is
    deliberately lossy: it is only ever used for comparison, never for output.
    """
    return "".join(ch for ch in word.lower() if ch.isalnum())


def _normalise_all(words: list[str]) -> list[str]:
    # Tokens that normalise to nothing (standalone punctuation) get a sentinel
    # that cannot compare equal to anything, so they never anchor a match.
    return [normalise_word(w) or "\x00" for w in words]


def find_new_index(
    prev_words: list[str],
    curr_words: list[str],
    min_overlap_words: int = 3,
) -> int:
    """Index into ``curr_words`` where content not present in ``prev_words`` begins.

    Returns 0 when no overlap is found (all of ``curr_words`` is new) and
    ``len(curr_words)`` when the current window added nothing.
    """
    if not prev_words or not curr_words:
        return 0

    prev_norm = _normalise_all(prev_words)
    curr_norm = _normalise_all(curr_words)

    sm = SequenceMatcher(None, prev_norm, curr_norm, autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size > 0]
    if not blocks:
        return 0

    # Anchor: the first block that starts at (or very near) the head of curr.
    anchor_idx = next(
        (i for i, b in enumerate(blocks) if b.b <= _HEAD_TOLERANCE),
        None,
    )
    if anchor_idx is None:
        # Nothing at the head of curr matches prev, so the windows do not
        # overlap textually — a silence gap, or a window of entirely new speech.
        return 0

    anchor = blocks[anchor_idx]
    end = anchor.b + anchor.size
    matched = anchor.size

    # Extend through subsequent blocks, tolerating small re-wording gaps.
    for block in blocks[anchor_idx + 1:]:
        gap = block.b - end
        if gap < 0:
            continue
        if gap > max(_MIN_GAP_TOLERANCE, matched // 2):
            break
        end = block.b + block.size
        matched += block.size

    # Require a convincing amount of matched content, unless the alignment
    # consumed the whole current window (nothing new was said).
    if matched < min_overlap_words and end < len(curr_words):
        return 0

    return min(end, len(curr_words))


def segment_words(segments: list[dict]) -> list[str]:
    """Flatten segment texts into one word list, in order."""
    words: list[str] = []
    for seg in segments:
        words.extend(seg.get("text", "").split())
    return words


def slice_segments(segments: list[dict], word_index: int) -> list[dict]:
    """Drop the first ``word_index`` words from a list of segments.

    Segments entirely inside the overlap are dropped; the one straddling the
    boundary is trimmed. This is what replaces re-searching the segment texts
    for a phrase — the previous approach failed whenever the boundary fell
    within a few words of a speaker change, and then fell back to re-emitting
    whole segments.
    """
    if word_index <= 0:
        return [dict(s) for s in segments]

    remaining = word_index
    out: list[dict] = []
    for seg in segments:
        words = seg.get("text", "").split()
        if remaining >= len(words):
            remaining -= len(words)
            continue
        if remaining > 0:
            trimmed = dict(seg)
            trimmed["text"] = " ".join(words[remaining:])
            out.append(trimmed)
            remaining = 0
        else:
            out.append(dict(seg))
    return out


def extract_new_text(prev_text: str, curr_text: str, min_overlap_words: int = 3) -> str:
    """String convenience wrapper around :func:`find_new_index`."""
    if not prev_text:
        return curr_text
    if not curr_text:
        return ""

    curr_words = curr_text.split()
    idx = find_new_index(prev_text.split(), curr_words, min_overlap_words)
    return " ".join(curr_words[idx:])
