"""Text deduplication for overlapping transcription windows.

Each transcription window overlaps with the previous one. We need to find
what's NEW in the current window compared to the previous transcription,
and only emit the new text.

Strategy:
- Normalise both texts to word lists
- Use SequenceMatcher to find the longest common block between prev and curr
- Everything in curr AFTER that block is new text
"""

from difflib import SequenceMatcher


def _norm(word: str) -> str:
    """Lowercase and strip punctuation for fuzzy comparison."""
    return word.lower().rstrip(".,!?;:'\")")


def extract_new_text(prev_text: str, curr_text: str, min_overlap_words: int = 3) -> str:
    """
    Given previous and current transcription (from overlapping windows),
    return only the NEW text from the current window.

    Args:
        prev_text: Previous window's full transcription
        curr_text: Current window's full transcription
        min_overlap_words: Minimum words to consider as a valid overlap match

    Returns:
        Only the new text not present in previous transcription
    """
    if not prev_text:
        return curr_text

    if not curr_text:
        return ""

    prev_words = prev_text.split()
    curr_words = curr_text.split()

    if not prev_words or not curr_words:
        return curr_text

    # Normalise for comparison
    prev_norm = [_norm(w) for w in prev_words]
    curr_norm = [_norm(w) for w in curr_words]

    # Find the longest matching block between prev and curr using
    # normalised words. This handles Whisper re-transcribing the
    # overlapping audio slightly differently.
    sm = SequenceMatcher(None, prev_norm, curr_norm, autojunk=False)
    match = sm.find_longest_match(0, len(prev_norm), 0, len(curr_norm))

    # match.a = start index in prev, match.b = start index in curr,
    # match.size = length of matching block
    if match.size >= min_overlap_words:
        # New text = everything in curr after the matching block
        after_idx = match.b + match.size
        new_words = curr_words[after_idx:]
        return " ".join(new_words) if new_words else ""

    # No significant overlap found — return all of current
    return curr_text
