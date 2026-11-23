"""Overlap-removal tests.

The cases under ``TestObservedDuplication`` are modelled on real failures seen
in captured transcripts — the previous implementation duplicated the tail of
the previous chunk on roughly every chunk. The wording here is synthetic (the
real captures are private meeting recordings) but each case preserves the exact
phenomenon that broke it:

* a token re-worded right at the window boundary,
* a re-wording *inside* the overlap that splits the alignment in two,
* a phrase that recurs later in the window but is not the overlap,
* a phrase that recurs within the previous window.
"""

from scribe_audio.dedup import (
    align_tail,
    extract_new_text,
    find_new_index,
    head_segments,
    normalise_word,
    segment_words,
    slice_segments,
)


class TestNormalise:
    def test_strips_punctuation_and_case(self):
        assert normalise_word("Production,") == "production"
        assert normalise_word("Mm-hmm.") == "mmhmm"
        assert normalise_word("You") == "you"

    def test_rewordings_that_should_not_collide(self):
        # These are genuinely different words; normalising must not merge them.
        assert normalise_word("product") != normalise_word("production")

    def test_punctuation_only_token(self):
        assert normalise_word("--") == ""


class TestObservedDuplication:
    """Shapes that the previous implementation got wrong in production."""

    def test_token_reworded_at_the_window_boundary(self):
        """Whisper renders the last word of the overlap differently each pass."""
        prev = (
            "we worked with the platform team to deliver the migration in stages. "
            "So we successfully deployed the first, first phase of the service "
            "into production, cache product."
        )
        curr = (
            "phase of the service into production, cache production, successfully "
            "at the end of August. And then we also had the chance to create new "
            "initiatives, which is caching and HTTPX projects."
        )
        new = extract_new_text(prev, curr)

        # The duplicated run must be gone...
        assert "phase of the service into production" not in new
        # ...and the genuinely new content must survive.
        assert "end of August" in new
        assert "caching and HTTPX projects" in new

    def test_rewording_inside_the_overlap_does_not_end_it_early(self):
        """'caching' -> 'Caching', 'HTTPX' -> 'HTTP X' splits the alignment.

        A head-anchored match alone stops at the first mismatch and leaves most
        of the overlap in place; the alignment has to survive internal gaps.
        """
        prev = (
            "at the end of August. And then we also had the chance to create new "
            "initiatives, which is caching and HTTPX projects."
        )
        curr = (
            "a chance to create new cached initiatives which is Caching and HTTP X "
            "projects. The dashboard was not using live metrics actively so we "
            "found a good opportunity."
        )
        new = extract_new_text(prev, curr)

        assert "create new" not in new
        assert "The dashboard was not using live metrics" in new

    def test_phrase_recurring_later_in_window_is_not_the_overlap(self):
        """The windows genuinely do not overlap, but share a phrase.

        'during this phase' ends the previous window and also appears mid-way
        through the current one. Anchoring on that later occurrence silently
        drops a sentence of real speech.
        """
        prev = (
            "there was a lot of review overhead from the platform team as well "
            "during this phase, which also"
        )
        curr = (
            "slowed the rollout and affected our estimates during this phase. "
            "And also we had some issues with the build cache again"
        )
        new = extract_new_text(prev, curr)

        assert new.startswith("slowed the rollout")
        assert "issues with the build cache" in new

    def test_phrase_recurring_within_previous_window(self):
        """'yeah so I mean' appears twice in prev.

        Anchoring on the first occurrence re-emits everything after it.
        """
        prev = "yeah so I mean the thing is yeah so I mean we should ship it"
        curr = "yeah so I mean we should ship it tomorrow morning"
        assert extract_new_text(prev, curr) == "tomorrow morning"


class TestEdgeCases:
    def test_no_previous_text_returns_everything(self):
        assert extract_new_text("", "hello world") == "hello world"

    def test_empty_current_returns_empty(self):
        assert extract_new_text("hello world", "") == ""

    def test_identical_windows_produce_nothing(self):
        text = "this is exactly the same sentence twice over"
        assert extract_new_text(text, text) == ""

    def test_no_overlap_returns_all_of_current(self):
        prev = "completely unrelated words about audio devices"
        curr = "a totally different sentence concerning transcripts"
        assert extract_new_text(prev, curr) == curr

    def test_pure_continuation(self):
        prev = "one two three four five six"
        curr = "four five six seven eight nine"
        assert extract_new_text(prev, curr) == "seven eight nine"

    def test_idempotent_under_repeat(self):
        """Feeding the same window twice must not emit anything the second time."""
        prev = "the quick brown fox jumps over the lazy dog"
        once = extract_new_text("", prev)
        assert once == prev
        assert extract_new_text(prev, prev) == ""

    def test_punctuation_only_tokens_do_not_anchor(self):
        prev = "some earlier sentence that ended"
        curr = "-- a brand new sentence entirely"
        assert extract_new_text(prev, curr) == curr


class TestSegmentSlicing:
    """Replaces the old phrase-search, whose fallback re-emitted whole segments."""

    SEGMENTS = [
        {"speaker": "You", "text": "hello there how are you doing today"},
        {"speaker": "Meeting", "text": "i am fine thanks"},
    ]

    def test_flatten_preserves_order(self):
        assert segment_words(self.SEGMENTS)[:3] == ["hello", "there", "how"]
        assert len(segment_words(self.SEGMENTS)) == 11

    def test_zero_index_keeps_everything(self):
        assert slice_segments(self.SEGMENTS, 0) == self.SEGMENTS

    def test_boundary_inside_a_segment_trims_it(self):
        out = slice_segments(self.SEGMENTS, 4)
        assert out[0]["text"] == "you doing today"
        assert out[0]["speaker"] == "You"
        assert out[1]["text"] == "i am fine thanks"

    def test_boundary_exactly_on_a_segment_edge_drops_it(self):
        out = slice_segments(self.SEGMENTS, 7)
        assert len(out) == 1
        assert out[0]["text"] == "i am fine thanks"

    def test_boundary_spanning_a_speaker_change(self):
        """The exact case the old fallback duplicated a whole turn on."""
        out = slice_segments(self.SEGMENTS, 6)
        assert out[0]["speaker"] == "You"
        assert out[0]["text"] == "today"
        assert out[1]["text"] == "i am fine thanks"

    def test_index_past_the_end_drops_everything(self):
        assert slice_segments(self.SEGMENTS, 99) == []

    def test_does_not_mutate_the_input(self):
        slice_segments(self.SEGMENTS, 4)
        assert self.SEGMENTS[0]["text"] == "hello there how are you doing today"

    def test_round_trip_against_find_new_index(self):
        """Flatten, find the boundary, slice — no text may be duplicated."""
        prev_segments = [{"speaker": "You", "text": "we should ship the build today"}]
        curr_segments = [
            {"speaker": "You", "text": "ship the build today"},
            {"speaker": "Meeting", "text": "sounds good to me"},
        ]
        prev_words = segment_words(prev_segments)
        curr_words = segment_words(curr_segments)
        idx = find_new_index(prev_words, curr_words)
        out = slice_segments(curr_segments, idx)

        text = " ".join(s["text"] for s in out)
        assert "ship the build" not in text
        assert text == "sounds good to me"


class TestAlignTail:
    """Whether the current window re-transcribes the last batch in full.

    Replacing that batch is only safe when the answer is yes: a partial match
    would delete the part the window does not cover, and a match reaching
    further back than the batch would duplicate text written before it.
    """

    def test_a_batch_fully_restated_is_covered(self):
        batch = "we should ship the build".split()
        curr = "we should ship the build today".split()
        assert align_tail(batch, curr) == 5

    def test_a_reworded_last_word_still_counts(self):
        """The whole point: the second pass corrects the boundary word."""
        batch = "Airbnb has the same purpose".split()
        curr = "Airbnb has the same problem when two guests book".split()
        assert align_tail(batch, curr) == 4

    def test_a_match_starting_later_in_the_window_is_refused(self):
        """Those earlier words belong to audio already written.

        Accepting this duplicated text: 'alpha beta' + 'alpha beta gamma'.
        """
        batch = ["gamma"]
        curr = "alpha beta gamma delta".split()
        assert align_tail(batch, curr) == 0

    def test_a_partially_covered_batch_is_refused(self):
        """Replacing it would delete 'alpha beta'."""
        batch = "alpha beta gamma delta".split()
        curr = "gamma delta epsilon".split()
        assert align_tail(batch, curr) == 0

    def test_an_unrelated_window_is_refused(self):
        assert align_tail("one two three".split(), "four five six".split()) == 0

    def test_a_batch_restated_with_nothing_new(self):
        batch = "the same words again".split()
        assert align_tail(batch, list(batch)) == 4

    def test_empty_inputs(self):
        assert align_tail([], ["a"]) == 0
        assert align_tail(["a"], []) == 0

    def test_covered_count_never_exceeds_the_window(self):
        batch = "a b c".split()
        curr = "a b".split()
        assert align_tail(batch, curr) <= len(curr)

    def test_a_long_batch_with_an_internal_rewording(self):
        batch = "the cache layer is returning stale keys".split()
        curr = "the cache layer is returning stale keys again today".split()
        assert align_tail(batch, curr) == 7

    def test_repeated_phrase_does_not_create_a_false_cover(self):
        batch = ["yeah"]
        curr = "something else entirely yeah".split()
        assert align_tail(batch, curr) == 0


class TestFindNewIndex:
    def test_index_points_at_first_new_word(self):
        prev = "one two three four five six".split()
        curr = "four five six seven eight".split()
        assert find_new_index(prev, curr) == 3

    def test_index_is_zero_when_nothing_matches(self):
        assert find_new_index("alpha beta".split(), "gamma delta".split()) == 0

    def test_index_is_length_when_all_is_old(self):
        words = "alpha beta gamma delta epsilon".split()
        assert find_new_index(words, words) == len(words)

    def test_index_never_exceeds_current_length(self):
        prev = "a b c d e f g h".split()
        curr = "f g h".split()
        assert find_new_index(prev, curr) <= len(curr)

    def test_empty_inputs(self):
        assert find_new_index([], ["a"]) == 0
        assert find_new_index(["a"], []) == 0
