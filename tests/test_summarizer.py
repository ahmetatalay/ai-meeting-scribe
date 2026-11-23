"""Summarisation tests.

A fake transport stands in for Ollama, so none of this needs a model or a
running server. What is worth testing is the chunking, the merge pass, the
grounding instructions and the failure messages.
"""

import pytest

from scribe_audio.summarizer import (
    FULL_PROMPT,
    LIVE_PROMPT,
    MAX_WORDS_PER_PASS,
    REFINE_PROMPT,
    OllamaSummarizer,
    SummarizerError,
    chunk_words,
    strip_preamble,
    transcript_for_model,
)


class Recorder:
    """Captures the prompts sent, and replies with a canned answer."""

    def __init__(self, reply="## Summary\nIt was a meeting."):
        self.prompts = []
        self.reply = reply

    def __call__(self, model, prompt):
        self.prompts.append((model, prompt))
        if callable(self.reply):
            return self.reply(len(self.prompts))
        return self.reply


def transcript(words: int = 200) -> str:
    return "\n".join(
        f"Meeting: sentence number {i} about the migration project"
        for i in range(words // 6)
    )


class TestSummarize:
    def test_a_full_summary_is_produced(self):
        rec = Recorder()
        s = OllamaSummarizer(transport=rec)
        out = s.summarize(transcript(), kind="full")

        assert out.text == "## Summary\nIt was a meeting."
        assert out.kind == "full"
        assert out.passes == 1
        assert len(rec.prompts) == 1

    def test_the_transcript_is_included_in_the_prompt(self):
        rec = Recorder()
        OllamaSummarizer(transport=rec).summarize("Meeting: " + "word " * 50)
        assert "word word" in rec.prompts[0][1]

    def test_full_and_live_use_different_instructions(self):
        rec = Recorder()
        s = OllamaSummarizer(transport=rec)
        s.summarize(transcript(), kind="full")
        s.summarize(transcript(), kind="live")

        assert "## Key points" in rec.prompts[0][1]
        assert "## Key points" not in rec.prompts[1][1]
        assert "still in progress" in rec.prompts[1][1]

    def test_an_unknown_kind_is_rejected(self):
        s = OllamaSummarizer(transport=Recorder())
        with pytest.raises(ValueError):
            s.summarize(transcript(), kind="poem")

    def test_the_model_name_is_passed_through(self):
        rec = Recorder()
        OllamaSummarizer(model="qwen2.5:32b", transport=rec).summarize(transcript())
        assert rec.prompts[0][0] == "qwen2.5:32b"

    def test_an_empty_transcript_is_refused(self):
        s = OllamaSummarizer(transport=Recorder())
        with pytest.raises(SummarizerError, match="empty"):
            s.summarize("   ")

    def test_too_little_speech_is_refused(self):
        """Asking for a summary two seconds into a call is not useful."""
        s = OllamaSummarizer(transport=Recorder())
        with pytest.raises(SummarizerError, match="Not enough"):
            s.summarize("Meeting: hello there")

    def test_an_empty_model_response_is_an_error(self):
        s = OllamaSummarizer(transport=lambda m, p: "   ")
        with pytest.raises(SummarizerError, match="empty"):
            s.summarize(transcript())


class TestGrounding:
    """The prompt has to actively suppress invention: these transcripts are
    noisy and a summary is read as a record of fact."""

    @pytest.mark.parametrize("prompt", [FULL_PROMPT, LIVE_PROMPT])
    def test_both_prompts_forbid_inference(self, prompt):
        assert "ONLY what the transcript states" in prompt
        assert "Never infer" in prompt
        assert '"None" is the correct' in prompt

    @pytest.mark.parametrize("prompt", [FULL_PROMPT, LIVE_PROMPT])
    def test_both_prompts_warn_about_transcription_errors(self, prompt):
        assert "machine-generated and contains errors" in prompt

    @pytest.mark.parametrize("prompt", [FULL_PROMPT, LIVE_PROMPT])
    def test_both_prompts_explain_the_speaker_labels(self, prompt):
        assert "audio channels, not voice recognition" in prompt
        assert "usually right but not always" in prompt

    def test_the_live_prompt_asks_only_for_actionable_sections(self):
        assert "## Decisions" in LIVE_PROMPT
        assert "## Action items" in LIVE_PROMPT
        assert "## Open questions" in LIVE_PROMPT
        assert "## Summary" not in LIVE_PROMPT


def flat(text: str) -> str:
    """Collapse whitespace, so an assertion does not depend on where the
    prompt's hand-wrapped lines happen to break."""
    return " ".join(text.split())


class TestSubstanceOverTopic:
    """Measured failure on a real recording of a systems-design talk.

    The transcript explained pre-signed URLs, chunked resumable uploads, S3
    event notifications and a 30-second load balancer timeout. The summary
    came back as "Discussing handling large blobs (large files like photos or
    videos)" — section headings restated as bullets, every mechanism and
    figure dropped. A summary that only names topics is no better than the
    meeting title.
    """

    PROMPTS = [FULL_PROMPT, LIVE_PROMPT, REFINE_PROMPT]

    @pytest.mark.parametrize("prompt", PROMPTS)
    def test_the_prompt_demands_content_not_topic_labels(self, prompt):
        assert "what was SAID, not what was discussed" in flat(prompt)
        assert "names a topic without its content is worthless" in flat(prompt)

    @pytest.mark.parametrize("prompt", PROMPTS)
    def test_topic_only_phrasings_are_named_and_banned(self, prompt):
        """Naming the exact phrasings to avoid works better than describing
        the fault in the abstract."""
        for banned in ('"discussed X"', '"talked about X"', '"X was mentioned"'):
            assert banned in flat(prompt), banned

    @pytest.mark.parametrize("prompt", PROMPTS)
    def test_the_prompt_carries_a_contrasting_example(self, prompt):
        assert "Bad:" in prompt and "Good:" in prompt

    @pytest.mark.parametrize("prompt", PROMPTS)
    def test_specifics_are_called_out_as_worth_keeping(self, prompt):
        assert "Keep the specifics" in flat(prompt)
        for kind in ("Numbers", "thresholds", "durations", "named tools"):
            assert kind in flat(prompt), kind

    @pytest.mark.parametrize("prompt", [FULL_PROMPT, REFINE_PROMPT])
    def test_key_points_ask_for_the_point_not_the_topic(self, prompt):
        section = prompt.split("## Key points")[1].split("##")[0]
        assert "not the topic it belongs to" in flat(section)


class TestEmptySectionPressure:
    """The same recording produced an invented action item: a speaker's "I am
    preparing for the presentation today with <colleague>" came back as a task
    assigned to both of them. That is current activity, not a commitment, and
    the colleague had never been assigned anything. With Decisions and Open
    questions genuinely empty, the model filled the one section it could.
    """

    PROMPTS = [FULL_PROMPT, LIVE_PROMPT, REFINE_PROMPT]

    @pytest.mark.parametrize("prompt", PROMPTS)
    def test_empty_sections_are_framed_as_the_normal_case(self, prompt):
        assert "Sections are usually empty" in flat(prompt)
        assert '"None" is the correct, expected answer' in flat(prompt)

    @pytest.mark.parametrize("prompt", PROMPTS)
    def test_filling_a_section_to_avoid_emptiness_is_forbidden(self, prompt):
        assert "Never put something in a section to avoid leaving it empty" \
            in flat(prompt)

    @pytest.mark.parametrize("prompt", PROMPTS)
    def test_a_presentation_is_named_as_having_neither(self, prompt):
        """Informational content is exactly where the invention happened."""
        assert "presentation, briefing or explanation" in flat(prompt)

    @pytest.mark.parametrize("prompt", PROMPTS)
    def test_an_action_item_requires_a_commitment(self, prompt):
        assert "committing to do something not yet done" in flat(prompt)

    @pytest.mark.parametrize("prompt", PROMPTS)
    def test_current_activity_is_excluded_by_example(self, prompt):
        assert "already doing or has done is NOT an action item" in flat(prompt)
        assert "never a task, however work-like it sounds" in flat(prompt)
        assert "Never restate a transcript sentence as a task" in flat(prompt)

    @pytest.mark.parametrize("prompt", PROMPTS)
    def test_no_negative_example_quotes_the_transcript_being_summarised(
        self, prompt
    ):
        """Using the offending sentence itself as the counter-example put it
        back in the output verbatim: the model copied the illustration rather
        than avoiding the pattern. Worked examples must not collide with
        plausible transcript content."""
        assert "preparing the presentation" not in flat(prompt).lower()

    @pytest.mark.parametrize("prompt", PROMPTS)
    def test_bystanders_are_not_owners(self, prompt):
        assert "someone merely mentioned is not an owner" in flat(prompt).lower()

    @pytest.mark.parametrize("prompt", PROMPTS)
    def test_topics_are_not_promoted_into_decisions(self, prompt):
        assert "never promote a topic" in flat(prompt)
        assert "into a decision or a task" in flat(prompt)


class TestLongTranscripts:
    def test_a_long_transcript_is_summarised_in_parts_then_merged(self):
        rec = Recorder(reply=lambda n: f"part {n}")
        s = OllamaSummarizer(transport=rec)
        long_text = "\n".join(
            f"Meeting: line {i} " + "filler " * 20 for i in range(900)
        )
        out = s.summarize(long_text)

        assert len(rec.prompts) > 2          # parts, plus the merge
        assert out.passes == len(rec.prompts)
        assert "PARTS:" in rec.prompts[-1][1]

    def test_nothing_is_silently_truncated(self):
        """Every word must appear in some request."""
        rec = Recorder()
        s = OllamaSummarizer(transport=rec)
        lines = [f"Meeting: unique-marker-{i} " + "filler " * 30 for i in range(600)]
        s.summarize("\n".join(lines))

        sent = " ".join(p for _, p in rec.prompts)
        for i in (0, 250, 599):
            assert f"unique-marker-{i}" in sent

    def test_the_merge_pass_also_forbids_additions(self):
        rec = Recorder()
        s = OllamaSummarizer(transport=rec)
        s.summarize("\n".join(f"Meeting: line {i} " + "x " * 40 for i in range(400)))
        assert "Do not add anything that is not in the parts" in rec.prompts[-1][1]


class TestChunking:
    def test_short_text_is_one_chunk(self):
        assert chunk_words("a b c", 10) == ["a b c"]

    def test_chunks_respect_the_limit(self):
        text = "\n".join(f"line {i} with a few words" for i in range(200))
        for chunk in chunk_words(text, 100):
            assert len(chunk.split()) <= 100

    def test_breaks_fall_on_line_boundaries(self):
        """Splitting mid-turn hands the model half a sentence with no speaker."""
        text = "\n".join(f"Meeting: turn number {i}" for i in range(100))
        for chunk in chunk_words(text, 40):
            for line in chunk.splitlines():
                assert line.startswith("Meeting:")

    def test_a_single_overlong_line_is_still_broken_up(self):
        text = " ".join(str(i) for i in range(500))
        chunks = chunk_words(text, 100)
        assert len(chunks) == 5
        assert all(len(c.split()) <= 100 for c in chunks)

    def test_the_default_limit_covers_an_hour_of_speech(self):
        # Around 150 words a minute of conversation.
        assert MAX_WORDS_PER_PASS >= 60 * 140


class TestStatus:
    def test_unreachable_ollama_reports_how_to_fix_it(self):
        # Port 1: nothing listens there, so this does not depend on whether
        # the developer has Ollama running.
        s = OllamaSummarizer(base_url="http://127.0.0.1:1")
        status = s.status()
        assert status["available"] is False
        assert "not reachable" in status["detail"]
        assert "brew install ollama" in status["hint"]

    def test_a_transport_is_always_considered_available(self):
        assert OllamaSummarizer(transport=Recorder()).status()["available"] is True


class TestRendering:
    def test_turns_become_speaker_labelled_lines(self):
        text = transcript_for_model([
            {"speaker": "You", "text": "my question"},
            {"speaker": "Meeting", "text": "their answer"},
        ])
        assert text == "You: my question\nMeeting: their answer"

    def test_blank_turns_are_dropped(self):
        assert transcript_for_model([{"speaker": "You", "text": "  "}]) == ""

    def test_a_missing_speaker_gets_a_neutral_label(self):
        assert transcript_for_model([{"text": "unattributed"}]) == "Transcript: unattributed"

    def test_conversational_preamble_is_stripped(self):
        assert strip_preamble("Sure! Here is the summary:\n\n## Summary\nx") \
            == "## Summary\nx"

    def test_a_clean_summary_is_untouched(self):
        assert strip_preamble("## Summary\nIt was fine.") == "## Summary\nIt was fine."

    def test_a_bulleted_opening_is_untouched(self):
        assert strip_preamble("- first point\n- second") == "- first point\n- second"


class TestCompaction:
    """The bound is enforced on the text, not asked for in the prompt.

    Measured on qwen2.5:14b, a running summary instructed to stay under 250
    words reached 555 after nine passes and per-pass latency grew from 15s to
    27s with it. A model cannot be relied on to respect a limit.
    """

    def bloated(self, bullets=20):
        return (
            "## Summary\nOne. Two. Three. Four. Five. Six.\n\n"
            "## Key points\n" + "\n".join(f"- point {i}" for i in range(bullets))
            + "\n\n## Decisions\n" + "\n".join(f"- decision {i}" for i in range(bullets))
            + "\n\n## Action items\n"
            + "\n".join(f"- [ ] task {i}" for i in range(bullets))
            + "\n\n## Open questions\n"
            + "\n".join(f"- question {i}" for i in range(bullets))
        )

    def test_each_section_is_capped(self):
        from scribe_audio.summarizer import MAX_BULLETS, compact

        out = compact(self.bloated())
        for section, limit in MAX_BULLETS.items():
            heading = "## " + section.title().replace("Points", "points") \
                .replace("Items", "items").replace("Questions", "questions")
            block = out.lower().split(f"## {section}")[1].split("##")[0]
            assert block.count("\n-") <= limit, section

    def test_the_prose_section_is_capped(self):
        from scribe_audio.summarizer import MAX_SUMMARY_SENTENCES, compact

        out = compact(self.bloated())
        prose = out.split("## Key points")[0]
        assert prose.count(".") <= MAX_SUMMARY_SENTENCES

    def test_the_earliest_bullets_are_the_ones_kept(self):
        """They have been folded and merged over more passes."""
        from scribe_audio.summarizer import compact

        out = compact(self.bloated())
        assert "- point 0" in out
        assert "- point 19" not in out

    def test_a_summary_inside_its_limits_is_untouched(self):
        from scribe_audio.summarizer import compact

        tidy = ("## Summary\nIt was short.\n\n## Key points\n- one\n- two\n\n"
                "## Decisions\nNone")
        assert compact(tidy) == tidy

    def test_section_headings_survive(self):
        from scribe_audio.summarizer import compact

        out = compact(self.bloated())
        for heading in ("## Summary", "## Key points", "## Decisions",
                        "## Action items", "## Open questions"):
            assert heading in out

    def test_empty_input(self):
        from scribe_audio.summarizer import compact

        assert compact("") == ""

    def test_refine_bounds_both_directions(self):
        """The cap applies to what goes in and to what comes out, so neither
        the input nor the stored summary can grow without bound."""
        from scribe_audio.summarizer import OllamaSummarizer

        seen = []

        def transport(model, prompt):
            seen.append(prompt)
            return self.bloated()

        s = OllamaSummarizer(transport=transport)
        out = s.refine(self.bloated(), "Meeting: " + "new words " * 30)

        assert "point 19" not in seen[0], "the fed-back summary must be trimmed"
        assert "point 19" not in out.text, "the returned summary must be trimmed"

    def test_repeated_refinement_does_not_grow(self):
        """Nine passes, as in the measurement that exposed the problem."""
        from scribe_audio.summarizer import OllamaSummarizer

        s = OllamaSummarizer(transport=lambda m, p: self.bloated())
        running, sizes = "", []
        for _ in range(9):
            running = s.refine(running, "Meeting: " + "more speech " * 40).text
            sizes.append(len(running.split()))

        assert sizes[0] == sizes[-1], f"summary grew: {sizes}"


class TestResourceFootprint:
    """A 14B model held 21GB resident and drove a 48GB machine 23GB into swap
    when it stayed loaded between summaries. Measured after these two changes:
    9.9GB while summarising, 0GB a minute later."""

    def _payload_of(self, **kwargs):
        """The JSON body the summariser would send to Ollama."""
        import json

        sent = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return json.dumps({"response": "## Summary\nx"}).encode()

        def fake_urlopen(request, timeout=None):
            sent.update(json.loads(request.data.decode()))
            return FakeResponse()

        import urllib.request

        original = urllib.request.urlopen
        urllib.request.urlopen = fake_urlopen
        try:
            OllamaSummarizer(**kwargs).refine("", "Meeting: " + "word " * 60)
        finally:
            urllib.request.urlopen = original
        return sent

    def test_the_model_is_released_after_a_request(self):
        from scribe_audio.summarizer import DEFAULT_KEEP_ALIVE

        assert self._payload_of()["keep_alive"] == DEFAULT_KEEP_ALIVE

    def test_the_release_window_is_short(self):
        """Long enough that two presses in a row do not both pay the load,
        short enough that the model is not resident between summaries."""
        from scribe_audio.summarizer import DEFAULT_KEEP_ALIVE

        assert DEFAULT_KEEP_ALIVE.endswith("s")
        assert int(DEFAULT_KEEP_ALIVE.rstrip("s")) <= 300

    def test_the_context_window_is_capped(self):
        from scribe_audio.summarizer import DEFAULT_CONTEXT

        payload = self._payload_of()
        assert payload["options"]["num_ctx"] == DEFAULT_CONTEXT
        # Large enough for a compacted summary plus a chunk of transcript,
        # small enough not to allocate a huge KV cache.
        assert 4096 <= DEFAULT_CONTEXT <= 16384

    def test_both_are_configurable(self):
        payload = self._payload_of(keep_alive="5m", context=4096)
        assert payload["keep_alive"] == "5m"
        assert payload["options"]["num_ctx"] == 4096

    def test_temperature_stays_low(self):
        """Summarising is extraction, not writing."""
        assert self._payload_of()["options"]["temperature"] <= 0.3
