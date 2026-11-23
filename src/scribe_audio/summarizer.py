"""Meeting summaries from a local LLM via Ollama.

Summarisation runs on the same machine as everything else, so a transcript
never leaves it. That matters more here than usual: these files are verbatim
records of performance conversations, interviews and private asides.

Two shapes of output, because they are read in different situations:

* ``live`` — decisions, action items and open questions, as bullets. What is
  actually useful to glance at mid-meeting; nobody reads a prose paragraph
  while a call is in progress.
* ``full`` — the complete write-up, generated once a recording is finished.

A summary covers **one** meeting. Feeding a model several unrelated transcripts
and asking it to summarise one of them produces confident invented
cross-references ("as agreed on 21 October...") for agreements that never
happened — and a meeting summary is treated as a record of fact. Connections
between meetings are surfaced separately, as links the reader verifies, by
:mod:`scribe_audio.related`.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass

OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen2.5:14b"
REQUEST_TIMEOUT = 300.0

# How long Ollama keeps the model resident after a request. A 14B model costs
# around 20GB once loaded, which is most of a 48GB machine and pushed it deep
# into swap when it stayed resident between summaries. Summarising is
# occasional, so the model is released shortly after each one; a short window
# still means two presses in a row do not both pay the load.
DEFAULT_KEEP_ALIVE = "60s"

# Context window requested from Ollama. The compaction above bounds the prompt
# to a few thousand tokens, so a large window only buys a much larger KV cache
# — which is a big part of why the resident size ran to 21GB.
DEFAULT_CONTEXT = 8192

# Transcripts are chunked below this word count per request. Roughly 13k tokens
# of transcript, comfortably inside a 32k context with room for the prompt and
# the answer.
MAX_WORDS_PER_PASS = 9000

# Grounding rules shared by both prompts. Transcripts are noisy — Whisper
# mishears names ("Airbnb" as "Arabic and B"), and the speaker labels are
# energy-based and occasionally wrong — so the model is told what it is reading
# rather than being left to trust it.
_GROUNDING = """\
You are summarising a transcript of one meeting. Follow these rules exactly:

- Use ONLY what the transcript states. Never infer, assume or add outside
  knowledge. Do not invent names, dates, numbers or decisions.
- Write what was SAID, not what was discussed. A bullet that names a topic
  without its content is worthless to someone who missed the meeting.
    Bad:  - Discussing how to handle large file uploads
    Good: - Large uploads go straight to blob storage via a short-lived
            pre-signed URL, in chunks, so a dropped connection resumes
            instead of restarting
  Never write a bullet whose only content is "discussed X", "talked about X"
  or "X was mentioned". Say what was actually claimed, decided or explained.
- Keep the specifics. Numbers, thresholds, durations, named tools, named
  mechanisms and concrete examples are the most valuable thing in a
  transcript and the first thing a vague summary loses. If the transcript
  gives a figure or a mechanism by name, it belongs in the summary.
  Dropping a figure the transcript supplies is an error:
    Bad:  - Offloaded to background work to prevent load balancer timeouts
    Good: - The load balancer cuts the connection at 30 seconds, so the API
            server queues the job and returns a job ID instead of rendering
  Do not generalise a stated quantity into "large", "several" or "some".
- Use the bullet budget you are given. A dense discussion should produce
  several bullets, not three: do not merge distinct mechanisms into one
  bullet while there is room left, because merging is what loses the
  specifics. Merge only once you are at the limit.
- Sections are usually empty, and "None" is the correct, expected answer.
  Most conversations contain no decisions and no action items at all; a
  presentation, briefing or explanation normally has none of either. Never
  put something in a section to avoid leaving it empty, and never promote a
  topic, an observation or a restated key point into a decision or a task.
- An action item requires someone committing to do something not yet done.
  A description of what someone is already doing or has done is NOT an action
  item. A sentence in the present or past tense reporting activity ("I'm
  looking at it", "we moved the host last week") is never a task, however
  work-like it sounds; only a commitment to future work is. Never restate a
  transcript sentence as a task by rewording it. Only name an owner the
  transcript actually assigns; someone merely mentioned is not an owner.
- The transcript is machine-generated and contains errors: misheard words,
  mangled names and occasional repeated phrases. Where wording is clearly
  garbled, describe the meaning rather than quoting it. Do not guess at names
  you cannot read.
- Speaker labels come from audio channels, not voice recognition. "You" is the
  person recording; "Meeting" is everyone else on the call. These labels are
  usually right but not always, so avoid attributing anything contentious to a
  specific speaker unless the transcript makes it unambiguous.
- Be concise and concrete. Prefer the transcript's own terms.
- Write in English, whatever language the meeting is held in.
"""

LIVE_PROMPT = _GROUNDING + """
This meeting is still in progress; you are seeing it so far. Produce only:

## Decisions
- Anything explicitly agreed or settled. "None" if nothing was.

## Action items
- [ ] Work someone committed to do — owner only if the transcript assigns one.
      "None" if nobody committed to anything.

## Open questions
- Anything raised but not resolved

No preamble, no closing remarks, no other sections.
"""

FULL_PROMPT = _GROUNDING + """
Produce:

## Summary
Two or three sentences on what this meeting was about.

## Key points
- One bullet per substantive point, stating the point itself — the claim, the
  mechanism, the number — not the topic it belongs to.

## Decisions
- Anything explicitly agreed or settled. "None" if nothing was.

## Action items
- [ ] Work someone committed to do — owner only if the transcript assigns one.
      "None" if nobody committed to anything.

## Open questions
- Anything raised but left unresolved

No preamble and no closing remarks.
"""

REFINE_PROMPT = _GROUNDING + """
You are maintaining a running summary of a meeting that is still going on. You
are given the summary so far and only the newest part of the transcript.

Return the updated summary of the whole meeting so far, with these sections:

## Summary
At most three sentences on what the meeting has been about.

## Key points
- At most 8 bullets, each stating a substantive point — the claim, the
  mechanism, the number — not the topic it belongs to.

## Decisions
- Anything explicitly agreed or settled. "None" if nothing was.

## Action items
- [ ] Work someone committed to do — owner only if the transcript assigns one.
      "None" if nobody committed to anything.

## Open questions
- Only things explicitly asked and not answered, or left unresolved. Not
  restatements of key points.

How to update it:

- Keep everything in the summary so far that the new transcript does not
  change. Never drop earlier material just because it is not in the new part.
- Fold the new material into the existing sections. Do not append a separate
  block for it and do not mark what is new.
- If the new transcript resolves or corrects something, revise that line rather
  than adding a second one. Move a resolved open question into Decisions.
- Never state the same point twice.
- Return the whole summary, not a diff. No preamble, no closing remarks.

Keep it short. The whole summary must stay under 250 words, and Key points
under 8 bullets: merge related bullets into one rather than letting the list
grow. This is a hard limit — a summary that grows without bound takes longer to
update on every pass until it cannot keep up with the meeting.
"""


REDUCE_PROMPT = _GROUNDING + """
Below are summaries of consecutive parts of ONE meeting. Merge them into a
single summary of the whole meeting, removing repetition and keeping the
section structure. Do not add anything that is not in the parts.
"""


class SummarizerError(RuntimeError):
    """Summarisation could not be completed."""


@dataclass
class Summary:
    text: str
    model: str
    kind: str
    passes: int = 1

    def as_dict(self) -> dict:
        from datetime import datetime

        return {
            "text": self.text,
            "model": self.model,
            "kind": self.kind,
            "passes": self.passes,
            "generated": datetime.now().isoformat(timespec="seconds"),
        }


def _post(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _get(url: str, timeout: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class OllamaSummarizer:
    """Summarises transcripts with a model served by a local Ollama.

    ``transport`` is the seam the tests use; it stands in for the two HTTP calls
    this needs, so nothing in the test suite requires Ollama to be running.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = OLLAMA_URL,
        timeout: float = REQUEST_TIMEOUT,
        keep_alive: str = DEFAULT_KEEP_ALIVE,
        context: int = DEFAULT_CONTEXT,
        transport=None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.keep_alive = keep_alive
        self.context = context
        self._transport = transport

    # ------------------------------------------------------------- transport

    def _generate(self, prompt: str) -> str:
        """Obtain one completion, validating the result the same way either way."""
        text = self._raw_generate(prompt).strip()
        if not text:
            raise SummarizerError("The model returned an empty response.")
        return strip_preamble(text)

    def _raw_generate(self, prompt: str) -> str:
        if self._transport is not None:
            return self._transport(self.model, prompt)
        try:
            data = _post(
                f"{self.base_url}/api/generate",
                {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    # Released shortly after the request rather than sitting on
                    # ~20GB between summaries.
                    "keep_alive": self.keep_alive,
                    "options": {
                        # Low temperature: this is extraction, not writing.
                        "temperature": 0.2,
                        "num_ctx": self.context,
                    },
                },
                self.timeout,
            )
        except urllib.error.URLError as exc:
            raise SummarizerError(
                f"Could not reach Ollama at {self.base_url}. Start it with "
                f"'ollama serve', or install it with 'brew install ollama'. ({exc})"
            ) from exc
        except TimeoutError as exc:
            raise SummarizerError(
                f"Ollama did not respond within {self.timeout:.0f}s. A smaller "
                f"model may help."
            ) from exc

        return data.get("response") or ""

    def status(self) -> dict:
        """Whether summarisation is usable right now, and why not if it isn't."""
        if self._transport is not None:
            return {"available": True, "model": self.model, "models": [self.model]}
        try:
            data = _get(f"{self.base_url}/api/tags", timeout=5.0)
        except Exception as exc:
            return {
                "available": False,
                "model": self.model,
                "models": [],
                "detail": f"Ollama is not reachable at {self.base_url}",
                "hint": "brew install ollama && ollama serve",
                "error": str(exc),
            }

        names = [m.get("name", "") for m in data.get("models", [])]
        # Ollama reports "qwen2.5:14b"; accept a bare name as matching any tag.
        installed = any(
            n == self.model or n.split(":")[0] == self.model.split(":")[0]
            for n in names
        )
        return {
            "available": installed,
            "model": self.model,
            "models": names,
            "detail": None if installed else f"Model '{self.model}' is not pulled",
            "hint": None if installed else f"ollama pull {self.model}",
        }

    # ------------------------------------------------------------ summarising

    def refine(self, existing: str, new_text: str) -> Summary:
        """Fold the newest transcript into an existing running summary.

        Re-reading the whole transcript every few seconds is both slow and
        unstable — the summary rewrites itself each pass. Refining keeps the
        earlier material intact and only has to read what is new, which is what
        makes a 10-15 second cadence affordable.
        """
        new_text = new_text.strip()
        if not new_text:
            raise SummarizerError("Nothing new has been said.")

        # Bounded in code before it goes back in: see compact().
        existing = compact((existing or "").strip()) or "(nothing yet)"
        prompt = (
            f"{REFINE_PROMPT}\n\nSUMMARY SO FAR:\n{existing}\n\n"
            f"NEW PART OF THE TRANSCRIPT:\n{new_text}"
        )
        return Summary(text=compact(self._generate(prompt)), model=self.model,
                       kind="live", passes=1)

    def summarize(self, transcript: str, kind: str = "full") -> Summary:
        """Summarise one transcript.

        Long transcripts are summarised in parts and the parts merged, rather
        than being silently truncated.
        """
        if kind not in ("full", "live"):
            raise ValueError(f"kind must be 'full' or 'live', got {kind!r}")

        body = transcript.strip()
        if not body:
            raise SummarizerError("The transcript is empty.")
        if len(body.split()) < 20:
            raise SummarizerError(
                "Not enough has been said yet to summarise."
            )

        instruction = LIVE_PROMPT if kind == "live" else FULL_PROMPT
        chunks = chunk_words(body, MAX_WORDS_PER_PASS)

        if len(chunks) == 1:
            return Summary(
                text=self._generate(f"{instruction}\n\nTRANSCRIPT:\n{chunks[0]}"),
                model=self.model, kind=kind, passes=1,
            )

        parts = [
            self._generate(
                f"{instruction}\n\nTRANSCRIPT (part {i} of {len(chunks)}):\n{chunk}"
            )
            for i, chunk in enumerate(chunks, start=1)
        ]
        merged = self._generate(
            REDUCE_PROMPT + "\n\nPARTS:\n\n" + "\n\n---\n\n".join(parts)
        )
        return Summary(text=merged, model=self.model, kind=kind,
                       passes=len(chunks) + 1)


def chunk_words(text: str, limit: int) -> list[str]:
    """Split text into chunks of at most ``limit`` words, on turn boundaries.

    Splitting mid-turn would hand the model half a sentence with no speaker, so
    breaks are taken between lines wherever possible.
    """
    words = text.split()
    if len(words) <= limit:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    count = 0
    for line in text.splitlines():
        line_words = len(line.split())
        if count + line_words > limit and current:
            chunks.append("\n".join(current))
            current, count = [], 0
        current.append(line)
        count += line_words
    if current:
        chunks.append("\n".join(current))

    # A single line longer than the limit still has to be broken somewhere.
    out: list[str] = []
    for chunk in chunks:
        chunk_words_list = chunk.split()
        if len(chunk_words_list) <= limit:
            out.append(chunk)
        else:
            for i in range(0, len(chunk_words_list), limit):
                out.append(" ".join(chunk_words_list[i:i + limit]))
    return out


def transcript_for_model(turns: list[dict]) -> str:
    """Render turns as the plain speaker-labelled text the prompts describe."""
    lines = []
    for turn in turns:
        text = (turn.get("text") or "").strip()
        if not text:
            continue
        speaker = turn.get("speaker") or "Transcript"
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


# Hard caps applied in code. Asking the model to respect a word limit does not
# work: measured on qwen2.5:14b, a running summary told to stay under 250 words
# reached 555 after nine passes, and per-pass latency grew from 15s to 27s with
# it. The bound has to be enforced on the text, not requested in the prompt.
MAX_BULLETS = {
    "key points": 8,
    "decisions": 8,
    "action items": 10,
    "open questions": 6,
}
MAX_SUMMARY_SENTENCES = 3


def compact(summary: str) -> str:
    """Trim a running summary back inside its limits.

    Keeps the earliest bullets in each section — they are the ones already
    folded and merged, so they carry the most consolidated meaning — and drops
    the overflow. Applied before the summary is fed back into the next pass, so
    the input to each pass stays bounded however long the meeting runs.
    """
    out: list[str] = []
    section = ""
    kept = 0

    for raw in summary.splitlines():
        line = raw.rstrip()
        stripped = line.strip()

        if stripped.startswith("#"):
            section = stripped.lstrip("#").strip().lower()
            kept = 0
            out.append(line)
            continue

        if not stripped:
            out.append(line)
            continue

        if stripped.startswith(("-", "*")):
            limit = MAX_BULLETS.get(section)
            if limit is not None and kept >= limit:
                continue
            kept += 1
            out.append(line)
            continue

        if section == "summary":
            # Keep only the first few sentences of the prose section.
            sentences = re.split(r"(?<=[.!?])\s+", stripped)
            out.append(" ".join(sentences[:MAX_SUMMARY_SENTENCES]))
            continue

        out.append(line)

    # Collapse any blank runs the trimming left behind.
    text = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def strip_preamble(summary: str) -> str:
    """Remove a leading conversational line some models add anyway.

    The prompt forbids preamble; smaller models still occasionally open with
    "Here is the summary:".
    """
    lines = summary.strip().splitlines()
    while lines and not lines[0].lstrip().startswith(("#", "-", "*", "[")):
        if re.match(r"^\s*(here|below|sure|certainly|okay)\b", lines[0], re.I):
            lines.pop(0)
            continue
        break
    return "\n".join(lines).strip()
