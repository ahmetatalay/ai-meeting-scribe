"""Finding earlier recordings that discuss the same things.

This exists so cross-meeting context can be offered *without* feeding several
transcripts to a model. Given a pile of unrelated meetings and asked to
summarise one, an LLM invents cross-references — "as agreed on 21 October" for
agreements that never happened — and a meeting summary is read as a record of
fact.

So the connection is found here, cheaply and deterministically, by term
overlap; the result is presented as links with the shared terms shown, for the
reader to verify. Nothing is asserted. If the reader decides a particular older
meeting really is relevant, they can include it deliberately.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

from . import library

WORD = re.compile(r"[A-Za-zÀ-ÿĞğİıŞşÖöÜüÇç][A-Za-zÀ-ÿĞğİıŞşÖöÜüÇç'-]{2,}")

# Terms carrying no topical signal. Deliberately small: a long hand-built list
# is a maintenance burden, and the document-frequency weighting below already
# suppresses words that appear in every meeting.
STOPWORDS = frozenset("""
the and that have for not with you this but his from they say her she will one
all would there their what out about who get which when make can like time just
him know take people into year your good some could them see other than then
now look only come its over also back after use two how our work first well way
even new want because any these give day most thing yeah okay right mean going
really think something actually maybe sure very much more lot bit kind sort
you're i'm it's that's we're don't doesn't didn't let's gonna
mean said says tell told talk talking speak speaking
are was were has had have been being does did doing done
should must may might shall cannot won't isn't aren't wasn't weren't
there's they're we've i've you've he's she's there are
thing things stuff quite pretty basically literally honestly
need needs needed let put got get getting
meeting transcript you meeting
bir bu ve için ile ama çok daha gibi olarak oldu olan var yok ben sen biz
şey evet hayır tamam yani sonra önce kadar diye zaten belki
""".split())

MIN_TERM_LENGTH = 3
TOP_TERMS = 25
DEFAULT_LIMIT = 5
MIN_SHARED_TERMS = 2


def terms(text: str) -> Counter:
    """Topical term frequencies for one transcript."""
    counts: Counter = Counter()
    for match in WORD.finditer(text.lower()):
        word = match.group(0).strip("'-")
        if len(word) < MIN_TERM_LENGTH or word in STOPWORDS:
            continue
        counts[word] += 1
    return counts


def salient(counts: Counter, document_frequency: Counter, documents: int,
            limit: int = TOP_TERMS) -> list[tuple[str, float]]:
    """The terms that distinguish this transcript from the others.

    Plain frequency surfaces words common to every meeting, so each count is
    weighted down by how many transcripts the term appears in.
    """
    scored = []
    for word, count in counts.items():
        appears_in = max(1, document_frequency.get(word, 1))
        weight = math.log((documents + 1) / appears_in)
        if weight <= 0:
            continue
        scored.append((word, count * weight))
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return scored[:limit]


def find(raw_dir: Path | str, name: str, limit: int = DEFAULT_LIMIT) -> list[dict]:
    """Recordings that share distinctive terms with ``name``, best first.

    Returns an empty list when nothing overlaps convincingly — which is the
    common and correct answer for unrelated meetings.
    """
    directory = Path(raw_dir).expanduser()
    target_path = library.resolve(directory, name)
    if not target_path.is_file():
        raise FileNotFoundError(name)

    entries = library.scan(directory)
    if len(entries) < 2:
        return []

    bodies: dict[str, Counter] = {}
    document_frequency: Counter = Counter()
    for entry in entries:
        try:
            raw = (directory / entry["name"]).read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError:
            continue
        # Only what was actually said. Reading the raw file would include the
        # frontmatter keys and headings, whose words ("title", "started") are
        # shared by every transcript and linked unrelated meetings together.
        text = " ".join(t["text"] for t in library.parse_turns(raw))
        counts = terms(text)
        bodies[entry["name"]] = counts
        document_frequency.update(counts.keys())

    if name not in bodies:
        return []

    documents = len(bodies)
    target_terms = dict(salient(bodies[name], document_frequency, documents))
    if not target_terms:
        return []

    by_name = {e["name"]: e for e in entries}
    results = []
    for other, counts in bodies.items():
        if other == name:
            continue
        other_terms = dict(salient(counts, document_frequency, documents))
        shared = set(target_terms) & set(other_terms)
        if len(shared) < MIN_SHARED_TERMS:
            continue

        # Weight by both sides' scores, so a term central to both meetings
        # counts for more than one mentioned in passing.
        score = sum(
            math.sqrt(target_terms[w] * other_terms[w]) for w in shared
        )
        entry = by_name[other]
        results.append({
            "name": other,
            "title": entry["title"],
            "started": entry["started"],
            "score": round(score, 2),
            "shared_terms": sorted(shared, key=lambda w: -target_terms[w])[:8],
        })

    results.sort(key=lambda r: -r["score"])
    return results[:limit]
