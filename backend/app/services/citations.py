"""Citations: turn [n] markers into verifiable file and line references.

Stage 7 of the RAG pipeline, and the third of the three defences against
invented code described in `generator.py`.

WHY THIS IS NOT JUST STRING FORMATTING
--------------------------------------
The model was handed numbered sources and told to cite them. It can still write
[7] when only three sources exist. That marker looks exactly as authoritative
as a real one, and a UI that renders it produces a citation pointing nowhere —
which is worse than no citation, because it carries the appearance of evidence.

So every marker is checked against the sources that were actually sent. Valid
markers become Citation objects carrying a real path and line range. Invalid
ones are removed from the displayed answer and recorded in `invalid_markers`,
so the failure is auditable rather than silently rendered.

THE TRAP: CODE CONTAINS BRACKETS
--------------------------------
An answer explaining code frequently contains code:

    The handler reads `items[0]` and returns `results[2]`.

A naive regex sees citations 0 and 2, strips `[0]` as invalid, and corrupts the
code it was meant to explain. Two defences:

1. Fenced blocks and inline code spans are masked before scanning, so anything
   the model marked as code is off limits.
2. A marker preceded by a word character, `)` or `]` is an index expression,
   not a citation: `arr[1]` and `foo()[2]` are skipped even outside code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.services.retriever import RetrievedChunk

# A marker is [1], [2,3] or [1-3], NOT preceded by an identifier or a closing
# paren -- `arr[1]` and `foo()[2]` are index expressions, not citations.
#
# A preceding `]` is deliberately NOT excluded here, because the system prompt
# asks the model to write adjacent citations as [2][3]. Telling that apart from
# a chained index like `grid[1][2]` needs position, not a lookbehind, so
# find_markers() handles it.
_MARKER = re.compile(r"(?<![\w\)])\[\s*(\d+(?:\s*[,\-]\s*\d+)*)\s*\]")

# Fenced blocks first, then inline spans; both are masked before scanning.
_FENCE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`\n]*`")

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Citation:
    """A validated reference from the answer back to real source code."""

    index: int          # the [n] the model wrote
    chunk_id: int
    file_path: str
    start_line: int
    end_line: int
    symbol: str | None
    language: str
    kind: str
    score: float

    @property
    def citation(self) -> str:
        return f"{self.file_path}:{self.start_line}-{self.end_line}"

    @property
    def label(self) -> str:
        return f"{self.citation} ({self.symbol})" if self.symbol else self.citation

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "chunk_id": self.chunk_id,
            "citation": self.citation,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "symbol": self.symbol,
            "language": self.language,
            "kind": self.kind,
            "score": round(self.score, 4),
        }

    @classmethod
    def from_chunk(cls, index: int, chunk: RetrievedChunk) -> Citation:
        return cls(
            index=index,
            chunk_id=chunk.chunk_id,
            file_path=chunk.file_path,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            symbol=chunk.symbol,
            language=chunk.language,
            kind=chunk.kind,
            score=chunk.score,
        )


@dataclass
class CitedAnswer:
    """An answer with its citations resolved and its bad markers removed."""

    answer: str                                  # display text, cleaned
    raw_answer: str                              # exactly what the model wrote
    citations: list[Citation] = field(default_factory=list)
    unused_sources: list[Citation] = field(default_factory=list)
    invalid_markers: list[int] = field(default_factory=list)
    citation_coverage: float = 0.0

    @property
    def has_citations(self) -> bool:
        return bool(self.citations)

    @property
    def is_grounded(self) -> bool:
        """Cited at least one real source and fabricated none.

        A weak signal, not proof: it says the answer POINTS at real code, not
        that the code supports the claim. Measuring the latter needs a judge
        over an evaluation set (spec section 20).
        """
        return self.has_citations and not self.invalid_markers

    def as_dict(self) -> dict:
        return {
            "answer": self.answer,
            "citations": [c.as_dict() for c in self.citations],
            "unused_sources": [c.as_dict() for c in self.unused_sources],
            "invalid_markers": self.invalid_markers,
            "citation_coverage": round(self.citation_coverage, 3),
            "is_grounded": self.is_grounded,
        }


def mask_code(text: str) -> str:
    """Blank out code spans, preserving length so offsets stay valid."""

    def blank(match: re.Match) -> str:
        # Keep newlines so line-based logic downstream still works.
        return "".join("\n" if ch == "\n" else " " for ch in match.group(0))

    return _INLINE_CODE.sub(blank, _FENCE.sub(blank, text))


def _expand(group: str) -> list[int]:
    """Expand a marker body: '1' -> [1], '1,3' -> [1,3], '1-3' -> [1,2,3]."""
    numbers: list[int] = []
    for part in group.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, _, end = part.partition("-")
            try:
                low, high = int(start), int(end)
            except ValueError:
                continue
            # Guard against a nonsense range like [1-9999] producing thousands
            # of entries; they would all be invalid anyway.
            if 0 <= high - low <= 100:
                numbers.extend(range(low, high + 1))
        else:
            try:
                numbers.append(int(part))
            except ValueError:
                continue
    return numbers


def find_markers(text: str) -> list[tuple[re.Match, list[int]]]:
    """Locate citation markers outside code, with the numbers they name.

    Resolves the [2][3] vs grid[1][2] ambiguity by position: a marker sitting
    immediately after a `]` counts only if that `]` closed a marker we already
    accepted. In `grid[1][2]` the first bracket is rejected by the lookbehind,
    so the second has nothing valid to chain from and is rejected too.
    """
    masked = mask_code(text)
    found: list[tuple[re.Match, list[int]]] = []
    previous_end = -1

    for match in _MARKER.finditer(masked):
        start = match.start()
        if start > 0 and masked[start - 1] == "]" and start != previous_end:
            continue
        found.append((match, _expand(match.group(1))))
        previous_end = match.end()

    return found


def citation_coverage(text: str) -> float:
    """Fraction of prose sentences carrying at least one citation.

    A groundedness proxy for the evaluation stage: an answer where only the
    first sentence is cited and the next five are not is worth flagging, even
    though the citation it does have is perfectly valid.
    """
    masked = mask_code(text)
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(masked) if s.strip()]
    # Ignore fragments too short to make a claim.
    sentences = [s for s in sentences if len(s.split()) >= 4]
    if not sentences:
        return 0.0
    cited = sum(1 for s in sentences if _MARKER.search(s))
    return cited / len(sentences)


def attach_citations(
    answer: str,
    sources: list[RetrievedChunk],
    *,
    strip_invalid: bool = True,
) -> CitedAnswer:
    """Validate the markers in `answer` against the sources actually sent.

    `sources` must be in the same order they were numbered into the prompt:
    position 0 is [1]. Passing a re-sorted list would silently repoint every
    citation at the wrong code.
    """
    raw = answer or ""
    valid_range = range(1, len(sources) + 1)

    used: dict[int, Citation] = {}
    invalid: list[int] = []
    removals: list[tuple[int, int]] = []

    for match, numbers in find_markers(raw):
        good = [n for n in numbers if n in valid_range]
        bad = [n for n in numbers if n not in valid_range]

        for n in good:
            used.setdefault(n, Citation.from_chunk(n, sources[n - 1]))
        for n in bad:
            if n not in invalid:
                invalid.append(n)

        # Only remove a marker when every number in it was bogus. A mixed
        # marker like [2,9] keeps its valid part rather than losing [2] too.
        if bad and not good:
            removals.append(match.span())

    display = _remove_spans(raw, removals) if strip_invalid else raw

    citations = [used[n] for n in sorted(used)]
    unused = [
        Citation.from_chunk(i, chunk)
        for i, chunk in enumerate(sources, start=1)
        if i not in used
    ]

    return CitedAnswer(
        answer=display,
        raw_answer=raw,
        citations=citations,
        unused_sources=unused,
        invalid_markers=sorted(invalid),
        citation_coverage=citation_coverage(display),
    )


def _remove_spans(text: str, spans: list[tuple[int, int]]) -> str:
    """Cut spans out of text and tidy the whitespace they leave behind."""
    if not spans:
        return text

    result: list[str] = []
    cursor = 0
    for start, end in sorted(spans):
        result.append(text[cursor:start])
        cursor = end
    result.append(text[cursor:])
    cleaned = "".join(result)

    # A removed marker leaves "the token [9] is checked" -> "the token  is
    # checked", and " ." when it sat before punctuation.
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)
    return cleaned.strip()


def format_sources_block(citations: list[Citation]) -> str:
    """Render the trailing 'Sources:' list shown under an answer (spec 13)."""
    if not citations:
        return ""
    lines = ["Sources:"]
    for citation in citations:
        symbol = f"  ({citation.symbol})" if citation.symbol else ""
        lines.append(f"- [{citation.index}] {citation.citation}{symbol}")
    return "\n".join(lines)
