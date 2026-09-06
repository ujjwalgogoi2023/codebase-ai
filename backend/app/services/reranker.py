"""Reranking: choose which retrieved chunks the LLM actually sees.

Stage 10, sitting between retrieval and generation:

    question -> vector search -> top 10 candidates -> [RERANK] -> top 3 -> LLM

WHY VECTOR SIMILARITY IS NOT ENOUGH
-----------------------------------
Embedding search compares a question to a chunk by squashing each into a single
vector and measuring the angle between them. That is fast enough to scan
100,000 chunks, and it is lossy for exactly that reason: the whole meaning of a
40-line function is compressed into one point before the question is ever seen.

So similarity is a proxy for relevance, not relevance itself. A chunk full of
authentication vocabulary scores well against "where do we validate JWTs?" even
when it only *imports* the JWT library. Meanwhile the middleware that actually
validates the token may score seventh, because most of its text is error
handling.

A reranker looks at the question and one candidate TOGETHER and asks the direct
question: does this chunk answer this? That is far more accurate and far more
expensive, which is why it runs over ten candidates rather than the whole
corpus. Retrieve wide and cheap, then re-score narrow and well.

    "Vector search gives us semantically similar documents, but similarity
     doesn't necessarily mean relevance to the specific question. So I retrieve
     a larger candidate set and rerank it before passing context to the LLM."

WHY LISTWISE, NOT POINTWISE
---------------------------
Scoring each candidate in its own request costs one LLM call per candidate: ten
calls per question, and the eval multiplies that by fifty questions and several
configurations. Sending all candidates in one prompt and asking for a ranked
order costs a single call, and gives the model the additional advantage of
seeing the candidates side by side, which is what ranking actually requires.

FAILING SOFT
------------
Reranking is an improvement, not a dependency. If the model returns malformed
output, or the call fails outright, we fall back to the vector ordering rather
than failing the question. A slightly worse answer beats an error page.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from app.config import Settings, get_settings
from app.services.llm import LLMProvider, get_llm_provider
from app.services.retriever import RetrievedChunk

RERANK_SYSTEM_PROMPT = """\
You rank code snippets by how well each one ANSWERS a question about a \
codebase.

Judge whether a snippet contains the answer, not whether it shares vocabulary \
with the question. A snippet that merely imports or mentions the relevant \
library is less useful than one that implements the behaviour being asked \
about.

Reply with ONLY a JSON array of candidate numbers, most relevant first, for \
example: [3, 1, 7]

Include every candidate number exactly once. Output no prose, no explanation \
and no code fences."""

# How much of each candidate the ranker sees. A signature, a docstring and the
# first few statements carry nearly all the signal about what code does, and
# sending whole chunks would make ranking cost as much as generation.
DEFAULT_PREVIEW_LINES = 18


@dataclass
class RerankResult:
    """Reranked chunks plus what the reranking actually did."""

    chunks: list[RetrievedChunk]
    strategy: str
    candidates: int = 0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float | None = None
    # True when the model's output could not be used and we fell back.
    fell_back: bool = False
    original_order: list[int] = field(default_factory=list)

    @property
    def changed_order(self) -> bool:
        """Did reranking actually change which chunks reach the LLM?"""
        return [c.chunk_id for c in self.chunks] != self.original_order[: len(self.chunks)]

    def as_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "candidates": self.candidates,
            "kept": len(self.chunks),
            "changed_order": self.changed_order,
            "fell_back": self.fell_back,
            "llm_calls": self.llm_calls,
            "usd": self.usd,
        }


class Reranker(ABC):
    """Re-scores a candidate set and returns the best `top_k`."""

    name: str

    @abstractmethod
    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int
    ) -> RerankResult:
        ...


class IdentityReranker(Reranker):
    """Keeps the vector ordering and truncates. The measurement baseline.

    This is not a placeholder. Section 21 asks whether reranking helps, and
    that question is unanswerable without running the identical pipeline with
    reranking switched off. This class IS the control group.
    """

    name = "identity"

    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int
    ) -> RerankResult:
        return RerankResult(
            chunks=chunks[:top_k],
            strategy=self.name,
            candidates=len(chunks),
            original_order=[c.chunk_id for c in chunks],
        )


class LLMReranker(Reranker):
    """Listwise reranking: one call, all candidates, ranked order back."""

    name = "llm"

    def __init__(
        self,
        llm: LLMProvider,
        preview_lines: int = DEFAULT_PREVIEW_LINES,
    ) -> None:
        self.llm = llm
        self.preview_lines = preview_lines

    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int
    ) -> RerankResult:
        original_order = [c.chunk_id for c in chunks]

        # Nothing to choose between: reranking N candidates down to N or more
        # cannot change what the LLM sees, so do not pay for the call.
        if len(chunks) <= top_k:
            return RerankResult(
                chunks=chunks[:top_k],
                strategy=self.name,
                candidates=len(chunks),
                original_order=original_order,
            )

        try:
            response = self.llm.complete(
                RERANK_SYSTEM_PROMPT,
                [{"role": "user", "content": self._build_prompt(query, chunks)}],
                temperature=0.0,
                max_output_tokens=256,
            )
        except Exception:
            # Reranking is an optimisation. Never let it break a question.
            return RerankResult(
                chunks=chunks[:top_k],
                strategy=self.name,
                candidates=len(chunks),
                fell_back=True,
                original_order=original_order,
            )

        order = parse_ranking(response.text, len(chunks))
        fell_back = not order

        ranked = _apply_order(chunks, order)

        return RerankResult(
            chunks=ranked[:top_k],
            strategy=self.name,
            candidates=len(chunks),
            llm_calls=1,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            usd=response.usd,
            fell_back=fell_back,
            original_order=original_order,
        )

    def _build_prompt(self, query: str, chunks: list[RetrievedChunk]) -> str:
        blocks = []
        for index, chunk in enumerate(chunks, start=1):
            lines = chunk.content.splitlines()
            preview = "\n".join(lines[: self.preview_lines])
            if len(lines) > self.preview_lines:
                preview += f"\n... ({len(lines) - self.preview_lines} more lines)"

            descriptor = f"{chunk.kind} {chunk.symbol}" if chunk.symbol else chunk.kind
            blocks.append(
                f"[{index}] {chunk.citation}  ({descriptor}, {chunk.language})\n"
                f"```{chunk.language}\n{preview}\n```"
            )

        return (
            f"Question: {query}\n\n"
            f"Candidates:\n\n" + "\n\n".join(blocks) + "\n\n"
            f"Rank all {len(chunks)} candidates, most relevant first."
        )


def parse_ranking(text: str, candidate_count: int) -> list[int]:
    """Extract a ranking from the model's reply.

    Returns 1-based indices, deduplicated and validated. An empty list means
    nothing usable came back, and the caller should keep the vector order.

    Deliberately forgiving about formatting (the model may wrap the array in a
    code fence or add a sentence), and deliberately strict about content: an
    index outside the candidate range is dropped rather than clamped, because
    a hallucinated candidate number must never select a chunk by accident.
    """
    if not text:
        return []

    match = re.search(r"\[[^\]]*\]", text, re.DOTALL)
    numbers: list[int] = []

    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                numbers = [int(n) for n in parsed if isinstance(n, (int, float))]
        except (ValueError, TypeError):
            numbers = []

    if not numbers:
        # Fall back to reading bare integers, which covers replies like
        # "3, 1, 7" or a numbered list.
        numbers = [int(n) for n in re.findall(r"\d+", text)]

    seen: set[int] = set()
    order: list[int] = []
    for number in numbers:
        if 1 <= number <= candidate_count and number not in seen:
            seen.add(number)
            order.append(number)

    return order


def _apply_order(chunks: list[RetrievedChunk], order: list[int]) -> list[RetrievedChunk]:
    """Reorder by 1-based indices, appending anything the model left out.

    Omissions are common when a model is asked to rank ten items and returns
    six. The missing candidates keep their original relative order at the end,
    so nothing is silently lost.
    """
    if not order:
        return list(chunks)

    ranked = [chunks[i - 1] for i in order]
    listed = set(order)
    ranked.extend(chunk for i, chunk in enumerate(chunks, start=1) if i not in listed)
    return ranked


def get_reranker(
    settings: Settings | None = None,
    *,
    llm: LLMProvider | None = None,
) -> Reranker:
    """Build the reranker named in configuration."""
    settings = settings or get_settings()
    name = settings.reranker.strip().lower()

    if name in ("identity", "none", "off"):
        return IdentityReranker()

    if name == "llm":
        return LLMReranker(
            llm or get_llm_provider(settings),
            preview_lines=settings.rerank_preview_lines,
        )

    # A cross-encoder would slot in here as another Reranker implementation:
    # same interface, no caller changes. It was not added because it needs a
    # local model download, and the LLM reranker reuses infrastructure that
    # already exists.
    raise ValueError(
        f"Unknown reranker {settings.reranker!r}. Supported: 'llm', 'identity'."
    )
