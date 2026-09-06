"""RAG generation: turn retrieved code into a grounded answer.

Stage 6, the last step of the query path:

    question -> retrieve -> rerank -> [BUILD PROMPT -> LLM] -> answer + citations

WHAT THE LLM ACTUALLY RECEIVES
------------------------------
Nothing magic. A system prompt with rules, then a user message containing the
retrieved code as numbered blocks, then the question:

    [1] backend/auth.py:42-61  (function authenticate_user, python)
    ```python
    def authenticate_user(username, password):
        ...
    ```

    [2] backend/middleware.py:18-39  (function require_auth, python)
    ...

    Question: where do we validate JWT tokens?

The numbering is the mechanism behind citations. The model is told to write
[1] and [2], and because we built that list we can map every marker back to an
exact file and line range. The model never invents a path; it picks an index.

GROUNDING (spec section 12)
---------------------------
The default behaviour of an LLM asked "where is the Stripe integration?" about
a repo with no Stripe is to invent `payment.py`, confidently. Three defences,
in order of reliability:

1. If retrieval returned nothing, do not call the model at all. A guaranteed
   correct refusal, for zero cost and zero latency. This is the only one that
   cannot fail.
2. The system prompt forbids outside knowledge and requires an explicit
   "not in this codebase" when the context does not answer the question.
3. Stage 7 validates the citations the model emitted, so a fabricated [7] when
   only three sources exist is caught rather than displayed.

Prompting alone is defence 2 of 3, and the weakest of them. It is a strong
nudge, not a guarantee, which is exactly why the other two exist.

HISTORY IS NOT A SUBSTITUTE FOR RETRIEVAL (spec section 14)
-----------------------------------------------------------
Every question runs its own search. Prior turns are replayed so the model can
resolve "it" and "that function", but the CONTEXT is always freshly retrieved.
Answering a follow-up from conversation history alone is how a system starts
confidently describing code it last saw three questions ago.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.services.embeddings import EmbeddingProvider, _get_encoder
from app.services.llm import LLMProvider, LLMResponse, get_llm_provider
from app.services.retriever import (
    RetrievedChunk,
    Retriever,
    SearchFilters,
    SearchResult,
)

SYSTEM_PROMPT = """\
You are CodebaseAI, an assistant that answers questions about one specific \
codebase.

Follow these rules exactly:

1. Answer ONLY from the code context provided in the user message. It is the \
only source of truth. Do not use general knowledge about libraries, \
frameworks, or common project layouts to fill gaps.

2. If the context does not contain enough information to answer, say so \
plainly, for example: "The retrieved code doesn't show a Stripe \
integration." Do not guess, and do not describe what the code "probably" or \
"likely" does. An honest "I don't know" is a correct answer; a plausible \
invention is not.

3. Cite every claim with the bracketed number of the source it came from, like \
[1] or [2][3]. Place citations immediately after the claim they support. Use \
ONLY numbers that appear in the context below.

4. Never invent file paths, function names, or line numbers. If you did not see \
it in the context, it does not exist as far as you are concerned.

5. Be concise and concrete. Name the actual functions and files involved. \
Prefer explaining the real control flow over restating the code line by line.

6. If the context is related but does not directly answer the question, say \
what the code DOES show and what is missing."""

NO_CONTEXT_ANSWER = (
    "I couldn't find anything relevant to that question in this repository. "
    "The search returned no matching code, so there is nothing for me to base "
    "an answer on. Try rephrasing the question, or check whether the relevant "
    "files were included when the repository was indexed."
)


@dataclass
class GeneratedAnswer:
    """An answer plus everything needed to verify it."""

    question: str
    answer: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    repository_id: int = 0
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float | None = None
    truncated: bool = False
    # True when we refused to answer because retrieval came back empty.
    no_context: bool = False
    # Sources dropped to fit the context budget.
    sources_omitted: int = 0

    def as_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "repository_id": self.repository_id,
            "model": self.model,
            "no_context": self.no_context,
            "truncated": self.truncated,
            "sources_omitted": self.sources_omitted,
            "usage": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "usd": self.usd,
            },
            "sources": [
                {"index": i, **chunk.as_dict()}
                for i, chunk in enumerate(self.sources, start=1)
            ],
        }


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as numbered, fenced blocks.

    The header carries path, line range, symbol and language. Those are not
    decoration: "which file is this?" is frequently the actual question, and
    the model can only answer it if the path is in the text it reads.
    """
    blocks: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        descriptor = f"{chunk.kind} {chunk.symbol}" if chunk.symbol else chunk.kind
        blocks.append(
            f"[{index}] {chunk.citation}  ({descriptor}, {chunk.language})\n"
            f"```{chunk.language}\n{chunk.content}\n```"
        )
    return "\n\n".join(blocks)


def build_user_message(question: str, chunks: list[RetrievedChunk]) -> str:
    return (
        "Code context retrieved from the repository:\n\n"
        f"{format_context(chunks)}\n\n"
        "---\n\n"
        f"Question: {question}"
    )


class Generator:
    """Retrieval + prompt assembly + LLM call."""

    def __init__(
        self,
        session: Session,
        *,
        llm: LLMProvider | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        settings: Settings | None = None,
        retriever: Retriever | None = None,
    ) -> None:
        self.session = session
        self.settings = settings or get_settings()
        self.llm = llm or get_llm_provider(self.settings)
        self.retriever = retriever or Retriever(
            session, provider=embedding_provider, settings=self.settings
        )

    def answer(
        self,
        question: str,
        repository_id: int,
        *,
        history: list[dict[str, str]] | None = None,
        top_k: int | None = None,
        filters: SearchFilters | None = None,
    ) -> GeneratedAnswer:
        history = history or []

        search = self.retriever.search(
            self._build_search_query(question, history),
            repository_id,
            top_k=top_k,
            filters=filters,
        )

        if not search.chunks:
            # Defence 1: the only one that cannot be talked out of.
            return GeneratedAnswer(
                question=question,
                answer=NO_CONTEXT_ANSWER,
                repository_id=repository_id,
                model=self.llm.model,
                no_context=True,
            )

        kept, omitted = self._fit_to_budget(search.chunks, question)

        messages = [
            *self._recent_history(history),
            {"role": "user", "content": build_user_message(question, kept)},
        ]

        response: LLMResponse = self.llm.complete(
            SYSTEM_PROMPT,
            messages,
            temperature=self.settings.llm_temperature,
            max_output_tokens=self.settings.llm_max_output_tokens,
        )

        return GeneratedAnswer(
            question=question,
            answer=response.text,
            sources=kept,
            repository_id=repository_id,
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            usd=response.usd,
            truncated=response.truncated,
            sources_omitted=omitted,
        )

    # -- internals ---------------------------------------------------------

    def _build_search_query(self, question: str, history: list[dict[str, str]]) -> str:
        """Expand a follow-up into something searchable.

        "How does it validate the token?" retrieves badly on its own: "it" is
        the entire subject, and it is not in the sentence. Prepending the most
        recent user turn restores enough context to search with.

        A production system would ask the LLM to rewrite the follow-up into a
        standalone question, which handles pronouns far better. That costs an
        extra call per turn, so this cheap heuristic is the deliberate starting
        point — and it is measurable later against the evaluation set.
        """
        if not history:
            return question

        previous = [m["content"] for m in history if m.get("role") == "user"]
        if not previous:
            return question
        return f"{previous[-1]}\n{question}"

    def _recent_history(self, history: list[dict[str, str]]) -> list[dict[str, str]]:
        limit = max(0, self.settings.max_history_turns)
        recent = history[-limit:] if limit else []
        return [
            {"role": m["role"], "content": m["content"]}
            for m in recent
            if m.get("role") in ("user", "assistant") and m.get("content")
        ]

    def _fit_to_budget(
        self, chunks: list[RetrievedChunk], question: str
    ) -> tuple[list[RetrievedChunk], int]:
        """Drop the weakest sources until the context fits.

        Chunks arrive best-first, so truncation removes the least relevant
        ones. Dropping from the end is what makes this safe: cutting the middle
        of a chunk would hand the model a syntactically broken fragment and
        invite it to guess at the rest.
        """
        budget = self.settings.max_context_tokens
        encoder = _get_encoder(self.settings.llm_model)

        def size(text: str) -> int:
            if encoder is None:
                return len(text) // 4
            return len(encoder.encode(text))

        overhead = size(SYSTEM_PROMPT) + size(question) + 64

        kept: list[RetrievedChunk] = []
        used = overhead
        for chunk in chunks:
            cost = size(format_context([chunk]))
            if kept and used + cost > budget:
                break
            used += cost
            kept.append(chunk)

        # Always keep at least the top hit, even if it alone blows the budget:
        # an answer from one oversized source beats refusing to answer at all.
        if not kept and chunks:
            kept = [chunks[0]]

        return kept, len(chunks) - len(kept)


def answer_question(
    session: Session,
    question: str,
    repository_id: int,
    *,
    history: list[dict[str, str]] | None = None,
    top_k: int | None = None,
    filters: SearchFilters | None = None,
    llm: LLMProvider | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    settings: Settings | None = None,
) -> GeneratedAnswer:
    """Convenience wrapper for CLI, tests and API handlers."""
    generator = Generator(
        session,
        llm=llm,
        embedding_provider=embedding_provider,
        settings=settings,
    )
    return generator.answer(
        question, repository_id, history=history, top_k=top_k, filters=filters
    )


__all__ = [
    "SYSTEM_PROMPT",
    "NO_CONTEXT_ANSWER",
    "GeneratedAnswer",
    "Generator",
    "answer_question",
    "build_user_message",
    "format_context",
    "SearchResult",
]
