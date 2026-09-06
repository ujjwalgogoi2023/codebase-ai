"""Tests for reranking (RAG pipeline stage 10).

Most of these need no database and no model: given a candidate list and a
model reply, reordering is pure logic.

What they prove is that reranking is SAFE — that a malformed reply, an omitted
candidate or a hallucinated index can never corrupt the result or break the
question. What they cannot prove is that reranking IMPROVES anything; that is
the stage 11 experiment's job, run against real embeddings and a real model.
"""

from __future__ import annotations

import pytest

from app.config import Settings
from app.services.llm import FakeLLMProvider
from app.services.reranker import (
    RERANK_SYSTEM_PROMPT,
    IdentityReranker,
    LLMReranker,
    get_reranker,
    parse_ranking,
)
from app.services.retriever import RetrievedChunk


def chunk(index: int, **overrides) -> RetrievedChunk:
    defaults = dict(
        chunk_id=index,
        file_path=f"backend/file{index}.py",
        language="python",
        symbol=f"func_{index}",
        kind="function",
        start_line=1,
        end_line=5,
        content=f"def func_{index}():\n    return {index}",
        # Descending relevance, so the vector order is 1, 2, 3, ...
        distance=0.1 * index,
    )
    defaults.update(overrides)
    return RetrievedChunk(**defaults)


CANDIDATES = [chunk(i) for i in range(1, 7)]


def ranker(response: str) -> LLMReranker:
    return LLMReranker(FakeLLMProvider(response=response))


# ---------------------------------------------------------------------------
# Identity: the control group
# ---------------------------------------------------------------------------

def test_identity_preserves_vector_order_and_truncates():
    result = IdentityReranker().rerank("q", CANDIDATES, top_k=3)

    assert [c.chunk_id for c in result.chunks] == [1, 2, 3]
    assert result.strategy == "identity"
    assert result.candidates == 6
    assert result.llm_calls == 0
    assert result.changed_order is False


def test_identity_with_fewer_candidates_than_top_k():
    result = IdentityReranker().rerank("q", CANDIDATES[:2], top_k=5)
    assert len(result.chunks) == 2


# ---------------------------------------------------------------------------
# Parsing the model's reply
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "reply,expected",
    [
        ("[3, 1, 2]", [3, 1, 2]),
        ("[3,1,2]", [3, 1, 2]),
        ("```json\n[3, 1, 2]\n```", [3, 1, 2]),
        ("Here is the ranking: [3, 1, 2]", [3, 1, 2]),
        ("3, 1, 2", [3, 1, 2]),          # bare integers
        ("1. chunk 3\n2. chunk 1", [1, 3, 2]),  # numbered list, read in order
    ],
)
def test_parse_ranking_is_forgiving_about_format(reply, expected):
    assert parse_ranking(reply, 6) == expected


def test_parse_ranking_drops_out_of_range_indices():
    """A hallucinated index must never select a chunk by accident."""
    assert parse_ranking("[3, 99, 1, 0, -2]", 6) == [3, 1]


def test_parse_ranking_deduplicates():
    assert parse_ranking("[2, 2, 1, 2]", 6) == [2, 1]


@pytest.mark.parametrize("reply", ["", "no numbers here", "I cannot rank these."])
def test_parse_ranking_returns_empty_on_unusable_output(reply):
    assert parse_ranking(reply, 6) == []


# ---------------------------------------------------------------------------
# LLM reranking
# ---------------------------------------------------------------------------

def test_reranking_reorders_the_candidates():
    result = ranker("[4, 2, 6]").rerank("q", CANDIDATES, top_k=3)

    assert [c.chunk_id for c in result.chunks] == [4, 2, 6]
    assert result.changed_order is True
    assert result.llm_calls == 1
    assert result.fell_back is False


def test_omitted_candidates_keep_their_original_relative_order():
    """Models routinely rank six of ten. Nothing may be silently lost.

    """
    result = ranker("[5, 3]").rerank("q", CANDIDATES, top_k=6)

    assert [c.chunk_id for c in result.chunks] == [5, 3, 1, 2, 4, 6]


def test_hallucinated_indices_are_ignored_not_clamped():
    result = ranker("[99, 4, 0]").rerank("q", CANDIDATES, top_k=3)

    # 99 and 0 dropped; 4 promoted; the rest follow in vector order.
    assert [c.chunk_id for c in result.chunks] == [4, 1, 2]


def test_unusable_output_falls_back_to_vector_order():
    result = ranker("I'm not able to rank these.").rerank("q", CANDIDATES, top_k=3)

    assert [c.chunk_id for c in result.chunks] == [1, 2, 3]
    assert result.fell_back is True
    assert result.changed_order is False


def test_a_failing_model_never_breaks_the_question():
    """Reranking is an optimisation. A slightly worse answer beats an error."""

    class ExplodingLLM(FakeLLMProvider):
        def complete(self, *args, **kwargs):
            raise RuntimeError("upstream is down")

    result = LLMReranker(ExplodingLLM()).rerank("q", CANDIDATES, top_k=3)

    assert [c.chunk_id for c in result.chunks] == [1, 2, 3]
    assert result.fell_back is True
    assert result.llm_calls == 0


def test_reranks_even_when_every_candidate_survives():
    """Order still matters when the set does not change.

    Citation numbering and which chunk the model reads first both depend on
    ordering, and the evaluation harness needs the full candidate list ranked
    in order to score where the correct answer landed.
    """
    llm = FakeLLMProvider(response="[3,2,1]")
    result = LLMReranker(llm).rerank("q", CANDIDATES[:3], top_k=3)

    assert len(llm.calls) == 1
    assert [c.chunk_id for c in result.chunks] == [3, 2, 1]


def test_no_call_for_a_single_candidate():
    """One candidate genuinely cannot be reordered."""
    llm = FakeLLMProvider(response="[1]")
    result = LLMReranker(llm).rerank("q", CANDIDATES[:1], top_k=10)

    assert llm.calls == [], "paid for a call that could not change anything"
    assert [c.chunk_id for c in result.chunks] == [1]


def test_empty_candidate_list():
    result = ranker("[1]").rerank("q", [], top_k=3)
    assert result.chunks == []


# ---------------------------------------------------------------------------
# The prompt sent to the ranker
# ---------------------------------------------------------------------------

def test_rerank_prompt_numbers_every_candidate():
    llm = FakeLLMProvider(response="[1]")
    LLMReranker(llm).rerank("where is auth?", CANDIDATES, top_k=3)

    prompt = llm.calls[0]["messages"][0]["content"]
    assert "where is auth?" in prompt
    for i in range(1, 7):
        assert f"[{i}] backend/file{i}.py" in prompt
    assert llm.calls[0]["system"] == RERANK_SYSTEM_PROMPT


def test_rerank_prompt_tells_the_model_to_judge_answers_not_vocabulary():
    assert "ANSWERS" in RERANK_SYSTEM_PROMPT
    assert "vocabulary" in RERANK_SYSTEM_PROMPT


def test_candidates_are_previewed_not_sent_whole():
    """Ranking must not cost as much as answering."""
    big = chunk(1, content="\n".join(f"line {i}" for i in range(200)))
    llm = FakeLLMProvider(response="[1]")

    LLMReranker(llm, preview_lines=5).rerank("q", [big] + CANDIDATES, top_k=2)

    prompt = llm.calls[0]["messages"][0]["content"]
    assert "line 4" in prompt
    assert "line 150" not in prompt
    assert "more lines)" in prompt, "truncation should be visible to the model"


def test_reranking_is_deterministic():
    llm = FakeLLMProvider(response="[2,1]")
    LLMReranker(llm).rerank("q", CANDIDATES, top_k=2)
    assert llm.calls[0]["temperature"] == 0.0


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def test_factory_builds_each_strategy():
    llm = FakeLLMProvider()
    assert isinstance(get_reranker(Settings(reranker="identity"), llm=llm), IdentityReranker)
    assert isinstance(get_reranker(Settings(reranker="llm"), llm=llm), LLMReranker)
    # "none" and "off" are natural things to write in a .env when turning it off.
    assert isinstance(get_reranker(Settings(reranker="none"), llm=llm), IdentityReranker)


def test_factory_rejects_an_unknown_strategy():
    with pytest.raises(ValueError, match="Unknown reranker"):
        get_reranker(Settings(reranker="magic"), llm=FakeLLMProvider())


# ---------------------------------------------------------------------------
# Integration with generation
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("_schema")
def test_generator_sends_only_the_reranked_top_k(db, sample_repo):
    from app.services.embeddings import FakeEmbeddingProvider
    from app.services.generator import SYSTEM_PROMPT, Generator
    from app.services.indexer import index_repository

    embedder = FakeEmbeddingProvider(dimension=1536, model="fake-embedding")
    indexed = index_repository(db, name="demo", root=sample_repo, provider=embedder)

    llm = FakeLLMProvider(response="[2, 1]")
    generator = Generator(
        db,
        llm=llm,
        embedding_provider=embedder,
        settings=Settings(
            llm_provider="fake", embedding_provider="fake", reranker="llm"
        ),
    )

    result = generator.answer("q", indexed.repository_id, top_k=5, rerank_top_k=2)

    assert len(result.sources) == 2, "the LLM saw more than rerank_top_k sources"
    assert result.rerank.candidates == 5
    assert result.rerank.strategy == "llm"

    generation = [c for c in llm.calls if c["system"] == SYSTEM_PROMPT][0]
    context = generation["messages"][-1]["content"]
    assert "[1]" in context and "[2]" in context
    assert "[3]" not in context, "a dropped candidate leaked into the prompt"


@pytest.mark.usefixtures("_schema")
def test_identity_reranker_makes_no_extra_call(db, sample_repo):
    from app.services.embeddings import FakeEmbeddingProvider
    from app.services.generator import SYSTEM_PROMPT, Generator
    from app.services.indexer import index_repository

    embedder = FakeEmbeddingProvider(dimension=1536, model="fake-embedding")
    indexed = index_repository(db, name="demo", root=sample_repo, provider=embedder)

    llm = FakeLLMProvider()
    generator = Generator(
        db,
        llm=llm,
        embedding_provider=embedder,
        settings=Settings(
            llm_provider="fake", embedding_provider="fake", reranker="identity"
        ),
    )
    generator.answer("q", indexed.repository_id, top_k=5, rerank_top_k=2)

    assert all(c["system"] == SYSTEM_PROMPT for c in llm.calls)
    assert len(llm.calls) == 1
