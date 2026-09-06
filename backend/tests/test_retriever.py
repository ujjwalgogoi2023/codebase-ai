"""Tests for semantic retrieval (RAG pipeline stage 5).

Requires Postgres (`docker compose up -d`); skips otherwise.

An important limit on what these can prove: with FakeEmbeddingProvider the
vectors are hashes, so "relevance" is meaningless. These tests verify that the
MECHANISM is correct — right SQL, right operator, right ordering, right
filters, right guardrails. Whether the results are actually GOOD is a question
only the stage 11 evaluation, running on real embeddings, can answer.

That distinction is the whole reason the evaluation harness exists.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select

from app.config import Settings
from app.models import Chunk, Repository
from app.services.embeddings import FakeEmbeddingProvider
from app.services.indexer import index_repository
from app.services.retriever import (
    RetrievalError,
    Retriever,
    SearchFilters,
    search_repository,
)

pytestmark = pytest.mark.usefixtures("_schema")


def provider(model: str = "fake-embedding") -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider(dimension=1536, model=model)


@pytest.fixture
def indexed(db, sample_repo):
    """A ready-to-search repository."""
    return index_repository(db, name="demo", root=sample_repo, provider=provider())


@pytest.fixture
def retriever(db):
    return Retriever(db, provider=provider(), settings=Settings(top_k=10))


# ---------------------------------------------------------------------------
# Core search behaviour
# ---------------------------------------------------------------------------

def test_search_returns_scored_results(indexed, retriever):
    result = retriever.search("how are passwords checked?", indexed.repository_id)

    assert len(result) > 0
    assert result.repository_id == indexed.repository_id
    assert result.embedding_model == "fake-embedding"
    for chunk in result:
        assert chunk.chunk_id > 0
        assert chunk.file_path
        assert chunk.start_line >= 1
        assert chunk.end_line >= chunk.start_line
        assert -1.0 <= chunk.score <= 1.0


def test_results_are_ordered_by_descending_relevance(indexed, retriever):
    result = retriever.search("anything at all", indexed.repository_id)

    scores = [c.score for c in result]
    assert scores == sorted(scores, reverse=True), "results not ranked best-first"


def test_score_is_the_complement_of_cosine_distance(indexed, retriever):
    result = retriever.search("x", indexed.repository_id, top_k=1)
    chunk = result.chunks[0]

    assert chunk.score == pytest.approx(1.0 - chunk.distance)


def test_searching_by_a_chunks_own_text_ranks_it_first(db, indexed):
    """The one relevance claim fake embeddings CAN support: identity."""
    target = db.scalars(
        select(Chunk).where(Chunk.symbol == "authenticate_user")
    ).one()
    exact_text = (
        f"File: {target.file_path}\nLanguage: {target.language}\n"
        f"Symbol: {target.symbol}\n\n{target.content}"
    )

    result = Retriever(db, provider=provider()).search(
        exact_text, indexed.repository_id, top_k=3
    )

    assert result.chunks[0].chunk_id == target.id
    assert result.chunks[0].score == pytest.approx(1.0, abs=1e-6)


def test_top_k_limits_the_result_count(indexed, retriever):
    assert len(retriever.search("q", indexed.repository_id, top_k=2)) == 2
    assert len(retriever.search("q", indexed.repository_id, top_k=5)) == 5


def test_top_k_defaults_to_settings(db, indexed):
    retriever = Retriever(db, provider=provider(), settings=Settings(top_k=3))
    result = retriever.search("q", indexed.repository_id)

    assert result.top_k == 3
    assert len(result) == 3


def test_top_k_larger_than_the_corpus_is_not_an_error(indexed, retriever):
    result = retriever.search("q", indexed.repository_id, top_k=10_000)
    assert 0 < len(result) < 10_000


def test_citations_are_line_accurate(db, indexed, retriever):
    result = retriever.search("bcrypt", indexed.repository_id, top_k=5)

    for hit in result:
        stored = db.get(Chunk, hit.chunk_id)
        assert hit.citation == f"{stored.file_path}:{stored.start_line}-{stored.end_line}"
        assert hit.content == stored.content


# ---------------------------------------------------------------------------
# Metadata filtering
# ---------------------------------------------------------------------------

def test_filter_by_language(indexed, retriever):
    result = retriever.search(
        "q", indexed.repository_id, top_k=50, filters=SearchFilters(languages=["python"])
    )

    assert len(result) > 0
    assert {c.language for c in result} == {"python"}


def test_filter_by_multiple_languages(indexed, retriever):
    result = retriever.search(
        "q",
        indexed.repository_id,
        top_k=50,
        filters=SearchFilters(languages=["javascript", "markdown"]),
    )

    assert {c.language for c in result} <= {"javascript", "markdown"}
    assert len(result) > 0


def test_filter_by_path_prefix(indexed, retriever):
    result = retriever.search(
        "q", indexed.repository_id, top_k=50, filters=SearchFilters(path_prefix="backend/")
    )

    assert len(result) > 0
    assert all(c.file_path.startswith("backend/") for c in result)


def test_filter_by_kind(indexed, retriever):
    result = retriever.search(
        "q", indexed.repository_id, top_k=50, filters=SearchFilters(kinds=["function"])
    )

    assert len(result) > 0
    assert {c.kind for c in result} == {"function"}


def test_min_score_drops_weak_matches(indexed, retriever):
    unfiltered = retriever.search("q", indexed.repository_id, top_k=50)
    threshold = max(c.score for c in unfiltered)

    filtered = retriever.search(
        "q", indexed.repository_id, top_k=50, filters=SearchFilters(min_score=threshold)
    )

    assert 0 < len(filtered) < len(unfiltered)
    assert all(c.score >= threshold for c in filtered)


def test_impossible_filter_returns_nothing_rather_than_failing(indexed, retriever):
    """An empty result is a valid answer, and stage 12 relies on it."""
    result = retriever.search(
        "q", indexed.repository_id, top_k=10, filters=SearchFilters(languages=["cobol"])
    )

    assert len(result) == 0
    assert result.citations() == []


def test_path_prefix_wildcards_are_matched_literally(db, indexed, retriever):
    """A '%' in a path must not become a LIKE wildcard."""
    result = retriever.search(
        "q", indexed.repository_id, top_k=50, filters=SearchFilters(path_prefix="%")
    )

    assert len(result) == 0, "'%' was treated as a wildcard instead of a literal"


# ---------------------------------------------------------------------------
# Isolation between repositories
# ---------------------------------------------------------------------------

def test_search_never_leaks_across_repositories(db, sample_repo):
    first = index_repository(db, name="repo-one", root=sample_repo, provider=provider())
    second = index_repository(db, name="repo-two", root=sample_repo, provider=provider())

    retriever = Retriever(db, provider=provider())
    result = retriever.search("q", second.repository_id, top_k=100)

    assert len(result) > 0
    ids = {c.chunk_id for c in result}
    owners = set(
        db.scalars(select(Chunk.repository_id).where(Chunk.id.in_(ids))).all()
    )
    assert owners == {second.repository_id}
    assert first.repository_id != second.repository_id


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------

def test_embedding_model_mismatch_is_refused(db, indexed):
    """The most destructive silent failure in RAG, caught loudly.

    Vectors from a different model have the same dimension, so every query
    still 'works' — it just returns noise with confident citations attached.
    """
    wrong = Retriever(db, provider=provider(model="some-other-model"))

    with pytest.raises(RetrievalError, match="Embedding model mismatch"):
        wrong.search("where is auth?", indexed.repository_id)


def test_unknown_repository_is_refused(retriever):
    with pytest.raises(RetrievalError, match="does not exist"):
        retriever.search("q", 999999)


def test_unfinished_repository_is_refused(db, sample_repo, retriever):
    repository = Repository(
        name="half-done",
        embedding_model="fake-embedding",
        index_status="indexing",
    )
    db.add(repository)
    db.commit()

    with pytest.raises(RetrievalError, match="not searchable"):
        retriever.search("q", repository.id)


def test_failed_repository_is_refused(db, tmp_path, retriever):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError):
        index_repository(db, name="broken", root=empty, provider=provider())

    failed = db.scalars(select(Repository).where(Repository.name == "broken")).one()
    with pytest.raises(RetrievalError, match="not searchable"):
        retriever.search("q", failed.id)


@pytest.mark.parametrize("bad_query", ["", "   ", "\n\t "])
def test_empty_query_is_refused(indexed, retriever, bad_query):
    with pytest.raises(RetrievalError, match="Query is empty"):
        retriever.search(bad_query, indexed.repository_id)


def test_invalid_top_k_is_refused(indexed, retriever):
    with pytest.raises(RetrievalError, match="top_k must be"):
        retriever.search("q", indexed.repository_id, top_k=0)


# ---------------------------------------------------------------------------
# Result shape
# ---------------------------------------------------------------------------

def test_search_result_serializes_for_the_api(indexed, retriever):
    payload = retriever.search("q", indexed.repository_id, top_k=2).as_dict()

    assert payload["count"] == 2
    assert payload["top_k"] == 2
    assert payload["embedding_model"] == "fake-embedding"
    first = payload["results"][0]
    assert set(first) >= {"chunk_id", "citation", "file_path", "score", "content"}


def test_module_level_helper_matches_the_class(db, indexed):
    direct = Retriever(db, provider=provider()).search(
        "passwords", indexed.repository_id, top_k=3
    )
    helper = search_repository(
        db, "passwords", indexed.repository_id, top_k=3, provider=provider()
    )

    assert [c.chunk_id for c in direct] == [c.chunk_id for c in helper]


def test_ef_search_is_clamped_to_the_pgvector_range(db, indexed):
    """pgvector rejects ef_search outside 1..1000; a big top_k must not crash."""
    retriever = Retriever(db, provider=provider(), settings=Settings(hnsw_ef_search=99999))

    result = retriever.search("q", indexed.repository_id, top_k=5000)

    assert len(result) > 0


@pytest.mark.parametrize("bad_top_k", [0, -1, -100])
def test_non_positive_top_k_is_refused_not_defaulted(indexed, retriever, bad_top_k):
    """`top_k or default` would turn 0 into the default instead of an error."""
    with pytest.raises(RetrievalError, match="top_k must be"):
        retriever.search("q", indexed.repository_id, top_k=bad_top_k)
