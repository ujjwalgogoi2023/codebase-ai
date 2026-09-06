"""Tests for pgvector storage and the indexing pipeline (stage 4).

These require a running Postgres with pgvector (`docker compose up -d`); they
skip otherwise. Embeddings come from FakeEmbeddingProvider, so these tests
verify PLUMBING — that vectors round-trip, that filters and cascades work, that
distance ordering is wired to the right operator. They deliberately assert
nothing about semantic quality, which fake vectors cannot provide.
"""

from __future__ import annotations

import pytest
from sqlalchemy import func, select, text

from app.models import Chunk, File, Repository
from app.services.embeddings import FakeEmbeddingProvider
from app.services.indexer import (
    count_chunks,
    delete_repository,
    index_repository,
    list_repositories,
)

pytestmark = pytest.mark.usefixtures("_schema")


def provider() -> FakeEmbeddingProvider:
    # Must match the column dimension declared on Chunk.embedding.
    return FakeEmbeddingProvider(dimension=1536, model="fake-embedding")


def index(db, sample_repo, **kwargs):
    defaults = dict(
        name="demo",
        root=sample_repo,
        provider=provider(),
    )
    defaults.update(kwargs)
    return index_repository(db, **defaults)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_index_repository_stores_files_and_chunks(db, sample_repo):
    result = index(db, sample_repo)

    assert result.files == 4, "expected 4 indexable files (node_modules excluded)"
    assert result.chunks > 0
    assert db.scalar(select(func.count()).select_from(File)) == 4
    assert db.scalar(select(func.count()).select_from(Chunk)) == result.chunks


def test_repository_status_and_metadata(db, sample_repo):
    result = index(db, sample_repo, url="https://github.com/demo/demo")

    repository = db.get(Repository, result.repository_id)
    assert repository.index_status == "ready"
    assert repository.error_message is None
    assert repository.indexed_at is not None
    assert repository.url == "https://github.com/demo/demo"
    assert repository.embedding_model == "fake-embedding"
    assert repository.chunk_strategy == "code_aware"
    assert repository.file_count == 4
    assert repository.chunk_count == result.chunks


def test_junk_directories_never_reach_the_database(db, sample_repo):
    index(db, sample_repo)
    paths = db.scalars(select(File.path)).all()

    assert not any("node_modules" in p for p in paths)


def test_chunk_metadata_round_trips(db, sample_repo):
    index(db, sample_repo)

    chunk = db.scalars(
        select(Chunk).where(Chunk.symbol == "authenticate_user")
    ).one()

    assert chunk.file_path == "backend/auth.py"
    assert chunk.language == "python"
    assert chunk.kind == "function"
    assert chunk.start_line >= 1
    assert chunk.end_line >= chunk.start_line
    assert "bcrypt.checkpw" in chunk.content
    assert chunk.citation == f"backend/auth.py:{chunk.start_line}-{chunk.end_line}"


def test_embedding_round_trips_with_correct_dimension(db, sample_repo):
    index(db, sample_repo)
    chunk = db.scalars(select(Chunk).limit(1)).one()

    assert len(chunk.embedding) == 1536


def test_denormalized_columns_match_their_source(db, sample_repo):
    """repository_id/language/file_path are copies; they must not drift."""
    index(db, sample_repo)

    rows = db.execute(
        select(Chunk.repository_id, Chunk.language, Chunk.file_path,
               File.repository_id, File.language, File.path)
        .join(File, Chunk.file_id == File.id)
    ).all()

    assert rows
    for c_repo, c_lang, c_path, f_repo, f_lang, f_path in rows:
        assert (c_repo, c_lang, c_path) == (f_repo, f_lang, f_path)


# ---------------------------------------------------------------------------
# Vector search: the reason the whole layer exists
# ---------------------------------------------------------------------------

def test_cosine_search_ranks_the_matching_chunk_first(db, sample_repo):
    """A chunk searched by its own embedding must rank first, at distance ~0."""
    index(db, sample_repo)

    target = db.scalars(select(Chunk).where(Chunk.symbol == "create_order")).one()
    query_vector = provider().embed_one(
        f"File: {target.file_path}\nLanguage: {target.language}\n"
        f"Symbol: {target.symbol}\n\n{target.content}"
    )

    rows = db.execute(
        select(Chunk.id, Chunk.symbol, Chunk.embedding.cosine_distance(query_vector))
        .order_by(Chunk.embedding.cosine_distance(query_vector))
        .limit(3)
    ).all()

    assert rows[0][0] == target.id
    assert rows[0][2] == pytest.approx(0.0, abs=1e-6)
    # Ordering must be ascending distance (nearest first).
    distances = [r[2] for r in rows]
    assert distances == sorted(distances)


def test_search_can_be_filtered_by_repository(db, sample_repo):
    """Metadata filtering: never leak results from another codebase."""
    first = index(db, sample_repo, name="repo-one")
    second = index(db, sample_repo, name="repo-two")

    query_vector = provider().embed_one("anything")
    rows = db.scalars(
        select(Chunk.repository_id)
        .where(Chunk.repository_id == second.repository_id)
        .order_by(Chunk.embedding.cosine_distance(query_vector))
        .limit(50)
    ).all()

    assert rows, "filtered search returned nothing"
    assert set(rows) == {second.repository_id}
    assert first.repository_id != second.repository_id


def test_search_can_be_filtered_by_language(db, sample_repo):
    index(db, sample_repo)
    query_vector = provider().embed_one("anything")

    rows = db.scalars(
        select(Chunk.language)
        .where(Chunk.language == "python")
        .order_by(Chunk.embedding.cosine_distance(query_vector))
        .limit(20)
    ).all()

    assert rows
    assert set(rows) == {"python"}


def test_hnsw_index_exists_on_the_embedding_column(db):
    """The index must use vector_cosine_ops to match the `<=>` operator.

    Asserting existence rather than usage on purpose: Postgres correctly
    prefers a sequential scan on a table this small, so an EXPLAIN assertion
    here would be testing the planner's heuristics, not our schema.
    """
    definition = db.execute(
        text("SELECT indexdef FROM pg_indexes WHERE indexname = 'ix_chunks_embedding_hnsw'")
    ).scalar_one()

    assert "USING hnsw" in definition
    assert "vector_cosine_ops" in definition


# ---------------------------------------------------------------------------
# Lifecycle: re-indexing, cascades, failures
# ---------------------------------------------------------------------------

def test_reindexing_same_config_replaces_the_previous_run(db, sample_repo):
    first = index(db, sample_repo)
    second = index(db, sample_repo)

    assert first.repository_id != second.repository_id
    repositories = list_repositories(db)
    assert len(repositories) == 1, "old run with identical config should be gone"
    assert repositories[0].id == second.repository_id


def test_different_chunk_strategies_coexist(db, sample_repo):
    """This is what makes the section 21 experiment possible."""
    aware = index(db, sample_repo, chunk_strategy="code_aware")
    naive = index(db, sample_repo, chunk_strategy="naive")

    repositories = list_repositories(db)
    assert len(repositories) == 2
    assert {r.chunk_strategy for r in repositories} == {"code_aware", "naive"}
    assert count_chunks(db, aware.repository_id) > 0
    assert count_chunks(db, naive.repository_id) > 0


def test_delete_repository_cascades(db, sample_repo):
    result = index(db, sample_repo)
    assert count_chunks(db, result.repository_id) > 0

    assert delete_repository(db, result.repository_id) is True

    assert db.scalar(select(func.count()).select_from(Repository)) == 0
    assert db.scalar(select(func.count()).select_from(File)) == 0
    assert db.scalar(select(func.count()).select_from(Chunk)) == 0, "orphaned vectors"


def test_delete_missing_repository_returns_false(db):
    assert delete_repository(db, 999999) is False


def test_failed_index_is_recorded_not_left_hanging(db, tmp_path):
    """A crash must leave a diagnosable row, not a permanent 'indexing'."""
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(ValueError, match="No indexable source files"):
        index_repository(db, name="broken", root=empty, provider=provider())

    repository = db.scalars(select(Repository)).one()
    assert repository.index_status == "failed"
    assert "No indexable source files" in repository.error_message


def test_duplicate_file_paths_are_rejected(db, sample_repo):
    from sqlalchemy.exc import IntegrityError

    result = index(db, sample_repo)
    repository_id = result.repository_id

    db.add(File(repository_id=repository_id, path="backend/auth.py", language="python"))
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()


def test_naive_strategy_produces_unnamed_chunks(db, sample_repo):
    result = index(db, sample_repo, chunk_strategy="naive")

    symbols = db.scalars(
        select(Chunk.symbol).where(Chunk.repository_id == result.repository_id)
    ).all()
    assert all(s is None for s in symbols)


def test_progress_callback_reports_each_stage(db, sample_repo):
    stages: list[str] = []
    index(db, sample_repo, progress=lambda stage, done, total: stages.append(stage))

    for expected in ("parsing", "chunking", "embedding", "storing", "done"):
        assert expected in stages, f"missing progress stage: {expected}"


def test_reindexing_the_same_row_replaces_its_index(db, sample_repo):
    """POST /repositories/{id}/index must not double every file and chunk."""
    from app.models import Repository
    from app.services.indexer import run_indexing

    first = index(db, sample_repo)
    repository = db.get(Repository, first.repository_id)

    second = run_indexing(db, repository, sample_repo, provider=provider())

    assert second.files == first.files
    assert second.chunks == first.chunks
    assert db.scalar(select(func.count()).select_from(File)) == first.files
    assert count_chunks(db, first.repository_id) == first.chunks
