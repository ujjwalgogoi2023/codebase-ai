"""Indexing: run the full ingestion pipeline and persist the result.

Stage 4 of the RAG pipeline, and the point where stages 1-3 become one command:

    parse -> chunk -> embed -> [STORE IN pgvector]

WHY POSTGRES + pgvector RATHER THAN A DEDICATED VECTOR DATABASE
---------------------------------------------------------------
A dedicated store (Pinecone, Chroma, Qdrant) holds vectors and a metadata blob.
This system needs both vectors AND ordinary relational data: repositories own
files, files own chunks, conversations own messages, and every one of those has
foreign keys and cascade deletes.

With pgvector, "delete a repository and everything under it" is one statement
and one transaction. With a separate vector store it is a distributed delete
across two systems with no shared transaction, and any failure between them
leaves orphaned vectors that retrieval will happily return forever.

One database, one transaction, one consistency story.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import func, insert, select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.models import Chunk, File, Repository
from app.services.chunker import CodeChunk, chunk_files
from app.services.embeddings import EmbeddingProvider, get_embedding_provider
from app.services.repository_parser import parse_repository

ProgressCallback = Callable[[str, int, int], None]


@dataclass
class IndexResult:
    """What one indexing run produced. Returned for CLI/API reporting."""

    repository_id: int
    name: str
    files: int
    chunks: int
    seconds: float
    embedding_model: str
    chunk_strategy: str
    cache_hits: int = 0
    cache_misses: int = 0

    def as_dict(self) -> dict:
        return {
            "repository_id": self.repository_id,
            "name": self.name,
            "files": self.files,
            "chunks": self.chunks,
            "seconds": round(self.seconds, 2),
            "embedding_model": self.embedding_model,
            "chunk_strategy": self.chunk_strategy,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
        }


def index_repository(
    session: Session,
    *,
    name: str,
    root: str | Path,
    url: str | None = None,
    source_type: str = "local",
    provider: EmbeddingProvider | None = None,
    settings: Settings | None = None,
    chunk_strategy: str = "code_aware",
    max_chunk_lines: int | None = None,
    replace_existing: bool = True,
    progress: ProgressCallback | None = None,
) -> IndexResult:
    """Parse, chunk, embed and store a codebase.

    Creates one Repository row per indexing run. `index_status` moves
    pending -> indexing -> ready, or -> failed with `error_message` set, so a
    caller polling the API can distinguish "still working" from "broken".
    """
    settings = settings or get_settings()
    provider = provider or get_embedding_provider(settings)
    max_chunk_lines = max_chunk_lines or settings.max_chunk_lines
    started = time.perf_counter()

    def report(stage: str, done: int = 0, total: int = 0) -> None:
        if progress:
            progress(stage, done, total)

    if replace_existing:
        _delete_matching(session, name, provider.model, chunk_strategy)

    repository = Repository(
        name=name,
        url=url,
        source_type=source_type,
        embedding_model=provider.model,
        embedding_dim=provider.dimension,
        chunk_strategy=chunk_strategy,
        max_chunk_lines=max_chunk_lines,
        index_status="indexing",
    )
    session.add(repository)
    session.commit()

    try:
        report("parsing")
        source_files, _ = parse_repository(root, max_file_bytes=settings.max_file_bytes)
        if not source_files:
            raise ValueError(f"No indexable source files found under {root!r}")

        report("chunking", 0, len(source_files))
        chunks = chunk_files(
            source_files,
            strategy=chunk_strategy,
            max_chunk_lines=max_chunk_lines,
        )
        if not chunks:
            raise ValueError("Parsing produced files but chunking produced nothing")

        report("embedding", 0, len(chunks))
        vectors = _embed_in_batches(chunks, provider, settings, report)

        report("storing", 0, len(chunks))
        _persist(session, repository, source_files, chunks, vectors)

        repository.index_status = "ready"
        repository.file_count = len(source_files)
        repository.chunk_count = len(chunks)
        repository.indexed_at = datetime.now(timezone.utc)
        session.commit()
        report("done", len(chunks), len(chunks))

        return IndexResult(
            repository_id=repository.id,
            name=repository.name,
            files=len(source_files),
            chunks=len(chunks),
            seconds=time.perf_counter() - started,
            embedding_model=provider.model,
            chunk_strategy=chunk_strategy,
            cache_hits=getattr(provider, "hits", 0),
            cache_misses=getattr(provider, "misses", 0),
        )

    except Exception as exc:
        # The failure rolled back the in-flight transaction, which would also
        # have discarded the status update. Start a clean transaction so the
        # repository is recorded as failed rather than stuck on "indexing"
        # forever.
        session.rollback()
        failed = session.get(Repository, repository.id)
        if failed is not None:
            failed.index_status = "failed"
            failed.error_message = f"{type(exc).__name__}: {exc}"[:2000]
            session.commit()
        raise


def _embed_in_batches(
    chunks: list[CodeChunk],
    provider: EmbeddingProvider,
    settings: Settings,
    report: Callable[[str, int, int], None],
) -> list[list[float]]:
    """Embed in visible batches.

    The provider batches internally too; this outer loop exists so a long run
    reports progress instead of appearing hung for several minutes.
    """
    texts = [chunk.embedding_text for chunk in chunks]
    size = max(1, settings.embedding_batch_size)
    vectors: list[list[float]] = []

    for start in range(0, len(texts), size):
        vectors.extend(provider.embed(texts[start : start + size]))
        report("embedding", min(start + size, len(texts)), len(texts))

    if len(vectors) != len(chunks):
        raise RuntimeError(
            f"Embedding count mismatch: {len(vectors)} vectors for {len(chunks)} chunks"
        )
    return vectors


def _persist(
    session: Session,
    repository: Repository,
    source_files,
    chunks: list[CodeChunk],
    vectors: list[list[float]],
) -> None:
    """Write files and chunks.

    Files go through the ORM (there are hundreds, and we need their generated
    ids). Chunks go through a bulk insert (there are thousands, and inserting
    them one ORM object at a time is dramatically slower).
    """
    file_rows = [
        File(
            repository_id=repository.id,
            path=source_file.path,
            language=source_file.language,
            num_lines=source_file.num_lines,
        )
        for source_file in source_files
    ]
    session.add_all(file_rows)
    session.flush()  # assigns file ids without committing

    file_id_by_path = {row.path: row.id for row in file_rows}

    payload = []
    for chunk, vector in zip(chunks, vectors):
        file_id = file_id_by_path.get(chunk.file_path)
        if file_id is None:
            # Would mean the chunker invented a path the parser never produced.
            raise RuntimeError(f"Chunk references unknown file {chunk.file_path!r}")
        payload.append(
            {
                "file_id": file_id,
                "repository_id": repository.id,
                "content": chunk.content,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "symbol": chunk.symbol,
                "kind": chunk.kind,
                "language": chunk.language,
                "file_path": chunk.file_path,
                "embedding": vector,
            }
        )

    if payload:
        session.execute(insert(Chunk), payload)


def _delete_matching(
    session: Session, name: str, embedding_model: str, chunk_strategy: str
) -> None:
    """Drop previous runs with the same (name, model, strategy).

    Only exact configuration matches are removed, so indexing the same repo
    with naive and code-aware chunking leaves BOTH rows in place — which is the
    whole point of the section 21 experiment.
    """
    stale = session.scalars(
        select(Repository).where(
            Repository.name == name,
            Repository.embedding_model == embedding_model,
            Repository.chunk_strategy == chunk_strategy,
        )
    ).all()
    for repository in stale:
        session.delete(repository)
    if stale:
        session.commit()


def delete_repository(session: Session, repository_id: int) -> bool:
    """Delete a repository and everything under it, in one transaction."""
    repository = session.get(Repository, repository_id)
    if repository is None:
        return False
    session.delete(repository)
    session.commit()
    return True


def list_repositories(session: Session) -> list[Repository]:
    return list(
        session.scalars(select(Repository).order_by(Repository.created_at.desc())).all()
    )


def count_chunks(session: Session, repository_id: int) -> int:
    return session.scalar(
        select(func.count()).select_from(Chunk).where(Chunk.repository_id == repository_id)
    ) or 0
