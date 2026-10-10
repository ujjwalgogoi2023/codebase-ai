"""Shared machinery for running the evaluation.

Indexes a repository under a given configuration, runs every dataset question
through retrieval (and optionally reranking), and scores where the correct file
landed.

The one rule this file exists to enforce: when comparing two configurations,
exactly one thing may differ. Same corpus, same questions, same scoring. If the
chunking strategy changes, the embedding model must not.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.config import Settings  # noqa: E402
from app.db.base import init_db, session_scope  # noqa: E402
from app.models import Repository  # noqa: E402
from app.services.embeddings import get_embedding_provider  # noqa: E402
from app.services.evaluation import (  # noqa: E402
    EvalQuestion,
    EvalReport,
    load_dataset,
    score_question,
)
from app.services.indexer import index_repository  # noqa: E402
from app.services.llm import get_llm_provider  # noqa: E402
from app.services.reranker import IdentityReranker, LLMReranker  # noqa: E402
from app.services.retriever import Retriever  # noqa: E402
from sqlalchemy import select  # noqa: E402


@dataclass
class RunConfig:
    """One point in the experiment grid."""

    label: str
    chunk_strategy: str = "code_aware"
    reranker: str = "identity"
    top_k: int = 10
    max_chunk_lines: int = 120

    def index_name(self, base: str) -> str:
        # The index is keyed by the things that change the STORED vectors.
        # Reranking happens at query time and does not need its own index, so
        # two reranker settings can share one.
        return f"eval:{base}:{self.chunk_strategy}:{self.max_chunk_lines}"


def ensure_indexed(session, config: RunConfig, base_name: str, source_path: Path, settings):
    """Index the repository for this config, reusing an existing index.

    Reusing matters for both cost and validity: the reranking comparison must
    run against the *same* vectors, or it would be measuring re-indexing noise
    as well as reranking.
    """
    provider = get_embedding_provider(settings)
    name = config.index_name(base_name)

    existing = session.scalars(
        select(Repository).where(
            Repository.name == name,
            Repository.embedding_model == provider.model,
            Repository.index_status == "ready",
        )
    ).first()
    if existing is not None:
        return existing.id, provider, True

    result = index_repository(
        session,
        name=name,
        root=source_path,
        provider=provider,
        settings=settings,
        chunk_strategy=config.chunk_strategy,
        max_chunk_lines=config.max_chunk_lines,
    )
    return result.repository_id, provider, False


def build_reranker(config: RunConfig, settings: Settings):
    if config.reranker in ("identity", "none", "off"):
        return IdentityReranker()
    return LLMReranker(
        get_llm_provider(settings), preview_lines=settings.rerank_preview_lines
    )


def run_config(
    config: RunConfig,
    questions: list[EvalQuestion],
    base_name: str,
    source_path: Path,
    settings: Settings,
    progress: bool = True,
) -> EvalReport:
    """Index if needed, run every question, and score the results."""
    with session_scope() as session:
        repository_id, provider, reused = ensure_indexed(
            session, config, base_name, source_path, settings
        )
        reranker = build_reranker(config, settings)
        retriever = Retriever(session, provider=provider, settings=settings)

        if progress:
            state = "reused index" if reused else "built index"
            print(f"  [{config.label}] {state} (repository {repository_id})")

        results = []
        for number, question in enumerate(questions, start=1):
            search = retriever.search(
                question.question, repository_id, top_k=config.top_k
            )

            # Rerank the WHOLE candidate list rather than truncating, so the
            # metric can see where the correct answer ended up rather than only
            # whether it survived.
            ranked = reranker.rerank(
                question.question, search.chunks, top_k=len(search.chunks)
            ).chunks

            results.append(
                score_question(
                    question,
                    [c.file_path for c in ranked],
                    [c.symbol for c in ranked],
                )
            )

            if progress and number % 10 == 0:
                print(f"    {number}/{len(questions)} questions", flush=True)

        return EvalReport(
            config={
                "label": config.label,
                "chunk_strategy": config.chunk_strategy,
                "reranker": config.reranker,
                "top_k": config.top_k,
                "max_chunk_lines": config.max_chunk_lines,
                "embedding_provider": settings.embedding_provider,
                "embedding_model": provider.model,
                "llm_provider": settings.llm_provider,
                "repository_id": repository_id,
            },
            results=results,
        )


def load_questions(dataset_path: Path) -> tuple[str, Path, list[EvalQuestion]]:
    """Load the dataset and resolve the repository path it refers to."""
    import json

    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    base_name, questions = load_dataset(dataset_path)

    source_path = (PROJECT_ROOT / data.get("source_path", "backend")).resolve()
    if not source_path.is_dir():
        raise SystemExit(f"Dataset source_path does not exist: {source_path}")

    return base_name, source_path, questions


def prepare(settings: Settings) -> None:
    init_db()
    if settings.embedding_provider.strip().lower() == "fake":
        print()
        print("  " + "=" * 66)
        print("  WARNING: EMBEDDING_PROVIDER=fake")
        print("  Fake embeddings are random vectors, so the scores below measure")
        print("  nothing. They prove the harness runs end to end, not that")
        print("  retrieval works. Set EMBEDDING_PROVIDER=ollama (or openai) for real")
        print("  numbers, and do not put these on a CV.")
        print("  " + "=" * 66)
