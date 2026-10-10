"""Retrieval: find the code chunks most relevant to a question.

Stage 5 of the RAG pipeline — the query side, where everything indexed so far
finally gets used:

    question -> embed -> [FILTERED ANN SEARCH] -> top-K candidates -> rerank -> LLM

WHAT ACTUALLY HAPPENS
---------------------
The question becomes a vector using the SAME model that embedded the chunks,
then Postgres ranks chunks by cosine distance:

    SELECT ... ORDER BY embedding <=> :query_vector LIMIT :k

`<=>` is pgvector's cosine distance operator. For unit-length vectors it equals
`1 - cosine_similarity`, so 0.0 means identical direction and 2.0 means
opposite. We report `score = 1 - distance` because "higher is better" is what
everyone expects to read.

WHY RETRIEVE MORE THAN WE SEND TO THE LLM
-----------------------------------------
This stage returns TOP_K (default 10), but only RERANK_TOP_K (default 3) chunks
ever reach the LLM. That gap is deliberate.

Vector search answers "which chunks are semantically similar to this question?"
That is not the same as "which chunks ANSWER this question". The similarity is
also computed between two fundamentally different kinds of text: a natural
language question ("where do we validate tokens?") versus source code. The
embedding model bridges that gap approximately, not exactly.

So we cast a wide net cheaply, then spend a more expensive, more accurate model
on the shortlist (stage 10). Retrieving only 3 would mean the reranker never
sees the chunk that a slightly-off similarity score ranked 7th.

APPROXIMATE, NOT EXACT
----------------------
HNSW is an approximate nearest-neighbour index: it can miss a true neighbour.
`hnsw.ef_search` controls how much of the graph is explored — higher means
better recall and slower queries. It is exposed as a setting because it is a
genuine tradeoff, not a constant.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import Select, select, text
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.models import Chunk, Repository
from app.services.embeddings import EmbeddingProvider, get_embedding_provider


# pgvector rejects anything outside this range for hnsw.ef_search.
HNSW_EF_SEARCH_MIN = 1
HNSW_EF_SEARCH_MAX = 1000


class RetrievalError(RuntimeError):
    """Raised when a search cannot be performed correctly."""


@dataclass(frozen=True)
class RetrievedChunk:
    """One search hit, with everything needed to cite it."""

    chunk_id: int
    file_path: str
    language: str
    symbol: str | None
    kind: str
    start_line: int
    end_line: int
    content: str
    distance: float
    context: str | None = None

    @property
    def score(self) -> float:
        """Cosine similarity in [-1, 1]; higher is more relevant."""
        return 1.0 - self.distance

    @property
    def citation(self) -> str:
        return f"{self.file_path}:{self.start_line}-{self.end_line}"

    @property
    def label(self) -> str:
        return f"{self.citation} ({self.symbol})" if self.symbol else self.citation

    def as_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "citation": self.citation,
            "file_path": self.file_path,
            "language": self.language,
            "symbol": self.symbol,
            "kind": self.kind,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "score": round(self.score, 4),
            "content": self.content,
        }


@dataclass
class SearchFilters:
    """Metadata narrowing applied alongside the vector search.

    This is the part a pure vector database makes awkward and a relational one
    makes trivial: these are ordinary SQL predicates on indexed columns, in the
    same query as the ANN scan.
    """

    languages: list[str] | None = None
    kinds: list[str] | None = None
    path_prefix: str | None = None
    min_score: float | None = None

    def is_empty(self) -> bool:
        return not any((self.languages, self.kinds, self.path_prefix, self.min_score))


@dataclass
class SearchResult:
    """Results plus the parameters that produced them.

    Retrieval is the stage that gets tuned most, so every result carries the
    configuration that produced it. Without that, comparing two evaluation runs
    is guesswork.
    """

    query: str
    repository_id: int
    chunks: list[RetrievedChunk] = field(default_factory=list)
    top_k: int = 0
    embedding_model: str = ""
    filters: SearchFilters | None = None

    def __len__(self) -> int:
        return len(self.chunks)

    def __iter__(self):
        return iter(self.chunks)

    def citations(self) -> list[str]:
        return [c.citation for c in self.chunks]

    def as_dict(self) -> dict:
        return {
            "query": self.query,
            "repository_id": self.repository_id,
            "top_k": self.top_k,
            "embedding_model": self.embedding_model,
            "count": len(self.chunks),
            "results": [c.as_dict() for c in self.chunks],
        }


class Retriever:
    """Embeds a question and finds the nearest chunks in one repository."""

    def __init__(
        self,
        session: Session,
        provider: EmbeddingProvider | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.session = session
        self.settings = settings or get_settings()
        self.provider = provider or get_embedding_provider(self.settings)

    def search(
        self,
        query: str,
        repository_id: int,
        *,
        top_k: int | None = None,
        filters: SearchFilters | None = None,
    ) -> SearchResult:
        query = (query or "").strip()
        if not query:
            raise RetrievalError("Query is empty")

        # `top_k or default` would be wrong: 0 is falsy, so an explicit
        # top_k=0 would silently become the default instead of being rejected.
        top_k = self.settings.top_k if top_k is None else top_k
        if top_k < 1:
            raise RetrievalError(f"top_k must be >= 1, got {top_k}")

        repository = self._load_ready_repository(repository_id)
        query_vector = self.provider.embed_one(query)

        self._apply_ef_search(top_k)

        statement = self._build_query(repository_id, query_vector, top_k, filters)
        rows = self.session.execute(statement).all()

        chunks = [
            RetrievedChunk(
                chunk_id=row.id,
                file_path=row.file_path,
                language=row.language,
                symbol=row.symbol,
                kind=row.kind,
                start_line=row.start_line,
                end_line=row.end_line,
                content=row.content,
                distance=float(row.distance),
                context=row.context,
            )
            for row in rows
        ]

        if filters and filters.min_score is not None:
            chunks = [c for c in chunks if c.score >= filters.min_score]

        return SearchResult(
            query=query,
            repository_id=repository_id,
            chunks=chunks,
            top_k=top_k,
            embedding_model=repository.embedding_model,
            filters=filters,
        )

    # -- internals ---------------------------------------------------------

    def _load_ready_repository(self, repository_id: int) -> Repository:
        repository = self.session.get(Repository, repository_id)
        if repository is None:
            raise RetrievalError(f"Repository {repository_id} does not exist")

        if repository.index_status != "ready":
            raise RetrievalError(
                f"Repository {repository_id} is not searchable "
                f"(index_status={repository.index_status!r})"
            )

        # The single most destructive silent failure in a RAG system: embedding
        # the question with a different model than the one that embedded the
        # chunks. Both vectors have the same dimension, so every query still
        # "works" — it just returns essentially random results, with confident
        # citations attached. Fail loudly instead.
        if repository.embedding_model != self.provider.model:
            raise RetrievalError(
                f"Embedding model mismatch: repository {repository_id} was indexed "
                f"with {repository.embedding_model!r} but the query would use "
                f"{self.provider.model!r}. Re-index, or switch the provider back."
            )

        return repository

    def _apply_ef_search(self, top_k: int) -> None:
        """Tune HNSW recall for this transaction.

        ef_search should be at least top_k, or the index explores fewer
        candidates than we asked to return. Scaling it above top_k improves the
        odds that the true nearest neighbours are actually found.

        pgvector enforces a hard range of 1..1000, so this clamps rather than
        letting an ambitious top_k turn into a database error. A top_k above
        1000 is well outside what this system does anyway — the LLM sees three.
        """
        ef_search = max(int(self.settings.hnsw_ef_search), top_k)
        ef_search = min(max(ef_search, HNSW_EF_SEARCH_MIN), HNSW_EF_SEARCH_MAX)
        # SET LOCAL does not accept bind parameters, so the value is coerced to
        # int and clamped above before interpolation. It never touches user input.
        self.session.execute(text(f"SET LOCAL hnsw.ef_search = {ef_search}"))

    def _build_query(
        self,
        repository_id: int,
        query_vector: list[float],
        top_k: int,
        filters: SearchFilters | None,
    ) -> Select:
        distance = Chunk.embedding.cosine_distance(query_vector).label("distance")

        statement = select(
            Chunk.id,
            Chunk.file_path,
            Chunk.language,
            Chunk.symbol,
            Chunk.kind,
            Chunk.start_line,
            Chunk.end_line,
            Chunk.content,
            Chunk.context,
            distance,
        ).where(Chunk.repository_id == repository_id)

        if filters:
            if filters.languages:
                statement = statement.where(Chunk.language.in_(filters.languages))
            if filters.kinds:
                statement = statement.where(Chunk.kind.in_(filters.kinds))
            if filters.path_prefix:
                # startswith() escapes LIKE wildcards, so a path containing '%'
                # is matched literally rather than as a pattern.
                statement = statement.where(
                    Chunk.file_path.startswith(filters.path_prefix, autoescape=True)
                )

        return statement.order_by(distance).limit(top_k)


def search_repository(
    session: Session,
    query: str,
    repository_id: int,
    *,
    top_k: int | None = None,
    filters: SearchFilters | None = None,
    provider: EmbeddingProvider | None = None,
    settings: Settings | None = None,
) -> SearchResult:
    """Convenience wrapper for one-off searches (CLI, tests, API handlers)."""
    return Retriever(session, provider=provider, settings=settings).search(
        query, repository_id, top_k=top_k, filters=filters
    )
