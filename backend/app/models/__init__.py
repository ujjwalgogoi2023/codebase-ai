"""Relational schema for CodebaseAI (spec section 15).

    repositories 1---* files 1---* chunks
    repositories 1---* conversations 1---* messages

The interesting design decision is on `chunks.repository_id`, which is
technically redundant — you could reach the repository through `files`. See the
comment on that column for why it is there anyway.
"""

from __future__ import annotations

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.config import get_settings
from app.db.base import Base

# pgvector needs a fixed dimension at DDL time to build an index, so this is
# read once from configuration rather than varying per row. Changing
# EMBEDDING_DIM therefore requires recreating the chunks table — which is fine,
# because vectors from two different models were never comparable anyway.
EMBEDDING_DIM = get_settings().embedding_dim


class Repository(Base):
    """One indexing run over one codebase.

    Indexing the same code twice with different settings creates two rows, not
    one. That is deliberate: it is what makes the section 21 experiments
    possible (index with naive chunking, index with code-aware chunking, then
    evaluate both without re-indexing between every question).
    """

    __tablename__ = "repositories"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    source_type: Mapped[str] = mapped_column(String(16), default="local")  # local|github|zip

    # Where the code lives on disk. Recorded so a repository created via the
    # API can be indexed later by a background task, which has no access to the
    # upload or clone that produced it.
    source_path: Mapped[str | None] = mapped_column(String(2048), nullable=True)

    # The configuration this index was built with. Stored because vectors are
    # only comparable within a single (model, dimension) pair, and because an
    # experiment is meaningless if you cannot say what produced each number.
    embedding_model: Mapped[str] = mapped_column(String(128))
    embedding_dim: Mapped[int] = mapped_column(Integer, default=EMBEDDING_DIM)
    chunk_strategy: Mapped[str] = mapped_column(String(32), default="code_aware")
    max_chunk_lines: Mapped[int] = mapped_column(Integer, default=120)

    # Indexing is slow enough that the API must report progress rather than
    # block (spec section 30: "how do you handle long-running indexing?").
    index_status: Mapped[str] = mapped_column(String(16), default="pending")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    file_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    files: Mapped[list[File]] = relationship(
        back_populates="repository", cascade="all, delete-orphan", passive_deletes=True
    )
    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="repository", cascade="all, delete-orphan", passive_deletes=True
    )
    conversations: Mapped[list[Conversation]] = relationship(
        back_populates="repository", cascade="all, delete-orphan", passive_deletes=True
    )

    def __repr__(self) -> str:
        return f"<Repository {self.id} {self.name!r} {self.index_status}>"


class File(Base):
    __tablename__ = "files"
    __table_args__ = (
        # Re-indexing must not silently create duplicate rows for one path.
        UniqueConstraint("repository_id", "path", name="uq_files_repo_path"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    repository_id: Mapped[int] = mapped_column(
        ForeignKey("repositories.id", ondelete="CASCADE"), index=True
    )
    path: Mapped[str] = mapped_column(String(1024))
    language: Mapped[str] = mapped_column(String(32), index=True)
    num_lines: Mapped[int] = mapped_column(Integer, default=0)

    repository: Mapped[Repository] = relationship(back_populates="files")
    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="file", cascade="all, delete-orphan", passive_deletes=True
    )

    def __repr__(self) -> str:
        return f"<File {self.id} {self.path!r}>"


class Chunk(Base):
    """A retrievable unit of code and its embedding."""

    __tablename__ = "chunks"
    __table_args__ = (
        # HNSW over cosine distance. Chosen over IVFFlat because HNSW needs no
        # training pass and works on an empty table, so indexing order does not
        # matter. `vector_cosine_ops` must match the `<=>` operator used at
        # query time, or Postgres silently ignores the index and falls back to
        # a sequential scan.
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("ix_chunks_repo_language", "repository_id", "language"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    file_id: Mapped[int] = mapped_column(
        ForeignKey("files.id", ondelete="CASCADE"), index=True
    )

    # Denormalized from files. Reaching the repository through a join would be
    # correct, but a filtered vector search then needs `JOIN files ON ...` in
    # the same query as the ANN scan, and the planner tends to abandon the HNSW
    # index for a sequential scan when it does. One redundant integer column
    # keeps "search only inside repository N" fast, which is the single most
    # common query this system runs.
    repository_id: Mapped[int] = mapped_column(
        ForeignKey("repositories.id", ondelete="CASCADE"), index=True
    )

    content: Mapped[str] = mapped_column(Text)
    # Module-level definitions the chunk references (see chunker). Shown to the
    # LLM beside the chunk, never embedded or cited. NULL for chunks indexed
    # before this existed, and for languages it is not computed for.
    context: Mapped[str | None] = mapped_column(Text, nullable=True)
    start_line: Mapped[int] = mapped_column(Integer)
    end_line: Mapped[int] = mapped_column(Integer)
    symbol: Mapped[str | None] = mapped_column(String(512), nullable=True)
    kind: Mapped[str] = mapped_column(String(32), default="module")

    # Denormalized alongside repository_id, for metadata filtering
    # ("only search Python files") without a join.
    language: Mapped[str] = mapped_column(String(32))
    file_path: Mapped[str] = mapped_column(String(1024))

    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))

    file: Mapped[File] = relationship(back_populates="chunks")
    repository: Mapped[Repository] = relationship(back_populates="chunks")

    @property
    def citation(self) -> str:
        return f"{self.file_path}:{self.start_line}-{self.end_line}"

    def __repr__(self) -> str:
        return f"<Chunk {self.id} {self.citation} {self.symbol!r}>"


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(primary_key=True)
    repository_id: Mapped[int] = mapped_column(
        ForeignKey("repositories.id", ondelete="CASCADE"), index=True
    )
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    repository: Mapped[Repository] = relationship(back_populates="conversations")
    messages: Mapped[list[Message]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Message.id",
    )


class Message(Base):
    """One turn of a conversation.

    Note what is NOT here: retrieved context is not stored as part of the
    message. Conversation history is not a substitute for retrieval (spec
    section 14) — every question runs its own search, so replaying history must
    not smuggle in stale context from an earlier turn.
    """

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(16))  # user | assistant
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


__all__ = [
    "EMBEDDING_DIM",
    "Repository",
    "File",
    "Chunk",
    "Conversation",
    "Message",
]
