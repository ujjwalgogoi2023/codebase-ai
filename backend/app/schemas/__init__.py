"""Pydantic request/response models for the API.

Separate from the SQLAlchemy models on purpose. ORM objects carry things the
wire should never see — the 1536-float embedding on every chunk being the
obvious one, which would make a search response megabytes of noise.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Repositories
# ---------------------------------------------------------------------------

class RepositoryCreate(BaseModel):
    """Register a repository from a local path or a git URL.

    ZIP uploads use the multipart endpoint instead, since a file cannot travel
    in a JSON body.
    """

    name: str | None = Field(
        default=None,
        max_length=255,
        description="Defaults to the folder name, or owner/repo for a git URL.",
    )
    url: str | None = Field(default=None, description="https://github.com/owner/repo")
    path: str | None = Field(default=None, description="Local directory (development only)")
    chunk_strategy: str = Field(default="code_aware", pattern="^(code_aware|naive)$")
    max_chunk_lines: int | None = Field(default=None, ge=10, le=1000)
    index_now: bool = Field(
        default=True,
        description="Start indexing immediately in the background.",
    )


class RepositoryOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    url: str | None
    source_type: str
    index_status: str
    error_message: str | None
    embedding_model: str
    chunk_strategy: str
    file_count: int
    chunk_count: int
    created_at: datetime
    indexed_at: datetime | None


class FileOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    path: str
    language: str
    num_lines: int


class IndexStatusOut(BaseModel):
    """What a polling client needs while indexing runs."""

    repository_id: int
    index_status: str
    file_count: int
    chunk_count: int
    error_message: str | None = None


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

class SearchHitOut(BaseModel):
    chunk_id: int
    citation: str
    file_path: str
    language: str
    symbol: str | None
    kind: str
    start_line: int
    end_line: int
    score: float
    content: str


class SearchResponse(BaseModel):
    query: str
    repository_id: int
    top_k: int
    embedding_model: str
    count: int
    results: list[SearchHitOut]


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    repository_id: int
    question: str = Field(min_length=1, max_length=4000)
    conversation_id: int | None = Field(
        default=None,
        description="Continue an existing conversation. Omitted starts a new one.",
    )
    top_k: int | None = Field(default=None, ge=1, le=100)
    languages: list[str] | None = None
    path_prefix: str | None = None


class CitationOut(BaseModel):
    index: int
    chunk_id: int
    citation: str
    file_path: str
    start_line: int
    end_line: int
    symbol: str | None
    language: str
    kind: str
    score: float


class UsageOut(BaseModel):
    input_tokens: int
    output_tokens: int
    usd: float | None


class ChatResponse(BaseModel):
    conversation_id: int
    repository_id: int
    question: str
    answer: str
    citations: list[CitationOut]
    sources: list[SearchHitOut]
    is_grounded: bool
    no_context: bool
    citation_coverage: float
    invalid_markers: list[int]
    model: str
    usage: UsageOut


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    role: str
    content: str
    created_at: datetime


class ConversationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    repository_id: int
    title: str | None
    created_at: datetime


class ConversationDetailOut(ConversationOut):
    messages: list[MessageOut] = []


class HealthOut(BaseModel):
    status: str
    database: str
    embedding_provider: str
    embedding_model: str
    llm_provider: str
    llm_model: str
