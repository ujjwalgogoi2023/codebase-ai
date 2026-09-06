"""FastAPI dependencies.

Everything the handlers need arrives through these, which is what makes the API
testable: the test suite overrides them to point at the test database and the
offline providers, and no handler needs to know.

`get_background_session` exists because a background task outlives the request.
The request's session is closed the moment the response is sent, so a task that
kept using it would fail on its first query.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager

from fastapi import Depends
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db.base import session_scope
from app.services.embeddings import EmbeddingProvider, get_embedding_provider
from app.services.llm import LLMProvider, get_llm_provider


def get_config() -> Settings:
    return get_settings()


def get_db() -> Iterator[Session]:
    """A session scoped to one request, committed or rolled back at the end."""
    with session_scope() as session:
        yield session


def get_background_session() -> Callable[[], AbstractContextManager[Session]]:
    """A session factory usable after the response has been sent."""
    return session_scope


def get_embeddings(
    settings: Settings = Depends(get_config),
) -> EmbeddingProvider:
    return get_embedding_provider(settings)


def get_llm(settings: Settings = Depends(get_config)) -> LLMProvider:
    return get_llm_provider(settings)
