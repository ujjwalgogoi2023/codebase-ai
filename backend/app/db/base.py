"""Database engine, session factory, and schema creation.

Engine and session factory are built lazily through lru_cache rather than at
import time, so importing a model never opens a socket. That keeps the unit
tests (parser, chunker, embeddings) runnable with no database at all.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import Settings, get_settings


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


@lru_cache
def get_engine(database_url: str | None = None) -> Engine:
    settings = get_settings()
    return create_engine(
        database_url or settings.database_url,
        # Recycles connections that the server closed underneath us, which
        # happens routinely when the container restarts during development.
        pool_pre_ping=True,
        future=True,
    )


@lru_cache
def get_session_factory(database_url: str | None = None) -> sessionmaker[Session]:
    return sessionmaker(
        bind=get_engine(database_url),
        autoflush=False,
        expire_on_commit=False,
    )


@contextmanager
def session_scope(database_url: str | None = None) -> Iterator[Session]:
    """Transactional scope: commit on success, roll back on any exception."""
    session = get_session_factory(database_url)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(database_url: str | None = None, settings: Settings | None = None) -> None:
    """Create the vector extension and all tables.

    `CREATE EXTENSION vector` must run before the tables, because the chunks
    table has a column whose type the extension defines. Ordinary Postgres has
    no idea what `vector(1536)` means until the extension is installed.

    Alembic is deliberately skipped for now (spec section 15: do not
    overengineer the initial schema). Schema changes during development are
    handled by dropping and re-indexing, which is cheap when the embedding
    cache means re-indexing costs nothing.
    """
    # Imported here rather than at module scope so that importing this module
    # does not drag in the ORM models, keeping the import graph acyclic.
    import app.models  # noqa: F401  (registers models on Base.metadata)

    engine = get_engine(database_url)
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    Base.metadata.create_all(engine)


def drop_db(database_url: str | None = None) -> None:
    """Drop every table. Used by tests and by `scripts/reset_db.py`."""
    import app.models  # noqa: F401

    Base.metadata.drop_all(get_engine(database_url))
