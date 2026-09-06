"""Shared pytest fixtures.

Database-backed tests are skipped (not failed) when Postgres is unreachable, so
the parser/chunker/embedding suites still run on a machine with no Docker. Run
`docker compose up -d` to enable them.
"""

from __future__ import annotations

import os

import pytest
from sqlalchemy import create_engine, text

from app.config import Settings
from app.db.base import get_session_factory, init_db


@pytest.fixture(scope="session")
def database_url() -> str:
    url = os.getenv("TEST_DATABASE_URL") or Settings().database_url
    try:
        engine = create_engine(url, pool_pre_ping=True)
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        engine.dispose()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"database unavailable ({type(exc).__name__}); run docker compose up -d")
    return url


@pytest.fixture(scope="session")
def _schema(database_url: str) -> None:
    init_db(database_url)


@pytest.fixture
def db(database_url: str, _schema: None):
    """A session with a clean database.

    Truncating up front rather than after means a failed test leaves its rows
    behind for inspection, which is far more useful when debugging.
    """
    session = get_session_factory(database_url)()
    session.execute(
        text("TRUNCATE repositories, files, chunks, conversations, messages "
             "RESTART IDENTITY CASCADE")
    )
    session.commit()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture
def fake_settings() -> Settings:
    """Settings pinned to the offline embedding provider."""
    return Settings(embedding_provider="fake", embedding_dim=1536)


@pytest.fixture
def sample_repo(tmp_path):
    """A small but realistic repository to index."""

    def write(rel: str, text_content: str) -> None:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text_content, encoding="utf-8")

    write(
        "backend/auth.py",
        "import bcrypt\n"
        "\n"
        "\n"
        "def authenticate_user(username, password):\n"
        "    user = db.get_user(username)\n"
        "    if user is None:\n"
        "        return None\n"
        "    if not bcrypt.checkpw(password, user.hash):\n"
        "        return None\n"
        "    return user\n"
        "\n"
        "\n"
        "def hash_password(raw):\n"
        "    return bcrypt.hashpw(raw, bcrypt.gensalt())\n",
    )
    write(
        "backend/orders.py",
        "def create_order(cart, user):\n"
        "    order = Order(user_id=user.id)\n"
        "    for item in cart.items:\n"
        "        order.add(item)\n"
        "    return order.save()\n",
    )
    write(
        "frontend/src/api.js",
        "export async function fetchOrders(userId) {\n"
        "  const res = await fetch(`/api/orders/${userId}`);\n"
        "  return res.json();\n"
        "}\n",
    )
    write("README.md", "# Demo\n\n## Auth\n\nPasswords are hashed with bcrypt.\n")
    # Junk that must never be indexed.
    write("node_modules/lib/index.js", "module.exports = {};\n")

    return tmp_path
