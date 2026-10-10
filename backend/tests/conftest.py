"""Shared pytest fixtures.

Database-backed tests are skipped (not failed) when Postgres is unreachable, so
the parser/chunker/embedding suites still run on a machine with no Docker. Run
`docker compose up -d` to enable them.
"""

from __future__ import annotations

import os

# The tests must not depend on whichever model the developer's .env selects.
# The chunks table's vector width is fixed from EMBEDDING_DIM at import time,
# and the suite embeds with 1536-d fakes, so pin it before app.models loads.
# Real environment variables take precedence over .env in pydantic-settings.
os.environ["EMBEDDING_DIM"] = "1536"
os.environ["EMBEDDING_PROVIDER"] = "fake"
os.environ["LLM_PROVIDER"] = "fake"

import pytest  # noqa: E402
from sqlalchemy import create_engine, text  # noqa: E402

from app.config import Settings  # noqa: E402
from app.db.base import drop_db, get_session_factory, init_db  # noqa: E402


def _derive_test_url(url: str) -> str:
    """Point at a sibling `<name>_test` database.

    Tests TRUNCATE between cases. Running them against the development database
    would silently destroy every repository you had indexed, which is a nasty
    surprise the first time it happens mid-demo.
    """
    base, separator, name = url.rpartition("/")
    if not separator:
        return url
    name, question, query = name.partition("?")
    return f"{base}/{name}_test{question}{query}"


def _ensure_database_exists(url: str) -> None:
    """CREATE DATABASE the test database if it is not there yet.

    CREATE DATABASE cannot run inside a transaction, hence AUTOCOMMIT, and it
    must be issued from a different database, hence the connection to
    `postgres`.
    """
    base, _, name = url.rpartition("/")
    name = name.partition("?")[0]
    admin = create_engine(f"{base}/postgres", isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as connection:
            exists = connection.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :name"), {"name": name}
            ).scalar()
            if not exists:
                connection.execute(text(f'CREATE DATABASE "{name}"'))
    finally:
        admin.dispose()


@pytest.fixture(scope="session")
def database_url() -> str:
    url = os.getenv("TEST_DATABASE_URL") or _derive_test_url(Settings().database_url)

    # Safety belt: these fixtures truncate. Never let them do that to a
    # database that was not explicitly designated for testing.
    name = url.rpartition("/")[2].partition("?")[0]
    if not name.endswith("_test"):
        pytest.fail(
            f"Refusing to run destructive tests against database {name!r}. "
            "Set TEST_DATABASE_URL to a database whose name ends in '_test'."
        )

    try:
        _ensure_database_exists(url)
        engine = create_engine(url, pool_pre_ping=True)
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        engine.dispose()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"database unavailable ({type(exc).__name__}); run docker compose up -d")
    return url


@pytest.fixture(scope="session")
def _schema(database_url: str) -> None:
    """Rebuild the test schema from scratch once per session.

    `create_all` only adds missing tables; it never alters existing ones. So
    adding a column to a model leaves an old test database silently stale, and
    every test fails with UndefinedColumn instead of anything informative.
    Dropping first makes the suite immune to schema drift. Safe because the
    `database_url` fixture refuses any database not named *_test.
    """
    drop_db(database_url)
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
