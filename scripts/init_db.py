"""Create the database schema (extension + tables + indexes).

Usage:
    python scripts/init_db.py
    python scripts/init_db.py --reset     # drop everything first
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.config import get_settings  # noqa: E402
from app.db.base import drop_db, get_engine, init_db  # noqa: E402
from sqlalchemy import text  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="drop all tables first")
    args = ap.parse_args()

    settings = get_settings()
    print(f"Database: {settings.database_url.rsplit('@', 1)[-1]}")

    if args.reset:
        drop_db()
        print("  dropped existing tables")

    init_db()
    print("  schema created")

    with get_engine().connect() as conn:
        tables = conn.execute(text(
            "SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY tablename"
        )).scalars().all()
        indexes = conn.execute(text(
            "SELECT indexname, indexdef FROM pg_indexes "
            "WHERE schemaname='public' AND indexdef ILIKE '%hnsw%'"
        )).all()

    print(f"\n  tables: {', '.join(tables)}")
    for name, definition in indexes:
        print(f"  vector index: {name}")
        print(f"      {definition}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
