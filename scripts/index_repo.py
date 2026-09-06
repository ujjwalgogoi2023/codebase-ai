"""Dev CLI for pipeline stage 4 — index a repository into pgvector.

Usage:
    python scripts/index_repo.py <path> --name my-app
    python scripts/index_repo.py <path> --name my-app --strategy naive
    python scripts/index_repo.py --list
    python scripts/index_repo.py --delete 3

Indexing the same path twice with the same (name, model, strategy) replaces the
earlier run. Changing the strategy creates a second run instead, so both remain
queryable for the chunking experiment.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.config import get_settings  # noqa: E402
from app.db.base import init_db, session_scope  # noqa: E402
from app.services.embeddings import get_embedding_provider  # noqa: E402
from app.services.indexer import (  # noqa: E402
    delete_repository,
    index_repository,
    list_repositories,
)


class ProgressPrinter:
    """Single-line progress, so a long index does not scroll the terminal."""

    def __init__(self) -> None:
        self.last = 0.0

    def __call__(self, stage: str, done: int, total: int) -> None:
        now = time.time()
        if stage not in ("done",) and now - self.last < 0.2:
            return
        self.last = now
        bar = f"{done}/{total}" if total else ""
        print(f"\r  {stage:<10} {bar:<16}", end="", flush=True)
        if stage == "done":
            print()


def do_list() -> int:
    with session_scope() as session:
        repositories = list_repositories(session)
        if not repositories:
            print("\n  (no repositories indexed yet)\n")
            return 0
        print(f"\n  {'id':<4} {'name':<24} {'strategy':<12} {'status':<9} {'files':>6} {'chunks':>7}")
        print("  " + "-" * 70)
        for repo in repositories:
            print(
                f"  {repo.id:<4} {repo.name[:23]:<24} {repo.chunk_strategy:<12} "
                f"{repo.index_status:<9} {repo.file_count:>6} {repo.chunk_count:>7}"
            )
            if repo.error_message:
                print(f"       error: {repo.error_message[:90]}")
        print()
    return 0


def do_delete(repository_id: int) -> int:
    with session_scope() as session:
        ok = delete_repository(session, repository_id)
    print(f"  {'deleted' if ok else 'not found'}: repository {repository_id}")
    return 0 if ok else 1


def do_index(args) -> int:
    settings = get_settings()
    root = Path(args.path).resolve()
    if not root.is_dir():
        print(f"  not a directory: {root}")
        return 2

    name = args.name or root.name
    provider = get_embedding_provider(settings)

    print(f"\n  repository : {root}")
    print(f"  name       : {name}")
    print(f"  provider   : {settings.embedding_provider} / {provider.model} "
          f"({provider.dimension}d)")
    print(f"  strategy   : {args.strategy}\n")

    with session_scope() as session:
        result = index_repository(
            session,
            name=name,
            root=root,
            url=args.url,
            provider=provider,
            settings=settings,
            chunk_strategy=args.strategy,
            progress=ProgressPrinter(),
        )

    print()
    for key, value in result.as_dict().items():
        print(f"  {key:<18} {value}")
    print()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Index a repository into pgvector.")
    ap.add_argument("path", nargs="?", help="path to the repository")
    ap.add_argument("--name", help="name for this index (defaults to the folder name)")
    ap.add_argument("--url", help="source URL, recorded as metadata")
    ap.add_argument("--strategy", default="code_aware", choices=["code_aware", "naive"])
    ap.add_argument("--list", action="store_true", help="list indexed repositories")
    ap.add_argument("--delete", type=int, metavar="ID", help="delete a repository")
    args = ap.parse_args()

    init_db()

    if args.list:
        return do_list()
    if args.delete is not None:
        return do_delete(args.delete)
    if not args.path:
        ap.error("provide a path, or use --list / --delete")
    return do_index(args)


if __name__ == "__main__":
    raise SystemExit(main())
