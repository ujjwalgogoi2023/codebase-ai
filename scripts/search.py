"""Dev CLI for pipeline stage 5 — inspect retrieval on its own.

Usage:
    python scripts/search.py 2 "where are passwords hashed?"
    python scripts/search.py 2 "jwt" --top-k 5 --language python
    python scripts/search.py 2 "orders" --path-prefix backend/ --show-content

This exists because retrieval must be debuggable WITHOUT the LLM in the way
(spec section 16). When an answer is wrong there are only two possibilities:
retrieval fetched the wrong code, or the LLM misread the right code. Being able
to look at the retrieved set alone tells you which — and they need completely
different fixes.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.config import get_settings  # noqa: E402
from app.db.base import session_scope  # noqa: E402
from app.services.embeddings import get_embedding_provider  # noqa: E402
from app.services.retriever import (  # noqa: E402
    RetrievalError,
    Retriever,
    SearchFilters,
)


def bar(score: float, width: int = 24) -> str:
    filled = max(0, min(width, round(score * width)))
    return "#" * filled + "." * (width - filled)


def main() -> int:
    ap = argparse.ArgumentParser(description="Search an indexed repository.")
    ap.add_argument("repository_id", type=int)
    ap.add_argument("query")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--language", action="append", dest="languages")
    ap.add_argument("--kind", action="append", dest="kinds")
    ap.add_argument("--path-prefix")
    ap.add_argument("--min-score", type=float)
    ap.add_argument("--show-content", action="store_true")
    args = ap.parse_args()

    settings = get_settings()
    provider = get_embedding_provider(settings)

    filters = SearchFilters(
        languages=args.languages,
        kinds=args.kinds,
        path_prefix=args.path_prefix,
        min_score=args.min_score,
    )

    with session_scope() as session:
        retriever = Retriever(session, provider=provider, settings=settings)
        try:
            result = retriever.search(
                args.query,
                args.repository_id,
                top_k=args.top_k,
                filters=None if filters.is_empty() else filters,
            )
        except RetrievalError as exc:
            print(f"\n  error: {exc}\n")
            return 1

        print(f'\n  query    "{result.query}"')
        print(f"  repo     {result.repository_id}   model {result.embedding_model}")
        print(f"  top_k    {result.top_k}   returned {len(result)}")
        if not filters.is_empty():
            print(f"  filters  {filters}")
        print("  " + "-" * 76)

        if not result.chunks:
            print("\n  no matching chunks\n")
            return 0

        for rank, hit in enumerate(result, start=1):
            print(
                f"  {rank:>2}. {hit.score:.4f} [{bar(hit.score)}] "
                f"{hit.citation}"
            )
            print(f"      {hit.kind:<9} {hit.language:<11} {hit.symbol or '-'}")
            if args.show_content:
                for line in hit.content.splitlines()[:6]:
                    print(f"        | {line[:72]}")
                print()
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
