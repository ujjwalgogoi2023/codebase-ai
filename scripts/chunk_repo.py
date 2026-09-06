"""Dev CLI for pipeline stage 2 — inspect and compare chunking strategies.

Usage:
    python scripts/chunk_repo.py <path-to-repo>
    python scripts/chunk_repo.py <path-to-repo> --show 10
    python scripts/chunk_repo.py <path-to-repo> --compare

`--compare` is the groundwork for the section 21 experiment: same corpus, one
variable changed. It reports the structural difference between strategies.
Whether that difference actually improves RETRIEVAL is a separate question that
only the evaluation harness can answer.
"""

import argparse
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.services.chunker import chunk_files  # noqa: E402
from app.services.repository_parser import parse_repository  # noqa: E402


def summarize(chunks, label: str) -> dict:
    sizes = [c.num_lines for c in chunks]
    named = [c for c in chunks if c.symbol]
    return {
        "label": label,
        "chunks": len(chunks),
        "named": len(named),
        "named_pct": round(100 * len(named) / len(chunks), 1) if chunks else 0.0,
        "mean_lines": round(statistics.mean(sizes), 1) if sizes else 0,
        "median_lines": round(statistics.median(sizes), 1) if sizes else 0,
        "max_lines": max(sizes) if sizes else 0,
        "kinds": Counter(c.kind for c in chunks),
    }


def print_summary(s: dict) -> None:
    print(f"\n  [{s['label']}]")
    print(f"    chunks            {s['chunks']}")
    print(f"    with a symbol     {s['named']} ({s['named_pct']}%)")
    print(f"    lines per chunk   mean {s['mean_lines']}  median {s['median_lines']}  max {s['max_lines']}")
    print(f"    kinds             {dict(s['kinds'].most_common())}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Chunk a repository and report the result.")
    ap.add_argument("repo")
    ap.add_argument("--show", type=int, default=0, help="print N sample chunks")
    ap.add_argument("--compare", action="store_true", help="naive vs code-aware")
    ap.add_argument("--max-chunk-lines", type=int, default=120)
    args = ap.parse_args()

    files, stats = parse_repository(args.repo)
    print(f"\nRepository: {Path(args.repo).resolve()}")
    print(f"Indexable files: {stats.kept}   lines: {sum(f.num_lines for f in files)}")
    print("-" * 74)

    aware = chunk_files(files, strategy="code_aware", max_chunk_lines=args.max_chunk_lines)

    if args.compare:
        naive = chunk_files(files, strategy="naive", max_chunk_lines=args.max_chunk_lines)
        print_summary(summarize(naive, "naive"))
        print_summary(summarize(aware, "code_aware"))
        print(
            "\n  A symbol is what makes a chunk describable. Naive chunking has none,"
            "\n  so its embeddings cannot encode 'this is the function that does X'."
        )
    else:
        print_summary(summarize(aware, "code_aware"))

    for chunk in aware[: args.show]:
        head = chunk.content.splitlines()[0][:66] if chunk.content else ""
        print(f"\n  {chunk.citation}  [{chunk.kind}] {chunk.symbol or '-'}")
        print(f"      {head}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
