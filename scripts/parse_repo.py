"""Dev CLI for pipeline stage 1 — inspect what the parser would index.

Usage:
    python scripts/parse_repo.py <path-to-repo> [--limit 20]

Every stage of this pipeline gets a way to be run and inspected on its own
(spec section 23: "each stage should be testable independently").
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.services.repository_parser import parse_repository  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Parse a repository and report what would be indexed.")
    ap.add_argument("repo", help="path to a repository on disk")
    ap.add_argument("--limit", type=int, default=15, help="how many files to list")
    args = ap.parse_args()

    files, stats = parse_repository(args.repo)

    print(f"\nRepository: {Path(args.repo).resolve()}")
    print("-" * 70)
    for f in files[: args.limit]:
        print(f"  {f.language:<11} {f.num_lines:>5} lines  {f.path}")
    if len(files) > args.limit:
        print(f"  ... and {len(files) - args.limit} more")

    print("-" * 70)
    for key, value in stats.as_dict().items():
        print(f"  {key:<26} {value}")
    total_lines = sum(f.num_lines for f in files)
    print(f"  {'total_lines_to_index':<26} {total_lines}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
