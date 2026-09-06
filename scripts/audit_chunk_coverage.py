"""Correctness harness: prove the chunker never loses source lines.

Usage:
    python scripts/audit_chunk_coverage.py <repo> [<repo> ...]

The unit tests check this invariant on hand-written fixtures. This script checks
it on whatever real code you point it at, which is where the weird cases live:
nested f-strings, generated files, decorators on decorators, regex literals full
of braces.

Why the invariant matters: if the chunker drops lines, the system will answer
"that code does not exist in this repository" about code that plainly does. That
is the most dangerous RAG failure mode, because a grounded-sounding refusal is
indistinguishable from a correct one.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.services.chunker import chunk_file  # noqa: E402
from app.services.repository_parser import parse_repository  # noqa: E402


def audit(repo: str) -> tuple[int, int, list[tuple[str, int]]]:
    files, _ = parse_repository(repo)
    lossy: list[tuple[str, int]] = []
    total_chunks = 0

    for source_file in files:
        chunks = chunk_file(source_file)
        total_chunks += len(chunks)

        covered: set[int] = set()
        for chunk in chunks:
            covered.update(range(chunk.start_line, chunk.end_line + 1))

        missing = [
            i + 1
            for i, line in enumerate(source_file.content.splitlines())
            if line.strip() and (i + 1) not in covered
        ]
        if missing:
            lossy.append((source_file.path, len(missing)))

    return len(files), total_chunks, lossy


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2

    grand_files = grand_chunks = grand_lossy = 0
    for repo in argv:
        if not Path(repo).is_dir():
            print(f"  SKIP  {repo} (not a directory)")
            continue

        n_files, n_chunks, lossy = audit(repo)
        grand_files += n_files
        grand_chunks += n_chunks
        grand_lossy += len(lossy)

        flag = "OK  " if not lossy else "LOSS"
        print(f"  [{flag}] {repo:<46} {n_files:>5} files -> {n_chunks:>6} chunks")
        for path, count in lossy[:5]:
            print(f"           {path} lost {count} lines")

    print(
        f"\n  TOTAL: {grand_files} files, {grand_chunks} chunks, "
        f"{grand_lossy} files with lost lines"
    )
    return 1 if grand_lossy else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
