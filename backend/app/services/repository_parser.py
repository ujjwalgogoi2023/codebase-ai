"""Repository parsing: walk a codebase and decide what is worth indexing.

This is the first stage of the RAG ingestion pipeline:

    Repository -> [Parsing] -> Chunking -> Embeddings -> pgvector

Its only job is to answer "which files are source code we care about, and what
is in them?". Everything downstream (chunking, embedding) operates on the
SourceFile records produced here.

The filtering matters more than it looks: a single `node_modules/` directory can
hold 100k files. Embedding those would cost real money and would flood retrieval
with third-party code that answers none of the user's questions about *their*
repository.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Directories that never contain first-party source worth answering questions
# about. Pruned during the walk so we never even descend into them.
IGNORED_DIRS: frozenset[str] = frozenset(
    {
        "node_modules",
        "venv",
        "env",
        "__pycache__",
        "dist",
        "build",
        "out",
        "target",          # Java/Maven, Rust
        "bin",
        "obj",
        "coverage",
        "htmlcov",
        "site-packages",
        "vendor",
        "migrations",      # usually generated, very repetitive
    }
)

# Files that are technically a supported language but are generated or minified,
# so they are noise for retrieval.
IGNORED_FILE_SUFFIXES: tuple[str, ...] = (
    ".min.js",
    ".min.css",
    ".bundle.js",
    ".d.ts",           # type declarations, not implementation
    ".pb.py",          # protobuf generated
    "_pb2.py",
)

IGNORED_FILENAMES: frozenset[str] = frozenset(
    {
        "package-lock.json",
        "yarn.lock",
        "poetry.lock",
        "pnpm-lock.yaml",
    }
)

# Start deliberately small (spec section 5.2: do NOT support 20+ languages).
# Markdown is included because README/docs genuinely answer "how does X work"
# questions and show up as legitimate retrieval candidates.
LANGUAGE_BY_EXTENSION: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".md": "markdown",
}


@dataclass(frozen=True)
class SourceFile:
    """One file we intend to index.

    `path` is repo-relative and always POSIX-style, so citations render the same
    on Windows and Linux ("backend/auth.py", never "backend\auth.py").
    """

    path: str
    language: str
    content: str
    num_lines: int
    size_bytes: int


@dataclass
class ParseStats:
    """Why files were dropped. Useful for debugging a disappointing index."""

    files_seen: int = 0
    dirs_pruned: int = 0
    skipped_unsupported_ext: int = 0
    skipped_ignored_name: int = 0
    skipped_too_large: int = 0
    skipped_binary: int = 0
    skipped_empty: int = 0
    kept: int = 0
    languages: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "files_seen": self.files_seen,
            "dirs_pruned": self.dirs_pruned,
            "skipped_unsupported_ext": self.skipped_unsupported_ext,
            "skipped_ignored_name": self.skipped_ignored_name,
            "skipped_too_large": self.skipped_too_large,
            "skipped_binary": self.skipped_binary,
            "skipped_empty": self.skipped_empty,
            "kept": self.kept,
            "languages": dict(sorted(self.languages.items())),
        }


def detect_language(filename: str) -> str | None:
    """Map a filename to a supported language, or None if we don't index it."""
    return LANGUAGE_BY_EXTENSION.get(Path(filename).suffix.lower())


def should_ignore_dir(name: str) -> bool:
    # Any dotted directory (.git, .venv, .next, .pytest_cache, .idea) plus the
    # explicit list above.
    return name.startswith(".") or name in IGNORED_DIRS


def is_virtualenv(path: Path) -> bool:
    """Detect a Python virtual environment by its marker file.

    Name matching alone is not enough: `venv/` and `.venv/` are conventions,
    not rules, and real projects contain `myenv/`, `env311/`, `.virtualenv/`.
    Every one of them holds a full copy of the standard library plus every
    installed package, so walking one is both slow and actively harmful to
    retrieval quality.

    `pyvenv.cfg` is written by the stdlib `venv` module and by virtualenv, so
    it identifies the directory regardless of what someone named it.
    """
    return (path / "pyvenv.cfg").is_file()


def should_ignore_file(name: str) -> bool:
    lowered = name.lower()
    if lowered in IGNORED_FILENAMES:
        return True
    return any(lowered.endswith(suffix) for suffix in IGNORED_FILE_SUFFIXES)


def _read_text(abs_path: Path) -> str | None:
    """Read a file as UTF-8. Returns None if it looks binary."""
    try:
        raw = abs_path.read_bytes()
    except OSError:
        return None
    # A NUL byte in the first block is the classic binary heuristic.
    if b"\x00" in raw[:8192]:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def parse_repository(
    root: str | Path,
    max_file_bytes: int = 1_000_000,
) -> tuple[list[SourceFile], ParseStats]:
    """Walk `root` and return the source files worth indexing, plus stats.

    Results are sorted by path so indexing runs are deterministic — which
    matters when we later compare evaluation results between configurations.
    """
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {root_path}")

    files: list[SourceFile] = []
    stats = ParseStats()

    for dirpath, dirnames, filenames in os.walk(root_path):
        # Prune in place so os.walk never descends into ignored trees.
        keep_dirs = [
            d
            for d in dirnames
            if not should_ignore_dir(d) and not is_virtualenv(Path(dirpath) / d)
        ]
        stats.dirs_pruned += len(dirnames) - len(keep_dirs)
        dirnames[:] = sorted(keep_dirs)

        for filename in sorted(filenames):
            stats.files_seen += 1

            # Explicit junk is checked first so the stats attribute the real
            # reason: package-lock.json is dropped because it is a lock file,
            # not merely because .json happens to be an unsupported language.
            if should_ignore_file(filename):
                stats.skipped_ignored_name += 1
                continue

            language = detect_language(filename)
            if language is None:
                stats.skipped_unsupported_ext += 1
                continue

            abs_path = Path(dirpath) / filename
            try:
                size_bytes = abs_path.stat().st_size
            except OSError:
                stats.skipped_binary += 1
                continue

            if size_bytes > max_file_bytes:
                stats.skipped_too_large += 1
                continue

            content = _read_text(abs_path)
            if content is None:
                stats.skipped_binary += 1
                continue
            if not content.strip():
                stats.skipped_empty += 1
                continue

            rel_path = abs_path.relative_to(root_path).as_posix()
            files.append(
                SourceFile(
                    path=rel_path,
                    language=language,
                    content=content,
                    num_lines=len(content.splitlines()),
                    size_bytes=size_bytes,
                )
            )
            stats.kept += 1
            stats.languages[language] = stats.languages.get(language, 0) + 1

    files.sort(key=lambda f: f.path)
    return files, stats
