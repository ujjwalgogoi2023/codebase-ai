"""Tests for the repository parser (RAG pipeline stage 1).

The fixture builds a miniature repo on disk containing exactly the traps a real
codebase has: vendored dependencies, generated files, binaries, and junk.
"""

from pathlib import Path

import pytest

from app.services.repository_parser import (
    ParseStats,
    SourceFile,
    detect_language,
    parse_repository,
)


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    """A tiny repo that exercises every filtering branch."""

    def write(rel: str, text: str) -> None:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")

    # --- files we DO want ---
    write(
        "backend/auth.py",
        "def authenticate_user(username, password):\n    return True\n",
    )
    write("backend/models/user.py", "class User:\n    pass\n")
    write("frontend/src/App.jsx", "export default function App() { return null }\n")
    write("frontend/src/api.ts", "export const get = (u: string) => fetch(u)\n")
    write("src/main/java/Service.java", "class Service { void run() {} }\n")
    write("README.md", "# Demo project\n\nAuth lives in backend/auth.py\n")

    # --- ignored directories (must never be descended into) ---
    write("node_modules/left-pad/index.js", "module.exports = () => {}\n")
    write("node_modules/react/deep/nested/thing.js", "// vendored\n")
    write(".git/hooks/pre-commit.py", "# git internals\n")
    write("venv/lib/site.py", "# virtualenv\n")
    write("backend/__pycache__/auth.py", "# compiled artifact\n")
    write("frontend/dist/bundle.js", "// built output\n")
    write("target/Generated.java", "// maven output\n")

    # --- ignored filenames / suffixes ---
    write("frontend/src/vendor.min.js", "!function(){}()\n")
    write("frontend/src/types.d.ts", "declare const x: number\n")
    write("package-lock.json", '{"lockfileVersion": 3}\n')

    # --- unsupported extensions ---
    write("assets/logo.svg", "<svg></svg>\n")
    write("config.yaml", "key: value\n")
    write("Makefile", "all:\n\techo hi\n")

    # --- empty / whitespace-only ---
    write("backend/empty.py", "")
    write("backend/blank.py", "   \n\n  \n")

    # --- binary file wearing a .py extension ---
    (tmp_path / "backend" / "blob.py").write_bytes(b"\x89PNG\x00\x1a\n\x00binary")

    return tmp_path


def test_keeps_only_supported_source_files(fake_repo: Path):
    files, _ = parse_repository(fake_repo)
    paths = [f.path for f in files]

    assert paths == [
        "README.md",
        "backend/auth.py",
        "backend/models/user.py",
        "frontend/src/App.jsx",
        "frontend/src/api.ts",
        "src/main/java/Service.java",
    ]


def test_prunes_ignored_directories(fake_repo: Path):
    files, stats = parse_repository(fake_repo)
    joined = " ".join(f.path for f in files)

    for junk in ("node_modules", ".git", "venv", "__pycache__", "dist", "target"):
        assert junk not in joined, f"{junk} leaked into the index"
    assert stats.dirs_pruned >= 6


def test_skips_generated_minified_and_lockfiles(fake_repo: Path):
    files, stats = parse_repository(fake_repo)
    names = {Path(f.path).name for f in files}

    assert "vendor.min.js" not in names
    assert "types.d.ts" not in names
    assert "package-lock.json" not in names
    assert stats.skipped_ignored_name == 3


def test_skips_binary_and_empty_files(fake_repo: Path):
    files, stats = parse_repository(fake_repo)
    names = {Path(f.path).name for f in files}

    assert "blob.py" not in names, "binary file was indexed"
    assert "empty.py" not in names
    assert "blank.py" not in names
    assert stats.skipped_binary == 1
    assert stats.skipped_empty == 2


def test_respects_max_file_bytes(fake_repo: Path):
    (fake_repo / "backend" / "huge.py").write_text("x = 1\n" * 5000, encoding="utf-8")

    files, stats = parse_repository(fake_repo, max_file_bytes=100)
    assert "backend/huge.py" not in [f.path for f in files]
    assert stats.skipped_too_large == 1


def test_language_detection_and_metadata(fake_repo: Path):
    files, stats = parse_repository(fake_repo)
    by_path = {f.path: f for f in files}

    assert by_path["backend/auth.py"].language == "python"
    assert by_path["frontend/src/App.jsx"].language == "javascript"
    assert by_path["frontend/src/api.ts"].language == "typescript"
    assert by_path["src/main/java/Service.java"].language == "java"
    assert by_path["README.md"].language == "markdown"

    auth = by_path["backend/auth.py"]
    assert auth.num_lines == 2
    assert "authenticate_user" in auth.content
    assert auth.size_bytes > 0

    assert stats.kept == 6
    assert stats.languages == {
        "java": 1,
        "javascript": 1,
        "markdown": 1,
        "python": 2,
        "typescript": 1,
    }


def test_paths_are_posix_and_relative(fake_repo: Path):
    files, _ = parse_repository(fake_repo)
    for f in files:
        assert "\\" not in f.path, "Windows separator leaked into a citation path"
        assert not Path(f.path).is_absolute()


def test_output_is_deterministic(fake_repo: Path):
    first, _ = parse_repository(fake_repo)
    second, _ = parse_repository(fake_repo)
    assert [f.path for f in first] == [f.path for f in second]


def test_detect_language_helper():
    assert detect_language("a.py") == "python"
    assert detect_language("a.PY") == "python"
    assert detect_language("a.tsx") == "typescript"
    assert detect_language("a.rb") is None


def test_rejects_non_directory(tmp_path: Path):
    f = tmp_path / "file.py"
    f.write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        parse_repository(f)


def test_virtualenv_is_detected_by_marker_not_by_name(tmp_path: Path):
    """`venv/` and `.venv/` are conventions, not rules."""
    for env_name in ("myenv", "env311", "toolchain"):
        env = tmp_path / env_name
        (env / "Lib").mkdir(parents=True)
        (env / "pyvenv.cfg").write_text("home = /usr\n", encoding="utf-8")
        (env / "Lib" / "installed_package.py").write_text("x = 1\n", encoding="utf-8")

    (tmp_path / "app.py").write_text("def go():\n    return 1\n", encoding="utf-8")

    files, stats = parse_repository(tmp_path)

    assert [f.path for f in files] == ["app.py"]
    assert stats.dirs_pruned == 3


def test_directory_named_like_a_venv_without_the_marker_is_still_indexed(tmp_path: Path):
    """Don't over-prune: a real source folder that happens to be called
    'environment' must not vanish just because the name looks suspicious."""
    src = tmp_path / "environment"
    src.mkdir()
    (src / "climate.py").write_text("def temperature():\n    return 21\n", encoding="utf-8")

    files, _ = parse_repository(tmp_path)

    assert [f.path for f in files] == ["environment/climate.py"]
