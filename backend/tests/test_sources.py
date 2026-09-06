"""Tests for source acquisition (ZIP upload, git clone).

This is the only code in the project that handles input from outside, so these
are mostly adversarial: an archive that tries to write outside its destination,
a clone URL that tries to make the server fetch something internal.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from app.services.sources import (
    MAX_ARCHIVE_MEMBERS,
    SourceError,
    _is_within,
    extract_zip,
    parse_git_url,
    repository_name_from_url,
)


def make_zip(path: Path, entries: dict[str, str]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return path


# ---------------------------------------------------------------------------
# Zip Slip and friends
# ---------------------------------------------------------------------------

def test_extracts_a_normal_archive(tmp_path: Path):
    archive = make_zip(
        tmp_path / "repo.zip",
        {"src/app.py": "print(1)\n", "README.md": "# hi\n"},
    )
    root = extract_zip(archive, tmp_path / "out")

    assert (root / "src" / "app.py").read_text() == "print(1)\n"
    assert (root / "README.md").exists()


def test_relative_traversal_is_rejected(tmp_path: Path):
    """The classic Zip Slip payload."""
    archive = make_zip(
        tmp_path / "evil.zip", {"../../escaped.py": "owned = True\n"}
    )

    with pytest.raises(SourceError, match="escapes the destination"):
        extract_zip(archive, tmp_path / "out")

    assert not (tmp_path.parent / "escaped.py").exists()
    assert not (tmp_path / "escaped.py").exists()


def test_deep_traversal_is_rejected(tmp_path: Path):
    archive = make_zip(
        tmp_path / "evil.zip", {"a/b/../../../../../../tmp/x.py": "x = 1\n"}
    )
    with pytest.raises(SourceError, match="escapes the destination"):
        extract_zip(archive, tmp_path / "out")


def test_absolute_path_is_rejected(tmp_path: Path):
    archive = make_zip(tmp_path / "evil.zip", {"/etc/passwd": "root\n"})

    with pytest.raises(SourceError, match="absolute path"):
        extract_zip(archive, tmp_path / "out")


def test_windows_drive_path_is_rejected(tmp_path: Path):
    archive = make_zip(tmp_path / "evil.zip", {"C:/Windows/System32/x.py": "x\n"})

    with pytest.raises(SourceError, match="absolute path"):
        extract_zip(archive, tmp_path / "out")


def test_zip_bomb_is_rejected_before_extraction(tmp_path: Path):
    """Size is checked from the header, not discovered by filling the disk."""
    archive_path = tmp_path / "bomb.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        # Highly compressible: tiny on disk, enormous when expanded.
        archive.writestr("big.py", "A" * (600 * 1024 * 1024))

    with pytest.raises(SourceError, match="over the"):
        extract_zip(archive_path, tmp_path / "out")


def test_too_many_members_is_rejected(tmp_path: Path):
    archive_path = tmp_path / "many.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for i in range(MAX_ARCHIVE_MEMBERS + 1):
            archive.writestr(f"f{i}.txt", "")

    with pytest.raises(SourceError, match="entries, over the"):
        extract_zip(archive_path, tmp_path / "out")


def test_non_zip_upload_is_rejected(tmp_path: Path):
    fake = tmp_path / "not.zip"
    fake.write_text("this is not a zip file", encoding="utf-8")

    with pytest.raises(SourceError, match="not a valid ZIP"):
        extract_zip(fake, tmp_path / "out")


def test_single_root_folder_is_collapsed(tmp_path: Path):
    """GitHub's Download ZIP wraps everything in reponame-main/."""
    archive = make_zip(
        tmp_path / "gh.zip",
        {"myrepo-main/src/app.py": "x = 1\n", "myrepo-main/README.md": "# hi\n"},
    )
    root = extract_zip(archive, tmp_path / "out")

    assert root.name == "myrepo-main"
    assert (root / "src" / "app.py").exists(), "citations would carry a useless prefix"


def test_multiple_top_level_entries_are_not_collapsed(tmp_path: Path):
    archive = make_zip(
        tmp_path / "flat.zip", {"src/app.py": "x\n", "README.md": "# hi\n"}
    )
    root = extract_zip(archive, tmp_path / "out")

    assert root.name == "out"


def test_is_within_rejects_sibling_prefix(tmp_path: Path):
    """/data/repo2 must not count as inside /data/repo."""
    base = tmp_path / "repo"
    base.mkdir()
    sibling = tmp_path / "repo2"
    sibling.mkdir()

    assert _is_within(base, base / "a" / "b")
    assert not _is_within(base, sibling)


# ---------------------------------------------------------------------------
# Clone URL validation: the SSRF boundary
# ---------------------------------------------------------------------------

def test_accepts_normal_github_urls():
    assert parse_git_url("https://github.com/psf/requests") == (
        "github.com", "psf", "requests"
    )
    assert parse_git_url("https://github.com/psf/requests.git")[2] == "requests"
    assert parse_git_url("https://github.com/psf/requests/")[2] == "requests"
    assert parse_git_url("https://gitlab.com/group/proj")[0] == "gitlab.com"


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ssh://git@internal-host/repo",
        "git://github.com/psf/requests",
        "http://github.com/psf/requests",              # plain http
        "https://169.254.169.254/latest/meta-data",    # cloud metadata
        "https://localhost:8080/x/y",
        "",
        "not a url",
    ],
)
def test_rejects_unsafe_or_malformed_urls(url):
    with pytest.raises(SourceError):
        parse_git_url(url)


def test_rejects_hosts_off_the_allowlist():
    with pytest.raises(SourceError, match="not allowed"):
        parse_git_url("https://evil.example.com/owner/repo")


def test_repository_name_from_url():
    assert repository_name_from_url("https://github.com/psf/requests") == "psf/requests"
