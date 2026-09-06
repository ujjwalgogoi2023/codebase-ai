"""Getting a codebase onto disk: ZIP uploads and git clones.

This is the only module that handles untrusted input from outside the system,
so the checks here matter more than anywhere else in the project.

ZIP EXTRACTION IS NOT SAFE BY DEFAULT
------------------------------------
`ZipFile.extractall()` writes wherever the archive says. An entry named
`../../../../etc/passwd` or `C:\\Windows\\System32\\x` escapes the destination
entirely. That is Zip Slip (CVE-2018-1002200 and a long tail of others), and
every member is checked against the resolved destination before anything is
written.

Two more archive hazards are handled:
  - Zip bombs: a few KB that expand to gigabytes, so total uncompressed size is
    capped before extraction rather than discovered during it.
  - Symlink entries, which can point outside the tree and are simply skipped.

GIT CLONE IS NOT SAFE BY DEFAULT EITHER
---------------------------------------
An arbitrary clone URL is a server-side request the user chose. `file:///etc`,
`ssh://internal-host/...` and `http://169.254.169.254/...` are all valid to git
and none of them should be reachable. So the scheme must be https and the host
must be on an allowlist, the command is built as an argument list (never a
shell string), and the clone is depth-1 with a timeout.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path

from app.config import PROJECT_ROOT

# Where acquired codebases live. Gitignored: this is scratch space, and the
# authoritative copy of the code is whatever the user uploaded.
REPO_STORAGE_ROOT = PROJECT_ROOT / "data" / "repos"

MAX_UNCOMPRESSED_BYTES = 500 * 1024 * 1024   # 500 MB
MAX_ARCHIVE_MEMBERS = 50_000
CLONE_TIMEOUT_SECONDS = 300

ALLOWED_GIT_HOSTS = frozenset(
    {"github.com", "www.github.com", "gitlab.com", "www.gitlab.com", "bitbucket.org"}
)

_GIT_URL = re.compile(
    r"^https://(?P<host>[A-Za-z0-9.\-]+)/(?P<owner>[\w.\-]+)/(?P<repo>[\w.\-]+?)(?:\.git)?/?$"
)


class SourceError(ValueError):
    """Raised when a source cannot be safely acquired."""


@dataclass
class AcquiredSource:
    """A codebase now present on local disk."""

    path: Path
    source_type: str          # zip | github | local
    name: str
    url: str | None = None


def storage_path_for(repository_id: int) -> Path:
    return REPO_STORAGE_ROOT / str(repository_id)


def cleanup_storage(repository_id: int) -> None:
    """Remove a repository's extracted/cloned files. Never raises."""
    shutil.rmtree(storage_path_for(repository_id), ignore_errors=True)


# ---------------------------------------------------------------------------
# ZIP
# ---------------------------------------------------------------------------

def _is_within(base: Path, target: Path) -> bool:
    """True when `target` resolves to somewhere inside `base`.

    Uses resolved absolute paths so `a/../../b` cannot slip through, and
    is_relative_to rather than string prefixes so `/data/repo2` is not
    mistaken for a child of `/data/repo`.
    """
    try:
        return target.resolve().is_relative_to(base.resolve())
    except (OSError, ValueError):
        return False


def extract_zip(archive_path: str | Path, destination: str | Path) -> Path:
    """Safely extract a ZIP archive, returning the directory to index."""
    archive_path = Path(archive_path)
    destination = Path(destination)

    if not zipfile.is_zipfile(archive_path):
        raise SourceError("Uploaded file is not a valid ZIP archive")

    destination.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()

        if len(members) > MAX_ARCHIVE_MEMBERS:
            raise SourceError(
                f"Archive has {len(members)} entries, over the "
                f"{MAX_ARCHIVE_MEMBERS} limit"
            )

        total = sum(m.file_size for m in members)
        if total > MAX_UNCOMPRESSED_BYTES:
            raise SourceError(
                f"Archive expands to {total // (1024 * 1024)} MB, over the "
                f"{MAX_UNCOMPRESSED_BYTES // (1024 * 1024)} MB limit"
            )

        for member in members:
            # Absolute paths and drive letters are rejected outright; anything
            # else is checked after resolution.
            name = member.filename
            if name.startswith(("/", "\\")) or (len(name) > 1 and name[1] == ":"):
                raise SourceError(f"Archive contains an absolute path: {name!r}")

            target = destination / name
            if not _is_within(destination, target):
                raise SourceError(f"Archive entry escapes the destination: {name!r}")

            # A symlink entry can point anywhere; skip rather than follow.
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                continue

            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, open(target, "wb") as handle:
                shutil.copyfileobj(source, handle, length=64 * 1024)

    return _collapse_single_root(destination)


def _collapse_single_root(destination: Path) -> Path:
    """Descend into a lone top-level folder.

    GitHub's "Download ZIP" wraps everything in `reponame-main/`. Indexing the
    wrapper would make every citation read `reponame-main/backend/auth.py`,
    which is noise in every path the user ever sees.
    """
    entries = [e for e in destination.iterdir() if not e.name.startswith("__MACOSX")]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return destination


# ---------------------------------------------------------------------------
# git
# ---------------------------------------------------------------------------

def parse_git_url(url: str) -> tuple[str, str, str]:
    """Validate a clone URL and return (host, owner, repo).

    Rejects anything that is not https on an allowlisted host. This is the
    SSRF boundary: git will happily accept file://, ssh:// and plain http to an
    internal address, and none of those are things a user should be able to
    make the server fetch.
    """
    url = (url or "").strip()
    match = _GIT_URL.match(url)
    if not match:
        raise SourceError(
            "Only https URLs of the form "
            "https://github.com/owner/repo are supported"
        )

    host = match.group("host").lower()
    if host not in ALLOWED_GIT_HOSTS:
        raise SourceError(
            f"Host {host!r} is not allowed. Allowed: "
            f"{', '.join(sorted(ALLOWED_GIT_HOSTS))}"
        )

    return host, match.group("owner"), match.group("repo")


def clone_repository(url: str, destination: str | Path) -> Path:
    """Shallow-clone a public repository."""
    _, owner, repo = parse_git_url(url)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    target = destination / repo
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)

    try:
        result = subprocess.run(
            # Argument list, never a shell string, so nothing in the URL can be
            # interpreted as a command. --depth 1 skips history we never read.
            ["git", "clone", "--depth", "1", "--quiet", url, str(target)],
            capture_output=True,
            text=True,
            timeout=CLONE_TIMEOUT_SECONDS,
            check=False,
            # Stops git from opening a credential prompt and hanging forever on
            # a private or nonexistent repository.
            env={"GIT_TERMINAL_PROMPT": "0", "PATH": _system_path()},
        )
    except FileNotFoundError as exc:
        raise SourceError("git is not installed or not on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        shutil.rmtree(target, ignore_errors=True)
        raise SourceError(
            f"Clone timed out after {CLONE_TIMEOUT_SECONDS}s"
        ) from exc

    if result.returncode != 0:
        shutil.rmtree(target, ignore_errors=True)
        detail = (result.stderr or "").strip().splitlines()
        message = detail[-1] if detail else f"git exited {result.returncode}"
        raise SourceError(f"Clone failed: {message}")

    # The .git directory is pure overhead here; the parser ignores it anyway.
    shutil.rmtree(target / ".git", ignore_errors=True)
    return target


def _system_path() -> str:
    import os

    return os.environ.get("PATH", "")


def repository_name_from_url(url: str) -> str:
    _, owner, repo = parse_git_url(url)
    return f"{owner}/{repo}"
