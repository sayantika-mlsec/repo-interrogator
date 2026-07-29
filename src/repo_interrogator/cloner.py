"""Fetch a repository at an exact commit, under hard limits, and clean up after.

This is the only code in the project that writes to disk from an external
source. Everything downstream assumes the working tree it is handed is: pinned
to a known commit, bounded in size, and deleted when the caller is done.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .errors import (
    CloneFailedError,
    CloneTimeoutError,
    GitNotAvailableError,
    InvalidShaError,
    RepoTooLargeError,
    ShaMismatchError,
    TooManyFilesError,
    UnsupportedLanguageError,
)
from .fsutil import iter_files, measure_tree, rmtree_robust

log = logging.getLogger(__name__)

_FULL_SHA_LEN = 40


class AnalysisMode(str, Enum):
    """How much of the toolchain a repository can support."""

    FULL = "full"
    """Enough Python to justify structural symbol extraction."""

    TEXT_ONLY = "text_only"
    """No usable Python. Text search and file reading only, no symbol index."""


@dataclass(frozen=True)
class CloneLimits:
    """Hard ceilings. Exceeding any of them is a rejection, never a truncation.

    A truncated repository would still produce questions, and those questions
    would look fine. That is the dangerous failure: a number in the results
    table computed against half a codebase, with nothing in the output saying
    so. Rejection is loud; truncation is silent.
    """

    max_bytes: int = 50 * 1024 * 1024
    max_files: int = 10_000
    timeout_s: int = 300
    min_python_files: int = 5
    """Below this, structural extraction is not worth running."""

    degrade_on_unsupported: bool = True
    """If False, a non-Python repository raises instead of degrading."""


@dataclass(frozen=True)
class ClonedRepo:
    """A verified working tree on disk."""

    name: str
    url: str
    sha: str
    path: Path
    size_bytes: int
    file_count: int
    python_file_count: int
    mode: AnalysisMode


def _run_git(args: list[str], *, cwd: Path | None, timeout_s: int) -> str:
    """Run git, or raise. Never returns on failure."""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except FileNotFoundError as exc:
        raise GitNotAvailableError("git executable not found on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise CloneTimeoutError(
            f"git {' '.join(args)} exceeded {timeout_s}s and was killed"
        ) from exc

    if proc.returncode != 0:
        raise CloneFailedError(
            f"git {' '.join(args)} failed (exit {proc.returncode}): "
            f"{proc.stderr.strip() or '<no stderr>'}"
        )
    return proc.stdout.strip()


def _fetch_pinned(url: str, sha: str, dest: Path, timeout_s: int) -> None:
    """Fetch exactly one commit, with no history.

    `git clone --depth 1` cannot do this. A shallow clone fetches the tip of the
    default branch, and the pinned commit is usually not that tip -- so a
    subsequent `git checkout <sha>` fails with "reference is not a tree",
    because the object was never downloaded.

    The working sequence asks the server for the commit by name instead:

        init -> add remote -> fetch --depth 1 <sha> -> checkout FETCH_HEAD

    This relies on the server permitting fetch-by-SHA (`allowAnySHA1InWant`).
    GitHub enables it. A server that does not will fail here, loudly, rather
    than silently handing back a different commit.
    """
    dest.mkdir(parents=True, exist_ok=True)
    _run_git(["init", "--quiet"], cwd=dest, timeout_s=30)
    _run_git(["remote", "add", "origin", url], cwd=dest, timeout_s=30)
    _run_git(
        ["fetch", "--depth", "1", "--no-tags", "--quiet", "origin", sha],
        cwd=dest,
        timeout_s=timeout_s,
    )
    _run_git(
        ["-c", "advice.detachedHead=false", "checkout", "--quiet", "--detach", "FETCH_HEAD"],
        cwd=dest,
        timeout_s=60,
    )


def _classify(python_files: int, limits: CloneLimits, name: str) -> AnalysisMode:
    if python_files >= limits.min_python_files:
        return AnalysisMode.FULL
    if limits.degrade_on_unsupported:
        log.warning(
            "%s: %d Python files (< %d). Degrading to text-only: no symbol index.",
            name,
            python_files,
            limits.min_python_files,
        )
        return AnalysisMode.TEXT_ONLY
    raise UnsupportedLanguageError(
        f"{name}: only {python_files} Python files, below minimum {limits.min_python_files}"
    )


@contextmanager
def cloned_repo(
    name: str,
    url: str,
    sha: str,
    *,
    limits: CloneLimits | None = None,
    workspace_root: Path | None = None,
) -> Generator[ClonedRepo, None, None]:
    """Clone at a pinned SHA, enforce limits, yield the tree, always clean up.

    Used as a context manager so that cleanup is tied to scope rather than to
    the caller remembering. It runs on the exception path too -- which is the
    path that matters, since a rejected oversized repo is exactly the case where
    a leaked clone hurts most.
    """
    limits = limits or CloneLimits()

    if len(sha) != _FULL_SHA_LEN or not all(c in "0123456789abcdef" for c in sha.lower()):
       raise InvalidShaError(
            f"{name}: sha must be a full 40-character hex commit id, got {sha!r}. "
            "Abbreviated SHAs are ambiguous and are rejected."
        )

    workspace_root = workspace_root or Path(tempfile.gettempdir()) / "repo-interrogator"
    workspace_root.mkdir(parents=True, exist_ok=True)
    dest = Path(tempfile.mkdtemp(prefix=f"{name}-", dir=workspace_root))

    try:
        _fetch_pinned(url, sha, dest, limits.timeout_s)

        actual = _run_git(["rev-parse", "HEAD"], cwd=dest, timeout_s=30)
        if actual.lower() != sha.lower():
            raise ShaMismatchError(
                f"{name}: requested {sha}, checked out {actual}. The pin is not holding."
            )

        size, count = measure_tree(
            dest, max_bytes=limits.max_bytes, max_files=limits.max_files
        )
        if size > limits.max_bytes:
            raise RepoTooLargeError(
                f"{name}: working tree exceeds {limits.max_bytes / 1e6:.0f} MB "
                f"(measured at least {size / 1e6:.1f} MB)"
            )
        if count > limits.max_files:
            raise TooManyFilesError(
                f"{name}: working tree exceeds {limits.max_files} files "
                f"(counted at least {count})"
            )

        py_count = sum(1 for f in iter_files(dest) if f.suffix == ".py")
        mode = _classify(py_count, limits, name)

        log.info(
            "%s @ %s: %.1f MB, %d files (%d python), mode=%s",
            name, sha[:8], size / 1e6, count, py_count, mode.value,
        )

        yield ClonedRepo(
            name=name, url=url, sha=sha, path=dest,
            size_bytes=size, file_count=count,
            python_file_count=py_count, mode=mode,
        )
    finally:
        rmtree_robust(dest)


def git_available() -> bool:
    return shutil.which("git") is not None