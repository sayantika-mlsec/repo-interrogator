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
    ReadableBytesTooLargeError,
    RepoTooLargeError,
    ShaMismatchError,
    TooManyFilesError,
    UnsupportedLanguageError,
)
from .fsutil import TreeMeasurement, measure_tree, rmtree_robust

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

    TWO BYTE BOUNDS, NOT ONE
    ------------------------
    A single total-bytes cap answers the wrong question. It rejected a pinned
    dev repository whose working tree was dominated by documentation GIFs --
    roughly 7 MB of readable content behind 65 MB of screenshots. The cap was
    measuring blob weight while claiming to measure code weight.

    So: ``max_total_bytes`` protects the machine from an enormous clone, and
    ``max_readable_bytes`` bounds what the agent actually has to contend with.
    They guard different resources and a repository can trip either alone.
    """

    max_total_bytes: int = 500 * 1024 * 1024
    """Machine guard. Generous by design.

    The largest tree in the pinned set is 77 MB, so this never fires on the
    current repositories. That is correct: its job is stopping a pathological
    clone, not filtering content. Tightening it toward the observed maximum
    would make it a second, cruder content filter and reintroduce exactly the
    conflation the readable cap replaced -- a repository would once again be
    rejected for the weight of its screenshots.
    """

    max_readable_bytes: int = 16 * 1024 * 1024
    """The bound that shapes the repository set.

    Set from measurement, not from judgement about what sounds reasonable.
    Readable content across the twelve pinned repositories runs from 0.3 MB to
    8.5 MB; this sits just under twice that maximum, so a repository half again
    larger than the largest in the set still enters, and one twice as large does
    not.

    It is the tight bound deliberately. Readable bytes are what the agent has to
    contend with, and that is the resource worth being strict about.
    """
    max_files: int = 10_000
    timeout_s: int = 300
    min_python_files: int = 5
    """Below this, structural extraction is not worth running."""

    degrade_on_unsupported: bool = True
    """If False, a non-Python repository raises instead of degrading."""

    enforce_caps: bool = True
    """When False, measure and report but never reject.

    Exists for one job: surveying the pinned set, where a repository that would
    be rejected is precisely the one whose numbers are needed. Turning it off
    logs a warning, because a results row produced with the caps disabled is not
    comparable to one produced with them on.
    """


@dataclass(frozen=True)
class ClonedRepo:
    """A verified working tree on disk, with the measurement that admitted it."""

    name: str
    url: str
    sha: str
    path: Path
    measurement: TreeMeasurement
    """Carried whole rather than unpacked into scalars, so a results row records
    the same object the gate decided on -- including whether it was complete."""

    mode: AnalysisMode

    @property
    def total_bytes(self) -> int:
        return self.measurement.total_bytes

    @property
    def readable_bytes(self) -> int:
        return self.measurement.readable_bytes

    @property
    def file_count(self) -> int:
        return self.measurement.file_count

    @property
    def python_file_count(self) -> int:
        return self.measurement.python_file_count


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


def _mb(value: int) -> str:
    return f"{value / 1e6:.1f} MB"


def _floor(measurement: TreeMeasurement, value: int) -> str:
    """Format a byte figure, saying 'at least' exactly when the walk was partial.

    A bounded walk stops at the first file past the bound, so its totals are
    lower bounds. Reporting one as a tree size is the error this helper exists
    to make impossible -- the phrasing is derived from the measurement rather
    than chosen by whoever writes the message.
    """
    return f"{'at least ' if measurement.partial else ''}{_mb(value)}"


def _enforce(measurement: TreeMeasurement, limits: CloneLimits, name: str) -> None:
    """Reject on any breached bound. Checked in the order the walk checks them."""
    if measurement.file_count > limits.max_files:
        raise TooManyFilesError(
            f"{name}: working tree exceeds {limits.max_files} files "
            f"(counted {'at least ' if measurement.partial else ''}"
            f"{measurement.file_count})"
        )

    if measurement.total_bytes > limits.max_total_bytes:
        raise RepoTooLargeError(
            f"{name}: working tree is {_floor(measurement, measurement.total_bytes)}, "
            f"over the {_mb(limits.max_total_bytes)} total-tree guard"
        )

    if measurement.readable_bytes > limits.max_readable_bytes:
        raise ReadableBytesTooLargeError(
            f"{name}: readable content is "
            f"{_floor(measurement, measurement.readable_bytes)}, over the "
            f"{_mb(limits.max_readable_bytes)} readable cap "
            f"(whole tree {_floor(measurement, measurement.total_bytes)}, "
            f"{measurement.readable_file_count} readable files)"
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

        if limits.enforce_caps:
            # Bounded: may stop early, figures are lower bounds, `breached` says so.
            measurement = measure_tree(
                dest,
                max_total_bytes=limits.max_total_bytes,
                max_readable_bytes=limits.max_readable_bytes,
                max_files=limits.max_files,
            )
            _enforce(measurement, limits, name)
        else:
            log.warning(
                "%s: cap enforcement disabled. Measuring only -- this clone is not "
                "comparable to one admitted under the caps.",
                name,
            )
            # Unbounded: walks to completion, every figure exact.
            measurement = measure_tree(dest)

        # Safe in both branches: an enforced walk that stopped early has already
        # raised, so the Python count reaching here is always a real total.
        mode = _classify(measurement.python_file_count, limits, name)

        log.info(
            "%s @ %s: %s tree / %s readable, %d files (%d readable, %d python), mode=%s",
            name,
            sha[:8],
            _mb(measurement.total_bytes),
            _mb(measurement.readable_bytes),
            measurement.file_count,
            measurement.readable_file_count,
            measurement.python_file_count,
            mode.value,
        )

        yield ClonedRepo(
            name=name,
            url=url,
            sha=sha,
            path=dest,
            measurement=measurement,
            mode=mode,
        )
    finally:
        rmtree_robust(dest)


def git_available() -> bool:
    return shutil.which("git") is not None