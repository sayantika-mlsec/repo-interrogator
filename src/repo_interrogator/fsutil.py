"""Filesystem primitives used by the workspace layer.

Four jobs: delete a git clone on Windows without failing, decide whether a file
holds text, measure a tree, and refuse to resolve a path outside a root
directory.

MEASUREMENT
-----------
A working tree has two sizes that matter, and they are not the same number.

The first is what lands on disk. That bounds clone time, temp-space use, and
how badly a pathological repository can hurt the machine.

The second is what the agent can actually put in front of a model: files that
decode as text. A repository of documentation videos has an enormous first
number and a small second one. Measuring only the first rejects repositories
for their screenshots.

So ``measure_tree`` returns both, from one traversal.
"""

from __future__ import annotations

import os
import shutil
import stat
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

from .errors import PathEscapeError

# Directories never walked when measuring or listing a repository.
# `.git` is excluded because the agent never reads it: what matters is the size
# of the working tree the agent can actually see, not the object store.
SKIP_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        ".venv",
        "venv",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        "dist",
        "build",
        ".idea",
        ".vscode",
    }
)


def _force_writable_and_retry(func, path: str, exc: BaseException) -> None:
    """Clear the read-only bit and retry a failed deletion.

    Git stores objects under ``.git/objects`` as read-only, because they are
    content-addressed and must never be modified in place. On Windows, the OS
    refuses to delete a read-only file at all, so ``shutil.rmtree`` raises
    ``PermissionError`` partway through and leaves a half-deleted directory.

    POSIX does not have this problem: deletion permission comes from the parent
    directory, not the file. So this handler only ever fires on Windows.

    Anything that is not a permission problem is re-raised untouched. Retrying
    a vanished file or a locked handle would replace the real error with a
    ``chmod`` failure two frames away from the cause.
    """
    if not isinstance(exc, PermissionError):
        raise exc
    os.chmod(path, stat.S_IWRITE)
    func(path)


def rmtree_robust(path: Path) -> None:
    """Delete a directory tree, tolerating read-only files. Idempotent."""
    if not path.exists():
        return

    shutil.rmtree(path, onexc=_force_writable_and_retry)


def iter_files(root: Path) -> Iterator[Path]:
    """Yield every file in the working tree, skipping noise directories.

    Symlinks are not followed. A symlinked directory pointing at ``/`` would
    otherwise make this walk the entire filesystem.
    """
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        base = Path(dirpath)
        for name in filenames:
            p = base / name
            if not p.is_symlink():
                yield p


def iter_entries(base: Path) -> Iterator[tuple[Path, bool]]:
    """Yield the immediate children of one directory, with a directory flag.

    The shallow counterpart to ``iter_files``, and it has to agree with it on
    what a repository contains: same skip set, same refusal to follow symlinks.
    A listing that showed a directory ``iter_files`` never walks would send the
    model somewhere the other tools cannot go.

    Sorted here rather than by the caller. The order a filesystem returns
    entries in is not stable across platforms, and an unstable listing means two
    runs of the same pinned commit are shown different first files.
    """
    try:
        entries = sorted(base.iterdir())
    except OSError:
        return

    for path in entries:
        if path.is_symlink():
            continue
        try:
            is_dir = path.is_dir()
        except OSError:
            continue
        if is_dir and path.name in SKIP_DIRS:
            continue
        yield path, is_dir

# --- readability -----------------------------------------------------------

SNIFF_BYTES = 8192
"""How much of a file is inspected to decide whether it is text."""

_NON_TEXT_RATIO = 0.30

_TEXT_BYTES = frozenset(bytes(range(32, 127)) + b"\n\r\t\f\b\x1b")


def is_probably_binary(path: Path, *, sniff: int = SNIFF_BYTES) -> bool:
    """Decide whether a file holds binary content, by looking at the content.

    Deliberately not an extension allowlist. An allowlist has to be maintained,
    and is wrong the first time a repository stores source under an unfamiliar
    suffix or a data file under ``.txt``.

    Three signals, in order of reliability:

    A NUL byte in the first chunk. This is git's own heuristic, and it settles
    every real image, archive and compiled artifact -- PNG carries one at byte 8,
    JPEG at byte 4.

    Failure to decode as UTF-8. A truncated multi-byte character at the sniff
    boundary is not evidence of anything, so a failure within the last three
    bytes is ignored; that is an artifact of where the read stopped.

    A high proportion of bytes outside the printable range. Only consulted when
    the decode already failed, so a UTF-8 file full of CJK text -- which would
    trip a naive ratio test badly -- never reaches it.

    An unreadable file counts as binary. The caller wanted to know whether it can
    be put in front of a model, and the answer is no either way.
    """
    try:
        with path.open("rb") as fh:
            chunk = fh.read(sniff)
    except OSError:
        return True

    if not chunk:
        return False

    if b"\x00" in chunk:
        return True

    try:
        chunk.decode("utf-8")
        return False
    except UnicodeDecodeError as exc:
        if exc.start >= len(chunk) - 3:
            return False

    non_text = sum(1 for b in chunk if b not in _TEXT_BYTES)
    return non_text / len(chunk) > _NON_TEXT_RATIO


# --- tree measurement ------------------------------------------------------


@dataclass(frozen=True)
class TreeMeasurement:
    """What one walk of a working tree found.

    ``breached`` is the whole point of this being a type rather than a tuple.
    A bounded walk stops at the first file past a bound, so every figure it
    returns is a lower bound rather than a total. That distinction cannot live
    in a docstring: a caller formatting an error message will read the number
    and call it the tree size. Here the object itself says whether the walk
    finished, and the rejection path is required to consult it.
    """

    total_bytes: int
    """Every file in the working tree, text or not."""

    readable_bytes: int
    """Only files that decode as text -- what the agent can actually contend with."""

    file_count: int
    readable_file_count: int
    python_file_count: int

    breached: str | None = None
    """Which bound stopped the walk: ``file_count``, ``total_bytes``,
    ``readable_bytes``, or ``None`` if the walk ran to completion."""

    @property
    def partial(self) -> bool:
        """True when the figures are lower bounds rather than totals."""
        return self.breached is not None

    @property
    def binary_bytes(self) -> int:
        return self.total_bytes - self.readable_bytes

    def to_dict(self) -> dict[str, object]:
        """Serialise onto a results row, ``partial`` included explicitly.

        A stored measurement that does not record whether it was complete is
        worse than no measurement, because it looks authoritative.
        """
        return asdict(self) | {"partial": self.partial}


def measure_tree(
    root: Path,
    *,
    max_total_bytes: int | None = None,
    max_readable_bytes: int | None = None,
    max_files: int | None = None,
) -> TreeMeasurement:
    """Walk a working tree once and return both byte totals plus three counts.

    **With no bounds passed, the walk runs to completion and every figure is
    exact.** That is the mode a survey wants. It is the same code path the gate
    uses, so a surveyed number and an enforced number can never come from two
    subtly different traversals.

    **With any bound passed, the walk may stop early** -- at the first file that
    carries a running total past the bound. The saving is real: discovering that
    a repository is forty times over the limit does not require walking it. The
    cost is that the figures are then lower bounds, which is why the return
    carries ``breached``.

    Readability is decided by content, via ``is_probably_binary``, which reads
    the first 8 KB of each file. That is one extra open per file and it is what
    the second bound is buying. A file whose ``stat`` fails -- a broken symlink,
    or one that vanished mid-walk -- is counted, contributes no bytes, and is
    never sniffed, so it lands on the unreadable side. The measurement is never
    silently abandoned for a single bad file.

    Bounds are checked in order: file count, total bytes, readable bytes. The
    first one crossed is the one reported.
    """
    total_bytes = 0
    readable_bytes = 0
    file_count = 0
    readable_file_count = 0
    python_file_count = 0
    breached: str | None = None

    for path in iter_files(root):
        file_count += 1
        if path.suffix == ".py":
            python_file_count += 1

        try:
            size = path.stat().st_size
        except OSError:
            size = None

        if size is not None:
            total_bytes += size
            if not is_probably_binary(path):
                readable_bytes += size
                readable_file_count += 1

        if max_files is not None and file_count > max_files:
            breached = "file_count"
            break
        if max_total_bytes is not None and total_bytes > max_total_bytes:
            breached = "total_bytes"
            break
        if max_readable_bytes is not None and readable_bytes > max_readable_bytes:
            breached = "readable_bytes"
            break

    return TreeMeasurement(
        total_bytes=total_bytes,
        readable_bytes=readable_bytes,
        file_count=file_count,
        readable_file_count=readable_file_count,
        python_file_count=python_file_count,
        breached=breached,
    )


# --- containment -----------------------------------------------------------


def resolve_within(root: Path, candidate: str | Path) -> Path:
    """Resolve ``candidate`` relative to ``root``, or raise if it escapes.

    This is the single containment check for the whole project. Every tool that
    accepts a path from the model routes through it.

    The check is done *after* resolution, not before, because string inspection
    is not sufficient. A path can contain no ``..`` at all and still escape:

        docs/link  ->  symlink to  /etc

    Only the resolved absolute path tells the truth about where a path points.
    """
    root = root.resolve()
    target = (root / Path(candidate)).resolve()
    if target != root and root not in target.parents:
        raise PathEscapeError(
            f"path {candidate!r} resolves to {target}, outside repository root {root}"
        )
    return target