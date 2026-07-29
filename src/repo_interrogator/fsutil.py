"""Filesystem primitives used by the workspace layer.

Three jobs: delete a git clone on Windows without failing, measure a tree
cheaply, and refuse to resolve a path outside a root directory.
"""

from __future__ import annotations

import os
import shutil
import stat
from collections.abc import Iterator
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


def measure_tree(root: Path, *, max_bytes: int, max_files: int) -> tuple[int, int]:
    """Return ``(total_bytes, file_count)``, stopping early once either cap is passed.

    Early exit matters: on a pathological repository, walking to completion to
    discover it is 40x over the limit costs real time for no extra information.
    The returned values are therefore a lower bound once a cap is exceeded,
    which is all the caller needs in order to reject.
    """
    total = 0
    count = 0
    for f in iter_files(root):
        try:
            total += f.stat().st_size
        except OSError:
            # Broken symlink or a file that vanished mid-walk. Count it, skip
            # its size — never silently abort the measurement.
            pass
        count += 1
        if total > max_bytes or count > max_files:
            break
    return total, count


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