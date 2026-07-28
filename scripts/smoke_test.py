"""Smoke test for the workspace safety layer.

Run once after installing, to confirm every guard actually fires on this
machine:

    uv run python scripts/smoke_clone.py

The Windows read-only cleanup path in particular cannot be verified on Linux,
because the failure it handles does not occur there. This script is the only
thing that proves it works before real runs depend on it.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from repo_interrogator.cloner import CloneLimits, cloned_repo, git_available
from repo_interrogator.errors import (
    CloneFailedError,
    PathEscapeError,
    RepoTooLargeError,
    TooManyFilesError,
    UnsupportedLanguageError,
)
from repo_interrogator.fsutil import resolve_within

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

# A small pinned repo. Any entry from repos.yaml would do; this one is chosen
# because it clones in a couple of seconds.
NAME = "structlog"
URL = "https://github.com/hynek/structlog"
SHA = "6651ae67d5790923bdaf110e08fb891b3ac706d8"

passed = 0
failed = 0


def check(label: str, condition: bool) -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {label}")
    else:
        failed += 1
        print(f"  FAIL  {label}")


def expect_raises(label: str, exc_type: type[BaseException], fn) -> None:
    global passed, failed
    try:
        fn()
    except exc_type as e:
        passed += 1
        print(f"  PASS  {label}  ({type(e).__name__})")
    except Exception as e:  # noqa: BLE001 - the point is to report the wrong type
        failed += 1
        print(f"  FAIL  {label}  (expected {exc_type.__name__}, got {type(e).__name__}: {e})")
    else:
        failed += 1
        print(f"  FAIL  {label}  (no exception raised)")


def main() -> int:
    if not git_available():
        print("git not found on PATH")
        return 2

    print("\n[1] pinned fetch, measurement, containment, cleanup")
    with cloned_repo(NAME, URL, SHA) as repo:
        root = repo.path
        check("sha matches request", repo.sha == SHA)
        check("tree exists on disk", (root / "pyproject.toml").is_file())
        check("size measured", repo.size_bytes > 0)
        check("python files found", repo.python_file_count > 0)
        check("mode is full", repo.mode.value == "full")

        expect_raises(
            "parent traversal blocked",
            PathEscapeError,
            lambda: resolve_within(root, "../../../etc/passwd"),
        )
        expect_raises(
            "absolute path blocked",
            PathEscapeError,
            lambda: resolve_within(root, "C:\\Windows\\System32" if sys.platform == "win32" else "/etc/passwd"),
        )
        check(
            "legitimate path resolves",
            resolve_within(root, "pyproject.toml").name == "pyproject.toml",
        )

    # This is the Windows-specific assertion. Git marks .git/objects read-only;
    # a naive shutil.rmtree raises PermissionError here and leaves the tree
    # behind. If this fails on Windows, rmtree_robust is not working.
    check("temp tree fully removed after exit", not root.exists())

    print("\n[2] caps reject rather than truncate")
    expect_raises(
        "size cap",
        RepoTooLargeError,
        lambda: _consume(cloned_repo(NAME, URL, SHA, limits=CloneLimits(max_bytes=100_000))),
    )
    expect_raises(
        "file-count cap",
        TooManyFilesError,
        lambda: _consume(cloned_repo(NAME, URL, SHA, limits=CloneLimits(max_files=5))),
    )

    print("\n[3] bad input fails loudly")
    expect_raises(
        "abbreviated sha rejected",
        ValueError,
        lambda: _consume(cloned_repo(NAME, URL, "6651ae6")),
    )
    expect_raises(
        "nonexistent commit rejected",
        CloneFailedError,
        lambda: _consume(cloned_repo(NAME, URL, "0" * 40)),
    )

    print("\n[4] non-python policy")
    expect_raises(
        "strict mode raises on too little python",
        UnsupportedLanguageError,
        lambda: _consume(
            cloned_repo(
                NAME, URL, SHA,
                limits=CloneLimits(min_python_files=10_000, degrade_on_unsupported=False),
            )
        ),
    )
    with cloned_repo(
        NAME, URL, SHA,
        limits=CloneLimits(min_python_files=10_000, degrade_on_unsupported=True),
    ) as repo:
        check("degrade mode falls back to text_only", repo.mode.value == "text_only")

    print(f"\n{passed} passed, {failed} failed")

    # Leftover directories are a silent failure mode: nothing crashes, disk just
    # fills up over a few hundred runs. Check explicitly.
    import tempfile
    ws = Path(tempfile.gettempdir()) / "repo-interrogator"
    leftovers = list(ws.iterdir()) if ws.exists() else []
    if leftovers:
        print(f"WARNING: {len(leftovers)} directories left in {ws}")
        for p in leftovers[:5]:
            print(f"   {p}")
        return 1

    print(f"no leftover clones in {ws}")
    return 0 if failed == 0 else 1


def _consume(cm) -> None:
    """Enter and exit a context manager, discarding the value."""
    with cm:
        pass


if __name__ == "__main__":
    sys.exit(main())