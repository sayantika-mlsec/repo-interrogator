"""Smoke test for the workspace safety layer.

Run once after installing, to confirm every guard actually fires on this
machine:

    uv run python scripts/smoke_clone.py

The Windows read-only cleanup path in particular cannot be verified on Linux,
because the failure it handles does not occur there. This script is the only
thing that proves it works before real runs depend on it.

Section [0] is offline and runs first. Two bounds cannot be exercised against a
real repository: a repository is whatever size it is, and cannot be made to
breach the readable cap while clearing the total guard on request. Synthetic
trees can. They also run in milliseconds, so a mistake in the measurement layer
fails before anything touches the network.
"""

from __future__ import annotations

import logging
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from repo_interrogator.cloner import CloneLimits, cloned_repo, git_available
from repo_interrogator.errors import (
    CloneFailedError,
    InvalidShaError,
    PathEscapeError,
    ReadableBytesTooLargeError,
    RepoTooLargeError,
    TooManyFilesError,
    UnsupportedLanguageError,
)
from repo_interrogator.fsutil import measure_tree, resolve_within

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

# A small pinned repo. Any entry from repos.yaml would do; this one is chosen
# because it clones in a couple of seconds.
NAME = "structlog"
URL = "https://github.com/hynek/structlog"
SHA = "6651ae67d5790923bdaf110e08fb891b3ac706d8"

# Synthetic file contents. Binary is a NUL run, which is git's own heuristic and
# the first signal is_probably_binary checks. Text is printable ASCII, which
# decodes cleanly and never reaches the ratio test.
BINARY = b"\x00" * 200_000
TEXT = b"x" * 50_000
SMALL_TEXT = b"print('hello')\n"

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


def expect_raises(label: str, exc_type: type[BaseException], fn) -> BaseException | None:
    global passed, failed
    try:
        fn()
    except exc_type as e:
        passed += 1
        print(f"  PASS  {label}  ({type(e).__name__})")
        return e
    except Exception as e:  # noqa: BLE001 - the point is to report the wrong type
        failed += 1
        print(f"  FAIL  {label}  (expected {exc_type.__name__}, got {type(e).__name__}: {e})")
    else:
        failed += 1
        print(f"  FAIL  {label}  (no exception raised)")
    return None


@contextmanager
def temp_tree(spec: dict[str, bytes]) -> Iterator[Path]:
    """Build a throwaway tree from ``relative path -> contents``."""
    with tempfile.TemporaryDirectory(prefix="ri-smoke-") as d:
        root = Path(d)
        for rel, data in spec.items():
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        yield root


def offline_measurement() -> None:
    """Everything the two-bound cap promises, proven without a network."""

    # --- .git is excluded, and was already excluded before this change. This
    # is a regression guard: the mechanism depends on the working tree being a
    # function of the pinned SHA, which the object store is not.
    with temp_tree({".git/objects/pack/big.pack": BINARY, "a.py": SMALL_TEXT}) as root:
        m = measure_tree(root)
        check(".git bytes excluded from total", m.total_bytes == len(SMALL_TEXT))
        check(".git files excluded from count", m.file_count == 1)

    # --- unreadable bytes are counted in the total and not in the readable
    # budget. This is the whole defect, in two numbers.
    with temp_tree({"docs/demo.gif": BINARY, "pkg/mod.py": SMALL_TEXT}) as root:
        m = measure_tree(root)
        check("binary counted in total", m.total_bytes == len(BINARY) + len(SMALL_TEXT))
        check("binary excluded from readable", m.readable_bytes == len(SMALL_TEXT))
        check("readable file count excludes binary", m.readable_file_count == 1)
        check("file count includes binary", m.file_count == 2)
        check("binary_bytes is the difference", m.binary_bytes == len(BINARY))
        check("unbounded walk is complete", m.partial is False)
        check("unbounded walk names no bound", m.breached is None)
        check("to_dict records completeness", m.to_dict()["partial"] is False)

    # --- a bounded walk stops early, and says so. The figures it returns are
    # lower bounds; nothing downstream may report them as totals.
    with temp_tree({"a.txt": TEXT, "b.txt": TEXT, "c.txt": TEXT}) as root:
        m = measure_tree(root, max_readable_bytes=len(TEXT))
        check("bounded breach marks the walk partial", m.partial is True)
        check("bounded breach names readable_bytes", m.breached == "readable_bytes")
        check("partial figures are a lower bound", m.readable_bytes <= 3 * len(TEXT))

        exact = measure_tree(root)
        check("same tree unbounded is exact", exact.readable_bytes == 3 * len(TEXT))
        check("same tree unbounded is complete", exact.partial is False)

    # --- the two bounds are independent. A tree of blobs trips the machine
    # guard; a tree of source trips the readable cap. Neither substitutes for
    # the other.
    with temp_tree({"video.mp4": BINARY, "tiny.py": SMALL_TEXT}) as root:
        m = measure_tree(root, max_total_bytes=1_000, max_readable_bytes=1_000_000)
        check("blob tree breaches the total guard", m.breached == "total_bytes")

    with temp_tree({f"src/mod{i}.py": TEXT for i in range(4)}) as root:
        m = measure_tree(root, max_total_bytes=100_000_000, max_readable_bytes=len(TEXT))
        check("source tree breaches the readable cap", m.breached == "readable_bytes")

    # --- bounds are reported in the order they are checked.
    # Both bounds have to breach on the same iteration for the order to be
    # observable. A readable bound of exactly one file's worth clears on file
    # one and falls on file two, which is also where the file count goes over.
    with temp_tree({"a.txt": TEXT, "b.txt": TEXT}) as root:
        m = measure_tree(root, max_files=1, max_readable_bytes=len(TEXT))
        check("file count is reported before byte bounds", m.breached == "file_count")

    # --- readability is decided by content. A suffix allowlist would get both
    # of these backwards, which is why none exists in either module.
    with temp_tree({"code.py": BINARY, "image.gif": SMALL_TEXT}) as root:
        m = measure_tree(root)
        check("a .py of binary is not readable", m.readable_bytes == len(SMALL_TEXT))
        check("a .gif of text is readable", m.readable_file_count == 1)
        check("python count stays suffix-based", m.python_file_count == 1)

    # --- an empty tree measures cleanly rather than raising or going negative.
    with temp_tree({}) as root:
        m = measure_tree(root)
        check(
            "empty tree measures zero and complete",
            m.total_bytes == 0 and m.file_count == 0 and m.partial is False,
        )


def main() -> int:
    print("\n[0] measurement: two bounds, offline")
    offline_measurement()

    if not git_available():
        print("\ngit not found on PATH; skipping sections [1]-[5]")
        print(f"\n{passed} passed, {failed} failed")
        return 2

    print("\n[1] pinned fetch, measurement, containment, cleanup")
    with cloned_repo(NAME, URL, SHA) as repo:
        root = repo.path
        check("sha matches request", repo.sha == SHA)
        check("tree exists on disk", (root / "pyproject.toml").is_file())
        check("total measured", repo.total_bytes > 0)
        check("readable measured", repo.readable_bytes > 0)
        check("readable never exceeds total", repo.readable_bytes <= repo.total_bytes)
        check("admitting walk was complete", repo.measurement.partial is False)
        check("python files found", repo.python_file_count > 0)
        check("mode is full", repo.mode.value == "full")
        check(
            "measurement is serialisable with completeness",
            set(repo.measurement.to_dict()) >= {"total_bytes", "readable_bytes", "partial"},
        )

        expect_raises(
            "parent traversal blocked",
            PathEscapeError,
            lambda: resolve_within(root, "../../../etc/passwd"),
        )
        expect_raises(
            "absolute path blocked",
            PathEscapeError,
            lambda: resolve_within(
                root,
                "C:\\Windows\\System32" if sys.platform == "win32" else "/etc/passwd",
            ),
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
    total_err = expect_raises(
        "total-tree guard",
        RepoTooLargeError,
        lambda: _consume(
            cloned_repo(NAME, URL, SHA, limits=CloneLimits(max_total_bytes=100_000))
        ),
    )
    readable_err = expect_raises(
        "readable-bytes cap",
        ReadableBytesTooLargeError,
        lambda: _consume(
            cloned_repo(NAME, URL, SHA, limits=CloneLimits(max_readable_bytes=100_000))
        ),
    )
    expect_raises(
        "file-count cap",
        TooManyFilesError,
        lambda: _consume(cloned_repo(NAME, URL, SHA, limits=CloneLimits(max_files=5))),
    )

    # A rejection is decided by a bounded walk, which always stops early, so its
    # figures are always lower bounds and the message must always say so.
    check("total guard message qualifies the figure", "at least" in str(total_err or ""))
    check("readable cap message qualifies the figure", "at least" in str(readable_err or ""))
    check("total guard message names its bound", "total-tree" in str(total_err or ""))
    check("readable cap message names its bound", "readable cap" in str(readable_err or ""))
    check(
        "the two bounds raise distinct types",
        not isinstance(readable_err, RepoTooLargeError),
    )

    print("\n[3] bad input fails loudly")
    expect_raises(
        "abbreviated sha rejected",
        InvalidShaError,
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

    print("\n[5] measurement-only mode")
    # Caps that would reject, with enforcement off: the repo is admitted and the
    # walk runs to completion, which is the only way the survey can record a
    # figure for a repository the caps would refuse.
    with cloned_repo(
        NAME, URL, SHA,
        limits=CloneLimits(max_readable_bytes=1, enforce_caps=False),
    ) as repo:
        check("enforcement off admits a breaching repo", repo.readable_bytes > 1)
        check("unenforced walk is complete", repo.measurement.partial is False)
        check("unenforced walk names no bound", repo.measurement.breached is None)

    print(f"\n{passed} passed, {failed} failed")

    # Leftover directories are a silent failure mode: nothing crashes, disk just
    # fills up over a few hundred runs. Check explicitly.
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