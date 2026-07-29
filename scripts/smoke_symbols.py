"""Smoke test for `get_symbols`: run it, don't trust it.

No network and no clone. A fixture tree is written to a temp directory and a
`ClonedRepo` is constructed by hand around it, so this runs offline in under a
second and exercises exactly the code under test.

Expected line numbers are derived from the fixture source by locating a unique
marker string, never hardcoded. Hardcoded numbers rot the moment anyone edits
the fixture, and a stale expectation that still passes is worse than no test.
The locator asserts the marker matched exactly one line, so it cannot silently
resolve to the wrong place -- and it shares no code with tree-sitter, so it
cannot fail in the same direction as the thing it is checking.

Run:  uv run python scripts/smoke_symbols.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from repo_interrogator.cloner import AnalysisMode, ClonedRepo
from repo_interrogator.errors import (
    PathEscapeError,
    SymbolDecodeError,
    SymbolIndexUnavailableError,
    SymbolParseError,
)
from repo_interrogator.fsutil import rmtree_robust
from repo_interrogator.symbols import (
    Symbol,
    SymbolKind,
    build_symbol_index,
    symbols_in_file,
)

# --- fixture ---------------------------------------------------------------

# Ends with a newline deliberately: that is what makes the last symbol's node
# end at column 0 of the following line, which is the off-by-one this file
# exists to catch.
GOOD_SRC = '''\
"""Fixture module docstring."""

import asyncio

CONSTANT = 42


def plain(a, b=1):
    """A module-level function."""
    return a + b


async def fetch_all(
    url: str,
    *,
    timeout: float = 1.0,
) -> list[str]:
    """Multi-line signature, async."""
    await asyncio.sleep(0)
    return [url]


@decorator_one
@decorator_two(arg=1)
def decorated(x):
    return x


def outer_fn():
    def inner_fn():
        return 1

    return inner_fn


class Outer:
    """A class."""

    attr = 1

    def method(self, q):
        return q

    @property
    def prop(self):
        return self.attr

    class Inner:
        def deep(self):
            return 2


def last_symbol():
    return 3
'''

BAD_SYNTAX_SRC = "def broken(:\n    pass\n"

# 0xFF is not a legal UTF-8 byte in any position.
NOT_UTF8_BYTES = b"x = '\xff'\n"


# --- harness ---------------------------------------------------------------

_passed = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global _passed
    if not condition:
        raise AssertionError(f"{label}{': ' + detail if detail else ''}")
    _passed += 1
    print(f"  [ok] {label}")


def check_raises(label: str, exc_type: type[BaseException], fn, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except exc_type:
        check(label, True)
        return
    except BaseException as exc:  # noqa: BLE001 - wrong type is the failure
        raise AssertionError(
            f"{label}: expected {exc_type.__name__}, got {type(exc).__name__}: {exc}"
        ) from exc
    raise AssertionError(f"{label}: expected {exc_type.__name__}, nothing raised")


def line_of(src: str, needle: str) -> int:
    """1-based line number of the single line containing `needle`.

    Raises if the marker is ambiguous, so an expectation can never quietly
    point at a line other than the one intended.
    """
    hits = [i for i, line in enumerate(src.splitlines(), start=1) if needle in line]
    if len(hits) != 1:
        raise AssertionError(f"marker {needle!r} matched {len(hits)} lines, need exactly 1")
    return hits[0]


def build_fixture(root: Path) -> None:
    (root / "good.py").write_text(GOOD_SRC, encoding="utf-8")
    (root / "bad_syntax.py").write_text(BAD_SYNTAX_SRC, encoding="utf-8")
    (root / "not_utf8.py").write_bytes(NOT_UTF8_BYTES)
    (root / "notes.md").write_text("# not python\n", encoding="utf-8")

    # Must be invisible to the indexer: same skip-list as every other tool.
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "cached.py").write_text("def ghost(): pass\n", encoding="utf-8")
    (root / ".venv").mkdir()
    (root / ".venv" / "vendored.py").write_text("def ghost2(): pass\n", encoding="utf-8")


def fake_repo(path: Path, mode: AnalysisMode) -> ClonedRepo:
    """A ClonedRepo around a fixture tree. Never cloned, never fetched."""
    return ClonedRepo(
        name="fixture",
        url="https://example.invalid/fixture.git",
        sha="0" * 40,
        path=path,
        size_bytes=1024,
        file_count=4,
        python_file_count=3,
        mode=mode,
    )


# --- tests -----------------------------------------------------------------


def test_premise() -> None:
    print("\npremise")
    check("fixture ends with a newline", GOOD_SRC.endswith("\n"))


def test_single_file(root: Path) -> None:
    print("\nsymbols_in_file")
    syms = symbols_in_file(root, "good.py")
    by_qual = {s.qualname: s for s in syms}

    check("10 definitions found", len(syms) == 10, f"got {len(syms)}: {sorted(by_qual)}")
    check(
        "qualnames are exactly the expected set",
        sorted(by_qual) == sorted(
            [
                "plain",
                "fetch_all",
                "decorated",
                "outer_fn",
                "Outer",
                "Outer.method",
                "Outer.prop",
                "Outer.Inner",
                "Outer.Inner.deep",
                "last_symbol",
            ]
        ),
        str(sorted(by_qual)),
    )
    check("assignments are not symbols", "attr" not in by_qual and "CONSTANT" not in by_qual)
    check("nested functions are not indexed", "inner_fn" not in by_qual)
    check("paths are repo-relative POSIX", all(s.path == "good.py" for s in syms))

    # --- line ranges: the contract this module exists to hold ---
    plain = by_qual["plain"]
    check(
        "plain starts at its def line",
        plain.start_line == line_of(GOOD_SRC, "def plain"),
        f"got {plain.start_line}",
    )
    check(
        "plain ends at its last statement, not the blank line after",
        plain.end_line == line_of(GOOD_SRC, "return a + b"),
        f"got {plain.end_line}",
    )

    last = by_qual["last_symbol"]
    check(
        "last symbol in file does not overrun by one",
        last.end_line == line_of(GOOD_SRC, "return 3"),
        f"got {last.end_line}",
    )
    check(
        "last symbol range is 2 lines",
        last.line_count == 2,
        f"got {last.line_count}",
    )

    outer = by_qual["Outer"]
    check(
        "class span covers its nested class body",
        outer.end_line == line_of(GOOD_SRC, "return 2"),
        f"got {outer.end_line}",
    )

    # --- decorators ---
    dec = by_qual["decorated"]
    check(
        "decorated symbol starts at the first decorator",
        dec.start_line == line_of(GOOD_SRC, "@decorator_one"),
        f"got {dec.start_line}",
    )
    check(
        "both decorators captured, in source order",
        dec.decorators == ("@decorator_one", "@decorator_two(arg=1)"),
        str(dec.decorators),
    )
    check("undecorated symbol has no decorators", plain.decorators == ())
    check(
        "property decorator captured on a method",
        by_qual["Outer.prop"].decorators == ("@property",),
        str(by_qual["Outer.prop"].decorators),
    )

    # --- kinds ---
    check("module-level def is FUNCTION", plain.kind is SymbolKind.FUNCTION)
    check("def in a class body is METHOD", by_qual["Outer.method"].kind is SymbolKind.METHOD)
    check("class is CLASS", outer.kind is SymbolKind.CLASS)
    check("nested class is CLASS", by_qual["Outer.Inner"].kind is SymbolKind.CLASS)
    check(
        "def in a nested class is METHOD",
        by_qual["Outer.Inner.deep"].kind is SymbolKind.METHOD,
    )

    # --- signatures ---
    check(
        "signature is source text, not reconstructed",
        by_qual["plain"].signature == "def plain(a, b=1):",
        repr(plain.signature),
    )
    fetch = by_qual["fetch_all"]
    check(
        "async def is captured",
        fetch.signature.startswith("async def fetch_all("),
        repr(fetch.signature),
    )
    check(
        "multi-line signature collapses to one line",
        "\n" not in fetch.signature and fetch.signature.endswith("-> list[str]:"),
        repr(fetch.signature),
    )
    check(
        "docstring is not part of the signature",
        "Multi-line signature" not in fetch.signature,
    )

    # --- ordering ---
    check(
        "symbols are returned in source order",
        [s.start_line for s in syms] == sorted(s.start_line for s in syms),
    )


def test_file_errors(root: Path, outside: Path) -> None:
    print("\nfile-level failures")
    check_raises(
        "unparseable file raises SymbolParseError",
        SymbolParseError,
        symbols_in_file,
        root,
        "bad_syntax.py",
    )
    check_raises(
        "non-UTF-8 file raises SymbolDecodeError",
        SymbolDecodeError,
        symbols_in_file,
        root,
        "not_utf8.py",
    )
    check_raises(
        "missing file raises SymbolParseError",
        SymbolParseError,
        symbols_in_file,
        root,
        "nope.py",
    )
    check_raises(
        "traversal outside the root is refused",
        PathEscapeError,
        symbols_in_file,
        root,
        f"../{outside.name}",
    )
    check_raises(
        "absolute path outside the root is refused",
        PathEscapeError,
        symbols_in_file,
        root,
        str(outside),
    )


def test_index(root: Path) -> None:
    print("\nbuild_symbol_index")
    index = build_symbol_index(fake_repo(root, AnalysisMode.FULL))

    check("only good.py indexed successfully", index.files_indexed == 1, str(index.files_indexed))
    check("all 10 symbols carried into the index", len(index.symbols) == 10)
    check(
        "both bad files recorded as failures, not dropped",
        sorted(p for p, _ in index.failures) == ["bad_syntax.py", "not_utf8.py"],
        str(index.failures),
    )
    check("failures carry a reason", all(reason for _, reason in index.failures))
    check("non-python files ignored", not any(s.path.endswith(".md") for s in index.symbols))
    check(
        "skip-list directories never walked",
        not any(s.name.startswith("ghost") for s in index.symbols),
        "indexer and list_files must see the same file set",
    )
    check("sha carried onto the index", index.sha == "0" * 40)
    check("by_file finds good.py", len(index.by_file("good.py")) == 10)
    check("by_file on an unknown path is empty", index.by_file("missing.py") == ())
    check("find resolves a qualname", len(index.find("Outer.Inner.deep")) == 1)

    check_raises(
        "text-only repo raises instead of returning an empty index",
        SymbolIndexUnavailableError,
        build_symbol_index,
        fake_repo(root, AnalysisMode.TEXT_ONLY),
    )


def test_invariant() -> None:
    print("\nline-numbering invariant")

    def make(start: int, end: int) -> Symbol:
        return Symbol(
            path="x.py",
            kind=SymbolKind.FUNCTION,
            name="f",
            qualname="f",
            signature="def f():",
            start_line=start,
            end_line=end,
        )

    check_raises("0-based start rejected", ValueError, make, 0, 5)
    check_raises("negative start rejected", ValueError, make, -1, 5)
    check_raises("inverted range rejected", ValueError, make, 10, 9)
    check("single-line symbol is valid", make(7, 7).line_count == 1)


def main() -> int:
    workspace = Path(tempfile.mkdtemp(prefix="smoke-symbols-"))
    root = workspace / "repo"
    root.mkdir()
    outside = workspace / "secret.py"
    outside.write_text("SECRET = 1\n", encoding="utf-8")

    try:
        build_fixture(root)
        test_premise()
        test_single_file(root)
        test_file_errors(root, outside)
        test_index(root)
        test_invariant()
    except AssertionError as exc:
        print(f"\nFAILED after {_passed} passing assertions:\n  {exc}")
        return 1
    finally:
        rmtree_robust(workspace)

    print(f"\n{_passed} assertions passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())