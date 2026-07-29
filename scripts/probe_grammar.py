"""One-shot diagnostic: does the symbol extractor actually see this repo?

Open question #5 asks whether the tree-sitter grammar, verified only against a
hand-written fixture, holds up against real code. That question has two halves,
and only one of them is about the grammar.

LOUD GAP -- the file does not parse. ``symbols_in_file`` checks
``root_node.has_error`` and raises, so these already land in
``SymbolIndex.failures`` and are already counted. A modern syntax feature the
installed grammar predates shows up here, as a whole file dropped. This probe
adds nothing but a breakdown by cause.

SILENT GAP -- the file parses cleanly, but ``_walk`` never reaches a definition
that is there. ``_walk`` descends into the module and into class bodies, and
nowhere else. A class defined under ``if TYPE_CHECKING:``, or inside
``try: ... except ImportError:``, sits behind an ``if_statement`` or a
``try_statement`` node and is skipped with no error and no failure row. For
function bodies that boundary is deliberate and documented. For conditional
top-level definitions it is probably not, and it is invisible to every number
the index reports about itself: ``files_indexed`` counts such a file as fully
indexed.

Reporting only ``len(failures)`` would close the first half and leave the second
open, which is the worse of the two. A symbol index that is quietly 8 percent
short still produces plausible questions, and Ablation C measures the value of
removing an index that was never complete.

METHOD
------
An independent traversal descends through statement containers as well as class
bodies, recording every definition together with whether ``_walk`` could have
reached it and, if not, which container hid it. Function bodies are not
descended into, so the documented boundary is respected and does not show up as
a false gap.

The probe checks itself: for every file, its own reachable-definition count is
compared against what ``symbols_in_file`` actually returned. A disagreement means
the probe is wrong, not the extractor, and it is reported in its own section
rather than folded into the totals.

USAGE
-----
    uv run python scripts/probe_grammar.py --fixture
    uv run python scripts/probe_grammar.py instructor <url> <40-hex-sha>
    uv run python scripts/probe_grammar.py instructor <url> <sha> --fixture

``repos.yaml`` is not read here; the repo is named on the command line so this
script does not depend on that file's schema.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser

from repo_interrogator.cloner import CloneLimits, cloned_repo, git_available
from repo_interrogator.errors import SymbolDecodeError, SymbolParseError
from repo_interrogator.fsutil import iter_files
from repo_interrogator.symbols import build_symbol_index, symbols_in_file

log = logging.getLogger("probe")

_PARSER = Parser(Language(tspython.language()))

_DEFINITIONS = ("function_definition", "class_definition")

_STATEMENT_CONTAINERS = frozenset(
    {
        "if_statement",
        "elif_clause",
        "else_clause",
        "try_statement",
        "except_clause",
        "except_group_clause",
        "finally_clause",
        "with_statement",
        "for_statement",
        "while_statement",
        "match_statement",
        "case_clause",
        "block",
    }
)
"""Nodes the reference scan descends through.

Deliberately a fixed list rather than "everything that is not a definition".
Descending indiscriminately would walk into expression trees and comprehensions,
which cannot hold a module-level definition and would only add noise. If a
container is missing from this set the probe under-reports the gap, which is the
safe direction to be wrong in: it will never invent a problem that is not there.
"""


@dataclass(frozen=True)
class Found:
    """One definition located by the reference scan."""

    path: str
    kind: str
    name: str
    line: int
    reachable: bool
    """True if ``_walk`` would have arrived here."""

    hidden_by: str
    """Outermost container that put it out of reach. Empty when reachable."""


def _name_of(source: bytes, node: Node) -> str:
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return "<unnamed>"
    return source[name_node.start_byte : name_node.end_byte].decode("utf-8", "replace")


def _reference_scan(
    container: Node,
    source: bytes,
    path: str,
    *,
    reachable: bool,
    hidden_by: str,
    out: list[Found],
) -> None:
    """Collect every definition, marking which ones ``_walk`` can see.

    Mirrors ``_walk``'s shape on purpose: same handling of
    ``decorated_definition``, same refusal to enter function bodies. The single
    difference is that this one also steps into ``if`` / ``try`` / ``with`` /
    ``match`` bodies, and flags everything found there as unreachable.
    """
    for child in container.named_children:
        target = child
        if child.type == "decorated_definition":
            target = child.child_by_field_name("definition")
            if target is None:
                continue

        if target.type in _DEFINITIONS:
            out.append(
                Found(
                    path=path,
                    kind=target.type.split("_")[0],
                    name=_name_of(source, target),
                    line=target.start_point[0] + 1,
                    reachable=reachable,
                    hidden_by=hidden_by,
                )
            )
            if target.type == "class_definition":
                body = target.child_by_field_name("body")
                if body is not None:
                    # Reachability is inherited: methods of a class that _walk
                    # never found are themselves unfound.
                    _reference_scan(
                        body,
                        source,
                        path,
                        reachable=reachable,
                        hidden_by=hidden_by,
                        out=out,
                    )
            # Function bodies are not entered. That boundary is the extractor's
            # documented decision, not a defect, and must not count as a gap.
            continue

        if child.type in _STATEMENT_CONTAINERS:
            _reference_scan(
                child,
                source,
                path,
                reachable=False,
                hidden_by=hidden_by or child.type,
                out=out,
            )


# --- repo pass -------------------------------------------------------------


def probe_repo(name: str, url: str, sha: str, *, limits: CloneLimits) -> int:
    """Clone, index, and measure both gaps. Returns a process exit code."""
    parse_errors: list[tuple[str, str]] = []
    decode_errors: list[str] = []
    os_errors: list[tuple[str, str]] = []
    disagreements: list[tuple[str, int, int]] = []
    found: list[Found] = []
    py_files = 0

    with cloned_repo(name, url, sha, limits=limits) as repo:
        index = build_symbol_index(repo)
        root = repo.path.resolve()

        for abs_path in iter_files(root):
            if abs_path.suffix != ".py":
                continue
            py_files += 1
            rel = abs_path.relative_to(root).as_posix()

            try:
                extracted = symbols_in_file(root, rel)
            except SymbolParseError as exc:
                parse_errors.append((rel, str(exc)))
                continue
            except SymbolDecodeError:
                decode_errors.append(rel)
                continue
            except OSError as exc:
                os_errors.append((rel, str(exc)))
                continue

            raw = abs_path.read_bytes()
            tree = _PARSER.parse(raw)

            per_file: list[Found] = []
            _reference_scan(
                tree.root_node, raw, rel, reachable=True, hidden_by="", out=per_file
            )
            found.extend(per_file)

            reachable = sum(1 for f in per_file if f.reachable)
            if reachable != len(extracted):
                disagreements.append((rel, len(extracted), reachable))

        _report(name, repo.sha, index, py_files, parse_errors, decode_errors,
                os_errors, disagreements, found)

    return _verdict(py_files, parse_errors, decode_errors, os_errors,
                    disagreements, found)


def _pct(part: int, whole: int) -> str:
    return "n/a" if whole == 0 else f"{100.0 * part / whole:.1f}%"


def _report(
    name: str,
    sha: str,
    index,
    py_files: int,
    parse_errors: list[tuple[str, str]],
    decode_errors: list[str],
    os_errors: list[tuple[str, str]],
    disagreements: list[tuple[str, int, int]],
    found: list[Found],
) -> None:
    missed = [f for f in found if not f.reachable]
    total_defs = len(found)

    print(f"\n{'=' * 68}")
    print(f"{name} @ {sha[:8]}")
    print("=" * 68)

    print("\nindex as reported")
    print(f"  python files seen   {py_files}")
    print(f"  files_indexed       {index.files_indexed}")
    print(f"  symbols             {len(index.symbols)}")
    print(f"  failures            {len(index.failures)}")

    print("\nloud gap  (file did not parse, whole file dropped)")
    print(f"  parse errors        {len(parse_errors)}  ({_pct(len(parse_errors), py_files)} of files)")
    print(f"  decode errors       {len(decode_errors)}")
    print(f"  os errors           {len(os_errors)}")
    for rel, reason in parse_errors[:10]:
        print(f"    {rel}\n      {reason}")
    if len(parse_errors) > 10:
        print(f"    ... and {len(parse_errors) - 10} more")

    print("\nsilent gap  (parsed cleanly, _walk never arrived)")
    print(f"  definitions present {total_defs}")
    print(f"  definitions missed  {len(missed)}  ({_pct(len(missed), total_defs)})")

    if missed:
        print("\n  by container")
        for container, n in Counter(f.hidden_by for f in missed).most_common():
            print(f"    {container:<24} {n}")

        print("\n  worst files")
        for rel, n in Counter(f.path for f in missed).most_common(8):
            print(f"    {rel:<52} {n}")

        print("\n  sample")
        for f in missed[:12]:
            print(f"    {f.path}:{f.line}  {f.kind} {f.name}  <- {f.hidden_by}")

    if disagreements:
        print("\nPROBE DISAGREEMENT  (this probe is wrong here, not the extractor)")
        for rel, extracted, reachable in disagreements[:12]:
            print(f"  {rel}: symbols_in_file={extracted} reference_reachable={reachable}")
        if len(disagreements) > 12:
            print(f"  ... and {len(disagreements) - 12} more")


def _verdict(
    py_files: int,
    parse_errors: list[tuple[str, str]],
    decode_errors: list[str],
    os_errors: list[tuple[str, str]],
    disagreements: list[tuple[str, int, int]],
    found: list[Found],
) -> int:
    missed = [f for f in found if not f.reachable]
    print("\nverdict")

    code = 0
    if disagreements:
        print("  ! probe self-check failed. Fix the probe before trusting any"
              " number above.")
        code = 2

    parse_rate = 0.0 if py_files == 0 else len(parse_errors) / py_files
    if parse_rate >= 0.01 or len(parse_errors) >= 12:
        print(f"  ! grammar: {len(parse_errors)} files unparseable. Check the"
              " installed tree-sitter-python against the failing constructs.")
        code = max(code, 1)
    else:
        print(f"  - grammar: {len(parse_errors)} unparseable. Holds up.")

    miss_rate = 0.0 if not found else len(missed) / len(found)
    if miss_rate >= 0.02 or len(missed) >= 25:
        print(f"  ! traversal: {len(missed)} definitions ({_pct(len(missed), len(found))})"
              " invisible to _walk. This needs a decision, not a note.")
        code = max(code, 1)
    else:
        print(f"  - traversal: {len(missed)} definitions missed. Within noise.")

    if decode_errors or os_errors:
        print(f"  - {len(decode_errors)} decode / {len(os_errors)} os errors.")

    return code


# --- fixture ---------------------------------------------------------------

# (label, source, expected_walk_symbols, expected_total_definitions)
#
# The last two entries are not grammar cases. They are the conditional-definition
# pattern, included so the two kinds of gap can be seen side by side in one run:
# both parse perfectly, and both yield nothing.
FEATURES: list[tuple[str, str, int, int]] = [
    ("match statement",
     "def f(x):\n    match x:\n        case 1:\n            return 'one'\n        case _:\n            return 'other'\n",
     1, 1),
    ("walrus operator",
     "def f(items):\n    if (n := len(items)) > 3:\n        return n\n    return 0\n",
     1, 1),
    ("PEP 695 generic function",
     "def first[T](xs: list[T]) -> T:\n    return xs[0]\n",
     1, 1),
    ("PEP 695 generic class",
     "class Box[T]:\n    def get(self) -> T:\n        ...\n",
     2, 2),
    ("PEP 695 type alias",
     "type Alias = int | str\n",
     0, 0),
    ("parenthesized context managers",
     "def f():\n    with (open('a') as a, open('b') as b):\n        return a, b\n",
     1, 1),
    ("except* group",
     "def f():\n    try:\n        pass\n    except* ValueError:\n        pass\n",
     1, 1),
    ("positional-only parameters",
     "def f(a, b, /, c, *, d):\n    return a\n",
     1, 1),
    ("async comprehension",
     "async def f(it):\n    return [x async for x in it]\n",
     1, 1),
    ("PEP 614 decorator expression",
     "@buttons[0].clicked.connect\ndef f():\n    ...\n",
     1, 1),
    ("PEP 701 nested f-string",
     "def f(x):\n    return f\"{f'{x!r}'}\"\n",
     1, 1),
    ("dataclass with slots and defaults",
     "import dataclasses\n\n@dataclasses.dataclass(frozen=True, slots=True)\nclass C:\n    a: int = 0\n\n    def m(self) -> int:\n        return self.a\n",
     2, 2),
    ("class under if TYPE_CHECKING",
     "import typing\n\nif typing.TYPE_CHECKING:\n    class Guarded:\n        ...\n",
     0, 1),
    ("class under try/except ImportError",
     "try:\n    import orjson\n\n    class Fast:\n        ...\nexcept ImportError:\n    class Fast:\n        ...\n",
     0, 2),
]


def probe_fixture() -> int:
    """Parse each snippet in memory. No clone, no network."""
    print(f"\n{'=' * 68}")
    print("syntax fixture")
    print("=" * 68)
    print(f"\n{'feature':<34} {'parses':<8} {'walk':<12} {'total':<12}")
    print("-" * 68)

    code = 0
    for label, src, expect_walk, expect_total in FEATURES:
        raw = src.encode("utf-8")
        tree = _PARSER.parse(raw)
        parses = not tree.root_node.has_error

        if parses:
            all_defs: list[Found] = []
            _reference_scan(
                tree.root_node, raw, "<fixture>", reachable=True, hidden_by="", out=all_defs
            )
            walk_n = sum(1 for f in all_defs if f.reachable)
            total_n = len(all_defs)
        else:
            walk_n = total_n = 0

        walk_ok = parses and walk_n == expect_walk
        total_ok = parses and total_n == expect_total

        if not parses:
            code = 1
        elif not (walk_ok and total_ok):
            code = max(code, 1)

        print(
            f"{label:<34} "
            f"{('yes' if parses else 'NO'):<8} "
            f"{f'{walk_n}/{expect_walk}' + ('' if walk_ok else '  <-'):<12} "
            f"{f'{total_n}/{expect_total}' + ('' if total_ok else '  <-'):<12}"
        )

    print(
        "\n'walk' is what the extractor sees; 'total' is what is in the file.\n"
        "A row where the two expectations differ is a traversal gap, not a\n"
        "grammar gap: the snippet parsed, and the definition is still missing."
    )
    return code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", nargs="?", help="repo label, e.g. instructor")
    parser.add_argument("url", nargs="?", help="clone url")
    parser.add_argument("sha", nargs="?", help="full 40-hex commit id")
    parser.add_argument("--fixture", action="store_true",
                        help="run the in-memory syntax fixture")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(name)s: %(message)s",
    )

    repo_args = [args.name, args.url, args.sha]
    if any(repo_args) and not all(repo_args):
        parser.error("name, url and sha must be given together")
    if not any(repo_args) and not args.fixture:
        parser.error("give a repo (name url sha), or --fixture, or both")

    code = 0
    if args.fixture:
        code = probe_fixture()

    if all(repo_args):
        if not git_available():
            print("git not found on PATH", file=sys.stderr)
            return 3
        code = max(code, probe_repo(args.name, args.url, args.sha,
                                    limits=CloneLimits()))

    return code


if __name__ == "__main__":
    raise SystemExit(main())