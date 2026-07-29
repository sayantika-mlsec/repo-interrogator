"""Structural symbol extraction: what a Python file defines, and where.

LINE NUMBERING CONTRACT
-----------------------
Every line number produced by this module is **1-based and inclusive on both
ends**. A definition occupying the first three lines of a file is ``(1, 3)``.

This is the convention used by editors, ``git blame`` and ``grep -n``, and it is
what every citation in this project resolves against. tree-sitter reports
0-based rows, so the conversion happens exactly once -- in ``_start_line`` and
``_end_line`` below -- and never again anywhere downstream. Code that slices a
Python list of lines must subtract 1 itself.

The contract is enforced, not merely documented: ``Symbol.__post_init__``
rejects any range that violates it.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser

from .cloner import AnalysisMode, ClonedRepo
from .errors import (
    SymbolDecodeError,
    SymbolIndexUnavailableError,
    SymbolParseError,
)
from .fsutil import iter_files, resolve_within

log = logging.getLogger(__name__)

PY_LANGUAGE = Language(tspython.language())
_PARSER = Parser(PY_LANGUAGE)

MAX_SIGNATURE_CHARS = 300
"""Signatures longer than this are truncated with a visible marker.

Truncation is acceptable here and not at the repo level, because a clipped
signature is obvious to the reader while a clipped repository is not.
"""


class SymbolKind(str, Enum):
    FUNCTION = "function"
    """Defined at module level."""

    METHOD = "method"
    """Defined inside a class body."""

    CLASS = "class"


@dataclass(frozen=True)
class Symbol:
    """One definition, located exactly.

    ``qualname`` disambiguates the many ``__init__`` and ``run`` names a real
    repository contains; ``name`` alone is not a key.
    """

    path: str
    """Repository-relative, POSIX separators. Never absolute."""

    kind: SymbolKind
    name: str
    qualname: str
    signature: str
    start_line: int
    end_line: int
    decorators: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.start_line < 1:
            raise ValueError(
                f"{self.path}:{self.qualname}: start_line={self.start_line}. "
                "Line numbers are 1-based; 0 means a conversion was missed."
            )
        if self.end_line < self.start_line:
            raise ValueError(
                f"{self.path}:{self.qualname}: end_line={self.end_line} precedes "
                f"start_line={self.start_line}. Ranges are inclusive."
            )

    @property
    def line_count(self) -> int:
        return self.end_line - self.start_line + 1


@dataclass(frozen=True)
class SymbolIndex:
    """Every definition in one repository, at one commit."""

    repo_name: str
    sha: str
    symbols: tuple[Symbol, ...]
    files_indexed: int
    failures: tuple[tuple[str, str], ...]
    """``(path, reason)`` for each file that could not be parsed.

    Carried in the index rather than logged and forgotten: a run against a repo
    where 40 files failed must not be reportable as if it saw the whole tree.
    """

    def by_file(self, path: str) -> tuple[Symbol, ...]:
        return tuple(s for s in self.symbols if s.path == path)

    def find(self, qualname: str) -> tuple[Symbol, ...]:
        return tuple(s for s in self.symbols if s.qualname == qualname)


def _start_line(node: Node) -> int:
    """0-based row -> 1-based line."""
    return node.start_point[0] + 1


def _end_line(node: Node) -> int:
    """0-based exclusive end point -> 1-based inclusive last line.

    tree-sitter's ``end_point`` is one past the last character. When a node ends
    with a newline -- which every multi-line definition does -- that lands on
    column 0 of the *following* line. Reporting it directly would claim one line
    too many, and every citation against the last symbol in a file would be off
    by one.
    """
    row, column = node.end_point
    if column == 0 and row > node.start_point[0]:
        row -= 1
    return row + 1


def _text(source: bytes, node: Node) -> str:
    """Slice source by byte offsets.

    Decoding is strict. tree-sitter never splits a UTF-8 character, so a failure
    here means the byte offsets are wrong, and that must surface immediately
    rather than be papered over with replacement characters.
    """
    return source[node.start_byte : node.end_byte].decode("utf-8")


def _signature(source: bytes, node: Node) -> str:
    """Everything up to the body: ``def f(a, b) -> int:``.

    Taken as a slice of real source rather than reassembled from child nodes.
    Reconstruction would have to re-implement Python's syntax for defaults,
    annotations, ``*args``, and generics -- and would silently drift from the
    file the citation points at. The source is the ground truth.
    """
    body = node.child_by_field_name("body")
    end = body.start_byte if body is not None else node.end_byte
    raw = source[node.start_byte : end].decode("utf-8").strip()
    collapsed = " ".join(raw.split())
    if len(collapsed) > MAX_SIGNATURE_CHARS:
        return collapsed[:MAX_SIGNATURE_CHARS] + " ...<truncated>"
    return collapsed


def _walk(
    container: Node,
    source: bytes,
    path: str,
    prefix: str,
) -> Iterator[Symbol]:
    """Yield definitions directly inside ``container``, recursing into classes only.

    Function bodies are not descended into. Closures and locally defined helpers
    are real code, but they are not navigation targets -- including them would
    multiply the map's size while adding entries no question can usefully cite.
    This is a deliberate boundary, and it is what ``by_file`` counts.
    """
    for child in container.named_children:
        target = child
        decorators: tuple[str, ...] = ()

        if child.type == "decorated_definition":
            target = child.child_by_field_name("definition")
            if target is None:
                raise SymbolParseError(
                    f"{path}:{_start_line(child)}: decorated definition has no definition"
                )
            decorators = tuple(
                _text(source, d) for d in child.children if d.type == "decorator"
            )

        if target.type not in ("function_definition", "class_definition"):
            continue

        name_node = target.child_by_field_name("name")
        if name_node is None:
            raise SymbolParseError(
                f"{path}:{_start_line(target)}: {target.type} has no name"
            )
        name = _text(source, name_node)
        qualname = f"{prefix}.{name}" if prefix else name

        is_class = target.type == "class_definition"
        if is_class:
            kind = SymbolKind.CLASS
        elif prefix:
            kind = SymbolKind.METHOD
        else:
            kind = SymbolKind.FUNCTION

        yield Symbol(
            path=path,
            kind=kind,
            name=name,
            qualname=qualname,
            signature=_signature(source, target),
            # Span starts at the decorator when there is one: the decorator is
            # part of the definition, and a citation that omits it points at
            # code whose behaviour it does not fully describe.
            start_line=_start_line(child),
            end_line=_end_line(child),
            decorators=decorators,
        )

        if is_class:
            body = target.child_by_field_name("body")
            if body is not None:
                yield from _walk(body, source, path, qualname)


def symbols_in_file(repo_root: Path, relative_path: str | Path) -> list[Symbol]:
    """Extract every definition from one Python file.

    ``relative_path`` is routed through the containment check, because this is a
    tool the model calls with paths it chose itself.
    """
    root = repo_root.resolve()
    abs_path = resolve_within(root, relative_path)

    if not abs_path.is_file():
        raise SymbolParseError(f"{relative_path}: not a file")

    raw = abs_path.read_bytes()
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SymbolDecodeError(f"{relative_path}: not valid UTF-8") from exc

    tree = _PARSER.parse(raw)
    if tree.root_node.has_error:
        raise SymbolParseError(
            f"{relative_path}: does not parse as Python 3 "
            "(Python 2 source, a template, or a truncated file)"
        )

    rel = abs_path.relative_to(root).as_posix()
    return list(_walk(tree.root_node, raw, rel, ""))


def build_symbol_index(repo: ClonedRepo) -> SymbolIndex:
    """Index every ``.py`` file in a cloned repository.

    Per-file failures are collected, not raised: one Python 2 file in a vendored
    directory should not void the other four hundred. They are counted and
    returned so the caller can see the shape of what was missed.
    """
    if repo.mode is AnalysisMode.TEXT_ONLY:
        raise SymbolIndexUnavailableError(
            f"{repo.name}: cloned in text-only mode, no symbol index exists. "
            "Use search_code and read_file instead."
        )

    root = repo.path.resolve()
    symbols: list[Symbol] = []
    failures: list[tuple[str, str]] = []
    indexed = 0

    # Same traversal as every other tool, so the file set the indexer sees and
    # the file set list_file s reports can never disagree.
    for abs_path in iter_files(root):
        if abs_path.suffix != ".py":
            continue
        rel = abs_path.relative_to(root).as_posix()
        try:
            symbols.extend(symbols_in_file(root, rel))
            indexed += 1
        except (SymbolParseError, SymbolDecodeError, OSError) as exc:
            failures.append((rel, str(exc)))

    if failures:
        log.warning(
            "%s @ %s: %d of %d python files failed to index",
            repo.name, repo.sha[:8], len(failures), indexed + len(failures),
        )

    log.info(
        "%s @ %s: %d symbols across %d files",
        repo.name, repo.sha[:8], len(symbols), indexed,
    )

    return SymbolIndex(
        repo_name=repo.name,
        sha=repo.sha,
        symbols=tuple(symbols),
        files_indexed=indexed,
        failures=tuple(failures),
    )