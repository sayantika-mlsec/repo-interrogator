"""Model-facing repository access: list, read, search, symbols.

These are four of the agent's five tools. Two properties shape every decision
below, and neither is obvious from the signatures.

**The paths come from the model.** Not from a config file, not from a
traversal -- from a language model that guessed. A guess can be ``../../etc``,
an absolute path, or a symlink that escapes while containing no traversal
segment at all. Every path entering these tools goes through ``resolve_within``
before anything touches the filesystem.

**The output goes into a context window.** So every response is bounded, and
truncation is marked in the same channel as the content, where the model reads
it. This is the opposite of the workspace layer, which rejects rather than
truncates: a half-read *repository* is invisible and corrupts a results row,
while a half-read *file* is visible to the reader and recoverable by asking for
the next range.

LINE NUMBERING
--------------
1-based, inclusive on both ends -- identical to the symbol layer. That is not a
stylistic match. A citation produced from ``read_file`` output and a citation
produced from a ``Symbol`` are resolved by the same verifier, so if the two
disagreed by one, half of every run's citations would fail and the failures
would be attributed to the model rather than to the tools.

Slicing a Python list of lines subtracts 1 exactly once, in ``read_file``.

THE CONFIG SEAM
---------------
``ToolConfig`` exists because two of these behaviours -- numbered output, and
the required line range -- are removed later to measure what they are worth.
Editing the tool at measurement time would invalidate the measurement, so the
toggle exists before the first baseline is recorded.

It is threaded explicitly rather than read from a global or the environment.
The reason is provenance, not testability: every run produces a results row, and
that row has to state which configuration produced it. Configuration living in
process state cannot be serialised into the record, and reconciling it after the
fact is guesswork.

The defaults are the design under test. ``ToolConfig()`` yields numbered output
and a required range, so forgetting to pass a config produces the real design
rather than a silently ablated one. An ablation has to be asked for.

One consequence worth stating: when ``number_lines`` is off, *no* line number
appears anywhere in ``read_file`` output -- not in a header, not in a truncation
marker. A header reading "lines 40-80 of 320" would hand back exactly the
information the ablation removes, leaving the model to count within a block
whose offset it was told. The ablation would then measure something much weaker
than it claims to.

A second consequence, recorded rather than fixed: ``get_symbols`` reports line
ranges regardless of ``number_lines``, so under that ablation the model can
still obtain line numbers from the symbol index -- as it can from
``search_code``. Suppressing them would leave a tool that names definitions
without locating them, which is not a tool at all. The interaction is reported
alongside the ablation result.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .errors import (
    BinaryFileError,
    FileDecodeError,
    FileNotFoundInRepoError,
    FileTooLargeError,
    InvalidLineRangeError,
    LineRangeRequiredError,
    NotAFileError,
    RipgrepNotAvailableError,
    SearchFailedError,
    SymbolsUnavailableError,
    ToolConfigurationError,
)
from .fsutil import SKIP_DIRS, is_probably_binary, iter_files, resolve_within

if TYPE_CHECKING:
    # Type-checking only. A runtime import would pull in the symbol layer, which
    # imports the workspace layer, and the tools would then depend on the cloner
    # in order to read a file.
    from .symbols import Symbol, SymbolIndex

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolConfig:
    """Tool behaviour, recorded alongside every result it produces."""

    number_lines: bool = True
    """Prefix each line of ``read_file`` output with its number. Ablation knob."""

    require_line_range: bool = True
    """Reject ``read_file`` calls that omit a range. Ablation knob."""

    max_read_lines: int = 400
    max_list_entries: int = 1500
    max_search_matches: int = 80
    max_symbol_entries: int = 200
    max_response_chars: int = 24_000
    max_line_chars: int = 400
    max_file_bytes: int = 8 * 1024 * 1024
    search_timeout_s: int = 60

    def to_dict(self) -> dict[str, object]:
        """Serialise for storage on a run record."""
        return asdict(self)

    @property
    def ablation_knobs(self) -> dict[str, bool]:
        """The two fields an ablation is permitted to vary.

        The response bounds are deliberately excluded. An ablation that also
        loosened the caps would confound the effect it is measuring with a
        context-length effect, and the two could not be separated afterwards.
        """
        return {
            "number_lines": self.number_lines,
            "require_line_range": self.require_line_range,
        }


@dataclass(frozen=True)
class ListResult:
    paths: tuple[str, ...]
    total_found: int
    truncated: bool
    text: str


@dataclass(frozen=True)
class ReadResult:
    path: str
    start_line: int
    end_line: int
    """Last line actually returned. May be below the requested end after clamping."""

    total_lines: int
    lines: tuple[str, ...]
    clamped: bool
    truncated: bool
    text: str


@dataclass(frozen=True)
class Match:
    path: str
    line_number: int
    line_text: str


@dataclass(frozen=True)
class SearchResult:
    pattern: str
    matches: tuple[Match, ...]
    total_found: int
    truncated: bool
    text: str


@dataclass(frozen=True)
class SymbolsResult:
    path: str
    symbols: tuple[Symbol, ...]
    total_found: int
    truncated: bool
    text: str


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


class FileTools:
    """The four repository-access tools, bound to one repository at one commit.

    A class rather than free functions so that the root, the configuration and
    the ripgrep dependency are resolved once, at construction. A missing
    ``rg`` discovered fifty steps into an agent run has already wasted the run.

    The symbol index is passed in rather than built here. Building it requires
    the symbol layer, which requires the workspace layer, and the tools would
    then transitively depend on the cloner in order to read a file. The caller
    already has the index; handing it over costs nothing.
    """

    def __init__(
        self,
        root: Path,
        config: ToolConfig | None = None,
        *,
        symbol_index: SymbolIndex | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ToolConfigurationError(f"repository root is not a directory: {self.root}")

        self.config = config or ToolConfig()
        self._index = symbol_index

        self._rg = shutil.which("rg")
        if self._rg is None:
            raise RipgrepNotAvailableError(
                "ripgrep (rg) not found on PATH. There is no Python-side fallback: "
                "a fallback would search a different file set with different regex "
                "semantics, so results would depend on the machine that ran them."
            )

    # --- shared path handling ---------------------------------------------

    def _resolve_file(self, path: str | Path) -> Path:
        """Contain, then check existence. Two separate questions.

        ``resolve_within`` answers "could this path escape", and it resolves
        non-strictly -- a path to a file that does not exist passes containment
        perfectly well. Existence is checked here, separately, so a nonexistent
        path inside the repository reports as a missing file rather than as an
        escape attempt.
        """
        abs_path = resolve_within(self.root, path)
        if not abs_path.exists():
            raise FileNotFoundInRepoError(f"{self._rel(abs_path)}: does not exist")
        if not abs_path.is_file():
            raise NotAFileError(f"{self._rel(abs_path)}: is a directory, not a file")
        return abs_path

    def _resolve_dir(self, path: str | Path) -> Path:
        abs_path = resolve_within(self.root, path)
        if not abs_path.exists():
            raise FileNotFoundInRepoError(f"{self._rel(abs_path)}: does not exist")
        if not abs_path.is_dir():
            raise NotAFileError(f"{self._rel(abs_path)}: is a file, not a directory")
        return abs_path

    def _rel(self, abs_path: Path) -> str:
        """Repository-relative, POSIX separators, on every platform."""
        try:
            return abs_path.relative_to(self.root).as_posix()
        except ValueError:
            return abs_path.as_posix()

    # --- list_files --------------------------------------------------------

    def list_files(
        self,
        subdir: str | None = None,
        *,
        pattern: str | None = None,
    ) -> ListResult:
        """List repository files, optionally under ``subdir`` or matching ``pattern``.

        Walks via ``iter_files`` -- the same traversal the symbol indexer uses.
        A second, independent walk would eventually disagree about which files
        the repository contains, and that disagreement would confound the
        ablation that removes the symbol index.

        ``pattern`` is matched against the repository-relative path with
        ``fnmatch``, whose ``*`` crosses directory separators. ``*.py`` therefore
        matches ``pkg/sub/mod.py``, which is not what a shell would do. This is
        the more forgiving reading of a pattern a model guessed at, and the
        looser behaviour fails toward showing too much rather than silently
        showing nothing.
        """
        cfg = self.config
        base = self._resolve_dir(subdir) if subdir else self.root

        rels: list[str] = []
        for abs_path in iter_files(base):
            rel = self._rel(abs_path)
            if pattern and not fnmatch.fnmatch(rel, pattern):
                continue
            rels.append(rel)

        rels.sort()
        total = len(rels)

        kept: list[str] = []
        used = 0
        truncated = False
        for rel in rels:
            if len(kept) >= cfg.max_list_entries or used + len(rel) + 1 > cfg.max_response_chars:
                truncated = True
                break
            kept.append(rel)
            used += len(rel) + 1

        body = "\n".join(kept)
        if truncated:
            body += f"\n… {total - len(kept)} more files not shown ({total} total)"
        elif not kept:
            body = "(no files matched)"

        return ListResult(
            paths=tuple(kept), total_found=total, truncated=truncated, text=body
        )

    # --- read_file ---------------------------------------------------------

    def read_file(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> ReadResult:
        """Read a line range, numbered 1-based and inclusive on both ends.

        A range is required by default. The alternative -- defaulting to the
        whole file -- would make the common case an unbounded read, and would
        remove the very behaviour whose value is being measured.
        """
        cfg = self.config
        abs_path = self._resolve_file(path)
        rel = self._rel(abs_path)

        size = abs_path.stat().st_size
        if size > cfg.max_file_bytes:
            raise FileTooLargeError(
                f"{rel}: {size / 1e6:.1f} MB exceeds the per-file read ceiling of "
                f"{cfg.max_file_bytes / 1e6:.0f} MB"
            )

        if is_probably_binary(abs_path):
            raise BinaryFileError(f"{rel}: binary content, cannot be read as text")

        raw = abs_path.read_bytes()
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise FileDecodeError(f"{rel}: not valid UTF-8") from exc

        all_lines = content.splitlines()
        total = len(all_lines)

        start, end = self._resolve_range(rel, start_line, end_line, total)

        clamped = end > total
        end_actual = min(end, total)

        # The only place a 1-based inclusive range meets a 0-based half-open
        # Python slice. Subtracting 1 here and nowhere else is what keeps the
        # contract honest.
        window = all_lines[start - 1 : end_actual]

        kept: list[str] = []
        rendered: list[str] = []
        used = 0
        truncated = False
        width = len(str(end_actual))

        for offset, line in enumerate(window):
            if len(kept) >= cfg.max_read_lines:
                truncated = True
                break
            clipped = _clip(line, cfg.max_line_chars)
            shown = f"{start + offset:>{width}} | {clipped}" if cfg.number_lines else clipped
            if used + len(shown) + 1 > cfg.max_response_chars and kept:
                truncated = True
                break
            kept.append(line)
            rendered.append(shown)
            used += len(shown) + 1

        last_line = start + len(kept) - 1 if kept else start
        text = self._render_read(rel, start, last_line, total, rendered, clamped, truncated)

        return ReadResult(
            path=rel,
            start_line=start,
            end_line=last_line,
            total_lines=total,
            lines=tuple(kept),
            clamped=clamped,
            truncated=truncated or clamped,
            text=text,
        )

    def _resolve_range(
        self, rel: str, start_line: int | None, end_line: int | None, total: int
    ) -> tuple[int, int]:
        cfg = self.config

        if start_line is None and end_line is None:
            if cfg.require_line_range:
                raise LineRangeRequiredError(
                    f"{rel}: read_file requires start_line and end_line "
                    f"(the file has {total} lines)"
                )
            return 1, total

        if start_line is None:
            raise InvalidLineRangeError(
                f"{rel}: end_line given without start_line; a range needs both ends"
            )

        if start_line < 1:
            raise InvalidLineRangeError(
                f"{rel}: start_line={start_line}. Line numbers are 1-based."
            )

        if end_line is None:
            end_line = min(total, start_line + cfg.max_read_lines - 1)

        if end_line < start_line:
            raise InvalidLineRangeError(
                f"{rel}: end_line={end_line} precedes start_line={start_line}"
            )

        if start_line > total:
            # Clamping the start would silently return lines other than the ones
            # requested. Nothing is readable here, so say so with the number the
            # caller needs in order to retry.
            raise InvalidLineRangeError(
                f"{rel}: start_line={start_line} is past the end of the file "
                f"({total} lines)"
            )

        return start_line, end_line

    def _render_read(
        self,
        rel: str,
        start: int,
        last: int,
        total: int,
        rendered: list[str],
        clamped: bool,
        truncated: bool,
    ) -> str:
        """Assemble model-facing text.

        When numbering is off, every line-number reference is suppressed as
        well. A header or a marker stating the offset would restore exactly the
        information the ablation exists to remove.
        """
        body = "\n".join(rendered)

        if not self.config.number_lines:
            if truncated:
                body += "\n… output truncated"
            return body

        header = f"{rel} lines {start}-{last} of {total}"
        parts = [header, body]
        if truncated and last < total:
            parts.append(f"… truncated. Continue with start_line={last + 1}.")
        elif clamped:
            parts.append(f"… end of file at line {total}.")
        return "\n".join(p for p in parts if p)

    # --- search_code -------------------------------------------------------

    def search_code(
        self,
        pattern: str,
        *,
        path: str | None = None,
        ignore_case: bool = False,
        fixed_strings: bool = False,
    ) -> SearchResult:
        """Search file contents with ripgrep.

        The flags exist to make ripgrep's file set match ``iter_files`` exactly.
        By default rg skips hidden paths and honours ``.gitignore``, while
        ``os.walk`` does neither -- so out of the box ``search_code`` would
        return hits in files ``list_files`` never shows, and miss files it does.
        Two traversals disagreeing about the repository is the precise failure
        the symbol layer already refused to allow.

        ``--no-config`` is not paranoia. A ``RIPGREP_CONFIG_PATH`` on one machine
        and not another would silently change results between runs, and nothing
        in the output would record it.

        Binary files are skipped by ripgrep's own default, which is also what
        keeps binary content out of the returned matches.
        """
        cfg = self.config
        target = self._resolve_dir(path) if path and (self.root / path).is_dir() else (
            self._resolve_file(path) if path else self.root
        )

        cmd = [
            self._rg,
            "--json",
            "--hidden",
            "--no-ignore",
            "--no-follow",
            "--no-config",
            "--case-sensitive" if not ignore_case else "--ignore-case",
        ]
        for skip in sorted(SKIP_DIRS):
            cmd += ["--glob", f"!**/{skip}/**"]
        if fixed_strings:
            cmd.append("--fixed-strings")
        cmd += ["--regexp", pattern, "--", str(target)]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                encoding="utf-8",
                errors="strict",
                timeout=cfg.search_timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise SearchFailedError(
                f"ripgrep exceeded {cfg.search_timeout_s}s for pattern {pattern!r}"
            ) from exc
        except UnicodeDecodeError as exc:
            raise SearchFailedError("ripgrep emitted output that is not valid UTF-8") from exc

        # rg exits 1 for "no matches found", which is an answer, not a failure.
        if proc.returncode not in (0, 1):
            raise SearchFailedError(
                f"ripgrep exited {proc.returncode} for pattern {pattern!r}: "
                f"{proc.stderr.strip() or '<no stderr>'}"
            )

        matches = self._parse_rg(proc.stdout)
        total = len(matches)

        kept: list[Match] = []
        rendered: list[str] = []
        used = 0
        truncated = False
        for m in matches:
            if len(kept) >= cfg.max_search_matches:
                truncated = True
                break
            line = f"{m.path}:{m.line_number}: {m.line_text}"
            if used + len(line) + 1 > cfg.max_response_chars and kept:
                truncated = True
                break
            kept.append(m)
            rendered.append(line)
            used += len(line) + 1

        body = "\n".join(rendered) if rendered else "(no matches)"
        if truncated:
            body += f"\n… {total - len(kept)} more matches not shown ({total} total)"

        return SearchResult(
            pattern=pattern,
            matches=tuple(kept),
            total_found=total,
            truncated=truncated,
            text=body,
        )

    def _parse_rg(self, stdout: str) -> list[Match]:
        """Read ripgrep's JSON stream.

        Unparseable output raises. A malformed line skipped quietly would mean
        a search silently returning fewer results than exist, which is
        indistinguishable in a results table from a search that found nothing.
        """
        out: list[Match] = []
        for raw in stdout.splitlines():
            if not raw.strip():
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise SearchFailedError(f"unparseable ripgrep output: {raw[:120]!r}") from exc

            if obj.get("type") != "match":
                continue
            data = obj["data"]

            path_text = data.get("path", {}).get("text")
            if path_text is None:
                # Path is not valid UTF-8; rg reports it as bytes. Nothing
                # downstream can cite it, so it is not a usable result.
                continue

            line_text = data.get("lines", {}).get("text")
            if line_text is None:
                continue

            out.append(
                Match(
                    path=self._rel(Path(path_text)),
                    line_number=int(data["line_number"]),
                    line_text=_clip(line_text.rstrip("\r\n"), self.config.max_line_chars),
                )
            )
        return out

    # --- get_symbols -------------------------------------------------------

    def get_symbols(self, path: str) -> SymbolsResult:
        """Every definition in one file, with its line range.

        The path is required. A whole-repo dump runs to thousands of entries on
        a real repository, and a tool that exists to save steps would end the
        run by consuming the context needed to read code.

        Ranges come from the index, so they are the same 1-based inclusive
        numbers ``read_file`` accepts. The model can go from a name to a read
        without counting anything.
        """
        cfg = self.config
        if self._index is None:
            raise SymbolsUnavailableError(
                "this repository has no symbol index (no parseable Python). "
                "Use list_files and search_code instead."
            )

        abs_path = self._resolve_file(path)
        rel = self._rel(abs_path)
        found = self._index.by_file(rel)
        total = len(found)

        kept: list[Symbol] = []
        rendered: list[str] = []
        used = 0
        truncated = False
        for sym in found:
            line = (
                f"{sym.qualname} ({sym.kind.value}) "
                f"lines {sym.start_line}-{sym.end_line} | {sym.signature}"
            )
            if len(kept) >= cfg.max_symbol_entries or (
                used + len(line) + 1 > cfg.max_response_chars and kept
            ):
                truncated = True
                break
            kept.append(sym)
            rendered.append(line)
            used += len(line) + 1

        if rendered:
            body = "\n".join(rendered)
        elif rel.endswith(".py"):
            body = "(no definitions in this file)"
        else:
            body = "(not a Python file; the symbol index covers .py only)"
        if truncated:
            body += f"\n… {total - len(kept)} more not shown ({total} total)"

        return SymbolsResult(
            path=rel, symbols=tuple(kept), total_found=total,
            truncated=truncated, text=body,
        )