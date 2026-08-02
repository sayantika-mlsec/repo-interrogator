"""Exception types.

One type per failure mode, deliberately. A single generic error would force
callers to parse message strings to find out what went wrong, and would make it
impossible to distinguish "this repo is too big" (skip it, keep going) from
"git is not installed" (stop everything).

Nothing here is ever caught and swallowed internally. Every one of these
propagates to the caller.
"""

from __future__ import annotations


class RepoInterrogatorError(Exception):
    """Root of every error this project raises. Never raised directly."""


# --- workspace layer -------------------------------------------------------


class WorkspaceError(RepoInterrogatorError):
    """Base class for the workspace layer. Never raised directly."""


class GitNotAvailableError(WorkspaceError):
    """The ``git`` executable was not found on PATH."""


class CloneFailedError(WorkspaceError):
    """git exited non-zero, or the requested commit could not be fetched."""


class CloneTimeoutError(WorkspaceError):
    """The clone exceeded its wall-clock budget and was killed."""


class InvalidShaError(WorkspaceError):
    """The caller supplied something that is not a full 40-hex commit id.

    Separate from ``ShaMismatchError``: this is a bad argument, that is a
    broken pin. Collapsing them would hide which one happened.
    """


class ShaMismatchError(WorkspaceError):
    """The checked-out commit is not the commit that was requested.

    This should be impossible. If it ever fires, the pin is meaningless and
    every number produced against this repo is suspect, so it is fatal rather
    than a warning.
    """


class RepoTooLargeError(WorkspaceError):
    """Working tree exceeds the configured byte ceiling."""


class ReadableBytesTooLargeError(WorkspaceError):
    """Text content the agent could actually read exceeds its ceiling.

    Distinct from ``RepoTooLargeError``, which guards the machine against an
    enormous clone. This one guards the agent against an enormous amount of
    readable surface. A repository of documentation videos trips the first and
    not the second; a monorepo of source trips the second and maybe not the
    first. Collapsing them would make the rejection reason unrecoverable.
    """


class TooManyFilesError(WorkspaceError):
    """Working tree exceeds the configured file-count ceiling."""


class UnsupportedLanguageError(WorkspaceError):
    """The repository does not contain enough parseable source to analyse."""


class PathEscapeError(WorkspaceError):
    """A requested path resolved outside the repository root.

    Raised by the containment check. Covers ``../`` traversal, absolute paths,
    and symlinks pointing out of the tree.
    """


# --- symbol layer ----------------------------------------------------------


class SymbolError(RepoInterrogatorError):
    """Base class for structural extraction. Never raised directly."""


class SymbolParseError(SymbolError):
    """A file could not be parsed into a usable syntax tree.

    Collected per-file by the index builder rather than aborting the run, but
    always counted and reported -- a repo whose symbol map is missing a third
    of its files must not look identical to one where everything parsed.
    """


class SymbolDecodeError(SymbolError):
    """A ``.py`` file is not valid UTF-8."""


class SymbolIndexUnavailableError(SymbolError):
    """A symbol index was requested for a repository that has none.

    Text-only repositories reach this. Raised rather than returning an empty
    index, because an empty index is indistinguishable from a repo that defines
    nothing.
    """

# --- tool layer ------------------------------------------------------------

class ToolError(RepoInterrogatorError):
    """Base class for the model-facing tools. Never raised directly.

    A separate branch from ``SymbolError`` on purpose. The symbol layer raises
    ``SymbolParseError`` for a missing file, which conflates "this path does not
    exist" with "this file is not Python". The agent loop needs to tell those
    apart: one is a recoverable mistake it should retry with a different path,
    the other is a fact about the repository.
    """


class ToolConfigurationError(ToolError):
    """The tools cannot be constructed at all.

    Raised at construction rather than on first use. A missing dependency
    discovered fifty steps into an agent run has already wasted the run.
    """


class RipgrepNotAvailableError(ToolConfigurationError):
    """The ``rg`` executable was not found on PATH.

    There is deliberately no Python-side fallback. A fallback would search a
    different file set with different regex semantics, so the same query would
    return different results depending on the machine -- and nothing in a
    results table would record which one ran.
    """


class FileNotFoundInRepoError(ToolError):
    """A path resolved inside the repository but does not exist."""


class NotAFileError(ToolError):
    """A path exists but is a directory."""


class BinaryFileError(ToolError):
    """A path points at binary content and cannot be read as text."""


class FileDecodeError(ToolError):
    """A file is text, but not valid UTF-8."""


class FileTooLargeError(ToolError):
    """A single file exceeds the per-file read ceiling."""


class LineRangeRequiredError(ToolError):
    """``read_file`` was called without a line range while one is required."""


class InvalidLineRangeError(ToolError):
    """A line range is malformed, or begins past the end of the file."""


class SearchFailedError(ToolError):
    """ripgrep exited with an error, or emitted output that could not be read."""

class SymbolsUnavailableError(ToolError):
    """``get_symbols`` was called where no symbol index exists.

    A ``ToolError`` and not a ``SymbolError`` on purpose: this reaches the model
    as an observation it can act on by switching to search, whereas the symbol
    layer's version of this is a fact about the repository that the caller must
    handle before a run starts.
    """

# --- agent layer -----------------------------------------------------------


class AgentError(RepoInterrogatorError):
    """Base class for the agent loop. Never raised directly."""


class AgentConfigurationError(AgentError):
    """The agent cannot be constructed. Raised before any spend."""


class StepBudgetExceededError(AgentError):
    """The run reached its model-call ceiling without finishing.

    Not a warning and not a partial result. A run that stopped short still
    produced questions, and those questions in a results table are
    indistinguishable from ones a completed run produced.
    """


class TokenBudgetExceededError(AgentError):
    """The run reached its token ceiling without finishing."""


class NoFinishError(AgentError):
    """The loop ended without ``finish`` being called, so nothing was produced."""