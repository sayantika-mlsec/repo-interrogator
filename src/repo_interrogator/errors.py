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