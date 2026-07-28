"""Exception types for the workspace layer.

One type per failure mode, deliberately. A single generic ``WorkspaceError``
would force callers to parse message strings to find out what went wrong, and
would make it impossible to distinguish "this repo is too big" (skip it, keep
going) from "git is not installed" (stop everything).

Nothing here is ever caught and swallowed internally. Every one of these
propagates to the caller.
"""

from __future__ import annotations


class WorkspaceError(Exception):
    """Base class. Never raised directly."""


class GitNotAvailableError(WorkspaceError):
    """The ``git`` executable was not found on PATH."""


class CloneFailedError(WorkspaceError):
    """git exited non-zero, or the requested commit could not be fetched."""


class CloneTimeoutError(WorkspaceError):
    """The clone exceeded its wall-clock budget and was killed."""


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