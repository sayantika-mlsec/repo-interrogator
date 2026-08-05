"""Resolve citations against the repository at the commit a run was pinned to.

WHY THIS LAYER HAS NO MODEL IN IT
---------------------------------
A question's answer can only be scored by another model, and a model scoring a
model is the arrangement this project exists to avoid. A citation is different:
it either points at a file that exists at the pinned commit, within a range that
file actually has, or it does not. That question is settled by reading a file,
so it is settled here.

Nothing in this module may ever call a model. There is no version of "just for
the ambiguous cases" that leaves the guarantee intact.

WHAT "RESOLVES" MEANS, AND WHAT IT DELIBERATELY DOES NOT
--------------------------------------------------------
A citation resolves when the file exists and the cited range lies inside it.
That is the whole test.

Whether the range overlaps a function or class definition is recorded on the
verdict and has no bearing on it. Requiring an overlap would fail correct
citations of documentation, configuration and module-level code, and the
resulting failure rate would mostly measure how much non-Python a run chose to
cite rather than whether it cited real code.

Recording the overlap separately keeps the second question askable without
letting it contaminate the first.

FAILURES ARE NAMED, NOT COUNTED TOGETHER
----------------------------------------
A path that does not exist, a range past the end of a real file, and a path that
escapes the repository are three different findings. The first says the model
invented a file, the second that it invented a location in a real one, and the
third is a containment concern that has nothing to do with grounding. A single
pass rate hides all three behind one number, and they call for different
responses.

CLONES ARE GROUPED BY PIN
-------------------------
A clone is a temporary directory that is deleted when its scope ends, so
verifying traces one at a time would re-fetch the same commit once per trace.
Traces are grouped by ``(url, sha)`` and each group is verified against one
working tree.

That also removes a class of doubt. Repeated runs of one repository are compared
against each other, and if each were checked against its own clone, a difference
between two runs could in principle come from a difference between two clones.
Grouping makes that impossible rather than unlikely.

A TRACE IS VERIFIABLE ON ITS OWN
--------------------------------
The URL and the commit come from the trace, never from the pin file, so a trace
stays checkable after the set is re-pinned. When a pin file is supplied its
commit is compared and any disagreement is reported as a finding -- a repository
re-pinned after a run was recorded is precisely the kind of change that
invalidates a results table without leaving any evidence in the table itself.
Verification still proceeds against the trace's own commit, because that is the
code the run actually read.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .cloner import AnalysisMode, CloneLimits, cloned_repo
from .errors import PathEscapeError
from .fsutil import resolve_within
from .repos import load_repos
from .symbols import SymbolIndex, build_index_at

log = logging.getLogger(__name__)


class CitationStatus(str, Enum):
    """The verdict on one citation. Exactly one applies."""

    RESOLVED = "resolved"
    """The file exists and the whole cited range lies inside it."""

    FILE_NOT_FOUND = "file_not_found"
    """No such path at the pinned commit."""

    NOT_A_FILE = "not_a_file"
    """The path exists but is a directory."""

    RANGE_PAST_EOF = "range_past_eof"
    """The file exists; the range extends beyond its last line."""

    PATH_ESCAPE = "path_escape"
    """The path resolves outside the repository root.

    Not a grounding failure. A citation is produced by the same model that was
    given contained tools, so this means either a tool boundary leaked or the
    model wrote a path it never read from. Either is worth knowing about on its
    own terms, which is why it is not folded into file_not_found.
    """

    UNREADABLE = "unreadable"
    """The file exists but is not decodable text, so it has no line count."""


@dataclass(frozen=True)
class CitationVerdict:
    """One citation, checked."""

    path: str
    start_line: int
    end_line: int
    status: CitationStatus
    total_lines: int | None = None
    """Lines in the cited file, where one could be counted."""

    symbols: tuple[str, ...] = ()
    """Qualified names of definitions the range overlaps. Recorded, never gating.

    Empty for a resolved citation into documentation, configuration, or
    module-level code, and empty for every citation in a repository with no
    symbol index. Absence is not a failure and must not be read as one.
    """

    detail: str = ""

    @property
    def resolved(self) -> bool:
        return self.status is CitationStatus.RESOLVED


@dataclass(frozen=True)
class QuestionVerdict:
    """One question and the verdicts on its citations."""

    index: int
    question: str
    citations: tuple[CitationVerdict, ...]

    @property
    def all_resolved(self) -> bool:
        return all(c.resolved for c in self.citations)

    @property
    def any_resolved(self) -> bool:
        return any(c.resolved for c in self.citations)


@dataclass(frozen=True)
class TraceVerdict:
    """Every citation in one run, checked against one working tree."""

    trace_path: Path
    repo: str
    sha: str
    url: str
    model_id: str
    questions: tuple[QuestionVerdict, ...]
    pin_mismatch: str | None = None
    """The pin file's commit, when it disagrees with the trace's. A finding."""

    symbol_index_available: bool = True
    """False for a repository with no symbol index. Makes an empty ``symbols``
    field on every verdict readable as absence of the index rather than absence
    of overlap."""

    @property
    def n_questions(self) -> int:
        return len(self.questions)

    @property
    def n_citations(self) -> int:
        return sum(len(q.citations) for q in self.questions)

    @property
    def n_resolved(self) -> int:
        return sum(c.resolved for q in self.questions for c in q.citations)

    @property
    def resolution_rate(self) -> float | None:
        """Resolved citations over all citations. ``None`` when there are none."""
        total = self.n_citations
        return self.n_resolved / total if total else None

    @property
    def n_questions_fully_resolved(self) -> int:
        return sum(q.all_resolved for q in self.questions)

    def failure_counts(self) -> Counter[str]:
        """Failures by kind. Resolved citations are not counted."""
        return Counter(
            c.status.value
            for q in self.questions
            for c in q.citations
            if not c.resolved
        )

    def to_row(self) -> dict[str, Any]:
        """One row of a results table."""
        row: dict[str, Any] = {
            "repo": self.repo,
            "sha": self.sha,
            "model_id": self.model_id,
            "n_questions": self.n_questions,
            "n_citations": self.n_citations,
            "n_resolved": self.n_resolved,
            "resolution_rate": self.resolution_rate,
            "n_questions_fully_resolved": self.n_questions_fully_resolved,
            "symbol_index_available": self.symbol_index_available,
            "pin_mismatch": self.pin_mismatch,
        }
        row.update(self.failure_counts())
        return row


@dataclass(frozen=True)
class LoadedTrace:
    """A completed run read from disk, with only what verification needs."""

    path: Path
    repo: str
    sha: str
    url: str
    model_id: str
    questions: tuple[dict[str, Any], ...]

    @property
    def pin(self) -> tuple[str, str]:
        """The clone key. Two traces sharing it are checked against one tree."""
        return (self.url, self.sha)


class TraceNotVerifiable(Exception):
    """A trace that cannot be checked, with the reason it cannot.

    Raised rather than returned as an empty verdict. A run that produced no
    questions and a run whose every citation failed are different facts, and a
    zero resolution rate would report them identically.
    """


# --- loading ---------------------------------------------------------------


def load_trace(path: Path) -> LoadedTrace:
    """Read one trace, or say why it cannot be verified."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TraceNotVerifiable(f"{path.name}: unreadable trace ({exc})") from exc

    row = data.get("row") or {}
    questions = data.get("questions")

    if not questions:
        outcome = data.get("outcome", "unknown")
        raise TraceNotVerifiable(
            f"{path.name}: no questions to check (outcome={outcome})"
        )

    for key, value in (("url", data.get("url")), ("repo", row.get("repo")), ("sha", row.get("sha"))):
        if not value:
            raise TraceNotVerifiable(f"{path.name}: trace has no {key}")

    return LoadedTrace(
        path=path,
        repo=row["repo"],
        sha=row["sha"],
        url=data["url"],
        model_id=row.get("model_id", ""),
        questions=tuple(questions),
    )


# --- the check -------------------------------------------------------------


def _count_lines(abs_path: Path) -> int | None:
    """Lines in a file, counted the way ``read_file`` counts them.

    ``splitlines`` on decoded text, identical to the read tool. A verifier that
    counted differently -- on bytes, or including a trailing empty line -- would
    reject ranges the model was shown as valid.

    Returns ``None`` when the file is not decodable text.
    """
    try:
        return len(abs_path.read_bytes().decode("utf-8").splitlines())
    except (OSError, UnicodeDecodeError):
        return None


def _overlapping_symbols(
    index: SymbolIndex | None, path: str, start: int, end: int
) -> tuple[str, ...]:
    """Definitions the cited range touches. Recorded, never gating."""
    if index is None:
        return ()
    return tuple(
        s.qualname
        for s in index.by_file(path)
        if s.start_line <= end and s.end_line >= start
    )


def verify_citation(
    root: Path, citation: dict[str, Any], index: SymbolIndex | None
) -> CitationVerdict:
    """Check one citation against the working tree.

    The path came from a model, so it goes through the same containment check
    the tools use before anything touches the filesystem.
    """
    path = str(citation.get("path", ""))
    start = int(citation.get("start_line", 0))
    end = int(citation.get("end_line", 0))

    try:
        abs_path = resolve_within(root, path)
    except PathEscapeError as exc:
        return CitationVerdict(path, start, end, CitationStatus.PATH_ESCAPE, detail=str(exc))

    if not abs_path.exists():
        return CitationVerdict(
            path, start, end, CitationStatus.FILE_NOT_FOUND,
            detail="no such path at this commit",
        )
    if not abs_path.is_file():
        return CitationVerdict(
            path, start, end, CitationStatus.NOT_A_FILE, detail="path is a directory",
        )

    total = _count_lines(abs_path)
    if total is None:
        return CitationVerdict(
            path, start, end, CitationStatus.UNREADABLE,
            detail="not decodable as UTF-8 text, so it has no line count",
        )

    symbols = _overlapping_symbols(index, path, start, end)

    if end > total:
        return CitationVerdict(
            path, start, end, CitationStatus.RANGE_PAST_EOF, total_lines=total,
            symbols=symbols,
            detail=f"cited through line {end}; the file has {total}",
        )

    return CitationVerdict(
        path, start, end, CitationStatus.RESOLVED, total_lines=total, symbols=symbols,
    )


def verify_trace_at(
    root: Path, trace: LoadedTrace, index: SymbolIndex | None, *, pin_mismatch: str | None = None
) -> TraceVerdict:
    """Check every citation in one trace against an already-cloned tree."""
    verdicts: list[QuestionVerdict] = []

    for i, raw in enumerate(trace.questions):
        cites = tuple(
            verify_citation(root, c, index) for c in (raw.get("citations") or [])
        )
        verdicts.append(
            QuestionVerdict(index=i, question=str(raw.get("question", "")), citations=cites)
        )

    return TraceVerdict(
        trace_path=trace.path,
        repo=trace.repo,
        sha=trace.sha,
        url=trace.url,
        model_id=trace.model_id,
        questions=tuple(verdicts),
        pin_mismatch=pin_mismatch,
        symbol_index_available=index is not None,
    )


# --- driving the clones ----------------------------------------------------


def group_by_pin(traces: list[LoadedTrace]) -> dict[tuple[str, str], list[LoadedTrace]]:
    """Traces sharing a URL and commit, so each group needs one clone."""
    groups: dict[tuple[str, str], list[LoadedTrace]] = {}
    for trace in traces:
        groups.setdefault(trace.pin, []).append(trace)
    return groups


def _pin_mismatch(trace: LoadedTrace, repos_path: Path | None) -> str | None:
    """The pin file's commit for this repository, when it disagrees.

    Reported, never enforced. The trace's own commit is the code the run read,
    and that is what the citations are checked against regardless.
    """
    if repos_path is None:
        return None
    try:
        entry = load_repos(repos_path).find(trace.repo)
    except (KeyError, OSError) as exc:
        log.warning("%s: could not be found in the pin file (%s)", trace.repo, exc)
        return None
    return None if entry.sha.lower() == trace.sha.lower() else entry.sha


def verify_traces(
    paths: list[Path],
    *,
    repos_path: Path | None = None,
    limits: CloneLimits | None = None,
) -> tuple[list[TraceVerdict], list[str]]:
    """Verify many traces, cloning once per pinned commit.

    Returns the verdicts and the traces that could not be checked, each with the
    reason. Unverifiable traces are returned rather than raised so that one
    failed run in a directory does not stop the rest from being scored.
    """
    loaded: list[LoadedTrace] = []
    skipped: list[str] = []

    for path in sorted(paths):
        try:
            loaded.append(load_trace(path))
        except TraceNotVerifiable as exc:
            skipped.append(str(exc))

    verdicts: list[TraceVerdict] = []

    for (url, sha), group in group_by_pin(loaded).items():
        name = group[0].repo
        log.info("cloning %s @ %s for %d trace(s)", name, sha[:8], len(group))
        try:
            with cloned_repo(name, url, sha, limits=limits) as repo:
                index = (
                    build_index_at(repo.path, repo.name, repo.sha)
                    if repo.mode is AnalysisMode.FULL
                    else None
                )
                for trace in group:
                    verdicts.append(
                        verify_trace_at(
                            repo.path, trace, index,
                            pin_mismatch=_pin_mismatch(trace, repos_path),
                        )
                    )
        except Exception as exc:  # noqa: BLE001
            # One unclonable pin must not discard the groups already verified,
            # and must not be silently absent from the output either.
            for trace in group:
                skipped.append(f"{trace.path.name}: clone failed ({type(exc).__name__}: {exc})")

    return verdicts, skipped


# --- reporting -------------------------------------------------------------


@dataclass
class RepoSummary:
    """Resolution across repeated runs of one repository at one commit.

    The spread is the point. A single run's rate is one draw; the range across
    identical runs is what any later comparison has to clear before a difference
    between configurations can be called a difference at all.
    """

    repo: str
    sha: str
    rates: list[float] = field(default_factory=list)

    @property
    def n_runs(self) -> int:
        return len(self.rates)

    @property
    def mean(self) -> float | None:
        return sum(self.rates) / len(self.rates) if self.rates else None

    @property
    def spread(self) -> tuple[float, float] | None:
        return (min(self.rates), max(self.rates)) if self.rates else None


def summarise(verdicts: list[TraceVerdict]) -> list[RepoSummary]:
    """Group verdicts by repository and commit, keeping every run's rate."""
    out: dict[tuple[str, str], RepoSummary] = {}
    for v in verdicts:
        rate = v.resolution_rate
        if rate is None:
            continue
        key = (v.repo, v.sha)
        out.setdefault(key, RepoSummary(repo=v.repo, sha=v.sha)).rates.append(rate)
    return list(out.values())