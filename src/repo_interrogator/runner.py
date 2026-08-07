"""One repository, end to end: clone at a pin, index, run, record.

WHY THIS IS A MODULE AND NOT A SCRIPT
-------------------------------------
Every measurement this project reports travels through this path. A sweep over
ten repositories calls it ten times, an ablation calls it once per
configuration, and a comparison calls it once per model. Code that a results row
depends on has to be importable, parameterised and unchanged between calls --
not edited in place between runs, which is what a script invites.

The script under ``scripts/`` parses arguments and delegates. It holds no
behaviour of its own.

WHAT THIS LAYER OWNS
--------------------
Sequencing, and nothing else. It clones through the workspace context manager,
indexes through the symbol layer, constructs the agent through ``build_agent``,
runs it, and writes the trajectory to disk. Limits, tool configuration, the task
string, the question target and the model id all arrive from the caller. The pin
file remains the only source of URLs and SHAs.

Two policies live here because there is nowhere lower to put them: the held-out
guard, and the rule that every run leaves an artifact.

THE QUESTION TARGET TRAVELS WITH THE TASK
-----------------------------------------
The target reaches the model twice: written into the task string, and reported
by the agent on every tool reply. They have to agree, and the agent refuses a
task that does not name its target rather than running with the model told two
different numbers.

So ``n_questions`` is passed to ``build_agent`` here, not left at a default. A
caller that supplies its own ``task`` still supplies the matching target; the
agent's guard is what catches the caller that forgets. The default is imported
rather than written again, because two independently authored tens drift.

THE HELD-OUT GUARD
------------------
Four repositories in the pinned set are scored exactly twice: once as a
baseline before anything is tuned, once at the end. Between those two reads they
are never run, never inspected, never labelled and never debugged against. The
dev/held-out gap is the only number in this project that cannot be re-earned --
one careless invocation and the second read is measuring a repository that has
already informed the design.

Until now that discipline was protected by remembering, on every invocation, for
a month. This module refuses a held-out repository unless the caller passes
``allow_held_out`` explicitly, and when the caller does, it appends the read to
an append-only ledger before the clone begins.

The ledger is the point. A flag alone converts the discipline from "remembered"
to "typed", which is better but still leaves "scored exactly twice" as a claim.
With a ledger it is a fact on disk that can be counted, and if the count ever
reads three the run that should not have happened is named with its date.

It is written **before** the clone, not after the run. A read that crashed
halfway is still a read: the repository was fetched, its questions may have been
printed, and nothing about that is undone by a later exception. Recording only
successful reads would leave exactly the reads worth knowing about unrecorded.

TEXT-ONLY REPOSITORIES DEGRADE, THEY DO NOT ABORT
-------------------------------------------------
A repository with too little Python for structural extraction is cloned in
text-only mode and has no symbol index. The agent runs anyway, with five tools
instead of six: ``get_symbols`` returns an observation saying no index exists,
and the model is expected to reach for ``search_code`` instead. That path is
already tested.

Refusing would mean this layer enforcing a constraint the layer below handles
deliberately. The degradation is logged at warning level and recorded on the
result, so a run with five tools is never mistaken for a run with six.

EVERY RUN LEAVES AN ARTIFACT, INCLUDING THE ONES THAT FAIL
----------------------------------------------------------
A run that breaches a budget costs exactly what a run that finishes costs. It is
also the more informative of the two: the trajectory is the only thing that says
where the tokens went, whether files were read twice, or whether one directory
listing was resent twenty times.

The write therefore happens in a ``finally``, from metadata assembled *before*
the agent is started. Nothing in that metadata -- the pin, the mode, the
measurement, the index counts, the limits -- depends on the run succeeding, so
there is no reason for a failure to take it down with the frame.

This does not soften the rule that a partial result is never returned as a
result. The exception still propagates, ``run_repo`` still returns nothing on
the failure path, and a failed run is a ``FailedRun`` rather than a
``RunRecord``: a different type, carrying no ``RunResult``, and carrying an
``outcome`` field that names which ceiling was hit. A results table built from
``RunRecord`` cannot accidentally contain a truncated run, and a trace file
cannot be read as a finished one -- the outcome is in the payload and ``-failed``
is in the filename.

A failed run can now carry questions. Writing is no longer the same act as
stopping, so a run that recorded seven questions and then hit a ceiling is a
different failure from one that recorded none. ``FailedRun`` reports the count
in its row and the questions in its trace, and it is still a ``FailedRun``:
questions produced by a run that never finished are evidence about the run, not
a result.

THE TRAJECTORY IS WRITTEN TO DISK
---------------------------------
Messages are serialised through their own ``model_dump``, not through ``str``.
A rendered message loses its block structure -- and on this provider a pure tool
call returns content as a list of zero blocks, with the thought signature living
in ``additional_kwargs``. Both facts are invisible in a rendered string, and both
are things the trace store will have to represent.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage

from .agent import (
    DEFAULT_N_QUESTIONS,
    Question,
    RunLimits,
    RunProgress,
    RunResult,
    build_agent,
    default_task,
)
from .cloner import CloneLimits, ClonedRepo, cloned_repo
from .errors import (
    HeldOutReadError,
    LedgerUnwritableError,
    NoFinishError,
    StepBudgetExceededError,
    TokenBudgetExceededError,
)
from .repos import RepoEntry
from .symbols import SymbolIndex, build_symbol_index
from .tools import ToolConfig

log = logging.getLogger(__name__)

DEFAULT_LEDGER = Path("docs/held-out-reads.md")
"""Append-only. Every held-out read, dated, one line each.

A file of its own rather than a stanza in the lab notes. The lab notes are a
narrative record read front to back; this is a ledger whose only job is to be
counted. Mixing them would mean answering "how many times has httpx been read"
by reading prose.
"""

_LEDGER_HEADER = """\
# Held-out reads

Append-only. One line per read of a held-out repository.

These four repositories are scored exactly twice: once as a baseline before any
tuning, once at the end. This file exists so that "exactly twice" is a fact that
can be counted rather than a claim that has to be remembered. A line is written
before the clone begins, so a read that failed partway through still appears.

If this file ever shows a third read of the same repository, the dev/held-out
comparison it feeds is no longer a held-out comparison, and the results that
depend on it say so.

| date (UTC) | repo | sha | model | reason |
|---|---|---|---|---|
"""

COMPLETED = "completed"
"""The one outcome that means a ``RunResult`` exists."""

TOOLS_WITH_INDEX = 6
"""list_files, get_symbols, read_file, search_code, record_questions, finish."""

TOOLS_WITHOUT_INDEX = TOOLS_WITH_INDEX - 1
"""``get_symbols`` still exists as a tool; it returns an observation saying so.

Counted as absent because what the number is for is telling two runs apart in a
results table, and a tool that can only refuse is not one the model can use.
"""

_OUTCOMES: dict[type[BaseException], str] = {
    TokenBudgetExceededError: "token-budget",
    StepBudgetExceededError: "step-budget",
    NoFinishError: "no-finish",
}
"""Short outcome names for the failures this project expects to see often.

Anything else records its own class name. A run killed by a provider 400, a
containment breach or a keyboard interrupt is still worth a trace, and naming it
by type keeps the outcome column groupable without pretending to have foreseen
every failure.

``no-finish`` now covers two cases the agent distinguishes in its message: a run
that never called the finishing tool, and one that called it holding nothing.
Both are runs that produced no result and both are worth the same column; the
message says which.
"""


def _outcome_of(exc: BaseException) -> str:
    return _OUTCOMES.get(type(exc), type(exc).__name__)


def _messages_as_dicts(messages: list[BaseMessage] | tuple[BaseMessage, ...]) -> list[dict[str, Any]]:
    """Messages as structured objects, never as rendered text.

    ``model_dump`` keeps the content blocks and ``additional_kwargs``. The
    thought signature this provider requires on every subsequent call lives in
    the latter, and a rendered string does not contain it.
    """
    out: list[dict[str, Any]] = []
    for msg in messages:
        try:
            out.append(msg.model_dump())
        except Exception as exc:  # noqa: BLE001
            # Never silently drop a turn. A trajectory missing a message is
            # worse than one carrying a marker saying which message is missing
            # and why.
            out.append(
                {
                    "__serialisation_failed__": True,
                    "type": type(msg).__name__,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return out


@dataclass
class RunMeta:
    """Everything about a run that is known before the model is called.

    Assembled up front rather than at the end. None of it depends on the run
    succeeding, so holding it until after ``agent.run`` returns would mean a
    breached run losing its pin, its mode and its measurement along with its
    trajectory -- for no reason other than where the constructor happened to sit.
    """

    repo: str
    url: str
    sha: str
    group: str
    domain: str | None
    pinned_on: str | None

    model_id: str
    location: str
    task: str
    n_questions: int
    """The target the model was given, in the task string and on every tool reply.

    On the row because it is a configuration knob like any other. A run asked
    for five questions and a run asked for twenty are not comparable, and the
    question count alone does not say which was asked for.
    """

    mode: str
    measurement: dict[str, Any]

    symbols_indexed: int | None
    symbol_files: int | None
    symbol_failures: int
    tools_available: int
    """Six normally, five where no symbol index exists."""

    limits: dict[str, int]
    started_at: str

    def base_row(self) -> dict[str, Any]:
        """The columns a completed run and a failed run share."""
        row: dict[str, Any] = {
            "repo": self.repo,
            "sha": self.sha,
            "group": self.group,
            "pinned_on": self.pinned_on,
            "model_id": self.model_id,
            "location": self.location,
            "mode": self.mode,
            "n_questions_requested": self.n_questions,
            "tools_available": self.tools_available,
            "symbols_indexed": self.symbols_indexed,
            "symbol_failures": self.symbol_failures,
            "started_at": self.started_at,
        }
        row.update(self.limits)
        return row


@dataclass
class RunRecord(RunMeta):
    """A completed run: everything it was, cost and produced.

    Constructed only where a ``RunResult`` exists. The type is the guarantee --
    anything holding a ``RunRecord`` is holding a run that called ``finish``
    while holding at least one recorded question.
    """

    duration_s: float
    result: RunResult = field(repr=False)
    trace_path: Path | None = None

    outcome: str = COMPLETED

    def to_row(self) -> dict[str, Any]:
        """The flat row a results table wants. Excludes the trajectory."""
        row = self.base_row()
        row["outcome"] = self.outcome
        row["duration_s"] = round(self.duration_s, 2)
        row.update(self.result.to_row())
        return row

    def questions_as_dicts(self) -> list[dict[str, Any]]:
        return [q.model_dump() for q in self.result.questions]

    def trajectory_as_dicts(self) -> list[dict[str, Any]]:
        return _messages_as_dicts(self.result.messages)


@dataclass
class FailedRun(RunMeta):
    """A run that cost money and returned no result.

    A separate type from ``RunRecord`` on purpose. The two carry the same
    metadata and the same cost columns, so a single type with a nullable
    ``result`` would have been shorter -- and would have made every consumer
    responsible for remembering to check it. One that forgot would tabulate a
    truncated run beside finished ones and nothing would say so.

    ``questions`` is no longer empty by construction. Writing and stopping are
    separate acts now, so a run can record seven questions and then hit a
    ceiling. They are kept and written to the trace, because "read for thirty
    calls and produced nothing" and "produced seven and ran out of room" are
    different failures that a step count cannot tell apart. They are not a
    result and this is not a ``RunRecord``.
    """

    duration_s: float
    outcome: str
    error_type: str
    error_message: str
    progress: RunProgress = field(repr=False)
    questions: tuple[Question, ...] = field(default=(), repr=False)
    trace_path: Path | None = None

    def to_row(self) -> dict[str, Any]:
        row = self.base_row()
        row["outcome"] = self.outcome
        row["duration_s"] = round(self.duration_s, 2)
        row.update(self.progress.to_dict())
        # Deliberately not the same column a completed run fills. A failed run
        # has no result, and a table that summed the two would be counting
        # questions that were never returned.
        row["n_questions"] = 0
        row["error_type"] = self.error_type
        return row

    def questions_as_dicts(self) -> list[dict[str, Any]]:
        return [q.model_dump() for q in self.questions]

    def trajectory_as_dicts(self) -> list[dict[str, Any]]:
        return _messages_as_dicts(self.progress.messages)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _stamp(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def record_held_out_read(
    entry: RepoEntry,
    model_id: str,
    reason: str,
    *,
    ledger: Path = DEFAULT_LEDGER,
) -> None:
    """Append one line to the held-out ledger, or refuse to proceed.

    Failure to write is fatal on purpose. A read that happens without being
    recorded is precisely the read the ledger exists to catch, so "the ledger
    was unwritable" must stop the run rather than produce an unrecorded read
    with a warning nobody reads afterwards.
    """
    if not reason or not reason.strip():
        raise HeldOutReadError(
            f"{entry.name}: a held-out read requires a stated reason. "
            "An unexplained line in the ledger cannot be checked against the "
            "two reads the design allows."
        )

    try:
        ledger.parent.mkdir(parents=True, exist_ok=True)
        if not ledger.exists():
            ledger.write_text(_LEDGER_HEADER, encoding="utf-8")
        line = (
            f"| {_stamp(_utc_now())} | {entry.name} | {entry.short_sha} | "
            f"{model_id} | {reason.strip()} |\n"
        )
        with ledger.open("a", encoding="utf-8") as fh:
            fh.write(line)
    except OSError as exc:
        raise LedgerUnwritableError(
            f"{entry.name}: could not append to {ledger} ({exc}). "
            "A held-out read that cannot be recorded does not proceed."
        ) from exc

    log.warning(
        "HELD-OUT READ recorded: %s @ %s (%s). Appended to %s.",
        entry.name,
        entry.short_sha,
        reason.strip(),
        ledger,
    )


def _guard_held_out(
    entry: RepoEntry,
    model_id: str,
    *,
    allow_held_out: bool,
    reason: str | None,
    ledger: Path,
) -> None:
    """Refuse a held-out repository unless the caller asked for it in words."""
    if not entry.is_held_out:
        return
    if not allow_held_out:
        raise HeldOutReadError(
            f"{entry.name} is held out. It is scored exactly twice -- once as a "
            "baseline before any tuning, once at the end -- and running it at any "
            "other point makes the final comparison meaningless. Pass "
            "allow_held_out with a stated reason if this is one of those two reads."
        )
    record_held_out_read(entry, model_id, reason or "", ledger=ledger)


def _build_index(repo: ClonedRepo) -> SymbolIndex | None:
    """Index the tree, or return ``None`` where the repository has no Python.

    The mode check lives in ``build_symbol_index``. Repeating it here would mean
    two places deciding what text-only means, and the second one would eventually
    be the one that was not updated.
    """
    from .errors import SymbolIndexUnavailableError

    try:
        return build_symbol_index(repo)
    except SymbolIndexUnavailableError:
        log.warning(
            "%s: no symbol index (mode=%s). Running with %d usable tools; "
            "get_symbols will return an observation and the model must use "
            "search_code.",
            repo.name,
            repo.mode.value,
            TOOLS_WITHOUT_INDEX,
        )
        return None


def _build_meta(
    entry: RepoEntry,
    repo: ClonedRepo,
    index: SymbolIndex | None,
    *,
    model_id: str,
    location: str,
    task: str,
    n_questions: int,
    pinned_on: str | None,
    limits: RunLimits,
    started: datetime,
) -> RunMeta:
    return RunMeta(
        repo=entry.name,
        url=entry.url,
        sha=entry.sha,
        group=entry.group,
        domain=entry.domain,
        pinned_on=pinned_on,
        model_id=model_id,
        location=location,
        task=task,
        n_questions=n_questions,
        mode=repo.mode.value,
        measurement=repo.measurement.to_dict(),
        symbols_indexed=len(index.symbols) if index else None,
        symbol_files=index.files_indexed if index else None,
        symbol_failures=len(index.failures) if index else 0,
        tools_available=TOOLS_WITH_INDEX if index else TOOLS_WITHOUT_INDEX,
        limits=limits.to_dict(),
        started_at=_stamp(started),
    )


def run_repo(
    entry: RepoEntry,
    model_id: str,
    *,
    pinned_on: str | None = None,
    task: str | None = None,
    n_questions: int = DEFAULT_N_QUESTIONS,
    run_limits: RunLimits | None = None,
    clone_limits: CloneLimits | None = None,
    tool_config: ToolConfig | None = None,
    location: str = "global",
    project: str | None = None,
    workspace_root: Path | None = None,
    allow_held_out: bool = False,
    held_out_reason: str | None = None,
    ledger: Path = DEFAULT_LEDGER,
    trace_dir: Path | None = None,
) -> RunRecord:
    """Clone at the pin, index, run the agent, and return the whole record.

    The clone is scoped to the context manager, so the working tree is removed
    on the exception path as well as the success path.

    ``n_questions`` reaches both the task string and the agent. A caller
    supplying its own ``task`` must state the same number in it; the agent
    refuses the mismatch rather than running with the model told two different
    targets.

    An agent failure is recorded and then re-raised. Nothing here converts a
    breached run into a returned value: the caller either receives a
    ``RunRecord`` for a run that called ``finish``, or an exception. What the
    failure path adds is a trace file, written from metadata that was complete
    before the model was ever called.

    ``trace_dir`` of ``None`` means write nothing. The run still costs the same.
    """
    _guard_held_out(
        entry,
        model_id,
        allow_held_out=allow_held_out,
        reason=held_out_reason,
        ledger=ledger,
    )

    task = task if task is not None else default_task(n_questions)
    run_limits = run_limits or RunLimits()
    started = _utc_now()
    clock = time.perf_counter()

    with cloned_repo(
        entry.name,
        entry.url,
        entry.sha,
        limits=clone_limits,
        workspace_root=workspace_root,
    ) as repo:
        index = _build_index(repo)

        meta = _build_meta(
            entry,
            repo,
            index,
            model_id=model_id,
            location=location,
            task=task,
            n_questions=n_questions,
            pinned_on=pinned_on,
            limits=run_limits,
            started=started,
        )

        agent = build_agent(
            repo.path,
            model_id,
            symbol_index=index,
            tool_config=tool_config,
            limits=run_limits,
            n_questions=n_questions,
            location=location,
            project=project,
        )

        record: RunRecord | None = None
        failed: FailedRun | None = None
        try:
            result = agent.run(task)
            record = RunRecord(
                **vars(meta),
                duration_s=time.perf_counter() - clock,
                result=result,
            )
        except BaseException as exc:  # noqa: BLE001
            # Caught to record, never to handle. Broad on purpose: a provider
            # 400, a containment breach and a keyboard interrupt all leave a
            # trajectory worth reading, and narrowing this to the project's own
            # error types would keep exactly the surprising failures unrecorded.
            #
            # The recorded questions come off the agent for the same reason the
            # counters do: they say how far the run got, and nothing else will.
            failed = FailedRun(
                **vars(meta),
                duration_s=time.perf_counter() - clock,
                outcome=_outcome_of(exc),
                error_type=type(exc).__name__,
                error_message=str(exc),
                progress=agent.progress,
                questions=agent.recorded,
            )
            raise
        finally:
            target = record or failed
            if target is not None and trace_dir is not None:
                try:
                    path = write_trace(target, trace_dir)
                    target.trace_path = path
                    if failed is not None:
                        # Logged rather than printed: the caller is about to see
                        # the exception, and the trace path is the one thing the
                        # exception does not carry.
                        log.error(
                            "%s failed (%s) after recording %d questions. "
                            "Trajectory kept: %s",
                            target.repo,
                            target.outcome,
                            len(target.questions),
                            path,
                        )
                except OSError as write_exc:
                    # Never mask the original failure with a filesystem one.
                    log.error("could not write trace for %s: %s", target.repo, write_exc)

        return record


def write_trace(record: RunRecord | FailedRun, trace_dir: Path) -> Path:
    """Write the full run to JSON and return the path.

    The filename carries repo, short SHA and timestamp. Two runs of the same
    repository at the same commit differ only in configuration, and a file that
    overwrote its predecessor would destroy the comparison being set up.

    A run that did not complete is marked in the filename as well as in the
    payload. The payload is authoritative; the filename is so that a directory
    listing does not read as ten finished runs when three of them died.

    A failed run's ``questions`` are the ones it recorded before it stopped, not
    ones it returned. The ``outcome`` field is what tells the two apart, and it
    is read before the questions are.
    """
    trace_dir.mkdir(parents=True, exist_ok=True)
    safe_time = record.started_at.replace(":", "").replace("-", "")
    suffix = "" if record.outcome == COMPLETED else "-failed"
    path = trace_dir / f"{record.repo}-{record.sha[:8]}-{safe_time}{suffix}.json"

    payload: dict[str, Any] = {
        "outcome": record.outcome,
        "row": record.to_row(),
        "task": record.task,
        "url": record.url,
        "domain": record.domain,
        "measurement": record.measurement,
        "questions": record.questions_as_dicts(),
        "trajectory": record.trajectory_as_dicts(),
    }
    if isinstance(record, FailedRun):
        payload["error"] = {
            "type": record.error_type,
            "message": record.error_message,
        }

    # default=str is a backstop, not a strategy: anything reaching it is a type
    # the trace store will also have to handle, and it will be visible in the
    # file as a string that should have been structured.
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def format_questions(record: RunRecord) -> str:
    """The human-readable view. Eyeballing is the whole point of the first run."""
    lines: list[str] = []
    for i, q in enumerate(record.result.questions, start=1):
        lines.append(f"{i:>2}. {q.question}")
        for c in q.citations:
            lines.append(f"    {c.path}:{c.start_line}-{c.end_line}")
        lines.append("")
    return "\n".join(lines)


def format_summary(record: RunRecord) -> str:
    """One block naming the run and what it cost."""
    r = record.result
    tools = f"{record.tools_available} tools"
    if record.tools_available == TOOLS_WITHOUT_INDEX:
        tools += " (no symbol index)"
    symbols = (
        f"{record.symbols_indexed} symbols / {record.symbol_files} files"
        if record.symbols_indexed is not None
        else "no symbol index"
    )
    failures = (
        f", {record.symbol_failures} parse failures" if record.symbol_failures else ""
    )
    return (
        f"{record.repo} @ {record.sha[:8]} ({record.group}, mode={record.mode})\n"
        f"  model      {record.model_id} @ {record.location}\n"
        f"  index      {symbols}{failures}\n"
        f"  run        {tools}, {r.steps} steps, {r.tool_calls} tool calls "
        f"({r.tool_errors} errors)\n"
        f"  cost       {r.total_tokens} tokens, {record.duration_s:.1f}s\n"
        f"  produced   {len(r.questions)} of {record.n_questions} questions"
    )