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
and runs it. Limits, tool configuration, the task string and the model id all
arrive from the caller. The pin file remains the only source of URLs and SHAs.

The one policy it does own is the held-out guard, below, because there is
nowhere lower to put it.

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
text-only mode and has no symbol index. The agent runs anyway, with four tools
instead of five: ``get_symbols`` returns an observation saying no index exists,
and the model is expected to reach for ``search_code`` instead. That path is
already tested.

Refusing would mean this layer enforcing a constraint the layer below handles
deliberately. The degradation is logged at warning level and recorded on the
result, so a run with four tools is never mistaken for a run with five.

THE TRAJECTORY IS WRITTEN TO DISK
---------------------------------
Every run costs money and produces one artifact. Printing it and discarding it
means paying twice to look at the same thing.

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

from .agent import RunLimits, RunResult, build_agent, default_task
from .cloner import CloneLimits, ClonedRepo, cloned_repo
from .errors import HeldOutReadError, LedgerUnwritableError
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


@dataclass
class RunRecord:
    """Everything one run was, cost and produced.

    Assembled rather than reconstructed later. A row that has to be pieced back
    together from a log file and a filename is a row whose provenance depends on
    the person doing the piecing.
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

    mode: str
    measurement: dict[str, Any]

    symbols_indexed: int | None
    symbol_files: int | None
    symbol_failures: int
    tools_available: int
    """Five normally, four where no symbol index exists."""

    limits: dict[str, int]
    started_at: str
    duration_s: float

    result: RunResult = field(repr=False)

    def to_row(self) -> dict[str, Any]:
        """The flat row a results table wants. Excludes the trajectory."""
        row: dict[str, Any] = {
            "repo": self.repo,
            "sha": self.sha,
            "group": self.group,
            "pinned_on": self.pinned_on,
            "model_id": self.model_id,
            "location": self.location,
            "mode": self.mode,
            "tools_available": self.tools_available,
            "symbols_indexed": self.symbols_indexed,
            "symbol_failures": self.symbol_failures,
            "started_at": self.started_at,
            "duration_s": round(self.duration_s, 2),
        }
        row.update(self.limits)
        row.update(self.result.to_row())
        return row

    def questions_as_dicts(self) -> list[dict[str, Any]]:
        return [q.model_dump() for q in self.result.questions]

    def trajectory_as_dicts(self) -> list[dict[str, Any]]:
        """Messages as structured objects, never as rendered text.

        ``model_dump`` keeps the content blocks and ``additional_kwargs``. The
        thought signature this provider requires on every subsequent call lives
        in the latter, and a rendered string does not contain it.
        """
        out: list[dict[str, Any]] = []
        for msg in self.result.messages:
            try:
                out.append(msg.model_dump())
            except Exception as exc:  # noqa: BLE001
                # Never silently drop a turn. A trajectory missing a message is
                # worse than one carrying a marker saying which message is
                # missing and why.
                out.append(
                    {
                        "__serialisation_failed__": True,
                        "type": type(msg).__name__,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        return out


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
            "%s: no symbol index (mode=%s). Running with four tools; get_symbols "
            "will return an observation and the model must use search_code.",
            repo.name,
            repo.mode.value,
        )
        return None


def run_repo(
    entry: RepoEntry,
    model_id: str,
    *,
    pinned_on: str | None = None,
    task: str | None = None,
    n_questions: int = 10,
    run_limits: RunLimits | None = None,
    clone_limits: CloneLimits | None = None,
    tool_config: ToolConfig | None = None,
    location: str = "global",
    project: str | None = None,
    workspace_root: Path | None = None,
    allow_held_out: bool = False,
    held_out_reason: str | None = None,
    ledger: Path = DEFAULT_LEDGER,
) -> RunRecord:
    """Clone at the pin, index, run the agent, and return the whole record.

    The clone is scoped to the context manager, so the working tree is removed
    on the exception path as well as the success path. Nothing here catches an
    agent failure: a run that breached a budget or never called ``finish``
    raises, and a partial result is never returned as a result.
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

        agent = build_agent(
            repo.path,
            model_id,
            symbol_index=index,
            tool_config=tool_config,
            limits=run_limits,
            location=location,
            project=project,
        )
        result = agent.run(task)

        return RunRecord(
            repo=entry.name,
            url=entry.url,
            sha=entry.sha,
            group=entry.group,
            domain=entry.domain,
            pinned_on=pinned_on,
            model_id=model_id,
            location=location,
            task=task,
            mode=repo.mode.value,
            measurement=repo.measurement.to_dict(),
            symbols_indexed=len(index.symbols) if index else None,
            symbol_files=index.files_indexed if index else None,
            symbol_failures=len(index.failures) if index else 0,
            tools_available=5 if index else 4,
            limits=run_limits.to_dict(),
            started_at=_stamp(started),
            duration_s=time.perf_counter() - clock,
            result=result,
        )


def write_trace(record: RunRecord, trace_dir: Path) -> Path:
    """Write the full run to JSON and return the path.

    The filename carries repo, short SHA and timestamp. Two runs of the same
    repository at the same commit differ only in configuration, and a file that
    overwrote its predecessor would destroy the comparison being set up.
    """
    trace_dir.mkdir(parents=True, exist_ok=True)
    safe_time = record.started_at.replace(":", "").replace("-", "")
    path = trace_dir / f"{record.repo}-{record.sha[:8]}-{safe_time}.json"

    payload = {
        "row": record.to_row(),
        "task": record.task,
        "url": record.url,
        "domain": record.domain,
        "measurement": record.measurement,
        "questions": record.questions_as_dicts(),
        "trajectory": record.trajectory_as_dicts(),
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
    if record.tools_available == 4:
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
        f"  produced   {len(r.questions)} questions"
    )