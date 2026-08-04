"""Run the agent against one pinned repository.

    uv run python scripts/run_repo.py --repo nids --model <GA_MODEL_ID>

This file parses arguments and delegates. Every decision it looks like it makes
-- which repository, which model, which limits -- it passes straight through to
``repo_interrogator.runner``. Behaviour that lives in a script cannot be called
by a sweep, cannot be imported by a test, and changes whenever someone edits the
script to get a different run out of it.

Writing the trajectory is the runner's job, not this file's. A failed run has a
trajectory worth keeping, and a script that wrote the trace after a successful
call could only ever keep the successful ones. This file passes a directory and
prints where the file landed.

Held-out repositories are refused unless ``--score-held-out`` is passed together
with ``--reason``, and the read is appended to the ledger before the clone
begins. See the runner's module docstring for why the ledger exists.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from repo_interrogator.agent import RunLimits, default_task
from repo_interrogator.cloner import git_available
from repo_interrogator.errors import RepoInterrogatorError
from repo_interrogator.repos import load_repos
from repo_interrogator.runner import (
    DEFAULT_LEDGER,
    format_questions,
    format_summary,
    run_repo,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
log = logging.getLogger("run")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True, help="repository name from the pin file")
    p.add_argument(
        "--model",
        required=True,
        help="GA model id. Required and has no default: a run that does not "
        "state its model is not reproducible.",
    )
    p.add_argument("--repos", type=Path, default=Path("repos.yaml"))
    p.add_argument("--questions", type=int, default=10)
    p.add_argument(
        "--task",
        default=None,
        help="the human turn, sent verbatim. Overrides --questions entirely.",
    )
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--trace-dir", type=Path, default=Path("traces"))
    p.add_argument(
        "--no-trace",
        action="store_true",
        help="print only. The run still costs the same; nothing is kept, "
        "including the trajectory of a run that fails.",
    )
    p.add_argument("--workspace", type=Path, default=None)
    p.add_argument(
        "--score-held-out",
        action="store_true",
        help="permit a held-out repository. One of exactly two reads. Requires --reason.",
    )
    p.add_argument(
        "--reason",
        default=None,
        help="why this held-out read is happening. Written to the ledger.",
    )
    p.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    return p


def main() -> int:
    load_dotenv()
    args = build_parser().parse_args()

    if not git_available():
        print("git not found on PATH")
        return 2
    if not args.repos.is_file():
        print(f"pin file not found: {args.repos}")
        return 2
    if args.score_held_out and not args.reason:
        print("--score-held-out requires --reason: an unexplained ledger line cannot be checked")
        return 2

    repo_set = load_repos(args.repos)
    try:
        entry = repo_set.find(args.repo)
    except KeyError as exc:
        print(exc.args[0])
        return 2

    limits = RunLimits(
        **{
            k: v
            for k, v in (
                ("max_steps", args.max_steps),
                ("max_total_tokens", args.max_tokens),
            )
            if v is not None
        }
    )

    task = args.task if args.task is not None else default_task(args.questions)

    try:
        record = run_repo(
            entry,
            args.model,
            pinned_on=repo_set.pinned_on,
            task=task,
            run_limits=limits,
            location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
            project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
            workspace_root=args.workspace,
            allow_held_out=args.score_held_out,
            held_out_reason=args.reason,
            ledger=args.ledger,
            trace_dir=None if args.no_trace else args.trace_dir,
        )
    except RepoInterrogatorError as exc:
        # Named rather than traced. Every error in this project is one type per
        # failure mode, so the type is the diagnosis.
        #
        # Narrow on purpose, and not the same catch as the runner's. That one is
        # broad because it records and re-raises; this one is narrow because it
        # decides what a person sees. An unexpected type should still arrive
        # here as a traceback -- its trajectory is already on disk by the time
        # it reaches this frame.
        print(f"\n{type(exc).__name__}: {exc}")
        return 1

    print()
    print(format_summary(record))
    print()
    print(format_questions(record))

    if record.trace_path is not None:
        print(f"trace written to {record.trace_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())