"""Check the citations in completed runs against the commits they were pinned to.

No model is called. Each distinct pinned commit is cloned once, however many
traces refer to it.

    uv run python scripts/verify_traces.py traces/
    uv run python scripts/verify_traces.py traces/nids-*.json --repos repos.yaml
    uv run python scripts/verify_traces.py traces/ --failures
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from repo_interrogator.verifier import summarise, verify_traces


def collect(paths: list[Path]) -> list[Path]:
    """Expand directories to the trace files inside them.

    Failed runs are written with a marked filename and carry no questions. They
    are left out here rather than reported as skipped for every invocation --
    a directory of traces normally contains several, and listing them each time
    would bury the traces that were genuinely unverifiable.
    """
    out: list[Path] = []
    for p in paths:
        if p.is_dir():
            out.extend(f for f in sorted(p.glob("*.json")) if "failed" not in f.name)
        else:
            out.append(p)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="trace files or a directory")
    parser.add_argument(
        "--repos", type=Path, default=None,
        help="pin file to cross-check each trace's commit against",
    )
    parser.add_argument("--failures", action="store_true", help="print every failed citation")
    parser.add_argument("--json", type=Path, default=None, help="write rows to this path")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    paths = collect(args.paths)
    if not paths:
        print("no trace files found")
        return 2

    verdicts, skipped = verify_traces(paths, repos_path=args.repos)

    print(f"{len(verdicts)} run(s) verified, {len(skipped)} skipped\n")

    print(f"  {'run':<44} {'cites':>6} {'ok':>4} {'rate':>7}  failures")
    for v in verdicts:
        rate = v.resolution_rate
        fails = v.failure_counts()
        detail = ", ".join(f"{k}={n}" for k, n in sorted(fails.items())) or "-"
        print(
            f"  {v.trace_path.name[:44]:<44} {v.n_citations:>6} {v.n_resolved:>4} "
            f"{rate:>6.1%}  {detail}" if rate is not None
            else f"  {v.trace_path.name[:44]:<44} {'-':>6} {'-':>4} {'-':>7}  no citations"
        )
        if v.pin_mismatch:
            print(f"      pin file says {v.pin_mismatch[:8]}, this run used {v.sha[:8]}")
        if not v.symbol_index_available:
            print("      no symbol index for this repository; symbol overlap not recorded")

    for summary in summarise(verdicts):
        if summary.n_runs < 2:
            continue
        low, high = summary.spread
        print(
            f"\n{summary.repo} @ {summary.sha[:8]} over {summary.n_runs} runs: "
            f"mean {summary.mean:.1%}, range {low:.1%}-{high:.1%}"
        )

    if args.failures:
        print("\nfailed citations")
        any_failed = False
        for v in verdicts:
            for q in v.questions:
                for c in q.citations:
                    if c.resolved:
                        continue
                    any_failed = True
                    print(f"  {v.trace_path.name}  Q{q.index + 1}")
                    print(f"    {c.path}:{c.start_line}-{c.end_line}  {c.status.value}")
                    if c.detail:
                        print(f"    {c.detail}")
        if not any_failed:
            print("  none")

    if skipped:
        print("\nskipped")
        for reason in skipped:
            print(f"  {reason}")

    if args.json:
        args.json.write_text(
            json.dumps([v.to_row() for v in verdicts], indent=2), encoding="utf-8"
        )
        print(f"\nrows written to {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())