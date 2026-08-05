"""Ask where each cited range came from, using only the trace.

No model, no clone, no network.

    uv run python scripts/check_provenance.py traces/
    uv run python scripts/check_provenance.py traces/ --uncovered
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from repo_interrogator.provenance import Provenance, trace_provenance

_ORDER = [p.value for p in Provenance]


def collect(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        if p.is_dir():
            out.extend(f for f in sorted(p.glob("*.json")) if "failed" not in f.name)
        else:
            out.append(p)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument(
        "--uncovered", action="store_true",
        help="list every citation nothing in the trajectory showed the model",
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    files = collect(args.paths)
    results = [r for r in (trace_provenance(f) for f in files) if r is not None]

    if not results:
        print("no runs with citations found")
        return 2

    header = "  " + f"{'run':<44} {'cites':>6} " + " ".join(f"{k:>13}" for k in _ORDER)
    print(header)
    for r in results:
        counts = r.counts()
        cells = " ".join(f"{counts.get(k, 0):>13}" for k in _ORDER)
        print(f"  {r.trace_path.name[:44]:<44} {r.n_citations:>6} {cells}")
        if r.header_fallbacks:
            print(
                f"      {r.header_fallbacks} read(s) had no returned range in the "
                "observation; the requested range was used instead"
            )

    total: Counter[str] = Counter()
    for r in results:
        total.update(r.counts())
    n = sum(total.values())

    print(f"\nacross {len(results)} run(s), {n} citations")
    for key in _ORDER:
        count = total.get(key, 0)
        print(f"  {key:>13}  {count:>4}  {count / n:>6.1%}")

    if args.uncovered:
        print("\nuncovered citations")
        found = False
        for r in results:
            for c in r.citations:
                if c.provenance is not Provenance.UNCOVERED:
                    continue
                found = True
                print(f"  {r.trace_path.name}  {c.path}:{c.start_line}-{c.end_line}")
        if not found:
            print("  none")

    if args.json:
        args.json.write_text(
            json.dumps([r.to_row() for r in results], indent=2), encoding="utf-8"
        )
        print(f"\nrows written to {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())