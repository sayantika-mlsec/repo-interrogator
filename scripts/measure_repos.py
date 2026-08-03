"""Measure every pinned repository and record the figures.

    uv run python scripts/measure_repos.py

The cap bounds cannot be chosen responsibly without seeing the distribution they
are meant to sit above. Setting a number first is the mistake the two-bound
change exists to correct, so this runs before the bounds are fixed and the
bounds are read off its output.

Clones run with cap enforcement disabled. That is the mode's entire purpose: a
repository the caps would reject is precisely the one whose figure is needed,
and under enforcement it would raise instead of reporting. With no bounds in
play the walk runs to completion, so every figure here is exact rather than a
lower bound.

Two outputs. A JSON record, which is what a later run reads; and a markdown
table, which is what a person reads when deciding where the bounds go.

Per-repository largest files are recorded alongside the totals. The rejection
that prompted all of this only became legible when someone listed the four
biggest files in the tree and saw they were animated GIFs. A survey reporting
only aggregates would reproduce exactly the blindness it exists to remove.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from repo_interrogator.cloner import CloneLimits, cloned_repo, git_available
from repo_interrogator.errors import RepoInterrogatorError
from repo_interrogator.fsutil import is_probably_binary, iter_files
from repo_interrogator.repos import RepoEntry, load_entries

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
log = logging.getLogger("survey")

PLACEHOLDER = "<FILL>"
TOP_FILES = 6


def largest_files(root: Path, n: int = TOP_FILES) -> list[dict[str, object]]:
    """The n largest files in the tree, each marked readable or not.

    Same traversal and same readability predicate the cap uses, so a file listed
    here as unreadable is a file the cap did not count.
    """
    sized: list[tuple[int, Path]] = []
    for path in iter_files(root):
        try:
            sized.append((path.stat().st_size, path))
        except OSError:
            continue
    sized.sort(key=lambda pair: pair[0], reverse=True)

    out: list[dict[str, object]] = []
    for size, path in sized[:n]:
        out.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": size,
                "readable": not is_probably_binary(path),
            }
        )
    return out


def measure_one(entry: RepoEntry) -> dict[str, object]:
    """Clone at the pin, measure without enforcing, and return one record."""
    limits = CloneLimits(enforce_caps=False)
    with cloned_repo(entry.name, entry.url, entry.sha, limits=limits) as repo:
        measurement = repo.measurement
        record: dict[str, object] = {
            "name": entry.name,
            "group": entry.group,
            "domain": entry.domain,
            "url": entry.url,
            "sha": entry.sha,
            "mode": repo.mode.value,
            "largest_files": largest_files(repo.path),
        }
        record.update(measurement.to_dict())
        return record


def mb(value: object) -> str:
    return f"{float(value) / 1e6:.1f}"


def write_markdown(
    records: list[dict[str, object]],
    errors: list[dict[str, str]],
    pinned_on: str | None,
    path: Path,
) -> None:
    lines: list[str] = ["# Pinned repository measurements", ""]
    if pinned_on:
        lines += [f"Repositories pinned {pinned_on}. Measured with cap enforcement", ""]
        lines[-2] = (
            f"Repositories pinned {pinned_on}. Measured with cap enforcement disabled, "
            "so every figure is a complete walk rather than a lower bound."
        )
    lines += [
        "| Repo | Group | Tree MB | Readable MB | Readable % | Files | Readable files | Python files | Mode |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]

    for r in sorted(records, key=lambda x: (x["group"], -float(x["readable_bytes"]))):
        total = float(r["total_bytes"])
        readable = float(r["readable_bytes"])
        pct = f"{100 * readable / total:.0f}%" if total else "-"
        lines.append(
            f"| {r['name']} | {r['group']} | {mb(total)} | {mb(readable)} | {pct} | "
            f"{r['file_count']} | {r['readable_file_count']} | {r['python_file_count']} | "
            f"{r['mode']} |"
        )

    lines += ["", "## Largest files per repository", ""]
    for r in sorted(records, key=lambda x: x["name"]):
        lines.append(f"**{r['name']}**")
        lines.append("")
        for f in r["largest_files"]:  # type: ignore[index]
            kind = "text" if f["readable"] else "binary"
            lines.append(f"- `{f['path']}` — {mb(f['bytes'])} MB ({kind})")
        lines.append("")

    if errors:
        lines += ["## Not measured", ""]
        for e in errors:
            lines.append(f"- **{e['name']}** ({e['group']}) — {e['error']}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def summarise(records: list[dict[str, object]]) -> None:
    """Print the distribution the bounds have to sit above.

    Deliberately does not propose values. The bound is a judgement about how
    much readable content the agent should ever face, and reading it off a
    maximum would make the largest repository in the set define the design.
    """
    analysis = [r for r in records if r["group"] in ("dev", "held_out")]
    if not analysis:
        return

    print("\nreadable bytes, analysis set, ascending:")
    for r in sorted(analysis, key=lambda x: float(x["readable_bytes"])):
        print(f"  {mb(r['readable_bytes']):>7} MB readable   {r['name']}  ({r['group']})")

    print("\ntree bytes, analysis set, ascending:")
    for r in sorted(analysis, key=lambda x: float(x["total_bytes"])):
        print(f"  {mb(r['total_bytes']):>7} MB tree       {r['name']}  ({r['group']})")

    print("\nfile counts, analysis set, descending:")
    for r in sorted(analysis, key=lambda x: -int(x["file_count"])):  # type: ignore[arg-type]
        print(f"  {r['file_count']:>7} files      {r['name']}  ({r['group']})")

    print(
        "\nA bound admitting every repository above must exceed each maximum. "
        "Whether it should is the decision this survey informs, not one it makes."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repos", type=Path, default=Path("repos.yaml"))
    parser.add_argument("--out-dir", type=Path, default=Path("docs"))
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        help="measure only the named repo; repeatable",
    )
    args = parser.parse_args()

    if not git_available():
        print("git not found on PATH")
        return 2
    if not args.repos.is_file():
        print(f"repos file not found: {args.repos}")
        return 2

    entries, pinned_on = load_entries(args.repos)
    if args.only:
        entries = [e for e in entries if e.name in set(args.only)]

    records: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []

    for entry in entries:
        if PLACEHOLDER in (entry.url, entry.sha):
            # Recorded rather than skipped quietly. An unpinned repository is a
            # hole in the set, and the set is what the baseline is read against.
            errors.append(
                {
                    "name": entry.name,
                    "group": entry.group,
                    "error": "url or sha is still a placeholder in repos.yaml",
                }
            )
            log.error("%s: not pinned, url or sha is %s", entry.name, PLACEHOLDER)
            continue

        log.info("measuring %s (%s)", entry.name, entry.group)
        try:
            records.append(measure_one(entry))
        except RepoInterrogatorError as exc:
            # Collected, not fatal. Aborting on the third repository means
            # re-cloning the first two to learn anything about the fourth.
            errors.append(
                {"name": entry.name, "group": entry.group, "error": f"{type(exc).__name__}: {exc}"}
            )
            log.error("%s: %s", entry.name, exc)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "repo-measurements.json"
    md_path = args.out_dir / "repo-measurements.md"

    json_path.write_text(
        json.dumps(
            {"pinned_on": pinned_on, "measured": records, "errors": errors},
            indent=2,
        ),
        encoding="utf-8",
    )
    write_markdown(records, errors, pinned_on, md_path)

    summarise(records)
    print(f"\n{len(records)} measured, {len(errors)} not measured")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")

    if errors:
        # Non-zero because a survey missing a repository cannot freeze the set,
        # and a frozen set is the deliverable.
        print("\nsurvey incomplete: the repository set cannot be frozen")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())