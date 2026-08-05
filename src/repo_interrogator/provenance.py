"""Where did each cited range come from?

THE QUESTION THIS ANSWERS
-------------------------
Citation resolution asks whether a cited range exists in the repository. On a
small repository every citation resolves, and a metric that never fails cannot
show damage when a tool is taken away either.

The likely explanation is structural rather than flattering: ``read_file`` and
``get_symbols`` only ever hand back ranges that are valid, so a citation copied
from one of them is valid by construction. If that is what is happening, the
resolution rate is measuring the tools and not the model.

This module separates the cases. For each citation it asks which earlier tool
call, if any, showed the model that range.

FOUR PROVENANCES, AND WHY THEY ARE NOT ONE
------------------------------------------
**Read.** The range lies inside something ``read_file`` returned. The model saw
the code. Resolution here is close to tautological.

**Symbol.** The range lies inside a definition ``get_symbols`` reported, and was
never read. The model knew where a function was without seeing its body. Any
question grounded here is grounded in a name and a location, not in behaviour.

**Search.** The range covers only lines ``search_code`` echoed as matches. The
model saw isolated lines out of context.

**Uncovered.** Nothing in the trajectory showed the model this range. These are
the citations the verifier is genuinely testing. If they resolve anyway, the
model is inferring valid line numbers rather than reporting observed ones --
which is worth knowing, and is invisible in a resolution rate that lumps them in
with the rest.

A partial category sits between read and uncovered, for a citation that extends
past what was read. Rounding it into either would hide the most interesting
shape: a citation anchored in real code that then reaches beyond it.

WHAT IS PARSED, AND WHY FROM THE OBSERVATION
--------------------------------------------
Ranges are taken from what the tools returned, not from what the model asked
for. A read can be clamped at end of file or truncated by a response bound, so
the requested range is an upper bound on what the model was actually shown.
Crediting the request would over-count coverage in exactly the cases where the
model saw least.

Where the returned range cannot be recovered -- under the ablation that removes
line numbers, ``read_file`` emits no header -- the requested range is used and
the trace is marked as having fallen back, so a coverage figure is never
silently mixing the two.

No model and no clone. This reads a trace and nothing else.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# `<path> lines <start>-<last> of <total>` -- read_file's header, numbered mode only.
_READ_HEADER = re.compile(r"^(?P<path>.+?) lines (?P<start>\d+)-(?P<end>\d+) of (?P<total>\d+)$")

# `<qualname> (<kind>) lines <start>-<end> | <signature>` -- one get_symbols entry.
_SYMBOL_LINE = re.compile(r"\blines (?P<start>\d+)-(?P<end>\d+) \|")

# `<path>:<line>: <text>` -- one search_code match.
_SEARCH_LINE = re.compile(r"^(?P<path>[^:]+):(?P<line>\d+):")


class Provenance(str, Enum):
    """Where a cited range came from. Exactly one applies."""

    READ = "read"
    """Entirely inside a range read_file returned."""

    PARTIAL_READ = "partial_read"
    """Overlaps a read range but extends beyond it."""

    SYMBOL = "symbol"
    """Entirely inside a definition get_symbols reported, and never read."""

    SEARCH = "search"
    """Covered only by lines search_code echoed as matches."""

    UNCOVERED = "uncovered"
    """Nothing in the trajectory showed the model this range."""


@dataclass
class Shown:
    """The line numbers a run was shown, per file, by each tool.

    Sets rather than ranges. A file read in three passes with a gap between them
    has a coverage shape no single interval describes, and treating it as one
    would credit the model with lines it never saw.
    """

    read: dict[str, set[int]] = field(default_factory=dict)
    symbol: dict[str, set[int]] = field(default_factory=dict)
    search: dict[str, set[int]] = field(default_factory=dict)
    header_fallbacks: int = 0
    """Reads whose returned range could not be recovered from the observation."""

    def add(self, bucket: dict[str, set[int]], path: str, start: int, end: int) -> None:
        if start < 1 or end < start:
            return
        bucket.setdefault(path, set()).update(range(start, end + 1))


@dataclass(frozen=True)
class CitationProvenance:
    """One citation, traced back to what the model was shown."""

    path: str
    start_line: int
    end_line: int
    provenance: Provenance
    read_fraction: float
    """Share of the cited lines that appeared in a read. 0.0 to 1.0."""

    @property
    def cited_lines(self) -> int:
        return self.end_line - self.start_line + 1


@dataclass(frozen=True)
class TraceProvenance:
    """Every citation in one run, traced."""

    trace_path: Path
    repo: str
    sha: str
    citations: tuple[CitationProvenance, ...]
    header_fallbacks: int = 0

    def counts(self) -> Counter[str]:
        return Counter(c.provenance.value for c in self.citations)

    @property
    def n_citations(self) -> int:
        return len(self.citations)

    @property
    def n_uncovered(self) -> int:
        return sum(c.provenance is Provenance.UNCOVERED for c in self.citations)

    def to_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "repo": self.repo,
            "sha": self.sha,
            "n_citations": self.n_citations,
            "header_fallbacks": self.header_fallbacks,
        }
        row.update({p.value: 0 for p in Provenance})
        row.update(self.counts())
        return row


# --- reading the trajectory ------------------------------------------------


def _text(content: Any) -> str:
    """Flatten message content to text.

    A pure tool call returns content as a list of zero blocks on this provider,
    so content is a list as often as it is a string. Used for parsing only,
    never to rebuild a message.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return ""


def collect_shown(trajectory: list[dict[str, Any]]) -> Shown:
    """Walk one trajectory and record every line number the model was shown.

    Tool calls are matched to their observations by ``tool_call_id`` rather than
    by position. Position happens to work while a run issues one call per turn
    and stops working the moment it issues two, which this provider can do.
    """
    shown = Shown()

    obs: dict[str, dict[str, Any]] = {
        m.get("tool_call_id", ""): m for m in trajectory if m.get("type") == "tool"
    }

    for msg in trajectory:
        if msg.get("type") != "ai":
            continue
        for call in msg.get("tool_calls") or []:
            name = call.get("name")
            args = call.get("args") or {}
            body = _text((obs.get(call.get("id", "")) or {}).get("content"))

            if name == "read_file":
                path = str(args.get("path", ""))
                first = body.split("\n", 1)[0]
                match = _READ_HEADER.match(first)
                if match:
                    # The returned range, after any clamping or truncation.
                    shown.add(shown.read, path, int(match["start"]), int(match["end"]))
                else:
                    # No header: numbering is off, or the read errored. Fall back
                    # to the request and record that the figure is an upper bound.
                    start, end = args.get("start_line"), args.get("end_line")
                    if isinstance(start, int) and isinstance(end, int):
                        shown.add(shown.read, path, start, end)
                    shown.header_fallbacks += 1

            elif name == "get_symbols":
                path = str(args.get("path", ""))
                for line in body.split("\n"):
                    m = _SYMBOL_LINE.search(line)
                    if m:
                        shown.add(shown.symbol, path, int(m["start"]), int(m["end"]))

            elif name == "search_code":
                for line in body.split("\n"):
                    m = _SEARCH_LINE.match(line)
                    if m:
                        shown.add(shown.search, m["path"], int(m["line"]), int(m["line"]))

    return shown


def classify(shown: Shown, path: str, start: int, end: int) -> CitationProvenance:
    """Assign one citation to the strongest provenance that covers it.

    Order matters and is not arbitrary. Read beats symbol beats search because
    each shows the model strictly more than the next: the code itself, then only
    where the code is, then only isolated lines. A range covered by two sources
    is credited to the stronger one, so the weaker categories mean "this and
    nothing better".
    """
    cited = set(range(start, end + 1)) if end >= start else set()
    if not cited:
        return CitationProvenance(path, start, end, Provenance.UNCOVERED, 0.0)

    read_hit = cited & shown.read.get(path, set())
    fraction = len(read_hit) / len(cited)

    if fraction == 1.0:
        return CitationProvenance(path, start, end, Provenance.READ, 1.0)
    if fraction > 0.0:
        return CitationProvenance(path, start, end, Provenance.PARTIAL_READ, fraction)
    if cited <= shown.symbol.get(path, set()):
        return CitationProvenance(path, start, end, Provenance.SYMBOL, 0.0)
    if cited <= shown.search.get(path, set()):
        return CitationProvenance(path, start, end, Provenance.SEARCH, 0.0)
    return CitationProvenance(path, start, end, Provenance.UNCOVERED, 0.0)


def trace_provenance(path: Path) -> TraceProvenance | None:
    """Trace one run's citations. ``None`` when the run produced none."""
    data = json.loads(path.read_text(encoding="utf-8"))
    questions = data.get("questions") or []
    if not questions:
        return None

    row = data.get("row") or {}
    shown = collect_shown(data.get("trajectory") or [])

    out: list[CitationProvenance] = []
    for q in questions:
        for c in q.get("citations") or []:
            out.append(
                classify(
                    shown,
                    str(c.get("path", "")),
                    int(c.get("start_line", 0)),
                    int(c.get("end_line", 0)),
                )
            )

    return TraceProvenance(
        trace_path=path,
        repo=row.get("repo", "?"),
        sha=row.get("sha", ""),
        citations=tuple(out),
        header_fallbacks=shown.header_fallbacks,
    )