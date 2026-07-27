# Ablation Plan

**Committed before any tool code was written. Frozen.**

The agent is built with a specific set of design choices in its tooling. This
document exists to make those choices *testable* rather than assumed.

The method is subtraction. Build the design believed to be correct, then remove
one piece at a time and measure the damage. An ablation that changes nothing is
evidence that the choice was decoration — and gets reported as such.

Each ablation below states a prediction, made in advance. The predictions are
recorded here so that the results can contradict them. A prediction written
after seeing the result is not a prediction, and revising one silently
invalidates the entire exercise.

---

## Ablation A — Unnumbered file reads

**Change:** `read_file` returns bare text. Line numbers are stripped, so the
model must count lines itself in order to cite a location.

**Prediction:** The largest single effect of the three. Citation verification
pass rate drops substantially — the model will emit plausible-looking line ranges
that do not resolve against the symbol index. Question *quality* should be
largely unaffected, because the model still reads identical content; only its
ability to point at that content degrades.

**Reasoning:** Counting is a serial, positional task with no mechanism behind it
in a language model — position has to be inferred from surrounding context.
Numbering moves that work out of the model and into the tool. If grounding
barely moves when numbering is removed, the numbering was never load-bearing and
the README should say so plainly.

---

## Ablation B — Unbounded file reads

**Change:** The line-range argument to `read_file` becomes optional. Entire files
may be pulled into context in a single call.

**Prediction:** The token budget exhausts earlier, so fewer steps complete within
budget and questions become shallower — concentrated across fewer files. Citation
accuracy roughly flat or slightly reduced (dilution rather than confusion). The
effect should scale with repository size: near-zero on small repos, clearly
visible on large multi-package trees.

**Reasoning:** Context is the scarce resource in an agent loop. An unbounded read
spends that resource on text the agent never asked for, and the cost is paid in
steps it can no longer afford to take.

---

## Ablation C — No structural symbol extraction

**Change:** The tree-sitter symbol tool is removed. Navigation proceeds using
file listing, file reading, and text search only.

**Prediction:** Step count per repository rises, as the agent performs more
exploratory reads to locate the same code. Question quality drops moderately on
unfamiliar repositories, and less on repositories where file and directory names
are already strongly descriptive. Citations that do resolve should remain
accurate — this is expected to be a cost in *finding* code, not in *pointing* at
it once found.

**Reasoning:** A symbol index is a map. Without one the agent searches; with one
it navigates. The prediction is that the difference shows up in effort before it
shows up in accuracy.

---

## Reporting rules

1. Every ablation runs against the full development set. No partial runs appear
   in the results table.
2. Ablations showing no measurable effect are reported with the same prominence
   as those that do. A flat row is a finding about the design.
3. No ablation touches the held-out repositories. Development set only.
4. No ablations are added to this list after it is committed. Ideas that surface
   later are recorded in the lab notes as future work, not folded into this table.