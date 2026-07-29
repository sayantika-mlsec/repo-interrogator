# Lab Notes

Running record of decisions, dead ends, and things that turned out to be wrong.

**Append-only.** New entries go at the bottom. Nothing above is edited or deleted.
When an earlier entry turns out to be mistaken, a later entry says so and explains
why — the mistaken entry stays where it is. A notebook that is tidied after the
fact is a story, not a record.

---

## Entry 1 — Project start

### Evaluation set pinned

Ten repositories pinned to commit SHAs, split six development / four held-out,
plus two further repositories reserved for a memorization check. Full set and
selection reasoning in `repos.yaml`.

Pinning happened before any tool code was written. This is deliberate and cannot
be retrofitted: a held-out score only means something if the held-out set was
chosen before there was anything to tune against it.

### Held-out protocol

The four held-out repositories are scored exactly twice — once as a baseline
before any tuning, once at the end — and are not inspected, labeled, or debugged
against at any point in between. Hand-labeling for the calibration set draws from
development repositories only; labeling a held-out question counts as looking at
it.

Two reads, not more. Each additional read is a chance to tune against the set
without noticing, and the number stops meaning anything after that.

### Held-out selection

An earlier candidate for the held-out set was `outlines`, which was rejected. It
occupies the same problem space as `instructor` in the development set — both
concern constrained and structured LLM output. Including it would have made the
final held-out number a blend of two distinct effects, "unseen repository" and
"unseen problem domain," with no way to separate them afterward.

`sqlglot` replaced it. The held-out set now spans four unrelated domains: async
HTTP, command-line parsing, structured logging, and SQL parsing.

### Ablation plan committed

Three ablations written down before any of the code they concern existed, each
with a prediction recorded in advance. See `docs/ablations.md`. The list is
frozen; nothing is added to it later.

### Prior project frozen

Three of the six development repositories are my own earlier projects.
All work on them is paused for the duration of this build, and they will receive
no commits. This means the pinned SHAs stay valid throughout rather than drifting
away from what the agent actually reads — a side effect of the pause, but a
useful one.

### Open questions carried forward

1. **Non-Python repositories.** All ten pinned repositories are Python, so
   structural symbol extraction is only ever exercised against a single grammar
   during development. The stated goal is a demo that works on an arbitrary
   repository supplied by a stranger, which will not always be Python. The
   clone/scaffolding layer must decide this explicitly — reject clearly, or
   degrade to text search without symbols. Decided in the scaffolding layer, not
   discovered during deployment.

2. **Size cap never exercised.** None of the pinned repositories comes close to
   the intended size ceiling, so the rejection path is untested by the evaluation
   set itself. It needs to be pointed at something oversized once, deliberately,
   and confirmed to fail loudly rather than truncate silently.

3. **Read-only cleanup on Windows.** Git marks objects under `.git/` read-only.
   Python's `shutil.rmtree` raises on those files on Windows, so temp-directory
   cleanup after each clone needs an explicit handler that clears the read-only
   bit and retries. Left unhandled, this leaks a full repository clone to disk on
   every run.


## Entry 2 — Workspace safety layer

Clone wrapper fetches a pinned commit directly (`fetch --depth 1 origin <sha>`)
rather than shallow-cloning and checking out. A shallow clone retrieves the tip
of the default branch, which is usually not the pinned commit, so the object is
never downloaded and checkout fails. The checked-out SHA is verified against the
requested SHA after every clone; a mismatch is fatal.

Caps reject rather than truncate. A truncated repository still produces
plausible questions, which is the failure that would silently corrupt a results
row.

Open question 2 (size cap never exercised) closed: the rejection path is tested
by lowering the cap rather than by finding an oversized repository.

Open question 3 (Windows read-only cleanup) closed: `shutil.rmtree` with a
handler that clears the read-only bit and retries. Verified on Windows.

Open question 1 (non-Python repositories) closed by policy: repositories with
too little Python degrade to text-only analysis with no symbol index, and the
degradation is logged. Strict rejection is available as a configuration flag.

All paths supplied to tools resolve through a single containment check that
rejects anything landing outside the repository root. The check runs after
path resolution, not before: string inspection cannot catch a symlink that
points out of the tree while containing no traversal segments.

Every guard has an assertion in a committed smoke script rather than a
one-off manual check, so the rejection paths stay tested as the code changes.

## Entry 3 — Structural symbol extraction

### Line numbering fixed as a contract

Symbol line ranges are 1-based and inclusive on both ends. A definition
occupying the first three lines of a file is `(1, 3)`.

This is the convention editors, `git blame`, and `grep -n` already use, and it is
what every citation in this project resolves against. The alternative — carrying
tree-sitter's native 0-based rows through the system — would have pushed the
conversion into the verifier, the reader, and the frontend, giving each of them
its own chance to get it wrong.

The convention is enforced rather than documented: constructing a symbol with a
start line below 1, or an end line preceding its start, raises. A convention
that is only written down drifts; one that is asserted cannot.

tree-sitter reports an exclusive end point, which for any definition ending in a
newline lands on column 0 of the *following* line. Uncorrected, every multi-line
symbol would claim one line too many — invisible in the middle of a file, and
visible only at the last symbol in it. The correction happens once, at the node
boundary, and there is a smoke assertion pinned specifically to the final symbol
in a file because that is the only position where the error shows.

### Signatures are sliced, never reconstructed

A signature is the literal source text from the definition keyword to the start
of the body. Reassembling it from syntax-tree children would mean
re-implementing Python's rules for defaults, annotations, star-args, and generic
parameters — and any gap between the reconstruction and the file would make the
citation subtly untrue. Source text cannot drift from itself.

Spans start at the first decorator where one is present. A decorator changes what
the code does, so a range that omits it points at code it does not fully describe.

### Nested definitions: a deliberate boundary

The walk descends into class bodies but not function bodies. Closures and locally
defined helpers are real code, but they are not navigation targets, and including
them would inflate the map with entries no question can usefully cite. This is a
choice, not an oversight, and it is what the reported symbol count means.

### One traversal, shared

The symbol indexer walks files through the same iterator every other tool uses,
rather than defining its own. Two traversals would eventually disagree about
which files a repository contains, and that disagreement would surface as a
difference in ablation results — appearing to measure the value of the symbol
map while actually measuring a file-set mismatch. The smoke script asserts that
skip-listed directories stay invisible to the indexer, with the reason recorded
in the assertion itself.

### Failure has a shape

A file that will not parse is recorded against its path and counted, not silently
skipped and not fatal. One unparseable vendored file should not void the other
four hundred, but a run that missed a third of a repository must not be
reportable as if it had seen the whole tree.

A repository in text-only mode raises when a symbol index is requested, rather
than returning an empty one. An empty index is indistinguishable from a
repository that defines nothing.

### Error hierarchy

A single project-wide root now sits above the exception tree, so a caller at the
API boundary can catch everything in one clause. Parsing failures are not
workspace failures and hang off their own branch beneath that root, which keeps
the layers distinguishable without forcing callers to name each one.

### Two corrections to the workspace layer

Found while reviewing it against the new code, both concerning errors that
disguised themselves as other errors:

A malformed commit id was raising an untyped error, which meant callers had to
match on message text to tell a bad argument from a pin that failed to hold.
Those need different responses — one is a bad entry in the pinned set, the other
means every number produced from that repository is suspect.

The Windows deletion handler retried on any failure, running its permission fix
unconditionally. A file that vanished mid-walk, or one held open by another
process, would surface as a confusing failure in the fix rather than as the
original cause. It now re-raises anything that is not a permission problem. This
matters because cleanup runs during exception handling, where a second error
buries the first.

### Open questions carried forward

4. **Symlinked files are invisible to the walk.** The file iterator skips
   symlinks so that a directory link cannot send it walking the whole
   filesystem. The side effect is that a repository storing its source behind
   symlinks would measure near zero and pass the size caps without ever being
   read. None of the pinned repositories do this, so it is recorded rather than
   solved.

5. **Grammar exercised only against fixture code.** The extractor is verified
   against a hand-written fixture covering async definitions, decorators, and
   nested classes. Real code contains structural pattern matching, walrus
   assignments, and PEP 695 generic parameter lists. Until the indexer has been
   pointed at a substantial third-party repository and its failure count
   inspected, the parse-failure rate on modern syntax is unknown.

   ## Entry 4 — Symbol extraction verified against real code

The extractor had only ever run against a hand-written fixture. Two questions
were open: whether the tree-sitter grammar handles current Python, and whether
the traversal reaches everything the grammar parses. They are different
failures, and only one of them is loud.

`symbols_in_file` rejects any file whose parse tree contains an error, so a
grammar gap drops a whole file and lands in `SymbolIndex.failures`. That half is
self-reporting.

The other half is not. The walk descends into the module and into class bodies,
and nowhere else. A definition inside a conditional block — `if TYPE_CHECKING:`,
or `try: ... except ImportError:` — parses cleanly and is silently absent. No
error, no failure row, and `files_indexed` still counts the file as fully
indexed. An index quietly short still produces plausible questions, and the
ablation that removes the index would then be measuring the value of removing
something incomplete.

`scripts/probe_grammar.py` measures both. It re-parses each file with an
independent traversal that also descends through statement containers, marking
every definition with whether the extractor could have reached it and which
container hid it if not. Function bodies are excluded from that traversal too,
so the documented closure boundary does not register as a gap. The probe
cross-checks itself per file: its own reachable count against what
`symbols_in_file` actually returned, with disagreements reported separately
rather than folded into the totals.

A syntax fixture covers `match`, walrus, PEP 695 generic functions and classes
and `type` aliases, `except*`, parenthesized context managers, positional-only
parameters, async comprehensions, PEP 614 decorator expressions, PEP 701 nested
f-strings, and a slotted dataclass. Two further cases hold a class inside a
conditional block, present specifically so the two kinds of gap can be seen side
by side.

Results. All fourteen fixture cases parse. Twelve yield exactly the expected
symbols; the two conditional cases parse and yield nothing, confirming the
silent gap as a mechanism. Against a pinned dev repository: 75 Python files,
1414 symbols, zero parse failures, zero definitions missed. The self-check
agreed on all 75 files, which is what makes the zero worth believing.

Both questions close. The grammar handles current syntax. The conditional-
definition gap exists but does not occur in this repository — `if TYPE_CHECKING:`
blocks in practice hold imports rather than definitions.

Two bounds on that conclusion, recorded rather than resolved. It is one
repository, not the whole set. And the probe's container list is fixed, so a
container type absent from it would cause under-reporting; the error direction is
conservative, but a zero is bounded by the probe rather than proven by it. The
probe is committed and offline-cheap, so it can be re-run across the full set
when repository sizes are next measured.