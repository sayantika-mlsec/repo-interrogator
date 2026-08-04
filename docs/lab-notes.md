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

## Entry 5 — File access tools

Three tools complete the model-facing surface: `list_files`, `read_file`,
`search_code`. Two facts drove every decision. The paths come from a model that
guessed, so all three route through the containment check before touching the
filesystem. The output lands in a context window, so all three bound their
response and mark truncation in the same channel as the content.

That bounding is deliberately the opposite of the workspace layer's. The
workspace rejects rather than truncates because a half-read repository is
invisible and corrupts a results row. A half-read file is visible to the reader
and recoverable by asking for the next range.

Line numbering is 1-based and inclusive, identical to the symbol layer. Not a
stylistic match: one verifier resolves citations from both, so a one-line
disagreement would fail half of every run's citations, and the failures would
look like model errors.

**The config seam.** Numbered output and the required line range are removed
later to measure what they are worth. Editing a tool at measurement time would
invalidate the measurement, so `ToolConfig` exists before any baseline is
recorded. It is threaded explicitly rather than read from a global or the
environment — for provenance, not testability: a results row has to state the
configuration that produced it, and configuration living in process state cannot
be serialised into the record. Defaults are the design under test, so a
forgotten config yields the real design rather than a silent ablation.

**Line-number metadata leaks, and nearly did.** A header reading
`core.py lines 40-80 of 320`, or a marker saying `Continue with start_line=81`,
hands back exactly what the numbering ablation removes — the model would only
have to count within a block whose offset it was told. Under
`number_lines=False` no line number appears anywhere; the marker degrades to a
bare `output truncated`. Three assertions verify no digit survives.

**One traversal.** ripgrep by default skips hidden paths and honours
`.gitignore`; `os.walk` does neither. Left alone, `search_code` would have
returned hits in files `list_files` never shows and missed files it does — the
same two-traversals-disagreeing failure the symbol indexer already refused.
`--hidden --no-ignore --no-follow` plus skip globs generated from `SKIP_DIRS`
makes the file sets identical, asserted by checking search hits are a subset of
listed paths. `--no-config` is there because a `RIPGREP_CONFIG_PATH` present on
one machine and not another would change results silently.

**A written expectation was wrong and the code was right.** Latin-1 text was
expected to fail as binary. The content predicate correctly calls it text, and
the UTF-8 decode then fails separately. `BinaryFileError` means "this is a PNG";
`FileDecodeError` means "this is source in the wrong encoding". An agent can act
on the difference. The assertion was corrected, not the behaviour.

**One deviation from the plan.** A range extending past end-of-file was
specified to clamp. `end_line` past EOF does clamp; `start_line` past EOF raises
instead, because clamping the start would silently return lines nobody asked
for. The error carries the file's real length so the call can be retried.

84 assertions, passing on Windows with 15.2.0. The 
JSON message schema, exit code 1 meaning "no matches", and every flag used
survived a major version bump — recorded as an observation, since the flag set
is what makes a search reproducible across machines.

Two limitations carried forward. `fnmatch` patterns cross directory separators
unlike shell glob, so `*.py` matches nested files; the looser reading fails
toward showing too much rather than silently showing nothing. And `search_code`
still returns line numbers under the numbering ablation, so the model can obtain
them from search even when `read_file` withholds them — a genuine interaction
between two tools, unfixable without making search useless, to be reported
alongside the ablation result.

## Entry 6 — the size cap measured the wrong bytes.

A pinned repository was rejected at the 50 MB working-tree ceiling. 
Its `docs/` directory alone exceeds that; its four largest files are 
documentation screen recordings of 12.0, 9.2, 7.5 and 3.7 MB, and everything 
readable totals roughly 7 MB. The cap was measuring blob weight while claiming 
to measure code weight.

The fix is two bounds rather than a bigger one. A total-tree guard protects 
the machine from an enormous clone; a readable-bytes cap bounds what the agent actually 
contends with. They guard different resources, and a repository can breach either alone.

A second defect surfaced while writing it. The measurement stops at the first file past a 
bound, which makes its figures lower bounds rather than totals — and nothing in the 
return recorded that, so a rejection message could report where counting stopped as though 
it were the tree size. The measurement now states which bound stopped it, and the "at least" 
phrasing is derived from that rather than chosen by whoever writes the message.

The bound values are placeholders. Setting a number before seeing the distribution across 
the pinned set is the mistake this change exists to correct, so calibration waits for the survey. 
First figures: `structlog` measures 2.2 MB tree against 0.9 MB readable.

## Entry 7 — the bounds, set from measurement

All twelve pinned repositories were measured at their commits with cap
enforcement disabled, so every figure is a complete walk rather than a lower
bound. Readable content runs from 0.3 MB to 8.5 MB; trees run from 0.3 MB to
77 MB; file counts from 15 to 979. Every repository is analyzable and the set is
frozen.

The two-bound change turned out to be load-bearing more narrowly than the
original rejection suggested. `instructor`, the repository that prompted it,
measures 77 MB of tree against 6.6 MB of readable content. But `ragas` measures
48.9 MB against 5.2 MB — it passed the old 50 MB ceiling with 1.1 MB to spare.
Two more documentation GIFs and it would have failed the same way, and there
would have been nothing in its rejection to connect it to the first.

The readable cap goes to 16 MB, just under twice the observed maximum: a
repository half again larger than the largest measured still enters, one twice as
large does not. The total guard stays at 500 MB and the file count at 10,000.
Neither fires on anything in this set, which is correct — their job is stopping a
pathological clone, not filtering content, and tightening them toward the
observed maxima would make them a second, cruder content filter and reintroduce
the conflation the readable cap replaced.

Two things the aggregate figures hid, both visible only because the largest files
were recorded per repository. `nids` reports 8.3 MB readable, of which 8.2 MB is
a single generated HTML report; its actual source is roughly 0.1 MB. That file
also exceeds the per-file read ceiling, so it counts toward the readable budget
and cannot be read — the interaction between the two limits was recorded earlier
as a small conservative overcount, and is in fact the largest single contributor
to the readable maximum in the development set. And the annotation predicting
that `sqlglot` would stress the file-count cap was wrong: it has 355 files
against `instructor`'s 979 and `typer`'s 773. The note is corrected rather than
deleted.

## Entry 8 — The agent loop

Two issues. The tool layer gained its fourth tool; the agent layer now exists.

### Symbol access as a tool

`get_symbols` returns the definitions in one file with their line ranges. The
path is required. A whole-repository dump runs to thousands of entries on a real
project, and a tool whose purpose is to save steps would end a run by consuming
the context needed to read code. `max_symbol_entries` (200) bounds it like every
other response.

`FileTools` now takes `symbol_index` as a keyword argument rather than building
one. Building an index requires the symbol layer, which requires the workspace
layer, and the tools would then depend transitively on the cloner in order to
read a file. The caller already holds the index.

`build_index_at(root, name, sha)` was split out of `build_symbol_index(repo)`,
which now delegates to it after checking the analysis mode. The mode check stays
with the clone record, the only thing that knows about modes. `build_index_at`
takes no mode argument: a parameter with exactly one legal value is a parameter
that will eventually be passed the wrong one.

`SymbolsUnavailableError` is a `ToolError`, not a `SymbolError`. It reaches the
model as an observation it can act on by switching to search. The symbol layer's
own version is a fact about the repository that the caller must handle before a
run starts.

The 84 existing tool assertions pass unchanged; the new parameter is
keyword-only.

### The loop

Five tools, `finish` among them. `finish` carries the questions as its argument
rather than the loop parsing them out of a final text turn. Termination becomes
a row in the same trace table as every other call, and output cannot be
separated from termination — there is no run that stopped without producing
questions, and none that produced questions without stopping.

`finish` writes validated objects into a collector rather than returning them. A
tool's return value is rendered to a string for the model; recovering questions
from that rendering would mean re-parsing our own formatting, and any lossy step
would be silent corruption. Validation happens once, at the boundary.

Questions are Pydantic-validated with at least one citation each. A question
with no citation cannot be verified, and an unverifiable question in an output
set would be scored as if it had passed a check it never took. Citations are
1-based inclusive, matching both the symbol layer and `read_file`.

`ToolError` is caught in each wrapper and returned to the model as an
observation, and still costs a step — otherwise a model failing every call runs
forever for free. Everything else propagates. `PathEscapeError` is a
`WorkspaceError`, so an escape attempt kills the run instead of becoming a hint.

A `ToolMessage` arriving with `status == "error"` raises. Legitimate tool errors
come back as ordinary content, so an error status means something escaped a
wrapper and was absorbed by the framework. That is the failure mode
`_safe_extract_text()` produced on the previous project, in a new place.

Both budgets raise. The check sits between receiving one stream state and
requesting the next, so a breach stops the run before the next call is
dispatched rather than after it has been billed. A test asserts the dispatch
count equals the ceiling; it would exceed it by one if the check ran afterwards.

### Verified against the provider, not assumed

A two-turn probe (`scripts/probe_signatures.py`) ran before any loop code was
written. Four findings, none of them derivable from documentation:

1. **Thinking tokens were 75 of 103 output tokens — 73%.** They bill as output.
   The budget counts `total_tokens`; a budget on visible output would have been
   wrong by roughly four times.
2. **`gemini-3.5-flash` is global-endpoint only.** A regional endpoint returns
   404. `location=global` is a property of the model and belongs on the results
   row.
3. **Thought signatures survive an append-whole message list**, stored in
   `additional_kwargs['__gemini_function_call_thought_signatures__']`. This is a
   standing constraint, not a note: assistant turns are appended, never rebuilt,
   filtered, or summarised. Any future context-trimming work operates on tool
   observations only. Rewriting an assistant turn breaks tool calling on the
   second call of a run.
4. **A pure tool call returns content as a list of zero blocks.** The trace
   store logs blocks, not rendered text, or most steps would record as empty
   strings and read as a broken model.

A live check confirmed the generated tool schema is accepted as written. The two
optional integer parameters on `read_file` serialise as `anyOf: [integer, null]`,
which older stacks rejected; this one does not. No sentinel workaround was
needed, and the honest schema ships.

### Recorded, not solved

Open question 7 gains a second instance. `get_symbols` reports line ranges
regardless of the numbering configuration, so the model can obtain line numbers
from the symbol index as well as from search. Suppressing them would leave a
tool that names definitions without locating them, which is not a tool. Reported
alongside the ablation result; the ablation set stays frozen.

### Assertions

`smoke_tools.py` 84 (unchanged). `smoke_agent.py` 37 offline, 42 with the live
run.

## Entry 9 — First full run

First end-to-end run. Until today every layer was tested alone: the workspace
layer cloned and verified a pin, the symbol layer indexed a tree, the tools read
and searched one, and the agent ran a bounded loop over tools built in a fixture
directory. Nothing had run all four in one call stack.

### What was built

`repos.py` and `runner.py` in the package, `run_repo.py` as a thin entry point.

The pin-file loader moved out of the survey script into the package. A package
cannot import from a script directory, so the alternative was a second YAML
parser, and two readers of a pin file drift on the field that matters — the
commit id. The survey's original function signature is kept as a wrapper so that
script changed by one import line.

The runner sequences and owns no policy except one. Held-out repositories are
refused unless the caller asks for them explicitly and states a reason, and the
read is appended to `docs/held-out-reads.md` before the clone begins. Written
before rather than after, because a read that crashed halfway is still a read:
the repository was fetched and its output may have been seen, and nothing about
that is undone by a later exception. Until now the discipline was protected by
remembering, on every invocation, for a month. It is now a file that can be
counted.

Text-only repositories degrade rather than abort — four tools instead of five,
logged, and recorded on the result so a four-tool run is never mistaken for a
five-tool one. Every run writes its full trajectory to JSON, serialised through
each message's own `model_dump` rather than rendered to text, because a rendered
message loses the content blocks and the thought signature this provider
requires on the following call.

Two changes ahead of the runner: the agent's task string is now sent verbatim
instead of being assembled from a question count that silently discarded the
caller's string, and an empty task raises rather than producing a run whose
prompt is not recoverable from its own trace.

### The run

`nids`, ten questions, no crashes. Twenty symbols across seven files, five tools,
seventeen model calls, seventeen tool calls, zero tool errors, 181,695 tokens,
63 seconds.

Two citations were resolved by hand. One names a single line and lands exactly on
it; one spans a seventeen-line region and covers what the question asks about.
The 1-based inclusive contract holds from the symbol layer through the tools,
through the model, and out into a citation.

### What the output shows

Three observations, recorded rather than acted on. No baseline has been read yet,
and changing anything now would make the first number meaningless.

**Questions state their own answers.** Two of ten name the identifier or quote
the phrase the question asks about. A question carrying its answer cannot
discriminate between a reader who understood the repository and one who did not.

**At least three questions are grounded in prose the repository's author wrote,
not in inferred behaviour.** The strongest case cites a line whose "why" exists
only in its trailing comment; another takes its framing from a comment two lines
above the cited region. The citations are valid — the comment is on the cited
line — but reading a comment and asking about it is a weaker capability than
reading code and working out what it does.

This matters for the documentation-exclusion control. That control removes
documentation files, and the paraphrasing seen here is happening in inline
comments, which it will not remove. The control as specified may therefore
measure less than intended. Recorded now, before the control runs, so its result
is interpretable rather than surprising.

**Several questions are compound**, asking two things joined by "and". Relevant
to any per-question scoring later.

### Cost

At roughly 180k tokens per run, a full sweep of the set is near 1.8M, and the
comparison work planned against it multiplies that several times over. A billing
budget alert should exist before the first full sweep, not after it.

The step ceiling was thirty and seventeen were used. It stays where it is: the
ablation that removes structural symbol extraction makes navigation more
expensive by design, and a ceiling tuned to the cheapest configuration would
raise instead of producing a measurable result.

### Found while working

Running the survey narrowed to one repository still overwrites the full
measurement record. The flag narrows the input and not the output, so a
convenience invocation destroys the frozen figures the size bounds were read
from. Restored from history; filed separately.

## Entry 10 — Full run on two other dev repos

Three things, in the order they happened: a run that cost full price and left
nothing behind, the fix for that, and then the two runs that fix made readable.

### A failed run kept no evidence

The first attempt at a large repository breached the token ceiling at 419,237
tokens after 23 model calls. Nothing was written to disk.

The trajectory was being written only on the success path. The counters and the
message list were locals inside the agent's run loop, and the trace was written
by the caller after a successful return, so a breach dropped both with the
frame. This is backwards: a run that fails is the more informative of the two
outcomes, because it is the only thing that says where the tokens went.

Fixed. The counters and the message list now live on the agent as a progress
object, a run's metadata is assembled before the model is called — none of it
depends on the run succeeding — and the trace is written from a `finally` on
every exit path. The progress object is deliberately not a result type: it has
no questions and no row method, so it cannot be handed to anything that builds a
results table. A failed run is a distinct type from a completed one, carries an
outcome naming which ceiling was hit, and is marked in the filename as well as
in the payload, so a directory listing does not read as ten finished runs when
three of them died. The exception still propagates and a partial result is still
never returned as a result.

Three assertions added covering retention on the breach path. Offline smoke
assertions for the agent layer: 40 → 43.

Committed directly to the default branch rather than through a pull request,
verified by the offline suite before landing, recorded rather than left
implicit.

### The token ceiling was unreachable by design

With the fix in place the same repository breached again — 21 model calls,
401,443 tokens — and this time the trajectory survived.

Reading it showed the ceiling was not measuring what it appeared to. Each model
call bills for the whole resent history, so cumulative spend grows with the
square of the step count while the context grows linearly. Input rose about
1,450 tokens per call and the context reached only 31.6k by the last call —
nowhere near any model limit — yet the cumulative total hit 400,000 at step 21.
The step ceiling of 30 could therefore never fire at any repository size. One of
the two budgets was decorative.

The ceiling is billed spend, not context size. Worth stating plainly because the
number looks like a context bound and is not one.

Raised to 800,000, extrapolated from the measured growth curve, on the estimate
that a run reaching 30 steps would cost roughly 745k. Applied uniformly to every
repository in the set: a per-repository budget would make the comparison between
the tuning set and the reserved set partly a measure of how much room each
repository was given.

Also noted from that trajectory: output was 2,500 of 401,443 tokens, six tenths
of one percent. The earlier probe finding that thinking dominates output tokens
is true and irrelevant at this scale. Counting total tokens remains correct, but
the reason is resent history, not thinking.

### The agent does not converge on a large repository

Re-run under the new ceiling: 30 model calls, 614,689 tokens, stopped by the
step budget without ever calling finish. Growth had decayed to about 800 tokens
per call, so the raised ceiling would have bought roughly 36 steps. The run
stopped because it was told to, not because it ran out of money.

The reading itself was clean, and cleaner than the earlier attempt. Twenty-one
file reads, no duplicated range, every read preceded by a structural listing of
the same file. The ranges are contiguous — 1416-1500 then 1501-1570, 1061-1130
then 1131-1168, 50-95 then 96-150 — so the agent is scanning regions in
consecutive chunks rather than jumping around. Seven files covered in a sensible
order. Output held flat near 110 tokens per call throughout, which is a pure
read loop with no visible deliberation about whether enough had been gathered.

The last two calls return to the file the run opened with, after six others.
That reads either as gathering up to finish or as beginning another pass, and a
single run cannot distinguish the two.

**Hypothesis, one repository, not tested.** The task says to investigate first
and finish when done, and never defines done. On a twenty-five-file repository
coverage is achievable and the run terminated naturally at 17 calls. On a
979-file repository coverage is unreachable, so nothing triggers termination.
If that is right it is a property of the prompt, not of the budget, and belongs
in the failure taxonomy rather than in a limit.

**The step ceiling stays at 30.** Raising a ceiling because a run hit it is how
a budget stops being a measurement and becomes whatever the model happened to
want. If it rises later it will be because a trajectory showed convergence, not
because a run was cut short.

### Resolved

Zero tool errors is not evidence that the error messages are good. Every file
read in both runs followed a structural listing of the same file, so the model
was always working from ranges the symbol tool had already reported and never
had to guess one. No error was provoked, so none was caught. The earlier
suspicion that a clean first run was suspiciously clean is answered: it was, and
the reason is that nothing tested the failure paths.

### Carried forward

- Whether the task is completable on a repository of this size at any budget is
  unanswered. Two mid-sized repositories will bracket it: if both terminate, the
  failure is size-dependent and the boundary is known for free.
- A billing alert on the cloud project is still not configured. At current cost
  a full sweep of the pinned set is near 8M tokens and the model comparison
  multiplies that.
