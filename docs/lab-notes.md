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