"""Smoke test for the file access tools: run them, don't trust them.

No network and no clone. A fixture tree is written to a temp directory and the
tools are pointed at it, so this runs offline in about a second.

Expected line numbers are derived from the fixture source by locating a unique
marker, never hardcoded -- the same discipline the symbol smoke test uses, and
for the same reason: a hardcoded number rots the moment the fixture is edited,
and a stale expectation that still passes is worse than no test at all.

The load-bearing assertion in this file is the interoperability one. A range
taken from a ``Symbol`` and a range read by ``read_file`` must return the same
text. If those two ever disagree by one, every citation in every run is wrong in
a way that looks like a model failure.

Run:  uv run python scripts/smoke_tools.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from repo_interrogator.errors import (
    BinaryFileError,
    FileDecodeError,
    FileNotFoundInRepoError,
    InvalidLineRangeError,
    LineRangeRequiredError,
    NotAFileError,
    PathEscapeError,
    ToolConfigurationError,
)
from repo_interrogator.fsutil import is_probably_binary, rmtree_robust
from repo_interrogator.symbols import symbols_in_file
from repo_interrogator.tools import FileTools, ToolConfig

# --- fixture ---------------------------------------------------------------

CORE_SRC = '''\
"""Fixture package module."""


def alpha(a, b=1):
    """First function."""
    return a + b


class Engine:
    """A class with a marker inside."""

    def run(self, payload):
        needle_in_method = payload
        return needle_in_method


def omega():
    return "last function in core"
'''

# No trailing newline: the last line must still be readable and counted.
NO_NEWLINE_SRC = "first = 1\nsecond = 2\nthird = 3"

# 40 numbered lines, for range and truncation assertions.
LONG_SRC = "".join(f"line_{i:03d} = {i}\n" for i in range(1, 41))

# PNG magic: the length field after the signature carries NUL bytes at offset 8.
PNG_BYTES = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"\x00" * 64

# 0xFF is not a legal UTF-8 byte in any position, but there is no NUL, so this
# exercises the decode branch of the binary check rather than the NUL branch.
LATIN1_BYTES = "caf\xe9 = 1\n".encode("latin-1")

GITIGNORED_SRC = "def hidden_from_git(): pass  # needle_gitignored\n"
WORKFLOW_SRC = "name: ci  # needle_hidden_dir\n"

# --- harness ---------------------------------------------------------------

_passed = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global _passed
    if not condition:
        raise AssertionError(f"{label}{': ' + detail if detail else ''}")
    _passed += 1
    print(f"  [ok] {label}")


def check_raises(label: str, exc_type: type[BaseException], fn, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except exc_type:
        check(label, True)
        return
    except BaseException as exc:  # noqa: BLE001 - wrong type is the failure
        raise AssertionError(
            f"{label}: expected {exc_type.__name__}, got {type(exc).__name__}: {exc}"
        ) from exc
    raise AssertionError(f"{label}: expected {exc_type.__name__}, nothing raised")


def line_of(src: str, needle: str) -> int:
    hits = [i for i, line in enumerate(src.splitlines(), start=1) if needle in line]
    if len(hits) != 1:
        raise AssertionError(f"marker {needle!r} matched {len(hits)} lines, need exactly 1")
    return hits[0]


def build_fixture(root: Path) -> None:
    (root / "pkg").mkdir()
    (root / "pkg" / "sub").mkdir()
    (root / ".github").mkdir()

    (root / "README.md").write_text("# fixture\nneedle_readme\n", encoding="utf-8")
    (root / "pkg" / "core.py").write_text(CORE_SRC, encoding="utf-8")
    (root / "pkg" / "sub" / "long.py").write_text(LONG_SRC, encoding="utf-8")
    (root / "pkg" / "sub" / "tail.py").write_text(NO_NEWLINE_SRC, encoding="utf-8")
    (root / "logo.png").write_bytes(PNG_BYTES)
    (root / "latin1.txt").write_bytes(LATIN1_BYTES)

    # Hidden directory: os.walk descends into it, so ripgrep must too.
    (root / ".github" / "ci.yml").write_text(WORKFLOW_SRC, encoding="utf-8")

    # Ignored by git but visible to os.walk, so search must find it.
    (root / ".gitignore").write_text("ignored.py\n", encoding="utf-8")
    (root / "ignored.py").write_text(GITIGNORED_SRC, encoding="utf-8")

    # Must be invisible to both list_files and search_code.
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "ghost.py").write_text(
        "def ghost(): pass  # needle_in_method\n", encoding="utf-8"
    )
    (root / "build").mkdir()
    (root / "build" / "artifact.py").write_text(
        "def built(): pass  # needle_in_method\n", encoding="utf-8"
    )


# --- tests -----------------------------------------------------------------


def test_construction(root: Path, outside: Path) -> None:
    print("\nconstruction")
    tools = FileTools(root)
    check("ripgrep resolved at construction", tools._rg is not None)
    check("default config is the design under test",
          tools.config.number_lines and tools.config.require_line_range)
    check("config is frozen", getattr(ToolConfig, "__dataclass_params__").frozen)
    check("config serialises for a run record",
          isinstance(ToolConfig().to_dict()["number_lines"], bool))
    check("ablation knobs are exactly the two toggles",
          set(ToolConfig().ablation_knobs) == {"number_lines", "require_line_range"})
    check_raises(
        "root that is not a directory is refused at construction",
        ToolConfigurationError,
        FileTools,
        outside,
    )


def test_binary_predicate(root: Path) -> None:
    print("\nreadability predicate")
    check("png is binary", is_probably_binary(root / "logo.png"))
    check("python source is not binary", not is_probably_binary(root / "pkg" / "core.py"))
    check("markdown is not binary", not is_probably_binary(root / "README.md"))
    # Not binary: it is text in the wrong encoding. The distinction matters,
    # because the two get different errors and only one of them is worth
    # reporting as unreadable content.
    check("latin-1 text is not binary", not is_probably_binary(root / "latin1.txt"))


def test_list_files(root: Path, outside: Path) -> None:
    print("\nlist_files")
    tools = FileTools(root)
    result = tools.list_files()
    paths = set(result.paths)

    check("posix separators on every platform", all("\\" not in p for p in result.paths))
    check("sorted deterministically", list(result.paths) == sorted(result.paths))
    check("nested file listed", "pkg/sub/long.py" in paths)
    check("hidden directory listed", ".github/ci.yml" in paths)
    check("gitignored file listed", "ignored.py" in paths)
    check("__pycache__ never listed", "__pycache__/ghost.py" not in paths)
    check("build never listed", "build/artifact.py" not in paths)
    check("nothing outside the root leaks in", outside.name not in paths)

    filtered = tools.list_files(pattern="pkg/**/*.py")
    check("pattern filters to python under pkg",
          set(filtered.paths) == {"pkg/sub/long.py", "pkg/sub/tail.py"},
          str(filtered.paths))

    sub = tools.list_files("pkg/sub")
    check("subdir listing is repo-relative, not subdir-relative",
          set(sub.paths) == {"pkg/sub/long.py", "pkg/sub/tail.py"}, str(sub.paths))

    check_raises("traversal outside the root refused", PathEscapeError,
                 tools.list_files, "../")
    check_raises("absolute path outside the root refused", PathEscapeError,
                 tools.list_files, str(outside.parent))
    check_raises("subdir that is a file refused", NotAFileError,
                 tools.list_files, "README.md")

    capped = FileTools(root, ToolConfig(max_list_entries=2)).list_files()
    check("entry cap truncates", capped.truncated and len(capped.paths) == 2)
    check("truncation is visible in the text", "more files not shown" in capped.text)
    check("total is reported even when truncated", capped.total_found > 2)


def test_read_file(root: Path, outside: Path) -> None:
    print("\nread_file")
    tools = FileTools(root)

    check_raises("range required by default", LineRangeRequiredError,
                 tools.read_file, "pkg/core.py")

    start = line_of(CORE_SRC, "def alpha")
    end = line_of(CORE_SRC, "return a + b")
    r = tools.read_file("pkg/core.py", start, end)

    check("returns exactly the requested range", r.start_line == start and r.end_line == end)
    check("first line is the def line", r.lines[0].strip().startswith("def alpha"))
    check("last line is the return", r.lines[-1].strip() == "return a + b")
    check("line count matches an inclusive range", len(r.lines) == end - start + 1)
    check("gutter carries the absolute line number", f"{start} | def alpha" in r.text)
    check("header states the range", r.text.splitlines()[0].startswith("pkg/core.py lines "))
    check("total lines reported", r.total_lines == len(CORE_SRC.splitlines()))

    single = tools.read_file("pkg/core.py", start, start)
    check("single-line range returns one line", len(single.lines) == 1)

    tail = tools.read_file("pkg/sub/tail.py", 1, 3)
    check("file without trailing newline keeps its last line",
          tail.lines[-1] == "third = 3" and tail.total_lines == 3)

    clamped = tools.read_file("pkg/sub/long.py", 38, 999)
    check("end past EOF clamps", clamped.clamped and clamped.end_line == 40)
    check("clamping is visible", "end of file at line 40" in clamped.text)

    check_raises("start past EOF raises", InvalidLineRangeError,
                 tools.read_file, "pkg/sub/long.py", 500, 600)
    check_raises("zero start raises", InvalidLineRangeError,
                 tools.read_file, "pkg/core.py", 0, 5)
    check_raises("negative start raises", InvalidLineRangeError,
                 tools.read_file, "pkg/core.py", -3, 5)
    check_raises("inverted range raises", InvalidLineRangeError,
                 tools.read_file, "pkg/core.py", 9, 4)
    check_raises("end without start raises", InvalidLineRangeError,
                 tools.read_file, "pkg/core.py", None, 5)

    check_raises("binary file refused", BinaryFileError, tools.read_file, "logo.png", 1, 5)
    check_raises("non-utf8 text refused as a decode failure, not as binary",
                 FileDecodeError, tools.read_file, "latin1.txt", 1, 2)
    check_raises("missing file is not a parse error", FileNotFoundInRepoError,
                 tools.read_file, "nope.py", 1, 2)
    check_raises("directory is not a file", NotAFileError, tools.read_file, "pkg", 1, 2)
    check_raises("traversal refused", PathEscapeError,
                 tools.read_file, f"../{outside.name}", 1, 2)
    check_raises("absolute path refused", PathEscapeError,
                 tools.read_file, str(outside), 1, 2)

    capped = FileTools(root, ToolConfig(max_read_lines=5)).read_file("pkg/sub/long.py", 1, 40)
    check("line cap truncates", capped.truncated and len(capped.lines) == 5)
    check("truncation names the resume point", "start_line=6" in capped.text)


def test_symbol_interop(root: Path) -> None:
    print("\ncitation interoperability")
    tools = FileTools(root)
    syms = {s.qualname: s for s in symbols_in_file(root, "pkg/core.py")}
    source_lines = CORE_SRC.splitlines()

    for qualname in ("alpha", "Engine", "Engine.run", "omega"):
        sym = syms[qualname]
        r = tools.read_file("pkg/core.py", sym.start_line, sym.end_line)
        expected = source_lines[sym.start_line - 1 : sym.end_line]
        check(
            f"symbol range and read_file agree for {qualname}",
            list(r.lines) == expected,
            f"symbol={sym.start_line}-{sym.end_line} read={r.start_line}-{r.end_line}",
        )

    last = syms["omega"]
    r = tools.read_file("pkg/core.py", last.start_line, last.end_line)
    check("last symbol in file does not overrun", r.lines[-1].strip().startswith("return"))


def test_ablations(root: Path) -> None:
    print("\nablation seam")
    start = line_of(CORE_SRC, "def alpha")
    end = line_of(CORE_SRC, "return a + b")

    bare = FileTools(root, ToolConfig(number_lines=False))
    r = bare.read_file("pkg/core.py", start, end)
    check("ablation A returns bare text", r.text.splitlines()[0].startswith("def alpha"))
    check("ablation A emits no gutter", " | " not in r.text)
    check("ablation A leaks no line numbers at all",
          not any(ch.isdigit() for ch in r.text.replace("b=1", "")),
          repr(r.text))
    check("ablation A still returns structured line numbers to code",
          r.start_line == start and r.end_line == end)

    bare_trunc = FileTools(root, ToolConfig(number_lines=False, max_read_lines=5))
    rt = bare_trunc.read_file("pkg/sub/long.py", 1, 40)
    check("ablation A truncation marker carries no resume number",
          "output truncated" in rt.text and "start_line=" not in rt.text)

    loose = FileTools(root, ToolConfig(require_line_range=False))
    whole = loose.read_file("pkg/core.py")
    check("ablation B allows an omitted range",
          whole.start_line == 1 and whole.total_lines == len(CORE_SRC.splitlines()))
    check("ablation B still numbers output", " | " in whole.text)

    both = FileTools(root, ToolConfig(number_lines=False, require_line_range=False))
    check("both ablations compose", both.read_file("pkg/core.py").lines[0].startswith("\"\"\""))

    check("response cap is identical across configs",
          ToolConfig().max_response_chars
          == ToolConfig(number_lines=False).max_response_chars
          == ToolConfig(require_line_range=False).max_response_chars)


def test_search_code(root: Path, outside: Path) -> None:
    print("\nsearch_code")
    tools = FileTools(root)

    r = tools.search_code("needle_in_method")
    hit_paths = {m.path for m in r.matches}
    check("finds the match in tracked source", "pkg/core.py" in hit_paths)
    check("__pycache__ excluded from search", "__pycache__/ghost.py" not in hit_paths)
    check("build excluded from search", "build/artifact.py" not in hit_paths)
    check("search and list agree on the file set",
          hit_paths <= set(tools.list_files().paths), str(hit_paths))

    check("hidden directories are searched",
          any(m.path == ".github/ci.yml" for m in tools.search_code("needle_hidden_dir").matches))
    check("gitignored files are searched",
          any(m.path == "ignored.py" for m in tools.search_code("needle_gitignored").matches))

    empty = tools.search_code("this_string_appears_nowhere_at_all")
    check("no matches is a result, not an error", empty.matches == () and empty.total_found == 0)
    check("no matches says so", "(no matches)" in empty.text)

    check("binary files produce no matches",
          tools.search_code("IHDR").matches == ())

    check("regex works", len(tools.search_code(r"needle_(readme|gitignored)").matches) == 2)
    check("fixed strings disable regex",
          tools.search_code("needle_(readme", fixed_strings=True).matches == ())
    check("case sensitive by default", tools.search_code("NEEDLE_README").matches == ())
    check("ignore_case finds it", len(tools.search_code("NEEDLE_README", ignore_case=True).matches) == 1)

    scoped = tools.search_code("needle_in_method", path="pkg")
    check("path scoping restricts the search", {m.path for m in scoped.matches} == {"pkg/core.py"})

    # The interoperability that makes search results actionable.
    m = tools.search_code("needle_in_method", path="pkg").matches[0]
    back = tools.read_file(m.path, m.line_number, m.line_number)
    check("search line numbers are directly readable by read_file",
          "needle_in_method" in back.lines[0], back.lines[0])
    check("search line number matches the fixture",
          m.line_number == line_of(CORE_SRC, "needle_in_method = payload"))

    check_raises("traversal refused", PathEscapeError,
                 tools.search_code, "x", path="../")

    capped = FileTools(root, ToolConfig(max_search_matches=1)).search_code("needle")
    check("match cap truncates", capped.truncated and len(capped.matches) == 1)
    check("truncation is visible", "more matches not shown" in capped.text)


def main() -> int:
    workspace = Path(tempfile.mkdtemp(prefix="smoke-tools-"))
    root = workspace / "repo"
    root.mkdir()
    outside = workspace / "secret.py"
    outside.write_text("SECRET = 1  # needle_in_method\n", encoding="utf-8")

    try:
        build_fixture(root)
        test_construction(root, outside)
        test_binary_predicate(root)
        test_list_files(root, outside)
        test_read_file(root, outside)
        test_symbol_interop(root)
        test_ablations(root)
        test_search_code(root, outside)
    except AssertionError as exc:
        print(f"\nFAILED after {_passed} passing assertions:\n  {exc}")
        return 1
    finally:
        rmtree_robust(workspace)

    print(f"\n{_passed} assertions passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())