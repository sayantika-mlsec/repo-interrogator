"""Smoke test for the agent layer.

Two halves, and the split is the point.

The **offline** half needs no network and no credentials. It exercises the tool
wrappers against a fixture tree, the output schema against malformed input, and
the budget arithmetic against a fake message stream. That covers everything in
this layer that is not the model, which is most of it -- and it runs in a second,
so it can run on every commit.

The **live** half needs a model and costs money, so it is opt-in behind
``--live``. It confirms two things nothing offline can: that the tool schemas are
acceptable to the provider, and that a two-turn tool exchange still works.

The schema check is the one worth being explicit about. ``read_file`` has two
optional integer parameters, which is the shape most likely to be rejected or
silently mangled in conversion. Ablation B -- removing the required line range --
depends on the model being able to omit them. If the schema cannot express that,
the ablation cannot be run, and finding out on the day it is scheduled is too
late.

Run:  uv run python scripts/smoke_agent.py
      uv run python scripts/smoke_agent.py --live --model <GA_MODEL_ID>
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import ValidationError

from repo_interrogator.agent import (
    Citation,
    Question,
    RepoAgent,
    RunLimits,
    RunProgress,
    build_tools,
    default_task
)
from repo_interrogator.errors import (
    AgentConfigurationError,
    StepBudgetExceededError,
    TokenBudgetExceededError,
)
from repo_interrogator.fsutil import rmtree_robust
from repo_interrogator.symbols import build_index_at
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
'''

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
    except BaseException as exc:  # noqa: BLE001
        raise AssertionError(
            f"{label}: expected {exc_type.__name__}, got {type(exc).__name__}: {exc}"
        ) from exc
    raise AssertionError(f"{label}: expected {exc_type.__name__}, nothing raised")


def build_fixture(root: Path) -> None:
    (root / "pkg").mkdir()
    (root / "pkg" / "core.py").write_text(CORE_SRC, encoding="utf-8")
    (root / "README.md").write_text("# fixture\n", encoding="utf-8")
    (root / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\r")


def named(tools: list, name: str):
    for t in tools:
        if t.name == name:
            return t
    raise AssertionError(f"no tool named {name}")


# --- schema ----------------------------------------------------------------


def test_schema() -> None:
    print("\noutput schema")

    c = Citation(path="pkg/core.py", start_line=4, end_line=6)
    check("a well-formed citation validates", c.end_line == 6)
    check("single-line citation is legal", Citation(
        path="a.py", start_line=3, end_line=3).start_line == 3)

    check_raises("zero start rejected at the boundary", ValidationError,
                 Citation, path="a.py", start_line=0, end_line=5)
    check_raises("negative start rejected", ValidationError,
                 Citation, path="a.py", start_line=-1, end_line=5)
    check_raises("inverted range rejected", ValidationError,
                 Citation, path="a.py", start_line=9, end_line=4)

    q = Question(question="Why does alpha default b to 1?", citations=[c])
    check("a question with a citation validates", len(q.citations) == 1)
    check_raises("a question with no citation is rejected", ValidationError,
                 Question, question="What does this repo do?", citations=[])

    schema = Question.model_json_schema()
    check("question schema names its fields",
          set(schema["properties"]) == {"question", "citations"}, str(schema["properties"].keys()))


# --- tool wrappers ---------------------------------------------------------


def test_wrappers(root: Path) -> None:
    print("\ntool wrappers")
    index = build_index_at(root, "fixture", "0" * 40)
    tools_obj = FileTools(root, symbol_index=index)
    sink: list[Question] = []
    tools = build_tools(tools_obj, sink)

    check("five tools, no more and no fewer", len(tools) == 5, str([t.name for t in tools]))
    check("finish is one of them", "finish" in {t.name for t in tools})
    check("every tool has a description",
          all(t.description and len(t.description) > 20 for t in tools))

    listed = named(tools, "list_files").invoke({})
    check("list_files returns text", "pkg/core.py" in listed)

    read = named(tools, "read_file").invoke(
        {"path": "pkg/core.py", "start_line": 4, "end_line": 6})
    check("read_file returns numbered text", "4 | def alpha" in read, read)

    syms = named(tools, "get_symbols").invoke({"path": "pkg/core.py"})
    check("get_symbols lists definitions with ranges", "Engine.run" in syms and "lines " in syms)

    found = named(tools, "search_code").invoke({"pattern": "needle_in_method"})
    check("search_code returns matches", "pkg/core.py:" in found)

    # --- errors the model should see ---------------------------------------

    missing = named(tools, "read_file").invoke(
        {"path": "nope.py", "start_line": 1, "end_line": 2})
    check("a missing file comes back as an observation, not an exception",
          missing.startswith("FileNotFoundInRepoError"), missing)

    no_range = named(tools, "read_file").invoke({"path": "pkg/core.py"})
    check("a missing range comes back as an observation",
          no_range.startswith("LineRangeRequiredError"), no_range)

    binary = named(tools, "read_file").invoke(
        {"path": "logo.png", "start_line": 1, "end_line": 2})
    check("binary content comes back as an observation",
          binary.startswith("BinaryFileError"), binary)

    check("the observation names the error type",
          missing.split(":")[0] == "FileNotFoundInRepoError")

    # --- errors the model must not see -------------------------------------

    check_raises(
        "an escape attempt is not handed back as a hint",
        Exception,
        named(tools, "read_file").invoke,
        {"path": "../outside.py", "start_line": 1, "end_line": 2},
    )

    # --- no index ----------------------------------------------------------

    bare = build_tools(FileTools(root), [])
    unavailable = named(bare, "get_symbols").invoke({"path": "pkg/core.py"})
    check("get_symbols without an index is an observation, not a crash",
          unavailable.startswith("SymbolsUnavailableError"), unavailable)

    # --- finish ------------------------------------------------------------

    payload = {"questions": [{
        "question": "What does Engine.run return?",
        "citations": [{"path": "pkg/core.py", "start_line": 12, "end_line": 14}],
    }]}
    ack = named(tools, "finish").invoke(payload)
    check("finish acknowledges", "1 questions" in ack or "Recorded 1" in ack)
    check("finish delivers validated objects to the sink",
          len(sink) == 1 and isinstance(sink[0], Question))
    check("the citation survived as a model, not a dict",
          isinstance(sink[0].citations[0], Citation))

    check_raises("finish rejects a citationless question", Exception,
                 named(tools, "finish").invoke,
                 {"questions": [{"question": "vague?", "citations": []}]})


def test_optional_parameter_shape(root: Path) -> None:
    """The shape Ablation B depends on.

    ``start_line`` and ``end_line`` must be omittable. If they are declared
    required, the required-range behaviour is enforced by the schema rather than
    by ``ToolConfig``, and turning the config knob off would change nothing --
    the ablation would silently measure zero.
    """
    print("\noptional-parameter shape (Ablation B depends on this)")
    index = build_index_at(root, "fixture", "0" * 40)
    tools = build_tools(FileTools(root, symbol_index=index), [])
    schema = named(tools, "read_file").args_schema.model_json_schema()

    required = set(schema.get("required", []))
    check("path is required", "path" in required)
    check("start_line is omittable", "start_line" not in required, str(required))
    check("end_line is omittable", "end_line" not in required, str(required))

    loose = build_tools(
        FileTools(root, ToolConfig(require_line_range=False), symbol_index=index), [])
    whole = named(loose, "read_file").invoke({"path": "pkg/core.py"})
    check("under the ablation, an omitted range reads the file",
          "def alpha" in whole and not whole.startswith("LineRangeRequiredError"))

    print("    read_file schema (inspect if the provider rejects it):")
    print("    " + json.dumps(schema.get("properties", {}), indent=2)[:600].replace("\n", "\n    "))


# --- budgets ---------------------------------------------------------------


class _FakeAgent:
    """A stand-in that emits states without calling a model.

    The budget logic is arithmetic over a message stream, and arithmetic is
    testable without spending anything. Making the loop take its stream from an
    object rather than reaching for a model is what allows this.
    """

    def __init__(self, states: list[dict]) -> None:
        self._states = states
        self.dispatched = 0

    def stream(self, _inputs, _config, stream_mode="values"):  # noqa: ANN001
        for state in self._states:
            self.dispatched += 1
            yield state


def _ai(tokens: int) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "list_files", "args": {}, "id": "call_1"}],
        usage_metadata={
            "input_tokens": tokens // 2,
            "output_tokens": tokens // 2,
            "total_tokens": tokens,
            "output_token_details": {"reasoning": tokens // 3},
        },
    )


def _make_agent(root: Path, states: list[dict], limits: RunLimits) -> RepoAgent:
    """A ``RepoAgent`` with the model half replaced and nothing else.

    ``__new__`` because ``__init__`` constructs a live client. The cost of that
    choice is that this function has to know every attribute ``run`` touches: new
    instance state on ``RepoAgent`` breaks it. It breaks loudly -- an
    ``AttributeError`` on the first run -- which is the acceptable version of
    that coupling, but it is coupling.
    """
    agent = RepoAgent.__new__(RepoAgent)
    agent.file_tools = FileTools(root)
    agent.model_id = "fixture-model"
    agent.limits = limits
    agent._sink = []
    agent._progress = RunProgress()
    agent._agent = _FakeAgent(states)
    return agent


def test_budgets(root: Path) -> None:
    print("\nbudgets")

    many = [{"messages": [_ai(100)] * (i + 1)} for i in range(50)]

    agent = _make_agent(root, many, RunLimits(max_steps=5, max_total_tokens=10_000_000))
    check_raises("the step ceiling raises rather than continuing",
                 StepBudgetExceededError, agent.run, "task")
    check("the run stopped at the ceiling, not past it",
          agent._agent.dispatched == 5, str(agent._agent.dispatched))

    # A breached run is the run most worth reading, so its counters and its
    # trajectory outlive the exception. Without these two assertions the
    # progress object could quietly go back to being loop-local and every
    # failed run would again cost full price and leave nothing behind.
    check("the breached run kept its counters",
          agent.progress.steps == 5 and agent.progress.tokens == 500,
          f"steps={agent.progress.steps} tokens={agent.progress.tokens}")
    check("the breached run kept its trajectory",
          len(agent.progress.messages) == 5, str(len(agent.progress.messages)))

    agent = _make_agent(root, many, RunLimits(max_steps=1000, max_total_tokens=250))
    check_raises("the token ceiling raises", TokenBudgetExceededError, agent.run, "task")
    check("tokens counted as total, not visible output",
          agent._agent.dispatched == 3, str(agent._agent.dispatched))
    check("a token breach keeps the spend that caused it",
          agent.progress.tokens == 300, str(agent.progress.tokens))

    def _messages_for(n: int) -> dict:
        return {"messages": [_ai(100)] * n}

    agent = _make_agent(root, [_messages_for(1), _messages_for(2)], RunLimits())
    check_raises("a run that never calls finish is a failure, not an empty result",
                 Exception, agent.run, "task")

    tool_err = ToolMessage(content="boom", tool_call_id="call_1", status="error")
    agent = _make_agent(root, [{"messages": [_ai(10), tool_err]}], RunLimits())
    check_raises("a framework-captured error is surfaced, not absorbed",
                 RuntimeError, agent.run, "task")


def test_construction(root: Path) -> None:
    print("\nconstruction")
    check_raises("an empty model id is refused at construction",
                 AgentConfigurationError, RepoAgent, FileTools(root), "")
    check("limits serialise for a results row",
          RunLimits(max_steps=7).to_dict()["max_steps"] == 7)

    agent = _make_agent(root, [], RunLimits())
    check_raises("an empty task is refused before any spend",
                 AgentConfigurationError, agent.run, "")
    check_raises("a whitespace task is refused too",
                 AgentConfigurationError, agent.run, "   ")
    check("the standard task states its question count",
          "3 questions" in default_task(3), default_task(3))


# --- live ------------------------------------------------------------------


def test_live(root: Path, model_id: str) -> None:
    print(f"\nlive run against {model_id}")
    import os

    from repo_interrogator.agent import build_agent

    index = build_index_at(root, "fixture", "0" * 40)
    agent = build_agent(
        root,
        model_id,
        symbol_index=index,
        limits=RunLimits(max_steps=12, max_total_tokens=120_000),
        location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
        project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
    )

    result = agent.run(default_task(3))

    check("questions came back", len(result.questions) > 0)
    check("every question carries a citation",
          all(q.citations for q in result.questions))
    check("steps were counted", result.steps > 0)
    check("tokens were counted", result.total_tokens > 0)
    check("at least one tool was called", result.tool_calls > 0)

    for q in result.questions:
        print(f"    Q: {q.question}")
        for c in q.citations:
            print(f"       {c.path}:{c.start_line}-{c.end_line}")

    print(f"\n    steps={result.steps} tokens={result.total_tokens} "
          f"tool_calls={result.tool_calls} tool_errors={result.tool_errors}")


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="also run against a real model")
    parser.add_argument("--model", help="GA model id, required with --live")
    args = parser.parse_args()

    if args.live and not args.model:
        print("--live requires --model")
        return 2

    if args.model and not args.live:
        print("--model without --live does nothing; add --live to run against the model")
        return 2

    workspace = Path(tempfile.mkdtemp(prefix="smoke-agent-"))
    root = workspace / "repo"
    root.mkdir()
    (workspace / "outside.py").write_text("SECRET = 1\n", encoding="utf-8")

    try:
        build_fixture(root)
        test_schema()
        test_wrappers(root)
        test_optional_parameter_shape(root)
        test_budgets(root)
        test_construction(root)
        if args.live:
            test_live(root, args.model)
    except AssertionError as exc:
        print(f"\nFAILED after {_passed} passing assertions:\n  {exc}")
        return 1
    finally:
        rmtree_robust(workspace)

    print(f"\n{_passed} assertions passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())