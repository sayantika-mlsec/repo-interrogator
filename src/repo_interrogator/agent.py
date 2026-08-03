"""The agent loop: five tools, a validated output schema, and two hard budgets.

WHAT THIS LAYER OWNS
--------------------
Everything below it is deterministic. ``FileTools`` reads bytes, ``SymbolIndex``
reports definitions, and both fail loudly on anything they cannot do. This layer
is the first one containing a model, so it is the first one where "it kept
going" is a possible failure rather than an impossible one. Its job is to make
that failure impossible too.

FIVE TOOLS, AND WHY ``finish`` IS ONE OF THEM
---------------------------------------------
``finish`` could have been a special case in the loop -- watch for a final text
turn, parse questions out of it. It is a tool instead, and it carries the
questions as its argument.

Two consequences follow, and both matter later. Termination becomes a row in the
same trace table as every other call, so "how did this run end" is answered by
the same query as "what did this run do". And output cannot be separated from
termination: there is no run that stopped without producing questions, and no
run that produced questions without stopping. A final-text-turn design allows
both, and both are silent.

THE SCHEMA IS THE CONTRACT
--------------------------
Questions come back Pydantic-validated. A citation that is missing a line range,
or carries a range that is inverted, is rejected at the boundary rather than
discovered by the verifier three days later. The verifier's job is to check
whether a citation *resolves*; it should never also be checking whether the
citation is *well formed*.

THE TASK STRING IS SENT VERBATIM
--------------------------------
``run`` takes one task string and sends it as the human turn unchanged. Nothing
is interpolated inside this layer.

The reason is that the message list is the run's record. If the prompt were
assembled here from separate arguments, reproducing a row would require knowing
every argument *and* the interpolation rule, and a prompt variant would exist
only as a parameter value that got formatted away. Sending an authored string
means the prompt in the trace is the prompt that was written, and a future
variant is visible in the message list rather than implied by a call signature.

``default_task()`` supplies the standard wording. Callers may substitute their
own.

BUDGETS ARE ENFORCED BEFORE THE SPEND, NOT AFTER
------------------------------------------------
Both budgets are checked before the next model call is dispatched, never after
it returns. Checking afterwards means the run stops having already spent past
the ceiling, and then records a number larger than the limit it claims to
enforce. That number is the one that goes in a results table.

The token budget counts ``total_tokens``, not ``output_tokens``. On this model
family, thinking tokens bill as output and were 73% of output tokens in the
first probe run -- a budget on visible output would have been wrong by roughly
four times. ``total_tokens`` also includes the whole message history resent on
every step, which is the actual billed spend and the thing worth bounding.

Neither budget is a soft target. Both raise.

ASSISTANT MESSAGES ARE APPENDED, NEVER REBUILT
----------------------------------------------
Gemini 3+ returns a thought signature with every function call, and the next
request must carry it back. Appending the ``AIMessage`` whole preserves it;
reconstructing one from ``.tool_calls`` drops it and the next call returns 400.

This is a standing constraint on future work, not just on this file. Any
context-trimming, deduplication or summarisation must operate on tool
observations only. The moment an assistant turn is rewritten, tool calling stops
working -- loudly, but only on the second call of a run.

ERRORS: WHICH ONES THE MODEL SEES
---------------------------------
A ``ToolError`` is returned to the model as an observation. It means the model
guessed a path that does not exist, or omitted a required range, or asked for
symbols in a repository that has none. That is the loop working: the model
should read the message and try something else.

Everything else propagates and kills the run. ``PathEscapeError`` is the clearest
case -- it is a ``WorkspaceError``, not a ``ToolError``, and it means the model
tried to leave the repository. Handing that back as an observation would turn a
containment breach into a hint.

A returned error still costs a step. Otherwise a model failing every call runs
forever for free.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

from .errors import (
    AgentConfigurationError,
    NoFinishError,
    StepBudgetExceededError,
    TokenBudgetExceededError,
    ToolError,
)
from .symbols import SymbolIndex
from .tools import FileTools, ToolConfig

log = logging.getLogger(__name__)


# --- output schema ---------------------------------------------------------


class Citation(BaseModel):
    """One pointer into the repository, at the pinned commit.

    Line numbers are 1-based and inclusive on both ends -- the same contract the
    symbol layer enforces and ``read_file`` accepts. The model reads ranges in
    that form from both tools, so it can only produce them in that form.
    """

    path: str = Field(description="Repository-relative path, e.g. src/pkg/mod.py")
    start_line: int = Field(description="First line, 1-based inclusive")
    end_line: int = Field(description="Last line, 1-based inclusive")

    @field_validator("start_line")
    @classmethod
    def _start_is_one_based(cls, v: int) -> int:
        if v < 1:
            raise ValueError("line numbers are 1-based; 0 means a conversion was missed")
        return v

    @field_validator("end_line")
    @classmethod
    def _end_is_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("line numbers are 1-based")
        return v

    def model_post_init(self, _: Any) -> None:
        if self.end_line < self.start_line:
            raise ValueError(
                f"{self.path}: end_line={self.end_line} precedes "
                f"start_line={self.start_line}. Ranges are inclusive."
            )


class Question(BaseModel):
    """One question about the repository, with the evidence it rests on.

    ``citations`` is required and non-empty. A question with no citation cannot
    be verified, and an unverifiable question in the output set would be scored
    as if it had passed a check it never took.
    """

    question: str = Field(description="A specific question about how this repository works")
    citations: list[Citation] = Field(
        description="The code this question is grounded in. At least one."
    )

    @field_validator("citations")
    @classmethod
    def _at_least_one(cls, v: list[Citation]) -> list[Citation]:
        if not v:
            raise ValueError("a question with no citation cannot be verified")
        return v


# --- run configuration and result ------------------------------------------


@dataclass(frozen=True)
class RunLimits:
    """Hard ceilings. Both raise; neither is advisory.

    Defaults are starting points, not findings. Set them from observed
    trajectories once real runs exist.
    """

    max_steps: int = 30
    """Model calls, not tool calls. A step that returns a tool error still counts."""

    max_total_tokens: int = 400_000
    """Cumulative billed tokens across the run, including thinking and resent history."""

    def to_dict(self) -> dict[str, int]:
        return {"max_steps": self.max_steps, "max_total_tokens": self.max_total_tokens}


@dataclass
class RunResult:
    """What one completed run produced, and what it cost.

    Only constructed on success. A run that breached a budget raises instead --
    a partial result recorded as a result is how a truncated run ends up in a
    table looking like a finished one.
    """

    questions: tuple[Question, ...]
    steps: int
    total_tokens: int
    tool_calls: int
    tool_errors: int
    messages: tuple[BaseMessage, ...] = field(repr=False)

    def to_row(self) -> dict[str, Any]:
        """The provenance half of a results row. Config and model id come from the caller."""
        return {
            "n_questions": len(self.questions),
            "steps": self.steps,
            "total_tokens": self.total_tokens,
            "tool_calls": self.tool_calls,
            "tool_errors": self.tool_errors,
        }


# --- tool construction -----------------------------------------------------


SYSTEM_PROMPT = """\
You are examining a repository you have never seen. Your task is to produce \
questions about how it actually works, each grounded in specific code.

You have five tools. Use them to navigate; do not guess at contents.

- list_files: see what exists
- get_symbols: definitions in one file, with line ranges
- read_file: read a line range
- search_code: find text across the repository
- finish: return your questions and stop

Work from structure to detail: find the files that matter, list their \
definitions, then read the ranges that look load-bearing. Prefer questions whose \
answer is visible in code you have actually read over questions any repository \
of this kind would produce.

Every citation must point at a range you read. When you are done, call finish."""


def default_task(n_questions: int = 10) -> str:
    """The standard task string. Callers may substitute their own.

    A function rather than a constant carrying a format placeholder: a caller
    who forgets to format a constant sends the model a literal brace, and the
    run still completes -- producing a real result against a prompt nobody
    intended. A function cannot be used un-called.
    """
    return (
        f"Produce {n_questions} questions about this repository. "
        "Investigate first; call finish when you are done."
    )


def _render_tool_error(exc: ToolError) -> str:
    """Turn a tool failure into an observation the model can act on.

    The class name is included deliberately. "does not exist" and "is a
    directory" call for different next moves, and the message alone does not
    always make the distinction legible to a model skimming an observation.
    """
    return f"{type(exc).__name__}: {exc}"


def build_tools(file_tools: FileTools, sink: list[Question]) -> list[Any]:
    """Wrap the file tools for the model, plus ``finish``.

    ``sink`` receives the validated questions when ``finish`` is called. A
    mutable collector rather than a parsed return value: the questions must
    survive whatever the framework does with the tool's own return string, and
    they must be identical to the objects Pydantic validated rather than a
    re-parse of their rendering.

    Every wrapper catches ``ToolError`` and returns the message as text.
    Anything else propagates -- see the module docstring.
    """

    @tool
    def list_files(subdir: str | None = None, pattern: str | None = None) -> str:
        """List files in the repository.

        Args:
            subdir: Optional directory to list under, repository-relative.
            pattern: Optional glob, matched against the full relative path.
                Note that * crosses directory separators, so "*.py" matches
                nested files.
        """
        try:
            return file_tools.list_files(subdir, pattern=pattern).text
        except ToolError as exc:
            return _render_tool_error(exc)

    @tool
    def read_file(path: str, start_line: int | None = None, end_line: int | None = None) -> str:
        """Read a range of lines from one file.

        Args:
            path: Repository-relative path.
            start_line: First line, 1-based inclusive.
            end_line: Last line, 1-based inclusive.
        """
        try:
            return file_tools.read_file(path, start_line, end_line).text
        except ToolError as exc:
            return _render_tool_error(exc)

    @tool
    def search_code(
        pattern: str,
        path: str | None = None,
        ignore_case: bool = False,
        fixed_strings: bool = False,
    ) -> str:
        """Search file contents across the repository.

        Args:
            pattern: A regular expression, or a literal if fixed_strings is true.
            path: Optional file or directory to restrict the search to.
            ignore_case: Match case-insensitively.
            fixed_strings: Treat pattern as a literal string, not a regex.
        """
        try:
            return file_tools.search_code(
                pattern, path=path, ignore_case=ignore_case, fixed_strings=fixed_strings
            ).text
        except ToolError as exc:
            return _render_tool_error(exc)

    @tool
    def get_symbols(path: str) -> str:
        """List every function, method and class defined in one Python file, with line ranges.

        Args:
            path: Repository-relative path to a .py file.
        """
        try:
            return file_tools.get_symbols(path).text
        except ToolError as exc:
            return _render_tool_error(exc)

    @tool
    def finish(questions: list[Question]) -> str:
        """Return the finished questions and end the run.

        Call this exactly once, when you have gathered enough evidence.

        Args:
            questions: The questions, each with at least one citation pointing
                at a line range you actually read.
        """
        sink.extend(questions)
        return f"Recorded {len(questions)} questions. Run complete."

    return [list_files, get_symbols, read_file, search_code, finish]


# --- the loop --------------------------------------------------------------


class RepoAgent:
    """One agent, bound to one repository at one commit, under one budget.

    Construction resolves the model and builds the graph. Both fail here rather
    than mid-run, for the same reason ``FileTools`` resolves ripgrep in its
    constructor.
    """

    def __init__(
        self,
        file_tools: FileTools,
        model_id: str,
        *,
        limits: RunLimits | None = None,
        location: str = "global",
        project: str | None = None,
    ) -> None:
        if not model_id:
            raise AgentConfigurationError(
                "model_id is required and has no default. A results row that does "
                "not state its model is not reproducible."
            )

        self.file_tools = file_tools
        self.model_id = model_id
        self.limits = limits or RunLimits()

        try:
            from langchain.agents import create_agent
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise AgentConfigurationError(
                "langchain and langchain-google-genai>=4.0 are required: "
                'uv add langchain "langchain-google-genai>=4.0"'
            ) from exc

        self._sink: list[Question] = []
        self._tools = build_tools(file_tools, self._sink)

        llm_kwargs: dict[str, Any] = {"model": model_id, "vertexai": True, "location": location}
        if project:
            llm_kwargs["project"] = project

        self._llm = ChatGoogleGenerativeAI(**llm_kwargs)
        self._agent = create_agent(model=self._llm, tools=self._tools)

    # --- budget accounting -------------------------------------------------

    @staticmethod
    def _tokens_of(msg: AIMessage) -> int:
        """Billed tokens for one model call.

        ``total_tokens`` rather than ``output_tokens``: thinking tokens bill as
        output and dominate on this model family, and the resent history is real
        spend that grows with every step.
        """
        usage = msg.usage_metadata or {}
        return int(usage.get("total_tokens") or 0)

    def _check_budgets(self, steps: int, tokens: int) -> None:
        """Raise before dispatching the next call, never after it returns."""
        if steps >= self.limits.max_steps:
            raise StepBudgetExceededError(
                f"{steps} model calls reached the ceiling of {self.limits.max_steps} "
                f"without finish() being called ({tokens} tokens spent)"
            )
        if tokens >= self.limits.max_total_tokens:
            raise TokenBudgetExceededError(
                f"{tokens} tokens reached the ceiling of {self.limits.max_total_tokens} "
                f"after {steps} model calls, without finish() being called"
            )

    # --- run ---------------------------------------------------------------

    def run(self, task: str) -> RunResult:
        """Generate questions about the repository. Raises on any budget breach.

        ``task`` is sent verbatim as the human turn. Nothing is interpolated
        here, so the prompt recorded in ``RunResult.messages`` is the prompt that
        was authored -- see the module docstring. ``default_task()`` supplies the
        standard wording.

        The stream is consumed lazily, and the budget check sits between
        receiving one state and asking for the next. Because the generator does
        no work until it is advanced, raising here stops the run *before* the
        next model call is dispatched rather than after it has been paid for.
        """
        if not task or not task.strip():
            raise AgentConfigurationError(
                "task is required and has no default. An empty human turn produces "
                "a run whose prompt is not recoverable from its own trace."
            )

        self._sink.clear()

        messages: list[BaseMessage] = [
            SystemMessage(SYSTEM_PROMPT),
            HumanMessage(task),
        ]

        steps = 0
        tokens = 0
        tool_calls = 0
        tool_errors = 0
        final: list[BaseMessage] = []

        # recursion_limit is a backstop, not the budget. It stops runaway
        # graph traversal; the checks below are what produce a reportable
        # number and a message naming which ceiling was hit.
        config = {"recursion_limit": self.limits.max_steps * 2 + 10}

        for state in self._agent.stream({"messages": messages}, config, stream_mode="values"):
            current: list[BaseMessage] = state.get("messages", [])
            if not current:
                continue
            final = current
            latest = current[-1]

            if isinstance(latest, AIMessage):
                steps += 1
                tokens += self._tokens_of(latest)
                tool_calls += len(latest.tool_calls or [])

            elif isinstance(latest, ToolMessage):
                # A ToolError was caught in the wrapper and returned as normal
                # content, so it never arrives with an error status. Anything
                # that does arrive with one escaped a wrapper and was swallowed
                # by the framework -- which is exactly the silent failure this
                # project refuses. Surface it.
                if getattr(latest, "status", "success") == "error":
                    raise RuntimeError(
                        f"a non-ToolError escaped the tool wrappers and was captured "
                        f"by the framework: {latest.content}"
                    )
                if isinstance(latest.content, str) and latest.content.startswith(
                    ("ToolError", "FileNotFoundInRepoError", "NotAFileError",
                     "BinaryFileError", "FileDecodeError", "FileTooLargeError",
                     "LineRangeRequiredError", "InvalidLineRangeError",
                     "SearchFailedError", "SymbolsUnavailableError")
                ):
                    tool_errors += 1

            if self._sink:
                break

            self._check_budgets(steps, tokens)

        if not self._sink:
            raise NoFinishError(
                f"the run ended after {steps} model calls and {tokens} tokens without "
                "calling finish(). No questions were produced."
            )

        log.info(
            "%s: %d questions, %d steps, %d tokens, %d tool calls (%d errors)",
            self.file_tools.root.name, len(self._sink), steps, tokens, tool_calls, tool_errors,
        )

        return RunResult(
            questions=tuple(self._sink),
            steps=steps,
            total_tokens=tokens,
            tool_calls=tool_calls,
            tool_errors=tool_errors,
            messages=tuple(final),
        )


def build_agent(
    root: Path,
    model_id: str,
    *,
    symbol_index: SymbolIndex | None = None,
    tool_config: ToolConfig | None = None,
    limits: RunLimits | None = None,
    location: str = "global",
    project: str | None = None,
) -> RepoAgent:
    """Construct an agent over an already-cloned tree.

    The tree is cloned, pinned and indexed by the caller. This layer never
    fetches anything: an agent that could clone could clone something other than
    the pinned commit, and the pin is the only reason any number here means
    anything.
    """
    tools = FileTools(root, tool_config, symbol_index=symbol_index)
    return RepoAgent(tools, model_id, limits=limits, location=location, project=project)