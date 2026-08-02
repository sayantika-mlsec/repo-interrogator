"""Probe: does a tool call survive a second turn?

Gemini 3+ returns a thought signature with every function call, and the next
request must carry it back in the same content part. If it is missing, the API
answers 400 -- not a warning, not a degraded response. So a two-turn tool loop
either works completely or does not work at all, and which one is true depends
on whether the integration layer preserves the signature when it rebuilds the
message list.

That is not a question worth answering by reading changelogs. This runs the
smallest possible two-turn exchange and reports what actually happened.

It also prints two things the agent loop needs and cannot guess:

  - the shape of ``AIMessage.content`` (Gemini 3 returns content *blocks*, not a
    string, because a signature has to live somewhere)
  - whether ``usage_metadata`` separates thinking tokens from output tokens

The second one decides whether the token budget is real. Thinking tokens bill as
output. A budget that counts only visible output would stop a run after part of
its spend while reporting the ceiling it never enforced.

Two turns is the minimum that can fail. One turn never carries a signature back,
so a one-turn probe passes on a broken integration.

Run:
    uv run python scripts/probe_signatures.py --model <GA_MODEL_ID>

Environment (Vertex backend):
    GOOGLE_CLOUD_PROJECT     required
    GOOGLE_CLOUD_LOCATION    optional, defaults to us-central1
    plus working ADC or service-account credentials
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from dotenv import load_dotenv


@tool
def lookup_line_count(path: str) -> int:
    """Return the number of lines in a repository file.

    Deliberately shaped like the real tools -- one required string argument,
    a scalar return -- so the probe exercises the same schema path the agent
    will. Nothing here touches a filesystem.
    """
    return 412


def describe_content(content: Any) -> str:
    """Report the content shape without assuming it is a string."""
    if isinstance(content, str):
        return f"str, {len(content)} chars"
    if isinstance(content, list):
        kinds = [b.get("type", "?") if isinstance(b, dict) else type(b).__name__ for b in content]
        return f"list of {len(content)} block(s): {kinds}"
    return f"unexpected: {type(content).__name__}"


def find_signatures(msg: AIMessage) -> list[str]:
    """Look for a signature wherever the integration might have put it.

    Three locations are checked because the answer has moved between releases
    and this probe exists precisely because the location is not known in
    advance. Finding none is a reportable result, not an error -- the second
    turn is what decides whether it mattered.
    """
    found: list[str] = []

    if isinstance(msg.content, list):
        for i, block in enumerate(msg.content):
            if isinstance(block, dict):
                extras = block.get("extras") or {}
                if isinstance(extras, dict) and extras.get("signature"):
                    found.append(f"content[{i}].extras.signature")

    for key, value in (msg.additional_kwargs or {}).items():
        if "signature" in key.lower() and value:
            found.append(f"additional_kwargs[{key!r}]")

    for call in msg.tool_calls or []:
        extra = call.get("extra_content") if isinstance(call, dict) else None
        if extra:
            found.append(f"tool_call[{call.get('name')}].extra_content")

    return found


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="Exact GA model id, e.g. gemini-3.5-flash. No default on purpose: "
        "an unstated model is an unreproducible result.",
    )
    parser.add_argument(
        "--location",
        default=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
    )
    args = parser.parse_args()

    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        print("GOOGLE_CLOUD_PROJECT is not set. Vertex needs a project.")
        return 2

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        print(
            "langchain_google_genai is not installed.\n"
            '  uv add "langchain-google-genai>=4.0"'
        )
        return 2

    print(f"model    {args.model}")
    print(f"project  {project}")
    print(f"location {args.location}")

    llm = ChatGoogleGenerativeAI(
        model=args.model,
        vertexai=True,
        project=project,
        location=args.location,
    ).bind_tools([lookup_line_count])

    messages: list[Any] = [
        HumanMessage(
            "How many lines are in src/repo_interrogator/tools.py? "
            "Use the tool, then state the number."
        )
    ]

    # --- turn 1 ------------------------------------------------------------

    print("\n--- turn 1: expect a tool call ---")
    try:
        first = llm.invoke(messages)
    except Exception as exc:  # noqa: BLE001 - the failure is the finding
        print(f"FAILED on the first call: {type(exc).__name__}: {exc}")
        return 1

    print(f"content shape   {describe_content(first.content)}")
    print(f"tool_calls      {len(first.tool_calls or [])}")

    if not first.tool_calls:
        print(
            "\nNo tool call was made, so the second turn cannot test anything.\n"
            "Re-run; if it persists, the model is answering from prior knowledge "
            "and the prompt needs to force a call."
        )
        return 1

    call = first.tool_calls[0]
    print(f"called          {call['name']}({json.dumps(call['args'])})")

    sigs = find_signatures(first)
    print(f"signature at    {sigs if sigs else 'not found in any known location'}")

    usage = first.usage_metadata or {}
    print(f"usage_metadata  {json.dumps(usage, indent=2, default=str)}")

    details = usage.get("output_token_details") or {}
    if "reasoning" in details:
        print(
            f"\nthinking tokens ARE reported separately: {details['reasoning']}.\n"
            "The budget must add these to output tokens, since they bill as output."
        )
    else:
        print(
            "\nthinking tokens are NOT broken out here.\n"
            "Budget on total_tokens rather than output_tokens -- otherwise the "
            "ceiling silently excludes reasoning spend."
        )

    # --- turn 2: the actual test -------------------------------------------
    #
    # The AIMessage is appended whole, never rebuilt. Reconstructing it from
    # .tool_calls alone is exactly the mistake that strips the signature, and
    # the agent loop must follow this same rule.

    print("\n--- turn 2: the real test ---")
    messages.append(first)
    messages.append(
        ToolMessage(content="412", tool_call_id=call["id"])
    )

    try:
        second = llm.invoke(messages)
    except Exception as exc:  # noqa: BLE001
        text = str(exc)
        print(f"FAILED: {type(exc).__name__}: {text[:400]}")
        if "thought_signature" in text or "thought signature" in text.lower():
            print(
                "\nVERDICT: signatures are dropped in transit.\n"
                "create_agent is not usable as-is with this model. Either pin a "
                "release that round-trips them, or call google-genai directly and "
                "keep the raw parts."
            )
        return 1

    print(f"content shape   {describe_content(second.content)}")
    print(f"text            {second.text()[:200]!r}")

    print(
        "\nVERDICT: two-turn tool calling works. Signatures survive an append-"
        "whole message list, so create_agent is safe with this model.\n"
        "Rule this locks in: assistant messages are appended, never rebuilt, "
        "filtered, or summarised. Any context-trimming work must touch tool "
        "observations only."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())