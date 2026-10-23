"""Manual LangChain tool-calling loop shared by Lab 6 exercises.

The model proposes calls. Python invokes the tools and sends results back
as ToolMessage objects. A model is never granted unrestricted tool execution.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage


def get_llm():
    from dotenv import load_dotenv
    from langchain_groq import ChatGroq

    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY missing; copy .env.example to .env")
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        temperature=0,
        max_tokens=2400,
    )


def json_text(value: Any) -> str:
    return json.dumps(value, default=str, ensure_ascii=False)


def run_tool_flow(
    user_query: str,
    tools: list,
    *,
    llm=None,
    max_rounds: int = 7,
    system_extra: str = "",
    before_execute: Callable[[str, dict, list[dict]], str | None] | None = None,
) -> dict:
    """Bind tools once, manually execute model-suggested calls until final text.

    Every proposed call (including failures) is reported and sent back to the
    model under its original tool_call_id. Validation hooks can block unsafe
    ordering or inconsistent arguments before the tool runs.
    """
    if not user_query.strip():
        raise ValueError("user_query cannot be empty")
    if len({tool.name for tool in tools}) != len(tools):
        raise ValueError("Tool names must be unique")
    model = llm if llm is not None else get_llm()
    bound = model.bind_tools(tools)
    registry = {tool.name: tool for tool in tools}
    messages = [
        SystemMessage(content=(
            "Use a bound tool whenever a user asks about factual data or a "
            "calculation provided by a tool. Do not make up tool results or "
            "claim a tool succeeded without its response. Tool data marked "
            "demo is illustrative, not live. After receiving tool results, "
            "answer the user clearly. " + system_extra
        )),
        HumanMessage(content=user_query),
    ]
    rounds = []
    executed = []

    for round_no in range(1, max_rounds + 1):
        response = bound.invoke(messages)
        messages.append(response)
        calls = response.tool_calls or []
        if not calls:
            return {
                "user_query": user_query,
                "rounds": rounds,
                "final_answer": str(response.content),
                "completed": True,
            }

        round_trace = {"round": round_no, "tool_calls": calls, "manual_results": []}
        for call in calls:
            name, args = call["name"], call["args"]
            error = None
            if name not in registry:
                error = f"Tool {name!r} is not bound."
            elif not isinstance(args, dict):
                error = "Tool arguments must be a JSON object."
            elif before_execute is not None:
                try:
                    error = before_execute(name, args, executed)
                except Exception as exc:
                    error = f"Argument consistency check failed: {type(exc).__name__}: {exc}"

            if error:
                result = {"error": error}
                success = False
            else:
                try:
                    result = registry[name].invoke(args)
                    success = True
                except Exception as exc:
                    result = {"error": f"{type(exc).__name__}: {exc}"}
                    success = False

            record = {"tool": name, "arguments": args, "result": result, "success": success}
            round_trace["manual_results"].append(record)
            executed.append(record)
            messages.append(ToolMessage(
                name=name,
                tool_call_id=call["id"],
                content=json_text(result),
            ))
        rounds.append(round_trace)

    return {
        "user_query": user_query,
        "rounds": rounds,
        "final_answer": "Stopped after maximum tool-calling rounds without a final answer.",
        "completed": False,
    }


def print_flow(result: dict) -> None:
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


def used_tool_names(flow: dict) -> list[str]:
    return [item["tool"] for round_ in flow["rounds"] for item in round_["manual_results"]]

