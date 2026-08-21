# Copyright © 2026

"""
Tool parser for Cohere North (cohere2_moe) action-block tool calls.

Format (from the North-Mini-Code chat template):
<|START_ACTION|>[
    {"tool_call_id": "0", "tool_name": "get_weather", "parameters": {"city": "SF"}}
]<|END_ACTION|>

The action block is a JSON array of calls; each has `tool_name` + `parameters`
and an optional `tool_call_id` that must be threaded through as the OpenAI
tool-call id. Thinking (`<|START_THINKING|>...<|END_THINKING|>`) and text
(`<|START_TEXT|>...<|END_TEXT|>`) live outside the action block and are
handled by the reasoning/detokenizer path, not here.
"""

import json
from typing import Any

import regex as re

tool_call_start = "<|START_ACTION|>"
tool_call_end = "<|END_ACTION|>"

_action_regex = re.compile(r"<\|START_ACTION\|>(.*?)<\|END_ACTION\|>", re.DOTALL)

# One bounded repair pass for the two North failure modes seen in the wild:
# the array wrapped in a markdown code fence, and a trailing comma before a
# closing `}`/`]`. Applied at most once per block -- no retry loop.
_fence_regex = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)

# Alternate between whole JSON string literals (left untouched, group 1) and a
# trailing comma before a closing bracket (group 2, comma dropped). Scanning
# left-to-right through this alternation -- rather than a bare `,(\s*[}\]])`
# -- keeps the repair from reaching into string *values*, e.g. an argument
# like {"msg": "hello, }bye"} must survive the repair pass unchanged.
_JSON_STR = r'"(?:[^"\\]|\\.)*"'
_trailing_comma_regex = re.compile(
    r"(" + _JSON_STR + r")" + r"|" + r",(\s*[}\]])", re.DOTALL
)


def _strip_code_fence(s: str) -> str:
    m = _fence_regex.match(s)
    return m.group(1) if m else s


def _drop_trailing_commas(s: str) -> str:
    def repl(m):
        return m.group(1) if m.group(1) is not None else m.group(2)

    return _trailing_comma_regex.sub(repl, s)


def _loads_with_repair(s: str):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        repaired = _drop_trailing_commas(_strip_code_fence(s))
        return json.loads(repaired)


def _normalize(call: Any):
    if not isinstance(call, dict):
        return None

    name = call.get("tool_name") or call.get("name") or call.get("function")
    if not isinstance(name, str) or not name:
        return None

    call_id = call.get("tool_call_id")
    if call_id is None:
        call_id = call.get("id")

    args = call.get("parameters")
    if args is None:
        args = call.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return None
    if args is None:
        args = {}
    if not isinstance(args, dict):
        # North arguments must be a JSON object; a list/string/scalar here
        # means the block is malformed -- drop this call, keep its siblings.
        return None

    out = {"name": name, "arguments": args}
    if call_id is not None:
        out["id"] = str(call_id)
    return out


def _parse_block(block: str):
    block = block.strip()
    if not block:
        return None
    try:
        data = _loads_with_repair(block)
    except json.JSONDecodeError:
        # tolerate a bare single object without the array wrapper
        try:
            data = _loads_with_repair(f"[{block}]")
        except json.JSONDecodeError:
            return None

    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return None

    calls = [c for c in (_normalize(item) for item in data) if c is not None]
    if not calls:
        return None
    return calls[0] if len(calls) == 1 else calls


def parse_tool_call(text: str, tools: list[Any] | None = None):
    matches = list(_action_regex.finditer(text))
    calls: list = []
    if matches:
        # Every complete block contributes its calls; one malformed block
        # never drops the calls parsed from its siblings.
        for m in matches:
            parsed = _parse_block(m.group(1))
            if parsed is not None:
                calls.extend(parsed if isinstance(parsed, list) else [parsed])
    else:
        # markers already stripped by the streaming layer -> parse the remainder
        parsed = _parse_block(text)
        if parsed is not None:
            calls.extend(parsed if isinstance(parsed, list) else [parsed])

    if not calls:
        raise ValueError("No valid North action block found in tool-call text.")
    return calls[0] if len(calls) == 1 else calls
