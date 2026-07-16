# Copyright © 2026 Apple Inc.

import json
from typing import Any

import regex as re

# A complete JSON string literal, so braces embedded inside string values do
# not confuse the balanced-brace matcher below.
_JSON_STR = r'"(?:[^"\\]|\\.)*"'

# Match each `name[ARGS]{...}` unit individually with *balanced* braces (via the
# regex module's recursive (?2) group). A greedy `\{.*\}` would span from the
# first call's `{` to the last call's `}`, merging several calls into a single
# invalid-JSON blob and dropping every call.
_tool_call_regex = re.compile(
    r"(\w+)\[ARGS\]\s*(\{(?:" + _JSON_STR + r"|[^{}]|(?2))*\})",
    re.DOTALL,
)

tool_call_start = "[TOOL_CALLS]"
tool_call_end = ""


def parse_tool_call(text: str, tools: Any | None = None):
    matches = list(_tool_call_regex.finditer(text))
    if not matches:
        raise ValueError(f"Could not parse tool call from: {text}")
    calls = [
        dict(name=m.group(1), arguments=json.loads(m.group(2))) for m in matches
    ]
    return calls[0] if len(calls) == 1 else calls
