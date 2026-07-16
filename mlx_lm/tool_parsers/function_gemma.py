# Copyright © 2025-2026 Apple Inc.

import json
from typing import Any, Optional

import regex as re

# A complete <escape>...<escape> string literal. It may contain braces or
# commas, which must NOT terminate the surrounding call:name{...} block.
_ESCAPE_STR = r"<escape>(?:(?!<escape>)[\s\S])*?<escape>"

# Match call:name{...} with *balanced* braces (via the regex module's
# recursive (?2) group) so dict-valued args are captured whole instead of
# being truncated at the first '}'. Escape literals are consumed atomically
# so braces inside string values never unbalance the match.
_tool_call_regex = re.compile(
    r"call:(\w+)(\{(?:" + _ESCAPE_STR + r"|[^{}]|(?2))*\})",
    re.DOTALL,
)


def _args_to_json(text: str) -> str:
    """Convert function_gemma args to valid JSON.

    Keys are unquoted and string values are wrapped in <escape>...<escape>
    delimiters instead of quotes. Nested objects are handled naturally because
    bare keys are quoted at every brace/comma boundary.
    """
    strings = []

    def _capture(m):
        strings.append(m.group(1))
        return f"\x00{len(strings) - 1}\x00"

    # Pull out <escape>-delimited strings first so their contents are inert.
    text = re.sub(r"<escape>(.*?)<escape>", _capture, text, flags=re.DOTALL)
    # Quote bare keys (any nesting level).
    text = re.sub(r"(?<=[{,])(\w+):", r'"\1":', text)
    # Restore captured strings as properly escaped JSON strings.
    for i, s in enumerate(strings):
        text = text.replace(f"\x00{i}\x00", json.dumps(s))
    return text


def _parse_single(match: "re.Match") -> dict:
    func_name = match.group(1)
    args_str = match.group(2)
    arguments = json.loads(_args_to_json(args_str))
    return dict(name=func_name, arguments=arguments)


def parse_tool_call(text: str, _: Optional[Any] = None):
    matches = list(_tool_call_regex.finditer(text))
    if not matches:
        raise ValueError("No function provided.")
    if len(matches) == 1:
        return _parse_single(matches[0])
    return [_parse_single(m) for m in matches]


tool_call_start = "<start_function_call>"
tool_call_end = "<end_function_call>"
