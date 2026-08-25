# Copyright © 2025-2026 Apple Inc.

"""
Modified from:
https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct/blob/main/qwen3coder_tool_parser.py
"""

import ast
import json
from typing import Any, Optional

import regex as re

from ._schema import infer_type_from_json_schema

# Match each <function=...>...</function> block individually (no trailing `$`
# anchor, which would otherwise merge several blocks into one greedy match and
# drop calls 2..n).
_function_regex = re.compile(r"<function=(.*?)</function>", re.DOTALL)
_parameter_regex = re.compile(r"<parameter=(.*?)</parameter>", re.DOTALL)

_string_types = {"string", "str", "text", "varchar", "char", "enum"}
_bool_types = {"boolean", "bool", "binary"}
_obj_types = {"object", "array", "arr"}


def _get_arguments_config(func_name: str, tools: Optional[Any]) -> dict:
    """Extract argument configuration for a function."""
    if tools is None:
        return {}
    for tool in tools:
        if not (function := tool.get("function", False)):
            continue
        if function["name"] == func_name:
            if not (params := function.get("parameters", False)):
                return {}
            return params.get("properties", {})
    return {}


def _convert_param_value(param_value: str, param_name: str, param_config: dict) -> Any:
    """Convert parameter value based on its type in the schema."""
    if param_value.lower() == "null":
        return None

    if not (param := param_config.get(param_name, False)):
        return param_value

    # Resolve anyOf/oneOf/list-form unions to a concrete non-null type; an
    # unresolved schema is treated as a string (values returned verbatim).
    inferred = infer_type_from_json_schema(param)
    param_type = inferred.strip().lower() if inferred else "string"
    if param_type in _string_types:
        return param_value
    elif (
        param_type.startswith("int")
        or param_type.startswith("uint")
        or param_type.startswith("long")
        or param_type.startswith("short")
        or param_type.startswith("unsigned")
    ):
        float_param_value = float(param_value)
        int_param_value = int(float_param_value)
        if float_param_value - int_param_value != 0:
            raise ValueError(f"Invalid integer literal {param_value!r}")
        return int_param_value
    elif param_type.startswith("num") or param_type.startswith("float"):
        float_param_value = float(param_value)
        int_param_value = int(float_param_value)
        return (
            float_param_value
            if (float_param_value - int_param_value) != 0
            else int_param_value
        )
    elif param_type in _bool_types:
        return param_value.lower() == "true"
    else:
        if (
            param_type in _obj_types
            or param_type.startswith("dict")
            or param_type.startswith("list")
        ):
            try:
                return json.loads(param_value, strict=False)
            except json.JSONDecodeError:
                return _safe_literal_eval(param_value)

        # Unknown / unresolved type: try a literal, but never let a malformed
        # value raise (e.g. SyntaxError) — fall back to the raw string.
        return _safe_literal_eval(param_value)


def _safe_literal_eval(param_value: str) -> Any:
    """ast.literal_eval that returns the raw string instead of raising."""
    try:
        return ast.literal_eval(param_value)
    except (ValueError, SyntaxError):
        return param_value


def _parse_xml_function_call(function_call_str: str, tools: Optional[Any]):
    end_index = function_call_str.index(">")
    function_name = function_call_str[:end_index]
    param_config = _get_arguments_config(function_name, tools)
    parameters = function_call_str[end_index + 1 :]
    param_dict = {}
    for match_text in _parameter_regex.findall(parameters):
        idx = match_text.index(">")
        param_name = match_text[:idx]
        param_value = str(match_text[idx + 1 :])
        if param_value.startswith("\n"):
            param_value = param_value[1:]
        if param_value.endswith("\n"):
            param_value = param_value[:-1]

        param_dict[param_name] = _convert_param_value(
            param_value, param_name, param_config
        )
    return dict(name=function_name, arguments=param_dict)


tool_call_start = "<tool_call>"

tool_call_end = "</tool_call>"


def parse_tool_call(
    model_output: str,
    tools: Optional[Any] = None,
):
    matches = _function_regex.findall(model_output)
    if not matches:
        raise ValueError("No function provided.")
    calls = [_parse_xml_function_call(m, tools) for m in matches]
    return calls[0] if len(calls) == 1 else calls
