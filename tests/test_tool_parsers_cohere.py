# Copyright © 2026

"""Regression tests for the North (cohere2_moe) action-block tool parser and
the shared ToolCallFormatter hardening that contains its failure modes.

Covers:
  * model-emitted `tool_call_id` threaded through as the OpenAI id (stable
    across repeated parses of the same text)
  * parallel calls in one action block, and multiple action blocks, all
    parsed -- none silently dropped
  * the one bounded repair pass (code fence, trailing comma)
  * non-object `parameters` rejected without crashing, valid siblings kept
  * total parse failure / truncated / empty output raises ValueError
  * ToolCallFormatter: no mutation of parser-owned dicts, a malformed call
    doesn't discard valid siblings, and `tool_parser=None` never crashes
"""

import unittest

from mlx_lm.server import ToolCallFormatter
from mlx_lm.tool_parsers import cohere


class TestCohereNorthParser(unittest.TestCase):
    def test_single_call_preserves_tool_call_id(self):
        text = (
            "<|START_ACTION|>["
            '{"tool_call_id": "0", "tool_name": "get_weather", '
            '"parameters": {"city": "SF"}}'
            "]<|END_ACTION|>"
        )
        call = cohere.parse_tool_call(text)
        self.assertEqual(
            call,
            {"id": "0", "name": "get_weather", "arguments": {"city": "SF"}},
        )

    def test_stable_ids_across_repeated_parses(self):
        text = (
            "<|START_ACTION|>["
            '{"tool_call_id": "42", "tool_name": "ping", "parameters": {}}'
            "]<|END_ACTION|>"
        )
        first = cohere.parse_tool_call(text)
        second = cohere.parse_tool_call(text)
        self.assertEqual(first["id"], "42")
        self.assertEqual(first, second)

    def test_parallel_calls_in_one_block_all_parsed(self):
        text = (
            "<|START_ACTION|>["
            '{"tool_call_id": "0", "tool_name": "get_weather", "parameters": {"city": "SF"}},'
            '{"tool_call_id": "1", "tool_name": "get_time", "parameters": {"tz": "PST"}}'
            "]<|END_ACTION|>"
        )
        calls = cohere.parse_tool_call(text)
        self.assertEqual(
            calls,
            [
                {"id": "0", "name": "get_weather", "arguments": {"city": "SF"}},
                {"id": "1", "name": "get_time", "arguments": {"tz": "PST"}},
            ],
        )

    def test_multiple_action_blocks_all_parsed(self):
        # A single `.search()` over START_ACTION/END_ACTION used to keep only
        # the first block; both must survive here.
        text = (
            "<|START_ACTION|>["
            '{"tool_call_id": "0", "tool_name": "search", "parameters": {"q": "weather"}}'
            "]<|END_ACTION|>"
            "<|START_ACTION|>["
            '{"tool_call_id": "1", "tool_name": "read_file", "parameters": {"path": "/tmp/x"}}'
            "]<|END_ACTION|>"
        )
        calls = cohere.parse_tool_call(text)
        self.assertEqual(
            calls,
            [
                {"id": "0", "name": "search", "arguments": {"q": "weather"}},
                {"id": "1", "name": "read_file", "arguments": {"path": "/tmp/x"}},
            ],
        )

    def test_repairable_trailing_comma(self):
        text = (
            "<|START_ACTION|>["
            '{"tool_call_id": "0", "tool_name": "get_weather", "parameters": {"city": "SF",}},'
            "]<|END_ACTION|>"
        )
        call = cohere.parse_tool_call(text)
        self.assertEqual(
            call,
            {"id": "0", "name": "get_weather", "arguments": {"city": "SF"}},
        )

    def test_trailing_comma_repair_does_not_corrupt_string_values(self):
        # The repair pass is triggered by the real trailing comma after the
        # object; it must not also treat ", }" *inside* the string argument
        # as a trailing comma to strip.
        text = (
            "<|START_ACTION|>["
            '{"tool_name": "say", "parameters": {"msg": "hello, }bye"}},'
            "]<|END_ACTION|>"
        )
        call = cohere.parse_tool_call(text)
        self.assertEqual(call, {"name": "say", "arguments": {"msg": "hello, }bye"}})

    def test_repairable_code_fence(self):
        text = (
            "<|START_ACTION|>```json\n["
            '{"tool_call_id": "0", "tool_name": "get_weather", "parameters": {"city": "SF"}}'
            "]\n```<|END_ACTION|>"
        )
        call = cohere.parse_tool_call(text)
        self.assertEqual(
            call,
            {"id": "0", "name": "get_weather", "arguments": {"city": "SF"}},
        )

    def test_non_object_arguments_rejected(self):
        text = (
            "<|START_ACTION|>["
            '{"tool_call_id": "0", "tool_name": "bad", "parameters": ["not", "an", "object"]}'
            "]<|END_ACTION|>"
        )
        with self.assertRaises(ValueError):
            cohere.parse_tool_call(text)

    def test_mixed_valid_invalid_siblings_keeps_valid(self):
        text = (
            "<|START_ACTION|>["
            '{"tool_call_id": "0", "tool_name": "bad", "parameters": "not-an-object"},'
            '{"tool_call_id": "1", "tool_name": "good", "parameters": {"x": 1}}'
            "]<|END_ACTION|>"
        )
        call = cohere.parse_tool_call(text)
        self.assertEqual(call, {"id": "1", "name": "good", "arguments": {"x": 1}})

    def test_total_parse_failure_raises_value_error(self):
        text = "<|START_ACTION|>not json at all<|END_ACTION|>"
        with self.assertRaises(ValueError):
            cohere.parse_tool_call(text)

    def test_truncated_block_raises_value_error(self):
        # No closing marker -- mid-generation truncation.
        text = '<|START_ACTION|>[{"tool_call_id": "0", "tool_name": "get_wea'
        with self.assertRaises(ValueError):
            cohere.parse_tool_call(text)

    def test_empty_output_raises_value_error(self):
        with self.assertRaises(ValueError):
            cohere.parse_tool_call("")

    def test_missing_id_omits_id_key(self):
        text = '<|START_ACTION|>[{"tool_name": "ping", "parameters": {}}]<|END_ACTION|>'
        call = cohere.parse_tool_call(text)
        self.assertNotIn("id", call)
        self.assertEqual(call, {"name": "ping", "arguments": {}})


class TestToolCallFormatterContainment(unittest.TestCase):
    def test_parser_none_returns_empty_without_crash(self):
        formatter = ToolCallFormatter(None, tools=None, streaming=False)
        self.assertEqual(formatter(["<|START_ACTION|>[]<|END_ACTION|>"]), [])

    def test_does_not_mutate_parser_owned_dict(self):
        owned = {"id": "7", "name": "get_weather", "arguments": {"city": "SF"}}

        def fake_parser(text, tools):
            return owned

        formatter = ToolCallFormatter(fake_parser, tools=None, streaming=False)
        out = formatter(["irrelevant"])
        self.assertEqual(
            out,
            [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "SF"}',
                    },
                    "type": "function",
                    "id": "7",
                }
            ],
        )
        # The dict returned by the parser must be untouched.
        self.assertEqual(
            owned, {"id": "7", "name": "get_weather", "arguments": {"city": "SF"}}
        )

    def test_malformed_call_does_not_discard_valid_siblings(self):
        def fake_parser(text, tools):
            return [
                {"name": "good_one", "arguments": {"x": 1}},
                {"name": "missing_arguments"},  # KeyError in _format
                {"name": "good_two", "arguments": {"y": 2}},
            ]

        formatter = ToolCallFormatter(fake_parser, tools=None, streaming=False)
        out = formatter(["irrelevant"])
        names = [tc["function"]["name"] for tc in out]
        self.assertEqual(names, ["good_one", "good_two"])

    def test_model_emitted_id_preserved_as_openai_id(self):
        def fake_parser(text, tools):
            return {"id": "call_abc", "name": "ping", "arguments": {}}

        formatter = ToolCallFormatter(fake_parser, tools=None, streaming=False)
        out = formatter(["irrelevant"])
        self.assertEqual(out[0]["id"], "call_abc")


if __name__ == "__main__":
    unittest.main()
