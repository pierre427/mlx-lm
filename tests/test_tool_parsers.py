# Copyright © 2026 Apple Inc.

import unittest

from mlx_lm.tool_parsers.pythonic import parse_tool_call


class TestPythonicToolParser(unittest.TestCase):
    def test_single_call_returns_dict(self):
        out = parse_tool_call('[get_weather(city="SF", units=2)]')
        self.assertEqual(
            out, {"name": "get_weather", "arguments": {"city": "SF", "units": 2}}
        )

    def test_multiple_calls_all_drained(self):
        # A single non-greedy search over the block used to merge the args of
        # call 1..n and silently drop calls 2..n.
        out = parse_tool_call('[get_weather(city="SF"), get_time(tz="PST")]')
        self.assertEqual(
            out,
            [
                {"name": "get_weather", "arguments": {"city": "SF"}},
                {"name": "get_time", "arguments": {"tz": "PST"}},
            ],
        )

    def test_paren_inside_quoted_arg(self):
        out = parse_tool_call('[run(cmd="echo )")]')
        self.assertEqual(out, {"name": "run", "arguments": {"cmd": "echo )"}})

    def test_no_function_raises(self):
        with self.assertRaises(ValueError):
            parse_tool_call("no calls here")


if __name__ == "__main__":
    unittest.main()
