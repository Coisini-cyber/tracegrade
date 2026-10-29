import unittest

from tracegrade.compile import (
    CompileError,
    CompileOptions,
    compile_trace,
    compile_traces,
    from_jsonl,
    normalize_text,
    to_jsonl,
    _subset_match,
)
from tracegrade.trace import from_canonical


def trace(**overrides):
    data = {
        "trace_id": "t1",
        "input": [{"role": "user", "content": "find the refund policy"}],
        "steps": [
            {"type": "tool_call", "id": "c1", "name": "search", "arguments": {"query": "refund"}},
            {"type": "tool_result", "id": "c1", "name": "search", "content": "30 days"},
        ],
        "final": "Refunds take 30 days.",
        "meta": {"model": "m"},
    }
    data.update(overrides)
    return from_canonical(data)


class NormalizeTest(unittest.TestCase):
    def test_collapses_whitespace(self):
        self.assertEqual(normalize_text("  a \n b\t c "), "a b c")

    def test_empty(self):
        self.assertEqual(normalize_text(""), "")


class SubsetTest(unittest.TestCase):
    def test_dict_subset(self):
        self.assertTrue(_subset_match({"a": 1}, {"a": 1, "b": 2}))
        self.assertFalse(_subset_match({"a": 1}, {"a": 2}))

    def test_nested(self):
        self.assertTrue(_subset_match({"a": {"b": 1}}, {"a": {"b": 1, "c": 2}}))
        self.assertFalse(_subset_match({"a": {"b": 1}}, {"a": {"c": 2}}))

    def test_lists_are_ordered_and_exact(self):
        self.assertTrue(_subset_match([1, 2], [1, 2]))
        self.assertFalse(_subset_match([1, 2], [2, 1]))
        self.assertFalse(_subset_match([1], [1, 2]))

    def test_scalars(self):
        self.assertTrue(_subset_match("x", "x"))
        self.assertFalse(_subset_match("x", "y"))

    def test_type_mismatch(self):
        self.assertFalse(_subset_match({"a": 1}, ["a"]))


class CompileTest(unittest.TestCase):
    def test_case_shape(self):
        case = compile_trace(trace())
        self.assertEqual(case.case_id, "t1")
        self.assertEqual(case.source_trace, "t1")
        self.assertEqual(case.tools, ["search"])
        self.assertEqual(len(case.fixtures), 1)
        self.assertEqual(case.fixtures[0].match, {"query": "refund"})
        self.assertEqual(case.fixtures[0].result, "30 days")
        self.assertEqual(case.expect.required_calls[0]["name"], "search")
        self.assertEqual(case.expect.final_value, "Refunds take 30 days.")

    def test_max_steps_defaults_to_recorded_plus_slack(self):
        case = compile_trace(trace(), CompileOptions(extra_steps=3))
        self.assertEqual(case.expect.max_steps, 4)

    def test_explicit_max_steps(self):
        case = compile_trace(trace(), CompileOptions(max_steps=1))
        self.assertEqual(case.expect.max_steps, 1)

    def test_tool_mode_none_drops_requirements(self):
        case = compile_trace(trace(), CompileOptions(tool_mode="none"))
        self.assertEqual(case.expect.required_calls, [])
        self.assertEqual(case.expect.tool_mode, "none")

    def test_final_mode_is_recorded(self):
        case = compile_trace(trace(), CompileOptions(final_mode="contains"))
        self.assertEqual(case.expect.final_type, "contains")

    def test_bad_options_rejected(self):
        with self.assertRaises(CompileError):
            compile_trace(trace(), CompileOptions(final_mode="vibes"))
        with self.assertRaises(CompileError):
            compile_trace(trace(), CompileOptions(tool_mode="sometimes"))

    def test_trace_without_input_is_rejected(self):
        with self.assertRaises(CompileError):
            compile_trace(trace(input=[]))

    def test_case_prefix(self):
        case = compile_trace(trace(), CompileOptions(case_prefix="suite-a/"))
        self.assertEqual(case.case_id, "suite-a/t1")

    def test_duplicate_ids_get_suffixed(self):
        cases = compile_traces([trace(), trace()])
        self.assertEqual([case.case_id for case in cases], ["t1", "t1-2"])

    def test_meta_is_carried_over(self):
        case = compile_trace(trace())
        self.assertEqual(case.meta["model"], "m")

    def test_missing_tool_result_still_produces_a_fixture(self):
        data = {
            "trace_id": "t2",
            "input": [{"role": "user", "content": "q"}],
            "steps": [{"type": "tool_call", "id": "c1", "name": "search", "arguments": {}}],
            "final": "done",
        }
        case = compile_trace(from_canonical(data))
        self.assertEqual(case.fixtures[0].result, "")


class SerialisationTest(unittest.TestCase):
    def test_round_trip(self):
        case = compile_trace(trace())
        text = to_jsonl([case])
        reloaded = from_jsonl(text)[0]
        self.assertEqual(reloaded.case_id, case.case_id)
        self.assertEqual(reloaded.expect.final_value, case.expect.final_value)
        self.assertEqual(reloaded.fixtures[0].match, {"query": "refund"})
        self.assertEqual(reloaded.input[0].content, case.input[0].content)

    def test_json_array_input(self):
        case = compile_trace(trace())
        import json

        text = json.dumps([case.as_dict()])
        self.assertEqual(from_jsonl(text)[0].case_id, "t1")

    def test_bad_json(self):
        with self.assertRaises(CompileError):
            from_jsonl("{oops}")

    def test_missing_case_id(self):
        with self.assertRaises(CompileError):
            from_jsonl('{"input": []}')

    def test_empty(self):
        self.assertEqual(from_jsonl(""), [])


if __name__ == "__main__":
    unittest.main()
