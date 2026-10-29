import json
import os
import tempfile
import unittest

from tracegrade.trace import (
    TraceError,
    from_canonical,
    from_openai,
    load_traces,
    load_traces_from_string,
    to_jsonl,
)


class CanonicalTest(unittest.TestCase):
    def test_minimal_trace(self):
        trace = from_canonical(
            {
                "trace_id": "t1",
                "input": [{"role": "user", "content": "hello"}],
                "final": "hi",
            }
        )
        self.assertEqual(trace.trace_id, "t1")
        self.assertEqual(trace.input[0].content, "hello")
        self.assertEqual(trace.final, "hi")
        self.assertEqual(trace.steps, [])

    def test_tool_call_and_result(self):
        trace = from_canonical(
            {
                "trace_id": "t1",
                "input": [{"role": "user", "content": "q"}],
                "steps": [
                    {"type": "tool_call", "id": "c1", "name": "search", "arguments": {"q": "x"}},
                    {"type": "tool_result", "id": "c1", "name": "search", "content": "found"},
                ],
                "final": "done",
            }
        )
        self.assertEqual(len(trace.tool_calls), 1)
        self.assertEqual(trace.tool_names, ["search"])
        self.assertEqual(trace.result_for(trace.tool_calls[0]).content, "found")

    def test_arguments_may_be_a_json_string(self):
        trace = from_canonical(
            {
                "trace_id": "t1",
                "input": [{"role": "user", "content": "q"}],
                "steps": [{"type": "tool_call", "name": "s", "arguments": '{"a": 1}'}],
            }
        )
        self.assertEqual(trace.tool_calls[0].arguments, {"a": 1})

    def test_final_falls_back_to_last_assistant_step(self):
        trace = from_canonical(
            {
                "trace_id": "t1",
                "input": [{"role": "user", "content": "q"}],
                "steps": [
                    {"type": "assistant", "content": "first"},
                    {"type": "assistant", "content": "second"},
                ],
            }
        )
        self.assertEqual(trace.final, "second")

    def test_content_parts_are_flattened(self):
        trace = from_canonical(
            {
                "trace_id": "t1",
                "input": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "one"}, {"type": "text", "text": "two"}],
                    }
                ],
            }
        )
        self.assertEqual(trace.input[0].content, "one\ntwo")

    def test_missing_id(self):
        with self.assertRaises(TraceError):
            from_canonical({"input": []})

    def test_unknown_step_type(self):
        with self.assertRaises(TraceError):
            from_canonical(
                {"trace_id": "t", "input": [], "steps": [{"type": "telepathy"}]}
            )

    def test_bad_arguments(self):
        with self.assertRaises(TraceError):
            from_canonical(
                {
                    "trace_id": "t",
                    "input": [],
                    "steps": [{"type": "tool_call", "name": "s", "arguments": "{oops}"}],
                }
            )


class OpenAITest(unittest.TestCase):
    def conversation(self):
        return {
            "id": "conv-1",
            "model": "gpt-test",
            "messages": [
                {"role": "system", "content": "be helpful"},
                {"role": "user", "content": "find the refund policy"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "search", "arguments": '{"query": "refund"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "30 days"},
                {"role": "assistant", "content": "Refunds take 30 days."},
            ],
        }

    def test_conversion(self):
        trace = from_openai(self.conversation())
        self.assertEqual(trace.trace_id, "conv-1")
        self.assertEqual([message.role for message in trace.input], ["system", "user"])
        self.assertEqual(trace.tool_names, ["search"])
        self.assertEqual(trace.tool_calls[0].arguments, {"query": "refund"})
        self.assertEqual(trace.tool_results[0].content, "30 days")
        self.assertEqual(trace.final, "Refunds take 30 days.")
        self.assertEqual(trace.meta["model"], "gpt-test")

    def test_missing_messages(self):
        with self.assertRaises(TraceError):
            from_openai({"id": "x"})

    def test_generated_id(self):
        data = self.conversation()
        del data["id"]
        trace = from_openai(data, line=7)
        self.assertEqual(trace.trace_id, "trace-7")


class LoaderTest(unittest.TestCase):
    def test_auto_detects_each_shape(self):
        text = (
            '{"id": "o1", "messages": [{"role": "user", "content": "hi"}]}\n'
            '{"trace_id": "c1", "input": [{"role": "user", "content": "hi"}]}\n'
        )
        traces = load_traces_from_string(text)
        self.assertEqual([trace.trace_id for trace in traces], ["o1", "c1"])

    def test_json_array(self):
        text = '[{"trace_id": "c1", "input": [{"role": "user", "content": "hi"}]}]'
        self.assertEqual(len(load_traces_from_string(text)), 1)

    def test_forced_format(self):
        text = '{"trace_id": "c1", "input": []}'
        self.assertEqual(len(load_traces_from_string(text, fmt="canonical")), 1)

    def test_bad_format(self):
        with self.assertRaises(TraceError):
            load_traces_from_string('{"trace_id": "c1"}', fmt="langchain")

    def test_missing_file(self):
        with self.assertRaises(TraceError):
            load_traces(os.path.join(tempfile.gettempdir(), "tracegrade-missing.jsonl"))

    def test_round_trip(self):
        trace = from_openai(OpenAITest().conversation())
        text = to_jsonl([trace])
        reloaded = load_traces_from_string(text, fmt="canonical")[0]
        self.assertEqual(reloaded.trace_id, trace.trace_id)
        self.assertEqual(reloaded.final, trace.final)
        self.assertEqual(len(reloaded.steps), len(trace.steps))

    def test_empty_file(self):
        with self.assertRaises(TraceError):
            load_traces_from_string("")


if __name__ == "__main__":
    unittest.main()
