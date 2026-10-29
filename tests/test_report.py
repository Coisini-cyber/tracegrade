import json
import os
import sys
import unittest

from tracegrade.compile import CompileOptions, compile_trace
from tracegrade.demo import build_traces
from tracegrade.report import failure_breakdown, render_console, render_json, render_markdown
from tracegrade.replay import CaseResult, ReplaySummary
from tracegrade.trace import from_canonical

AGENT = os.path.join(os.path.dirname(__file__), "agents", "probe_agent.py")


def summary_with(pass_count, fail_count):
    results = []
    for index in range(pass_count):
        results.append(CaseResult(case_id="ok-%d" % index, passed=True, steps=1, duration_ms=12.0))
    for index in range(fail_count):
        results.append(
            CaseResult(
                case_id="bad-%d" % index,
                passed=False,
                steps=1,
                duration_ms=10.0,
                failures=[
                    "final answer did not match (normalized): expected 'a', got 'b'",
                    "missing required tool call: search({})",
                ],
                tool_calls=[{"name": "search", "arguments": {}}],
            )
        )
    return ReplaySummary(results)


class MarkdownTest(unittest.TestCase):
    def setUp(self):
        self.text = render_markdown(summary_with(3, 2))

    def test_header_and_totals(self):
        self.assertIn("# Trace replay", self.text)
        self.assertIn("| cases | 5 |", self.text)
        self.assertIn("| passed | 3 |", self.text)
        self.assertIn("| failed | 2 |", self.text)
        self.assertIn("60.0%", self.text)

    def test_case_table(self):
        self.assertIn("| `ok-0` | pass |", self.text)
        self.assertIn("| `bad-0` | FAIL |", self.text)

    def test_failure_breakdown(self):
        self.assertIn("## Failure breakdown", self.text)
        self.assertIn("| final answer did not match | 2 |", self.text)
        self.assertIn("| missing required tool call | 2 |", self.text)

    def test_failure_details(self):
        self.assertIn("### `bad-0`", self.text)
        self.assertIn("Tool calls made:", self.text)

    def test_all_passing_has_no_failure_section(self):
        text = render_markdown(summary_with(2, 0))
        self.assertNotIn("## Failures", text)


class JsonTest(unittest.TestCase):
    def test_structure(self):
        data = json.loads(render_json(summary_with(1, 1)))
        self.assertEqual(data["total"], 2)
        self.assertEqual(data["passed"], 1)
        self.assertEqual(len(data["results"]), 2)
        self.assertIn("failure_breakdown", data)

    def test_failure_kinds(self):
        breakdown = failure_breakdown(summary_with(0, 2))
        kinds = {entry["kind"] for entry in breakdown}
        self.assertIn("final answer did not match", kinds)


class ConsoleTest(unittest.TestCase):
    def test_lines(self):
        text = render_console(summary_with(1, 1))
        self.assertIn("PASS  ok-0", text)
        self.assertIn("FAIL  bad-0", text)
        self.assertIn("1/2 passed", text)


class EndToEndReportTest(unittest.TestCase):
    def test_report_from_a_real_replay(self):
        from tracegrade.replay import replay_case

        case = compile_trace(from_canonical(build_traces()[0]), CompileOptions())
        result = replay_case(
            case,
            '"%s" "%s"' % (sys.executable, AGENT),
            timeout=20,
            env=dict(os.environ, AGENT_MODE="good"),
        )
        text = render_markdown(ReplaySummary([result]))
        self.assertIn("100.0%", text)
        self.assertIn("pass", text)


if __name__ == "__main__":
    unittest.main()
