import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

from tracegrade.cli import main


def run(argv):
    out = io.StringIO()
    err = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


class EndToEndTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        code, out, _ = run(["demo", "-o", self.directory.name])
        self.assertEqual(code, 0)
        self.traces = os.path.join(self.directory.name, "traces.jsonl")
        self.agent = os.path.join(self.directory.name, "mock_agent.py")
        self.cases = os.path.join(self.directory.name, "cases.jsonl")
        self.command = '"%s" "%s"' % (sys.executable, self.agent)

    def tearDown(self):
        self.directory.cleanup()

    def test_demo_writes_both_files(self):
        self.assertTrue(os.path.exists(self.traces))
        self.assertTrue(os.path.exists(self.agent))

    def test_compile_then_replay_passes(self):
        code, _, err = run(["compile", self.traces, "-o", self.cases])
        self.assertEqual(code, 0)
        self.assertIn("compiled 6 case(s)", err)

        code, out, _ = run(["replay", self.cases, "--agent", self.command])
        self.assertEqual(code, 0)
        self.assertIn("6/6 passed", out)

    def test_replay_writes_markdown_and_json(self):
        run(["compile", self.traces, "-o", self.cases])
        report = os.path.join(self.directory.name, "report.md")
        data = os.path.join(self.directory.name, "report.json")
        code, _, _ = run(
            [
                "replay",
                self.cases,
                "--agent",
                self.command,
                "-o",
                report,
                "--json",
                data,
                "--quiet",
            ]
        )
        self.assertEqual(code, 0)
        with open(report, encoding="utf-8") as handle:
            self.assertIn("# Trace replay", handle.read())
        with open(data, encoding="utf-8") as handle:
            payload = json.load(handle)
        self.assertEqual(payload["passed"], 6)

    def test_replay_failure_exits_nonzero(self):
        run(["compile", self.traces, "-o", self.cases])
        broken = os.path.join(self.directory.name, "broken.jsonl")
        with open(self.cases, encoding="utf-8") as handle:
            lines = [json.loads(line) for line in handle if line.strip()]
        for case in lines:
            case["expect"]["final"]["value"] = "a different answer"
        with open(broken, "w", encoding="utf-8") as handle:
            for case in lines:
                handle.write(json.dumps(case) + "\n")
        code, out, _ = run(["replay", broken, "--agent", self.command])
        self.assertEqual(code, 1)
        self.assertIn("FAIL", out)

    def test_min_pass_rate_gate(self):
        run(["compile", self.traces, "-o", self.cases])
        code, _, err = run(
            [
                "replay",
                self.cases,
                "--agent",
                self.command,
                "--min-pass-rate",
                "1.0",
                "--quiet",
            ]
        )
        self.assertEqual(code, 0)

        broken = os.path.join(self.directory.name, "broken.jsonl")
        with open(self.cases, encoding="utf-8") as handle:
            lines = [json.loads(line) for line in handle if line.strip()]
        lines[0]["expect"]["final"]["value"] = "nope"
        with open(broken, "w", encoding="utf-8") as handle:
            for case in lines:
                handle.write(json.dumps(case) + "\n")
        code, _, err = run(
            ["replay", broken, "--agent", self.command, "--min-pass-rate", "1.0", "--quiet"]
        )
        self.assertEqual(code, 3)
        self.assertIn("gate failed", err)

    def test_single_case_selection(self):
        run(["compile", self.traces, "-o", self.cases])
        code, out, _ = run(
            ["replay", self.cases, "--agent", self.command, "--case", "demo-01"]
        )
        self.assertEqual(code, 0)
        self.assertIn("1/1 passed", out)

    def test_unknown_case(self):
        run(["compile", self.traces, "-o", self.cases])
        code, _, err = run(
            ["replay", self.cases, "--agent", self.command, "--case", "nope"]
        )
        self.assertEqual(code, 1)
        self.assertIn("no matching cases", err)

    def test_lint_clean(self):
        run(["compile", self.traces, "-o", self.cases])
        code, out, _ = run(["lint", self.cases])
        self.assertEqual(code, 0)
        self.assertIn("0 problem(s)", out)

    def test_lint_flags_an_empty_case(self):
        empty = os.path.join(self.directory.name, "empty.jsonl")
        with open(empty, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "case_id": "x",
                        "input": [{"role": "user", "content": "hi"}],
                        "expect": {
                            "final": {"type": "none", "value": ""},
                            "tool_calls": {"mode": "none", "required": [], "max_steps": 3},
                        },
                    }
                )
                + "\n"
            )
        code, out, _ = run(["lint", empty])
        self.assertEqual(code, 1)
        self.assertIn("never fail", out)

    def test_show(self):
        run(["compile", self.traces, "-o", self.cases])
        code, out, _ = run(["show", self.cases, "demo-01"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["case_id"], "demo-01")

    def test_show_unknown(self):
        run(["compile", self.traces, "-o", self.cases])
        code, _, err = run(["show", self.cases, "nope"])
        self.assertEqual(code, 1)
        self.assertIn("no case named", err)


class CompileOptionsTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        run(["demo", "-o", self.directory.name])
        self.traces = os.path.join(self.directory.name, "traces.jsonl")
        self.cases = os.path.join(self.directory.name, "cases.jsonl")

    def tearDown(self):
        self.directory.cleanup()

    def test_tool_mode_none(self):
        code, _, _ = run(["compile", self.traces, "-o", self.cases, "--tool-mode", "none"])
        self.assertEqual(code, 0)
        with open(self.cases, encoding="utf-8") as handle:
            case = json.loads(handle.readline())
        self.assertEqual(case["expect"]["tool_calls"]["mode"], "none")

    def test_bad_final_mode_rejected_by_argparse(self):
        with self.assertRaises(SystemExit):
            run(["compile", self.traces, "--final-mode", "vibes"])


class ErrorTest(unittest.TestCase):
    def test_missing_traces(self):
        code, _, err = run(["compile", "does-not-exist.jsonl"])
        self.assertEqual(code, 1)
        self.assertIn("error:", err)

    def test_missing_cases(self):
        code, _, err = run(["replay", "does-not-exist.jsonl", "--agent", "true"])
        self.assertEqual(code, 1)
        self.assertIn("error:", err)

    def test_no_command(self):
        code, out, _ = run([])
        self.assertEqual(code, 2)
        self.assertIn("usage:", out)


if __name__ == "__main__":
    unittest.main()
