import os
import sys
import unittest

from tracegrade.compile import CompileOptions, compile_trace
from tracegrade.demo import build_traces
from tracegrade.replay import (
    AgentError,
    SubprocessAgent,
    replay_case,
    replay_cases,
)
from tracegrade.trace import from_canonical

AGENT = os.path.join(os.path.dirname(__file__), "agents", "probe_agent.py")


def agent_command():
    return '"%s" "%s"' % (sys.executable, AGENT)


def agent_env(mode):
    env = dict(os.environ)
    env["AGENT_MODE"] = mode
    return env


def demo_case(index=0, **options):
    trace = from_canonical(build_traces()[index])
    return compile_trace(trace, CompileOptions(**options))


class HappyPathTest(unittest.TestCase):
    def test_replay_passes(self):
        case = demo_case()
        result = replay_case(case, agent_command(), timeout=20, env=agent_env("good"))
        self.assertTrue(result.passed, result.failures)
        self.assertEqual(result.steps, 1)
        self.assertEqual(result.tool_calls[0]["name"], "search")
        self.assertIn("According to the docs", result.final)

    def test_all_demo_cases_pass(self):
        cases = [demo_case(index) for index in range(6)]
        summary = replay_cases(cases, agent_command(), timeout=20, env=agent_env("good"))
        self.assertEqual(summary.total, 6)
        self.assertEqual(summary.passed, 6)
        self.assertAlmostEqual(summary.pass_rate, 1.0)

    def test_each_case_gets_a_fresh_process(self):
        # If state leaked between cases, the second replay would see a consumed
        # fixture and report a miss.
        cases = [demo_case(0), demo_case(0)]
        summary = replay_cases(cases, agent_command(), timeout=20, env=agent_env("good"))
        self.assertEqual(summary.passed, 2)


class FailureTest(unittest.TestCase):
    def test_wrong_final_fails(self):
        result = replay_case(demo_case(), agent_command(), timeout=20, env=agent_env("wrong_final"))
        self.assertFalse(result.passed)
        self.assertTrue(any("final answer" in failure for failure in result.failures))

    def test_offscript_tool_call_fails(self):
        result = replay_case(demo_case(), agent_command(), timeout=20, env=agent_env("offscript"))
        self.assertFalse(result.passed)
        self.assertGreater(result.fixture_misses, 0)
        self.assertTrue(any("no recorded fixture" in failure for failure in result.failures))
        self.assertTrue(any("missing required tool call" in failure for failure in result.failures))

    def test_agent_error_is_recorded(self):
        result = replay_case(demo_case(), agent_command(), timeout=20, env=agent_env("error"))
        self.assertFalse(result.passed)
        self.assertIn("upstream model unavailable", result.error)

    def test_agent_crash_is_recorded(self):
        result = replay_case(demo_case(), agent_command(), timeout=20, env=agent_env("crash"))
        self.assertFalse(result.passed)
        self.assertIsNotNone(result.error)

    def test_non_json_output_is_recorded(self):
        result = replay_case(demo_case(), agent_command(), timeout=20, env=agent_env("garbage"))
        self.assertFalse(result.passed)
        self.assertIn("non-JSON", result.error)

    def test_unexpected_message_type(self):
        result = replay_case(demo_case(), agent_command(), timeout=20, env=agent_env("weird"))
        self.assertFalse(result.passed)
        self.assertIn("unexpected message type", result.error)

    def test_timeout(self):
        case = demo_case()
        result = replay_case(case, agent_command(), timeout=1, env=agent_env("silent"))
        self.assertFalse(result.passed)
        self.assertIn("did not respond", result.error)

    def test_step_limit(self):
        case = demo_case(max_steps=2)
        result = replay_case(case, agent_command(), timeout=20, env=agent_env("looping"))
        self.assertFalse(result.passed)
        self.assertIn("step limit", result.error)

    def test_fail_fast_stops_early(self):
        cases = [demo_case(0), demo_case(1), demo_case(2)]
        summary = replay_cases(
            cases, agent_command(), timeout=20, fail_fast=True, env=agent_env("wrong_final")
        )
        self.assertEqual(summary.total, 1)


class MatcherTest(unittest.TestCase):
    def test_exact_final_mode_is_strict(self):
        case = demo_case(final_mode="exact")
        result = replay_case(case, agent_command(), timeout=20, env=agent_env("good"))
        self.assertTrue(result.passed, result.failures)

    def test_contains_final_mode(self):
        case = demo_case(final_mode="contains")
        case.expect.final_value = "According to the docs"
        result = replay_case(case, agent_command(), timeout=20, env=agent_env("good"))
        self.assertTrue(result.passed, result.failures)

    def test_none_final_mode_ignores_the_answer(self):
        case = demo_case(final_mode="none")
        result = replay_case(case, agent_command(), timeout=20, env=agent_env("wrong_final"))
        self.assertTrue(result.passed, result.failures)

    def test_forbidden_tool(self):
        case = demo_case(tool_mode="required")
        case.expect.forbidden_calls = ["search"]
        result = replay_case(case, agent_command(), timeout=20, env=agent_env("good"))
        self.assertFalse(result.passed)
        self.assertTrue(any("forbidden" in failure for failure in result.failures))

    def test_exact_tool_mode_rejects_extra_calls(self):
        case = demo_case(tool_mode="exact")
        case.expect.required_calls = []
        result = replay_case(case, agent_command(), timeout=20, env=agent_env("good"))
        self.assertFalse(result.passed)
        self.assertTrue(any("tool calls differ" in failure for failure in result.failures))

    def test_require_all_fixtures(self):
        data = {
            "trace_id": "t",
            "input": [{"role": "user", "content": "hello there friend"}],
            "steps": [
                {
                    "type": "tool_call",
                    "id": "c1",
                    "name": "search",
                    "arguments": {"query": "hello there friend", "limit": 5},
                },
                {"type": "tool_result", "id": "c1", "name": "search", "content": "x"},
            ],
            "final": "According to the docs: x",
        }
        case = compile_trace(
            from_canonical(data), CompileOptions(require_all_fixtures_used=True)
        )
        result = replay_case(case, agent_command(), timeout=20, env=agent_env("good"))
        # The recorded call asked for a limit the agent never sends, so the
        # fixture is never consumed.
        self.assertFalse(result.passed)
        self.assertTrue(any("not replayed" in failure for failure in result.failures))


class ProtocolTest(unittest.TestCase):
    def test_subprocess_agent_round_trip(self):
        with SubprocessAgent(agent_command(), timeout=20, env=agent_env("good")) as agent:
            response = agent.send(
                {
                    "type": "start",
                    "case_id": "x",
                    "input": [{"role": "user", "content": "hello there"}],
                    "tools": ["search"],
                    "max_steps": 3,
                }
            )
            self.assertEqual(response["type"], "tool_call")
            self.assertEqual(response["arguments"], {"query": "hello there"})
            final = agent.send(
                {"type": "tool_result", "content": "answer", "is_error": False}
            )
            self.assertEqual(final["type"], "final")
            self.assertIn("answer", final["content"])

    def test_missing_command_fails(self):
        agent = SubprocessAgent("definitely-not-a-real-command-xyz", timeout=5)
        agent.start()
        try:
            with self.assertRaises(AgentError):
                agent.send({"type": "start"})
        finally:
            agent.close()


if __name__ == "__main__":
    unittest.main()
