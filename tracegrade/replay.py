"""Replay compiled cases against an agent and evaluate the assertions.

The agent is any process that speaks the line-delimited JSON protocol in
``docs/protocol.md``.  A fresh process is started per case, so one crashed
case cannot poison the next.

Tool calls are answered from the recorded fixtures instead of the real tools.
That is what makes the replay deterministic and offline: the agent sees the
same tool output it saw in production, and the only thing that can change is
the agent's own decisions.
"""

from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .compile import Case, Fixture, _subset_match, normalize_text


class AgentError(RuntimeError):
    """The agent process failed, timed out, or broke the protocol."""


@dataclass
class CaseResult:
    case_id: str
    passed: bool
    failures: List[str] = field(default_factory=list)
    steps: int = 0
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    final: str = ""
    fixture_misses: int = 0
    error: Optional[str] = None
    duration_ms: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "passed": self.passed,
            "failures": self.failures,
            "steps": self.steps,
            "tool_calls": self.tool_calls,
            "final": self.final,
            "fixture_misses": self.fixture_misses,
            "error": self.error,
            "duration_ms": round(self.duration_ms, 1),
        }


class _LineReader:
    """Read lines from a pipe with a timeout, on any platform."""

    def __init__(self, stream) -> None:
        self._queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self._stream = stream
        thread = threading.Thread(target=self._pump, daemon=True)
        thread.start()

    def _pump(self) -> None:
        try:
            for line in self._stream:
                self._queue.put(line)
        except Exception:
            pass
        finally:
            self._queue.put(None)

    def read(self, timeout: float) -> str:
        try:
            line = self._queue.get(timeout=timeout)
        except queue.Empty:
            raise AgentError("agent did not respond within %.1fs" % timeout)
        if line is None:
            raise AgentError("agent closed its output stream")
        return line


class SubprocessAgent:
    """An agent subprocess speaking the replay protocol."""

    def __init__(self, command: str, timeout: float = 30.0, env: Optional[Dict[str, str]] = None):
        self.command = command
        self.timeout = timeout
        self.env = env
        self.process: Optional[subprocess.Popen] = None
        self._stdout: Optional[_LineReader] = None
        self._stderr: List[str] = []

    def __enter__(self) -> "SubprocessAgent":
        self.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def start(self) -> None:
        try:
            self.process = subprocess.Popen(
                self.command,
                shell=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=self.env,
            )
        except OSError as error:
            raise AgentError("could not start agent: %s" % error)
        assert self.process.stdout is not None
        assert self.process.stderr is not None
        self._stdout = _LineReader(self.process.stdout)
        threading.Thread(target=self._pump_stderr, daemon=True).start()

    def _pump_stderr(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        try:
            for line in self.process.stderr:
                line = line.rstrip()
                if line:
                    self._stderr.append(line)
                    if len(self._stderr) > 100:
                        del self._stderr[:50]
        except Exception:
            pass

    def send(self, payload: Dict[str, Any], timeout: Optional[float] = None) -> Dict[str, Any]:
        if self.process is None or self.process.stdin is None or self._stdout is None:
            raise AgentError("agent is not running")
        try:
            self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise AgentError("agent closed its input: %s" % error)
        line = self._stdout.read(timeout or self.timeout)
        try:
            message = json.loads(line)
        except ValueError:
            raise AgentError("agent wrote non-JSON output: %r" % line[:200])
        if not isinstance(message, dict):
            raise AgentError("agent output must be a JSON object")
        return message

    def close(self) -> None:
        if self.process is None:
            return
        try:
            if self.process.stdin:
                self.process.stdin.close()
        except Exception:
            pass
        try:
            self.process.terminate()
            self.process.wait(timeout=3)
        except Exception:
            try:
                self.process.kill()
            except Exception:
                pass
        for stream in (self.process.stdout, self.process.stderr):
            try:
                if stream:
                    stream.close()
            except Exception:
                pass
        self.process = None

    @property
    def stderr_tail(self) -> str:
        return "\n".join(self._stderr[-10:])


def reset_fixtures(case: Case) -> None:
    for fixture in case.fixtures:
        fixture.used = False


def _find_fixture(case: Case, name: str, arguments: Dict[str, Any]) -> Optional[Fixture]:
    fallback: Optional[Fixture] = None
    for fixture in case.fixtures:
        if fixture.name != name:
            continue
        if not _subset_match(fixture.match, arguments):
            continue
        if not fixture.used:
            fixture.used = True
            return fixture
        if fallback is None:
            fallback = fixture
    return fallback


def _final_matches(expect, final: str) -> bool:
    if expect.final_type == "none":
        return True
    if expect.final_type == "exact":
        return final == expect.final_value
    if expect.final_type == "contains":
        return normalize_text(expect.final_value) in normalize_text(final)
    return normalize_text(final) == normalize_text(expect.final_value)


def _call_matches(required: Dict[str, Any], observed: Dict[str, Any]) -> bool:
    if required.get("name") != observed.get("name"):
        return False
    subset = required.get("arguments_subset") or {}
    return _subset_match(subset, observed.get("arguments") or {})


def evaluate(case: Case, result: CaseResult) -> CaseResult:
    """Fill in failures for a finished replay."""
    expect = case.expect
    failures: List[str] = []

    if result.error:
        failures.append("agent error: %s" % result.error)
    else:
        if not _final_matches(expect, result.final):
            failures.append(
                "final answer did not match (%s): expected %r, got %r"
                % (
                    expect.final_type,
                    normalize_text(expect.final_value)[:160],
                    normalize_text(result.final)[:160],
                )
            )

    if expect.tool_mode != "none":
        for required in expect.required_calls:
            if not any(_call_matches(required, observed) for observed in result.tool_calls):
                failures.append(
                    "missing required tool call: %s(%s)"
                    % (required.get("name"), json.dumps(required.get("arguments_subset") or {}))
                )
        for forbidden in expect.forbidden_calls:
            if any(observed.get("name") == forbidden for observed in result.tool_calls):
                failures.append("forbidden tool call was made: %s" % forbidden)

    if expect.tool_mode == "exact":
        expected_names = [call.get("name") for call in expect.required_calls]
        observed_names = [call.get("name") for call in result.tool_calls]
        if sorted(expected_names) != sorted(observed_names):
            failures.append(
                "tool calls differ: expected %s, got %s"
                % (expected_names, observed_names)
            )

    if result.fixture_misses:
        failures.append(
            "%d tool call(s) had no recorded fixture; the agent went off-script"
            % result.fixture_misses
        )

    if expect.require_all_fixtures_used:
        unused = [fixture.name for fixture in case.fixtures if not fixture.used]
        if unused:
            failures.append("recorded tool calls were not replayed: %s" % ", ".join(unused))

    if result.steps > expect.max_steps:
        failures.append(
            "took %d steps, limit is %d" % (result.steps, expect.max_steps)
        )

    result.failures = failures
    result.passed = not failures
    return result


def replay_case(
    case: Case,
    command: str,
    timeout: float = 30.0,
    env: Optional[Dict[str, str]] = None,
) -> CaseResult:
    reset_fixtures(case)
    result = CaseResult(case_id=case.case_id, passed=False)
    started = time.monotonic()
    agent = SubprocessAgent(command, timeout=timeout, env=env)
    try:
        agent.start()
        response = agent.send(
            {
                "type": "start",
                "case_id": case.case_id,
                "input": [message.as_dict() for message in case.input],
                "tools": case.tools,
                "max_steps": case.expect.max_steps,
            }
        )
        while True:
            kind = response.get("type")
            if kind == "final":
                result.final = str(response.get("content") or "")
                break
            if kind == "error":
                result.error = str(response.get("message") or "agent reported an error")
                break
            if kind != "tool_call":
                result.error = "unexpected message type %r" % kind
                break

            name = str(response.get("name") or "")
            arguments = response.get("arguments") or {}
            if not isinstance(arguments, dict):
                result.error = "tool arguments must be an object"
                break
            result.steps += 1
            result.tool_calls.append({"name": name, "arguments": arguments})

            if result.steps > case.expect.max_steps:
                result.error = "step limit exceeded"
                break

            fixture = _find_fixture(case, name, arguments)
            if fixture is None:
                result.fixture_misses += 1
                reply = {
                    "type": "tool_result",
                    "name": name,
                    "arguments": arguments,
                    "content": "no recorded result for this call",
                    "is_error": True,
                }
            else:
                reply = {
                    "type": "tool_result",
                    "name": name,
                    "arguments": arguments,
                    "content": fixture.result,
                    "is_error": fixture.is_error,
                }
            response = agent.send(reply)
    except AgentError as error:
        result.error = str(error)
        tail = agent.stderr_tail
        if tail:
            result.error += " (stderr: %s)" % tail.replace("\n", " | ")[:400]
    finally:
        agent.close()
        result.duration_ms = (time.monotonic() - started) * 1000
    return evaluate(case, result)


@dataclass
class ReplaySummary:
    results: List[CaseResult]

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for result in self.results if result.passed)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": self.pass_rate,
            "results": [result.as_dict() for result in self.results],
        }


def replay_cases(
    cases: Sequence[Case],
    command: str,
    timeout: float = 30.0,
    fail_fast: bool = False,
    env: Optional[Dict[str, str]] = None,
) -> ReplaySummary:
    results: List[CaseResult] = []
    for case in cases:
        result = replay_case(case, command, timeout=timeout, env=env)
        results.append(result)
        if fail_fast and not result.passed:
            break
    return ReplaySummary(results)
