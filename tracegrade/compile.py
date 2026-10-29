"""Compile traces into replayable eval cases.

A case is what a regression test needs and nothing else:

- the opening input,
- the tools that were in play,
- the recorded tool results as fixtures, so the agent can be re-run offline,
- assertions derived from what the agent actually did.

The assertions are a starting point, not a verdict. They are written to a
readable JSON file precisely so a human can loosen or tighten them.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .trace import Message, Trace

WHITESPACE = re.compile(r"\s+")


class CompileError(ValueError):
    """Raised when a trace cannot be turned into a case."""


def normalize_text(value: str) -> str:
    """Collapse whitespace and strip. Used by the ``normalized`` matcher."""
    return WHITESPACE.sub(" ", (value or "").strip())


def _subset_match(expected: Any, actual: Any) -> bool:
    """True when ``expected`` is a recursive subset of ``actual``.

    Lists must match element-wise in order and length; dicts may omit keys;
    everything else is compared for equality. This is what lets a case say
    "the call must include query=q" without pinning unrelated options.
    """
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        return all(
            key in actual and _subset_match(value, actual[key])
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            return False
        return all(_subset_match(left, right) for left, right in zip(expected, actual))
    return expected == actual


@dataclass
class Fixture:
    name: str
    match: Dict[str, Any]
    result: str
    is_error: bool = False
    used: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "match": self.match,
            "result": self.result,
            "is_error": self.is_error,
        }


@dataclass
class Expectation:
    final_type: str = "normalized"
    final_value: str = ""
    required_calls: List[Dict[str, Any]] = field(default_factory=list)
    forbidden_calls: List[str] = field(default_factory=list)
    tool_mode: str = "required"
    max_steps: int = 8
    require_all_fixtures_used: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "final": {"type": self.final_type, "value": self.final_value},
            "tool_calls": {
                "mode": self.tool_mode,
                "required": self.required_calls,
                "forbidden": self.forbidden_calls,
                "max_steps": self.max_steps,
            },
            "require_all_fixtures_used": self.require_all_fixtures_used,
        }


@dataclass
class Case:
    case_id: str
    input: List[Message]
    tools: List[str]
    fixtures: List[Fixture]
    expect: Expectation
    meta: Dict[str, Any] = field(default_factory=dict)
    source_trace: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "source_trace": self.source_trace or self.case_id,
            "input": [message.as_dict() for message in self.input],
            "tools": self.tools,
            "fixtures": [fixture.as_dict() for fixture in self.fixtures],
            "expect": self.expect.as_dict(),
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Case":
        if not isinstance(raw, dict):
            raise CompileError("case must be an object")
        case_id = raw.get("case_id")
        if not case_id:
            raise CompileError("case is missing case_id")
        messages = [
            Message(
                role=str(item.get("role") or "user"),
                content=str(item.get("content") or ""),
                name=item.get("name"),
            )
            for item in raw.get("input") or []
        ]
        fixtures = [
            Fixture(
                name=str(item.get("name") or ""),
                match=item.get("match") or {},
                result=str(item.get("result") or ""),
                is_error=bool(item.get("is_error")),
            )
            for item in raw.get("fixtures") or []
        ]
        expect_raw = raw.get("expect") or {}
        final = expect_raw.get("final") or {}
        tools = expect_raw.get("tool_calls") or {}
        expect = Expectation(
            final_type=str(final.get("type") or "normalized"),
            final_value=str(final.get("value") or ""),
            required_calls=list(tools.get("required") or []),
            forbidden_calls=list(tools.get("forbidden") or []),
            tool_mode=str(tools.get("mode") or "required"),
            max_steps=int(tools.get("max_steps") or 8),
            require_all_fixtures_used=bool(expect_raw.get("require_all_fixtures_used")),
        )
        return cls(
            case_id=str(case_id),
            input=messages,
            tools=list(raw.get("tools") or []),
            fixtures=fixtures,
            expect=expect,
            meta=raw.get("meta") or {},
            source_trace=str(raw.get("source_trace") or case_id),
        )


@dataclass
class CompileOptions:
    final_mode: str = "normalized"
    tool_mode: str = "required"
    max_steps: Optional[int] = None
    extra_steps: int = 2
    require_all_fixtures_used: bool = False
    case_prefix: str = ""

    def validate(self) -> None:
        if self.final_mode not in ("exact", "normalized", "contains", "none"):
            raise CompileError(
                "final mode must be exact, normalized, contains, or none"
            )
        if self.tool_mode not in ("required", "exact", "none"):
            raise CompileError("tool mode must be required, exact, or none")


def compile_trace(trace: Trace, options: Optional[CompileOptions] = None) -> Case:
    options = options or CompileOptions()
    options.validate()

    if not trace.input:
        raise CompileError(
            "trace %r has no input messages, so there is nothing to replay" % trace.trace_id
        )

    fixtures: List[Fixture] = []
    required: List[Dict[str, Any]] = []
    for call in trace.tool_calls:
        result = trace.result_for(call)
        fixtures.append(
            Fixture(
                name=call.name,
                match=dict(call.arguments),
                result=result.content if result else "",
                is_error=result.is_error if result else False,
            )
        )
        required.append({"name": call.name, "arguments_subset": dict(call.arguments)})

    if options.tool_mode == "none":
        required = []

    recorded_steps = len(trace.tool_calls)
    if options.max_steps is not None:
        max_steps = options.max_steps
    else:
        max_steps = max(recorded_steps + options.extra_steps, 1)

    case_id = options.case_prefix + trace.trace_id
    return Case(
        case_id=case_id,
        input=list(trace.input),
        tools=trace.tool_names,
        fixtures=fixtures,
        expect=Expectation(
            final_type=options.final_mode,
            final_value=trace.final,
            required_calls=required,
            forbidden_calls=[],
            tool_mode=options.tool_mode,
            max_steps=max_steps,
            require_all_fixtures_used=options.require_all_fixtures_used,
        ),
        meta=dict(trace.meta),
        source_trace=trace.trace_id,
    )


def compile_traces(
    traces: Sequence[Trace], options: Optional[CompileOptions] = None
) -> List[Case]:
    options = options or CompileOptions()
    options.validate()
    cases: List[Case] = []
    seen: Dict[str, int] = {}
    for trace in traces:
        case = compile_trace(trace, options)
        if case.case_id in seen:
            seen[case.case_id] += 1
            case.case_id = "%s-%d" % (case.case_id, seen[case.case_id])
        else:
            seen[case.case_id] = 1
        cases.append(case)
    return cases


def to_jsonl(cases: Sequence[Case]) -> str:
    return (
        "\n".join(
            json.dumps(case.as_dict(), ensure_ascii=False, sort_keys=True) for case in cases
        )
        + "\n"
    )


def from_jsonl(text: str) -> List[Case]:
    cases: List[Case] = []
    stripped = text.lstrip()
    if not stripped:
        return cases
    if stripped[0] == "[":
        try:
            data = json.loads(stripped)
        except ValueError as error:
            raise CompileError("invalid JSON: %s" % error)
        return [Case.from_dict(row) for row in data]
    for index, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            row = json.loads(line)
        except ValueError as error:
            raise CompileError("line %d: invalid JSON: %s" % (index, error))
        cases.append(Case.from_dict(row))
    return cases
