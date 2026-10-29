"""Canonical trace model and loaders.

A trace is one recorded run of an agent: the input messages, the tool calls it
made, the tool results it received, and its final answer.

Two on-disk shapes are supported:

``canonical``
    ``{"trace_id": ..., "input": [...], "steps": [...], "final": ...}``

``openai``
    A chat-completions conversation: ``{"id": ..., "messages": [...]}`` with
    ``tool_calls`` on assistant messages and ``role: "tool"`` results.

Both load into the same :class:`Trace`, so everything downstream is
format-agnostic.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


class TraceError(ValueError):
    """Raised when a trace cannot be parsed."""


@dataclass
class Message:
    role: str
    content: str = ""
    name: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            data["name"] = self.name
        return data


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}


@dataclass
class ToolResult:
    id: str
    name: str
    content: str = ""
    is_error: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "content": self.content,
            "is_error": self.is_error,
        }


@dataclass
class Step:
    """One step of a trace: a tool call or a tool result."""

    kind: str  # "tool_call" | "tool_result"
    call: Optional[ToolCall] = None
    result: Optional[ToolResult] = None

    def as_dict(self) -> Dict[str, Any]:
        if self.kind == "tool_call":
            assert self.call is not None
            return {"type": "tool_call", **self.call.as_dict()}
        assert self.result is not None
        return {"type": "tool_result", **self.result.as_dict()}


@dataclass
class Trace:
    trace_id: str
    input: List[Message]
    steps: List[Step] = field(default_factory=list)
    final: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)
    source: str = "<memory>"

    @property
    def tool_calls(self) -> List[ToolCall]:
        return [step.call for step in self.steps if step.kind == "tool_call" and step.call]

    @property
    def tool_results(self) -> List[ToolResult]:
        return [
            step.result for step in self.steps if step.kind == "tool_result" and step.result
        ]

    @property
    def tool_names(self) -> List[str]:
        names: List[str] = []
        for call in self.tool_calls:
            if call.name not in names:
                names.append(call.name)
        return names

    def result_for(self, call: ToolCall) -> Optional[ToolResult]:
        for result in self.tool_results:
            if result.id == call.id:
                return result
        return None


def _text_of(content: Any) -> str:
    """Flatten the several shapes chat APIs use for message content."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text" and "text" in item:
                    parts.append(str(item["text"]))
                elif "content" in item:
                    parts.append(str(item["content"]))
        return "\n".join(part for part in parts if part)
    return str(content)


def _arguments(value: Any, line: int) -> Dict[str, Any]:
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError as error:
            raise TraceError("line %d: tool arguments are not valid JSON: %s" % (line, error))
        if not isinstance(parsed, dict):
            raise TraceError("line %d: tool arguments must be an object" % line)
        return parsed
    raise TraceError("line %d: tool arguments must be an object or JSON string" % line)


def from_canonical(raw: Dict[str, Any], line: int = 0, source: str = "<memory>") -> Trace:
    if not isinstance(raw, dict):
        raise TraceError("line %d: expected an object" % line)
    identifier = raw.get("trace_id") or raw.get("id")
    if not identifier:
        raise TraceError("line %d: missing trace_id" % line)

    input_messages: List[Message] = []
    for item in raw.get("input") or []:
        if not isinstance(item, dict):
            raise TraceError("line %d: input messages must be objects" % line)
        input_messages.append(
            Message(
                role=str(item.get("role") or "user"),
                content=_text_of(item.get("content")),
                name=item.get("name"),
            )
        )

    steps: List[Step] = []
    for item in raw.get("steps") or []:
        if not isinstance(item, dict):
            raise TraceError("line %d: steps must be objects" % line)
        kind = item.get("type") or item.get("kind")
        if kind in ("tool_call", "call"):
            name = item.get("name")
            if not name:
                raise TraceError("line %d: tool call without a name" % line)
            steps.append(
                Step(
                    kind="tool_call",
                    call=ToolCall(
                        id=str(item.get("id") or "call-%d" % (len(steps) + 1)),
                        name=str(name),
                        arguments=_arguments(item.get("arguments"), line),
                    ),
                )
            )
        elif kind in ("tool_result", "result"):
            steps.append(
                Step(
                    kind="tool_result",
                    result=ToolResult(
                        id=str(item.get("id") or ""),
                        name=str(item.get("name") or ""),
                        content=_text_of(item.get("content")),
                        is_error=bool(item.get("is_error")),
                    ),
                )
            )
        elif kind in ("assistant", "message"):
            # Assistant text is kept only when it is the final answer.
            continue
        else:
            raise TraceError("line %d: unknown step type %r" % (line, kind))

    final = _text_of(raw.get("final"))
    if not final:
        # Fall back to the last assistant message in the step list.
        for item in reversed(raw.get("steps") or []):
            if isinstance(item, dict) and item.get("type") in ("assistant", "message"):
                final = _text_of(item.get("content"))
                break

    meta = raw.get("meta") or {}
    if not isinstance(meta, dict):
        raise TraceError("line %d: meta must be an object" % line)

    return Trace(
        trace_id=str(identifier),
        input=input_messages,
        steps=steps,
        final=final,
        meta=meta,
        source=source,
    )


def from_openai(raw: Dict[str, Any], line: int = 0, source: str = "<memory>") -> Trace:
    """Convert a chat-completions conversation into a trace."""
    if not isinstance(raw, dict):
        raise TraceError("line %d: expected an object" % line)
    identifier = raw.get("id") or raw.get("trace_id")
    messages = raw.get("messages")
    if not isinstance(messages, list) or not messages:
        raise TraceError("line %d: openai traces need a non-empty messages list" % line)
    if not identifier:
        identifier = "trace-%d" % line

    input_messages: List[Message] = []
    steps: List[Step] = []
    final = ""

    for item in messages:
        if not isinstance(item, dict):
            raise TraceError("line %d: messages must be objects" % line)
        role = str(item.get("role") or "")
        if role == "tool":
            steps.append(
                Step(
                    kind="tool_result",
                    result=ToolResult(
                        id=str(item.get("tool_call_id") or ""),
                        name=str(item.get("name") or ""),
                        content=_text_of(item.get("content")),
                        is_error=bool(item.get("is_error")),
                    ),
                )
            )
            continue

        calls = item.get("tool_calls") or []
        if role == "assistant" and calls:
            for call in calls:
                if not isinstance(call, dict):
                    raise TraceError("line %d: tool_calls must be objects" % line)
                function = call.get("function") or {}
                steps.append(
                    Step(
                        kind="tool_call",
                        call=ToolCall(
                            id=str(call.get("id") or ""),
                            name=str(function.get("name") or call.get("name") or ""),
                            arguments=_arguments(function.get("arguments"), line),
                        ),
                    )
                )
            text = _text_of(item.get("content"))
            if text:
                final = text
            continue

        message = Message(role=role, content=_text_of(item.get("content")), name=item.get("name"))
        if role == "assistant":
            final = message.content
        elif not steps:
            input_messages.append(message)
        else:
            # Later user turns are part of the conversation, not the opening input.
            input_messages.append(message)

    meta = raw.get("meta") or {}
    if not isinstance(meta, dict):
        raise TraceError("line %d: meta must be an object" % line)
    for key in ("model", "latency_ms", "cost_usd", "timestamp"):
        if key in raw and key not in meta:
            meta[key] = raw[key]

    return Trace(
        trace_id=str(identifier),
        input=input_messages,
        steps=steps,
        final=final,
        meta=meta,
        source=source,
    )


def detect_format(raw: Dict[str, Any]) -> str:
    if isinstance(raw.get("messages"), list):
        return "openai"
    return "canonical"


def _iter_json(text: str) -> Iterable[Tuple[int, Dict[str, Any]]]:
    stripped = text.lstrip()
    if not stripped:
        return
    if stripped[0] == "[":
        try:
            data = json.loads(stripped)
        except ValueError as error:
            raise TraceError("invalid JSON: %s" % error)
        if not isinstance(data, list):
            raise TraceError("top-level JSON must be an array or one object per line")
        for index, row in enumerate(data, start=1):
            yield index, row
        return
    for index, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            row = json.loads(line)
        except ValueError as error:
            raise TraceError("line %d: invalid JSON: %s" % (index, error))
        yield index, row


def load_traces(path: str, fmt: str = "auto") -> List[Trace]:
    if not os.path.exists(path):
        raise TraceError("file not found: %s" % path)
    with open(path, "r", encoding="utf-8-sig") as handle:
        text = handle.read()
    return load_traces_from_string(text, fmt=fmt, source=os.path.basename(path))


def load_traces_from_string(
    text: str, fmt: str = "auto", source: str = "<memory>"
) -> List[Trace]:
    traces: List[Trace] = []
    for line, raw in _iter_json(text):
        if not isinstance(raw, dict):
            raise TraceError("line %d: expected an object" % line)
        kind = detect_format(raw) if fmt == "auto" else fmt
        if kind == "openai":
            traces.append(from_openai(raw, line, source))
        elif kind == "canonical":
            traces.append(from_canonical(raw, line, source))
        else:
            raise TraceError("unknown trace format %r" % fmt)
    if not traces:
        raise TraceError("no traces found in %s" % source)
    return traces


def to_jsonl(traces: Sequence[Trace]) -> str:
    lines = []
    for trace in traces:
        row = {
            "trace_id": trace.trace_id,
            "input": [message.as_dict() for message in trace.input],
            "steps": [step.as_dict() for step in trace.steps],
            "final": trace.final,
            "meta": trace.meta,
        }
        lines.append(json.dumps(row, ensure_ascii=False, sort_keys=True))
    return "\n".join(lines) + "\n"
