"""A runnable demo: sample traces plus an agent that reproduces them.

``tracegrade demo`` writes two files:

- ``traces.jsonl``: six canonical traces of a tiny search-and-answer agent.
- ``mock_agent.py``: a deterministic agent speaking the replay protocol, which
  reproduces those traces.

Compile the traces and replay them against the mock agent and everything
passes. Then edit one final answer in ``traces.jsonl``, recompile, and watch a
case fail. That is the whole workflow in two commands.
"""

from __future__ import annotations

import json
from typing import Dict, List

QUESTIONS = [
    ("how do I reset my password", "Open Settings, then Security, then Reset password."),
    ("what is the refund window", "Refunds are accepted within 30 days of purchase."),
    ("which plans include SSO", "SSO is available on the Team and Enterprise plans."),
    ("how do I export my data", "Use Settings, then Data, then Export, to download a ZIP."),
    ("what is the rate limit", "The API allows 600 requests per minute per key."),
    ("how do I invite a teammate", "Open Settings, then Members, then Invite, and enter an email."),
]


def query_for(text: str) -> str:
    words = [word.strip("?.,!").lower() for word in text.split()]
    words = [word for word in words if word]
    return " ".join(words[:3])


def build_traces() -> List[Dict[str, object]]:
    traces: List[Dict[str, object]] = []
    for index, (question, answer) in enumerate(QUESTIONS, start=1):
        traces.append(
            {
                "trace_id": "demo-%02d" % index,
                "input": [{"role": "user", "content": question}],
                "steps": [
                    {
                        "type": "tool_call",
                        "id": "call-%d" % index,
                        "name": "search",
                        "arguments": {"query": query_for(question)},
                    },
                    {
                        "type": "tool_result",
                        "id": "call-%d" % index,
                        "name": "search",
                        "content": answer,
                    },
                ],
                "final": "According to the docs: " + answer,
                "meta": {"model": "demo-agent", "latency_ms": 400 + index * 30},
            }
        )
    return traces


def to_jsonl(traces: List[Dict[str, object]]) -> str:
    return "\n".join(json.dumps(trace, ensure_ascii=False, sort_keys=True) for trace in traces) + "\n"


MOCK_AGENT = '''"""A deterministic agent for the tracegrade demo.

It speaks the replay protocol: newline-delimited JSON on stdin and stdout.
See docs/protocol.md in the tracegrade repository.
"""

import json
import sys


def query_for(text):
    words = [word.strip("?.,!").lower() for word in text.split()]
    words = [word for word in words if word]
    return " ".join(words[:3])


def send(payload):
    sys.stdout.write(json.dumps(payload) + "\\n")
    sys.stdout.flush()


for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    message = json.loads(line)
    kind = message.get("type")

    if kind == "start":
        messages = message.get("input") or []
        text = messages[-1]["content"] if messages else ""
        send({"type": "tool_call", "name": "search", "arguments": {"query": query_for(text)}})
    elif kind == "tool_result":
        if message.get("is_error"):
            send({"type": "final", "content": "I could not look that up."})
        else:
            send({"type": "final", "content": "According to the docs: " + message.get("content", "")})
    else:
        send({"type": "error", "message": "unexpected message: %r" % kind})
'''


def write_demo(directory: str) -> Dict[str, str]:
    import os

    traces_path = os.path.join(directory, "traces.jsonl")
    agent_path = os.path.join(directory, "mock_agent.py")
    with open(traces_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(to_jsonl(build_traces()))
    with open(agent_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(MOCK_AGENT)
    return {"traces": traces_path, "agent": agent_path}
