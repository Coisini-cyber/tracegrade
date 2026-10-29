"""A deterministic agent for the tracegrade demo.

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
    sys.stdout.write(json.dumps(payload) + "\n")
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
