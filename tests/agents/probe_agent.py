"""A configurable agent used by the replay tests.

Set ``AGENT_MODE`` to change its behaviour. It speaks the tracegrade replay
protocol: one JSON object per line on stdin and stdout.
"""

import json
import os
import sys
import time

MODE = os.environ.get("AGENT_MODE", "good")


def query_for(text):
    words = [word.strip("?.,!").lower() for word in text.split()]
    words = [word for word in words if word]
    return " ".join(words[:3])


def send(payload):
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


if MODE == "crash":
    sys.exit(3)

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    message = json.loads(line)
    kind = message.get("type")

    if kind == "start":
        if MODE == "garbage":
            sys.stdout.write("this is not json\n")
            sys.stdout.flush()
            continue
        if MODE == "silent":
            time.sleep(30)
            continue
        if MODE == "error":
            send({"type": "error", "message": "upstream model unavailable"})
            continue
        if MODE == "weird":
            send({"type": "telepathy"})
            continue
        if MODE == "looping":
            send({"type": "tool_call", "name": "search", "arguments": {"query": "loop"}})
            continue
        if MODE == "offscript":
            send({"type": "tool_call", "name": "search", "arguments": {"query": "different"}})
            continue
        messages = message.get("input") or []
        text = messages[-1]["content"] if messages else ""
        send({"type": "tool_call", "name": "search", "arguments": {"query": query_for(text)}})

    elif kind == "tool_result":
        if MODE == "looping":
            send({"type": "tool_call", "name": "search", "arguments": {"query": "loop"}})
        elif MODE == "wrong_final":
            send({"type": "final", "content": "Something else entirely."})
        else:
            send(
                {
                    "type": "final",
                    "content": "According to the docs: " + message.get("content", ""),
                }
            )
    else:
        send({"type": "error", "message": "unexpected message: %r" % kind})
