# The replay protocol

tracegrade does not import your agent, link against a framework, or assume a
model provider. It starts a process and talks to it over stdin/stdout with one
JSON object per line.

This keeps the harness language-agnostic: a Python agent, a Node agent, a Go
binary, or a shell script wrapping an HTTP call all work the same way.

## Framing

- One JSON object per line, UTF-8, `\n` terminated.
- The harness writes a request, then blocks until the agent writes exactly one
  response line.
- Anything the agent writes to **stderr** is captured for diagnostics and shown
  only when the case fails.
- The process is started fresh for every case and terminated afterwards.

## Harness to agent

### `start`

Sent once, immediately after the process starts.

```json
{
  "type": "start",
  "case_id": "demo-01",
  "input": [{"role": "user", "content": "how do I reset my password"}],
  "tools": ["search"],
  "max_steps": 3
}
```

`input` is the conversation that preceded the agent's first action, in
OpenAI chat format. `tools` lists the tool names the agent may call.

### `tool_result`

Sent in response to every `tool_call`.

```json
{
  "type": "tool_result",
  "name": "search",
  "arguments": {"query": "how do i"},
  "content": "Open Settings, then Security, then Reset password.",
  "is_error": false
}
```

The content is the recorded output from the original run. `is_error: true`
means the harness had no fixture for the call: the agent went off-script, and
the case will be marked failed even if the agent recovers.

## Agent to harness

### `tool_call`

```json
{"type": "tool_call", "name": "search", "arguments": {"query": "how do i"}}
```

The harness matches the call against the case fixtures by name and by
recursive subset of the arguments, consumes it, and replies with
`tool_result`. Calls beyond `max_steps` are rejected.

### `final`

```json
{"type": "final", "content": "According to the docs: ..."}
```

Ends the case. The harness evaluates the final matcher and the tool-call
assertions.

### `error`

```json
{"type": "error", "message": "upstream model unavailable"}
```

Ends the case as a failure, with the message recorded verbatim.

## Minimal agent

```python
import json, sys

for line in sys.stdin:
    message = json.loads(line)
    if message["type"] == "start":
        print(json.dumps({"type": "tool_call", "name": "search",
                          "arguments": {"query": "password"}}), flush=True)
    elif message["type"] == "tool_result":
        print(json.dumps({"type": "final",
                          "content": "According to the docs: " + message["content"]}),
              flush=True)
```

Run it with:

```bash
tracegrade replay cases.jsonl --agent "python agent.py"
```

## Notes

- Nothing is required beyond a process that reads and writes lines. There is no
  SDK to install and no version to pin.
- The harness enforces `--timeout` per response, so a hung agent fails the case
  instead of hanging CI.
- Assertions live in the case file, not in the agent, so the same agent can be
  checked against several suites.
