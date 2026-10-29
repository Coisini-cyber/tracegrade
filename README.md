# tracegrade

**Turn production agent traces into replayable regression evals.**

You already have the test cases. They are sitting in your logs: real inputs,
real tool calls, real tool outputs, real answers. `tracegrade` compiles them
into a deterministic suite you can run in CI without touching a live model,
a live tool, or the network.

```bash
tracegrade demo -o demo/            # sample traces + a mock agent
tracegrade compile demo/traces.jsonl -o cases.jsonl
tracegrade replay cases.jsonl --agent "python demo/mock_agent.py"
```

```text
PASS  demo-01  (1 steps)
PASS  demo-02  (1 steps)
...
6/6 passed (100.0%)
```

Change the agent, run it again, and the cases that regressed fail.

## The idea

An eval is only useful if it is cheap to run and stable over time. Hand-written
evals are expensive and drift from what users actually do. Production traces
are the opposite: they are free, current, and cover the long tail.

The obstacle is that a trace is not a test. It records *one* run of a
non-deterministic system with live side effects. `tracegrade` removes both
problems:

1. **Fixtures instead of side effects.** Every tool call in the trace becomes a
   recorded fixture. During replay the agent's tool calls are answered from the
   fixtures, so nothing hits the network and the world does not move.
2. **Assertions from behaviour.** The compiler records what the agent did —
   which tools it called with which arguments, and what it finally said — and
   turns that into explicit, editable expectations.

The only thing that varies between runs is the agent's own decisions, which is
exactly what you want to test.

## Workflow

```text
production logs ──► compile ──► cases.jsonl ──► replay ──► report
                       │                          │
                 fixtures + assertions      your agent, offline
```

### 1. Export traces

Two shapes are accepted, auto-detected per line.

**Canonical** — you control the export:

```json
{
  "trace_id": "req-8842",
  "input": [{"role": "user", "content": "find the refund policy"}],
  "steps": [
    {"type": "tool_call", "id": "c1", "name": "search", "arguments": {"query": "refund"}},
    {"type": "tool_result", "id": "c1", "name": "search", "content": "30 days"}
  ],
  "final": "Refunds take 30 days.",
  "meta": {"model": "gpt-4.1", "latency_ms": 830}
}
```

**OpenAI** — a chat-completions conversation, as stored by most gateways:

```json
{"id": "req-8842", "model": "gpt-4.1", "messages": [
  {"role": "user", "content": "find the refund policy"},
  {"role": "assistant", "content": null, "tool_calls": [
    {"id": "c1", "type": "function",
     "function": {"name": "search", "arguments": "{\"query\": \"refund\"}"}}]},
  {"role": "tool", "tool_call_id": "c1", "content": "30 days"},
  {"role": "assistant", "content": "Refunds take 30 days."}]}
```

### 2. Compile

```bash
tracegrade compile traces.jsonl -o cases.jsonl
```

Each case contains the opening input, the tool names, one fixture per recorded
tool call, and an `expect` block:

```json
{
  "case_id": "req-8842",
  "input": [{"role": "user", "content": "find the refund policy"}],
  "tools": ["search"],
  "fixtures": [{"name": "search", "match": {"query": "refund"}, "result": "30 days"}],
  "expect": {
    "final": {"type": "normalized", "value": "Refunds take 30 days."},
    "tool_calls": {
      "mode": "required",
      "required": [{"name": "search", "arguments_subset": {"query": "refund"}}],
      "forbidden": [],
      "max_steps": 3
    }
  }
}
```

Flags that matter:

| flag | effect |
| --- | --- |
| `--final-mode normalized` | default: compare after collapsing whitespace |
| `--final-mode exact` | byte-for-byte |
| `--final-mode contains` | recorded answer must appear in the new one |
| `--final-mode none` | do not check the answer |
| `--tool-mode required` | recorded calls must appear, extras allowed |
| `--tool-mode exact` | nothing more, nothing less |
| `--tool-mode none` | do not check tools |
| `--extra-steps N` | allow N more steps than the original run (default 2) |
| `--require-all-fixtures` | fail if a recorded tool call is never replayed |

The output is plain JSON on purpose. Loosen an over-specific assertion by
editing one line; no DSL, no code generation, no framework.

### 3. Replay

Your agent is a process that speaks a line-delimited JSON protocol — see
[docs/protocol.md](docs/protocol.md). It reads a `start` message, emits
`tool_call` or `final`, and receives `tool_result` messages from the fixtures.

```bash
tracegrade replay cases.jsonl --agent "python agent.py"
tracegrade replay cases.jsonl --agent "node dist/agent.js" --timeout 20
tracegrade replay cases.jsonl --agent "python agent.py" -o report.md --json report.json
```

```bash
# fail CI when the pass rate drops
tracegrade replay cases.jsonl --agent "python agent.py" --min-pass-rate 0.95 --quiet
```

A fresh process is started per case, so a crash, a hang, or a leaked global
cannot affect the next case. `--timeout` applies to every response.

## What a case can fail on

| failure | cause |
| --- | --- |
| `final answer did not match` | the agent answered differently |
| `missing required tool call` | the agent skipped a tool the trace used |
| `forbidden tool call was made` | the agent used a tool you banned |
| `tool calls differ` | `--tool-mode exact` and the call set changed |
| `no recorded fixture` | the agent called a tool with arguments the trace never saw |
| `took N steps, limit is M` | the agent looped |
| `agent error` | the process crashed, timed out, or broke the protocol |

`no recorded fixture` is the interesting one: it means the agent went
off-script. The case fails even if the agent recovers, because the recorded
tool output no longer describes what happened.

## Commands

```bash
tracegrade compile FILE [-o cases.jsonl] [--format auto|canonical|openai] [...]
tracegrade replay cases.jsonl --agent CMD [-o report.md] [--json report.json]
tracegrade lint cases.jsonl        # find cases that can never fail
tracegrade show cases.jsonl CASE   # print one case
tracegrade demo -o DIR             # sample traces and a mock agent
```

Exit codes: `0` ok, `1` a case failed or input was invalid, `2` usage,
`3` the pass-rate gate failed.

## Keeping the suite honest

A compiled suite inherits the behaviour of the run it came from, including its
bugs. Two habits keep it useful:

- **Review the cases you commit.** A trace where the agent did the wrong thing
  becomes a test that enshrines the wrong thing. Delete those, or fix the
  expectation.
- **Refresh from production.** Traces age. Recompile from a recent window after
  a prompt or tool change, and keep a small hand-curated core that never
  changes.

`tracegrade lint` catches the most common rot: cases with no assertions at all.

## Limitations

- Replay is single-turn per case. Multi-turn conversations are supported as
  input history, but the agent is only asked to act once.
- Fixtures match by tool name and recursive subset of arguments. A tool whose
  output depends on hidden state (time, user id) needs those values pinned in
  the recorded arguments.
- The protocol is line-delimited JSON over stdio. Agents that only expose an
  HTTP API need a small adapter process.
- No third-party packages, no network calls, Python 3.9+.

## Development

```bash
python -m unittest discover -s tests -v
python -m compileall -q tracegrade
```

86 tests, including end-to-end replays against real subprocesses in seven
failure modes.

## License

MIT.
