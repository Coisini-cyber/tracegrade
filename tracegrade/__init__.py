"""tracegrade: compile production agent traces into replayable regression evals.

    >>> from tracegrade import load_traces, compile_traces, replay_cases
    >>> cases = compile_traces(load_traces("traces.jsonl"))
    >>> summary = replay_cases(cases, "python agent.py")
    >>> print(summary.passed, "/", summary.total)

The compiler keeps the opening input, turns recorded tool results into
fixtures, and derives assertions from what the agent actually did. The replayer
runs any process that speaks the line-delimited JSON protocol in
``docs/protocol.md`` and answers its tool calls from those fixtures.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .compile import (  # noqa: E402
    Case,
    CompileError,
    CompileOptions,
    Expectation,
    Fixture,
    compile_trace,
    compile_traces,
    from_jsonl,
    normalize_text,
    to_jsonl,
)
from .replay import (  # noqa: E402
    AgentError,
    CaseResult,
    ReplaySummary,
    SubprocessAgent,
    replay_case,
    replay_cases,
)
from .report import render_console, render_json, render_markdown  # noqa: E402
from .trace import (  # noqa: E402
    Message,
    ToolCall,
    ToolResult,
    Trace,
    TraceError,
    load_traces,
    load_traces_from_string,
    to_jsonl as traces_to_jsonl,
)

__all__ = [
    "__version__",
    "AgentError",
    "Case",
    "CaseResult",
    "CompileError",
    "CompileOptions",
    "Expectation",
    "Fixture",
    "Message",
    "ReplaySummary",
    "SubprocessAgent",
    "ToolCall",
    "ToolResult",
    "Trace",
    "TraceError",
    "compile_trace",
    "compile_traces",
    "from_jsonl",
    "load_traces",
    "load_traces_from_string",
    "normalize_text",
    "render_console",
    "render_json",
    "render_markdown",
    "replay_case",
    "replay_cases",
    "to_jsonl",
    "traces_to_jsonl",
]
