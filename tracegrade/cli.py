"""tracegrade command line interface."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional, Sequence

from . import __version__
from .compile import (
    Case,
    CompileError,
    CompileOptions,
    compile_traces,
    from_jsonl,
    to_jsonl,
)
from .demo import write_demo
from .report import render_console, render_json, render_markdown
from .replay import replay_cases
from .trace import TraceError, load_traces


def _load_cases(path: str):
    if not os.path.exists(path):
        raise CompileError("file not found: %s" % path)
    with open(path, "r", encoding="utf-8-sig") as handle:
        return from_jsonl(handle.read())


def _write(text: str, path: Optional[str]) -> None:
    if path:
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        print("wrote %s" % path, file=sys.stderr)
    else:
        sys.stdout.write(text)


def cmd_compile(args: argparse.Namespace) -> int:
    traces = load_traces(args.file, fmt=args.format)
    options = CompileOptions(
        final_mode=args.final_mode,
        tool_mode=args.tool_mode,
        max_steps=args.max_steps,
        extra_steps=args.extra_steps,
        require_all_fixtures_used=args.require_all_fixtures,
        case_prefix=args.case_prefix or "",
    )
    cases = compile_traces(traces, options)
    _write(to_jsonl(cases), args.out)
    print("compiled %d case(s) from %d trace(s)" % (len(cases), len(traces)), file=sys.stderr)
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    cases = _load_cases(args.file)
    if args.case:
        wanted = set(args.case)
        cases = [case for case in cases if case.case_id in wanted]
        if not cases:
            print("error: no matching cases", file=sys.stderr)
            return 1
    summary = replay_cases(
        cases,
        args.agent,
        timeout=args.timeout,
        fail_fast=args.fail_fast,
    )

    if args.json:
        _write(render_json(summary), args.json)
    if args.report:
        _write(render_markdown(summary, title=args.title), args.report)
    if not args.json and not args.report:
        print(render_console(summary))
    elif not args.quiet:
        print(render_console(summary), file=sys.stderr)

    if args.min_pass_rate is not None and summary.pass_rate < args.min_pass_rate:
        print(
            "gate failed: pass rate %.1f%% is below %.1f%%"
            % (100 * summary.pass_rate, 100 * args.min_pass_rate),
            file=sys.stderr,
        )
        return 3
    return 0 if summary.failed == 0 else 1


def cmd_lint(args: argparse.Namespace) -> int:
    cases = _load_cases(args.file)
    problems = 0
    for case in cases:
        issues = []
        if not case.input:
            issues.append("no input messages")
        if not case.expect.final_type:
            issues.append("no final matcher")
        if case.expect.tool_mode != "none" and not case.expect.required_calls:
            issues.append("tool checks are on but no required calls are recorded")
        if case.expect.max_steps < 1:
            issues.append("max_steps must be at least 1")
        if case.expect.final_type == "none" and case.expect.tool_mode == "none":
            issues.append("nothing is asserted, so this case can never fail")
        if issues:
            problems += len(issues)
            print("%s:" % case.case_id)
            for issue in issues:
                print("  - %s" % issue)
    print("%d case(s), %d problem(s)" % (len(cases), problems))
    return 0 if problems == 0 else 1


def cmd_show(args: argparse.Namespace) -> int:
    cases = _load_cases(args.file)
    for case in cases:
        if case.case_id == args.case:
            print(json.dumps(case.as_dict(), indent=2, ensure_ascii=False))
            return 0
    print("error: no case named %r" % args.case, file=sys.stderr)
    return 1


def cmd_demo(args: argparse.Namespace) -> int:
    directory = args.out or "."
    os.makedirs(directory, exist_ok=True)
    paths = write_demo(directory)
    print("wrote %s" % paths["traces"])
    print("wrote %s" % paths["agent"])
    print("")
    print("next:")
    print("  tracegrade compile %s -o cases.jsonl" % paths["traces"])
    print('  tracegrade replay cases.jsonl --agent "python %s"' % paths["agent"])
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tracegrade",
        description="Compile production agent traces into replayable regression evals.",
    )
    parser.add_argument("--version", action="version", version="tracegrade %s" % __version__)
    subparsers = parser.add_subparsers(dest="command")

    compile_parser = subparsers.add_parser("compile", help="turn traces into eval cases")
    compile_parser.add_argument("file", help="traces as JSONL or a JSON array")
    compile_parser.add_argument("-o", "--out", help="write cases here instead of stdout")
    compile_parser.add_argument(
        "--format",
        choices=("auto", "canonical", "openai"),
        default="auto",
        help="input trace shape (default: auto-detect)",
    )
    compile_parser.add_argument(
        "--final-mode",
        choices=("exact", "normalized", "contains", "none"),
        default="normalized",
        help="how strictly the final answer must match (default: normalized)",
    )
    compile_parser.add_argument(
        "--tool-mode",
        choices=("required", "exact", "none"),
        default="required",
        help="required = recorded calls must appear; exact = nothing more, nothing less",
    )
    compile_parser.add_argument("--max-steps", type=int, default=None)
    compile_parser.add_argument("--extra-steps", type=int, default=2)
    compile_parser.add_argument("--case-prefix", default="")
    compile_parser.add_argument(
        "--require-all-fixtures",
        action="store_true",
        help="fail a case if a recorded tool call was never replayed",
    )
    compile_parser.set_defaults(func=cmd_compile)

    replay_parser = subparsers.add_parser("replay", help="run cases against an agent")
    replay_parser.add_argument("file", help="cases produced by `tracegrade compile`")
    replay_parser.add_argument(
        "--agent",
        required=True,
        help='command that speaks the replay protocol, e.g. "python agent.py"',
    )
    replay_parser.add_argument("--timeout", type=float, default=30.0)
    replay_parser.add_argument("--case", action="append", help="only run these case ids")
    replay_parser.add_argument("--fail-fast", action="store_true")
    replay_parser.add_argument("-o", "--report", help="write a Markdown report here")
    replay_parser.add_argument("--json", help="write a JSON report here")
    replay_parser.add_argument("--title", default="Trace replay")
    replay_parser.add_argument("--quiet", action="store_true")
    replay_parser.add_argument(
        "--min-pass-rate",
        type=float,
        default=None,
        help="exit with status 3 if the pass rate is below this value (0-1)",
    )
    replay_parser.set_defaults(func=cmd_replay)

    lint_parser = subparsers.add_parser("lint", help="find cases that assert nothing")
    lint_parser.add_argument("file")
    lint_parser.set_defaults(func=cmd_lint)

    show_parser = subparsers.add_parser("show", help="print one case as JSON")
    show_parser.add_argument("file")
    show_parser.add_argument("case")
    show_parser.set_defaults(func=cmd_show)

    demo_parser = subparsers.add_parser("demo", help="write sample traces and a mock agent")
    demo_parser.add_argument("-o", "--out", help="directory (default: current)")
    demo_parser.set_defaults(func=cmd_demo)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 2
    try:
        return int(args.func(args))
    except (TraceError, CompileError) as error:
        print("error: %s" % error, file=sys.stderr)
        return 1
    except BrokenPipeError:
        return 0
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
