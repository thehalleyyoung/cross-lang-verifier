#!/usr/bin/env python3
"""Main CLI entry point for the Cross-Language Equivalence Verifier (XLEV).

Provides subcommands: verify, fuzz, analyze, benchmark.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import List, Optional

from .config import VerifyConfig, OutputFormat, Verbosity, get_profile, list_profiles
from .reporter import (
    VerificationReport, ReportWriter, VerdictKind,
    EquivalenceVerdict, TimingInfo,
)
from .pipeline import VerificationPipeline, PipelinePhase, PipelineStatus, PipelineBuilder


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="xlev",
        description="Cross-Language Equivalence Verifier: "
                    "verify equivalence of C and Rust function implementations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  xlev verify --c-source add.c --rust-source add.rs
  xlev verify --c-source lib.c --rust-source lib.rs --function add_numbers --profile thorough
  xlev fuzz --c-source add.c --rust-source add.rs --iterations 10000
  xlev analyze --c-source lib.c --rust-source lib.rs
  xlev benchmark --suite examples/benchmarks/
""",
    )
    parser.add_argument("--version", action="version", version="xlev 0.1.0")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="Increase verbosity (-v, -vv, -vvv)")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress non-essential output")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config file (JSON or YAML)")
    parser.add_argument("--profile", type=str, default=None,
                        choices=list_profiles(),
                        help="Use a named config profile")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- verify ---
    verify_parser = subparsers.add_parser("verify", help="Verify equivalence of C and Rust functions")
    _add_source_args(verify_parser)
    verify_parser.add_argument("--function", "-f", type=str, default=None,
                               help="Function name to verify (used for both C and Rust)")
    verify_parser.add_argument("--c-function", type=str, default=None,
                               help="C function name (overrides --function)")
    verify_parser.add_argument("--rust-function", type=str, default=None,
                               help="Rust function name (overrides --function)")
    verify_parser.add_argument("--output", "-o", type=str, default=None,
                               help="Output file path")
    verify_parser.add_argument("--format", type=str, default="json",
                               choices=["json", "text", "html"],
                               help="Output format (default: json)")
    verify_parser.add_argument("--timeout", type=float, default=None,
                               help="Total timeout in seconds")
    verify_parser.add_argument("--smt-timeout", type=float, default=None,
                               help="SMT solver timeout in seconds")
    verify_parser.add_argument("--loop-bound", type=int, default=None,
                               help="Loop unrolling bound")
    verify_parser.add_argument("--no-fuzz", action="store_true",
                               help="Disable fuzzing phase")
    verify_parser.add_argument("--report-dir", type=str, default=None,
                               help="Directory for HTML reports")

    # --- fuzz ---
    fuzz_parser = subparsers.add_parser("fuzz", help="Run differential fuzzing only")
    _add_source_args(fuzz_parser)
    fuzz_parser.add_argument("--function", "-f", type=str, default=None,
                             help="Function name to fuzz")
    fuzz_parser.add_argument("--iterations", "-n", type=int, default=None,
                             help="Maximum fuzzing iterations")
    fuzz_parser.add_argument("--seed-count", type=int, default=None,
                             help="Number of initial seeds")
    fuzz_parser.add_argument("--timeout", type=float, default=None,
                             help="Fuzzing timeout in seconds")
    fuzz_parser.add_argument("--output", "-o", type=str, default=None,
                             help="Output file for results")
    fuzz_parser.add_argument("--coverage-target", type=float, default=None,
                             help="Target coverage ratio (0.0-1.0)")

    # --- analyze ---
    analyze_parser = subparsers.add_parser("analyze", help="Run analysis passes on source pair")
    _add_source_args(analyze_parser)
    analyze_parser.add_argument("--output", "-o", type=str, default=None,
                                help="Output file for analysis results")
    analyze_parser.add_argument("--dot", type=str, default=None,
                                help="Output CFG in DOT format to file")
    analyze_parser.add_argument("--no-alias", action="store_true",
                                help="Skip alias analysis")
    analyze_parser.add_argument("--interprocedural", action="store_true",
                                help="Run interprocedural analysis")

    # --- benchmark ---
    bench_parser = subparsers.add_parser("benchmark", help="Run verification on a benchmark suite")
    bench_parser.add_argument("--suite", "-s", type=str, required=True,
                              help="Path to benchmark suite directory")
    bench_parser.add_argument("--output", "-o", type=str, default=None,
                              help="Output file for benchmark results")
    bench_parser.add_argument("--timeout", type=float, default=None,
                              help="Per-benchmark timeout in seconds")
    bench_parser.add_argument("--parallel", type=int, default=1,
                              help="Number of parallel workers")
    bench_parser.add_argument("--filter", type=str, default=None,
                              help="Filter benchmarks by name pattern")

    return parser


def _add_source_args(parser: argparse.ArgumentParser) -> None:
    """Add common source file arguments to a subparser."""
    parser.add_argument("--c-source", "-c", type=str, required=True,
                        help="Path to C source file")
    parser.add_argument("--rust-source", "-r", type=str, required=True,
                        help="Path to Rust source file")


def _setup_logging(verbosity: int, quiet: bool) -> None:
    """Configure logging based on verbosity level."""
    if quiet:
        level = logging.ERROR
    elif verbosity >= 3:
        level = logging.DEBUG
    elif verbosity >= 2:
        level = logging.DEBUG
    elif verbosity >= 1:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_config(args: argparse.Namespace) -> VerifyConfig:
    """Load config from file, profile, or defaults, then apply CLI overrides."""
    if args.config:
        config = VerifyConfig.load_file(args.config)
    elif args.profile:
        config = get_profile(args.profile)
    else:
        config = VerifyConfig.find_and_load()

    # Apply verbosity
    if args.quiet:
        config.verbosity = Verbosity.QUIET
    elif args.verbose >= 3:
        config.verbosity = Verbosity.DEBUG
    elif args.verbose >= 2:
        config.verbosity = Verbosity.VERBOSE
    elif args.verbose >= 1:
        config.verbosity = Verbosity.VERBOSE

    return config


def _make_progress_callback(quiet: bool):
    """Create a progress callback for terminal display."""
    if quiet:
        return None

    start = time.time()

    def callback(phase: PipelinePhase, status: PipelineStatus, msg: str = ""):
        elapsed = time.time() - start
        if status == PipelineStatus.RUNNING:
            print(f"  [{elapsed:6.1f}s] {phase.value}...", end="", flush=True)
        elif status == PipelineStatus.COMPLETED:
            print(f" done {msg}", flush=True)
        elif status == PipelineStatus.FAILED:
            print(f" FAILED: {msg}", flush=True)
        elif status == PipelineStatus.TIMED_OUT:
            print(f" TIMEOUT", flush=True)

    return callback


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def cmd_verify(args: argparse.Namespace) -> int:
    """Handle the 'verify' subcommand."""
    config = _load_config(args)

    # Apply verify-specific CLI overrides
    if args.timeout is not None:
        config.timeouts.total_timeout = args.timeout
    if args.smt_timeout is not None:
        config.timeouts.smt_timeout = args.smt_timeout
    if args.loop_bound is not None:
        config.loops.default_bound = args.loop_bound
    if args.no_fuzz:
        config.fuzzer.enabled = False
    if args.report_dir:
        config.report_dir = args.report_dir

    c_func = args.c_function or args.function
    rust_func = args.rust_function or args.function

    config.output_format = OutputFormat(args.format)

    if not os.path.isfile(args.c_source):
        print(f"Error: C source file not found: {args.c_source}", file=sys.stderr)
        return 1
    if not os.path.isfile(args.rust_source):
        print(f"Error: Rust source file not found: {args.rust_source}", file=sys.stderr)
        return 1

    pipeline = VerificationPipeline(config)
    progress = _make_progress_callback(args.quiet)
    if progress:
        pipeline.set_progress_callback(progress)

    if not args.quiet:
        print(f"Verifying: {args.c_source} ↔ {args.rust_source}")
        if c_func:
            print(f"Functions: {c_func} (C) ↔ {rust_func or c_func} (Rust)")
        print()

    report = pipeline.verify_from_files(args.c_source, args.rust_source, c_func, rust_func)

    _output_report(report, args)

    return 0 if report.verdict.kind == VerdictKind.EQUIVALENT else (
        2 if report.verdict.kind == VerdictKind.DIVERGENT else 3
    )


def cmd_fuzz(args: argparse.Namespace) -> int:
    """Handle the 'fuzz' subcommand."""
    config = _load_config(args)
    config.fuzzer.enabled = True

    if args.iterations is not None:
        config.fuzzer.max_iterations = args.iterations
    if args.seed_count is not None:
        config.fuzzer.seed_count = args.seed_count
    if args.timeout is not None:
        config.timeouts.fuzz_timeout = args.timeout
        config.timeouts.total_timeout = args.timeout + 60
    if args.coverage_target is not None:
        config.fuzzer.coverage_target = args.coverage_target

    func_name = args.function

    if not os.path.isfile(args.c_source):
        print(f"Error: C source file not found: {args.c_source}", file=sys.stderr)
        return 1
    if not os.path.isfile(args.rust_source):
        print(f"Error: Rust source file not found: {args.rust_source}", file=sys.stderr)
        return 1

    with open(args.c_source) as f:
        c_source = f.read()
    with open(args.rust_source) as f:
        rust_source = f.read()

    pipeline = VerificationPipeline(config)
    progress = _make_progress_callback(args.quiet)
    if progress:
        pipeline.set_progress_callback(progress)

    if not args.quiet:
        print(f"Fuzzing: {args.c_source} ↔ {args.rust_source}")
        print(f"Max iterations: {config.fuzzer.max_iterations}")
        print()

    report = pipeline.fuzz_only(c_source, rust_source, func_name, func_name)

    if args.output:
        ReportWriter.write_json(report, args.output)
        if not args.quiet:
            print(f"\nResults written to {args.output}")
    else:
        print(report.format_terminal())

    return 0 if report.verdict.kind != VerdictKind.DIVERGENT else 2


def cmd_analyze(args: argparse.Namespace) -> int:
    """Handle the 'analyze' subcommand."""
    config = _load_config(args)

    if args.no_alias:
        config.analysis.run_alias_analysis = False
    if args.interprocedural:
        config.analysis.interprocedural = True

    if not os.path.isfile(args.c_source):
        print(f"Error: C source file not found: {args.c_source}", file=sys.stderr)
        return 1
    if not os.path.isfile(args.rust_source):
        print(f"Error: Rust source file not found: {args.rust_source}", file=sys.stderr)
        return 1

    with open(args.c_source) as f:
        c_source = f.read()
    with open(args.rust_source) as f:
        rust_source = f.read()

    pipeline = VerificationPipeline(config)
    if not args.quiet:
        print(f"Analyzing: {args.c_source} ↔ {args.rust_source}")

    results = pipeline.analyze_only(c_source, rust_source)

    output = json.dumps(results, indent=2, default=str)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        if not args.quiet:
            print(f"Analysis results written to {args.output}")
    else:
        print(output)

    return 0 if "error" not in results else 1


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Handle the 'benchmark' subcommand."""
    config = _load_config(args)
    if args.timeout is not None:
        config.timeouts.total_timeout = args.timeout

    suite_dir = args.suite
    if not os.path.isdir(suite_dir):
        print(f"Error: Benchmark suite directory not found: {suite_dir}", file=sys.stderr)
        return 1

    # Discover benchmark pairs
    benchmarks = _discover_benchmarks(suite_dir, args.filter)
    if not benchmarks:
        print("No benchmarks found.", file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"Running {len(benchmarks)} benchmark(s) from {suite_dir}")
        print()

    results = []
    for i, (name, c_path, rust_path) in enumerate(benchmarks):
        if not args.quiet:
            print(f"[{i + 1}/{len(benchmarks)}] {name}...", end="", flush=True)

        t0 = time.time()
        try:
            pipeline = VerificationPipeline(config)
            report = pipeline.verify_from_files(c_path, rust_path)
            elapsed = time.time() - t0
            result = {
                "name": name,
                "verdict": report.verdict.kind.value,
                "confidence": report.verdict.confidence,
                "time_seconds": round(elapsed, 3),
                "counterexamples": len(report.counterexamples),
                "warnings": len(report.warnings),
            }
            if not args.quiet:
                symbol = report.verdict.kind.symbol
                print(f" {symbol} {report.verdict.kind.value} ({elapsed:.2f}s)")
        except Exception as e:
            elapsed = time.time() - t0
            result = {
                "name": name,
                "verdict": "error",
                "error": str(e),
                "time_seconds": round(elapsed, 3),
            }
            if not args.quiet:
                print(f" ERROR: {e}")

        results.append(result)

    # Summary
    summary = _benchmark_summary(results)
    if not args.quiet:
        print()
        print("=" * 50)
        print(f"Results: {summary['equivalent']} equivalent, "
              f"{summary['divergent']} divergent, "
              f"{summary['unknown']} unknown, "
              f"{summary['error']} errors")
        print(f"Total time: {summary['total_time']:.2f}s")
        print("=" * 50)

    output = json.dumps({"benchmarks": results, "summary": summary}, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        if not args.quiet:
            print(f"\nResults written to {args.output}")
    else:
        if args.quiet:
            print(output)

    return 0


def _discover_benchmarks(suite_dir: str, pattern: Optional[str] = None):
    """Discover C/Rust benchmark pairs in a directory."""
    benchmarks = []
    for entry in sorted(os.listdir(suite_dir)):
        entry_path = os.path.join(suite_dir, entry)

        if os.path.isdir(entry_path):
            c_file = os.path.join(entry_path, "impl.c")
            rust_file = os.path.join(entry_path, "impl.rs")
            if not os.path.isfile(c_file):
                c_candidates = [f for f in os.listdir(entry_path) if f.endswith(".c")]
                c_file = os.path.join(entry_path, c_candidates[0]) if c_candidates else None
            if not os.path.isfile(rust_file):
                rs_candidates = [f for f in os.listdir(entry_path) if f.endswith(".rs")]
                rust_file = os.path.join(entry_path, rs_candidates[0]) if rs_candidates else None

            if c_file and rust_file and os.path.isfile(c_file) and os.path.isfile(rust_file):
                name = entry
                if pattern and pattern not in name:
                    continue
                benchmarks.append((name, c_file, rust_file))

    return benchmarks


def _benchmark_summary(results: list) -> dict:
    verdicts = [r.get("verdict", "error") for r in results]
    return {
        "total": len(results),
        "equivalent": verdicts.count("equivalent"),
        "divergent": verdicts.count("divergent"),
        "unknown": verdicts.count("unknown"),
        "error": verdicts.count("error"),
        "total_time": sum(r.get("time_seconds", 0) for r in results),
    }


def _output_report(report: VerificationReport, args: argparse.Namespace) -> None:
    """Output the report in the requested format."""
    fmt = getattr(args, "format", "json")

    if hasattr(args, "output") and args.output:
        ReportWriter.write(report, args.output, fmt)
        if not args.quiet:
            print(f"\nReport written to {args.output}")
    else:
        if fmt == "json":
            print(report.to_json())
        elif fmt == "html":
            print(report.format_html())
        else:
            print(report.format_terminal())

    # Write HTML report to report_dir if specified
    if hasattr(args, "report_dir") and args.report_dir:
        os.makedirs(args.report_dir, exist_ok=True)
        html_path = os.path.join(args.report_dir, "report.html")
        ReportWriter.write_html(report, html_path)
        if not args.quiet:
            print(f"HTML report: {html_path}")


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    _setup_logging(args.verbose, args.quiet)

    commands = {
        "verify": cmd_verify,
        "fuzz": cmd_fuzz,
        "analyze": cmd_analyze,
        "benchmark": cmd_benchmark,
    }

    handler = commands.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    try:
        return handler(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as e:
        logging.getLogger(__name__).exception("Fatal error")
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
