#!/usr/bin/env python3
"""
SemRec CLI — Verification Oracle for LLM Translation Pipelines.

Commands:
  semrec verify  --source C --target Rust --c-file X --rs-file Y
  semrec cegar   --source-file X [--model gpt-4.1-nano] [--max-iter 5]
  semrec bench   [--output results.json]

Examples:
  semrec verify --c-file add.c --rs-file add.rs
  semrec cegar --source-file overflow.c --max-iter 5
  semrec bench --output results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import logging
from typing import List, Optional

logger = logging.getLogger("semrec")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="semrec",
        description="SemRec: Verification Oracle for Cross-Language Equivalence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  semrec verify --c-file add.c --rs-file add.rs
  semrec verify --c-code 'int f(int x){return x+1;}' --rs-code 'pub fn f(x:i32)->i32{x.wrapping_add(1)}'
  semrec cegar --source-file overflow.c --model gpt-4.1-nano
  semrec bench --output results.json --pairs 200
""",
    )
    parser.add_argument("--version", action="version", version="semrec 0.2.0")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    parser.add_argument("-q", "--quiet", action="store_true")

    sub = parser.add_subparsers(dest="command", help="Available commands")

    # --- verify ---
    vp = sub.add_parser("verify", help="Verify equivalence of C and Rust functions")
    vp.add_argument("positional", nargs="*", help="C and Rust source files (positional)")
    vp.add_argument("--source", default="C", help="Source language (default: C)")
    vp.add_argument("--target", default="Rust", help="Target language (default: Rust)")
    vp.add_argument("--c-file", type=str, help="Path to C source file")
    vp.add_argument("--rs-file", type=str, help="Path to Rust source file")
    vp.add_argument("--c-code", type=str, help="Inline C code")
    vp.add_argument("--rs-code", type=str, help="Inline Rust code")
    vp.add_argument("--function", "-f", type=str, help="Function name")
    vp.add_argument("--timeout", type=int, default=10000, help="SMT timeout (ms)")
    vp.add_argument("--output", "-o", type=str, help="Output file (JSON)")
    vp.add_argument("--format", choices=["json", "text"], default="json")

    # --- cegar ---
    cp = sub.add_parser("cegar", help="CEGAR loop: translate C→Rust with LLM + verification")
    cp.add_argument("--source-file", type=str, help="C source file to translate")
    cp.add_argument("--source-code", type=str, help="Inline C code to translate")
    cp.add_argument("--model", default="gpt-4.1-nano", help="LLM model (default: gpt-4.1-nano)")
    cp.add_argument("--max-iter", type=int, default=5, help="Max CEGAR iterations")
    cp.add_argument("--timeout", type=int, default=10000, help="SMT timeout (ms)")
    cp.add_argument("--output", "-o", type=str, help="Output file (JSON)")
    cp.add_argument("--function", "-f", type=str, help="Function name")

    # --- bench ---
    bp = sub.add_parser("bench", help="Run benchmark suite")
    bp.add_argument("--output", "-o", type=str, help="Output file (JSON)")
    bp.add_argument("--pairs", type=int, default=0, help="Number of pairs to run (0=all)")
    bp.add_argument("--category", type=str, help="Filter by category")
    bp.add_argument("--cegar", action="store_true", help="Run CEGAR evaluation")
    bp.add_argument("--model", default="gpt-4.1-nano", help="LLM model for CEGAR")
    bp.add_argument("--max-iter", type=int, default=5, help="Max CEGAR iterations")

    return parser


def _setup_logging(verbose: int, quiet: bool):
    level = logging.WARNING if quiet else (
        logging.DEBUG if verbose >= 2 else
        logging.INFO if verbose >= 1 else logging.WARNING
    )
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def _ensure_path():
    """Ensure implementation src is on sys.path."""
    impl_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if impl_dir not in sys.path:
        sys.path.insert(0, impl_dir)


def cmd_verify(args) -> int:
    """Verify equivalence of C and Rust functions."""
    _ensure_path()
    from src.oracle.oracle import VerificationOracle

    # Support positional args: semrec verify file.c file.rs
    if hasattr(args, 'positional') and args.positional:
        for p in args.positional:
            if p.endswith('.c') or p.endswith('.h'):
                if not args.c_file:
                    args.c_file = p
            elif p.endswith('.rs'):
                if not args.rs_file:
                    args.rs_file = p

    # Get source code
    c_code = args.c_code
    rs_code = args.rs_code
    if args.c_file:
        if not os.path.isfile(args.c_file):
            print(f"Error: C source file not found: {args.c_file}", file=sys.stderr)
            return 1
        with open(args.c_file) as f:
            c_code = f.read()
    if args.rs_file:
        if not os.path.isfile(args.rs_file):
            print(f"Error: Rust source file not found: {args.rs_file}", file=sys.stderr)
            return 1
        with open(args.rs_file) as f:
            rs_code = f.read()

    if not c_code or not rs_code:
        print("Error: provide both C and Rust code (--c-file/--rs-file or --c-code/--rs-code)",
              file=sys.stderr)
        return 1

    oracle = VerificationOracle(timeout_ms=args.timeout)
    result = oracle.verify(c_code, rs_code, args.function)

    if args.format == "text":
        print(f"Verdict: {result.verdict}")
        if result.counterexample:
            print(f"Counterexample: {result.counterexample.format_human()}")
        if result.repair_hint:
            print(f"Repair: {result.repair_hint.description}")
        print(f"Time: {result.time_ms:.1f}ms")
    else:
        output = result.to_json()
        if args.output:
            with open(args.output, "w") as f:
                f.write(output)
            print(f"Result written to {args.output}", file=sys.stderr)
        else:
            print(output)

    return 0 if result.verdict in ("equivalent", "divergent") else 1


def cmd_cegar(args) -> int:
    """Run CEGAR loop on C source."""
    _ensure_path()
    from src.cegar_engine import CEGAREngine

    c_code = args.source_code
    if args.source_file:
        with open(args.source_file) as f:
            c_code = f.read()
    if not c_code:
        print("Error: provide C code (--source-file or --source-code)", file=sys.stderr)
        return 1

    func_name = args.function or "func"
    engine = CEGAREngine(
        model=args.model,
        max_iterations=args.max_iter,
        timeout_ms=args.timeout,
    )

    if not args.quiet:
        print(f"Running CEGAR loop (model={args.model}, max_iter={args.max_iter})...",
              file=sys.stderr)

    result = engine.run(c_code, func_name)

    if not args.quiet:
        status = "✓ CONVERGED" if result.converged else "✗ DIVERGENT"
        print(f"{status} after {result.total_iterations} iterations "
              f"({result.total_time_ms:.0f}ms)", file=sys.stderr)

    output = result.to_json()
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Result written to {args.output}", file=sys.stderr)
    else:
        print(output)

    return 0 if result.converged else 1


def cmd_bench(args) -> int:
    """Run benchmark suite."""
    _ensure_path()
    from src.oracle.oracle import VerificationOracle
    from benchmarks.pairs.benchmark_pairs import get_all_pairs
    try:
        from benchmarks.pairs.scaled_benchmark_pairs import get_scaled_pairs
        all_pairs = get_all_pairs() + get_scaled_pairs()
    except ImportError:
        all_pairs = get_all_pairs()

    if args.category:
        all_pairs = [p for p in all_pairs if p.category == args.category]

    if args.pairs > 0:
        all_pairs = all_pairs[:args.pairs]

    if not args.quiet:
        print(f"Running {len(all_pairs)} benchmark pairs...", file=sys.stderr)

    oracle = VerificationOracle(timeout_ms=10000)
    results = []
    correct = 0
    total = len(all_pairs)

    for i, pair in enumerate(all_pairs):
        r = oracle.verify(pair.c_source, pair.rust_source, pair.name)
        match = (r.verdict == pair.expected_result) or (
            pair.expected_result == "conditional"
        )
        if match:
            correct += 1
        results.append({
            "name": pair.name,
            "category": pair.category,
            "expected": pair.expected_result,
            "actual": r.verdict,
            "correct": match,
            "time_ms": round(r.time_ms, 2),
            "divergence_class": r.counterexample.divergence_class if r.counterexample else "",
        })
        if not args.quiet and (i + 1) % 20 == 0:
            print(f"  [{i+1}/{total}] accuracy so far: {correct}/{i+1}", file=sys.stderr)

    if args.cegar:
        if not args.quiet:
            print(f"\nRunning CEGAR evaluation on divergent pairs...", file=sys.stderr)
        from src.cegar_engine import CEGAREngine, analyze_cegar_results
        engine = CEGAREngine(model=args.model, max_iterations=args.max_iter)
        div_pairs = [(p.name, p.c_source) for p in all_pairs
                     if p.expected_result == "divergent"]
        if args.pairs > 0:
            div_pairs = div_pairs[:args.pairs]
        cegar_results = engine.run_batch(
            div_pairs,
            progress_callback=lambda i, n, name: (
                print(f"  CEGAR [{i+1}/{n}] {name}", file=sys.stderr)
                if not args.quiet else None
            ),
        )
        cegar_analysis = analyze_cegar_results(cegar_results)
    else:
        cegar_results = []
        cegar_analysis = {}

    summary = {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / max(total, 1) * 100, 1),
        "by_verdict": {
            v: sum(1 for r in results if r["actual"] == v)
            for v in ["equivalent", "divergent", "unknown", "error"]
        },
    }

    output_data = {
        "benchmark_results": results,
        "summary": summary,
    }
    if cegar_analysis:
        output_data["cegar_analysis"] = cegar_analysis

    output = json.dumps(output_data, indent=2, default=str)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        if not args.quiet:
            print(f"\nResults written to {args.output}", file=sys.stderr)

    if not args.quiet:
        print(f"\n{'='*50}", file=sys.stderr)
        print(f"Benchmark: {correct}/{total} correct ({summary['accuracy']}%)", file=sys.stderr)
        for v, c in summary["by_verdict"].items():
            if c > 0:
                print(f"  {v}: {c}", file=sys.stderr)
        if cegar_analysis:
            print(f"\nCEGAR: {cegar_analysis.get('converged', 0)}/{cegar_analysis.get('total_pairs', 0)} "
                  f"converged ({cegar_analysis.get('convergence_rate', 0)}%)", file=sys.stderr)
        print(f"{'='*50}", file=sys.stderr)

    if not args.output:
        print(output)

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    _setup_logging(args.verbose, getattr(args, 'quiet', False))

    commands = {"verify": cmd_verify, "cegar": cmd_cegar, "bench": cmd_bench}
    handler = commands.get(args.command)
    if not handler:
        parser.print_help()
        return 1

    try:
        return handler(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as e:
        logger.exception("Fatal error")
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
