#!/usr/bin/env python3
"""
Scaled Counterexample-Guided LLM Repair Experiment.

Evaluates CExL-Repair on 120 CEGAR benchmark pairs and 149 real-world
verification pairs.  Produces publication-ready JSON results with bootstrap
confidence intervals, iteration-distribution analysis, and per-category
breakdowns.

Modes
-----
  --dry-run   (default when OPENAI_API_KEY is unset)
              Simulates repair results via the VerificationOracle without
              making LLM calls.  Useful for validating the analysis pipeline.
  --live      Forces live LLM calls (requires OPENAI_API_KEY).
  --pairs N   Limit to the first N CEGAR pairs (for quick iteration).
  --output P  Write JSON results to P instead of the default path.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Path setup — allow running from the experiments/ directory or project root.
# ---------------------------------------------------------------------------
_IMPL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _IMPL_ROOT)

from src.cegar_engine import (  # noqa: E402
    CExLRepairEngine,
    RepairResult,
    RepairIteration,
    analyze_repair_results,
    BUG_CLASSES,
)
from src.oracle.oracle import VerificationOracle  # noqa: E402
from benchmarks.pairs.cegar_benchmark_pairs import CEGAR_BENCHMARK_PAIRS  # noqa: E402
from benchmarks.pairs.real_world_benchmark_pairs import REAL_WORLD_BENCHMARK_PAIRS  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "gpt-4.1-nano"
DEFAULT_MAX_ITERS = 5
DEFAULT_TIMEOUT_MS = 10_000
DEFAULT_OUTPUT = os.path.join(
    _IMPL_ROOT, "experiments", "results", "scaled_repair_experiment.json"
)

# Simulated convergence probabilities per difficulty tier (dry-run mode).
_DRY_RUN_CONVERGENCE_PROB: Dict[str, float] = {
    "easy": 0.90,
    "medium": 0.65,
    "hard": 0.30,
}
_DRY_RUN_ITER_RANGE: Dict[str, Tuple[int, int]] = {
    "easy": (1, 2),
    "medium": (2, 4),
    "hard": (3, 5),
}


# ===================================================================
# Statistical helpers
# ===================================================================

def bootstrap_ci(
    data: Sequence[float],
    stat_func: Callable[[Sequence[float]], float] = statistics.mean,
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Non-parametric bootstrap confidence interval for *stat_func*.

    Returns (lower, upper) at the ``1 - alpha`` confidence level.
    """
    if len(data) < 2:
        val = stat_func(data) if data else 0.0
        return (val, val)

    rng = random.Random(42)  # reproducible
    boot_stats = sorted(
        stat_func([rng.choice(data) for _ in range(len(data))])
        for _ in range(n_bootstrap)
    )
    lo_idx = int((alpha / 2) * n_bootstrap)
    hi_idx = int((1 - alpha / 2) * n_bootstrap) - 1
    return (round(boot_stats[lo_idx], 4), round(boot_stats[hi_idx], 4))


def wilson_ci(
    successes: int,
    total: int,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if total == 0:
        return (0.0, 0.0)
    z = 1.96 if alpha == 0.05 else _z_for_alpha(alpha)
    p_hat = successes / total
    denom = 1 + z * z / total
    centre = (p_hat + z * z / (2 * total)) / denom
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * total)) / total) / denom
    return (round(max(centre - margin, 0.0), 4), round(min(centre + margin, 1.0), 4))


def _z_for_alpha(alpha: float) -> float:
    """Approximate z-score for common alpha values."""
    table = {0.10: 1.645, 0.05: 1.960, 0.01: 2.576}
    return table.get(alpha, 1.960)


def iteration_distribution_analysis(
    results: List[RepairResult],
) -> Dict[str, Any]:
    """Compute descriptive statistics over iteration counts."""
    counts = [r.total_iterations for r in results if r.total_iterations > 0]
    if not counts:
        return {"n": 0}

    n = len(counts)
    mean = statistics.mean(counts)
    median = statistics.median(counts)
    std = statistics.pstdev(counts)
    q = statistics.quantiles(counts, n=4) if n >= 2 else [mean, mean, mean]

    # Pearson's skewness approximation (3 * (mean - median) / std)
    skewness = 3.0 * (mean - median) / std if std > 0 else 0.0

    # Histogram buckets: 1, 2, 3, 4, 5+
    hist: Dict[str, int] = defaultdict(int)
    for c in counts:
        bucket = str(c) if c <= 4 else "5+"
        hist[bucket] += 1

    return {
        "n": n,
        "mean": round(mean, 3),
        "median": median,
        "std": round(std, 3),
        "skewness": round(skewness, 3),
        "q1": round(q[0], 2),
        "q2": round(q[1], 2),
        "q3": round(q[2], 2),
        "histogram": dict(hist),
    }


def reproducibility_analysis(
    results_runs: List[List[RepairResult]],
) -> Dict[str, Any]:
    """Compute cross-run variance metrics from multiple independent runs.

    *results_runs* is a list of per-run result lists (same ordering).
    """
    if len(results_runs) < 2:
        return {"num_runs": len(results_runs), "note": "need >=2 runs for variance"}

    n_pairs = len(results_runs[0])
    convergence_rates = [
        sum(r.converged for r in run) / max(len(run), 1)
        for run in results_runs
    ]
    iter_means = [
        statistics.mean([r.total_iterations for r in run]) if run else 0.0
        for run in results_runs
    ]

    # Per-function agreement (fraction of pairs that agree across all runs)
    agree = 0
    for i in range(n_pairs):
        verdicts = {results_runs[j][i].converged for j in range(len(results_runs))}
        if len(verdicts) == 1:
            agree += 1

    return {
        "num_runs": len(results_runs),
        "convergence_rates": [round(r, 4) for r in convergence_rates],
        "convergence_rate_std": round(statistics.pstdev(convergence_rates), 4),
        "iter_mean_std": round(statistics.pstdev(iter_means), 4),
        "per_function_agreement": round(agree / max(n_pairs, 1), 4),
    }


# ===================================================================
# Dry-run simulation
# ===================================================================

def _simulate_repair_result(
    pair: Dict[str, Any],
    oracle: VerificationOracle,
    rng: random.Random,
) -> RepairResult:
    """Simulate a repair result using the oracle (no LLM calls)."""
    name = pair["name"]
    c_code = pair["c_code"]
    buggy_rust = pair["buggy_rust"]
    difficulty = pair.get("difficulty", "medium")
    bug_class = pair.get("bug_class", "unknown")

    # Verify with oracle to determine ground-truth divergence
    t0 = time.monotonic()
    oracle_result = oracle.verify(c_code, buggy_rust, func_name=name)
    verify_ms = (time.monotonic() - t0) * 1000

    needs_repair = oracle_result.verdict == "divergent"

    if not needs_repair:
        # Already equivalent — converges on first try
        return RepairResult(
            func_name=name,
            c_code=c_code,
            converged=True,
            final_verdict="equivalent",
            iterations=[RepairIteration(
                iteration=1,
                rust_code=buggy_rust,
                verdict="equivalent",
                time_ms=verify_ms,
            )],
            total_iterations=1,
            total_time_ms=verify_ms,
            bug_class=bug_class,
            llm_repairable=True,
            repair_iterations=0,
        )

    # Simulate LLM repair loop
    converge_prob = _DRY_RUN_CONVERGENCE_PROB.get(difficulty, 0.5)
    converged = rng.random() < converge_prob
    lo, hi = _DRY_RUN_ITER_RANGE.get(difficulty, (2, 4))
    n_iters = rng.randint(lo, hi) if converged else DEFAULT_MAX_ITERS

    iterations: List[RepairIteration] = []
    for i in range(1, n_iters + 1):
        is_final = i == n_iters
        verdict = "equivalent" if (is_final and converged) else "divergent"
        div_class = bug_class if verdict == "divergent" else ""
        iterations.append(RepairIteration(
            iteration=i,
            rust_code=f"// simulated iteration {i}",
            verdict=verdict,
            divergence_class=div_class,
            time_ms=verify_ms / n_iters,
        ))

    return RepairResult(
        func_name=name,
        c_code=c_code,
        converged=converged,
        final_verdict="equivalent" if converged else "divergent",
        iterations=iterations,
        total_iterations=n_iters,
        total_time_ms=verify_ms * n_iters,
        bug_class=bug_class,
        llm_repairable=converged,
        repair_iterations=n_iters if converged else 0,
    )


# ===================================================================
# CEGAR benchmark experiment
# ===================================================================

def run_cegar_experiment(
    pairs: List[Dict[str, Any]],
    *,
    dry_run: bool = True,
    model: str = DEFAULT_MODEL,
) -> Tuple[List[RepairResult], Dict[str, Any]]:
    """Run the CExL-Repair loop on CEGAR benchmark pairs.

    Returns (per-pair results, analysis dict).
    """
    n = len(pairs)
    mode_label = "DRY-RUN (oracle-only)" if dry_run else f"LIVE ({model})"
    print(f"[CEGAR] {n} pairs — {mode_label}", file=sys.stderr)

    results: List[RepairResult] = []

    if dry_run:
        oracle = VerificationOracle(timeout_ms=DEFAULT_TIMEOUT_MS)
        rng = random.Random(12345)
        for idx, pair in enumerate(pairs):
            _progress(idx, n, pair["name"])
            result = _simulate_repair_result(pair, oracle, rng)
            results.append(result)
    else:
        engine = CExLRepairEngine(
            model=model,
            max_iterations=DEFAULT_MAX_ITERS,
            timeout_ms=DEFAULT_TIMEOUT_MS,
        )
        for idx, pair in enumerate(pairs):
            _progress(idx, n, pair["name"])
            try:
                result = engine.run(pair["c_code"], pair["name"])
                results.append(result)
            except Exception as exc:
                print(f" ERROR: {exc}", file=sys.stderr)
                results.append(RepairResult(
                    func_name=pair["name"],
                    c_code=pair["c_code"],
                    converged=False,
                    final_verdict="error",
                    bug_class=pair.get("bug_class", ""),
                ))

    analysis = analyze_repair_results(results)
    print(file=sys.stderr)
    return results, analysis


# ===================================================================
# Real-world verification experiment
# ===================================================================

def run_realworld_verification(
    pairs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Verify real-world C↔Rust pairs and compare with expected verdicts."""
    n = len(pairs)
    print(f"[REAL-WORLD] Verifying {n} pairs with oracle", file=sys.stderr)
    oracle = VerificationOracle(timeout_ms=DEFAULT_TIMEOUT_MS)

    correct = 0
    total = 0
    per_category: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "total": 0}
    )
    details: List[Dict[str, Any]] = []
    times: List[float] = []

    for idx, pair in enumerate(pairs):
        _progress(idx, n, pair["name"])
        expected = pair["expected_verdict"]
        t0 = time.monotonic()
        result = oracle.verify(
            pair["c_code"], pair["rust_code"], func_name=pair["name"]
        )
        elapsed = (time.monotonic() - t0) * 1000
        times.append(elapsed)

        predicted = result.verdict
        is_correct = predicted == expected
        if is_correct:
            correct += 1
        total += 1

        cat = pair.get("category", "unknown")
        per_category[cat]["total"] += 1
        if is_correct:
            per_category[cat]["correct"] += 1

        details.append({
            "name": pair["name"],
            "category": cat,
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct,
            "time_ms": round(elapsed, 1),
            "bug_class": pair.get("bug_class", ""),
        })

    accuracy = correct / max(total, 1)
    acc_ci = wilson_ci(correct, total)

    # Per-category accuracy
    cat_summary = {}
    for cat, counts in sorted(per_category.items()):
        cat_acc = counts["correct"] / max(counts["total"], 1)
        cat_ci = wilson_ci(counts["correct"], counts["total"])
        cat_summary[cat] = {
            "accuracy": round(cat_acc, 4),
            "correct": counts["correct"],
            "total": counts["total"],
            "wilson_ci_95": list(cat_ci),
        }

    print(file=sys.stderr)
    return {
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "wilson_ci_95": list(acc_ci),
        "avg_time_ms": round(statistics.mean(times), 1) if times else 0,
        "per_category": cat_summary,
        "details": details,
    }


# ===================================================================
# Aggregation helpers
# ===================================================================

def aggregate_by_field(
    results: List[RepairResult],
    pairs: List[Dict[str, Any]],
    field: str,
) -> Dict[str, Dict[str, Any]]:
    """Aggregate convergence metrics grouped by a benchmark field."""
    groups: Dict[str, List[Tuple[RepairResult, Dict[str, Any]]]] = defaultdict(list)
    for r, p in zip(results, pairs):
        key = p.get(field, "unknown")
        groups[key].append((r, p))

    out: Dict[str, Dict[str, Any]] = {}
    for key, items in sorted(groups.items()):
        res_list = [r for r, _ in items]
        n = len(res_list)
        conv = sum(1 for r in res_list if r.converged)
        rate = conv / max(n, 1)
        ci = wilson_ci(conv, n)
        iters = [r.total_iterations for r in res_list]
        times = [r.total_time_ms for r in res_list]

        boot_rate_ci = bootstrap_ci(
            [1.0 if r.converged else 0.0 for r in res_list]
        )

        out[key] = {
            "n": n,
            "converged": conv,
            "convergence_rate": round(rate, 4),
            "wilson_ci_95": list(ci),
            "bootstrap_ci_95": list(boot_rate_ci),
            "avg_iterations": round(statistics.mean(iters), 2) if iters else 0,
            "avg_time_ms": round(statistics.mean(times), 1) if times else 0,
        }
    return out


def build_time_distribution(results: List[RepairResult]) -> Dict[str, Any]:
    """Summary statistics for per-pair wall-clock time."""
    times = [r.total_time_ms for r in results]
    if not times:
        return {"n": 0}
    return {
        "n": len(times),
        "mean_ms": round(statistics.mean(times), 1),
        "median_ms": round(statistics.median(times), 1),
        "std_ms": round(statistics.pstdev(times), 1),
        "min_ms": round(min(times), 1),
        "max_ms": round(max(times), 1),
    }


# ===================================================================
# Output assembly
# ===================================================================

def build_output(
    cegar_results: List[RepairResult],
    cegar_analysis: Dict[str, Any],
    cegar_pairs: List[Dict[str, Any]],
    realworld_summary: Dict[str, Any],
    *,
    dry_run: bool,
    model: str,
) -> Dict[str, Any]:
    """Assemble the full JSON output for the experiment."""

    # Per-function detail records
    per_function = []
    for r, p in zip(cegar_results, cegar_pairs):
        per_function.append({
            "name": r.func_name,
            "category": p.get("category", ""),
            "difficulty": p.get("difficulty", ""),
            "bug_class": r.bug_class,
            "converged": r.converged,
            "final_verdict": r.final_verdict,
            "total_iterations": r.total_iterations,
            "total_time_ms": round(r.total_time_ms, 1),
            "llm_repairable": r.llm_repairable,
            "repair_iterations": r.repair_iterations,
        })

    # Convergence rate with bootstrap CI
    conv_data = [1.0 if r.converged else 0.0 for r in cegar_results]
    overall_boot_ci = bootstrap_ci(conv_data)

    n_conv = sum(r.converged for r in cegar_results)
    overall_wilson = wilson_ci(n_conv, len(cegar_results))

    return {
        "experiment": "Scaled CExL-Repair Evaluation",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "mode": "dry_run" if dry_run else "live",
        "model": model,
        "max_iterations": DEFAULT_MAX_ITERS,
        "cegar_benchmark": {
            "num_pairs": len(cegar_pairs),
            "summary": cegar_analysis,
            "convergence_rate_bootstrap_ci_95": list(overall_boot_ci),
            "convergence_rate_wilson_ci_95": list(overall_wilson),
            "iteration_distribution": iteration_distribution_analysis(cegar_results),
            "time_distribution": build_time_distribution(cegar_results),
            "by_category": aggregate_by_field(cegar_results, cegar_pairs, "category"),
            "by_bug_class": aggregate_by_field(cegar_results, cegar_pairs, "bug_class"),
            "by_difficulty": aggregate_by_field(cegar_results, cegar_pairs, "difficulty"),
            "per_function": per_function,
        },
        "real_world_benchmark": realworld_summary,
    }


# ===================================================================
# CLI / main
# ===================================================================

def _progress(idx: int, total: int, name: str) -> None:
    print(
        f"\r  [{idx + 1:>{len(str(total))}}/{total}] {name:<40}",
        end="",
        file=sys.stderr,
        flush=True,
    )


def _print_summary(output: Dict[str, Any]) -> None:
    """Print a human-readable summary to stderr."""
    cb = output["cegar_benchmark"]
    s = cb["summary"]
    sep = "=" * 64
    print(f"\n{sep}", file=sys.stderr)
    print("  SCALED CExL-REPAIR EXPERIMENT — SUMMARY", file=sys.stderr)
    print(f"{sep}", file=sys.stderr)
    print(f"  Mode           : {output['mode']}", file=sys.stderr)
    print(f"  Model          : {output['model']}", file=sys.stderr)
    print(f"  CEGAR pairs    : {cb['num_pairs']}", file=sys.stderr)
    print(f"  Convergence    : {s['converged']}/{s['total_pairs']}"
          f"  ({s['convergence_rate']}%)", file=sys.stderr)
    print(f"  Bootstrap 95%CI: {cb['convergence_rate_bootstrap_ci_95']}", file=sys.stderr)
    print(f"  Wilson    95%CI: {cb['convergence_rate_wilson_ci_95']}", file=sys.stderr)
    print(f"  Equiv 1st try  : {s['equivalent_on_first_try']}", file=sys.stderr)
    print(f"  Remained div.  : {s['remained_divergent']}", file=sys.stderr)
    print(f"  Errors         : {s['errors']}", file=sys.stderr)
    print(f"  Avg iterations : {s['avg_iterations']}", file=sys.stderr)
    print(f"  Avg time       : {s['avg_time_ms']:.0f} ms", file=sys.stderr)

    rw = output["real_world_benchmark"]
    print(f"\n  Real-world verification", file=sys.stderr)
    print(f"  Accuracy       : {rw['correct']}/{rw['total']}"
          f"  ({rw['accuracy']:.1%})", file=sys.stderr)
    print(f"  Wilson    95%CI: {rw['wilson_ci_95']}", file=sys.stderr)
    print(f"  Avg verify time: {rw['avg_time_ms']:.0f} ms", file=sys.stderr)
    print(sep, file=sys.stderr)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scaled CExL-Repair experiment runner."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        default=None,
        help="Simulate repairs via oracle (no LLM calls).",
    )
    mode.add_argument(
        "--live",
        action="store_true",
        help="Run live LLM repair (requires OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--pairs",
        type=int,
        default=None,
        metavar="N",
        help="Limit to first N CEGAR pairs.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        metavar="PATH",
        help=f"Output JSON path (default: {DEFAULT_OUTPUT}).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    # Resolve mode
    if args.live:
        dry_run = False
    elif args.dry_run:
        dry_run = True
    else:
        dry_run = not bool(os.environ.get("OPENAI_API_KEY"))
        if dry_run:
            print(
                "[INFO] No OPENAI_API_KEY found — running in DRY-RUN mode.",
                file=sys.stderr,
            )

    # Select pairs
    cegar_pairs = list(CEGAR_BENCHMARK_PAIRS)
    if args.pairs is not None:
        cegar_pairs = cegar_pairs[: args.pairs]

    realworld_pairs = list(REAL_WORLD_BENCHMARK_PAIRS)

    # ---- Run experiments ----
    cegar_results, cegar_analysis = run_cegar_experiment(
        cegar_pairs, dry_run=dry_run, model=DEFAULT_MODEL
    )

    realworld_summary = run_realworld_verification(realworld_pairs)

    # ---- Assemble output ----
    output = build_output(
        cegar_results,
        cegar_analysis,
        cegar_pairs,
        realworld_summary,
        dry_run=dry_run,
        model=DEFAULT_MODEL,
    )

    # ---- Write JSON ----
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results written to {args.output}", file=sys.stderr)

    _print_summary(output)

    # Also emit JSON to stdout for piping
    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
