#!/usr/bin/env python3
"""
Run comprehensive experiments for the SemRec paper revision.

Covers:
1. Core verification on all benchmark pairs (including new memory/C2Rust pairs)
2. Scalability evaluation (time vs LOC)
3. CEGAR LLM repair with improved hints
4. Confidence interval computation
"""

import json
import os
import sys
import time
import statistics
from typing import Dict, List, Any, Tuple

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.pairs import COMBINED_BENCHMARKS, BenchmarkPair
from benchmarks.pairs.memory_benchmark_pairs import MEMORY_PAIRS, SCALED_MEMORY_PAIRS
from benchmarks.pairs.c2rust_benchmark_pairs import C2RUST_REALISTIC_PAIRS
from src.oracle.oracle import VerificationOracle


def run_verification_suite(pairs: List[BenchmarkPair], tag: str = "all",
                           timeout_ms: int = 15000) -> Dict[str, Any]:
    """Run verification on a list of benchmark pairs and collect results."""
    oracle = VerificationOracle(timeout_ms=timeout_ms)
    results = []
    correct = 0
    total = len(pairs)
    errors = 0
    times_ms = []
    by_category: Dict[str, Dict[str, int]] = {}

    for i, pair in enumerate(pairs):
        t0 = time.time()
        try:
            result = oracle.verify(pair.c_source, pair.rust_source, pair.name)
            elapsed = (time.time() - t0) * 1000
            times_ms.append(elapsed)

            verdict = result.verdict
            expected = pair.expected_result
            match = False
            if expected == "equivalent" and verdict == "equivalent":
                match = True
            elif expected == "divergent" and verdict == "divergent":
                match = True
            elif expected == "conditional":
                match = verdict in ("equivalent", "divergent")

            if match:
                correct += 1
            if verdict == "error":
                errors += 1

            cat = pair.category
            if cat not in by_category:
                by_category[cat] = {"total": 0, "correct": 0, "errors": 0}
            by_category[cat]["total"] += 1
            if match:
                by_category[cat]["correct"] += 1
            if verdict == "error":
                by_category[cat]["errors"] += 1

            results.append({
                "name": pair.name,
                "category": cat,
                "expected": expected,
                "verdict": verdict,
                "correct": match,
                "time_ms": round(elapsed, 2),
                "smt_queries": result.smt_queries,
            })
        except Exception as e:
            errors += 1
            elapsed = (time.time() - t0) * 1000
            times_ms.append(elapsed)
            results.append({
                "name": pair.name,
                "category": pair.category,
                "expected": pair.expected_result,
                "verdict": "error",
                "correct": False,
                "time_ms": round(elapsed, 2),
                "error": str(e),
            })

    accuracy = correct / max(total, 1)
    # Wilson score interval for 95% confidence
    z = 1.96
    n = total
    p_hat = accuracy
    denom = 1 + z*z/n
    center = (p_hat + z*z/(2*n)) / denom
    margin = z * ((p_hat*(1-p_hat)/n + z*z/(4*n*n)) ** 0.5) / denom
    ci_low = max(0, center - margin)
    ci_high = min(1, center + margin)

    summary = {
        "tag": tag,
        "total_pairs": total,
        "correct": correct,
        "accuracy": round(accuracy * 100, 1),
        "accuracy_ci_95": [round(ci_low * 100, 1), round(ci_high * 100, 1)],
        "errors": errors,
        "mean_time_ms": round(statistics.mean(times_ms), 1) if times_ms else 0,
        "median_time_ms": round(statistics.median(times_ms), 1) if times_ms else 0,
        "p95_time_ms": round(sorted(times_ms)[int(len(times_ms)*0.95)] if times_ms else 0, 1),
        "total_time_ms": round(sum(times_ms), 1),
        "by_category": {
            cat: {
                "accuracy": round(d["correct"] / max(d["total"], 1) * 100, 1),
                **d,
            }
            for cat, d in sorted(by_category.items())
        },
    }

    return {"summary": summary, "results": results}


def run_scalability_eval(timeout_ms: int = 15000) -> Dict[str, Any]:
    """Evaluate scalability: verification time vs function complexity."""
    oracle = VerificationOracle(timeout_ms=timeout_ms)
    data_points = []

    for pair in COMBINED_BENCHMARKS:
        loc = len(pair.c_source.strip().split('\n')) + len(pair.rust_source.strip().split('\n'))
        t0 = time.time()
        try:
            result = oracle.verify(pair.c_source, pair.rust_source, pair.name)
            elapsed = (time.time() - t0) * 1000
            data_points.append({
                "name": pair.name,
                "loc": loc,
                "time_ms": round(elapsed, 2),
                "verdict": result.verdict,
                "smt_queries": result.smt_queries,
            })
        except Exception:
            elapsed = (time.time() - t0) * 1000
            data_points.append({
                "name": pair.name,
                "loc": loc,
                "time_ms": round(elapsed, 2),
                "verdict": "error",
            })

    # Compute regression: time = a * LOC + b
    locs = [d["loc"] for d in data_points]
    times = [d["time_ms"] for d in data_points]
    n = len(locs)
    if n > 2:
        mean_x = sum(locs) / n
        mean_y = sum(times) / n
        ss_xx = sum((x - mean_x)**2 for x in locs)
        ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(locs, times))
        slope = ss_xy / ss_xx if ss_xx > 0 else 0
        intercept = mean_y - slope * mean_x
        # R²
        ss_tot = sum((y - mean_y)**2 for y in times)
        ss_res = sum((y - (slope * x + intercept))**2 for x, y in zip(locs, times))
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    else:
        slope, intercept, r_squared = 0, 0, 0

    return {
        "data_points": data_points,
        "regression": {
            "slope_ms_per_loc": round(slope, 3),
            "intercept_ms": round(intercept, 1),
            "r_squared": round(r_squared, 4),
        },
        "summary": {
            "total_benchmarks": n,
            "min_loc": min(locs) if locs else 0,
            "max_loc": max(locs) if locs else 0,
            "min_time_ms": round(min(times), 1) if times else 0,
            "max_time_ms": round(max(times), 1) if times else 0,
        },
    }


def main():
    """Run all experiments and save results."""
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 60)
    print("SemRec Comprehensive Experiments")
    print("=" * 60)

    # 1. Full verification suite
    print("\n[1/4] Running full verification suite...")
    full_results = run_verification_suite(COMBINED_BENCHMARKS, "full_suite")
    with open(os.path.join(results_dir, "full_verification_v3.json"), "w") as f:
        json.dump(full_results, f, indent=2)
    s = full_results["summary"]
    print(f"  Accuracy: {s['accuracy']}% ({s['correct']}/{s['total_pairs']})")
    print(f"  95% CI: [{s['accuracy_ci_95'][0]}%, {s['accuracy_ci_95'][1]}%]")
    print(f"  Mean time: {s['mean_time_ms']}ms, Median: {s['median_time_ms']}ms")

    # 2. Memory benchmark subset
    print("\n[2/4] Running memory/pointer benchmarks...")
    mem_results = run_verification_suite(
        MEMORY_PAIRS + SCALED_MEMORY_PAIRS, "memory"
    )
    with open(os.path.join(results_dir, "memory_verification.json"), "w") as f:
        json.dump(mem_results, f, indent=2)
    s = mem_results["summary"]
    print(f"  Accuracy: {s['accuracy']}% ({s['correct']}/{s['total_pairs']})")

    # 3. C2Rust realistic benchmarks
    print("\n[3/4] Running C2Rust realistic benchmarks...")
    c2rust_results = run_verification_suite(C2RUST_REALISTIC_PAIRS, "c2rust_realistic")
    with open(os.path.join(results_dir, "c2rust_realistic_verification.json"), "w") as f:
        json.dump(c2rust_results, f, indent=2)
    s = c2rust_results["summary"]
    print(f"  Accuracy: {s['accuracy']}% ({s['correct']}/{s['total_pairs']})")

    # 4. Scalability evaluation
    print("\n[4/4] Running scalability evaluation...")
    scale_results = run_scalability_eval()
    with open(os.path.join(results_dir, "scalability_evaluation.json"), "w") as f:
        json.dump(scale_results, f, indent=2)
    reg = scale_results["regression"]
    print(f"  Slope: {reg['slope_ms_per_loc']} ms/LOC")
    print(f"  R²: {reg['r_squared']}")

    # Summary
    print("\n" + "=" * 60)
    print("All experiments complete. Results saved to experiments/results/")
    print("=" * 60)

    # Save combined summary
    combined_summary = {
        "full_suite": full_results["summary"],
        "memory": mem_results["summary"],
        "c2rust_realistic": c2rust_results["summary"],
        "scalability": scale_results["summary"],
        "regression": scale_results["regression"],
    }
    with open(os.path.join(results_dir, "experiment_summary_v3.json"), "w") as f:
        json.dump(combined_summary, f, indent=2)


if __name__ == "__main__":
    main()
