#!/usr/bin/env python3
"""
SemRec Scaled CEGAR Evaluation Framework.

The central experiment: systematically evaluate formal-methods-guided LLM
code repair across 215 real C functions, comparing:
  1. CEGAR (counterexample-guided repair) vs
  2. Naive retry (re-translate without feedback) vs
  3. Spec-only (tell LLM the semantic rules without counterexample)

This creates the first systematic study of formal-methods-guided LLM code
repair for cross-language translation.

Outputs:
  - Per-function results with bug taxonomy
  - Convergence curves (success rate vs iteration)
  - Bug class repairability analysis
  - Statistical analysis with bootstrap confidence intervals
"""

import json
import os
import sys
import time
import random
import traceback
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple

# Setup paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from real_benchmarks.benchmark_suite import ALL_BENCHMARKS, Benchmark, suite_summary

# Import oracle
from src.oracle.oracle import (
    VerificationOracle, OracleResult, classify_divergence,
    generate_repair_hint, Verdict,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# LLM interface
# ---------------------------------------------------------------------------

def get_llm_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    from openai import OpenAI
    return OpenAI()


def llm_translate(client, c_code: str, func_name: str,
                  model: str = "gpt-4.1-nano") -> str:
    """Initial LLM translation of C to Rust."""
    prompt = f"""Translate this C function to Rust. Return ONLY the Rust function, no explanation.
Use idiomatic Rust. The function should have the same semantics as the C version.
Handle edge cases correctly (overflow, division by zero, etc.).

C function:
```c
{c_code}
```

Rust function:"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=512,
    )
    return _strip_code_fences(response.choices[0].message.content.strip())


def llm_repair_cegar(client, c_code: str, rust_code: str,
                     counterexample: Dict, repair_hint: Dict,
                     func_name: str, iteration: int,
                     model: str = "gpt-4.1-nano") -> str:
    """CEGAR repair: feed counterexample + repair hint to LLM."""
    cex_inputs = {k: v for k, v in counterexample.get("inputs", {}).items()}
    reason = counterexample.get("reason", "semantic divergence")
    hint = repair_hint.get("suggested_fix", "") if repair_hint else ""

    prompt = f"""The following Rust translation of a C function has a semantic bug.

## C function (ground truth):
```c
{c_code}
```

## Current Rust translation (BUGGY):
```rust
{rust_code}
```

## Bug found by formal verification:
- Counterexample input: {json.dumps(cex_inputs, default=str)}
- Divergence reason: {reason}
- Repair hint: {hint}

## Key C-to-Rust semantic differences:
- C signed overflow is UB; use wrapping_add/wrapping_sub/wrapping_mul in Rust
- C division by zero is UB; Rust panics
- C shift by >= width is UB; Rust wraps shift amount or panics
- C negation of INT_MIN is UB; use wrapping_neg() in Rust
- C signed integer division truncates toward zero

Return ONLY the fixed Rust function, no explanation."""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=512,
    )
    return _strip_code_fences(response.choices[0].message.content.strip())


def llm_repair_naive(client, c_code: str, func_name: str,
                     model: str = "gpt-4.1-nano") -> str:
    """Naive retry: just re-translate without any feedback."""
    return llm_translate(client, c_code, func_name, model)


def llm_repair_spec_only(client, c_code: str, rust_code: str,
                         func_name: str,
                         model: str = "gpt-4.1-nano") -> str:
    """Spec-only repair: tell LLM the rules but no counterexample."""
    prompt = f"""The following Rust translation of a C function may have semantic bugs.

## C function (ground truth):
```c
{c_code}
```

## Current Rust translation (may be buggy):
```rust
{rust_code}
```

## Key C-to-Rust semantic rules to check:
- C signed overflow is UB; use wrapping_add/wrapping_sub/wrapping_mul
- C division by zero is UB; Rust panics
- C shift by >= bit width is UB; mask shift amount in Rust
- C negation of INT_MIN is UB; use wrapping_neg()
- Ensure wrapping semantics for all arithmetic that could overflow

Fix any semantic bugs. Return ONLY the fixed Rust function, no explanation."""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=512,
    )
    return _strip_code_fences(response.choices[0].message.content.strip())


def _strip_code_fences(code: str) -> str:
    if code.startswith("```"):
        lines = code.split('\n')
        code = '\n'.join(lines[1:])
        if code.endswith("```"):
            code = code[:-3].strip()
    return code


# ---------------------------------------------------------------------------
# CEGAR loop
# ---------------------------------------------------------------------------

@dataclass
class CEGARTrialResult:
    func_name: str
    category: str
    source: str
    method: str  # "cegar", "naive", "spec_only"
    success: bool
    iterations: int
    max_iterations: int
    bugs_found: List[str]
    bug_classes: List[str]
    counterexamples: List[Dict]
    repair_hints: List[Dict]
    initial_verdict: str
    final_verdict: str
    total_time_ms: float
    verify_time_ms: float
    llm_time_ms: float
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def run_cegar_trial(oracle: VerificationOracle, client, benchmark: Benchmark,
                    max_iterations: int = 5,
                    model: str = "gpt-4.1-nano") -> CEGARTrialResult:
    """Run one CEGAR trial for a single benchmark function."""
    name = benchmark.name
    c_code = benchmark.code
    total_start = time.time()
    verify_time = 0.0
    llm_time = 0.0

    # Step 1: Initial translation
    t0 = time.time()
    try:
        rust_code = llm_translate(client, c_code, name, model)
    except Exception as e:
        return CEGARTrialResult(
            func_name=name, category=benchmark.category, source=benchmark.source,
            method="cegar", success=False, iterations=0,
            max_iterations=max_iterations, bugs_found=[], bug_classes=[],
            counterexamples=[], repair_hints=[],
            initial_verdict="error", final_verdict="error",
            total_time_ms=(time.time() - total_start) * 1000,
            verify_time_ms=0, llm_time_ms=0, error=str(e),
        )
    llm_time += (time.time() - t0) * 1000

    all_bugs = []
    all_bug_classes = []
    all_cex = []
    all_hints = []
    initial_verdict = ""

    for iteration in range(max_iterations):
        # Step 2: Verify
        t0 = time.time()
        result = oracle.verify(c_code, rust_code, name)
        verify_time += (time.time() - t0) * 1000

        if iteration == 0:
            initial_verdict = result.verdict

        if result.verdict == Verdict.EQUIVALENT.value:
            return CEGARTrialResult(
                func_name=name, category=benchmark.category, source=benchmark.source,
                method="cegar", success=True, iterations=iteration + 1,
                max_iterations=max_iterations, bugs_found=all_bugs,
                bug_classes=all_bug_classes, counterexamples=all_cex,
                repair_hints=all_hints,
                initial_verdict=initial_verdict, final_verdict="equivalent",
                total_time_ms=(time.time() - total_start) * 1000,
                verify_time_ms=verify_time, llm_time_ms=llm_time,
            )

        if result.verdict == Verdict.DIVERGENT.value and result.counterexample:
            reason = result.counterexample.reason
            all_bugs.append(reason)
            all_bug_classes.append(result.counterexample.divergence_class)
            all_cex.append(result.counterexample.to_dict())
            if result.repair_hint:
                all_hints.append(result.repair_hint.to_dict())

            # Step 3: CEGAR repair
            t0 = time.time()
            try:
                rust_code = llm_repair_cegar(
                    client, c_code, rust_code,
                    result.counterexample.to_dict(),
                    result.repair_hint.to_dict() if result.repair_hint else {},
                    name, iteration, model,
                )
            except Exception as e:
                return CEGARTrialResult(
                    func_name=name, category=benchmark.category, source=benchmark.source,
                    method="cegar", success=False, iterations=iteration + 1,
                    max_iterations=max_iterations, bugs_found=all_bugs,
                    bug_classes=all_bug_classes, counterexamples=all_cex,
                    repair_hints=all_hints,
                    initial_verdict=initial_verdict, final_verdict="error",
                    total_time_ms=(time.time() - total_start) * 1000,
                    verify_time_ms=verify_time, llm_time_ms=llm_time,
                    error=str(e),
                )
            llm_time += (time.time() - t0) * 1000
        else:
            break

    # Final verification
    t0 = time.time()
    final = oracle.verify(c_code, rust_code, name)
    verify_time += (time.time() - t0) * 1000

    return CEGARTrialResult(
        func_name=name, category=benchmark.category, source=benchmark.source,
        method="cegar", success=(final.verdict == Verdict.EQUIVALENT.value),
        iterations=min(len(all_bugs) + 1, max_iterations),
        max_iterations=max_iterations, bugs_found=all_bugs,
        bug_classes=all_bug_classes, counterexamples=all_cex,
        repair_hints=all_hints,
        initial_verdict=initial_verdict, final_verdict=final.verdict,
        total_time_ms=(time.time() - total_start) * 1000,
        verify_time_ms=verify_time, llm_time_ms=llm_time,
    )


def run_naive_trial(oracle: VerificationOracle, client, benchmark: Benchmark,
                    max_iterations: int = 5,
                    model: str = "gpt-4.1-nano") -> CEGARTrialResult:
    """Run naive retry baseline (re-translate without feedback)."""
    name = benchmark.name
    c_code = benchmark.code
    total_start = time.time()
    verify_time = 0.0
    llm_time = 0.0
    initial_verdict = ""

    for iteration in range(max_iterations):
        t0 = time.time()
        try:
            rust_code = llm_translate(client, c_code, name, model)
        except Exception as e:
            return CEGARTrialResult(
                func_name=name, category=benchmark.category, source=benchmark.source,
                method="naive", success=False, iterations=iteration + 1,
                max_iterations=max_iterations, bugs_found=[], bug_classes=[],
                counterexamples=[], repair_hints=[],
                initial_verdict=initial_verdict, final_verdict="error",
                total_time_ms=(time.time() - total_start) * 1000,
                verify_time_ms=verify_time, llm_time_ms=llm_time, error=str(e),
            )
        llm_time += (time.time() - t0) * 1000

        t0 = time.time()
        result = oracle.verify(c_code, rust_code, name)
        verify_time += (time.time() - t0) * 1000

        if iteration == 0:
            initial_verdict = result.verdict

        if result.verdict == Verdict.EQUIVALENT.value:
            return CEGARTrialResult(
                func_name=name, category=benchmark.category, source=benchmark.source,
                method="naive", success=True, iterations=iteration + 1,
                max_iterations=max_iterations, bugs_found=[], bug_classes=[],
                counterexamples=[], repair_hints=[],
                initial_verdict=initial_verdict, final_verdict="equivalent",
                total_time_ms=(time.time() - total_start) * 1000,
                verify_time_ms=verify_time, llm_time_ms=llm_time,
            )

    return CEGARTrialResult(
        func_name=name, category=benchmark.category, source=benchmark.source,
        method="naive", success=False, iterations=max_iterations,
        max_iterations=max_iterations, bugs_found=[], bug_classes=[],
        counterexamples=[], repair_hints=[],
        initial_verdict=initial_verdict, final_verdict=result.verdict,
        total_time_ms=(time.time() - total_start) * 1000,
        verify_time_ms=verify_time, llm_time_ms=llm_time,
    )


def run_spec_only_trial(oracle: VerificationOracle, client, benchmark: Benchmark,
                        max_iterations: int = 5,
                        model: str = "gpt-4.1-nano") -> CEGARTrialResult:
    """Run spec-only baseline (semantic rules but no counterexample)."""
    name = benchmark.name
    c_code = benchmark.code
    total_start = time.time()
    verify_time = 0.0
    llm_time = 0.0

    # Initial translation
    t0 = time.time()
    try:
        rust_code = llm_translate(client, c_code, name, model)
    except Exception as e:
        return CEGARTrialResult(
            func_name=name, category=benchmark.category, source=benchmark.source,
            method="spec_only", success=False, iterations=0,
            max_iterations=max_iterations, bugs_found=[], bug_classes=[],
            counterexamples=[], repair_hints=[],
            initial_verdict="error", final_verdict="error",
            total_time_ms=(time.time() - total_start) * 1000,
            verify_time_ms=0, llm_time_ms=0, error=str(e),
        )
    llm_time += (time.time() - t0) * 1000
    initial_verdict = ""

    for iteration in range(max_iterations):
        t0 = time.time()
        result = oracle.verify(c_code, rust_code, name)
        verify_time += (time.time() - t0) * 1000

        if iteration == 0:
            initial_verdict = result.verdict

        if result.verdict == Verdict.EQUIVALENT.value:
            return CEGARTrialResult(
                func_name=name, category=benchmark.category, source=benchmark.source,
                method="spec_only", success=True, iterations=iteration + 1,
                max_iterations=max_iterations, bugs_found=[], bug_classes=[],
                counterexamples=[], repair_hints=[],
                initial_verdict=initial_verdict, final_verdict="equivalent",
                total_time_ms=(time.time() - total_start) * 1000,
                verify_time_ms=verify_time, llm_time_ms=llm_time,
            )

        # Spec-only repair (no counterexample, just rules)
        t0 = time.time()
        try:
            rust_code = llm_repair_spec_only(client, c_code, rust_code, name, model)
        except Exception as e:
            break
        llm_time += (time.time() - t0) * 1000

    # Final check
    t0 = time.time()
    final = oracle.verify(c_code, rust_code, name)
    verify_time += (time.time() - t0) * 1000

    return CEGARTrialResult(
        func_name=name, category=benchmark.category, source=benchmark.source,
        method="spec_only", success=(final.verdict == Verdict.EQUIVALENT.value),
        iterations=max_iterations,
        max_iterations=max_iterations, bugs_found=[], bug_classes=[],
        counterexamples=[], repair_hints=[],
        initial_verdict=initial_verdict, final_verdict=final.verdict,
        total_time_ms=(time.time() - total_start) * 1000,
        verify_time_ms=verify_time, llm_time_ms=llm_time,
    )


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------

def bootstrap_ci(data: List[float], n_bootstrap: int = 1000,
                 ci: float = 0.95) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval."""
    if not data:
        return (0.0, 0.0, 0.0)
    n = len(data)
    means = []
    for _ in range(n_bootstrap):
        sample = [data[random.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / len(sample))
    means.sort()
    alpha = (1 - ci) / 2
    lo = means[int(alpha * n_bootstrap)]
    hi = means[int((1 - alpha) * n_bootstrap)]
    mean = sum(data) / n
    return (mean, lo, hi)


def compute_statistics(results: List[CEGARTrialResult]) -> Dict[str, Any]:
    """Compute comprehensive statistics from trial results."""
    if not results:
        return {}

    total = len(results)
    successes = [1.0 if r.success else 0.0 for r in results]
    mean, lo, hi = bootstrap_ci(successes)

    # By category
    by_cat = {}
    for r in results:
        cat = r.category
        if cat not in by_cat:
            by_cat[cat] = {"total": 0, "success": 0, "bugs": []}
        by_cat[cat]["total"] += 1
        if r.success:
            by_cat[cat]["success"] += 1
        by_cat[cat]["bugs"].extend(r.bug_classes)

    for cat in by_cat:
        n = by_cat[cat]["total"]
        s = by_cat[cat]["success"]
        by_cat[cat]["rate"] = s / n if n > 0 else 0
        by_cat[cat]["bugs"] = dict(
            sorted(
                {b: by_cat[cat]["bugs"].count(b) for b in set(by_cat[cat]["bugs"])}.items(),
                key=lambda x: -x[1]
            )
        )

    # Bug class repairability
    bug_repair = {}
    for r in results:
        for bc in r.bug_classes:
            if bc not in bug_repair:
                bug_repair[bc] = {"found": 0, "repaired": 0}
            bug_repair[bc]["found"] += 1
        if r.success:
            for bc in set(r.bug_classes):
                if bc in bug_repair:
                    bug_repair[bc]["repaired"] += 1

    for bc in bug_repair:
        f = bug_repair[bc]["found"]
        rp = bug_repair[bc]["repaired"]
        bug_repair[bc]["repair_rate"] = rp / f if f > 0 else 0

    # Convergence curve
    iter_counts = [r.iterations for r in results if r.success]
    convergence = {}
    for k in range(1, 6):
        n_converged = sum(1 for r in results if r.success and r.iterations <= k)
        convergence[k] = n_converged / total if total > 0 else 0

    # Timing
    verify_times = [r.verify_time_ms for r in results]
    llm_times = [r.llm_time_ms for r in results]

    return {
        "total": total,
        "success": sum(1 for r in results if r.success),
        "success_rate": mean,
        "success_rate_ci_95": [lo, hi],
        "by_category": by_cat,
        "bug_class_repairability": bug_repair,
        "convergence_curve": convergence,
        "avg_iterations_success": (sum(iter_counts) / len(iter_counts)) if iter_counts else 0,
        "avg_verify_time_ms": sum(verify_times) / len(verify_times) if verify_times else 0,
        "avg_llm_time_ms": sum(llm_times) / len(llm_times) if llm_times else 0,
        "total_time_ms": sum(r.total_time_ms for r in results),
    }


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_scaled_experiment(
    benchmarks: Optional[List[Benchmark]] = None,
    max_iterations: int = 5,
    model: str = "gpt-4.1-nano",
    run_baselines: bool = True,
    sample_size: Optional[int] = None,
    seed: int = 42,
):
    """Run the full scaled CEGAR evaluation experiment."""
    if benchmarks is None:
        benchmarks = ALL_BENCHMARKS
    if sample_size and sample_size < len(benchmarks):
        random.seed(seed)
        benchmarks = random.sample(benchmarks, sample_size)

    print("=" * 70)
    print("SemRec Scaled CEGAR Evaluation Framework")
    print("=" * 70)
    s = suite_summary()
    print(f"Benchmark suite: {len(benchmarks)} C functions")
    print(f"Model: {model}")
    print(f"Max CEGAR iterations: {max_iterations}")
    print(f"Baselines: {'CEGAR + Naive + Spec-only' if run_baselines else 'CEGAR only'}")
    print()

    client = get_llm_client()
    if client is None:
        print("ERROR: OPENAI_API_KEY not set. Cannot run LLM experiments.")
        return None

    oracle = VerificationOracle(timeout_ms=10000)

    # --- Run CEGAR trials ---
    print("\n--- CEGAR (counterexample-guided repair) ---")
    cegar_results = []
    for i, bench in enumerate(benchmarks):
        print(f"  [{i+1}/{len(benchmarks)}] {bench.name} ({bench.category})...", end="", flush=True)
        try:
            r = run_cegar_trial(oracle, client, bench, max_iterations, model)
            cegar_results.append(r)
            sym = "✓" if r.success else "✗"
            print(f" {sym} ({r.iterations} iter, {r.total_time_ms:.0f}ms)")
        except Exception as e:
            print(f" ERROR: {e}")
            cegar_results.append(CEGARTrialResult(
                func_name=bench.name, category=bench.category, source=bench.source,
                method="cegar", success=False, iterations=0,
                max_iterations=max_iterations, bugs_found=[], bug_classes=[],
                counterexamples=[], repair_hints=[],
                initial_verdict="error", final_verdict="error",
                total_time_ms=0, verify_time_ms=0, llm_time_ms=0,
                error=str(e),
            ))

    cegar_stats = compute_statistics(cegar_results)

    # --- Run baselines ---
    naive_stats = {}
    spec_stats = {}
    naive_results = []
    spec_results = []

    if run_baselines:
        # Only run baselines on a stratified subsample (40 functions) for cost efficiency
        random.seed(seed)
        cats = list(set(b.category for b in benchmarks))
        baseline_sample = []
        for cat in cats:
            cat_benchmarks = [b for b in benchmarks if b.category == cat]
            n_sample = max(1, min(len(cat_benchmarks), 3))
            baseline_sample.extend(random.sample(cat_benchmarks, min(n_sample, len(cat_benchmarks))))

        print(f"\n--- Naive retry baseline ({len(baseline_sample)} functions) ---")
        for i, bench in enumerate(baseline_sample):
            print(f"  [{i+1}/{len(baseline_sample)}] {bench.name}...", end="", flush=True)
            try:
                r = run_naive_trial(oracle, client, bench, max_iterations, model)
                naive_results.append(r)
                sym = "✓" if r.success else "✗"
                print(f" {sym} ({r.iterations} iter)")
            except Exception as e:
                print(f" ERROR: {e}")

        print(f"\n--- Spec-only baseline ({len(baseline_sample)} functions) ---")
        for i, bench in enumerate(baseline_sample):
            print(f"  [{i+1}/{len(baseline_sample)}] {bench.name}...", end="", flush=True)
            try:
                r = run_spec_only_trial(oracle, client, bench, max_iterations, model)
                spec_results.append(r)
                sym = "✓" if r.success else "✗"
                print(f" {sym} ({r.iterations} iter)")
            except Exception as e:
                print(f" ERROR: {e}")

        naive_stats = compute_statistics(naive_results)
        spec_stats = compute_statistics(spec_results)

    # --- Summary ---
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nCEGAR: {cegar_stats['success']}/{cegar_stats['total']} "
          f"({cegar_stats['success_rate']:.1%}) "
          f"[95% CI: {cegar_stats['success_rate_ci_95'][0]:.1%}–{cegar_stats['success_rate_ci_95'][1]:.1%}]")
    if naive_stats:
        print(f"Naive: {naive_stats['success']}/{naive_stats['total']} "
              f"({naive_stats['success_rate']:.1%}) "
              f"[95% CI: {naive_stats['success_rate_ci_95'][0]:.1%}–{naive_stats['success_rate_ci_95'][1]:.1%}]")
    if spec_stats:
        print(f"Spec:  {spec_stats['success']}/{spec_stats['total']} "
              f"({spec_stats['success_rate']:.1%}) "
              f"[95% CI: {spec_stats['success_rate_ci_95'][0]:.1%}–{spec_stats['success_rate_ci_95'][1]:.1%}]")

    print(f"\nConvergence curve (CEGAR):")
    for k, rate in cegar_stats.get("convergence_curve", {}).items():
        print(f"  ≤{k} iterations: {rate:.1%}")

    print(f"\nBug class repairability:")
    for bc, info in sorted(cegar_stats.get("bug_class_repairability", {}).items(),
                           key=lambda x: -x[1]["found"]):
        print(f"  {bc:25s}: {info['repaired']}/{info['found']} repaired "
              f"({info['repair_rate']:.0%})")

    # --- Save results ---
    output = {
        "experiment": "scaled_cegar_evaluation",
        "model": model,
        "max_iterations": max_iterations,
        "benchmark_suite_size": len(benchmarks),
        "cegar": {
            "statistics": cegar_stats,
            "results": [r.to_dict() for r in cegar_results],
        },
    }
    if naive_stats:
        output["naive_retry"] = {
            "statistics": naive_stats,
            "results": [r.to_dict() for r in naive_results],
        }
    if spec_stats:
        output["spec_only"] = {
            "statistics": spec_stats,
            "results": [r.to_dict() for r in spec_results],
        }

    outpath = os.path.join(RESULTS_DIR, "scaled_cegar_results.json")
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nFull results saved to {outpath}")

    return output


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SemRec Scaled CEGAR Evaluation")
    parser.add_argument("--sample", type=int, default=None,
                        help="Run on a random subsample of N benchmarks")
    parser.add_argument("--no-baselines", action="store_true",
                        help="Skip baseline comparisons")
    parser.add_argument("--model", default="gpt-4.1-nano",
                        help="LLM model to use")
    parser.add_argument("--max-iter", type=int, default=5,
                        help="Maximum CEGAR iterations")
    args = parser.parse_args()

    run_scaled_experiment(
        sample_size=args.sample,
        run_baselines=not args.no_baselines,
        model=args.model,
        max_iterations=args.max_iter,
    )
