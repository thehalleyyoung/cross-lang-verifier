#!/usr/bin/env python3
"""
Statistical analysis module for SemRec benchmark results.

Provides:
  - Bootstrap confidence intervals for accuracy metrics
  - FP/FN decomposition (false equivalence vs false divergence)
  - Stratified per-category performance reporting
  - K-sensitivity analysis for bounded unrolling
  - Confusion matrix generation

All statistics are computed from actual verification results, not hallucinated.
"""

import json
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any


@dataclass
class ConfusionMatrix:
    """2x2 confusion matrix for equivalence verification."""
    true_positive: int = 0   # Correctly identified as divergent
    true_negative: int = 0   # Correctly identified as equivalent
    false_positive: int = 0  # False divergence (said divergent, actually equivalent)
    false_negative: int = 0  # False equivalence (said equivalent, actually divergent) — DANGEROUS

    @property
    def total(self) -> int:
        return self.true_positive + self.true_negative + self.false_positive + self.false_negative

    @property
    def accuracy(self) -> float:
        return (self.true_positive + self.true_negative) / max(self.total, 1)

    @property
    def precision(self) -> float:
        """Of those we called divergent, how many actually are?"""
        denom = self.true_positive + self.false_positive
        return self.true_positive / max(denom, 1)

    @property
    def recall(self) -> float:
        """Of those that are actually divergent, how many did we find?"""
        denom = self.true_positive + self.false_negative
        return self.true_positive / max(denom, 1)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / max(p + r, 1e-10)

    @property
    def false_positive_rate(self) -> float:
        """Rate of incorrectly reporting divergence."""
        return self.false_positive / max(self.total, 1)

    @property
    def false_negative_rate(self) -> float:
        """Rate of incorrectly reporting equivalence — the DANGEROUS direction."""
        return self.false_negative / max(self.total, 1)

    def to_dict(self) -> dict:
        return {
            "true_positive": self.true_positive,
            "true_negative": self.true_negative,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "false_positive_rate": round(self.false_positive_rate, 4),
            "false_negative_rate": round(self.false_negative_rate, 4),
        }


def classify_verdict_match(actual: str, expected: str) -> str:
    """Classify a verdict against ground truth.

    Returns one of: TP, TN, FP, FN, unknown, error
    """
    actual_l = actual.lower()
    expected_l = expected.lower()

    # Map expected categories
    expected_is_equiv = "equivalent" in expected_l
    expected_is_div = "divergent" in expected_l

    actual_is_equiv = actual_l == "equivalent"
    actual_is_div = "divergent" in actual_l

    if actual_l in ("unknown", "error", "pipeline_fail"):
        return "unknown"

    if expected_is_div and actual_is_div:
        return "TP"  # Correctly found divergence
    if expected_is_equiv and actual_is_equiv:
        return "TN"  # Correctly confirmed equivalence
    if expected_is_equiv and actual_is_div:
        return "FP"  # False alarm: said divergent but actually equivalent
    if expected_is_div and actual_is_equiv:
        return "FN"  # Missed divergence: said equivalent but actually divergent (DANGEROUS)

    # Handle "divergent_on_overflow" expected with "equivalent" actual
    # This is an FN if the expected is divergent and we said equivalent
    if "overflow" in expected_l and actual_is_equiv:
        return "FN"
    if "overflow" in expected_l and actual_is_div:
        return "TP"

    return "unknown"


def build_confusion_matrix(results: List[dict]) -> ConfusionMatrix:
    """Build confusion matrix from verification results."""
    cm = ConfusionMatrix()
    for r in results:
        classification = classify_verdict_match(
            r.get("verdict", "unknown"),
            r.get("expected_verdict", "unknown")
        )
        if classification == "TP":
            cm.true_positive += 1
        elif classification == "TN":
            cm.true_negative += 1
        elif classification == "FP":
            cm.false_positive += 1
        elif classification == "FN":
            cm.false_negative += 1
    return cm


def bootstrap_ci(values: List[float], n_bootstrap: int = 10000,
                 confidence: float = 0.95, seed: int = 42) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval for a metric.

    Returns (mean, lower_bound, upper_bound).
    """
    if not values:
        return (0.0, 0.0, 0.0)

    rng = random.Random(seed)
    n = len(values)
    boot_means = []

    for _ in range(n_bootstrap):
        sample = [rng.choice(values) for _ in range(n)]
        boot_means.append(sum(sample) / n)

    boot_means.sort()
    alpha = (1 - confidence) / 2
    lo_idx = int(alpha * n_bootstrap)
    hi_idx = int((1 - alpha) * n_bootstrap) - 1
    lo_idx = max(0, min(lo_idx, n_bootstrap - 1))
    hi_idx = max(0, min(hi_idx, n_bootstrap - 1))

    mean = sum(values) / n
    return (round(mean, 4), round(boot_means[lo_idx], 4), round(boot_means[hi_idx], 4))


def stratified_report(results: List[dict]) -> Dict[str, dict]:
    """Generate per-category accuracy breakdown with CIs.

    Returns dict mapping category → {total, accuracy, ci_lower, ci_upper, cm}.
    """
    by_category = defaultdict(list)
    for r in results:
        cat = r.get("category", "unknown")
        by_category[cat].append(r)

    report = {}
    for cat, cat_results in sorted(by_category.items()):
        cm = build_confusion_matrix(cat_results)
        correct_flags = []
        for r in cat_results:
            cls = classify_verdict_match(
                r.get("verdict", "unknown"),
                r.get("expected_verdict", "unknown")
            )
            if cls in ("TP", "TN"):
                correct_flags.append(1.0)
            elif cls in ("FP", "FN"):
                correct_flags.append(0.0)
            # Skip unknowns

        if correct_flags:
            mean, lo, hi = bootstrap_ci(correct_flags)
        else:
            mean, lo, hi = (0.0, 0.0, 0.0)

        report[cat] = {
            "total": len(cat_results),
            "accuracy": mean,
            "ci_95_lower": lo,
            "ci_95_upper": hi,
            "confusion_matrix": cm.to_dict(),
        }

    return report


def fp_fn_decomposition(results: List[dict]) -> dict:
    """Decompose error rate into false positive and false negative components.

    False Positive (FP): Tool says divergent, truth is equivalent.
      → Conservative error; wastes developer time but safe.

    False Negative (FN): Tool says equivalent, truth is divergent.
      → DANGEROUS error; allows buggy translation to pass.
    """
    cm = build_confusion_matrix(results)
    total_errors = cm.false_positive + cm.false_negative

    return {
        "total_pairs": cm.total,
        "total_errors": total_errors,
        "error_rate": round(total_errors / max(cm.total, 1), 4),
        "false_positive_count": cm.false_positive,
        "false_positive_rate": round(cm.false_positive / max(cm.total, 1), 4),
        "false_positive_description": "Said divergent, actually equivalent (conservative, safe)",
        "false_negative_count": cm.false_negative,
        "false_negative_rate": round(cm.false_negative / max(cm.total, 1), 4),
        "false_negative_description": "Said equivalent, actually divergent (DANGEROUS)",
        "safety_assessment": (
            "SAFE: All errors are conservative (false positives)"
            if cm.false_negative == 0
            else f"WARNING: {cm.false_negative} false negative(s) — tool missed real divergences"
        ),
        "confusion_matrix": cm.to_dict(),
    }


def k_sensitivity_analysis(results_by_k: Dict[int, List[dict]]) -> dict:
    """Analyze sensitivity to bounded unrolling parameter K.

    Input: dict mapping K value → list of verification results at that K.
    """
    analysis = {}
    for k, results in sorted(results_by_k.items()):
        cm = build_confusion_matrix(results)
        correct_flags = []
        for r in results:
            cls = classify_verdict_match(
                r.get("verdict", "unknown"),
                r.get("expected_verdict", "unknown")
            )
            if cls in ("TP", "TN"):
                correct_flags.append(1.0)
            elif cls in ("FP", "FN"):
                correct_flags.append(0.0)

        if correct_flags:
            mean, lo, hi = bootstrap_ci(correct_flags)
        else:
            mean, lo, hi = (0.0, 0.0, 0.0)

        unknowns = sum(1 for r in results if r.get("verdict") in ("unknown", "error", "pipeline_fail"))
        definitive = len(results) - unknowns

        analysis[str(k)] = {
            "K": k,
            "total_pairs": len(results),
            "accuracy": mean,
            "ci_95_lower": lo,
            "ci_95_upper": hi,
            "definitive_verdicts": definitive,
            "definitive_rate": round(definitive / max(len(results), 1), 4),
            "unknown_count": unknowns,
            "confusion_matrix": cm.to_dict(),
        }

    return analysis


def compute_full_statistics(results: List[dict]) -> dict:
    """Compute comprehensive statistical analysis of verification results.

    Returns a dict with all statistical measures.
    """
    # Overall confusion matrix
    cm = build_confusion_matrix(results)

    # Overall accuracy with bootstrap CI
    correct_flags = []
    for r in results:
        cls = classify_verdict_match(
            r.get("verdict", "unknown"),
            r.get("expected_verdict", "unknown")
        )
        if cls in ("TP", "TN"):
            correct_flags.append(1.0)
        elif cls in ("FP", "FN"):
            correct_flags.append(0.0)

    if correct_flags:
        acc_mean, acc_lo, acc_hi = bootstrap_ci(correct_flags)
    else:
        acc_mean, acc_lo, acc_hi = (0.0, 0.0, 0.0)

    # Timing CI
    times = [r.get("time_ms", 0) for r in results if r.get("time_ms", 0) > 0]
    if times:
        time_mean, time_lo, time_hi = bootstrap_ci(times)
    else:
        time_mean, time_lo, time_hi = (0.0, 0.0, 0.0)

    return {
        "n_pairs": len(results),
        "overall_accuracy": {
            "mean": acc_mean,
            "ci_95_lower": acc_lo,
            "ci_95_upper": acc_hi,
            "n_evaluated": len(correct_flags),
        },
        "confusion_matrix": cm.to_dict(),
        "fp_fn_decomposition": fp_fn_decomposition(results),
        "stratified_by_category": stratified_report(results),
        "timing_ms": {
            "mean": time_mean,
            "ci_95_lower": time_lo,
            "ci_95_upper": time_hi,
        },
    }


def print_statistical_report(stats: dict) -> None:
    """Pretty-print the statistical analysis."""
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS REPORT")
    print("=" * 70)

    acc = stats["overall_accuracy"]
    print(f"\n  Overall Accuracy: {acc['mean']*100:.1f}% "
          f"(95% CI: [{acc['ci_95_lower']*100:.1f}%, {acc['ci_95_upper']*100:.1f}%])")
    print(f"  Evaluated: {acc['n_evaluated']} / {stats['n_pairs']} pairs")

    cm = stats["confusion_matrix"]
    print(f"\n  Confusion Matrix:")
    print(f"    {'':20s} Predicted Equiv  Predicted Div")
    print(f"    {'Actually Equiv':20s} TN={cm['true_negative']:4d}         FP={cm['false_positive']:4d}")
    print(f"    {'Actually Div':20s} FN={cm['false_negative']:4d}         TP={cm['true_positive']:4d}")

    fpfn = stats["fp_fn_decomposition"]
    print(f"\n  FP/FN Decomposition:")
    print(f"    Total errors: {fpfn['total_errors']} ({fpfn['error_rate']*100:.1f}%)")
    print(f"    False Positives: {fpfn['false_positive_count']} ({fpfn['false_positive_rate']*100:.1f}%) — {fpfn['false_positive_description']}")
    print(f"    False Negatives: {fpfn['false_negative_count']} ({fpfn['false_negative_rate']*100:.1f}%) — {fpfn['false_negative_description']}")
    print(f"    Safety: {fpfn['safety_assessment']}")

    print(f"\n  Precision: {cm['precision']*100:.1f}%")
    print(f"  Recall:    {cm['recall']*100:.1f}%")
    print(f"  F1 Score:  {cm['f1']*100:.1f}%")

    strat = stats.get("stratified_by_category", {})
    if strat:
        print(f"\n  Stratified Per-Category Performance:")
        print(f"    {'Category':20s} {'N':>4s} {'Acc':>8s} {'95% CI':>20s} {'TP':>4s} {'TN':>4s} {'FP':>4s} {'FN':>4s}")
        print(f"    {'-'*20} {'----':>4s} {'--------':>8s} {'--------------------':>20s} {'----':>4s} {'----':>4s} {'----':>4s} {'----':>4s}")
        for cat, data in sorted(strat.items()):
            ci_str = f"[{data['ci_95_lower']*100:.1f}%, {data['ci_95_upper']*100:.1f}%]"
            ccm = data["confusion_matrix"]
            print(f"    {cat:20s} {data['total']:4d} {data['accuracy']*100:7.1f}% {ci_str:>20s} "
                  f"{ccm['true_positive']:4d} {ccm['true_negative']:4d} {ccm['false_positive']:4d} {ccm['false_negative']:4d}")

    timing = stats.get("timing_ms", {})
    if timing:
        print(f"\n  Timing: {timing['mean']:.1f}ms mean "
              f"(95% CI: [{timing['ci_95_lower']:.1f}ms, {timing['ci_95_upper']:.1f}ms])")

    print("=" * 70)


if __name__ == "__main__":
    # Load existing results and compute statistics
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    results_file = os.path.join(results_dir, "verification_results.json")

    if os.path.exists(results_file):
        with open(results_file) as f:
            results = json.load(f)
        stats = compute_full_statistics(results)
        print_statistical_report(stats)

        # Save
        out_file = os.path.join(results_dir, "statistical_analysis.json")
        with open(out_file, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\nSaved to {out_file}")
    else:
        print(f"No results file found at {results_file}")
        print("Run run_experiments.py first.")
