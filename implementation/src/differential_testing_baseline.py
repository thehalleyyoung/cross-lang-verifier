#!/usr/bin/env python3
"""
Differential testing baseline for comparison with SemRec verification oracle.

Strategy: for each benchmark pair, use the same IR + SMT pipeline to build z3
expressions for the C and Rust functions, then evaluate on random concrete
inputs (substitution + simplify) and check whether outputs match.

Three budgets: 100, 1000, 10000 random inputs per function.
Edge cases (0, 1, -1, INT_MIN, INT_MAX, powers of 2) always included.

Key research question: can random testing find the divergences that the
SMT-based oracle finds?  Especially UB-related bugs that only trigger at
extreme values (INT_MIN, INT_MAX).
"""

import json
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import z3

from src.oracle.oracle import VerificationOracle
from benchmarks.pairs.benchmark_pairs import get_all_pairs, BenchmarkPair

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1
UINT32_MAX = 2**32 - 1

EDGE_CASES_I32 = [
    0, 1, -1, 2, -2,
    127, -128, 255, 256,
    2**15 - 1, -(2**15),       # INT16 boundaries
    2**16 - 1, 2**16,
    INT32_MAX, INT32_MIN,      # INT32 boundaries
    INT32_MAX - 1, INT32_MIN + 1,
    42, -42, 100, -100,
    1000, -1000, 10000,
    0x55555555, 0x7F7F7F7F,    # bit patterns
]

RANDOM_BUDGETS = [100, 1000, 10000]


def random_i32():
    return random.randint(INT32_MIN, INT32_MAX)


# ---------------------------------------------------------------------------
# Core: build z3 expressions and evaluate on concrete inputs
# ---------------------------------------------------------------------------

def build_z3_pair(c_code: str, rust_code: str, name: str):
    """Parse both functions through the SemRec pipeline and return
    (shared_vars, c_ret_expr, r_ret_expr, c_ctx, r_ctx) or None on failure."""
    from src.frontend_c.parser import CParser
    from src.frontend_c.ir_lowering import CIRLowering
    from src.frontend_rust.parser import RustParser
    from src.frontend_rust.ir_lowering import RustIRLowering
    from src.frontend_rust.type_resolver import RustTypeResolver
    from src.semantics.semantic_config import SemanticConfig
    from src.smt.encoder import SMTEncoder, EncodingContext
    from src.oracle.oracle import preprocess_rust_code

    cleaned_rust = preprocess_rust_code(rust_code)

    try:
        c_ast = CParser(c_code, f"{name}.c").parse()
        c_module = CIRLowering().lower(c_ast)
    except Exception:
        return None

    try:
        r_ast = RustParser(cleaned_rust, f"{name}.rs").parse()
        r_module = RustIRLowering(RustTypeResolver()).lower(r_ast)
    except Exception:
        import re
        if re.match(r'\s*fn\s+', cleaned_rust):
            try:
                r_ast = RustParser("pub " + cleaned_rust, f"{name}.rs").parse()
                r_module = RustIRLowering(RustTypeResolver()).lower(r_ast)
            except Exception:
                return None
        else:
            return None

    c_funcs = list(c_module.functions.values())
    r_funcs = list(r_module.functions.values())
    if not c_funcs or not r_funcs:
        return None

    c_func, r_func = c_funcs[0], r_funcs[0]
    c_config = SemanticConfig.c11()
    r_config = SemanticConfig.rust_release()

    c_args = list(c_func.arguments)
    r_args = list(r_func.arguments)
    n_args = min(len(c_args), len(r_args))
    if n_args == 0:
        return None

    dummy_encoder = SMTEncoder(config=c_config)
    shared_vars = []
    c_input_map: Dict[str, z3.ExprRef] = {}
    r_input_map: Dict[str, z3.ExprRef] = {}

    for i in range(n_args):
        sort = dummy_encoder.encode_type(c_args[i].type) if c_args[i].type else z3.BitVecSort(32)
        z3_var = z3.BitVec(f"input_{i}", sort.size()) if z3.is_bv_sort(sort) else z3.Const(f"input_{i}", sort)
        shared_vars.append((f"input_{i}", z3_var))
        ca_name = c_args[i].name or f"arg_{c_args[i].index}"
        ra_name = r_args[i].name or f"arg_{r_args[i].index}"
        c_input_map[ca_name] = z3_var
        r_input_map[ra_name] = z3_var

    # Encode C
    c_ctx = EncodingContext()
    for nm, var in c_input_map.items():
        c_ctx.declarations[nm] = var
        c_ctx._alloca_values[nm] = var
    c_encoder = SMTEncoder(config=c_config)
    _, c_ret = c_encoder.encode_function(c_func, c_ctx)

    # Encode Rust
    r_ctx = EncodingContext()
    for nm, var in r_input_map.items():
        r_ctx.declarations[nm] = var
        r_ctx._alloca_values[nm] = var
    r_encoder = SMTEncoder(config=r_config)
    _, r_ret = r_encoder.encode_function(r_func, r_ctx)

    if c_ret is None or r_ret is None:
        return None

    # Coerce return types to match
    c_r, r_r = c_ret, r_ret
    try:
        if z3.is_bv(c_r) and z3.is_bv(r_r):
            cw, rw = c_r.size(), r_r.size()
            if cw != rw:
                if cw < rw:
                    c_r = z3.SignExt(rw - cw, c_r)
                else:
                    r_r = z3.SignExt(cw - rw, r_r)
        elif z3.is_bool(c_r) and z3.is_bv(r_r):
            c_r = z3.If(c_r, z3.BitVecVal(1, r_r.size()), z3.BitVecVal(0, r_r.size()))
        elif z3.is_bv(c_r) and z3.is_bool(r_r):
            r_r = z3.If(r_r, z3.BitVecVal(1, c_r.size()), z3.BitVecVal(0, c_r.size()))
    except Exception:
        return None

    return shared_vars, c_r, r_r, c_ctx, r_ctx


def eval_on_concrete(shared_vars, c_expr, r_expr, c_ctx, r_ctx, inputs: List[int],
                     ub_aware: bool = False) -> Optional[bool]:
    """Evaluate both z3 expressions on a concrete input vector.
    Returns True if outputs differ, False if same, None if can't evaluate.

    If ub_aware=True, also checks whether C assumptions are violated on this
    input (meaning C has UB here, so the pair diverges regardless of output).
    This models an ideal differential tester that knows about UB.
    """
    subs = []
    for i, (vname, var) in enumerate(shared_vars):
        if i < len(inputs):
            val = z3.BitVecVal(inputs[i], var.sort().size()) if z3.is_bv(var) else z3.IntVal(inputs[i])
            subs.append((var, val))

    # Check output mismatch
    try:
        c_val = z3.simplify(z3.substitute(c_expr, *subs))
        r_val = z3.simplify(z3.substitute(r_expr, *subs))
    except Exception:
        return None

    def _is_concrete(v):
        return z3.is_bv_value(v) or z3.is_int_value(v) or z3.is_true(v) or z3.is_false(v)

    output_differs = False
    if _is_concrete(c_val) and _is_concrete(r_val):
        try:
            s = z3.Solver()
            s.add(c_val != r_val)
            output_differs = (s.check() == z3.sat)
        except Exception:
            pass
    else:
        try:
            s = z3.Solver()
            s.set("timeout", 1000)
            s.add(c_val != r_val)
            output_differs = (s.check() == z3.sat)
        except Exception:
            pass

    if output_differs:
        return True

    # UB-aware mode: check if C assumptions are violated on this input.
    # If so, C has undefined behavior here — the functions diverge in
    # specification even if the concrete outputs happen to match.
    if ub_aware and c_ctx.assumptions:
        for assumption in c_ctx.assumptions:
            try:
                concrete_asn = z3.simplify(z3.substitute(assumption, *subs))
                if z3.is_false(concrete_asn):
                    return True  # UB triggered
                if not z3.is_true(concrete_asn):
                    s = z3.Solver()
                    s.set("timeout", 500)
                    s.add(z3.Not(concrete_asn))
                    if s.check() == z3.sat:
                        return True
            except Exception:
                continue

    return output_differs


def differential_test(pair: BenchmarkPair, n_random: int = 1000):
    """Run differential testing on a single pair.
    Returns dict with results, or None if can't test."""
    z3_data = build_z3_pair(pair.c_source, pair.rust_source, pair.name)
    if z3_data is None:
        return None

    shared_vars, c_expr, r_expr, c_ctx, r_ctx = z3_data
    n_args = len(shared_vars)

    # Generate test inputs
    test_inputs = []

    # 1. Single edge cases
    for ec in EDGE_CASES_I32:
        test_inputs.append([ec] * n_args)

    # 2. Pairs of edge cases (combinatorial)
    if n_args >= 2:
        important_edges = [0, 1, -1, INT32_MAX, INT32_MIN, INT32_MAX - 1, INT32_MIN + 1, 42, -42]
        for e1 in important_edges:
            for e2 in important_edges:
                inp = [e1, e2] + [0] * (n_args - 2)
                test_inputs.append(inp)
    if n_args >= 3:
        for e1 in [0, 1, -1, INT32_MAX, INT32_MIN]:
            for e2 in [0, 1, -1, INT32_MAX, INT32_MIN]:
                for e3 in [0, 1, -1, INT32_MAX, INT32_MIN]:
                    test_inputs.append([e1, e2, e3] + [0] * (n_args - 3))

    # 3. Random inputs
    for _ in range(n_random):
        test_inputs.append([random_i32() for _ in range(n_args)])

    # Run tests
    found_divergence = False
    divergent_input = None
    n_tested = 0
    n_edge_tested = 0
    n_edge_total = len(test_inputs) - n_random
    edge_found = False

    for idx, inputs in enumerate(test_inputs):
        is_edge = idx < (len(test_inputs) - n_random)
        result = eval_on_concrete(shared_vars, c_expr, r_expr, c_ctx, r_ctx, inputs)
        if result is None:
            continue
        n_tested += 1
        if is_edge:
            n_edge_tested += 1
        if result:
            found_divergence = True
            divergent_input = inputs
            if is_edge:
                edge_found = True
            break

    return {
        "found": found_divergence,
        "input": divergent_input,
        "n_total": len(test_inputs),
        "n_tested": n_tested,
        "n_random": n_random,
        "n_edge": n_edge_total,
        "edge_found": edge_found,
    }


# ---------------------------------------------------------------------------
# Main: run comparison at all budgets
# ---------------------------------------------------------------------------

def run_comparison():
    pairs = get_all_pairs()
    oracle = VerificationOracle(timeout_ms=5000)

    print(f"Running differential testing baseline on {len(pairs)} benchmark pairs")
    print(f"Random budgets: {RANDOM_BUDGETS}")
    print("=" * 80)

    # First, get oracle results for all pairs
    oracle_results = {}
    print("\n--- Oracle (SMT) verification ---")
    oracle_correct = 0
    for p in pairs:
        r = oracle.verify(p.c_source, p.rust_source, p.name)
        oracle_match = (r.verdict == p.expected_result)
        if oracle_match:
            oracle_correct += 1
        oracle_results[p.name] = {
            "verdict": r.verdict,
            "correct": oracle_match,
            "time_ms": r.time_ms,
        }
        status = "✓" if oracle_match else "✗"
        print(f"  {status} {p.name}: expected={p.expected_result}, oracle={r.verdict} ({r.time_ms:.0f}ms)")

    print(f"\nOracle accuracy: {oracle_correct}/{len(pairs)} ({oracle_correct/len(pairs)*100:.1f}%)")

    # Pre-build z3 expressions (shared across budgets)
    print("\n--- Building z3 expressions for differential testing ---")
    z3_cache = {}
    for p in pairs:
        z3_data = build_z3_pair(p.c_source, p.rust_source, p.name)
        z3_cache[p.name] = z3_data
        status = "OK" if z3_data else "SKIP"
        print(f"  {status}: {p.name}")

    # Run differential testing at each budget, in TWO modes:
    # 1. Naive: only compare outputs (standard differential testing)
    # 2. UB-aware: also flag inputs that trigger C undefined behavior
    all_budget_results = {}
    all_budget_results_ub = {}
    for budget in RANDOM_BUDGETS:
        for ub_mode, results_dict in [(False, all_budget_results), (True, all_budget_results_ub)]:
            mode_label = "UB-aware" if ub_mode else "naive"
            print(f"\n--- Differential testing ({mode_label}, n_random={budget}) ---")
            random.seed(42)  # reproducible
            diff_correct = 0
            budget_results = []

            for p in pairs:
                z3_data = z3_cache[p.name]
                if z3_data is None:
                    diff_verdict = "error"
                    diff_match = False
                    diff_detail = {"found": False, "n_tested": 0, "n_random": budget, "error": True}
                else:
                    shared_vars, c_expr, r_expr, c_ctx, r_ctx = z3_data
                    n_args = len(shared_vars)

                    # Generate test inputs
                    test_inputs = []
                    # Edge cases
                    for ec in EDGE_CASES_I32:
                        test_inputs.append([ec] * n_args)
                    if n_args >= 2:
                        important_edges = [0, 1, -1, INT32_MAX, INT32_MIN, INT32_MAX - 1, INT32_MIN + 1, 42, -42]
                        for e1 in important_edges:
                            for e2 in important_edges:
                                test_inputs.append([e1, e2] + [0] * (n_args - 2))
                    if n_args >= 3:
                        for e1 in [0, 1, -1, INT32_MAX, INT32_MIN]:
                            for e2 in [0, 1, -1, INT32_MAX, INT32_MIN]:
                                for e3 in [0, 1, -1, INT32_MAX, INT32_MIN]:
                                    test_inputs.append([e1, e2, e3] + [0] * (n_args - 3))

                    n_edge = len(test_inputs)
                    # Random
                    for _ in range(budget):
                        test_inputs.append([random_i32() for _ in range(n_args)])

                    found = False
                    divergent_input = None
                    n_tested = 0
                    edge_found = False

                    for idx, inputs in enumerate(test_inputs):
                        is_edge = idx < n_edge
                        result = eval_on_concrete(shared_vars, c_expr, r_expr, c_ctx, r_ctx, inputs,
                                                  ub_aware=ub_mode)
                        if result is None:
                            continue
                        n_tested += 1
                        if result:
                            found = True
                            divergent_input = inputs
                            edge_found = is_edge
                            break

                    diff_detail = {
                        "found": found,
                        "input": divergent_input,
                        "n_tested": n_tested,
                        "n_total": len(test_inputs),
                        "n_random": budget,
                        "n_edge": n_edge,
                        "edge_found": edge_found,
                    }

                    if found:
                        diff_verdict = "divergent"
                    else:
                        diff_verdict = "equivalent"

                    diff_match = (diff_verdict == p.expected_result)

                if diff_match:
                    diff_correct += 1

                status = "✓" if diff_match else "✗"
                extra = ""
                if diff_detail.get("found") and diff_detail.get("edge_found"):
                    extra = " [edge case]"
                elif diff_detail.get("found"):
                    extra = " [random]"
                print(f"  {status} {p.name}: expected={p.expected_result}, diff={diff_verdict}{extra}")

                budget_results.append({
                    "name": p.name,
                    "category": p.category,
                    "expected": p.expected_result,
                    "diff_verdict": diff_verdict,
                    "diff_correct": diff_match,
                    "oracle_verdict": oracle_results[p.name]["verdict"],
                    "oracle_correct": oracle_results[p.name]["correct"],
                    "diff_detail": {k: (str(v) if isinstance(v, list) else v)
                                    for k, v in diff_detail.items()},
                })

            results_dict[budget] = budget_results
            print(f"\n  Diff testing accuracy ({mode_label}, n={budget}): "
                  f"{diff_correct}/{len(pairs)} ({diff_correct/len(pairs)*100:.1f}%)")

    # ---------------------------------------------------------------------------
    # Analysis
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(f"\nOracle accuracy: {oracle_correct}/{len(pairs)} ({oracle_correct/len(pairs)*100:.1f}%)")

    for label, res_dict in [("Naive diff testing", all_budget_results),
                             ("UB-aware diff testing", all_budget_results_ub)]:
        print(f"\n  {label}:")
        for budget in RANDOM_BUDGETS:
            results = res_dict[budget]
            diff_correct = sum(1 for r in results if r["diff_correct"])
            print(f"    n={budget}: {diff_correct}/{len(pairs)} ({diff_correct/len(pairs)*100:.1f}%)")

    # Detailed analysis for both modes at largest budget
    for mode_label, res_dict in [("NAIVE", all_budget_results), ("UB-AWARE", all_budget_results_ub)]:
        final_results = res_dict[RANDOM_BUDGETS[-1]]

        oracle_only = [r for r in final_results if r["oracle_correct"] and not r["diff_correct"]]
        diff_only = [r for r in final_results if r["diff_correct"] and not r["oracle_correct"]]
        both_correct = [r for r in final_results if r["oracle_correct"] and r["diff_correct"]]
        neither = [r for r in final_results if not r["oracle_correct"] and not r["diff_correct"]]

        print(f"\n--- {mode_label} diff testing analysis (n={RANDOM_BUDGETS[-1]}) ---")
        print(f"  Both correct:       {len(both_correct)}")
        print(f"  Oracle-only:        {len(oracle_only)}")
        print(f"  Diff-only:          {len(diff_only)}")
        print(f"  Neither:            {len(neither)}")

        if oracle_only:
            print(f"\n  Oracle catches that {mode_label} diff testing MISSES:")
            for r in oracle_only:
                print(f"    {r['name']} ({r['category']}): expected={r['expected']}, "
                      f"oracle={r['oracle_verdict']}, diff={r['diff_verdict']}")

        if diff_only:
            print(f"\n  {mode_label} diff testing catches that oracle MISSES:")
            for r in diff_only:
                print(f"    {r['name']} ({r['category']}): expected={r['expected']}, "
                      f"oracle={r['oracle_verdict']}, diff={r['diff_verdict']}")

        # Divergent-only analysis
        divergent_pairs = [r for r in final_results if r["expected"] == "divergent"]
        o_div_found = sum(1 for r in divergent_pairs if r["oracle_correct"])
        d_div_found = sum(1 for r in divergent_pairs if r["diff_correct"])
        print(f"\n  Divergence detection ({mode_label}, divergent pairs n={len(divergent_pairs)}):")
        print(f"    Oracle finds: {o_div_found}/{len(divergent_pairs)} divergences")
        print(f"    Diff testing finds: {d_div_found}/{len(divergent_pairs)} divergences")

        missed_divs = [r for r in divergent_pairs if r["oracle_correct"] and not r["diff_correct"]]
        if missed_divs:
            print(f"\n  Divergences that ONLY the oracle detects ({len(missed_divs)}):")
            for r in missed_divs:
                print(f"    - {r['name']} ({r['category']})")

    # Per-category breakdown (using naive mode for the main comparison)
    final_results_naive = all_budget_results[RANDOM_BUDGETS[-1]]
    final_results_ub = all_budget_results_ub[RANDOM_BUDGETS[-1]]
    print(f"\n  Per-category breakdown (n={RANDOM_BUDGETS[-1]}):")
    categories = sorted(set(r["category"] for r in final_results_naive))
    print(f"    {'Category':20s}  Oracle    Naive     UB-aware")
    for cat in categories:
        cat_naive = [r for r in final_results_naive if r["category"] == cat]
        cat_ub = [r for r in final_results_ub if r["category"] == cat]
        o_correct = sum(1 for r in cat_naive if r["oracle_correct"])
        n_correct = sum(1 for r in cat_naive if r["diff_correct"])
        u_correct = sum(1 for r in cat_ub if r["diff_correct"])
        n = len(cat_naive)
        print(f"    {cat:20s}  {o_correct}/{n}       {n_correct}/{n}       {u_correct}/{n}")

    # Budget scaling analysis
    print(f"\n  Budget scaling (divergent pairs only, n=29):")
    print(f"    {'Budget':>7s}  {'Naive':>10s}  {'UB-aware':>10s}")
    for budget in RANDOM_BUDGETS:
        naive_div = [r for r in all_budget_results[budget] if r["expected"] == "divergent"]
        ub_div = [r for r in all_budget_results_ub[budget] if r["expected"] == "divergent"]
        nf = sum(1 for r in naive_div if r["diff_correct"])
        uf = sum(1 for r in ub_div if r["diff_correct"])
        nd = len(naive_div)
        print(f"    n={budget:>5d}  {nf}/{nd} ({nf/nd*100:4.1f}%)  {uf}/{nd} ({uf/nd*100:4.1f}%)")

    # Save results
    output = {
        "summary": {
            "n_pairs": len(pairs),
            "n_divergent": sum(1 for p in pairs if p.expected_result == "divergent"),
            "n_equivalent": sum(1 for p in pairs if p.expected_result == "equivalent"),
            "oracle_accuracy_pct": round(oracle_correct / len(pairs) * 100, 1),
            "budgets": {},
        },
        "naive_results": {},
        "ub_aware_results": {},
        "oracle_results": oracle_results,
    }

    for budget in RANDOM_BUDGETS:
        for label, res_dict, key in [("naive", all_budget_results, "naive_results"),
                                      ("ub_aware", all_budget_results_ub, "ub_aware_results")]:
            bres = res_dict[budget]
            dc = sum(1 for r in bres if r["diff_correct"])
            div_bres = [r for r in bres if r["expected"] == "divergent"]
            div_found = sum(1 for r in div_bres if r["diff_correct"])
            output["summary"]["budgets"][f"{label}_n{budget}"] = {
                "diff_accuracy_pct": round(dc / len(pairs) * 100, 1),
                "divergences_found": div_found,
                "divergences_total": len(div_bres),
            }
            output[key][str(budget)] = bres

    # Oracle-unique finds (vs naive at max budget)
    final_naive = all_budget_results[RANDOM_BUDGETS[-1]]
    final_ub = all_budget_results_ub[RANDOM_BUDGETS[-1]]
    output["oracle_unique_vs_naive"] = [
        r["name"] for r in final_naive if r["oracle_correct"] and not r["diff_correct"]
    ]
    output["oracle_unique_vs_ub_aware"] = [
        r["name"] for r in final_ub if r["oracle_correct"] and not r["diff_correct"]
    ]

    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "experiments", "results")
    os.makedirs(results_dir, exist_ok=True)
    outpath = os.path.join(results_dir, "diff_testing_comparison.json")
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {outpath}")


if __name__ == "__main__":
    run_comparison()
