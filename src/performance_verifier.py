"""
Performance equivalence verification for C-to-Rust migration.
Provides static analysis, cost modeling, SIMD intrinsic mapping,
and benchmark generation for cross-language performance comparison.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple


class OptimizationKind(Enum):
    BOUNDS_CHECK_ELIM = auto()
    ITERATOR_FUSION = auto()
    ZERO_COST_ABSTRACTION = auto()
    SIMD_VECTORIZATION = auto()
    BRANCH_ELIMINATION = auto()
    ALLOCATION_ELISION = auto()
    INLINE_OPPORTUNITY = auto()
    CACHE_OPTIMIZATION = auto()


@dataclass
class PerfMetric:
    name: str
    value: float
    unit: str
    confidence: float = 0.0
    source: str = "static_analysis"

    def normalized(self, baseline: float) -> float:
        if baseline == 0.0:
            return 0.0
        return self.value / baseline

    def within_tolerance(self, other: PerfMetric, tolerance: float = 0.05) -> bool:
        if other.value == 0.0:
            return self.value == 0.0
        ratio = abs(self.value - other.value) / max(abs(other.value), 1e-12)
        return ratio <= tolerance


@dataclass
class PerfComparison:
    c_metrics: List[PerfMetric]
    rust_metrics: List[PerfMetric]
    speedup: float
    regression_risk: float
    summary: str
    detailed_notes: List[str] = field(default_factory=list)

    def is_regression(self, threshold: float = 0.95) -> bool:
        return self.speedup < threshold

    def is_improvement(self, threshold: float = 1.05) -> bool:
        return self.speedup > threshold


@dataclass
class PerfPrediction:
    predicted_speedup: float
    confidence: float
    factors: List[str]
    bottlenecks: List[str]
    recommendations: List[str]

    def expected_range(self) -> Tuple[float, float]:
        margin = (1.0 - self.confidence) * self.predicted_speedup
        return (self.predicted_speedup - margin, self.predicted_speedup + margin)


@dataclass
class PerfRegression:
    location: str
    description: str
    severity: float
    category: str
    mitigation: str

    def is_critical(self) -> bool:
        return self.severity > 0.7


@dataclass
class Optimization:
    kind: OptimizationKind
    location: str
    description: str
    estimated_improvement: float
    code_suggestion: str

    def priority_score(self) -> float:
        kind_weights = {
            OptimizationKind.BOUNDS_CHECK_ELIM: 0.8,
            OptimizationKind.ITERATOR_FUSION: 0.7,
            OptimizationKind.ZERO_COST_ABSTRACTION: 0.6,
            OptimizationKind.SIMD_VECTORIZATION: 0.9,
            OptimizationKind.BRANCH_ELIMINATION: 0.5,
            OptimizationKind.ALLOCATION_ELISION: 0.85,
            OptimizationKind.INLINE_OPPORTUNITY: 0.4,
            OptimizationKind.CACHE_OPTIMIZATION: 0.75,
        }
        return self.estimated_improvement * kind_weights.get(self.kind, 0.5)


@dataclass
class SIMDMapping:
    c_intrinsic: str
    rust_equivalent: str
    arch: str
    feature_gate: str
    notes: str = ""


@dataclass
class LoopAnalysis:
    c_loop_count: int
    rust_loop_count: int
    vectorizable_c: int
    vectorizable_rust: int
    iterator_chains: int
    bounds_checks_in_loops: int
    recommendations: List[str] = field(default_factory=list)

    def vectorization_ratio(self) -> float:
        total = self.c_loop_count + self.rust_loop_count
        if total == 0:
            return 1.0
        vectorizable = self.vectorizable_c + self.vectorizable_rust
        return vectorizable / total


@dataclass
class AllocPattern:
    malloc_count: int
    free_count: int
    realloc_count: int
    stack_arrays: int
    heap_arrays: int
    box_allocations: int
    vec_allocations: int
    potential_leaks: int

    def total_heap_ops(self) -> int:
        return (self.malloc_count + self.free_count + self.realloc_count
                + self.box_allocations + self.vec_allocations)


@dataclass
class BenchmarkResult:
    name: str
    mean_ns: float
    std_dev_ns: float
    throughput_ops_per_sec: float
    iterations: int

    def coefficient_of_variation(self) -> float:
        if self.mean_ns == 0.0:
            return 0.0
        return self.std_dev_ns / self.mean_ns


@dataclass
class CostModel:
    arithmetic_ops: int
    memory_loads: int
    memory_stores: int
    branches: int
    function_calls: int
    estimated_cycles: float
    memory_bandwidth_bytes: int

    def compute_intensity(self) -> float:
        total_mem = self.memory_loads + self.memory_stores
        if total_mem == 0:
            return float("inf")
        return self.arithmetic_ops / total_mem

    def estimated_time_ns(self, clock_ghz: float = 3.0) -> float:
        if clock_ghz <= 0.0:
            return 0.0
        cycles_per_ns = clock_ghz
        return self.estimated_cycles / cycles_per_ns


# ---------------------------------------------------------------------------
# SIMD intrinsic mapping table
# ---------------------------------------------------------------------------

_SIMD_MAP: Dict[str, SIMDMapping] = {
    "_mm_add_ps": SIMDMapping(
        "_mm_add_ps", "_mm_add_ps", "x86_64", "sse",
        "use std::arch::x86_64::_mm_add_ps;",
    ),
    "_mm_sub_ps": SIMDMapping(
        "_mm_sub_ps", "_mm_sub_ps", "x86_64", "sse",
        "use std::arch::x86_64::_mm_sub_ps;",
    ),
    "_mm_mul_ps": SIMDMapping(
        "_mm_mul_ps", "_mm_mul_ps", "x86_64", "sse",
        "use std::arch::x86_64::_mm_mul_ps;",
    ),
    "_mm_div_ps": SIMDMapping(
        "_mm_div_ps", "_mm_div_ps", "x86_64", "sse",
        "use std::arch::x86_64::_mm_div_ps;",
    ),
    "_mm_set1_ps": SIMDMapping(
        "_mm_set1_ps", "_mm_set1_ps", "x86_64", "sse",
        "use std::arch::x86_64::_mm_set1_ps;",
    ),
    "_mm_setzero_ps": SIMDMapping(
        "_mm_setzero_ps", "_mm_setzero_ps", "x86_64", "sse",
        "use std::arch::x86_64::_mm_setzero_ps;",
    ),
    "_mm_load_ps": SIMDMapping(
        "_mm_load_ps", "_mm_load_ps", "x86_64", "sse",
        "use std::arch::x86_64::_mm_load_ps;",
    ),
    "_mm_store_ps": SIMDMapping(
        "_mm_store_ps", "_mm_store_ps", "x86_64", "sse",
        "use std::arch::x86_64::_mm_store_ps;",
    ),
    "_mm_loadu_ps": SIMDMapping(
        "_mm_loadu_ps", "_mm_loadu_ps", "x86_64", "sse",
        "use std::arch::x86_64::_mm_loadu_ps;",
    ),
    "_mm_storeu_ps": SIMDMapping(
        "_mm_storeu_ps", "_mm_storeu_ps", "x86_64", "sse",
        "use std::arch::x86_64::_mm_storeu_ps;",
    ),
    "_mm_max_ps": SIMDMapping(
        "_mm_max_ps", "_mm_max_ps", "x86_64", "sse",
        "use std::arch::x86_64::_mm_max_ps;",
    ),
    "_mm_min_ps": SIMDMapping(
        "_mm_min_ps", "_mm_min_ps", "x86_64", "sse",
        "use std::arch::x86_64::_mm_min_ps;",
    ),
    "_mm_sqrt_ps": SIMDMapping(
        "_mm_sqrt_ps", "_mm_sqrt_ps", "x86_64", "sse",
        "use std::arch::x86_64::_mm_sqrt_ps;",
    ),
    "_mm_and_ps": SIMDMapping(
        "_mm_and_ps", "_mm_and_ps", "x86_64", "sse",
        "use std::arch::x86_64::_mm_and_ps;",
    ),
    "_mm_or_ps": SIMDMapping(
        "_mm_or_ps", "_mm_or_ps", "x86_64", "sse",
        "use std::arch::x86_64::_mm_or_ps;",
    ),
    "_mm_xor_ps": SIMDMapping(
        "_mm_xor_ps", "_mm_xor_ps", "x86_64", "sse",
        "use std::arch::x86_64::_mm_xor_ps;",
    ),
    "_mm_cmpeq_ps": SIMDMapping(
        "_mm_cmpeq_ps", "_mm_cmpeq_ps", "x86_64", "sse",
        "use std::arch::x86_64::_mm_cmpeq_ps;",
    ),
    "_mm_cmplt_ps": SIMDMapping(
        "_mm_cmplt_ps", "_mm_cmplt_ps", "x86_64", "sse",
        "use std::arch::x86_64::_mm_cmplt_ps;",
    ),
    "_mm_add_epi32": SIMDMapping(
        "_mm_add_epi32", "_mm_add_epi32", "x86_64", "sse2",
        "use std::arch::x86_64::_mm_add_epi32;",
    ),
    "_mm_sub_epi32": SIMDMapping(
        "_mm_sub_epi32", "_mm_sub_epi32", "x86_64", "sse2",
        "use std::arch::x86_64::_mm_sub_epi32;",
    ),
    "_mm_mullo_epi32": SIMDMapping(
        "_mm_mullo_epi32", "_mm_mullo_epi32", "x86_64", "sse4.1",
        "use std::arch::x86_64::_mm_mullo_epi32;",
    ),
    "_mm256_add_ps": SIMDMapping(
        "_mm256_add_ps", "_mm256_add_ps", "x86_64", "avx",
        "use std::arch::x86_64::_mm256_add_ps;",
    ),
    "_mm256_mul_ps": SIMDMapping(
        "_mm256_mul_ps", "_mm256_mul_ps", "x86_64", "avx",
        "use std::arch::x86_64::_mm256_mul_ps;",
    ),
    "_mm256_sub_ps": SIMDMapping(
        "_mm256_sub_ps", "_mm256_sub_ps", "x86_64", "avx",
        "use std::arch::x86_64::_mm256_sub_ps;",
    ),
    "_mm256_load_ps": SIMDMapping(
        "_mm256_load_ps", "_mm256_load_ps", "x86_64", "avx",
        "use std::arch::x86_64::_mm256_load_ps;",
    ),
    "_mm256_store_ps": SIMDMapping(
        "_mm256_store_ps", "_mm256_store_ps", "x86_64", "avx",
        "use std::arch::x86_64::_mm256_store_ps;",
    ),
    "_mm256_set1_ps": SIMDMapping(
        "_mm256_set1_ps", "_mm256_set1_ps", "x86_64", "avx",
        "use std::arch::x86_64::_mm256_set1_ps;",
    ),
    "_mm256_setzero_ps": SIMDMapping(
        "_mm256_setzero_ps", "_mm256_setzero_ps", "x86_64", "avx",
        "use std::arch::x86_64::_mm256_setzero_ps;",
    ),
    "_mm256_fmadd_ps": SIMDMapping(
        "_mm256_fmadd_ps", "_mm256_fmadd_ps", "x86_64", "fma",
        "use std::arch::x86_64::_mm256_fmadd_ps;",
    ),
}

# ---------------------------------------------------------------------------
# Cost constants for the simple cost model
# ---------------------------------------------------------------------------

_COST_ARITH = 1.0
_COST_MUL = 3.0
_COST_DIV = 12.0
_COST_BRANCH = 2.0
_COST_CALL = 5.0
_COST_LOAD = 4.0
_COST_STORE = 4.0
_COST_BOUNDS_CHECK = 1.5
_COST_UTF8_VALIDATION = 8.0
_COST_ALLOC = 50.0
_COST_FREE = 30.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_pattern(code: str, pattern: str) -> int:
    return len(re.findall(pattern, code))


def _extract_loops_c(code: str) -> List[str]:
    loops: List[str] = []
    for m in re.finditer(r'\bfor\s*\([^)]*\)\s*\{', code):
        start = m.start()
        depth = 0
        end = start
        for i in range(m.end() - 1, len(code)):
            if code[i] == '{':
                depth += 1
            elif code[i] == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        loops.append(code[start:end])
    for m in re.finditer(r'\bwhile\s*\([^)]*\)\s*\{', code):
        start = m.start()
        depth = 0
        end = start
        for i in range(m.end() - 1, len(code)):
            if code[i] == '{':
                depth += 1
            elif code[i] == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        loops.append(code[start:end])
    return loops


def _extract_loops_rust(code: str) -> List[str]:
    loops: List[str] = []
    for m in re.finditer(r'\bfor\s+\w+\s+in\s+', code):
        start = m.start()
        brace = code.find('{', m.end())
        if brace == -1:
            continue
        depth = 0
        end = brace
        for i in range(brace, len(code)):
            if code[i] == '{':
                depth += 1
            elif code[i] == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        loops.append(code[start:end])
    for m in re.finditer(r'\bloop\s*\{', code):
        start = m.start()
        depth = 0
        end = start
        for i in range(m.end() - 1, len(code)):
            if code[i] == '{':
                depth += 1
            elif code[i] == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        loops.append(code[start:end])
    for m in re.finditer(r'\bwhile\s+[^{]*\{', code):
        start = m.start()
        depth = 0
        end = start
        for i in range(m.end() - 1, len(code)):
            if code[i] == '{':
                depth += 1
            elif code[i] == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        loops.append(code[start:end])
    return loops


def _is_vectorizable_c(loop_body: str) -> bool:
    if re.search(r'\bif\s*\(', loop_body):
        return False
    if re.search(r'\b(break|continue|goto|return)\b', loop_body):
        return False
    if _count_pattern(loop_body, r'\b(malloc|calloc|realloc|free)\b') > 0:
        return False
    arith = _count_pattern(loop_body, r'[+\-*/]=?')
    array_access = _count_pattern(loop_body, r'\w+\[')
    return arith > 0 and array_access > 0


def _is_vectorizable_rust(loop_body: str) -> bool:
    if re.search(r'\bif\s+', loop_body) and 'if let' not in loop_body:
        return False
    if re.search(r'\b(break|continue|return)\b', loop_body):
        return False
    if _count_pattern(loop_body, r'\b(Box::new|Vec::new|vec!)\b') > 0:
        return False
    arith = _count_pattern(loop_body, r'[+\-*/]=?')
    index_access = _count_pattern(loop_body, r'\w+\[')
    iter_call = _count_pattern(loop_body, r'\.\b(map|filter|fold|sum|zip)\b')
    return arith > 0 and (index_access > 0 or iter_call > 0)


def _count_iterator_chains(code: str) -> int:
    chain_pattern = r'\.\b(iter|into_iter|iter_mut)\b\(\)'
    chains = re.findall(chain_pattern, code)
    multi_step = re.findall(
        r'\.\b(map|filter|fold|flat_map|filter_map|take|skip|zip|chain|'
        r'enumerate|peekable|collect|sum|product|any|all|find|position|'
        r'max|min|count|for_each|inspect|cloned|copied)\b\s*\(',
        code,
    )
    return max(len(chains), len(multi_step) // 2)


def _count_bounds_checks_in_loops(code: str) -> int:
    loops = _extract_loops_rust(code)
    count = 0
    for loop in loops:
        index_accesses = _count_pattern(loop, r'\w+\[\w+\]')
        has_get_unchecked = 'get_unchecked' in loop
        has_iter = re.search(r'\.\b(iter|into_iter|iter_mut)\b', loop) is not None
        if index_accesses > 0 and not has_get_unchecked and not has_iter:
            count += index_accesses
    return count


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def estimate_cost(code: str, lang: str) -> CostModel:
    """Build a simple static cost model by counting operations."""
    add_sub = _count_pattern(code, r'(?<![=!<>])[+\-](?!=)')
    mul = _count_pattern(code, r'\*(?!=)')
    div_mod = _count_pattern(code, r'[/%](?!=)')
    arithmetic_ops = add_sub + mul + div_mod

    if lang == "c":
        loads = _count_pattern(code, r'\*\w+') + _count_pattern(code, r'\w+\[')
        stores = _count_pattern(code, r'\w+\[\w+\]\s*=') + _count_pattern(code, r'\*\w+\s*=')
        branches = _count_pattern(code, r'\bif\s*\(') + _count_pattern(code, r'\bswitch\s*\(')
        calls = _count_pattern(code, r'\b\w+\s*\(') - branches
        alloc_ops = _count_pattern(code, r'\b(malloc|calloc|realloc)\b')
        free_ops = _count_pattern(code, r'\bfree\b')
    else:
        loads = _count_pattern(code, r'\w+\[') + _count_pattern(code, r'&\w+')
        stores = _count_pattern(code, r'\w+\[\w+\]\s*=') + _count_pattern(code, r'\blet\s+mut\b')
        branches = _count_pattern(code, r'\bif\s+') + _count_pattern(code, r'\bmatch\s+')
        calls = _count_pattern(code, r'\b\w+\s*\(') - branches
        bounds_checks = _count_bounds_checks_in_loops(code)
        loads += bounds_checks
        alloc_ops = _count_pattern(code, r'\b(Box::new|Vec::new|vec!|String::new)\b')
        free_ops = 0  # Rust drops are implicit

    calls = max(calls, 0)
    cycles = (
        arithmetic_ops * _COST_ARITH
        + mul * (_COST_MUL - _COST_ARITH)
        + div_mod * (_COST_DIV - _COST_ARITH)
        + loads * _COST_LOAD
        + stores * _COST_STORE
        + branches * _COST_BRANCH
        + calls * _COST_CALL
        + alloc_ops * _COST_ALLOC
        + free_ops * _COST_FREE
    )

    mem_bytes = (loads + stores) * 8  # assume 64-bit words

    return CostModel(
        arithmetic_ops=arithmetic_ops,
        memory_loads=loads,
        memory_stores=stores,
        branches=branches,
        function_calls=calls,
        estimated_cycles=cycles,
        memory_bandwidth_bytes=mem_bytes,
    )


def benchmark_comparison(
    c_code: str, rust_code: str, inputs: List[Dict],
) -> PerfComparison:
    """Compare estimated performance characteristics of C vs Rust code."""
    c_cost = estimate_cost(c_code, "c")
    rust_cost = estimate_cost(rust_code, "rust")

    c_metrics: List[PerfMetric] = [
        PerfMetric("estimated_cycles", c_cost.estimated_cycles, "cycles", 0.7),
        PerfMetric("arithmetic_ops", float(c_cost.arithmetic_ops), "ops", 0.9),
        PerfMetric("memory_ops", float(c_cost.memory_loads + c_cost.memory_stores), "ops", 0.8),
        PerfMetric("branches", float(c_cost.branches), "count", 0.9),
        PerfMetric("function_calls", float(c_cost.function_calls), "count", 0.9),
    ]
    rust_metrics: List[PerfMetric] = [
        PerfMetric("estimated_cycles", rust_cost.estimated_cycles, "cycles", 0.7),
        PerfMetric("arithmetic_ops", float(rust_cost.arithmetic_ops), "ops", 0.9),
        PerfMetric("memory_ops", float(rust_cost.memory_loads + rust_cost.memory_stores), "ops", 0.8),
        PerfMetric("branches", float(rust_cost.branches), "count", 0.9),
        PerfMetric("function_calls", float(rust_cost.function_calls), "count", 0.9),
    ]

    if rust_cost.estimated_cycles == 0.0:
        speedup = 1.0
    else:
        speedup = c_cost.estimated_cycles / rust_cost.estimated_cycles

    # Factor in input-dependent scaling
    input_scale = 1.0
    for inp in inputs:
        size = inp.get("size", 1)
        if isinstance(size, (int, float)) and size > 1000:
            # Large inputs amplify allocation differences
            c_allocs = _count_pattern(c_code, r'\b(malloc|calloc)\b')
            rust_allocs = _count_pattern(rust_code, r'\b(Vec::new|vec!)\b')
            if c_allocs > rust_allocs:
                input_scale *= 1.02
            elif rust_allocs > c_allocs:
                input_scale *= 0.98

    speedup *= input_scale

    bounds_in_loops = _count_bounds_checks_in_loops(rust_code)
    regression_risk = min(1.0, bounds_in_loops * 0.05 + (0.1 if speedup < 1.0 else 0.0))

    notes: List[str] = []
    if bounds_in_loops > 0:
        notes.append(f"Found {bounds_in_loops} potential bounds checks inside loops")
    iter_chains = _count_iterator_chains(rust_code)
    if iter_chains > 0:
        notes.append(f"Found {iter_chains} iterator chains (potential fusion)")
    if c_cost.compute_intensity() < rust_cost.compute_intensity():
        notes.append("Rust version has higher compute intensity (memory-bound likely)")

    summary_parts = []
    if speedup > 1.05:
        summary_parts.append(f"Rust is ~{speedup:.2f}x faster (estimated)")
    elif speedup < 0.95:
        summary_parts.append(f"C is ~{1/speedup:.2f}x faster (estimated)")
    else:
        summary_parts.append("Performance is roughly equivalent")
    summary = "; ".join(summary_parts)

    return PerfComparison(
        c_metrics=c_metrics,
        rust_metrics=rust_metrics,
        speedup=speedup,
        regression_risk=regression_risk,
        summary=summary,
        detailed_notes=notes,
    )


def predict_performance_difference(
    c_code: str, rust_code: str,
) -> PerfPrediction:
    """Static-analysis-based prediction of performance differences."""
    factors: List[str] = []
    bottlenecks: List[str] = []
    recommendations: List[str] = []
    speedup_multiplier = 1.0
    confidence = 0.6

    # Bounds check overhead
    bounds_checks = _count_bounds_checks_in_loops(rust_code)
    if bounds_checks > 0:
        overhead = bounds_checks * 0.02
        speedup_multiplier -= overhead
        factors.append(f"Bounds checks in loops: ~{overhead*100:.1f}% overhead")
        bottlenecks.append("Array indexing inside loops without iterator usage")
        recommendations.append(
            "Replace indexed loops with iterators or use get_unchecked in unsafe blocks"
        )

    # UTF-8 validation overhead
    utf8_points = _count_pattern(rust_code, r'\bString::from\b|\bto_string\(\)')
    c_str_ops = _count_pattern(c_code, r'\bstrcpy\b|\bstrcat\b|\bstrlen\b')
    if utf8_points > 0 and c_str_ops > 0:
        overhead = min(utf8_points * 0.03, 0.15)
        speedup_multiplier -= overhead
        factors.append(f"UTF-8 validation on {utf8_points} string conversions")
        bottlenecks.append("String conversion with UTF-8 validation")
        recommendations.append("Use byte slices (&[u8]) where UTF-8 is not required")

    # Iterator fusion opportunities
    iter_chains = _count_iterator_chains(rust_code)
    if iter_chains > 0:
        gain = min(iter_chains * 0.03, 0.12)
        speedup_multiplier += gain
        factors.append(f"Iterator fusion on {iter_chains} chains: ~{gain*100:.1f}% gain")

    # Allocation patterns
    c_allocs = _count_pattern(c_code, r'\b(malloc|calloc)\b')
    c_frees = _count_pattern(c_code, r'\bfree\b')
    rust_heap = _count_pattern(rust_code, r'\b(Box::new|Vec::new|vec!|String::new)\b')
    if c_allocs > rust_heap:
        gain = min((c_allocs - rust_heap) * 0.01, 0.08)
        speedup_multiplier += gain
        factors.append(f"Fewer heap allocations in Rust ({rust_heap} vs {c_allocs})")
    elif rust_heap > c_allocs:
        overhead = min((rust_heap - c_allocs) * 0.01, 0.08)
        speedup_multiplier -= overhead
        factors.append(f"More heap allocations in Rust ({rust_heap} vs {c_allocs})")
        recommendations.append("Reduce heap allocations; consider stack allocation or arena")

    # SIMD usage
    c_simd = _count_pattern(c_code, r'\b_mm\d*_\w+')
    rust_simd = _count_pattern(rust_code, r'\b_mm\d*_\w+')
    if c_simd > 0 and rust_simd == 0:
        overhead = min(c_simd * 0.05, 0.30)
        speedup_multiplier -= overhead
        factors.append(f"C uses {c_simd} SIMD intrinsics not present in Rust")
        bottlenecks.append("Missing SIMD vectorization")
        recommendations.append("Port SIMD intrinsics using std::arch or use auto-vectorizable patterns")
        confidence -= 0.1

    # Inline hints
    c_inline = _count_pattern(c_code, r'\b(inline|__inline|__forceinline)\b')
    rust_inline = _count_pattern(rust_code, r'#\[inline')
    if c_inline > rust_inline:
        factors.append(f"C has {c_inline} inline hints vs Rust {rust_inline}")
        recommendations.append("Add #[inline] or #[inline(always)] to hot functions")

    # Function call overhead comparison
    c_cost = estimate_cost(c_code, "c")
    rust_cost = estimate_cost(rust_code, "rust")
    if rust_cost.function_calls > c_cost.function_calls * 1.5:
        overhead = min((rust_cost.function_calls - c_cost.function_calls) * 0.005, 0.10)
        speedup_multiplier -= overhead
        factors.append(f"Rust has more function calls ({rust_cost.function_calls} vs {c_cost.function_calls})")

    speedup_multiplier = max(speedup_multiplier, 0.3)
    confidence = max(min(confidence, 1.0), 0.1)

    if not factors:
        factors.append("No significant performance differences detected")

    return PerfPrediction(
        predicted_speedup=round(speedup_multiplier, 4),
        confidence=round(confidence, 2),
        factors=factors,
        bottlenecks=bottlenecks,
        recommendations=recommendations,
    )


def detect_performance_regressions(
    c_code: str, rust_code: str,
) -> List[PerfRegression]:
    """Find locations where the Rust translation would be slower than C."""
    regressions: List[PerfRegression] = []

    # 1. Bounds checking in indexed loops
    rust_loops = _extract_loops_rust(rust_code)
    for i, loop in enumerate(rust_loops):
        index_accesses = _count_pattern(loop, r'\w+\[\w+\]')
        if index_accesses > 0 and 'get_unchecked' not in loop:
            uses_iter = re.search(r'\.\b(iter|into_iter)\b', loop) is not None
            if not uses_iter:
                regressions.append(PerfRegression(
                    location=f"rust_loop_{i}",
                    description=(
                        f"Loop has {index_accesses} array index accesses with implicit "
                        f"bounds checks. Each check adds a conditional branch."
                    ),
                    severity=min(0.3 + index_accesses * 0.1, 0.9),
                    category="bounds_check",
                    mitigation="Use iterators, slice::get_unchecked, or assert length before loop",
                ))

    # 2. UTF-8 validation overhead
    utf8_conversions = re.findall(
        r'\b(String::from|to_string|from_utf8|to_owned)\s*\(', rust_code,
    )
    c_str_ops = _count_pattern(c_code, r'\b(strcpy|strcat|strdup|strncpy)\b')
    if utf8_conversions and c_str_ops > 0:
        regressions.append(PerfRegression(
            location="string_handling",
            description=(
                f"Rust performs UTF-8 validation on {len(utf8_conversions)} string "
                f"conversions whereas C operates on raw bytes ({c_str_ops} string ops). "
                f"Each validation scans the full string."
            ),
            severity=min(0.2 + len(utf8_conversions) * 0.05, 0.7),
            category="utf8_validation",
            mitigation="Use &[u8], CStr, or String::from_utf8_unchecked where safe",
        ))

    # 3. Drop / destructor overhead
    rust_drop_types = _count_pattern(rust_code, r'\b(Vec|String|Box|HashMap|BTreeMap|Rc|Arc)\b')
    c_frees = _count_pattern(c_code, r'\bfree\b')
    if rust_drop_types > c_frees + 3:
        regressions.append(PerfRegression(
            location="destructor_overhead",
            description=(
                f"Rust uses {rust_drop_types} owned heap types vs {c_frees} explicit frees. "
                f"Automatic drop may run more destructors than the C version's manual frees."
            ),
            severity=0.3,
            category="drop_overhead",
            mitigation="Use references (&) instead of owned types where possible; use ManuallyDrop for hot paths",
        ))

    # 4. Enum / match dispatch overhead vs C switch
    rust_matches = _count_pattern(rust_code, r'\bmatch\s+')
    c_switches = _count_pattern(c_code, r'\bswitch\s*\(')
    if rust_matches > c_switches * 2:
        regressions.append(PerfRegression(
            location="match_dispatch",
            description=(
                f"Rust has {rust_matches} match expressions vs {c_switches} C switch statements. "
                f"Complex match patterns may not compile to simple jump tables."
            ),
            severity=0.25,
            category="dispatch_overhead",
            mitigation="Ensure matches are on contiguous integer-like enums for jump table generation",
        ))

    # 5. Overflow checking in debug builds
    arith_ops = _count_pattern(rust_code, r'[+\-*](?!=)')
    if arith_ops > 20:
        regressions.append(PerfRegression(
            location="overflow_checks",
            description=(
                f"Rust debug builds check {arith_ops} arithmetic operations for overflow. "
                f"Release builds optimize these away, but debug perf will differ."
            ),
            severity=0.15,
            category="debug_overhead",
            mitigation="Use wrapping_add/wrapping_mul or compile with overflow-checks=false",
        ))

    # 6. Missing SIMD
    c_simd_count = _count_pattern(c_code, r'\b_mm\d*_\w+')
    rust_simd_count = _count_pattern(rust_code, r'\b_mm\d*_\w+')
    if c_simd_count > 0 and rust_simd_count == 0:
        regressions.append(PerfRegression(
            location="simd_missing",
            description=(
                f"C code uses {c_simd_count} SIMD intrinsics that are absent from the Rust "
                f"translation. This can cause significant throughput loss on data-parallel workloads."
            ),
            severity=min(0.5 + c_simd_count * 0.05, 0.95),
            category="simd_regression",
            mitigation="Use std::arch intrinsics or safe SIMD wrappers (packed_simd / std::simd)",
        ))

    # 7. Lock / atomic overhead
    rust_atomics = _count_pattern(rust_code, r'\b(Mutex|RwLock|AtomicU|AtomicI|Arc)\b')
    c_atomics = _count_pattern(c_code, r'\b(pthread_mutex|atomic_|__sync_)\b')
    if rust_atomics > c_atomics + 2:
        regressions.append(PerfRegression(
            location="synchronization",
            description=(
                f"Rust uses {rust_atomics} synchronization primitives vs {c_atomics} in C. "
                f"Extra locking adds contention and cache-line bouncing."
            ),
            severity=min(0.3 + (rust_atomics - c_atomics) * 0.05, 0.8),
            category="sync_overhead",
            mitigation="Reduce lock granularity; use lock-free structures where appropriate",
        ))

    return regressions


def optimization_opportunity(rust_code: str) -> List[Optimization]:
    """Find where Rust code can be made faster with idiomatic patterns."""
    optimizations: List[Optimization] = []

    # 1. Bounds check elimination via iterators
    rust_loops = _extract_loops_rust(rust_code)
    for i, loop in enumerate(rust_loops):
        index_accesses = _count_pattern(loop, r'\w+\[\w+\]')
        uses_iter = re.search(r'\.\b(iter|into_iter|iter_mut)\b', loop) is not None
        if index_accesses > 0 and not uses_iter:
            optimizations.append(Optimization(
                kind=OptimizationKind.BOUNDS_CHECK_ELIM,
                location=f"loop_{i}",
                description=f"Replace indexed loop with iterator to eliminate {index_accesses} bounds checks",
                estimated_improvement=min(index_accesses * 0.03, 0.15),
                code_suggestion="for item in slice.iter() { /* use item instead of slice[i] */ }",
            ))

    # 2. Iterator fusion
    chain_matches = list(re.finditer(
        r'\.iter\(\)\s*(?:\.\w+\([^)]*\)\s*){3,}', rust_code,
    ))
    for m in chain_matches:
        optimizations.append(Optimization(
            kind=OptimizationKind.ITERATOR_FUSION,
            location=f"char_{m.start()}",
            description="Long iterator chain can be fused by the compiler into a single loop",
            estimated_improvement=0.05,
            code_suggestion="Ensure the chain ends with .collect() or .for_each() for best fusion",
        ))

    # 3. Zero-cost abstractions — trait objects to generics
    dyn_traits = re.findall(r'\bdyn\s+\w+', rust_code)
    if dyn_traits:
        optimizations.append(Optimization(
            kind=OptimizationKind.ZERO_COST_ABSTRACTION,
            location="trait_objects",
            description=f"Found {len(dyn_traits)} dynamic dispatch sites; monomorphize for static dispatch",
            estimated_improvement=min(len(dyn_traits) * 0.02, 0.10),
            code_suggestion="fn process<T: Trait>(item: &T) instead of fn process(item: &dyn Trait)",
        ))

    # 4. SIMD vectorization hints
    numeric_loops = 0
    for loop in rust_loops:
        if re.search(r'[+\-*/]\s*=', loop) and re.search(r'f(32|64)', rust_code):
            numeric_loops += 1
    if numeric_loops > 0:
        optimizations.append(Optimization(
            kind=OptimizationKind.SIMD_VECTORIZATION,
            location="numeric_loops",
            description=f"{numeric_loops} numeric loops may benefit from explicit SIMD or auto-vectorization",
            estimated_improvement=min(numeric_loops * 0.10, 0.40),
            code_suggestion="Use iterators with .chunks_exact(4) or std::simd for explicit vectorization",
        ))

    # 5. Branch elimination via branchless ops
    if_in_loops = 0
    for loop in rust_loops:
        if_in_loops += _count_pattern(loop, r'\bif\s+')
    if if_in_loops > 2:
        optimizations.append(Optimization(
            kind=OptimizationKind.BRANCH_ELIMINATION,
            location="loop_branches",
            description=f"{if_in_loops} conditional branches inside loops; consider branchless alternatives",
            estimated_improvement=min(if_in_loops * 0.02, 0.10),
            code_suggestion="Use bool as usize, conditional moves, or select operations instead of if/else",
        ))

    # 6. Allocation elision
    vec_in_loops = 0
    for loop in rust_loops:
        vec_in_loops += _count_pattern(loop, r'\b(Vec::new|vec!|String::new|to_string)\b')
    if vec_in_loops > 0:
        optimizations.append(Optimization(
            kind=OptimizationKind.ALLOCATION_ELISION,
            location="loop_allocations",
            description=f"{vec_in_loops} heap allocations inside loops; hoist outside or use stack buffers",
            estimated_improvement=min(vec_in_loops * 0.05, 0.25),
            code_suggestion="Move Vec/String creation outside the loop and use .clear() to reuse",
        ))

    # 7. Inline opportunities
    small_fns = re.findall(r'\bfn\s+(\w+)\s*\([^)]*\)[^{]*\{[^}]{1,80}\}', rust_code)
    for fn_name in small_fns:
        call_count = _count_pattern(rust_code, rf'\b{re.escape(fn_name)}\s*\(')
        if call_count > 2:
            optimizations.append(Optimization(
                kind=OptimizationKind.INLINE_OPPORTUNITY,
                location=f"fn_{fn_name}",
                description=f"Small function '{fn_name}' called {call_count} times; mark for inlining",
                estimated_improvement=min(call_count * 0.01, 0.05),
                code_suggestion=f"#[inline] fn {fn_name}(...) or #[inline(always)] for hot paths",
            ))

    # 8. Cache optimization — struct field reordering
    struct_defs = re.findall(
        r'struct\s+(\w+)\s*\{([^}]+)\}', rust_code,
    )
    for name, body in struct_defs:
        fields = re.findall(r'(\w+)\s*:\s*(\w+)', body)
        if len(fields) >= 4:
            sizes = []
            for _, ftype in fields:
                if ftype in ("u8", "i8", "bool"):
                    sizes.append(1)
                elif ftype in ("u16", "i16"):
                    sizes.append(2)
                elif ftype in ("u32", "i32", "f32"):
                    sizes.append(4)
                else:
                    sizes.append(8)
            if sizes != sorted(sizes, reverse=True):
                optimizations.append(Optimization(
                    kind=OptimizationKind.CACHE_OPTIMIZATION,
                    location=f"struct_{name}",
                    description=f"Struct '{name}' fields not ordered by size; may have padding",
                    estimated_improvement=0.02,
                    code_suggestion=f"#[repr(C)] or reorder fields largest-first in struct {name}",
                ))

    return optimizations


def simd_migration(c_code: str) -> str:
    """Convert C SIMD intrinsics to Rust std::arch equivalents."""
    intrinsic_pattern = re.compile(r'\b(_mm\d*_\w+)\s*\(')
    found_intrinsics = set(intrinsic_pattern.findall(c_code))

    if not found_intrinsics:
        return "// No SIMD intrinsics found in the C code.\n"

    required_features: set[str] = set()
    imports: List[str] = []
    unmapped: List[str] = []

    for intr in sorted(found_intrinsics):
        mapping = _SIMD_MAP.get(intr)
        if mapping:
            required_features.add(mapping.feature_gate)
            imports.append(mapping.notes)
        else:
            unmapped.append(intr)

    lines: List[str] = [
        "// Auto-generated Rust SIMD migration",
        "// Source: C code with x86 SIMD intrinsics",
        "",
    ]

    if required_features:
        lines.append("#![allow(unused_imports)]")
        lines.append("")

    for feat in sorted(required_features):
        lines.append(f'#[cfg(target_arch = "x86_64")]')

    lines.append("")

    seen_imports: set[str] = set()
    for imp in sorted(set(imports)):
        if imp not in seen_imports:
            lines.append(imp)
            seen_imports.add(imp)

    lines.append("")

    # Build the target_feature attribute
    feat_list = ", ".join(f'enable = "{f}"' for f in sorted(required_features))
    lines.append(f'#[target_feature({feat_list})]')
    lines.append("pub unsafe fn simd_kernel(")
    lines.append("    input: *const f32,")
    lines.append("    output: *mut f32,")
    lines.append("    len: usize,")
    lines.append(") {")

    # Translate the C function body line by line
    in_function = False
    brace_depth = 0
    for raw_line in c_code.splitlines():
        stripped = raw_line.strip()
        if not in_function:
            if re.match(r'^\w[\w\s*]*\w+\s*\([^)]*\)\s*\{', stripped):
                in_function = True
                brace_depth = 1
            continue

        brace_depth += stripped.count('{') - stripped.count('}')
        if brace_depth <= 0:
            break

        converted = stripped
        # Replace C types
        converted = re.sub(r'\b__m128\b', '__m128', converted)
        converted = re.sub(r'\b__m256\b', '__m256', converted)
        converted = re.sub(r'\bfloat\b', 'f32', converted)
        converted = re.sub(r'\bint\b', 'i32', converted)
        # Replace pointer dereference with offset
        converted = re.sub(r'\*\((\w+)\s*\+\s*(\w+)\)', r'\1.add(\2).read()', converted)
        # Variable declarations
        converted = re.sub(r'^(f32|i32|__m128|__m256)\s+(\w+)\s*=', r'let \2 =', converted)
        converted = re.sub(r'^(f32|i32|__m128|__m256)\s+(\w+)\s*;', r'let \2;', converted)

        lines.append(f"    {converted}")

    lines.append("}")

    if unmapped:
        lines.append("")
        lines.append("// WARNING: The following intrinsics have no automatic mapping:")
        for u in unmapped:
            lines.append(f"//   {u} — manually translate or find a crate equivalent")

    lines.append("")
    return "\n".join(lines)


def analyze_loop_performance(
    c_code: str, rust_code: str,
) -> LoopAnalysis:
    """Compare loop structures and vectorization potential between C and Rust."""
    c_loops = _extract_loops_c(c_code)
    rust_loops = _extract_loops_rust(rust_code)

    vec_c = sum(1 for lp in c_loops if _is_vectorizable_c(lp))
    vec_rust = sum(1 for lp in rust_loops if _is_vectorizable_rust(lp))
    iter_chains = _count_iterator_chains(rust_code)
    bounds_in_loops = _count_bounds_checks_in_loops(rust_code)

    recommendations: List[str] = []
    if bounds_in_loops > 0:
        recommendations.append(
            f"Eliminate {bounds_in_loops} bounds checks by using iterators or slice patterns"
        )
    if vec_c > vec_rust:
        recommendations.append(
            f"C has {vec_c} vectorizable loops vs Rust {vec_rust}; "
            f"refactor Rust loops for better auto-vectorization"
        )
    if iter_chains > 0:
        recommendations.append(
            f"{iter_chains} iterator chains detected — ensure they compile to fused loops"
        )
    if len(c_loops) > 0 and len(rust_loops) == 0:
        recommendations.append("Rust version has no explicit loops; verify iterator perf is adequate")

    # Check for loop unrolling hints
    c_unroll = _count_pattern(c_code, r'#pragma\s+(unroll|GCC\s+unroll)')
    if c_unroll > 0:
        recommendations.append(
            f"C code has {c_unroll} unroll pragmas; Rust relies on LLVM auto-unrolling"
        )

    return LoopAnalysis(
        c_loop_count=len(c_loops),
        rust_loop_count=len(rust_loops),
        vectorizable_c=vec_c,
        vectorizable_rust=vec_rust,
        iterator_chains=iter_chains,
        bounds_checks_in_loops=bounds_in_loops,
        recommendations=recommendations,
    )


def allocation_analysis(c_code: str, rust_code: str) -> Dict:
    """Compare heap allocation patterns between C and Rust code."""
    c_malloc = _count_pattern(c_code, r'\bmalloc\s*\(')
    c_calloc = _count_pattern(c_code, r'\bcalloc\s*\(')
    c_realloc = _count_pattern(c_code, r'\brealloc\s*\(')
    c_free = _count_pattern(c_code, r'\bfree\s*\(')
    c_stack_arrays = _count_pattern(c_code, r'\b\w+\s+\w+\s*\[\s*\d+\s*\]')
    c_total_alloc = c_malloc + c_calloc

    rust_box = _count_pattern(rust_code, r'\bBox::new\b')
    rust_vec = _count_pattern(rust_code, r'\b(Vec::new|vec!)\b')
    rust_string = _count_pattern(rust_code, r'\b(String::new|String::from|to_string|to_owned)\b')
    rust_rc = _count_pattern(rust_code, r'\b(Rc::new|Arc::new)\b')
    rust_with_capacity = _count_pattern(rust_code, r'\bwith_capacity\s*\(')
    rust_stack_arrays = _count_pattern(rust_code, r'\[\s*\w+\s*;\s*\d+\s*\]')
    rust_total_alloc = rust_box + rust_vec + rust_string + rust_rc

    # Leak detection heuristics
    c_potential_leaks = max(0, c_total_alloc - c_free)
    rust_potential_leaks = 0  # Rust's ownership prevents leaks (except Rc cycles)
    rc_count = _count_pattern(rust_code, r'\bRc::new\b')
    if rc_count > 2:
        rust_potential_leaks = 1  # possible Rc cycle

    # Allocation in loops
    c_alloc_in_loop = 0
    for loop in _extract_loops_c(c_code):
        c_alloc_in_loop += _count_pattern(loop, r'\b(malloc|calloc)\s*\(')

    rust_alloc_in_loop = 0
    for loop in _extract_loops_rust(rust_code):
        rust_alloc_in_loop += _count_pattern(loop, r'\b(Vec::new|vec!|Box::new|String::new)\b')

    notes: List[str] = []
    if c_potential_leaks > 0:
        notes.append(f"C code has {c_potential_leaks} unmatched malloc/free pairs (potential leaks)")
    if rust_alloc_in_loop > 0:
        notes.append(f"Rust allocates {rust_alloc_in_loop} times inside loops; consider hoisting")
    if c_alloc_in_loop > 0:
        notes.append(f"C allocates {c_alloc_in_loop} times inside loops")
    if rust_with_capacity > 0:
        notes.append(f"Rust uses with_capacity {rust_with_capacity} times (good: avoids reallocations)")

    overall_verdict = "equivalent"
    if rust_total_alloc < c_total_alloc:
        overall_verdict = "rust_fewer_allocations"
    elif rust_total_alloc > c_total_alloc:
        overall_verdict = "rust_more_allocations"

    return {
        "c_pattern": {
            "malloc": c_malloc,
            "calloc": c_calloc,
            "realloc": c_realloc,
            "free": c_free,
            "stack_arrays": c_stack_arrays,
            "total_heap_allocs": c_total_alloc,
            "alloc_in_loops": c_alloc_in_loop,
            "potential_leaks": c_potential_leaks,
        },
        "rust_pattern": {
            "box_new": rust_box,
            "vec_new": rust_vec,
            "string_new": rust_string,
            "rc_arc_new": rust_rc,
            "with_capacity": rust_with_capacity,
            "stack_arrays": rust_stack_arrays,
            "total_heap_allocs": rust_total_alloc,
            "alloc_in_loops": rust_alloc_in_loop,
            "potential_leaks": rust_potential_leaks,
        },
        "comparison": {
            "verdict": overall_verdict,
            "c_total": c_total_alloc,
            "rust_total": rust_total_alloc,
            "delta": rust_total_alloc - c_total_alloc,
        },
        "notes": notes,
    }


def generate_benchmark_harness(c_code: str, rust_code: str) -> str:
    """Generate a criterion.rs benchmark harness comparing C and Rust implementations."""
    # Extract function signatures from C
    c_fns = re.findall(
        r'(?:static\s+)?(?:inline\s+)?(\w+)\s+(\w+)\s*\(([^)]*)\)\s*\{',
        c_code,
    )
    # Extract function signatures from Rust
    rust_fns = re.findall(
        r'(?:pub\s+)?fn\s+(\w+)\s*\(([^)]*)\)(?:\s*->\s*(\w+))?\s*\{',
        rust_code,
    )

    lines: List[str] = [
        "// Auto-generated benchmark harness for C-to-Rust migration verification",
        "// Uses criterion.rs for statistical benchmarking",
        "",
        "use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};",
        "",
    ]

    # Generate extern C bindings
    if c_fns:
        lines.append('extern "C" {')
        for ret_type, fn_name, params in c_fns:
            rust_params = _convert_c_params_to_rust(params)
            rust_ret = _convert_c_type_to_rust(ret_type)
            ret_clause = f" -> {rust_ret}" if rust_ret != "()" else ""
            lines.append(f"    fn c_{fn_name}({rust_params}){ret_clause};")
        lines.append("}")
        lines.append("")

    # Import Rust functions
    if rust_fns:
        lines.append("// Import Rust implementation")
        lines.append("use crate::{")
        for fn_name, _, _ in rust_fns:
            lines.append(f"    {fn_name},")
        lines.append("};")
        lines.append("")

    # Generate test data setup
    lines.extend([
        "fn setup_test_data(size: usize) -> Vec<f64> {",
        "    (0..size).map(|i| (i as f64) * 0.1).collect()",
        "}",
        "",
        "fn setup_test_ints(size: usize) -> Vec<i32> {",
        "    (0..size).map(|i| i as i32).collect()",
        "}",
        "",
    ])

    # Generate benchmark functions for each pair
    paired = _pair_functions(c_fns, rust_fns)
    for c_fn, rust_fn in paired:
        c_name = c_fn[1] if c_fn else None
        r_name = rust_fn[0] if rust_fn else None
        bench_name = c_name or r_name or "unknown"

        lines.append(f"fn bench_{bench_name}(c: &mut Criterion) {{")
        lines.append(f'    let mut group = c.benchmark_group("{bench_name}");')
        lines.append("")
        lines.append("    for size in [100, 1_000, 10_000, 100_000].iter() {")
        lines.append("        let data = setup_test_data(*size);")
        lines.append("        let int_data = setup_test_ints(*size);")
        lines.append("")

        if c_name:
            lines.append(f'        group.bench_with_input(BenchmarkId::new("c_{c_name}", size), size, |b, _| {{')
            lines.append(f"            b.iter(|| unsafe {{")
            lines.append(f"                c_{c_name}(black_box(data.as_ptr()), black_box(data.len()))")
            lines.append(f"            }});")
            lines.append(f"        }});")
            lines.append("")

        if r_name:
            lines.append(f'        group.bench_with_input(BenchmarkId::new("rust_{r_name}", size), size, |b, _| {{')
            lines.append(f"            b.iter(|| {{")
            lines.append(f"                {r_name}(black_box(&data))")
            lines.append(f"            }});")
            lines.append(f"        }});")

        lines.append("    }")
        lines.append("    group.finish();")
        lines.append("}")
        lines.append("")

    # Generate the criterion_group and criterion_main macros
    bench_fn_names = [f"bench_{c_fn[1] if c_fn else rust_fn[0]}" for c_fn, rust_fn in paired]
    if not bench_fn_names:
        bench_fn_names = ["bench_default"]
        lines.extend([
            "fn bench_default(c: &mut Criterion) {",
            '    c.bench_function("default", |b| b.iter(|| {',
            "        black_box(42)",
            "    }));",
            "}",
            "",
        ])

    fn_list = ", ".join(bench_fn_names)
    lines.append(f"criterion_group!(benches, {fn_list});")
    lines.append("criterion_main!(benches);")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers for benchmark generation
# ---------------------------------------------------------------------------

def _convert_c_type_to_rust(c_type: str) -> str:
    """Map a C type name to its Rust equivalent."""
    c_type = c_type.strip()
    mapping = {
        "void": "()",
        "int": "i32",
        "unsigned": "u32",
        "unsigned int": "u32",
        "long": "i64",
        "unsigned long": "u64",
        "long long": "i64",
        "unsigned long long": "u64",
        "float": "f32",
        "double": "f64",
        "char": "i8",
        "unsigned char": "u8",
        "size_t": "usize",
        "ssize_t": "isize",
        "int8_t": "i8",
        "int16_t": "i16",
        "int32_t": "i32",
        "int64_t": "i64",
        "uint8_t": "u8",
        "uint16_t": "u16",
        "uint32_t": "u32",
        "uint64_t": "u64",
        "bool": "bool",
        "_Bool": "bool",
    }
    # Handle pointer types
    if c_type.endswith("*"):
        inner = c_type[:-1].strip()
        rust_inner = _convert_c_type_to_rust(inner)
        if rust_inner == "()":
            return "*mut std::ffi::c_void"
        return f"*mut {rust_inner}"
    if "const" in c_type and "*" in c_type:
        inner = c_type.replace("const", "").replace("*", "").strip()
        rust_inner = _convert_c_type_to_rust(inner)
        return f"*const {rust_inner}"
    return mapping.get(c_type, c_type)


def _convert_c_params_to_rust(params: str) -> str:
    """Convert a C parameter list to Rust syntax."""
    if not params.strip() or params.strip() == "void":
        return ""
    parts = []
    for param in params.split(","):
        param = param.strip()
        if not param:
            continue
        # Try to split into type and name
        tokens = param.rsplit(None, 1)
        if len(tokens) == 2:
            c_type, name = tokens
            # Handle pointer in name (e.g., "float *data")
            if name.startswith("*"):
                name = name[1:]
                c_type = c_type + " *"
            rust_type = _convert_c_type_to_rust(c_type)
            parts.append(f"{name}: {rust_type}")
        else:
            rust_type = _convert_c_type_to_rust(tokens[0])
            parts.append(f"_arg: {rust_type}")
    return ", ".join(parts)


def _pair_functions(
    c_fns: List[tuple],
    rust_fns: List[tuple],
) -> List[Tuple[Optional[tuple], Optional[tuple]]]:
    """Pair C functions with their Rust equivalents by name similarity."""
    paired: List[Tuple[Optional[tuple], Optional[tuple]]] = []
    used_rust: set[int] = set()

    for c_fn in c_fns:
        c_name = c_fn[1].lower()
        best_match: Optional[int] = None
        best_score = 0.0
        for j, r_fn in enumerate(rust_fns):
            if j in used_rust:
                continue
            r_name = r_fn[0].lower()
            score = _name_similarity(c_name, r_name)
            if score > best_score:
                best_score = score
                best_match = j
        if best_match is not None and best_score > 0.3:
            used_rust.add(best_match)
            paired.append((c_fn, rust_fns[best_match]))
        else:
            paired.append((c_fn, None))

    for j, r_fn in enumerate(rust_fns):
        if j not in used_rust:
            paired.append((None, r_fn))

    return paired


def _name_similarity(a: str, b: str) -> float:
    """Compute simple name similarity between two identifiers."""
    if a == b:
        return 1.0
    # Normalize underscores
    a_parts = set(a.replace("_", " ").split())
    b_parts = set(b.replace("_", " ").split())
    if not a_parts or not b_parts:
        return 0.0
    intersection = a_parts & b_parts
    union = a_parts | b_parts
    return len(intersection) / len(union)
