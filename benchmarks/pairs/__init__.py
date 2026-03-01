"""Benchmark pairs for cross-language equivalence verification."""
from .benchmark_pairs import (
    ALL_BENCHMARKS, BenchmarkPair,
    get_benchmarks_by_category, get_equivalent_benchmarks,
    get_divergent_benchmarks, get_all_categories,
    ARITHMETIC_PAIRS, DIVISION_SHIFT_PAIRS, LOOP_PAIRS,
    ERROR_HANDLING_PAIRS, BITWISE_PAIRS, STRING_MEMORY_PAIRS,
)

__all__ = [
    "ALL_BENCHMARKS", "BenchmarkPair",
    "get_benchmarks_by_category", "get_equivalent_benchmarks",
    "get_divergent_benchmarks", "get_all_categories",
]
