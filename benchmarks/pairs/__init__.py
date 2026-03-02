"""Benchmark pairs for cross-language equivalence verification."""
from .benchmark_pairs import (
    ALL_BENCHMARKS, BenchmarkPair,
    get_benchmarks_by_category, get_equivalent_benchmarks,
    get_divergent_benchmarks, get_all_categories,
    ARITHMETIC_PAIRS, DIVISION_SHIFT_PAIRS, LOOP_PAIRS,
    ERROR_HANDLING_PAIRS, BITWISE_PAIRS, STRING_MEMORY_PAIRS,
)
from .expanded_benchmark_pairs import (
    EXPANDED_BENCHMARKS,
    STRUCT_PAIRS, ENUM_PAIRS, FLOAT_PAIRS, C2RUST_PAIRS,
    ITERATOR_PAIRS, CAST_PAIRS, COMPOUND_PAIRS, CONTROL_FLOW_PAIRS,
    get_expanded_benchmarks, get_expanded_by_category,
    get_expanded_categories,
)
from .memory_benchmark_pairs import (
    MEMORY_PAIRS, SCALED_MEMORY_PAIRS, get_all_memory_pairs,
)
from .c2rust_benchmark_pairs import (
    C2RUST_REALISTIC_PAIRS, get_all_c2rust_pairs,
)

# Combined: original 52 + expanded 150 + memory 20 + c2rust_realistic 13 = 235+ pairs
COMBINED_BENCHMARKS = ALL_BENCHMARKS + EXPANDED_BENCHMARKS + MEMORY_PAIRS + SCALED_MEMORY_PAIRS + C2RUST_REALISTIC_PAIRS

__all__ = [
    "ALL_BENCHMARKS", "EXPANDED_BENCHMARKS", "COMBINED_BENCHMARKS",
    "BenchmarkPair",
    "get_benchmarks_by_category", "get_equivalent_benchmarks",
    "get_divergent_benchmarks", "get_all_categories",
    "get_expanded_benchmarks", "get_expanded_by_category",
    "get_expanded_categories",
    "STRUCT_PAIRS", "ENUM_PAIRS", "FLOAT_PAIRS", "C2RUST_PAIRS",
    "ITERATOR_PAIRS", "CAST_PAIRS", "COMPOUND_PAIRS", "CONTROL_FLOW_PAIRS",
    "MEMORY_PAIRS", "SCALED_MEMORY_PAIRS", "get_all_memory_pairs",
    "C2RUST_REALISTIC_PAIRS", "get_all_c2rust_pairs",
]
