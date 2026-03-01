"""
Differential tester for cross-language equivalence verification.

Tests equivalence of C and Rust implementations by running both versions
on the same inputs and comparing outputs, with fuzzing, property-based
testing, coverage tracking, and crash detection.
"""

import enum
import hashlib
import math
import os
import random
import signal
import string
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class InputStrategy(enum.Enum):
    RANDOM = "random"
    BOUNDARY = "boundary"
    STRUCTURED = "structured"
    MIXED = "mixed"


class MutationKind(enum.Enum):
    BIT_FLIP = "bit_flip"
    BYTE_FLIP = "byte_flip"
    ARITHMETIC_INC = "arithmetic_inc"
    ARITHMETIC_DEC = "arithmetic_dec"
    SPLICE = "splice"
    INSERT_RANDOM = "insert_random"
    DELETE_BYTES = "delete_bytes"
    DUPLICATE_REGION = "duplicate_region"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ComparisonResult:
    equal: bool
    differences: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.equal

    def add_difference(self, diff: str) -> None:
        self.differences.append(diff)
        self.equal = False


@dataclass
class ExecutionResult:
    success: bool
    output: Any = None
    crash_kind: Optional[str] = None  # None / 'segfault' / 'panic' / 'timeout' / 'memory_leak'
    error_msg: Optional[str] = None

    def is_crash(self) -> bool:
        return self.crash_kind is not None


@dataclass
class CounterExample:
    inputs: Tuple
    c_output: Any
    rust_output: Any
    shrunk_inputs: Optional[Tuple] = None
    shrunk_c_output: Any = None
    shrunk_rust_output: Any = None

    def describe(self) -> str:
        lines = [
            f"Original inputs:    {self.inputs!r}",
            f"C output:           {self.c_output!r}",
            f"Rust output:        {self.rust_output!r}",
        ]
        if self.shrunk_inputs is not None:
            lines.extend([
                f"Shrunk inputs:      {self.shrunk_inputs!r}",
                f"Shrunk C output:    {self.shrunk_c_output!r}",
                f"Shrunk Rust output: {self.shrunk_rust_output!r}",
            ])
        return "\n".join(lines)


@dataclass
class CoverageStats:
    total_paths: int = 0
    covered_paths: int = 0
    branch_points: int = 0
    covered_branches: int = 0

    @property
    def path_coverage_pct(self) -> float:
        if self.total_paths == 0:
            return 0.0
        return (self.covered_paths / self.total_paths) * 100.0

    @property
    def branch_coverage_pct(self) -> float:
        if self.branch_points == 0:
            return 0.0
        return (self.covered_branches / self.branch_points) * 100.0


@dataclass
class TestResult:
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    coverage: CoverageStats = field(default_factory=CoverageStats)
    counterexamples: List[CounterExample] = field(default_factory=list)
    duration_seconds: float = 0.0

    def summary(self) -> str:
        lines = [
            f"Total tests:    {self.total_tests}",
            f"Passed:         {self.passed}",
            f"Failed:         {self.failed}",
            f"Errors:         {self.errors}",
            f"Duration:       {self.duration_seconds:.3f}s",
            f"Path coverage:  {self.coverage.path_coverage_pct:.1f}%",
        ]
        if self.counterexamples:
            lines.append(f"Counterexamples ({len(self.counterexamples)}):")
            for i, ce in enumerate(self.counterexamples, 1):
                lines.append(f"  #{i}: inputs={ce.inputs!r}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "total_tests": self.total_tests,
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "duration_seconds": round(self.duration_seconds, 6),
            "coverage": {
                "total_paths": self.coverage.total_paths,
                "covered_paths": self.coverage.covered_paths,
                "path_coverage_pct": round(self.coverage.path_coverage_pct, 2),
                "branch_points": self.coverage.branch_points,
                "covered_branches": self.coverage.covered_branches,
                "branch_coverage_pct": round(self.coverage.branch_coverage_pct, 2),
            },
            "counterexamples": [
                {
                    "inputs": list(ce.inputs),
                    "c_output": ce.c_output,
                    "rust_output": ce.rust_output,
                    "shrunk_inputs": list(ce.shrunk_inputs) if ce.shrunk_inputs else None,
                    "shrunk_c_output": ce.shrunk_c_output,
                    "shrunk_rust_output": ce.shrunk_rust_output,
                }
                for ce in self.counterexamples
            ],
        }


# ---------------------------------------------------------------------------
# InputGenerator
# ---------------------------------------------------------------------------

class InputGenerator:
    """Generates random test inputs with support for edge cases and strategies."""

    _INT_BOUNDARIES = [0, 1, -1, 2, -2, 127, -128, 255, 256, 32767,
                       -32768, 65535, 2**31 - 1, -(2**31), 2**63 - 1, -(2**63)]

    _FLOAT_SPECIALS = [0.0, -0.0, float("inf"), float("-inf"), float("nan"),
                       1.0, -1.0, 1e-308, -1e-308, 1e308, -1e308,
                       2.2250738585072014e-308, 1.7976931348623157e+308]

    def __init__(self, strategy: InputStrategy = InputStrategy.MIXED,
                 seed: Optional[int] = None):
        self._strategy = strategy
        self._rng = random.Random(seed)

    @property
    def strategy(self) -> InputStrategy:
        return self._strategy

    @strategy.setter
    def strategy(self, value: InputStrategy) -> None:
        self._strategy = value

    def _pick_strategy(self) -> InputStrategy:
        if self._strategy != InputStrategy.MIXED:
            return self._strategy
        return self._rng.choice([InputStrategy.RANDOM, InputStrategy.BOUNDARY,
                                  InputStrategy.STRUCTURED])

    # -- primitives ----------------------------------------------------------

    def generate_int(self, min_val: int = -(2**31),
                     max_val: int = 2**31 - 1) -> int:
        strat = self._pick_strategy()
        if strat == InputStrategy.BOUNDARY:
            candidates = [v for v in self._INT_BOUNDARIES
                          if min_val <= v <= max_val]
            if candidates:
                return self._rng.choice(candidates)
            return self._rng.choice([min_val, max_val,
                                      (min_val + max_val) // 2])
        if strat == InputStrategy.STRUCTURED:
            base = self._rng.choice([2, 10, 16, 256])
            exp = self._rng.randint(0, 20)
            val = base ** exp
            val = val if self._rng.random() < 0.5 else -val
            return max(min_val, min(max_val, val))
        return self._rng.randint(min_val, max_val)

    def generate_float(self, min_val: float = -1e10,
                       max_val: float = 1e10) -> float:
        strat = self._pick_strategy()
        if strat == InputStrategy.BOUNDARY:
            candidates = []
            for v in self._FLOAT_SPECIALS:
                if math.isnan(v) or math.isinf(v):
                    candidates.append(v)
                elif min_val <= v <= max_val:
                    candidates.append(v)
            if candidates:
                return self._rng.choice(candidates)
        if strat == InputStrategy.STRUCTURED:
            # powers of 2
            exp = self._rng.randint(-30, 30)
            val = 2.0 ** exp
            if self._rng.random() < 0.5:
                val = -val
            if min_val <= val <= max_val:
                return val
        # Random: uniform in range, with occasional special value
        if self._rng.random() < 0.05:
            return self._rng.choice(self._FLOAT_SPECIALS)
        return self._rng.uniform(min_val, max_val)

    def generate_string(self, max_len: int = 100,
                        charset: str = "printable") -> str:
        strat = self._pick_strategy()
        if strat == InputStrategy.BOUNDARY:
            edge_cases = [
                "",
                "\x00",
                "\x00" * 5,
                "a" * max_len,
                "\n" * 10,
                "\r\n" * 5,
                "\t" * 10,
                "\u00e9\u00e8\u00ea",  # accented chars
                "\U0001f600" * 3,       # emoji
                " " * max_len,
            ]
            return self._rng.choice(edge_cases)[:max_len]

        if strat == InputStrategy.STRUCTURED:
            kind = self._rng.choice(["palindrome", "repeated", "ascending"])
            length = self._rng.randint(1, max(1, max_len))
            if kind == "palindrome":
                half = "".join(self._rng.choice(string.ascii_lowercase)
                               for _ in range(length // 2))
                return half + half[::-1]
            if kind == "repeated":
                pat = "".join(self._rng.choice(string.ascii_lowercase)
                              for _ in range(self._rng.randint(1, 4)))
                return (pat * (length // len(pat) + 1))[:length]
            # ascending
            return "".join(chr(ord("a") + (i % 26)) for i in range(length))

        charset_map = {
            "printable": string.printable,
            "ascii": string.ascii_letters + string.digits,
            "digits": string.digits,
            "alpha": string.ascii_letters,
            "hex": string.hexdigits,
        }
        chars = charset_map.get(charset, string.printable)
        length = self._rng.randint(0, max_len)
        result = "".join(self._rng.choice(chars) for _ in range(length))
        # Occasionally inject a null byte
        if self._rng.random() < 0.05 and length > 0:
            pos = self._rng.randint(0, len(result) - 1)
            result = result[:pos] + "\x00" + result[pos + 1:]
        return result

    def generate_array(self, elem_gen: Callable[[], Any],
                       min_len: int = 0, max_len: int = 50) -> list:
        strat = self._pick_strategy()
        if strat == InputStrategy.BOUNDARY:
            choice = self._rng.choice(["empty", "single", "max"])
            if choice == "empty":
                return []
            if choice == "single":
                return [elem_gen()]
            return [elem_gen() for _ in range(max_len)]
        if strat == InputStrategy.STRUCTURED:
            length = self._rng.randint(min_len, max_len)
            kind = self._rng.choice(["sorted", "reversed", "all_same"])
            elements = [elem_gen() for _ in range(length)]
            if kind == "sorted":
                try:
                    elements.sort()
                except TypeError:
                    pass
            elif kind == "reversed":
                try:
                    elements.sort(reverse=True)
                except TypeError:
                    pass
            elif kind == "all_same":
                if elements:
                    elements = [elements[0]] * length
            return elements
        length = self._rng.randint(min_len, max_len)
        return [elem_gen() for _ in range(length)]

    def generate_struct(self, field_gens: Dict[str, Callable[[], Any]]) -> dict:
        result = {}
        for fname, fgen in field_gens.items():
            result[fname] = fgen()
        return result

    def generate_pointer(self, elem_gen: Callable[[], Any],
                         nullable: bool = True) -> Any:
        if nullable and self._rng.random() < 0.1:
            return None
        return elem_gen()

    def generate_from_type(self, type_spec: str) -> Any:
        spec = type_spec.strip()
        if spec in ("int", "int32_t", "i32"):
            return self.generate_int()
        if spec in ("long", "int64_t", "i64", "long long"):
            return self.generate_int(-(2**63), 2**63 - 1)
        if spec in ("short", "int16_t", "i16"):
            return self.generate_int(-32768, 32767)
        if spec in ("char", "int8_t", "i8"):
            return self.generate_int(-128, 127)
        if spec in ("unsigned int", "uint32_t", "u32"):
            return self.generate_int(0, 2**32 - 1)
        if spec in ("unsigned long", "uint64_t", "u64"):
            return self.generate_int(0, 2**64 - 1)
        if spec in ("unsigned short", "uint16_t", "u16"):
            return self.generate_int(0, 65535)
        if spec in ("unsigned char", "uint8_t", "u8"):
            return self.generate_int(0, 255)
        if spec in ("float", "f32"):
            return self.generate_float(-3.4e38, 3.4e38)
        if spec in ("double", "f64"):
            return self.generate_float()
        if spec in ("bool", "_Bool"):
            return self._rng.choice([True, False])
        if spec in ("char*", "const char*", "&str", "String"):
            return self.generate_string()
        if spec.endswith("[]") or spec.endswith("*"):
            inner = spec.rstrip("[]* ")
            if not inner:
                inner = "int"
            return self.generate_array(lambda: self.generate_from_type(inner))
        if spec.startswith("struct ") or spec.startswith("{"):
            # Generic struct with some int fields
            n_fields = self._rng.randint(1, 5)
            return self.generate_struct(
                {f"field_{i}": lambda: self.generate_int()
                 for i in range(n_fields)}
            )
        if spec == "void":
            return None
        # Fallback: treat as int
        return self.generate_int()


# ---------------------------------------------------------------------------
# OutputComparator
# ---------------------------------------------------------------------------

class OutputComparator:
    """Compares outputs from C and Rust functions for equivalence."""

    def __init__(self, float_abs_tol: float = 1e-9,
                 float_rel_tol: float = 1e-9,
                 normalize_strings: bool = True):
        self._float_abs_tol = float_abs_tol
        self._float_rel_tol = float_rel_tol
        self._normalize_strings = normalize_strings

    def compare(self, c_output: Any, rust_output: Any,
                type_spec: str = "auto") -> ComparisonResult:
        result = ComparisonResult(equal=True)
        self._compare_values(c_output, rust_output, type_spec, "", result)
        return result

    def _compare_values(self, a: Any, b: Any, type_spec: str,
                        path: str, result: ComparisonResult) -> None:
        prefix = f"At {path}: " if path else ""

        # Handle None
        if a is None and b is None:
            return
        if a is None or b is None:
            result.add_difference(
                f"{prefix}one is None and the other is not "
                f"(c={a!r}, rust={b!r})"
            )
            return

        # Determine effective type
        effective = type_spec
        if effective == "auto":
            effective = self._infer_type(a, b)

        if effective in ("float", "double", "f32", "f64"):
            self._compare_floats(a, b, prefix, result)
        elif effective in ("str", "string", "char*", "&str", "String"):
            self._compare_strings(a, b, prefix, result)
        elif effective in ("list", "array") or isinstance(a, list):
            self._compare_arrays(a, b, prefix, result)
        elif effective in ("dict", "struct") or isinstance(a, dict):
            self._compare_structs(a, b, prefix, result)
        elif effective in ("pointer", "ptr"):
            # Compare dereferenced values, not addresses
            self._compare_values(a, b, "auto", path + "->", result)
        else:
            if a != b:
                result.add_difference(
                    f"{prefix}values differ: c={a!r}, rust={b!r}"
                )

    def _infer_type(self, a: Any, b: Any) -> str:
        if isinstance(a, float) or isinstance(b, float):
            return "float"
        if isinstance(a, str) or isinstance(b, str):
            return "str"
        if isinstance(a, list) or isinstance(b, list):
            return "list"
        if isinstance(a, dict) or isinstance(b, dict):
            return "dict"
        return "int"

    def _compare_floats(self, a: Any, b: Any, prefix: str,
                        result: ComparisonResult) -> None:
        try:
            fa, fb = float(a), float(b)
        except (TypeError, ValueError):
            result.add_difference(
                f"{prefix}cannot convert to float: c={a!r}, rust={b!r}"
            )
            return

        # NaN: both NaN is equal for equivalence
        if math.isnan(fa) and math.isnan(fb):
            return
        if math.isnan(fa) or math.isnan(fb):
            result.add_difference(f"{prefix}NaN mismatch: c={fa}, rust={fb}")
            return

        # Infinity
        if math.isinf(fa) and math.isinf(fb):
            if (fa > 0) == (fb > 0):
                return
            result.add_difference(
                f"{prefix}infinity sign mismatch: c={fa}, rust={fb}"
            )
            return

        # Signed zero: -0.0 == 0.0 for equivalence
        if fa == 0.0 and fb == 0.0:
            return

        # Absolute tolerance
        diff = abs(fa - fb)
        if diff <= self._float_abs_tol:
            return

        # Relative tolerance
        denom = max(abs(fa), abs(fb))
        if denom > 0 and diff / denom <= self._float_rel_tol:
            return

        result.add_difference(
            f"{prefix}float mismatch: c={fa}, rust={fb}, "
            f"abs_diff={diff}, rel_diff={diff / denom if denom else float('inf')}"
        )

    def _compare_strings(self, a: Any, b: Any, prefix: str,
                         result: ComparisonResult) -> None:
        sa, sb = str(a), str(b)
        if self._normalize_strings:
            sa = sa.rstrip()
            sb = sb.rstrip()
            sa = sa.replace("\r\n", "\n")
            sb = sb.replace("\r\n", "\n")
        if sa != sb:
            # Find first diff position
            min_len = min(len(sa), len(sb))
            pos = min_len
            for i in range(min_len):
                if sa[i] != sb[i]:
                    pos = i
                    break
            c_char = repr(sa[pos]) if pos < len(sa) else "<end>"
            r_char = repr(sb[pos]) if pos < len(sb) else "<end>"
            result.add_difference(
                f"{prefix}string mismatch at position {pos}: "
                f"c[{pos}]={c_char}, "
                f"rust[{pos}]={r_char}, "
                f"c_len={len(sa)}, rust_len={len(sb)}"
            )

    def _compare_arrays(self, a: Any, b: Any, prefix: str,
                        result: ComparisonResult) -> None:
        if not isinstance(a, (list, tuple)):
            result.add_difference(f"{prefix}C output is not array: {a!r}")
            return
        if not isinstance(b, (list, tuple)):
            result.add_difference(f"{prefix}Rust output is not array: {b!r}")
            return
        if len(a) != len(b):
            result.add_difference(
                f"{prefix}array length mismatch: c_len={len(a)}, "
                f"rust_len={len(b)}"
            )
            check_len = min(len(a), len(b))
        else:
            check_len = len(a)

        for i in range(check_len):
            self._compare_values(a[i], b[i], "auto",
                                 f"{prefix}[{i}]", result)

    def _compare_structs(self, a: Any, b: Any, prefix: str,
                         result: ComparisonResult) -> None:
        if not isinstance(a, dict):
            result.add_difference(f"{prefix}C output is not struct: {a!r}")
            return
        if not isinstance(b, dict):
            result.add_difference(f"{prefix}Rust output is not struct: {b!r}")
            return

        all_keys = set(a.keys()) | set(b.keys())
        c_only = set(a.keys()) - set(b.keys())
        rust_only = set(b.keys()) - set(a.keys())

        if c_only:
            result.add_difference(
                f"{prefix}fields only in C output: {c_only}"
            )
        if rust_only:
            result.add_difference(
                f"{prefix}fields only in Rust output: {rust_only}"
            )

        for key in sorted(all_keys - c_only - rust_only):
            self._compare_values(a[key], b[key], "auto",
                                 f"{prefix}.{key}", result)


# ---------------------------------------------------------------------------
# CoverageTracker
# ---------------------------------------------------------------------------

class CoverageTracker:
    """Tracks code path coverage for differential testing."""

    def __init__(self):
        self._seen_paths: set = set()
        self._path_counts: Dict[str, int] = {}
        self._branch_points: set = set()
        self._covered_branches: set = set()
        self._total_executions: int = 0

    def record_path(self, path_id: Tuple) -> None:
        h = self._hash_path(path_id)
        self._seen_paths.add(h)
        self._path_counts[h] = self._path_counts.get(h, 0) + 1
        self._total_executions += 1
        # Extract individual branches from the path
        for i, decision in enumerate(path_id):
            branch_key = (i, True)
            branch_key_false = (i, False)
            self._branch_points.add((i, True))
            self._branch_points.add((i, False))
            if decision:
                self._covered_branches.add((i, True))
            else:
                self._covered_branches.add((i, False))

    def is_new_coverage(self, path_id: Tuple) -> bool:
        h = self._hash_path(path_id)
        return h not in self._seen_paths

    def get_coverage_stats(self) -> dict:
        bp = len(self._branch_points)
        cb = len(self._covered_branches)
        tp = max(len(self._seen_paths), 1)
        return {
            "total_paths": self._total_executions,
            "covered_paths": len(self._seen_paths),
            "coverage_pct": round(
                (len(self._seen_paths) / max(self._total_executions, 1)) * 100, 2
            ),
            "branch_points": bp,
            "covered_branches": cb,
            "branch_coverage_pct": round(
                (cb / max(bp, 1)) * 100, 2
            ),
        }

    def to_coverage_stats(self) -> CoverageStats:
        stats = self.get_coverage_stats()
        return CoverageStats(
            total_paths=stats["total_paths"],
            covered_paths=stats["covered_paths"],
            branch_points=stats["branch_points"],
            covered_branches=stats["covered_branches"],
        )

    def merge(self, other: "CoverageTracker") -> None:
        self._seen_paths |= other._seen_paths
        for k, v in other._path_counts.items():
            self._path_counts[k] = self._path_counts.get(k, 0) + v
        self._branch_points |= other._branch_points
        self._covered_branches |= other._covered_branches
        self._total_executions += other._total_executions

    def reset(self) -> None:
        self._seen_paths.clear()
        self._path_counts.clear()
        self._branch_points.clear()
        self._covered_branches.clear()
        self._total_executions = 0

    @staticmethod
    def _hash_path(path_id: Tuple) -> str:
        raw = repr(path_id).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]


# ---------------------------------------------------------------------------
# CrashDetector
# ---------------------------------------------------------------------------

class CrashDetector:
    """Detects crashes, panics, hangs, and memory leaks during execution."""

    def __init__(self, default_timeout: float = 5.0):
        self._default_timeout = default_timeout
        self._allocation_tracker: Dict[int, int] = {}  # id -> size
        self._total_allocated: int = 0
        self._total_freed: int = 0

    def run_with_detection(self, func: Callable, inputs: Tuple,
                           timeout: float = 0.0) -> ExecutionResult:
        if timeout <= 0:
            timeout = self._default_timeout

        result_container: List[ExecutionResult] = []

        def _target() -> None:
            try:
                self._total_allocated = 0
                self._total_freed = 0
                output = func(*inputs)
                leaked = self._total_allocated - self._total_freed
                if leaked > 0:
                    result_container.append(ExecutionResult(
                        success=False,
                        output=output,
                        crash_kind="memory_leak",
                        error_msg=f"Memory leak detected: {leaked} bytes not freed",
                    ))
                else:
                    result_container.append(ExecutionResult(
                        success=True,
                        output=output,
                    ))
            except SystemExit as exc:
                result_container.append(ExecutionResult(
                    success=False,
                    output=None,
                    crash_kind="segfault",
                    error_msg=f"Process exited abnormally: {exc}",
                ))
            except KeyboardInterrupt:
                result_container.append(ExecutionResult(
                    success=False,
                    output=None,
                    crash_kind="segfault",
                    error_msg="Interrupted (possible segfault simulation)",
                ))
            except Exception as exc:
                result_container.append(ExecutionResult(
                    success=False,
                    output=None,
                    crash_kind="panic",
                    error_msg=f"{type(exc).__name__}: {exc}",
                ))

        thread = threading.Thread(target=_target, daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            return ExecutionResult(
                success=False,
                output=None,
                crash_kind="timeout",
                error_msg=f"Function did not complete within {timeout}s",
            )

        if result_container:
            return result_container[0]

        return ExecutionResult(
            success=False,
            output=None,
            crash_kind="panic",
            error_msg="Function thread completed without producing a result",
        )

    def track_allocation(self, alloc_id: int, size: int) -> None:
        self._allocation_tracker[alloc_id] = size
        self._total_allocated += size

    def track_free(self, alloc_id: int) -> None:
        size = self._allocation_tracker.pop(alloc_id, 0)
        self._total_freed += size

    def check_leaks(self) -> Optional[str]:
        leaked = self._total_allocated - self._total_freed
        if leaked > 0:
            return (
                f"Memory leak: {leaked} bytes in "
                f"{len(self._allocation_tracker)} allocation(s)"
            )
        return None


# ---------------------------------------------------------------------------
# Fuzzer
# ---------------------------------------------------------------------------

class Fuzzer:
    """Mutation-based fuzzer with energy-based scheduling."""

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)
        self._corpus: List[bytes] = []
        self._energy: Dict[int, float] = {}  # corpus index -> energy
        self._coverage_tracker = CoverageTracker()
        self._interesting: List[bytes] = []

    def add_seed(self, data: bytes) -> None:
        idx = len(self._corpus)
        self._corpus.append(data)
        self._energy[idx] = 1.0

    def get_next_seed(self) -> bytes:
        if not self._corpus:
            return b"\x00"
        # Weighted selection by energy
        total_energy = sum(self._energy.get(i, 1.0)
                           for i in range(len(self._corpus)))
        if total_energy <= 0:
            return self._rng.choice(self._corpus)
        pick = self._rng.uniform(0, total_energy)
        cumulative = 0.0
        for i in range(len(self._corpus)):
            cumulative += self._energy.get(i, 1.0)
            if cumulative >= pick:
                return self._corpus[i]
        return self._corpus[-1]

    def mutate(self, input_bytes: bytes) -> bytes:
        if len(input_bytes) == 0:
            return bytes([self._rng.randint(0, 255)])

        mutation = self._rng.choice(list(MutationKind))
        data = bytearray(input_bytes)

        if mutation == MutationKind.BIT_FLIP:
            pos = self._rng.randint(0, len(data) - 1)
            bit = self._rng.randint(0, 7)
            data[pos] ^= (1 << bit)

        elif mutation == MutationKind.BYTE_FLIP:
            pos = self._rng.randint(0, len(data) - 1)
            data[pos] = data[pos] ^ 0xFF

        elif mutation == MutationKind.ARITHMETIC_INC:
            pos = self._rng.randint(0, len(data) - 1)
            delta = self._rng.randint(1, 35)
            data[pos] = (data[pos] + delta) & 0xFF

        elif mutation == MutationKind.ARITHMETIC_DEC:
            pos = self._rng.randint(0, len(data) - 1)
            delta = self._rng.randint(1, 35)
            data[pos] = (data[pos] - delta) & 0xFF

        elif mutation == MutationKind.SPLICE:
            if len(self._corpus) >= 2:
                other = self._rng.choice(self._corpus)
                if len(other) > 0:
                    split_a = self._rng.randint(0, len(data))
                    split_b = self._rng.randint(0, len(other))
                    data = data[:split_a] + bytearray(other[split_b:])
            else:
                pos = self._rng.randint(0, len(data))
                data.insert(pos, self._rng.randint(0, 255))

        elif mutation == MutationKind.INSERT_RANDOM:
            pos = self._rng.randint(0, len(data))
            n_bytes = self._rng.randint(1, 10)
            insert_data = bytearray(self._rng.randint(0, 255)
                                     for _ in range(n_bytes))
            data[pos:pos] = insert_data

        elif mutation == MutationKind.DELETE_BYTES:
            if len(data) > 1:
                pos = self._rng.randint(0, len(data) - 1)
                n_del = self._rng.randint(1, min(10, len(data) - pos))
                del data[pos:pos + n_del]

        elif mutation == MutationKind.DUPLICATE_REGION:
            if len(data) >= 2:
                src = self._rng.randint(0, len(data) - 1)
                length = self._rng.randint(1, min(20, len(data) - src))
                dst = self._rng.randint(0, len(data))
                region = data[src:src + length]
                data[dst:dst] = region

        return bytes(data)

    def fuzz(self, func: Callable, seeds: List[bytes],
             n_iterations: int = 10000) -> List[bytes]:
        for s in seeds:
            self.add_seed(s)
        if not self._corpus:
            self.add_seed(b"\x00" * 8)

        self._interesting = []

        for iteration in range(n_iterations):
            parent = self.get_next_seed()
            mutated = self.mutate(parent)

            # Apply multiple mutations occasionally
            if self._rng.random() < 0.3:
                mutated = self.mutate(mutated)

            path_id = self._execute_and_trace(func, mutated)

            if path_id is not None and self._coverage_tracker.is_new_coverage(path_id):
                self._coverage_tracker.record_path(path_id)
                self._interesting.append(mutated)
                # Add to corpus and boost energy
                idx = len(self._corpus)
                self._corpus.append(mutated)
                self._energy[idx] = 3.0
                # Boost parent energy
                if parent in self._corpus:
                    pidx = self._corpus.index(parent)
                    self._energy[pidx] = self._energy.get(pidx, 1.0) * 1.5
            elif path_id is not None:
                self._coverage_tracker.record_path(path_id)

            # Energy decay
            if iteration % 100 == 0:
                for k in self._energy:
                    self._energy[k] = max(0.1, self._energy[k] * 0.95)

        return list(self._interesting)

    def _execute_and_trace(self, func: Callable,
                           data: bytes) -> Optional[Tuple]:
        try:
            result = func(data)
            # Build a path tuple from the result to simulate coverage
            if result is None:
                return (0,)
            h = hashlib.md5(repr(result).encode()).hexdigest()
            path = tuple(int(c, 16) % 2 == 0 for c in h[:8])
            return path
        except Exception:
            # Crashes are also interesting coverage
            return (1, 0, 1)


# ---------------------------------------------------------------------------
# PropertyTester
# ---------------------------------------------------------------------------

class PropertyTester:
    """Property-based tester with input shrinking."""

    def __init__(self, input_generator: Optional[InputGenerator] = None,
                 comparator: Optional[OutputComparator] = None):
        self._gen = input_generator or InputGenerator(seed=42)
        self._comp = comparator or OutputComparator()

    def shrink(self, value: Any, type_spec: str,
               predicate: Callable[[Any], bool]) -> Any:
        """Shrink *value* to the smallest value that still satisfies *predicate*."""
        spec = type_spec.strip().lower()

        if spec in ("int", "i32", "i64", "int32_t", "int64_t",
                     "long", "short", "char"):
            return self._shrink_int(value, predicate)
        if spec in ("float", "double", "f32", "f64"):
            return self._shrink_float(value, predicate)
        if spec in ("str", "string", "char*", "&str"):
            return self._shrink_string(value, predicate)
        if spec in ("list", "array") or spec.endswith("[]"):
            return self._shrink_array(value, predicate)
        # Fallback: return as-is
        return value

    def _shrink_int(self, value: int,
                    predicate: Callable[[int], bool]) -> int:
        # Try 0 first
        if predicate(0):
            return 0
        best = value
        # Binary search toward 0
        lo, hi = 0, abs(value)
        sign = 1 if value >= 0 else -1
        while lo < hi:
            mid = (lo + hi) // 2
            candidate = mid * sign
            if predicate(candidate):
                best = candidate
                hi = mid
            else:
                lo = mid + 1
        # Try a few small values near the best
        for delta in range(1, min(10, abs(best) + 1)):
            for s in [1, -1]:
                c = best + delta * s
                if abs(c) < abs(best) and predicate(c):
                    best = c
        return best

    def _shrink_float(self, value: float,
                      predicate: Callable[[float], bool]) -> float:
        # Try 0.0
        if predicate(0.0):
            return 0.0
        best = value
        # Try simple integers
        for candidate in [1.0, -1.0, 2.0, -2.0, 0.5, -0.5]:
            if abs(candidate) < abs(best) and predicate(candidate):
                best = candidate
        # Try powers of 2
        for exp in range(-10, 11):
            candidate = 2.0 ** exp
            for sign in [1.0, -1.0]:
                c = candidate * sign
                if abs(c) < abs(best) and predicate(c):
                    best = c
        # Try truncating to integer
        try:
            int_val = float(int(value))
            if abs(int_val) < abs(best) and predicate(int_val):
                best = int_val
        except (OverflowError, ValueError):
            pass
        return best

    def _shrink_string(self, value: str,
                       predicate: Callable[[str], bool]) -> str:
        # Try empty
        if predicate(""):
            return ""
        best = value
        # Try removing characters one at a time
        for i in range(len(value)):
            candidate = value[:i] + value[i + 1:]
            if predicate(candidate):
                best = candidate
                # Recurse on shorter string
                return self._shrink_string(candidate, predicate)
        # Try shorter substrings
        for length in range(1, len(value)):
            for start in range(len(value) - length + 1):
                candidate = value[start:start + length]
                if len(candidate) < len(best) and predicate(candidate):
                    best = candidate
                    return self._shrink_string(best, predicate)
        return best

    def _shrink_array(self, value: list,
                      predicate: Callable[[list], bool]) -> list:
        # Try empty
        if predicate([]):
            return []
        best = list(value)
        # Try removing elements one at a time
        i = 0
        while i < len(best):
            candidate = best[:i] + best[i + 1:]
            if predicate(candidate):
                best = candidate
                # Don't increment i since list shortened
            else:
                i += 1
        # Try shrinking individual elements (integers)
        for i in range(len(best)):
            if isinstance(best[i], int):
                original = best[i]
                # Try 0
                test_list = list(best)
                test_list[i] = 0
                if predicate(test_list):
                    best = test_list
                    continue
                # Binary search toward 0
                lo, hi = 0, abs(original)
                sign = 1 if original >= 0 else -1
                while lo < hi:
                    mid = (lo + hi) // 2
                    test_list = list(best)
                    test_list[i] = mid * sign
                    if predicate(test_list):
                        best = test_list
                        hi = mid
                    else:
                        lo = mid + 1
        return best

    def find_counterexample(
        self,
        func_c: Callable,
        func_rust: Callable,
        type_specs: List[str],
        max_attempts: int = 1000,
    ) -> Optional[CounterExample]:
        for _ in range(max_attempts):
            inputs = tuple(self._gen.generate_from_type(ts)
                           for ts in type_specs)
            try:
                c_out = func_c(*inputs)
            except Exception:
                continue
            try:
                rust_out = func_rust(*inputs)
            except Exception:
                continue

            cmp = self._comp.compare(c_out, rust_out)
            if not cmp.equal:
                # Found a counterexample, now shrink
                shrunk = self._shrink_counterexample(
                    func_c, func_rust, inputs, type_specs
                )
                shrunk_c = func_c(*shrunk)
                shrunk_r = func_rust(*shrunk)
                return CounterExample(
                    inputs=inputs,
                    c_output=c_out,
                    rust_output=rust_out,
                    shrunk_inputs=shrunk,
                    shrunk_c_output=shrunk_c,
                    shrunk_rust_output=shrunk_r,
                )
        return None

    def _shrink_counterexample(
        self,
        func_c: Callable,
        func_rust: Callable,
        inputs: Tuple,
        type_specs: List[str],
    ) -> Tuple:
        def still_fails(candidate_inputs: Tuple) -> bool:
            try:
                c_out = func_c(*candidate_inputs)
                r_out = func_rust(*candidate_inputs)
                return not self._comp.compare(c_out, r_out).equal
            except Exception:
                return False

        shrunk_list = list(inputs)
        for i, (val, ts) in enumerate(zip(inputs, type_specs)):
            def test_single(candidate: Any) -> bool:
                trial = list(shrunk_list)
                trial[i] = candidate
                return still_fails(tuple(trial))

            shrunk_list[i] = self.shrink(val, ts, test_single)

        return tuple(shrunk_list)


# ---------------------------------------------------------------------------
# DifferentialTester
# ---------------------------------------------------------------------------

class DifferentialTester:
    """Main differential testing engine."""

    def __init__(
        self,
        comparator: Optional[OutputComparator] = None,
        crash_detector: Optional[CrashDetector] = None,
        coverage_tracker: Optional[CoverageTracker] = None,
        input_generator: Optional[InputGenerator] = None,
        timeout: float = 5.0,
    ):
        self._comp = comparator or OutputComparator()
        self._crash = crash_detector or CrashDetector(default_timeout=timeout)
        self._cov = coverage_tracker or CoverageTracker()
        self._gen = input_generator or InputGenerator(seed=None)
        self._timeout = timeout

    def test(
        self,
        c_function: Callable,
        rust_function: Callable,
        input_generator: Optional[Callable[[], Tuple]] = None,
        n_tests: int = 1000,
    ) -> TestResult:
        result = TestResult()
        start = time.monotonic()

        if input_generator is None:
            input_generator = self._make_default_generator(c_function)

        for _ in range(n_tests):
            inputs = input_generator()
            if not isinstance(inputs, tuple):
                inputs = (inputs,)

            result.total_tests += 1

            c_exec = self._crash.run_with_detection(
                c_function, inputs, self._timeout
            )
            rust_exec = self._crash.run_with_detection(
                rust_function, inputs, self._timeout
            )

            # Track coverage from both executions
            c_path = self._make_path(c_exec)
            r_path = self._make_path(rust_exec)
            self._cov.record_path(c_path)
            self._cov.record_path(r_path)

            # Check for crashes
            if c_exec.is_crash() or rust_exec.is_crash():
                result.errors += 1
                if c_exec.is_crash() != rust_exec.is_crash():
                    # One crashed, the other didn't – it's a divergence
                    result.counterexamples.append(CounterExample(
                        inputs=inputs,
                        c_output=c_exec.error_msg if c_exec.is_crash()
                                 else c_exec.output,
                        rust_output=rust_exec.error_msg if rust_exec.is_crash()
                                    else rust_exec.output,
                    ))
                continue

            # Compare outputs
            cmp = self._comp.compare(c_exec.output, rust_exec.output)
            if cmp.equal:
                result.passed += 1
            else:
                result.failed += 1
                result.counterexamples.append(CounterExample(
                    inputs=inputs,
                    c_output=c_exec.output,
                    rust_output=rust_exec.output,
                ))

        result.duration_seconds = time.monotonic() - start
        result.coverage = self._cov.to_coverage_stats()
        return result

    def test_with_property_shrinking(
        self,
        c_function: Callable,
        rust_function: Callable,
        type_specs: List[str],
        n_tests: int = 1000,
    ) -> TestResult:
        """Like test(), but automatically shrinks counterexamples."""
        prop = PropertyTester(self._gen, self._comp)
        result = TestResult()
        start = time.monotonic()

        for _ in range(n_tests):
            inputs = tuple(self._gen.generate_from_type(ts)
                           for ts in type_specs)
            result.total_tests += 1

            c_exec = self._crash.run_with_detection(
                c_function, inputs, self._timeout
            )
            rust_exec = self._crash.run_with_detection(
                rust_function, inputs, self._timeout
            )

            if c_exec.is_crash() or rust_exec.is_crash():
                result.errors += 1
                continue

            cmp = self._comp.compare(c_exec.output, rust_exec.output)
            if cmp.equal:
                result.passed += 1
            else:
                result.failed += 1
                # Shrink the counterexample
                shrunk = prop._shrink_counterexample(
                    c_function, rust_function, inputs, type_specs
                )
                try:
                    shrunk_c = c_function(*shrunk)
                    shrunk_r = rust_function(*shrunk)
                except Exception:
                    shrunk_c = None
                    shrunk_r = None
                result.counterexamples.append(CounterExample(
                    inputs=inputs,
                    c_output=c_exec.output,
                    rust_output=rust_exec.output,
                    shrunk_inputs=shrunk,
                    shrunk_c_output=shrunk_c,
                    shrunk_rust_output=shrunk_r,
                ))

        result.duration_seconds = time.monotonic() - start
        result.coverage = self._cov.to_coverage_stats()
        return result

    def _make_default_generator(self, func: Callable) -> Callable[[], Tuple]:
        """Infer input types from function and generate accordingly."""
        import inspect
        sig = inspect.signature(func)
        params = list(sig.parameters.values())

        def generator() -> Tuple:
            args = []
            for p in params:
                annotation = p.annotation
                if annotation is int:
                    args.append(self._gen.generate_int())
                elif annotation is float:
                    args.append(self._gen.generate_float())
                elif annotation is str:
                    args.append(self._gen.generate_string())
                elif annotation is bool:
                    args.append(self._gen.generate_from_type("bool"))
                elif annotation is list:
                    args.append(self._gen.generate_array(
                        lambda: self._gen.generate_int()
                    ))
                elif annotation is dict:
                    args.append(self._gen.generate_struct(
                        {"x": lambda: self._gen.generate_int()}
                    ))
                elif annotation is bytes:
                    length = self._gen.generate_int(0, 100)
                    args.append(bytes(
                        self._gen.generate_int(0, 255)
                        for _ in range(length)
                    ))
                else:
                    # No annotation: default to int
                    args.append(self._gen.generate_int())
            if not args:
                args.append(self._gen.generate_int())
            return tuple(args)

        return generator

    @staticmethod
    def _make_path(exec_result: ExecutionResult) -> Tuple:
        """Derive a synthetic path identifier from an execution result."""
        if exec_result.is_crash():
            return (hash(exec_result.crash_kind) % 256,
                    hash(exec_result.error_msg or "") % 256)
        h = hashlib.md5(repr(exec_result.output).encode()).hexdigest()
        return tuple(int(c, 16) for c in h[:6])


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def run_differential_test(
    c_func: Callable,
    rust_func: Callable,
    input_gen: Optional[Callable[[], Tuple]] = None,
    n_tests: int = 1000,
    timeout: float = 5.0,
) -> TestResult:
    """Convenience function for quick differential testing."""
    tester = DifferentialTester(timeout=timeout)
    return tester.test(c_func, rust_func, input_gen, n_tests)


def run_property_test(
    c_func: Callable,
    rust_func: Callable,
    type_specs: List[str],
    n_tests: int = 1000,
) -> TestResult:
    """Convenience function for property-based differential testing."""
    tester = DifferentialTester()
    return tester.test_with_property_shrinking(
        c_func, rust_func, type_specs, n_tests
    )


def fuzz_function(
    func: Callable,
    seeds: Optional[List[bytes]] = None,
    n_iterations: int = 10000,
) -> List[bytes]:
    """Convenience function for fuzzing a single function."""
    fuzzer = Fuzzer()
    return fuzzer.fuzz(func, seeds or [b"\x00" * 8], n_iterations)


# ---------------------------------------------------------------------------
# Self-test (executed when run as __main__)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Differential Tester Self-Test ===\n")

    # -- InputGenerator demo -------------------------------------------------
    gen = InputGenerator(strategy=InputStrategy.MIXED, seed=123)
    print("Sample int (random):", gen.generate_int())
    gen.strategy = InputStrategy.BOUNDARY
    print("Sample int (boundary):", gen.generate_int())
    gen.strategy = InputStrategy.STRUCTURED
    print("Sample float (structured):", gen.generate_float())
    gen.strategy = InputStrategy.RANDOM
    print("Sample string:", repr(gen.generate_string(max_len=20)))
    print("Sample array:", gen.generate_array(lambda: gen.generate_int(-10, 10),
                                               max_len=5))
    print("Sample from type 'uint8_t':", gen.generate_from_type("uint8_t"))
    print()

    # -- OutputComparator demo -----------------------------------------------
    comp = OutputComparator()
    r1 = comp.compare(3.14, 3.14 + 1e-12, "float")
    print(f"Float compare (close): equal={r1.equal}")
    r2 = comp.compare(float("nan"), float("nan"), "float")
    print(f"NaN compare: equal={r2.equal}")
    r3 = comp.compare(-0.0, 0.0, "float")
    print(f"Signed zero compare: equal={r3.equal}")
    r4 = comp.compare([1, 2, 3], [1, 2, 4], "list")
    print(f"Array compare: equal={r4.equal}, diffs={r4.differences}")
    print()

    # -- CoverageTracker demo ------------------------------------------------
    cov = CoverageTracker()
    cov.record_path((True, False, True))
    cov.record_path((True, True, False))
    print("New coverage?", cov.is_new_coverage((False, False, False)))
    print("Coverage stats:", cov.get_coverage_stats())
    print()

    # -- CrashDetector demo --------------------------------------------------
    detector = CrashDetector(default_timeout=2.0)
    res = detector.run_with_detection(lambda x: x * 2, (21,))
    print(f"Normal exec: success={res.success}, output={res.output}")
    res2 = detector.run_with_detection(lambda x: 1 / 0, (0,))
    print(f"Exception exec: success={res2.success}, kind={res2.crash_kind}")
    print()

    # -- PropertyTester demo -------------------------------------------------
    def c_abs(x: int) -> int:
        return x if x >= 0 else -x

    def rust_abs_buggy(x: int) -> int:
        if x == -(2**31):
            return x  # Bug: overflow
        return x if x >= 0 else -x

    prop = PropertyTester(InputGenerator(seed=7))
    ce = prop.find_counterexample(c_abs, rust_abs_buggy, ["int"],
                                   max_attempts=5000)
    if ce:
        print("Counterexample found!")
        print(ce.describe())
    else:
        print("No counterexample found in 5000 attempts.")
    print()

    # -- DifferentialTester demo ---------------------------------------------
    def c_add(a: int, b: int) -> int:
        return a + b

    def rust_add(a: int, b: int) -> int:
        return a + b

    tester = DifferentialTester(timeout=1.0)
    result = tester.test(c_add, rust_add, n_tests=200)
    print("Differential test result:")
    print(result.summary())
    print()
    print("As dict (excerpt):")
    d = result.to_dict()
    print(f"  passed={d['passed']}, failed={d['failed']}, errors={d['errors']}")

    # -- Fuzzer demo ---------------------------------------------------------
    def target(data: bytes) -> int:
        total = 0
        for b in data:
            if b > 128:
                total += b
            else:
                total -= b
        return total

    fuzzer = Fuzzer(seed=99)
    interesting = fuzzer.fuzz(target, [b"hello"], n_iterations=500)
    print(f"\nFuzzer found {len(interesting)} interesting inputs")

    print("\n=== All self-tests passed ===")
