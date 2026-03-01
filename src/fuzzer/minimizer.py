"""
Input minimization for differential fuzzing.

Given a divergence-triggering input, finds a minimal input that still
triggers the divergence using delta debugging and type-aware minimization.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict, Tuple, Any, Callable

from .engine import FuzzInput, FuzzResult


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class MinimizationStrategy(Enum):
    """Strategy for input minimization."""
    DELTA_DEBUGGING = auto()
    HIERARCHICAL = auto()
    TYPE_AWARE = auto()
    BINARY_SEARCH = auto()


@dataclass
class MinimizationResult:
    """Result of input minimization."""
    original_input: FuzzInput
    minimized_input: FuzzInput
    original_size: int = 0
    minimized_size: int = 0
    iterations: int = 0
    strategy: MinimizationStrategy = MinimizationStrategy.DELTA_DEBUGGING
    still_triggers: bool = True

    @property
    def reduction_ratio(self) -> float:
        if self.original_size == 0:
            return 0.0
        return 1.0 - (self.minimized_size / self.original_size)

    def summary(self) -> str:
        return (
            f"Minimization: {self.original_size}B → {self.minimized_size}B "
            f"({self.reduction_ratio * 100:.1f}% reduction, "
            f"{self.iterations} iterations, {self.strategy.name})"
        )


# ---------------------------------------------------------------------------
# Input minimizer
# ---------------------------------------------------------------------------

class InputMinimizer:
    """
    Minimizes divergence-triggering inputs.
    
    Supports delta debugging, hierarchical minimization (minimize
    struct fields independently), and type-aware minimization.
    """

    def __init__(
        self,
        test_function: Optional[Callable[[FuzzInput], bool]] = None,
        max_iterations: int = 1000,
        strategy: MinimizationStrategy = MinimizationStrategy.DELTA_DEBUGGING,
    ):
        self.test_function = test_function
        self.max_iterations = max_iterations
        self.strategy = strategy
        self._iterations = 0

    def minimize(
        self,
        input: FuzzInput,
        test_fn: Optional[Callable[[FuzzInput], bool]] = None,
    ) -> MinimizationResult:
        """Minimize an input while preserving the divergence-triggering property."""
        tester = test_fn or self.test_function
        if tester is None:
            return MinimizationResult(
                original_input=input,
                minimized_input=input,
                original_size=input.size,
                minimized_size=input.size,
            )

        self._iterations = 0

        if self.strategy == MinimizationStrategy.DELTA_DEBUGGING:
            minimized = self._delta_debug(input, tester)
        elif self.strategy == MinimizationStrategy.HIERARCHICAL:
            minimized = self._hierarchical_minimize(input, tester)
        elif self.strategy == MinimizationStrategy.TYPE_AWARE:
            minimized = self._type_aware_minimize(input, tester)
        elif self.strategy == MinimizationStrategy.BINARY_SEARCH:
            minimized = self._binary_search_minimize(input, tester)
        else:
            minimized = self._delta_debug(input, tester)

        still_triggers = tester(minimized)

        return MinimizationResult(
            original_input=input,
            minimized_input=minimized,
            original_size=input.size,
            minimized_size=minimized.size,
            iterations=self._iterations,
            strategy=self.strategy,
            still_triggers=still_triggers,
        )

    # -- Delta debugging (Zeller's algorithm) --

    def _delta_debug(
        self,
        input: FuzzInput,
        test_fn: Callable[[FuzzInput], bool],
    ) -> FuzzInput:
        """Classic delta debugging: find a 1-minimal subset."""
        data = bytearray(input.data)
        n = 2  # Number of chunks

        while len(data) >= 2 and self._iterations < self.max_iterations:
            chunk_size = max(len(data) // n, 1)
            reduced = False

            for i in range(n):
                start = i * chunk_size
                end = min(start + chunk_size, len(data))

                # Try removing this chunk
                candidate = data[:start] + data[end:]
                if not candidate:
                    continue

                self._iterations += 1
                candidate_input = FuzzInput(
                    data=bytes(candidate),
                    typed_values=input.typed_values,
                    source="minimized",
                    generation=input.generation,
                )

                if test_fn(candidate_input):
                    data = bytearray(candidate)
                    n = max(n - 1, 2)
                    reduced = True
                    break

            if not reduced:
                if n >= len(data):
                    break
                n = min(n * 2, len(data))

        return FuzzInput(
            data=bytes(data),
            typed_values=input.typed_values,
            source="delta_debug",
            generation=input.generation,
        )

    # -- Hierarchical minimization --

    def _hierarchical_minimize(
        self,
        input: FuzzInput,
        test_fn: Callable[[FuzzInput], bool],
    ) -> FuzzInput:
        """Hierarchical minimization: minimize typed fields independently."""
        if not input.typed_values:
            return self._delta_debug(input, test_fn)

        current = FuzzInput(
            data=input.data,
            typed_values=dict(input.typed_values),
            source="hierarchical",
            generation=input.generation,
        )

        # Minimize each typed field independently
        for field_name, field_value in list(input.typed_values.items()):
            if isinstance(field_value, int):
                current = self._minimize_int_field(current, field_name, field_value, test_fn)
            elif isinstance(field_value, float):
                current = self._minimize_float_field(current, field_name, field_value, test_fn)

        # Then do delta debugging on the raw data
        current = self._delta_debug(current, test_fn)

        return current

    def _minimize_int_field(
        self,
        input: FuzzInput,
        field_name: str,
        value: int,
        test_fn: Callable[[FuzzInput], bool],
    ) -> FuzzInput:
        """Minimize a single integer field."""
        best = input

        # Try simpler values
        candidates = [0, 1, -1, value // 2, value - 1, value + 1]
        candidates = [c for c in candidates if c != value]

        for candidate_val in candidates:
            self._iterations += 1
            if self._iterations >= self.max_iterations:
                break

            new_typed = dict(input.typed_values)
            new_typed[field_name] = candidate_val

            candidate = FuzzInput(
                data=self._rebuild_data(new_typed, input),
                typed_values=new_typed,
                source="minimize_int",
                generation=input.generation,
            )

            if test_fn(candidate):
                if abs(candidate_val) < abs(value):
                    best = candidate
                    value = candidate_val

        # Binary search toward zero
        if value != 0:
            lo = 0
            hi = abs(value)
            sign = 1 if value > 0 else -1

            while lo < hi and self._iterations < self.max_iterations:
                mid = (lo + hi) // 2
                self._iterations += 1

                new_typed = dict(best.typed_values)
                new_typed[field_name] = mid * sign

                candidate = FuzzInput(
                    data=self._rebuild_data(new_typed, input),
                    typed_values=new_typed,
                    source="minimize_int_bsearch",
                    generation=input.generation,
                )

                if test_fn(candidate):
                    hi = mid
                    best = candidate
                else:
                    lo = mid + 1

        return best

    def _minimize_float_field(
        self,
        input: FuzzInput,
        field_name: str,
        value: float,
        test_fn: Callable[[FuzzInput], bool],
    ) -> FuzzInput:
        """Minimize a single float field."""
        import math
        best = input

        # Try simpler values
        candidates = [0.0, 1.0, -1.0, 0.5, -0.5]
        if not math.isnan(value) and not math.isinf(value):
            candidates.extend([value / 2, value * 2, int(value) * 1.0])

        candidates = [c for c in candidates if c != value]

        for candidate_val in candidates:
            self._iterations += 1
            if self._iterations >= self.max_iterations:
                break

            new_typed = dict(input.typed_values)
            new_typed[field_name] = candidate_val

            candidate = FuzzInput(
                data=self._rebuild_data(new_typed, input),
                typed_values=new_typed,
                source="minimize_float",
                generation=input.generation,
            )

            if test_fn(candidate):
                best = candidate

        return best

    # -- Type-aware minimization --

    def _type_aware_minimize(
        self,
        input: FuzzInput,
        test_fn: Callable[[FuzzInput], bool],
    ) -> FuzzInput:
        """Type-aware minimization using knowledge of parameter types."""
        current = input

        if input.typed_values:
            # First pass: try to zero out each field
            for field_name in list(input.typed_values.keys()):
                self._iterations += 1
                if self._iterations >= self.max_iterations:
                    break

                new_typed = dict(current.typed_values)
                old_val = new_typed[field_name]

                if isinstance(old_val, (int, float)):
                    new_typed[field_name] = type(old_val)(0)
                else:
                    continue

                candidate = FuzzInput(
                    data=self._rebuild_data(new_typed, input),
                    typed_values=new_typed,
                    source="type_aware_zero",
                    generation=input.generation,
                )

                if test_fn(candidate):
                    current = candidate

            # Second pass: hierarchical on remaining non-zero fields
            current = self._hierarchical_minimize(current, test_fn)
        else:
            current = self._delta_debug(input, test_fn)

        return current

    # -- Binary search minimization --

    def _binary_search_minimize(
        self,
        input: FuzzInput,
        test_fn: Callable[[FuzzInput], bool],
    ) -> FuzzInput:
        """Binary search on input length."""
        data = input.data
        lo = 0
        hi = len(data)
        best = input

        while lo < hi and self._iterations < self.max_iterations:
            mid = (lo + hi) // 2
            self._iterations += 1

            candidate = FuzzInput(
                data=data[:mid],
                typed_values=input.typed_values,
                source="binary_search",
                generation=input.generation,
            )

            if candidate.data and test_fn(candidate):
                hi = mid
                best = candidate
            else:
                lo = mid + 1

        # Refine with delta debugging
        best = self._delta_debug(best, test_fn)
        return best

    # -- Helpers --

    def _rebuild_data(self, typed_values: Dict[str, Any], original: FuzzInput) -> bytes:
        """Rebuild raw data from typed values."""
        parts: List[bytes] = []
        for name, val in typed_values.items():
            if isinstance(val, int):
                # Determine byte width from original
                byte_count = 4
                orig_val = original.typed_values.get(name)
                if isinstance(orig_val, int):
                    if abs(orig_val) > 0xFFFFFFFF:
                        byte_count = 8
                    elif abs(orig_val) > 0xFFFF:
                        byte_count = 4
                    elif abs(orig_val) > 0xFF:
                        byte_count = 2
                    else:
                        byte_count = 1
                unsigned_val = val & ((1 << (byte_count * 8)) - 1)
                parts.append(unsigned_val.to_bytes(byte_count, 'little'))
            elif isinstance(val, float):
                try:
                    parts.append(struct.pack('<d', val))
                except (OverflowError, ValueError, struct.error):
                    parts.append(b'\x00' * 8)
            else:
                parts.append(b'\x00\x00\x00\x00')

        return b''.join(parts) if parts else original.data
