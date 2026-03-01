"""
Semantic seed generation from the divergence table.

For each divergence class, generates boundary-value inputs that are
likely to trigger divergent behavior between C and Rust.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any, Set

from ..ir.types import IRType, IntType, FloatType, PointerType, VoidType, Signedness, FloatKind
from ..ir.function import Function
from ..semantics.divergence_table import (
    DivergenceClass, DivergenceType, DivergenceTable, DivergenceEntry,
)
from .engine import FuzzInput


# ---------------------------------------------------------------------------
# Seed set
# ---------------------------------------------------------------------------

@dataclass
class SeedSet:
    """A collection of seeds for a specific divergence class."""
    divergence_class: DivergenceClass
    seeds: List[FuzzInput] = field(default_factory=list)
    description: str = ""

    @property
    def size(self) -> int:
        return len(self.seeds)

    def __repr__(self) -> str:
        return f"SeedSet({self.divergence_class.name}, {self.size} seeds)"


# ---------------------------------------------------------------------------
# Boundary value generators for specific types
# ---------------------------------------------------------------------------

class BoundaryValueGenerator:
    """Generates boundary values for specific types and divergence classes."""

    @staticmethod
    def signed_int_boundaries(width: int) -> List[int]:
        """Generate boundary values for signed integers."""
        max_val = (1 << (width - 1)) - 1
        min_val = -(1 << (width - 1))
        values = [
            0, 1, -1, 2, -2,
            max_val, max_val - 1, max_val - 2,
            min_val, min_val + 1, min_val + 2,
            max_val // 2, min_val // 2,
            42, -42, 100, -100,
        ]
        # Add powers of 2
        for i in range(1, width - 1):
            values.append(1 << i)
            values.append(-(1 << i))
            values.append((1 << i) - 1)
            values.append(-((1 << i) - 1))
        return [v for v in values if min_val <= v <= max_val]

    @staticmethod
    def unsigned_int_boundaries(width: int) -> List[int]:
        """Generate boundary values for unsigned integers."""
        max_val = (1 << width) - 1
        values = [
            0, 1, 2,
            max_val, max_val - 1, max_val - 2,
            max_val // 2, max_val // 2 + 1,
            42, 100, 255,
        ]
        for i in range(1, width):
            values.append(1 << i)
            values.append((1 << i) - 1)
        return [v for v in values if 0 <= v <= max_val]

    @staticmethod
    def float_boundaries(width: int = 64) -> List[float]:
        """Generate boundary values for floating-point numbers."""
        values = [
            0.0, -0.0,
            1.0, -1.0,
            0.5, -0.5,
            float('inf'), float('-inf'),
            float('nan'),
        ]

        if width <= 32:
            # f32 boundaries
            values.extend([
                3.4028235e+38,    # FLT_MAX
                -3.4028235e+38,
                1.1754944e-38,    # FLT_MIN (smallest normal)
                -1.1754944e-38,
                1.4012985e-45,    # FLT_TRUE_MIN (smallest denormal)
                -1.4012985e-45,
                1.1920929e-07,    # FLT_EPSILON
                16777216.0,       # 2^24 (float precision boundary)
                16777217.0,       # 2^24 + 1 (not exactly representable)
            ])
        else:
            # f64 boundaries
            values.extend([
                1.7976931348623157e+308,   # DBL_MAX
                -1.7976931348623157e+308,
                2.2250738585072014e-308,   # DBL_MIN (smallest normal)
                -2.2250738585072014e-308,
                5e-324,                     # DBL_TRUE_MIN (smallest denormal)
                -5e-324,
                2.220446049250313e-16,     # DBL_EPSILON
                9007199254740992.0,         # 2^53 (double precision boundary)
                9007199254740993.0,         # 2^53 + 1
            ])

        # Near-zero values
        for exp in range(-20, 20):
            try:
                values.append(10.0 ** exp)
                values.append(-(10.0 ** exp))
            except OverflowError:
                pass

        return values

    @staticmethod
    def shift_amount_boundaries(width: int) -> List[int]:
        """Generate boundary values for shift amounts."""
        return [
            0, 1, width - 2, width - 1,
            width, width + 1,
            -1, -width,
            31 if width >= 32 else width - 1,
            32 if width >= 32 else width,
            63 if width >= 64 else width - 1,
            64 if width >= 64 else width,
        ]

    @staticmethod
    def divisor_boundaries(width: int, signed: bool) -> List[int]:
        """Generate boundary values for divisors."""
        values = [0, 1, -1, 2, -2]
        if signed:
            max_val = (1 << (width - 1)) - 1
            min_val = -(1 << (width - 1))
            values.extend([max_val, min_val, max_val // 2, min_val // 2])
        else:
            max_val = (1 << width) - 1
            values.extend([max_val, max_val // 2, 1])
        return [v for v in values if not signed or (-(1 << (width - 1)) <= v < (1 << (width - 1)))]

    @staticmethod
    def pointer_boundaries() -> List[int]:
        """Generate boundary values for pointers."""
        return [
            0,              # NULL
            1,              # Very low (likely invalid)
            4,              # Misaligned for 8-byte
            7,              # Misaligned
            8,              # Aligned
            16,             # Aligned
            0xFFFFFFFF,     # 32-bit max
            0xDEADBEEF,     # Common sentinel
            0x7FFFFFFF,     # Signed 32-bit max
            0x80000000,     # Signed 32-bit min (unsigned)
        ]


# ---------------------------------------------------------------------------
# Seed generator
# ---------------------------------------------------------------------------

class SeedGenerator:
    """
    Generates seeds from the divergence table.
    
    For each divergence class, creates inputs that target the specific
    boundary conditions where C and Rust are likely to diverge.
    """

    def __init__(self, table: Optional[DivergenceTable] = None):
        self.table = table or DivergenceTable()
        self.boundary_gen = BoundaryValueGenerator()

    def generate_all_seeds(
        self,
        param_types: Optional[List[Tuple[str, IRType]]] = None,
    ) -> List[SeedSet]:
        """Generate seeds for all divergence classes."""
        seed_sets: List[SeedSet] = []

        for entry in self.table:
            seeds = self.generate_seeds_for_class(entry.cls, param_types)
            if seeds.seeds:
                seed_sets.append(seeds)

        return seed_sets

    def generate_seeds_for_class(
        self,
        cls: DivergenceClass,
        param_types: Optional[List[Tuple[str, IRType]]] = None,
    ) -> SeedSet:
        """Generate seeds for a specific divergence class."""
        dispatch = {
            DivergenceClass.SignedOverflow: self._gen_signed_overflow_seeds,
            DivergenceClass.UnsignedWrap: self._gen_unsigned_wrap_seeds,
            DivergenceClass.IntPromotion: self._gen_int_promotion_seeds,
            DivergenceClass.NegativeShift: self._gen_shift_seeds,
            DivergenceClass.DivisionByZero: self._gen_division_seeds,
            DivergenceClass.FloatToIntOOB: self._gen_float_to_int_seeds,
            DivergenceClass.NullDeref: self._gen_null_deref_seeds,
            DivergenceClass.ArrayOOB: self._gen_array_oob_seeds,
            DivergenceClass.PointerArith: self._gen_pointer_arith_seeds,
            DivergenceClass.FloatPrecision: self._gen_float_precision_seeds,
            DivergenceClass.ErrorHandling: self._gen_error_handling_seeds,
        }

        gen_fn = dispatch.get(cls, self._gen_default_seeds)
        return gen_fn(cls, param_types)

    def generate_from_signature(
        self,
        func: Function,
    ) -> List[FuzzInput]:
        """Generate seeds from a function's type signature."""
        param_types = [
            (arg.name or f"arg_{arg.index}", arg.type)
            for arg in func.arguments
        ]
        return self.generate_typed_inputs(param_types)

    def generate_typed_inputs(
        self,
        param_types: List[Tuple[str, IRType]],
    ) -> List[FuzzInput]:
        """Generate type-aware inputs for given parameter types."""
        inputs: List[FuzzInput] = []

        # Generate boundary-value combinations
        param_values: List[List[Any]] = []
        for name, ty in param_types:
            if isinstance(ty, IntType):
                if ty.signed:
                    vals = self.boundary_gen.signed_int_boundaries(ty.width)
                else:
                    vals = self.boundary_gen.unsigned_int_boundaries(ty.width)
                param_values.append(vals[:20])  # Limit per-param
            elif isinstance(ty, FloatType):
                vals = self.boundary_gen.float_boundaries(ty.kind.value)
                param_values.append(vals[:20])
            elif isinstance(ty, PointerType):
                param_values.append(self.boundary_gen.pointer_boundaries())
            else:
                param_values.append([0])

        # Generate combinations (bounded product)
        max_combos = 1000
        combos = self._bounded_product(param_values, max_combos)

        for combo in combos:
            typed_values: Dict[str, Any] = {}
            data_parts: List[bytes] = []

            for i, (name, ty) in enumerate(param_types):
                val = combo[i]
                typed_values[name] = val

                if isinstance(ty, IntType):
                    byte_count = max(ty.width // 8, 1)
                    int_val = int(val)
                    if int_val < 0:
                        int_val = int_val & ((1 << ty.width) - 1)
                    data_parts.append(int_val.to_bytes(byte_count, 'little'))
                elif isinstance(ty, FloatType):
                    try:
                        if ty.kind == FloatKind.F32:
                            data_parts.append(struct.pack('<f', float(val)))
                        else:
                            data_parts.append(struct.pack('<d', float(val)))
                    except (OverflowError, ValueError, struct.error):
                        data_parts.append(b'\x00' * (4 if ty.kind == FloatKind.F32 else 8))
                else:
                    data_parts.append(int(val).to_bytes(8, 'little'))

            data = b''.join(data_parts)
            inputs.append(FuzzInput(
                data=data, typed_values=typed_values, source="typed_boundary",
            ))

        return inputs

    def _bounded_product(
        self,
        lists: List[List[Any]],
        max_count: int,
    ) -> List[List[Any]]:
        """Generate bounded cartesian product of lists."""
        if not lists:
            return [[]]

        result: List[List[Any]] = [[]]
        for lst in lists:
            new_result: List[List[Any]] = []
            for combo in result:
                for val in lst:
                    new_result.append(combo + [val])
                    if len(new_result) >= max_count:
                        return new_result
            result = new_result

        return result[:max_count]

    # -- Per-class seed generators --

    def _gen_signed_overflow_seeds(
        self,
        cls: DivergenceClass,
        param_types: Optional[List[Tuple[str, IRType]]],
    ) -> SeedSet:
        """Generate seeds targeting signed overflow."""
        seeds: List[FuzzInput] = []

        for width in [8, 16, 32, 64]:
            max_val = (1 << (width - 1)) - 1
            min_val = -(1 << (width - 1))

            # Add pairs that overflow: a + b
            overflow_pairs = [
                (max_val, 1), (max_val, max_val),
                (min_val, -1), (min_val, min_val),
                (max_val // 2 + 1, max_val // 2 + 1),
                (min_val // 2 - 1, min_val // 2 - 1),
            ]

            for a, b in overflow_pairs:
                typed_values = {"a": a, "b": b}
                byte_count = max(width // 8, 1)
                a_bytes = (a & ((1 << width) - 1)).to_bytes(byte_count, 'little')
                b_bytes = (b & ((1 << width) - 1)).to_bytes(byte_count, 'little')
                seeds.append(FuzzInput(
                    data=a_bytes + b_bytes,
                    typed_values=typed_values,
                    source=f"signed_overflow_i{width}",
                ))

        return SeedSet(
            divergence_class=cls,
            seeds=seeds,
            description="Signed overflow boundary values",
        )

    def _gen_unsigned_wrap_seeds(
        self,
        cls: DivergenceClass,
        param_types: Optional[List[Tuple[str, IRType]]],
    ) -> SeedSet:
        seeds: List[FuzzInput] = []

        for width in [8, 16, 32, 64]:
            max_val = (1 << width) - 1
            wrap_pairs = [
                (max_val, 1), (max_val, max_val),
                (0, 0), (1, max_val),
                (max_val // 2, max_val // 2 + 2),
            ]

            for a, b in wrap_pairs:
                typed_values = {"a": a, "b": b}
                byte_count = max(width // 8, 1)
                a_bytes = a.to_bytes(byte_count, 'little')
                b_bytes = b.to_bytes(byte_count, 'little')
                seeds.append(FuzzInput(
                    data=a_bytes + b_bytes,
                    typed_values=typed_values,
                    source=f"unsigned_wrap_u{width}",
                ))

        return SeedSet(divergence_class=cls, seeds=seeds, description="Unsigned wrap boundary values")

    def _gen_int_promotion_seeds(
        self,
        cls: DivergenceClass,
        param_types: Optional[List[Tuple[str, IRType]]],
    ) -> SeedSet:
        seeds: List[FuzzInput] = []

        # Values that behave differently under different promotion rules
        values = [127, 128, 255, 256, -1, -128, 32767, 32768, 65535]
        for val in values:
            unsigned = val & 0xFFFF
            seeds.append(FuzzInput(
                data=unsigned.to_bytes(2, 'little'),
                typed_values={"x": val},
                source="int_promotion",
            ))

        return SeedSet(divergence_class=cls, seeds=seeds, description="Integer promotion boundary values")

    def _gen_shift_seeds(
        self,
        cls: DivergenceClass,
        param_types: Optional[List[Tuple[str, IRType]]],
    ) -> SeedSet:
        seeds: List[FuzzInput] = []

        for width in [8, 16, 32, 64]:
            values = [1, -1, (1 << (width - 1)) - 1, -(1 << (width - 1))]
            shift_amounts = self.boundary_gen.shift_amount_boundaries(width)

            for val in values:
                for shift in shift_amounts:
                    unsigned_val = val & ((1 << width) - 1)
                    unsigned_shift = shift & ((1 << width) - 1)
                    byte_count = max(width // 8, 1)
                    seeds.append(FuzzInput(
                        data=unsigned_val.to_bytes(byte_count, 'little') +
                             unsigned_shift.to_bytes(byte_count, 'little'),
                        typed_values={"value": val, "shift": shift},
                        source=f"shift_i{width}",
                    ))

        return SeedSet(divergence_class=cls, seeds=seeds, description="Shift operation boundary values")

    def _gen_division_seeds(
        self,
        cls: DivergenceClass,
        param_types: Optional[List[Tuple[str, IRType]]],
    ) -> SeedSet:
        seeds: List[FuzzInput] = []

        for width in [32, 64]:
            max_val = (1 << (width - 1)) - 1
            min_val = -(1 << (width - 1))

            div_pairs = [
                (1, 0), (max_val, 0), (min_val, 0),  # Div by zero
                (min_val, -1),  # INT_MIN / -1
                (max_val, -1), (0, 1), (1, 1),
                (max_val, 2), (min_val, 2),
                (-1, -1), (min_val + 1, -1),
            ]

            for a, b in div_pairs:
                byte_count = max(width // 8, 1)
                a_u = a & ((1 << width) - 1)
                b_u = b & ((1 << width) - 1)
                seeds.append(FuzzInput(
                    data=a_u.to_bytes(byte_count, 'little') + b_u.to_bytes(byte_count, 'little'),
                    typed_values={"dividend": a, "divisor": b},
                    source=f"division_i{width}",
                ))

        return SeedSet(divergence_class=cls, seeds=seeds, description="Division boundary values")

    def _gen_float_to_int_seeds(
        self,
        cls: DivergenceClass,
        param_types: Optional[List[Tuple[str, IRType]]],
    ) -> SeedSet:
        seeds: List[FuzzInput] = []

        float_vals = [
            float('inf'), float('-inf'), float('nan'),
            2147483648.0, -2147483649.0,  # Just outside i32 range
            2147483647.0, -2147483648.0,  # i32 max/min
            4294967296.0, -1.0,            # u32 boundaries
            9223372036854775808.0,          # Just outside i64
            1e20, -1e20,                    # Way outside
            0.5, -0.5,                     # Truncation
            0.0, -0.0,
            1.9999999, -1.9999999,
            3.4028235e+38, -3.4028235e+38, # f32 max
        ]

        for val in float_vals:
            try:
                data = struct.pack('<d', val)
                seeds.append(FuzzInput(
                    data=data,
                    typed_values={"x": val},
                    source="float_to_int",
                ))
            except (OverflowError, ValueError, struct.error):
                pass

        return SeedSet(divergence_class=cls, seeds=seeds, description="Float-to-int conversion boundary values")

    def _gen_null_deref_seeds(
        self,
        cls: DivergenceClass,
        param_types: Optional[List[Tuple[str, IRType]]],
    ) -> SeedSet:
        seeds: List[FuzzInput] = []

        for ptr_val in self.boundary_gen.pointer_boundaries():
            seeds.append(FuzzInput(
                data=ptr_val.to_bytes(8, 'little'),
                typed_values={"ptr": ptr_val},
                source="null_deref",
            ))

        return SeedSet(divergence_class=cls, seeds=seeds, description="Null/invalid pointer values")

    def _gen_array_oob_seeds(
        self,
        cls: DivergenceClass,
        param_types: Optional[List[Tuple[str, IRType]]],
    ) -> SeedSet:
        seeds: List[FuzzInput] = []

        for array_len in [0, 1, 10, 100, 1024]:
            indices = [
                -1, 0, 1, array_len - 1, array_len, array_len + 1,
                2147483647, -2147483648,
            ]
            for idx in indices:
                idx_u = idx & 0xFFFFFFFF
                seeds.append(FuzzInput(
                    data=array_len.to_bytes(4, 'little') + idx_u.to_bytes(4, 'little'),
                    typed_values={"length": array_len, "index": idx},
                    source="array_oob",
                ))

        return SeedSet(divergence_class=cls, seeds=seeds, description="Array out-of-bounds indices")

    def _gen_pointer_arith_seeds(
        self,
        cls: DivergenceClass,
        param_types: Optional[List[Tuple[str, IRType]]],
    ) -> SeedSet:
        seeds: List[FuzzInput] = []

        base_ptrs = [0x10000, 0x7FFFFFFF, 0xFFFFFFFF]
        offsets = [0, 1, -1, 4, -4, 1000, -1000, 2147483647, -2147483648]

        for base in base_ptrs:
            for offset in offsets:
                offset_u = offset & 0xFFFFFFFF
                seeds.append(FuzzInput(
                    data=base.to_bytes(8, 'little') + offset_u.to_bytes(4, 'little'),
                    typed_values={"base": base, "offset": offset},
                    source="pointer_arith",
                ))

        return SeedSet(divergence_class=cls, seeds=seeds, description="Pointer arithmetic boundary values")

    def _gen_float_precision_seeds(
        self,
        cls: DivergenceClass,
        param_types: Optional[List[Tuple[str, IRType]]],
    ) -> SeedSet:
        seeds: List[FuzzInput] = []

        float_vals = self.boundary_gen.float_boundaries(64)

        for a in float_vals[:15]:
            for b in float_vals[:15]:
                try:
                    data = struct.pack('<dd', a, b)
                    seeds.append(FuzzInput(
                        data=data,
                        typed_values={"a": a, "b": b},
                        source="float_precision",
                    ))
                except (OverflowError, ValueError, struct.error):
                    pass

        return SeedSet(divergence_class=cls, seeds=seeds, description="Float precision boundary values")

    def _gen_error_handling_seeds(
        self,
        cls: DivergenceClass,
        param_types: Optional[List[Tuple[str, IRType]]],
    ) -> SeedSet:
        seeds: List[FuzzInput] = []

        # Error code values
        error_codes = [0, -1, 1, -2, 255, -128, 2147483647, -2147483648]
        for code in error_codes:
            code_u = code & 0xFFFFFFFF
            seeds.append(FuzzInput(
                data=code_u.to_bytes(4, 'little'),
                typed_values={"error_code": code},
                source="error_handling",
            ))

        return SeedSet(divergence_class=cls, seeds=seeds, description="Error handling boundary values")

    def _gen_default_seeds(
        self,
        cls: DivergenceClass,
        param_types: Optional[List[Tuple[str, IRType]]],
    ) -> SeedSet:
        seeds: List[FuzzInput] = []

        if param_types:
            seeds = self.generate_typed_inputs(param_types)

        return SeedSet(divergence_class=cls, seeds=seeds, description=f"Default seeds for {cls.name}")
