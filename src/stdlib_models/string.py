"""
String function models: C ↔ Rust equivalences.

Models strlen ↔ str::len, strcmp ↔ str cmp, strcpy ↔ String::clone,
strcat ↔ String::push_str, sprintf ↔ format!, atoi ↔ str::parse.
Includes null-terminator vs length-prefix divergence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import z3

from .memory import DivergenceLevel, FunctionEquivalence, ModelResult


# ---------------------------------------------------------------------------
# strlen model
# ---------------------------------------------------------------------------

class StrlenModel:
    """
    Model for strlen(s) ↔ str::len / String::len.
    
    C: strlen counts bytes until null terminator.
    Rust: str::len returns byte length stored in fat pointer.
    Divergence: Embedded nulls, missing null terminator.
    """

    equivalence = FunctionEquivalence(
        c_function="strlen",
        rust_equivalent="str::len / String::len",
        divergence_level=DivergenceLevel.MODERATE,
        preconditions=["C string is null-terminated"],
        divergence_points=[
            "Embedded NUL: C stops at first NUL, Rust includes all bytes",
            "Missing NUL terminator: C reads past buffer (UB), Rust uses stored length",
            "UTF-8 vs bytes: Rust str::len is byte count, C strlen is byte count (same)",
            "NULL pointer: C is UB, Rust panics",
        ],
    )

    @staticmethod
    def apply(
        string_ptr: z3.BitVecRef,
        memory: z3.ArrayRef,
        max_length: int = 256,
        addr_width: int = 64,
    ) -> ModelResult:
        result = ModelResult()
        null = z3.BitVecVal(0, addr_width)

        result.error_condition = string_ptr == null

        # Model: search for null terminator byte
        c_len = z3.BitVec("strlen_result", 32)

        # Build constraint: there exists a position where byte is 0
        # and all bytes before it are non-zero
        found_null = z3.BoolVal(False)
        for i in range(max_length):
            offset = z3.BitVecVal(i, addr_width)
            byte_val = z3.Select(memory, string_ptr + offset)
            is_null_byte = byte_val == z3.BitVecVal(0, 8)

            all_prev_nonzero = z3.BoolVal(True)
            for j in range(i):
                prev_offset = z3.BitVecVal(j, addr_width)
                prev_byte = z3.Select(memory, string_ptr + prev_offset)
                all_prev_nonzero = z3.And(all_prev_nonzero, prev_byte != z3.BitVecVal(0, 8))

            found_at_i = z3.And(is_null_byte, all_prev_nonzero)
            result.constraints.append(
                z3.Implies(found_at_i, c_len == z3.BitVecVal(i, 32))
            )

        result.return_value = c_len
        return result

    @staticmethod
    def rust_len_model(
        string_len: z3.BitVecRef,
    ) -> ModelResult:
        """Rust str::len simply returns the stored length."""
        result = ModelResult()
        result.return_value = string_len
        return result

    @staticmethod
    def divergence_condition(
        string_ptr: z3.BitVecRef,
        string_len: z3.BitVecRef,
        memory: z3.ArrayRef,
        addr_width: int = 64,
    ) -> z3.BoolRef:
        """Condition under which strlen and str::len diverge: embedded nulls."""
        # Check if there's a null byte before the Rust-reported length
        has_embedded_null = z3.BoolVal(False)
        max_check = 64
        for i in range(max_check):
            offset = z3.BitVecVal(i, addr_width)
            idx = z3.BitVecVal(i, 32)
            byte_val = z3.Select(memory, string_ptr + offset)
            within_len = z3.ULT(idx, string_len)
            is_null = byte_val == z3.BitVecVal(0, 8)
            has_embedded_null = z3.Or(has_embedded_null, z3.And(within_len, is_null))

        return has_embedded_null


# ---------------------------------------------------------------------------
# strcmp model
# ---------------------------------------------------------------------------

class StrcmpModel:
    """
    Model for strcmp/strncmp ↔ str comparison.
    
    C: Lexicographic comparison of bytes until null terminator.
    Rust: Lexicographic comparison of byte slices.
    Divergence: Return value (C: any int, Rust: Ordering), embedded nulls.
    """

    equivalence = FunctionEquivalence(
        c_function="strcmp / strncmp",
        rust_equivalent="str::cmp / str::eq / PartialOrd",
        divergence_level=DivergenceLevel.LOW,
        preconditions=["Both strings are null-terminated"],
        divergence_points=[
            "Return value: C returns any int, Rust returns Ordering enum (-1/0/1)",
            "Embedded NUL: C stops comparison, Rust compares all bytes",
            "NULL pointer: C is UB, Rust panics",
            "Locale: C can be locale-dependent (strcoll), Rust is byte-wise",
        ],
    )

    @staticmethod
    def apply(
        a_ptr: z3.BitVecRef,
        b_ptr: z3.BitVecRef,
        memory: z3.ArrayRef,
        max_length: int = 256,
        addr_width: int = 64,
    ) -> ModelResult:
        result = ModelResult()
        null = z3.BitVecVal(0, addr_width)

        result.error_condition = z3.Or(a_ptr == null, b_ptr == null)

        ret = z3.BitVec("strcmp_result", 32)

        # Model byte-by-byte comparison
        equal_so_far = z3.BoolVal(True)
        for i in range(min(max_length, 32)):
            offset = z3.BitVecVal(i, addr_width)
            byte_a = z3.Select(memory, a_ptr + offset)
            byte_b = z3.Select(memory, b_ptr + offset)

            both_zero = z3.And(byte_a == z3.BitVecVal(0, 8), byte_b == z3.BitVecVal(0, 8))
            a_less = z3.ULT(byte_a, byte_b)
            b_less = z3.ULT(byte_b, byte_a)

            # If first difference found
            result.constraints.append(z3.Implies(
                z3.And(equal_so_far, a_less),
                ret < z3.BitVecVal(0, 32),
            ))
            result.constraints.append(z3.Implies(
                z3.And(equal_so_far, b_less),
                ret > z3.BitVecVal(0, 32),
            ))
            result.constraints.append(z3.Implies(
                z3.And(equal_so_far, both_zero),
                ret == z3.BitVecVal(0, 32),
            ))

            equal_so_far = z3.And(equal_so_far, byte_a == byte_b, z3.Not(both_zero))

        # Divergence: sign of return matches but value may differ
        result.divergence_condition = z3.And(
            ret != z3.BitVecVal(0, 32),
            z3.Or(ret != z3.BitVecVal(1, 32), ret != z3.BitVecVal(-1 & 0xFFFFFFFF, 32)),
        )

        result.return_value = ret
        return result


# ---------------------------------------------------------------------------
# strcpy model
# ---------------------------------------------------------------------------

class StrcpyModel:
    """
    Model for strcpy/strncpy ↔ String::clone / str::to_string.
    
    C: Copies until null terminator, no bounds checking.
    Rust: Clone/to_string allocates and copies with known length.
    """

    equivalence = FunctionEquivalence(
        c_function="strcpy / strncpy",
        rust_equivalent="String::clone / str::to_string / to_owned",
        divergence_level=DivergenceLevel.HIGH,
        preconditions=["dst has enough space", "src is null-terminated"],
        divergence_points=[
            "Buffer overflow: C writes past dst end (UB), Rust allocates correctly",
            "strncpy padding: C pads with NUL up to n, Rust doesn't pad",
            "Missing NUL: C reads past buffer, Rust uses known length",
            "Overlap: C is UB for strcpy, Rust copies via new allocation",
        ],
    )

    @staticmethod
    def apply(
        dst: z3.BitVecRef,
        src: z3.BitVecRef,
        memory: z3.ArrayRef,
        n: Optional[z3.BitVecRef] = None,
        addr_width: int = 64,
    ) -> ModelResult:
        result = ModelResult()
        null = z3.BitVecVal(0, addr_width)

        result.error_condition = z3.Or(dst == null, src == null)

        # Memory effect: copy bytes from src to dst until NUL
        result.memory_effects.append({
            "type": "string_copy",
            "dst": dst,
            "src": src,
            "max_len": n,
        })

        result.return_value = dst
        return result


# ---------------------------------------------------------------------------
# strcat model
# ---------------------------------------------------------------------------

class StrcatModel:
    """
    Model for strcat/strncat ↔ String::push_str.
    
    C: Appends src to dst, requires dst has enough space.
    Rust: String::push_str automatically grows the buffer.
    """

    equivalence = FunctionEquivalence(
        c_function="strcat / strncat",
        rust_equivalent="String::push_str / String::push",
        divergence_level=DivergenceLevel.HIGH,
        preconditions=["dst has enough space for concatenation"],
        divergence_points=[
            "Buffer overflow: C writes past dst (UB), Rust auto-grows",
            "NULL pointer: C is UB, Rust panics",
            "Overlap: C is UB, Rust via separate String is safe",
        ],
    )

    @staticmethod
    def apply(
        dst: z3.BitVecRef,
        src: z3.BitVecRef,
        dst_len: z3.BitVecRef,
        src_len: z3.BitVecRef,
        dst_capacity: z3.BitVecRef,
        addr_width: int = 64,
    ) -> ModelResult:
        result = ModelResult()
        null = z3.BitVecVal(0, addr_width)

        result.error_condition = z3.Or(dst == null, src == null)

        # Buffer overflow: dst_len + src_len + 1 > dst_capacity
        total_needed = dst_len + src_len + z3.BitVecVal(1, dst_len.size())
        overflow = z3.UGT(total_needed, dst_capacity)
        result.divergence_condition = overflow

        result.memory_effects.append({
            "type": "string_concat",
            "dst": dst,
            "src": src,
            "dst_len": dst_len,
            "src_len": src_len,
        })

        result.return_value = dst
        return result


# ---------------------------------------------------------------------------
# sprintf model
# ---------------------------------------------------------------------------

class SprintfModel:
    """
    Model for sprintf/snprintf ↔ format! / write!.
    
    C: sprintf writes to buffer (no bounds check), snprintf is bounded.
    Rust: format! allocates a String, write! to a buffer.
    """

    equivalence = FunctionEquivalence(
        c_function="sprintf / snprintf",
        rust_equivalent="format! / write! / format_args!",
        divergence_level=DivergenceLevel.MODERATE,
        preconditions=["Buffer is large enough (for sprintf)"],
        divergence_points=[
            "Buffer overflow: sprintf has no limit (UB), format! allocates",
            "Return value: snprintf returns chars needed, Rust returns Result",
            "Format strings: %d vs {} syntax (semantic, not runtime)",
            "Locale: C can be locale-dependent, Rust is not",
        ],
    )

    FORMAT_SPECIFIERS = {
        "%d": ("i32", "{}"),
        "%u": ("u32", "{}"),
        "%ld": ("i64", "{}"),
        "%lu": ("u64", "{}"),
        "%f": ("f64", "{}"),
        "%e": ("f64", "{:e}"),
        "%g": ("f64", "{}"),
        "%s": ("&str", "{}"),
        "%c": ("char", "{}"),
        "%p": ("*const ()", "{:p}"),
        "%x": ("u32", "{:x}"),
        "%X": ("u32", "{:X}"),
        "%o": ("u32", "{:o}"),
        "%%": ("", "%"),
    }

    @staticmethod
    def apply(
        dst: z3.BitVecRef,
        n: Optional[z3.BitVecRef],
        format_len: z3.BitVecRef,
        addr_width: int = 64,
    ) -> ModelResult:
        result = ModelResult()
        null = z3.BitVecVal(0, addr_width)

        result.error_condition = dst == null

        ret = z3.BitVec("sprintf_result", 32)

        # snprintf: return value is the number of chars that would be written
        result.return_value = ret

        if n is not None:
            # snprintf: truncates to n-1 chars + NUL
            result.constraints.append(z3.ULE(ret, n))
        else:
            # sprintf: no bounds checking, potential buffer overflow
            result.divergence_condition = z3.BoolVal(True)  # Always potentially dangerous

        result.memory_effects.append({
            "type": "string_format",
            "dst": dst,
            "max_len": n,
        })

        return result


# ---------------------------------------------------------------------------
# atoi model
# ---------------------------------------------------------------------------

class AtoiModel:
    """
    Model for atoi/strtol ↔ str::parse.
    
    C: atoi has UB on overflow, strtol sets errno.
    Rust: str::parse returns Result, overflow returns Err.
    """

    equivalence = FunctionEquivalence(
        c_function="atoi / strtol / strtoul",
        rust_equivalent="str::parse::<i32> / str::parse::<i64>",
        divergence_level=DivergenceLevel.HIGH,
        preconditions=["String represents a valid integer"],
        divergence_points=[
            "Overflow: atoi is UB, strtol sets errno/returns LONG_MAX; Rust returns Err",
            "Leading whitespace: C skips whitespace, Rust returns Err",
            "Trailing chars: C (strtol) stops, Rust returns Err",
            "Empty string: C returns 0, Rust returns Err",
            "Invalid format: C returns 0 (atoi) or sets errno, Rust returns Err",
        ],
    )

    @staticmethod
    def apply(
        string_ptr: z3.BitVecRef,
        string_len: z3.BitVecRef,
        target_width: int = 32,
        addr_width: int = 64,
    ) -> ModelResult:
        result = ModelResult()
        null = z3.BitVecVal(0, addr_width)

        result.error_condition = string_ptr == null

        c_result = z3.BitVec("atoi_c_result", target_width)
        rust_result = z3.BitVec("parse_rust_result", target_width)
        rust_ok = z3.Bool("parse_rust_ok")

        result.return_value = c_result

        # Empty string: C returns 0, Rust returns Err
        empty = string_len == z3.BitVecVal(0, string_len.size())
        result.constraints.append(z3.Implies(empty, c_result == z3.BitVecVal(0, target_width)))
        result.constraints.append(z3.Implies(empty, z3.Not(rust_ok)))

        # When parsing succeeds, values should match
        result.constraints.append(z3.Implies(rust_ok, c_result == rust_result))

        # Overflow: C is UB (atoi) or clamped (strtol), Rust is Err
        max_val = z3.BitVecVal((1 << (target_width - 1)) - 1, target_width)
        min_val = z3.BitVecVal(1 << (target_width - 1), target_width)
        overflow = z3.Or(c_result == max_val, c_result == min_val)
        result.divergence_condition = z3.And(overflow, z3.Not(rust_ok))

        return result


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

class StringFunctionModels:
    """Registry of all string function models."""

    models = {
        "strlen": StrlenModel,
        "strcmp": StrcmpModel,
        "strncmp": StrcmpModel,
        "strcpy": StrcpyModel,
        "strncpy": StrcpyModel,
        "strcat": StrcatModel,
        "strncat": StrcatModel,
        "sprintf": SprintfModel,
        "snprintf": SprintfModel,
        "atoi": AtoiModel,
        "strtol": AtoiModel,
        "strtoul": AtoiModel,
    }

    @classmethod
    def get_model(cls, func_name: str) -> Optional[type]:
        return cls.models.get(func_name)

    @classmethod
    def get_equivalence(cls, func_name: str) -> Optional[FunctionEquivalence]:
        model = cls.get_model(func_name)
        if model and hasattr(model, 'equivalence'):
            return model.equivalence
        return None

    @classmethod
    def all_equivalences(cls) -> List[FunctionEquivalence]:
        seen = set()
        result = []
        for model_cls in cls.models.values():
            if id(model_cls) not in seen and hasattr(model_cls, 'equivalence'):
                seen.add(id(model_cls))
                result.append(model_cls.equivalence)
        return result

    @classmethod
    def summary(cls) -> str:
        lines = ["String Function Models:"]
        for eq in cls.all_equivalences():
            lines.append(f"  {eq.summary()}")
            for dp in eq.divergence_points:
                lines.append(f"    ⚠ {dp}")
        return "\n".join(lines)
