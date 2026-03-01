"""
Memory function models: C ↔ Rust equivalences.

Models malloc/free ↔ Box::new/drop, calloc ↔ vec![0; n],
realloc ↔ Vec::resize, memcpy/memmove ↔ slice::copy_from_slice,
memset ↔ slice::fill, memcmp ↔ slice cmp.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict, Any, Tuple

import z3


# ---------------------------------------------------------------------------
# Base model class
# ---------------------------------------------------------------------------

class DivergenceLevel(Enum):
    """How likely a divergence is for this function pair."""
    NONE = auto()        # Semantically identical
    LOW = auto()         # Minor differences (e.g., alignment)
    MODERATE = auto()    # Input-dependent differences
    HIGH = auto()        # Fundamental semantic differences


@dataclass
class FunctionEquivalence:
    """Specification of equivalence between a C and Rust function."""
    c_function: str
    rust_equivalent: str
    divergence_level: DivergenceLevel
    preconditions: List[str] = field(default_factory=list)
    divergence_points: List[str] = field(default_factory=list)
    notes: str = ""

    def summary(self) -> str:
        return (
            f"{self.c_function} ↔ {self.rust_equivalent} "
            f"[{self.divergence_level.name}]"
        )


@dataclass
class ModelResult:
    """Result of applying a function model."""
    constraints: List[z3.BoolRef] = field(default_factory=list)
    return_value: Optional[z3.ExprRef] = None
    memory_effects: List[Dict[str, Any]] = field(default_factory=list)
    error_condition: Optional[z3.BoolRef] = None
    divergence_condition: Optional[z3.BoolRef] = None


# ---------------------------------------------------------------------------
# malloc model
# ---------------------------------------------------------------------------

class MallocModel:
    """
    Model for malloc(size) ↔ Box::new / Vec::with_capacity.
    
    C: malloc returns NULL on failure, undefined on size=0.
    Rust: Box::new panics on OOM (abort in most configs).
    Divergence: OOM handling, zero-size allocation behavior.
    """

    equivalence = FunctionEquivalence(
        c_function="malloc",
        rust_equivalent="Box::new / Vec::with_capacity",
        divergence_level=DivergenceLevel.MODERATE,
        preconditions=["size > 0 for defined behavior"],
        divergence_points=[
            "OOM: C returns NULL, Rust aborts",
            "Zero-size: C impl-defined, Rust allocates unique pointer",
            "Alignment: C aligns to max_align_t, Rust aligns to type",
        ],
    )

    @staticmethod
    def apply(
        size: z3.BitVecRef,
        addr_width: int = 64,
    ) -> ModelResult:
        """Apply the malloc model symbolically."""
        result = ModelResult()
        ptr = z3.BitVec("malloc_result", addr_width)

        # C semantics: ptr == NULL || ptr is valid
        null = z3.BitVecVal(0, addr_width)
        is_valid = z3.And(
            ptr != null,
            z3.UGT(ptr, z3.BitVecVal(0x1000, addr_width)),
        )
        result.constraints.append(z3.Or(ptr == null, is_valid))

        # Zero-size: implementation-defined
        zero_size = size == z3.BitVecVal(0, size.size())
        result.divergence_condition = zero_size

        result.return_value = ptr
        result.memory_effects.append({
            "type": "alloc",
            "address": ptr,
            "size": size,
        })

        return result


# ---------------------------------------------------------------------------
# free model
# ---------------------------------------------------------------------------

class FreeModel:
    """
    Model for free(ptr) ↔ drop(Box) / Vec::drop.
    
    C: free(NULL) is no-op, double-free is UB.
    Rust: Drop is called exactly once, double-free impossible via ownership.
    """

    equivalence = FunctionEquivalence(
        c_function="free",
        rust_equivalent="drop(Box<T>) / Vec::drop",
        divergence_level=DivergenceLevel.HIGH,
        preconditions=["ptr was returned by malloc/calloc/realloc or is NULL"],
        divergence_points=[
            "Double-free: C is UB, Rust prevented by type system",
            "Invalid pointer: C is UB, Rust prevented by type system",
            "NULL: C no-op, Rust N/A (Option<Box>)",
        ],
    )

    @staticmethod
    def apply(
        ptr: z3.BitVecRef,
        is_valid_alloc: z3.BoolRef,
        addr_width: int = 64,
    ) -> ModelResult:
        result = ModelResult()
        null = z3.BitVecVal(0, addr_width)

        # free(NULL) is a no-op in C
        is_null = ptr == null

        # Double-free is UB in C
        double_free = z3.And(z3.Not(is_null), z3.Not(is_valid_alloc))
        result.error_condition = double_free

        result.memory_effects.append({
            "type": "free",
            "address": ptr,
        })

        return result


# ---------------------------------------------------------------------------
# calloc model
# ---------------------------------------------------------------------------

class CallocModel:
    """
    Model for calloc(count, size) ↔ vec![0; count].
    
    C: calloc zeros memory, returns NULL on failure.
    Rust: Vec initializes to zeros, panics on OOM.
    Divergence: Overflow in count*size, OOM handling.
    """

    equivalence = FunctionEquivalence(
        c_function="calloc",
        rust_equivalent="vec![0; n]",
        divergence_level=DivergenceLevel.MODERATE,
        preconditions=["count * size does not overflow"],
        divergence_points=[
            "Overflow in count*size: C returns NULL, Rust panics",
            "OOM: C returns NULL, Rust aborts",
            "Zero elements: C impl-defined, Rust returns empty Vec",
        ],
    )

    @staticmethod
    def apply(
        count: z3.BitVecRef,
        size: z3.BitVecRef,
        addr_width: int = 64,
    ) -> ModelResult:
        result = ModelResult()
        ptr = z3.BitVec("calloc_result", addr_width)
        null = z3.BitVecVal(0, addr_width)

        # Check for multiplication overflow
        width = count.size()
        ext_count = z3.ZeroExt(width, count)
        ext_size = z3.ZeroExt(width, size)
        product = ext_count * ext_size
        high_bits = z3.Extract(2 * width - 1, width, product)
        overflow = high_bits != z3.BitVecVal(0, width)

        # On overflow: C returns NULL
        result.constraints.append(z3.Implies(overflow, ptr == null))
        result.divergence_condition = overflow

        # Normal case: valid pointer or NULL
        total_size = count * size
        result.constraints.append(z3.Or(ptr == null, z3.UGT(ptr, z3.BitVecVal(0x1000, addr_width))))

        result.return_value = ptr
        result.memory_effects.append({
            "type": "alloc_zeroed",
            "address": ptr,
            "size": total_size,
        })

        return result


# ---------------------------------------------------------------------------
# realloc model
# ---------------------------------------------------------------------------

class ReallocModel:
    """
    Model for realloc(ptr, size) ↔ Vec::resize.
    
    C: realloc(NULL, size) == malloc(size), realloc(ptr, 0) == free(ptr).
    Rust: Vec::resize always succeeds or panics.
    """

    equivalence = FunctionEquivalence(
        c_function="realloc",
        rust_equivalent="Vec::resize / Vec::reserve",
        divergence_level=DivergenceLevel.MODERATE,
        preconditions=["ptr was returned by malloc/calloc/realloc or is NULL"],
        divergence_points=[
            "OOM: C returns NULL (original ptr still valid), Rust panics",
            "Size 0: C impl-defined, Rust creates empty Vec",
            "NULL ptr: C behaves like malloc, Rust N/A",
        ],
    )

    @staticmethod
    def apply(
        ptr: z3.BitVecRef,
        new_size: z3.BitVecRef,
        addr_width: int = 64,
    ) -> ModelResult:
        result = ModelResult()
        new_ptr = z3.BitVec("realloc_result", addr_width)
        null = z3.BitVecVal(0, addr_width)

        # realloc(NULL, size) == malloc(size)
        is_null = ptr == null

        # realloc(ptr, 0): C impl-defined
        is_zero_size = new_size == z3.BitVecVal(0, new_size.size())
        result.divergence_condition = z3.Or(is_null, is_zero_size)

        # Result: NULL on failure, valid pointer on success
        result.constraints.append(
            z3.Or(new_ptr == null, z3.UGT(new_ptr, z3.BitVecVal(0x1000, addr_width)))
        )

        result.return_value = new_ptr
        result.memory_effects.append({
            "type": "realloc",
            "old_address": ptr,
            "new_address": new_ptr,
            "size": new_size,
        })

        return result


# ---------------------------------------------------------------------------
# memcpy model
# ---------------------------------------------------------------------------

class MemcpyModel:
    """
    Model for memcpy(dst, src, n) ↔ slice::copy_from_slice.
    
    C: UB if regions overlap, UB if NULL, UB if out of bounds.
    Rust: copy_from_slice panics if lengths differ, safe.
    """

    equivalence = FunctionEquivalence(
        c_function="memcpy / memmove",
        rust_equivalent="slice::copy_from_slice / slice::copy_within",
        divergence_level=DivergenceLevel.HIGH,
        preconditions=[
            "dst and src do not overlap (for memcpy)",
            "dst and src are valid for n bytes",
        ],
        divergence_points=[
            "Overlap: memcpy is UB, memmove handles it; Rust copy_from_slice panics if overlap",
            "NULL pointers: C is UB, Rust panics",
            "Out of bounds: C is UB, Rust panics",
        ],
    )

    @staticmethod
    def apply(
        dst: z3.BitVecRef,
        src: z3.BitVecRef,
        n: z3.BitVecRef,
        memory: z3.ArrayRef,
        addr_width: int = 64,
    ) -> ModelResult:
        result = ModelResult()
        null = z3.BitVecVal(0, addr_width)

        # NULL check
        dst_null = dst == null
        src_null = src == null
        result.error_condition = z3.Or(dst_null, src_null)

        # Overlap check (for memcpy vs memmove)
        n_ext = z3.ZeroExt(addr_width - n.size(), n) if n.size() < addr_width else n
        overlap = z3.And(
            z3.ULT(dst, src + n_ext),
            z3.ULT(src, dst + n_ext),
            dst != src,
        )
        result.divergence_condition = overlap

        # Memory effect: copy n bytes from src to dst
        result.memory_effects.append({
            "type": "copy",
            "dst": dst,
            "src": src,
            "size": n,
        })

        result.return_value = dst
        return result


# ---------------------------------------------------------------------------
# memset model
# ---------------------------------------------------------------------------

class MemsetModel:
    """
    Model for memset(ptr, value, n) ↔ slice::fill.
    
    C: UB if ptr is NULL or out of bounds.
    Rust: slice::fill is safe, panics if slice is invalid.
    """

    equivalence = FunctionEquivalence(
        c_function="memset",
        rust_equivalent="slice::fill / [val; n]",
        divergence_level=DivergenceLevel.LOW,
        preconditions=["ptr is valid for n bytes"],
        divergence_points=[
            "NULL pointer: C is UB, Rust panics",
            "Out of bounds: C is UB, Rust panics",
            "Value truncation: C truncates to unsigned char",
        ],
    )

    @staticmethod
    def apply(
        ptr: z3.BitVecRef,
        value: z3.BitVecRef,
        n: z3.BitVecRef,
        addr_width: int = 64,
    ) -> ModelResult:
        result = ModelResult()
        null = z3.BitVecVal(0, addr_width)

        result.error_condition = ptr == null

        # Value is truncated to byte
        byte_val = z3.Extract(7, 0, value) if value.size() > 8 else value

        result.memory_effects.append({
            "type": "fill",
            "address": ptr,
            "value": byte_val,
            "size": n,
        })

        result.return_value = ptr
        return result


# ---------------------------------------------------------------------------
# memcmp model
# ---------------------------------------------------------------------------

class MemcmpModel:
    """
    Model for memcmp(a, b, n) ↔ slice comparison.
    
    C: Returns <0, 0, >0 (exact value is impl-defined).
    Rust: Returns Ordering::Less/Equal/Greater.
    Divergence: Sign of non-zero return value.
    """

    equivalence = FunctionEquivalence(
        c_function="memcmp",
        rust_equivalent="slice::cmp / slice::eq",
        divergence_level=DivergenceLevel.LOW,
        preconditions=["Both pointers valid for n bytes"],
        divergence_points=[
            "Return value: C returns any negative/positive int, Rust returns -1/0/1",
            "NULL pointers: C is UB, Rust panics",
        ],
    )

    @staticmethod
    def apply(
        a: z3.BitVecRef,
        b: z3.BitVecRef,
        n: z3.BitVecRef,
        memory: z3.ArrayRef,
        addr_width: int = 64,
    ) -> ModelResult:
        result = ModelResult()
        null = z3.BitVecVal(0, addr_width)

        result.error_condition = z3.Or(a == null, b == null)

        # Return value: sign matches but exact value may differ
        ret = z3.BitVec("memcmp_result", 32)

        # The sign of the return value is defined: negative if a < b, etc.
        # But exact value is implementation-defined
        result.return_value = ret

        # Divergence: C might return 42, Rust returns 1
        result.divergence_condition = z3.And(
            ret != z3.BitVecVal(0, 32),
            ret != z3.BitVecVal(1, 32),
            ret != z3.BitVecVal(-1, 32),  # Wraps in unsigned
        )

        return result


# ---------------------------------------------------------------------------
# Collection of all memory function models
# ---------------------------------------------------------------------------

class MemoryFunctionModels:
    """Registry of all memory function models."""

    models = {
        "malloc": MallocModel,
        "free": FreeModel,
        "calloc": CallocModel,
        "realloc": ReallocModel,
        "memcpy": MemcpyModel,
        "memmove": MemcpyModel,
        "memset": MemsetModel,
        "memcmp": MemcmpModel,
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
        lines = ["Memory Function Models:"]
        for eq in cls.all_equivalences():
            lines.append(f"  {eq.summary()}")
            for dp in eq.divergence_points:
                lines.append(f"    ⚠ {dp}")
        return "\n".join(lines)
