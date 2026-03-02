"""
Symbolic memory model for pointer, struct, and heap reasoning.

Uses separation-logic concepts and Z3's array theory to extend the SMT
encoder with a byte-addressable heap, allocation tracking, provenance
tracking for Rust's ownership model, struct layout computation, and
cross-language heap equivalence checking.

Classes
-------
SymbolicHeap
    Byte-level memory modelled as a Z3 array (address → byte).
SeparationLogicEncoder
    Encodes spatial separation, frame conditions, and points-to assertions.
StructLayout
    Computes field offsets with C/Rust ABI padding and alignment rules.
ProvenanceTracker
    Tracks pointer provenance for Rust's ownership/borrowing model.
PointerArithmeticEncoder
    Encodes GEP, pointer comparison/difference with bounds checking.
HeapEquivalenceChecker
    Asserts value-level equivalence of two heaps across shared pointers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Sequence, Tuple

import z3

from ..ir.types import (
    IRType,
    IntType,
    PointerType,
    StructType,
    ArrayType,
    FloatType,
    VoidType,
    Signedness,
)
from ..semantics.semantic_config import (
    SemanticConfig,
    PointerModel,
    ArrayBoundsModel,
)
from .encoder import EncodingContext


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BYTE_BITS = 8
_NULL_ADDR = 0


# ---------------------------------------------------------------------------
# Allocation descriptor
# ---------------------------------------------------------------------------

@dataclass
class Allocation:
    """Metadata for a single heap allocation.

    Attributes
    ----------
    alloc_id : int
        Unique allocation identifier.
    base : z3.BitVecRef
        Symbolic base address.
    size : z3.BitVecRef
        Symbolic size in bytes.
    align : int
        Required alignment in bytes.
    freed : z3.BoolRef
        Symbolic flag: ``True`` once the region has been freed.
    """

    alloc_id: int
    base: z3.BitVecRef
    size: z3.BitVecRef
    align: int
    freed: z3.BoolRef


# ---------------------------------------------------------------------------
# SymbolicHeap
# ---------------------------------------------------------------------------

class SymbolicHeap:
    """Byte-addressable symbolic heap backed by Z3 arrays.

    Memory is modelled as ``Array(BitVec(ptr_width), BitVec(8))``.  Reads
    and writes at arbitrary widths are composed from individual byte
    accesses according to the configured endianness.

    Parameters
    ----------
    ctx : EncodingContext
        Shared encoding context for declarations and assertions.
    config : SemanticConfig
        Semantic configuration (pointer size, endianness, …).
    name : str
        Namespace prefix for Z3 symbols produced by this heap.
    """

    def __init__(
        self,
        ctx: EncodingContext,
        config: SemanticConfig,
        name: str = "heap",
    ) -> None:
        self.ctx = ctx
        self.config = config
        self.name = name
        self.ptr_width: int = config.pointer_size
        self.little_endian: bool = config.endianness == "little"

        # Memory state (SSA-style versioning for functional updates)
        self._version: int = 0
        self._mem: z3.ArrayRef = ctx.declare_array(
            f"{name}_mem_0",
            z3.BitVecSort(self.ptr_width),
            z3.BitVecSort(_BYTE_BITS),
        )

        # Allocation bookkeeping
        self._alloc_counter: int = 0
        self.allocations: Dict[int, Allocation] = {}

    # -- low-level byte access ---------------------------------------------

    def _addr_const(self, val: int) -> z3.BitVecRef:
        """Create a concrete address constant."""
        return z3.BitVecVal(val, self.ptr_width)

    def _byte_offset(self, base: z3.BitVecRef, offset: int) -> z3.BitVecRef:
        """Return ``base + offset`` as a pointer-width bitvector."""
        return base + z3.BitVecVal(offset, self.ptr_width)

    # -- multi-byte read / write -------------------------------------------

    def read_bytes(self, addr: z3.BitVecRef, width_bytes: int) -> z3.BitVecRef:
        """Read *width_bytes* bytes starting at *addr* and return a
        ``BitVec(width_bytes * 8)`` value.

        Bytes are composed according to the configured endianness.
        """
        if width_bytes <= 0:
            raise ValueError("width_bytes must be positive")

        byte_exprs: List[z3.BitVecRef] = []
        for i in range(width_bytes):
            byte_addr = self._byte_offset(addr, i)
            byte_val = z3.Select(self._mem, byte_addr)
            byte_exprs.append(byte_val)

        if self.little_endian:
            byte_exprs = list(reversed(byte_exprs))

        result = byte_exprs[0]
        for b in byte_exprs[1:]:
            result = z3.Concat(result, b)
        return result

    def write_bytes(
        self,
        addr: z3.BitVecRef,
        value: z3.BitVecRef,
        width_bytes: int,
    ) -> None:
        """Write *width_bytes* bytes of *value* starting at *addr*.

        A new memory version is created (functional SSA update).
        """
        if width_bytes <= 0:
            raise ValueError("width_bytes must be positive")

        total_bits = width_bytes * _BYTE_BITS
        if value.size() != total_bits:
            if value.size() < total_bits:
                value = z3.ZeroExt(total_bits - value.size(), value)
            else:
                value = z3.Extract(total_bits - 1, 0, value)

        new_mem = self._mem
        for i in range(width_bytes):
            if self.little_endian:
                low = i * _BYTE_BITS
                high = low + _BYTE_BITS - 1
            else:
                high = total_bits - 1 - i * _BYTE_BITS
                low = high - _BYTE_BITS + 1

            byte_val = z3.Extract(high, low, value)
            byte_addr = self._byte_offset(addr, i)
            new_mem = z3.Store(new_mem, byte_addr, byte_val)

        self._version += 1
        mem_name = f"{self.name}_mem_{self._version}"
        self.ctx.declarations[mem_name] = new_mem
        self._mem = new_mem

    # -- allocation management ---------------------------------------------

    def allocate(
        self,
        size: z3.BitVecRef,
        align: int = 1,
    ) -> Allocation:
        """Model a fresh heap allocation.

        Returns an :class:`Allocation` whose ``base`` is a fresh symbolic
        pointer constrained to be non-null, properly aligned, and
        non-overlapping with every prior allocation.

        Parameters
        ----------
        size : z3.BitVecRef
            Symbolic allocation size in bytes (pointer-width bitvector).
        align : int
            Required alignment in bytes (must be a power of two).
        """
        self._alloc_counter += 1
        aid = self._alloc_counter

        base = self.ctx.declare_bv(
            f"{self.name}_alloc_{aid}_base", self.ptr_width
        )
        freed = self.ctx.declare_bool(f"{self.name}_alloc_{aid}_freed")

        # Initially not freed
        self.ctx.assert_hard(z3.Not(freed))

        # Non-null
        self.ctx.assert_hard(base != self._addr_const(_NULL_ADDR))

        # Alignment: base % align == 0
        if align > 1:
            align_mask = self._addr_const(align - 1)
            self.ctx.assert_hard((base & align_mask) == self._addr_const(0))

        # Positive size
        self.ctx.assert_hard(z3.UGT(size, z3.BitVecVal(0, self.ptr_width)))

        # No wrap-around: base + size does not wrap past the address space
        self.ctx.assert_hard(
            z3.UGE(base + size, base)
        )

        alloc = Allocation(
            alloc_id=aid, base=base, size=size, align=align, freed=freed
        )

        # Spatial separation from all prior allocations
        for prev in self.allocations.values():
            self._assert_disjoint(alloc, prev)

        self.allocations[aid] = alloc
        return alloc

    def free(self, alloc: Allocation) -> z3.BoolRef:
        """Mark *alloc* as freed.

        Returns a ``BoolRef`` that is ``True`` when the free is valid
        (i.e. the allocation was not already freed).
        """
        valid = z3.Not(alloc.freed)

        # Create a new freed flag to model the post-free state
        new_freed_name = self.ctx.fresh(f"{self.name}_freed_{alloc.alloc_id}")
        new_freed = self.ctx.declare_bool(new_freed_name)
        self.ctx.assert_hard(new_freed)

        # Update the allocation record
        self.allocations[alloc.alloc_id] = Allocation(
            alloc_id=alloc.alloc_id,
            base=alloc.base,
            size=alloc.size,
            align=alloc.align,
            freed=new_freed,
        )
        return valid

    # -- disjointness helpers ----------------------------------------------

    def _assert_disjoint(self, a: Allocation, b: Allocation) -> None:
        """Assert that allocations *a* and *b* do not overlap."""
        a_end = a.base + a.size
        b_end = b.base + b.size
        self.ctx.assert_hard(
            z3.Or(z3.UGE(a.base, b_end), z3.UGE(b.base, a_end))
        )

    # -- convenience -------------------------------------------------------

    @property
    def mem(self) -> z3.ArrayRef:
        """Return the current memory state."""
        return self._mem

    def snapshot(self) -> z3.ArrayRef:
        """Return a snapshot of the current memory for later comparison."""
        return self._mem


# ---------------------------------------------------------------------------
# SeparationLogicEncoder
# ---------------------------------------------------------------------------

class SeparationLogicEncoder:
    """Encodes separation-logic style assertions over a symbolic heap.

    Provides primitives for allocation, load, store, free, frame
    conditions, and null checks that integrate with the shared
    :class:`EncodingContext`.

    Parameters
    ----------
    ctx : EncodingContext
        Shared encoding context.
    config : SemanticConfig
        Language-level semantics (pointer model, bounds checking, …).
    heap_name : str
        Namespace prefix for the underlying :class:`SymbolicHeap`.
    """

    def __init__(
        self,
        ctx: EncodingContext,
        config: SemanticConfig,
        heap_name: str = "heap",
    ) -> None:
        self.ctx = ctx
        self.config = config
        self.heap = SymbolicHeap(ctx, config, name=heap_name)
        self.ptr_width: int = config.pointer_size

    # -- core operations ---------------------------------------------------

    def encode_alloc(
        self,
        size: z3.BitVecRef,
        align: int = 1,
    ) -> Tuple[z3.BitVecRef, Allocation]:
        """Encode a heap allocation.

        Parameters
        ----------
        size : z3.BitVecRef
            Allocation size in bytes (pointer-width bitvector).
        align : int
            Required alignment in bytes.

        Returns
        -------
        (base_addr, allocation)
            The symbolic base address and the allocation record.
        """
        alloc = self.heap.allocate(size, align)
        return alloc.base, alloc

    def encode_load(
        self,
        addr: z3.BitVecRef,
        width: int,
    ) -> z3.BitVecRef:
        """Encode a load of *width* bytes from *addr*.

        If array bounds checking is enabled, an in-bounds assertion is
        added for every live allocation that could contain *addr*.

        Returns the loaded bitvector value.
        """
        if self.config.array_bounds != ArrayBoundsModel.NoCheck:
            self._assert_addr_in_bounds(addr, width)

        return self.heap.read_bytes(addr, width)

    def encode_store(
        self,
        addr: z3.BitVecRef,
        value: z3.BitVecRef,
        width: int,
    ) -> None:
        """Encode a store of *width* bytes of *value* at *addr*.

        When bounds checking is enabled the store address is asserted to
        be inside a live allocation.
        """
        if self.config.array_bounds != ArrayBoundsModel.NoCheck:
            self._assert_addr_in_bounds(addr, width)

        self.heap.write_bytes(addr, value, width)

    def encode_free(self, addr: z3.BitVecRef) -> z3.BoolRef:
        """Encode a ``free(addr)``.

        Finds the allocation whose base matches *addr* and marks it
        freed.  Returns a boolean indicating the free was valid
        (double-free detection).
        """
        conditions: List[z3.BoolRef] = []
        for alloc in self.heap.allocations.values():
            match_cond = addr == alloc.base
            valid_cond = self.heap.free(alloc)
            conditions.append(z3.And(match_cond, valid_cond))

        if not conditions:
            return z3.BoolVal(False)

        # At least one allocation must match
        valid_free = z3.Or(*conditions) if len(conditions) > 1 else conditions[0]
        self.ctx.assert_hard(valid_free)
        return valid_free

    def encode_frame_condition(
        self,
        alloc1: Allocation,
        alloc2: Allocation,
    ) -> z3.BoolRef:
        """Assert that *alloc1* and *alloc2* are spatially separate.

        This is the core *frame rule*: a write to one allocation does
        not affect the other.

        Returns the disjointness constraint (also added to the context).
        """
        a_end = alloc1.base + alloc1.size
        b_end = alloc2.base + alloc2.size
        frame = z3.Or(
            z3.UGE(alloc1.base, b_end),
            z3.UGE(alloc2.base, a_end),
        )
        self.ctx.assert_hard(frame)
        return frame

    def encode_null_check(self, addr: z3.BitVecRef) -> z3.BoolRef:
        """Return a constraint that is ``True`` when *addr* is null."""
        null = z3.BitVecVal(_NULL_ADDR, self.ptr_width)
        return addr == null

    def encode_memcpy(
        self,
        dst: z3.BitVecRef,
        src: z3.BitVecRef,
        length: int,
    ) -> None:
        """Encode ``memcpy(dst, src, length)`` as byte-by-byte array copy."""
        for i in range(min(length, 256)):
            off = z3.BitVecVal(i, self.ptr_width)
            byte_val = self.heap.read_bytes(src + off, 1)
            self.heap.write_bytes(dst + off, byte_val, 1)

    def encode_memset(
        self,
        dst: z3.BitVecRef,
        value: z3.BitVecRef,
        length: int,
    ) -> None:
        """Encode ``memset(dst, value, length)`` as byte-by-byte fill."""
        byte_val = z3.Extract(7, 0, value) if value.size() > 8 else value
        if byte_val.size() < 8:
            byte_val = z3.ZeroExt(8 - byte_val.size(), byte_val)
        for i in range(min(length, 256)):
            off = z3.BitVecVal(i, self.ptr_width)
            self.heap.write_bytes(dst + off, byte_val, 1)

    def encode_points_to(
        self,
        addr: z3.BitVecRef,
        value: z3.BitVecRef,
        width: int,
    ) -> z3.BoolRef:
        """Assert that memory at *addr* contains *value*.

        This is the separation-logic ``addr ↦ value`` predicate.
        """
        loaded = self.heap.read_bytes(addr, width)
        # Coerce sizes if needed
        if loaded.size() != value.size():
            if value.size() < loaded.size():
                value = z3.ZeroExt(loaded.size() - value.size(), value)
            else:
                value = z3.Extract(loaded.size() - 1, 0, value)
        constraint = loaded == value
        self.ctx.assert_hard(constraint)
        return constraint

    # -- internal helpers --------------------------------------------------

    def _assert_addr_in_bounds(self, addr: z3.BitVecRef, width: int) -> None:
        """Assert *addr .. addr+width* falls within some live allocation."""
        access_end = addr + z3.BitVecVal(width, self.ptr_width)
        in_some_alloc: List[z3.BoolRef] = []
        for alloc in self.heap.allocations.values():
            alloc_end = alloc.base + alloc.size
            in_bounds = z3.And(
                z3.UGE(addr, alloc.base),
                z3.ULE(access_end, alloc_end),
                z3.Not(alloc.freed),
            )
            in_some_alloc.append(in_bounds)

        if in_some_alloc:
            cond = (
                z3.Or(*in_some_alloc)
                if len(in_some_alloc) > 1
                else in_some_alloc[0]
            )
            self.ctx.assert_hard(cond)


# ---------------------------------------------------------------------------
# StructLayout
# ---------------------------------------------------------------------------

class StructLayout:
    """Computes field offsets and encodes field-access pointer arithmetic
    for struct types, respecting C or Rust ABI padding/alignment rules.

    Parameters
    ----------
    config : SemanticConfig
        Determines ABI (C vs Rust) and pointer width.
    """

    def __init__(self, config: SemanticConfig) -> None:
        self.config = config
        self.ptr_width: int = config.pointer_size

    # -- layout computation ------------------------------------------------

    def compute_offsets(
        self,
        struct_ty: StructType,
    ) -> List[int]:
        """Return field offsets in *bytes* for *struct_ty*.

        Uses the struct's own ``packed`` flag.  When the language is Rust
        and the struct is not ``#[repr(C)]``, fields may be reordered
        for optimal packing; however, since ``StructType.fields`` is
        already in declared order, we rely on the IR having resolved any
        reordering and compute offsets sequentially.
        """
        offsets: List[int] = []
        current_byte = 0

        for sf in struct_ty.fields:
            field_align = sf.type.align_bytes(self.ptr_width)

            if struct_ty.packed:
                field_align = 1

            current_byte = _align_up_int(current_byte, field_align)
            offsets.append(current_byte)
            current_byte += sf.type.size_bytes(self.ptr_width)

        return offsets

    def compute_size(self, struct_ty: StructType) -> int:
        """Total size of the struct in bytes (including tail padding)."""
        if not struct_ty.fields:
            return 0
        offsets = self.compute_offsets(struct_ty)
        last_field = struct_ty.fields[-1]
        raw_end = offsets[-1] + last_field.type.size_bytes(self.ptr_width)

        if struct_ty.packed:
            return raw_end

        overall_align = max(
            f.type.align_bytes(self.ptr_width) for f in struct_ty.fields
        )
        return _align_up_int(raw_end, overall_align)

    def compute_alignment(self, struct_ty: StructType) -> int:
        """Overall alignment of the struct in bytes."""
        if not struct_ty.fields:
            return 1
        if struct_ty.packed:
            return 1
        return max(
            f.type.align_bytes(self.ptr_width) for f in struct_ty.fields
        )

    # -- Z3 field access ---------------------------------------------------

    def encode_field_access(
        self,
        struct_ptr: z3.BitVecRef,
        struct_ty: StructType,
        field_idx: int,
    ) -> z3.BitVecRef:
        """Return a Z3 expression for the address of field *field_idx*.

        Parameters
        ----------
        struct_ptr : z3.BitVecRef
            Pointer to the beginning of the struct.
        struct_ty : StructType
            The IR struct type.
        field_idx : int
            Zero-based index of the target field.

        Returns
        -------
        z3.BitVecRef
            ``struct_ptr + field_offset`` as a pointer-width bitvector.

        Raises
        ------
        IndexError
            If *field_idx* is out of range.
        """
        if field_idx < 0 or field_idx >= len(struct_ty.fields):
            raise IndexError(
                f"Field index {field_idx} out of range for struct "
                f"with {len(struct_ty.fields)} fields"
            )
        offsets = self.compute_offsets(struct_ty)
        offset_val = z3.BitVecVal(offsets[field_idx], self.ptr_width)
        return struct_ptr + offset_val

    def encode_all_field_addrs(
        self,
        struct_ptr: z3.BitVecRef,
        struct_ty: StructType,
    ) -> List[z3.BitVecRef]:
        """Return Z3 expressions for the addresses of all fields."""
        offsets = self.compute_offsets(struct_ty)
        return [
            struct_ptr + z3.BitVecVal(off, self.ptr_width)
            for off in offsets
        ]


# ---------------------------------------------------------------------------
# ProvenanceTracker
# ---------------------------------------------------------------------------

class ProvenanceTracker:
    """Tracks pointer provenance for Rust's ownership/borrowing model.

    Each live pointer is associated with a symbolic *provenance tag*
    (an integer).  Derived pointers inherit their parent's tag, and
    freed/invalidated pointers have their tags removed from the valid
    set so that subsequent use is detected as an error.

    Parameters
    ----------
    ctx : EncodingContext
        Shared encoding context.
    config : SemanticConfig
        Determines whether provenance tracking is active
        (``PointerModel.Provenance``).
    """

    def __init__(self, ctx: EncodingContext, config: SemanticConfig) -> None:
        self.ctx = ctx
        self.config = config
        self.ptr_width: int = config.pointer_size
        self._enabled: bool = config.pointer_model == PointerModel.Provenance

        # Tag counter
        self._tag_counter: int = 0

        # Map from pointer name → provenance tag (Z3 int)
        self._provenance: Dict[str, z3.BitVecRef] = {}

        # Valid-tag set modelled as an uninterpreted function
        # valid_tag : BitVec(32) → Bool
        self._valid_fn = z3.Function(
            "provenance_valid",
            z3.BitVecSort(32),
            z3.BoolSort(),
        )

        # Derivation relation: child_tag → parent_tag
        self._parent: Dict[str, str] = {}

    # -- public API --------------------------------------------------------

    def new_provenance(self, ptr_name: str) -> z3.BitVecRef:
        """Create a fresh provenance tag for *ptr_name*.

        The tag is asserted to be valid.

        Returns the symbolic tag value.
        """
        self._tag_counter += 1
        tag_name = f"prov_tag_{self._tag_counter}"
        tag = self.ctx.declare_bv(tag_name, 32)

        # Tag is unique (different from all previous tags)
        for existing_tag in self._provenance.values():
            self.ctx.assert_hard(tag != existing_tag)

        # Tag is valid
        if self._enabled:
            self.ctx.assert_hard(self._valid_fn(tag))

        self._provenance[ptr_name] = tag
        return tag

    def derive(
        self,
        parent_name: str,
        child_name: str,
        offset: z3.BitVecRef,
    ) -> z3.BitVecRef:
        """Derive a child pointer's provenance from a parent.

        The child inherits the parent's tag.  This models pointer
        arithmetic and field access within a single allocation.

        Parameters
        ----------
        parent_name : str
            Name of the parent pointer in the encoding context.
        child_name : str
            Name for the derived pointer.
        offset : z3.BitVecRef
            Byte offset from the parent pointer (unused for provenance
            but recorded for the derivation chain).

        Returns
        -------
        z3.BitVecRef
            The child's provenance tag (same as parent's).
        """
        parent_tag = self._provenance.get(parent_name)
        if parent_tag is None:
            return self.new_provenance(child_name)

        self._provenance[child_name] = parent_tag
        self._parent[child_name] = parent_name
        return parent_tag

    def check_valid(self, ptr_name: str) -> z3.BoolRef:
        """Return (and assert) that *ptr_name*'s provenance is valid.

        In flat-pointer mode this is trivially ``True``.
        """
        if not self._enabled:
            return z3.BoolVal(True)

        tag = self._provenance.get(ptr_name)
        if tag is None:
            return z3.BoolVal(False)

        validity = self._valid_fn(tag)
        self.ctx.assert_hard(validity)
        return validity

    def invalidate(self, ptr_name: str) -> None:
        """Mark *ptr_name*'s provenance as dead (e.g. after ``free``).

        Also invalidates all pointers derived from *ptr_name*.
        """
        if not self._enabled:
            return

        tag = self._provenance.get(ptr_name)
        if tag is None:
            return

        self.ctx.assert_hard(z3.Not(self._valid_fn(tag)))

        # Transitively invalidate all children
        for child, parent in list(self._parent.items()):
            if parent == ptr_name:
                self.invalidate(child)

    def get_tag(self, ptr_name: str) -> Optional[z3.BitVecRef]:
        """Return the provenance tag for *ptr_name*, or ``None``."""
        return self._provenance.get(ptr_name)


# ---------------------------------------------------------------------------
# PointerArithmeticEncoder
# ---------------------------------------------------------------------------

class PointerArithmeticEncoder:
    """Encodes pointer arithmetic with proper bounds checking.

    Handles GEP (``GetElementPtr``), pointer comparison, pointer
    difference, and null-pointer semantics.

    Parameters
    ----------
    ctx : EncodingContext
        Shared encoding context.
    config : SemanticConfig
        Controls bounds checking and pointer model.
    heap : SymbolicHeap
        The heap that owns the allocations (used for bounds queries).
    provenance : ProvenanceTracker or None
        Optional provenance tracker for Rust-style pointer reasoning.
    """

    def __init__(
        self,
        ctx: EncodingContext,
        config: SemanticConfig,
        heap: SymbolicHeap,
        provenance: Optional[ProvenanceTracker] = None,
    ) -> None:
        self.ctx = ctx
        self.config = config
        self.heap = heap
        self.provenance = provenance
        self.ptr_width: int = config.pointer_size

    # -- GEP ---------------------------------------------------------------

    def encode_gep(
        self,
        base: z3.BitVecRef,
        indices: Sequence[z3.BitVecRef],
        pointee_type: IRType,
        struct_ty: Optional[StructType] = None,
    ) -> z3.BitVecRef:
        """Encode a GEP (GetElementPtr) instruction.

        Computes ``base + sum(index_i * stride_i)`` where the stride is
        derived from *pointee_type* (and optionally *struct_ty* for
        struct field access).

        Parameters
        ----------
        base : z3.BitVecRef
            Base pointer.
        indices : sequence of z3.BitVecRef
            Index operands.  For a simple ``ptr[i]`` there is one index.
            For nested struct/array access there may be multiple.
        pointee_type : IRType
            The type the base pointer points to.
        struct_ty : StructType or None
            If the GEP traverses struct fields, supply the struct type
            so that field-offset computation uses ABI-aware layout.

        Returns
        -------
        z3.BitVecRef
            The resulting pointer value.
        """
        result = base
        current_ty: IRType = pointee_type

        layout = StructLayout(self.config)

        for idx_pos, idx in enumerate(indices):
            # Coerce index to pointer width
            idx = self._coerce_to_ptr_width(idx)

            if isinstance(current_ty, StructType):
                # For struct GEPs the index must be a constant
                if z3.is_bv_value(idx):
                    field_idx = idx.as_long()
                else:
                    field_idx = 0
                offsets = layout.compute_offsets(current_ty)
                if field_idx < len(offsets):
                    offset_bv = z3.BitVecVal(offsets[field_idx], self.ptr_width)
                    result = result + offset_bv
                    current_ty = current_ty.fields[field_idx].type
                continue

            if isinstance(current_ty, ArrayType):
                elem_size = current_ty.element.size_bytes(self.config.pointer_size)
                stride = z3.BitVecVal(elem_size, self.ptr_width)
                result = result + idx * stride

                if self.config.array_bounds != ArrayBoundsModel.NoCheck:
                    length_bv = z3.BitVecVal(current_ty.length, self.ptr_width)
                    self.ctx.assert_hard(z3.ULT(idx, length_bv))

                current_ty = current_ty.element
                continue

            # Scalar pointee: simple offset by element size
            elem_size = current_ty.size_bytes(self.config.pointer_size)
            if elem_size == 0:
                elem_size = 1
            stride = z3.BitVecVal(elem_size, self.ptr_width)
            result = result + idx * stride

        return result

    # -- pointer comparison ------------------------------------------------

    def encode_ptr_eq(
        self,
        lhs: z3.BitVecRef,
        rhs: z3.BitVecRef,
    ) -> z3.BoolRef:
        """Pointer equality comparison."""
        return lhs == rhs

    def encode_ptr_ne(
        self,
        lhs: z3.BitVecRef,
        rhs: z3.BitVecRef,
    ) -> z3.BoolRef:
        """Pointer inequality comparison."""
        return lhs != rhs

    def encode_ptr_lt(
        self,
        lhs: z3.BitVecRef,
        rhs: z3.BitVecRef,
    ) -> z3.BoolRef:
        """Unsigned less-than comparison on pointers.

        In C, comparing pointers from different allocations is UB.
        In Rust (safe code), it is not allowed.  We model the
        comparison unconditionally but add a same-allocation
        assertion when provenance tracking is enabled.
        """
        if self.provenance and self.config.pointer_model == PointerModel.Provenance:
            self._assert_same_provenance(lhs, rhs)
        return z3.ULT(lhs, rhs)

    def encode_ptr_le(
        self,
        lhs: z3.BitVecRef,
        rhs: z3.BitVecRef,
    ) -> z3.BoolRef:
        """Unsigned less-than-or-equal comparison on pointers."""
        if self.provenance and self.config.pointer_model == PointerModel.Provenance:
            self._assert_same_provenance(lhs, rhs)
        return z3.ULE(lhs, rhs)

    def encode_ptr_gt(
        self,
        lhs: z3.BitVecRef,
        rhs: z3.BitVecRef,
    ) -> z3.BoolRef:
        """Unsigned greater-than comparison on pointers."""
        if self.provenance and self.config.pointer_model == PointerModel.Provenance:
            self._assert_same_provenance(lhs, rhs)
        return z3.UGT(lhs, rhs)

    def encode_ptr_ge(
        self,
        lhs: z3.BitVecRef,
        rhs: z3.BitVecRef,
    ) -> z3.BoolRef:
        """Unsigned greater-than-or-equal comparison on pointers."""
        if self.provenance and self.config.pointer_model == PointerModel.Provenance:
            self._assert_same_provenance(lhs, rhs)
        return z3.UGE(lhs, rhs)

    # -- pointer difference ------------------------------------------------

    def encode_ptr_diff(
        self,
        lhs: z3.BitVecRef,
        rhs: z3.BitVecRef,
        pointee_type: IRType,
    ) -> z3.BitVecRef:
        """Encode ``(lhs - rhs) / sizeof(pointee)``.

        Result is a signed pointer-width integer.  When provenance
        tracking is active, both pointers must belong to the same
        allocation.
        """
        if self.provenance and self.config.pointer_model == PointerModel.Provenance:
            self._assert_same_provenance(lhs, rhs)

        byte_diff = lhs - rhs
        elem_size = pointee_type.size_bytes(self.config.pointer_size)
        if elem_size <= 1:
            return byte_diff

        divisor = z3.BitVecVal(elem_size, self.ptr_width)
        return byte_diff / divisor  # signed division

    # -- null semantics ----------------------------------------------------

    def encode_null(self) -> z3.BitVecRef:
        """Return the null pointer constant."""
        return z3.BitVecVal(_NULL_ADDR, self.ptr_width)

    def encode_is_null(self, addr: z3.BitVecRef) -> z3.BoolRef:
        """Return ``True`` iff *addr* is null."""
        return addr == self.encode_null()

    def encode_is_not_null(self, addr: z3.BitVecRef) -> z3.BoolRef:
        """Return ``True`` iff *addr* is non-null."""
        return addr != self.encode_null()

    # -- bounds checking ---------------------------------------------------

    def encode_in_bounds(
        self,
        ptr: z3.BitVecRef,
        access_size: int,
    ) -> z3.BoolRef:
        """Assert that ``ptr .. ptr + access_size`` is within a live
        allocation.  Returns the disjunction of all per-allocation
        in-bounds predicates.
        """
        access_end = ptr + z3.BitVecVal(access_size, self.ptr_width)
        options: List[z3.BoolRef] = []
        for alloc in self.heap.allocations.values():
            alloc_end = alloc.base + alloc.size
            in_bounds = z3.And(
                z3.UGE(ptr, alloc.base),
                z3.ULE(access_end, alloc_end),
                z3.Not(alloc.freed),
            )
            options.append(in_bounds)

        if not options:
            return z3.BoolVal(False)
        return z3.Or(*options) if len(options) > 1 else options[0]

    # -- internal helpers --------------------------------------------------

    def _coerce_to_ptr_width(self, bv: z3.BitVecRef) -> z3.BitVecRef:
        """Extend or truncate *bv* to pointer width."""
        if bv.size() == self.ptr_width:
            return bv
        if bv.size() < self.ptr_width:
            return z3.SignExt(self.ptr_width - bv.size(), bv)
        return z3.Extract(self.ptr_width - 1, 0, bv)

    def _assert_same_provenance(
        self,
        lhs: z3.BitVecRef,
        rhs: z3.BitVecRef,
    ) -> None:
        """Assert that *lhs* and *rhs* belong to the same allocation.

        This is a simplified model: we assert that both addresses fall
        within the bounds of at least one common allocation.
        """
        for alloc in self.heap.allocations.values():
            alloc_end = alloc.base + alloc.size
            lhs_in = z3.And(
                z3.UGE(lhs, alloc.base), z3.ULT(lhs, alloc_end)
            )
            rhs_in = z3.And(
                z3.UGE(rhs, alloc.base), z3.ULT(rhs, alloc_end)
            )
            both_in = z3.And(lhs_in, rhs_in, z3.Not(alloc.freed))
            # We add this as an assumption so the solver can use it
            self.ctx.assert_assume(both_in)
            return  # only need one matching allocation

        # No allocations tracked — nothing to assert
        return


# ---------------------------------------------------------------------------
# HeapEquivalenceChecker
# ---------------------------------------------------------------------------

class HeapEquivalenceChecker:
    """Compares two symbolic heaps (e.g. C vs Rust) for observational
    equivalence over a set of shared pointers.

    The checker asserts that for every shared pointer, the values
    stored at the corresponding addresses in both heaps are equal,
    taking into account type widths and struct layouts.

    Parameters
    ----------
    ctx : EncodingContext
        Shared encoding context.
    config : SemanticConfig
        Semantic configuration (pointer width, endianness, …).
    """

    def __init__(
        self,
        ctx: EncodingContext,
        config: SemanticConfig,
    ) -> None:
        self.ctx = ctx
        self.config = config
        self.ptr_width: int = config.pointer_size

    def encode_heap_equivalence(
        self,
        c_heap: SymbolicHeap,
        rust_heap: SymbolicHeap,
        shared_ptrs: List[Tuple[z3.BitVecRef, z3.BitVecRef, IRType]],
    ) -> z3.BoolRef:
        """Assert value-level equivalence for all shared pointers.

        Parameters
        ----------
        c_heap : SymbolicHeap
            The C-side heap.
        rust_heap : SymbolicHeap
            The Rust-side heap.
        shared_ptrs : list of (c_addr, rust_addr, type)
            Each tuple maps a C pointer to its Rust counterpart and the
            IR type of the pointed-to value.

        Returns
        -------
        z3.BoolRef
            Conjunction of all per-pointer equivalence assertions.
        """
        if not shared_ptrs:
            return z3.BoolVal(True)

        equiv_parts: List[z3.BoolRef] = []

        for c_addr, rust_addr, ir_type in shared_ptrs:
            eq_constraint = self._encode_value_equiv(
                c_heap, rust_heap, c_addr, rust_addr, ir_type
            )
            equiv_parts.append(eq_constraint)

        conjunction = z3.And(*equiv_parts) if len(equiv_parts) > 1 else equiv_parts[0]
        self.ctx.assert_hard(conjunction)
        return conjunction

    def encode_allocation_correspondence(
        self,
        c_heap: SymbolicHeap,
        rust_heap: SymbolicHeap,
        alloc_pairs: List[Tuple[Allocation, Allocation]],
    ) -> z3.BoolRef:
        """Assert that paired allocations have equal sizes and that
        their contents are byte-for-byte equal.

        This is a stronger equivalence than :meth:`encode_heap_equivalence`:
        it requires *all* bytes to match, not just typed values.

        Parameters
        ----------
        c_heap, rust_heap : SymbolicHeap
            The two heaps.
        alloc_pairs : list of (c_alloc, rust_alloc)
            Paired allocations.

        Returns
        -------
        z3.BoolRef
            Conjunction of size-equality and content-equality constraints.
        """
        if not alloc_pairs:
            return z3.BoolVal(True)

        parts: List[z3.BoolRef] = []

        for c_alloc, rust_alloc in alloc_pairs:
            # Sizes must match
            parts.append(c_alloc.size == rust_alloc.size)

            # Byte-level content equivalence via universally quantified
            # index within the allocation.
            idx = z3.BitVec(
                self.ctx.fresh("eq_idx"), self.ptr_width
            )
            c_byte = z3.Select(c_heap.mem, c_alloc.base + idx)
            r_byte = z3.Select(rust_heap.mem, rust_alloc.base + idx)

            in_bounds = z3.And(
                z3.UGE(idx, z3.BitVecVal(0, self.ptr_width)),
                z3.ULT(idx, c_alloc.size),
            )
            byte_eq = z3.ForAll(
                [idx],
                z3.Implies(in_bounds, c_byte == r_byte),
            )
            parts.append(byte_eq)

        result = z3.And(*parts) if len(parts) > 1 else parts[0]
        self.ctx.assert_hard(result)
        return result

    # -- per-type equivalence ----------------------------------------------

    def _encode_value_equiv(
        self,
        c_heap: SymbolicHeap,
        rust_heap: SymbolicHeap,
        c_addr: z3.BitVecRef,
        rust_addr: z3.BitVecRef,
        ir_type: IRType,
    ) -> z3.BoolRef:
        """Recursively encode value equivalence for *ir_type*."""
        if isinstance(ir_type, IntType):
            width_bytes = ir_type.size_bytes(self.config.pointer_size)
            c_val = c_heap.read_bytes(c_addr, width_bytes)
            r_val = rust_heap.read_bytes(rust_addr, width_bytes)
            return c_val == r_val

        if isinstance(ir_type, FloatType):
            width_bytes = ir_type.size_bytes(self.config.pointer_size)
            c_val = c_heap.read_bytes(c_addr, width_bytes)
            r_val = rust_heap.read_bytes(rust_addr, width_bytes)
            return c_val == r_val

        if isinstance(ir_type, PointerType):
            ptr_bytes = ir_type.size_bytes(self.config.pointer_size)
            c_val = c_heap.read_bytes(c_addr, ptr_bytes)
            r_val = rust_heap.read_bytes(rust_addr, ptr_bytes)
            return c_val == r_val

        if isinstance(ir_type, StructType):
            return self._encode_struct_equiv(
                c_heap, rust_heap, c_addr, rust_addr, ir_type
            )

        if isinstance(ir_type, ArrayType):
            return self._encode_array_equiv(
                c_heap, rust_heap, c_addr, rust_addr, ir_type
            )

        # Fallback: byte-level comparison of the full size
        width_bytes = ir_type.size_bytes(self.config.pointer_size)
        if width_bytes > 0:
            c_val = c_heap.read_bytes(c_addr, width_bytes)
            r_val = rust_heap.read_bytes(rust_addr, width_bytes)
            return c_val == r_val

        return z3.BoolVal(True)

    def _encode_struct_equiv(
        self,
        c_heap: SymbolicHeap,
        rust_heap: SymbolicHeap,
        c_addr: z3.BitVecRef,
        rust_addr: z3.BitVecRef,
        struct_ty: StructType,
    ) -> z3.BoolRef:
        """Field-by-field equivalence for a struct value.

        Only data-carrying bytes are compared; padding is ignored.
        """
        layout = StructLayout(self.config)
        offsets = layout.compute_offsets(struct_ty)

        parts: List[z3.BoolRef] = []
        for i, sf in enumerate(struct_ty.fields):
            off_bv = z3.BitVecVal(offsets[i], self.ptr_width)
            c_field = c_addr + off_bv
            r_field = rust_addr + off_bv
            parts.append(
                self._encode_value_equiv(
                    c_heap, rust_heap, c_field, r_field, sf.type
                )
            )

        if not parts:
            return z3.BoolVal(True)
        return z3.And(*parts) if len(parts) > 1 else parts[0]

    def _encode_array_equiv(
        self,
        c_heap: SymbolicHeap,
        rust_heap: SymbolicHeap,
        c_addr: z3.BitVecRef,
        rust_addr: z3.BitVecRef,
        array_ty: ArrayType,
    ) -> z3.BoolRef:
        """Element-by-element equivalence for a fixed-length array.

        For arrays with known small length the elements are unrolled.
        For larger arrays a universally quantified assertion is used.
        """
        elem_size = array_ty.element.size_bytes(self.config.pointer_size)
        length = array_ty.length

        _UNROLL_LIMIT = 32

        if 0 < length <= _UNROLL_LIMIT:
            parts: List[z3.BoolRef] = []
            for i in range(length):
                off = z3.BitVecVal(i * elem_size, self.ptr_width)
                parts.append(
                    self._encode_value_equiv(
                        c_heap, rust_heap,
                        c_addr + off, rust_addr + off,
                        array_ty.element,
                    )
                )
            return z3.And(*parts) if len(parts) > 1 else parts[0]

        # Quantified fallback for large/unknown-length arrays
        idx = z3.BitVec(self.ctx.fresh("arr_eq_idx"), self.ptr_width)
        off = idx * z3.BitVecVal(elem_size, self.ptr_width)
        c_elem_bytes = c_heap.read_bytes(c_addr + off, elem_size)
        r_elem_bytes = rust_heap.read_bytes(rust_addr + off, elem_size)
        in_bounds = z3.And(
            z3.UGE(idx, z3.BitVecVal(0, self.ptr_width)),
            z3.ULT(idx, z3.BitVecVal(length, self.ptr_width)),
        )
        return z3.ForAll(
            [idx],
            z3.Implies(in_bounds, c_elem_bytes == r_elem_bytes),
        )


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _align_up_int(value: int, alignment: int) -> int:
    """Round *value* up to the next multiple of *alignment*."""
    if alignment <= 1:
        return value
    return (value + alignment - 1) & ~(alignment - 1)


# ---------------------------------------------------------------------------
# Pointer pair encoding (base_id, offset)
# ---------------------------------------------------------------------------

class PointerPairEncoding:
    """Encodes pointers as (base_id, offset) pairs for provenance-aware
    pointer reasoning.

    Each pointer is represented as two bitvectors:
    - ``base_id``: identifies the allocation (0 = null)
    - ``offset``: byte offset within that allocation

    This separates provenance from address arithmetic, allowing
    the solver to reason about pointer identity and arithmetic
    independently.

    Parameters
    ----------
    ctx : EncodingContext
        Shared encoding context.
    ptr_width : int
        Pointer width in bits.
    id_width : int
        Width of the base_id field in bits (default 32).
    """

    def __init__(
        self,
        ctx: EncodingContext,
        ptr_width: int = 64,
        id_width: int = 32,
    ) -> None:
        self.ctx = ctx
        self.ptr_width = ptr_width
        self.id_width = id_width
        self._alloc_counter = 0

    def null_pointer(self) -> Tuple[z3.BitVecRef, z3.BitVecRef]:
        """Return the null pointer as ``(0, 0)``."""
        return (
            z3.BitVecVal(0, self.id_width),
            z3.BitVecVal(0, self.ptr_width),
        )

    def fresh_pointer(self, name: str) -> Tuple[z3.BitVecRef, z3.BitVecRef]:
        """Create a fresh symbolic pointer ``(base_id, offset)``."""
        base_id = self.ctx.declare_bv(f"{name}_base_id", self.id_width)
        offset = self.ctx.declare_bv(f"{name}_offset", self.ptr_width)
        return base_id, offset

    def allocate(self, name: str) -> Tuple[z3.BitVecRef, z3.BitVecRef]:
        """Allocate a new pointer with a unique non-zero base_id and zero
        offset.  The base_id is constrained to be distinct from all
        prior allocations.
        """
        self._alloc_counter += 1
        base_id = self.ctx.declare_bv(f"{name}_base_id", self.id_width)
        # Non-null
        self.ctx.assert_hard(base_id != z3.BitVecVal(0, self.id_width))
        offset = z3.BitVecVal(0, self.ptr_width)
        return base_id, offset

    def pointer_add(
        self,
        ptr: Tuple[z3.BitVecRef, z3.BitVecRef],
        n: z3.BitVecRef,
        elem_size: int,
    ) -> Tuple[z3.BitVecRef, z3.BitVecRef]:
        """Pointer arithmetic: ``p + n`` → ``(base_id, offset + n * sizeof(T))``."""
        base_id, offset = ptr
        stride = z3.BitVecVal(elem_size, self.ptr_width)
        if n.size() != self.ptr_width:
            if n.size() < self.ptr_width:
                n = z3.SignExt(self.ptr_width - n.size(), n)
            else:
                n = z3.Extract(self.ptr_width - 1, 0, n)
        new_offset = offset + n * stride
        return base_id, new_offset

    def pointer_eq(
        self,
        a: Tuple[z3.BitVecRef, z3.BitVecRef],
        b: Tuple[z3.BitVecRef, z3.BitVecRef],
    ) -> z3.BoolRef:
        """Pointer equality: same base_id and same offset."""
        return z3.And(a[0] == b[0], a[1] == b[1])

    def pointer_ne(
        self,
        a: Tuple[z3.BitVecRef, z3.BitVecRef],
        b: Tuple[z3.BitVecRef, z3.BitVecRef],
    ) -> z3.BoolRef:
        """Pointer inequality."""
        return z3.Or(a[0] != b[0], a[1] != b[1])

    def pointer_lt(
        self,
        a: Tuple[z3.BitVecRef, z3.BitVecRef],
        b: Tuple[z3.BitVecRef, z3.BitVecRef],
    ) -> z3.BoolRef:
        """Pointer less-than (only meaningful within same allocation)."""
        same_base = a[0] == b[0]
        return z3.And(same_base, z3.ULT(a[1], b[1]))

    def is_null(
        self,
        ptr: Tuple[z3.BitVecRef, z3.BitVecRef],
    ) -> z3.BoolRef:
        """Check if pointer is null ``(0, 0)``."""
        return z3.And(
            ptr[0] == z3.BitVecVal(0, self.id_width),
            ptr[1] == z3.BitVecVal(0, self.ptr_width),
        )

    def to_flat(
        self,
        ptr: Tuple[z3.BitVecRef, z3.BitVecRef],
    ) -> z3.BitVecRef:
        """Convert a pointer pair to a flat address (for interop)."""
        # Widen base_id to ptr_width and combine
        base = z3.ZeroExt(self.ptr_width - self.id_width, ptr[0]) \
            if self.id_width < self.ptr_width else ptr[0]
        return base + ptr[1]
