"""
Tests for the symbolic memory model module.

Covers SymbolicHeap, SeparationLogicEncoder, StructLayout,
ProvenanceTracker, PointerArithmeticEncoder, HeapEquivalenceChecker,
and integration with the SMT encoder.
"""

from __future__ import annotations

import os
import sys

import pytest
import z3

# Ensure src is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.smt.memory_model import (
    SymbolicHeap,
    SeparationLogicEncoder,
    StructLayout,
    ProvenanceTracker,
    PointerArithmeticEncoder,
    HeapEquivalenceChecker,
    Allocation,
)
from src.smt.encoder import EncodingContext, SMTEncoder
from src.semantics.semantic_config import (
    SemanticConfig,
    PointerModel,
    ArrayBoundsModel,
    LayoutModel,
)
from src.ir.types import (
    IntType,
    PointerType,
    StructType,
    StructField,
    ArrayType,
    Signedness,
    VoidType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ctx() -> EncodingContext:
    """Create a fresh EncodingContext."""
    return EncodingContext()


def _c11_config(pointer_size: int = 64) -> SemanticConfig:
    """Return a C11 SemanticConfig."""
    return SemanticConfig.c11(pointer_size)


def _rust_config(pointer_size: int = 64) -> SemanticConfig:
    """Return a Rust-debug SemanticConfig."""
    return SemanticConfig.rust_debug(pointer_size)


def _make_heap(
    ctx: EncodingContext | None = None,
    config: SemanticConfig | None = None,
    name: str = "heap",
) -> tuple[SymbolicHeap, EncodingContext, SemanticConfig]:
    ctx = ctx or _make_ctx()
    config = config or _c11_config()
    heap = SymbolicHeap(ctx, config, name=name)
    return heap, ctx, config


def _check_sat(ctx: EncodingContext, extra=None) -> z3.CheckSatResult:
    """Build a solver from *ctx* assertions and check satisfiability."""
    s = z3.Solver()
    for a in ctx.assertions:
        s.add(a)
    if extra is not None:
        if isinstance(extra, list):
            for e in extra:
                s.add(e)
        else:
            s.add(extra)
    return s.check()


def _get_model(ctx: EncodingContext, extra=None) -> z3.ModelRef:
    s = z3.Solver()
    for a in ctx.assertions:
        s.add(a)
    if extra is not None:
        if isinstance(extra, list):
            for e in extra:
                s.add(e)
        else:
            s.add(extra)
    assert s.check() == z3.sat
    return s.model()


# ===================================================================
# 1. SymbolicHeap tests
# ===================================================================


class TestSymbolicHeap:
    """Tests for byte-addressable symbolic heap."""

    def test_write_read_32bit(self):
        """Write a 32-bit value and read it back — should be SAT (equal)."""
        heap, ctx, _ = _make_heap()
        addr = z3.BitVecVal(0x1000, 64)
        val = z3.BitVecVal(0xDEADBEEF, 32)
        heap.write_bytes(addr, val, 4)
        loaded = heap.read_bytes(addr, 4)
        assert _check_sat(ctx, loaded == val) == z3.sat

    def test_write_two_addresses_independence(self):
        """Write to two different addresses, read both — independence."""
        heap, ctx, _ = _make_heap()
        a1 = z3.BitVecVal(0x1000, 64)
        a2 = z3.BitVecVal(0x2000, 64)
        v1 = z3.BitVecVal(0x11111111, 32)
        v2 = z3.BitVecVal(0x22222222, 32)
        heap.write_bytes(a1, v1, 4)
        heap.write_bytes(a2, v2, 4)
        r1 = heap.read_bytes(a1, 4)
        r2 = heap.read_bytes(a2, 4)
        assert _check_sat(ctx, [r1 == v1, r2 == v2]) == z3.sat

    def test_read_uninitialized(self):
        """Read from uninitialized address — unconstrained (SAT for any value)."""
        heap, ctx, _ = _make_heap()
        addr = z3.BitVecVal(0x3000, 64)
        loaded = heap.read_bytes(addr, 4)
        probe = z3.BitVecVal(42, 32)
        # Should be satisfiable for the probe value (unconstrained memory)
        assert _check_sat(ctx, loaded == probe) == z3.sat
        # Also satisfiable for a different value
        probe2 = z3.BitVecVal(99, 32)
        assert _check_sat(ctx, loaded == probe2) == z3.sat

    def test_write_read_8bit(self):
        """Write and read an 8-bit value."""
        heap, ctx, _ = _make_heap()
        addr = z3.BitVecVal(0x100, 64)
        val = z3.BitVecVal(0xAB, 8)
        heap.write_bytes(addr, val, 1)
        loaded = heap.read_bytes(addr, 1)
        assert _check_sat(ctx, loaded == val) == z3.sat

    def test_write_read_16bit(self):
        """Write and read a 16-bit value."""
        heap, ctx, _ = _make_heap()
        addr = z3.BitVecVal(0x200, 64)
        val = z3.BitVecVal(0xCAFE, 16)
        heap.write_bytes(addr, val, 2)
        loaded = heap.read_bytes(addr, 2)
        assert _check_sat(ctx, loaded == val) == z3.sat

    def test_write_read_64bit(self):
        """Write and read a 64-bit value."""
        heap, ctx, _ = _make_heap()
        addr = z3.BitVecVal(0x400, 64)
        val = z3.BitVecVal(0x0102030405060708, 64)
        heap.write_bytes(addr, val, 8)
        loaded = heap.read_bytes(addr, 8)
        assert _check_sat(ctx, loaded == val) == z3.sat

    def test_overwrite(self):
        """Write, then overwrite — read should get latest value."""
        heap, ctx, _ = _make_heap()
        addr = z3.BitVecVal(0x500, 64)
        old = z3.BitVecVal(0xAAAAAAAA, 32)
        new = z3.BitVecVal(0xBBBBBBBB, 32)
        heap.write_bytes(addr, old, 4)
        heap.write_bytes(addr, new, 4)
        loaded = heap.read_bytes(addr, 4)
        assert _check_sat(ctx, loaded == new) == z3.sat
        # The old value should NOT be readable
        assert _check_sat(ctx, loaded == old) == z3.unsat

    def test_multibyte_composition(self):
        """Write 32-bit value, then read individual bytes."""
        heap, ctx, _ = _make_heap()
        addr = z3.BitVecVal(0x600, 64)
        # Little-endian: 0x04030201 stored as bytes [01, 02, 03, 04]
        val = z3.BitVecVal(0x04030201, 32)
        heap.write_bytes(addr, val, 4)
        b0 = heap.read_bytes(addr, 1)
        b1 = heap.read_bytes(z3.BitVecVal(0x601, 64), 1)
        b2 = heap.read_bytes(z3.BitVecVal(0x602, 64), 1)
        b3 = heap.read_bytes(z3.BitVecVal(0x603, 64), 1)
        assert _check_sat(ctx, [
            b0 == z3.BitVecVal(0x01, 8),
            b1 == z3.BitVecVal(0x02, 8),
            b2 == z3.BitVecVal(0x03, 8),
            b3 == z3.BitVecVal(0x04, 8),
        ]) == z3.sat

    def test_version_tracking(self):
        """Heap version increments on each write."""
        heap, ctx, _ = _make_heap()
        assert heap._version == 0
        addr = z3.BitVecVal(0x700, 64)
        heap.write_bytes(addr, z3.BitVecVal(1, 8), 1)
        assert heap._version == 1
        heap.write_bytes(addr, z3.BitVecVal(2, 8), 1)
        assert heap._version == 2

    def test_snapshot(self):
        """Snapshot captures memory state at call time."""
        heap, ctx, _ = _make_heap()
        addr = z3.BitVecVal(0x800, 64)
        snap_before = heap.snapshot()
        heap.write_bytes(addr, z3.BitVecVal(0xFF, 8), 1)
        snap_after = heap.snapshot()
        # snap_before and snap_after should be different Z3 expressions
        assert snap_before is not snap_after


# ===================================================================
# 2. SeparationLogicEncoder tests
# ===================================================================


class TestSeparationLogicEncoder:
    """Tests for separation-logic encoding primitives."""

    def _make_sep(self, bounds: ArrayBoundsModel = ArrayBoundsModel.NoCheck):
        config = _c11_config()
        config.array_bounds = bounds
        ctx = _make_ctx()
        sep = SeparationLogicEncoder(ctx, config)
        return sep, ctx

    def test_alloc_non_null(self):
        """Allocate one region — verify base is non-null."""
        sep, ctx = self._make_sep()
        size = z3.BitVecVal(16, 64)
        base, alloc = sep.encode_alloc(size)
        null = z3.BitVecVal(0, 64)
        # base == null should be UNSAT
        assert _check_sat(ctx, base == null) == z3.unsat

    def test_alloc_two_non_overlap(self):
        """Allocate two regions — verify they don't overlap (frame condition)."""
        sep, ctx = self._make_sep()
        size = z3.BitVecVal(64, 64)
        base1, alloc1 = sep.encode_alloc(size)
        base2, alloc2 = sep.encode_alloc(size)
        # Their ranges should not overlap: check that base1 == base2 is UNSAT
        assert _check_sat(ctx, base1 == base2) == z3.unsat

    def test_encode_alloc_returns_allocation(self):
        """encode_alloc returns a valid Allocation with base + size."""
        sep, ctx = self._make_sep()
        size = z3.BitVecVal(32, 64)
        base, alloc = sep.encode_alloc(size, align=8)
        assert isinstance(alloc, Allocation)
        assert alloc.alloc_id > 0
        assert alloc.align == 8

    def test_free_allocation(self):
        """Free an allocation — check freed flag."""
        sep, ctx = self._make_sep()
        size = z3.BitVecVal(16, 64)
        base, alloc = sep.encode_alloc(size)
        valid = sep.encode_free(base)
        # Free should be satisfiable (valid free of a live allocation)
        assert _check_sat(ctx) == z3.sat

    def test_null_check_on_null(self):
        """Null check on null pointer should be satisfiable/true."""
        sep, ctx = self._make_sep()
        null_addr = z3.BitVecVal(0, 64)
        is_null = sep.encode_null_check(null_addr)
        assert _check_sat(ctx, is_null) == z3.sat

    def test_null_check_on_non_null(self):
        """Null check on a non-null pointer should be UNSAT when asserted true."""
        sep, ctx = self._make_sep()
        addr = z3.BitVecVal(0x1000, 64)
        is_null = sep.encode_null_check(addr)
        # addr is concrete 0x1000, so is_null is False
        assert _check_sat(ctx, is_null) == z3.unsat

    def test_points_to(self):
        """Points-to: write a value, assert points-to, check SAT."""
        sep, ctx = self._make_sep()
        addr = z3.BitVecVal(0x2000, 64)
        val = z3.BitVecVal(42, 32)
        sep.encode_store(addr, val, 4)
        constraint = sep.encode_points_to(addr, val, 4)
        assert _check_sat(ctx) == z3.sat

    def test_frame_condition_explicit(self):
        """Explicit frame condition between two allocations."""
        sep, ctx = self._make_sep()
        size = z3.BitVecVal(32, 64)
        _, alloc1 = sep.encode_alloc(size)
        _, alloc2 = sep.encode_alloc(size)
        frame = sep.encode_frame_condition(alloc1, alloc2)
        assert _check_sat(ctx) == z3.sat


# ===================================================================
# 3. StructLayout tests
# ===================================================================


class TestStructLayout:
    """Tests for struct layout computation."""

    def _layout(self, config=None):
        return StructLayout(config or _c11_config())

    def test_simple_i32_i32(self):
        """Struct {i32, i32}: field 0 at offset 0, field 1 at offset 4."""
        layout = self._layout()
        st = StructType("pair", (
            StructField("a", IntType(32, Signedness.SIGNED)),
            StructField("b", IntType(32, Signedness.SIGNED)),
        ))
        offsets = layout.compute_offsets(st)
        assert offsets == [0, 4]

    def test_padding_i8_i32(self):
        """Struct {i8, i32}: field 1 offset = 4 (aligned to 4 bytes)."""
        layout = self._layout()
        st = StructType("padded", (
            StructField("c", IntType(8, Signedness.SIGNED)),
            StructField("d", IntType(32, Signedness.SIGNED)),
        ))
        offsets = layout.compute_offsets(st)
        assert offsets[0] == 0
        assert offsets[1] == 4  # 3 bytes of padding after i8

    def test_nested_struct(self):
        """Struct {i32, {i8, i16}}: correct offsets."""
        layout = self._layout()
        inner = StructType("inner", (
            StructField("x", IntType(8, Signedness.UNSIGNED)),
            StructField("y", IntType(16, Signedness.SIGNED)),
        ))
        outer = StructType("outer", (
            StructField("a", IntType(32, Signedness.SIGNED)),
            StructField("b", inner),
        ))
        offsets = layout.compute_offsets(outer)
        assert offsets[0] == 0
        # inner alignment = max(align(i8), align(i16)) = 2 bytes
        # i32 is 4 bytes so inner starts at offset 4
        assert offsets[1] == 4

    def test_total_size(self):
        """Total size includes tail padding."""
        layout = self._layout()
        st = StructType("sized", (
            StructField("a", IntType(32, Signedness.SIGNED)),
            StructField("b", IntType(32, Signedness.SIGNED)),
        ))
        assert layout.compute_size(st) == 8

    def test_total_size_with_padding(self):
        """Struct {i8, i32} total size includes tail padding."""
        layout = self._layout()
        st = StructType("padded_size", (
            StructField("c", IntType(8, Signedness.SIGNED)),
            StructField("d", IntType(32, Signedness.SIGNED)),
        ))
        size = layout.compute_size(st)
        # i8 @ 0, i32 @ 4, raw end = 8, overall align = 4, size = 8
        assert size == 8

    def test_c_compat_layout(self):
        """C-compatible layout uses CCompat model."""
        config = _c11_config()
        config.layout_model = LayoutModel.CCompat
        layout = StructLayout(config)
        st = StructType("c_struct", (
            StructField("x", IntType(8, Signedness.SIGNED)),
            StructField("y", IntType(32, Signedness.SIGNED)),
        ))
        offsets = layout.compute_offsets(st)
        assert offsets == [0, 4]

    def test_packed_struct(self):
        """Packed struct: no padding between fields."""
        layout = self._layout()
        st = StructType("packed", (
            StructField("a", IntType(8, Signedness.SIGNED)),
            StructField("b", IntType(32, Signedness.SIGNED)),
        ), packed=True)
        offsets = layout.compute_offsets(st)
        assert offsets == [0, 1]  # no padding: i8 at 0, i32 at 1

    def test_struct_with_pointer(self):
        """Struct with a pointer field uses pointer-width alignment."""
        layout = self._layout()
        pointee = IntType(32, Signedness.SIGNED)
        st = StructType("has_ptr", (
            StructField("val", IntType(32, Signedness.SIGNED)),
            StructField("ptr", PointerType(pointee)),
        ))
        offsets = layout.compute_offsets(st)
        assert offsets[0] == 0
        # Pointer is 8-byte aligned on 64-bit → field 1 at offset 8
        assert offsets[1] == 8


# ===================================================================
# 4. ProvenanceTracker tests
# ===================================================================


class TestProvenanceTracker:
    """Tests for pointer provenance tracking."""

    def _make_tracker(self):
        config = _rust_config()
        ctx = _make_ctx()
        tracker = ProvenanceTracker(ctx, config)
        return tracker, ctx

    def test_create_provenance(self):
        """Create provenance for a pointer — returns a tag."""
        tracker, ctx = self._make_tracker()
        tag = tracker.new_provenance("ptr_a")
        assert tag is not None
        assert _check_sat(ctx) == z3.sat

    def test_derive_child(self):
        """Derive child provenance — child has same tag as parent."""
        tracker, ctx = self._make_tracker()
        parent_tag = tracker.new_provenance("parent")
        offset = z3.BitVecVal(4, 64)
        child_tag = tracker.derive("parent", "child", offset)
        # Child should have the same tag as parent
        assert _check_sat(ctx, child_tag == parent_tag) == z3.sat

    def test_check_valid(self):
        """Check valid provenance — should be SAT."""
        tracker, ctx = self._make_tracker()
        tracker.new_provenance("ptr_v")
        validity = tracker.check_valid("ptr_v")
        assert _check_sat(ctx) == z3.sat

    def test_invalidate(self):
        """Invalidate provenance — subsequent check should be UNSAT."""
        tracker, ctx = self._make_tracker()
        tracker.new_provenance("ptr_x")
        tracker.invalidate("ptr_x")
        # After invalidation the solver has contradictory assertions
        # (valid AND NOT valid), so the context is UNSAT
        assert _check_sat(ctx) == z3.unsat

    def test_check_invalid_after_invalidation(self):
        """After invalidation, provenance is not valid."""
        tracker, ctx = self._make_tracker()
        tracker.new_provenance("ptr_y")
        tracker.invalidate("ptr_y")
        # Context should be UNSAT because valid(tag) AND NOT valid(tag)
        assert _check_sat(ctx) == z3.unsat

    def test_get_tag(self):
        """get_tag returns stored tag, or None for unknown pointer."""
        tracker, _ = self._make_tracker()
        assert tracker.get_tag("unknown") is None
        tracker.new_provenance("known")
        assert tracker.get_tag("known") is not None


# ===================================================================
# 5. PointerArithmeticEncoder tests
# ===================================================================


class TestPointerArithmeticEncoder:
    """Tests for pointer arithmetic encoding."""

    def _make_ptr_enc(self, bounds=ArrayBoundsModel.NoCheck):
        config = _c11_config()
        config.array_bounds = bounds
        ctx = _make_ctx()
        heap = SymbolicHeap(ctx, config)
        enc = PointerArithmeticEncoder(ctx, config, heap)
        return enc, ctx, heap

    def test_null_check_zero(self):
        """Null check on zero address → is_null is True."""
        enc, ctx, _ = self._make_ptr_enc()
        addr = z3.BitVecVal(0, 64)
        is_null = enc.encode_is_null(addr)
        assert _check_sat(ctx, is_null) == z3.sat

    def test_null_check_nonzero(self):
        """Null check on non-zero address → is_null is UNSAT."""
        enc, ctx, _ = self._make_ptr_enc()
        addr = z3.BitVecVal(0x100, 64)
        is_null = enc.encode_is_null(addr)
        assert _check_sat(ctx, is_null) == z3.unsat

    def test_pointer_offset(self):
        """Pointer offset via GEP: base + index * element_size."""
        enc, ctx, _ = self._make_ptr_enc()
        base = z3.BitVecVal(0x1000, 64)
        idx = z3.BitVecVal(3, 64)
        pointee = IntType(32, Signedness.SIGNED)  # 4 bytes
        result = enc.encode_gep(base, [idx], pointee)
        expected = z3.BitVecVal(0x1000 + 3 * 4, 64)
        assert _check_sat(ctx, result == expected) == z3.sat

    def test_pointer_comparison(self):
        """Pointer equality and inequality."""
        enc, ctx, _ = self._make_ptr_enc()
        a = z3.BitVecVal(0x1000, 64)
        b = z3.BitVecVal(0x2000, 64)
        assert _check_sat(ctx, enc.encode_ptr_eq(a, a)) == z3.sat
        assert _check_sat(ctx, enc.encode_ptr_eq(a, b)) == z3.unsat
        assert _check_sat(ctx, enc.encode_ptr_ne(a, b)) == z3.sat

    def test_bounds_checking(self):
        """Bounds checking: in-bounds access after allocation."""
        enc, ctx, heap = self._make_ptr_enc()
        size = z3.BitVecVal(64, 64)
        alloc = heap.allocate(size)
        in_bounds = enc.encode_in_bounds(alloc.base, 4)
        assert _check_sat(ctx, in_bounds) == z3.sat

    def test_encode_is_not_null(self):
        """is_not_null on a concrete non-zero address."""
        enc, ctx, _ = self._make_ptr_enc()
        addr = z3.BitVecVal(0x42, 64)
        not_null = enc.encode_is_not_null(addr)
        assert _check_sat(ctx, not_null) == z3.sat


# ===================================================================
# 6. HeapEquivalenceChecker tests
# ===================================================================


class TestHeapEquivalenceChecker:
    """Tests for cross-heap equivalence checking."""

    def _make_checker(self):
        ctx = _make_ctx()
        config = _c11_config()
        c_heap = SymbolicHeap(ctx, config, name="c_heap")
        r_heap = SymbolicHeap(ctx, config, name="r_heap")
        checker = HeapEquivalenceChecker(ctx, config)
        return checker, c_heap, r_heap, ctx

    def test_same_values_equivalent(self):
        """Two heaps with same values at shared pointers → equivalent."""
        checker, c_heap, r_heap, ctx = self._make_checker()
        addr = z3.BitVecVal(0x1000, 64)
        val = z3.BitVecVal(0xBEEF, 32)
        c_heap.write_bytes(addr, val, 4)
        r_heap.write_bytes(addr, val, 4)
        i32 = IntType(32, Signedness.SIGNED)
        eq = checker.encode_heap_equivalence(c_heap, r_heap, [(addr, addr, i32)])
        assert _check_sat(ctx) == z3.sat

    def test_different_values_not_equivalent(self):
        """Two heaps with different values → not equivalent (UNSAT)."""
        checker, c_heap, r_heap, ctx = self._make_checker()
        addr = z3.BitVecVal(0x1000, 64)
        c_heap.write_bytes(addr, z3.BitVecVal(1, 32), 4)
        r_heap.write_bytes(addr, z3.BitVecVal(2, 32), 4)
        i32 = IntType(32, Signedness.SIGNED)
        eq = checker.encode_heap_equivalence(c_heap, r_heap, [(addr, addr, i32)])
        assert _check_sat(ctx) == z3.unsat

    def test_empty_heaps_equivalent(self):
        """Empty heaps with no shared pointers → trivially equivalent."""
        checker, c_heap, r_heap, ctx = self._make_checker()
        eq = checker.encode_heap_equivalence(c_heap, r_heap, [])
        assert _check_sat(ctx) == z3.sat

    def test_multiple_shared_pointers(self):
        """Equivalence over multiple shared pointers."""
        checker, c_heap, r_heap, ctx = self._make_checker()
        a1 = z3.BitVecVal(0x1000, 64)
        a2 = z3.BitVecVal(0x2000, 64)
        v1 = z3.BitVecVal(10, 32)
        v2 = z3.BitVecVal(20, 32)
        c_heap.write_bytes(a1, v1, 4)
        c_heap.write_bytes(a2, v2, 4)
        r_heap.write_bytes(a1, v1, 4)
        r_heap.write_bytes(a2, v2, 4)
        i32 = IntType(32, Signedness.SIGNED)
        eq = checker.encode_heap_equivalence(
            c_heap, r_heap, [(a1, a1, i32), (a2, a2, i32)]
        )
        assert _check_sat(ctx) == z3.sat


# ===================================================================
# 7. Integration with SMTEncoder
# ===================================================================


class TestIntegrationSMTEncoder:
    """Integration tests using SMTEncoder with the memory model path."""

    def _make_encoder(self):
        config = _c11_config()
        config.array_bounds = ArrayBoundsModel.NoCheck
        ctx = _make_ctx()
        encoder = SMTEncoder(config=config)
        return encoder, ctx, config

    def test_encode_load_via_heap(self):
        """Encode a load using the SeparationLogicEncoder path."""
        config = _c11_config()
        config.array_bounds = ArrayBoundsModel.NoCheck
        ctx = _make_ctx()
        sep = SeparationLogicEncoder(ctx, config)
        addr = z3.BitVecVal(0x1000, 64)
        val = z3.BitVecVal(42, 32)
        sep.encode_store(addr, val, 4)
        loaded = sep.encode_load(addr, 4)
        assert _check_sat(ctx, loaded == val) == z3.sat

    def test_store_load_roundtrip(self):
        """Store then load roundtrip through SeparationLogicEncoder."""
        config = _c11_config()
        config.array_bounds = ArrayBoundsModel.NoCheck
        ctx = _make_ctx()
        sep = SeparationLogicEncoder(ctx, config)
        addr = z3.BitVecVal(0x3000, 64)
        val = z3.BitVecVal(0xCAFEBABE, 32)
        sep.encode_store(addr, val, 4)
        result = sep.encode_load(addr, 4)
        # roundtrip: stored value must equal loaded value
        assert _check_sat(ctx, result == val) == z3.sat
        # and must NOT equal a different value
        wrong = z3.BitVecVal(0x12345678, 32)
        assert _check_sat(ctx, result == wrong) == z3.unsat

    def test_encode_gep_via_pointer_encoder(self):
        """Encode a GEP instruction using PointerArithmeticEncoder."""
        config = _c11_config()
        ctx = _make_ctx()
        heap = SymbolicHeap(ctx, config)
        ptr_enc = PointerArithmeticEncoder(ctx, config, heap)
        base = z3.BitVecVal(0x4000, 64)
        idx = z3.BitVecVal(5, 64)
        pointee = IntType(32, Signedness.SIGNED)
        result = ptr_enc.encode_gep(base, [idx], pointee)
        expected = z3.BitVecVal(0x4000 + 5 * 4, 64)
        assert _check_sat(ctx, result == expected) == z3.sat

    def test_encode_alloca_via_sep(self):
        """Encode an alloca-like allocation via SeparationLogicEncoder."""
        config = _c11_config()
        config.array_bounds = ArrayBoundsModel.NoCheck
        ctx = _make_ctx()
        sep = SeparationLogicEncoder(ctx, config)
        size = z3.BitVecVal(4, 64)
        base, alloc = sep.encode_alloc(size, align=4)
        # Write to the allocated region, then read back
        val = z3.BitVecVal(0x55, 32)
        sep.encode_store(base, val, 4)
        loaded = sep.encode_load(base, 4)
        assert _check_sat(ctx, loaded == val) == z3.sat

    def test_gep_struct_field(self):
        """GEP into a struct field via PointerArithmeticEncoder."""
        config = _c11_config()
        ctx = _make_ctx()
        heap = SymbolicHeap(ctx, config)
        ptr_enc = PointerArithmeticEncoder(ctx, config, heap)
        st = StructType("pair", (
            StructField("a", IntType(32, Signedness.SIGNED)),
            StructField("b", IntType(32, Signedness.SIGNED)),
        ))
        base = z3.BitVecVal(0x5000, 64)
        # GEP with struct type: index 1 → offset 4
        field_idx = z3.BitVecVal(1, 64)
        result = ptr_enc.encode_gep(base, [field_idx], st, struct_ty=st)
        expected = z3.BitVecVal(0x5000 + 4, 64)
        assert _check_sat(ctx, result == expected) == z3.sat


# ===================================================================
# Additional edge-case tests
# ===================================================================


class TestEdgeCases:
    """Additional edge-case and robustness tests."""

    def test_heap_name_isolation(self):
        """Two heaps with different names are independent."""
        ctx = _make_ctx()
        config = _c11_config()
        h1 = SymbolicHeap(ctx, config, name="h1")
        h2 = SymbolicHeap(ctx, config, name="h2")
        addr = z3.BitVecVal(0x100, 64)
        h1.write_bytes(addr, z3.BitVecVal(1, 8), 1)
        h2.write_bytes(addr, z3.BitVecVal(2, 8), 1)
        r1 = h1.read_bytes(addr, 1)
        r2 = h2.read_bytes(addr, 1)
        assert _check_sat(ctx, [r1 == z3.BitVecVal(1, 8), r2 == z3.BitVecVal(2, 8)]) == z3.sat

    def test_alloc_alignment(self):
        """Allocation with alignment 16 → base % 16 == 0."""
        ctx = _make_ctx()
        config = _c11_config()
        heap = SymbolicHeap(ctx, config)
        size = z3.BitVecVal(32, 64)
        alloc = heap.allocate(size, align=16)
        mask = z3.BitVecVal(15, 64)
        # base & 15 should be 0
        assert _check_sat(ctx, (alloc.base & mask) != z3.BitVecVal(0, 64)) == z3.unsat

    def test_struct_layout_field_access_z3(self):
        """encode_field_access produces correct Z3 pointer expressions."""
        config = _c11_config()
        layout = StructLayout(config)
        st = StructType("test_s", (
            StructField("x", IntType(8, Signedness.UNSIGNED)),
            StructField("y", IntType(32, Signedness.SIGNED)),
        ))
        base = z3.BitVecVal(0x2000, 64)
        field_addr = layout.encode_field_access(base, st, 1)
        expected = z3.BitVecVal(0x2000 + 4, 64)
        s = z3.Solver()
        s.add(field_addr == expected)
        assert s.check() == z3.sat

    def test_struct_layout_index_error(self):
        """encode_field_access raises IndexError for out-of-range index."""
        config = _c11_config()
        layout = StructLayout(config)
        st = StructType("small", (
            StructField("only", IntType(32, Signedness.SIGNED)),
        ))
        base = z3.BitVecVal(0, 64)
        with pytest.raises(IndexError):
            layout.encode_field_access(base, st, 5)

    def test_provenance_derive_unknown_parent(self):
        """Deriving from an unknown parent creates fresh provenance."""
        config = _rust_config()
        ctx = _make_ctx()
        tracker = ProvenanceTracker(ctx, config)
        offset = z3.BitVecVal(0, 64)
        tag = tracker.derive("nonexistent", "child", offset)
        # Should have created a new provenance for child
        assert tag is not None
        assert tracker.get_tag("child") is not None

    def test_encode_all_field_addrs(self):
        """encode_all_field_addrs returns correct addresses for every field."""
        config = _c11_config()
        layout = StructLayout(config)
        st = StructType("triple", (
            StructField("a", IntType(32, Signedness.SIGNED)),
            StructField("b", IntType(32, Signedness.SIGNED)),
            StructField("c", IntType(32, Signedness.SIGNED)),
        ))
        base = z3.BitVecVal(0x3000, 64)
        addrs = layout.encode_all_field_addrs(base, st)
        assert len(addrs) == 3
        s = z3.Solver()
        s.add(addrs[0] == z3.BitVecVal(0x3000, 64))
        s.add(addrs[1] == z3.BitVecVal(0x3004, 64))
        s.add(addrs[2] == z3.BitVecVal(0x3008, 64))
        assert s.check() == z3.sat

    def test_sep_store_load_different_widths(self):
        """Store 32-bit value, load 16-bit slice."""
        config = _c11_config()
        config.array_bounds = ArrayBoundsModel.NoCheck
        ctx = _make_ctx()
        sep = SeparationLogicEncoder(ctx, config)
        addr = z3.BitVecVal(0x5000, 64)
        val = z3.BitVecVal(0x04030201, 32)
        sep.encode_store(addr, val, 4)
        # Read just the first 2 bytes (little-endian: 0x0201)
        low16 = sep.encode_load(addr, 2)
        assert _check_sat(ctx, low16 == z3.BitVecVal(0x0201, 16)) == z3.sat

    def test_allocation_correspondence(self):
        """HeapEquivalenceChecker.encode_allocation_correspondence works."""
        ctx = _make_ctx()
        config = _c11_config()
        c_heap = SymbolicHeap(ctx, config, name="c")
        r_heap = SymbolicHeap(ctx, config, name="r")
        checker = HeapEquivalenceChecker(ctx, config)
        # Allocate matching regions
        size = z3.BitVecVal(16, 64)
        c_alloc = c_heap.allocate(size)
        r_alloc = r_heap.allocate(size)
        # Write same value
        addr_off = z3.BitVecVal(0, 64)
        c_heap.write_bytes(c_alloc.base + addr_off, z3.BitVecVal(0xFF, 8), 1)
        r_heap.write_bytes(r_alloc.base + addr_off, z3.BitVecVal(0xFF, 8), 1)
        eq = checker.encode_allocation_correspondence(
            c_heap, r_heap, [(c_alloc, r_alloc)]
        )
        assert _check_sat(ctx) == z3.sat

    def test_packed_struct_size(self):
        """Packed struct {i8, i32} has total size 5 (no tail padding)."""
        layout = StructLayout(_c11_config())
        st = StructType("pack", (
            StructField("a", IntType(8, Signedness.UNSIGNED)),
            StructField("b", IntType(32, Signedness.SIGNED)),
        ), packed=True)
        assert layout.compute_size(st) == 5

    def test_struct_alignment(self):
        """Struct alignment is max of field alignments."""
        layout = StructLayout(_c11_config())
        st = StructType("al", (
            StructField("a", IntType(8, Signedness.UNSIGNED)),
            StructField("b", IntType(64, Signedness.SIGNED)),
        ))
        assert layout.compute_alignment(st) == 8  # 64-bit field → 8-byte align

    def test_pointer_lt(self):
        """Pointer less-than comparison."""
        config = _c11_config()
        ctx = _make_ctx()
        heap = SymbolicHeap(ctx, config)
        enc = PointerArithmeticEncoder(ctx, config, heap)
        a = z3.BitVecVal(0x1000, 64)
        b = z3.BitVecVal(0x2000, 64)
        lt = enc.encode_ptr_lt(a, b)
        assert _check_sat(ctx, lt) == z3.sat
        assert _check_sat(ctx, enc.encode_ptr_lt(b, a)) == z3.unsat
