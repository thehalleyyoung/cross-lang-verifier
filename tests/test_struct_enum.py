"""
Tests for struct/enum SMT encoding, expanded alignment patterns,
and new benchmark categories.
"""
import pytest
import z3

from src.ir.types import (
    IntType, FloatType, VoidType, StructType, StructField, EnumType,
    UnionType, ArrayType, Signedness, FloatKind, type_from_dict,
    check_compatibility, TypeCompatibility, collect_types,
    contains_pointer, _align_up,
)
from src.ir.instructions import (
    Value, Constant, BinaryOp, CompareOp, ReturnInst,
    ExtractValueInst, InsertValueInst, SwitchInst,
    BinOpKind, CmpPredicate, InstructionMetadata,
)
from src.ir.basic_block import BasicBlock
from src.ir.function import Function
from src.smt.encoder import SMTEncoder, EncodingContext


# ---------------------------------------------------------------------------
# EnumType IR tests
# ---------------------------------------------------------------------------

class TestEnumType:
    """Tests for the new EnumType."""

    def test_c_like_enum(self):
        """C-like enum with unit variants."""
        e = EnumType("Color", (
            ("Red", VoidType()),
            ("Green", VoidType()),
            ("Blue", VoidType()),
        ))
        assert e.num_variants == 3
        assert e.is_c_like
        assert e.variant_index("Red") == 0
        assert e.variant_index("Blue") == 2
        assert e._effective_tag_width() == 8

    def test_data_enum(self):
        """Rust-style enum with data payloads."""
        e = EnumType("Shape", (
            ("Circle", IntType(32)),
            ("Rectangle", StructType("Rect", (
                StructField("w", IntType(32)),
                StructField("h", IntType(32)),
            ))),
            ("None", VoidType()),
        ))
        assert not e.is_c_like
        assert e.num_variants == 3
        assert e.variant_type("Circle") == IntType(32)
        assert e.is_sized()

    def test_enum_size(self):
        """Enum size includes tag + max payload."""
        e = EnumType("Opt", (
            ("Some", IntType(32)),
            ("None", VoidType()),
        ))
        # Tag = 8 bits, payload = 32 bits
        total = e.size_bits()
        assert total >= 40  # at least tag + payload

    def test_enum_serialization(self):
        """Round-trip serialization."""
        e = EnumType("Color", (
            ("Red", VoidType()),
            ("Green", VoidType()),
        ))
        d = e.to_dict()
        assert d["kind"] == "enum"
        assert d["name"] == "Color"
        restored = type_from_dict(d)
        assert isinstance(restored, EnumType)
        assert restored.num_variants == 2

    def test_enum_is_aggregate(self):
        e = EnumType("X", (("A", VoidType()),))
        assert e.is_aggregate()

    def test_enum_variant_names(self):
        e = EnumType("Dir", (
            ("North", VoidType()),
            ("South", VoidType()),
            ("East", VoidType()),
            ("West", VoidType()),
        ))
        assert e.variant_names == ("North", "South", "East", "West")

    def test_enum_missing_variant(self):
        e = EnumType("X", (("A", VoidType()),))
        with pytest.raises(KeyError):
            e.variant_index("B")

    def test_enum_equality(self):
        e1 = EnumType("X", (("A", VoidType()), ("B", IntType(32))))
        e2 = EnumType("X", (("A", VoidType()), ("B", IntType(32))))
        assert e1 == e2

    def test_enum_hash(self):
        e1 = EnumType("X", (("A", VoidType()),))
        e2 = EnumType("X", (("A", VoidType()),))
        assert hash(e1) == hash(e2)

    def test_enum_str(self):
        e = EnumType("Color", (
            ("Red", VoidType()),
            ("Green", IntType(32)),
        ))
        s = str(e)
        assert "Red" in s
        assert "Green" in s

    def test_collect_types_enum(self):
        e = EnumType("Opt", (
            ("Some", IntType(32)),
            ("None", VoidType()),
        ))
        types = collect_types(e)
        assert any(isinstance(t, EnumType) for t in types)
        assert any(isinstance(t, IntType) for t in types)

    def test_large_enum_tag_width(self):
        """Enum with > 256 variants needs 16-bit tag."""
        variants = tuple((f"V{i}", VoidType()) for i in range(300))
        e = EnumType("Big", variants)
        assert e._effective_tag_width() == 16


# ---------------------------------------------------------------------------
# SMT Encoder: Struct/Enum tests
# ---------------------------------------------------------------------------

class TestSMTStructEnum:
    """Tests for struct and enum SMT encoding."""

    def setup_method(self):
        self.encoder = SMTEncoder(pointer_width=64)
        self.ctx = EncodingContext(pointer_width=64)

    def test_encode_struct_type(self):
        """StructType encodes to BitVecSort."""
        st = StructType("Point", (
            StructField("x", IntType(32)),
            StructField("y", IntType(32)),
        ))
        sort = self.encoder.encode_type(st)
        assert z3.is_bv_sort(sort)
        assert sort.size() == st.size_bits()

    def test_encode_enum_type(self):
        """EnumType encodes to BitVecSort."""
        et = EnumType("Opt", (
            ("Some", IntType(32)),
            ("None", VoidType()),
        ))
        sort = self.encoder.encode_type(et)
        assert z3.is_bv_sort(sort)
        assert sort.size() >= 40  # tag + payload

    def test_struct_literal(self):
        """Construct a struct bitvector from field values."""
        st = StructType("Point", (
            StructField("x", IntType(32)),
            StructField("y", IntType(32)),
        ))
        x_val = z3.BitVecVal(10, 32)
        y_val = z3.BitVecVal(20, 32)
        result = self.encoder.encode_struct_literal(st, [x_val, y_val], self.ctx)
        assert z3.is_bv(result)
        assert result.size() == st.size_bits()

    def test_enum_construct(self):
        """Construct an enum bitvector with tag and payload."""
        et = EnumType("Opt", (
            ("Some", IntType(32)),
            ("None", VoidType()),
        ))
        payload = z3.BitVecVal(42, 32)
        result = self.encoder.encode_enum_construct(et, "Some", payload, self.ctx)
        assert z3.is_bv(result)

    def test_enum_construct_none(self):
        """Construct a None variant with no payload."""
        et = EnumType("Opt", (
            ("Some", IntType(32)),
            ("None", VoidType()),
        ))
        result = self.encoder.encode_enum_construct(et, "None", None, self.ctx)
        assert z3.is_bv(result)

    def test_enum_discriminant(self):
        """Extract discriminant from enum bitvector."""
        et = EnumType("Dir", (
            ("North", VoidType()),
            ("South", VoidType()),
        ))
        bv = self.encoder.encode_enum_construct(et, "South", None, self.ctx)
        tag = self.encoder.encode_enum_discriminant(bv, et)
        # Solve to verify tag == 1
        s = z3.Solver()
        s.add(tag == z3.BitVecVal(1, et._effective_tag_width()))
        assert s.check() == z3.sat

    def test_encode_extract_value_struct(self):
        """ExtractValue on a struct — extract field by index."""
        st = StructType("Pair", (
            StructField("a", IntType(32)),
            StructField("b", IntType(32)),
        ))
        # Create a struct value
        agg_val = Value(st, "my_struct")
        struct_bv = z3.BitVec("my_struct", st.size_bits())
        self.ctx.declarations["my_struct"] = struct_bv

        # Extract field 0
        inst = ExtractValueInst(agg_val, (0,), IntType(32), "field_a")
        result = self.encoder.encode_extract_value(inst, self.ctx)
        assert z3.is_bv(result)
        assert result.size() == 32

    def test_encode_insert_value_struct(self):
        """InsertValue on a struct — insert a value at field index."""
        st = StructType("Pair", (
            StructField("a", IntType(32)),
            StructField("b", IntType(32)),
        ))
        agg_val = Value(st, "my_struct")
        struct_bv = z3.BitVec("my_struct", st.size_bits())
        self.ctx.declarations["my_struct"] = struct_bv

        new_val = Value(IntType(32), "new_a")
        new_bv = z3.BitVecVal(99, 32)
        self.ctx.declarations["new_a"] = new_bv

        inst = InsertValueInst(agg_val, new_val, (0,), "updated_struct")
        result = self.encoder.encode_insert_value(inst, self.ctx)
        assert z3.is_bv(result)
        assert result.size() == st.size_bits()

    def test_encode_switch_instruction(self):
        """SwitchInst sets branch conditions for each case."""
        cond_val = Value(IntType(32), "switch_cond")
        cond_bv = z3.BitVec("switch_cond", 32)
        self.ctx.declarations["switch_cond"] = cond_bv

        bb_default = BasicBlock("default")
        bb_case0 = BasicBlock("case0")
        bb_case1 = BasicBlock("case1")

        case0_const = Constant(IntType(32), 0, "c0")
        case1_const = Constant(IntType(32), 1, "c1")

        inst = SwitchInst(
            cond_val, bb_default,
            [(case0_const, bb_case0), (case1_const, bb_case1)],
        )

        self.encoder.encode_switch(inst, self.ctx)

        # Check that path conditions were set
        assert self.ctx.get("_branch_cond_case0") is not None
        assert self.ctx.get("_branch_cond_case1") is not None
        assert self.ctx.get("_branch_cond_default") is not None

    def test_struct_extract_insert_roundtrip(self):
        """Verify that insert followed by extract returns the inserted value."""
        st = StructType("Pair", (
            StructField("a", IntType(32)),
            StructField("b", IntType(32)),
        ))

        # Start with zero struct
        zero_struct = z3.BitVecVal(0, st.size_bits())
        val = z3.BitVecVal(42, 32)

        # Build struct literal
        struct_bv = self.encoder.encode_struct_literal(st, [val, z3.BitVecVal(0, 32)], self.ctx)

        # Extract field 0
        agg_val = Value(st, "test_struct")
        self.ctx.declarations["test_struct"] = struct_bv
        inst = ExtractValueInst(agg_val, (0,), IntType(32), "extracted")
        result = self.encoder.encode_extract_value(inst, self.ctx)

        # Verify result == 42
        solver = z3.Solver()
        solver.add(result == z3.BitVecVal(42, 32))
        assert solver.check() == z3.sat


# ---------------------------------------------------------------------------
# Alignment: Structural pattern tests
# ---------------------------------------------------------------------------

class TestAlignmentPatterns:
    """Tests for improved alignment patterns."""

    def test_switch_match_similarity(self):
        """Switch and match blocks should have non-zero similarity."""
        from src.product_program.alignment import _instruction_similarity

        # Create a SwitchInst and a BranchInst
        cond = Value(IntType(32), "cond")
        bb1 = BasicBlock("target1")
        bb2 = BasicBlock("target2")
        case_const = Constant(IntType(32), 0, "c0")

        switch_inst = SwitchInst(cond, bb1, [(case_const, bb2)])

        cond2 = Value(IntType(1), "br_cond")
        from src.ir.instructions import BranchInst
        branch_inst = BranchInst(bb1, condition=cond2, false_target=bb2, name="br")

        sim = _instruction_similarity(switch_inst, branch_inst)
        assert sim > 0.0, "Switch and conditional branch should have some similarity"

    def test_extractvalue_gep_similarity(self):
        """ExtractValue and GEP should have structural similarity."""
        from src.product_program.alignment import _instruction_similarity
        from src.ir.instructions import GetElementPtrInst
        from src.ir.types import PointerType

        st = StructType("S", (StructField("x", IntType(32)),))
        agg = Value(st, "agg")
        ev = ExtractValueInst(agg, (0,), IntType(32), "ev")

        ptr = Value(PointerType(IntType(32)), "ptr")
        idx = Value(IntType(32), "idx")
        gep = GetElementPtrInst(IntType(32), ptr, [idx], name="gep")

        sim = _instruction_similarity(ev, gep)
        assert sim > 0.0, "ExtractValue and GEP should have structural similarity"

    def test_type_similarity_struct(self):
        """Struct type similarity should be computed correctly."""
        from src.product_program.alignment import _type_similarity

        st1 = StructType("A", (
            StructField("x", IntType(32)),
            StructField("y", IntType(32)),
        ))
        st2 = StructType("B", (
            StructField("a", IntType(32)),
            StructField("b", IntType(32)),
        ))
        sim = _type_similarity(st1, st2)
        assert sim > 0.7, "Same-shape structs should be very similar"

    def test_type_similarity_enum(self):
        """Enum type similarity."""
        from src.product_program.alignment import _type_similarity

        e1 = EnumType("A", (("X", VoidType()), ("Y", VoidType())))
        e2 = EnumType("B", (("P", VoidType()), ("Q", VoidType())))
        sim = _type_similarity(e1, e2)
        assert sim > 0.5

    def test_type_similarity_int_enum_cross(self):
        """IntType vs EnumType should have partial similarity."""
        from src.product_program.alignment import _type_similarity

        i = IntType(32)
        e = EnumType("Dir", (("N", VoidType()), ("S", VoidType())))
        sim = _type_similarity(i, e)
        assert sim > 0.0, "Int and enum should have partial cross-type similarity"

    def test_opcode_tag_new_instructions(self):
        """New instructions should have proper opcode tags."""
        from src.product_program.alignment import _instruction_opcode_tag

        st = StructType("S", (StructField("x", IntType(32)),))
        agg = Value(st, "agg")
        ev = ExtractValueInst(agg, (0,), IntType(32), "ev")
        assert _instruction_opcode_tag(ev) == "extractvalue"

        val = Value(IntType(32), "v")
        iv = InsertValueInst(agg, val, (0,), "iv")
        assert _instruction_opcode_tag(iv) == "insertvalue"

        cond = Value(IntType(32), "cond")
        bb = BasicBlock("bb")
        sw = SwitchInst(cond, bb, [])
        assert _instruction_opcode_tag(sw) == "switch"


# ---------------------------------------------------------------------------
# Expanded benchmark tests
# ---------------------------------------------------------------------------

class TestExpandedBenchmarks:
    """Tests for expanded benchmark pairs."""

    def test_expanded_pair_count(self):
        from benchmarks.pairs.expanded_benchmark_pairs import EXPANDED_BENCHMARKS
        assert len(EXPANDED_BENCHMARKS) >= 130

    def test_expanded_categories(self):
        from benchmarks.pairs.expanded_benchmark_pairs import get_expanded_categories
        cats = get_expanded_categories()
        assert "struct" in cats
        assert "enum" in cats
        assert "float" in cats
        assert "c2rust" in cats
        assert "iterator" in cats
        assert "cast" in cats
        assert "compound" in cats
        assert "control_flow" in cats

    def test_combined_benchmarks(self):
        from benchmarks.pairs import COMBINED_BENCHMARKS
        assert len(COMBINED_BENCHMARKS) >= 180

    def test_struct_pairs_count(self):
        from benchmarks.pairs.expanded_benchmark_pairs import STRUCT_PAIRS
        assert len(STRUCT_PAIRS) >= 15

    def test_enum_pairs_count(self):
        from benchmarks.pairs.expanded_benchmark_pairs import ENUM_PAIRS
        assert len(ENUM_PAIRS) >= 15

    def test_all_pairs_have_sources(self):
        from benchmarks.pairs.expanded_benchmark_pairs import EXPANDED_BENCHMARKS
        for bp in EXPANDED_BENCHMARKS:
            assert bp.c_source.strip(), f"{bp.name} missing C source"
            assert bp.rust_source.strip(), f"{bp.name} missing Rust source"
            assert bp.expected_result in ("equivalent", "divergent", "conditional"), \
                f"{bp.name} has invalid expected_result: {bp.expected_result}"

    def test_unique_names(self):
        from benchmarks.pairs.expanded_benchmark_pairs import EXPANDED_BENCHMARKS
        names = [bp.name for bp in EXPANDED_BENCHMARKS]
        assert len(names) == len(set(names)), "Duplicate benchmark names"

    def test_divergent_pairs_have_kind(self):
        from benchmarks.pairs.expanded_benchmark_pairs import EXPANDED_BENCHMARKS
        for bp in EXPANDED_BENCHMARKS:
            if bp.expected_result == "divergent":
                assert bp.divergence_kind, f"{bp.name} is divergent but has no divergence_kind"


# ---------------------------------------------------------------------------
# Union/Struct encoding edge cases
# ---------------------------------------------------------------------------

class TestUnionEncoding:
    """Test that UnionType encodes properly as bitvector."""

    def setup_method(self):
        self.encoder = SMTEncoder(pointer_width=64)

    def test_union_type_encoding(self):
        ut = UnionType("MyUnion", (
            ("int_val", IntType(32)),
            ("long_val", IntType(64)),
        ))
        sort = self.encoder.encode_type(ut)
        assert z3.is_bv_sort(sort)
        assert sort.size() >= 64


# ---------------------------------------------------------------------------
# Constant encoding for struct/enum
# ---------------------------------------------------------------------------

class TestConstantEncoding:
    """Test constant encoding for aggregate types."""

    def setup_method(self):
        self.encoder = SMTEncoder(pointer_width=64)
        self.ctx = EncodingContext(pointer_width=64)

    def test_struct_constant(self):
        """Struct constant encodes to bitvector."""
        st = StructType("S", (StructField("x", IntType(32)),))
        c = Constant(st, 0, "zero_struct")
        result = self.encoder.encode_constant(c, self.ctx)
        # Should get a bitvector of struct size
        assert z3.is_bv(result)

    def test_enum_constant(self):
        """Enum constant encodes to bitvector."""
        et = EnumType("E", (("A", VoidType()),))
        c = Constant(et, 0, "enum_val")
        result = self.encoder.encode_constant(c, self.ctx)
        assert z3.is_bv(result)
