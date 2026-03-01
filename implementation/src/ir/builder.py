"""
IRBuilder for programmatic construction of Cross-Language Equivalence Verifier IR.

Provides a stateful builder with insert-point management, instruction creation
helpers for every instruction type, automatic type inference, and SSA value
numbering.
"""

from __future__ import annotations

from typing import Optional, Sequence

from .basic_block import BasicBlock
from .function import Function
from .module import Module, GlobalVariable, ExternalDeclaration
from .instructions import (
    AllocaInst,
    Argument,
    AtomicCmpXchgInst,
    AtomicOrdering,
    AtomicRMWInst,
    AtomicRMWOp,
    BinOpKind,
    BinaryOp,
    BranchInst,
    CallInst,
    CastInst,
    CastKind,
    CmpPredicate,
    CompareOp,
    Constant,
    ExtractValueInst,
    FenceInst,
    GetElementPtrInst,
    InsertValueInst,
    Instruction,
    InstructionMetadata,
    LoadInst,
    MemcpyInst,
    MemsetInst,
    PhiInst,
    ReturnInst,
    SelectInst,
    SourceLocation,
    StoreInst,
    SwitchInst,
    UnaryOp,
    UnaryOpKind,
    Value,
)
from .types import (
    ArrayType,
    FloatKind,
    FloatType,
    FunctionType,
    IRType,
    IntType,
    PointerType,
    Signedness,
    StructType,
    VoidType,
    OverflowBehavior,
)


class IRBuilder:
    """Stateful builder for constructing IR programmatically.

    Usage::

        mod = Module("example")
        fn_type = FunctionType(IntType(32), (IntType(32), IntType(32)))
        func = mod.create_function("add", fn_type)
        entry = func.create_block("entry")

        builder = IRBuilder()
        builder.position_at_end(entry)
        result = builder.add(func.get_argument(0), func.get_argument(1), name="sum")
        builder.ret(result)
    """

    def __init__(self) -> None:
        self._insert_block: BasicBlock | None = None
        self._insert_point: int | None = None  # None means "at end"
        self._default_metadata: InstructionMetadata = InstructionMetadata()
        self._name_counter: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Insert point management
    # ------------------------------------------------------------------

    @property
    def insert_block(self) -> BasicBlock | None:
        return self._insert_block

    def position_at_end(self, block: BasicBlock) -> None:
        """Set the insert point to the end of *block*."""
        self._insert_block = block
        self._insert_point = None

    def position_at_start(self, block: BasicBlock) -> None:
        """Set the insert point to the beginning of *block*."""
        self._insert_block = block
        self._insert_point = 0

    def position_before(self, inst: Instruction) -> None:
        """Set the insert point just before *inst*."""
        if inst.parent is None:
            raise ValueError("Instruction has no parent block")
        self._insert_block = inst.parent
        idx = inst.parent.instructions.index(inst)
        self._insert_point = idx

    def position_after(self, inst: Instruction) -> None:
        """Set the insert point just after *inst*."""
        if inst.parent is None:
            raise ValueError("Instruction has no parent block")
        self._insert_block = inst.parent
        idx = inst.parent.instructions.index(inst)
        self._insert_point = idx + 1

    def clear_insertion_point(self) -> None:
        self._insert_block = None
        self._insert_point = None

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def set_default_source_location(self, loc: SourceLocation) -> None:
        self._default_metadata.source_loc = loc

    def set_default_overflow(self, behavior: OverflowBehavior) -> None:
        self._default_metadata.overflow = behavior

    def _make_metadata(self) -> InstructionMetadata:
        """Clone the current default metadata for a new instruction."""
        return InstructionMetadata(
            source_loc=self._default_metadata.source_loc,
            overflow=self._default_metadata.overflow,
            tags=dict(self._default_metadata.tags),
        )

    # ------------------------------------------------------------------
    # Name generation
    # ------------------------------------------------------------------

    def _auto_name(self, prefix: str) -> str:
        """Generate a unique name with the given prefix."""
        count = self._name_counter.get(prefix, 0)
        self._name_counter[prefix] = count + 1
        if count == 0:
            return prefix
        return f"{prefix}.{count}"

    # ------------------------------------------------------------------
    # Insertion helper
    # ------------------------------------------------------------------

    def _insert(self, inst: Instruction) -> Instruction:
        """Insert *inst* at the current insertion point."""
        if self._insert_block is None:
            raise RuntimeError("No insertion point set")
        if self._insert_point is None:
            self._insert_block.append(inst)
        else:
            self._insert_block.insert(self._insert_point, inst)
            self._insert_point += 1
        return inst

    # ------------------------------------------------------------------
    # Binary arithmetic
    # ------------------------------------------------------------------

    def _binop(self, op: BinOpKind, lhs: Value, rhs: Value, name: str = "") -> BinaryOp:
        inst = BinaryOp(op, lhs, rhs, name=name or self._auto_name(op.value),
                        metadata=self._make_metadata())
        self._insert(inst)
        return inst

    def binop(self, op: BinOpKind, lhs: Value, rhs: Value, name: str = "",
              metadata: InstructionMetadata | None = None) -> BinaryOp:
        inst = BinaryOp(op, lhs, rhs, name=name or self._auto_name(op.value),
                        metadata=metadata or self._make_metadata())
        self._insert(inst)
        return inst

    def add(self, lhs: Value, rhs: Value, name: str = "") -> BinaryOp:
        return self._binop(BinOpKind.ADD, lhs, rhs, name)

    def sub(self, lhs: Value, rhs: Value, name: str = "") -> BinaryOp:
        return self._binop(BinOpKind.SUB, lhs, rhs, name)

    def mul(self, lhs: Value, rhs: Value, name: str = "") -> BinaryOp:
        return self._binop(BinOpKind.MUL, lhs, rhs, name)

    def sdiv(self, lhs: Value, rhs: Value, name: str = "") -> BinaryOp:
        return self._binop(BinOpKind.SDIV, lhs, rhs, name)

    def udiv(self, lhs: Value, rhs: Value, name: str = "") -> BinaryOp:
        return self._binop(BinOpKind.UDIV, lhs, rhs, name)

    def srem(self, lhs: Value, rhs: Value, name: str = "") -> BinaryOp:
        return self._binop(BinOpKind.SREM, lhs, rhs, name)

    def urem(self, lhs: Value, rhs: Value, name: str = "") -> BinaryOp:
        return self._binop(BinOpKind.UREM, lhs, rhs, name)

    def fadd(self, lhs: Value, rhs: Value, name: str = "") -> BinaryOp:
        return self._binop(BinOpKind.FADD, lhs, rhs, name)

    def fsub(self, lhs: Value, rhs: Value, name: str = "") -> BinaryOp:
        return self._binop(BinOpKind.FSUB, lhs, rhs, name)

    def fmul(self, lhs: Value, rhs: Value, name: str = "") -> BinaryOp:
        return self._binop(BinOpKind.FMUL, lhs, rhs, name)

    def fdiv(self, lhs: Value, rhs: Value, name: str = "") -> BinaryOp:
        return self._binop(BinOpKind.FDIV, lhs, rhs, name)

    def frem(self, lhs: Value, rhs: Value, name: str = "") -> BinaryOp:
        return self._binop(BinOpKind.FREM, lhs, rhs, name)

    def shl(self, lhs: Value, rhs: Value, name: str = "") -> BinaryOp:
        return self._binop(BinOpKind.SHL, lhs, rhs, name)

    def lshr(self, lhs: Value, rhs: Value, name: str = "") -> BinaryOp:
        return self._binop(BinOpKind.LSHR, lhs, rhs, name)

    def ashr(self, lhs: Value, rhs: Value, name: str = "") -> BinaryOp:
        return self._binop(BinOpKind.ASHR, lhs, rhs, name)

    def and_(self, lhs: Value, rhs: Value, name: str = "") -> BinaryOp:
        return self._binop(BinOpKind.AND, lhs, rhs, name)

    def or_(self, lhs: Value, rhs: Value, name: str = "") -> BinaryOp:
        return self._binop(BinOpKind.OR, lhs, rhs, name)

    def xor(self, lhs: Value, rhs: Value, name: str = "") -> BinaryOp:
        return self._binop(BinOpKind.XOR, lhs, rhs, name)

    # ------------------------------------------------------------------
    # Unary operations
    # ------------------------------------------------------------------

    def neg(self, operand: Value, name: str = "") -> UnaryOp:
        inst = UnaryOp(UnaryOpKind.NEG, operand, name=name or self._auto_name("neg"),
                       metadata=self._make_metadata())
        self._insert(inst)
        return inst

    def not_(self, operand: Value, name: str = "") -> UnaryOp:
        inst = UnaryOp(UnaryOpKind.NOT, operand, name=name or self._auto_name("not"),
                       metadata=self._make_metadata())
        self._insert(inst)
        return inst

    def bitwise_not(self, operand: Value, name: str = "") -> UnaryOp:
        inst = UnaryOp(UnaryOpKind.BITWISE_NOT, operand,
                       name=name or self._auto_name("bitnot"),
                       metadata=self._make_metadata())
        self._insert(inst)
        return inst

    def fneg(self, operand: Value, name: str = "") -> UnaryOp:
        inst = UnaryOp(UnaryOpKind.FNEG, operand, name=name or self._auto_name("fneg"),
                       metadata=self._make_metadata())
        self._insert(inst)
        return inst

    # ------------------------------------------------------------------
    # Comparisons
    # ------------------------------------------------------------------

    def _cmp(self, pred: CmpPredicate, lhs: Value, rhs: Value, name: str = "") -> CompareOp:
        inst = CompareOp(pred, lhs, rhs, name=name or self._auto_name("cmp"),
                         metadata=self._make_metadata())
        self._insert(inst)
        return inst

    def icmp(self, pred: CmpPredicate, lhs: Value, rhs: Value, name: str = "") -> CompareOp:
        return self._cmp(pred, lhs, rhs, name)

    def fcmp(self, pred: CmpPredicate, lhs: Value, rhs: Value, name: str = "") -> CompareOp:
        return self._cmp(pred, lhs, rhs, name)

    def icmp_eq(self, lhs: Value, rhs: Value, name: str = "") -> CompareOp:
        return self._cmp(CmpPredicate.EQ, lhs, rhs, name)

    def icmp_ne(self, lhs: Value, rhs: Value, name: str = "") -> CompareOp:
        return self._cmp(CmpPredicate.NE, lhs, rhs, name)

    def icmp_slt(self, lhs: Value, rhs: Value, name: str = "") -> CompareOp:
        return self._cmp(CmpPredicate.SLT, lhs, rhs, name)

    def icmp_sle(self, lhs: Value, rhs: Value, name: str = "") -> CompareOp:
        return self._cmp(CmpPredicate.SLE, lhs, rhs, name)

    def icmp_sgt(self, lhs: Value, rhs: Value, name: str = "") -> CompareOp:
        return self._cmp(CmpPredicate.SGT, lhs, rhs, name)

    def icmp_sge(self, lhs: Value, rhs: Value, name: str = "") -> CompareOp:
        return self._cmp(CmpPredicate.SGE, lhs, rhs, name)

    def icmp_ult(self, lhs: Value, rhs: Value, name: str = "") -> CompareOp:
        return self._cmp(CmpPredicate.ULT, lhs, rhs, name)

    def icmp_ule(self, lhs: Value, rhs: Value, name: str = "") -> CompareOp:
        return self._cmp(CmpPredicate.ULE, lhs, rhs, name)

    def icmp_ugt(self, lhs: Value, rhs: Value, name: str = "") -> CompareOp:
        return self._cmp(CmpPredicate.UGT, lhs, rhs, name)

    def icmp_uge(self, lhs: Value, rhs: Value, name: str = "") -> CompareOp:
        return self._cmp(CmpPredicate.UGE, lhs, rhs, name)

    def fcmp_oeq(self, lhs: Value, rhs: Value, name: str = "") -> CompareOp:
        return self._cmp(CmpPredicate.OEQ, lhs, rhs, name)

    def fcmp_one(self, lhs: Value, rhs: Value, name: str = "") -> CompareOp:
        return self._cmp(CmpPredicate.ONE, lhs, rhs, name)

    def fcmp_olt(self, lhs: Value, rhs: Value, name: str = "") -> CompareOp:
        return self._cmp(CmpPredicate.OLT, lhs, rhs, name)

    def fcmp_ole(self, lhs: Value, rhs: Value, name: str = "") -> CompareOp:
        return self._cmp(CmpPredicate.OLE, lhs, rhs, name)

    def fcmp_ogt(self, lhs: Value, rhs: Value, name: str = "") -> CompareOp:
        return self._cmp(CmpPredicate.OGT, lhs, rhs, name)

    def fcmp_oge(self, lhs: Value, rhs: Value, name: str = "") -> CompareOp:
        return self._cmp(CmpPredicate.OGE, lhs, rhs, name)

    # ------------------------------------------------------------------
    # Memory operations
    # ------------------------------------------------------------------

    def alloca(
        self, ty: IRType, num_elements: int = 1, alignment: int = 0, name: str = ""
    ) -> AllocaInst:
        inst = AllocaInst(ty, num_elements, alignment,
                          name=name or self._auto_name("alloca"),
                          metadata=self._make_metadata())
        self._insert(inst)
        return inst

    def load(
        self, address: Value, result_type: IRType | None = None,
        volatile: bool = False, alignment: int = 0, name: str = ""
    ) -> LoadInst:
        if result_type is None:
            if isinstance(address.type, PointerType):
                result_type = address.type.pointee
            else:
                result_type = VoidType()
        inst = LoadInst(address, result_type, volatile, alignment,
                        name=name or self._auto_name("load"),
                        metadata=self._make_metadata())
        self._insert(inst)
        return inst

    def store(
        self, value: Value, address: Value,
        volatile: bool = False, alignment: int = 0
    ) -> StoreInst:
        inst = StoreInst(value, address, volatile, alignment,
                         metadata=self._make_metadata())
        self._insert(inst)
        return inst

    def gep(
        self,
        source_type_or_base,
        base_or_indices=None,
        indices=None,
        inbounds: bool = True,
        name: str = "",
    ) -> GetElementPtrInst:
        # Support both old gep(base, indices) and new gep(source_type, base, indices)
        if indices is None and isinstance(base_or_indices, (list, tuple)):
            # Old-style: gep(base, indices)
            base = source_type_or_base
            indices = base_or_indices
            source_type = VoidType()  # placeholder
        else:
            source_type = source_type_or_base
            base = base_or_indices
        inst = GetElementPtrInst(
            source_type, base, indices, inbounds=inbounds,
            name=name or self._auto_name("gep"),
            metadata=self._make_metadata(),
        )
        self._insert(inst)
        return inst

    def gep_struct(
        self, base: Value, struct_type: StructType, field_index: int,
        name: str = ""
    ) -> GetElementPtrInst:
        """Convenience GEP for struct field access."""
        zero = Constant.int_const(0, IntType(32, Signedness.SIGNED))
        idx = Constant.int_const(field_index, IntType(32, Signedness.SIGNED))
        result_type = PointerType(struct_type.fields[field_index].type)
        inst = GetElementPtrInst(
            struct_type, base, [zero, idx],
            result_type=result_type, inbounds=True,
            name=name or self._auto_name("gep.field"),
            metadata=self._make_metadata(),
        )
        self._insert(inst)
        return inst

    def gep_array(
        self, base: Value, element_type: IRType, index: Value,
        name: str = ""
    ) -> GetElementPtrInst:
        """Convenience GEP for array element access."""
        zero = Constant.int_const(0, IntType(64, Signedness.SIGNED))
        result_type = PointerType(element_type)
        arr_type = base.type
        if isinstance(arr_type, PointerType):
            arr_type = arr_type.pointee
        inst = GetElementPtrInst(
            arr_type, base, [zero, index],
            result_type=result_type, inbounds=True,
            name=name or self._auto_name("gep.elem"),
            metadata=self._make_metadata(),
        )
        self._insert(inst)
        return inst

    # ------------------------------------------------------------------
    # Casts
    # ------------------------------------------------------------------

    def cast(self, kind: CastKind, val: Value, dest_ty: IRType, name: str = "") -> CastInst:
        return self._cast(kind, val, dest_ty, name)

    def _cast(self, kind: CastKind, val: Value, dest_ty: IRType, name: str = "") -> CastInst:
        inst = CastInst(kind, val, dest_ty,
                        name=name or self._auto_name(kind.value),
                        metadata=self._make_metadata())
        self._insert(inst)
        return inst

    def trunc(self, val: Value, dest_ty: IntType, name: str = "") -> CastInst:
        return self._cast(CastKind.TRUNC, val, dest_ty, name)

    def zext(self, val: Value, dest_ty: IntType, name: str = "") -> CastInst:
        return self._cast(CastKind.ZEXT, val, dest_ty, name)

    def sext(self, val: Value, dest_ty: IntType, name: str = "") -> CastInst:
        return self._cast(CastKind.SEXT, val, dest_ty, name)

    def fptrunc(self, val: Value, dest_ty: FloatType, name: str = "") -> CastInst:
        return self._cast(CastKind.FPTRUNC, val, dest_ty, name)

    def fpext(self, val: Value, dest_ty: FloatType, name: str = "") -> CastInst:
        return self._cast(CastKind.FPEXT, val, dest_ty, name)

    def fptosi(self, val: Value, dest_ty: IntType, name: str = "") -> CastInst:
        return self._cast(CastKind.FPTOSI, val, dest_ty, name)

    def fptoui(self, val: Value, dest_ty: IntType, name: str = "") -> CastInst:
        return self._cast(CastKind.FPTOUI, val, dest_ty, name)

    def sitofp(self, val: Value, dest_ty: FloatType, name: str = "") -> CastInst:
        return self._cast(CastKind.SITOFP, val, dest_ty, name)

    def uitofp(self, val: Value, dest_ty: FloatType, name: str = "") -> CastInst:
        return self._cast(CastKind.UITOFP, val, dest_ty, name)

    def bitcast(self, val: Value, dest_ty: IRType, name: str = "") -> CastInst:
        return self._cast(CastKind.BITCAST, val, dest_ty, name)

    def ptrtoint(self, val: Value, dest_ty: IntType, name: str = "") -> CastInst:
        return self._cast(CastKind.PTRTOINT, val, dest_ty, name)

    def inttoptr(self, val: Value, dest_ty: PointerType, name: str = "") -> CastInst:
        return self._cast(CastKind.INTTOPTR, val, dest_ty, name)

    # ------------------------------------------------------------------
    # Control flow
    # ------------------------------------------------------------------

    def call(
        self,
        callee: Value,
        args: Sequence[Value],
        return_type: IRType | None = None,
        callee_name: str = "",
        is_tail: bool = False,
        name: str = "",
    ) -> CallInst:
        if return_type is None:
            callee_ty = callee.type
            if isinstance(callee_ty, PointerType):
                callee_ty = callee_ty.pointee
            if isinstance(callee_ty, FunctionType):
                return_type = callee_ty.return_type
            else:
                return_type = VoidType()
        inst = CallInst(
            callee, args, return_type, callee_name, is_tail,
            name=name or (self._auto_name("call") if not isinstance(return_type, VoidType) else ""),
            metadata=self._make_metadata(),
        )
        self._insert(inst)
        return inst

    def ret(self, value: Value | None = None) -> ReturnInst:
        inst = ReturnInst(value, metadata=self._make_metadata())
        self._insert(inst)
        return inst

    def ret_void(self) -> ReturnInst:
        return self.ret(None)

    def br(self, target: BasicBlock) -> BranchInst:
        inst = BranchInst(target, metadata=self._make_metadata())
        self._insert(inst)
        return inst

    def cond_br(
        self, condition: Value, true_bb: BasicBlock, false_bb: BasicBlock
    ) -> BranchInst:
        inst = BranchInst(true_bb, condition, false_bb, metadata=self._make_metadata())
        self._insert(inst)
        return inst

    def switch(
        self,
        condition: Value,
        default_target: BasicBlock,
        cases: Sequence[tuple[Constant, BasicBlock]],
    ) -> SwitchInst:
        inst = SwitchInst(condition, default_target, cases, metadata=self._make_metadata())
        self._insert(inst)
        return inst

    # ------------------------------------------------------------------
    # SSA helpers
    # ------------------------------------------------------------------

    def phi(
        self,
        ty: IRType,
        incoming: Sequence[tuple[Value, BasicBlock]] | None = None,
        name: str = "",
    ) -> PhiInst:
        inc = incoming or []
        inst = PhiInst(ty, inc, name=name or self._auto_name("phi"),
                       metadata=self._make_metadata())
        self._insert(inst)
        return inst

    def select(
        self, cond: Value, true_val: Value, false_val: Value, name: str = ""
    ) -> SelectInst:
        inst = SelectInst(cond, true_val, false_val,
                          name=name or self._auto_name("sel"),
                          metadata=self._make_metadata())
        self._insert(inst)
        return inst

    # ------------------------------------------------------------------
    # Aggregate operations
    # ------------------------------------------------------------------

    def extract_value(
        self, aggregate: Value, indices: tuple[int, ...], result_type: IRType,
        name: str = ""
    ) -> ExtractValueInst:
        inst = ExtractValueInst(
            aggregate, indices, result_type,
            name=name or self._auto_name("extract"),
            metadata=self._make_metadata(),
        )
        self._insert(inst)
        return inst

    def insert_value(
        self, aggregate: Value, value: Value, indices: tuple[int, ...],
        name: str = ""
    ) -> InsertValueInst:
        inst = InsertValueInst(
            aggregate, value, indices,
            name=name or self._auto_name("insert"),
            metadata=self._make_metadata(),
        )
        self._insert(inst)
        return inst

    # ------------------------------------------------------------------
    # Memory intrinsics
    # ------------------------------------------------------------------

    def memcpy(
        self, dest: Value, src: Value, length: Value,
        is_volatile: bool = False
    ) -> MemcpyInst:
        inst = MemcpyInst(dest, src, length, is_volatile,
                          metadata=self._make_metadata())
        self._insert(inst)
        return inst

    def memset(
        self, dest: Value, value: Value, length: Value,
        is_volatile: bool = False
    ) -> MemsetInst:
        inst = MemsetInst(dest, value, length, is_volatile,
                          metadata=self._make_metadata())
        self._insert(inst)
        return inst

    # ------------------------------------------------------------------
    # Atomics
    # ------------------------------------------------------------------

    def fence(self, ordering: AtomicOrdering = AtomicOrdering.SEQ_CST) -> FenceInst:
        inst = FenceInst(ordering, metadata=self._make_metadata())
        self._insert(inst)
        return inst

    def atomic_rmw(
        self, op: AtomicRMWOp, address: Value, value: Value,
        ordering: AtomicOrdering = AtomicOrdering.SEQ_CST,
        name: str = "",
    ) -> AtomicRMWInst:
        inst = AtomicRMWInst(
            op, address, value, ordering,
            name=name or self._auto_name("atomicrmw"),
            metadata=self._make_metadata(),
        )
        self._insert(inst)
        return inst

    def cmpxchg(
        self, address: Value, expected: Value, desired: Value,
        success_ordering: AtomicOrdering = AtomicOrdering.SEQ_CST,
        failure_ordering: AtomicOrdering = AtomicOrdering.SEQ_CST,
        name: str = "",
    ) -> AtomicCmpXchgInst:
        inst = AtomicCmpXchgInst(
            address, expected, desired, success_ordering, failure_ordering,
            name=name or self._auto_name("cmpxchg"),
            metadata=self._make_metadata(),
        )
        self._insert(inst)
        return inst

    # ------------------------------------------------------------------
    # Constant helpers
    # ------------------------------------------------------------------

    @staticmethod
    def const_int(value: int, ty: IntType | None = None) -> Constant:
        return Constant.int_const(value, ty)

    @staticmethod
    def const_float(value: float, ty: FloatType | None = None) -> Constant:
        return Constant.float_const(value, ty)

    @staticmethod
    def const_null(pointee: IRType | None = None) -> Constant:
        return Constant.null_ptr(pointee)

    @staticmethod
    def const_bool(value: bool) -> Constant:
        return Constant.bool_const(value)

    @staticmethod
    def const_undef(ty: IRType) -> Constant:
        return Constant.undef(ty)

    # ------------------------------------------------------------------
    # Block creation helpers
    # ------------------------------------------------------------------

    def create_block(self, name: str = "", func: Function | None = None) -> BasicBlock:
        """Create a new basic block. If *func* is given, add it to the function."""
        block = BasicBlock(name=name)
        if func is not None:
            func.add_block(block)
        elif self._insert_block is not None and self._insert_block.parent is not None:
            self._insert_block.parent.add_block(block)
        return block

    # ------------------------------------------------------------------
    # Common patterns
    # ------------------------------------------------------------------

    def build_increment(self, val: Value, name: str = "") -> BinaryOp:
        """val + 1"""
        one = Constant.int_const(1, val.type) if isinstance(val.type, IntType) else Constant.float_const(1.0, val.type)
        if isinstance(val.type, FloatType):
            return self.fadd(val, one, name)
        return self.add(val, one, name)

    def build_decrement(self, val: Value, name: str = "") -> BinaryOp:
        """val - 1"""
        one = Constant.int_const(1, val.type) if isinstance(val.type, IntType) else Constant.float_const(1.0, val.type)
        if isinstance(val.type, FloatType):
            return self.fsub(val, one, name)
        return self.sub(val, one, name)

    def build_is_null(self, ptr: Value, name: str = "") -> CompareOp:
        """ptr == null"""
        null = Constant.null_ptr()
        null_cast = self.bitcast(null, ptr.type, name=self._auto_name("null_cast"))
        return self.icmp_eq(ptr, null_cast, name)

    def build_is_not_null(self, ptr: Value, name: str = "") -> CompareOp:
        """ptr != null"""
        null = Constant.null_ptr()
        null_cast = self.bitcast(null, ptr.type, name=self._auto_name("null_cast"))
        return self.icmp_ne(ptr, null_cast, name)

    def build_abs(self, val: Value, name: str = "") -> SelectInst:
        """abs(val) for signed integers."""
        zero = Constant.int_const(0, val.type) if isinstance(val.type, IntType) else Constant.float_const(0.0, val.type)
        if isinstance(val.type, FloatType):
            neg_val = self.fneg(val, name=self._auto_name("abs.neg"))
            is_neg = self.fcmp_olt(val, zero, name=self._auto_name("abs.cmp"))
        else:
            neg_val = self.neg(val, name=self._auto_name("abs.neg"))
            is_neg = self.icmp_slt(val, zero, name=self._auto_name("abs.cmp"))
        return self.select(is_neg, neg_val, val, name or self._auto_name("abs"))

    def build_min(self, a: Value, b: Value, signed: bool = True, name: str = "") -> SelectInst:
        """min(a, b)"""
        if isinstance(a.type, FloatType):
            cmp = self.fcmp_olt(a, b, name=self._auto_name("min.cmp"))
        elif signed:
            cmp = self.icmp_slt(a, b, name=self._auto_name("min.cmp"))
        else:
            cmp = self.icmp_ult(a, b, name=self._auto_name("min.cmp"))
        return self.select(cmp, a, b, name or self._auto_name("min"))

    def build_max(self, a: Value, b: Value, signed: bool = True, name: str = "") -> SelectInst:
        """max(a, b)"""
        if isinstance(a.type, FloatType):
            cmp = self.fcmp_ogt(a, b, name=self._auto_name("max.cmp"))
        elif signed:
            cmp = self.icmp_sgt(a, b, name=self._auto_name("max.cmp"))
        else:
            cmp = self.icmp_ugt(a, b, name=self._auto_name("max.cmp"))
        return self.select(cmp, a, b, name or self._auto_name("max"))
