"""
SSA instruction types for the Cross-Language Equivalence Verifier IR.

Defines ~40 instruction types covering arithmetic, memory, control flow,
atomics, and aggregate operations. Each instruction carries operand types,
result type, metadata (source location, semantic config), and supports a
visitor pattern for analysis passes.
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Optional, Sequence

from .types import (
    IRType,
    IntType,
    FloatType,
    PointerType,
    ArrayType,
    StructType,
    UnionType,
    FunctionType,
    VoidType,
    OverflowBehavior,
    Signedness,
)

if TYPE_CHECKING:
    from .basic_block import BasicBlock


# ---------------------------------------------------------------------------
# Counters for unique value IDs
# ---------------------------------------------------------------------------

_NEXT_VALUE_ID = itertools.count()


def _fresh_id() -> int:
    return next(_NEXT_VALUE_ID)


def reset_value_counter(start: int = 0) -> None:
    """Reset the global value counter (useful for tests)."""
    global _NEXT_VALUE_ID
    _NEXT_VALUE_ID = itertools.count(start)


# ---------------------------------------------------------------------------
# Source location metadata
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SourceLocation:
    """Source-level location information attached to instructions."""
    file: str = ""
    line: int = 0
    column: int = 0
    language: str = ""  # "c" or "rust"

    def __str__(self) -> str:
        if not self.file:
            return "<unknown>"
        return f"{self.file}:{self.line}:{self.column}"


@dataclass
class InstructionMetadata:
    """Bag of metadata for an instruction."""
    source_loc: Optional[SourceLocation] = None
    comment: str = ""
    overflow: OverflowBehavior = OverflowBehavior.WRAP
    # Provenance / semantic tag – free-form for analysis passes
    tags: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Value – the base class for anything that produces a typed result
# ---------------------------------------------------------------------------

class Value:
    """An SSA value – every instruction result, function parameter, or
    constant is a Value.

    Attributes:
        id: unique numeric identifier for SSA naming (e.g. %42).
        name: optional human-readable name.
        type: the IR type of this value.
        users: set of Instructions that consume this value.
    """
    __slots__ = ("id", "name", "type", "users")

    def __init__(self, ty: IRType, name: str = "") -> None:
        self.id: int = _fresh_id()
        self.name: str = name
        self.type: IRType = ty
        self.users: set[Instruction] = set()

    @property
    def display_name(self) -> str:
        if self.name:
            return f"%{self.name}"
        return f"%{self.id}"

    def replace_all_uses_with(self, new_val: "Value") -> None:
        """RAUW – replace this value with *new_val* in all users."""
        for user in list(self.users):
            user._replace_operand(self, new_val)
            new_val.users.add(user)
        self.users.clear()

    def has_users(self) -> bool:
        return len(self.users) > 0

    def __repr__(self) -> str:
        return f"Value({self.display_name}: {self.type})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Value):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)


class Argument(Value):
    """A function argument value."""
    __slots__ = ("index",)

    def __init__(self, ty: IRType, index: int, name: str = "") -> None:
        super().__init__(ty, name or f"arg{index}")
        self.index = index


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class Constant(Value):
    """A compile-time constant value."""
    __slots__ = ("_value",)

    def __init__(self, ty: IRType, value: object, name: str = "") -> None:
        super().__init__(ty, name)
        self._value = value

    @property
    def value(self) -> object:
        return self._value

    def __repr__(self) -> str:
        return f"Constant({self.type}, {self._value})"

    @staticmethod
    def int_const(value: int, ty: IntType | None = None) -> "Constant":
        if ty is None:
            ty = IntType(64, Signedness.SIGNED)
        return Constant(ty, value)

    @staticmethod
    def float_const(value: float, ty: FloatType | None = None) -> "Constant":
        from .types import FloatKind
        if ty is None:
            ty = FloatType(FloatKind.F64)
        return Constant(ty, value)

    @staticmethod
    def null_ptr(pointee: IRType | None = None) -> "Constant":
        if pointee is None:
            pointee = VoidType()
        return Constant(PointerType(pointee), 0, "null")

    @staticmethod
    def bool_const(value: bool) -> "Constant":
        return Constant(IntType(1, Signedness.UNSIGNED), int(value))

    @staticmethod
    def undef(ty: IRType) -> "Constant":
        return Constant(ty, None, "undef")

    @property
    def is_zero(self) -> bool:
        return self._value == 0

    @property
    def is_null(self) -> bool:
        return isinstance(self.type, PointerType) and self._value == 0

    @property
    def is_undef(self) -> bool:
        return self._value is None


# ---------------------------------------------------------------------------
# Instruction base class
# ---------------------------------------------------------------------------

class Instruction(Value, ABC):
    """Abstract base for all SSA instructions.

    Every Instruction is also a Value (it defines the result of the
    instruction). Instructions that produce no result (e.g., StoreInst,
    BranchInst) use VoidType.
    """
    __slots__ = ("_operands", "parent", "metadata")

    def __init__(
        self,
        result_type: IRType,
        operands: Sequence[Value],
        name: str = "",
        metadata: InstructionMetadata | None = None,
    ) -> None:
        super().__init__(result_type, name)
        self._operands: list[Value] = list(operands)
        self.parent: Optional[BasicBlock] = None
        self.metadata: InstructionMetadata = metadata or InstructionMetadata()
        # Register as user of each operand
        for op in self._operands:
            op.users.add(self)

    @property
    def operands(self) -> tuple[Value, ...]:
        return tuple(self._operands)

    @property
    def num_operands(self) -> int:
        return len(self._operands)

    def get_operand(self, index: int) -> Value:
        return self._operands[index]

    def set_operand(self, index: int, val: Value) -> None:
        old = self._operands[index]
        old.users.discard(self)
        self._operands[index] = val
        val.users.add(self)

    def _replace_operand(self, old: Value, new: Value) -> None:
        for i, op in enumerate(self._operands):
            if op is old:
                self._operands[i] = new

    @abstractmethod
    def opcode_name(self) -> str:
        """Return a human-readable opcode string."""
        ...

    def validate(self) -> list[str]:
        """Validate this instruction, returning a list of error messages."""
        return []

    @abstractmethod
    def accept(self, visitor: "InstructionVisitor") -> None:
        """Accept an InstructionVisitor."""
        ...

    def clone(self) -> "Instruction":
        """Create a shallow copy with a new ID but same operands/metadata."""
        import copy
        inst = copy.copy(self)
        inst.id = _fresh_id()
        inst.users = set()
        inst.parent = None
        for op in inst._operands:
            op.users.add(inst)
        return inst

    def erase_from_parent(self) -> None:
        """Remove this instruction from its parent basic block."""
        if self.parent is not None:
            self.parent.remove(self)
        for op in self._operands:
            op.users.discard(self)

    def __repr__(self) -> str:
        ops = ", ".join(v.display_name for v in self._operands)
        return f"{self.display_name} = {self.opcode_name()}({ops})"


# ---------------------------------------------------------------------------
# Binary operations
# ---------------------------------------------------------------------------

class BinOpKind(Enum):
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    SDIV = "sdiv"
    UDIV = "udiv"
    SREM = "srem"
    UREM = "urem"
    FADD = "fadd"
    FSUB = "fsub"
    FMUL = "fmul"
    FDIV = "fdiv"
    FREM = "frem"
    SHL = "shl"
    LSHR = "lshr"
    ASHR = "ashr"
    AND = "and"
    OR = "or"
    XOR = "xor"


class BinaryOp(Instruction):
    """Binary arithmetic / bitwise / shift instruction."""
    __slots__ = ("op",)

    def __init__(
        self,
        op: BinOpKind,
        lhs: Value,
        rhs: Value,
        result_type: IRType | None = None,
        name: str = "",
        metadata: InstructionMetadata | None = None,
    ) -> None:
        rtype = result_type or lhs.type
        super().__init__(rtype, [lhs, rhs], name, metadata)
        self.op = op

    @property
    def lhs(self) -> Value:
        return self._operands[0]

    @property
    def rhs(self) -> Value:
        return self._operands[1]

    def opcode_name(self) -> str:
        return self.op.value

    def is_commutative(self) -> bool:
        return self.op in {
            BinOpKind.ADD, BinOpKind.MUL,
            BinOpKind.FADD, BinOpKind.FMUL,
            BinOpKind.AND, BinOpKind.OR, BinOpKind.XOR,
        }

    def is_shift(self) -> bool:
        return self.op in {BinOpKind.SHL, BinOpKind.LSHR, BinOpKind.ASHR}

    def is_floating(self) -> bool:
        return self.op in {
            BinOpKind.FADD, BinOpKind.FSUB,
            BinOpKind.FMUL, BinOpKind.FDIV, BinOpKind.FREM,
        }

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.is_floating():
            if not isinstance(self.lhs.type, FloatType):
                errors.append(f"{self.op.value}: lhs must be float, got {self.lhs.type}")
            if not isinstance(self.rhs.type, FloatType):
                errors.append(f"{self.op.value}: rhs must be float, got {self.rhs.type}")
        else:
            if not isinstance(self.lhs.type, IntType):
                errors.append(f"{self.op.value}: lhs must be int, got {self.lhs.type}")
            if not isinstance(self.rhs.type, IntType):
                errors.append(f"{self.op.value}: rhs must be int, got {self.rhs.type}")
        if self.lhs.type != self.rhs.type:
            errors.append(f"{self.op.value}: operand type mismatch: {self.lhs.type} vs {self.rhs.type}")
        return errors

    def accept(self, visitor: "InstructionVisitor") -> None:
        visitor.visit_binary_op(self)


# ---------------------------------------------------------------------------
# Unary operations
# ---------------------------------------------------------------------------

class UnaryOpKind(Enum):
    NEG = "neg"
    NOT = "not"
    BITWISE_NOT = "bitwise_not"
    FNEG = "fneg"


class UnaryOp(Instruction):
    """Unary operation."""
    __slots__ = ("op",)

    def __init__(
        self,
        op: UnaryOpKind,
        operand: Value,
        result_type: IRType | None = None,
        name: str = "",
        metadata: InstructionMetadata | None = None,
    ) -> None:
        rtype = result_type or operand.type
        super().__init__(rtype, [operand], name, metadata)
        self.op = op

    @property
    def operand(self) -> Value:
        return self._operands[0]

    def opcode_name(self) -> str:
        return self.op.value

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.op is UnaryOpKind.FNEG:
            if not isinstance(self.operand.type, FloatType):
                errors.append(f"fneg: operand must be float, got {self.operand.type}")
        elif self.op in (UnaryOpKind.NEG, UnaryOpKind.BITWISE_NOT):
            if not isinstance(self.operand.type, IntType):
                errors.append(f"{self.op.value}: operand must be int, got {self.operand.type}")
        elif self.op is UnaryOpKind.NOT:
            if not isinstance(self.operand.type, IntType):
                errors.append(f"not: operand must be int, got {self.operand.type}")
        return errors

    def accept(self, visitor: "InstructionVisitor") -> None:
        visitor.visit_unary_op(self)


# ---------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------

class CmpPredicate(Enum):
    EQ = "eq"
    NE = "ne"
    SLT = "slt"
    SLE = "sle"
    SGT = "sgt"
    SGE = "sge"
    ULT = "ult"
    ULE = "ule"
    UGT = "ugt"
    UGE = "uge"
    # Float-specific ordered comparisons
    OEQ = "oeq"
    ONE = "one"
    OLT = "olt"
    OLE = "ole"
    OGT = "ogt"
    OGE = "oge"
    # Float-specific unordered comparisons
    UEQ = "ueq"
    UNE = "une"
    # Ordering predicates
    ORD = "ord"   # both operands are not NaN
    UNO = "uno"   # at least one operand is NaN

    def is_signed(self) -> bool:
        return self in {CmpPredicate.SLT, CmpPredicate.SLE, CmpPredicate.SGT, CmpPredicate.SGE}

    def is_unsigned(self) -> bool:
        return self in {CmpPredicate.ULT, CmpPredicate.ULE, CmpPredicate.UGT, CmpPredicate.UGE}

    def is_float(self) -> bool:
        return self in {
            CmpPredicate.OEQ, CmpPredicate.ONE, CmpPredicate.OLT, CmpPredicate.OLE,
            CmpPredicate.OGT, CmpPredicate.OGE, CmpPredicate.UEQ, CmpPredicate.UNE,
            CmpPredicate.ORD, CmpPredicate.UNO,
        }

    def negate(self) -> "CmpPredicate":
        _neg = {
            CmpPredicate.EQ: CmpPredicate.NE, CmpPredicate.NE: CmpPredicate.EQ,
            CmpPredicate.SLT: CmpPredicate.SGE, CmpPredicate.SLE: CmpPredicate.SGT,
            CmpPredicate.SGT: CmpPredicate.SLE, CmpPredicate.SGE: CmpPredicate.SLT,
            CmpPredicate.ULT: CmpPredicate.UGE, CmpPredicate.ULE: CmpPredicate.UGT,
            CmpPredicate.UGT: CmpPredicate.ULE, CmpPredicate.UGE: CmpPredicate.ULT,
            CmpPredicate.OEQ: CmpPredicate.UNE, CmpPredicate.ONE: CmpPredicate.UEQ,
            CmpPredicate.OLT: CmpPredicate.UGE, CmpPredicate.OLE: CmpPredicate.UGT,
            CmpPredicate.OGT: CmpPredicate.ULE, CmpPredicate.OGE: CmpPredicate.ULT,
            CmpPredicate.UEQ: CmpPredicate.ONE, CmpPredicate.UNE: CmpPredicate.OEQ,
            CmpPredicate.ORD: CmpPredicate.UNO, CmpPredicate.UNO: CmpPredicate.ORD,
        }
        return _neg[self]

    def swap(self) -> "CmpPredicate":
        _swap = {
            CmpPredicate.EQ: CmpPredicate.EQ, CmpPredicate.NE: CmpPredicate.NE,
            CmpPredicate.SLT: CmpPredicate.SGT, CmpPredicate.SLE: CmpPredicate.SGE,
            CmpPredicate.SGT: CmpPredicate.SLT, CmpPredicate.SGE: CmpPredicate.SLE,
            CmpPredicate.ULT: CmpPredicate.UGT, CmpPredicate.ULE: CmpPredicate.UGE,
            CmpPredicate.UGT: CmpPredicate.ULT, CmpPredicate.UGE: CmpPredicate.ULE,
            CmpPredicate.OEQ: CmpPredicate.OEQ, CmpPredicate.ONE: CmpPredicate.ONE,
            CmpPredicate.OLT: CmpPredicate.OGT, CmpPredicate.OLE: CmpPredicate.OGE,
            CmpPredicate.OGT: CmpPredicate.OLT, CmpPredicate.OGE: CmpPredicate.OLE,
            CmpPredicate.UEQ: CmpPredicate.UEQ, CmpPredicate.UNE: CmpPredicate.UNE,
            CmpPredicate.ORD: CmpPredicate.ORD, CmpPredicate.UNO: CmpPredicate.UNO,
        }
        return _swap[self]


class CompareOp(Instruction):
    """Integer or floating-point comparison – result is always i1."""
    __slots__ = ("predicate",)

    def __init__(
        self,
        predicate: CmpPredicate,
        lhs: Value,
        rhs: Value,
        name: str = "",
        metadata: InstructionMetadata | None = None,
    ) -> None:
        result = IntType(1, Signedness.UNSIGNED)
        super().__init__(result, [lhs, rhs], name, metadata)
        self.predicate = predicate

    @property
    def lhs(self) -> Value:
        return self._operands[0]

    @property
    def rhs(self) -> Value:
        return self._operands[1]

    def opcode_name(self) -> str:
        return f"cmp.{self.predicate.value}"

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.lhs.type != self.rhs.type:
            errors.append(f"cmp: operand type mismatch: {self.lhs.type} vs {self.rhs.type}")
        if self.predicate.is_float() and not isinstance(self.lhs.type, FloatType):
            errors.append(f"cmp.{self.predicate.value}: requires float operands, got {self.lhs.type}")
        if self.predicate.is_signed() and isinstance(self.lhs.type, IntType) and self.lhs.type.is_unsigned:
            # Warning-level: signed comparison on unsigned operands
            pass
        return errors

    def accept(self, visitor: "InstructionVisitor") -> None:
        visitor.visit_compare_op(self)


# ---------------------------------------------------------------------------
# Memory instructions
# ---------------------------------------------------------------------------

class LoadInst(Instruction):
    """Load a value from memory."""
    __slots__ = ("volatile", "alignment")

    def __init__(
        self,
        address: Value,
        result_type: IRType,
        volatile: bool = False,
        alignment: int = 0,
        name: str = "",
        metadata: InstructionMetadata | None = None,
    ) -> None:
        super().__init__(result_type, [address], name, metadata)
        self.volatile = volatile
        self.alignment = alignment

    @property
    def address(self) -> Value:
        return self._operands[0]

    def opcode_name(self) -> str:
        return "load.volatile" if self.volatile else "load"

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not isinstance(self.address.type, PointerType):
            errors.append(f"load: address must be pointer, got {self.address.type}")
        return errors

    def accept(self, visitor: "InstructionVisitor") -> None:
        visitor.visit_load(self)


class StoreInst(Instruction):
    """Store a value to memory (no result)."""
    __slots__ = ("volatile", "alignment")

    def __init__(
        self,
        value: Value,
        address: Value,
        volatile: bool = False,
        alignment: int = 0,
        name: str = "",
        metadata: InstructionMetadata | None = None,
    ) -> None:
        super().__init__(VoidType(), [value, address], name, metadata)
        self.volatile = volatile
        self.alignment = alignment

    @property
    def value(self) -> Value:
        return self._operands[0]

    @property
    def address(self) -> Value:
        return self._operands[1]

    def opcode_name(self) -> str:
        return "store.volatile" if self.volatile else "store"

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not isinstance(self.address.type, PointerType):
            errors.append(f"store: address must be pointer, got {self.address.type}")
        return errors

    def accept(self, visitor: "InstructionVisitor") -> None:
        visitor.visit_store(self)


class AllocaInst(Instruction):
    """Stack allocation – produces a pointer to the allocated type."""
    __slots__ = ("alloc_type", "num_elements", "alignment")

    def __init__(
        self,
        alloc_type: IRType,
        num_elements: int = 1,
        alignment: int = 0,
        name: str = "",
        metadata: InstructionMetadata | None = None,
    ) -> None:
        ptr_type = PointerType(alloc_type)
        super().__init__(ptr_type, [], name, metadata)
        self.alloc_type = alloc_type
        self.num_elements = num_elements
        self.alignment = alignment

    def opcode_name(self) -> str:
        return "alloca"

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.num_elements < 1:
            errors.append(f"alloca: num_elements must be >= 1, got {self.num_elements}")
        if not self.alloc_type.is_sized():
            errors.append(f"alloca: type must be sized, got {self.alloc_type}")
        return errors

    def accept(self, visitor: "InstructionVisitor") -> None:
        visitor.visit_alloca(self)


class GetElementPtrInst(Instruction):
    """Compute the address of a sub-element of an aggregate or array."""
    __slots__ = ("source_element_type", "inbounds")

    def __init__(
        self,
        source_element_type: IRType,
        base: Value,
        indices: Sequence[Value],
        result_type: IRType | None = None,
        inbounds: bool = True,
        name: str = "",
        metadata: InstructionMetadata | None = None,
    ) -> None:
        rtype = result_type or _compute_gep_result_type(source_element_type, indices)
        super().__init__(rtype, [base, *indices], name, metadata)
        self.source_element_type = source_element_type
        self.inbounds = inbounds

    @property
    def base(self) -> Value:
        return self._operands[0]

    @property
    def indices(self) -> tuple[Value, ...]:
        return tuple(self._operands[1:])

    def opcode_name(self) -> str:
        return "getelementptr" + (".inbounds" if self.inbounds else "")

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not isinstance(self.base.type, PointerType):
            errors.append(f"gep: base must be pointer, got {self.base.type}")
        for i, idx in enumerate(self.indices):
            if not isinstance(idx.type, IntType):
                errors.append(f"gep: index {i} must be integer, got {idx.type}")
        return errors

    def accept(self, visitor: "InstructionVisitor") -> None:
        visitor.visit_gep(self)


def _compute_gep_result_type(source_type: IRType, indices: Sequence[Value]) -> IRType:
    """Walk through the indices to determine the result pointer type."""
    current = source_type
    for i, idx in enumerate(indices):
        if i == 0:
            # First index dereferences the pointer into the source type
            continue
        if isinstance(current, ArrayType):
            current = current.element
        elif isinstance(current, StructType):
            # For struct GEP, the index must be a constant
            if isinstance(idx, Constant) and isinstance(idx.value, int):
                field_idx = idx.value
                if 0 <= field_idx < len(current.fields):
                    current = current.fields[field_idx].type
                else:
                    break
            else:
                break
        else:
            break
    return PointerType(current)


# ---------------------------------------------------------------------------
# Cast instructions
# ---------------------------------------------------------------------------

class CastKind(Enum):
    TRUNC = "trunc"
    ZEXT = "zext"
    SEXT = "sext"
    FPTRUNC = "fptrunc"
    FPEXT = "fpext"
    FPTOSI = "fptosi"
    FPTOUI = "fptoui"
    SITOFP = "sitofp"
    UITOFP = "uitofp"
    BITCAST = "bitcast"
    PTRTOINT = "ptrtoint"
    INTTOPTR = "inttoptr"


class CastInst(Instruction):
    """Type cast / conversion instruction."""
    __slots__ = ("cast_kind",)

    def __init__(
        self,
        cast_kind: CastKind,
        operand: Value,
        dest_type: IRType,
        name: str = "",
        metadata: InstructionMetadata | None = None,
    ) -> None:
        super().__init__(dest_type, [operand], name, metadata)
        self.cast_kind = cast_kind

    @property
    def operand(self) -> Value:
        return self._operands[0]

    @property
    def src_type(self) -> IRType:
        return self.operand.type

    @property
    def dest_type(self) -> IRType:
        return self.type

    def opcode_name(self) -> str:
        return self.cast_kind.value

    def validate(self) -> list[str]:
        errors: list[str] = []
        src, dst = self.src_type, self.dest_type
        match self.cast_kind:
            case CastKind.TRUNC:
                if not (isinstance(src, IntType) and isinstance(dst, IntType)):
                    errors.append("trunc: both types must be integer")
                elif src.width <= dst.width:
                    errors.append(f"trunc: source width {src.width} must be > dest width {dst.width}")
            case CastKind.ZEXT | CastKind.SEXT:
                if not (isinstance(src, IntType) and isinstance(dst, IntType)):
                    errors.append(f"{self.cast_kind.value}: both types must be integer")
                elif src.width >= dst.width:
                    errors.append(f"{self.cast_kind.value}: source width must be < dest width")
            case CastKind.FPTRUNC:
                if not (isinstance(src, FloatType) and isinstance(dst, FloatType)):
                    errors.append("fptrunc: both types must be float")
                elif src.width <= dst.width:
                    errors.append("fptrunc: source must be wider than dest")
            case CastKind.FPEXT:
                if not (isinstance(src, FloatType) and isinstance(dst, FloatType)):
                    errors.append("fpext: both types must be float")
                elif src.width >= dst.width:
                    errors.append("fpext: source must be narrower than dest")
            case CastKind.FPTOSI | CastKind.FPTOUI:
                if not isinstance(src, FloatType):
                    errors.append(f"{self.cast_kind.value}: source must be float")
                if not isinstance(dst, IntType):
                    errors.append(f"{self.cast_kind.value}: dest must be integer")
            case CastKind.SITOFP | CastKind.UITOFP:
                if not isinstance(src, IntType):
                    errors.append(f"{self.cast_kind.value}: source must be integer")
                if not isinstance(dst, FloatType):
                    errors.append(f"{self.cast_kind.value}: dest must be float")
            case CastKind.BITCAST:
                if src.is_sized() and dst.is_sized():
                    if src.size_bits() != dst.size_bits():
                        errors.append(f"bitcast: size mismatch {src.size_bits()} vs {dst.size_bits()}")
            case CastKind.PTRTOINT:
                if not isinstance(src, PointerType):
                    errors.append("ptrtoint: source must be pointer")
                if not isinstance(dst, IntType):
                    errors.append("ptrtoint: dest must be integer")
            case CastKind.INTTOPTR:
                if not isinstance(src, IntType):
                    errors.append("inttoptr: source must be integer")
                if not isinstance(dst, PointerType):
                    errors.append("inttoptr: dest must be pointer")
        return errors

    def accept(self, visitor: "InstructionVisitor") -> None:
        visitor.visit_cast(self)


# ---------------------------------------------------------------------------
# Control flow instructions
# ---------------------------------------------------------------------------

class CallInst(Instruction):
    """Function call."""
    __slots__ = ("callee_name", "is_tail_call")

    def __init__(
        self,
        callee: Value,
        args: Sequence[Value],
        return_type: IRType,
        callee_name: str = "",
        is_tail_call: bool = False,
        name: str = "",
        metadata: InstructionMetadata | None = None,
    ) -> None:
        super().__init__(return_type, [callee, *args], name, metadata)
        self.callee_name = callee_name
        self.is_tail_call = is_tail_call

    @property
    def callee(self) -> Value:
        return self._operands[0]

    @property
    def args(self) -> tuple[Value, ...]:
        return tuple(self._operands[1:])

    def opcode_name(self) -> str:
        return "tail call" if self.is_tail_call else "call"

    def validate(self) -> list[str]:
        errors: list[str] = []
        callee_ty = self.callee.type
        if isinstance(callee_ty, PointerType):
            callee_ty = callee_ty.pointee
        if isinstance(callee_ty, FunctionType):
            expected = len(callee_ty.param_types)
            got = len(self.args)
            if not callee_ty.is_variadic and got != expected:
                errors.append(f"call: expected {expected} args, got {got}")
            elif got < expected:
                errors.append(f"call: too few args: expected >= {expected}, got {got}")
        return errors

    def accept(self, visitor: "InstructionVisitor") -> None:
        visitor.visit_call(self)


class ReturnInst(Instruction):
    """Return from function."""

    def __init__(
        self,
        value: Value | None = None,
        name: str = "",
        metadata: InstructionMetadata | None = None,
    ) -> None:
        ops = [value] if value is not None else []
        rtype = value.type if value is not None else VoidType()
        super().__init__(rtype, ops, name, metadata)

    @property
    def return_value(self) -> Value | None:
        return self._operands[0] if self._operands else None

    @property
    def is_void_return(self) -> bool:
        return len(self._operands) == 0

    def opcode_name(self) -> str:
        return "ret"

    def validate(self) -> list[str]:
        return []

    def accept(self, visitor: "InstructionVisitor") -> None:
        visitor.visit_return(self)


class BranchInst(Instruction):
    """Conditional or unconditional branch."""
    __slots__ = ("_true_target", "_false_target")

    def __init__(
        self,
        target: "BasicBlock",
        condition: Value | None = None,
        false_target: "BasicBlock | None" = None,
        name: str = "",
        metadata: InstructionMetadata | None = None,
    ) -> None:
        ops: list[Value] = []
        if condition is not None:
            ops.append(condition)
        super().__init__(VoidType(), ops, name, metadata)
        self._true_target = target
        self._false_target = false_target

    @property
    def is_conditional(self) -> bool:
        return self._false_target is not None

    @property
    def condition(self) -> Value | None:
        return self._operands[0] if self._operands else None

    @property
    def true_target(self) -> "BasicBlock":
        return self._true_target

    @property
    def false_target(self) -> "BasicBlock | None":
        return self._false_target

    @property
    def successors(self) -> list["BasicBlock"]:
        result = [self._true_target]
        if self._false_target is not None:
            result.append(self._false_target)
        return result

    def opcode_name(self) -> str:
        return "br.cond" if self.is_conditional else "br"

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.is_conditional:
            if not self._operands:
                errors.append("br.cond: missing condition")
            elif not (isinstance(self._operands[0].type, IntType) and self._operands[0].type.width == 1):
                errors.append(f"br.cond: condition must be i1, got {self._operands[0].type}")
        return errors

    def accept(self, visitor: "InstructionVisitor") -> None:
        visitor.visit_branch(self)


class SwitchInst(Instruction):
    """Multi-way branch (switch statement)."""
    __slots__ = ("_cases", "_default_target")

    def __init__(
        self,
        condition: Value,
        default_target: "BasicBlock",
        cases: Sequence[tuple[Constant, "BasicBlock"]],
        name: str = "",
        metadata: InstructionMetadata | None = None,
    ) -> None:
        case_vals = [c for c, _ in cases]
        super().__init__(VoidType(), [condition, *case_vals], name, metadata)
        self._cases = list(cases)
        self._default_target = default_target

    @property
    def condition(self) -> Value:
        return self._operands[0]

    @property
    def default_target(self) -> "BasicBlock":
        return self._default_target

    @property
    def cases(self) -> list[tuple[Constant, "BasicBlock"]]:
        return list(self._cases)

    @property
    def successors(self) -> list["BasicBlock"]:
        targets = [self._default_target]
        targets.extend(bb for _, bb in self._cases)
        return targets

    def opcode_name(self) -> str:
        return "switch"

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not isinstance(self.condition.type, IntType):
            errors.append(f"switch: condition must be integer, got {self.condition.type}")
        for val, _ in self._cases:
            if val.type != self.condition.type:
                errors.append(f"switch: case value type {val.type} != condition type {self.condition.type}")
        return errors

    def accept(self, visitor: "InstructionVisitor") -> None:
        visitor.visit_switch(self)


# ---------------------------------------------------------------------------
# SSA-specific instructions
# ---------------------------------------------------------------------------

class PhiInst(Instruction):
    """Phi node – merges values from different predecessor blocks."""
    __slots__ = ("_incoming",)

    def __init__(
        self,
        result_type: IRType,
        incoming: Sequence[tuple[Value, "BasicBlock"]],
        name: str = "",
        metadata: InstructionMetadata | None = None,
    ) -> None:
        values = [v for v, _ in incoming]
        super().__init__(result_type, values, name, metadata)
        self._incoming = list(incoming)

    @property
    def incoming(self) -> list[tuple[Value, "BasicBlock"]]:
        return list(self._incoming)

    @property
    def incoming_values(self) -> list[Value]:
        return [v for v, _ in self._incoming]

    @property
    def incoming_blocks(self) -> list["BasicBlock"]:
        return [b for _, b in self._incoming]

    def add_incoming(self, value: Value, block: "BasicBlock") -> None:
        self._incoming.append((value, block))
        self._operands.append(value)
        value.users.add(self)

    def remove_incoming_block(self, block: "BasicBlock") -> None:
        new_incoming = [(v, b) for v, b in self._incoming if b is not block]
        removed = [(v, b) for v, b in self._incoming if b is block]
        for v, _ in removed:
            v.users.discard(self)
        self._incoming = new_incoming
        self._operands = [v for v, _ in new_incoming]

    def get_value_for_block(self, block: "BasicBlock") -> Value | None:
        for v, b in self._incoming:
            if b is block:
                return v
        return None

    def opcode_name(self) -> str:
        return "phi"

    def validate(self) -> list[str]:
        errors: list[str] = []
        for v, b in self._incoming:
            if v.type != self.type:
                errors.append(f"phi: incoming value type {v.type} != phi type {self.type}")
        if self.parent is not None:
            pred_set = set(self.parent.predecessors)
            phi_blocks = set(self.incoming_blocks)
            if pred_set != phi_blocks:
                missing = pred_set - phi_blocks
                extra = phi_blocks - pred_set
                if missing:
                    errors.append(f"phi: missing predecessors: {[b.name for b in missing]}")
                if extra:
                    errors.append(f"phi: extra blocks: {[b.name for b in extra]}")
        return errors

    def accept(self, visitor: "InstructionVisitor") -> None:
        visitor.visit_phi(self)


class SelectInst(Instruction):
    """Conditional select (ternary) – like C's ?: operator."""

    def __init__(
        self,
        condition: Value,
        true_val: Value,
        false_val: Value,
        name: str = "",
        metadata: InstructionMetadata | None = None,
    ) -> None:
        super().__init__(true_val.type, [condition, true_val, false_val], name, metadata)

    @property
    def condition(self) -> Value:
        return self._operands[0]

    @property
    def true_value(self) -> Value:
        return self._operands[1]

    @property
    def false_value(self) -> Value:
        return self._operands[2]

    def opcode_name(self) -> str:
        return "select"

    def validate(self) -> list[str]:
        errors: list[str] = []
        cond_ty = self.condition.type
        if not (isinstance(cond_ty, IntType) and cond_ty.width == 1):
            errors.append(f"select: condition must be i1, got {cond_ty}")
        if self.true_value.type != self.false_value.type:
            errors.append(f"select: value type mismatch: {self.true_value.type} vs {self.false_value.type}")
        return errors

    def accept(self, visitor: "InstructionVisitor") -> None:
        visitor.visit_select(self)


# ---------------------------------------------------------------------------
# Aggregate operations
# ---------------------------------------------------------------------------

class ExtractValueInst(Instruction):
    """Extract a field from an aggregate value."""
    __slots__ = ("indices",)

    def __init__(
        self,
        aggregate: Value,
        indices: tuple[int, ...],
        result_type: IRType,
        name: str = "",
        metadata: InstructionMetadata | None = None,
    ) -> None:
        super().__init__(result_type, [aggregate], name, metadata)
        self.indices = indices

    @property
    def aggregate(self) -> Value:
        return self._operands[0]

    def opcode_name(self) -> str:
        return "extractvalue"

    def validate(self) -> list[str]:
        errors: list[str] = []
        ty = self.aggregate.type
        for i, idx in enumerate(self.indices):
            if isinstance(ty, StructType):
                if idx < 0 or idx >= len(ty.fields):
                    errors.append(f"extractvalue: index {idx} out of range for struct with {len(ty.fields)} fields")
                    break
                ty = ty.fields[idx].type
            elif isinstance(ty, ArrayType):
                ty = ty.element
            else:
                errors.append(f"extractvalue: cannot index into {ty}")
                break
        return errors

    def accept(self, visitor: "InstructionVisitor") -> None:
        visitor.visit_extract_value(self)


class InsertValueInst(Instruction):
    """Insert a value into an aggregate."""
    __slots__ = ("indices",)

    def __init__(
        self,
        aggregate: Value,
        value: Value,
        indices: tuple[int, ...],
        name: str = "",
        metadata: InstructionMetadata | None = None,
    ) -> None:
        super().__init__(aggregate.type, [aggregate, value], name, metadata)
        self.indices = indices

    @property
    def aggregate(self) -> Value:
        return self._operands[0]

    @property
    def inserted_value(self) -> Value:
        return self._operands[1]

    def opcode_name(self) -> str:
        return "insertvalue"

    def validate(self) -> list[str]:
        errors: list[str] = []
        ty = self.aggregate.type
        for idx in self.indices:
            if isinstance(ty, StructType):
                if idx < 0 or idx >= len(ty.fields):
                    errors.append(f"insertvalue: index {idx} out of range")
                    break
                ty = ty.fields[idx].type
            elif isinstance(ty, ArrayType):
                ty = ty.element
            else:
                errors.append(f"insertvalue: cannot index into {ty}")
                break
        return errors

    def accept(self, visitor: "InstructionVisitor") -> None:
        visitor.visit_insert_value(self)


# ---------------------------------------------------------------------------
# Memory intrinsics
# ---------------------------------------------------------------------------

class MemcpyInst(Instruction):
    """Memory copy intrinsic (like C memcpy / memmove)."""
    __slots__ = ("is_volatile",)

    def __init__(
        self,
        dest: Value,
        src: Value,
        length: Value,
        is_volatile: bool = False,
        name: str = "",
        metadata: InstructionMetadata | None = None,
    ) -> None:
        super().__init__(VoidType(), [dest, src, length], name, metadata)
        self.is_volatile = is_volatile

    @property
    def dest(self) -> Value:
        return self._operands[0]

    @property
    def src(self) -> Value:
        return self._operands[1]

    @property
    def length(self) -> Value:
        return self._operands[2]

    def opcode_name(self) -> str:
        return "memcpy.volatile" if self.is_volatile else "memcpy"

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not isinstance(self.dest.type, PointerType):
            errors.append(f"memcpy: dest must be pointer, got {self.dest.type}")
        if not isinstance(self.src.type, PointerType):
            errors.append(f"memcpy: src must be pointer, got {self.src.type}")
        if not isinstance(self.length.type, IntType):
            errors.append(f"memcpy: length must be integer, got {self.length.type}")
        return errors

    def accept(self, visitor: "InstructionVisitor") -> None:
        visitor.visit_memcpy(self)


class MemsetInst(Instruction):
    """Memory set intrinsic (like C memset)."""
    __slots__ = ("is_volatile",)

    def __init__(
        self,
        dest: Value,
        value: Value,
        length: Value,
        is_volatile: bool = False,
        name: str = "",
        metadata: InstructionMetadata | None = None,
    ) -> None:
        super().__init__(VoidType(), [dest, value, length], name, metadata)
        self.is_volatile = is_volatile

    @property
    def dest(self) -> Value:
        return self._operands[0]

    @property
    def fill_value(self) -> Value:
        return self._operands[1]

    @property
    def length(self) -> Value:
        return self._operands[2]

    def opcode_name(self) -> str:
        return "memset.volatile" if self.is_volatile else "memset"

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not isinstance(self.dest.type, PointerType):
            errors.append(f"memset: dest must be pointer, got {self.dest.type}")
        if not isinstance(self.fill_value.type, IntType):
            errors.append(f"memset: value must be integer, got {self.fill_value.type}")
        if not isinstance(self.length.type, IntType):
            errors.append(f"memset: length must be integer, got {self.length.type}")
        return errors

    def accept(self, visitor: "InstructionVisitor") -> None:
        visitor.visit_memset(self)


# ---------------------------------------------------------------------------
# Atomic / concurrency instructions
# ---------------------------------------------------------------------------

class AtomicOrdering(Enum):
    NOT_ATOMIC = auto()
    UNORDERED = auto()
    MONOTONIC = auto()
    ACQUIRE = auto()
    RELEASE = auto()
    ACQ_REL = auto()
    SEQ_CST = auto()


class FenceInst(Instruction):
    """Memory fence (barrier)."""
    __slots__ = ("ordering",)

    def __init__(
        self,
        ordering: AtomicOrdering = AtomicOrdering.SEQ_CST,
        name: str = "",
        metadata: InstructionMetadata | None = None,
    ) -> None:
        super().__init__(VoidType(), [], name, metadata)
        self.ordering = ordering

    def opcode_name(self) -> str:
        return f"fence.{self.ordering.name.lower()}"

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.ordering in (AtomicOrdering.NOT_ATOMIC, AtomicOrdering.UNORDERED):
            errors.append(f"fence: invalid ordering {self.ordering.name}")
        return errors

    def accept(self, visitor: "InstructionVisitor") -> None:
        visitor.visit_fence(self)


class AtomicRMWOp(Enum):
    XCHG = "xchg"
    ADD = "add"
    SUB = "sub"
    AND = "and"
    NAND = "nand"
    OR = "or"
    XOR = "xor"
    MAX = "max"
    MIN = "min"
    UMAX = "umax"
    UMIN = "umin"


class AtomicRMWInst(Instruction):
    """Atomic read-modify-write instruction."""
    __slots__ = ("rmw_op", "ordering")

    def __init__(
        self,
        rmw_op: AtomicRMWOp,
        address: Value,
        value: Value,
        ordering: AtomicOrdering = AtomicOrdering.SEQ_CST,
        name: str = "",
        metadata: InstructionMetadata | None = None,
    ) -> None:
        super().__init__(value.type, [address, value], name, metadata)
        self.rmw_op = rmw_op
        self.ordering = ordering

    @property
    def address(self) -> Value:
        return self._operands[0]

    @property
    def value(self) -> Value:
        return self._operands[1]

    def opcode_name(self) -> str:
        return f"atomicrmw.{self.rmw_op.value}"

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not isinstance(self.address.type, PointerType):
            errors.append(f"atomicrmw: address must be pointer, got {self.address.type}")
        if not isinstance(self.value.type, IntType):
            errors.append(f"atomicrmw: value must be integer, got {self.value.type}")
        return errors

    def accept(self, visitor: "InstructionVisitor") -> None:
        visitor.visit_atomic_rmw(self)


class AtomicCmpXchgInst(Instruction):
    """Atomic compare-and-exchange instruction.

    Returns a struct {value_type, i1} where the i1 indicates success.
    """
    __slots__ = ("success_ordering", "failure_ordering")

    def __init__(
        self,
        address: Value,
        expected: Value,
        desired: Value,
        success_ordering: AtomicOrdering = AtomicOrdering.SEQ_CST,
        failure_ordering: AtomicOrdering = AtomicOrdering.SEQ_CST,
        name: str = "",
        metadata: InstructionMetadata | None = None,
    ) -> None:
        from .types import StructField
        result_type = StructType(
            None,
            (
                StructField("value", expected.type),
                StructField("success", IntType(1, Signedness.UNSIGNED)),
            ),
        )
        super().__init__(result_type, [address, expected, desired], name, metadata)
        self.success_ordering = success_ordering
        self.failure_ordering = failure_ordering

    @property
    def address(self) -> Value:
        return self._operands[0]

    @property
    def expected(self) -> Value:
        return self._operands[1]

    @property
    def desired(self) -> Value:
        return self._operands[2]

    def opcode_name(self) -> str:
        return "cmpxchg"

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not isinstance(self.address.type, PointerType):
            errors.append(f"cmpxchg: address must be pointer, got {self.address.type}")
        if self.expected.type != self.desired.type:
            errors.append(f"cmpxchg: expected/desired type mismatch: {self.expected.type} vs {self.desired.type}")
        return errors

    def accept(self, visitor: "InstructionVisitor") -> None:
        visitor.visit_atomic_cmpxchg(self)


# ---------------------------------------------------------------------------
# Terminator helpers
# ---------------------------------------------------------------------------

def is_terminator(inst: Instruction) -> bool:
    """Return True if *inst* is a terminator instruction."""
    return isinstance(inst, (ReturnInst, BranchInst, SwitchInst))


def get_successors(inst: Instruction) -> list["BasicBlock"]:
    """Return successor blocks of a terminator instruction."""
    if isinstance(inst, BranchInst):
        return inst.successors
    if isinstance(inst, SwitchInst):
        return inst.successors
    return []


# ---------------------------------------------------------------------------
# Instruction visitor
# ---------------------------------------------------------------------------

class InstructionVisitor(ABC):
    """Visitor pattern for IR instructions. Override only the methods you need."""

    def visit_binary_op(self, inst: BinaryOp) -> None: ...
    def visit_unary_op(self, inst: UnaryOp) -> None: ...
    def visit_compare_op(self, inst: CompareOp) -> None: ...
    def visit_load(self, inst: LoadInst) -> None: ...
    def visit_store(self, inst: StoreInst) -> None: ...
    def visit_alloca(self, inst: AllocaInst) -> None: ...
    def visit_gep(self, inst: GetElementPtrInst) -> None: ...
    def visit_cast(self, inst: CastInst) -> None: ...
    def visit_call(self, inst: CallInst) -> None: ...
    def visit_return(self, inst: ReturnInst) -> None: ...
    def visit_branch(self, inst: BranchInst) -> None: ...
    def visit_switch(self, inst: SwitchInst) -> None: ...
    def visit_phi(self, inst: PhiInst) -> None: ...
    def visit_select(self, inst: SelectInst) -> None: ...
    def visit_extract_value(self, inst: ExtractValueInst) -> None: ...
    def visit_insert_value(self, inst: InsertValueInst) -> None: ...
    def visit_memcpy(self, inst: MemcpyInst) -> None: ...
    def visit_memset(self, inst: MemsetInst) -> None: ...
    def visit_fence(self, inst: FenceInst) -> None: ...
    def visit_atomic_rmw(self, inst: AtomicRMWInst) -> None: ...
    def visit_atomic_cmpxchg(self, inst: AtomicCmpXchgInst) -> None: ...

    def visit(self, inst: Instruction) -> None:
        """Generic dispatch via the visitor pattern."""
        inst.accept(self)


class InstructionCounter(InstructionVisitor):
    """Counts instructions by category."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}

    def _inc(self, name: str) -> None:
        self.counts[name] = self.counts.get(name, 0) + 1

    def visit_binary_op(self, inst: BinaryOp) -> None:
        self._inc("binary")

    def visit_unary_op(self, inst: UnaryOp) -> None:
        self._inc("unary")

    def visit_compare_op(self, inst: CompareOp) -> None:
        self._inc("compare")

    def visit_load(self, inst: LoadInst) -> None:
        self._inc("load")

    def visit_store(self, inst: StoreInst) -> None:
        self._inc("store")

    def visit_alloca(self, inst: AllocaInst) -> None:
        self._inc("alloca")

    def visit_gep(self, inst: GetElementPtrInst) -> None:
        self._inc("gep")

    def visit_cast(self, inst: CastInst) -> None:
        self._inc("cast")

    def visit_call(self, inst: CallInst) -> None:
        self._inc("call")

    def visit_return(self, inst: ReturnInst) -> None:
        self._inc("return")

    def visit_branch(self, inst: BranchInst) -> None:
        self._inc("branch")

    def visit_switch(self, inst: SwitchInst) -> None:
        self._inc("switch")

    def visit_phi(self, inst: PhiInst) -> None:
        self._inc("phi")

    def visit_select(self, inst: SelectInst) -> None:
        self._inc("select")

    def visit_extract_value(self, inst: ExtractValueInst) -> None:
        self._inc("extractvalue")

    def visit_insert_value(self, inst: InsertValueInst) -> None:
        self._inc("insertvalue")

    def visit_memcpy(self, inst: MemcpyInst) -> None:
        self._inc("memcpy")

    def visit_memset(self, inst: MemsetInst) -> None:
        self._inc("memset")

    def visit_fence(self, inst: FenceInst) -> None:
        self._inc("fence")

    def visit_atomic_rmw(self, inst: AtomicRMWInst) -> None:
        self._inc("atomicrmw")

    def visit_atomic_cmpxchg(self, inst: AtomicCmpXchgInst) -> None:
        self._inc("cmpxchg")

    @property
    def total(self) -> int:
        return sum(self.counts.values())
