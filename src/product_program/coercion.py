"""
Coercion point insertion for product programs.

At every aligned instruction pair, checks the semantic divergence table and
inserts coercion assertions where C and Rust semantics differ. Generates
SMT assertions for each coercion point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict, Tuple, Any, Callable

from ..ir.types import (
    IRType, IntType, FloatType, PointerType, VoidType,
    Signedness, FloatKind, OverflowBehavior, Language,
)
from ..ir.instructions import (
    Instruction, BinaryOp, UnaryOp, CompareOp, CastInst,
    LoadInst, StoreInst, CallInst, ReturnInst, BranchInst,
    PhiInst, SelectInst, Value, Constant, BinOpKind, CmpPredicate, CastKind,
)
from ..ir.basic_block import BasicBlock
from ..ir.function import Function
from ..semantics.divergence_table import (
    DivergenceClass, DivergenceType, DivergenceEntry,
    DivergenceTable, CSemantics, RustSemantics,
)
from ..semantics.semantic_config import SemanticConfig, OverflowMode
from .alignment import InstructionAlignment, AlignmentKind, AlignmentResult


class CoercionKind(Enum):
    """Category of coercion needed."""
    OVERFLOW_CHECK = auto()
    DIVISION_CHECK = auto()
    SHIFT_CHECK = auto()
    FLOAT_PRECISION = auto()
    FLOAT_TO_INT = auto()
    POINTER_PROVENANCE = auto()
    NULL_CHECK = auto()
    BOUNDS_CHECK = auto()
    ERROR_HANDLING = auto()
    TYPE_WIDTH = auto()
    SIGNEDNESS = auto()
    RETURN_COERCION = auto()
    CAST_COERCION = auto()
    CALLING_CONVENTION = auto()


class AssertionStrength(Enum):
    """Strength of a coercion assertion."""
    HARD = auto()      # Must hold for equivalence
    SOFT = auto()      # Advisory, may be relaxed
    ASSUME = auto()    # Assumed precondition


@dataclass
class CoercionAssertion:
    """An SMT assertion for a coercion point."""
    smt_expression: str
    description: str
    strength: AssertionStrength = AssertionStrength.HARD
    variables: List[str] = field(default_factory=list)

    def negate(self) -> CoercionAssertion:
        """Return negated assertion (for counterexample search)."""
        return CoercionAssertion(
            smt_expression=f"(not {self.smt_expression})",
            description=f"NOT({self.description})",
            strength=self.strength,
            variables=self.variables,
        )

    def __repr__(self) -> str:
        return f"Assertion({self.description}, {self.strength.name})"


@dataclass
class CoercionPoint:
    """A point where C and Rust semantics diverge, requiring a coercion assertion."""
    kind: CoercionKind
    left_instruction: Optional[Instruction]
    right_instruction: Optional[Instruction]
    divergence_class: DivergenceClass
    c_semantics: CSemantics
    rust_semantics: RustSemantics
    assertions: List[CoercionAssertion] = field(default_factory=list)
    operation: str = ""
    bit_width: int = 32
    source_location: str = ""
    notes: List[str] = field(default_factory=list)

    @property
    def is_critical(self) -> bool:
        return any(a.strength == AssertionStrength.HARD for a in self.assertions)

    @property
    def num_assertions(self) -> int:
        return len(self.assertions)

    def summary(self) -> str:
        inst_l = self.left_instruction.name if self.left_instruction else "---"
        inst_r = self.right_instruction.name if self.right_instruction else "---"
        return (
            f"CoercionPoint({self.kind.name}, {inst_l}<->{inst_r}, "
            f"{self.divergence_class.name}, {len(self.assertions)} assertions)"
        )

    def __repr__(self) -> str:
        return self.summary()


# ---------------------------------------------------------------------------
# SMT expression helpers
# ---------------------------------------------------------------------------

def _var(name, width):
    return f"((_ extract {width-1} 0) {name})"

def _bvadd(a, b): return f"(bvadd {a} {b})"
def _bvsub(a, b): return f"(bvsub {a} {b})"
def _bvmul(a, b): return f"(bvmul {a} {b})"
def _bvsdiv(a, b): return f"(bvsdiv {a} {b})"
def _bvudiv(a, b): return f"(bvudiv {a} {b})"
def _bvshl(a, b): return f"(bvshl {a} {b})"
def _bvashr(a, b): return f"(bvashr {a} {b})"
def _bvlshr(a, b): return f"(bvlshr {a} {b})"
def _bvnot(a): return f"(bvnot {a})"
def _bvneg(a): return f"(bvneg {a})"
def _bvslt(a, b): return f"(bvslt {a} {b})"
def _bvsle(a, b): return f"(bvsle {a} {b})"
def _bvsgt(a, b): return f"(bvsgt {a} {b})"
def _bvsge(a, b): return f"(bvsge {a} {b})"
def _bvult(a, b): return f"(bvult {a} {b})"
def _eq(a, b): return f"(= {a} {b})"
def _and(*args): return f"(and {' '.join(args)})" if len(args) > 1 else args[0]
def _or(*args): return f"(or {' '.join(args)})" if len(args) > 1 else args[0]
def _not(a): return f"(not {a})"
def _implies(a, b): return f"(=> {a} {b})"
def _ite(c, t, f): return f"(ite {c} {t} {f})"
def _bvconst(val, width): return f"(_ bv{val} {width})"
def _sign_max(width): return _bvconst((1 << (width - 1)) - 1, width)
def _sign_min(width): return _bvconst(1 << (width - 1), width)
def _unsigned_max(width): return _bvconst((1 << width) - 1, width)


# ---------------------------------------------------------------------------
# Divergence-specific assertion generators
# ---------------------------------------------------------------------------

def _gen_signed_overflow_assertions(
    left: Instruction, right: Instruction, width: int, var_prefix: str,
) -> List[CoercionAssertion]:
    """Generate assertions for signed integer overflow divergence."""
    lhs = f"{var_prefix}_lhs"
    rhs = f"{var_prefix}_rhs"
    c_result = f"{var_prefix}_c_result"
    rust_result = f"{var_prefix}_rust_result"
    variables = [lhs, rhs, c_result, rust_result]

    if isinstance(left, BinaryOp):
        op_name = left.op.name
    else:
        op_name = "ADD"

    if op_name == "ADD":
        result_expr = _bvadd(lhs, rhs)
    elif op_name == "SUB":
        result_expr = _bvsub(lhs, rhs)
    elif op_name == "MUL":
        result_expr = _bvmul(lhs, rhs)
    else:
        result_expr = _bvadd(lhs, rhs)

    no_overflow = _and(
        _bvsle(_sign_min(width), result_expr),
        _bvsle(result_expr, _sign_max(width)),
    )

    return [
        CoercionAssertion(
            smt_expression=no_overflow,
            description=f"No signed overflow on {op_name.lower()} (i{width})",
            strength=AssertionStrength.ASSUME,
            variables=variables,
        ),
        CoercionAssertion(
            smt_expression=_implies(no_overflow, _eq(c_result, rust_result)),
            description=f"Signed {op_name.lower()} equivalence when no overflow (i{width})",
            strength=AssertionStrength.HARD,
            variables=variables,
        ),
    ]


def _gen_unsigned_wrap_assertions(
    left: Instruction, right: Instruction, width: int, var_prefix: str,
) -> List[CoercionAssertion]:
    """Generate assertions for unsigned wrapping behavior divergence."""
    lhs = f"{var_prefix}_lhs"
    rhs = f"{var_prefix}_rhs"
    c_result = f"{var_prefix}_c_result"
    rust_result = f"{var_prefix}_rust_result"
    variables = [lhs, rhs, c_result, rust_result]

    if isinstance(left, BinaryOp):
        op_name = left.op.name
    else:
        op_name = "ADD"

    if op_name == "ADD":
        result_expr = _bvadd(lhs, rhs)
    elif op_name == "SUB":
        result_expr = _bvsub(lhs, rhs)
    elif op_name == "MUL":
        result_expr = _bvmul(lhs, rhs)
    else:
        result_expr = _bvadd(lhs, rhs)

    return [
        CoercionAssertion(
            smt_expression=_eq(c_result, rust_result),
            description=f"Unsigned {op_name.lower()} wrap equivalence (i{width})",
            strength=AssertionStrength.HARD,
            variables=variables,
        ),
    ]


def _gen_division_assertions(
    left: Instruction, right: Instruction, width: int, var_prefix: str,
) -> List[CoercionAssertion]:
    """Generate assertions for division-by-zero and signed division overflow."""
    lhs = f"{var_prefix}_lhs"
    rhs = f"{var_prefix}_rhs"
    c_result = f"{var_prefix}_c_result"
    rust_result = f"{var_prefix}_rust_result"
    variables = [lhs, rhs, c_result, rust_result]

    zero = _bvconst(0, width)
    assertions = [
        CoercionAssertion(
            smt_expression=_not(_eq(rhs, zero)),
            description=f"Divisor is non-zero (i{width})",
            strength=AssertionStrength.ASSUME,
            variables=variables,
        ),
    ]

    if isinstance(left, BinaryOp) and left.op.name in ("SDIV", "SREM"):
        neg_one = _unsigned_max(width)
        int_min_div = _and(
            _eq(lhs, _sign_min(width)),
            _eq(rhs, neg_one),
        )
        assertions.append(CoercionAssertion(
            smt_expression=_not(int_min_div),
            description=f"No INT_MIN / -1 overflow (i{width})",
            strength=AssertionStrength.ASSUME,
            variables=variables,
        ))

    assertions.append(CoercionAssertion(
        smt_expression=_eq(c_result, rust_result),
        description=f"Division result equivalence (i{width})",
        strength=AssertionStrength.HARD,
        variables=variables,
    ))

    return assertions


def _gen_shift_assertions(
    left: Instruction, right: Instruction, width: int, var_prefix: str,
) -> List[CoercionAssertion]:
    """Generate assertions for shift amount validity."""
    lhs = f"{var_prefix}_lhs"
    rhs = f"{var_prefix}_rhs"
    c_result = f"{var_prefix}_c_result"
    rust_result = f"{var_prefix}_rust_result"
    variables = [lhs, rhs, c_result, rust_result]

    zero = _bvconst(0, width)
    max_shift = _bvconst(width - 1, width)

    valid_shift = _and(
        _bvsge(rhs, zero),
        _bvsle(rhs, max_shift),
    )

    return [
        CoercionAssertion(
            smt_expression=valid_shift,
            description=f"Shift amount in valid range [0, {width - 1}] (i{width})",
            strength=AssertionStrength.ASSUME,
            variables=variables,
        ),
        CoercionAssertion(
            smt_expression=_implies(valid_shift, _eq(c_result, rust_result)),
            description=f"Shift result equivalence when amount is valid (i{width})",
            strength=AssertionStrength.HARD,
            variables=variables,
        ),
    ]


def _gen_float_precision_assertions(
    left: Instruction, right: Instruction, width: int, var_prefix: str,
) -> List[CoercionAssertion]:
    """Generate assertions for floating-point precision differences."""
    c_result = f"{var_prefix}_c_result"
    rust_result = f"{var_prefix}_rust_result"
    variables = [c_result, rust_result]

    return [
        CoercionAssertion(
            smt_expression=_eq(c_result, rust_result),
            description=f"Float result equivalence (f{width})",
            strength=AssertionStrength.SOFT,
            variables=variables,
        ),
    ]


def _gen_float_to_int_assertions(
    left: Instruction, right: Instruction, width: int, var_prefix: str,
) -> List[CoercionAssertion]:
    """Generate assertions for float-to-int conversion out-of-bounds."""
    src = f"{var_prefix}_float_src"
    c_result = f"{var_prefix}_c_result"
    rust_result = f"{var_prefix}_rust_result"
    variables = [src, c_result, rust_result]

    in_range = f"{var_prefix}_in_range"

    return [
        CoercionAssertion(
            smt_expression=in_range,
            description=f"Float value is within target integer range (i{width})",
            strength=AssertionStrength.ASSUME,
            variables=variables + [in_range],
        ),
        CoercionAssertion(
            smt_expression=_implies(in_range, _eq(c_result, rust_result)),
            description=f"Float-to-int conversion equivalence when in range (i{width})",
            strength=AssertionStrength.HARD,
            variables=variables,
        ),
    ]


def _gen_null_deref_assertions(
    left: Instruction, right: Instruction, width: int, var_prefix: str,
) -> List[CoercionAssertion]:
    """Generate assertions for null pointer dereference checks."""
    ptr = f"{var_prefix}_ptr"
    c_result = f"{var_prefix}_c_result"
    rust_result = f"{var_prefix}_rust_result"
    variables = [ptr, c_result, rust_result]
    ptr_width = 64

    null_ptr = _bvconst(0, ptr_width)

    return [
        CoercionAssertion(
            smt_expression=_not(_eq(ptr, null_ptr)),
            description="Pointer is non-null",
            strength=AssertionStrength.ASSUME,
            variables=variables,
        ),
        CoercionAssertion(
            smt_expression=_implies(
                _not(_eq(ptr, null_ptr)),
                _eq(c_result, rust_result),
            ),
            description="Memory access equivalence when pointer is non-null",
            strength=AssertionStrength.HARD,
            variables=variables,
        ),
    ]


def _gen_bounds_check_assertions(
    left: Instruction, right: Instruction, width: int, var_prefix: str,
) -> List[CoercionAssertion]:
    """Generate assertions for array bounds checking."""
    idx = f"{var_prefix}_idx"
    length = f"{var_prefix}_len"
    c_result = f"{var_prefix}_c_result"
    rust_result = f"{var_prefix}_rust_result"
    variables = [idx, length, c_result, rust_result]

    in_bounds = _and(
        _bvsge(idx, _bvconst(0, width)),
        _bvslt(idx, length),
    )

    return [
        CoercionAssertion(
            smt_expression=in_bounds,
            description="Array index is within bounds",
            strength=AssertionStrength.ASSUME,
            variables=variables,
        ),
        CoercionAssertion(
            smt_expression=_implies(in_bounds, _eq(c_result, rust_result)),
            description="Array access equivalence when index is in bounds",
            strength=AssertionStrength.HARD,
            variables=variables,
        ),
    ]


def _gen_return_coercion_assertions(
    left: Instruction, right: Instruction, width: int, var_prefix: str,
) -> List[CoercionAssertion]:
    """Generate assertions for return value coercions."""
    c_result = f"{var_prefix}_c_result"
    rust_result = f"{var_prefix}_rust_result"
    variables = [c_result, rust_result]

    return [
        CoercionAssertion(
            smt_expression=_eq(c_result, rust_result),
            description=f"Return value equivalence (i{width})",
            strength=AssertionStrength.HARD,
            variables=variables,
        ),
    ]


def _gen_error_handling_assertions(
    left: Instruction, right: Instruction, width: int, var_prefix: str,
) -> List[CoercionAssertion]:
    """Generate assertions for error handling differences."""
    c_result = f"{var_prefix}_c_result"
    rust_result = f"{var_prefix}_rust_result"
    c_err = f"{var_prefix}_c_err"
    rust_err = f"{var_prefix}_rust_err"
    variables = [c_result, rust_result, c_err, rust_err]

    return [
        CoercionAssertion(
            smt_expression=_eq(c_err, rust_err),
            description="Error status equivalence",
            strength=AssertionStrength.SOFT,
            variables=variables,
        ),
        CoercionAssertion(
            smt_expression=_implies(
                _not(c_err),
                _eq(c_result, rust_result),
            ),
            description="Result equivalence when no error",
            strength=AssertionStrength.HARD,
            variables=variables,
        ),
    ]


# ---------------------------------------------------------------------------
# Generator dispatch table
# ---------------------------------------------------------------------------

_COERCION_GENERATORS: Dict[DivergenceClass, Callable] = {
    DivergenceClass.SignedOverflow: _gen_signed_overflow_assertions,
    DivergenceClass.UnsignedWrap: _gen_unsigned_wrap_assertions,
    DivergenceClass.DivisionByZero: _gen_division_assertions,
    DivergenceClass.NegativeShift: _gen_shift_assertions,
    DivergenceClass.FloatPrecision: _gen_float_precision_assertions,
    DivergenceClass.FloatToIntOOB: _gen_float_to_int_assertions,
    DivergenceClass.NullDeref: _gen_null_deref_assertions,
    DivergenceClass.ArrayOOB: _gen_bounds_check_assertions,
    DivergenceClass.ErrorHandling: _gen_error_handling_assertions,
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _get_instruction_width(inst: Instruction) -> int:
    """Get the bit width of an instruction's result type."""
    if inst.type is None:
        return 32
    if isinstance(inst.type, IntType):
        return inst.type.width
    if isinstance(inst.type, FloatType):
        return inst.type.width
    if isinstance(inst.type, PointerType):
        return 64
    return 32


def _detect_applicable_divergences(
    left: Instruction,
    right: Instruction,
    table: DivergenceTable,
) -> List[Tuple[DivergenceClass, DivergenceEntry]]:
    """Detect which divergences are applicable to the instruction pair."""
    applicable = []
    for entry in table:
        if _is_divergence_relevant(entry.cls, left, right):
            applicable.append((entry.cls, entry))
    return applicable


def _is_divergence_relevant(
    cls: DivergenceClass,
    left: Instruction,
    right: Instruction,
) -> bool:
    """Check if a divergence class is actually relevant for the instruction pair."""
    if cls == DivergenceClass.SignedOverflow:
        if isinstance(left, BinaryOp) and left.op.name in ("ADD", "SUB", "MUL"):
            if isinstance(left.type, IntType) and left.type.is_signed:
                return True
    elif cls == DivergenceClass.UnsignedWrap:
        if isinstance(left, BinaryOp) and left.op.name in ("ADD", "SUB", "MUL"):
            if isinstance(left.type, IntType) and not left.type.is_signed:
                return True
    elif cls == DivergenceClass.DivisionByZero:
        if isinstance(left, BinaryOp) and left.op.name in ("SDIV", "UDIV", "SREM", "UREM"):
            return True
    elif cls == DivergenceClass.NegativeShift:
        if isinstance(left, BinaryOp) and left.op.name in ("SHL", "LSHR", "ASHR"):
            return True
    elif cls == DivergenceClass.FloatPrecision:
        if isinstance(left, BinaryOp) and left.op.name in ("FADD", "FSUB", "FMUL", "FDIV"):
            return True
    elif cls == DivergenceClass.FloatToIntOOB:
        if isinstance(left, CastInst) and left.cast_kind in (CastKind.FPTOSI, CastKind.FPTOUI):
            return True
    elif cls == DivergenceClass.NullDeref:
        if isinstance(left, LoadInst) or isinstance(left, StoreInst):
            return True
    elif cls == DivergenceClass.ArrayOOB:
        if isinstance(left, LoadInst) or isinstance(left, StoreInst):
            return True
    elif cls == DivergenceClass.ErrorHandling:
        if isinstance(left, CallInst):
            return True
    return False


# ---------------------------------------------------------------------------
# Main generator class
# ---------------------------------------------------------------------------

class CoercionGenerator:
    """
    Generates coercion points for aligned instruction pairs.

    For each aligned pair, consults the divergence table and inserts
    coercion assertions where C and Rust semantics may differ.
    """

    def __init__(
        self,
        divergence_table: Optional[DivergenceTable] = None,
        c_config: Optional[SemanticConfig] = None,
        rust_config: Optional[SemanticConfig] = None,
    ):
        self.table = divergence_table or DivergenceTable()
        self.c_config = c_config or SemanticConfig.c11()
        self.rust_config = rust_config or SemanticConfig.rust_release()
        self._var_counter = 0

    def _fresh_prefix(self) -> str:
        self._var_counter += 1
        return f"cp_{self._var_counter}"

    def generate_for_alignment(
        self,
        alignment: AlignmentResult,
    ) -> List[CoercionPoint]:
        """Generate all coercion points for a complete alignment."""
        coercion_points: List[CoercionPoint] = []

        for ba in alignment.block_alignments:
            for ia in ba.instruction_alignments:
                if ia.is_matched and ia.left is not None and ia.right is not None:
                    points = self.generate_for_pair(ia.left, ia.right)
                    coercion_points.extend(points)

        # Check return type coercion
        ret_points = self._generate_return_coercions(
            alignment.left_function, alignment.right_function
        )
        coercion_points.extend(ret_points)

        return coercion_points

    def generate_for_pair(
        self,
        left: Instruction,
        right: Instruction,
    ) -> List[CoercionPoint]:
        """Generate coercion points for a single aligned instruction pair."""
        points: List[CoercionPoint] = []

        applicable = _detect_applicable_divergences(left, right, self.table)

        for cls, entry in applicable:
            prefix = self._fresh_prefix()
            width = _get_instruction_width(left)

            generator = _COERCION_GENERATORS.get(cls)
            if generator is not None:
                assertions = generator(left, right, width, prefix)
            else:
                assertions = self._generate_generic_assertions(
                    left, right, width, prefix, entry
                )

            if assertions:
                kind = self._classify_coercion_kind(cls)
                point = CoercionPoint(
                    kind=kind,
                    left_instruction=left,
                    right_instruction=right,
                    divergence_class=cls,
                    c_semantics=entry.c_semantics,
                    rust_semantics=entry.rust_semantics,
                    assertions=assertions,
                    operation=self._describe_operation(left),
                    bit_width=width,
                    source_location=self._get_source_location(left),
                )
                points.append(point)

        return points

    def summary(self, points: List[CoercionPoint]) -> str:
        """Generate a summary of coercion points."""
        if not points:
            return "No coercion points generated."
        lines = [f"Coercion summary: {len(points)} points"]
        by_kind: Dict[CoercionKind, int] = {}
        for p in points:
            by_kind[p.kind] = by_kind.get(p.kind, 0) + 1
        for kind, count in sorted(by_kind.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"  {kind.name}: {count}")
        total_assertions = sum(p.num_assertions for p in points)
        lines.append(f"Total assertions: {total_assertions}")
        return "\n".join(lines)

    def _generate_generic_assertions(self, left, right, width, var_prefix, entry):
        """Generate generic assertions when no specific generator exists."""
        lhs = f"{var_prefix}_c_result"
        rhs = f"{var_prefix}_rust_result"
        return [CoercionAssertion(
            smt_expression=_eq(lhs, rhs),
            description=f"Generic equivalence for {entry.cls.name}",
            strength=AssertionStrength.SOFT,
            variables=[lhs, rhs],
        )]

    def _generate_return_coercions(self, left_func, right_func):
        """Generate coercions for return type differences."""
        points = []
        lt = left_func.return_type
        rt = right_func.return_type

        if isinstance(lt, VoidType) and isinstance(rt, VoidType):
            return points
        if isinstance(lt, VoidType) or isinstance(rt, VoidType):
            return points

        if type(lt) is type(rt):
            if isinstance(lt, IntType) and isinstance(rt, IntType):
                assertions = []
                result_c = "ret_c"
                result_rust = "ret_rust"
                width = max(lt.width, rt.width)

                if lt.width != rt.width:
                    assertions.append(CoercionAssertion(
                        smt_expression=_eq(result_c, result_rust),
                        description=f"Return width coercion: i{lt.width} vs i{rt.width}",
                        strength=AssertionStrength.HARD,
                        variables=[result_c, result_rust],
                    ))

                if lt.is_signed or rt.is_signed:
                    assertions.append(CoercionAssertion(
                        smt_expression=_eq(result_c, result_rust),
                        description=f"Return value equivalence (signed context)",
                        strength=AssertionStrength.HARD,
                        variables=[result_c, result_rust],
                    ))
                else:
                    assertions.append(CoercionAssertion(
                        smt_expression=_eq(result_c, result_rust),
                        description=f"Return value equivalence",
                        strength=AssertionStrength.HARD,
                        variables=[result_c, result_rust],
                    ))

                if lt.is_signed != rt.is_signed:
                    assertions.append(CoercionAssertion(
                        smt_expression=_eq(result_c, result_rust),
                        description=f"Return signedness coercion: {'signed' if lt.is_signed else 'unsigned'} "
                                    f"vs {'signed' if rt.is_signed else 'unsigned'}",
                        strength=AssertionStrength.HARD,
                        variables=[result_c, result_rust],
                    ))

                if assertions:
                    points.append(CoercionPoint(
                        kind=CoercionKind.RETURN_COERCION,
                        left_instruction=None,
                        right_instruction=None,
                        divergence_class=DivergenceClass.SignedOverflow,
                        c_semantics=CSemantics(summary="C return semantics"),
                        rust_semantics=RustSemantics(summary="Rust return semantics"),
                        assertions=assertions,
                        operation="return",
                        bit_width=width,
                    ))
        return points

    def _generate_type_coercion(self, left, right):
        """Generate type coercion points."""
        points = []
        lt = left.type if hasattr(left, 'type') else None
        rt = right.type if hasattr(right, 'type') else None
        if lt is None or rt is None:
            return points
        if isinstance(lt, IntType) and isinstance(rt, IntType) and (lt.width != rt.width or lt.is_signed != rt.is_signed):
            prefix = self._fresh_prefix()
            width = max(lt.width, rt.width)
            assertions = []
            lhs = f"{prefix}_c"
            rhs = f"{prefix}_rust"
            assertions.append(CoercionAssertion(
                smt_expression=_eq(lhs, rhs),
                description=f"Type width coercion: i{lt.width} vs i{rt.width}",
                strength=AssertionStrength.HARD,
                variables=[lhs, rhs],
            ))
            if lt.is_signed or rt.is_signed:
                assertions.append(CoercionAssertion(
                    smt_expression=_implies(
                        _bvsge(lhs, _bvconst(0, width)),
                        _eq(lhs, rhs)
                    ),
                    description="Sign extension equivalence for non-negative values",
                    strength=AssertionStrength.SOFT,
                    variables=[lhs, rhs],
                ))
            if assertions:
                points.append(CoercionPoint(
                    kind=CoercionKind.TYPE_WIDTH,
                    left_instruction=left,
                    right_instruction=right,
                    divergence_class=DivergenceClass.IntPromotion,
                    c_semantics=CSemantics(summary="C integer promotion"),
                    rust_semantics=RustSemantics(summary="Rust explicit cast required"),
                    assertions=assertions,
                    operation="type_coercion",
                    bit_width=width,
                ))
        return points

    @staticmethod
    def _classify_coercion_kind(cls: DivergenceClass) -> CoercionKind:
        mapping = {
            DivergenceClass.SignedOverflow: CoercionKind.OVERFLOW_CHECK,
            DivergenceClass.UnsignedWrap: CoercionKind.OVERFLOW_CHECK,
            DivergenceClass.DivisionByZero: CoercionKind.DIVISION_CHECK,
            DivergenceClass.NegativeShift: CoercionKind.SHIFT_CHECK,
            DivergenceClass.FloatPrecision: CoercionKind.FLOAT_PRECISION,
            DivergenceClass.FloatToIntOOB: CoercionKind.FLOAT_TO_INT,
            DivergenceClass.NullDeref: CoercionKind.NULL_CHECK,
            DivergenceClass.ArrayOOB: CoercionKind.BOUNDS_CHECK,
            DivergenceClass.ErrorHandling: CoercionKind.ERROR_HANDLING,
            DivergenceClass.IntPromotion: CoercionKind.TYPE_WIDTH,
            DivergenceClass.PointerArith: CoercionKind.POINTER_PROVENANCE,
        }
        return mapping.get(cls, CoercionKind.OVERFLOW_CHECK)

    @staticmethod
    def _describe_operation(inst: Instruction) -> str:
        if isinstance(inst, BinaryOp):
            return f"{inst.op.name.lower()}"
        if isinstance(inst, CastInst):
            return f"cast.{inst.cast_kind.name.lower()}"
        if isinstance(inst, LoadInst):
            return "load"
        if isinstance(inst, StoreInst):
            return "store"
        if isinstance(inst, CallInst):
            return f"call"
        if isinstance(inst, ReturnInst):
            return "return"
        return type(inst).__name__.lower()

    @staticmethod
    def _get_source_location(inst: Instruction) -> str:
        if hasattr(inst, 'metadata') and inst.metadata:
            if hasattr(inst.metadata, 'source_loc') and inst.metadata.source_loc:
                return str(inst.metadata.source_loc)
        return ""
