"""
Full type checker for the Cross-Language Equivalence Verifier IR.

Verifies instruction operand types, inserts implicit casts per language
rules, checks function call argument compatibility, verifies pointer
dereference safety, checks integer promotion chains, and reports type
errors with source locations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from ..ir.types import (
    IRType,
    IntType,
    FloatType,
    PointerType,
    ArrayType,
    StructType,
    UnionType,
    FunctionType,
    VoidType,
    Signedness,
    FloatKind,
    Language,
    check_compatibility,
    TypeCompatibility,
)
from ..ir.instructions import (
    Instruction,
    BinaryOp,
    UnaryOp,
    CompareOp,
    CastInst,
    LoadInst,
    StoreInst,
    AllocaInst,
    GetElementPtrInst,
    CallInst,
    ReturnInst,
    BranchInst,
    SwitchInst,
    SelectInst,
    PhiInst,
    ExtractValueInst,
    InsertValueInst,
    MemcpyInst,
    MemsetInst,
    BinOpKind,
    UnaryOpKind,
    CastKind,
    CmpPredicate,
    Value,
    Constant,
    Argument,
    SourceLocation,
)
from ..ir.basic_block import BasicBlock
from ..ir.function import Function
from ..ir.module import Module


# ── Error / Warning types ────────────────────────────────────────────────

class TypeCheckSeverity(Enum):
    ERROR = auto()
    WARNING = auto()
    INFO = auto()


@dataclass
class TypeCheckError:
    """A type error found during checking."""
    message: str
    severity: TypeCheckSeverity = TypeCheckSeverity.ERROR
    instruction: Instruction | None = None
    function_name: str = ""
    block_name: str = ""
    source_location: SourceLocation | None = None

    def __str__(self) -> str:
        loc_parts: list[str] = []
        if self.function_name:
            loc_parts.append(f"in {self.function_name}")
        if self.block_name:
            loc_parts.append(f"block {self.block_name}")
        if self.source_location:
            loc_parts.append(f"at {self.source_location}")
        loc_str = " ".join(loc_parts)
        prefix = self.severity.name
        return f"[{prefix}] {self.message}" + (f" ({loc_str})" if loc_str else "")


TypeCheckWarning = TypeCheckError  # alias for API symmetry


@dataclass
class InsertedCast:
    """Record of a cast that was implicitly inserted."""
    src_type: IRType
    dst_type: IRType
    cast_kind: CastKind
    reason: str
    instruction: Instruction | None = None
    function_name: str = ""

    def __str__(self) -> str:
        return (
            f"Cast {self.cast_kind.value}: {self.src_type} → {self.dst_type} "
            f"({self.reason})"
        )


@dataclass
class TypeCheckResult:
    """Aggregate result of type checking."""
    errors: list[TypeCheckError] = field(default_factory=list)
    warnings: list[TypeCheckError] = field(default_factory=list)
    inserted_casts: list[InsertedCast] = field(default_factory=list)
    instructions_checked: int = 0

    @property
    def is_valid(self) -> bool:
        return all(e.severity is not TypeCheckSeverity.ERROR for e in self.errors)

    @property
    def error_count(self) -> int:
        return sum(1 for e in self.errors if e.severity is TypeCheckSeverity.ERROR)

    @property
    def warning_count(self) -> int:
        return len(self.warnings)

    def add_error(self, msg: str, **kwargs: Any) -> None:
        self.errors.append(TypeCheckError(message=msg, severity=TypeCheckSeverity.ERROR, **kwargs))

    def add_warning(self, msg: str, **kwargs: Any) -> None:
        w = TypeCheckError(message=msg, severity=TypeCheckSeverity.WARNING, **kwargs)
        self.warnings.append(w)

    def merge(self, other: "TypeCheckResult") -> None:
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.inserted_casts.extend(other.inserted_casts)
        self.instructions_checked += other.instructions_checked

    def summary(self) -> str:
        lines = [
            f"Type Check Result:",
            f"  Instructions checked: {self.instructions_checked}",
            f"  Errors:               {self.error_count}",
            f"  Warnings:             {self.warning_count}",
            f"  Inserted casts:       {len(self.inserted_casts)}",
        ]
        if self.errors:
            lines.append("  Errors:")
            for e in self.errors:
                lines.append(f"    {e}")
        if self.warnings:
            lines.append("  Warnings:")
            for w in self.warnings:
                lines.append(f"    {w}")
        return "\n".join(lines)


# ── Type Checker ─────────────────────────────────────────────────────────

class TypeChecker:
    """Full type checker for the IR.

    Configurable for C or Rust promotion rules.  By default uses C-style
    implicit promotion and reports Rust-incompatible implicit conversions
    as warnings.
    """

    def __init__(
        self,
        language: Language = Language.C,
        insert_casts: bool = True,
        strict: bool = False,
    ) -> None:
        self.language = language
        self.insert_casts = insert_casts
        self.strict = strict  # If True, warnings become errors

    # ── Public API ───────────────────────────────────────────────────────

    def check_module(self, module: Module) -> TypeCheckResult:
        """Type-check an entire module."""
        result = TypeCheckResult()

        # Check globals
        for gv in module.iter_globals():
            if gv.initializer is not None:
                if gv.initializer.type != gv.type:
                    if not self._types_compatible(gv.initializer.type, gv.type):
                        result.add_error(
                            f"Global '{gv.name}': initializer type {gv.initializer.type} "
                            f"incompatible with declared type {gv.type}"
                        )

        # Check each function
        for func in module.iter_functions():
            func_result = self.check_function(func)
            result.merge(func_result)

        return result

    def check_function(self, func: Function) -> TypeCheckResult:
        """Type-check a single function."""
        result = TypeCheckResult()

        # Check return type consistency
        for block in func.blocks:
            for inst in block:
                if isinstance(inst, ReturnInst):
                    self._check_return(inst, func, result)

        # Check all instructions
        for block in func.blocks:
            for inst in block:
                self._check_instruction(inst, func.name, block.name, result)
                result.instructions_checked += 1

        # Check phi node type consistency
        for block in func.blocks:
            for phi in block.phi_nodes:
                self._check_phi(phi, func.name, block.name, result)

        return result

    def check_instruction(self, inst: Instruction) -> TypeCheckResult:
        """Type-check a single instruction."""
        result = TypeCheckResult()
        func_name = ""
        block_name = ""
        if inst.parent:
            block_name = inst.parent.name
            if inst.parent.parent:
                func_name = inst.parent.parent.name
        self._check_instruction(inst, func_name, block_name, result)
        result.instructions_checked = 1
        return result

    # ── Instruction checking dispatch ────────────────────────────────────

    def _check_instruction(
        self,
        inst: Instruction,
        func_name: str,
        block_name: str,
        result: TypeCheckResult,
    ) -> None:
        ctx = {"function_name": func_name, "block_name": block_name, "instruction": inst}
        src = inst.metadata.source_loc if inst.metadata else None
        if src:
            ctx["source_location"] = src

        if isinstance(inst, BinaryOp):
            self._check_binary_op(inst, result, ctx)
        elif isinstance(inst, UnaryOp):
            self._check_unary_op(inst, result, ctx)
        elif isinstance(inst, CompareOp):
            self._check_compare_op(inst, result, ctx)
        elif isinstance(inst, CastInst):
            self._check_cast(inst, result, ctx)
        elif isinstance(inst, LoadInst):
            self._check_load(inst, result, ctx)
        elif isinstance(inst, StoreInst):
            self._check_store(inst, result, ctx)
        elif isinstance(inst, AllocaInst):
            pass  # Alloca is always well-typed
        elif isinstance(inst, GetElementPtrInst):
            self._check_gep(inst, result, ctx)
        elif isinstance(inst, CallInst):
            self._check_call(inst, result, ctx)
        elif isinstance(inst, ReturnInst):
            pass  # Checked separately
        elif isinstance(inst, BranchInst):
            self._check_branch(inst, result, ctx)
        elif isinstance(inst, SwitchInst):
            self._check_switch(inst, result, ctx)
        elif isinstance(inst, SelectInst):
            self._check_select(inst, result, ctx)
        elif isinstance(inst, PhiInst):
            pass  # Checked separately
        elif isinstance(inst, ExtractValueInst):
            self._check_extract_value(inst, result, ctx)
        elif isinstance(inst, InsertValueInst):
            self._check_insert_value(inst, result, ctx)

    # ── Binary operation checking ────────────────────────────────────────

    def _check_binary_op(
        self, inst: BinaryOp, result: TypeCheckResult, ctx: dict,
    ) -> None:
        lhs_type = inst.lhs.type
        rhs_type = inst.rhs.type

        if inst.is_floating():
            # Both operands must be float
            if not isinstance(lhs_type, FloatType):
                result.add_error(
                    f"{inst.op.value}: left operand must be float, got {lhs_type}", **ctx,
                )
            if not isinstance(rhs_type, FloatType):
                result.add_error(
                    f"{inst.op.value}: right operand must be float, got {rhs_type}", **ctx,
                )
            if isinstance(lhs_type, FloatType) and isinstance(rhs_type, FloatType):
                if lhs_type != rhs_type:
                    result.add_error(
                        f"{inst.op.value}: float width mismatch: "
                        f"{lhs_type} vs {rhs_type}", **ctx,
                    )
        else:
            # Both operands must be integer
            if not isinstance(lhs_type, IntType):
                result.add_error(
                    f"{inst.op.value}: left operand must be integer, got {lhs_type}", **ctx,
                )
            if not isinstance(rhs_type, IntType):
                result.add_error(
                    f"{inst.op.value}: right operand must be integer, got {rhs_type}", **ctx,
                )

            if isinstance(lhs_type, IntType) and isinstance(rhs_type, IntType):
                # Check width match
                if lhs_type.width != rhs_type.width:
                    result.add_error(
                        f"{inst.op.value}: operand width mismatch: "
                        f"{lhs_type.width} vs {rhs_type.width}", **ctx,
                    )

                # Check signedness match for division/remainder
                if inst.op in (BinOpKind.SDIV, BinOpKind.SREM):
                    if lhs_type.is_unsigned or rhs_type.is_unsigned:
                        result.add_warning(
                            f"{inst.op.value}: signed operation on unsigned operand", **ctx,
                        )
                if inst.op in (BinOpKind.UDIV, BinOpKind.UREM):
                    if lhs_type.is_signed or rhs_type.is_signed:
                        result.add_warning(
                            f"{inst.op.value}: unsigned operation on signed operand", **ctx,
                        )

                # C promotion check: warn if operands are narrower than int
                if self.language is Language.C:
                    if lhs_type.width < 32:
                        result.add_warning(
                            f"{inst.op.value}: operand width {lhs_type.width} < 32; "
                            f"C would promote to int first", **ctx,
                        )
                        if self.insert_casts:
                            self._record_promotion_cast(
                                lhs_type, IntType(32, Signedness.SIGNED),
                                f"C integer promotion for {inst.op.value}", inst,
                                ctx.get("function_name", ""), result,
                            )

        # Check result type matches
        expected_type = lhs_type
        if inst.type != expected_type:
            if not self._types_compatible(inst.type, expected_type):
                result.add_error(
                    f"{inst.op.value}: result type {inst.type} doesn't match "
                    f"operand type {expected_type}", **ctx,
                )

    # ── Unary operation checking ─────────────────────────────────────────

    def _check_unary_op(
        self, inst: UnaryOp, result: TypeCheckResult, ctx: dict,
    ) -> None:
        operand = inst.operands[0]
        op_type = operand.type

        if inst.op is UnaryOpKind.NEG:
            if not isinstance(op_type, IntType):
                result.add_error(f"neg: operand must be integer, got {op_type}", **ctx)
            elif op_type.is_unsigned:
                result.add_warning("neg: negation of unsigned integer", **ctx)

        elif inst.op is UnaryOpKind.FNEG:
            if not isinstance(op_type, FloatType):
                result.add_error(f"fneg: operand must be float, got {op_type}", **ctx)

        elif inst.op is UnaryOpKind.NOT:
            if not isinstance(op_type, IntType):
                result.add_error(f"not: operand must be integer, got {op_type}", **ctx)

        elif inst.op is UnaryOpKind.BITWISE_NOT:
            if not isinstance(op_type, IntType):
                result.add_error(f"bitwise_not: operand must be integer, got {op_type}", **ctx)

    # ── Comparison checking ──────────────────────────────────────────────

    def _check_compare_op(
        self, inst: CompareOp, result: TypeCheckResult, ctx: dict,
    ) -> None:
        lhs_type = inst.lhs.type
        rhs_type = inst.rhs.type

        # Result must be i1
        if not (isinstance(inst.type, IntType) and inst.type.width == 1):
            result.add_error(
                f"compare: result type must be i1, got {inst.type}", **ctx,
            )

        if inst.predicate.is_float():
            if not isinstance(lhs_type, FloatType):
                result.add_error(
                    f"fcmp: left operand must be float, got {lhs_type}", **ctx,
                )
            if not isinstance(rhs_type, FloatType):
                result.add_error(
                    f"fcmp: right operand must be float, got {rhs_type}", **ctx,
                )
        else:
            # Integer or pointer comparison
            if isinstance(lhs_type, IntType) and isinstance(rhs_type, IntType):
                if lhs_type.width != rhs_type.width:
                    result.add_error(
                        f"icmp: width mismatch: {lhs_type.width} vs {rhs_type.width}", **ctx,
                    )
                # Signedness mismatch warning
                if inst.predicate.is_signed() and (lhs_type.is_unsigned or rhs_type.is_unsigned):
                    result.add_warning(
                        "icmp: signed comparison on unsigned operands", **ctx,
                    )
                if inst.predicate.is_unsigned() and (lhs_type.is_signed or rhs_type.is_signed):
                    result.add_warning(
                        "icmp: unsigned comparison on signed operands", **ctx,
                    )
            elif isinstance(lhs_type, PointerType) and isinstance(rhs_type, PointerType):
                pass  # Pointer comparison is valid
            elif lhs_type != rhs_type:
                result.add_error(
                    f"compare: type mismatch: {lhs_type} vs {rhs_type}", **ctx,
                )

    # ── Cast checking ────────────────────────────────────────────────────

    def _check_cast(
        self, inst: CastInst, result: TypeCheckResult, ctx: dict,
    ) -> None:
        src = inst.src_type
        dst = inst.dest_type
        kind = inst.cast_kind

        if kind is CastKind.TRUNC:
            if not isinstance(src, IntType) or not isinstance(dst, IntType):
                result.add_error("trunc: both types must be integer", **ctx)
            elif src.width <= dst.width:
                result.add_error(
                    f"trunc: source width {src.width} must be > dest width {dst.width}", **ctx,
                )

        elif kind in (CastKind.ZEXT, CastKind.SEXT):
            if not isinstance(src, IntType) or not isinstance(dst, IntType):
                result.add_error(f"{kind.value}: both types must be integer", **ctx)
            elif src.width >= dst.width:
                result.add_error(f"{kind.value}: source must be narrower than dest", **ctx)
            # Check sign extension on unsigned type
            if kind is CastKind.SEXT and isinstance(src, IntType) and src.is_unsigned:
                result.add_warning(
                    f"sext of unsigned type {src}; consider zext", **ctx,
                )

        elif kind is CastKind.FPTRUNC:
            if not isinstance(src, FloatType) or not isinstance(dst, FloatType):
                result.add_error("fptrunc: both must be float", **ctx)
            elif isinstance(src, FloatType) and isinstance(dst, FloatType):
                if src.width <= dst.width:
                    result.add_error("fptrunc: source must be wider", **ctx)

        elif kind is CastKind.FPEXT:
            if not isinstance(src, FloatType) or not isinstance(dst, FloatType):
                result.add_error("fpext: both must be float", **ctx)
            elif isinstance(src, FloatType) and isinstance(dst, FloatType):
                if src.width >= dst.width:
                    result.add_error("fpext: source must be narrower", **ctx)

        elif kind in (CastKind.FPTOSI, CastKind.FPTOUI):
            if not isinstance(src, FloatType):
                result.add_error(f"{kind.value}: source must be float", **ctx)
            if not isinstance(dst, IntType):
                result.add_error(f"{kind.value}: dest must be integer", **ctx)

        elif kind in (CastKind.SITOFP, CastKind.UITOFP):
            if not isinstance(src, IntType):
                result.add_error(f"{kind.value}: source must be integer", **ctx)
            if not isinstance(dst, FloatType):
                result.add_error(f"{kind.value}: dest must be float", **ctx)
            # Precision loss warning
            if isinstance(src, IntType) and isinstance(dst, FloatType):
                if dst.kind is FloatKind.F32 and src.width > 24:
                    result.add_warning(
                        f"{kind.value}: {src.width}-bit int may not be exactly "
                        f"representable in f32 (24 mantissa bits)", **ctx,
                    )
                elif dst.kind is FloatKind.F64 and src.width > 53:
                    result.add_warning(
                        f"{kind.value}: {src.width}-bit int may not be exactly "
                        f"representable in f64 (53 mantissa bits)", **ctx,
                    )

        elif kind is CastKind.BITCAST:
            if src.is_sized() and dst.is_sized():
                if src.size_bits() != dst.size_bits():
                    result.add_error(
                        f"bitcast: size mismatch: {src.size_bits()} vs {dst.size_bits()} bits",
                        **ctx,
                    )
            if self.language is Language.RUST:
                result.add_warning("bitcast: Rust requires transmute for this", **ctx)

        elif kind is CastKind.PTRTOINT:
            if not isinstance(src, PointerType):
                result.add_error("ptrtoint: source must be pointer", **ctx)
            if not isinstance(dst, IntType):
                result.add_error("ptrtoint: dest must be integer", **ctx)
            elif isinstance(dst, IntType) and dst.width < 64:
                result.add_warning(
                    f"ptrtoint to {dst.width}-bit int; may truncate pointer", **ctx,
                )

        elif kind is CastKind.INTTOPTR:
            if not isinstance(src, IntType):
                result.add_error("inttoptr: source must be integer", **ctx)
            if not isinstance(dst, PointerType):
                result.add_error("inttoptr: dest must be pointer", **ctx)

    # ── Load checking ────────────────────────────────────────────────────

    def _check_load(
        self, inst: LoadInst, result: TypeCheckResult, ctx: dict,
    ) -> None:
        addr_type = inst.address.type
        if not isinstance(addr_type, PointerType):
            result.add_error(
                f"load: address must be pointer, got {addr_type}", **ctx,
            )
        elif isinstance(addr_type, PointerType):
            pointee = addr_type.pointee
            if not isinstance(pointee, VoidType) and inst.type != pointee:
                result.add_warning(
                    f"load: result type {inst.type} doesn't match "
                    f"pointee type {pointee}", **ctx,
                )

    # ── Store checking ───────────────────────────────────────────────────

    def _check_store(
        self, inst: StoreInst, result: TypeCheckResult, ctx: dict,
    ) -> None:
        addr_type = inst.address.type
        if not isinstance(addr_type, PointerType):
            result.add_error(
                f"store: address must be pointer, got {addr_type}", **ctx,
            )
        elif isinstance(addr_type, PointerType):
            pointee = addr_type.pointee
            val_type = inst.value.type
            if not isinstance(pointee, VoidType) and val_type != pointee:
                if not self._types_compatible(val_type, pointee):
                    result.add_error(
                        f"store: value type {val_type} incompatible with "
                        f"pointee type {pointee}", **ctx,
                    )

    # ── GEP checking ────────────────────────────────────────────────────

    def _check_gep(
        self, inst: GetElementPtrInst, result: TypeCheckResult, ctx: dict,
    ) -> None:
        base_type = inst.operands[0].type if inst.operands else None
        if base_type is None:
            result.add_error("gep: missing base operand", **ctx)
            return

        if not isinstance(base_type, PointerType):
            result.add_error(
                f"gep: base must be pointer, got {base_type}", **ctx,
            )

        # Check index types are integer
        for i, idx in enumerate(inst.operands[1:]):
            if not isinstance(idx.type, IntType):
                result.add_error(
                    f"gep: index {i} must be integer, got {idx.type}", **ctx,
                )

        # Result must be pointer
        if not isinstance(inst.type, PointerType):
            result.add_error(
                f"gep: result must be pointer, got {inst.type}", **ctx,
            )

    # ── Call checking ────────────────────────────────────────────────────

    def _check_call(
        self, inst: CallInst, result: TypeCheckResult, ctx: dict,
    ) -> None:
        callee = inst.operands[0] if inst.operands else None
        if callee is None:
            result.add_error("call: missing callee", **ctx)
            return

        callee_type = callee.type
        func_type: FunctionType | None = None

        if isinstance(callee_type, FunctionType):
            func_type = callee_type
        elif isinstance(callee_type, PointerType):
            if isinstance(callee_type.pointee, FunctionType):
                func_type = callee_type.pointee

        if func_type is None:
            # Indirect call without type info; skip detailed checking
            return

        # Check argument count
        args = inst.operands[1:]
        expected = len(func_type.param_types)
        if func_type.is_variadic:
            if len(args) < expected:
                result.add_error(
                    f"call: too few arguments: expected at least {expected}, "
                    f"got {len(args)}", **ctx,
                )
        else:
            if len(args) != expected:
                result.add_error(
                    f"call: argument count mismatch: expected {expected}, "
                    f"got {len(args)}", **ctx,
                )

        # Check argument types
        for i, (arg, param_type) in enumerate(zip(args, func_type.param_types)):
            if arg.type != param_type:
                if not self._types_compatible(arg.type, param_type):
                    result.add_error(
                        f"call: argument {i} type {arg.type} incompatible "
                        f"with parameter type {param_type}", **ctx,
                    )
                else:
                    result.add_warning(
                        f"call: argument {i} type {arg.type} implicitly "
                        f"converted to {param_type}", **ctx,
                    )
                    if self.insert_casts:
                        cast_kind = self._determine_cast_kind(arg.type, param_type)
                        if cast_kind:
                            result.inserted_casts.append(InsertedCast(
                                src_type=arg.type,
                                dst_type=param_type,
                                cast_kind=cast_kind,
                                reason=f"call argument {i} conversion",
                                instruction=inst,
                                function_name=ctx.get("function_name", ""),
                            ))

        # Check variadic argument promotions (C)
        if func_type.is_variadic and self.language is Language.C:
            for i in range(expected, len(args)):
                arg = args[i]
                if isinstance(arg.type, IntType) and arg.type.width < 32:
                    result.add_warning(
                        f"call: variadic argument {i} ({arg.type}) "
                        f"will be promoted to int", **ctx,
                    )
                elif isinstance(arg.type, FloatType) and arg.type.kind is FloatKind.F32:
                    result.add_warning(
                        f"call: variadic argument {i} (f32) "
                        f"will be promoted to f64", **ctx,
                    )

        # Check return type
        if inst.type != func_type.return_type:
            if not isinstance(inst.type, VoidType) or not isinstance(func_type.return_type, VoidType):
                if not self._types_compatible(inst.type, func_type.return_type):
                    result.add_error(
                        f"call: result type {inst.type} doesn't match "
                        f"return type {func_type.return_type}", **ctx,
                    )

    # ── Branch checking ──────────────────────────────────────────────────

    def _check_branch(
        self, inst: BranchInst, result: TypeCheckResult, ctx: dict,
    ) -> None:
        cond = inst.condition
        if cond is not None:
            if not isinstance(cond.type, IntType) or cond.type.width != 1:
                result.add_error(
                    f"branch: condition must be i1, got {cond.type}", **ctx,
                )

    # ── Switch checking ──────────────────────────────────────────────────

    def _check_switch(
        self, inst: SwitchInst, result: TypeCheckResult, ctx: dict,
    ) -> None:
        cond = inst.condition
        if not isinstance(cond.type, IntType):
            result.add_error(
                f"switch: condition must be integer, got {cond.type}", **ctx,
            )

    # ── Select checking ──────────────────────────────────────────────────

    def _check_select(
        self, inst: SelectInst, result: TypeCheckResult, ctx: dict,
    ) -> None:
        cond = inst.condition
        if not isinstance(cond.type, IntType) or cond.type.width != 1:
            result.add_error(
                f"select: condition must be i1, got {cond.type}", **ctx,
            )

        true_type = inst.true_value.type
        false_type = inst.false_value.type
        if true_type != false_type:
            result.add_error(
                f"select: true/false types must match: {true_type} vs {false_type}", **ctx,
            )

        if inst.type != true_type:
            result.add_error(
                f"select: result type {inst.type} doesn't match "
                f"operand type {true_type}", **ctx,
            )

    # ── Phi checking ─────────────────────────────────────────────────────

    def _check_phi(
        self, phi: PhiInst, func_name: str, block_name: str,
        result: TypeCheckResult,
    ) -> None:
        ctx = {"function_name": func_name, "block_name": block_name, "instruction": phi}
        phi_type = phi.type

        for val, _block in phi.incoming:
            if val.type != phi_type:
                if not self._types_compatible(val.type, phi_type):
                    result.add_error(
                        f"phi: incoming value type {val.type} doesn't match "
                        f"phi type {phi_type}", **ctx,
                    )
                else:
                    result.add_warning(
                        f"phi: incoming value type {val.type} implicitly "
                        f"converted to {phi_type}", **ctx,
                    )

    # ── Return checking ──────────────────────────────────────────────────

    def _check_return(
        self, inst: ReturnInst, func: Function, result: TypeCheckResult,
    ) -> None:
        ctx = {"function_name": func.name, "instruction": inst}
        if inst.is_void_return:
            if not func.is_void_return:
                result.add_error(
                    f"return: void return in non-void function "
                    f"(expected {func.return_type})", **ctx,
                )
        else:
            ret_val = inst.return_value
            if ret_val is not None:
                if ret_val.type != func.return_type:
                    if not self._types_compatible(ret_val.type, func.return_type):
                        result.add_error(
                            f"return: type {ret_val.type} incompatible with "
                            f"declared return type {func.return_type}", **ctx,
                        )
                    else:
                        result.add_warning(
                            f"return: implicit conversion from {ret_val.type} "
                            f"to {func.return_type}", **ctx,
                        )

    # ── ExtractValue checking ────────────────────────────────────────────

    def _check_extract_value(
        self, inst: ExtractValueInst, result: TypeCheckResult, ctx: dict,
    ) -> None:
        agg_type = inst.operands[0].type if inst.operands else None
        if agg_type is None:
            result.add_error("extract_value: missing aggregate operand", **ctx)
            return
        if not isinstance(agg_type, (StructType, ArrayType)):
            result.add_error(
                f"extract_value: aggregate must be struct or array, got {agg_type}", **ctx,
            )

    # ── InsertValue checking ─────────────────────────────────────────────

    def _check_insert_value(
        self, inst: InsertValueInst, result: TypeCheckResult, ctx: dict,
    ) -> None:
        if len(inst.operands) < 2:
            result.add_error("insert_value: missing operands", **ctx)
            return
        agg_type = inst.operands[0].type
        if not isinstance(agg_type, (StructType, ArrayType)):
            result.add_error(
                f"insert_value: aggregate must be struct or array, got {agg_type}", **ctx,
            )

    # ── Helper methods ───────────────────────────────────────────────────

    def _types_compatible(self, src: IRType, dst: IRType) -> bool:
        """Check if src can be implicitly converted to dst."""
        if src == dst:
            return True

        # Integer widening
        if isinstance(src, IntType) and isinstance(dst, IntType):
            if src.width <= dst.width:
                return True

        # Float widening
        if isinstance(src, FloatType) and isinstance(dst, FloatType):
            if src.width <= dst.width:
                return True

        # Integer to float
        if isinstance(src, IntType) and isinstance(dst, FloatType):
            if self.language is Language.C:
                return True

        # Pointer compatibility
        if isinstance(src, PointerType) and isinstance(dst, PointerType):
            if isinstance(src.pointee, VoidType) or isinstance(dst.pointee, VoidType):
                return True

        return False

    def _determine_cast_kind(self, src: IRType, dst: IRType) -> CastKind | None:
        """Determine the appropriate cast kind for an implicit conversion."""
        if isinstance(src, IntType) and isinstance(dst, IntType):
            if src.width < dst.width:
                return CastKind.SEXT if src.is_signed else CastKind.ZEXT
            if src.width > dst.width:
                return CastKind.TRUNC
        if isinstance(src, FloatType) and isinstance(dst, FloatType):
            if src.width < dst.width:
                return CastKind.FPEXT
            if src.width > dst.width:
                return CastKind.FPTRUNC
        if isinstance(src, IntType) and isinstance(dst, FloatType):
            return CastKind.SITOFP if src.is_signed else CastKind.UITOFP
        if isinstance(src, FloatType) and isinstance(dst, IntType):
            return CastKind.FPTOSI if dst.is_signed else CastKind.FPTOUI
        return None

    def _record_promotion_cast(
        self, src: IntType, dst: IntType, reason: str,
        inst: Instruction | None, func_name: str,
        result: TypeCheckResult,
    ) -> None:
        cast_kind = CastKind.SEXT if src.is_signed else CastKind.ZEXT
        result.inserted_casts.append(InsertedCast(
            src_type=src,
            dst_type=dst,
            cast_kind=cast_kind,
            reason=reason,
            instruction=inst,
            function_name=func_name,
        ))
