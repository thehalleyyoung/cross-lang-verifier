"""
IR Validator for the Cross-Language Equivalence Verifier.

Performs structural and semantic validation on the IR including:
- SSA dominance checking
- Type consistency across instructions
- Control flow validity (terminators, reachability)
- Phi node correctness
- Module-level consistency
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Sequence

from .basic_block import BasicBlock
from .function import Function
from .module import Module
from .instructions import (
    AllocaInst,
    Argument,
    AtomicCmpXchgInst,
    AtomicRMWInst,
    BinaryOp,
    BranchInst,
    CallInst,
    CastInst,
    CompareOp,
    Constant,
    ExtractValueInst,
    FenceInst,
    GetElementPtrInst,
    InsertValueInst,
    Instruction,
    InstructionVisitor,
    LoadInst,
    MemcpyInst,
    MemsetInst,
    PhiInst,
    ReturnInst,
    SelectInst,
    StoreInst,
    SwitchInst,
    UnaryOp,
    Value,
    is_terminator,
)
from .types import (
    ArrayType,
    FloatType,
    FunctionType,
    IRType,
    IntType,
    PointerType,
    StructType,
    VoidType,
)


class Severity(Enum):
    ERROR = auto()
    WARNING = auto()
    INFO = auto()


@dataclass
class ValidationMessage:
    """A single validation finding."""
    severity: Severity
    message: str
    location: str = ""  # e.g. "function 'foo', block 'bb0', instruction %3"

    def __str__(self) -> str:
        prefix = self.severity.name
        loc = f" at {self.location}" if self.location else ""
        return f"[{prefix}]{loc}: {self.message}"


@dataclass
class ValidationResult:
    """Aggregate result of validation."""
    messages: list[ValidationMessage] = field(default_factory=list)

    def error(self, msg: str, location: str = "") -> None:
        self.messages.append(ValidationMessage(Severity.ERROR, msg, location))

    def warning(self, msg: str, location: str = "") -> None:
        self.messages.append(ValidationMessage(Severity.WARNING, msg, location))

    def info(self, msg: str, location: str = "") -> None:
        self.messages.append(ValidationMessage(Severity.INFO, msg, location))

    @property
    def errors(self) -> list[ValidationMessage]:
        return [m for m in self.messages if m.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[ValidationMessage]:
        return [m for m in self.messages if m.severity is Severity.WARNING]

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    @property
    def num_errors(self) -> int:
        return len(self.errors)

    @property
    def num_warnings(self) -> int:
        return len(self.warnings)

    def merge(self, other: "ValidationResult") -> None:
        self.messages.extend(other.messages)

    def __str__(self) -> str:
        if self.is_valid:
            w = self.num_warnings
            return f"Valid ({w} warning{'s' if w != 1 else ''})"
        return "\n".join(str(m) for m in self.messages)


class IRValidator:
    """Validates IR modules, functions, and blocks.

    Usage::

        validator = IRValidator()
        result = validator.validate_module(module)
        if not result.is_valid:
            for err in result.errors:
                print(err)
    """

    def __init__(self, strict: bool = True) -> None:
        self._strict = strict

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_module(self, module: Module) -> ValidationResult:
        """Validate an entire module."""
        result = ValidationResult()
        self._check_module_structure(module, result)
        for func in module.functions.values():
            func_result = self.validate_function(func)
            result.merge(func_result)
        return result

    def validate_function(self, func: Function) -> ValidationResult:
        """Validate a single function."""
        result = ValidationResult()
        loc_prefix = f"function '{func.name}'"

        if not func.blocks:
            result.error("Function has no basic blocks", loc_prefix)
            return result

        self._check_function_signature(func, result, loc_prefix)
        self._check_entry_block(func, result, loc_prefix)
        self._check_block_structure(func, result, loc_prefix)
        self._check_cfg_consistency(func, result, loc_prefix)
        self._check_reachability(func, result, loc_prefix)
        self._check_phi_nodes(func, result, loc_prefix)
        self._check_ssa_dominance(func, result, loc_prefix)
        self._check_type_consistency(func, result, loc_prefix)
        self._check_return_types(func, result, loc_prefix)

        return result

    def validate_block(self, block: BasicBlock) -> ValidationResult:
        """Validate a single basic block in isolation."""
        result = ValidationResult()
        loc = f"block '{block.name}'"

        if block.is_empty:
            result.error("Block is empty", loc)
            return result

        self._check_phi_placement(block, result, loc)
        self._check_terminator(block, result, loc)

        # Per-instruction validation
        for inst in block:
            inst_loc = f"{loc}, {inst.display_name}"
            for err in inst.validate():
                result.error(err, inst_loc)

        return result

    # ------------------------------------------------------------------
    # Module-level checks
    # ------------------------------------------------------------------

    def _check_module_structure(self, module: Module, result: ValidationResult) -> None:
        loc = f"module '{module.name}'"

        # Check for symbol conflicts
        func_names = set(module.functions.keys())
        global_names = set(module.globals.keys())
        conflicts = func_names & global_names
        for c in conflicts:
            result.error(f"Symbol '{c}' is both a function and a global variable", loc)

        # Validate global variables
        for gv in module.globals.values():
            gv_loc = f"{loc}, global '{gv.name}'"
            for err in gv.validate():
                result.error(err, gv_loc)

        # Validate external declarations
        for ext in module.externals.values():
            ext_loc = f"{loc}, external '{ext.name}'"
            for err in ext.validate():
                result.error(err, ext_loc)

        # Check that externals don't shadow definitions
        for ext_name in module.externals:
            if ext_name in func_names:
                result.warning(
                    f"External declaration '{ext_name}' shadows function definition",
                    loc,
                )

    # ------------------------------------------------------------------
    # Function-level checks
    # ------------------------------------------------------------------

    def _check_function_signature(
        self, func: Function, result: ValidationResult, loc: str
    ) -> None:
        ft = func.func_type
        if len(func.arguments) != len(ft.param_types):
            result.error(
                f"Argument count mismatch: function type has {len(ft.param_types)} "
                f"params but function has {len(func.arguments)} arguments",
                loc,
            )
        for i, (arg, pty) in enumerate(zip(func.arguments, ft.param_types)):
            if arg.type != pty:
                result.error(
                    f"Argument {i} type mismatch: declared {pty}, actual {arg.type}",
                    loc,
                )

    def _check_entry_block(
        self, func: Function, result: ValidationResult, loc: str
    ) -> None:
        entry = func.entry_block
        if entry is None:
            return
        if entry.num_predecessors > 0:
            result.error("Entry block has predecessors", loc)
        if entry.phi_nodes:
            result.error("Entry block contains phi nodes", loc)

    def _check_block_structure(
        self, func: Function, result: ValidationResult, loc: str
    ) -> None:
        for block in func.blocks:
            blk_loc = f"{loc}, block '{block.name}'"

            if block.is_empty:
                result.error("Block is empty", blk_loc)
                continue

            # Check terminator
            self._check_terminator(block, result, blk_loc)

            # Check no terminator in middle
            for i, inst in enumerate(block.instructions):
                if is_terminator(inst) and i < len(block) - 1:
                    result.error(
                        f"Terminator '{inst.opcode_name()}' is not the last instruction",
                        blk_loc,
                    )

            # Check phi placement
            self._check_phi_placement(block, result, blk_loc)

    def _check_phi_placement(
        self, block: BasicBlock, result: ValidationResult, loc: str
    ) -> None:
        seen_non_phi = False
        for inst in block:
            if isinstance(inst, PhiInst):
                if seen_non_phi:
                    result.error(
                        f"Phi node {inst.display_name} appears after non-phi instruction",
                        loc,
                    )
            else:
                seen_non_phi = True

    def _check_terminator(
        self, block: BasicBlock, result: ValidationResult, loc: str
    ) -> None:
        if not block.has_terminator:
            result.error("Block does not end with a terminator", loc)
            return

        term = block.terminator
        assert term is not None

        if isinstance(term, BranchInst):
            if term.is_conditional:
                cond = term.condition
                if cond is not None:
                    if not (isinstance(cond.type, IntType) and cond.type.width == 1):
                        result.error(
                            f"Branch condition must be i1, got {cond.type}", loc
                        )
        elif isinstance(term, SwitchInst):
            if not isinstance(term.condition.type, IntType):
                result.error(
                    f"Switch condition must be integer, got {term.condition.type}",
                    loc,
                )

    # ------------------------------------------------------------------
    # CFG checks
    # ------------------------------------------------------------------

    def _check_cfg_consistency(
        self, func: Function, result: ValidationResult, loc: str
    ) -> None:
        """Check that predecessor/successor edges are consistent."""
        for block in func.blocks:
            blk_loc = f"{loc}, block '{block.name}'"

            # Every successor should list this block as a predecessor
            for succ in block.successors:
                if block not in succ.predecessors:
                    result.error(
                        f"Successor '{succ.name}' does not list '{block.name}' as predecessor",
                        blk_loc,
                    )

            # Every predecessor should list this block as a successor
            for pred in block.predecessors:
                if block not in pred.successors:
                    result.error(
                        f"Predecessor '{pred.name}' does not list '{block.name}' as successor",
                        blk_loc,
                    )

    def _check_reachability(
        self, func: Function, result: ValidationResult, loc: str
    ) -> None:
        """Check that all blocks are reachable from the entry."""
        entry = func.entry_block
        if entry is None:
            return

        reachable: set[int] = set()
        queue: deque[BasicBlock] = deque([entry])
        reachable.add(entry.id)

        while queue:
            block = queue.popleft()
            for succ in block.successors:
                if succ.id not in reachable:
                    reachable.add(succ.id)
                    queue.append(succ)

        for block in func.blocks:
            if block.id not in reachable:
                if self._strict:
                    result.error(
                        f"Block '{block.name}' is unreachable from entry", loc
                    )
                else:
                    result.warning(
                        f"Block '{block.name}' is unreachable from entry", loc
                    )

    # ------------------------------------------------------------------
    # Phi node checks
    # ------------------------------------------------------------------

    def _check_phi_nodes(
        self, func: Function, result: ValidationResult, loc: str
    ) -> None:
        """Validate phi node incoming edges match predecessors."""
        for block in func.blocks:
            blk_loc = f"{loc}, block '{block.name}'"
            preds = set(id(p) for p in block.predecessors)

            for phi in block.phi_nodes:
                phi_loc = f"{blk_loc}, {phi.display_name}"
                incoming_blocks = [id(b) for b in phi.incoming_blocks]
                incoming_set = set(incoming_blocks)

                # Check for duplicate incoming blocks
                if len(incoming_blocks) != len(incoming_set):
                    result.error("Phi has duplicate incoming blocks", phi_loc)

                # Check that incoming blocks match predecessors
                missing = preds - incoming_set
                extra = incoming_set - preds
                if missing:
                    missing_names = [
                        b.name for b in block.predecessors if id(b) in missing
                    ]
                    result.error(
                        f"Phi missing incoming from predecessors: {missing_names}",
                        phi_loc,
                    )
                if extra:
                    extra_names = [
                        b.name for _, b in phi.incoming if id(b) in extra
                    ]
                    result.error(
                        f"Phi has incoming from non-predecessors: {extra_names}",
                        phi_loc,
                    )

                # Check type consistency
                for val, blk in phi.incoming:
                    if val.type != phi.type:
                        result.error(
                            f"Phi incoming value type {val.type} != phi type {phi.type} "
                            f"(from block '{blk.name}')",
                            phi_loc,
                        )

    # ------------------------------------------------------------------
    # SSA dominance
    # ------------------------------------------------------------------

    def _check_ssa_dominance(
        self, func: Function, result: ValidationResult, loc: str
    ) -> None:
        """Check that every use is dominated by its definition."""
        func.compute_dominators()

        # Build definition map: value id -> defining block
        def_block: dict[int, BasicBlock] = {}
        def_position: dict[int, int] = {}  # value id -> position within block

        entry = func.entry_block
        for arg in func.arguments:
            if entry is not None:
                def_block[arg.id] = entry
                def_position[arg.id] = -1  # before all instructions

        for block in func.blocks:
            for pos, inst in enumerate(block):
                def_block[inst.id] = block
                def_position[inst.id] = pos

        # Check each use
        for block in func.blocks:
            for pos, inst in enumerate(block):
                inst_loc = f"{loc}, block '{block.name}', {inst.display_name}"

                for op in inst.operands:
                    if isinstance(op, Constant):
                        continue
                    if isinstance(op, Argument):
                        continue

                    op_def = def_block.get(op.id)
                    if op_def is None:
                        # Value not defined in this function — could be external
                        continue

                    if isinstance(inst, PhiInst):
                        # Phi: value must dominate the incoming edge source
                        for val, pred_block in inst.incoming:
                            if val is op:
                                if not op_def.dominates(pred_block):
                                    result.error(
                                        f"SSA dominance violation: {op.display_name} "
                                        f"(defined in '{op_def.name}') does not "
                                        f"dominate phi predecessor '{pred_block.name}'",
                                        inst_loc,
                                    )
                    else:
                        if op_def is block:
                            # Same block: definition must precede use
                            op_pos = def_position.get(op.id, -1)
                            if op_pos >= pos:
                                result.error(
                                    f"SSA dominance violation: {op.display_name} "
                                    f"used before definition in same block",
                                    inst_loc,
                                )
                        elif not op_def.dominates(block):
                            result.error(
                                f"SSA dominance violation: {op.display_name} "
                                f"(defined in '{op_def.name}') does not dominate "
                                f"use in '{block.name}'",
                                inst_loc,
                            )

    # ------------------------------------------------------------------
    # Type consistency
    # ------------------------------------------------------------------

    def _check_type_consistency(
        self, func: Function, result: ValidationResult, loc: str
    ) -> None:
        """Run per-instruction type checks."""
        checker = _TypeChecker(result, loc)
        for block in func.blocks:
            for inst in block:
                checker.check(inst, block)

    # ------------------------------------------------------------------
    # Return type checks
    # ------------------------------------------------------------------

    def _check_return_types(
        self, func: Function, result: ValidationResult, loc: str
    ) -> None:
        expected_ret = func.return_type
        for block in func.exit_blocks:
            term = block.terminator
            if not isinstance(term, ReturnInst):
                continue
            blk_loc = f"{loc}, block '{block.name}'"
            if term.is_void_return:
                if not isinstance(expected_ret, VoidType):
                    result.error(
                        f"Void return in function returning {expected_ret}",
                        blk_loc,
                    )
            else:
                rv = term.return_value
                if rv is not None and rv.type != expected_ret:
                    result.error(
                        f"Return type mismatch: expected {expected_ret}, got {rv.type}",
                        blk_loc,
                    )


# ---------------------------------------------------------------------------
# Per-instruction type checker (used internally by IRValidator)
# ---------------------------------------------------------------------------

class _TypeChecker:
    """Checks type consistency of individual instructions."""

    def __init__(self, result: ValidationResult, func_loc: str) -> None:
        self._result = result
        self._func_loc = func_loc

    def check(self, inst: Instruction, block: BasicBlock) -> None:
        loc = f"{self._func_loc}, block '{block.name}', {inst.display_name}"
        # Delegate to per-instruction validate()
        for err in inst.validate():
            self._result.error(err, loc)

        # Additional cross-instruction type checks
        if isinstance(inst, BinaryOp):
            self._check_binop(inst, loc)
        elif isinstance(inst, LoadInst):
            self._check_load(inst, loc)
        elif isinstance(inst, StoreInst):
            self._check_store(inst, loc)
        elif isinstance(inst, CallInst):
            self._check_call(inst, loc)
        elif isinstance(inst, GetElementPtrInst):
            self._check_gep(inst, loc)

    def _check_binop(self, inst: BinaryOp, loc: str) -> None:
        if inst.type != inst.lhs.type:
            self._result.warning(
                f"Binary op result type {inst.type} != operand type {inst.lhs.type}",
                loc,
            )

    def _check_load(self, inst: LoadInst, loc: str) -> None:
        addr_ty = inst.address.type
        if isinstance(addr_ty, PointerType):
            if addr_ty.pointee != inst.type and not isinstance(addr_ty.pointee, VoidType):
                self._result.warning(
                    f"Load type {inst.type} != pointer pointee {addr_ty.pointee}",
                    loc,
                )

    def _check_store(self, inst: StoreInst, loc: str) -> None:
        addr_ty = inst.address.type
        if isinstance(addr_ty, PointerType):
            if addr_ty.pointee != inst.value.type and not isinstance(addr_ty.pointee, VoidType):
                self._result.warning(
                    f"Store value type {inst.value.type} != pointer pointee {addr_ty.pointee}",
                    loc,
                )

    def _check_call(self, inst: CallInst, loc: str) -> None:
        callee_ty = inst.callee.type
        if isinstance(callee_ty, PointerType):
            callee_ty = callee_ty.pointee
        if isinstance(callee_ty, FunctionType):
            if inst.type != callee_ty.return_type:
                self._result.error(
                    f"Call result type {inst.type} != callee return type {callee_ty.return_type}",
                    loc,
                )
            for i, (arg, param_ty) in enumerate(
                zip(inst.args, callee_ty.param_types)
            ):
                if arg.type != param_ty:
                    self._result.error(
                        f"Call argument {i} type {arg.type} != parameter type {param_ty}",
                        loc,
                    )

    def _check_gep(self, inst: GetElementPtrInst, loc: str) -> None:
        if not isinstance(inst.base.type, PointerType):
            self._result.error(
                f"GEP base must be pointer, got {inst.base.type}", loc
            )


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def validate_module(module: Module, strict: bool = True) -> ValidationResult:
    """Validate a module and return the result."""
    return IRValidator(strict=strict).validate_module(module)


def validate_function(func: Function, strict: bool = True) -> ValidationResult:
    """Validate a function and return the result."""
    return IRValidator(strict=strict).validate_function(func)


def assert_valid(module: Module) -> None:
    """Validate a module and raise ValueError if invalid."""
    result = validate_module(module)
    if not result.is_valid:
        raise ValueError(f"IR validation failed:\n{result}")
