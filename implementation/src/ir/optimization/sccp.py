"""
Sparse Conditional Constant Propagation for the Cross-Language Equivalence Verifier.

Implements the SCCP algorithm using a lattice (top/constant/bottom),
worklist on SSA def-use graph, phi merging, branch propagation,
and dead branch elimination.

Provides:
- SparseConditionalConstantPropagation: main SCCP pass
- LatticeValue: three-level lattice for constant tracking
- SCCPSolver: core solver engine
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple

from ...ir.function import Function
from ...ir.basic_block import BasicBlock
from ...ir.instructions import (
    Instruction, Value, Constant, Argument,
    BinaryOp, BinOpKind, UnaryOp, UnaryOpKind,
    CompareOp, CmpPredicate,
    CastInst, CastKind,
    LoadInst, StoreInst, AllocaInst,
    CallInst, ReturnInst, BranchInst, SwitchInst,
    PhiInst, SelectInst,
    GetElementPtrInst,
)
from ...ir.types import IRType, IntType, FloatType, Signedness
from .pass_manager import FunctionPass, PassResult, AnalysisManager

logger = logging.getLogger(__name__)


# ─── Lattice ───────────────────────────────────────────────────────────

class LatticeState(Enum):
    """Three-level lattice for SCCP."""
    TOP = auto()       # Undefined / not yet reached
    CONSTANT = auto()  # Known constant value
    BOTTOM = auto()    # Overdefined (multiple possible values)


class LatticeValue:
    """A value in the SCCP lattice.

    TOP → CONSTANT → BOTTOM (monotonically decreasing)

    TOP means the value hasn't been computed yet.
    CONSTANT means the value is a known constant.
    BOTTOM means the value could be anything.
    """

    __slots__ = ('_state', '_constant')

    def __init__(self, state: LatticeState = LatticeState.TOP,
                 constant: Optional[Constant] = None) -> None:
        self._state = state
        self._constant = constant

    @staticmethod
    def top() -> "LatticeValue":
        return LatticeValue(LatticeState.TOP)

    @staticmethod
    def bottom() -> "LatticeValue":
        return LatticeValue(LatticeState.BOTTOM)

    @staticmethod
    def constant(c: Constant) -> "LatticeValue":
        return LatticeValue(LatticeState.CONSTANT, c)

    @property
    def state(self) -> LatticeState:
        return self._state

    @property
    def is_top(self) -> bool:
        return self._state == LatticeState.TOP

    @property
    def is_constant(self) -> bool:
        return self._state == LatticeState.CONSTANT

    @property
    def is_bottom(self) -> bool:
        return self._state == LatticeState.BOTTOM

    @property
    def constant_value(self) -> Optional[Constant]:
        return self._constant

    def get_int_value(self) -> Optional[int]:
        if self._constant is not None and hasattr(self._constant, 'value'):
            v = self._constant.value
            if isinstance(v, (int, bool)):
                return int(v)
        return None

    def meet(self, other: "LatticeValue") -> "LatticeValue":
        """Compute the meet (greatest lower bound) of two lattice values."""
        if self.is_top:
            return other
        if other.is_top:
            return self
        if self.is_bottom or other.is_bottom:
            return LatticeValue.bottom()
        # Both constant: check if same
        if (self._constant is not None and other._constant is not None and
                hasattr(self._constant, 'value') and hasattr(other._constant, 'value') and
                self._constant.value == other._constant.value):
            return self
        return LatticeValue.bottom()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LatticeValue):
            return NotImplemented
        if self._state != other._state:
            return False
        if self._state == LatticeState.CONSTANT:
            return (self._constant is not None and other._constant is not None and
                    hasattr(self._constant, 'value') and hasattr(other._constant, 'value') and
                    self._constant.value == other._constant.value)
        return True

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __str__(self) -> str:
        if self.is_top:
            return "⊤"
        elif self.is_bottom:
            return "⊥"
        else:
            return f"C({self._constant.value if self._constant and hasattr(self._constant, 'value') else '?'})"

    def __repr__(self) -> str:
        return f"LatticeValue({self._state.name}, {self._constant})"


# ─── SCCP Solver ──────────────────────────────────────────────────────

class SCCPSolver:
    """Core SCCP solver using dual worklist algorithm.

    Maintains two worklists:
    1. SSA worklist: instructions whose inputs have changed
    2. CFG worklist: blocks that have become executable

    The algorithm proceeds by processing items from either worklist,
    evaluating instructions, and propagating results through the
    SSA def-use graph and CFG.
    """

    def __init__(self, function: Function) -> None:
        self._function = function
        self._lattice: Dict[int, LatticeValue] = {}
        self._executable_edges: Set[Tuple[int, int]] = set()
        self._executable_blocks: Set[int] = set()
        self._ssa_worklist: List[Instruction] = []
        self._cfg_worklist: List[BasicBlock] = []
        self._block_map: Dict[int, BasicBlock] = {}
        self._inst_map: Dict[int, Instruction] = {}

    @property
    def lattice(self) -> Dict[int, LatticeValue]:
        return self._lattice

    @property
    def executable_blocks(self) -> Set[int]:
        return self._executable_blocks

    def get_value(self, val: Value) -> LatticeValue:
        """Get the lattice value for any IR value."""
        if isinstance(val, Constant):
            return LatticeValue.constant(val)
        vid = val.id if hasattr(val, 'id') else id(val)
        return self._lattice.get(vid, LatticeValue.top())

    def _set_value(self, val: Value, lv: LatticeValue) -> bool:
        """Set lattice value, returning True if it changed."""
        vid = val.id if hasattr(val, 'id') else id(val)
        old = self._lattice.get(vid, LatticeValue.top())
        if old == lv:
            return False
        self._lattice[vid] = lv
        return True

    def solve(self) -> None:
        """Run the SCCP algorithm to completion."""
        # Initialize
        self._block_map = {b.id: b for b in self._function.blocks}
        for block in self._function.blocks:
            for inst in block.instructions:
                self._inst_map[inst.id] = inst

        # Arguments are BOTTOM (unknown input)
        for arg in self._function.arguments:
            self._lattice[arg.id] = LatticeValue.bottom()

        # Start from entry block
        entry = self._function.entry_block
        if entry is None:
            return
        self._cfg_worklist.append(entry)

        # Main loop
        max_iterations = 50000
        iteration = 0

        while (self._ssa_worklist or self._cfg_worklist) and iteration < max_iterations:
            iteration += 1

            # Process CFG worklist
            if self._cfg_worklist:
                block = self._cfg_worklist.pop()
                self._visit_block(block)
                continue

            # Process SSA worklist
            if self._ssa_worklist:
                inst = self._ssa_worklist.pop()
                self._visit_instruction(inst)

        if iteration >= max_iterations:
            logger.warning("SCCP solver hit iteration limit")

    def _visit_block(self, block: BasicBlock) -> None:
        """Visit a basic block: evaluate all instructions."""
        first_visit = block.id not in self._executable_blocks
        self._executable_blocks.add(block.id)

        if first_visit:
            # Evaluate all instructions
            for inst in block.instructions:
                self._visit_instruction(inst)
        else:
            # Only re-evaluate phi nodes (CFG edge change)
            for phi in block.phi_nodes:
                self._visit_phi(phi)

    def _visit_instruction(self, inst: Instruction) -> None:
        """Evaluate a single instruction and propagate results."""
        if inst.parent is None or inst.parent.id not in self._executable_blocks:
            return

        old = self.get_value(inst)
        if old.is_bottom:
            return  # Already overdefined, can't improve

        new = self._evaluate(inst)
        if self._set_value(inst, new):
            # Value changed: add users to SSA worklist
            if hasattr(inst, 'users'):
                for user in inst.users:
                    if isinstance(user, Instruction):
                        self._ssa_worklist.append(user)

            # If this is a terminator, propagate control flow
            if isinstance(inst, (BranchInst, SwitchInst)):
                self._propagate_cf(inst)

    def _visit_phi(self, phi: PhiInst) -> None:
        """Evaluate a phi node considering only executable incoming edges."""
        if phi.parent is None or phi.parent.id not in self._executable_blocks:
            return

        old = self.get_value(phi)
        if old.is_bottom:
            return

        result = LatticeValue.top()
        for val, block in phi.incoming:
            edge = (block.id, phi.parent.id)
            if edge not in self._executable_edges:
                continue
            incoming = self.get_value(val)
            result = result.meet(incoming)
            if result.is_bottom:
                break

        if self._set_value(phi, result):
            if hasattr(phi, 'users'):
                for user in phi.users:
                    if isinstance(user, Instruction):
                        self._ssa_worklist.append(user)

    def _evaluate(self, inst: Instruction) -> LatticeValue:
        """Evaluate an instruction to produce a lattice value."""
        if isinstance(inst, PhiInst):
            return self._eval_phi(inst)
        elif isinstance(inst, BinaryOp):
            return self._eval_binop(inst)
        elif isinstance(inst, UnaryOp):
            return self._eval_unaryop(inst)
        elif isinstance(inst, CompareOp):
            return self._eval_cmp(inst)
        elif isinstance(inst, CastInst):
            return self._eval_cast(inst)
        elif isinstance(inst, SelectInst):
            return self._eval_select(inst)
        elif isinstance(inst, (LoadInst, CallInst)):
            return LatticeValue.bottom()
        elif isinstance(inst, AllocaInst):
            return LatticeValue.bottom()
        elif isinstance(inst, GetElementPtrInst):
            return LatticeValue.bottom()
        elif isinstance(inst, (ReturnInst, BranchInst, SwitchInst, StoreInst)):
            return LatticeValue.bottom()
        return LatticeValue.bottom()

    def _eval_phi(self, phi: PhiInst) -> LatticeValue:
        result = LatticeValue.top()
        for val, block in phi.incoming:
            edge = (block.id, phi.parent.id) if phi.parent else None
            if edge is not None and edge not in self._executable_edges:
                continue
            result = result.meet(self.get_value(val))
            if result.is_bottom:
                break
        return result

    def _eval_binop(self, inst: BinaryOp) -> LatticeValue:
        lv = self.get_value(inst.left)
        rv = self.get_value(inst.right)

        if lv.is_bottom or rv.is_bottom:
            # Special cases where one overdefined operand still yields constant
            if inst.op == BinOpKind.MUL:
                if (lv.is_constant and lv.get_int_value() == 0) or \
                   (rv.is_constant and rv.get_int_value() == 0):
                    return LatticeValue.constant(
                        Constant(value=0, ir_type=inst.ir_type))
            if inst.op == BinOpKind.AND:
                if (lv.is_constant and lv.get_int_value() == 0) or \
                   (rv.is_constant and rv.get_int_value() == 0):
                    return LatticeValue.constant(
                        Constant(value=0, ir_type=inst.ir_type))
            return LatticeValue.bottom()

        if lv.is_top or rv.is_top:
            return LatticeValue.top()

        # Both constant
        li = lv.get_int_value()
        ri = rv.get_int_value()
        if li is not None and ri is not None and isinstance(inst.ir_type, IntType):
            result = self._compute_int_binop(inst.op, li, ri, inst.ir_type)
            if result is not None:
                return LatticeValue.constant(
                    Constant(value=result, ir_type=inst.ir_type))

        return LatticeValue.bottom()

    def _eval_unaryop(self, inst: UnaryOp) -> LatticeValue:
        ov = self.get_value(inst.operand)
        if ov.is_bottom:
            return LatticeValue.bottom()
        if ov.is_top:
            return LatticeValue.top()

        iv = ov.get_int_value()
        if iv is not None and isinstance(inst.ir_type, IntType):
            if inst.op == UnaryOpKind.NEG:
                width = inst.ir_type.width
                return LatticeValue.constant(
                    Constant(value=(-iv) & ((1 << width) - 1), ir_type=inst.ir_type))
            elif inst.op == UnaryOpKind.NOT:
                width = inst.ir_type.width
                mask = (1 << width) - 1
                return LatticeValue.constant(
                    Constant(value=iv ^ mask, ir_type=inst.ir_type))

        return LatticeValue.bottom()

    def _eval_cmp(self, inst: CompareOp) -> LatticeValue:
        lv = self.get_value(inst.left)
        rv = self.get_value(inst.right)

        if lv.is_bottom or rv.is_bottom:
            return LatticeValue.bottom()
        if lv.is_top or rv.is_top:
            return LatticeValue.top()

        li = lv.get_int_value()
        ri = rv.get_int_value()
        if li is not None and ri is not None:
            src_type = inst.left.ir_type if hasattr(inst.left, 'ir_type') else None
            if isinstance(src_type, IntType):
                result = self._compute_int_cmp(inst.predicate, li, ri, src_type)
                if result is not None:
                    bool_type = IntType(width=1, signedness=Signedness.UNSIGNED)
                    return LatticeValue.constant(
                        Constant(value=1 if result else 0, ir_type=bool_type))

        return LatticeValue.bottom()

    def _eval_cast(self, inst: CastInst) -> LatticeValue:
        ov = self.get_value(inst.operand)
        if ov.is_bottom:
            return LatticeValue.bottom()
        if ov.is_top:
            return LatticeValue.top()

        iv = ov.get_int_value()
        if iv is not None:
            result = self._compute_cast(inst.cast_kind, iv,
                                         inst.operand.ir_type if hasattr(inst.operand, 'ir_type') else None,
                                         inst.ir_type)
            if result is not None:
                return LatticeValue.constant(
                    Constant(value=result, ir_type=inst.ir_type))

        return LatticeValue.bottom()

    def _eval_select(self, inst: SelectInst) -> LatticeValue:
        cv = self.get_value(inst.condition)
        tv = self.get_value(inst.true_value)
        fv = self.get_value(inst.false_value)

        if cv.is_constant:
            ci = cv.get_int_value()
            if ci is not None:
                return tv if ci != 0 else fv

        if cv.is_bottom:
            return tv.meet(fv)

        return LatticeValue.top()

    def _propagate_cf(self, inst: Instruction) -> None:
        """Propagate control flow from a terminator instruction."""
        if inst.parent is None:
            return
        block = inst.parent

        if isinstance(inst, BranchInst):
            if not inst.is_conditional:
                target = inst.target if hasattr(inst, 'target') else inst.true_block
                if target is not None:
                    self._mark_edge_executable(block, target)
            else:
                cv = self.get_value(inst.condition)
                if cv.is_constant:
                    ci = cv.get_int_value()
                    if ci is not None:
                        taken = inst.true_block if ci != 0 else inst.false_block
                        if taken is not None:
                            self._mark_edge_executable(block, taken)
                        return
                # Unknown: both edges executable
                if inst.true_block is not None:
                    self._mark_edge_executable(block, inst.true_block)
                if inst.false_block is not None:
                    self._mark_edge_executable(block, inst.false_block)

        elif isinstance(inst, SwitchInst):
            sv = self.get_value(inst.value)
            if sv.is_constant:
                si = sv.get_int_value()
                if si is not None and hasattr(inst, 'cases'):
                    for case_val, case_block in inst.cases:
                        cv = case_val.value if isinstance(case_val, Constant) and hasattr(case_val, 'value') else None
                        if cv == si:
                            self._mark_edge_executable(block, case_block)
                            return
                    if hasattr(inst, 'default_block') and inst.default_block is not None:
                        self._mark_edge_executable(block, inst.default_block)
                    return
            # Unknown: all edges executable
            for succ in block.successors:
                self._mark_edge_executable(block, succ)

        elif isinstance(inst, ReturnInst):
            pass

    def _mark_edge_executable(self, src: BasicBlock, dst: BasicBlock) -> None:
        edge = (src.id, dst.id)
        if edge in self._executable_edges:
            return
        self._executable_edges.add(edge)
        self._cfg_worklist.append(dst)

    def _compute_int_binop(self, op: BinOpKind, left: int, right: int,
                            typ: IntType) -> Optional[int]:
        width = typ.width
        mask = (1 << width) - 1
        signed = typ.signedness == Signedness.SIGNED

        def to_signed(v: int) -> int:
            v = v & mask
            return v - (1 << width) if v >= (1 << (width - 1)) else v

        l = to_signed(left) if signed else (left & mask)
        r = to_signed(right) if signed else (right & mask)

        if op == BinOpKind.ADD:
            return (l + r) & mask
        elif op == BinOpKind.SUB:
            return (l - r) & mask
        elif op == BinOpKind.MUL:
            return (l * r) & mask
        elif op == BinOpKind.SDIV:
            return None if r == 0 else (int(l / r) & mask)
        elif op == BinOpKind.UDIV:
            ul, ur = left & mask, right & mask
            return None if ur == 0 else ((ul // ur) & mask)
        elif op == BinOpKind.SREM:
            return None if r == 0 else ((l - int(l / r) * r) & mask)
        elif op == BinOpKind.UREM:
            ul, ur = left & mask, right & mask
            return None if ur == 0 else ((ul % ur) & mask)
        elif op == BinOpKind.SHL:
            shift = right & mask
            return 0 if shift >= width else ((left & mask) << shift) & mask
        elif op == BinOpKind.LSHR:
            shift = right & mask
            return 0 if shift >= width else ((left & mask) >> shift) & mask
        elif op == BinOpKind.ASHR:
            shift = right & mask
            return mask if shift >= width and l < 0 else (l >> shift) & mask
        elif op == BinOpKind.AND:
            return ((left & mask) & (right & mask)) & mask
        elif op == BinOpKind.OR:
            return ((left & mask) | (right & mask)) & mask
        elif op == BinOpKind.XOR:
            return ((left & mask) ^ (right & mask)) & mask
        return None

    def _compute_int_cmp(self, pred: CmpPredicate, left: int, right: int,
                          typ: IntType) -> Optional[bool]:
        width = typ.width
        mask = (1 << width) - 1

        def to_signed(v: int) -> int:
            v = v & mask
            return v - (1 << width) if v >= (1 << (width - 1)) else v

        sl, sr = to_signed(left), to_signed(right)
        ul, ur = left & mask, right & mask

        cmp_map = {
            CmpPredicate.EQ: ul == ur,
            CmpPredicate.NE: ul != ur,
            CmpPredicate.SLT: sl < sr,
            CmpPredicate.SLE: sl <= sr,
            CmpPredicate.SGT: sl > sr,
            CmpPredicate.SGE: sl >= sr,
            CmpPredicate.ULT: ul < ur,
            CmpPredicate.ULE: ul <= ur,
            CmpPredicate.UGT: ul > ur,
            CmpPredicate.UGE: ul >= ur,
        }
        return cmp_map.get(pred)

    def _compute_cast(self, kind: CastKind, value: int,
                       src_type: Optional[IRType], dst_type: IRType) -> Optional[int]:
        if not isinstance(dst_type, IntType):
            return None
        dst_mask = (1 << dst_type.width) - 1

        if kind == CastKind.ZEXT:
            if isinstance(src_type, IntType):
                return (value & ((1 << src_type.width) - 1)) & dst_mask
        elif kind == CastKind.SEXT:
            if isinstance(src_type, IntType):
                src_mask = (1 << src_type.width) - 1
                v = value & src_mask
                if v >= (1 << (src_type.width - 1)):
                    v -= (1 << src_type.width)
                return v & dst_mask
        elif kind == CastKind.TRUNC:
            return value & dst_mask
        return None


# ─── SCCP Pass ────────────────────────────────────────────────────────

class SparseConditionalConstantPropagation(FunctionPass):
    """Sparse Conditional Constant Propagation pass.

    Uses the SCCP algorithm to simultaneously propagate constants
    and determine executable edges, then:
    1. Replaces instructions with known constant values
    2. Folds constant-condition branches to unconditional
    3. Removes non-executable blocks
    """

    _name = "sccp"
    _description = "Sparse conditional constant propagation"
    _invalidated_analyses = ["cfg", "domtree", "loops"]

    def run_on_function(self, function: Function, analyses: AnalysisManager) -> PassResult:
        solver = SCCPSolver(function)
        solver.solve()

        changed = False

        # Phase 1: Replace instructions with constants
        for block in list(function.blocks):
            if block.id not in solver.executable_blocks:
                continue
            for inst in list(block.instructions):
                lv = solver.get_value(inst)
                if lv.is_constant and lv.constant_value is not None:
                    if self._can_replace(inst):
                        self._replace_with_constant(inst, lv.constant_value, block)
                        changed = True

        # Phase 2: Fold constant branches
        for block in list(function.blocks):
            if block.id not in solver.executable_blocks:
                continue
            term = block.terminator
            if isinstance(term, BranchInst) and term.is_conditional:
                cv = solver.get_value(term.condition)
                if cv.is_constant:
                    ci = cv.get_int_value()
                    if ci is not None:
                        self._fold_branch(block, term, ci != 0)
                        changed = True

        # Phase 3: Remove non-executable blocks
        entry = function.entry_block
        for block in list(function.blocks):
            if block is entry:
                continue
            if block.id not in solver.executable_blocks:
                self._remove_block(block, function)
                changed = True

        if changed:
            self.stats.increment("sccp_constants_found")
            logger.debug(f"SCCP: propagated constants in {function.name}")

        return PassResult.CHANGED if changed else PassResult.UNCHANGED

    def _can_replace(self, inst: Instruction) -> bool:
        """Check if an instruction can be replaced with a constant."""
        if isinstance(inst, (ReturnInst, BranchInst, SwitchInst, StoreInst)):
            return False
        if isinstance(inst, (CallInst, LoadInst)):
            return False
        if isinstance(inst, PhiInst):
            return True
        return True

    def _replace_with_constant(self, inst: Instruction, const: Constant,
                                block: BasicBlock) -> None:
        """Replace all uses of inst with const, then remove inst."""
        if hasattr(inst, 'users'):
            for user in list(inst.users):
                if isinstance(user, Instruction):
                    self._substitute(user, inst, const)

        try:
            block.remove(inst)
            self.stats.instructions_removed += 1
        except ValueError:
            pass

    def _fold_branch(self, block: BasicBlock, br: BranchInst, taken: bool) -> None:
        """Replace a conditional branch with an unconditional one."""
        target = br.true_block if taken else br.false_block
        not_taken = br.false_block if taken else br.true_block

        if not_taken is not None:
            for phi in not_taken.phi_nodes:
                phi.remove_incoming(block)
            not_taken.remove_predecessor(block)

        block.remove(br)
        new_br = BranchInst(target=target)
        new_br._parent = block
        block.append(new_br)
        self.stats.increment("branches_folded")

    def _remove_block(self, block: BasicBlock, function: Function) -> None:
        """Remove a non-executable block."""
        for succ in list(block.successors):
            for phi in succ.phi_nodes:
                phi.remove_incoming(block)
            succ.remove_predecessor(block)

        try:
            function.remove_block(block)
            self.stats.blocks_removed += 1
        except ValueError:
            pass

    def _substitute(self, user: Instruction, old: Value, new: Value) -> None:
        if isinstance(user, BinaryOp):
            if user.left is old:
                user.left = new
            if user.right is old:
                user.right = new
        elif isinstance(user, UnaryOp):
            if user.operand is old:
                user.operand = new
        elif isinstance(user, CompareOp):
            if user.left is old:
                user.left = new
            if user.right is old:
                user.right = new
        elif isinstance(user, CastInst):
            if user.operand is old:
                user.operand = new
        elif isinstance(user, SelectInst):
            if user.condition is old:
                user.condition = new
            if user.true_value is old:
                user.true_value = new
            if user.false_value is old:
                user.false_value = new
        elif isinstance(user, PhiInst):
            user.incoming = [(new if v is old else v, b) for v, b in user.incoming]
        elif isinstance(user, BranchInst) and user.is_conditional:
            if user.condition is old:
                user.condition = new
        elif isinstance(user, ReturnInst):
            if user.value is old:
                user.value = new
        elif isinstance(user, StoreInst):
            if user.value is old:
                user.value = new
        elif isinstance(user, CallInst):
            user.arguments = [new if a is old else a for a in user.arguments]
