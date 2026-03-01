"""
Dead Code Elimination for the Cross-Language Equivalence Verifier.

Provides:
- DeadCodeElimination: mark-sweep DCE using liveness analysis
- AggressiveDCE: assume side-effect-free unless proven otherwise
- DeadBlockElimination: remove blocks with no predecessors
- UnreachableCodeRemoval: remove code unreachable from entry
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from ...ir.function import Function
from ...ir.module import Module
from ...ir.basic_block import BasicBlock
from ...ir.instructions import (
    Instruction, Value, Constant, Argument,
    BinaryOp, UnaryOp, CompareOp,
    LoadInst, StoreInst, AllocaInst,
    GetElementPtrInst, CastInst,
    CallInst, ReturnInst, BranchInst, SwitchInst,
    PhiInst, SelectInst,
    ExtractValueInst, InsertValueInst,
    MemcpyInst, MemsetInst,
    FenceInst, AtomicRMWInst, AtomicCmpXchgInst,
)
from .pass_manager import FunctionPass, ModulePass, PassResult, AnalysisManager

logger = logging.getLogger(__name__)


def _has_side_effects(inst: Instruction) -> bool:
    """Determine if an instruction has observable side effects."""
    if isinstance(inst, (StoreInst, CallInst, ReturnInst, BranchInst, SwitchInst)):
        return True
    if isinstance(inst, (MemcpyInst, MemsetInst)):
        return True
    if isinstance(inst, (FenceInst, AtomicRMWInst, AtomicCmpXchgInst)):
        return True
    if isinstance(inst, LoadInst) and inst.is_volatile:
        return True
    return False


def _is_trivially_dead(inst: Instruction) -> bool:
    """Check if an instruction is trivially dead (no users and no side effects)."""
    if _has_side_effects(inst):
        return False
    if not hasattr(inst, 'users'):
        return False
    return len(inst.users) == 0


def _get_operands(inst: Instruction) -> List[Value]:
    """Extract all Value operands from an instruction."""
    operands: List[Value] = []
    if isinstance(inst, BinaryOp):
        operands.extend([inst.left, inst.right])
    elif isinstance(inst, UnaryOp):
        operands.append(inst.operand)
    elif isinstance(inst, CompareOp):
        operands.extend([inst.left, inst.right])
    elif isinstance(inst, LoadInst):
        operands.append(inst.address)
    elif isinstance(inst, StoreInst):
        operands.extend([inst.value, inst.address])
    elif isinstance(inst, AllocaInst):
        if inst.array_size is not None:
            operands.append(inst.array_size)
    elif isinstance(inst, GetElementPtrInst):
        operands.append(inst.base)
        operands.extend(inst.indices)
    elif isinstance(inst, CastInst):
        operands.append(inst.operand)
    elif isinstance(inst, CallInst):
        if inst.callee is not None:
            operands.append(inst.callee)
        operands.extend(inst.arguments)
    elif isinstance(inst, ReturnInst):
        if inst.value is not None:
            operands.append(inst.value)
    elif isinstance(inst, BranchInst):
        if inst.is_conditional and inst.condition is not None:
            operands.append(inst.condition)
    elif isinstance(inst, SwitchInst):
        operands.append(inst.value)
    elif isinstance(inst, PhiInst):
        for val, _ in inst.incoming:
            operands.append(val)
    elif isinstance(inst, SelectInst):
        operands.extend([inst.condition, inst.true_value, inst.false_value])
    elif isinstance(inst, ExtractValueInst):
        operands.append(inst.aggregate)
    elif isinstance(inst, InsertValueInst):
        operands.extend([inst.aggregate, inst.value])
    elif isinstance(inst, MemcpyInst):
        operands.extend([inst.dest, inst.src])
    elif isinstance(inst, MemsetInst):
        operands.extend([inst.dest, inst.value])
    return [op for op in operands if isinstance(op, Instruction)]


# ─── Mark-Sweep DCE ─────────────────────────────────────────────────────

class LivenessMarker:
    """Mark-sweep liveness computation for DCE.

    Starting from roots (instructions with side effects or that are used
    externally), mark all transitively needed instructions as live.
    """

    def __init__(self, function: Function) -> None:
        self._function = function
        self._live: Set[int] = set()
        self._worklist: List[Instruction] = []

    def compute(self) -> Set[int]:
        """Compute the set of live instruction IDs."""
        self._seed_roots()
        self._propagate()
        return self._live

    def _seed_roots(self) -> None:
        """Find root instructions that must be kept (side-effectful)."""
        for block in self._function.blocks:
            for inst in block.instructions:
                if _has_side_effects(inst):
                    self._mark_live(inst)

    def _mark_live(self, inst: Instruction) -> None:
        if inst.id not in self._live:
            self._live.add(inst.id)
            self._worklist.append(inst)

    def _propagate(self) -> None:
        """Propagate liveness backward through operands."""
        while self._worklist:
            inst = self._worklist.pop()
            for operand in _get_operands(inst):
                if isinstance(operand, Instruction):
                    self._mark_live(operand)
            # Phi nodes: mark the block's terminator as live to keep CFG intact
            if isinstance(inst, PhiInst) and inst.parent is not None:
                for _, block in inst.incoming:
                    if block.terminator is not None:
                        self._mark_live(block.terminator)


class DeadCodeElimination(FunctionPass):
    """Mark-sweep dead code elimination using liveness.

    Starts from roots (side-effectful instructions), marks all
    transitively needed instructions, and removes the rest.
    """

    _name = "dce"
    _description = "Dead code elimination using mark-sweep liveness"
    _invalidated_analyses = ["cfg", "domtree", "loops"]

    def run_on_function(self, function: Function, analyses: AnalysisManager) -> PassResult:
        marker = LivenessMarker(function)
        live_ids = marker.compute()

        dead_instructions: List[tuple[BasicBlock, Instruction]] = []
        for block in function.blocks:
            for inst in block.instructions:
                if inst.id not in live_ids:
                    dead_instructions.append((block, inst))

        if not dead_instructions:
            return PassResult.UNCHANGED

        removed = 0
        for block, inst in reversed(dead_instructions):
            self._remove_from_users(inst)
            block.remove(inst)
            removed += 1

        self.stats.instructions_removed += removed
        self.stats.increment("dead_instructions_removed", removed)
        logger.debug(f"DCE: removed {removed} dead instructions from {function.name}")
        return PassResult.CHANGED

    def _remove_from_users(self, inst: Instruction) -> None:
        """Remove inst from the user lists of its operands."""
        for operand in _get_operands(inst):
            if isinstance(operand, Instruction) and hasattr(operand, 'users'):
                try:
                    operand.users.remove(inst)
                except (ValueError, KeyError):
                    pass


# ─── Aggressive DCE ─────────────────────────────────────────────────────

class AggressiveDCE(FunctionPass):
    """Aggressive dead code elimination.

    Assumes instructions are dead unless proven to have side effects.
    More aggressive than standard DCE: also removes unused allocas,
    loads from provably-dead addresses, etc.
    """

    _name = "adce"
    _description = "Aggressive DCE assuming side-effect-free unless proven otherwise"
    _invalidated_analyses = ["cfg", "domtree", "loops"]

    def __init__(self) -> None:
        super().__init__()
        self._known_pure_calls: Set[str] = {
            "abs", "fabs", "sqrt", "sin", "cos", "tan",
            "exp", "log", "pow", "ceil", "floor", "round",
            "strlen", "strcmp", "memcmp",
        }

    def run_on_function(self, function: Function, analyses: AnalysisManager) -> PassResult:
        live: Set[int] = set()
        worklist: List[Instruction] = []

        # Seed: only truly side-effectful instructions
        for block in function.blocks:
            for inst in block.instructions:
                if self._is_essential(inst):
                    if inst.id not in live:
                        live.add(inst.id)
                        worklist.append(inst)

        # Propagate
        while worklist:
            inst = worklist.pop()
            for op in _get_operands(inst):
                if isinstance(op, Instruction) and op.id not in live:
                    live.add(op.id)
                    worklist.append(op)
            if isinstance(inst, PhiInst) and inst.parent is not None:
                for _, block in inst.incoming:
                    term = block.terminator
                    if term is not None and term.id not in live:
                        live.add(term.id)
                        worklist.append(term)

        # Sweep
        dead: List[tuple[BasicBlock, Instruction]] = []
        for block in function.blocks:
            for inst in block.instructions:
                if inst.id not in live:
                    dead.append((block, inst))

        if not dead:
            return PassResult.UNCHANGED

        removed = 0
        for block, inst in reversed(dead):
            self._detach_instruction(inst)
            block.remove(inst)
            removed += 1

        self.stats.instructions_removed += removed
        self.stats.increment("adce_removed", removed)

        # Clean up trivially unreachable blocks after aggressive removal
        self._cleanup_empty_blocks(function)

        return PassResult.CHANGED

    def _is_essential(self, inst: Instruction) -> bool:
        """Determine if an instruction is essential (must be kept)."""
        if isinstance(inst, (ReturnInst, BranchInst, SwitchInst)):
            return True
        if isinstance(inst, (StoreInst, MemcpyInst, MemsetInst)):
            return True
        if isinstance(inst, (FenceInst, AtomicRMWInst, AtomicCmpXchgInst)):
            return True
        if isinstance(inst, LoadInst) and inst.is_volatile:
            return True
        if isinstance(inst, CallInst):
            return not self._is_pure_call(inst)
        return False

    def _is_pure_call(self, call: CallInst) -> bool:
        """Check if a call is to a known pure function."""
        if hasattr(call, 'callee_name'):
            return call.callee_name in self._known_pure_calls
        return False

    def _detach_instruction(self, inst: Instruction) -> None:
        for operand in _get_operands(inst):
            if isinstance(operand, Instruction) and hasattr(operand, 'users'):
                try:
                    operand.users.remove(inst)
                except (ValueError, KeyError):
                    pass

    def _cleanup_empty_blocks(self, function: Function) -> None:
        """Remove blocks that have become empty (no instructions at all)."""
        to_remove = []
        entry = function.entry_block
        for block in function.blocks:
            if block is entry:
                continue
            if block.is_empty:
                to_remove.append(block)

        for block in to_remove:
            for pred in list(block.predecessors):
                pred.remove_predecessor(block)
            function.remove_block(block)
            self.stats.blocks_removed += 1


# ─── Dead Block Elimination ────────────────────────────────────────────

class DeadBlockElimination(FunctionPass):
    """Remove basic blocks with no predecessors (except entry).

    Iteratively removes blocks that have become unreachable due to
    other optimizations removing branches.
    """

    _name = "dead_block_elim"
    _description = "Remove unreachable basic blocks"
    _invalidated_analyses = ["cfg", "domtree", "loops"]

    def run_on_function(self, function: Function, analyses: AnalysisManager) -> PassResult:
        if function.num_blocks <= 1:
            return PassResult.UNCHANGED

        entry = function.entry_block
        if entry is None:
            return PassResult.UNCHANGED

        changed = False
        iterate = True
        total_removed = 0

        while iterate:
            iterate = False
            reachable = self._compute_reachable(entry)
            to_remove = [b for b in function.blocks if b not in reachable]

            if not to_remove:
                break

            for block in to_remove:
                self._disconnect_block(block, function)
                function.remove_block(block)
                total_removed += 1
                iterate = True
                changed = True

        if changed:
            self.stats.blocks_removed += total_removed
            self.stats.increment("dead_blocks_removed", total_removed)
            logger.debug(f"Dead block elimination: removed {total_removed} blocks from {function.name}")

        return PassResult.CHANGED if changed else PassResult.UNCHANGED

    def _compute_reachable(self, entry: BasicBlock) -> Set[BasicBlock]:
        """BFS from entry to find all reachable blocks."""
        reachable: Set[BasicBlock] = set()
        worklist = [entry]
        while worklist:
            block = worklist.pop()
            if block in reachable:
                continue
            reachable.add(block)
            for succ in block.successors:
                if succ not in reachable:
                    worklist.append(succ)
        return reachable

    def _disconnect_block(self, block: BasicBlock, function: Function) -> None:
        """Remove all edges to/from a block and update phi nodes in successors."""
        for succ in list(block.successors):
            for phi in succ.phi_nodes:
                phi.remove_incoming(block)
            succ.remove_predecessor(block)

        block._clear_successor_edges()

        for inst in list(block.instructions):
            self._detach_uses(inst)

    def _detach_uses(self, inst: Instruction) -> None:
        for op in _get_operands(inst):
            if isinstance(op, Instruction) and hasattr(op, 'users'):
                try:
                    op.users.remove(inst)
                except (ValueError, KeyError):
                    pass


# ─── Unreachable Code Removal ──────────────────────────────────────────

class UnreachableCodeRemoval(FunctionPass):
    """Remove code that is provably unreachable.

    Handles:
    - Code after unconditional branches (within a block)
    - Code after return statements
    - Blocks only reachable through proven-false conditions
    """

    _name = "unreachable_removal"
    _description = "Remove provably unreachable code"
    _invalidated_analyses = ["cfg", "domtree", "loops"]

    def run_on_function(self, function: Function, analyses: AnalysisManager) -> PassResult:
        changed = False
        result = self._remove_instructions_after_terminator(function)
        if result:
            changed = True

        result = self._remove_unreachable_branches(function)
        if result:
            changed = True

        result = self._remove_dead_blocks(function)
        if result:
            changed = True

        return PassResult.CHANGED if changed else PassResult.UNCHANGED

    def _remove_instructions_after_terminator(self, function: Function) -> bool:
        """Remove any instructions after the first terminator in a block."""
        changed = False
        for block in function.blocks:
            found_terminator = False
            to_remove = []
            for inst in block.instructions:
                if found_terminator:
                    to_remove.append(inst)
                elif isinstance(inst, (ReturnInst, BranchInst, SwitchInst)):
                    found_terminator = True

            for inst in reversed(to_remove):
                block.remove(inst)
                self.stats.instructions_removed += 1
                changed = True

        return changed

    def _remove_unreachable_branches(self, function: Function) -> bool:
        """Replace conditional branches with known-constant conditions."""
        changed = False
        for block in function.blocks:
            term = block.terminator
            if not isinstance(term, BranchInst) or not term.is_conditional:
                continue

            cond = term.condition
            if not isinstance(cond, Constant):
                continue

            # Condition is a constant: convert to unconditional branch
            if hasattr(cond, 'value'):
                taken = term.true_block if cond.value else term.false_block
                not_taken = term.false_block if cond.value else term.true_block

                # Remove phi entries in the not-taken block
                if not_taken is not None:
                    for phi in not_taken.phi_nodes:
                        phi.remove_incoming(block)
                    not_taken.remove_predecessor(block)

                # Create unconditional branch
                from ...ir.builder import IRBuilder
                block.remove(term)
                new_br = BranchInst(target=taken)
                new_br._parent = block
                block.append(new_br)
                changed = True
                self.stats.increment("branches_folded")

        return changed

    def _remove_dead_blocks(self, function: Function) -> bool:
        """Remove blocks not reachable from entry."""
        entry = function.entry_block
        if entry is None:
            return False

        reachable: Set[BasicBlock] = set()
        worklist = [entry]
        while worklist:
            b = worklist.pop()
            if b in reachable:
                continue
            reachable.add(b)
            for s in b.successors:
                worklist.append(s)

        to_remove = [b for b in function.blocks if b not in reachable]
        if not to_remove:
            return False

        for block in to_remove:
            for succ in list(block.successors):
                for phi in succ.phi_nodes:
                    phi.remove_incoming(block)
                succ.remove_predecessor(block)
            function.remove_block(block)
            self.stats.blocks_removed += 1

        return True
