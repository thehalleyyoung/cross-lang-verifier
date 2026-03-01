"""
Loop-Invariant Code Motion for the Cross-Language Equivalence Verifier.

Identifies loop-invariant instructions and hoists them to the loop
preheader, or sinks them below the loop when beneficial.

Provides:
- LoopInvariantCodeMotion: main LICM pass
- LoopInvariantAnalysis: analyze which instructions are loop-invariant
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from ...ir.function import Function
from ...ir.basic_block import BasicBlock
from ...ir.instructions import (
    Instruction, Value, Constant, Argument,
    BinaryOp, UnaryOp, CompareOp,
    LoadInst, StoreInst, AllocaInst,
    GetElementPtrInst, CastInst,
    CallInst, ReturnInst, BranchInst, SwitchInst,
    PhiInst, SelectInst,
    MemcpyInst, MemsetInst,
    FenceInst, AtomicRMWInst, AtomicCmpXchgInst,
)
from ...ir.types import IRType
from .pass_manager import FunctionPass, PassResult, AnalysisManager

logger = logging.getLogger(__name__)


# ─── Loop Invariant Analysis ──────────────────────────────────────────

@dataclass
class LoopDescriptor:
    """Describes a natural loop for LICM analysis."""
    header: BasicBlock
    body: Set[int]  # block IDs
    exits: Set[int]  # block IDs of exit blocks
    preheader: Optional[BasicBlock] = None
    latch: Optional[BasicBlock] = None
    depth: int = 0

    def contains_block(self, block: BasicBlock) -> bool:
        return block.id in self.body

    def contains_inst(self, inst: Instruction) -> bool:
        return inst.parent is not None and inst.parent.id in self.body


class LoopInvariantAnalysis:
    """Analyze which instructions in a loop are loop-invariant.

    An instruction is loop-invariant if:
    1. It has no side effects (not a store, call, etc.)
    2. All its operands are either:
       a. Constants or arguments
       b. Defined outside the loop
       c. Themselves loop-invariant (fixed-point computation)
    """

    def __init__(self, loop: LoopDescriptor) -> None:
        self._loop = loop
        self._invariant: Set[int] = set()
        self._not_invariant: Set[int] = set()

    @property
    def invariant_instructions(self) -> Set[int]:
        return self._invariant

    def analyze(self, function: Function) -> Set[int]:
        """Compute the set of loop-invariant instruction IDs."""
        self._invariant.clear()
        self._not_invariant.clear()

        # Iterate until fixed point
        changed = True
        while changed:
            changed = False
            for block in function.blocks:
                if block.id not in self._loop.body:
                    continue
                for inst in block.instructions:
                    if inst.id in self._invariant or inst.id in self._not_invariant:
                        continue
                    if self._is_invariant(inst):
                        self._invariant.add(inst.id)
                        changed = True
                    elif self._is_definitely_not_invariant(inst):
                        self._not_invariant.add(inst.id)

        return self._invariant

    def _is_invariant(self, inst: Instruction) -> bool:
        """Check if an instruction is loop-invariant."""
        # Side-effectful instructions are never invariant
        if self._has_side_effects(inst):
            return False

        # Phi nodes at loop header are not invariant
        if isinstance(inst, PhiInst) and inst.parent is self._loop.header:
            return False

        # Check all operands
        operands = self._get_operands(inst)
        for op in operands:
            if not self._operand_is_invariant(op):
                return False

        return True

    def _is_definitely_not_invariant(self, inst: Instruction) -> bool:
        """Check if an instruction can never be invariant."""
        if self._has_side_effects(inst):
            return True
        if isinstance(inst, PhiInst) and inst.parent is self._loop.header:
            return True
        return False

    def _operand_is_invariant(self, op: Value) -> bool:
        """Check if an operand value is loop-invariant."""
        if isinstance(op, (Constant, Argument)):
            return True
        if isinstance(op, Instruction):
            # Defined outside loop
            if op.parent is not None and op.parent.id not in self._loop.body:
                return True
            # Already determined to be invariant
            if op.id in self._invariant:
                return True
        return False

    def _has_side_effects(self, inst: Instruction) -> bool:
        if isinstance(inst, (StoreInst, CallInst, ReturnInst, BranchInst, SwitchInst)):
            return True
        if isinstance(inst, (MemcpyInst, MemsetInst)):
            return True
        if isinstance(inst, (FenceInst, AtomicRMWInst, AtomicCmpXchgInst)):
            return True
        if isinstance(inst, LoadInst) and getattr(inst, 'is_volatile', False):
            return True
        return False

    def _get_operands(self, inst: Instruction) -> List[Value]:
        ops: List[Value] = []
        if isinstance(inst, BinaryOp):
            ops.extend([inst.left, inst.right])
        elif isinstance(inst, UnaryOp):
            ops.append(inst.operand)
        elif isinstance(inst, CompareOp):
            ops.extend([inst.left, inst.right])
        elif isinstance(inst, CastInst):
            ops.append(inst.operand)
        elif isinstance(inst, SelectInst):
            ops.extend([inst.condition, inst.true_value, inst.false_value])
        elif isinstance(inst, LoadInst):
            ops.append(inst.address)
        elif isinstance(inst, GetElementPtrInst):
            ops.append(inst.base)
            ops.extend(inst.indices)
        elif isinstance(inst, PhiInst):
            for val, _ in inst.incoming:
                ops.append(val)
        return ops


# ─── Preheader Management ─────────────────────────────────────────────

class PreheaderManager:
    """Create and manage loop preheaders."""

    @staticmethod
    def ensure_preheader(loop: LoopDescriptor, function: Function) -> BasicBlock:
        """Ensure the loop has a preheader block. Create one if needed.

        A preheader is a single-entry block that is the only predecessor
        of the loop header coming from outside the loop.
        """
        if loop.preheader is not None:
            return loop.preheader

        header = loop.header
        outside_preds = [p for p in header.predecessors if p.id not in loop.body]

        if len(outside_preds) == 1:
            # Check if it's a valid preheader (only successor is header)
            pred = outside_preds[0]
            if len(pred.successors) == 1 and pred.successors[0] is header:
                loop.preheader = pred
                return pred

        # Create a new preheader block
        preheader = BasicBlock(name=f"preheader.{header.name}")
        preheader._parent = function

        # Add branch from preheader to header
        br = BranchInst(target=header)
        br._parent = preheader
        preheader.append(br)

        # Redirect outside predecessors to go through preheader
        for pred in outside_preds:
            PreheaderManager._redirect_edge(pred, header, preheader)
            preheader.add_predecessor(pred)

        # Update header's predecessors
        for pred in outside_preds:
            header.remove_predecessor(pred)
        header.add_predecessor(preheader)

        # Update phi nodes in header
        for phi in header.phi_nodes:
            outside_incoming = [(v, b) for v, b in phi.incoming if b in outside_preds]
            inside_incoming = [(v, b) for v, b in phi.incoming if b not in outside_preds]

            if len(outside_incoming) == 1:
                # Single outside predecessor: just change the block
                phi.incoming = inside_incoming + [(outside_incoming[0][0], preheader)]
            elif len(outside_incoming) > 1:
                # Multiple outside predecessors: create phi in preheader
                pre_phi = PhiInst(
                    ir_type=phi.ir_type,
                    incoming=outside_incoming,
                    name=f"preheader.{phi.name}" if phi.name else "preheader.phi",
                )
                pre_phi._parent = preheader
                preheader.add_phi(pre_phi)
                phi.incoming = inside_incoming + [(pre_phi, preheader)]
            else:
                phi.incoming = inside_incoming

        # Insert preheader before header in function
        function.insert_block_before(header, preheader)

        loop.preheader = preheader
        return preheader

    @staticmethod
    def _redirect_edge(src: BasicBlock, old_dst: BasicBlock,
                        new_dst: BasicBlock) -> None:
        """Redirect an edge from src→old_dst to src→new_dst."""
        term = src.terminator
        if term is None:
            return

        if isinstance(term, BranchInst):
            if term.is_conditional:
                if term.true_block is old_dst:
                    term.true_block = new_dst
                if term.false_block is old_dst:
                    term.false_block = new_dst
            else:
                target = term.target if hasattr(term, 'target') else term.true_block
                if target is old_dst:
                    if hasattr(term, 'target'):
                        term.target = new_dst
                    else:
                        term.true_block = new_dst

        elif isinstance(term, SwitchInst):
            if hasattr(term, 'default_block') and term.default_block is old_dst:
                term.default_block = new_dst
            if hasattr(term, 'cases'):
                for i, (val, block) in enumerate(term.cases):
                    if block is old_dst:
                        term.cases[i] = (val, new_dst)


# ─── Safety Analysis ─────────────────────────────────────────────────

class HoistSafetyChecker:
    """Check if it's safe to hoist an instruction out of a loop.

    Hoisting is safe if:
    1. The instruction has no side effects
    2. The instruction dominates all loop exits (guarantees execution)
    3. Moving it won't change semantics (no exception risk, etc.)
    """

    def __init__(self, loop: LoopDescriptor) -> None:
        self._loop = loop
        self._exit_blocks: Set[int] = loop.exits

    def is_safe_to_hoist(self, inst: Instruction) -> bool:
        """Check if it's safe to hoist an instruction to the preheader."""
        # Never hoist side-effectful instructions
        if isinstance(inst, (StoreInst, CallInst, MemcpyInst, MemsetInst,
                             FenceInst, AtomicRMWInst, AtomicCmpXchgInst)):
            return False

        if isinstance(inst, (ReturnInst, BranchInst, SwitchInst, PhiInst)):
            return False

        # Volatile loads can't be hoisted
        if isinstance(inst, LoadInst) and getattr(inst, 'is_volatile', False):
            return False

        # Check that instruction dominates all exits
        if inst.parent is not None:
            if not self._dominates_exits(inst.parent):
                # Non-dominating instructions can still be hoisted if they
                # are guaranteed not to trap/fault
                if not self._is_speculatable(inst):
                    return False

        return True

    def _dominates_exits(self, block: BasicBlock) -> bool:
        """Check if a block dominates all loop exit blocks."""
        for exit_id in self._exit_blocks:
            # Simple check: if the block is the header, it dominates all
            if block is self._loop.header:
                return True
            # Would need proper dominator tree for precise check
            # Conservative: only hoist from header or its direct dominators
        return block is self._loop.header

    def _is_speculatable(self, inst: Instruction) -> bool:
        """Check if an instruction can be speculatively executed safely."""
        if isinstance(inst, (BinaryOp, UnaryOp, CompareOp, SelectInst)):
            # Check for potential div-by-zero
            if isinstance(inst, BinaryOp) and inst.op in (
                    BinOpKind.SDIV, BinOpKind.UDIV, BinOpKind.SREM, BinOpKind.UREM):
                return False
            return True
        if isinstance(inst, CastInst):
            return True
        if isinstance(inst, GetElementPtrInst):
            return True
        return False


# ─── LICM Pass ────────────────────────────────────────────────────────

class LoopInvariantCodeMotion(FunctionPass):
    """Loop-Invariant Code Motion optimization pass.

    For each loop (innermost first):
    1. Ensure a preheader exists
    2. Identify loop-invariant instructions
    3. Check hoisting safety
    4. Move invariant instructions to preheader
    5. Optionally sink cold code below loop
    """

    _name = "licm"
    _description = "Hoist loop-invariant computations to preheader"
    _required_analyses = ["loops", "domtree"]
    _invalidated_analyses = ["cfg", "domtree"]

    def __init__(self) -> None:
        super().__init__()
        self._hoisted = 0
        self._sunk = 0

    def run_on_function(self, function: Function, analyses: AnalysisManager) -> PassResult:
        self._hoisted = 0
        self._sunk = 0

        # Get loop info
        loops = self._find_loops(function)
        if not loops:
            return PassResult.UNCHANGED

        # Process innermost loops first
        loops.sort(key=lambda l: l.depth, reverse=True)

        changed = False
        for loop in loops:
            if self._process_loop(loop, function):
                changed = True

        if changed:
            self.stats.increment("licm_hoisted", self._hoisted)
            self.stats.increment("licm_sunk", self._sunk)
            logger.debug(f"LICM: hoisted {self._hoisted}, sunk {self._sunk} in {function.name}")

        return PassResult.CHANGED if changed else PassResult.UNCHANGED

    def _find_loops(self, function: Function) -> List[LoopDescriptor]:
        """Find all natural loops in the function."""
        loops: List[LoopDescriptor] = []

        try:
            from ...analysis.cfg import CFG, LoopInfo
            cfg = CFG(function)
            loop_info = LoopInfo.build(cfg)

            for natural_loop in loop_info.loops:
                body_ids = {b.id for b in natural_loop.body}
                exit_ids = {b.id for b in natural_loop.exits}

                desc = LoopDescriptor(
                    header=natural_loop.header,
                    body=body_ids,
                    exits=exit_ids,
                    depth=natural_loop.depth,
                )

                # Find latch (back edge source)
                for block_id in body_ids:
                    for block in function.blocks:
                        if block.id == block_id:
                            for succ in block.successors:
                                if succ is natural_loop.header:
                                    desc.latch = block
                            break

                loops.append(desc)

        except (ImportError, Exception) as e:
            logger.debug(f"Falling back to simple loop detection: {e}")
            loops = self._simple_loop_detection(function)

        return loops

    def _simple_loop_detection(self, function: Function) -> List[LoopDescriptor]:
        """Simple loop detection using back edge identification."""
        loops: List[LoopDescriptor] = []

        # Find back edges using DFS
        visited: Set[int] = set()
        in_stack: Set[int] = set()
        back_edges: List[Tuple[BasicBlock, BasicBlock]] = []

        def dfs(block: BasicBlock) -> None:
            visited.add(block.id)
            in_stack.add(block.id)
            for succ in block.successors:
                if succ.id not in visited:
                    dfs(succ)
                elif succ.id in in_stack:
                    back_edges.append((block, succ))
            in_stack.discard(block.id)

        entry = function.entry_block
        if entry is not None:
            dfs(entry)

        # Build loops from back edges
        for latch, header in back_edges:
            body = self._compute_loop_body(header, latch, function)
            exits: Set[int] = set()
            for block in function.blocks:
                if block.id in body:
                    for succ in block.successors:
                        if succ.id not in body:
                            exits.add(succ.id)

            desc = LoopDescriptor(
                header=header,
                body=body,
                exits=exits,
                latch=latch,
                depth=0,
            )
            loops.append(desc)

        return loops

    def _compute_loop_body(self, header: BasicBlock, latch: BasicBlock,
                            function: Function) -> Set[int]:
        """Compute the body of a natural loop given header and latch."""
        body: Set[int] = {header.id}
        if latch is header:
            return body

        worklist = [latch]
        while worklist:
            block = worklist.pop()
            if block.id in body:
                continue
            body.add(block.id)
            for pred in block.predecessors:
                if pred.id not in body:
                    worklist.append(pred)

        return body

    def _process_loop(self, loop: LoopDescriptor, function: Function) -> bool:
        """Process a single loop for LICM."""
        # Ensure preheader
        preheader = PreheaderManager.ensure_preheader(loop, function)

        # Find invariant instructions
        analysis = LoopInvariantAnalysis(loop)
        invariant_ids = analysis.analyze(function)

        if not invariant_ids:
            return False

        # Check safety and hoist
        safety = HoistSafetyChecker(loop)
        hoisted = False

        # Collect instructions to hoist (maintain order)
        to_hoist: List[Instruction] = []
        for block in function.blocks:
            if block.id not in loop.body:
                continue
            for inst in list(block.instructions):
                if inst.id in invariant_ids and safety.is_safe_to_hoist(inst):
                    to_hoist.append(inst)

        # Sort by dependency order (operands before users)
        to_hoist = self._sort_by_dependencies(to_hoist)

        for inst in to_hoist:
            if inst.parent is None:
                continue
            self._hoist_instruction(inst, preheader)
            hoisted = True
            self._hoisted += 1

        # Try to sink code below loop
        sunk = self._try_sink_instructions(loop, function)
        if sunk:
            hoisted = True

        return hoisted

    def _hoist_instruction(self, inst: Instruction, preheader: BasicBlock) -> None:
        """Move an instruction to the preheader."""
        if inst.parent is None:
            return

        # Remove from current block
        inst.parent.remove(inst)

        # Insert before the terminator of preheader
        preheader.insert_before_terminator(inst)
        inst._parent = preheader

    def _sort_by_dependencies(self, instructions: List[Instruction]) -> List[Instruction]:
        """Sort instructions so that dependencies come first."""
        inst_ids = {inst.id for inst in instructions}
        sorted_list: List[Instruction] = []
        visited: Set[int] = set()

        def visit(inst: Instruction) -> None:
            if inst.id in visited:
                return
            visited.add(inst.id)

            # Visit operands first
            for op in self._get_operands(inst):
                if isinstance(op, Instruction) and op.id in inst_ids:
                    visit(op)

            sorted_list.append(inst)

        for inst in instructions:
            visit(inst)

        return sorted_list

    def _try_sink_instructions(self, loop: LoopDescriptor,
                                function: Function) -> bool:
        """Try to sink instructions from the loop to after the loop.

        An instruction can be sunk if:
        1. It has no uses inside the loop
        2. It's only used after the loop
        3. It dominates all exit blocks where it's used
        """
        sunk = False
        exit_blocks = [b for b in function.blocks if b.id in loop.exits]

        if len(exit_blocks) != 1:
            return False

        exit_block = exit_blocks[0]

        for block in function.blocks:
            if block.id not in loop.body:
                continue
            for inst in list(block.instructions):
                if self._can_sink(inst, loop, exit_block):
                    self._sink_instruction(inst, exit_block)
                    sunk = True
                    self._sunk += 1

        return sunk

    def _can_sink(self, inst: Instruction, loop: LoopDescriptor,
                   exit_block: BasicBlock) -> bool:
        """Check if an instruction can be sunk below the loop."""
        if isinstance(inst, (StoreInst, CallInst, ReturnInst, BranchInst,
                             SwitchInst, PhiInst)):
            return False

        if not hasattr(inst, 'users'):
            return False

        # All uses must be outside the loop
        for user in inst.users:
            if isinstance(user, Instruction) and user.parent is not None:
                if user.parent.id in loop.body:
                    return False

        return True

    def _sink_instruction(self, inst: Instruction, exit_block: BasicBlock) -> None:
        """Sink an instruction to the exit block."""
        if inst.parent is None:
            return

        inst.parent.remove(inst)
        exit_block.insert_at_front(inst)
        inst._parent = exit_block

    def _get_operands(self, inst: Instruction) -> List[Value]:
        ops: List[Value] = []
        if isinstance(inst, BinaryOp):
            ops.extend([inst.left, inst.right])
        elif isinstance(inst, UnaryOp):
            ops.append(inst.operand)
        elif isinstance(inst, CompareOp):
            ops.extend([inst.left, inst.right])
        elif isinstance(inst, CastInst):
            ops.append(inst.operand)
        elif isinstance(inst, SelectInst):
            ops.extend([inst.condition, inst.true_value, inst.false_value])
        elif isinstance(inst, LoadInst):
            ops.append(inst.address)
        elif isinstance(inst, GetElementPtrInst):
            ops.append(inst.base)
            ops.extend(inst.indices)
        return ops
