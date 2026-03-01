"""
Loop transformations for the Cross-Language Equivalence Verifier.

Provides:
- LoopUnroller: full and partial loop unrolling
- LoopPeeler: peel first/last iterations
- LoopRotator: convert to do-while form
- LoopUnswitcher: hoist loop-invariant conditions
- LoopVectorizationPrep: dependence analysis for vectorization
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple

from ...ir.function import Function
from ...ir.basic_block import BasicBlock
from ...ir.instructions import (
    Instruction, Value, Constant, Argument,
    BinaryOp, BinOpKind, UnaryOp,
    CompareOp, CmpPredicate,
    LoadInst, StoreInst, AllocaInst,
    GetElementPtrInst, CastInst,
    CallInst, ReturnInst, BranchInst, SwitchInst,
    PhiInst, SelectInst,
)
from ...ir.types import IRType, IntType

logger = logging.getLogger(__name__)


# ─── Unroll Strategy ──────────────────────────────────────────────────

class UnrollStrategy(Enum):
    """Strategy for loop unrolling."""
    FULL = auto()       # Completely unroll (known trip count)
    PARTIAL = auto()    # Unroll by a factor
    RUNTIME = auto()    # Unroll with runtime trip count check


@dataclass
class LoopBounds:
    """Describes the bounds of a countable loop."""
    induction_var: Optional[PhiInst] = None
    init_value: Optional[Value] = None
    step_value: Optional[Value] = None
    bound_value: Optional[Value] = None
    comparison: Optional[CmpPredicate] = None
    trip_count: Optional[int] = None
    is_countable: bool = False

    def __str__(self) -> str:
        if not self.is_countable:
            return "LoopBounds(not countable)"
        return f"LoopBounds(trip={self.trip_count}, step={self.step_value})"


@dataclass
class LoopDesc:
    """Descriptor for a natural loop."""
    header: BasicBlock
    body: Set[int]
    exits: Set[int]
    latch: Optional[BasicBlock] = None
    preheader: Optional[BasicBlock] = None
    depth: int = 0


# ─── Loop Analysis ────────────────────────────────────────────────────

class LoopBoundsAnalyzer:
    """Analyze loop bounds to determine trip count."""

    def analyze(self, loop: LoopDesc, function: Function) -> LoopBounds:
        """Analyze a loop to extract its bounds."""
        bounds = LoopBounds()

        # Find the induction variable
        iv = self._find_induction_variable(loop)
        if iv is None:
            return bounds

        bounds.induction_var = iv

        # Find init, step, and bound
        init, step = self._analyze_iv_recurrence(iv, loop)
        if init is None or step is None:
            return bounds

        bounds.init_value = init
        bounds.step_value = step

        # Find the loop exit condition
        bound, pred = self._find_exit_condition(iv, loop, function)
        if bound is None:
            return bounds

        bounds.bound_value = bound
        bounds.comparison = pred
        bounds.is_countable = True

        # Compute trip count if all values are constants
        bounds.trip_count = self._compute_trip_count(init, step, bound, pred)

        return bounds

    def _find_induction_variable(self, loop: LoopDesc) -> Optional[PhiInst]:
        """Find the primary induction variable in the loop header."""
        for phi in loop.header.phi_nodes:
            # An IV has exactly one incoming from outside the loop (init)
            # and one from inside (step)
            outside = [(v, b) for v, b in phi.incoming if b.id not in loop.body]
            inside = [(v, b) for v, b in phi.incoming if b.id in loop.body]

            if len(outside) == 1 and len(inside) == 1:
                step_val = inside[0][0]
                if isinstance(step_val, BinaryOp) and step_val.op in (BinOpKind.ADD, BinOpKind.SUB):
                    if step_val.left is phi or step_val.right is phi:
                        return phi
        return None

    def _analyze_iv_recurrence(self, iv: PhiInst,
                                loop: LoopDesc) -> Tuple[Optional[Value], Optional[Value]]:
        """Extract init and step from an IV phi node."""
        init_val: Optional[Value] = None
        step_val: Optional[Value] = None

        for val, block in iv.incoming:
            if block.id not in loop.body:
                init_val = val
            else:
                # The step should be an add/sub of the IV and a constant
                if isinstance(val, BinaryOp):
                    if val.left is iv:
                        step_val = val.right
                    elif val.right is iv:
                        step_val = val.left

        return init_val, step_val

    def _find_exit_condition(self, iv: PhiInst, loop: LoopDesc,
                              function: Function) -> Tuple[Optional[Value], Optional[CmpPredicate]]:
        """Find the loop exit condition comparing the IV to a bound."""
        if loop.latch is None:
            return None, None

        term = loop.latch.terminator
        if not isinstance(term, BranchInst) or not term.is_conditional:
            # Check header instead
            term = loop.header.terminator
            if not isinstance(term, BranchInst) or not term.is_conditional:
                return None, None

        cond = term.condition
        if not isinstance(cond, CompareOp):
            return None, None

        # Check if the comparison uses the IV
        if cond.left is iv:
            return cond.right, cond.predicate
        elif cond.right is iv:
            return cond.left, self._swap_predicate(cond.predicate)

        # Check if comparison uses IV+step
        for operand in [cond.left, cond.right]:
            if isinstance(operand, BinaryOp):
                if operand.left is iv or operand.right is iv:
                    bound = cond.right if operand is cond.left else cond.left
                    return bound, cond.predicate

        return None, None

    def _compute_trip_count(self, init: Value, step: Value, bound: Value,
                             pred: Optional[CmpPredicate]) -> Optional[int]:
        """Compute the trip count from constant bounds."""
        if not all(isinstance(v, Constant) and hasattr(v, 'value') for v in [init, step, bound]):
            return None
        if pred is None:
            return None

        i = init.value
        s = step.value
        b = bound.value

        if not all(isinstance(v, int) for v in [i, s, b]):
            return None

        if s == 0:
            return None

        if pred in (CmpPredicate.SLT, CmpPredicate.ULT):
            if s > 0:
                count = max(0, (b - i + s - 1) // s)
                return count if count >= 0 else None
        elif pred in (CmpPredicate.SLE, CmpPredicate.ULE):
            if s > 0:
                count = max(0, (b - i + s) // s)
                return count if count >= 0 else None
        elif pred in (CmpPredicate.SGT, CmpPredicate.UGT):
            if s < 0:
                count = max(0, (i - b - s - 1) // (-s))
                return count if count >= 0 else None
        elif pred == CmpPredicate.NE:
            if s != 0 and (b - i) % s == 0:
                count = (b - i) // s
                return count if count >= 0 else None

        return None

    def _swap_predicate(self, pred: CmpPredicate) -> CmpPredicate:
        swap = {
            CmpPredicate.SLT: CmpPredicate.SGT,
            CmpPredicate.SGT: CmpPredicate.SLT,
            CmpPredicate.SLE: CmpPredicate.SGE,
            CmpPredicate.SGE: CmpPredicate.SLE,
            CmpPredicate.ULT: CmpPredicate.UGT,
            CmpPredicate.UGT: CmpPredicate.ULT,
            CmpPredicate.ULE: CmpPredicate.UGE,
            CmpPredicate.UGE: CmpPredicate.ULE,
            CmpPredicate.EQ: CmpPredicate.EQ,
            CmpPredicate.NE: CmpPredicate.NE,
        }
        return swap.get(pred, pred)


# ─── Block Cloner ────────────────────────────────────────────────────

class BlockCloner:
    """Clone basic blocks for loop transformations."""

    def __init__(self) -> None:
        self._value_map: Dict[int, Value] = {}
        self._block_map: Dict[int, BasicBlock] = {}

    def clone_blocks(self, blocks: List[BasicBlock], function: Function,
                      suffix: str = "") -> List[BasicBlock]:
        """Clone a list of blocks, creating fresh copies."""
        self._value_map.clear()
        self._block_map.clear()

        cloned: List[BasicBlock] = []
        for block in blocks:
            new_block = BasicBlock(name=f"{block.name}{suffix}")
            new_block._parent = function
            self._block_map[block.id] = new_block
            cloned.append(new_block)

        # Clone instructions
        for orig, new_block in zip(blocks, cloned):
            for inst in orig.instructions:
                new_inst = self._clone_inst(inst, new_block)
                if new_inst is not None:
                    new_block.append(new_inst)
                    self._value_map[inst.id] = new_inst

        # Fix up references
        for new_block in cloned:
            for inst in new_block.instructions:
                self._remap_operands(inst)

        return cloned

    @property
    def value_map(self) -> Dict[int, Value]:
        return self._value_map

    @property
    def block_map(self) -> Dict[int, BasicBlock]:
        return self._block_map

    def remap_value(self, val: Value) -> Value:
        if isinstance(val, Constant):
            return val
        mapped = self._value_map.get(val.id if hasattr(val, 'id') else id(val))
        return mapped if mapped is not None else val

    def remap_block(self, block: Optional[BasicBlock]) -> Optional[BasicBlock]:
        if block is None:
            return None
        return self._block_map.get(block.id, block)

    def _clone_inst(self, inst: Instruction, parent: BasicBlock) -> Optional[Instruction]:
        if isinstance(inst, BinaryOp):
            c = BinaryOp(op=inst.op, left=inst.left, right=inst.right,
                         ir_type=inst.ir_type, name=inst.name)
            c._parent = parent
            return c
        elif isinstance(inst, UnaryOp):
            c = UnaryOp(op=inst.op, operand=inst.operand,
                        ir_type=inst.ir_type, name=inst.name)
            c._parent = parent
            return c
        elif isinstance(inst, CompareOp):
            c = CompareOp(predicate=inst.predicate, left=inst.left, right=inst.right,
                          ir_type=inst.ir_type, name=inst.name)
            c._parent = parent
            return c
        elif isinstance(inst, LoadInst):
            c = LoadInst(address=inst.address, ir_type=inst.ir_type, name=inst.name)
            c._parent = parent
            return c
        elif isinstance(inst, StoreInst):
            c = StoreInst(value=inst.value, address=inst.address)
            c._parent = parent
            return c
        elif isinstance(inst, BranchInst):
            if inst.is_conditional:
                c = BranchInst(condition=inst.condition,
                               true_block=inst.true_block,
                               false_block=inst.false_block)
            else:
                target = inst.target if hasattr(inst, 'target') else inst.true_block
                c = BranchInst(target=target)
            c._parent = parent
            return c
        elif isinstance(inst, PhiInst):
            c = PhiInst(ir_type=inst.ir_type, incoming=list(inst.incoming),
                        name=inst.name)
            c._parent = parent
            return c
        elif isinstance(inst, SelectInst):
            c = SelectInst(condition=inst.condition, true_value=inst.true_value,
                           false_value=inst.false_value, ir_type=inst.ir_type,
                           name=inst.name)
            c._parent = parent
            return c
        elif isinstance(inst, CastInst):
            c = CastInst(cast_kind=inst.cast_kind, operand=inst.operand,
                         ir_type=inst.ir_type, name=inst.name)
            c._parent = parent
            return c
        elif isinstance(inst, ReturnInst):
            c = ReturnInst(value=inst.value, ir_type=inst.ir_type)
            c._parent = parent
            return c
        elif isinstance(inst, CallInst):
            c = CallInst(callee=inst.callee, arguments=list(inst.arguments),
                         ir_type=inst.ir_type, name=inst.name)
            c._parent = parent
            return c
        return None

    def _remap_operands(self, inst: Instruction) -> None:
        if isinstance(inst, BinaryOp):
            inst.left = self.remap_value(inst.left)
            inst.right = self.remap_value(inst.right)
        elif isinstance(inst, UnaryOp):
            inst.operand = self.remap_value(inst.operand)
        elif isinstance(inst, CompareOp):
            inst.left = self.remap_value(inst.left)
            inst.right = self.remap_value(inst.right)
        elif isinstance(inst, LoadInst):
            inst.address = self.remap_value(inst.address)
        elif isinstance(inst, StoreInst):
            inst.value = self.remap_value(inst.value)
            inst.address = self.remap_value(inst.address)
        elif isinstance(inst, CastInst):
            inst.operand = self.remap_value(inst.operand)
        elif isinstance(inst, SelectInst):
            inst.condition = self.remap_value(inst.condition)
            inst.true_value = self.remap_value(inst.true_value)
            inst.false_value = self.remap_value(inst.false_value)
        elif isinstance(inst, PhiInst):
            inst.incoming = [(self.remap_value(v), self.remap_block(b))
                             for v, b in inst.incoming]
        elif isinstance(inst, BranchInst):
            if inst.is_conditional:
                inst.condition = self.remap_value(inst.condition)
                inst.true_block = self.remap_block(inst.true_block)
                inst.false_block = self.remap_block(inst.false_block)
            else:
                target = inst.target if hasattr(inst, 'target') else inst.true_block
                remapped = self.remap_block(target)
                if hasattr(inst, 'target'):
                    inst.target = remapped
                else:
                    inst.true_block = remapped
        elif isinstance(inst, ReturnInst):
            if inst.value is not None:
                inst.value = self.remap_value(inst.value)
        elif isinstance(inst, CallInst):
            inst.arguments = [self.remap_value(a) for a in inst.arguments]


# ─── Loop Unroller ────────────────────────────────────────────────────

class LoopUnroller:
    """Unroll loops by a given factor or completely.

    Full unrolling: replicate the loop body trip_count times and
    remove the loop structure entirely.

    Partial unrolling: replicate the loop body factor times within
    the loop, reducing the number of iterations.
    """

    def __init__(self, max_full_unroll_count: int = 64,
                 default_unroll_factor: int = 4,
                 max_unrolled_size: int = 500) -> None:
        self._max_full = max_full_unroll_count
        self._default_factor = default_unroll_factor
        self._max_size = max_unrolled_size

    def unroll(self, loop: LoopDesc, function: Function,
               strategy: UnrollStrategy = UnrollStrategy.PARTIAL,
               factor: Optional[int] = None) -> bool:
        """Unroll a loop with the given strategy."""
        analyzer = LoopBoundsAnalyzer()
        bounds = analyzer.analyze(loop, function)

        if strategy == UnrollStrategy.FULL:
            return self._full_unroll(loop, bounds, function)
        else:
            f = factor or self._default_factor
            return self._partial_unroll(loop, bounds, function, f)

    def _full_unroll(self, loop: LoopDesc, bounds: LoopBounds,
                      function: Function) -> bool:
        """Fully unroll a loop with known trip count."""
        if not bounds.is_countable or bounds.trip_count is None:
            return False
        if bounds.trip_count > self._max_full:
            return False

        body_blocks = [b for b in function.blocks if b.id in loop.body]
        body_size = sum(len(b.instructions) for b in body_blocks)
        if body_size * bounds.trip_count > self._max_size:
            return False

        # Clone the body trip_count times
        preheader = loop.preheader
        if preheader is None:
            return False

        all_cloned: List[List[BasicBlock]] = []
        for i in range(bounds.trip_count):
            cloner = BlockCloner()
            cloned = cloner.clone_blocks(body_blocks, function, suffix=f".unroll.{i}")
            all_cloned.append(cloned)
            for b in cloned:
                function.add_block(b)

        # Wire iterations together
        for i in range(len(all_cloned) - 1):
            last_block = all_cloned[i][-1]
            first_block = all_cloned[i + 1][0]
            # Remove back edge, add forward edge
            term = last_block.terminator
            if isinstance(term, BranchInst):
                last_block.remove(term)
                br = BranchInst(target=first_block)
                br._parent = last_block
                last_block.append(br)
                first_block.add_predecessor(last_block)

        # Wire preheader to first iteration
        if all_cloned and preheader.terminator is not None:
            preheader.remove(preheader.terminator)
            br = BranchInst(target=all_cloned[0][0])
            br._parent = preheader
            preheader.append(br)

        # Wire last iteration to exit
        exit_blocks = [b for b in function.blocks if b.id in loop.exits]
        if all_cloned and exit_blocks:
            last = all_cloned[-1][-1]
            if last.terminator is not None:
                last.remove(last.terminator)
            br = BranchInst(target=exit_blocks[0])
            br._parent = last
            last.append(br)

        # Remove original loop body
        for b in body_blocks:
            try:
                function.remove_block(b)
            except ValueError:
                pass

        return True

    def _partial_unroll(self, loop: LoopDesc, bounds: LoopBounds,
                         function: Function, factor: int) -> bool:
        """Partially unroll a loop by a given factor."""
        body_blocks = [b for b in function.blocks if b.id in loop.body]
        body_size = sum(len(b.instructions) for b in body_blocks)

        if body_size * factor > self._max_size:
            factor = max(2, self._max_size // max(body_size, 1))

        if factor < 2:
            return False

        # Clone body (factor-1) times inside the loop
        cloned_iterations: List[List[BasicBlock]] = []
        for i in range(1, factor):
            cloner = BlockCloner()
            cloned = cloner.clone_blocks(body_blocks, function, suffix=f".unroll.{i}")
            cloned_iterations.append(cloned)
            for b in cloned:
                function.add_block(b)

        if not cloned_iterations:
            return False

        # Wire original latch to first cloned iteration
        if loop.latch is not None and loop.latch.terminator is not None:
            # The latch back edge should now go to the first cloned block
            first_cloned = cloned_iterations[0][0]
            latch_term = loop.latch.terminator
            if isinstance(latch_term, BranchInst):
                if latch_term.is_conditional:
                    if latch_term.true_block is loop.header:
                        latch_term.true_block = first_cloned
                    if latch_term.false_block is loop.header:
                        latch_term.false_block = first_cloned
                else:
                    target = latch_term.target if hasattr(latch_term, 'target') else latch_term.true_block
                    if target is loop.header:
                        if hasattr(latch_term, 'target'):
                            latch_term.target = first_cloned
                        else:
                            latch_term.true_block = first_cloned

        # Wire cloned iterations together
        for i in range(len(cloned_iterations) - 1):
            last = cloned_iterations[i][-1]
            first = cloned_iterations[i + 1][0]
            term = last.terminator
            if isinstance(term, BranchInst):
                last.remove(term)
                br = BranchInst(target=first)
                br._parent = last
                last.append(br)

        # Wire last cloned iteration back to header
        last_cloned = cloned_iterations[-1][-1]
        last_term = last_cloned.terminator
        if isinstance(last_term, BranchInst):
            last_cloned.remove(last_term)
            br = BranchInst(target=loop.header)
            br._parent = last_cloned
            last_cloned.append(br)
            loop.header.add_predecessor(last_cloned)

        return True


# ─── Loop Peeler ─────────────────────────────────────────────────────

class LoopPeeler:
    """Peel iterations from the beginning or end of a loop.

    Peeling separates the first N iterations from the rest of the loop,
    which can enable further optimizations.
    """

    def __init__(self, max_peel_count: int = 4) -> None:
        self._max_peel = max_peel_count

    def peel_first(self, loop: LoopDesc, function: Function,
                    count: int = 1) -> bool:
        """Peel the first `count` iterations from the loop."""
        if count > self._max_peel:
            return False

        body_blocks = [b for b in function.blocks if b.id in loop.body]
        if not body_blocks:
            return False

        preheader = loop.preheader
        if preheader is None:
            return False

        for i in range(count):
            cloner = BlockCloner()
            peeled = cloner.clone_blocks(body_blocks, function, suffix=f".peel.{i}")
            for b in peeled:
                function.insert_block_before(loop.header, b)

            # Wire preheader → peeled entry
            if preheader.terminator is not None:
                preheader.remove(preheader.terminator)
            br = BranchInst(target=peeled[0])
            br._parent = preheader
            preheader.append(br)
            peeled[0].add_predecessor(preheader)

            # Wire peeled exit → header
            peeled_latch = peeled[-1]
            if peeled_latch.terminator is not None:
                peeled_latch.remove(peeled_latch.terminator)
            br = BranchInst(target=loop.header)
            br._parent = peeled_latch
            peeled_latch.append(br)
            loop.header.add_predecessor(peeled_latch)

            # Update preheader for next iteration
            preheader = peeled_latch

        return True


# ─── Loop Rotator ────────────────────────────────────────────────────

class LoopRotator:
    """Rotate a while loop into do-while form.

    Transforms:
        while (cond) { body; }
    Into:
        if (cond) { do { body; } while (cond); }

    This simplifies the loop structure and enables better optimization.
    """

    def __init__(self, max_header_size: int = 20) -> None:
        self._max_header_size = max_header_size

    def rotate(self, loop: LoopDesc, function: Function) -> bool:
        """Rotate a loop from while to do-while form."""
        header = loop.header
        if header is None:
            return False

        # Check if header ends with a conditional branch
        term = header.terminator
        if not isinstance(term, BranchInst) or not term.is_conditional:
            return False

        # Determine which successor is the loop body and which is exit
        body_succ = None
        exit_succ = None
        for succ in header.successors:
            if succ.id in loop.body and succ is not header:
                body_succ = succ
            elif succ.id not in loop.body:
                exit_succ = succ

        if body_succ is None or exit_succ is None:
            return False

        # Don't rotate if header is too large
        if len(header.instructions) > self._max_header_size:
            return False

        # Clone the header as a guard
        cloner = BlockCloner()
        guard_blocks = cloner.clone_blocks([header], function, suffix=".guard")
        if not guard_blocks:
            return False

        guard = guard_blocks[0]
        function.insert_block_before(header, guard)

        # Wire preheader → guard instead of header
        if loop.preheader is not None:
            for pred in list(header.predecessors):
                if pred.id not in loop.body:
                    self._redirect_terminator(pred, header, guard)
                    guard.add_predecessor(pred)
                    header.remove_predecessor(pred)

        # Guard's body edge goes to header, exit edge stays
        guard_term = guard.terminator
        if isinstance(guard_term, BranchInst) and guard_term.is_conditional:
            body_target = cloner.remap_block(body_succ) or body_succ
            # Redirect guard's body edge to actual header
            if guard_term.true_block is body_target:
                guard_term.true_block = header
            elif guard_term.false_block is body_target:
                guard_term.false_block = header
            header.add_predecessor(guard)

        return True

    def _redirect_terminator(self, block: BasicBlock,
                              old_target: BasicBlock,
                              new_target: BasicBlock) -> None:
        term = block.terminator
        if isinstance(term, BranchInst):
            if term.is_conditional:
                if term.true_block is old_target:
                    term.true_block = new_target
                if term.false_block is old_target:
                    term.false_block = new_target
            else:
                if hasattr(term, 'target') and term.target is old_target:
                    term.target = new_target
                elif term.true_block is old_target:
                    term.true_block = new_target


# ─── Loop Unswitcher ─────────────────────────────────────────────────

class LoopUnswitcher:
    """Hoist loop-invariant conditional branches out of the loop.

    If a conditional branch inside the loop has a loop-invariant
    condition, duplicate the loop with the condition resolved
    in each copy.
    """

    def __init__(self, max_loop_size: int = 200) -> None:
        self._max_size = max_loop_size

    def unswitch(self, loop: LoopDesc, function: Function) -> bool:
        """Attempt to unswitch a loop-invariant condition."""
        body_blocks = [b for b in function.blocks if b.id in loop.body]
        body_size = sum(len(b.instructions) for b in body_blocks)

        if body_size * 2 > self._max_size:
            return False

        # Find a loop-invariant conditional branch
        for block in body_blocks:
            term = block.terminator
            if not isinstance(term, BranchInst) or not term.is_conditional:
                continue

            if self._is_loop_invariant(term.condition, loop):
                return self._do_unswitch(term, block, loop, function, body_blocks)

        return False

    def _is_loop_invariant(self, val: Value, loop: LoopDesc) -> bool:
        """Check if a value is loop-invariant."""
        if isinstance(val, (Constant, Argument)):
            return True
        if isinstance(val, Instruction) and val.parent is not None:
            return val.parent.id not in loop.body
        return False

    def _do_unswitch(self, branch: BranchInst, branch_block: BasicBlock,
                      loop: LoopDesc, function: Function,
                      body_blocks: List[BasicBlock]) -> bool:
        """Perform the unswitching transformation."""
        preheader = loop.preheader
        if preheader is None:
            return False

        # Clone the loop for the false case
        cloner = BlockCloner()
        false_loop = cloner.clone_blocks(body_blocks, function, suffix=".unswitch.false")
        for b in false_loop:
            function.add_block(b)

        # Insert conditional branch in preheader
        if preheader.terminator is not None:
            preheader.remove(preheader.terminator)

        true_entry = loop.header
        false_entry = false_loop[0] if false_loop else true_entry

        cond_br = BranchInst(
            condition=branch.condition,
            true_block=true_entry,
            false_block=false_entry,
        )
        cond_br._parent = preheader
        preheader.append(cond_br)

        return True


# ─── Loop Vectorization Prep ────────────────────────────────────────

@dataclass
class MemoryAccess:
    """A memory access within a loop."""
    instruction: Instruction
    address: Value
    is_write: bool
    stride: Optional[int] = None


@dataclass
class DependenceInfo:
    """Dependence between two memory accesses."""
    source: MemoryAccess
    sink: MemoryAccess
    distance: Optional[int] = None
    is_loop_carried: bool = False
    prevents_vectorization: bool = False


class LoopVectorizationPrep:
    """Analyze loop dependences for vectorization readiness.

    Performs memory dependence analysis to determine if a loop
    can be safely vectorized.
    """

    def __init__(self) -> None:
        self._accesses: List[MemoryAccess] = []
        self._dependences: List[DependenceInfo] = []

    @property
    def can_vectorize(self) -> bool:
        return not any(d.prevents_vectorization for d in self._dependences)

    @property
    def dependences(self) -> List[DependenceInfo]:
        return self._dependences

    def analyze(self, loop: LoopDesc, function: Function) -> bool:
        """Analyze loop for vectorizability. Returns True if vectorizable."""
        self._accesses.clear()
        self._dependences.clear()

        body_blocks = [b for b in function.blocks if b.id in loop.body]

        # Collect memory accesses
        for block in body_blocks:
            for inst in block.instructions:
                if isinstance(inst, LoadInst):
                    access = MemoryAccess(instruction=inst, address=inst.address,
                                          is_write=False)
                    access.stride = self._compute_stride(inst.address, loop)
                    self._accesses.append(access)
                elif isinstance(inst, StoreInst):
                    access = MemoryAccess(instruction=inst, address=inst.address,
                                          is_write=True)
                    access.stride = self._compute_stride(inst.address, loop)
                    self._accesses.append(access)

        # Check pairwise dependences
        for i, a1 in enumerate(self._accesses):
            for a2 in self._accesses[i + 1:]:
                if not a1.is_write and not a2.is_write:
                    continue  # Read-read: no dependence
                dep = self._check_dependence(a1, a2, loop)
                if dep is not None:
                    self._dependences.append(dep)

        return self.can_vectorize

    def _compute_stride(self, address: Value, loop: LoopDesc) -> Optional[int]:
        """Compute the stride of a memory access relative to the IV."""
        if isinstance(address, GetElementPtrInst):
            for idx in address.indices:
                if isinstance(idx, BinaryOp) and idx.op == BinOpKind.MUL:
                    if isinstance(idx.right, Constant) and hasattr(idx.right, 'value'):
                        return idx.right.value
            if len(address.indices) > 0:
                return 1  # Default stride of 1
        return None

    def _check_dependence(self, a1: MemoryAccess, a2: MemoryAccess,
                           loop: LoopDesc) -> Optional[DependenceInfo]:
        """Check for memory dependence between two accesses."""
        # Simple check: if addresses are provably different, no dependence
        if a1.address is a2.address:
            dep = DependenceInfo(source=a1, sink=a2, distance=0,
                                is_loop_carried=False)
            if a1.is_write or a2.is_write:
                dep.prevents_vectorization = True
            return dep

        # If both have known strides and same base, check for overlap
        if a1.stride is not None and a2.stride is not None:
            if a1.stride == a2.stride and a1.stride > 0:
                return None  # Same stride, likely no cross-iteration dependence

        # Conservative: assume dependence if both are writes or one write
        if a1.is_write or a2.is_write:
            dep = DependenceInfo(source=a1, sink=a2, is_loop_carried=True)
            dep.prevents_vectorization = True
            return dep

        return None
