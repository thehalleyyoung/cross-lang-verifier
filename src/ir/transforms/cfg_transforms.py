"""
CFG transformations for the Cross-Language Equivalence Verifier.

Provides:
- BlockMerger: merge blocks with single pred/succ
- BlockSplitter: split blocks at specified points
- CriticalEdgeSplitter: split critical edges in CFG
- JumpThreader: thread jumps through conditional blocks
- TailDuplicator: duplicate tail blocks for optimization
- UnreachableBlockEliminator: remove unreachable blocks
- EdgeProfiler: annotate edges with estimated frequencies
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
    LoadInst, StoreInst,
    CastInst, CallInst,
    ReturnInst, BranchInst, SwitchInst,
    PhiInst, SelectInst,
)
from ...ir.types import IRType

logger = logging.getLogger(__name__)


# ─── Block Merger ────────────────────────────────────────────────────

class BlockMerger:
    """Merge basic blocks when one has a single successor and the successor
    has a single predecessor.

    B1 → B2 (B1 sole pred of B2, B2 sole succ of B1) → merged B1+B2
    """

    def __init__(self) -> None:
        self._merged_count = 0

    @property
    def merged_count(self) -> int:
        return self._merged_count

    def merge_blocks(self, function: Function) -> bool:
        """Merge all eligible block pairs. Returns True if any merged."""
        changed = False
        iterate = True

        while iterate:
            iterate = False
            for block in list(function.blocks):
                if block not in function.blocks:
                    continue
                succ = self._get_merge_candidate(block)
                if succ is not None:
                    self._merge(block, succ, function)
                    iterate = True
                    changed = True

        return changed

    def _get_merge_candidate(self, block: BasicBlock) -> Optional[BasicBlock]:
        """Check if block has a single successor that can be merged."""
        if len(block.successors) != 1:
            return None

        succ = block.successors[0]

        # Successor must have exactly one predecessor (this block)
        if len(succ.predecessors) != 1:
            return None

        # Don't merge with self (single-block loop)
        if succ is block:
            return None

        # Successor must not be the entry block
        if succ is function.entry_block if hasattr(function, 'entry_block') else False:
            return None

        return succ

    def _merge(self, pred: BasicBlock, succ: BasicBlock, function: Function) -> None:
        """Merge succ into pred."""
        # Remove the unconditional branch terminator of pred
        term = pred.terminator
        if term is not None:
            pred.remove(term)

        # Move all instructions from succ to pred
        for inst in list(succ.instructions):
            succ.remove(inst)
            inst._parent = pred
            pred.append(inst)

        # Transfer succ's successors to pred
        for s in succ.successors:
            # Update phi nodes in s
            for phi in s.phi_nodes:
                phi.incoming = [
                    (v, pred if b is succ else b) for v, b in phi.incoming
                ]
            s.remove_predecessor(succ)
            s.add_predecessor(pred)

        # Remove succ from function
        function.remove_block(succ)
        self._merged_count += 1


# ─── Block Splitter ─────────────────────────────────────────────────

class BlockSplitter:
    """Split a basic block at a specified instruction."""

    def __init__(self) -> None:
        self._split_count = 0

    @property
    def split_count(self) -> int:
        return self._split_count

    def split_at(self, block: BasicBlock, inst: Instruction,
                  function: Function) -> BasicBlock:
        """Split block at the given instruction.

        Creates a new block containing inst and all following instructions.
        The original block ends just before inst.
        """
        new_block = BasicBlock(name=f"{block.name}.split")
        new_block._parent = function

        # Find the split point
        found = False
        to_move: List[Instruction] = []
        for i in block.instructions:
            if found:
                to_move.append(i)
            elif i is inst:
                found = True
                to_move.append(i)

        if not found:
            return block

        # Move instructions to new block
        for i in to_move:
            block.remove(i)
            i._parent = new_block
            new_block.append(i)

        # Transfer successors
        for succ in list(block.successors):
            for phi in succ.phi_nodes:
                phi.incoming = [
                    (v, new_block if b is block else b) for v, b in phi.incoming
                ]
            succ.remove_predecessor(block)
            succ.add_predecessor(new_block)

        # Add unconditional branch from block to new_block
        br = BranchInst(target=new_block)
        br._parent = block
        block.append(br)
        new_block.add_predecessor(block)

        function.insert_block_after(block, new_block)
        self._split_count += 1

        return new_block

    def split_before_terminator(self, block: BasicBlock,
                                 function: Function) -> Optional[BasicBlock]:
        """Split a block just before its terminator."""
        term = block.terminator
        if term is None:
            return None
        return self.split_at(block, term, function)


# ─── Critical Edge Splitter ─────────────────────────────────────────

class CriticalEdgeSplitter:
    """Split critical edges in the CFG.

    A critical edge is one from a block with multiple successors
    to a block with multiple predecessors. Splitting these edges
    is necessary for correct phi node placement.
    """

    def __init__(self) -> None:
        self._split_count = 0

    @property
    def split_count(self) -> int:
        return self._split_count

    def split_critical_edges(self, function: Function) -> List[BasicBlock]:
        """Split all critical edges. Returns list of new blocks created."""
        new_blocks: List[BasicBlock] = []

        for block in list(function.blocks):
            if len(block.successors) <= 1:
                continue

            for succ in list(block.successors):
                if len(succ.predecessors) <= 1:
                    continue

                # This is a critical edge
                new_block = self._split_edge(block, succ, function)
                if new_block is not None:
                    new_blocks.append(new_block)

        return new_blocks

    def _split_edge(self, src: BasicBlock, dst: BasicBlock,
                     function: Function) -> Optional[BasicBlock]:
        """Split the edge from src to dst by inserting a new block."""
        new_block = BasicBlock(name=f"crit.{src.name}.{dst.name}")
        new_block._parent = function

        # Add branch from new_block to dst
        br = BranchInst(target=dst)
        br._parent = new_block
        new_block.append(br)

        # Redirect src → new_block instead of src → dst
        self._redirect_terminator(src, dst, new_block)

        # Update predecessor lists
        new_block.add_predecessor(src)
        dst.remove_predecessor(src)
        dst.add_predecessor(new_block)

        # Update phi nodes in dst
        for phi in dst.phi_nodes:
            phi.incoming = [
                (v, new_block if b is src else b) for v, b in phi.incoming
            ]

        function.insert_block_after(src, new_block)
        self._split_count += 1

        return new_block

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
        elif isinstance(term, SwitchInst):
            if hasattr(term, 'default_block') and term.default_block is old_target:
                term.default_block = new_target
            if hasattr(term, 'cases'):
                for i, (val, target) in enumerate(term.cases):
                    if target is old_target:
                        term.cases[i] = (val, new_target)


# ─── Jump Threading ─────────────────────────────────────────────────

class JumpThreader:
    """Thread jumps through blocks that just test a value.

    If a block B has a conditional branch on a value that is known
    to be true/false along a specific incoming edge, redirect that
    edge directly to the appropriate successor.
    """

    def __init__(self, max_block_size: int = 10) -> None:
        self._max_size = max_block_size
        self._threaded_count = 0

    @property
    def threaded_count(self) -> int:
        return self._threaded_count

    def thread_jumps(self, function: Function) -> bool:
        """Perform jump threading. Returns True if any changes made."""
        changed = False

        for block in list(function.blocks):
            term = block.terminator
            if not isinstance(term, BranchInst) or not term.is_conditional:
                continue

            # Check if condition comes from a phi node
            cond = term.condition
            if not isinstance(cond, PhiInst) or cond.parent is not block:
                continue

            # For each incoming edge where the condition is constant
            for val, pred in list(cond.incoming):
                if not isinstance(val, Constant):
                    continue

                cv = val.value if hasattr(val, 'value') else None
                if cv is None:
                    continue

                # Thread this edge
                target = term.true_block if cv != 0 else term.false_block
                if target is None:
                    continue

                if self._thread_edge(pred, block, target, function):
                    changed = True
                    self._threaded_count += 1

        return changed

    def _thread_edge(self, pred: BasicBlock, through: BasicBlock,
                      target: BasicBlock, function: Function) -> bool:
        """Thread the edge pred→through to go directly to target."""
        # Redirect pred's terminator
        pred_term = pred.terminator
        if isinstance(pred_term, BranchInst):
            if pred_term.is_conditional:
                if pred_term.true_block is through:
                    pred_term.true_block = target
                if pred_term.false_block is through:
                    pred_term.false_block = target
            else:
                if hasattr(pred_term, 'target') and pred_term.target is through:
                    pred_term.target = target
                elif pred_term.true_block is through:
                    pred_term.true_block = target

        # Update phi nodes in target
        for phi in target.phi_nodes:
            # Find the value that would come from 'through'
            through_val = None
            for val, block in phi.incoming:
                if block is through:
                    through_val = val
                    break
            if through_val is not None:
                phi.incoming.append((through_val, pred))

        # Update phi nodes in 'through'
        for phi in through.phi_nodes:
            phi.incoming = [(v, b) for v, b in phi.incoming if b is not pred]

        # Update predecessor lists
        through.remove_predecessor(pred)
        target.add_predecessor(pred)

        return True


# ─── Tail Duplicator ────────────────────────────────────────────────

class TailDuplicator:
    """Duplicate tail blocks to enable better optimization.

    When a block has multiple predecessors and ends with a branch,
    duplicating it for each predecessor can enable jump threading
    and other optimizations.
    """

    def __init__(self, max_dup_size: int = 15) -> None:
        self._max_size = max_dup_size
        self._dup_count = 0

    @property
    def dup_count(self) -> int:
        return self._dup_count

    def duplicate_tails(self, function: Function) -> bool:
        """Duplicate eligible tail blocks. Returns True if any changes."""
        changed = False

        for block in list(function.blocks):
            if len(block.predecessors) < 2:
                continue
            if len(block.instructions) > self._max_size:
                continue
            if block is function.entry_block:
                continue

            # Duplicate for all but the first predecessor
            preds = list(block.predecessors)
            for pred in preds[1:]:
                new_block = self._duplicate_for_pred(block, pred, function)
                if new_block is not None:
                    changed = True
                    self._dup_count += 1

        return changed

    def _duplicate_for_pred(self, block: BasicBlock, pred: BasicBlock,
                             function: Function) -> Optional[BasicBlock]:
        """Create a copy of block for a specific predecessor."""
        new_block = BasicBlock(name=f"{block.name}.dup.{pred.name}")
        new_block._parent = function

        value_map: Dict[int, Value] = {}

        # Resolve phi values for this predecessor
        for phi in block.phi_nodes:
            for val, b in phi.incoming:
                if b is pred:
                    value_map[phi.id] = val
                    break

        # Clone non-phi instructions
        for inst in block.non_phi_instructions():
            cloned = self._clone_inst(inst, new_block, value_map)
            if cloned is not None:
                new_block.append(cloned)
                value_map[inst.id] = cloned

        # Redirect pred to new_block
        self._redirect_terminator(pred, block, new_block)
        block.remove_predecessor(pred)
        new_block.add_predecessor(pred)

        # Update phi nodes in block's successors
        for succ in block.successors:
            for phi in succ.phi_nodes:
                for val, b in phi.incoming:
                    if b is block:
                        mapped = value_map.get(val.id if hasattr(val, 'id') else id(val), val)
                        phi.incoming.append((mapped, new_block))
                        break

        # Remove this pred from block's phi nodes
        for phi in block.phi_nodes:
            phi.incoming = [(v, b) for v, b in phi.incoming if b is not pred]

        function.insert_block_after(block, new_block)
        return new_block

    def _clone_inst(self, inst: Instruction, parent: BasicBlock,
                     value_map: Dict[int, Value]) -> Optional[Instruction]:
        """Clone an instruction, remapping through value_map."""
        def remap(v: Value) -> Value:
            if isinstance(v, Constant):
                return v
            mapped = value_map.get(v.id if hasattr(v, 'id') else id(v))
            return mapped if mapped is not None else v

        if isinstance(inst, BinaryOp):
            c = BinaryOp(op=inst.op, left=remap(inst.left), right=remap(inst.right),
                         ir_type=inst.ir_type, name=inst.name)
            c._parent = parent
            return c
        elif isinstance(inst, CompareOp):
            c = CompareOp(predicate=inst.predicate, left=remap(inst.left),
                          right=remap(inst.right), ir_type=inst.ir_type, name=inst.name)
            c._parent = parent
            return c
        elif isinstance(inst, BranchInst):
            if inst.is_conditional:
                c = BranchInst(condition=remap(inst.condition),
                               true_block=inst.true_block,
                               false_block=inst.false_block)
            else:
                target = inst.target if hasattr(inst, 'target') else inst.true_block
                c = BranchInst(target=target)
            c._parent = parent
            return c
        elif isinstance(inst, ReturnInst):
            c = ReturnInst(value=remap(inst.value) if inst.value else None,
                           ir_type=inst.ir_type)
            c._parent = parent
            return c
        elif isinstance(inst, LoadInst):
            c = LoadInst(address=remap(inst.address), ir_type=inst.ir_type, name=inst.name)
            c._parent = parent
            return c
        elif isinstance(inst, StoreInst):
            c = StoreInst(value=remap(inst.value), address=remap(inst.address))
            c._parent = parent
            return c
        elif isinstance(inst, CastInst):
            c = CastInst(cast_kind=inst.cast_kind, operand=remap(inst.operand),
                         ir_type=inst.ir_type, name=inst.name)
            c._parent = parent
            return c
        elif isinstance(inst, SelectInst):
            c = SelectInst(condition=remap(inst.condition),
                           true_value=remap(inst.true_value),
                           false_value=remap(inst.false_value),
                           ir_type=inst.ir_type, name=inst.name)
            c._parent = parent
            return c
        return None

    def _redirect_terminator(self, block: BasicBlock,
                              old: BasicBlock, new: BasicBlock) -> None:
        term = block.terminator
        if isinstance(term, BranchInst):
            if term.is_conditional:
                if term.true_block is old:
                    term.true_block = new
                if term.false_block is old:
                    term.false_block = new
            else:
                if hasattr(term, 'target') and term.target is old:
                    term.target = new
                elif term.true_block is old:
                    term.true_block = new


# ─── Unreachable Block Eliminator ──────────────────────────────────

class UnreachableBlockEliminator:
    """Remove blocks unreachable from the entry block."""

    def __init__(self) -> None:
        self._removed_count = 0

    @property
    def removed_count(self) -> int:
        return self._removed_count

    def eliminate(self, function: Function) -> bool:
        """Remove unreachable blocks. Returns True if any removed."""
        entry = function.entry_block
        if entry is None:
            return False

        reachable = self._compute_reachable(entry)
        to_remove = [b for b in function.blocks if b not in reachable]

        if not to_remove:
            return False

        for block in to_remove:
            for succ in list(block.successors):
                for phi in succ.phi_nodes:
                    phi.incoming = [(v, b) for v, b in phi.incoming if b is not block]
                succ.remove_predecessor(block)
            function.remove_block(block)
            self._removed_count += 1

        return True

    def _compute_reachable(self, entry: BasicBlock) -> Set[BasicBlock]:
        reachable: Set[BasicBlock] = set()
        worklist = [entry]
        while worklist:
            block = worklist.pop()
            if block in reachable:
                continue
            reachable.add(block)
            for succ in block.successors:
                worklist.append(succ)
        return reachable


# ─── Edge Profiler ──────────────────────────────────────────────────

@dataclass
class EdgeProfile:
    """Estimated edge frequency profile."""
    edge_counts: Dict[Tuple[int, int], float] = field(default_factory=dict)
    block_counts: Dict[int, float] = field(default_factory=dict)
    total_executions: float = 0.0

    def get_edge_frequency(self, src_id: int, dst_id: int) -> float:
        return self.edge_counts.get((src_id, dst_id), 0.0)

    def get_block_frequency(self, block_id: int) -> float:
        return self.block_counts.get(block_id, 0.0)

    def hottest_edge(self) -> Optional[Tuple[int, int]]:
        if not self.edge_counts:
            return None
        return max(self.edge_counts, key=self.edge_counts.get)

    def summary(self) -> str:
        lines = [f"Edge Profile ({self.total_executions:.0f} total)"]
        for (src, dst), count in sorted(self.edge_counts.items(),
                                         key=lambda x: -x[1])[:10]:
            lines.append(f"  {src} → {dst}: {count:.1f}")
        return "\n".join(lines)


class EdgeProfiler:
    """Estimate edge execution frequencies using static heuristics.

    Uses branch probability heuristics to estimate how often
    each edge is taken:
    - Loop back edges: ~90% taken
    - Error/return paths: ~10%
    - Equal probability for unknown branches
    """

    LOOP_BACK_EDGE_PROB = 0.9
    LOOP_EXIT_PROB = 0.1
    DEFAULT_BRANCH_PROB = 0.5

    def __init__(self) -> None:
        self._profile: Optional[EdgeProfile] = None

    @property
    def profile(self) -> Optional[EdgeProfile]:
        return self._profile

    def estimate(self, function: Function) -> EdgeProfile:
        """Estimate edge frequencies for all edges in the function."""
        profile = EdgeProfile()

        # Identify back edges
        back_edges = self._find_back_edges(function)

        entry = function.entry_block
        if entry is None:
            self._profile = profile
            return profile

        # Start with entry count = 1.0
        profile.block_counts[entry.id] = 1.0

        # Process blocks in RPO
        rpo = function.iter_blocks_rpo()
        for block in rpo:
            block_count = profile.block_counts.get(block.id, 0.0)
            if block_count == 0.0 and block is not entry:
                # Estimate from predecessors
                for pred in block.predecessors:
                    block_count += profile.edge_counts.get(
                        (pred.id, block.id), 0.0)
                profile.block_counts[block.id] = block_count

            succs = block.successors
            if len(succs) == 0:
                continue
            elif len(succs) == 1:
                profile.edge_counts[(block.id, succs[0].id)] = block_count
            elif len(succs) == 2:
                s0, s1 = succs[0], succs[1]
                e0 = (block.id, s0.id)
                e1 = (block.id, s1.id)

                if e0 in back_edges:
                    profile.edge_counts[e0] = block_count * self.LOOP_BACK_EDGE_PROB
                    profile.edge_counts[e1] = block_count * self.LOOP_EXIT_PROB
                elif e1 in back_edges:
                    profile.edge_counts[e1] = block_count * self.LOOP_BACK_EDGE_PROB
                    profile.edge_counts[e0] = block_count * self.LOOP_EXIT_PROB
                else:
                    profile.edge_counts[e0] = block_count * self.DEFAULT_BRANCH_PROB
                    profile.edge_counts[e1] = block_count * self.DEFAULT_BRANCH_PROB
            else:
                prob = 1.0 / len(succs)
                for succ in succs:
                    profile.edge_counts[(block.id, succ.id)] = block_count * prob

        profile.total_executions = sum(profile.block_counts.values())
        self._profile = profile
        return profile

    def _find_back_edges(self, function: Function) -> Set[Tuple[int, int]]:
        """Find back edges using DFS."""
        back_edges: Set[Tuple[int, int]] = set()
        visited: Set[int] = set()
        in_stack: Set[int] = set()

        def dfs(block: BasicBlock) -> None:
            visited.add(block.id)
            in_stack.add(block.id)
            for succ in block.successors:
                if succ.id not in visited:
                    dfs(succ)
                elif succ.id in in_stack:
                    back_edges.add((block.id, succ.id))
            in_stack.discard(block.id)

        entry = function.entry_block
        if entry is not None:
            dfs(entry)

        return back_edges
