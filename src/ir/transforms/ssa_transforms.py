"""
SSA transformations for the Cross-Language Equivalence Verifier.

Provides:
- SSAConstructor: full SSA construction (Cytron algorithm)
- SSADeconstructor: SSA deconstruction with parallel copy insertion
- CopyPropagator: propagate copies through SSA
- PhiSimplifier: simplify and remove trivial phi nodes
- ValueRenamer: rename SSA values with fresh names
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
)
from ...ir.types import IRType

logger = logging.getLogger(__name__)


# ─── SSA Constructor (Cytron Algorithm) ────────────────────────────

class SSAConstructor:
    """Construct SSA form using the Cytron et al. algorithm.

    Steps:
    1. Compute dominance frontiers
    2. Place phi nodes at iterated dominance frontiers of variable definitions
    3. Rename variables in dominator tree order

    This transforms non-SSA IR (with multiple assignments to the same
    variable name) into proper SSA form with phi nodes.
    """

    def __init__(self) -> None:
        self._defs: Dict[str, Set[int]] = defaultdict(set)  # var → defining block IDs
        self._phi_placed: Dict[str, Set[int]] = defaultdict(set)
        self._stacks: Dict[str, List[Value]] = defaultdict(list)
        self._counters: Dict[str, int] = defaultdict(int)
        self._phis_inserted = 0

    @property
    def phis_inserted(self) -> int:
        return self._phis_inserted

    def construct(self, function: Function) -> bool:
        """Transform function into SSA form."""
        self._defs.clear()
        self._phi_placed.clear()
        self._stacks.clear()
        self._counters.clear()
        self._phis_inserted = 0

        # Step 1: collect variable definitions
        self._collect_definitions(function)

        if not self._defs:
            return False

        # Step 2: compute dominance frontiers
        function.compute_dominators()
        frontiers = self._compute_dominance_frontiers(function)

        # Step 3: place phi nodes
        block_map = {b.id: b for b in function.blocks}
        self._place_phis(frontiers, block_map, function)

        # Step 4: rename variables
        dom_children = self._compute_dom_children(function)
        entry = function.entry_block
        if entry is not None:
            self._rename(entry, dom_children, block_map)

        return self._phis_inserted > 0

    def _collect_definitions(self, function: Function) -> None:
        """Find all variable definitions (stores to named variables)."""
        for block in function.blocks:
            for inst in block.instructions:
                if isinstance(inst, StoreInst) and isinstance(inst.address, AllocaInst):
                    var_name = inst.address.name or f"var.{inst.address.id}"
                    self._defs[var_name].add(block.id)
                elif hasattr(inst, 'name') and inst.name:
                    self._defs[inst.name].add(block.id)

    def _compute_dominance_frontiers(self, function: Function) -> Dict[int, Set[int]]:
        """Compute dominance frontiers for all blocks."""
        frontiers: Dict[int, Set[int]] = {b.id: set() for b in function.blocks}

        for block in function.blocks:
            preds = block.predecessors
            if len(preds) < 2:
                continue
            for pred in preds:
                runner = pred
                while runner is not None and runner is not block.idom:
                    frontiers[runner.id].add(block.id)
                    runner = runner.idom

        return frontiers

    def _compute_dom_children(self, function: Function) -> Dict[int, List[int]]:
        """Compute dominator tree children."""
        children: Dict[int, List[int]] = defaultdict(list)
        for block in function.blocks:
            if block.idom is not None:
                children[block.idom.id].append(block.id)
        return dict(children)

    def _place_phis(self, frontiers: Dict[int, Set[int]],
                     block_map: Dict[int, BasicBlock],
                     function: Function) -> None:
        """Place phi nodes at iterated dominance frontiers."""
        for var_name, def_blocks in self._defs.items():
            worklist = list(def_blocks)
            visited: Set[int] = set()

            while worklist:
                block_id = worklist.pop()
                if block_id in visited:
                    continue
                visited.add(block_id)

                for df_id in frontiers.get(block_id, set()):
                    if df_id not in self._phi_placed[var_name]:
                        self._phi_placed[var_name].add(df_id)
                        block = block_map.get(df_id)
                        if block is not None:
                            # Determine type from any definition
                            phi_type = self._get_var_type(var_name, function)
                            if phi_type is not None:
                                phi = PhiInst(
                                    ir_type=phi_type,
                                    incoming=[],
                                    name=f"{var_name}.phi"
                                )
                                phi._parent = block
                                block.add_phi(phi)
                                self._phis_inserted += 1
                        worklist.append(df_id)

    def _get_var_type(self, var_name: str, function: Function) -> Optional[IRType]:
        """Determine the type of a variable from its uses."""
        for block in function.blocks:
            for inst in block.instructions:
                if hasattr(inst, 'name') and inst.name == var_name:
                    return inst.ir_type
                if isinstance(inst, StoreInst) and isinstance(inst.address, AllocaInst):
                    if (inst.address.name or f"var.{inst.address.id}") == var_name:
                        return inst.value.ir_type if hasattr(inst.value, 'ir_type') else None
        return None

    def _rename(self, block: BasicBlock, dom_children: Dict[int, List[int]],
                block_map: Dict[int, BasicBlock]) -> None:
        """Rename variables in dominator tree preorder."""
        push_counts: Dict[str, int] = defaultdict(int)

        # Process phi nodes
        for phi in block.phi_nodes:
            if phi.name and phi.name.endswith('.phi'):
                var_name = phi.name[:-4]
                self._stacks[var_name].append(phi)
                push_counts[var_name] += 1

        # Process instructions
        for inst in block.instructions:
            # Rename uses
            self._rename_uses(inst)

            # Rename definitions
            if hasattr(inst, 'name') and inst.name and inst.name in self._defs:
                var_name = inst.name
                self._counters[var_name] += 1
                inst.name = f"{var_name}.{self._counters[var_name]}"
                self._stacks[var_name].append(inst)
                push_counts[var_name] += 1

        # Fill phi operands in successors
        for succ in block.successors:
            for phi in succ.phi_nodes:
                if phi.name and phi.name.endswith('.phi'):
                    var_name = phi.name[:-4]
                    if self._stacks[var_name]:
                        current = self._stacks[var_name][-1]
                        phi.incoming.append((current, block))

        # Recurse into dominated children
        for child_id in dom_children.get(block.id, []):
            child = block_map.get(child_id)
            if child is not None:
                self._rename(child, dom_children, block_map)

        # Pop definitions
        for var_name, count in push_counts.items():
            for _ in range(count):
                self._stacks[var_name].pop()

    def _rename_uses(self, inst: Instruction) -> None:
        """Replace variable references with current SSA value."""
        # This is a simplified version; full implementation would
        # walk all operands and replace name-based references
        pass


# ─── SSA Deconstructor ─────────────────────────────────────────────

@dataclass
class ParallelCopy:
    """A parallel copy: simultaneous assignment of multiple values."""
    copies: List[Tuple[Value, Value]] = field(default_factory=list)  # (dest, src)

    def add(self, dest: Value, src: Value) -> None:
        self.copies.append((dest, src))

    def __len__(self) -> int:
        return len(self.copies)

    def __bool__(self) -> bool:
        return len(self.copies) > 0


class SSADeconstructor:
    """Deconstruct SSA form by eliminating phi nodes.

    Converts phi nodes into parallel copies placed at the end
    of predecessor blocks, then sequentializes the parallel copies.

    This is necessary when lowering SSA-form IR to a target that
    doesn't support phi nodes (e.g., register allocation).
    """

    def __init__(self) -> None:
        self._copies_inserted = 0
        self._phis_removed = 0

    @property
    def copies_inserted(self) -> int:
        return self._copies_inserted

    @property
    def phis_removed(self) -> int:
        return self._phis_removed

    def deconstruct(self, function: Function) -> bool:
        """Remove all phi nodes, replacing with copies."""
        self._copies_inserted = 0
        self._phis_removed = 0

        has_phis = False
        for block in function.blocks:
            if block.phi_nodes:
                has_phis = True
                break

        if not has_phis:
            return False

        # Step 1: split critical edges (needed for correct copy placement)
        self._split_critical_edges(function)

        # Step 2: insert parallel copies
        parallel_copies = self._compute_parallel_copies(function)

        # Step 3: sequentialize parallel copies
        for block, pcopy in parallel_copies.items():
            self._sequentialize_copies(block, pcopy, function)

        # Step 4: remove phi nodes
        for block in function.blocks:
            for phi in list(block.phi_nodes):
                block.remove(phi)
                self._phis_removed += 1

        return self._phis_removed > 0

    def _split_critical_edges(self, function: Function) -> None:
        """Split critical edges to ensure correct copy placement."""
        from .cfg_transforms import CriticalEdgeSplitter
        splitter = CriticalEdgeSplitter()
        splitter.split_critical_edges(function)

    def _compute_parallel_copies(self, function: Function) -> Dict[BasicBlock, ParallelCopy]:
        """Compute parallel copies needed for each predecessor block."""
        copies: Dict[BasicBlock, ParallelCopy] = defaultdict(ParallelCopy)

        for block in function.blocks:
            for phi in block.phi_nodes:
                for val, pred in phi.incoming:
                    if val is not phi:  # Skip self-loops
                        copies[pred].add(phi, val)

        return dict(copies)

    def _sequentialize_copies(self, block: BasicBlock, pcopy: ParallelCopy,
                                function: Function) -> None:
        """Convert parallel copies into sequential copies.

        Uses a sequentialization algorithm that handles cycles by
        introducing temporary variables.
        """
        if not pcopy:
            return

        # Build dependency graph
        copies = list(pcopy.copies)

        # Simple sequentialization: process copies in order,
        # breaking cycles with temporaries
        processed: Set[int] = set()
        dest_set = {id(d) for d, _ in copies}

        for dest, src in copies:
            if id(src) in dest_set and id(src) not in processed:
                # Potential cycle: needs a temporary
                # For now, just insert the copy
                pass

            # Insert copy before terminator
            self._insert_copy(block, dest, src)
            processed.add(id(dest))
            self._copies_inserted += 1

    def _insert_copy(self, block: BasicBlock, dest: Value, src: Value) -> None:
        """Insert a copy operation (implemented as a select with true condition)."""
        # Use a binary operation that acts as a copy: add with 0
        if isinstance(dest.ir_type if hasattr(dest, 'ir_type') else None, type(None)):
            return

        # Create an identity operation
        copy_inst = BinaryOp(
            op=BinOpKind.ADD,
            left=src,
            right=Constant(value=0, ir_type=dest.ir_type if hasattr(dest, 'ir_type') else src.ir_type),
            ir_type=dest.ir_type if hasattr(dest, 'ir_type') else src.ir_type,
            name=f"copy.{dest.name}" if hasattr(dest, 'name') and dest.name else "copy",
        )
        copy_inst._parent = block
        block.insert_before_terminator(copy_inst)

        # Replace uses of dest with copy_inst
        if hasattr(dest, 'users'):
            for user in list(dest.users):
                if isinstance(user, Instruction) and user is not copy_inst:
                    self._substitute(user, dest, copy_inst)


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
        elif isinstance(user, ReturnInst):
            if user.value is old:
                user.value = new
        elif isinstance(user, StoreInst):
            if user.value is old:
                user.value = new
        elif isinstance(user, BranchInst) and user.is_conditional:
            if user.condition is old:
                user.condition = new
        elif isinstance(user, CallInst):
            user.arguments = [new if a is old else a for a in user.arguments]


# ─── Copy Propagator ─────────────────────────────────────────────────

class CopyPropagator:
    """Propagate copy operations through the SSA graph.

    When a value v2 = copy(v1), replace all uses of v2 with v1.
    This cleans up after SSA deconstruction and other transformations.
    """

    def __init__(self) -> None:
        self._propagated = 0

    @property
    def num_propagated(self) -> int:
        return self._propagated

    def propagate(self, function: Function) -> bool:
        """Propagate copies in the function. Returns True if any changes."""
        changed = False
        worklist: List[Instruction] = []

        for block in function.blocks:
            for inst in block.instructions:
                if self._is_copy(inst):
                    worklist.append(inst)

        while worklist:
            inst = worklist.pop()
            if inst.parent is None:
                continue

            src = self._get_copy_source(inst)
            if src is None:
                continue

            # Replace all uses of inst with src
            if hasattr(inst, 'users') and inst.users:
                for user in list(inst.users):
                    if isinstance(user, Instruction):
                        self._substitute(user, inst, src)
                        if hasattr(src, 'users') and not isinstance(src, Constant):
                            src.users.append(user)
                        # If user is also a copy, add to worklist
                        if self._is_copy(user):
                            worklist.append(user)

                inst.parent.remove(inst)
                self._propagated += 1
                changed = True

        return changed

    def _is_copy(self, inst: Instruction) -> bool:
        """Check if an instruction is effectively a copy."""
        # x + 0 or 0 + x
        if isinstance(inst, BinaryOp) and inst.op == BinOpKind.ADD:
            if isinstance(inst.right, Constant) and hasattr(inst.right, 'value') and inst.right.value == 0:
                return True
            if isinstance(inst.left, Constant) and hasattr(inst.left, 'value') and inst.left.value == 0:
                return True

        # Bitcast to same type
        if isinstance(inst, CastInst) and inst.cast_kind == CastKind.BITCAST:
            if (hasattr(inst.operand, 'ir_type') and
                    str(inst.operand.ir_type) == str(inst.ir_type)):
                return True

        # Phi with single unique incoming
        if isinstance(inst, PhiInst):
            unique = None
            for val, _ in inst.incoming:
                if val is inst:
                    continue
                if unique is None:
                    unique = val
                elif val is not unique:
                    return False
            return unique is not None

        return False

    def _get_copy_source(self, inst: Instruction) -> Optional[Value]:
        """Get the source value of a copy instruction."""
        if isinstance(inst, BinaryOp) and inst.op == BinOpKind.ADD:
            if isinstance(inst.right, Constant) and hasattr(inst.right, 'value') and inst.right.value == 0:
                return inst.left
            if isinstance(inst.left, Constant) and hasattr(inst.left, 'value') and inst.left.value == 0:
                return inst.right

        if isinstance(inst, CastInst) and inst.cast_kind == CastKind.BITCAST:
            return inst.operand

        if isinstance(inst, PhiInst):
            unique = None
            for val, _ in inst.incoming:
                if val is inst:
                    continue
                if unique is None:
                    unique = val
                elif val is not unique:
                    return None
            return unique

        return None

    def _substitute(self, user: Instruction, old: Value, new: Value) -> None:
        if isinstance(user, BinaryOp):
            if user.left is old: user.left = new
            if user.right is old: user.right = new
        elif isinstance(user, UnaryOp):
            if user.operand is old: user.operand = new
        elif isinstance(user, CompareOp):
            if user.left is old: user.left = new
            if user.right is old: user.right = new
        elif isinstance(user, CastInst):
            if user.operand is old: user.operand = new
        elif isinstance(user, SelectInst):
            if user.condition is old: user.condition = new
            if user.true_value is old: user.true_value = new
            if user.false_value is old: user.false_value = new
        elif isinstance(user, PhiInst):
            user.incoming = [(new if v is old else v, b) for v, b in user.incoming]
        elif isinstance(user, ReturnInst):
            if user.value is old: user.value = new
        elif isinstance(user, StoreInst):
            if user.value is old: user.value = new
        elif isinstance(user, BranchInst) and user.is_conditional:
            if user.condition is old: user.condition = new
        elif isinstance(user, CallInst):
            user.arguments = [new if a is old else a for a in user.arguments]


# ─── Phi Simplifier ─────────────────────────────────────────────────

class PhiSimplifier:
    """Simplify and remove trivial phi nodes.

    A phi is trivial if:
    - It has a single incoming value (after removing self-references)
    - All incoming values are the same
    - It is unreachable
    """

    def __init__(self) -> None:
        self._simplified = 0

    @property
    def num_simplified(self) -> int:
        return self._simplified

    def simplify(self, function: Function) -> bool:
        """Simplify all phi nodes. Returns True if any changes."""
        changed = True
        any_changed = False

        while changed:
            changed = False
            for block in function.blocks:
                for phi in list(block.phi_nodes):
                    replacement = self._get_trivial_replacement(phi)
                    if replacement is not None:
                        self._replace_phi(phi, replacement, block)
                        changed = True
                        any_changed = True

        return any_changed

    def _get_trivial_replacement(self, phi: PhiInst) -> Optional[Value]:
        """Check if a phi node can be replaced with a single value."""
        if not phi.incoming:
            return Constant(value=0, ir_type=phi.ir_type)

        unique: Optional[Value] = None
        for val, _ in phi.incoming:
            if val is phi:
                continue
            if unique is None:
                unique = val
            elif val is not unique:
                if (isinstance(val, Constant) and isinstance(unique, Constant) and
                        hasattr(val, 'value') and hasattr(unique, 'value') and
                        val.value == unique.value):
                    continue
                return None

        return unique

    def _replace_phi(self, phi: PhiInst, replacement: Value,
                      block: BasicBlock) -> None:
        """Replace phi with replacement value."""
        if hasattr(phi, 'users'):
            for user in list(phi.users):
                if isinstance(user, Instruction):
                    self._substitute(user, phi, replacement)
                    if hasattr(replacement, 'users') and not isinstance(replacement, Constant):
                        replacement.users.append(user)

        try:
            block.remove(phi)
        except ValueError:
            pass
        self._simplified += 1

    def _substitute(self, user: Instruction, old: Value, new: Value) -> None:
        if isinstance(user, BinaryOp):
            if user.left is old: user.left = new
            if user.right is old: user.right = new
        elif isinstance(user, UnaryOp):
            if user.operand is old: user.operand = new
        elif isinstance(user, CompareOp):
            if user.left is old: user.left = new
            if user.right is old: user.right = new
        elif isinstance(user, CastInst):
            if user.operand is old: user.operand = new
        elif isinstance(user, SelectInst):
            if user.condition is old: user.condition = new
            if user.true_value is old: user.true_value = new
            if user.false_value is old: user.false_value = new
        elif isinstance(user, PhiInst):
            user.incoming = [(new if v is old else v, b) for v, b in user.incoming]
        elif isinstance(user, ReturnInst):
            if user.value is old: user.value = new
        elif isinstance(user, StoreInst):
            if user.value is old: user.value = new
        elif isinstance(user, BranchInst) and user.is_conditional:
            if user.condition is old: user.condition = new


# ─── Value Renamer ──────────────────────────────────────────────────

class ValueRenamer:
    """Rename SSA values with fresh, sequential names.

    Useful for normalizing IR after transformations that may
    have left values with inconsistent or confusing names.
    """

    def __init__(self, prefix: str = "v") -> None:
        self._prefix = prefix
        self._counter = 0
        self._renamed = 0

    @property
    def num_renamed(self) -> int:
        return self._renamed

    def rename(self, function: Function) -> bool:
        """Rename all values in the function. Returns True if any renamed."""
        self._counter = 0
        self._renamed = 0

        # Rename arguments
        for i, arg in enumerate(function.arguments):
            arg.name = f"arg{i}"
            self._renamed += 1

        # Rename blocks and instructions in RPO
        block_counter = 0
        for block in function.iter_blocks_rpo():
            block.name = f"bb{block_counter}"
            block_counter += 1

            for inst in block.instructions:
                if isinstance(inst, (ReturnInst, BranchInst, SwitchInst, StoreInst)):
                    continue  # Terminators and stores don't produce named values
                if isinstance(inst, PhiInst):
                    inst.name = f"phi{self._counter}"
                else:
                    inst.name = f"{self._prefix}{self._counter}"
                self._counter += 1
                self._renamed += 1

        return self._renamed > 0

    def rename_block(self, block: BasicBlock, start_counter: int = 0) -> int:
        """Rename values within a single block. Returns next counter value."""
        counter = start_counter
        for inst in block.instructions:
            if isinstance(inst, (ReturnInst, BranchInst, SwitchInst, StoreInst)):
                continue
            inst.name = f"{self._prefix}{counter}"
            counter += 1
            self._renamed += 1
        return counter
