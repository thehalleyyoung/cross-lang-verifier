"""
Memory-to-register promotion for the Cross-Language Equivalence Verifier.

Implements the Cytron et al. algorithm for promoting stack allocations
(alloca instructions) to SSA registers with proper phi node placement.

Provides:
- Mem2Reg: main pass combining analysis + phi insertion + renaming
- PromotableAllocaAnalysis: identify allocas safe to promote
- PhiInsertion: insert phi nodes at dominance frontiers
- SSARenamer: rename variables to SSA form
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
    AllocaInst, LoadInst, StoreInst,
    PhiInst, BranchInst, ReturnInst,
    GetElementPtrInst, CallInst,
    MemcpyInst, MemsetInst,
    CastInst, BinaryOp,
)
from ...ir.types import IRType, IntType, PointerType
from .pass_manager import FunctionPass, PassResult, AnalysisManager

logger = logging.getLogger(__name__)


# ─── Promotable Alloca Analysis ────────────────────────────────────────

@dataclass
class AllocaInfo:
    """Information about an alloca instruction for promotion analysis."""
    alloca: AllocaInst
    stores: List[StoreInst] = field(default_factory=list)
    loads: List[LoadInst] = field(default_factory=list)
    defining_blocks: Set[int] = field(default_factory=set)
    using_blocks: Set[int] = field(default_factory=set)
    is_promotable: bool = True
    reason_not_promotable: str = ""
    promoted_type: Optional[IRType] = None
    store_count: int = 0
    load_count: int = 0


class PromotableAllocaAnalysis:
    """Analyze which alloca instructions can be safely promoted to registers.

    An alloca is promotable if:
    - It is only used by load and store instructions (no address taken)
    - No GEP (no partial access)
    - Not used in calls (not passed by pointer)
    - Not used in memcpy/memset
    - All loads and stores use the entire alloca (no partial access)
    """

    def __init__(self) -> None:
        self._alloca_infos: Dict[int, AllocaInfo] = {}

    @property
    def promotable_allocas(self) -> List[AllocaInfo]:
        return [info for info in self._alloca_infos.values() if info.is_promotable]

    @property
    def all_allocas(self) -> List[AllocaInfo]:
        return list(self._alloca_infos.values())

    def analyze(self, function: Function) -> List[AllocaInfo]:
        """Analyze all allocas in the function for promotability."""
        self._alloca_infos.clear()

        # Collect all allocas
        for block in function.blocks:
            for inst in block.instructions:
                if isinstance(inst, AllocaInst):
                    info = AllocaInfo(alloca=inst)
                    info.promoted_type = self._get_promoted_type(inst)
                    self._alloca_infos[inst.id] = info

        if not self._alloca_infos:
            return []

        # Analyze uses of each alloca
        alloca_ids = set(self._alloca_infos.keys())
        for block in function.blocks:
            for inst in block.instructions:
                self._check_use(inst, alloca_ids, block)

        return self.promotable_allocas

    def _get_promoted_type(self, alloca: AllocaInst) -> Optional[IRType]:
        """Determine the type to promote the alloca to."""
        if isinstance(alloca.ir_type, PointerType):
            return alloca.ir_type.pointee
        return alloca.ir_type

    def _check_use(self, inst: Instruction, alloca_ids: Set[int], block: BasicBlock) -> None:
        """Check if an instruction uses any tracked alloca in a non-promotable way."""
        if isinstance(inst, LoadInst):
            addr = inst.address
            if isinstance(addr, Instruction) and addr.id in alloca_ids:
                info = self._alloca_infos[addr.id]
                info.loads.append(inst)
                info.load_count += 1
                info.using_blocks.add(block.id)

        elif isinstance(inst, StoreInst):
            addr = inst.address
            if isinstance(addr, Instruction) and addr.id in alloca_ids:
                info = self._alloca_infos[addr.id]
                info.stores.append(inst)
                info.store_count += 1
                info.defining_blocks.add(block.id)
            # Check if alloca is stored AS a value (address escapes)
            val = inst.value
            if isinstance(val, Instruction) and val.id in alloca_ids:
                self._mark_not_promotable(val.id, "address stored to memory (escapes)")

        elif isinstance(inst, GetElementPtrInst):
            base = inst.base
            if isinstance(base, Instruction) and base.id in alloca_ids:
                self._mark_not_promotable(base.id, "used in GEP (partial access)")

        elif isinstance(inst, CallInst):
            for arg in inst.arguments:
                if isinstance(arg, Instruction) and arg.id in alloca_ids:
                    self._mark_not_promotable(arg.id, "passed to function call (address escapes)")

        elif isinstance(inst, (MemcpyInst, MemsetInst)):
            for operand in [inst.dest]:
                if isinstance(operand, Instruction) and operand.id in alloca_ids:
                    self._mark_not_promotable(operand.id, "used in memcpy/memset")
            if isinstance(inst, MemcpyInst):
                if isinstance(inst.src, Instruction) and inst.src.id in alloca_ids:
                    self._mark_not_promotable(inst.src.id, "used as memcpy source")

        elif isinstance(inst, CastInst):
            if isinstance(inst.operand, Instruction) and inst.operand.id in alloca_ids:
                self._mark_not_promotable(inst.operand.id, "used in cast (address escapes)")

    def _mark_not_promotable(self, alloca_id: int, reason: str) -> None:
        if alloca_id in self._alloca_infos:
            info = self._alloca_infos[alloca_id]
            info.is_promotable = False
            info.reason_not_promotable = reason


# ─── Phi Node Insertion ────────────────────────────────────────────────

class PhiInsertion:
    """Insert phi nodes at dominance frontiers for promoted allocas.

    Implements the phi placement algorithm from Cytron et al.:
    For each alloca, place phi nodes at the iterated dominance frontier
    of blocks containing stores to that alloca.
    """

    def __init__(self, function: Function) -> None:
        self._function = function
        self._inserted_phis: Dict[int, Dict[int, PhiInst]] = {}  # alloca_id -> block_id -> phi

    def insert_phis(self, alloca_infos: List[AllocaInfo],
                     dom_frontiers: Dict[int, Set[int]]) -> Dict[int, Dict[int, PhiInst]]:
        """Insert phi nodes for all promotable allocas.

        Args:
            alloca_infos: list of promotable allocas with their store locations
            dom_frontiers: mapping from block_id to its dominance frontier block_ids

        Returns:
            mapping from alloca_id to (block_id -> phi instruction)
        """
        self._inserted_phis.clear()
        block_map = {b.id: b for b in self._function.blocks}

        for info in alloca_infos:
            if not info.is_promotable:
                continue
            phis = self._insert_phis_for_alloca(info, dom_frontiers, block_map)
            self._inserted_phis[info.alloca.id] = phis

        return self._inserted_phis

    def _insert_phis_for_alloca(
        self, info: AllocaInfo,
        dom_frontiers: Dict[int, Set[int]],
        block_map: Dict[int, BasicBlock]
    ) -> Dict[int, PhiInst]:
        """Insert phi nodes for a single alloca using iterated dominance frontiers."""
        promoted_type = info.promoted_type
        if promoted_type is None:
            return {}

        # Compute iterated dominance frontier of defining blocks
        defining_blocks = info.defining_blocks
        phi_blocks: Set[int] = set()
        worklist = list(defining_blocks)
        visited: Set[int] = set()

        while worklist:
            block_id = worklist.pop()
            if block_id in visited:
                continue
            visited.add(block_id)

            frontier = dom_frontiers.get(block_id, set())
            for df_id in frontier:
                if df_id not in phi_blocks:
                    phi_blocks.add(df_id)
                    worklist.append(df_id)

        # Create phi nodes
        phis: Dict[int, PhiInst] = {}
        for block_id in phi_blocks:
            block = block_map.get(block_id)
            if block is None:
                continue

            phi = PhiInst(
                ir_type=promoted_type,
                incoming=[],
                name=f"mem2reg.{info.alloca.name or info.alloca.id}"
            )
            phi._parent = block
            block.add_phi(phi)
            phis[block_id] = phi

        return phis


# ─── SSA Variable Renaming ─────────────────────────────────────────────

class SSARenamer:
    """Rename variables according to SSA form after phi insertion.

    Walks the dominator tree in preorder, maintaining a stack of
    definitions for each alloca. Loads are replaced with the current
    definition, stores push new definitions, and phi nodes are wired up.
    """

    def __init__(self, function: Function) -> None:
        self._function = function
        self._stacks: Dict[int, List[Value]] = defaultdict(list)
        self._removed_instructions: List[Tuple[BasicBlock, Instruction]] = []
        self._alloca_types: Dict[int, IRType] = {}

    def rename(self, alloca_infos: List[AllocaInfo],
               inserted_phis: Dict[int, Dict[int, PhiInst]],
               dom_tree_children: Dict[int, List[int]],
               block_map: Dict[int, BasicBlock]) -> None:
        """Perform SSA renaming.

        Args:
            alloca_infos: promotable allocas
            inserted_phis: phi nodes inserted by PhiInsertion
            dom_tree_children: mapping from block_id to child block_ids in dom tree
            block_map: mapping from block_id to BasicBlock
        """
        self._stacks.clear()
        self._removed_instructions.clear()

        for info in alloca_infos:
            alloca_id = info.alloca.id
            self._alloca_types[alloca_id] = info.promoted_type
            # Initialize stack with undef
            undef = Constant(value=0, ir_type=info.promoted_type)
            self._stacks[alloca_id].append(undef)

        entry = self._function.entry_block
        if entry is None:
            return

        self._rename_block(entry, alloca_infos, inserted_phis,
                           dom_tree_children, block_map)

        # Remove loads, stores, and allocas
        for block, inst in reversed(self._removed_instructions):
            try:
                block.remove(inst)
            except ValueError:
                pass

    def _rename_block(
        self, block: BasicBlock,
        alloca_infos: List[AllocaInfo],
        inserted_phis: Dict[int, Dict[int, PhiInst]],
        dom_tree_children: Dict[int, List[int]],
        block_map: Dict[int, BasicBlock]
    ) -> None:
        """Rename variables in a block and recurse into dominated blocks."""
        # Track how many definitions we push for each alloca in this block
        push_counts: Dict[int, int] = defaultdict(int)
        alloca_ids = {info.alloca.id for info in alloca_infos}

        # Process phi nodes inserted for this block
        for alloca_id, phi_map in inserted_phis.items():
            if block.id in phi_map:
                phi = phi_map[block.id]
                self._stacks[alloca_id].append(phi)
                push_counts[alloca_id] += 1

        # Process instructions in this block
        for inst in list(block.instructions):
            if isinstance(inst, LoadInst):
                addr = inst.address
                if isinstance(addr, Instruction) and addr.id in alloca_ids:
                    # Replace load with current definition
                    current_def = self._current_def(addr.id)
                    self._replace_uses(inst, current_def)
                    self._removed_instructions.append((block, inst))

            elif isinstance(inst, StoreInst):
                addr = inst.address
                if isinstance(addr, Instruction) and addr.id in alloca_ids:
                    # Push new definition
                    self._stacks[addr.id].append(inst.value)
                    push_counts[addr.id] += 1
                    self._removed_instructions.append((block, inst))

            elif isinstance(inst, AllocaInst) and inst.id in alloca_ids:
                self._removed_instructions.append((block, inst))

        # Fill in phi operands in successor blocks
        for succ in block.successors:
            for alloca_id, phi_map in inserted_phis.items():
                if succ.id in phi_map:
                    phi = phi_map[succ.id]
                    current_def = self._current_def(alloca_id)
                    phi.incoming.append((current_def, block))
                    if hasattr(current_def, 'users'):
                        current_def.users.append(phi)

        # Recurse into dominated children
        children = dom_tree_children.get(block.id, [])
        for child_id in children:
            child = block_map.get(child_id)
            if child is not None:
                self._rename_block(child, alloca_infos, inserted_phis,
                                   dom_tree_children, block_map)

        # Pop definitions pushed in this block
        for alloca_id, count in push_counts.items():
            for _ in range(count):
                self._stacks[alloca_id].pop()

    def _current_def(self, alloca_id: int) -> Value:
        """Get the current SSA definition for an alloca."""
        stack = self._stacks.get(alloca_id)
        if stack:
            return stack[-1]
        typ = self._alloca_types.get(alloca_id)
        return Constant(value=0, ir_type=typ)

    def _replace_uses(self, old_val: Value, new_val: Value) -> None:
        """Replace all uses of old_val with new_val."""
        if not hasattr(old_val, 'users'):
            return
        for user in list(old_val.users):
            if isinstance(user, Instruction):
                self._substitute_in_instruction(user, old_val, new_val)
                if hasattr(new_val, 'users'):
                    new_val.users.append(user)

    def _substitute_in_instruction(self, inst: Instruction, old: Value, new: Value) -> None:
        if isinstance(inst, BinaryOp):
            if inst.left is old:
                inst.left = new
            if inst.right is old:
                inst.right = new
        elif isinstance(inst, LoadInst):
            if inst.address is old:
                inst.address = new
        elif isinstance(inst, StoreInst):
            if inst.value is old:
                inst.value = new
            if inst.address is old:
                inst.address = new
        elif isinstance(inst, CastInst):
            if inst.operand is old:
                inst.operand = new
        elif isinstance(inst, PhiInst):
            inst.incoming = [(new if v is old else v, b) for v, b in inst.incoming]
        elif isinstance(inst, ReturnInst):
            if inst.value is old:
                inst.value = new
        elif isinstance(inst, BranchInst) and inst.is_conditional:
            if inst.condition is old:
                inst.condition = new
        elif isinstance(inst, CallInst):
            inst.arguments = [new if a is old else a for a in inst.arguments]


# ─── Dominance Frontier Computation ────────────────────────────────────

class DominanceFrontierComputer:
    """Compute dominance frontiers for all blocks.

    Uses the algorithm from Cooper, Harvey, and Kennedy:
    For each join point (block with >1 predecessor), walk up the
    dominator tree from each predecessor until hitting the immediate
    dominator of the join point.
    """

    def __init__(self, function: Function) -> None:
        self._function = function

    def compute(self) -> Tuple[Dict[int, Set[int]], Dict[int, int], Dict[int, List[int]]]:
        """Compute dominance frontiers, idom map, and dom tree children.

        Returns:
            (frontiers, idom_map, children_map) where:
            - frontiers: block_id -> set of block_ids in its dominance frontier
            - idom_map: block_id -> immediate dominator block_id
            - children_map: block_id -> list of block_ids of dominator tree children
        """
        self._function.compute_dominators()

        blocks = self._function.blocks
        block_map = {b.id: b for b in blocks}

        idom_map: Dict[int, int] = {}
        children_map: Dict[int, List[int]] = defaultdict(list)
        frontiers: Dict[int, Set[int]] = {b.id: set() for b in blocks}

        for block in blocks:
            if block.idom is not None:
                idom_map[block.id] = block.idom.id
                children_map[block.idom.id].append(block.id)

        # Cooper-Harvey-Kennedy algorithm
        for block in blocks:
            preds = block.predecessors
            if len(preds) < 2:
                continue
            for pred in preds:
                runner_id = pred.id
                block_idom_id = idom_map.get(block.id)
                while runner_id != block_idom_id and runner_id is not None:
                    frontiers[runner_id].add(block.id)
                    runner_id = idom_map.get(runner_id)

        return frontiers, idom_map, dict(children_map)


# ─── Mem2Reg Pass ──────────────────────────────────────────────────────

class Mem2Reg(FunctionPass):
    """Memory-to-register promotion using the Cytron et al. algorithm.

    Promotes stack allocations to SSA registers by:
    1. Identifying promotable allocas
    2. Computing dominance frontiers
    3. Inserting phi nodes at iterated dominance frontiers
    4. Renaming variables in dominator tree preorder
    """

    _name = "mem2reg"
    _description = "Promote memory to registers (SSA construction)"
    _invalidated_analyses = ["cfg", "domtree", "loops", "alias"]

    def run_on_function(self, function: Function, analyses: AnalysisManager) -> PassResult:
        # Step 1: Find promotable allocas
        alloca_analysis = PromotableAllocaAnalysis()
        promotable = alloca_analysis.analyze(function)

        if not promotable:
            return PassResult.UNCHANGED

        logger.debug(f"mem2reg: found {len(promotable)} promotable allocas in {function.name}")

        # Step 2: Compute dominance frontiers
        df_computer = DominanceFrontierComputer(function)
        frontiers, idom_map, children_map = df_computer.compute()

        # Step 3: Insert phi nodes
        phi_inserter = PhiInsertion(function)
        inserted_phis = phi_inserter.insert_phis(promotable, frontiers)

        # Step 4: Rename variables
        block_map = {b.id: b for b in function.blocks}
        renamer = SSARenamer(function)
        renamer.rename(promotable, inserted_phis, children_map, block_map)

        # Step 5: Simplify trivial phis
        self._simplify_trivial_phis(function)

        # Record statistics
        total_phis = sum(len(pm) for pm in inserted_phis.values())
        self.stats.increment("allocas_promoted", len(promotable))
        self.stats.increment("phis_inserted", total_phis)
        self.stats.instructions_removed += len(renamer._removed_instructions)

        return PassResult.CHANGED

    def _simplify_trivial_phis(self, function: Function) -> None:
        """Remove phi nodes that are trivially reducible.

        A phi is trivial if:
        - All incoming values are the same (or self-referential)
        - It has exactly one unique non-self incoming value
        """
        changed = True
        while changed:
            changed = False
            for block in function.blocks:
                for phi in list(block.phi_nodes):
                    replacement = self._get_trivial_replacement(phi)
                    if replacement is not None:
                        self._replace_phi(phi, replacement, block)
                        changed = True

    def _get_trivial_replacement(self, phi: PhiInst) -> Optional[Value]:
        """Check if a phi is trivial and return its replacement value."""
        unique: Optional[Value] = None
        for val, _ in phi.incoming:
            if val is phi:
                continue  # Skip self-references
            if unique is None:
                unique = val
            elif val is not unique:
                # Check if they're the same constant
                if (isinstance(val, Constant) and isinstance(unique, Constant) and
                        hasattr(val, 'value') and hasattr(unique, 'value') and
                        val.value == unique.value):
                    continue
                return None  # Multiple distinct values
        return unique

    def _replace_phi(self, phi: PhiInst, replacement: Value, block: BasicBlock) -> None:
        """Replace a trivial phi with its replacement value."""
        if hasattr(phi, 'users'):
            for user in list(phi.users):
                if isinstance(user, Instruction):
                    self._substitute(user, phi, replacement)
                    if hasattr(replacement, 'users'):
                        replacement.users.append(user)

        try:
            block.remove(phi)
        except ValueError:
            pass
        self.stats.increment("trivial_phis_removed")

    def _substitute(self, inst: Instruction, old: Value, new: Value) -> None:
        if isinstance(inst, BinaryOp):
            if inst.left is old:
                inst.left = new
            if inst.right is old:
                inst.right = new
        elif isinstance(inst, PhiInst):
            inst.incoming = [(new if v is old else v, b) for v, b in inst.incoming]
        elif isinstance(inst, ReturnInst):
            if inst.value is old:
                inst.value = new
        elif isinstance(inst, BranchInst) and inst.is_conditional:
            if inst.condition is old:
                inst.condition = new
        elif isinstance(inst, StoreInst):
            if inst.value is old:
                inst.value = new
        elif isinstance(inst, CastInst):
            if inst.operand is old:
                inst.operand = new
        elif isinstance(inst, CallInst):
            inst.arguments = [new if a is old else a for a in inst.arguments]


# ─── Single-Store Optimization ─────────────────────────────────────────

class SingleStorePromoter:
    """Optimized promotion for allocas with a single store.

    If an alloca has exactly one store, and all loads are dominated
    by that store, we can directly replace loads with the stored value
    without inserting any phi nodes.
    """

    def __init__(self, function: Function) -> None:
        self._function = function

    def try_promote(self, info: AllocaInfo) -> bool:
        """Try to promote an alloca with a single store.

        Returns True if promotion was successful.
        """
        if info.store_count != 1 or not info.stores:
            return False

        store = info.stores[0]
        stored_value = store.value
        store_block = store.parent

        if store_block is None:
            return False

        # Check that all loads are dominated by the store
        for load in info.loads:
            if load.parent is None:
                return False
            if not self._is_dominated_by_store(load, store, store_block):
                return False

        # All loads dominated by the single store: replace loads directly
        for load in info.loads:
            self._replace_load(load, stored_value)

        # Remove the store and alloca
        if store.parent is not None:
            store.parent.remove(store)
        alloca = info.alloca
        if alloca.parent is not None:
            alloca.parent.remove(alloca)

        return True

    def _is_dominated_by_store(self, load: LoadInst, store: StoreInst,
                                store_block: BasicBlock) -> bool:
        """Check if a load is dominated by a store."""
        load_block = load.parent
        if load_block is None:
            return False

        if load_block is store_block:
            # Same block: check instruction order
            for inst in store_block.instructions:
                if inst is store:
                    return True
                if inst is load:
                    return False
            return False

        # Different blocks: use dominator relation
        if store_block.idom is None and load_block.idom is None:
            return False
        return self._dominates(store_block, load_block)

    def _dominates(self, a: BasicBlock, b: BasicBlock) -> bool:
        """Check if block a dominates block b."""
        current = b
        visited: Set[int] = set()
        while current is not None:
            if current is a:
                return True
            if current.id in visited:
                break
            visited.add(current.id)
            current = current.idom
        return False

    def _replace_load(self, load: LoadInst, replacement: Value) -> None:
        if hasattr(load, 'users'):
            for user in list(load.users):
                if isinstance(user, Instruction):
                    self._substitute(user, load, replacement)
        if load.parent is not None:
            load.parent.remove(load)

    def _substitute(self, inst: Instruction, old: Value, new: Value) -> None:
        if isinstance(inst, BinaryOp):
            if inst.left is old:
                inst.left = new
            if inst.right is old:
                inst.right = new
        elif isinstance(inst, PhiInst):
            inst.incoming = [(new if v is old else v, b) for v, b in inst.incoming]
        elif isinstance(inst, ReturnInst):
            if inst.value is old:
                inst.value = new
        elif isinstance(inst, BranchInst) and inst.is_conditional:
            if inst.condition is old:
                inst.condition = new
        elif isinstance(inst, StoreInst):
            if inst.value is old:
                inst.value = new
        elif isinstance(inst, CastInst):
            if inst.operand is old:
                inst.operand = new
        elif isinstance(inst, CallInst):
            inst.arguments = [new if a is old else a for a in inst.arguments]


# ─── Multi-Load Single-Block Optimization ──────────────────────────────

class LocalPromoter:
    """Promote allocas that are only used within a single basic block.

    When all loads and stores to an alloca are in the same block,
    we can simply forward stores to loads in program order without
    needing phi nodes or dominance analysis.
    """

    def __init__(self) -> None:
        self._promotions = 0

    @property
    def num_promotions(self) -> int:
        return self._promotions

    def try_promote(self, info: AllocaInfo) -> bool:
        """Try to promote an alloca with all uses in a single block."""
        if not info.is_promotable:
            return False

        blocks = info.defining_blocks | info.using_blocks
        if len(blocks) > 1:
            return False
        if not blocks:
            # No uses at all: just remove the alloca
            if info.alloca.parent is not None:
                info.alloca.parent.remove(info.alloca)
                self._promotions += 1
            return True

        block_id = next(iter(blocks))
        block = info.alloca.parent
        if block is None:
            return False

        # Find the block containing the uses
        for b in [block] + list(block.successors) + list(block.predecessors):
            if b.id == block_id:
                block = b
                break

        # Process instructions in order, forwarding stores to loads
        current_value: Optional[Value] = Constant(value=0, ir_type=info.promoted_type)
        alloca_id = info.alloca.id
        to_remove: List[Instruction] = []

        for inst in list(block.instructions):
            if isinstance(inst, StoreInst):
                if isinstance(inst.address, Instruction) and inst.address.id == alloca_id:
                    current_value = inst.value
                    to_remove.append(inst)

            elif isinstance(inst, LoadInst):
                if isinstance(inst.address, Instruction) and inst.address.id == alloca_id:
                    if current_value is not None:
                        self._replace_uses(inst, current_value)
                    to_remove.append(inst)

        # Remove processed instructions
        for inst in reversed(to_remove):
            try:
                block.remove(inst)
            except ValueError:
                pass

        # Remove the alloca
        if info.alloca.parent is not None:
            try:
                info.alloca.parent.remove(info.alloca)
            except ValueError:
                pass

        self._promotions += 1
        return True

    def _replace_uses(self, old: Value, new: Value) -> None:
        if not hasattr(old, 'users'):
            return
        for user in list(old.users):
            if isinstance(user, Instruction):
                if isinstance(user, BinaryOp):
                    if user.left is old:
                        user.left = new
                    if user.right is old:
                        user.right = new
                elif isinstance(user, ReturnInst):
                    if user.value is old:
                        user.value = new
                elif isinstance(user, StoreInst):
                    if user.value is old:
                        user.value = new
                elif isinstance(user, CastInst):
                    if user.operand is old:
                        user.operand = new
                elif isinstance(user, PhiInst):
                    user.incoming = [(new if v is old else v, b) for v, b in user.incoming]
