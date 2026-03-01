"""
Function inlining for the Cross-Language Equivalence Verifier.

Provides:
- FunctionInliner: main pass with cost model and inlining transformation
- InlineCostModel: compute inline cost/benefit for call sites
- InlineDecision: enum of inlining decisions
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple

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
    MemcpyInst, MemsetInst,
)
from ...ir.types import IRType, IntType, FloatType, PointerType, FunctionType, VoidType
from .pass_manager import ModulePass, PassResult, AnalysisManager

logger = logging.getLogger(__name__)


# ─── Inline Decision ───────────────────────────────────────────────────

class InlineDecision(Enum):
    """Decision about whether to inline a call site."""
    INLINE = auto()          # Should inline
    NO_INLINE = auto()       # Should not inline
    ALWAYS_INLINE = auto()   # Must inline (annotated)
    NEVER_INLINE = auto()    # Must not inline (recursive, too large, etc.)


# ─── Cost Model ────────────────────────────────────────────────────────

@dataclass
class InlineCost:
    """Cost estimate for inlining a function at a call site."""
    instruction_cost: int = 0
    call_overhead_saved: int = 0
    argument_bonus: int = 0
    alloca_bonus: int = 0
    constant_bonus: int = 0
    single_block_bonus: int = 0
    small_function_bonus: int = 0
    recursive_penalty: int = 0
    cold_penalty: int = 0

    @property
    def total_cost(self) -> int:
        return (self.instruction_cost
                - self.call_overhead_saved
                - self.argument_bonus
                - self.alloca_bonus
                - self.constant_bonus
                - self.single_block_bonus
                - self.small_function_bonus
                + self.recursive_penalty
                + self.cold_penalty)

    def __str__(self) -> str:
        return (f"InlineCost(total={self.total_cost}, "
                f"inst={self.instruction_cost}, "
                f"saved={self.call_overhead_saved}, "
                f"bonuses={self.argument_bonus + self.alloca_bonus + self.constant_bonus})")


@dataclass
class InlineCostConfig:
    """Configuration for the inline cost model."""
    threshold: int = 225
    call_overhead: int = 40
    instruction_cost_default: int = 5
    alloca_cost: int = 10
    call_cost: int = 25
    branch_cost: int = 3
    load_store_cost: int = 7
    small_function_threshold: int = 30
    small_function_bonus_value: int = 75
    single_block_bonus_value: int = 50
    constant_arg_bonus_per_arg: int = 20
    max_callee_size: int = 500
    max_inline_depth: int = 5


class InlineCostModel:
    """Compute inline cost/benefit analysis for call sites.

    Uses instruction counting with bonuses for small functions,
    constant arguments, and single basic block functions.
    """

    def __init__(self, config: Optional[InlineCostConfig] = None) -> None:
        self._config = config or InlineCostConfig()

    @property
    def config(self) -> InlineCostConfig:
        return self._config

    def compute_cost(self, call: CallInst, callee: Function,
                      caller: Function, depth: int = 0) -> InlineCost:
        """Compute the inline cost for a specific call site."""
        cost = InlineCost()

        # Count instructions in callee
        inst_count = 0
        for block in callee.blocks:
            for inst in block.instructions:
                inst_count += 1
                cost.instruction_cost += self._instruction_cost(inst)

        # Call overhead saved
        cost.call_overhead_saved = self._config.call_overhead

        # Small function bonus
        if inst_count <= self._config.small_function_threshold:
            cost.small_function_bonus = self._config.small_function_bonus_value

        # Single block bonus
        if callee.num_blocks == 1:
            cost.single_block_bonus = self._config.single_block_bonus_value

        # Constant argument bonus
        for arg in call.arguments:
            if isinstance(arg, Constant):
                cost.constant_bonus += self._config.constant_arg_bonus_per_arg

        # Alloca bonus: if callee has allocas, inlining can merge them with caller's frame
        for block in callee.blocks:
            for inst in block.instructions:
                if isinstance(inst, AllocaInst):
                    cost.alloca_bonus += 5

        # Recursive penalty
        if self._is_recursive(call, callee):
            cost.recursive_penalty = 500

        # Depth penalty
        if depth > 0:
            cost.recursive_penalty += depth * 50

        return cost

    def should_inline(self, call: CallInst, callee: Function,
                       caller: Function, depth: int = 0) -> InlineDecision:
        """Decide whether to inline a call site."""
        # Never inline functions that are too large
        if callee.instruction_count > self._config.max_callee_size:
            return InlineDecision.NEVER_INLINE

        # Never inline at excessive depth
        if depth >= self._config.max_inline_depth:
            return InlineDecision.NEVER_INLINE

        # Never inline recursive calls
        if self._is_recursive(call, callee):
            return InlineDecision.NEVER_INLINE

        # Check for always-inline annotation
        if hasattr(callee, 'attributes') and 'always_inline' in getattr(callee, 'attributes', []):
            return InlineDecision.ALWAYS_INLINE

        # Check for no-inline annotation
        if hasattr(callee, 'attributes') and 'no_inline' in getattr(callee, 'attributes', []):
            return InlineDecision.NEVER_INLINE

        # Compute cost
        cost = self.compute_cost(call, callee, caller, depth)

        if cost.total_cost <= self._config.threshold:
            return InlineDecision.INLINE
        else:
            return InlineDecision.NO_INLINE

    def _instruction_cost(self, inst: Instruction) -> int:
        if isinstance(inst, (BinaryOp, UnaryOp, CompareOp)):
            return self._config.instruction_cost_default
        elif isinstance(inst, (LoadInst, StoreInst)):
            return self._config.load_store_cost
        elif isinstance(inst, AllocaInst):
            return self._config.alloca_cost
        elif isinstance(inst, CallInst):
            return self._config.call_cost
        elif isinstance(inst, (BranchInst, SwitchInst)):
            return self._config.branch_cost
        elif isinstance(inst, (CastInst, SelectInst, PhiInst)):
            return self._config.instruction_cost_default
        elif isinstance(inst, ReturnInst):
            return 1
        return self._config.instruction_cost_default

    def _is_recursive(self, call: CallInst, callee: Function) -> bool:
        """Check if a call creates direct recursion."""
        if hasattr(call, 'callee_name'):
            return call.callee_name == callee.name
        return False


# ─── Function Cloner ───────────────────────────────────────────────────

class FunctionCloner:
    """Clone a function body for inlining, remapping values and blocks."""

    def __init__(self) -> None:
        self._value_map: Dict[int, Value] = {}
        self._block_map: Dict[int, BasicBlock] = {}

    def clone_into(self, callee: Function, call: CallInst,
                    caller: Function, call_block: BasicBlock) -> Tuple[BasicBlock, BasicBlock, Optional[Value]]:
        """Clone callee body into caller at the call site.

        Returns:
            (entry_block, exit_block, return_value) of the cloned body.
            entry_block is the cloned entry of the callee.
            exit_block is the merge block after the inlined body.
            return_value is the value returned by the inlined function (or None for void).
        """
        self._value_map.clear()
        self._block_map.clear()

        # Map callee arguments to call arguments
        for i, param in enumerate(callee.arguments):
            if i < len(call.arguments):
                self._value_map[param.id] = call.arguments[i]

        # Clone blocks
        cloned_blocks: List[BasicBlock] = []
        for block in callee.blocks:
            new_block = BasicBlock(name=f"inline.{callee.name}.{block.name}")
            new_block._parent = caller
            self._block_map[block.id] = new_block
            cloned_blocks.append(new_block)

        # Clone instructions
        return_values: List[Tuple[Value, BasicBlock]] = []
        for orig_block, new_block in zip(callee.blocks, cloned_blocks):
            for inst in orig_block.instructions:
                if isinstance(inst, ReturnInst):
                    # Collect return values
                    if inst.value is not None:
                        mapped = self._remap_value(inst.value)
                        return_values.append((mapped, new_block))
                    else:
                        return_values.append((None, new_block))
                else:
                    cloned = self._clone_instruction(inst, new_block)
                    if cloned is not None:
                        new_block.append(cloned)
                        self._value_map[inst.id] = cloned

        # Fix up phi nodes and branch targets
        for orig_block, new_block in zip(callee.blocks, cloned_blocks):
            for inst in new_block.instructions:
                self._remap_instruction_operands(inst)

        # Create merge block
        merge_block = BasicBlock(name=f"inline.{callee.name}.merge")
        merge_block._parent = caller

        # Add branches from return points to merge block
        for ret_val, ret_block in return_values:
            br = BranchInst(target=merge_block)
            br._parent = ret_block
            ret_block.append(br)
            merge_block.add_predecessor(ret_block)

        # Create phi for return value if needed
        return_value: Optional[Value] = None
        if return_values and return_values[0][0] is not None:
            if len(return_values) == 1:
                return_value = return_values[0][0]
            else:
                # Create phi node for multiple return values
                incoming = [(rv, rb) for rv, rb in return_values if rv is not None]
                if incoming:
                    ret_type = callee.return_type
                    phi = PhiInst(ir_type=ret_type, incoming=incoming,
                                  name=f"inline.{callee.name}.retval")
                    phi._parent = merge_block
                    merge_block.add_phi(phi)
                    return_value = phi

        # Add cloned blocks to caller
        for block in cloned_blocks:
            caller.add_block(block)
        caller.add_block(merge_block)

        entry = cloned_blocks[0] if cloned_blocks else merge_block
        return entry, merge_block, return_value

    def _clone_instruction(self, inst: Instruction, new_parent: BasicBlock) -> Optional[Instruction]:
        """Create a clone of an instruction."""
        if isinstance(inst, BinaryOp):
            cloned = BinaryOp(
                op=inst.op,
                left=self._remap_value(inst.left),
                right=self._remap_value(inst.right),
                ir_type=inst.ir_type,
                name=f"inline.{inst.name}" if inst.name else "",
            )
            cloned._parent = new_parent
            return cloned

        elif isinstance(inst, UnaryOp):
            cloned = UnaryOp(
                op=inst.op,
                operand=self._remap_value(inst.operand),
                ir_type=inst.ir_type,
                name=f"inline.{inst.name}" if inst.name else "",
            )
            cloned._parent = new_parent
            return cloned

        elif isinstance(inst, CompareOp):
            cloned = CompareOp(
                predicate=inst.predicate,
                left=self._remap_value(inst.left),
                right=self._remap_value(inst.right),
                ir_type=inst.ir_type,
                name=f"inline.{inst.name}" if inst.name else "",
            )
            cloned._parent = new_parent
            return cloned

        elif isinstance(inst, LoadInst):
            cloned = LoadInst(
                address=self._remap_value(inst.address),
                ir_type=inst.ir_type,
                name=f"inline.{inst.name}" if inst.name else "",
            )
            cloned._parent = new_parent
            return cloned

        elif isinstance(inst, StoreInst):
            cloned = StoreInst(
                value=self._remap_value(inst.value),
                address=self._remap_value(inst.address),
            )
            cloned._parent = new_parent
            return cloned

        elif isinstance(inst, AllocaInst):
            cloned = AllocaInst(
                ir_type=inst.ir_type,
                name=f"inline.{inst.name}" if inst.name else "",
            )
            cloned._parent = new_parent
            return cloned

        elif isinstance(inst, CastInst):
            cloned = CastInst(
                cast_kind=inst.cast_kind,
                operand=self._remap_value(inst.operand),
                ir_type=inst.ir_type,
                name=f"inline.{inst.name}" if inst.name else "",
            )
            cloned._parent = new_parent
            return cloned

        elif isinstance(inst, CallInst):
            cloned = CallInst(
                callee=self._remap_value(inst.callee) if inst.callee else None,
                arguments=[self._remap_value(a) for a in inst.arguments],
                ir_type=inst.ir_type,
                name=f"inline.{inst.name}" if inst.name else "",
            )
            cloned._parent = new_parent
            return cloned

        elif isinstance(inst, BranchInst):
            if inst.is_conditional:
                cloned = BranchInst(
                    condition=self._remap_value(inst.condition),
                    true_block=self._remap_block(inst.true_block),
                    false_block=self._remap_block(inst.false_block),
                )
            else:
                target = inst.target if hasattr(inst, 'target') else inst.true_block
                cloned = BranchInst(target=self._remap_block(target))
            cloned._parent = new_parent
            return cloned

        elif isinstance(inst, SelectInst):
            cloned = SelectInst(
                condition=self._remap_value(inst.condition),
                true_value=self._remap_value(inst.true_value),
                false_value=self._remap_value(inst.false_value),
                ir_type=inst.ir_type,
                name=f"inline.{inst.name}" if inst.name else "",
            )
            cloned._parent = new_parent
            return cloned

        elif isinstance(inst, PhiInst):
            cloned = PhiInst(
                ir_type=inst.ir_type,
                incoming=[(self._remap_value(v), self._remap_block(b))
                          for v, b in inst.incoming],
                name=f"inline.{inst.name}" if inst.name else "",
            )
            cloned._parent = new_parent
            return cloned

        elif isinstance(inst, GetElementPtrInst):
            cloned = GetElementPtrInst(
                base=self._remap_value(inst.base),
                indices=[self._remap_value(i) for i in inst.indices],
                ir_type=inst.ir_type,
                name=f"inline.{inst.name}" if inst.name else "",
            )
            cloned._parent = new_parent
            return cloned

        # Fallback: skip unknown instructions
        logger.debug(f"Cannot clone instruction type: {type(inst).__name__}")
        return None

    def _remap_value(self, val: Value) -> Value:
        """Remap a value through the value map."""
        if val is None:
            return val
        if isinstance(val, Constant):
            return val
        mapped = self._value_map.get(val.id if hasattr(val, 'id') else id(val))
        return mapped if mapped is not None else val

    def _remap_block(self, block: Optional[BasicBlock]) -> Optional[BasicBlock]:
        """Remap a block through the block map."""
        if block is None:
            return None
        return self._block_map.get(block.id, block)

    def _remap_instruction_operands(self, inst: Instruction) -> None:
        """Remap all operands of an instruction (second pass for forward references)."""
        if isinstance(inst, PhiInst):
            inst.incoming = [
                (self._remap_value(v), self._remap_block(b))
                for v, b in inst.incoming
            ]
        elif isinstance(inst, BranchInst):
            if inst.is_conditional:
                inst.condition = self._remap_value(inst.condition)
                inst.true_block = self._remap_block(inst.true_block)
                inst.false_block = self._remap_block(inst.false_block)
            else:
                target = inst.target if hasattr(inst, 'target') else inst.true_block
                remapped = self._remap_block(target)
                if hasattr(inst, 'target'):
                    inst.target = remapped
                else:
                    inst.true_block = remapped


# ─── Function Inliner Pass ────────────────────────────────────────────

class FunctionInliner(ModulePass):
    """Inline function calls based on cost model analysis.

    Processes call sites in bottom-up order (callees before callers),
    applies cost model to decide inlining, then performs the transformation.
    """

    _name = "inline"
    _description = "Function inlining with cost model"
    _invalidated_analyses = ["cfg", "domtree", "loops", "callgraph", "alias"]

    def __init__(self, config: Optional[InlineCostConfig] = None) -> None:
        super().__init__()
        self._cost_model = InlineCostModel(config)
        self._max_module_growth: float = 3.0
        self._inline_count = 0

    def run_on_module(self, module: Module, analyses: AnalysisManager) -> PassResult:
        changed = False
        self._inline_count = 0

        # Build function lookup
        func_map: Dict[str, Function] = {}
        for func in module.functions:
            func_map[func.name] = func

        # Collect all call sites
        call_sites = self._collect_call_sites(module, func_map)

        # Process in bottom-up order
        call_order = self._compute_bottom_up_order(func_map)

        for caller_name in call_order:
            caller = func_map.get(caller_name)
            if caller is None:
                continue

            sites = call_sites.get(caller_name, [])
            for call, call_block, callee_name in sites:
                callee = func_map.get(callee_name)
                if callee is None or callee.num_blocks == 0:
                    continue

                decision = self._cost_model.should_inline(call, callee, caller)
                if decision in (InlineDecision.INLINE, InlineDecision.ALWAYS_INLINE):
                    success = self._perform_inline(call, call_block, callee, caller)
                    if success:
                        changed = True
                        self._inline_count += 1
                        self.stats.increment("functions_inlined")

        if changed:
            self.stats.increment("total_inlined", self._inline_count)

        return PassResult.CHANGED if changed else PassResult.UNCHANGED

    def _collect_call_sites(self, module: Module,
                             func_map: Dict[str, Function]) -> Dict[str, List[Tuple[CallInst, BasicBlock, str]]]:
        """Collect all direct call sites grouped by caller."""
        sites: Dict[str, List[Tuple[CallInst, BasicBlock, str]]] = defaultdict(list)

        for func in module.functions:
            for block in func.blocks:
                for inst in block.instructions:
                    if isinstance(inst, CallInst):
                        callee_name = self._get_callee_name(inst)
                        if callee_name and callee_name in func_map:
                            sites[func.name].append((inst, block, callee_name))

        return sites

    def _get_callee_name(self, call: CallInst) -> Optional[str]:
        """Extract the callee function name from a call instruction."""
        if hasattr(call, 'callee_name'):
            return call.callee_name
        if call.callee is not None and hasattr(call.callee, 'name'):
            return call.callee.name
        return None

    def _compute_bottom_up_order(self, func_map: Dict[str, Function]) -> List[str]:
        """Compute bottom-up processing order (callees before callers)."""
        # Build call graph
        calls: Dict[str, Set[str]] = defaultdict(set)
        for name, func in func_map.items():
            for block in func.blocks:
                for inst in block.instructions:
                    if isinstance(inst, CallInst):
                        callee = self._get_callee_name(inst)
                        if callee and callee in func_map and callee != name:
                            calls[name].add(callee)

        # Topological sort (reverse = bottom-up)
        visited: Set[str] = set()
        order: List[str] = []

        def visit(name: str) -> None:
            if name in visited:
                return
            visited.add(name)
            for callee in calls.get(name, set()):
                visit(callee)
            order.append(name)

        for name in func_map:
            visit(name)

        return order  # Already bottom-up due to post-order

    def _perform_inline(self, call: CallInst, call_block: BasicBlock,
                         callee: Function, caller: Function) -> bool:
        """Perform the actual inlining transformation."""
        try:
            # Split call_block at the call site
            before_block, after_block = self._split_at_call(call, call_block, caller)

            # Clone callee body
            cloner = FunctionCloner()
            entry, merge, ret_val = cloner.clone_into(callee, call, caller, before_block)

            # Wire up: before_block → inlined entry
            br_to_entry = BranchInst(target=entry)
            br_to_entry._parent = before_block
            before_block.append(br_to_entry)
            entry.add_predecessor(before_block)

            # Wire up: merge → after_block
            br_to_after = BranchInst(target=after_block)
            br_to_after._parent = merge
            merge.append(br_to_after)
            after_block.add_predecessor(merge)

            # Replace uses of call result with return value
            if ret_val is not None and hasattr(call, 'users'):
                for user in list(call.users):
                    if isinstance(user, Instruction):
                        self._substitute(user, call, ret_val)

            # Fix up phi nodes in after_block
            for phi in after_block.phi_nodes:
                new_incoming = []
                for val, block in phi.incoming:
                    if block is call_block:
                        new_incoming.append((val, merge))
                    else:
                        new_incoming.append((val, block))
                phi.incoming = new_incoming

            return True

        except Exception as e:
            logger.error(f"Failed to inline {callee.name}: {e}")
            return False

    def _split_at_call(self, call: CallInst, block: BasicBlock,
                        caller: Function) -> Tuple[BasicBlock, BasicBlock]:
        """Split a block at a call instruction into before and after blocks."""
        after_block = BasicBlock(name=f"{block.name}.after_inline")
        after_block._parent = caller

        # Move instructions after the call to the new block
        found_call = False
        to_move: List[Instruction] = []
        for inst in list(block.instructions):
            if found_call:
                to_move.append(inst)
            elif inst is call:
                found_call = True
                to_move.append(inst)  # Include the call itself for removal

        # Remove moved instructions from original block
        for inst in to_move:
            block.remove(inst)

        # Add non-call instructions to after_block
        for inst in to_move:
            if inst is not call:
                inst._parent = after_block
                after_block.append(inst)

        # Transfer successors
        for succ in list(block.successors):
            after_block._successors = block._successors[:]
            for phi in succ.phi_nodes:
                phi.incoming = [
                    (v, after_block if b is block else b)
                    for v, b in phi.incoming
                ]

        block._successors = []
        block._clear_successor_edges()

        caller.add_block(after_block)
        return block, after_block

    def _substitute(self, user: Instruction, old: Value, new: Value) -> None:
        if isinstance(user, BinaryOp):
            if user.left is old:
                user.left = new
            if user.right is old:
                user.right = new
        elif isinstance(user, PhiInst):
            user.incoming = [(new if v is old else v, b) for v, b in user.incoming]
        elif isinstance(user, ReturnInst):
            if user.value is old:
                user.value = new
        elif isinstance(user, StoreInst):
            if user.value is old:
                user.value = new
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
        elif isinstance(user, BranchInst) and user.is_conditional:
            if user.condition is old:
                user.condition = new
        elif isinstance(user, CompareOp):
            if user.left is old:
                user.left = new
            if user.right is old:
                user.right = new
        elif isinstance(user, CallInst):
            user.arguments = [new if a is old else a for a in user.arguments]
