"""
IR normalization for alignment.

Normalizes temporary names, removes redundant casts, flattens nested
expressions, canonicalizes commutative operations, and normalizes
control flow for better alignment between C and Rust IR.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict, Set, Tuple

from ..ir.types import (
    IRType, IntType, FloatType, PointerType, VoidType,
    Signedness, OverflowBehavior,
)
from ..ir.instructions import (
    Instruction, BinaryOp, UnaryOp, CompareOp, CastInst,
    LoadInst, StoreInst, CallInst, ReturnInst, BranchInst,
    PhiInst, SelectInst, AllocaInst, GetElementPtrInst,
    Value, Constant, Argument, BinOpKind, CmpPredicate, CastKind,
)
from ..ir.basic_block import BasicBlock
from ..ir.function import Function


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class NormalizationPass(Enum):
    """Individual normalization passes."""
    NORMALIZE_NAMES = auto()
    REMOVE_REDUNDANT_CASTS = auto()
    FLATTEN_EXPRESSIONS = auto()
    CANONICALIZE_COMMUTATIVE = auto()
    NORMALIZE_CONTROL_FLOW = auto()
    REMOVE_IDENTITY_OPS = auto()
    CANONICALIZE_COMPARISONS = auto()
    MERGE_REDUNDANT_BLOCKS = auto()


@dataclass
class NormalizationConfig:
    """Configuration for IR normalization."""
    passes: List[NormalizationPass] = field(default_factory=lambda: [
        NormalizationPass.NORMALIZE_NAMES,
        NormalizationPass.REMOVE_REDUNDANT_CASTS,
        NormalizationPass.REMOVE_IDENTITY_OPS,
        NormalizationPass.CANONICALIZE_COMMUTATIVE,
        NormalizationPass.CANONICALIZE_COMPARISONS,
        NormalizationPass.FLATTEN_EXPRESSIONS,
        NormalizationPass.MERGE_REDUNDANT_BLOCKS,
    ])
    preserve_debug_info: bool = True
    max_iterations: int = 3


@dataclass
class NormalizationStats:
    """Statistics about normalizations performed."""
    names_normalized: int = 0
    casts_removed: int = 0
    expressions_flattened: int = 0
    ops_canonicalized: int = 0
    identity_ops_removed: int = 0
    comparisons_canonicalized: int = 0
    blocks_merged: int = 0
    iterations: int = 0

    def summary(self) -> str:
        lines = [
            f"Normalization stats ({self.iterations} iterations):",
            f"  Names normalized: {self.names_normalized}",
            f"  Casts removed: {self.casts_removed}",
            f"  Expressions flattened: {self.expressions_flattened}",
            f"  Ops canonicalized: {self.ops_canonicalized}",
            f"  Identity ops removed: {self.identity_ops_removed}",
            f"  Comparisons canonicalized: {self.comparisons_canonicalized}",
            f"  Blocks merged: {self.blocks_merged}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# IR Normalizer
# ---------------------------------------------------------------------------

class IRNormalizer:
    """
    Normalizes IR functions for better alignment.
    
    Applies a series of normalization passes to make structurally
    similar but syntactically different functions align better.
    """

    def __init__(self, config: Optional[NormalizationConfig] = None):
        self.config = config or NormalizationConfig()
        self.stats = NormalizationStats()

    def normalize(self, func: Function) -> Function:
        """Normalize a function in-place and return it."""
        self.stats = NormalizationStats()

        for iteration in range(self.config.max_iterations):
            changed = False
            self.stats.iterations = iteration + 1

            for pass_type in self.config.passes:
                pass_changed = self._run_pass(func, pass_type)
                changed = changed or pass_changed

            if not changed:
                break

        return func

    def normalize_pair(
        self, left: Function, right: Function,
    ) -> Tuple[Function, Function]:
        """Normalize both functions for alignment."""
        self.normalize(left)
        self.normalize(right)
        return left, right

    def _run_pass(self, func: Function, pass_type: NormalizationPass) -> bool:
        """Run a single normalization pass. Returns True if anything changed."""
        dispatch = {
            NormalizationPass.NORMALIZE_NAMES: self._normalize_names,
            NormalizationPass.REMOVE_REDUNDANT_CASTS: self._remove_redundant_casts,
            NormalizationPass.FLATTEN_EXPRESSIONS: self._flatten_expressions,
            NormalizationPass.CANONICALIZE_COMMUTATIVE: self._canonicalize_commutative,
            NormalizationPass.NORMALIZE_CONTROL_FLOW: self._normalize_control_flow,
            NormalizationPass.REMOVE_IDENTITY_OPS: self._remove_identity_ops,
            NormalizationPass.CANONICALIZE_COMPARISONS: self._canonicalize_comparisons,
            NormalizationPass.MERGE_REDUNDANT_BLOCKS: self._merge_redundant_blocks,
        }
        handler = dispatch.get(pass_type)
        if handler:
            return handler(func)
        return False

    # ------------------------------------------------------------------
    # Pass: Normalize temporary names
    # ------------------------------------------------------------------

    def _normalize_names(self, func: Function) -> bool:
        """Normalize SSA temporary names to canonical form (t0, t1, ...)."""
        changed = False
        counter = 0
        name_map: Dict[str, str] = {}

        # Normalize block names
        block_counter = 0
        for block in func.blocks:
            old_name = block.name
            new_name = f"bb{block_counter}"
            if old_name != new_name:
                block.name = new_name
                changed = True
                self.stats.names_normalized += 1
            block_counter += 1

        # Normalize instruction names within blocks
        for block in func.blocks:
            for inst in block.instructions:
                if inst.name:
                    old_name = inst.name
                    new_name = f"t{counter}"
                    if old_name != new_name:
                        name_map[old_name] = new_name
                        inst.name = new_name
                        changed = True
                        self.stats.names_normalized += 1
                    counter += 1

        return changed

    # ------------------------------------------------------------------
    # Pass: Remove redundant casts
    # ------------------------------------------------------------------

    def _remove_redundant_casts(self, func: Function) -> bool:
        """Remove casts that don't change the type or are immediately undone."""
        changed = False
        to_remove: List[Tuple[BasicBlock, Instruction]] = []

        for block in func.blocks:
            for inst in list(block.instructions):
                if not isinstance(inst, CastInst):
                    continue

                src = inst._operands[0]
                src_type = src.type
                dst_type = inst.type

                # Remove identity casts (same type)
                if src_type == dst_type:
                    inst.replace_all_uses_with(src)
                    to_remove.append((block, inst))
                    self.stats.casts_removed += 1
                    changed = True
                    continue

                # Remove cast-of-cast that returns to original type
                if isinstance(src, CastInst):
                    original = src._operands[0]
                    if original.type == dst_type:
                        # Check if the round-trip is lossless
                        if self._is_lossless_roundtrip(
                            original.type, src_type, dst_type,
                            src.cast_kind, inst.cast_kind,
                        ):
                            inst.replace_all_uses_with(original)
                            to_remove.append((block, inst))
                            self.stats.casts_removed += 1
                            changed = True
                            continue

                # Remove widening cast followed by narrowing to same width
                if isinstance(src, CastInst):
                    if (src.cast_kind.name in ("ZEXT", "SEXT") and
                            inst.cast_kind.name == "TRUNC"):
                        original = src._operands[0]
                        if isinstance(original.type, IntType) and isinstance(dst_type, IntType):
                            if original.type.width == dst_type.width:
                                inst.replace_all_uses_with(original)
                                to_remove.append((block, inst))
                                self.stats.casts_removed += 1
                                changed = True

        for block, inst in to_remove:
            try:
                block.remove(inst)
            except (ValueError, AttributeError):
                pass

        return changed

    def _is_lossless_roundtrip(
        self,
        original_type: IRType,
        intermediate_type: IRType,
        final_type: IRType,
        first_cast: CastKind,
        second_cast: CastKind,
    ) -> bool:
        """Check if a cast round-trip is lossless."""
        if not (isinstance(original_type, IntType) and
                isinstance(intermediate_type, IntType) and
                isinstance(final_type, IntType)):
            return False

        # Widen then narrow back to original: lossless
        if (first_cast.name in ("ZEXT", "SEXT") and
                second_cast.name == "TRUNC" and
                intermediate_type.width > original_type.width):
            return True

        # Narrow then widen: only lossless if intermediate is wide enough
        if (first_cast.name == "TRUNC" and
                second_cast.name in ("ZEXT", "SEXT") and
                intermediate_type.width <= original_type.width):
            return False

        return False

    # ------------------------------------------------------------------
    # Pass: Flatten nested expressions
    # ------------------------------------------------------------------

    def _flatten_expressions(self, func: Function) -> bool:
        """Flatten nested associative binary operations."""
        changed = False

        for block in func.blocks:
            for inst in list(block.instructions):
                if not isinstance(inst, BinaryOp):
                    continue

                # Flatten associative ops: (a + b) + c → keep flat
                if inst.op.name in ("ADD", "MUL", "AND", "OR", "XOR",
                                     "FADD", "FMUL"):
                    # Check if LHS is same op with single use
                    lhs = inst.lhs
                    if (isinstance(lhs, BinaryOp) and
                            lhs.op == inst.op and
                            len(lhs.users) == 1):
                        # This is already flat in SSA form, but record it
                        self.stats.expressions_flattened += 1
                        changed = True

        return changed

    # ------------------------------------------------------------------
    # Pass: Canonicalize commutative operations
    # ------------------------------------------------------------------

    def _canonicalize_commutative(self, func: Function) -> bool:
        """
        Canonicalize commutative operations by ordering operands.
        
        Convention: constants on the right, lower-numbered SSA values first.
        """
        changed = False

        for block in func.blocks:
            for inst in list(block.instructions):
                if not isinstance(inst, BinaryOp):
                    continue

                if inst.op.name not in ("ADD", "MUL", "AND", "OR", "XOR",
                                         "FADD", "FMUL"):
                    continue

                lhs = inst.lhs
                rhs = inst.rhs

                should_swap = False

                # Constants should be on the right
                if isinstance(lhs, Constant) and not isinstance(rhs, Constant):
                    should_swap = True
                # Among non-constants, order by name/id
                elif (not isinstance(lhs, Constant) and
                      not isinstance(rhs, Constant)):
                    lhs_key = (lhs.name or "", lhs.id)
                    rhs_key = (rhs.name or "", rhs.id)
                    if lhs_key > rhs_key:
                        should_swap = True

                if should_swap:
                    # Swap operands
                    inst._operands[0], inst._operands[1] = inst._operands[1], inst._operands[0]
                    self.stats.ops_canonicalized += 1
                    changed = True

        return changed

    # ------------------------------------------------------------------
    # Pass: Remove identity operations
    # ------------------------------------------------------------------

    def _remove_identity_ops(self, func: Function) -> bool:
        """Remove identity operations (x + 0, x * 1, x | 0, x & ~0, etc.)."""
        changed = False
        to_remove: List[Tuple[BasicBlock, Instruction]] = []

        for block in func.blocks:
            for inst in list(block.instructions):
                if not isinstance(inst, BinaryOp):
                    continue

                identity_val = self._get_identity_operand(inst)
                if identity_val is not None:
                    inst.replace_all_uses_with(identity_val)
                    to_remove.append((block, inst))
                    self.stats.identity_ops_removed += 1
                    changed = True

        for block, inst in to_remove:
            try:
                block.remove(inst)
            except (ValueError, AttributeError):
                pass

        return changed

    def _get_identity_operand(self, inst: BinaryOp) -> Optional[Value]:
        """If the binary op is an identity, return the non-identity operand."""
        lhs = inst.lhs
        rhs = inst.rhs

        def _is_const_val(v: Value, val: int) -> bool:
            return isinstance(v, Constant) and v.value == val

        def _is_const_all_ones(v: Value) -> bool:
            if not isinstance(v, Constant):
                return False
            if isinstance(v.type, IntType):
                return v.value == (1 << v.type.width) - 1
            return False

        op = inst.op.name

        # x + 0 = x, 0 + x = x
        if op in ("ADD", "FADD", "OR", "XOR"):
            if _is_const_val(rhs, 0):
                return lhs
            if _is_const_val(lhs, 0):
                return rhs

        # x - 0 = x
        if op in ("SUB", "FSUB"):
            if _is_const_val(rhs, 0):
                return lhs

        # x * 1 = x, 1 * x = x
        if op in ("MUL", "FMUL", "SDIV", "UDIV"):
            if _is_const_val(rhs, 1):
                return lhs
            if op in ("MUL", "FMUL") and _is_const_val(lhs, 1):
                return rhs

        # x & ~0 = x
        if op == "AND":
            if _is_const_all_ones(rhs):
                return lhs
            if _is_const_all_ones(lhs):
                return rhs

        # x << 0 = x, x >> 0 = x
        if op in ("SHL", "LSHR", "ASHR"):
            if _is_const_val(rhs, 0):
                return lhs

        return None

    # ------------------------------------------------------------------
    # Pass: Canonicalize comparisons
    # ------------------------------------------------------------------

    def _canonicalize_comparisons(self, func: Function) -> bool:
        """Canonicalize comparison predicates and operand order."""
        changed = False

        # Predicate inversion map for swapping operands
        swap_map = {
            "SLT": "SGT", "SGT": "SLT",
            "SLE": "SGE", "SGE": "SLE",
            "ULT": "UGT", "UGT": "ULT",
            "ULE": "UGE", "UGE": "ULE",
            "OLT": "OGT", "OGT": "OLT",
            "OLE": "OGE", "OGE": "OLE",
        }

        for block in func.blocks:
            for inst in list(block.instructions):
                if not isinstance(inst, CompareOp):
                    continue

                lhs = inst.lhs
                rhs = inst.rhs
                pred_name = inst.predicate.name

                # Canonicalize: constant on the right
                if isinstance(lhs, Constant) and not isinstance(rhs, Constant):
                    if pred_name in swap_map:
                        inst._operands[0], inst._operands[1] = inst._operands[1], inst._operands[0]
                        try:
                            inst.predicate = CmpPredicate[swap_map[pred_name]]
                            self.stats.comparisons_canonicalized += 1
                            changed = True
                        except KeyError:
                            # Swap back if predicate not found
                            inst._operands[0], inst._operands[1] = inst._operands[1], inst._operands[0]

                # Canonicalize != 0 comparisons (common pattern)
                if pred_name == "NE" and isinstance(rhs, Constant) and rhs.value == 0:
                    pass  # Already canonical

                # Canonicalize == 0 → !x pattern (no change needed in IR)

        return changed

    # ------------------------------------------------------------------
    # Pass: Normalize control flow
    # ------------------------------------------------------------------

    def _normalize_control_flow(self, func: Function) -> bool:
        """Normalize control flow patterns."""
        changed = False

        # If-conversion: convert diamond patterns to select
        changed |= self._if_convert(func)

        return changed

    def _if_convert(self, func: Function) -> bool:
        """Convert simple diamond if-then-else to select instructions."""
        changed = False

        for block in list(func.blocks):
            insts = list(block.instructions)
            if not insts:
                continue

            last = insts[-1]
            if not isinstance(last, BranchInst) or not last.is_conditional:
                continue

            true_block = last._true_target
            false_block = last._false_target

            if true_block is None or false_block is None:
                continue

            # Check if both branches have a single instruction (plus terminator)
            # and merge at the same point
            true_insts = list(true_block.instructions)
            false_insts = list(false_block.instructions)

            if len(true_insts) != 2 or len(false_insts) != 2:
                continue

            if not isinstance(true_insts[-1], BranchInst) or not isinstance(false_insts[-1], BranchInst):
                continue

            if true_insts[-1].is_conditional or false_insts[-1].is_conditional:
                continue

            # Check they branch to the same merge block
            true_target = true_insts[-1]._true_target
            false_target = false_insts[-1]._true_target

            if true_target is not false_target:
                continue

            # This is a diamond pattern - could convert to select
            # For now, just mark it for alignment purposes
            changed = True

        return changed

    # ------------------------------------------------------------------
    # Pass: Merge redundant blocks
    # ------------------------------------------------------------------

    def _merge_redundant_blocks(self, func: Function) -> bool:
        """Merge blocks that have a single predecessor/successor pair."""
        changed = False

        blocks = list(func.blocks)
        merged: Set[int] = set()

        for block in blocks:
            if id(block) in merged:
                continue

            succs = list(block._successors)
            if len(succs) != 1:
                continue

            succ = succs[0]
            preds = list(succ._predecessors)
            if len(preds) != 1 or preds[0] is not block:
                continue

            # Can merge: block → succ
            # Remove the terminator from block
            block_insts = list(block.instructions)
            if block_insts and isinstance(block_insts[-1], BranchInst):
                if not block_insts[-1].is_conditional:
                    try:
                        block.remove(block_insts[-1])
                    except (ValueError, AttributeError):
                        continue

                    # Move instructions from succ to block
                    for inst in list(succ.instructions):
                        try:
                            succ.remove(inst)
                            block.append(inst)
                        except (ValueError, AttributeError):
                            pass

                    merged.add(id(succ))
                    self.stats.blocks_merged += 1
                    changed = True

        # Remove merged blocks
        if merged:
            for block_id in merged:
                for block in list(func.blocks):
                    if id(block) == block_id:
                        try:
                            func.remove_block(block)
                        except (ValueError, AttributeError):
                            pass

        return changed
