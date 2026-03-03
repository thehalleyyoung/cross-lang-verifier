"""
Structural alignment of two IR functions for product program construction.

Implements greedy LCS-based block alignment, instruction-level alignment
within matched blocks, cost computation, and similarity metrics.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple, Dict, Sequence

from ..ir.types import IRType, IntType, FloatType, VoidType, PointerType, StructType, EnumType
from ..ir.instructions import (
    Instruction, BinaryOp, UnaryOp, CompareOp, LoadInst, StoreInst,
    CastInst, CallInst, ReturnInst, BranchInst, PhiInst, SelectInst,
    AllocaInst, GetElementPtrInst, ExtractValueInst, InsertValueInst,
    SwitchInst, Value,
)
from ..ir.basic_block import BasicBlock
from ..ir.function import Function


# ---------------------------------------------------------------------------
# Enums & lightweight types
# ---------------------------------------------------------------------------

class AlignmentKind(Enum):
    """How two entities were aligned."""
    MATCHED = auto()
    LEFT_ONLY = auto()
    RIGHT_ONLY = auto()
    REORDERED = auto()


class SimilarityMetric(Enum):
    STRUCTURAL = auto()
    TYPE_BASED = auto()
    OPCODE_BASED = auto()
    COMBINED = auto()


# ---------------------------------------------------------------------------
# Instruction-level alignment
# ---------------------------------------------------------------------------

@dataclass
class InstructionAlignment:
    """Alignment of a single instruction pair."""
    left: Optional[Instruction]
    right: Optional[Instruction]
    kind: AlignmentKind
    similarity: float = 0.0
    notes: List[str] = field(default_factory=list)

    @property
    def is_matched(self) -> bool:
        return self.kind == AlignmentKind.MATCHED

    @property
    def is_left_only(self) -> bool:
        return self.kind == AlignmentKind.LEFT_ONLY

    @property
    def is_right_only(self) -> bool:
        return self.kind == AlignmentKind.RIGHT_ONLY

    def __repr__(self) -> str:
        l_name = self.left.name if self.left else "---"
        r_name = self.right.name if self.right else "---"
        return f"InstructionAlignment({l_name} <-> {r_name}, {self.kind.name}, sim={self.similarity:.2f})"


# ---------------------------------------------------------------------------
# Block-level alignment
# ---------------------------------------------------------------------------

@dataclass
class BlockAlignment:
    """Alignment between two basic blocks."""
    left: Optional[BasicBlock]
    right: Optional[BasicBlock]
    kind: AlignmentKind
    instruction_alignments: List[InstructionAlignment] = field(default_factory=list)
    similarity: float = 0.0

    @property
    def is_matched(self) -> bool:
        return self.kind == AlignmentKind.MATCHED

    @property
    def matched_instructions(self) -> List[InstructionAlignment]:
        return [ia for ia in self.instruction_alignments if ia.is_matched]

    @property
    def left_name(self) -> str:
        return self.left.name if self.left else "---"

    @property
    def right_name(self) -> str:
        return self.right.name if self.right else "---"

    def __repr__(self) -> str:
        return (f"BlockAlignment({self.left_name} <-> {self.right_name}, "
                f"{self.kind.name}, sim={self.similarity:.2f}, "
                f"insts={len(self.instruction_alignments)})")


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------

@dataclass
class AlignmentCost:
    """Cost of an alignment, lower is better."""
    block_mismatches: int = 0
    instruction_mismatches: int = 0
    type_mismatches: int = 0
    opcode_mismatches: int = 0
    reorder_penalty: float = 0.0
    extra_temporaries: int = 0

    @property
    def total(self) -> float:
        return (
            self.block_mismatches * 10.0
            + self.instruction_mismatches * 3.0
            + self.type_mismatches * 2.0
            + self.opcode_mismatches * 5.0
            + self.reorder_penalty
            + self.extra_temporaries * 0.5
        )

    def __add__(self, other: AlignmentCost) -> AlignmentCost:
        return AlignmentCost(
            block_mismatches=self.block_mismatches + other.block_mismatches,
            instruction_mismatches=self.instruction_mismatches + other.instruction_mismatches,
            type_mismatches=self.type_mismatches + other.type_mismatches,
            opcode_mismatches=self.opcode_mismatches + other.opcode_mismatches,
            reorder_penalty=self.reorder_penalty + other.reorder_penalty,
            extra_temporaries=self.extra_temporaries + other.extra_temporaries,
        )

    def __repr__(self) -> str:
        return f"AlignmentCost(total={self.total:.1f})"


# ---------------------------------------------------------------------------
# Full alignment result
# ---------------------------------------------------------------------------

@dataclass
class AlignmentResult:
    """Complete alignment of two functions."""
    left_function: Function
    right_function: Function
    block_alignments: List[BlockAlignment] = field(default_factory=list)
    cost: AlignmentCost = field(default_factory=AlignmentCost)
    structural_similarity: float = 0.0

    @property
    def matched_blocks(self) -> List[BlockAlignment]:
        return [ba for ba in self.block_alignments if ba.is_matched]

    @property
    def left_only_blocks(self) -> List[BlockAlignment]:
        return [ba for ba in self.block_alignments if ba.kind == AlignmentKind.LEFT_ONLY]

    @property
    def right_only_blocks(self) -> List[BlockAlignment]:
        return [ba for ba in self.block_alignments if ba.kind == AlignmentKind.RIGHT_ONLY]

    @property
    def all_instruction_alignments(self) -> List[InstructionAlignment]:
        result: List[InstructionAlignment] = []
        for ba in self.block_alignments:
            result.extend(ba.instruction_alignments)
        return result

    def summary(self) -> str:
        lines = [
            f"Alignment: {self.left_function.name} <-> {self.right_function.name}",
            f"  Blocks: {len(self.matched_blocks)} matched, "
            f"{len(self.left_only_blocks)} left-only, "
            f"{len(self.right_only_blocks)} right-only",
            f"  Instructions: {len(self.all_instruction_alignments)} total alignments",
            f"  Structural similarity: {self.structural_similarity:.3f}",
            f"  Cost: {self.cost}",
        ]
        return "\n".join(lines)

    def visualize(self, max_width: int = 80) -> str:
        lines: List[str] = []
        half = max_width // 2 - 2
        sep = " | "
        header = f"{'LEFT':^{half}}{sep}{'RIGHT':^{half}}"
        lines.append("=" * max_width)
        lines.append(header)
        lines.append("=" * max_width)

        for ba in self.block_alignments:
            left_label = f"[{ba.left_name}]" if ba.left else "[---]"
            right_label = f"[{ba.right_name}]" if ba.right else "[---]"
            kind_tag = f" ({ba.kind.name})"
            lines.append(f"{left_label:<{half}}{sep}{right_label:<{half}}{kind_tag}")
            lines.append("-" * max_width)

            for ia in ba.instruction_alignments:
                l_str = _inst_summary(ia.left) if ia.left else "---"
                r_str = _inst_summary(ia.right) if ia.right else "---"
                sim_str = f" [{ia.similarity:.2f}]"
                l_str = l_str[:half - 1]
                r_str = r_str[:half - len(sim_str) - 1]
                lines.append(f"  {l_str:<{half - 2}}{sep}  {r_str:<{half - len(sim_str) - 2}}{sim_str}")

            lines.append("")

        lines.append("=" * max_width)
        lines.append(f"Overall similarity: {self.structural_similarity:.3f}  Cost: {self.cost.total:.1f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _inst_summary(inst: Instruction) -> str:
    """One-line summary of an instruction for visualization."""
    name = inst.name or "?"
    cls_name = type(inst).__name__
    if isinstance(inst, BinaryOp):
        return f"{name} = {inst.op.name} {inst.lhs.name}, {inst.rhs.name}"
    if isinstance(inst, CompareOp):
        return f"{name} = cmp {inst.predicate.name} {inst.lhs.name}, {inst.rhs.name}"
    if isinstance(inst, CastInst):
        return f"{name} = cast.{inst.cast_kind.name} {inst._operands[0].name}"
    if isinstance(inst, LoadInst):
        return f"{name} = load {inst.address.name}"
    if isinstance(inst, StoreInst):
        return f"store {inst.value.name} -> {inst.address.name}"
    if isinstance(inst, ReturnInst):
        rv = inst.return_value
        return f"ret {rv.name if rv else 'void'}"
    if isinstance(inst, BranchInst):
        if inst.is_conditional:
            return f"br {inst._operands[0].name}, {inst._true_target.name}, {inst._false_target.name}"
        return f"br {inst._true_target.name}"
    if isinstance(inst, SwitchInst):
        return f"switch {inst.condition.name}, {len(inst.cases)} cases"
    if isinstance(inst, PhiInst):
        srcs = ", ".join(f"{v.name}:{bb.name}" for v, bb in inst.incoming)
        return f"{name} = phi [{srcs}]"
    if isinstance(inst, CallInst):
        return f"{name} = call {inst.callee_name}(...)"
    if isinstance(inst, ExtractValueInst):
        return f"{name} = extractvalue {inst.aggregate.name}, {inst.indices}"
    if isinstance(inst, InsertValueInst):
        return f"{name} = insertvalue {inst.aggregate.name}, {inst.inserted_value.name}, {inst.indices}"
    return f"{name} = {cls_name}"


def _block_signature(block: BasicBlock) -> Tuple:
    """Compute a signature for a basic block based on instruction types and branching."""
    inst_types: List[str] = []
    for inst in block.instructions:
        tag = type(inst).__name__
        if isinstance(inst, BinaryOp):
            tag += f".{inst.op.name}"
        elif isinstance(inst, CompareOp):
            tag += f".{inst.predicate.name}"
        elif isinstance(inst, CastInst):
            tag += f".{inst.cast_kind.name}"
        inst_types.append(tag)

    branch_pattern = "none"
    if block.instructions:
        last = block.instructions[-1]
        if isinstance(last, BranchInst):
            branch_pattern = "cond" if last.is_conditional else "uncond"
        elif isinstance(last, ReturnInst):
            branch_pattern = "ret"

    num_preds = len(list(block._predecessors)) if hasattr(block, '_predecessors') else 0
    num_succs = len(list(block._successors)) if hasattr(block, '_successors') else 0

    return (tuple(inst_types), branch_pattern, num_preds, num_succs)


def _instruction_opcode_tag(inst: Instruction) -> str:
    """Canonical opcode tag for an instruction."""
    if isinstance(inst, BinaryOp):
        return f"binop.{inst.op.name}"
    if isinstance(inst, UnaryOp):
        return f"unaryop.{inst.op.name}"
    if isinstance(inst, CompareOp):
        return f"cmp.{inst.predicate.name}"
    if isinstance(inst, CastInst):
        return f"cast.{inst.cast_kind.name}"
    if isinstance(inst, LoadInst):
        return "load"
    if isinstance(inst, StoreInst):
        return "store"
    if isinstance(inst, AllocaInst):
        return "alloca"
    if isinstance(inst, GetElementPtrInst):
        return "gep"
    if isinstance(inst, ExtractValueInst):
        return "extractvalue"
    if isinstance(inst, InsertValueInst):
        return "insertvalue"
    if isinstance(inst, CallInst):
        return f"call.{inst.callee_name}"
    if isinstance(inst, ReturnInst):
        return "ret"
    if isinstance(inst, BranchInst):
        return "br.cond" if inst.is_conditional else "br"
    if isinstance(inst, SwitchInst):
        return "switch"
    if isinstance(inst, PhiInst):
        return "phi"
    if isinstance(inst, SelectInst):
        return "select"
    return type(inst).__name__.lower()


def _type_similarity(t1: Optional[IRType], t2: Optional[IRType]) -> float:
    """Similarity score between two IR types, 0.0-1.0."""
    if t1 is None or t2 is None:
        return 0.0 if (t1 is None) != (t2 is None) else 1.0
    if type(t1) is type(t2):
        if isinstance(t1, IntType) and isinstance(t2, IntType):
            width_match = 1.0 if t1.width == t2.width else 0.5
            sign_match = 1.0 if t1.is_signed == t2.is_signed else 0.7
            return width_match * sign_match
        if isinstance(t1, FloatType) and isinstance(t2, FloatType):
            return 1.0 if t1.kind == t2.kind else 0.7
        if isinstance(t1, PointerType) and isinstance(t2, PointerType):
            return 0.8 + 0.2 * _type_similarity(t1.pointee, t2.pointee)
        if isinstance(t1, StructType) and isinstance(t2, StructType):
            if t1.num_fields == t2.num_fields and t1.num_fields > 0:
                field_sims = [
                    _type_similarity(f1.type, f2.type)
                    for f1, f2 in zip(t1.fields, t2.fields)
                ]
                return 0.7 + 0.3 * (sum(field_sims) / len(field_sims))
            return 0.5
        if isinstance(t1, EnumType) and isinstance(t2, EnumType):
            if t1.num_variants == t2.num_variants:
                return 0.8
            return 0.5
        if isinstance(t1, VoidType):
            return 1.0
        return 0.8
    # Cross-type similarities for common C-to-Rust patterns
    if isinstance(t1, IntType) and isinstance(t2, IntType):
        return 0.5
    # switch (int) vs match (enum) — partial similarity
    if (isinstance(t1, IntType) and isinstance(t2, EnumType)) or \
       (isinstance(t1, EnumType) and isinstance(t2, IntType)):
        return 0.4
    # C struct vs Rust struct — high similarity if same category
    if (isinstance(t1, StructType) and isinstance(t2, StructType)):
        return 0.6
    return 0.0


def _operand_similarity(left: Instruction, right: Instruction) -> float:
    """Compute similarity between operands of two instructions using type matching."""
    left_ops = left._operands
    right_ops = right._operands
    if not left_ops and not right_ops:
        return 1.0
    if not left_ops or not right_ops:
        return 0.0

    # Count ratio for basic similarity
    max_ops = max(len(left_ops), len(right_ops))
    min_ops = min(len(left_ops), len(right_ops))
    count_sim = min_ops / max_ops

    # Pairwise type similarity for matched operands
    type_sims = []
    for i in range(min_ops):
        lo = left_ops[i]
        ro = right_ops[i]
        lo_type = lo.type if hasattr(lo, 'type') else None
        ro_type = ro.type if hasattr(ro, 'type') else None
        type_sims.append(_type_similarity(lo_type, ro_type))

    avg_type_sim = sum(type_sims) / len(type_sims) if type_sims else 0.0

    # Bonus: check if both operands are constants with same value
    const_bonus = 0.0
    for i in range(min_ops):
        lo = left_ops[i]
        ro = right_ops[i]
        if (hasattr(lo, 'value') and hasattr(ro, 'value') and
                isinstance(lo, Value) and isinstance(ro, Value)):
            try:
                if lo.value == ro.value:
                    const_bonus += 0.1
            except (TypeError, AttributeError):
                pass

    return min(0.4 * count_sim + 0.4 * avg_type_sim + 0.2 * min(const_bonus, 1.0), 1.0)


def _instruction_similarity(left: Instruction, right: Instruction) -> float:
    """Compute similarity between two instructions, 0.0-1.0."""
    tag_l = _instruction_opcode_tag(left)
    tag_r = _instruction_opcode_tag(right)

    if tag_l == tag_r:
        opcode_sim = 1.0
    elif tag_l.split(".")[0] == tag_r.split(".")[0]:
        opcode_sim = 0.6
    else:
        # Cross-language structural translation patterns
        opcode_sim = _structural_pattern_similarity(tag_l, tag_r, left, right)

    type_sim = _type_similarity(left.type, right.type)

    operand_sim = _operand_similarity(left, right)

    base = 0.5 * opcode_sim + 0.3 * type_sim + 0.2 * operand_sim

    # Boost when types are compatible but opcodes differ (common in C↔Rust)
    if opcode_sim < 0.5 and type_sim > 0.7:
        base = max(base, 0.15 + 0.3 * type_sim)

    return min(base, 1.0)


# ---------------------------------------------------------------------------
# Structural translation pattern matching
# ---------------------------------------------------------------------------

# Common cross-language patterns: C switch ↔ Rust match, C error codes ↔
# Rust Result, C for-loop ↔ Rust iterator, etc.
_STRUCTURAL_EQUIVALENCES = {
    # switch ↔ extractvalue (match discriminant extraction)
    ("switch", "extractvalue"): 0.5,
    ("extractvalue", "switch"): 0.5,
    # switch ↔ branch (match arm → conditional branch)
    ("switch", "br.cond"): 0.4,
    ("br.cond", "switch"): 0.4,
    # C error return ↔ Rust Result construction
    ("ret", "insertvalue"): 0.3,
    ("insertvalue", "ret"): 0.3,
    # GEP ↔ extractvalue (struct field access patterns)
    ("gep", "extractvalue"): 0.5,
    ("extractvalue", "gep"): 0.5,
    # Load from struct ↔ extractvalue
    ("load", "extractvalue"): 0.4,
    ("extractvalue", "load"): 0.4,
    # Store to struct ↔ insertvalue
    ("store", "insertvalue"): 0.4,
    ("insertvalue", "store"): 0.4,
    # C comparison+branch ↔ Rust match
    ("cmp", "extractvalue"): 0.3,
    ("extractvalue", "cmp"): 0.3,
    # Cast ↔ cast (C implicit vs Rust explicit `as`)
    ("cast", "cast"): 0.7,
    # GEP ↔ GEP (index operations)
    ("gep", "gep"): 0.8,
    # GEP ↔ load (index operation mapping to element access)
    ("gep", "load"): 0.4,
    ("load", "gep"): 0.4,
    # ExtractValue/InsertValue ↔ field access patterns
    ("extractvalue", "extractvalue"): 0.8,
    ("insertvalue", "insertvalue"): 0.8,
    ("extractvalue", "insertvalue"): 0.3,
    ("insertvalue", "extractvalue"): 0.3,
    # Call ↔ binop (wrapping_add/sub/mul → +/-/*)
    ("call", "binop"): 0.4,
    ("binop", "call"): 0.4,
    # Select ↔ br.cond (conditional patterns)
    ("select", "br.cond"): 0.35,
    ("br.cond", "select"): 0.35,
    # Phi ↔ select (phi node vs select)
    ("phi", "select"): 0.45,
    ("select", "phi"): 0.45,
}

# Rust wrapping/checked/saturating method names → corresponding BinOpKind names
_WRAPPING_METHOD_TO_OP = {
    "wrapping_add": "ADD", "wrapping_sub": "SUB", "wrapping_mul": "MUL",
    "wrapping_shl": "SHL", "wrapping_shr": "LSHR",
    "checked_add": "ADD", "checked_sub": "SUB", "checked_mul": "MUL",
    "saturating_add": "ADD", "saturating_sub": "SUB", "saturating_mul": "MUL",
    "overflowing_add": "ADD", "overflowing_sub": "SUB", "overflowing_mul": "MUL",
}


def _structural_pattern_similarity(
    tag_l: str, tag_r: str,
    left: Instruction, right: Instruction,
) -> float:
    """Score similarity for cross-language structural translations."""
    base_l = tag_l.split(".")[0]
    base_r = tag_r.split(".")[0]
    key = (base_l, base_r)

    # Direct table lookup
    if key in _STRUCTURAL_EQUIVALENCES:
        base_score = _STRUCTURAL_EQUIVALENCES[key]
    else:
        base_score = 0.0

    # Cast instructions: C implicit casts vs Rust explicit `as`
    if isinstance(left, CastInst) and isinstance(right, CastInst):
        if left.cast_kind == right.cast_kind:
            base_score = max(base_score, 0.8)
        else:
            base_score = max(base_score, 0.5)

    # Method calls mapping to operators (e.g. .wrapping_add() → +)
    if isinstance(left, CallInst) and isinstance(right, BinaryOp):
        callee = left.callee_name
        for method, op_name in _WRAPPING_METHOD_TO_OP.items():
            if method in callee and right.op.name == op_name:
                base_score = max(base_score, 0.7)
                break
    if isinstance(left, BinaryOp) and isinstance(right, CallInst):
        callee = right.callee_name
        for method, op_name in _WRAPPING_METHOD_TO_OP.items():
            if method in callee and left.op.name == op_name:
                base_score = max(base_score, 0.7)
                break

    # GEP ↔ index operations
    if isinstance(left, GetElementPtrInst) and isinstance(right, GetElementPtrInst):
        base_score = max(base_score, 0.8)
    if (isinstance(left, GetElementPtrInst) and isinstance(right, (LoadInst, ExtractValueInst))) or \
       (isinstance(left, (LoadInst, ExtractValueInst)) and isinstance(right, GetElementPtrInst)):
        base_score = max(base_score, 0.45)

    # ExtractValue/InsertValue ↔ field access patterns
    if isinstance(left, ExtractValueInst) and isinstance(right, ExtractValueInst):
        if left.indices == right.indices:
            base_score = max(base_score, 0.9)
        else:
            base_score = max(base_score, 0.6)
    if isinstance(left, InsertValueInst) and isinstance(right, InsertValueInst):
        if left.indices == right.indices:
            base_score = max(base_score, 0.9)
        else:
            base_score = max(base_score, 0.6)

    # Compatible result types boost even for different opcodes
    if base_score > 0.0:
        ts = _type_similarity(left.type, right.type)
        if ts > 0.7:
            base_score = min(base_score + 0.1, 1.0)

    return base_score


def _block_similarity(left: BasicBlock, right: BasicBlock) -> float:
    """Compute structural similarity between two basic blocks."""
    sig_l = _block_signature(left)
    sig_r = _block_signature(right)

    # Compare branching patterns
    branch_sim = 1.0 if sig_l[1] == sig_r[1] else 0.0

    # Compare instruction type sequences via LCS ratio
    inst_types_l = sig_l[0]
    inst_types_r = sig_r[0]

    if not inst_types_l and not inst_types_r:
        inst_sim = 1.0
    elif not inst_types_l or not inst_types_r:
        inst_sim = 0.0
    else:
        lcs_len = _lcs_length(inst_types_l, inst_types_r)
        inst_sim = (2.0 * lcs_len) / (len(inst_types_l) + len(inst_types_r))

    # Connectivity similarity
    pred_diff = abs(sig_l[2] - sig_r[2])
    succ_diff = abs(sig_l[3] - sig_r[3])
    conn_sim = 1.0 / (1.0 + pred_diff + succ_diff)

    # Pairwise instruction similarity (graph-aware, not just sequential)
    left_insts = list(left.instructions)
    right_insts = list(right.instructions)
    pairwise_sim = 0.0
    if left_insts and right_insts:
        min_len = min(len(left_insts), len(right_insts))
        max_len = max(len(left_insts), len(right_insts))
        sim_sum = 0.0
        for i in range(min_len):
            sim_sum += _instruction_similarity(left_insts[i], right_insts[i])
        pairwise_sim = sim_sum / max_len if max_len > 0 else 0.0

    return 0.3 * inst_sim + 0.25 * branch_sim + 0.2 * conn_sim + 0.25 * pairwise_sim


def _lcs_length(a: Sequence, b: Sequence) -> int:
    """Compute length of longest common subsequence."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    # Optimised 1D DP
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def _lcs_indices(a: Sequence, b: Sequence) -> List[Tuple[int, int]]:
    """Compute LCS and return list of matched (i, j) index pairs."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return []

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Backtrack
    pairs: List[Tuple[int, int]] = []
    i, j = m, n
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1]:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


# ---------------------------------------------------------------------------
# Instruction alignment within a block pair
# ---------------------------------------------------------------------------

def _align_instructions(
    left_insts: List[Instruction],
    right_insts: List[Instruction],
) -> Tuple[List[InstructionAlignment], AlignmentCost]:
    """Align instructions within two matched blocks using similarity-weighted LCS."""
    m = len(left_insts)
    n = len(right_insts)
    cost = AlignmentCost()
    alignments: List[InstructionAlignment] = []

    if m == 0 and n == 0:
        return alignments, cost

    if m == 0:
        for inst in right_insts:
            alignments.append(InstructionAlignment(
                left=None, right=inst, kind=AlignmentKind.RIGHT_ONLY
            ))
            cost.instruction_mismatches += 1
        return alignments, cost

    if n == 0:
        for inst in left_insts:
            alignments.append(InstructionAlignment(
                left=inst, right=None, kind=AlignmentKind.LEFT_ONLY
            ))
            cost.instruction_mismatches += 1
        return alignments, cost

    # Build similarity matrix
    sim_matrix = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            sim_matrix[i][j] = _instruction_similarity(left_insts[i], right_insts[j])

    # DP for optimal alignment (like sequence alignment / Needleman-Wunsch)
    MATCH_THRESHOLD = 0.3
    GAP_PENALTY = -0.5

    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] + GAP_PENALTY
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j - 1] + GAP_PENALTY

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match_score = dp[i - 1][j - 1] + sim_matrix[i - 1][j - 1]
            gap_left = dp[i - 1][j] + GAP_PENALTY
            gap_right = dp[i][j - 1] + GAP_PENALTY
            dp[i][j] = max(match_score, gap_left, gap_right)

    # Traceback
    i, j = m, n
    raw_alignments: List[InstructionAlignment] = []
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            match_score = dp[i - 1][j - 1] + sim_matrix[i - 1][j - 1]
            if abs(dp[i][j] - match_score) < 1e-9:
                sim = sim_matrix[i - 1][j - 1]
                if sim >= MATCH_THRESHOLD:
                    notes: List[str] = []
                    left_inst = left_insts[i - 1]
                    right_inst = right_insts[j - 1]
                    if _instruction_opcode_tag(left_inst) != _instruction_opcode_tag(right_inst):
                        notes.append("opcode_mismatch")
                        cost.opcode_mismatches += 1
                    if left_inst.type != right_inst.type:
                        notes.append("type_mismatch")
                        cost.type_mismatches += 1
                    raw_alignments.append(InstructionAlignment(
                        left=left_inst,
                        right=right_inst,
                        kind=AlignmentKind.MATCHED,
                        similarity=sim,
                        notes=notes,
                    ))
                else:
                    raw_alignments.append(InstructionAlignment(
                        left=left_insts[i - 1], right=None,
                        kind=AlignmentKind.LEFT_ONLY,
                    ))
                    raw_alignments.append(InstructionAlignment(
                        left=None, right=right_insts[j - 1],
                        kind=AlignmentKind.RIGHT_ONLY,
                    ))
                    cost.instruction_mismatches += 2
                i -= 1
                j -= 1
                continue

        if i > 0 and (j == 0 or dp[i - 1][j] + GAP_PENALTY >= dp[i][j] - 1e-9):
            raw_alignments.append(InstructionAlignment(
                left=left_insts[i - 1], right=None,
                kind=AlignmentKind.LEFT_ONLY,
            ))
            cost.instruction_mismatches += 1
            i -= 1
        elif j > 0:
            raw_alignments.append(InstructionAlignment(
                left=None, right=right_insts[j - 1],
                kind=AlignmentKind.RIGHT_ONLY,
            ))
            cost.instruction_mismatches += 1
            j -= 1
        else:
            break

    raw_alignments.reverse()
    return raw_alignments, cost


# ---------------------------------------------------------------------------
# Greedy block alignment via signature LCS
# ---------------------------------------------------------------------------

def _greedy_block_alignment(
    left_blocks: List[BasicBlock],
    right_blocks: List[BasicBlock],
    similarity_threshold: float = 0.25,
) -> List[Tuple[Optional[int], Optional[int], float]]:
    """
    Block alignment using graph similarity for better matching.

    When blocks differ in count, uses bipartite best-match approach
    instead of purely sequential alignment. Returns list of
    (left_idx | None, right_idx | None, similarity).
    """
    m = len(left_blocks)
    n = len(right_blocks)

    if m == 0 and n == 0:
        return []
    if m == 0:
        return [(None, j, 0.0) for j in range(n)]
    if n == 0:
        return [(i, None, 0.0) for i in range(m)]

    # Compute pairwise block similarity
    sim = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            sim[i][j] = _block_similarity(left_blocks[i], right_blocks[j])

    # When block counts differ significantly, use bipartite greedy matching
    # to find best matches rather than forcing sequential alignment
    if abs(m - n) > max(m, n) * 0.5 and min(m, n) >= 1:
        return _bipartite_block_alignment(sim, m, n, similarity_threshold)

    # DP to find best-score alignment (like global sequence alignment)
    GAP = -0.2
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] + GAP
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j - 1] + GAP

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match = dp[i - 1][j - 1] + sim[i - 1][j - 1]
            gap_l = dp[i - 1][j] + GAP
            gap_r = dp[i][j - 1] + GAP
            dp[i][j] = max(match, gap_l, gap_r)

    # Traceback
    result: List[Tuple[Optional[int], Optional[int], float]] = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            match = dp[i - 1][j - 1] + sim[i - 1][j - 1]
            if abs(dp[i][j] - match) < 1e-9 and sim[i - 1][j - 1] >= similarity_threshold:
                result.append((i - 1, j - 1, sim[i - 1][j - 1]))
                i -= 1
                j -= 1
                continue
        if i > 0 and (j == 0 or dp[i - 1][j] >= dp[i][j - 1]):
            result.append((i - 1, None, 0.0))
            i -= 1
        elif j > 0:
            result.append((None, j - 1, 0.0))
            j -= 1
        else:
            break

    result.reverse()
    return result


def _bipartite_block_alignment(
    sim: List[List[float]],
    m: int,
    n: int,
    threshold: float,
) -> List[Tuple[Optional[int], Optional[int], float]]:
    """
    Greedy bipartite matching for blocks with significantly different counts.

    Greedily matches highest-similarity pairs first, then emits unmatched
    blocks as single-sided entries.
    """
    # Build candidate list sorted by similarity (descending)
    candidates: List[Tuple[float, int, int]] = []
    for i in range(m):
        for j in range(n):
            if sim[i][j] >= threshold:
                candidates.append((sim[i][j], i, j))
    candidates.sort(reverse=True)

    used_left: set = set()
    used_right: set = set()
    matched: List[Tuple[int, int, float]] = []

    for s, i, j in candidates:
        if i not in used_left and j not in used_right:
            matched.append((i, j, s))
            used_left.add(i)
            used_right.add(j)

    # Build result in block order (by min of left/right index)
    result: List[Tuple[Optional[int], Optional[int], float]] = []
    # Interleave matched and unmatched in order
    all_events: List[Tuple[float, Optional[int], Optional[int], float]] = []
    for li, ri, s in matched:
        all_events.append((li + ri * 0.001, li, ri, s))
    for i in range(m):
        if i not in used_left:
            all_events.append((i, i, None, 0.0))
    for j in range(n):
        if j not in used_right:
            all_events.append((j + 0.5, None, j, 0.0))
    all_events.sort()

    for _, li, ri, s in all_events:
        result.append((li, ri, s))

    return result


# ---------------------------------------------------------------------------
# Detect reordered blocks
# ---------------------------------------------------------------------------

def _detect_reordered_blocks(
    left_blocks: List[BasicBlock],
    right_blocks: List[BasicBlock],
    unmatched_left: List[int],
    unmatched_right: List[int],
    threshold: float = 0.5,
) -> List[Tuple[int, int, float]]:
    """Try to match remaining unmatched blocks that may be reordered."""
    if not unmatched_left or not unmatched_right:
        return []

    candidates: List[Tuple[float, int, int]] = []
    for li in unmatched_left:
        for ri in unmatched_right:
            sim = _block_similarity(left_blocks[li], right_blocks[ri])
            if sim >= threshold:
                candidates.append((sim, li, ri))

    candidates.sort(reverse=True)

    used_l: set = set()
    used_r: set = set()
    result: List[Tuple[int, int, float]] = []

    for sim, li, ri in candidates:
        if li not in used_l and ri not in used_r:
            result.append((li, ri, sim))
            used_l.add(li)
            used_r.add(ri)

    return result


# ---------------------------------------------------------------------------
# Block alignment helpers for structural differences
# ---------------------------------------------------------------------------

def _is_fallthrough_block(block: BasicBlock) -> bool:
    """Check if a block is a simple fallthrough (unconditional branch only or single-op + branch)."""
    insts = list(block.instructions)
    if not insts:
        return True
    if len(insts) == 1 and isinstance(insts[0], BranchInst) and not insts[0].is_conditional:
        return True
    if len(insts) == 2 and isinstance(insts[-1], BranchInst) and not insts[-1].is_conditional:
        return True
    return False


def _is_short_circuit_block(block: BasicBlock) -> bool:
    """Check if a block is part of a short-circuit evaluation expansion (C && / || patterns)."""
    insts = list(block.instructions)
    if not insts:
        return False
    last = insts[-1]
    if isinstance(last, BranchInst) and last.is_conditional:
        non_branch = [i for i in insts if not isinstance(i, BranchInst)]
        if len(non_branch) <= 1:
            return True
        if all(isinstance(i, (CompareOp, BranchInst, PhiInst)) for i in insts):
            return True
    return False


def _try_merge_fallthrough(
    alignment_triples: List[Tuple[Optional[int], Optional[int], float]],
    left_blocks: List[BasicBlock],
    right_blocks: List[BasicBlock],
) -> List[Tuple[Optional[int], Optional[int], float]]:
    """
    If one side has exactly one more block and it's a fallthrough, merge it
    into the adjacent matched block.
    """
    left_only = [(i, t) for i, t in enumerate(alignment_triples) if t[0] is not None and t[1] is None]
    right_only = [(i, t) for i, t in enumerate(alignment_triples) if t[0] is None and t[1] is not None]

    # Heuristic: if exactly one extra block on one side and it's a fallthrough, remove it
    if len(left_only) == 1 and len(right_only) == 0:
        idx, (li, _, _) = left_only[0]
        if _is_fallthrough_block(left_blocks[li]):
            result = list(alignment_triples)
            result.pop(idx)
            return result

    if len(right_only) == 1 and len(left_only) == 0:
        idx, (_, ri, _) = right_only[0]
        if _is_fallthrough_block(right_blocks[ri]):
            result = list(alignment_triples)
            result.pop(idx)
            return result

    return alignment_triples


# ---------------------------------------------------------------------------
# Extra temporary detection
# ---------------------------------------------------------------------------

def _count_extra_temporaries(block: BasicBlock) -> int:
    """Count instructions that are trivial temporaries (single-use casts, identity ops)."""
    count = 0
    for inst in block.instructions:
        if isinstance(inst, CastInst):
            if len(inst.users) <= 1:
                count += 1
        elif isinstance(inst, BinaryOp):
            # Identity operations like add 0, mul 1
            if isinstance(inst.rhs, Value) and hasattr(inst.rhs, 'value'):
                from ..ir.instructions import Constant
                if isinstance(inst.rhs, Constant):
                    if inst.op.name in ('ADD', 'SUB', 'OR', 'XOR') and inst.rhs.value == 0:
                        count += 1
                    elif inst.op.name in ('MUL', 'SDIV', 'UDIV') and inst.rhs.value == 1:
                        count += 1
    return count


# ---------------------------------------------------------------------------
# FunctionAligner — main entry point
# ---------------------------------------------------------------------------

class FunctionAligner:
    """Aligns two IR functions for product program construction."""

    def __init__(
        self,
        similarity_threshold: float = 0.25,
        reorder_threshold: float = 0.5,
        instruction_match_threshold: float = 0.3,
        block_similarity_threshold: float = 0.2,
    ):
        self.similarity_threshold = similarity_threshold
        self.reorder_threshold = reorder_threshold
        self.instruction_match_threshold = instruction_match_threshold
        self.block_similarity_threshold = block_similarity_threshold

    def align(self, left: Function, right: Function) -> AlignmentResult:
        """Align two functions and return a complete AlignmentResult."""
        result = AlignmentResult(
            left_function=left,
            right_function=right,
        )

        left_blocks = list(left.blocks)
        right_blocks = list(right.blocks)

        # Phase 1: Greedy LCS-based block alignment
        raw_alignment = _greedy_block_alignment(
            left_blocks, right_blocks, self.similarity_threshold
        )

        # Phase 1b: Merge fallthrough blocks when one side has one extra
        raw_alignment = _try_merge_fallthrough(raw_alignment, left_blocks, right_blocks)

        matched_left: set = set()
        matched_right: set = set()
        block_alignments: List[BlockAlignment] = []

        for li, ri, sim in raw_alignment:
            if li is not None and ri is not None:
                # Apply block similarity threshold: below threshold → single-sided
                if sim < self.block_similarity_threshold:
                    ba_left = BlockAlignment(
                        left=left_blocks[li],
                        right=None,
                        kind=AlignmentKind.LEFT_ONLY,
                        instruction_alignments=[
                            InstructionAlignment(left=inst, right=None, kind=AlignmentKind.LEFT_ONLY)
                            for inst in left_blocks[li].instructions
                        ],
                    )
                    block_alignments.append(ba_left)
                    result.cost.block_mismatches += 1
                    ba_right = BlockAlignment(
                        left=None,
                        right=right_blocks[ri],
                        kind=AlignmentKind.RIGHT_ONLY,
                        instruction_alignments=[
                            InstructionAlignment(left=None, right=inst, kind=AlignmentKind.RIGHT_ONLY)
                            for inst in right_blocks[ri].instructions
                        ],
                    )
                    block_alignments.append(ba_right)
                    result.cost.block_mismatches += 1
                    continue

                matched_left.add(li)
                matched_right.add(ri)

                inst_aligns, inst_cost = _align_instructions(
                    list(left_blocks[li].instructions),
                    list(right_blocks[ri].instructions),
                )

                ba = BlockAlignment(
                    left=left_blocks[li],
                    right=right_blocks[ri],
                    kind=AlignmentKind.MATCHED,
                    instruction_alignments=inst_aligns,
                    similarity=sim,
                )
                block_alignments.append(ba)
                result.cost = result.cost + inst_cost
            elif li is not None:
                ba = BlockAlignment(
                    left=left_blocks[li],
                    right=None,
                    kind=AlignmentKind.LEFT_ONLY,
                    instruction_alignments=[
                        InstructionAlignment(left=inst, right=None, kind=AlignmentKind.LEFT_ONLY)
                        for inst in left_blocks[li].instructions
                    ],
                )
                block_alignments.append(ba)
                result.cost.block_mismatches += 1
            else:
                assert ri is not None
                ba = BlockAlignment(
                    left=None,
                    right=right_blocks[ri],
                    kind=AlignmentKind.RIGHT_ONLY,
                    instruction_alignments=[
                        InstructionAlignment(left=None, right=inst, kind=AlignmentKind.RIGHT_ONLY)
                        for inst in right_blocks[ri].instructions
                    ],
                )
                block_alignments.append(ba)
                result.cost.block_mismatches += 1

        # Phase 2: Try to match reordered blocks among unmatched
        unmatched_left = [i for i in range(len(left_blocks)) if i not in matched_left]
        unmatched_right = [i for i in range(len(right_blocks)) if i not in matched_right]

        reordered = _detect_reordered_blocks(
            left_blocks, right_blocks, unmatched_left, unmatched_right,
            self.reorder_threshold,
        )

        for li, ri, sim in reordered:
            inst_aligns, inst_cost = _align_instructions(
                list(left_blocks[li].instructions),
                list(right_blocks[ri].instructions),
            )
            ba = BlockAlignment(
                left=left_blocks[li],
                right=right_blocks[ri],
                kind=AlignmentKind.REORDERED,
                instruction_alignments=inst_aligns,
                similarity=sim,
            )
            block_alignments.append(ba)
            result.cost = result.cost + inst_cost
            result.cost.reorder_penalty += 1.0
            unmatched_left.remove(li)
            unmatched_right.remove(ri)

        # Phase 3: Add remaining unmatched blocks
        for li in unmatched_left:
            ba = BlockAlignment(
                left=left_blocks[li],
                right=None,
                kind=AlignmentKind.LEFT_ONLY,
                instruction_alignments=[
                    InstructionAlignment(left=inst, right=None, kind=AlignmentKind.LEFT_ONLY)
                    for inst in left_blocks[li].instructions
                ],
            )
            block_alignments.append(ba)
            result.cost.block_mismatches += 1

        for ri in unmatched_right:
            ba = BlockAlignment(
                left=None,
                right=right_blocks[ri],
                kind=AlignmentKind.RIGHT_ONLY,
                instruction_alignments=[
                    InstructionAlignment(left=None, right=inst, kind=AlignmentKind.RIGHT_ONLY)
                    for inst in right_blocks[ri].instructions
                ],
            )
            block_alignments.append(ba)
            result.cost.block_mismatches += 1

        # Phase 4: Count extra temporaries
        for ba in block_alignments:
            if ba.left:
                result.cost.extra_temporaries += _count_extra_temporaries(ba.left)
            if ba.right:
                result.cost.extra_temporaries += _count_extra_temporaries(ba.right)

        result.block_alignments = block_alignments

        # Compute overall structural similarity
        result.structural_similarity = self._compute_structural_similarity(result)

        return result

    def _compute_structural_similarity(self, result: AlignmentResult) -> float:
        """Compute overall structural similarity score 0.0-1.0."""
        total_blocks = (
            len(list(result.left_function.blocks))
            + len(list(result.right_function.blocks))
        )
        if total_blocks == 0:
            return 1.0

        matched_weight = sum(ba.similarity for ba in result.matched_blocks)
        reordered_weight = sum(
            ba.similarity * 0.8
            for ba in result.block_alignments
            if ba.kind == AlignmentKind.REORDERED
        )

        block_sim = (2.0 * (matched_weight + reordered_weight)) / total_blocks

        total_insts = len(result.all_instruction_alignments)
        if total_insts == 0:
            inst_sim = 1.0
        else:
            matched_inst_sim = sum(
                ia.similarity for ia in result.all_instruction_alignments if ia.is_matched
            )
            inst_sim = matched_inst_sim / total_insts

        return 0.6 * min(block_sim, 1.0) + 0.4 * inst_sim

    def align_blocks_only(self, left: Function, right: Function) -> List[BlockAlignment]:
        """Quick block-level alignment without instruction alignment."""
        left_blocks = list(left.blocks)
        right_blocks = list(right.blocks)

        raw = _greedy_block_alignment(left_blocks, right_blocks, self.similarity_threshold)
        result: List[BlockAlignment] = []

        for li, ri, sim in raw:
            if li is not None and ri is not None:
                result.append(BlockAlignment(
                    left=left_blocks[li], right=right_blocks[ri],
                    kind=AlignmentKind.MATCHED, similarity=sim,
                ))
            elif li is not None:
                result.append(BlockAlignment(
                    left=left_blocks[li], right=None,
                    kind=AlignmentKind.LEFT_ONLY,
                ))
            else:
                assert ri is not None
                result.append(BlockAlignment(
                    left=None, right=right_blocks[ri],
                    kind=AlignmentKind.RIGHT_ONLY,
                ))

        return result

    def compute_similarity(self, left: Function, right: Function) -> float:
        """Quick structural similarity without full alignment."""
        result = self.align(left, right)
        return result.structural_similarity

    def compute_cost(self, left: Function, right: Function) -> AlignmentCost:
        """Compute alignment cost."""
        result = self.align(left, right)
        return result.cost
