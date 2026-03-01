"""
Automated product program construction and verification.

Takes two IR functions (C and Rust), structurally aligns them,
inserts coercion points at semantic divergence sites, generates
verification conditions, and feeds them to Z3 for real results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any

import z3

from ..ir.types import (
    IRType, IntType, FloatType, PointerType, VoidType,
    Signedness, FloatKind, OverflowBehavior,
)
from ..ir.instructions import (
    Instruction, BinaryOp, UnaryOp, CompareOp, CastInst,
    ReturnInst, BranchInst, PhiInst, SelectInst,
    LoadInst, StoreInst, CallInst,
    Value, Constant, Argument, BinOpKind,
    InstructionMetadata,
)
from ..ir.basic_block import BasicBlock
from ..ir.function import Function
from ..semantics.semantic_config import (
    SemanticConfig, OverflowMode, ShiftModel, DivisionModel,
)
from ..smt.auto_encoder import AutoSMTEncoder, EquivalenceResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Coercion point
# ---------------------------------------------------------------------------

class CoercionKind(Enum):
    OVERFLOW = auto()
    DIVISION = auto()
    SHIFT = auto()
    NEGATION = auto()
    FLOAT_TO_INT = auto()
    NULL_CHECK = auto()
    BOUNDS_CHECK = auto()
    TYPE_WIDTH = auto()
    SIGNEDNESS = auto()
    RETURN_COERCION = auto()


@dataclass
class AutoCoercionPoint:
    """A point where C and Rust semantics may diverge."""
    kind: CoercionKind
    c_instruction: Optional[Instruction]
    rust_instruction: Optional[Instruction]
    description: str
    z3_condition: Optional[z3.BoolRef] = None
    severity: str = "warning"  # "info", "warning", "error"


# ---------------------------------------------------------------------------
# Structural alignment
# ---------------------------------------------------------------------------

@dataclass
class InstructionPair:
    """A pair of aligned instructions from C and Rust."""
    c_inst: Optional[Instruction]
    rust_inst: Optional[Instruction]
    similarity: float = 0.0
    coercions: List[AutoCoercionPoint] = field(default_factory=list)


@dataclass
class BlockPair:
    """A pair of aligned basic blocks."""
    c_block: Optional[BasicBlock]
    rust_block: Optional[BasicBlock]
    instruction_pairs: List[InstructionPair] = field(default_factory=list)
    similarity: float = 0.0


@dataclass
class StructuralAlignment:
    """Complete structural alignment of two functions."""
    c_func: Function
    rust_func: Function
    block_pairs: List[BlockPair] = field(default_factory=list)
    overall_similarity: float = 0.0
    coercion_points: List[AutoCoercionPoint] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Automated product program builder
# ---------------------------------------------------------------------------

class AutoProductBuilder:
    """
    Builds and verifies product programs automatically.

    Steps:
    1. Structurally align C and Rust functions
    2. Identify semantic divergence sites
    3. Insert coercion points
    4. Generate Z3 verification conditions
    5. Check and report results
    """

    def __init__(self, c_config: Optional[SemanticConfig] = None,
                 rust_config: Optional[SemanticConfig] = None):
        self.c_config = c_config or SemanticConfig.c11()
        self.rust_config = rust_config or SemanticConfig.rust_release()
        self.encoder = AutoSMTEncoder(self.c_config, self.rust_config)

    def align_and_verify(
        self,
        c_func: Function,
        rust_func: Function,
        timeout_ms: int = 30000,
    ) -> Tuple[StructuralAlignment, EquivalenceResult]:
        """
        Align two functions, construct product program, and verify.

        Returns (alignment, equivalence_result).
        """
        # Step 1: Structural alignment
        alignment = self.align_functions(c_func, rust_func)

        # Step 2: Identify coercion points
        self._identify_coercions(alignment)

        # Step 3: Verify via Z3
        result = self.encoder.check_equivalence(c_func, rust_func, timeout_ms)

        return alignment, result

    def align_functions(self, c_func: Function, rust_func: Function) -> StructuralAlignment:
        """Structurally align two functions."""
        alignment = StructuralAlignment(c_func=c_func, rust_func=rust_func)

        c_blocks = list(c_func.blocks)
        r_blocks = list(rust_func.blocks)

        # Align blocks by position (simple strategy for straight-line code)
        n = max(len(c_blocks), len(r_blocks))
        for i in range(n):
            c_block = c_blocks[i] if i < len(c_blocks) else None
            r_block = r_blocks[i] if i < len(r_blocks) else None

            bp = BlockPair(c_block=c_block, rust_block=r_block)

            if c_block and r_block:
                bp.instruction_pairs = self._align_instructions(c_block, r_block)
                bp.similarity = self._block_similarity(c_block, r_block)

            alignment.block_pairs.append(bp)

        if alignment.block_pairs:
            alignment.overall_similarity = sum(
                bp.similarity for bp in alignment.block_pairs
            ) / len(alignment.block_pairs)

        return alignment

    def _align_instructions(
        self, c_block: BasicBlock, r_block: BasicBlock
    ) -> List[InstructionPair]:
        """Align instructions within two blocks using LCS-like approach."""
        c_insts = list(c_block.instructions)
        r_insts = list(r_block.instructions)
        pairs = []

        # Simple greedy alignment by opcode
        ci, ri = 0, 0
        while ci < len(c_insts) and ri < len(r_insts):
            c = c_insts[ci]
            r = r_insts[ri]

            sim = self._instruction_similarity(c, r)
            if sim > 0.3:
                pairs.append(InstructionPair(c_inst=c, rust_inst=r, similarity=sim))
                ci += 1
                ri += 1
            else:
                # Look ahead
                if ri + 1 < len(r_insts) and self._instruction_similarity(c, r_insts[ri + 1]) > 0.5:
                    pairs.append(InstructionPair(c_inst=None, rust_inst=r))
                    ri += 1
                elif ci + 1 < len(c_insts) and self._instruction_similarity(c_insts[ci + 1], r) > 0.5:
                    pairs.append(InstructionPair(c_inst=c, rust_inst=None))
                    ci += 1
                else:
                    pairs.append(InstructionPair(c_inst=c, rust_inst=r, similarity=sim))
                    ci += 1
                    ri += 1

        # Remaining
        while ci < len(c_insts):
            pairs.append(InstructionPair(c_inst=c_insts[ci], rust_inst=None))
            ci += 1
        while ri < len(r_insts):
            pairs.append(InstructionPair(c_inst=None, rust_inst=r_insts[ri]))
            ri += 1

        return pairs

    def _instruction_similarity(self, a: Instruction, b: Instruction) -> float:
        """Compute similarity between two instructions."""
        if type(a) != type(b):
            return 0.0
        score = 0.5  # Same type
        if isinstance(a, BinaryOp) and isinstance(b, BinaryOp):
            if a.op == b.op:
                score += 0.3
            if a.type and b.type and type(a.type) == type(b.type):
                score += 0.2
        elif isinstance(a, CompareOp) and isinstance(b, CompareOp):
            if a.predicate == b.predicate:
                score += 0.5
        elif isinstance(a, ReturnInst) and isinstance(b, ReturnInst):
            score += 0.5
        elif isinstance(a, BranchInst) and isinstance(b, BranchInst):
            score += 0.3
        else:
            score += 0.2
        return min(score, 1.0)

    def _block_similarity(self, c_block: BasicBlock, r_block: BasicBlock) -> float:
        c_n = len(list(c_block.instructions))
        r_n = len(list(r_block.instructions))
        if c_n == 0 and r_n == 0:
            return 1.0
        return 1.0 - abs(c_n - r_n) / max(c_n, r_n, 1)

    def _identify_coercions(self, alignment: StructuralAlignment) -> None:
        """Identify points where semantic coercions are needed."""
        for bp in alignment.block_pairs:
            for ip in bp.instruction_pairs:
                coercions = []
                if ip.c_inst and ip.rust_inst:
                    coercions = self._check_divergence_sites(ip.c_inst, ip.rust_inst)
                elif ip.c_inst:
                    coercions = [AutoCoercionPoint(
                        kind=CoercionKind.RETURN_COERCION,
                        c_instruction=ip.c_inst,
                        rust_instruction=None,
                        description=f"C-only instruction: {ip.c_inst.opcode_name()}",
                    )]
                elif ip.rust_inst:
                    coercions = [AutoCoercionPoint(
                        kind=CoercionKind.RETURN_COERCION,
                        c_instruction=None,
                        rust_instruction=ip.rust_inst,
                        description=f"Rust-only instruction: {ip.rust_inst.opcode_name()}",
                    )]
                ip.coercions = coercions
                alignment.coercion_points.extend(coercions)

    def _check_divergence_sites(
        self, c_inst: Instruction, r_inst: Instruction
    ) -> List[AutoCoercionPoint]:
        """Check for semantic divergence between aligned instructions."""
        coercions = []

        if isinstance(c_inst, BinaryOp) and isinstance(r_inst, BinaryOp):
            op = c_inst.op.name
            # Arithmetic overflow
            if op in ("ADD", "SUB", "MUL"):
                is_signed = isinstance(c_inst.type, IntType) and c_inst.type.is_signed
                if is_signed:
                    if (self.c_config.signed_overflow != self.rust_config.signed_overflow):
                        coercions.append(AutoCoercionPoint(
                            kind=CoercionKind.OVERFLOW,
                            c_instruction=c_inst,
                            rust_instruction=r_inst,
                            description=f"Signed overflow on {op}: C={self.c_config.signed_overflow}, "
                                       f"Rust={self.rust_config.signed_overflow}",
                            severity="error",
                        ))

            # Shift divergence
            if op in ("SHL", "LSHR", "ASHR"):
                if self.c_config.shift_model != self.rust_config.shift_model:
                    coercions.append(AutoCoercionPoint(
                        kind=CoercionKind.SHIFT,
                        c_instruction=c_inst,
                        rust_instruction=r_inst,
                        description=f"Shift semantics: C={self.c_config.shift_model}, "
                                   f"Rust={self.rust_config.shift_model}",
                        severity="warning",
                    ))

            # Division
            if op in ("SDIV", "UDIV", "SREM", "UREM"):
                if self.c_config.division_model != self.rust_config.division_model:
                    coercions.append(AutoCoercionPoint(
                        kind=CoercionKind.DIVISION,
                        c_instruction=c_inst,
                        rust_instruction=r_inst,
                        description=f"Division semantics: C={self.c_config.division_model}, "
                                   f"Rust={self.rust_config.division_model}",
                        severity="error",
                    ))

        if isinstance(c_inst, UnaryOp) and isinstance(r_inst, UnaryOp):
            if c_inst.op.name == "NEG":
                coercions.append(AutoCoercionPoint(
                    kind=CoercionKind.NEGATION,
                    c_instruction=c_inst,
                    rust_instruction=r_inst,
                    description="Negation of INT_MIN: C=UB, Rust=wrap/panic",
                    severity="warning",
                ))

        if isinstance(c_inst, CastInst) and isinstance(r_inst, CastInst):
            if c_inst.cast_kind.name in ("FPTOSI", "FPTOUI"):
                coercions.append(AutoCoercionPoint(
                    kind=CoercionKind.FLOAT_TO_INT,
                    c_instruction=c_inst,
                    rust_instruction=r_inst,
                    description="Float-to-int: C=UB for OOB, Rust=saturate",
                    severity="warning",
                ))

        return coercions
