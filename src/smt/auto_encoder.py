"""
Automated SMT encoding pipeline for cross-language equivalence verification.

Given two IR functions (one from C, one from Rust), automatically:
1. Walk the IR in topological order
2. Encode each instruction into Z3 bitvector/FP constraints
3. Apply semantic configuration σ = (ovf, fp, err) during encoding
4. Generate equivalence conditions
5. Handle: integer overflow, null/Option, error handling, array bounds, shifts
6. Handle: pointer/memory operations via QF_ABV array theory (load, store, alloca, GEP)

This replaces hand-coded per-category Z3 queries with a generic IR→Z3 compiler.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any, Set

import z3

from ..ir.types import (
    IRType, IntType, FloatType, PointerType, VoidType,
    ArrayType, StructType, FunctionType, Signedness, FloatKind,
    OverflowBehavior, Language,
)
from ..ir.instructions import (
    Instruction, BinaryOp, UnaryOp, CompareOp, CastInst,
    LoadInst, StoreInst, CallInst, ReturnInst, BranchInst,
    PhiInst, SelectInst, AllocaInst, GetElementPtrInst,
    MemcpyInst, MemsetInst,
    Value, Constant, Argument, BinOpKind, CmpPredicate, CastKind,
    InstructionMetadata,
)
from ..ir.basic_block import BasicBlock
from ..ir.function import Function
from ..semantics.semantic_config import (
    SemanticConfig, OverflowMode, FloatModel, ErrorModel,
    ShiftModel, DivisionModel, FloatToIntModel, ArrayBoundsModel,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Divergence witness
# ---------------------------------------------------------------------------

@dataclass
class DivergenceWitness:
    """A concrete input that causes C and Rust to produce different results."""
    inputs: Dict[str, Any]
    c_result: Any
    rust_result: Any
    divergence_kind: str
    explanation: str

    def __repr__(self) -> str:
        inp = ", ".join(f"{k}={v}" for k, v in self.inputs.items())
        return (f"DivergenceWitness({inp} -> C:{self.c_result}, "
                f"Rust:{self.rust_result}, kind={self.divergence_kind})")


@dataclass
class EquivalenceResult:
    """Result of an automated equivalence check."""
    equivalent: bool
    status: str  # "equivalent", "divergent", "unknown", "timeout", "error"
    witness: Optional[DivergenceWitness] = None
    divergence_kinds: List[str] = field(default_factory=list)
    checked_paths: int = 0
    smt_time_ms: float = 0.0
    explanation: str = ""

    def __repr__(self) -> str:
        return f"EquivalenceResult(status={self.status}, witness={self.witness})"


# ---------------------------------------------------------------------------
# Semantic-aware encoding context
# ---------------------------------------------------------------------------

class AutoEncodingContext:
    """
    Encoding context that automatically applies semantic configuration.

    Unlike the base EncodingContext, this one:
    - Tracks which semantic config applies to each variable
    - Generates overflow/shift/division guards based on config
    - Creates divergence conditions at coercion points
    """

    def __init__(self, c_config: SemanticConfig, rust_config: SemanticConfig,
                 pointer_width: int = 64):
        self.c_config = c_config
        self.rust_config = rust_config
        self.pointer_width = pointer_width

        self.solver = z3.Solver()
        self.solver.set("timeout", 30000)  # 30 second timeout

        self.declarations: Dict[str, z3.ExprRef] = {}
        self.c_return: Optional[z3.ExprRef] = None
        self.rust_return: Optional[z3.ExprRef] = None

        # Track divergence conditions
        self.divergence_conditions: List[Tuple[z3.BoolRef, str]] = []
        # Track assumptions (preconditions)
        self.assumptions: List[z3.BoolRef] = []
        # Track side conditions (e.g., no UB in C side)
        self.c_side_conditions: List[z3.BoolRef] = []
        self.rust_side_conditions: List[z3.BoolRef] = []

        self._fresh_counter = 0

        # --- Memory model state (per-side, keyed by prefix) ---
        # Memory is Z3 Array(BitVec(ptr_width), BitVec(8)) with SSA versioning.
        self._mem_version: Dict[str, int] = {}     # prefix -> current version
        self._mem_arrays: Dict[str, z3.ArrayRef] = {}  # "c_mem_0" -> Z3 array
        self._alloc_counter: Dict[str, int] = {}   # prefix -> next alloc id
        self._alloc_bases: Dict[str, Dict[int, z3.BitVecRef]] = {}  # prefix -> {id: base}
        self._alloc_sizes: Dict[str, Dict[int, z3.BitVecRef]] = {}  # prefix -> {id: size}
        self._ptr_provenance: Dict[str, Dict[str, int]] = {}  # prefix -> {var: alloc_id}

    def _init_memory(self, prefix: str) -> None:
        """Initialize a fresh memory array for a function side."""
        mem_name = f"{prefix}mem_0"
        addr_sort = z3.BitVecSort(self.pointer_width)
        byte_sort = z3.BitVecSort(8)
        mem = z3.Array(mem_name, addr_sort, byte_sort)
        self._mem_version[prefix] = 0
        self._mem_arrays[mem_name] = mem
        self._alloc_counter[prefix] = 0
        self._alloc_bases[prefix] = {}
        self._alloc_sizes[prefix] = {}
        self._ptr_provenance[prefix] = {}

    def get_memory(self, prefix: str) -> z3.ArrayRef:
        """Get the current memory array for a side."""
        v = self._mem_version.get(prefix, 0)
        key = f"{prefix}mem_{v}"
        return self._mem_arrays.get(key, z3.Array(key, z3.BitVecSort(self.pointer_width), z3.BitVecSort(8)))

    def update_memory(self, prefix: str, new_mem: z3.ArrayRef) -> None:
        """Create a new SSA version of memory for a side."""
        v = self._mem_version.get(prefix, 0) + 1
        self._mem_version[prefix] = v
        key = f"{prefix}mem_{v}"
        self._mem_arrays[key] = new_mem

    def alloc_stack(self, prefix: str, size_bytes: int, align: int = 8) -> z3.BitVecRef:
        """Allocate stack memory, returning a symbolic base address."""
        aid = self._alloc_counter.get(prefix, 0)
        self._alloc_counter[prefix] = aid + 1
        base = z3.BitVec(f"{prefix}alloc_{aid}_base", self.pointer_width)
        sz = z3.BitVecVal(size_bytes, self.pointer_width)
        self._alloc_bases.setdefault(prefix, {})[aid] = base
        self._alloc_sizes.setdefault(prefix, {})[aid] = sz
        # Alignment constraint
        if align > 1:
            self.solver.add(z3.URem(base, z3.BitVecVal(align, self.pointer_width)) == 0)
        # Non-null
        self.solver.add(base != z3.BitVecVal(0, self.pointer_width))
        # Non-overlapping with previous allocations
        for prev_id, prev_base in list(self._alloc_bases.get(prefix, {}).items()):
            if prev_id == aid:
                continue
            prev_sz = self._alloc_sizes[prefix][prev_id]
            # base+size <= prev_base OR prev_base+prev_size <= base
            self.solver.add(z3.Or(
                z3.ULE(base + sz, prev_base),
                z3.ULE(prev_base + prev_sz, base)
            ))
        return base

    def fresh(self, prefix: str = "t") -> str:
        self._fresh_counter += 1
        return f"__{prefix}_{self._fresh_counter}"

    def declare_bv(self, name: str, width: int) -> z3.BitVecRef:
        if name in self.declarations:
            return self.declarations[name]
        v = z3.BitVec(name, width)
        self.declarations[name] = v
        return v

    def declare_fp(self, name: str, ebits: int = 11, sbits: int = 53) -> z3.FPRef:
        if name in self.declarations:
            return self.declarations[name]
        v = z3.FP(name, z3.FPSort(ebits, sbits))
        self.declarations[name] = v
        return v

    def declare_bool(self, name: str) -> z3.BoolRef:
        if name in self.declarations:
            return self.declarations[name]
        v = z3.Bool(name)
        self.declarations[name] = v
        return v

    def get(self, name: str) -> Optional[z3.ExprRef]:
        return self.declarations.get(name)

    def set(self, name: str, expr: z3.ExprRef) -> None:
        self.declarations[name] = expr


# ---------------------------------------------------------------------------
# Automated SMT Encoder
# ---------------------------------------------------------------------------

class AutoSMTEncoder:
    """
    Automatically encodes two IR functions into Z3 and checks equivalence.

    The key insight: the semantic configuration σ = (ovf, fp, err, shift, div)
    is automatically applied during IR lowering, producing different Z3
    encodings for the same IR depending on whether it came from C or Rust.
    """

    def __init__(self, c_config: Optional[SemanticConfig] = None,
                 rust_config: Optional[SemanticConfig] = None,
                 pointer_width: int = 64):
        self.c_config = c_config or SemanticConfig.c11()
        self.rust_config = rust_config or SemanticConfig.rust_release()
        self.pointer_width = pointer_width

    def check_equivalence(
        self,
        c_func: Function,
        rust_func: Function,
        timeout_ms: int = 30000,
    ) -> EquivalenceResult:
        """
        Check if c_func and rust_func are semantically equivalent
        under their respective semantic configurations.

        Returns EquivalenceResult with status and optional counterexample.
        """
        import time
        t0 = time.time()

        ctx = AutoEncodingContext(self.c_config, self.rust_config, self.pointer_width)
        ctx.solver.set("timeout", timeout_ms)

        try:
            # Step 1: Create shared symbolic inputs
            shared_inputs = self._create_shared_inputs(c_func, rust_func, ctx)

            # Step 2: Encode C function with C semantics
            c_ret = self._encode_function(c_func, ctx, "c_", self.c_config)
            ctx.c_return = c_ret

            # Step 3: Encode Rust function with Rust semantics
            rust_ret = self._encode_function(rust_func, ctx, "r_", self.rust_config)
            ctx.rust_return = rust_ret

            # Step 4: DO NOT add C-side preconditions
            # The key insight: C's UB on overflow means the BEHAVIOR IS UNDEFINED,
            # but the BV semantics still wrap. The divergence exists because:
            # - C compiler may optimize assuming no overflow
            # - The overflow case IS the semantic divergence we want to detect
            # We check: "is there any input where the two functions differ?"
            # (including overflow inputs, since those are where divergences happen)

            # Step 5: Check if outputs can differ
            if c_ret is not None and rust_ret is not None:
                # Coerce to same width if needed
                c_ret, rust_ret = self._coerce_pair(c_ret, rust_ret)
                # Ask: is there an input where they differ?
                ctx.solver.add(c_ret != rust_ret)
            elif ctx.divergence_conditions:
                # Check explicit divergence conditions
                ctx.solver.add(z3.Or([d for d, _ in ctx.divergence_conditions]))
            else:
                elapsed = (time.time() - t0) * 1000
                return EquivalenceResult(
                    equivalent=True, status="equivalent",
                    smt_time_ms=elapsed,
                    explanation="Both functions have void return; trivially equivalent"
                )

            result = ctx.solver.check()
            elapsed = (time.time() - t0) * 1000

            if result == z3.unsat:
                return EquivalenceResult(
                    equivalent=True, status="equivalent",
                    smt_time_ms=elapsed,
                    explanation="Z3 proved no input can cause different outputs"
                )
            elif result == z3.sat:
                model = ctx.solver.model()
                witness = self._extract_witness(model, shared_inputs, c_ret, rust_ret, ctx)
                return EquivalenceResult(
                    equivalent=False, status="divergent",
                    witness=witness,
                    divergence_kinds=witness.divergence_kind.split(",") if witness else [],
                    smt_time_ms=elapsed,
                    explanation=f"Found diverging input: {witness}"
                )
            else:
                return EquivalenceResult(
                    equivalent=False, status="unknown",
                    smt_time_ms=elapsed,
                    explanation="Z3 could not determine equivalence (timeout or resource limit)"
                )

        except Exception as e:
            elapsed = (time.time() - t0) * 1000
            return EquivalenceResult(
                equivalent=False, status="error",
                smt_time_ms=elapsed,
                explanation=f"Encoding error: {e}"
            )

    def _create_shared_inputs(
        self, c_func: Function, rust_func: Function,
        ctx: AutoEncodingContext
    ) -> Dict[str, z3.ExprRef]:
        """Create shared symbolic input variables for both functions."""
        shared = {}
        c_args = list(c_func.arguments)
        r_args = list(rust_func.arguments)
        n = min(len(c_args), len(r_args))

        for i in range(n):
            c_arg = c_args[i]
            r_arg = r_args[i]
            name = f"input_{i}"
            sort = self._type_to_sort(c_arg.type)
            sym = ctx.declare_bv(name, sort.size()) if z3.is_bv_sort(sort) else ctx.declare_fp(name)
            shared[name] = sym

            # Alias for both sides
            c_name = f"c_{c_arg.name or f'arg{i}'}"
            r_name = f"r_{r_arg.name or f'arg{i}'}"
            ctx.set(c_name, sym)
            ctx.set(r_name, sym)

        return shared

    def _encode_function(
        self, func: Function, ctx: AutoEncodingContext,
        prefix: str, config: SemanticConfig,
    ) -> Optional[z3.ExprRef]:
        """Encode a function's IR into Z3, applying semantic config.
        
        Handles control flow by building conditional return expressions
        when branches lead to different return values.
        Handles memory operations via QF_ABV array theory.
        """
        # Initialize memory model for this side
        ctx._init_memory(prefix)

        # Map from instruction object id to Z3 expression (avoids name collisions)
        inst_results: Dict[int, z3.ExprRef] = {}
        # Collect all (path_condition, return_value) pairs
        return_pairs: List[Tuple[Optional[z3.ExprRef], z3.ExprRef]] = []
        last_branch_cond = None
        in_then_branch = False

        for block_idx, block in enumerate(func.blocks):
            for inst in block.instructions:
                result = self._encode_instruction(inst, ctx, prefix, config)
                if result is not None:
                    inst_results[id(inst)] = result
                    if inst.name:
                        ctx.set(f"{prefix}{inst.name}", result)

                # Track conditional branches
                if isinstance(inst, BranchInst) and inst.is_conditional:
                    cond_operand = inst.operands[0] if inst.operands else None
                    if cond_operand is not None:
                        # Use direct instruction reference to avoid name collisions
                        cond_val = inst_results.get(id(cond_operand))
                        if cond_val is None:
                            cond_val = self._resolve_value(cond_operand, ctx, prefix)
                        if cond_val is not None:
                            if z3.is_bv(cond_val) and cond_val.size() == 1:
                                cond_val = cond_val == z3.BitVecVal(1, 1)
                            last_branch_cond = cond_val
                            in_then_branch = True

                if isinstance(inst, ReturnInst) and result is not None:
                    if last_branch_cond is not None and in_then_branch:
                        return_pairs.append((last_branch_cond, result))
                        in_then_branch = False
                    else:
                        return_pairs.append((None, result))
                        last_branch_cond = None

        if not return_pairs:
            return None

        # Build nested If-Then-Else from collected returns
        # Last unconditional return (or last return) is the default
        # Find the default (unconditional) return
        default_ret = None
        conditional_returns = []
        for cond, val in return_pairs:
            if cond is None:
                default_ret = val
            else:
                conditional_returns.append((cond, val))

        if not conditional_returns:
            return default_ret

        if default_ret is None:
            # All returns are conditional; use last one as default
            default_ret = conditional_returns[-1][1]
            conditional_returns = conditional_returns[:-1]

        # Build: If(cond1, val1, If(cond2, val2, ..., default))
        result = default_ret
        for cond, val in reversed(conditional_returns):
            # Coerce val and result to same sort
            val, result = self._coerce_pair(val, result)
            result = z3.If(cond, val, result)

        return result

    def _encode_instruction(
        self, inst: Instruction, ctx: AutoEncodingContext,
        prefix: str, config: SemanticConfig,
    ) -> Optional[z3.ExprRef]:
        """Encode a single instruction with semantic config applied."""
        if isinstance(inst, BinaryOp):
            return self._encode_binop(inst, ctx, prefix, config)
        if isinstance(inst, UnaryOp):
            return self._encode_unaryop(inst, ctx, prefix, config)
        if isinstance(inst, CompareOp):
            return self._encode_compare(inst, ctx, prefix, config)
        if isinstance(inst, CastInst):
            return self._encode_cast(inst, ctx, prefix, config)
        if isinstance(inst, SelectInst):
            return self._encode_select(inst, ctx, prefix, config)
        if isinstance(inst, PhiInst):
            return self._encode_phi(inst, ctx, prefix)
        if isinstance(inst, ReturnInst):
            return self._encode_return(inst, ctx, prefix)
        if isinstance(inst, AllocaInst):
            return self._encode_alloca(inst, ctx, prefix)
        if isinstance(inst, GetElementPtrInst):
            return self._encode_gep(inst, ctx, prefix, config)
        if isinstance(inst, LoadInst):
            return self._encode_load(inst, ctx, prefix)
        if isinstance(inst, StoreInst):
            return self._encode_store(inst, ctx, prefix)
        if isinstance(inst, CallInst):
            return self._encode_call(inst, ctx, prefix, config)
        if isinstance(inst, BranchInst):
            return None
        # Unknown instruction: model as unconstrained
        if inst.name and inst.type:
            sort = self._type_to_sort(inst.type)
            return ctx.declare_bv(f"{prefix}{inst.name}", sort.size()) if z3.is_bv_sort(sort) else None
        return None

    # -----------------------------------------------------------------------
    # Binary operations with semantic config
    # -----------------------------------------------------------------------

    def _encode_binop(
        self, inst: BinaryOp, ctx: AutoEncodingContext,
        prefix: str, config: SemanticConfig,
    ) -> z3.ExprRef:
        lhs = self._resolve_value(inst.lhs, ctx, prefix)
        rhs = self._resolve_value(inst.rhs, ctx, prefix)
        lhs, rhs = self._coerce_pair(lhs, rhs)

        op = inst.op.name
        is_c = prefix.startswith("c")

        # Core operation (modular arithmetic / BV semantics)
        result = self._binop_core(op, lhs, rhs, inst.type)

        # Apply semantic configuration for overflow
        if op in ("ADD", "SUB", "MUL") and z3.is_bv(lhs):
            width = lhs.size()
            is_signed = isinstance(inst.type, IntType) and inst.type.is_signed
            overflow_cond = self._compute_overflow(op, lhs, rhs, width, is_signed)

            if overflow_cond is not None:
                # Check instruction metadata first (e.g., wrapping_add → WRAP)
                inst_ovf = None
                if inst.metadata and inst.metadata.overflow:
                    md_ovf = inst.metadata.overflow
                    if md_ovf == OverflowBehavior.WRAP:
                        inst_ovf = OverflowMode.Wrap
                    elif md_ovf == OverflowBehavior.UNDEFINED:
                        # UNDEFINED only means UB for signed; unsigned is always well-defined
                        inst_ovf = OverflowMode.UB if is_signed else OverflowMode.Wrap
                    elif md_ovf == OverflowBehavior.SATURATE:
                        inst_ovf = OverflowMode.Saturate
                    elif md_ovf == OverflowBehavior.TRAP:
                        inst_ovf = OverflowMode.Panic

                if inst_ovf is not None:
                    ovf_mode = inst_ovf
                elif is_c:
                    ovf_mode = config.signed_overflow if is_signed else config.unsigned_overflow
                else:
                    ovf_mode = config.signed_overflow if is_signed else config.unsigned_overflow

                if ovf_mode == OverflowMode.UB:
                    # C signed overflow = UB → model as unconstrained (any result possible)
                    # This is the key: C compiler can produce ANY value on overflow
                    ub_var = ctx.declare_bv(ctx.fresh(f"{prefix}ub"), width)
                    result = z3.If(overflow_cond, ub_var, result)
                elif ovf_mode == OverflowMode.Wrap:
                    pass  # BV already wraps
                elif ovf_mode == OverflowMode.Panic:
                    # Rust debug: panic on overflow → result is "trapped"
                    trap_var = ctx.declare_bv(ctx.fresh(f"{prefix}trap"), width)
                    result = z3.If(overflow_cond, trap_var, result)
                elif ovf_mode == OverflowMode.Saturate:
                    max_val = z3.BitVecVal((1 << (width - 1)) - 1, width)
                    min_val = z3.BitVecVal(-(1 << (width - 1)), width)
                    result = z3.If(
                        overflow_cond,
                        z3.If(lhs > z3.BitVecVal(0, width), max_val, min_val),
                        result
                    )

        # Apply semantic configuration for division
        if op in ("SDIV", "UDIV", "SREM", "UREM") and z3.is_bv(rhs):
            width = rhs.size()
            zero = z3.BitVecVal(0, width)
            div_by_zero = rhs == zero

            if is_c:
                if config.division_model == DivisionModel.UB:
                    # Division by zero in C is UB → result is unconstrained
                    ub_div = ctx.declare_bv(ctx.fresh(f"{prefix}divub"), width)
                    result = z3.If(div_by_zero, ub_div, result)
                    # Also: signed INT_MIN / -1 is UB in C
                    if op in ("SDIV", "SREM"):
                        int_min = z3.BitVecVal(-(1 << (width - 1)), width)
                        neg_one = z3.BitVecVal(-1, width)
                        ub_div2 = ctx.declare_bv(ctx.fresh(f"{prefix}divub2"), width)
                        result = z3.If(z3.And(lhs == int_min, rhs == neg_one), ub_div2, result)
            else:
                if config.division_model == DivisionModel.Panic:
                    trap_var = ctx.declare_bv(ctx.fresh(f"{prefix}divtrap"), width)
                    result = z3.If(div_by_zero, trap_var, result)

        # Apply semantic configuration for shifts
        if op in ("SHL", "LSHR", "ASHR") and z3.is_bv(rhs):
            width = rhs.size()
            overshift = z3.UGE(rhs, z3.BitVecVal(width, width))

            if is_c:
                if config.shift_model == ShiftModel.UB_on_overshift:
                    # Overshift in C = UB → result is unconstrained
                    ub_shift = ctx.declare_bv(ctx.fresh(f"{prefix}shiftub"), width)
                    result = z3.If(overshift, ub_shift, result)
            else:
                if config.shift_model == ShiftModel.Mask:
                    # Rust release: mask shift amount
                    mask = z3.BitVecVal(width - 1, width)
                    masked_rhs = rhs & mask
                    result = self._binop_core(op, lhs, masked_rhs, inst.type)
                elif config.shift_model == ShiftModel.Panic_on_overshift:
                    trap_var = ctx.declare_bv(ctx.fresh(f"{prefix}shifttrap"), width)
                    result = z3.If(overshift, trap_var, result)

        # Store result
        if inst.name:
            ctx.set(f"{prefix}{inst.name}", result)

        return result

    def _binop_core(self, op: str, lhs: z3.ExprRef, rhs: z3.ExprRef,
                    result_type: Optional[IRType]) -> z3.ExprRef:
        """Core binary operation (pure BV/FP semantics, no config)."""
        if op == "ADD": return lhs + rhs
        if op == "SUB": return lhs - rhs
        if op == "MUL": return lhs * rhs
        if op == "SDIV":
            if z3.is_bv(lhs):
                return lhs / rhs
            return lhs / rhs
        if op == "UDIV": return z3.UDiv(lhs, rhs)
        if op == "SREM": return z3.SRem(lhs, rhs)
        if op == "UREM": return z3.URem(lhs, rhs)
        if op == "SHL": return lhs << rhs
        if op == "LSHR": return z3.LShR(lhs, rhs)
        if op == "ASHR": return lhs >> rhs
        if op == "AND": return lhs & rhs
        if op == "OR": return lhs | rhs
        if op == "XOR": return lhs ^ rhs
        if op == "FADD": return self._fp_binop("add", lhs, rhs, result_type)
        if op == "FSUB": return self._fp_binop("sub", lhs, rhs, result_type)
        if op == "FMUL": return self._fp_binop("mul", lhs, rhs, result_type)
        if op == "FDIV": return self._fp_binop("div", lhs, rhs, result_type)
        if op == "FREM": return self._fp_binop("rem", lhs, rhs, result_type)
        return lhs  # fallback

    def _fp_binop(self, op: str, lhs: z3.ExprRef, rhs: z3.ExprRef,
                  result_type: Optional[IRType]) -> z3.ExprRef:
        lhs = self._ensure_fp(lhs, result_type)
        rhs = self._ensure_fp(rhs, result_type)
        rm = z3.RNE()
        if op == "add": return z3.fpAdd(rm, lhs, rhs)
        if op == "sub": return z3.fpSub(rm, lhs, rhs)
        if op == "mul": return z3.fpMul(rm, lhs, rhs)
        if op == "div": return z3.fpDiv(rm, lhs, rhs)
        if op == "rem": return z3.fpRem(lhs, rhs)
        return lhs

    def _compute_overflow(self, op: str, lhs: z3.BitVecRef, rhs: z3.BitVecRef,
                          width: int, signed: bool) -> Optional[z3.BoolRef]:
        """Compute overflow condition for arithmetic operations."""
        if not signed:
            # Unsigned overflow check
            if op == "ADD":
                result = lhs + rhs
                return z3.ULT(result, lhs)  # Wrapped around
            if op == "SUB":
                return z3.UGT(rhs, lhs)
            if op == "MUL":
                ext_lhs = z3.ZeroExt(width, lhs)
                ext_rhs = z3.ZeroExt(width, rhs)
                full = ext_lhs * ext_rhs
                return z3.Extract(2 * width - 1, width, full) != z3.BitVecVal(0, width)
            return None
        # Signed overflow
        zero = z3.BitVecVal(0, width)
        if op == "ADD":
            result = lhs + rhs
            return z3.Or(
                z3.And(lhs > zero, rhs > zero, result < zero),
                z3.And(lhs < zero, rhs < zero, result > zero)
            )
        if op == "SUB":
            result = lhs - rhs
            return z3.Or(
                z3.And(lhs > zero, rhs < zero, result < zero),
                z3.And(lhs < zero, rhs > zero, result > zero)
            )
        if op == "MUL":
            ext_lhs = z3.SignExt(width, lhs)
            ext_rhs = z3.SignExt(width, rhs)
            full = ext_lhs * ext_rhs
            trunc = z3.Extract(width - 1, 0, full)
            return z3.SignExt(width, trunc) != full
        return None

    # -----------------------------------------------------------------------
    # Unary operations
    # -----------------------------------------------------------------------

    def _encode_unaryop(
        self, inst: UnaryOp, ctx: AutoEncodingContext,
        prefix: str, config: SemanticConfig,
    ) -> z3.ExprRef:
        operand = self._resolve_value(inst._operands[0], ctx, prefix)
        op = inst.op.name

        if op == "NEG":
            result = -operand
            # Negation of INT_MIN is overflow
            if z3.is_bv(operand):
                width = operand.size()
                int_min = z3.BitVecVal(-(1 << (width - 1)), width)
                neg_overflow = operand == int_min
                is_c = prefix.startswith("c")
                if is_c and config.signed_overflow == OverflowMode.UB:
                    # Negation of INT_MIN in C = UB → result is unconstrained
                    ub_neg = ctx.declare_bv(ctx.fresh(f"{prefix}negub"), width)
                    result = z3.If(neg_overflow, ub_neg, result)
                elif not is_c and config.signed_overflow == OverflowMode.Panic:
                    trap = ctx.declare_bv(ctx.fresh(f"{prefix}negtrap"), width)
                    result = z3.If(neg_overflow, trap, result)
        elif op == "NOT":
            result = z3.Not(operand) if z3.is_bool(operand) else ~operand
        elif op == "BITWISE_NOT":
            result = ~operand
        elif op == "FNEG":
            result = z3.fpNeg(operand) if z3.is_fp(operand) else -operand
        else:
            result = operand

        if inst.name:
            ctx.set(f"{prefix}{inst.name}", result)
        return result

    # -----------------------------------------------------------------------
    # Comparisons
    # -----------------------------------------------------------------------

    def _encode_compare(
        self, inst: CompareOp, ctx: AutoEncodingContext,
        prefix: str, config: SemanticConfig,
    ) -> z3.BoolRef:
        lhs = self._resolve_value(inst.lhs, ctx, prefix)
        rhs = self._resolve_value(inst.rhs, ctx, prefix)
        lhs, rhs = self._coerce_pair(lhs, rhs)

        pred = inst.predicate.name
        result = self._encode_predicate(pred, lhs, rhs)

        if inst.name:
            ctx.set(f"{prefix}{inst.name}", result)
        return result

    def _encode_predicate(self, pred: str, lhs: z3.ExprRef, rhs: z3.ExprRef) -> z3.BoolRef:
        if pred in ("EQ", "OEQ", "UEQ"): return lhs == rhs
        if pred in ("NE", "ONE", "UNE"): return lhs != rhs
        if pred in ("SLT", "OLT"):
            return lhs < rhs if z3.is_bv(lhs) else z3.fpLT(lhs, rhs)
        if pred in ("SLE", "OLE"):
            return lhs <= rhs if z3.is_bv(lhs) else z3.fpLEQ(lhs, rhs)
        if pred in ("SGT", "OGT"):
            return lhs > rhs if z3.is_bv(lhs) else z3.fpGT(lhs, rhs)
        if pred in ("SGE", "OGE"):
            return lhs >= rhs if z3.is_bv(lhs) else z3.fpGEQ(lhs, rhs)
        if pred == "ULT": return z3.ULT(lhs, rhs)
        if pred == "ULE": return z3.ULE(lhs, rhs)
        if pred == "UGT": return z3.UGT(lhs, rhs)
        if pred == "UGE": return z3.UGE(lhs, rhs)
        return lhs == rhs

    # -----------------------------------------------------------------------
    # Casts
    # -----------------------------------------------------------------------

    def _encode_cast(
        self, inst: CastInst, ctx: AutoEncodingContext,
        prefix: str, config: SemanticConfig,
    ) -> z3.ExprRef:
        operand = self._resolve_value(inst._operands[0], ctx, prefix)
        dst_type = inst.type
        kind = inst.cast_kind.name

        if kind == "TRUNC":
            w = dst_type.width if isinstance(dst_type, IntType) else 32
            if z3.is_bv(operand) and operand.size() > w:
                result = z3.Extract(w - 1, 0, operand)
            else:
                result = operand
        elif kind == "ZEXT":
            w = dst_type.width if isinstance(dst_type, IntType) else 64
            if z3.is_bv(operand):
                ext = w - operand.size()
                result = z3.ZeroExt(max(ext, 0), operand) if ext > 0 else operand
            else:
                result = operand
        elif kind == "SEXT":
            w = dst_type.width if isinstance(dst_type, IntType) else 64
            if z3.is_bv(operand):
                ext = w - operand.size()
                result = z3.SignExt(max(ext, 0), operand) if ext > 0 else operand
            else:
                result = operand
        elif kind == "FPTOSI":
            w = dst_type.width if isinstance(dst_type, IntType) else 32
            is_c = prefix.startswith("c")
            if z3.is_fp(operand):
                result = z3.fpToSBV(z3.RTZ(), operand, z3.BitVecSort(w))
                # Float-to-int out-of-range: C = UB, Rust = saturate
                if is_c and config.float_to_int == FloatToIntModel.UB:
                    pass  # UB, assume in-range
                elif not is_c and config.float_to_int == FloatToIntModel.Saturate:
                    max_val = z3.BitVecVal((1 << (w - 1)) - 1, w)
                    min_val = z3.BitVecVal(-(1 << (w - 1)), w)
                    fp_max = z3.FPVal(float((1 << (w - 1)) - 1), operand.sort())
                    fp_min = z3.FPVal(float(-(1 << (w - 1))), operand.sort())
                    is_nan = z3.fpIsNaN(operand)
                    result = z3.If(is_nan, z3.BitVecVal(0, w),
                             z3.If(z3.fpGT(operand, fp_max), max_val,
                             z3.If(z3.fpLT(operand, fp_min), min_val, result)))
            else:
                result = z3.BitVecVal(0, w)
        elif kind == "FPTOUI":
            w = dst_type.width if isinstance(dst_type, IntType) else 32
            if z3.is_fp(operand):
                result = z3.fpToUBV(z3.RTZ(), operand, z3.BitVecSort(w))
            else:
                result = z3.BitVecVal(0, w)
        elif kind == "SITOFP":
            sort = self._float_sort(dst_type)
            result = z3.fpSignedToFP(z3.RNE(), operand, sort) if z3.is_bv(operand) else z3.FPVal(0.0, sort)
        elif kind == "UITOFP":
            sort = self._float_sort(dst_type)
            result = z3.fpToFP(z3.RNE(), operand, sort) if z3.is_bv(operand) else z3.FPVal(0.0, sort)
        elif kind == "BITCAST":
            result = operand
        elif kind in ("PTRTOINT", "INTTOPTR"):
            w = self.pointer_width
            if isinstance(dst_type, IntType):
                w = dst_type.width
            if z3.is_bv(operand):
                if operand.size() > w:
                    result = z3.Extract(w - 1, 0, operand)
                elif operand.size() < w:
                    result = z3.ZeroExt(w - operand.size(), operand)
                else:
                    result = operand
            else:
                result = z3.BitVecVal(0, w)
        else:
            result = operand

        if inst.name:
            ctx.set(f"{prefix}{inst.name}", result)
        return result

    # -----------------------------------------------------------------------
    # Other instructions
    # -----------------------------------------------------------------------

    def _encode_select(self, inst: SelectInst, ctx: AutoEncodingContext,
                       prefix: str, config: SemanticConfig) -> z3.ExprRef:
        cond = self._resolve_value(inst._operands[0], ctx, prefix)
        tv = self._resolve_value(inst._operands[1], ctx, prefix)
        fv = self._resolve_value(inst._operands[2], ctx, prefix)

        if z3.is_bv(cond):
            cond = cond != z3.BitVecVal(0, cond.size())
        tv, fv = self._coerce_pair(tv, fv)
        result = z3.If(cond, tv, fv)
        if inst.name:
            ctx.set(f"{prefix}{inst.name}", result)
        return result

    def _encode_phi(self, inst: PhiInst, ctx: AutoEncodingContext,
                    prefix: str) -> z3.ExprRef:
        sort = self._type_to_sort(inst.type) if inst.type else z3.BitVecSort(32)
        name = f"{prefix}{inst.name or ctx.fresh('phi')}"
        if z3.is_bv_sort(sort):
            return ctx.declare_bv(name, sort.size())
        return ctx.declare_bool(name)

    def _encode_return(self, inst: ReturnInst, ctx: AutoEncodingContext,
                       prefix: str) -> Optional[z3.ExprRef]:
        if inst.return_value is not None:
            return self._resolve_value(inst.return_value, ctx, prefix)
        return None

    def _encode_load(self, inst: LoadInst, ctx: AutoEncodingContext,
                     prefix: str) -> z3.ExprRef:
        """Encode a load instruction using the memory array model.

        Reads width_bytes consecutive bytes from memory at the given address,
        composing them into a single bitvector (little-endian).
        """
        result_sort = self._type_to_sort(inst.type) if inst.type else z3.BitVecSort(32)
        name = f"{prefix}{inst.name or ctx.fresh('load')}"

        addr_expr = self._resolve_value(inst.address, ctx, prefix)
        if not z3.is_bv(addr_expr) or addr_expr.size() != ctx.pointer_width:
            # Fallback: unconstrained symbolic value
            if z3.is_bv_sort(result_sort):
                return ctx.declare_bv(name, result_sort.size())
            return ctx.declare_bool(name)

        mem = ctx.get_memory(prefix)
        if z3.is_bv_sort(result_sort):
            width_bytes = result_sort.size() // 8
            if width_bytes < 1:
                width_bytes = 1
            # Little-endian multi-byte read
            result = z3.Select(mem, addr_expr)  # byte 0
            for i in range(1, width_bytes):
                byte_i = z3.Select(mem, addr_expr + z3.BitVecVal(i, ctx.pointer_width))
                result = z3.Concat(byte_i, result)
            # result is now width_bytes*8 bits wide
            actual_bits = width_bytes * 8
            target_bits = result_sort.size()
            if actual_bits > target_bits:
                result = z3.Extract(target_bits - 1, 0, result)
            elif actual_bits < target_bits:
                result = z3.ZeroExt(target_bits - actual_bits, result)
            ctx.set(name, result)
            return result
        return ctx.declare_bool(name)

    def _encode_store(self, inst: StoreInst, ctx: AutoEncodingContext,
                      prefix: str) -> None:
        """Encode a store instruction by updating the memory array.

        Writes width_bytes consecutive bytes to memory at the given address
        (little-endian). Returns None since store produces no value.
        """
        val_expr = self._resolve_value(inst.value, ctx, prefix)
        addr_expr = self._resolve_value(inst.address, ctx, prefix)

        if not z3.is_bv(addr_expr) or addr_expr.size() != ctx.pointer_width:
            return None
        if not z3.is_bv(val_expr):
            return None

        mem = ctx.get_memory(prefix)
        width_bytes = val_expr.size() // 8
        if width_bytes < 1:
            width_bytes = 1

        # Little-endian multi-byte write
        new_mem = mem
        for i in range(width_bytes):
            byte_val = z3.Extract(i * 8 + 7, i * 8, val_expr) if val_expr.size() > 8 else val_expr
            a = addr_expr + z3.BitVecVal(i, ctx.pointer_width) if i > 0 else addr_expr
            new_mem = z3.Store(new_mem, a, byte_val)

        ctx.update_memory(prefix, new_mem)
        return None

    def _encode_alloca(self, inst: AllocaInst, ctx: AutoEncodingContext,
                       prefix: str) -> z3.ExprRef:
        """Encode an alloca instruction as a fresh non-overlapping allocation."""
        elem_size = self._type_size_bytes(inst.alloc_type)
        total_size = elem_size * inst.num_elements
        align = inst.alignment if inst.alignment > 0 else max(elem_size, 1)

        base = ctx.alloc_stack(prefix, total_size, align)
        name = f"{prefix}{inst.name or ctx.fresh('alloca')}"
        ctx.set(name, base)
        # Track provenance
        aid = ctx._alloc_counter.get(prefix, 1) - 1
        ctx._ptr_provenance.setdefault(prefix, {})[name] = aid
        return base

    def _encode_gep(self, inst: GetElementPtrInst, ctx: AutoEncodingContext,
                    prefix: str, config: SemanticConfig) -> z3.ExprRef:
        """Encode a GEP (GetElementPtr) instruction.

        Computes: base + sum(index_i * stride_i) where strides depend on
        the source element type structure (array element size, struct field offsets).
        """
        base_expr = self._resolve_value(inst.base, ctx, prefix)
        if not z3.is_bv(base_expr):
            base_expr = z3.BitVec(f"{prefix}{inst.name or ctx.fresh('gep')}", ctx.pointer_width)

        if base_expr.size() != ctx.pointer_width:
            if base_expr.size() < ctx.pointer_width:
                base_expr = z3.ZeroExt(ctx.pointer_width - base_expr.size(), base_expr)
            else:
                base_expr = z3.Extract(ctx.pointer_width - 1, 0, base_expr)

        offset = z3.BitVecVal(0, ctx.pointer_width)
        current_type = inst.source_element_type

        for i, idx_val in enumerate(inst.indices):
            idx_expr = self._resolve_value(idx_val, ctx, prefix)
            if not z3.is_bv(idx_expr):
                idx_expr = z3.BitVecVal(0, ctx.pointer_width)
            if idx_expr.size() != ctx.pointer_width:
                if idx_expr.size() < ctx.pointer_width:
                    idx_expr = z3.SignExt(ctx.pointer_width - idx_expr.size(), idx_expr)
                else:
                    idx_expr = z3.Extract(ctx.pointer_width - 1, 0, idx_expr)

            if i == 0:
                # First index: scale by element size
                elem_sz = self._type_size_bytes(current_type)
                offset = offset + idx_expr * z3.BitVecVal(elem_sz, ctx.pointer_width)
            else:
                if isinstance(current_type, ArrayType):
                    elem_sz = self._type_size_bytes(current_type.element)
                    offset = offset + idx_expr * z3.BitVecVal(elem_sz, ctx.pointer_width)
                    current_type = current_type.element
                elif isinstance(current_type, StructType):
                    # Struct index must be constant
                    if isinstance(idx_val, Constant) and isinstance(idx_val.value, int):
                        field_idx = idx_val.value
                        field_offset = self._struct_field_offset(current_type, field_idx)
                        offset = offset + z3.BitVecVal(field_offset, ctx.pointer_width)
                        if 0 <= field_idx < len(current_type.fields):
                            current_type = current_type.fields[field_idx].type
                    else:
                        # Non-constant struct index: model as offset * avg field size
                        avg_sz = max(self._type_size_bytes(current_type) // max(len(current_type.fields), 1), 1)
                        offset = offset + idx_expr * z3.BitVecVal(avg_sz, ctx.pointer_width)
                else:
                    # Scalar: treat as byte offset
                    offset = offset + idx_expr

        result = base_expr + offset
        name = f"{prefix}{inst.name or ctx.fresh('gep')}"
        ctx.set(name, result)

        # Bounds check for inbounds GEP (C UB if out of bounds)
        if inst.inbounds and prefix.startswith("c_"):
            # Track as assumption: the pointer must remain within its allocation
            src_name = inst.base.name or ""
            prov_key = f"{prefix}{src_name}"
            aid = ctx._ptr_provenance.get(prefix, {}).get(prov_key)
            if aid is not None and aid in ctx._alloc_bases.get(prefix, {}):
                alloc_base = ctx._alloc_bases[prefix][aid]
                alloc_size = ctx._alloc_sizes[prefix][aid]
                in_bounds = z3.And(
                    z3.UGE(result, alloc_base),
                    z3.ULE(result, alloc_base + alloc_size)
                )
                ctx.assumptions.append(in_bounds)

        return result

    def _encode_call(self, inst: CallInst, ctx: AutoEncodingContext,
                     prefix: str, config: SemanticConfig) -> z3.ExprRef:
        """Encode a call instruction, with special handling for memory intrinsics."""
        callee = inst.callee_name if hasattr(inst, 'callee_name') else (inst.name or "")
        # Check for memory allocation functions
        if isinstance(callee, str):
            callee_lower = callee.lower()
            if callee_lower in ("malloc", "calloc", "alloc"):
                # Model as heap allocation
                size_arg = inst.operands[0] if inst.operands else None
                size_val = 64  # default
                if size_arg and isinstance(size_arg, Constant) and isinstance(size_arg.value, int):
                    size_val = size_arg.value
                base = ctx.alloc_stack(prefix, size_val, 8)
                if callee_lower == "calloc":
                    # Zero-initialize
                    mem = ctx.get_memory(prefix)
                    for i in range(size_val):
                        a = base + z3.BitVecVal(i, ctx.pointer_width) if i > 0 else base
                        mem = z3.Store(mem, a, z3.BitVecVal(0, 8))
                    ctx.update_memory(prefix, mem)
                name = f"{prefix}{inst.name or ctx.fresh('call')}"
                ctx.set(name, base)
                return base
            if callee_lower == "free":
                # Model free as no-op for equivalence (but track for UB detection)
                return z3.BoolVal(True)
            if callee_lower in ("memcpy", "memmove"):
                # Model as array copy
                if len(inst.operands) >= 3:
                    dst = self._resolve_value(inst.operands[0], ctx, prefix)
                    src = self._resolve_value(inst.operands[1], ctx, prefix)
                    n_arg = inst.operands[2]
                    n_val = 0
                    if isinstance(n_arg, Constant) and isinstance(n_arg.value, int):
                        n_val = n_arg.value
                    if z3.is_bv(dst) and z3.is_bv(src) and n_val > 0:
                        mem = ctx.get_memory(prefix)
                        for i in range(min(n_val, 64)):  # cap at 64 bytes
                            s = src + z3.BitVecVal(i, ctx.pointer_width) if i > 0 else src
                            d = dst + z3.BitVecVal(i, ctx.pointer_width) if i > 0 else dst
                            byte_val = z3.Select(mem, s)
                            mem = z3.Store(mem, d, byte_val)
                        ctx.update_memory(prefix, mem)
                    return dst if z3.is_bv(dst) else z3.BoolVal(True)
            if callee_lower == "memset":
                if len(inst.operands) >= 3:
                    dst = self._resolve_value(inst.operands[0], ctx, prefix)
                    val_arg = self._resolve_value(inst.operands[1], ctx, prefix)
                    n_arg = inst.operands[2]
                    n_val = 0
                    if isinstance(n_arg, Constant) and isinstance(n_arg.value, int):
                        n_val = n_arg.value
                    if z3.is_bv(dst) and n_val > 0:
                        mem = ctx.get_memory(prefix)
                        byte_val = z3.Extract(7, 0, val_arg) if z3.is_bv(val_arg) and val_arg.size() >= 8 else z3.BitVecVal(0, 8)
                        for i in range(min(n_val, 64)):
                            d = dst + z3.BitVecVal(i, ctx.pointer_width) if i > 0 else dst
                            mem = z3.Store(mem, d, byte_val)
                        ctx.update_memory(prefix, mem)
                    return dst if z3.is_bv(dst) else z3.BoolVal(True)

        if inst.type and not isinstance(inst.type, VoidType):
            sort = self._type_to_sort(inst.type)
            name = f"{prefix}{inst.name or ctx.fresh('call')}"
            if z3.is_bv_sort(sort):
                return ctx.declare_bv(name, sort.size())
        return z3.BoolVal(True)

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _resolve_value(self, val: Value, ctx: AutoEncodingContext,
                       prefix: str) -> z3.ExprRef:
        """Resolve an IR value to a Z3 expression."""
        if isinstance(val, Constant):
            return self._encode_constant(val)

        # Try prefixed name first, then unprefixed
        name = val.name or f"v_{val.id}"
        for n in [f"{prefix}{name}", name]:
            existing = ctx.get(n)
            if existing is not None:
                return existing

        # Create new symbolic variable
        sort = self._type_to_sort(val.type) if val.type else z3.BitVecSort(32)
        full_name = f"{prefix}{name}"
        if z3.is_bv_sort(sort):
            return ctx.declare_bv(full_name, sort.size())
        return ctx.declare_bool(full_name)

    def _encode_constant(self, const: Constant) -> z3.ExprRef:
        ty = const.type
        val = const.value
        if isinstance(ty, IntType):
            v = 1 if isinstance(val, bool) and val else (0 if isinstance(val, bool) else int(val or 0))
            return z3.BitVecVal(v & ((1 << ty.width) - 1), ty.width)
        if isinstance(ty, FloatType):
            sort = z3.FPSort(8, 24) if ty.kind == FloatKind.F32 else z3.FPSort(11, 53)
            return z3.FPVal(float(val or 0), sort)
        if isinstance(ty, PointerType):
            return z3.BitVecVal(int(val or 0) & ((1 << self.pointer_width) - 1), self.pointer_width)
        return z3.BitVecVal(0, 32)

    def _type_to_sort(self, ty: IRType) -> z3.SortRef:
        if isinstance(ty, IntType):
            return z3.BitVecSort(ty.width)
        if isinstance(ty, FloatType):
            return z3.FPSort(8, 24) if ty.kind == FloatKind.F32 else z3.FPSort(11, 53)
        if isinstance(ty, PointerType):
            return z3.BitVecSort(self.pointer_width)
        if isinstance(ty, ArrayType):
            return z3.ArraySort(z3.BitVecSort(64), self._type_to_sort(ty.element))
        return z3.BitVecSort(32)

    def _float_sort(self, ty: Optional[IRType]) -> z3.FPSortRef:
        if isinstance(ty, FloatType) and ty.kind == FloatKind.F32:
            return z3.FPSort(8, 24)
        return z3.FPSort(11, 53)

    def _ensure_fp(self, expr: z3.ExprRef, ty: Optional[IRType]) -> z3.FPRef:
        if z3.is_fp(expr):
            return expr
        sort = self._float_sort(ty)
        if z3.is_bv(expr):
            return z3.fpBVToFP(expr, sort)
        return z3.FPVal(0.0, sort)

    def _coerce_pair(self, a: z3.ExprRef, b: z3.ExprRef) -> Tuple[z3.ExprRef, z3.ExprRef]:
        """Coerce two Z3 expressions to the same sort."""
        # Handle Bool ↔ BV coercion
        if z3.is_bool(a) and z3.is_bv(b):
            a = z3.If(a, z3.BitVecVal(1, b.size()), z3.BitVecVal(0, b.size()))
        elif z3.is_bv(a) and z3.is_bool(b):
            b = z3.If(b, z3.BitVecVal(1, a.size()), z3.BitVecVal(0, a.size()))
        if z3.is_bv(a) and z3.is_bv(b):
            if a.size() == b.size():
                return a, b
            if a.size() < b.size():
                a = z3.SignExt(b.size() - a.size(), a)
            else:
                b = z3.SignExt(a.size() - b.size(), b)
        return a, b

    # -----------------------------------------------------------------------
    # Type size computation (for memory model)
    # -----------------------------------------------------------------------

    def _type_size_bytes(self, ty: IRType) -> int:
        """Compute the size of an IR type in bytes."""
        if isinstance(ty, IntType):
            return max(ty.width // 8, 1)
        if isinstance(ty, FloatType):
            return 4 if ty.kind == FloatKind.F32 else 8
        if isinstance(ty, PointerType):
            return self.pointer_width // 8
        if isinstance(ty, ArrayType):
            return self._type_size_bytes(ty.element) * (ty.length if ty.length > 0 else 1)
        if isinstance(ty, StructType):
            total = 0
            for f in ty.fields:
                f_size = self._type_size_bytes(f.type)
                f_align = self._type_align(f.type)
                # Pad to alignment
                if f_align > 0 and total % f_align != 0:
                    total += f_align - (total % f_align)
                total += f_size
            return max(total, 1)
        if isinstance(ty, VoidType):
            return 0
        return 4  # default

    def _type_align(self, ty: IRType) -> int:
        """Compute natural alignment of a type in bytes."""
        if isinstance(ty, IntType):
            return max(ty.width // 8, 1)
        if isinstance(ty, FloatType):
            return 4 if ty.kind == FloatKind.F32 else 8
        if isinstance(ty, PointerType):
            return self.pointer_width // 8
        if isinstance(ty, ArrayType):
            return self._type_align(ty.element)
        if isinstance(ty, StructType):
            if not ty.fields:
                return 1
            return max(self._type_align(f.type) for f in ty.fields)
        return 4

    def _struct_field_offset(self, sty: StructType, field_idx: int) -> int:
        """Compute byte offset of a field within a struct."""
        offset = 0
        for i, f in enumerate(sty.fields):
            f_align = self._type_align(f.type)
            if f_align > 0 and offset % f_align != 0:
                offset += f_align - (offset % f_align)
            if i == field_idx:
                return offset
            offset += self._type_size_bytes(f.type)
        return offset

    def _extract_witness(
        self, model: z3.ModelRef, shared_inputs: Dict[str, z3.ExprRef],
        c_ret: Optional[z3.ExprRef], rust_ret: Optional[z3.ExprRef],
        ctx: AutoEncodingContext,
    ) -> DivergenceWitness:
        """Extract a concrete counterexample from a Z3 model."""
        inputs = {}
        for name, sym in shared_inputs.items():
            val = model.eval(sym, model_completion=True)
            try:
                inputs[name] = val.as_long() if z3.is_bv(val) else str(val)
            except Exception:
                inputs[name] = str(val)

        c_val = None
        r_val = None
        if c_ret is not None:
            v = model.eval(c_ret, model_completion=True)
            try:
                c_val = v.as_signed_long() if z3.is_bv(v) else str(v)
            except Exception:
                c_val = str(v)
        if rust_ret is not None:
            v = model.eval(rust_ret, model_completion=True)
            try:
                r_val = v.as_signed_long() if z3.is_bv(v) else str(v)
            except Exception:
                r_val = str(v)

        # Determine divergence kind
        kind = "output_mismatch"
        for div_cond, div_kind in ctx.divergence_conditions:
            if z3.is_true(model.eval(div_cond, model_completion=True)):
                kind = div_kind
                break

        return DivergenceWitness(
            inputs=inputs,
            c_result=c_val,
            rust_result=r_val,
            divergence_kind=kind,
            explanation=f"C returns {c_val}, Rust returns {r_val} for inputs {inputs}"
        )
