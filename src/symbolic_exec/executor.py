"""
Symbolic executor for IR instructions.

Executes IR instructions symbolically, handling all instruction types
with path exploration strategies (DFS, BFS, random) and configurable
loop unrolling bounds.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict, Tuple, Set, Any, Callable

import z3

from ..ir.types import (
    IRType, IntType, FloatType, PointerType, VoidType,
    ArrayType, StructType, Signedness, FloatKind,
    OverflowBehavior, Language,
)
from ..ir.instructions import (
    Instruction, BinaryOp, UnaryOp, CompareOp, CastInst,
    LoadInst, StoreInst, CallInst, ReturnInst, BranchInst,
    PhiInst, SelectInst, AllocaInst, GetElementPtrInst,
    Value, Constant, Argument, BinOpKind, CmpPredicate, CastKind,
)
from ..ir.basic_block import BasicBlock
from ..ir.function import Function
from .state import SymbolicState, SymbolicValue, PathConstraint


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class PathExplorationStrategy(Enum):
    DFS = auto()
    BFS = auto()
    RANDOM = auto()
    COVERAGE_GUIDED = auto()
    DIVERGENCE_SEEKING = auto()


@dataclass
class ExecutionConfig:
    """Configuration for symbolic execution."""
    max_paths: int = 1000
    max_depth: int = 100
    loop_unroll_bound: int = 5
    max_call_depth: int = 10
    solver_timeout_ms: int = 10000
    strategy: PathExplorationStrategy = PathExplorationStrategy.DFS
    check_overflows: bool = True
    check_memory_safety: bool = True
    inline_calls: bool = True
    pointer_width: int = 64


@dataclass
class ExecutionResult:
    """Result of symbolic execution."""
    completed_paths: List[SymbolicState] = field(default_factory=list)
    error_paths: List[SymbolicState] = field(default_factory=list)
    infeasible_paths: int = 0
    timeout_paths: int = 0
    total_instructions: int = 0
    total_paths_explored: int = 0
    covered_blocks: Set[str] = field(default_factory=set)

    @property
    def num_completed(self) -> int:
        return len(self.completed_paths)

    @property
    def num_errors(self) -> int:
        return len(self.error_paths)

    def summary(self) -> str:
        lines = [
            f"Execution Result:",
            f"  Completed paths: {self.num_completed}",
            f"  Error paths: {self.num_errors}",
            f"  Infeasible: {self.infeasible_paths}",
            f"  Timeouts: {self.timeout_paths}",
            f"  Total instructions: {self.total_instructions}",
            f"  Total paths explored: {self.total_paths_explored}",
            f"  Block coverage: {len(self.covered_blocks)} blocks",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Z3 type/sort helpers
# ---------------------------------------------------------------------------

def _ir_type_to_z3_sort(ty: IRType, ptr_width: int = 64) -> z3.SortRef:
    """Convert IR type to Z3 sort."""
    if isinstance(ty, IntType):
        return z3.BitVecSort(ty.width)
    if isinstance(ty, FloatType):
        if ty.kind == FloatKind.F32:
            return z3.FPSort(8, 24)
        return z3.FPSort(11, 53)
    if isinstance(ty, PointerType):
        return z3.BitVecSort(ptr_width)
    if isinstance(ty, VoidType):
        return z3.BoolSort()
    # Default
    return z3.BitVecSort(32)


def _make_z3_constant(val: Any, ty: IRType, ptr_width: int = 64) -> z3.ExprRef:
    """Create a Z3 constant from a concrete value and IR type."""
    if isinstance(ty, IntType):
        if isinstance(val, bool):
            val = 1 if val else 0
        return z3.BitVecVal(int(val) & ((1 << ty.width) - 1), ty.width)
    if isinstance(ty, FloatType):
        sort = z3.FPSort(8, 24) if ty.kind == FloatKind.F32 else z3.FPSort(11, 53)
        return z3.FPVal(float(val), sort)
    if isinstance(ty, PointerType):
        return z3.BitVecVal(int(val) & ((1 << ptr_width) - 1), ptr_width)
    return z3.BitVecVal(0, 32)


# ---------------------------------------------------------------------------
# Symbolic executor
# ---------------------------------------------------------------------------

class SymbolicExecutor:
    """
    Symbolic executor for IR programs.
    
    Executes IR instructions symbolically, forking on branches,
    with configurable path exploration and loop bounding.
    """

    def __init__(
        self,
        config: Optional[ExecutionConfig] = None,
        function_summaries: Optional[Dict[str, Callable]] = None,
    ):
        self.config = config or ExecutionConfig()
        self.function_summaries = function_summaries or {}

        # Worklist of states to explore
        self._worklist: List[SymbolicState] = []
        self._result = ExecutionResult()

        # Module-level function lookup
        self._functions: Dict[str, Function] = {}

        # Solver for feasibility checks
        self._solver = z3.Solver()
        self._solver.set("timeout", self.config.solver_timeout_ms)

    def execute_function(
        self,
        func: Function,
        module_functions: Optional[Dict[str, Function]] = None,
    ) -> ExecutionResult:
        """Execute a function symbolically, exploring all feasible paths."""
        self._functions = module_functions or {}
        self._result = ExecutionResult()
        self._worklist = []

        # Create initial state
        initial_state = self._create_initial_state(func)
        self._worklist.append(initial_state)

        # Main exploration loop
        while self._worklist and self._result.total_paths_explored < self.config.max_paths:
            state = self._pick_next_state()
            if state is None:
                break

            self._explore_state(state, func)
            self._result.total_paths_explored += 1

        # Collect coverage
        for s in self._result.completed_paths + self._result.error_paths:
            self._result.covered_blocks.update(s.covered_blocks)

        return self._result

    def _create_initial_state(self, func: Function) -> SymbolicState:
        """Create the initial symbolic state for a function."""
        state = SymbolicState(
            function_name=func.name,
            address_width=self.config.pointer_width,
        )

        # Create symbolic arguments
        for arg in func.arguments:
            sort = _ir_type_to_z3_sort(arg.type, self.config.pointer_width)
            if z3.is_bv_sort(sort):
                sym = z3.BitVec(f"arg_{arg.name or arg.index}", sort.size())
            elif isinstance(sort, z3.FPSortRef):
                sym = z3.FP(f"arg_{arg.name or arg.index}", sort)
            else:
                sym = z3.BitVec(f"arg_{arg.name or arg.index}", 32)

            state.set_var(
                arg.name or f"arg_{arg.index}",
                SymbolicValue(name=arg.name or f"arg_{arg.index}", z3_expr=sym, ir_type=arg.type),
            )

        # Set entry block
        blocks = list(func.blocks)
        if blocks:
            state.current_block = blocks[0].name

        return state

    def _pick_next_state(self) -> Optional[SymbolicState]:
        """Pick the next state to explore based on the strategy."""
        if not self._worklist:
            return None

        if self.config.strategy == PathExplorationStrategy.DFS:
            return self._worklist.pop()  # Stack: LIFO
        elif self.config.strategy == PathExplorationStrategy.BFS:
            return self._worklist.pop(0)  # Queue: FIFO
        elif self.config.strategy == PathExplorationStrategy.RANDOM:
            idx = random.randint(0, len(self._worklist) - 1)
            return self._worklist.pop(idx)
        elif self.config.strategy == PathExplorationStrategy.COVERAGE_GUIDED:
            # Prefer states that cover new blocks
            best_idx = 0
            best_score = -1
            all_covered = self._result.covered_blocks
            for i, s in enumerate(self._worklist):
                new_blocks = len(s.covered_blocks - all_covered)
                if new_blocks > best_score:
                    best_score = new_blocks
                    best_idx = i
            return self._worklist.pop(best_idx)
        else:
            return self._worklist.pop()

    def _explore_state(self, state: SymbolicState, func: Function) -> None:
        """Explore a single state until termination or branching."""
        block_map = {b.name: b for b in func.blocks}
        step_count = 0

        while not state.is_terminated and step_count < self.config.max_depth:
            block_name = state.current_block
            if block_name is None:
                state.terminate("no_current_block")
                break

            block = block_map.get(block_name)
            if block is None:
                state.terminate(f"unknown_block:{block_name}")
                break

            # Check loop bound
            visit_count = state.visit_block(block_name)
            if visit_count > self.config.loop_unroll_bound:
                state.terminate("loop_bound_exceeded")
                self._result.completed_paths.append(state)
                return

            # Execute all instructions in the block
            instructions = list(block.instructions)
            branched = False

            for i, inst in enumerate(instructions):
                state.current_inst_index = i
                self._result.total_instructions += 1

                if isinstance(inst, BranchInst):
                    branched = self._execute_branch(state, inst)
                    break
                elif isinstance(inst, ReturnInst):
                    self._execute_return(state, inst)
                    self._result.completed_paths.append(state)
                    return
                else:
                    error = self._execute_instruction(state, inst)
                    if error:
                        state.terminate(f"error:{error}")
                        self._result.error_paths.append(state)
                        return

                step_count += 1

            if not branched and not state.is_terminated:
                # Fell through without a terminator
                state.terminate("no_terminator")
                self._result.completed_paths.append(state)
                return

            if branched:
                return  # States were added to worklist

    def _execute_instruction(self, state: SymbolicState, inst: Instruction) -> Optional[str]:
        """Execute a single instruction. Returns error string or None."""
        if isinstance(inst, BinaryOp):
            return self._execute_binop(state, inst)
        elif isinstance(inst, UnaryOp):
            return self._execute_unaryop(state, inst)
        elif isinstance(inst, CompareOp):
            return self._execute_compare(state, inst)
        elif isinstance(inst, CastInst):
            return self._execute_cast(state, inst)
        elif isinstance(inst, LoadInst):
            return self._execute_load(state, inst)
        elif isinstance(inst, StoreInst):
            return self._execute_store(state, inst)
        elif isinstance(inst, AllocaInst):
            return self._execute_alloca(state, inst)
        elif isinstance(inst, GetElementPtrInst):
            return self._execute_gep(state, inst)
        elif isinstance(inst, CallInst):
            return self._execute_call(state, inst)
        elif isinstance(inst, PhiInst):
            return self._execute_phi(state, inst)
        elif isinstance(inst, SelectInst):
            return self._execute_select(state, inst)
        else:
            # Unknown instruction: create unconstrained result
            if inst.name and inst.type:
                sort = _ir_type_to_z3_sort(inst.type, self.config.pointer_width)
                sym = state.fresh_bv(f"unknown_{inst.name}", sort.size() if z3.is_bv_sort(sort) else 32)
                state.set_var(inst.name, SymbolicValue(name=inst.name, z3_expr=sym, ir_type=inst.type))
            return None

    # -- Binary operations --

    def _execute_binop(self, state: SymbolicState, inst: BinaryOp) -> Optional[str]:
        """Execute a binary operation symbolically."""
        lhs = self._resolve_value(state, inst.lhs)
        rhs = self._resolve_value(state, inst.rhs)

        if lhs is None or rhs is None:
            # Create fresh unconstrained result
            if inst.name:
                width = inst.type.width if isinstance(inst.type, IntType) else 32
                sym = state.fresh_bv(inst.name, width)
                state.set_var(inst.name, SymbolicValue(name=inst.name, z3_expr=sym, ir_type=inst.type))
            return None

        lhs_expr = lhs.z3_expr
        rhs_expr = rhs.z3_expr

        # Ensure compatible sorts
        lhs_expr, rhs_expr = self._coerce_bv_sorts(lhs_expr, rhs_expr)

        op = inst.op.name
        result: Optional[z3.ExprRef] = None

        if op == "ADD":
            result = lhs_expr + rhs_expr
        elif op == "SUB":
            result = lhs_expr - rhs_expr
        elif op == "MUL":
            result = lhs_expr * rhs_expr
        elif op == "SDIV":
            if self.config.check_overflows:
                # Check division by zero
                is_zero = rhs_expr == z3.BitVecVal(0, rhs_expr.size())
                if self._is_feasible(state, is_zero):
                    pass  # Continue; division by zero is handled at a higher level
            result = lhs_expr / rhs_expr  # Z3 signed div
        elif op == "UDIV":
            result = z3.UDiv(lhs_expr, rhs_expr)
        elif op == "SREM":
            result = z3.SRem(lhs_expr, rhs_expr)
        elif op == "UREM":
            result = z3.URem(lhs_expr, rhs_expr)
        elif op == "SHL":
            result = lhs_expr << rhs_expr
        elif op == "LSHR":
            result = z3.LShR(lhs_expr, rhs_expr)
        elif op == "ASHR":
            result = lhs_expr >> rhs_expr
        elif op == "AND":
            result = lhs_expr & rhs_expr
        elif op == "OR":
            result = lhs_expr | rhs_expr
        elif op == "XOR":
            result = lhs_expr ^ rhs_expr
        elif op == "FADD":
            result = self._fp_binop(lhs_expr, rhs_expr, "add", inst.type)
        elif op == "FSUB":
            result = self._fp_binop(lhs_expr, rhs_expr, "sub", inst.type)
        elif op == "FMUL":
            result = self._fp_binop(lhs_expr, rhs_expr, "mul", inst.type)
        elif op == "FDIV":
            result = self._fp_binop(lhs_expr, rhs_expr, "div", inst.type)
        elif op == "FREM":
            result = self._fp_binop(lhs_expr, rhs_expr, "rem", inst.type)
        else:
            # Unknown op: unconstrained
            if inst.name and isinstance(inst.type, IntType):
                result = state.fresh_bv(inst.name, inst.type.width)
            else:
                result = state.fresh_bv(inst.name or "unknown", 32)

        if result is not None and inst.name:
            state.set_var(inst.name, SymbolicValue(name=inst.name, z3_expr=result, ir_type=inst.type))

        return None

    def _fp_binop(
        self, lhs: z3.ExprRef, rhs: z3.ExprRef, op: str, result_type: Optional[IRType],
    ) -> z3.ExprRef:
        """Execute a floating point binary operation."""
        # Coerce to FP if needed
        lhs_fp = self._to_fp(lhs, result_type)
        rhs_fp = self._to_fp(rhs, result_type)
        rm = z3.RNE()  # Round to nearest, ties to even

        if op == "add":
            return z3.fpAdd(rm, lhs_fp, rhs_fp)
        elif op == "sub":
            return z3.fpSub(rm, lhs_fp, rhs_fp)
        elif op == "mul":
            return z3.fpMul(rm, lhs_fp, rhs_fp)
        elif op == "div":
            return z3.fpDiv(rm, lhs_fp, rhs_fp)
        elif op == "rem":
            return z3.fpRem(lhs_fp, rhs_fp)
        else:
            return lhs_fp

    def _to_fp(self, expr: z3.ExprRef, ty: Optional[IRType]) -> z3.FPRef:
        """Convert an expression to floating point if needed."""
        if z3.is_fp(expr):
            return expr
        sort = z3.FPSort(11, 53)
        if isinstance(ty, FloatType) and ty.kind == FloatKind.F32:
            sort = z3.FPSort(8, 24)
        if z3.is_bv(expr):
            return z3.fpBVToFP(expr, sort)
        return z3.FPVal(0.0, sort)

    # -- Unary operations --

    def _execute_unaryop(self, state: SymbolicState, inst: UnaryOp) -> Optional[str]:
        operand = self._resolve_value(state, inst._operands[0])
        if operand is None:
            if inst.name and inst.type:
                width = inst.type.width if isinstance(inst.type, IntType) else 32
                sym = state.fresh_bv(inst.name, width)
                state.set_var(inst.name, SymbolicValue(name=inst.name, z3_expr=sym, ir_type=inst.type))
            return None

        expr = operand.z3_expr
        op = inst.op.name

        if op == "NEG":
            result = -expr
        elif op == "NOT" or op == "BITWISE_NOT":
            result = ~expr if z3.is_bv(expr) else z3.Not(expr)
        elif op == "FNEG":
            result = z3.fpNeg(expr) if z3.is_fp(expr) else -expr
        else:
            result = state.fresh_bv(inst.name or "unknown", 32)

        if inst.name:
            state.set_var(inst.name, SymbolicValue(name=inst.name, z3_expr=result, ir_type=inst.type))
        return None

    # -- Comparison --

    def _execute_compare(self, state: SymbolicState, inst: CompareOp) -> Optional[str]:
        lhs = self._resolve_value(state, inst.lhs)
        rhs = self._resolve_value(state, inst.rhs)

        if lhs is None or rhs is None:
            if inst.name:
                sym = state.fresh_bool(inst.name)
                state.set_var(inst.name, SymbolicValue(
                    name=inst.name, z3_expr=sym, ir_type=inst.type))
            return None

        lhs_expr = lhs.z3_expr
        rhs_expr = rhs.z3_expr

        # Coerce sorts
        if z3.is_bv(lhs_expr) and z3.is_bv(rhs_expr):
            lhs_expr, rhs_expr = self._coerce_bv_sorts(lhs_expr, rhs_expr)

        pred = inst.predicate.name
        result: z3.BoolRef

        if pred == "EQ" or pred == "OEQ" or pred == "UEQ":
            result = lhs_expr == rhs_expr
        elif pred == "NE" or pred == "ONE" or pred == "UNE":
            result = lhs_expr != rhs_expr
        elif pred == "SLT" or pred == "OLT":
            result = lhs_expr < rhs_expr if z3.is_bv(lhs_expr) else z3.fpLT(lhs_expr, rhs_expr)
        elif pred == "SLE" or pred == "OLE":
            result = lhs_expr <= rhs_expr if z3.is_bv(lhs_expr) else z3.fpLEQ(lhs_expr, rhs_expr)
        elif pred == "SGT" or pred == "OGT":
            result = lhs_expr > rhs_expr if z3.is_bv(lhs_expr) else z3.fpGT(lhs_expr, rhs_expr)
        elif pred == "SGE" or pred == "OGE":
            result = lhs_expr >= rhs_expr if z3.is_bv(lhs_expr) else z3.fpGEQ(lhs_expr, rhs_expr)
        elif pred == "ULT":
            result = z3.ULT(lhs_expr, rhs_expr)
        elif pred == "ULE":
            result = z3.ULE(lhs_expr, rhs_expr)
        elif pred == "UGT":
            result = z3.UGT(lhs_expr, rhs_expr)
        elif pred == "UGE":
            result = z3.UGE(lhs_expr, rhs_expr)
        elif pred == "ORD":
            result = z3.And(z3.Not(z3.fpIsNaN(lhs_expr)), z3.Not(z3.fpIsNaN(rhs_expr)))
        elif pred == "UNO":
            result = z3.Or(z3.fpIsNaN(lhs_expr), z3.fpIsNaN(rhs_expr))
        else:
            result = state.fresh_bool(inst.name or "cmp")

        if inst.name:
            state.set_var(inst.name, SymbolicValue(
                name=inst.name, z3_expr=result, ir_type=inst.type))
        return None

    # -- Cast --

    def _execute_cast(self, state: SymbolicState, inst: CastInst) -> Optional[str]:
        operand = self._resolve_value(state, inst._operands[0])
        if operand is None:
            if inst.name:
                sort = _ir_type_to_z3_sort(inst.type, self.config.pointer_width)
                if z3.is_bv_sort(sort):
                    sym = state.fresh_bv(inst.name, sort.size())
                else:
                    sym = state.fresh_bv(inst.name, 32)
                state.set_var(inst.name, SymbolicValue(name=inst.name, z3_expr=sym, ir_type=inst.type))
            return None

        expr = operand.z3_expr
        src_type = operand.ir_type or (inst._operands[0].type if hasattr(inst._operands[0], 'type') else None)
        dst_type = inst.type
        kind = inst.cast_kind.name

        result: z3.ExprRef

        if kind == "TRUNC":
            dst_width = dst_type.width if isinstance(dst_type, IntType) else 32
            if z3.is_bv(expr) and expr.size() > dst_width:
                result = z3.Extract(dst_width - 1, 0, expr)
            else:
                result = expr
        elif kind == "ZEXT":
            dst_width = dst_type.width if isinstance(dst_type, IntType) else 64
            if z3.is_bv(expr):
                ext_bits = dst_width - expr.size()
                result = z3.ZeroExt(max(ext_bits, 0), expr) if ext_bits > 0 else expr
            else:
                result = expr
        elif kind == "SEXT":
            dst_width = dst_type.width if isinstance(dst_type, IntType) else 64
            if z3.is_bv(expr):
                ext_bits = dst_width - expr.size()
                result = z3.SignExt(max(ext_bits, 0), expr) if ext_bits > 0 else expr
            else:
                result = expr
        elif kind == "FPTRUNC":
            dst_sort = z3.FPSort(8, 24)  # f64→f32
            if z3.is_fp(expr):
                result = z3.fpFPToFP(z3.RNE(), expr, dst_sort)
            else:
                result = z3.FPVal(0.0, dst_sort)
        elif kind == "FPEXT":
            dst_sort = z3.FPSort(11, 53)  # f32→f64
            if z3.is_fp(expr):
                result = z3.fpFPToFP(z3.RNE(), expr, dst_sort)
            else:
                result = z3.FPVal(0.0, dst_sort)
        elif kind == "FPTOSI":
            dst_width = dst_type.width if isinstance(dst_type, IntType) else 32
            if z3.is_fp(expr):
                result = z3.fpToSBV(z3.RTZ(), expr, z3.BitVecSort(dst_width))
            else:
                result = z3.BitVecVal(0, dst_width)
        elif kind == "FPTOUI":
            dst_width = dst_type.width if isinstance(dst_type, IntType) else 32
            if z3.is_fp(expr):
                result = z3.fpToUBV(z3.RTZ(), expr, z3.BitVecSort(dst_width))
            else:
                result = z3.BitVecVal(0, dst_width)
        elif kind == "SITOFP":
            dst_sort = z3.FPSort(11, 53)
            if isinstance(dst_type, FloatType) and dst_type.kind == FloatKind.F32:
                dst_sort = z3.FPSort(8, 24)
            if z3.is_bv(expr):
                result = z3.fpSignedToFP(z3.RNE(), expr, dst_sort)
            else:
                result = z3.FPVal(0.0, dst_sort)
        elif kind == "UITOFP":
            dst_sort = z3.FPSort(11, 53)
            if isinstance(dst_type, FloatType) and dst_type.kind == FloatKind.F32:
                dst_sort = z3.FPSort(8, 24)
            if z3.is_bv(expr):
                result = z3.fpToFP(z3.RNE(), expr, dst_sort)
            else:
                result = z3.FPVal(0.0, dst_sort)
        elif kind == "BITCAST":
            result = expr  # Bit-level reinterpretation
        elif kind == "PTRTOINT":
            dst_width = dst_type.width if isinstance(dst_type, IntType) else self.config.pointer_width
            if z3.is_bv(expr):
                if expr.size() > dst_width:
                    result = z3.Extract(dst_width - 1, 0, expr)
                elif expr.size() < dst_width:
                    result = z3.ZeroExt(dst_width - expr.size(), expr)
                else:
                    result = expr
            else:
                result = z3.BitVecVal(0, dst_width)
        elif kind == "INTTOPTR":
            ptr_width = self.config.pointer_width
            if z3.is_bv(expr):
                if expr.size() > ptr_width:
                    result = z3.Extract(ptr_width - 1, 0, expr)
                elif expr.size() < ptr_width:
                    result = z3.ZeroExt(ptr_width - expr.size(), expr)
                else:
                    result = expr
            else:
                result = z3.BitVecVal(0, ptr_width)
        else:
            result = expr

        if inst.name:
            state.set_var(inst.name, SymbolicValue(
                name=inst.name, z3_expr=result, ir_type=inst.type))
        return None

    # -- Memory operations --

    def _execute_load(self, state: SymbolicState, inst: LoadInst) -> Optional[str]:
        addr = self._resolve_value(state, inst.address)

        if addr is None:
            if inst.name:
                sort = _ir_type_to_z3_sort(inst.type, self.config.pointer_width)
                if z3.is_bv_sort(sort):
                    sym = state.fresh_bv(inst.name, sort.size())
                else:
                    sym = state.fresh_bv(inst.name, 32)
                state.set_var(inst.name, SymbolicValue(
                    name=inst.name, z3_expr=sym, ir_type=inst.type))
            return None

        addr_expr = addr.z3_expr
        if not z3.is_bv(addr_expr):
            addr_expr = z3.BitVecVal(0, self.config.pointer_width)

        # Check null pointer if enabled
        if self.config.check_memory_safety:
            is_null = addr_expr == z3.BitVecVal(0, addr_expr.size())
            if self._is_feasible(state, is_null):
                # Null dereference possible but don't terminate; add constraint
                pass

        num_bytes = inst.type.size_bits() // 8 if inst.type and inst.type.is_sized() else 4
        loaded = state.memory.load(addr_expr, max(num_bytes, 1))

        if inst.name:
            state.set_var(inst.name, SymbolicValue(
                name=inst.name, z3_expr=loaded, ir_type=inst.type))
        return None

    def _execute_store(self, state: SymbolicState, inst: StoreInst) -> Optional[str]:
        val = self._resolve_value(state, inst.value)
        addr = self._resolve_value(state, inst.address)

        if val is None or addr is None:
            return None

        addr_expr = addr.z3_expr
        val_expr = val.z3_expr

        if not z3.is_bv(addr_expr):
            addr_expr = z3.BitVecVal(0, self.config.pointer_width)
        if not z3.is_bv(val_expr):
            if z3.is_bool(val_expr):
                val_expr = z3.If(val_expr, z3.BitVecVal(1, 8), z3.BitVecVal(0, 8))
            else:
                return None

        num_bytes = val_expr.size() // 8 if z3.is_bv(val_expr) else 4
        state.memory.store(addr_expr, val_expr, max(num_bytes, 1))
        return None

    def _execute_alloca(self, state: SymbolicState, inst: AllocaInst) -> Optional[str]:
        alloc_type = inst.type
        if isinstance(alloc_type, PointerType):
            alloc_type = alloc_type.pointee
        size = alloc_type.size_bits() // 8 if alloc_type and alloc_type.is_sized() else 8

        alloc_id, base = state.memory.allocate(size, name=inst.name or "alloca")

        if inst.name:
            state.set_var(inst.name, SymbolicValue(
                name=inst.name, z3_expr=base, ir_type=inst.type))
        return None

    def _execute_gep(self, state: SymbolicState, inst: GetElementPtrInst) -> Optional[str]:
        base = self._resolve_value(state, inst._operands[0])
        if base is None:
            if inst.name:
                sym = state.fresh_bv(inst.name, self.config.pointer_width)
                state.set_var(inst.name, SymbolicValue(
                    name=inst.name, z3_expr=sym, ir_type=inst.type))
            return None

        base_expr = base.z3_expr
        if not z3.is_bv(base_expr):
            base_expr = z3.BitVecVal(0, self.config.pointer_width)

        # Compute offset from indices
        offset = z3.BitVecVal(0, self.config.pointer_width)
        for i, idx_val in enumerate(inst._operands[1:]):
            idx = self._resolve_value(state, idx_val)
            if idx is not None:
                idx_expr = idx.z3_expr
                if z3.is_bv(idx_expr):
                    if idx_expr.size() < self.config.pointer_width:
                        idx_expr = z3.SignExt(self.config.pointer_width - idx_expr.size(), idx_expr)
                    elif idx_expr.size() > self.config.pointer_width:
                        idx_expr = z3.Extract(self.config.pointer_width - 1, 0, idx_expr)
                    # Assume element size of 1 byte for simplicity (would need type info)
                    element_size = z3.BitVecVal(1, self.config.pointer_width)
                    offset = offset + idx_expr * element_size

        result = base_expr + offset
        if inst.name:
            state.set_var(inst.name, SymbolicValue(
                name=inst.name, z3_expr=result, ir_type=inst.type))
        return None

    # -- Call --

    def _execute_call(self, state: SymbolicState, inst: CallInst) -> Optional[str]:
        callee_name = inst.callee_name

        # Check for function summary
        if callee_name in self.function_summaries:
            summary_fn = self.function_summaries[callee_name]
            args = [self._resolve_value(state, a) for a in inst._operands[1:]]
            try:
                result = summary_fn(state, args)
                if inst.name and result is not None:
                    state.set_var(inst.name, result)
            except Exception:
                if inst.name:
                    sort = _ir_type_to_z3_sort(inst.type, self.config.pointer_width)
                    sym = state.fresh_bv(inst.name, sort.size() if z3.is_bv_sort(sort) else 32)
                    state.set_var(inst.name, SymbolicValue(
                        name=inst.name, z3_expr=sym, ir_type=inst.type))
            return None

        # Check for inlining
        if self.config.inline_calls and callee_name in self._functions:
            if state.call_depth < self.config.max_call_depth:
                callee = self._functions[callee_name]
                return self._inline_call(state, inst, callee)

        # Default: create unconstrained return value
        if inst.name and inst.type and not isinstance(inst.type, VoidType):
            sort = _ir_type_to_z3_sort(inst.type, self.config.pointer_width)
            if z3.is_bv_sort(sort):
                sym = state.fresh_bv(f"call_{callee_name}", sort.size())
            elif isinstance(sort, z3.FPSortRef):
                sym = state.fresh_fp(f"call_{callee_name}", sort)
            else:
                sym = state.fresh_bv(f"call_{callee_name}", 32)
            state.set_var(inst.name, SymbolicValue(
                name=inst.name, z3_expr=sym, ir_type=inst.type))

        return None

    def _inline_call(
        self, state: SymbolicState, inst: CallInst, callee: Function,
    ) -> Optional[str]:
        """Inline a function call."""
        state.push_frame(
            callee.name,
            return_block=state.current_block,
            return_var=inst.name,
        )

        # Bind arguments
        callee_args = list(callee.arguments)
        call_args = list(inst._operands[1:])
        for i, carg in enumerate(callee_args):
            if i < len(call_args):
                val = self._resolve_value(state, call_args[i])
                if val is not None:
                    state.set_var(
                        carg.name or f"arg_{i}",
                        SymbolicValue(name=carg.name or f"arg_{i}",
                                      z3_expr=val.z3_expr, ir_type=carg.type),
                    )

        # Execute callee (recursive execution via sub-executor)
        sub_executor = SymbolicExecutor(
            config=ExecutionConfig(
                max_paths=min(self.config.max_paths // 2, 100),
                max_depth=self.config.max_depth,
                loop_unroll_bound=self.config.loop_unroll_bound,
                max_call_depth=self.config.max_call_depth,
                solver_timeout_ms=self.config.solver_timeout_ms,
                strategy=self.config.strategy,
                pointer_width=self.config.pointer_width,
            ),
            function_summaries=self.function_summaries,
        )
        sub_result = sub_executor.execute_function(callee, self._functions)

        # Use first completed path's return value
        state.pop_frame()

        if sub_result.completed_paths:
            ret_state = sub_result.completed_paths[0]
            if ret_state.return_value and inst.name:
                state.set_var(inst.name, ret_state.return_value)
        elif inst.name and inst.type:
            sort = _ir_type_to_z3_sort(inst.type, self.config.pointer_width)
            sym = state.fresh_bv(f"call_{callee.name}", sort.size() if z3.is_bv_sort(sort) else 32)
            state.set_var(inst.name, SymbolicValue(
                name=inst.name, z3_expr=sym, ir_type=inst.type))

        return None

    # -- Phi --

    def _execute_phi(self, state: SymbolicState, inst: PhiInst) -> Optional[str]:
        prev_block = state.current_block

        # Find the incoming value from the predecessor block
        for val, block in inst.incoming:
            if block.name == prev_block:
                resolved = self._resolve_value(state, val)
                if resolved is not None and inst.name:
                    state.set_var(inst.name, SymbolicValue(
                        name=inst.name, z3_expr=resolved.z3_expr, ir_type=inst.type))
                return None

        # No matching predecessor found; use first incoming or fresh
        if inst.incoming:
            val, _ = inst.incoming[0]
            resolved = self._resolve_value(state, val)
            if resolved is not None and inst.name:
                state.set_var(inst.name, SymbolicValue(
                    name=inst.name, z3_expr=resolved.z3_expr, ir_type=inst.type))
        elif inst.name:
            sort = _ir_type_to_z3_sort(inst.type, self.config.pointer_width)
            sym = state.fresh_bv(inst.name, sort.size() if z3.is_bv_sort(sort) else 32)
            state.set_var(inst.name, SymbolicValue(name=inst.name, z3_expr=sym, ir_type=inst.type))

        return None

    # -- Select --

    def _execute_select(self, state: SymbolicState, inst: SelectInst) -> Optional[str]:
        cond = self._resolve_value(state, inst._operands[0])
        true_val = self._resolve_value(state, inst._operands[1])
        false_val = self._resolve_value(state, inst._operands[2])

        if cond is None or true_val is None or false_val is None:
            if inst.name:
                sort = _ir_type_to_z3_sort(inst.type, self.config.pointer_width)
                sym = state.fresh_bv(inst.name, sort.size() if z3.is_bv_sort(sort) else 32)
                state.set_var(inst.name, SymbolicValue(
                    name=inst.name, z3_expr=sym, ir_type=inst.type))
            return None

        cond_expr = cond.z3_expr
        if z3.is_bv(cond_expr):
            cond_expr = cond_expr != z3.BitVecVal(0, cond_expr.size())

        t_expr, f_expr = true_val.z3_expr, false_val.z3_expr
        if z3.is_bv(t_expr) and z3.is_bv(f_expr):
            t_expr, f_expr = self._coerce_bv_sorts(t_expr, f_expr)

        result = z3.If(cond_expr, t_expr, f_expr)
        if inst.name:
            state.set_var(inst.name, SymbolicValue(
                name=inst.name, z3_expr=result, ir_type=inst.type))
        return None

    # -- Branch --

    def _execute_branch(self, state: SymbolicState, inst: BranchInst) -> bool:
        """Execute a branch. Returns True if states were forked."""
        if not inst.is_conditional:
            # Unconditional branch
            state.current_block = inst._true_target.name
            self._worklist.append(state)
            return True

        cond = self._resolve_value(state, inst._operands[0])

        true_target = inst._true_target.name
        false_target = inst._false_target.name if inst._false_target else None

        if cond is None:
            # Can't resolve condition; explore both paths
            if false_target:
                true_state = state.fork()
                true_state.current_block = true_target
                true_state.add_constraint(
                    state.fresh_bool("unknown_cond"),
                    is_branch_taken=True,
                    source_block=state.current_block,
                )
                self._worklist.append(true_state)

                false_state = state.fork()
                false_state.current_block = false_target
                false_state.add_constraint(
                    state.fresh_bool("unknown_cond_neg"),
                    is_branch_taken=False,
                    source_block=state.current_block,
                )
                self._worklist.append(false_state)
            else:
                state.current_block = true_target
                self._worklist.append(state)
            return True

        cond_expr = cond.z3_expr
        if z3.is_bv(cond_expr):
            cond_expr = cond_expr != z3.BitVecVal(0, cond_expr.size())

        if false_target is None:
            state.current_block = true_target
            self._worklist.append(state)
            return True

        # Fork: try both branches
        true_feasible = self._is_feasible(state, cond_expr)
        false_feasible = self._is_feasible(state, z3.Not(cond_expr))

        if true_feasible and false_feasible:
            # Both feasible: fork
            true_state = state.fork()
            true_state.add_constraint(cond_expr, True, state.current_block, "branch_true")
            true_state.current_block = true_target
            self._worklist.append(true_state)

            false_state = state.fork()
            false_state.add_constraint(z3.Not(cond_expr), False, state.current_block, "branch_false")
            false_state.current_block = false_target
            self._worklist.append(false_state)
        elif true_feasible:
            state.add_constraint(cond_expr, True, state.current_block, "branch_true_only")
            state.current_block = true_target
            self._worklist.append(state)
        elif false_feasible:
            state.add_constraint(z3.Not(cond_expr), False, state.current_block, "branch_false_only")
            state.current_block = false_target
            self._worklist.append(state)
        else:
            # Both infeasible (shouldn't happen normally)
            state.terminate("infeasible_branch")
            self._result.infeasible_paths += 1

        return True

    # -- Return --

    def _execute_return(self, state: SymbolicState, inst: ReturnInst) -> None:
        if inst.return_value is not None:
            val = self._resolve_value(state, inst.return_value)
            if val is not None:
                state.return_value = val
        state.terminate("return")

    # -- Helpers --

    def _resolve_value(self, state: SymbolicState, val: Value) -> Optional[SymbolicValue]:
        """Resolve an IR Value to a SymbolicValue."""
        if isinstance(val, Constant):
            expr = _make_z3_constant(val.value, val.type, self.config.pointer_width)
            return SymbolicValue(name=val.name or "const", z3_expr=expr, ir_type=val.type)

        name = val.name
        if name:
            sv = state.get_var(name)
            if sv is not None:
                return sv

        # Try by display name
        dn = val.display_name
        if dn:
            sv = state.get_var(dn)
            if sv is not None:
                return sv

        return None

    def _coerce_bv_sorts(
        self, a: z3.BitVecRef, b: z3.BitVecRef,
    ) -> Tuple[z3.BitVecRef, z3.BitVecRef]:
        """Coerce two bitvectors to the same width."""
        if a.size() == b.size():
            return a, b
        if a.size() < b.size():
            a = z3.SignExt(b.size() - a.size(), a)
        else:
            b = z3.SignExt(a.size() - b.size(), b)
        return a, b

    def _is_feasible(self, state: SymbolicState, cond: z3.BoolRef) -> bool:
        """Check if a condition is feasible under current path constraints."""
        self._solver.push()
        for pc in state.path_constraints:
            self._solver.add(pc.condition)
        self._solver.add(cond)
        result = self._solver.check()
        self._solver.pop()

        if result == z3.sat:
            return True
        elif result == z3.unsat:
            return False
        else:
            # Timeout: assume feasible
            self._result.timeout_paths += 1
            return True
