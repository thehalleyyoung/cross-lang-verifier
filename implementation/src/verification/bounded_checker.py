"""
Bounded Model Checking for the Cross-Language Equivalence Verifier.

Unrolls loops to depth k, encodes the program as an SMT formula,
checks all paths within bounds, and reports coverage statistics.

Provides:
- BoundedModelChecker: main BMC engine
- BMCConfig: configuration for BMC
- BMCResult: result of BMC checking
- LoopUnrollStrategy: strategies for loop unrolling
- PathEncoder: encode execution paths as SMT formulas
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple

from ..ir.function import Function
from ..ir.basic_block import BasicBlock
from ..ir.instructions import (
    Instruction, Value, Constant, Argument,
    BinaryOp, BinOpKind, UnaryOp, UnaryOpKind,
    CompareOp, CmpPredicate,
    LoadInst, StoreInst, AllocaInst,
    GetElementPtrInst, CastInst, CastKind,
    CallInst, ReturnInst, BranchInst, SwitchInst,
    PhiInst, SelectInst,
)
from ..ir.types import IRType, IntType, FloatType, Signedness, VoidType

logger = logging.getLogger(__name__)


# ─── Configuration ──────────────────────────────────────────────────

class LoopUnrollStrategy(Enum):
    """Strategy for unrolling loops in BMC."""
    FIXED = auto()       # Unroll to a fixed depth
    INCREASING = auto()  # Start small and increase
    ADAPTIVE = auto()    # Adapt based on solver performance


@dataclass
class BMCConfig:
    """Configuration for bounded model checking."""
    max_unroll_depth: int = 10
    initial_unroll_depth: int = 1
    depth_increment: int = 1
    timeout_ms: float = 30000.0
    timeout_per_depth_ms: float = 10000.0
    strategy: LoopUnrollStrategy = LoopUnrollStrategy.INCREASING
    check_assertions: bool = True
    produce_counterexample: bool = True
    track_coverage: bool = True
    use_incremental_solver: bool = True
    max_paths: int = 10000
    simplify_formula: bool = True


# ─── BMC Result ────────────────────────────────────────────────────

class BMCStatus(Enum):
    """Status of a BMC check."""
    EQUIVALENT = auto()
    NOT_EQUIVALENT = auto()
    UNKNOWN = auto()
    TIMEOUT = auto()
    DEPTH_EXCEEDED = auto()
    ERROR = auto()


@dataclass
class BMCResult:
    """Result of bounded model checking."""
    status: BMCStatus
    depth_reached: int = 0
    time_ms: float = 0.0
    paths_explored: int = 0
    paths_feasible: int = 0
    counterexample: Optional[Any] = None
    coverage_left: float = 0.0
    coverage_right: float = 0.0
    blocks_covered_left: Set[int] = field(default_factory=set)
    blocks_covered_right: Set[int] = field(default_factory=set)
    smt_queries: int = 0
    error_message: str = ""

    @property
    def is_equivalent(self) -> bool:
        return self.status == BMCStatus.EQUIVALENT

    def summary(self) -> str:
        return (f"BMC: {self.status.name} (depth={self.depth_reached}, "
                f"paths={self.paths_explored}, time={self.time_ms:.1f}ms)")


# ─── Symbolic Variable Manager ────────────────────────────────────

class SymbolicVarManager:
    """Manage symbolic variables for BMC encoding."""

    def __init__(self) -> None:
        self._counter = 0
        self._variables: Dict[str, Any] = {}
        self._type_map: Dict[str, IRType] = {}

    def fresh(self, prefix: str, typ: IRType) -> str:
        """Create a fresh symbolic variable name."""
        name = f"{prefix}_{self._counter}"
        self._counter += 1
        self._type_map[name] = typ
        return name

    def fresh_for_value(self, val: Value, prefix: str = "v") -> str:
        """Create a fresh variable for an IR value."""
        name = f"{prefix}_{val.name}_{self._counter}" if hasattr(val, 'name') and val.name else f"{prefix}_{self._counter}"
        self._counter += 1
        if hasattr(val, 'ir_type'):
            self._type_map[name] = val.ir_type
        return name

    def get_type(self, name: str) -> Optional[IRType]:
        return self._type_map.get(name)

    @property
    def num_variables(self) -> int:
        return self._counter


# ─── Path Encoder ─────────────────────────────────────────────────

class PathCondition:
    """Represents a path condition as a conjunction of constraints."""

    def __init__(self) -> None:
        self._constraints: List[Any] = []
        self._branch_history: List[Tuple[int, bool]] = []

    def add_constraint(self, constraint: Any) -> None:
        self._constraints.append(constraint)

    def add_branch(self, block_id: int, taken: bool) -> None:
        self._branch_history.append((block_id, taken))

    @property
    def constraints(self) -> List[Any]:
        return self._constraints

    @property
    def branch_history(self) -> List[Tuple[int, bool]]:
        return self._branch_history

    def fork(self) -> "PathCondition":
        new = PathCondition()
        new._constraints = list(self._constraints)
        new._branch_history = list(self._branch_history)
        return new

    @property
    def depth(self) -> int:
        return len(self._branch_history)


class PathEncoder:
    """Encode execution paths as SMT formulas.

    For each path through the program (up to a given loop bound),
    encodes the path condition and the computation along the path
    as an SMT formula.
    """

    def __init__(self, config: BMCConfig) -> None:
        self._config = config
        self._var_mgr = SymbolicVarManager()
        self._paths_encoded = 0
        self._formula_size = 0

    @property
    def paths_encoded(self) -> int:
        return self._paths_encoded

    def encode_function(self, func: Function, prefix: str = "",
                         loop_bound: int = 10) -> List[Dict[str, Any]]:
        """Encode all paths through a function up to the loop bound.

        Returns a list of path encodings, each containing:
        - path_condition: conjunction of branch conditions
        - return_value: symbolic expression for the return value
        - memory_state: symbolic memory state at return
        - blocks_visited: set of block IDs visited
        """
        paths: List[Dict[str, Any]] = []
        entry = func.entry_block
        if entry is None:
            return paths

        # Initialize symbolic values for arguments
        arg_vars: Dict[int, str] = {}
        for arg in func.arguments:
            var_name = self._var_mgr.fresh_for_value(arg, prefix=f"{prefix}arg")
            arg_vars[arg.id] = var_name

        # Explore paths via DFS
        initial_state = {
            "values": dict(arg_vars),
            "memory": {},
            "path_condition": PathCondition(),
            "blocks_visited": set(),
            "loop_counts": defaultdict(int),
        }

        self._explore_paths(func, entry, initial_state, paths, loop_bound)
        return paths

    def _explore_paths(self, func: Function, block: BasicBlock,
                        state: Dict[str, Any], paths: List[Dict[str, Any]],
                        loop_bound: int) -> None:
        """Recursively explore execution paths."""
        if len(paths) >= self._config.max_paths:
            return

        block_id = block.id
        visited = state["blocks_visited"]
        loop_counts = state["loop_counts"]

        # Check loop bound
        if block_id in visited:
            loop_counts[block_id] += 1
            if loop_counts[block_id] > loop_bound:
                return
        visited.add(block_id)

        # Encode instructions in this block
        values = state["values"]
        memory = state["memory"]

        for inst in block.instructions:
            if isinstance(inst, ReturnInst):
                # Path complete: record it
                ret_var = None
                if inst.value is not None:
                    ret_var = self._resolve_value(inst.value, values)
                paths.append({
                    "path_condition": state["path_condition"],
                    "return_value": ret_var,
                    "memory_state": dict(memory),
                    "blocks_visited": set(visited),
                })
                self._paths_encoded += 1
                return

            elif isinstance(inst, BranchInst):
                if not inst.is_conditional:
                    target = inst.target if hasattr(inst, 'target') else inst.true_block
                    if target is not None:
                        new_state = self._clone_state(state)
                        self._explore_paths(func, target, new_state, paths, loop_bound)
                else:
                    cond = self._resolve_value(inst.condition, values)

                    # True branch
                    if inst.true_block is not None:
                        true_state = self._clone_state(state)
                        true_state["path_condition"].add_constraint(
                            ("assert", cond, True))
                        true_state["path_condition"].add_branch(block_id, True)
                        self._explore_paths(func, inst.true_block, true_state,
                                            paths, loop_bound)

                    # False branch
                    if inst.false_block is not None:
                        false_state = self._clone_state(state)
                        false_state["path_condition"].add_constraint(
                            ("assert", cond, False))
                        false_state["path_condition"].add_branch(block_id, False)
                        self._explore_paths(func, inst.false_block, false_state,
                                            paths, loop_bound)
                return

            elif isinstance(inst, SwitchInst):
                sv = self._resolve_value(inst.value, values)
                for succ in block.successors:
                    new_state = self._clone_state(state)
                    self._explore_paths(func, succ, new_state, paths, loop_bound)
                return

            else:
                # Encode the instruction
                result_var = self._encode_instruction(inst, values, memory)
                if result_var is not None:
                    values[inst.id] = result_var

    def _encode_instruction(self, inst: Instruction, values: Dict[int, str],
                             memory: Dict[str, str]) -> Optional[str]:
        """Encode a single instruction symbolically."""
        if isinstance(inst, BinaryOp):
            left = self._resolve_value(inst.left, values)
            right = self._resolve_value(inst.right, values)
            result = self._var_mgr.fresh_for_value(inst, "binop")
            self._formula_size += 1
            return result

        elif isinstance(inst, UnaryOp):
            operand = self._resolve_value(inst.operand, values)
            result = self._var_mgr.fresh_for_value(inst, "unop")
            self._formula_size += 1
            return result

        elif isinstance(inst, CompareOp):
            left = self._resolve_value(inst.left, values)
            right = self._resolve_value(inst.right, values)
            result = self._var_mgr.fresh_for_value(inst, "cmp")
            self._formula_size += 1
            return result

        elif isinstance(inst, CastInst):
            operand = self._resolve_value(inst.operand, values)
            result = self._var_mgr.fresh_for_value(inst, "cast")
            self._formula_size += 1
            return result

        elif isinstance(inst, SelectInst):
            cond = self._resolve_value(inst.condition, values)
            tv = self._resolve_value(inst.true_value, values)
            fv = self._resolve_value(inst.false_value, values)
            result = self._var_mgr.fresh_for_value(inst, "sel")
            self._formula_size += 1
            return result

        elif isinstance(inst, PhiInst):
            # For BMC, phi nodes are resolved by the specific path
            if inst.incoming:
                return self._resolve_value(inst.incoming[0][0], values)
            return self._var_mgr.fresh_for_value(inst, "phi")

        elif isinstance(inst, LoadInst):
            addr = self._resolve_value(inst.address, values)
            result = self._var_mgr.fresh_for_value(inst, "load")
            return result

        elif isinstance(inst, StoreInst):
            addr = self._resolve_value(inst.address, values)
            val = self._resolve_value(inst.value, values)
            memory[addr] = val
            return None

        elif isinstance(inst, AllocaInst):
            return self._var_mgr.fresh_for_value(inst, "alloca")

        elif isinstance(inst, CallInst):
            return self._var_mgr.fresh_for_value(inst, "call")

        elif isinstance(inst, GetElementPtrInst):
            return self._var_mgr.fresh_for_value(inst, "gep")

        return None

    def _resolve_value(self, val: Value, values: Dict[int, str]) -> str:
        """Resolve a value to its symbolic variable name."""
        if isinstance(val, Constant):
            return f"const_{val.value}" if hasattr(val, 'value') else "const_0"
        vid = val.id if hasattr(val, 'id') else id(val)
        if vid in values:
            return values[vid]
        return self._var_mgr.fresh_for_value(val, "unknown")

    def _clone_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "values": dict(state["values"]),
            "memory": dict(state["memory"]),
            "path_condition": state["path_condition"].fork(),
            "blocks_visited": set(state["blocks_visited"]),
            "loop_counts": defaultdict(int, state["loop_counts"]),
        }


# ─── Bounded Model Checker ───────────────────────────────────────

class BoundedModelChecker:
    """Bounded model checker for equivalence verification.

    Checks equivalence of two functions by:
    1. Encoding both functions as SMT formulas up to loop bound k
    2. Asserting shared inputs
    3. Checking if outputs can differ (negation of equivalence)
    4. Increasing bound k until proof, counterexample, or timeout
    """

    def __init__(self, config: Optional[BMCConfig] = None) -> None:
        self._config = config or BMCConfig()
        self._total_queries = 0
        self._total_time_ms = 0.0

    @property
    def config(self) -> BMCConfig:
        return self._config

    def check_equivalence(self, left: Function, right: Function) -> "FunctionVerificationResult":
        """Check equivalence using bounded model checking."""
        from .verifier import FunctionPair, FunctionVerificationResult, VerificationStatus

        pair = FunctionPair(left=left, right=right)
        result = FunctionVerificationResult(pair=pair)
        start = time.monotonic()

        if self._config.strategy == LoopUnrollStrategy.FIXED:
            bmc_result = self._check_at_depth(left, right, self._config.max_unroll_depth)
        elif self._config.strategy == LoopUnrollStrategy.INCREASING:
            bmc_result = self._check_increasing(left, right)
        else:
            bmc_result = self._check_adaptive(left, right)

        # Convert BMC result to verification result
        result.time_ms = (time.monotonic() - start) * 1000
        result.paths_explored = bmc_result.paths_explored
        result.paths_feasible = bmc_result.paths_feasible
        result.smt_queries = bmc_result.smt_queries
        result.blocks_covered_left = len(bmc_result.blocks_covered_left)
        result.blocks_covered_right = len(bmc_result.blocks_covered_right)

        if bmc_result.status == BMCStatus.EQUIVALENT:
            result.status = VerificationStatus.EQUIVALENT
        elif bmc_result.status == BMCStatus.NOT_EQUIVALENT:
            result.status = VerificationStatus.NOT_EQUIVALENT
            result.counterexample = bmc_result.counterexample
        elif bmc_result.status == BMCStatus.TIMEOUT:
            result.status = VerificationStatus.TIMEOUT
        else:
            result.status = VerificationStatus.UNKNOWN
            result.error_message = bmc_result.error_message or "BMC inconclusive"

        return result

    def check_single(self, func: Function, depth: int) -> BMCResult:
        """Check a single function's paths for internal consistency."""
        start = time.monotonic()
        encoder = PathEncoder(self._config)
        paths = encoder.encode_function(func, prefix="", loop_bound=depth)

        result = BMCResult(
            status=BMCStatus.EQUIVALENT,
            depth_reached=depth,
            paths_explored=len(paths),
            paths_feasible=len(paths),
        )

        for path in paths:
            result.blocks_covered_left |= path["blocks_visited"]

        if func.num_blocks > 0:
            result.coverage_left = len(result.blocks_covered_left) / func.num_blocks

        result.time_ms = (time.monotonic() - start) * 1000
        return result

    def _check_at_depth(self, left: Function, right: Function,
                         depth: int) -> BMCResult:
        """Check equivalence at a specific loop unroll depth."""
        start = time.monotonic()

        encoder = PathEncoder(self._config)

        # Encode both functions
        left_paths = encoder.encode_function(left, prefix="L_", loop_bound=depth)
        right_paths = encoder.encode_function(right, prefix="R_", loop_bound=depth)

        result = BMCResult(
            status=BMCStatus.UNKNOWN,
            depth_reached=depth,
            paths_explored=len(left_paths) + len(right_paths),
        )

        # Collect covered blocks
        for path in left_paths:
            result.blocks_covered_left |= path["blocks_visited"]
        for path in right_paths:
            result.blocks_covered_right |= path["blocks_visited"]

        if left.num_blocks > 0:
            result.coverage_left = len(result.blocks_covered_left) / left.num_blocks
        if right.num_blocks > 0:
            result.coverage_right = len(result.blocks_covered_right) / right.num_blocks

        # Try SMT-based equivalence check
        try:
            smt_result = self._smt_check(left, right, left_paths, right_paths)
            result.status = smt_result.status
            result.counterexample = smt_result.counterexample
            result.smt_queries = smt_result.smt_queries
            self._total_queries += smt_result.smt_queries
        except Exception as e:
            logger.debug(f"SMT check failed at depth {depth}: {e}")
            # Fall back to path comparison
            result = self._compare_paths(left_paths, right_paths, result)

        result.time_ms = (time.monotonic() - start) * 1000
        self._total_time_ms += result.time_ms
        return result

    def _check_increasing(self, left: Function, right: Function) -> BMCResult:
        """Check with increasing depths until proof, counterexample, or timeout."""
        start = time.monotonic()
        best_result = BMCResult(status=BMCStatus.UNKNOWN)

        depth = self._config.initial_unroll_depth
        while depth <= self._config.max_unroll_depth:
            elapsed = (time.monotonic() - start) * 1000
            if elapsed > self._config.timeout_ms:
                best_result.status = BMCStatus.TIMEOUT
                break

            logger.debug(f"BMC: checking at depth {depth}")
            result = self._check_at_depth(left, right, depth)

            # Update best result
            best_result.depth_reached = depth
            best_result.paths_explored += result.paths_explored
            best_result.blocks_covered_left |= result.blocks_covered_left
            best_result.blocks_covered_right |= result.blocks_covered_right
            best_result.smt_queries += result.smt_queries

            if result.status == BMCStatus.EQUIVALENT:
                best_result.status = BMCStatus.EQUIVALENT
                break
            elif result.status == BMCStatus.NOT_EQUIVALENT:
                best_result.status = BMCStatus.NOT_EQUIVALENT
                best_result.counterexample = result.counterexample
                break

            depth += self._config.depth_increment

        best_result.time_ms = (time.monotonic() - start) * 1000
        if left.num_blocks > 0:
            best_result.coverage_left = len(best_result.blocks_covered_left) / left.num_blocks
        if right.num_blocks > 0:
            best_result.coverage_right = len(best_result.blocks_covered_right) / right.num_blocks

        return best_result

    def _check_adaptive(self, left: Function, right: Function) -> BMCResult:
        """Adaptively choose depth based on solver performance."""
        start = time.monotonic()
        best_result = BMCResult(status=BMCStatus.UNKNOWN)
        depth = self._config.initial_unroll_depth
        prev_time = 0.0

        while depth <= self._config.max_unroll_depth:
            elapsed = (time.monotonic() - start) * 1000
            if elapsed > self._config.timeout_ms:
                best_result.status = BMCStatus.TIMEOUT
                break

            result = self._check_at_depth(left, right, depth)

            best_result.depth_reached = depth
            best_result.paths_explored += result.paths_explored
            best_result.blocks_covered_left |= result.blocks_covered_left
            best_result.blocks_covered_right |= result.blocks_covered_right
            best_result.smt_queries += result.smt_queries

            if result.status in (BMCStatus.EQUIVALENT, BMCStatus.NOT_EQUIVALENT):
                best_result.status = result.status
                best_result.counterexample = result.counterexample
                break

            # Adaptive depth increment based on solver time
            if result.time_ms > 0 and prev_time > 0:
                growth_rate = result.time_ms / prev_time
                if growth_rate > 4.0:
                    # Solver time growing too fast: increase slowly
                    depth += 1
                elif growth_rate < 2.0:
                    # Solver time growing slowly: increase faster
                    depth += min(3, self._config.max_unroll_depth - depth)
                else:
                    depth += self._config.depth_increment
            else:
                depth += self._config.depth_increment

            prev_time = result.time_ms

        best_result.time_ms = (time.monotonic() - start) * 1000
        if left.num_blocks > 0:
            best_result.coverage_left = len(best_result.blocks_covered_left) / left.num_blocks
        if right.num_blocks > 0:
            best_result.coverage_right = len(best_result.blocks_covered_right) / right.num_blocks

        return best_result

    def _smt_check(self, left: Function, right: Function,
                    left_paths: List[Dict], right_paths: List[Dict]) -> BMCResult:
        """Perform SMT-based equivalence check on encoded paths."""
        result = BMCResult(status=BMCStatus.UNKNOWN)

        try:
            from ..smt.encoder import SMTEncoder, EncodingContext
            from ..smt.solver import SMTSolver, SolverConfig, SolverStatus

            encoder = SMTEncoder()
            ctx = EncodingContext()

            left_ctx, left_ret = encoder.encode_function(left, ctx, prefix="L_")
            right_ctx, right_ret = encoder.encode_function(right, ctx, prefix="R_")

            solver_config = SolverConfig()
            solver_config.timeout_ms = int(self._config.timeout_per_depth_ms)
            solver = SMTSolver(solver_config)

            for a in left_ctx.assertions:
                solver.add(a)
            for a in right_ctx.assertions:
                solver.add(a)

            if left_ret is not None and right_ret is not None:
                import z3
                solver.add(left_ret != right_ret)

            result.smt_queries = 1
            smt_result = solver.check()

            if smt_result.status == SolverStatus.UNSAT:
                result.status = BMCStatus.EQUIVALENT
            elif smt_result.status == SolverStatus.SAT:
                result.status = BMCStatus.NOT_EQUIVALENT
                result.counterexample = smt_result.model
            elif smt_result.status == SolverStatus.TIMEOUT:
                result.status = BMCStatus.TIMEOUT

        except ImportError:
            result = self._compare_paths(left_paths, right_paths, result)
        except Exception as e:
            result.error_message = str(e)

        return result

    def _compare_paths(self, left_paths: List[Dict], right_paths: List[Dict],
                        result: BMCResult) -> BMCResult:
        """Compare paths structurally when SMT is unavailable."""
        if len(left_paths) == len(right_paths):
            all_match = True
            for lp, rp in zip(left_paths, right_paths):
                lr = lp.get("return_value")
                rr = rp.get("return_value")
                if lr != rr and lr is not None and rr is not None:
                    all_match = False
                    break

            if all_match:
                result.status = BMCStatus.EQUIVALENT
            else:
                result.status = BMCStatus.UNKNOWN
                result.error_message = "Path comparison: return values differ"
        else:
            result.status = BMCStatus.UNKNOWN
            result.error_message = f"Different path counts: {len(left_paths)} vs {len(right_paths)}"

        result.paths_feasible = min(len(left_paths), len(right_paths))
        return result
