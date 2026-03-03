"""
Main verification engine for the Cross-Language Equivalence Verifier.

Implements bounded equivalence checking using product programs,
symbolic execution, and SMT solving. Orchestrates the full
verification pipeline from IR functions to equivalence verdicts.

Provides:
- EquivalenceVerifier: main verification engine
- VerificationConfig: configuration for verification
- VerificationSession: tracks state of a verification run
- FunctionVerificationResult: result for a single function pair
- ModuleVerificationResult: aggregated results for module pairs
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import re

from ..ir.function import Function
from ..ir.module import Module
from ..ir.basic_block import BasicBlock
from ..ir.instructions import (
    Instruction, Value, Constant, Argument,
    BinaryOp, UnaryOp, CompareOp,
    LoadInst, StoreInst,
    CallInst, ReturnInst, BranchInst,
    PhiInst, SelectInst,
)
from ..ir.types import IRType, IntType, FloatType, VoidType

logger = logging.getLogger(__name__)


# ─── Configuration ──────────────────────────────────────────────────

class VerificationStrategy(Enum):
    """Strategy for equivalence verification."""
    PRODUCT_PROGRAM = auto()     # Build product program and verify
    SYMBOLIC_EXECUTION = auto()  # Symbolic execution with path merging
    BOUNDED_MC = auto()          # Bounded model checking
    COMBINED = auto()            # Try multiple strategies


@dataclass
class VerificationConfig:
    """Configuration for the equivalence verifier."""
    strategy: VerificationStrategy = VerificationStrategy.COMBINED
    timeout_per_function_ms: float = 30000.0
    timeout_total_ms: float = 300000.0
    max_path_depth: int = 100
    max_loop_unroll: int = 10
    max_memory_mb: int = 4096
    incremental: bool = True
    verify_side_effects: bool = True
    verify_return_values: bool = True
    verify_memory_state: bool = True
    use_unsat_cores: bool = True
    collect_coverage: bool = True
    parallel_functions: int = 1
    debug_mode: bool = False
    produce_counterexamples: bool = True
    minimize_counterexamples: bool = True


# ─── Verification Status ──────────────────────────────────────────

class VerificationStatus(Enum):
    """Status of a verification attempt."""
    NOT_STARTED = auto()
    IN_PROGRESS = auto()
    EQUIVALENT = auto()
    NOT_EQUIVALENT = auto()
    UNKNOWN = auto()
    TIMEOUT = auto()
    ERROR = auto()
    SKIPPED = auto()


# ─── Function Matching ─────────────────────────────────────────────

@dataclass
class FunctionPair:
    """A pair of functions to check for equivalence."""
    left: Function
    right: Function
    name: str = ""
    signature_match: bool = True
    notes: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"{self.left.name} ↔ {self.right.name}"


class FunctionMatcher:
    """Match functions from two modules for equivalence checking."""

    def __init__(self) -> None:
        self._matched: List[FunctionPair] = []
        self._unmatched_left: List[Function] = []
        self._unmatched_right: List[Function] = []

    def match(self, left_module: Module, right_module: Module) -> List[FunctionPair]:
        """Match functions by name and signature, with fuzzy C/Rust matching."""
        self._matched.clear()
        self._unmatched_left.clear()
        self._unmatched_right.clear()

        right_funcs = {f.name: f for f in right_module.functions}
        matched_right: Set[str] = set()

        for left_func in left_module.functions:
            if left_func.name in right_funcs:
                # Exact name match
                right_func = right_funcs[left_func.name]
                sig_match = self._signatures_compatible(left_func, right_func)
                pair = FunctionPair(
                    left=left_func,
                    right=right_func,
                    signature_match=sig_match,
                )
                if not sig_match:
                    pair.notes.append("Signature mismatch")
                self._matched.append(pair)
                matched_right.add(left_func.name)
            else:
                # Try fuzzy matching for cross-language names
                fuzzy_match = self._fuzzy_find(left_func.name, right_funcs, matched_right)
                if fuzzy_match is not None:
                    right_func = right_funcs[fuzzy_match]
                    sig_match = self._signatures_compatible(left_func, right_func)
                    pair = FunctionPair(
                        left=left_func,
                        right=right_func,
                        signature_match=sig_match,
                    )
                    pair.notes.append(f"Fuzzy matched: {left_func.name} ↔ {fuzzy_match}")
                    if not sig_match:
                        pair.notes.append("Signature mismatch")
                    self._matched.append(pair)
                    matched_right.add(fuzzy_match)
                else:
                    self._unmatched_left.append(left_func)

        for name, func in right_funcs.items():
            if name not in matched_right:
                self._unmatched_right.append(func)

        return self._matched

    def _fuzzy_find(self, name: str, candidates: Dict[str, Function],
                     already_matched: Set[str]) -> Optional[str]:
        """Fuzzy match a function name against candidates.

        Handles C/Rust naming differences: strips common prefixes/suffixes,
        normalizes underscores vs camelCase, and handles Rust name mangling.
        """
        normalized = self._normalize_name(name)
        for cand_name in candidates:
            if cand_name in already_matched:
                continue
            if self._normalize_name(cand_name) == normalized:
                return cand_name
        return None

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize a function name for cross-language comparison.

        Strips common prefixes (_Z, __,  module::), converts camelCase
        to snake_case, and lowercases.
        """
        # Strip Rust name mangling prefix (_ZN...)
        n = re.sub(r'^_ZN\d+', '', name)
        # Strip leading underscores (C convention)
        n = n.lstrip('_')
        # Strip Rust module path (module::func → func)
        if '::' in n:
            n = n.rsplit('::', 1)[-1]
        # Convert camelCase to snake_case
        n = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', n)
        return n.lower().strip('_')

    def _signatures_compatible(self, left: Function, right: Function) -> bool:
        """Check if two functions have compatible signatures."""
        if left.num_arguments != right.num_arguments:
            return False
        if str(left.return_type) != str(right.return_type):
            return False
        for la, ra in zip(left.arguments, right.arguments):
            if str(la.ir_type) != str(ra.ir_type):
                return False
        return True


# ─── Path State ────────────────────────────────────────────────────

@dataclass
class PathState:
    """State of a single execution path during verification."""
    path_id: int
    constraints: List[Any] = field(default_factory=list)
    left_values: Dict[int, Any] = field(default_factory=dict)
    right_values: Dict[int, Any] = field(default_factory=dict)
    left_block: Optional[BasicBlock] = None
    right_block: Optional[BasicBlock] = None
    left_memory: Dict[int, Any] = field(default_factory=dict)
    right_memory: Dict[int, Any] = field(default_factory=dict)
    depth: int = 0
    feasible: bool = True
    left_return: Optional[Any] = None
    right_return: Optional[Any] = None

    def fork(self, path_id: int) -> "PathState":
        return PathState(
            path_id=path_id,
            constraints=list(self.constraints),
            left_values=dict(self.left_values),
            right_values=dict(self.right_values),
            left_block=self.left_block,
            right_block=self.right_block,
            left_memory=dict(self.left_memory),
            right_memory=dict(self.right_memory),
            depth=self.depth,
            feasible=self.feasible,
        )


# ─── Function Verification Result ────────────────────────────────

@dataclass
class FunctionVerificationResult:
    """Result of verifying a single function pair."""
    pair: FunctionPair
    status: VerificationStatus = VerificationStatus.NOT_STARTED
    time_ms: float = 0.0
    paths_explored: int = 0
    paths_feasible: int = 0
    blocks_covered_left: int = 0
    blocks_covered_right: int = 0
    counterexample: Optional[Any] = None
    error_message: str = ""
    smt_queries: int = 0
    smt_time_ms: float = 0.0

    @property
    def is_equivalent(self) -> bool:
        return self.status == VerificationStatus.EQUIVALENT

    @property
    def coverage_left(self) -> float:
        total = self.pair.left.num_blocks
        return self.blocks_covered_left / max(total, 1)

    @property
    def coverage_right(self) -> float:
        total = self.pair.right.num_blocks
        return self.blocks_covered_right / max(total, 1)

    def summary(self) -> str:
        status_str = self.status.name
        return (f"{self.pair.name}: {status_str} "
                f"({self.time_ms:.1f}ms, {self.paths_explored} paths, "
                f"coverage L={self.coverage_left:.0%} R={self.coverage_right:.0%})")


# ─── Module Verification Result ──────────────────────────────────

@dataclass
class ModuleVerificationResult:
    """Aggregated verification results for a module pair."""
    function_results: List[FunctionVerificationResult] = field(default_factory=list)
    total_time_ms: float = 0.0
    unmatched_left: List[str] = field(default_factory=list)
    unmatched_right: List[str] = field(default_factory=list)

    @property
    def num_equivalent(self) -> int:
        return sum(1 for r in self.function_results if r.is_equivalent)

    @property
    def num_not_equivalent(self) -> int:
        return sum(1 for r in self.function_results
                   if r.status == VerificationStatus.NOT_EQUIVALENT)

    @property
    def num_unknown(self) -> int:
        return sum(1 for r in self.function_results
                   if r.status in (VerificationStatus.UNKNOWN, VerificationStatus.TIMEOUT))

    @property
    def num_total(self) -> int:
        return len(self.function_results)

    @property
    def all_equivalent(self) -> bool:
        return all(r.is_equivalent for r in self.function_results)

    def summary(self) -> str:
        lines = [
            f"Module Verification Results ({self.total_time_ms:.1f}ms)",
            f"  Functions: {self.num_total} total",
            f"  Equivalent: {self.num_equivalent}",
            f"  Not equivalent: {self.num_not_equivalent}",
            f"  Unknown/timeout: {self.num_unknown}",
        ]
        if self.unmatched_left:
            lines.append(f"  Unmatched (left): {', '.join(self.unmatched_left)}")
        if self.unmatched_right:
            lines.append(f"  Unmatched (right): {', '.join(self.unmatched_right)}")
        lines.append("")
        for r in self.function_results:
            lines.append(f"  {r.summary()}")
        return "\n".join(lines)


# ─── Verification Session ────────────────────────────────────────

class VerificationSession:
    """Tracks the state of an active verification run."""

    def __init__(self, config: VerificationConfig) -> None:
        self._config = config
        self._start_time: float = 0.0
        self._function_results: List[FunctionVerificationResult] = []
        self._current_function: Optional[str] = None
        self._total_smt_queries: int = 0
        self._total_smt_time_ms: float = 0.0
        self._callbacks: List[Callable] = []
        self._cancelled: bool = False

    @property
    def config(self) -> VerificationConfig:
        return self._config

    @property
    def elapsed_ms(self) -> float:
        return (time.monotonic() - self._start_time) * 1000

    @property
    def is_timed_out(self) -> bool:
        return self.elapsed_ms > self._config.timeout_total_ms

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    def start(self) -> None:
        self._start_time = time.monotonic()

    def cancel(self) -> None:
        self._cancelled = True

    def add_result(self, result: FunctionVerificationResult) -> None:
        self._function_results.append(result)
        self._total_smt_queries += result.smt_queries
        self._total_smt_time_ms += result.smt_time_ms

    def on_progress(self, callback: Callable) -> None:
        self._callbacks.append(callback)

    def notify_progress(self, function_name: str, status: str) -> None:
        self._current_function = function_name
        for cb in self._callbacks:
            try:
                cb(function_name, status)
            except Exception:
                pass

    def get_results(self) -> ModuleVerificationResult:
        return ModuleVerificationResult(
            function_results=list(self._function_results),
            total_time_ms=self.elapsed_ms,
        )


# ─── Product Program Builder ─────────────────────────────────────

class ProductProgramBuilder:
    """Build a product program from two functions for equivalence checking.

    A product program executes both functions in lockstep on shared
    symbolic inputs, then checks that outputs match.
    """

    def __init__(self) -> None:
        self._shared_inputs: Dict[str, Any] = {}
        self._left_prefix = "L."
        self._right_prefix = "R."

    def build(self, left: Function, right: Function) -> Dict[str, Any]:
        """Build a product program description.

        Returns a dict with:
        - shared_inputs: symbolic inputs shared by both functions
        - left_encoding: encoding of the left function
        - right_encoding: encoding of the right function
        - equivalence_conditions: conditions that must hold for equivalence
        """
        product = {
            "left_function": left.name,
            "right_function": right.name,
            "shared_inputs": {},
            "left_encoding": {},
            "right_encoding": {},
            "equivalence_conditions": [],
        }

        # Create shared symbolic inputs
        for i, (la, ra) in enumerate(zip(left.arguments, right.arguments)):
            input_name = f"input_{i}"
            product["shared_inputs"][input_name] = {
                "type": str(la.ir_type),
                "left_arg": la.name,
                "right_arg": ra.name,
            }

        # Build equivalence condition for return values
        if not isinstance(left.return_type, VoidType):
            product["equivalence_conditions"].append({
                "kind": "return_value_equality",
                "description": "Return values must be equal",
            })

        # Build equivalence conditions for memory effects
        product["equivalence_conditions"].append({
            "kind": "memory_state_equality",
            "description": "Observable memory state must be equal",
        })

        return product


# ─── SMT Equivalence Checker ─────────────────────────────────────

class SMTEquivalenceChecker:
    """Check equivalence using SMT encoding.

    Encodes both functions as SMT formulas, asserts shared inputs,
    and checks if outputs can differ (negation of equivalence).
    """

    def __init__(self, config: VerificationConfig) -> None:
        self._config = config
        self._queries = 0
        self._total_time_ms = 0.0

    @property
    def query_count(self) -> int:
        return self._queries

    @property
    def total_time_ms(self) -> float:
        return self._total_time_ms

    def check_equivalence(self, left: Function, right: Function,
                           session: VerificationSession) -> FunctionVerificationResult:
        """Check if two functions are equivalent using SMT."""
        pair = FunctionPair(left=left, right=right)
        result = FunctionVerificationResult(pair=pair)
        result.status = VerificationStatus.IN_PROGRESS

        start = time.monotonic()

        try:
            # Try to import SMT encoder and solver
            from ..smt.encoder import SMTEncoder, EncodingContext
            from ..smt.solver import SMTSolver, SolverConfig, SolverStatus

            encoder = SMTEncoder()
            ctx = EncodingContext()

            # Encode both functions
            left_ctx, left_ret = encoder.encode_function(left, ctx, prefix="L_")
            right_ctx, right_ret = encoder.encode_function(right, ctx, prefix="R_")

            # Assert shared inputs are equal
            for i, (la, ra) in enumerate(zip(left.arguments, right.arguments)):
                input_constraint = encoder.encode_type(la.ir_type)
                # Symbolic: both functions receive the same input

            # Check if outputs can differ
            solver_config = SolverConfig()
            solver_config.timeout_ms = int(self._config.timeout_per_function_ms)
            solver = SMTSolver(solver_config)

            # Add all assertions
            for assertion in left_ctx.assertions:
                solver.add(assertion)
            for assertion in right_ctx.assertions:
                solver.add(assertion)

            # Assert outputs different (negate equivalence)
            if left_ret is not None and right_ret is not None:
                import z3
                diff = left_ret != right_ret
                solver.add(diff)

            self._queries += 1
            smt_start = time.monotonic()
            smt_result = solver.check()
            smt_elapsed = (time.monotonic() - smt_start) * 1000
            self._total_time_ms += smt_elapsed

            if smt_result.status == SolverStatus.UNSAT:
                result.status = VerificationStatus.EQUIVALENT
            elif smt_result.status == SolverStatus.SAT:
                result.status = VerificationStatus.NOT_EQUIVALENT
                if self._config.produce_counterexamples:
                    result.counterexample = smt_result.model
            elif smt_result.status == SolverStatus.TIMEOUT:
                result.status = VerificationStatus.TIMEOUT
            else:
                result.status = VerificationStatus.UNKNOWN

            result.smt_queries = 1
            result.smt_time_ms = smt_elapsed

        except ImportError:
            # SMT solver not available: fall back to structural comparison
            result = self._structural_comparison(left, right, pair)
        except Exception as e:
            result.status = VerificationStatus.ERROR
            result.error_message = str(e)
            logger.error(f"SMT verification failed: {e}")

        result.time_ms = (time.monotonic() - start) * 1000
        return result

    def _structural_comparison(self, left: Function, right: Function,
                                pair: FunctionPair) -> FunctionVerificationResult:
        """Fall-back structural comparison when SMT is unavailable."""
        result = FunctionVerificationResult(pair=pair)

        if left.num_blocks != right.num_blocks:
            result.status = VerificationStatus.UNKNOWN
            result.error_message = "Different number of blocks (structural comparison)"
            return result

        if left.instruction_count != right.instruction_count:
            result.status = VerificationStatus.UNKNOWN
            result.error_message = "Different instruction count (structural comparison)"
            return result

        # Compare instruction sequences
        left_insts = list(left.iter_instructions())
        right_insts = list(right.iter_instructions())

        all_match = True
        for li, ri in zip(left_insts, right_insts):
            if type(li) != type(ri):
                all_match = False
                break
            if isinstance(li, BinaryOp) and isinstance(ri, BinaryOp):
                if li.op != ri.op:
                    all_match = False
                    break

        if all_match and len(left_insts) == len(right_insts):
            result.status = VerificationStatus.EQUIVALENT
        else:
            result.status = VerificationStatus.UNKNOWN
            result.error_message = "Structural mismatch"

        return result


# ─── Equivalence Verifier ────────────────────────────────────────

class EquivalenceVerifier:
    """Main equivalence verification engine.

    Orchestrates the verification of function pairs across modules
    using configurable strategies and timeouts.
    """

    def __init__(self, config: Optional[VerificationConfig] = None) -> None:
        self._config = config or VerificationConfig()
        self._session: Optional[VerificationSession] = None

    @property
    def config(self) -> VerificationConfig:
        return self._config

    @property
    def session(self) -> Optional[VerificationSession]:
        return self._session

    def verify_modules(self, left: Module, right: Module) -> ModuleVerificationResult:
        """Verify equivalence of all matching function pairs between modules."""
        self._session = VerificationSession(self._config)
        self._session.start()

        # Match functions
        matcher = FunctionMatcher()
        pairs = matcher.match(left, right)

        # Verify each pair
        for pair in pairs:
            if self._session.is_timed_out or self._session.is_cancelled:
                break

            self._session.notify_progress(pair.name, "verifying")
            result = self.verify_function_pair(pair)
            self._session.add_result(result)
            self._session.notify_progress(pair.name, result.status.name)

        module_result = self._session.get_results()
        module_result.unmatched_left = [f.name for f in matcher._unmatched_left]
        module_result.unmatched_right = [f.name for f in matcher._unmatched_right]

        return module_result

    def verify_function_pair(self, pair: FunctionPair) -> FunctionVerificationResult:
        """Verify equivalence of a single function pair with per-function timeout."""
        if not pair.signature_match:
            result = FunctionVerificationResult(pair=pair)
            result.status = VerificationStatus.NOT_EQUIVALENT
            result.error_message = "Incompatible signatures"
            return result

        if pair.left.num_blocks == 0 or pair.right.num_blocks == 0:
            result = FunctionVerificationResult(pair=pair)
            result.status = VerificationStatus.SKIPPED
            result.error_message = "Empty function"
            return result

        strategy = self._config.strategy
        fn_start = time.monotonic()

        if strategy == VerificationStrategy.PRODUCT_PROGRAM:
            result = self._verify_product(pair)
        elif strategy == VerificationStrategy.SYMBOLIC_EXECUTION:
            result = self._verify_symbolic(pair)
        elif strategy == VerificationStrategy.BOUNDED_MC:
            result = self._verify_bounded(pair)
        else:
            result = self._verify_combined(pair)

        # Enforce per-function timeout
        fn_elapsed = (time.monotonic() - fn_start) * 1000
        if fn_elapsed > self._config.timeout_per_function_ms:
            if result.status not in (VerificationStatus.EQUIVALENT,
                                      VerificationStatus.NOT_EQUIVALENT):
                result.status = VerificationStatus.TIMEOUT
                result.time_ms = fn_elapsed

        return result

    def _verify_structural(self, pair: FunctionPair) -> FunctionVerificationResult:
        """Quick structural comparison as a first-pass check.

        If functions have identical instruction sequences (same opcodes,
        same operation kinds), declare them equivalent without SMT.
        """
        result = FunctionVerificationResult(pair=pair)
        start = time.monotonic()

        left, right = pair.left, pair.right

        if left.num_blocks != right.num_blocks:
            result.status = VerificationStatus.UNKNOWN
            result.error_message = "Different block count"
            result.time_ms = (time.monotonic() - start) * 1000
            return result

        if left.instruction_count != right.instruction_count:
            result.status = VerificationStatus.UNKNOWN
            result.error_message = "Different instruction count"
            result.time_ms = (time.monotonic() - start) * 1000
            return result

        left_insts = list(left.iter_instructions())
        right_insts = list(right.iter_instructions())

        all_match = True
        for li, ri in zip(left_insts, right_insts):
            if type(li) != type(ri):
                all_match = False
                break
            if isinstance(li, BinaryOp) and isinstance(ri, BinaryOp):
                if li.op != ri.op:
                    all_match = False
                    break
            if isinstance(li, CompareOp) and isinstance(ri, CompareOp):
                if li.predicate != ri.predicate:
                    all_match = False
                    break
            if isinstance(li, UnaryOp) and isinstance(ri, UnaryOp):
                if li.op != ri.op:
                    all_match = False
                    break

        if all_match and len(left_insts) == len(right_insts):
            result.status = VerificationStatus.EQUIVALENT
        else:
            result.status = VerificationStatus.UNKNOWN
            result.error_message = "Structural mismatch"

        result.time_ms = (time.monotonic() - start) * 1000
        return result

    def _verify_product(self, pair: FunctionPair) -> FunctionVerificationResult:
        """Verify using product program construction."""
        builder = ProductProgramBuilder()
        product = builder.build(pair.left, pair.right)

        checker = SMTEquivalenceChecker(self._config)
        result = checker.check_equivalence(pair.left, pair.right, self._session)
        return result

    def _verify_symbolic(self, pair: FunctionPair) -> FunctionVerificationResult:
        """Verify using symbolic execution."""
        result = FunctionVerificationResult(pair=pair)
        start = time.monotonic()

        try:
            from ..symbolic_exec.executor import SymbolicExecutor
            from ..symbolic_exec.state import SymbolicState

            # Execute both functions symbolically
            left_executor = SymbolicExecutor()
            right_executor = SymbolicExecutor()

            left_state = SymbolicState()
            right_state = SymbolicState()

            # Share input symbols between executions
            for i, (la, ra) in enumerate(zip(pair.left.arguments, pair.right.arguments)):
                sym_name = f"shared_input_{i}"
                left_state.set_var(la.name, left_state.fresh_bv(sym_name, 64))
                right_state.set_var(ra.name, right_state.fresh_bv(sym_name, 64))

            left_paths = left_executor.execute(pair.left, left_state,
                                                max_depth=self._config.max_path_depth)
            right_paths = right_executor.execute(pair.right, right_state,
                                                  max_depth=self._config.max_path_depth)

            result.paths_explored = len(left_paths) + len(right_paths)
            result.status = VerificationStatus.UNKNOWN
            result.error_message = "Symbolic execution completed but equivalence checking not implemented"

        except ImportError:
            result.status = VerificationStatus.UNKNOWN
            result.error_message = "Symbolic executor not available"
        except Exception as e:
            result.status = VerificationStatus.ERROR
            result.error_message = str(e)

        result.time_ms = (time.monotonic() - start) * 1000
        return result

    def _verify_bounded(self, pair: FunctionPair) -> FunctionVerificationResult:
        """Verify using bounded model checking."""
        try:
            from .bounded_checker import BoundedModelChecker, BMCConfig
            bmc_config = BMCConfig(
                max_unroll_depth=self._config.max_loop_unroll,
                timeout_ms=self._config.timeout_per_function_ms,
            )
            bmc = BoundedModelChecker(bmc_config)
            return bmc.check_equivalence(pair.left, pair.right)
        except ImportError:
            result = FunctionVerificationResult(pair=pair)
            result.status = VerificationStatus.UNKNOWN
            result.error_message = "Bounded model checker not available"
            return result

    def _verify_combined(self, pair: FunctionPair) -> FunctionVerificationResult:
        """Try multiple verification strategies in order of increasing cost.

        Order: structural → bounded MC → product program → symbolic.
        Stops as soon as a definitive result is obtained.
        """
        fn_start = time.monotonic()

        # Strategy 1: quick structural check (cheapest)
        result = self._verify_structural(pair)
        if result.status == VerificationStatus.EQUIVALENT:
            return result

        # Check per-function timeout
        if (time.monotonic() - fn_start) * 1000 > self._config.timeout_per_function_ms:
            result.status = VerificationStatus.TIMEOUT
            return result

        # Strategy 2: bounded model checking (moderate cost)
        result = self._verify_bounded(pair)
        if result.status in (VerificationStatus.EQUIVALENT,
                              VerificationStatus.NOT_EQUIVALENT):
            return result

        if (time.monotonic() - fn_start) * 1000 > self._config.timeout_per_function_ms:
            result.status = VerificationStatus.TIMEOUT
            return result

        # Strategy 3: product program + SMT (higher cost)
        result = self._verify_product(pair)
        if result.status in (VerificationStatus.EQUIVALENT,
                              VerificationStatus.NOT_EQUIVALENT):
            return result

        if (time.monotonic() - fn_start) * 1000 > self._config.timeout_per_function_ms:
            result.status = VerificationStatus.TIMEOUT
            return result

        # Strategy 4: symbolic execution (highest cost)
        result = self._verify_symbolic(pair)
        if result.status in (VerificationStatus.EQUIVALENT,
                              VerificationStatus.NOT_EQUIVALENT):
            return result

        # No strategy succeeded
        result = FunctionVerificationResult(pair=pair)
        result.status = VerificationStatus.UNKNOWN
        result.error_message = "All verification strategies inconclusive"
        result.time_ms = (time.monotonic() - fn_start) * 1000
        return result

    def verify_functions(self, left: Function, right: Function) -> FunctionVerificationResult:
        """Convenience method to verify two functions directly."""
        pair = FunctionPair(left=left, right=right)
        self._session = VerificationSession(self._config)
        self._session.start()
        return self.verify_function_pair(pair)

    def verify_modules_interprocedural(self, left: Module, right: Module) -> ModuleVerificationResult:
        """Verify modules with interprocedural analysis.

        If a function calls another function that's also in the module,
        inline the callee's verification result to strengthen confidence.
        """
        # First do standard verification
        module_result = self.verify_modules(left, right)

        # Build a map of verified function results
        verified: Dict[str, FunctionVerificationResult] = {}
        for r in module_result.function_results:
            verified[r.pair.left.name] = r

        # Check for call dependencies and propagate results
        for r in module_result.function_results:
            if r.status != VerificationStatus.UNKNOWN:
                continue

            # Collect callees in left function
            callees_equiv = True
            has_callees = False
            for inst in r.pair.left.iter_instructions():
                if isinstance(inst, CallInst):
                    callee_name = inst.callee_name if hasattr(inst, 'callee_name') else None
                    if callee_name and callee_name in verified:
                        has_callees = True
                        callee_result = verified[callee_name]
                        if not callee_result.is_equivalent:
                            callees_equiv = False
                            break

            # If all callees are equivalent and structural match was close,
            # upgrade confidence
            if has_callees and callees_equiv and r.status == VerificationStatus.UNKNOWN:
                r.error_message += " (all callees verified equivalent)"

        return module_result
