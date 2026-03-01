"""
Z3 solver interface.

Manages solver instances with theory selection, assertions, satisfiability
checking with timeout, model extraction, incremental solving, and
unsat core extraction.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict, Tuple, Any, Set

import z3


# ---------------------------------------------------------------------------
# Configuration and result types
# ---------------------------------------------------------------------------

class SolverStatus(Enum):
    SAT = auto()
    UNSAT = auto()
    UNKNOWN = auto()
    TIMEOUT = auto()


class Theory(Enum):
    """SMT theories."""
    QF_BV = auto()       # Quantifier-free bitvectors
    QF_FP = auto()       # Quantifier-free floating point
    QF_ABV = auto()      # Quantifier-free arrays + bitvectors
    QF_AUFBV = auto()    # Quantifier-free arrays + uninterpreted functions + BV
    ALL = auto()         # No restriction


@dataclass
class SolverConfig:
    """Configuration for the SMT solver."""
    timeout_ms: int = 10000
    theory: Theory = Theory.QF_BV
    incremental: bool = True
    produce_models: bool = True
    produce_unsat_cores: bool = False
    random_seed: int = 42
    max_memory_mb: int = 4096


@dataclass
class SolverResult:
    """Result of a satisfiability check."""
    status: SolverStatus
    model: Optional[z3.ModelRef] = None
    unsat_core: Optional[List[z3.ExprRef]] = None
    time_ms: float = 0.0
    statistics: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_sat(self) -> bool:
        return self.status == SolverStatus.SAT

    @property
    def is_unsat(self) -> bool:
        return self.status == SolverStatus.UNSAT

    @property
    def is_unknown(self) -> bool:
        return self.status in (SolverStatus.UNKNOWN, SolverStatus.TIMEOUT)

    def get_value(self, expr: z3.ExprRef) -> Optional[z3.ExprRef]:
        """Get the value of an expression in the model."""
        if self.model is None:
            return None
        try:
            return self.model.evaluate(expr, model_completion=True)
        except z3.Z3Exception:
            return None

    def summary(self) -> str:
        lines = [
            f"Solver Result: {self.status.name}",
            f"  Time: {self.time_ms:.1f}ms",
        ]
        if self.unsat_core:
            lines.append(f"  Unsat core size: {len(self.unsat_core)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# SMT Solver
# ---------------------------------------------------------------------------

class SMTSolver:
    """
    Z3 solver interface with scope management, incremental solving,
    and theory selection.
    """

    def __init__(self, config: Optional[SolverConfig] = None):
        self.config = config or SolverConfig()
        self._scope_depth = 0
        self._assertion_count = 0
        self._check_count = 0
        self._total_time_ms = 0.0

        # Create solver based on theory
        if self.config.produce_unsat_cores:
            self._solver = z3.Solver()
            self._solver.set("unsat_core", True)
        else:
            self._solver = self._create_solver()

        self._solver.set("timeout", self.config.timeout_ms)
        if self.config.random_seed:
            self._solver.set("random_seed", self.config.random_seed)

        # Tracked assertions for unsat core
        self._tracked: Dict[str, z3.BoolRef] = {}
        self._track_counter = 0

    def _create_solver(self) -> z3.Solver:
        """Create a Z3 solver with appropriate tactic."""
        theory = self.config.theory

        if theory == Theory.QF_BV:
            return z3.SolverFor("QF_BV")
        elif theory == Theory.QF_FP:
            return z3.SolverFor("QF_FP")
        elif theory == Theory.QF_ABV:
            return z3.SolverFor("QF_ABV")
        elif theory == Theory.QF_AUFBV:
            return z3.SolverFor("QF_AUFBV")
        else:
            return z3.Solver()

    # -- Assertions --

    def add(self, *assertions: z3.BoolRef) -> None:
        """Add assertions to the solver."""
        for a in assertions:
            self._solver.add(a)
            self._assertion_count += 1

    def add_tracked(self, assertion: z3.BoolRef, name: Optional[str] = None) -> str:
        """Add a tracked assertion for unsat core extraction."""
        if name is None:
            self._track_counter += 1
            name = f"track_{self._track_counter}"

        track_var = z3.Bool(name)
        self._solver.add(z3.Implies(track_var, assertion))
        self._tracked[name] = track_var
        self._assertion_count += 1
        return name

    def add_soft(self, assertion: z3.BoolRef, weight: int = 1, group: str = "") -> None:
        """Add a soft assertion (for optimization)."""
        # Z3's Optimize supports soft constraints
        self._solver.add(assertion)
        self._assertion_count += 1

    # -- Scope management --

    def push(self) -> None:
        """Push a new scope level."""
        self._solver.push()
        self._scope_depth += 1

    def pop(self, n: int = 1) -> None:
        """Pop scope levels."""
        actual = min(n, self._scope_depth)
        if actual > 0:
            self._solver.pop(actual)
            self._scope_depth -= actual

    @property
    def scope_depth(self) -> int:
        return self._scope_depth

    # -- Checking --

    def check(self, *assumptions: z3.BoolRef) -> SolverResult:
        """Check satisfiability with optional assumptions."""
        self._check_count += 1
        start = time.time()

        try:
            if assumptions:
                result = self._solver.check(*assumptions)
            elif self._tracked and self.config.produce_unsat_cores:
                track_vars = list(self._tracked.values())
                result = self._solver.check(*track_vars)
            else:
                result = self._solver.check()
        except z3.Z3Exception:
            elapsed = (time.time() - start) * 1000
            self._total_time_ms += elapsed
            return SolverResult(status=SolverStatus.UNKNOWN, time_ms=elapsed)

        elapsed = (time.time() - start) * 1000
        self._total_time_ms += elapsed

        if result == z3.sat:
            model = self._solver.model() if self.config.produce_models else None
            return SolverResult(
                status=SolverStatus.SAT,
                model=model,
                time_ms=elapsed,
            )
        elif result == z3.unsat:
            core = None
            if self.config.produce_unsat_cores:
                try:
                    core = list(self._solver.unsat_core())
                except z3.Z3Exception:
                    core = None
            return SolverResult(
                status=SolverStatus.UNSAT,
                unsat_core=core,
                time_ms=elapsed,
            )
        else:
            status = SolverStatus.TIMEOUT if elapsed >= self.config.timeout_ms else SolverStatus.UNKNOWN
            return SolverResult(status=status, time_ms=elapsed)

    def check_sat(self) -> bool:
        """Quick satisfiability check."""
        result = self.check()
        return result.is_sat

    def check_valid(self, formula: z3.BoolRef) -> SolverResult:
        """Check if a formula is valid (always true) by checking negation."""
        self.push()
        self.add(z3.Not(formula))
        result = self.check()
        self.pop()

        if result.is_unsat:
            return SolverResult(status=SolverStatus.SAT, time_ms=result.time_ms)
        elif result.is_sat:
            return SolverResult(
                status=SolverStatus.UNSAT,
                model=result.model,
                time_ms=result.time_ms,
            )
        else:
            return result

    def check_implies(
        self,
        premise: z3.BoolRef,
        conclusion: z3.BoolRef,
    ) -> SolverResult:
        """Check if premise implies conclusion."""
        return self.check_valid(z3.Implies(premise, conclusion))

    # -- Model extraction --

    def get_model(self) -> Optional[z3.ModelRef]:
        """Get the model from the last SAT check."""
        try:
            return self._solver.model()
        except z3.Z3Exception:
            return None

    def evaluate(self, expr: z3.ExprRef, model: Optional[z3.ModelRef] = None) -> Optional[z3.ExprRef]:
        """Evaluate an expression in a model."""
        if model is None:
            model = self.get_model()
        if model is None:
            return None
        try:
            return model.evaluate(expr, model_completion=True)
        except z3.Z3Exception:
            return None

    def get_unsat_core(self) -> List[z3.ExprRef]:
        """Get the unsatisfiable core from the last UNSAT check."""
        try:
            return list(self._solver.unsat_core())
        except z3.Z3Exception:
            return []

    # -- Convenience --

    def find_counterexample(
        self,
        formula: z3.BoolRef,
        variables: List[z3.ExprRef],
    ) -> Optional[Dict[str, z3.ExprRef]]:
        """Find a counterexample to a formula (values that make it False)."""
        self.push()
        self.add(z3.Not(formula))
        result = self.check()
        self.pop()

        if result.is_sat and result.model:
            cex: Dict[str, z3.ExprRef] = {}
            for v in variables:
                val = result.model.evaluate(v, model_completion=True)
                cex[str(v)] = val
            return cex
        return None

    def find_satisfying(
        self,
        formula: z3.BoolRef,
        variables: List[z3.ExprRef],
    ) -> Optional[Dict[str, z3.ExprRef]]:
        """Find values satisfying a formula."""
        self.push()
        self.add(formula)
        result = self.check()
        self.pop()

        if result.is_sat and result.model:
            vals: Dict[str, z3.ExprRef] = {}
            for v in variables:
                val = result.model.evaluate(v, model_completion=True)
                vals[str(v)] = val
            return vals
        return None

    def all_solutions(
        self,
        formula: z3.BoolRef,
        variables: List[z3.ExprRef],
        max_solutions: int = 100,
    ) -> List[Dict[str, z3.ExprRef]]:
        """Enumerate all solutions up to a maximum count."""
        solutions: List[Dict[str, z3.ExprRef]] = []
        self.push()
        self.add(formula)

        for _ in range(max_solutions):
            result = self.check()
            if not result.is_sat or result.model is None:
                break

            sol: Dict[str, z3.ExprRef] = {}
            blocking: List[z3.BoolRef] = []
            for v in variables:
                val = result.model.evaluate(v, model_completion=True)
                sol[str(v)] = val
                blocking.append(v != val)
            solutions.append(sol)

            # Block this solution
            if blocking:
                self.add(z3.Or(*blocking))

        self.pop()
        return solutions

    # -- Reset --

    def reset(self) -> None:
        """Reset the solver to initial state."""
        self._solver.reset()
        self._scope_depth = 0
        self._assertion_count = 0
        self._tracked.clear()
        self._track_counter = 0

    # -- Statistics --

    @property
    def num_assertions(self) -> int:
        return self._assertion_count

    @property
    def num_checks(self) -> int:
        return self._check_count

    @property
    def total_time_ms(self) -> float:
        return self._total_time_ms

    def statistics(self) -> Dict[str, Any]:
        return {
            "assertions": self._assertion_count,
            "checks": self._check_count,
            "total_time_ms": self._total_time_ms,
            "scope_depth": self._scope_depth,
            "tracked_assertions": len(self._tracked),
        }

    def summary(self) -> str:
        stats = self.statistics()
        return (
            f"SMTSolver: {stats['assertions']} assertions, "
            f"{stats['checks']} checks, "
            f"{stats['total_time_ms']:.1f}ms total"
        )

    def __repr__(self) -> str:
        return f"SMTSolver(theory={self.config.theory.name}, depth={self._scope_depth})"
