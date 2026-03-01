"""
Path management for symbolic execution.

Handles path constraint tracking, feasibility checking, path merging
at join points, prioritization strategies, and path explosion mitigation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict, Tuple, Set, Any

import z3

from .state import SymbolicState, PathConstraint, SymbolicValue


# ---------------------------------------------------------------------------
# Path strategy
# ---------------------------------------------------------------------------

class PathStrategy(Enum):
    """Strategy for path prioritization."""
    DEPTH_FIRST = auto()
    BREADTH_FIRST = auto()
    COVERAGE_GUIDED = auto()
    DIVERGENCE_SEEKING = auto()
    RANDOM = auto()
    SHORTEST_FIRST = auto()


# ---------------------------------------------------------------------------
# Path info
# ---------------------------------------------------------------------------

@dataclass
class PathInfo:
    """Information about an explored path."""
    path_id: int
    constraints: List[z3.BoolRef] = field(default_factory=list)
    covered_blocks: Set[str] = field(default_factory=set)
    depth: int = 0
    is_feasible: bool = True
    is_complete: bool = False
    return_value: Optional[z3.ExprRef] = None
    error: Optional[str] = None
    branch_history: List[Tuple[str, bool]] = field(default_factory=list)
    priority: float = 0.0

    @property
    def num_constraints(self) -> int:
        return len(self.constraints)

    @property
    def coverage_count(self) -> int:
        return len(self.covered_blocks)

    def summary(self) -> str:
        status = "complete" if self.is_complete else "partial"
        if self.error:
            status = f"error:{self.error}"
        return (
            f"Path[{self.path_id}]: {status}, depth={self.depth}, "
            f"constraints={self.num_constraints}, coverage={self.coverage_count}"
        )


# ---------------------------------------------------------------------------
# Path manager
# ---------------------------------------------------------------------------

class PathManager:
    """
    Manages paths during symbolic execution.
    
    Tracks path constraints, checks feasibility via Z3,
    merges paths at join points, and prioritizes exploration.
    """

    def __init__(
        self,
        strategy: PathStrategy = PathStrategy.DEPTH_FIRST,
        solver_timeout_ms: int = 10000,
        max_paths: int = 1000,
        merge_threshold: float = 0.8,
    ):
        self.strategy = strategy
        self.solver_timeout_ms = solver_timeout_ms
        self.max_paths = max_paths
        self.merge_threshold = merge_threshold

        # Active paths (worklist)
        self._active_paths: List[SymbolicState] = []

        # Completed paths
        self._completed: List[PathInfo] = []
        self._error_paths: List[PathInfo] = []

        # Path ID counter
        self._next_path_id = 0

        # Coverage tracking
        self._global_coverage: Set[str] = set()

        # Solver
        self._solver = z3.Solver()
        self._solver.set("timeout", solver_timeout_ms)

        # Statistics
        self._feasibility_checks = 0
        self._infeasible_count = 0
        self._merge_count = 0
        self._prune_count = 0

    # -- Path operations --

    def new_path_id(self) -> int:
        pid = self._next_path_id
        self._next_path_id += 1
        return pid

    def add_active(self, state: SymbolicState) -> None:
        """Add a state to the active worklist."""
        self._active_paths.append(state)

    def pick_next(self) -> Optional[SymbolicState]:
        """Pick the next state to explore based on strategy."""
        if not self._active_paths:
            return None

        if self.strategy == PathStrategy.DEPTH_FIRST:
            return self._active_paths.pop()
        elif self.strategy == PathStrategy.BREADTH_FIRST:
            return self._active_paths.pop(0)
        elif self.strategy == PathStrategy.COVERAGE_GUIDED:
            return self._pick_coverage_guided()
        elif self.strategy == PathStrategy.DIVERGENCE_SEEKING:
            return self._pick_divergence_seeking()
        elif self.strategy == PathStrategy.RANDOM:
            return self._pick_random()
        elif self.strategy == PathStrategy.SHORTEST_FIRST:
            return self._pick_shortest()
        else:
            return self._active_paths.pop()

    def _pick_coverage_guided(self) -> Optional[SymbolicState]:
        """Pick the state that is likely to cover new blocks."""
        best_idx = 0
        best_new = -1

        for i, state in enumerate(self._active_paths):
            new_blocks = len(state.covered_blocks - self._global_coverage)
            if new_blocks > best_new:
                best_new = new_blocks
                best_idx = i

        return self._active_paths.pop(best_idx)

    def _pick_divergence_seeking(self) -> Optional[SymbolicState]:
        """Pick states near potential divergence points."""
        # Prioritize states with shorter constraint paths (closer to root)
        # and those at blocks we haven't seen much
        best_idx = 0
        best_score = float('-inf')

        for i, state in enumerate(self._active_paths):
            new_coverage = len(state.covered_blocks - self._global_coverage)
            depth_penalty = state.path_length * 0.1
            score = new_coverage * 2.0 - depth_penalty
            if score > best_score:
                best_score = score
                best_idx = i

        return self._active_paths.pop(best_idx)

    def _pick_random(self) -> Optional[SymbolicState]:
        import random
        idx = random.randint(0, len(self._active_paths) - 1)
        return self._active_paths.pop(idx)

    def _pick_shortest(self) -> Optional[SymbolicState]:
        """Pick the state with fewest path constraints."""
        best_idx = 0
        best_len = float('inf')
        for i, state in enumerate(self._active_paths):
            if state.path_length < best_len:
                best_len = state.path_length
                best_idx = i
        return self._active_paths.pop(best_idx)

    @property
    def has_active(self) -> bool:
        return len(self._active_paths) > 0

    @property
    def num_active(self) -> int:
        return len(self._active_paths)

    @property
    def total_explored(self) -> int:
        return len(self._completed) + len(self._error_paths)

    # -- Feasibility --

    def check_feasibility(self, state: SymbolicState) -> bool:
        """Check if the current path constraints are satisfiable."""
        self._feasibility_checks += 1

        self._solver.push()
        for pc in state.path_constraints:
            self._solver.add(pc.condition)

        # Add memory constraints
        mem_constraints = state.memory.allocation_constraints()
        for mc in mem_constraints:
            self._solver.add(mc)

        result = self._solver.check()
        self._solver.pop()

        if result == z3.unsat:
            self._infeasible_count += 1
            return False
        # sat or unknown → treat as feasible
        return True

    def check_condition_feasibility(
        self,
        state: SymbolicState,
        condition: z3.BoolRef,
    ) -> bool:
        """Check if a condition is feasible under current path constraints."""
        self._feasibility_checks += 1

        self._solver.push()
        for pc in state.path_constraints:
            self._solver.add(pc.condition)
        self._solver.add(condition)

        result = self._solver.check()
        self._solver.pop()

        if result == z3.unsat:
            self._infeasible_count += 1
            return False
        return True

    def check_both_branches(
        self,
        state: SymbolicState,
        condition: z3.BoolRef,
    ) -> Tuple[bool, bool]:
        """Check feasibility of both branches. Returns (true_feasible, false_feasible)."""
        true_feas = self.check_condition_feasibility(state, condition)
        false_feas = self.check_condition_feasibility(state, z3.Not(condition))
        return true_feas, false_feas

    def fork_on_condition(
        self,
        state: SymbolicState,
        condition: z3.BoolRef,
        true_block: str,
        false_block: str,
    ) -> List[SymbolicState]:
        """Fork a state on a condition, returning feasible successors."""
        true_feas, false_feas = self.check_both_branches(state, condition)
        result: List[SymbolicState] = []

        if true_feas:
            true_state = state.fork()
            true_state.add_constraint(
                condition, True, state.current_block, f"br→{true_block}"
            )
            true_state.current_block = true_block
            result.append(true_state)

        if false_feas:
            false_state = state.fork()
            false_state.add_constraint(
                z3.Not(condition), False, state.current_block, f"br→{false_block}"
            )
            false_state.current_block = false_block
            result.append(false_state)

        if not result:
            self._infeasible_count += 1

        return result

    # -- Path merging --

    def try_merge_at_join(
        self,
        states: List[SymbolicState],
        join_block: str,
    ) -> List[SymbolicState]:
        """
        Try to merge states at a join point.
        
        Merges states that have compatible path constraints and
        similar variable sets to reduce path explosion.
        """
        if len(states) <= 1:
            return states

        # Group states by similarity
        groups = self._group_mergeable(states)

        result: List[SymbolicState] = []
        for group in groups:
            if len(group) == 1:
                result.append(group[0])
            else:
                merged = SymbolicState.merge(group)
                if merged is not None:
                    merged.current_block = join_block
                    self._merge_count += 1
                    result.append(merged)
                else:
                    result.extend(group)

        return result

    def _group_mergeable(
        self,
        states: List[SymbolicState],
    ) -> List[List[SymbolicState]]:
        """Group states that can be merged together."""
        if not states:
            return []

        groups: List[List[SymbolicState]] = [[states[0]]]

        for state in states[1:]:
            merged_into_group = False
            for group in groups:
                if self._are_mergeable(group[0], state):
                    group.append(state)
                    merged_into_group = True
                    break
            if not merged_into_group:
                groups.append([state])

        return groups

    def _are_mergeable(self, a: SymbolicState, b: SymbolicState) -> bool:
        """Check if two states can be merged."""
        # Must be at the same block
        if a.current_block != b.current_block:
            return False

        # Must have similar variable sets
        a_vars = set(a.all_variables().keys())
        b_vars = set(b.all_variables().keys())

        if not a_vars or not b_vars:
            return True

        intersection = a_vars & b_vars
        union = a_vars | b_vars

        similarity = len(intersection) / len(union) if union else 1.0
        return similarity >= self.merge_threshold

    # -- Pruning --

    def prune_infeasible(self) -> int:
        """Remove infeasible paths from the worklist."""
        original_count = len(self._active_paths)
        feasible: List[SymbolicState] = []

        for state in self._active_paths:
            if self.check_feasibility(state):
                feasible.append(state)
            else:
                self._prune_count += 1

        self._active_paths = feasible
        return original_count - len(feasible)

    def prune_by_coverage(self, min_new_coverage: int = 0) -> int:
        """Prune paths that are unlikely to cover new blocks."""
        if min_new_coverage <= 0:
            return 0

        original_count = len(self._active_paths)
        kept: List[SymbolicState] = []

        for state in self._active_paths:
            new_blocks = state.covered_blocks - self._global_coverage
            if len(new_blocks) >= min_new_coverage:
                kept.append(state)
            else:
                self._prune_count += 1

        self._active_paths = kept
        return original_count - len(kept)

    def limit_active_paths(self, max_active: Optional[int] = None) -> int:
        """Limit the number of active paths, pruning lowest priority."""
        limit = max_active or self.max_paths
        if len(self._active_paths) <= limit:
            return 0

        # Sort by priority and keep top N
        scored: List[Tuple[float, int, SymbolicState]] = []
        for i, state in enumerate(self._active_paths):
            score = self._compute_priority(state)
            scored.append((score, i, state))

        scored.sort(reverse=True)
        pruned = len(self._active_paths) - limit
        self._active_paths = [s for _, _, s in scored[:limit]]
        self._prune_count += pruned
        return pruned

    def _compute_priority(self, state: SymbolicState) -> float:
        """Compute priority score for a state."""
        new_coverage = len(state.covered_blocks - self._global_coverage)
        depth_penalty = state.path_length * 0.05
        return new_coverage * 2.0 - depth_penalty

    # -- Path completion --

    def complete_path(self, state: SymbolicState) -> PathInfo:
        """Record a completed path."""
        info = PathInfo(
            path_id=self.new_path_id(),
            constraints=[pc.condition for pc in state.path_constraints],
            covered_blocks=set(state.covered_blocks),
            depth=state.path_length,
            is_feasible=True,
            is_complete=True,
            return_value=state.return_value.z3_expr if state.return_value else None,
            branch_history=[
                (pc.source_block or "", pc.is_branch_taken)
                for pc in state.path_constraints
            ],
        )
        self._completed.append(info)
        self._global_coverage.update(state.covered_blocks)
        return info

    def error_path(self, state: SymbolicState, error: str) -> PathInfo:
        """Record an error path."""
        info = PathInfo(
            path_id=self.new_path_id(),
            constraints=[pc.condition for pc in state.path_constraints],
            covered_blocks=set(state.covered_blocks),
            depth=state.path_length,
            is_feasible=True,
            is_complete=False,
            error=error,
            branch_history=[
                (pc.source_block or "", pc.is_branch_taken)
                for pc in state.path_constraints
            ],
        )
        self._error_paths.append(info)
        self._global_coverage.update(state.covered_blocks)
        return info

    # -- Statistics --

    @property
    def global_coverage(self) -> Set[str]:
        return set(self._global_coverage)

    def statistics(self) -> Dict[str, Any]:
        return {
            "active_paths": len(self._active_paths),
            "completed_paths": len(self._completed),
            "error_paths": len(self._error_paths),
            "feasibility_checks": self._feasibility_checks,
            "infeasible_count": self._infeasible_count,
            "merge_count": self._merge_count,
            "prune_count": self._prune_count,
            "global_coverage": len(self._global_coverage),
        }

    def summary(self) -> str:
        stats = self.statistics()
        lines = [
            f"PathManager ({self.strategy.name}):",
            f"  Active: {stats['active_paths']}",
            f"  Completed: {stats['completed_paths']}",
            f"  Errors: {stats['error_paths']}",
            f"  Feasibility checks: {stats['feasibility_checks']}",
            f"  Infeasible: {stats['infeasible_count']}",
            f"  Merges: {stats['merge_count']}",
            f"  Pruned: {stats['prune_count']}",
            f"  Coverage: {stats['global_coverage']} blocks",
        ]
        return "\n".join(lines)
