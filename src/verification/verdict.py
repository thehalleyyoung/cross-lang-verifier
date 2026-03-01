"""
Verification verdict for the Cross-Language Equivalence Verifier.

Provides verdict data structures with evidence (proof, counterexample,
unknown+coverage), verdict aggregation across functions, and confidence
scoring based on coverage and path exploration.

Provides:
- EquivalenceVerdict: verdict with evidence
- VerdictKind: enum of verdict outcomes
- VerdictEvidence: supporting evidence
- VerdictAggregator: aggregate verdicts across functions
- ConfidenceScore: confidence scoring model
- CoverageInfo: code coverage information
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ─── Verdict Kind ──────────────────────────────────────────────────

class VerdictKind(Enum):
    """Kind of equivalence verdict."""
    EQUIVALENT = auto()         # Proven equivalent
    NOT_EQUIVALENT = auto()     # Proven not equivalent (counterexample found)
    UNKNOWN = auto()            # Could not determine
    TIMEOUT = auto()            # Timed out
    ERROR = auto()              # Internal error
    PARTIAL_EQUIVALENT = auto() # Equivalent within explored bounds
    CONDITIONALLY_EQUIVALENT = auto()  # Equivalent under certain assumptions


# ─── Coverage Information ─────────────────────────────────────────

@dataclass
class CoverageInfo:
    """Code coverage information from verification."""
    total_blocks_left: int = 0
    covered_blocks_left: int = 0
    total_blocks_right: int = 0
    covered_blocks_right: int = 0
    total_paths: int = 0
    explored_paths: int = 0
    feasible_paths: int = 0
    total_instructions_left: int = 0
    covered_instructions_left: int = 0
    total_instructions_right: int = 0
    covered_instructions_right: int = 0
    loop_iterations_explored: Dict[str, int] = field(default_factory=dict)
    uncovered_blocks_left: List[str] = field(default_factory=list)
    uncovered_blocks_right: List[str] = field(default_factory=list)

    @property
    def block_coverage_left(self) -> float:
        return self.covered_blocks_left / max(self.total_blocks_left, 1)

    @property
    def block_coverage_right(self) -> float:
        return self.covered_blocks_right / max(self.total_blocks_right, 1)

    @property
    def overall_block_coverage(self) -> float:
        total = self.total_blocks_left + self.total_blocks_right
        covered = self.covered_blocks_left + self.covered_blocks_right
        return covered / max(total, 1)

    @property
    def instruction_coverage_left(self) -> float:
        return self.covered_instructions_left / max(self.total_instructions_left, 1)

    @property
    def instruction_coverage_right(self) -> float:
        return self.covered_instructions_right / max(self.total_instructions_right, 1)

    @property
    def path_coverage(self) -> float:
        return self.explored_paths / max(self.total_paths, 1)

    @property
    def path_feasibility_rate(self) -> float:
        return self.feasible_paths / max(self.explored_paths, 1)

    def merge(self, other: "CoverageInfo") -> "CoverageInfo":
        """Merge coverage from another run."""
        return CoverageInfo(
            total_blocks_left=max(self.total_blocks_left, other.total_blocks_left),
            covered_blocks_left=max(self.covered_blocks_left, other.covered_blocks_left),
            total_blocks_right=max(self.total_blocks_right, other.total_blocks_right),
            covered_blocks_right=max(self.covered_blocks_right, other.covered_blocks_right),
            total_paths=max(self.total_paths, other.total_paths),
            explored_paths=self.explored_paths + other.explored_paths,
            feasible_paths=self.feasible_paths + other.feasible_paths,
            total_instructions_left=max(self.total_instructions_left, other.total_instructions_left),
            covered_instructions_left=max(self.covered_instructions_left, other.covered_instructions_left),
            total_instructions_right=max(self.total_instructions_right, other.total_instructions_right),
            covered_instructions_right=max(self.covered_instructions_right, other.covered_instructions_right),
        )

    def summary(self) -> str:
        lines = [
            f"Coverage Summary:",
            f"  Block coverage: L={self.block_coverage_left:.0%} R={self.block_coverage_right:.0%}",
            f"  Instruction coverage: L={self.instruction_coverage_left:.0%} R={self.instruction_coverage_right:.0%}",
            f"  Paths: {self.explored_paths}/{self.total_paths} explored, "
            f"{self.feasible_paths} feasible ({self.path_feasibility_rate:.0%})",
        ]
        if self.loop_iterations_explored:
            lines.append("  Loop iterations:")
            for loop, iters in sorted(self.loop_iterations_explored.items()):
                lines.append(f"    {loop}: {iters}")
        if self.uncovered_blocks_left:
            lines.append(f"  Uncovered (left): {', '.join(self.uncovered_blocks_left[:5])}")
        if self.uncovered_blocks_right:
            lines.append(f"  Uncovered (right): {', '.join(self.uncovered_blocks_right[:5])}")
        return "\n".join(lines)


# ─── Confidence Score ─────────────────────────────────────────────

class ConfidenceScore:
    """Confidence scoring for verification verdicts.

    Computes a confidence score [0, 1] based on:
    - Code coverage achieved
    - Number of paths explored
    - Verification strategy used
    - Loop unroll depth
    - SMT solver result quality
    """

    # Weight factors for different aspects
    COVERAGE_WEIGHT = 0.4
    PATH_WEIGHT = 0.3
    STRATEGY_WEIGHT = 0.2
    DEPTH_WEIGHT = 0.1

    # Strategy confidence bonuses
    STRATEGY_SCORES = {
        "smt_proof": 1.0,
        "bounded_mc": 0.7,
        "symbolic_exec": 0.6,
        "structural": 0.3,
        "fuzzing": 0.2,
    }

    def __init__(self) -> None:
        self._components: Dict[str, float] = {}

    @property
    def components(self) -> Dict[str, float]:
        return self._components

    def compute(self, coverage: CoverageInfo, strategy: str = "unknown",
                loop_depth: int = 0, max_loop_depth: int = 10,
                smt_result_quality: float = 1.0) -> float:
        """Compute overall confidence score."""
        # Coverage component
        coverage_score = coverage.overall_block_coverage
        self._components["coverage"] = coverage_score

        # Path exploration component
        path_score = min(1.0, coverage.explored_paths / max(coverage.total_paths, 1))
        self._components["path_exploration"] = path_score

        # Strategy component
        strategy_score = self.STRATEGY_SCORES.get(strategy, 0.5) * smt_result_quality
        self._components["strategy"] = strategy_score

        # Depth component (deeper exploration → higher confidence)
        depth_score = min(1.0, loop_depth / max(max_loop_depth, 1))
        self._components["depth"] = depth_score

        # Weighted combination
        total = (
            self.COVERAGE_WEIGHT * coverage_score +
            self.PATH_WEIGHT * path_score +
            self.STRATEGY_WEIGHT * strategy_score +
            self.DEPTH_WEIGHT * depth_score
        )

        return min(1.0, max(0.0, total))

    def compute_from_proof(self) -> float:
        """For SMT-proven results, confidence is 1.0."""
        self._components = {"smt_proof": 1.0}
        return 1.0

    def compute_from_counterexample(self, validated: bool) -> float:
        """Confidence for counterexample-based verdicts."""
        if validated:
            self._components = {"validated_cex": 1.0}
            return 1.0
        self._components = {"unvalidated_cex": 0.8}
        return 0.8

    def explain(self) -> str:
        """Explain the confidence score components."""
        lines = ["Confidence breakdown:"]
        for name, score in sorted(self._components.items()):
            lines.append(f"  {name}: {score:.2f}")
        return "\n".join(lines)


# ─── Verdict Evidence ─────────────────────────────────────────────

class EvidenceKind(Enum):
    """Kind of evidence supporting a verdict."""
    SMT_PROOF = auto()         # Formal proof from SMT solver
    COUNTEREXAMPLE = auto()    # Concrete counterexample
    COVERAGE_REPORT = auto()   # Coverage information
    STRUCTURAL_MATCH = auto()  # Structural comparison
    TIMEOUT_INFO = auto()      # Timeout information
    ERROR_INFO = auto()        # Error details
    ASSUMPTION = auto()        # Assumptions made


@dataclass
class VerdictEvidence:
    """Evidence supporting a verification verdict."""
    kind: EvidenceKind
    description: str = ""
    data: Optional[Any] = None
    confidence: float = 1.0

    def __str__(self) -> str:
        return f"[{self.kind.name}] {self.description} (confidence: {self.confidence:.2f})"


# ─── Equivalence Verdict ──────────────────────────────────────────

@dataclass
class EquivalenceVerdict:
    """Complete verdict for an equivalence check.

    Contains the verdict kind, supporting evidence, confidence score,
    coverage information, and timing data.
    """
    kind: VerdictKind
    function_name: str = ""
    left_function: str = ""
    right_function: str = ""
    evidence: List[VerdictEvidence] = field(default_factory=list)
    coverage: Optional[CoverageInfo] = None
    confidence: float = 0.0
    time_ms: float = 0.0
    assumptions: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def add_evidence(self, evidence: VerdictEvidence) -> None:
        self.evidence.append(evidence)

    def add_assumption(self, assumption: str) -> None:
        self.assumptions.append(assumption)

    @property
    def is_definitive(self) -> bool:
        return self.kind in (VerdictKind.EQUIVALENT, VerdictKind.NOT_EQUIVALENT)

    @property
    def has_counterexample(self) -> bool:
        return any(e.kind == EvidenceKind.COUNTEREXAMPLE for e in self.evidence)

    @property
    def has_proof(self) -> bool:
        return any(e.kind == EvidenceKind.SMT_PROOF for e in self.evidence)

    def get_counterexample(self) -> Optional[Any]:
        for e in self.evidence:
            if e.kind == EvidenceKind.COUNTEREXAMPLE:
                return e.data
        return None

    def summary(self) -> str:
        lines = [
            f"Verdict: {self.kind.name}",
            f"  Functions: {self.left_function} ↔ {self.right_function}",
            f"  Confidence: {self.confidence:.0%}",
            f"  Time: {self.time_ms:.1f}ms",
        ]
        if self.evidence:
            lines.append("  Evidence:")
            for e in self.evidence:
                lines.append(f"    {e}")
        if self.assumptions:
            lines.append("  Assumptions:")
            for a in self.assumptions:
                lines.append(f"    - {a}")
        if self.coverage:
            lines.append(self.coverage.summary())
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


# ─── Verdict Builder ─────────────────────────────────────────────

class VerdictBuilder:
    """Builder pattern for constructing verdicts."""

    def __init__(self) -> None:
        self._verdict = EquivalenceVerdict(kind=VerdictKind.UNKNOWN)
        self._scorer = ConfidenceScore()

    def set_functions(self, left: str, right: str) -> "VerdictBuilder":
        self._verdict.left_function = left
        self._verdict.right_function = right
        self._verdict.function_name = f"{left} ↔ {right}"
        return self

    def set_equivalent(self) -> "VerdictBuilder":
        self._verdict.kind = VerdictKind.EQUIVALENT
        return self

    def set_not_equivalent(self) -> "VerdictBuilder":
        self._verdict.kind = VerdictKind.NOT_EQUIVALENT
        return self

    def set_unknown(self, reason: str = "") -> "VerdictBuilder":
        self._verdict.kind = VerdictKind.UNKNOWN
        if reason:
            self._verdict.notes.append(reason)
        return self

    def set_timeout(self, elapsed_ms: float) -> "VerdictBuilder":
        self._verdict.kind = VerdictKind.TIMEOUT
        self._verdict.time_ms = elapsed_ms
        return self

    def add_smt_proof(self, description: str = "SMT proof of equivalence") -> "VerdictBuilder":
        self._verdict.add_evidence(VerdictEvidence(
            kind=EvidenceKind.SMT_PROOF,
            description=description,
            confidence=1.0,
        ))
        return self

    def add_counterexample(self, cex: Any,
                            description: str = "Counterexample found") -> "VerdictBuilder":
        self._verdict.add_evidence(VerdictEvidence(
            kind=EvidenceKind.COUNTEREXAMPLE,
            description=description,
            data=cex,
            confidence=1.0,
        ))
        return self

    def set_coverage(self, coverage: CoverageInfo) -> "VerdictBuilder":
        self._verdict.coverage = coverage
        self._verdict.add_evidence(VerdictEvidence(
            kind=EvidenceKind.COVERAGE_REPORT,
            description=coverage.summary(),
            confidence=coverage.overall_block_coverage,
        ))
        return self

    def add_assumption(self, assumption: str) -> "VerdictBuilder":
        self._verdict.add_assumption(assumption)
        return self

    def set_time(self, time_ms: float) -> "VerdictBuilder":
        self._verdict.time_ms = time_ms
        return self

    def compute_confidence(self, strategy: str = "unknown") -> "VerdictBuilder":
        if self._verdict.kind == VerdictKind.EQUIVALENT and self._verdict.has_proof:
            self._verdict.confidence = self._scorer.compute_from_proof()
        elif self._verdict.kind == VerdictKind.NOT_EQUIVALENT:
            self._verdict.confidence = self._scorer.compute_from_counterexample(
                self._verdict.has_counterexample)
        elif self._verdict.coverage is not None:
            self._verdict.confidence = self._scorer.compute(
                self._verdict.coverage, strategy)
        else:
            self._verdict.confidence = 0.0
        return self

    def build(self) -> EquivalenceVerdict:
        return self._verdict


# ─── Verdict Aggregator ──────────────────────────────────────────

class VerdictAggregator:
    """Aggregate verdicts across multiple function pairs.

    Combines individual function verdicts into an overall module-level
    verdict with aggregate statistics.
    """

    def __init__(self) -> None:
        self._verdicts: List[EquivalenceVerdict] = []

    def add(self, verdict: EquivalenceVerdict) -> None:
        self._verdicts.append(verdict)

    @property
    def verdicts(self) -> List[EquivalenceVerdict]:
        return self._verdicts

    @property
    def num_equivalent(self) -> int:
        return sum(1 for v in self._verdicts if v.kind == VerdictKind.EQUIVALENT)

    @property
    def num_not_equivalent(self) -> int:
        return sum(1 for v in self._verdicts if v.kind == VerdictKind.NOT_EQUIVALENT)

    @property
    def num_unknown(self) -> int:
        return sum(1 for v in self._verdicts
                   if v.kind in (VerdictKind.UNKNOWN, VerdictKind.TIMEOUT))

    @property
    def num_total(self) -> int:
        return len(self._verdicts)

    @property
    def all_equivalent(self) -> bool:
        return all(v.kind == VerdictKind.EQUIVALENT for v in self._verdicts)

    @property
    def average_confidence(self) -> float:
        if not self._verdicts:
            return 0.0
        return sum(v.confidence for v in self._verdicts) / len(self._verdicts)

    @property
    def total_time_ms(self) -> float:
        return sum(v.time_ms for v in self._verdicts)

    def aggregate_verdict(self) -> EquivalenceVerdict:
        """Produce an aggregate verdict for the entire module."""
        if not self._verdicts:
            return EquivalenceVerdict(kind=VerdictKind.UNKNOWN)

        if self.all_equivalent:
            kind = VerdictKind.EQUIVALENT
        elif self.num_not_equivalent > 0:
            kind = VerdictKind.NOT_EQUIVALENT
        else:
            kind = VerdictKind.UNKNOWN

        aggregate = EquivalenceVerdict(
            kind=kind,
            function_name="module-level",
            confidence=self.average_confidence,
            time_ms=self.total_time_ms,
        )

        # Aggregate coverage
        if any(v.coverage is not None for v in self._verdicts):
            total_coverage = CoverageInfo()
            for v in self._verdicts:
                if v.coverage is not None:
                    total_coverage = total_coverage.merge(v.coverage)
            aggregate.coverage = total_coverage

        # Collect all counterexamples
        for v in self._verdicts:
            cex = v.get_counterexample()
            if cex is not None:
                aggregate.add_evidence(VerdictEvidence(
                    kind=EvidenceKind.COUNTEREXAMPLE,
                    description=f"From {v.function_name}",
                    data=cex,
                ))

        return aggregate

    def summary(self) -> str:
        lines = [
            f"Verdict Aggregation ({self.num_total} functions):",
            f"  Equivalent: {self.num_equivalent}",
            f"  Not equivalent: {self.num_not_equivalent}",
            f"  Unknown: {self.num_unknown}",
            f"  Average confidence: {self.average_confidence:.0%}",
            f"  Total time: {self.total_time_ms:.1f}ms",
        ]
        if not self.all_equivalent:
            lines.append("  Non-equivalent functions:")
            for v in self._verdicts:
                if v.kind == VerdictKind.NOT_EQUIVALENT:
                    lines.append(f"    {v.function_name}")
        return "\n".join(lines)

    def filter_by_kind(self, kind: VerdictKind) -> List[EquivalenceVerdict]:
        return [v for v in self._verdicts if v.kind == kind]

    def get_lowest_confidence(self) -> Optional[EquivalenceVerdict]:
        if not self._verdicts:
            return None
        return min(self._verdicts, key=lambda v: v.confidence)

    def get_highest_confidence(self) -> Optional[EquivalenceVerdict]:
        if not self._verdicts:
            return None
        return max(self._verdicts, key=lambda v: v.confidence)
