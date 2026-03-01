"""
Verification Core for the Cross-Language Equivalence Verifier.

Provides the main verification infrastructure including:
- Equivalence verification engine (product programs + symbolic execution + SMT)
- Counterexample generation and validation
- Verification verdicts with evidence and confidence scoring
- Bounded model checking with loop unrolling
"""

from .verifier import (
    EquivalenceVerifier,
    VerificationConfig,
    VerificationSession,
    FunctionVerificationResult,
    ModuleVerificationResult,
)
from .counterexample import (
    CounterexampleGenerator,
    Counterexample,
    ConcreteInput,
    CounterexampleValidator,
    CounterexampleMinimizer,
)
from .verdict import (
    EquivalenceVerdict,
    VerdictKind,
    VerdictEvidence,
    VerdictAggregator,
    ConfidenceScore,
    CoverageInfo,
)
from .bounded_checker import (
    BoundedModelChecker,
    BMCConfig,
    BMCResult,
    LoopUnrollStrategy,
    PathEncoder,
)

__all__ = [
    "EquivalenceVerifier", "VerificationConfig", "VerificationSession",
    "FunctionVerificationResult", "ModuleVerificationResult",
    "CounterexampleGenerator", "Counterexample", "ConcreteInput",
    "CounterexampleValidator", "CounterexampleMinimizer",
    "EquivalenceVerdict", "VerdictKind", "VerdictEvidence",
    "VerdictAggregator", "ConfidenceScore", "CoverageInfo",
    "BoundedModelChecker", "BMCConfig", "BMCResult",
    "LoopUnrollStrategy", "PathEncoder",
]
