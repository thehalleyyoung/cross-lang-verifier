"""
SMT Interface for Cross-Language Equivalence Verification.

Provides Z3-based encoding, solving, decoding, and theory-specific helpers
for bitvectors, floating point, and arrays.
"""

from .encoder import (
    SMTEncoder,
    EncodingContext,
)
from .solver import (
    SMTSolver,
    SolverResult,
    SolverConfig,
)
from .decoder import (
    ModelDecoder,
    ConcreteValue,
    Counterexample,
    TestCase,
)
from .theories import (
    BitvectorTheory,
    FloatingPointTheory,
    ArrayTheory,
)

__all__ = [
    "SMTEncoder",
    "EncodingContext",
    "SMTSolver",
    "SolverResult",
    "SolverConfig",
    "ModelDecoder",
    "ConcreteValue",
    "Counterexample",
    "TestCase",
    "BitvectorTheory",
    "FloatingPointTheory",
    "ArrayTheory",
]
