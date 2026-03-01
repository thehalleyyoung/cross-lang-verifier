"""CLI package for the Cross-Language Equivalence Verifier."""

from .config import VerifyConfig
from .reporter import EquivalenceVerdict, VerdictKind, VerificationReport
from .pipeline import VerificationPipeline

__all__ = [
    "VerifyConfig",
    "EquivalenceVerdict",
    "VerdictKind",
    "VerificationReport",
    "VerificationPipeline",
]
