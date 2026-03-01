"""
Semantics package for the Cross-Language Equivalence Verifier.

Provides semantic divergence tables, configurable execution semantics,
and concrete/abstract evaluation of IR instructions under different
language models (C vs Rust).
"""

from .divergence_table import (
    DivergenceClass,
    DivergenceType,
    CSemantics,
    RustSemantics,
    DivergenceEntry,
    DivergenceTable,
    DivergenceAnalyzer,
)
from .semantic_config import (
    OverflowMode,
    FloatModel,
    ErrorModel,
    PointerModel,
    IntegerPromotionModel,
    SemanticConfig,
)
from .eval import (
    EvalResult,
    EvalFlags,
    SemanticEvaluator,
    Interval,
)

__all__ = [
    "DivergenceClass", "DivergenceType", "CSemantics", "RustSemantics",
    "DivergenceEntry", "DivergenceTable", "DivergenceAnalyzer",
    "OverflowMode", "FloatModel", "ErrorModel", "PointerModel",
    "IntegerPromotionModel", "SemanticConfig",
    "EvalResult", "EvalFlags", "SemanticEvaluator", "Interval",
]
