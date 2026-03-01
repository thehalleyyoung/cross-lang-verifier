"""
Type system package for the Cross-Language Equivalence Verifier.

Provides type checking, integer promotion rules, and type coercion
generation for the IR, supporting both C and Rust type semantics.
"""

from .checker import TypeChecker, TypeCheckResult, TypeCheckError, TypeCheckWarning
from .promotions import (
    PromotionRules,
    CPromotionRules,
    RustPromotionRules,
    PromotionChain,
    PromotionComparison,
)
from .coercions import (
    CoercionGenerator,
    CoercionStep,
    CoercionChain,
    InformationLoss,
)

__all__ = [
    "TypeChecker", "TypeCheckResult", "TypeCheckError", "TypeCheckWarning",
    "PromotionRules", "CPromotionRules", "RustPromotionRules",
    "PromotionChain", "PromotionComparison",
    "CoercionGenerator", "CoercionStep", "CoercionChain", "InformationLoss",
]
