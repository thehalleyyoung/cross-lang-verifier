"""
Product Program Construction for Cross-Language Equivalence Verification.

Constructs a single product program from two aligned IR functions (C and Rust),
with coercion assertions at semantic divergence points.
"""

from .alignment import (
    AlignmentResult,
    BlockAlignment,
    InstructionAlignment,
    AlignmentCost,
    FunctionAligner,
)
from .coercion import (
    CoercionPoint,
    CoercionKind,
    CoercionGenerator,
)
from .product import (
    ProductProgram,
    ProductBlock,
    ProductInstruction,
    ProductBuilder,
)
from .normalizer import (
    IRNormalizer,
    NormalizationPass,
    NormalizationConfig,
)

__all__ = [
    "AlignmentResult",
    "BlockAlignment",
    "InstructionAlignment",
    "AlignmentCost",
    "FunctionAligner",
    "CoercionPoint",
    "CoercionKind",
    "CoercionGenerator",
    "ProductProgram",
    "ProductBlock",
    "ProductInstruction",
    "ProductBuilder",
    "IRNormalizer",
    "NormalizationPass",
    "NormalizationConfig",
]
