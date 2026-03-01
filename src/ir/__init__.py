"""
IR (Intermediate Representation) package for the Cross-Language Equivalence Verifier.

Provides a shared typed SSA-based IR that can represent both C and Rust program
semantics. The IR is designed to be language-neutral while preserving enough
information to detect semantic divergences introduced by transpilation.

Modules:
    types        - Complete type system with C/Rust interop
    instructions - All SSA instruction types
    basic_block  - Basic block with CFG edges
    function     - Function with SSA form validation
    module       - Top-level module container
    builder      - Programmatic IR construction
    printer      - Pretty-printing in LLVM-like syntax
    validator    - Structural and semantic IR validation
"""

from .types import (
    IRType,
    IntType,
    FloatType,
    PointerType,
    ArrayType,
    StructType,
    StructField,
    UnionType,
    FunctionType,
    VoidType,
    Signedness,
    FloatKind,
    ProvenanceTag,
    OverflowBehavior,
    Language,
    TypeCompatibility,
    check_compatibility,
    compute_common_type,
    type_join,
    type_meet,
    are_layout_compatible,
    type_from_dict,
)
from .instructions import Instruction, Value, Constant, BinaryOp, UnaryOp, CompareOp
from .basic_block import BasicBlock
from .function import Function
from .module import Module
from .builder import IRBuilder
from .printer import IRPrinter
from .validator import IRValidator

__all__ = [
    "IRType", "IntType", "FloatType", "PointerType", "ArrayType",
    "StructType", "StructField", "UnionType", "FunctionType", "VoidType",
    "Signedness", "FloatKind", "ProvenanceTag", "OverflowBehavior", "Language",
    "TypeCompatibility", "check_compatibility", "compute_common_type",
    "type_join", "type_meet", "are_layout_compatible", "type_from_dict",
    "Instruction", "Value", "Constant", "BinaryOp", "UnaryOp", "CompareOp",
    "BasicBlock", "Function", "Module", "IRBuilder", "IRPrinter", "IRValidator",
]
