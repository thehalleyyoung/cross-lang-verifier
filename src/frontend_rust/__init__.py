"""
Rust Frontend for the Cross-Language Equivalence Verifier.

Parses Rust source code (targeting C2Rust output) into a Rust-specific AST,
resolves types, and lowers to the shared IR.

Modules:
    lexer          - Tokenize Rust source into a token stream
    rust_ast       - Rust-specific AST node types
    parser         - Recursive descent Rust parser
    type_resolver  - Rust type resolution
    ir_lowering    - Lower Rust AST to shared IR
"""

from .lexer import RustLexer, Token, TokenKind, SourcePos
from .rust_ast import (
    Crate, FnItem, StructItem, EnumItem, ImplItem,
)
from .parser import RustParser, ParseError
from .type_resolver import RustTypeResolver
from .ir_lowering import RustIRLowering

try:
    from .tree_sitter_parser import TreeSitterRustParser
except ImportError:
    TreeSitterRustParser = None  # tree-sitter-rust not installed

__all__ = [
    "RustLexer", "Token", "TokenKind", "SourcePos",
    "Crate", "FnItem", "StructItem", "EnumItem", "ImplItem",
    "RustParser", "ParseError",
    "TreeSitterRustParser",
    "RustTypeResolver",
    "RustIRLowering",
]
