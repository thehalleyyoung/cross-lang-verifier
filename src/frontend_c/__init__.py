"""
C Frontend for the Cross-Language Equivalence Verifier.

Parses C source code (targeting C2Rust output) into a C-specific AST,
resolves types, and lowers to the shared IR.

Modules:
    lexer          - Tokenize C source into a token stream
    c_ast          - C-specific AST node types
    parser         - Recursive descent C parser
    type_resolver  - C type resolution and layout computation
    ir_lowering    - Lower C AST to shared IR
    preprocessor   - Minimal C preprocessor for C2Rust output
"""

from .lexer import CLexer, Token, TokenKind, SourcePos
from .c_ast import (
    TranslationUnit, FunctionDecl, VarDecl, TypedefDecl,
    StructDecl, UnionDecl, EnumDecl,
)
from .parser import CParser, ParseError
from .type_resolver import CTypeResolver
from .ir_lowering import CIRLowering
from .preprocessor import CPreprocessor

try:
    from .tree_sitter_parser import TreeSitterCParser
except ImportError:
    TreeSitterCParser = None  # tree-sitter-c not installed

__all__ = [
    "CLexer", "Token", "TokenKind", "SourcePos",
    "TranslationUnit", "FunctionDecl", "VarDecl", "TypedefDecl",
    "StructDecl", "UnionDecl", "EnumDecl",
    "CParser", "ParseError",
    "TreeSitterCParser",
    "CTypeResolver",
    "CIRLowering",
    "CPreprocessor",
]
