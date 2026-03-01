"""Rust parser for cross-language equivalence verifier.

Implements a tokenizer and recursive descent parser for a substantial
subset of Rust, producing a typed AST suitable for equivalence analysis.
"""

from __future__ import annotations
import enum
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Token types
# ---------------------------------------------------------------------------

class TokenType(enum.Enum):
    # Literals
    IDENT = "IDENT"
    INT_LIT = "INT_LIT"
    FLOAT_LIT = "FLOAT_LIT"
    STRING_LIT = "STRING_LIT"
    RAW_STRING_LIT = "RAW_STRING_LIT"
    CHAR_LIT = "CHAR_LIT"
    BOOL_LIT = "BOOL_LIT"
    LIFETIME = "LIFETIME"

    # Keywords
    KW_FN = "fn"
    KW_LET = "let"
    KW_MUT = "mut"
    KW_PUB = "pub"
    KW_STRUCT = "struct"
    KW_ENUM = "enum"
    KW_IMPL = "impl"
    KW_TRAIT = "trait"
    KW_MATCH = "match"
    KW_IF = "if"
    KW_ELSE = "else"
    KW_LOOP = "loop"
    KW_WHILE = "while"
    KW_FOR = "for"
    KW_IN = "in"
    KW_RETURN = "return"
    KW_BREAK = "break"
    KW_CONTINUE = "continue"
    KW_USE = "use"
    KW_MOD = "mod"
    KW_CRATE = "crate"
    KW_SELF_LOWER = "self"
    KW_SUPER = "super"
    KW_SELF_UPPER = "Self"
    KW_AS = "as"
    KW_TYPE = "type"
    KW_WHERE = "where"
    KW_UNSAFE = "unsafe"
    KW_ASYNC = "async"
    KW_AWAIT = "await"
    KW_DYN = "dyn"
    KW_REF = "ref"
    KW_MOVE = "move"
    KW_CONST = "const"
    KW_STATIC = "static"
    KW_EXTERN = "extern"
    KW_TRUE = "true"
    KW_FALSE = "false"

    # Punctuation / operators
    PLUS = "+"
    MINUS = "-"
    STAR = "*"
    SLASH = "/"
    PERCENT = "%"
    AMP = "&"
    PIPE = "|"
    CARET = "^"
    TILDE = "~"
    BANG = "!"
    LT = "<"
    GT = ">"
    EQ = "="
    DOT = "."
    COMMA = ","
    SEMI = ";"
    COLON = ":"
    LPAREN = "("
    RPAREN = ")"
    LBRACE = "{"
    RBRACE = "}"
    LBRACKET = "["
    RBRACKET = "]"
    HASH = "#"
    QUESTION = "?"
    AT = "@"
    UNDERSCORE = "_"

    # Multi-char operators
    ARROW = "->"
    FAT_ARROW = "=>"
    DOUBLE_COLON = "::"
    DOT_DOT = ".."
    DOT_DOT_EQ = "..="
    AMP_AMP = "&&"
    PIPE_PIPE = "||"
    EQ_EQ = "=="
    BANG_EQ = "!="
    LT_EQ = "<="
    GT_EQ = ">="
    PLUS_EQ = "+="
    MINUS_EQ = "-="
    STAR_EQ = "*="
    SLASH_EQ = "/="
    PERCENT_EQ = "%="
    AMP_EQ = "&="
    PIPE_EQ = "|="
    CARET_EQ = "^="
    SHL = "<<"
    SHR = ">>"
    SHL_EQ = "<<="
    SHR_EQ = ">>="
    TURBOFISH = "::<"

    EOF = "EOF"


@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    col: int

    def __repr__(self) -> str:
        return f"Token({self.type}, {self.value!r}, {self.line}:{self.col})"


# ---------------------------------------------------------------------------
# Keyword map
# ---------------------------------------------------------------------------

_KEYWORDS: Dict[str, TokenType] = {
    "fn": TokenType.KW_FN, "let": TokenType.KW_LET, "mut": TokenType.KW_MUT,
    "pub": TokenType.KW_PUB, "struct": TokenType.KW_STRUCT, "enum": TokenType.KW_ENUM,
    "impl": TokenType.KW_IMPL, "trait": TokenType.KW_TRAIT, "match": TokenType.KW_MATCH,
    "if": TokenType.KW_IF, "else": TokenType.KW_ELSE, "loop": TokenType.KW_LOOP,
    "while": TokenType.KW_WHILE, "for": TokenType.KW_FOR, "in": TokenType.KW_IN,
    "return": TokenType.KW_RETURN, "break": TokenType.KW_BREAK,
    "continue": TokenType.KW_CONTINUE, "use": TokenType.KW_USE, "mod": TokenType.KW_MOD,
    "crate": TokenType.KW_CRATE, "self": TokenType.KW_SELF_LOWER,
    "super": TokenType.KW_SUPER, "Self": TokenType.KW_SELF_UPPER,
    "as": TokenType.KW_AS, "type": TokenType.KW_TYPE, "where": TokenType.KW_WHERE,
    "unsafe": TokenType.KW_UNSAFE, "async": TokenType.KW_ASYNC,
    "await": TokenType.KW_AWAIT, "dyn": TokenType.KW_DYN, "ref": TokenType.KW_REF,
    "move": TokenType.KW_MOVE, "const": TokenType.KW_CONST,
    "static": TokenType.KW_STATIC, "extern": TokenType.KW_EXTERN,
    "true": TokenType.KW_TRUE, "false": TokenType.KW_FALSE,
}


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class Tokenizer:
    """Produces a list of Token objects from Rust source code."""

    def __init__(self, source: str) -> None:
        self.source = source
        self.pos = 0
        self.line = 1
        self.col = 1
        self.tokens: List[Token] = []

    # -- helpers --

    def _peek(self, offset: int = 0) -> str:
        idx = self.pos + offset
        if idx < len(self.source):
            return self.source[idx]
        return "\0"

    def _advance(self) -> str:
        ch = self.source[self.pos]
        self.pos += 1
        if ch == "\n":
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return ch

    def _match(self, expected: str) -> bool:
        if self.pos < len(self.source) and self.source[self.pos] == expected:
            self._advance()
            return True
        return False

    def _emit(self, tt: TokenType, value: str, line: int, col: int) -> None:
        self.tokens.append(Token(tt, value, line, col))

    # -- public --

    def tokenize(self) -> List[Token]:
        while self.pos < len(self.source):
            self._skip_whitespace_and_comments()
            if self.pos >= len(self.source):
                break
            start_line, start_col = self.line, self.col
            ch = self._peek()

            if ch == "r" and self._peek(1) in ('"', '#'):
                self._read_raw_string(start_line, start_col)
            elif ch == "b" and self._peek(1) == '"':
                self._read_string(start_line, start_col, prefix="b")
            elif ch == '"':
                self._read_string(start_line, start_col)
            elif ch == "'":
                self._read_char_or_lifetime(start_line, start_col)
            elif ch.isdigit():
                self._read_number(start_line, start_col)
            elif ch == '_' or ch.isalpha():
                self._read_ident_or_keyword(start_line, start_col)
            else:
                self._read_punctuation(start_line, start_col)

        self._emit(TokenType.EOF, "", self.line, self.col)
        return self.tokens

    # -- whitespace / comments --

    def _skip_whitespace_and_comments(self) -> None:
        while self.pos < len(self.source):
            ch = self._peek()
            if ch in (" ", "\t", "\r", "\n"):
                self._advance()
            elif ch == "/" and self._peek(1) == "/":
                while self.pos < len(self.source) and self._peek() != "\n":
                    self._advance()
            elif ch == "/" and self._peek(1) == "*":
                self._advance()
                self._advance()
                depth = 1
                while self.pos < len(self.source) and depth > 0:
                    if self._peek() == "/" and self._peek(1) == "*":
                        self._advance()
                        self._advance()
                        depth += 1
                    elif self._peek() == "*" and self._peek(1) == "/":
                        self._advance()
                        self._advance()
                        depth -= 1
                    else:
                        self._advance()
            else:
                break

    # -- string literals --

    def _read_string(self, sl: int, sc: int, prefix: str = "") -> None:
        if prefix:
            self._advance()  # skip 'b'
        self._advance()  # skip opening '"'
        buf: List[str] = []
        while self.pos < len(self.source) and self._peek() != '"':
            if self._peek() == "\\":
                buf.append(self._advance())
                if self.pos < len(self.source):
                    buf.append(self._advance())
            else:
                buf.append(self._advance())
        if self.pos < len(self.source):
            self._advance()  # skip closing '"'
        self._emit(TokenType.STRING_LIT, prefix + '"' + "".join(buf) + '"', sl, sc)

    def _read_raw_string(self, sl: int, sc: int) -> None:
        self._advance()  # skip 'r'
        hashes = 0
        while self.pos < len(self.source) and self._peek() == '#':
            hashes += 1
            self._advance()
        if self.pos < len(self.source) and self._peek() == '"':
            self._advance()
        buf: List[str] = []
        closing = '"' + '#' * hashes
        while self.pos < len(self.source):
            remaining = self.source[self.pos:]
            if remaining.startswith(closing):
                for _ in range(len(closing)):
                    self._advance()
                break
            buf.append(self._advance())
        self._emit(TokenType.RAW_STRING_LIT, 'r' + '#' * hashes + '"' + "".join(buf) + closing, sl, sc)

    # -- char / lifetime --

    def _read_char_or_lifetime(self, sl: int, sc: int) -> None:
        self._advance()  # skip opening '
        if self.pos < len(self.source) and (self._peek().isalpha() or self._peek() == '_'):
            start = self.pos
            ident_chars: List[str] = []
            while self.pos < len(self.source) and (self._peek().isalnum() or self._peek() == '_'):
                ident_chars.append(self._advance())
            ident = "".join(ident_chars)
            if self.pos < len(self.source) and self._peek() == "'":
                self._advance()
                self._emit(TokenType.CHAR_LIT, "'" + ident + "'", sl, sc)
            else:
                self._emit(TokenType.LIFETIME, "'" + ident, sl, sc)
        elif self.pos < len(self.source) and self._peek() == "\\":
            esc: List[str] = [self._advance()]
            if self.pos < len(self.source):
                esc.append(self._advance())
            if self.pos < len(self.source) and self._peek() == "'":
                self._advance()
            self._emit(TokenType.CHAR_LIT, "'" + "".join(esc) + "'", sl, sc)
        elif self.pos < len(self.source):
            ch = self._advance()
            if self.pos < len(self.source) and self._peek() == "'":
                self._advance()
            self._emit(TokenType.CHAR_LIT, "'" + ch + "'", sl, sc)

    # -- numbers --

    def _read_number(self, sl: int, sc: int) -> None:
        buf: List[str] = []
        is_float = False
        if self._peek() == '0' and self._peek(1) in ('x', 'X'):
            buf.append(self._advance())
            buf.append(self._advance())
            while self.pos < len(self.source) and (self._peek() in "0123456789abcdefABCDEF_"):
                buf.append(self._advance())
            self._emit(TokenType.INT_LIT, "".join(buf), sl, sc)
            return
        if self._peek() == '0' and self._peek(1) in ('b', 'B'):
            buf.append(self._advance())
            buf.append(self._advance())
            while self.pos < len(self.source) and self._peek() in "01_":
                buf.append(self._advance())
            self._emit(TokenType.INT_LIT, "".join(buf), sl, sc)
            return
        if self._peek() == '0' and self._peek(1) in ('o', 'O'):
            buf.append(self._advance())
            buf.append(self._advance())
            while self.pos < len(self.source) and self._peek() in "01234567_":
                buf.append(self._advance())
            self._emit(TokenType.INT_LIT, "".join(buf), sl, sc)
            return
        while self.pos < len(self.source) and (self._peek().isdigit() or self._peek() == '_'):
            buf.append(self._advance())
        if self.pos < len(self.source) and self._peek() == '.' and self._peek(1) != '.':
            if self._peek(1).isdigit():
                is_float = True
                buf.append(self._advance())  # '.'
                while self.pos < len(self.source) and (self._peek().isdigit() or self._peek() == '_'):
                    buf.append(self._advance())
        if self.pos < len(self.source) and self._peek() in ('e', 'E'):
            is_float = True
            buf.append(self._advance())
            if self.pos < len(self.source) and self._peek() in ('+', '-'):
                buf.append(self._advance())
            while self.pos < len(self.source) and (self._peek().isdigit() or self._peek() == '_'):
                buf.append(self._advance())
        # type suffix (i32, u64, f64, etc.)
        if self.pos < len(self.source) and self._peek() in ('i', 'u', 'f'):
            suffix_start = self.pos
            suffix_chars: List[str] = [self._advance()]
            while self.pos < len(self.source) and self._peek().isdigit():
                suffix_chars.append(self._advance())
            buf.extend(suffix_chars)
            if suffix_chars[0] == 'f':
                is_float = True
        self._emit(TokenType.FLOAT_LIT if is_float else TokenType.INT_LIT, "".join(buf), sl, sc)

    # -- identifiers / keywords --

    def _read_ident_or_keyword(self, sl: int, sc: int) -> None:
        buf: List[str] = []
        while self.pos < len(self.source) and (self._peek().isalnum() or self._peek() == '_'):
            buf.append(self._advance())
        word = "".join(buf)
        if word in _KEYWORDS:
            self._emit(_KEYWORDS[word], word, sl, sc)
        elif word == "_":
            self._emit(TokenType.UNDERSCORE, word, sl, sc)
        else:
            self._emit(TokenType.IDENT, word, sl, sc)

    # -- punctuation / operators --

    def _read_punctuation(self, sl: int, sc: int) -> None:
        ch = self._advance()
        two = ch + self._peek() if self.pos < len(self.source) else ch
        three = two + (self.source[self.pos + 1] if self.pos + 1 < len(self.source) else "")

        # three-char operators
        if three == "..=":
            self._advance(); self._advance()
            self._emit(TokenType.DOT_DOT_EQ, "..=", sl, sc); return
        if three == "<<=":
            self._advance(); self._advance()
            self._emit(TokenType.SHL_EQ, "<<=", sl, sc); return
        if three == ">>=":
            self._advance(); self._advance()
            self._emit(TokenType.SHR_EQ, ">>=", sl, sc); return
        if three == "::<":
            self._advance(); self._advance()
            self._emit(TokenType.TURBOFISH, "::<", sl, sc); return

        # two-char operators
        _two_map = {
            "->": TokenType.ARROW, "=>": TokenType.FAT_ARROW, "::": TokenType.DOUBLE_COLON,
            "..": TokenType.DOT_DOT, "&&": TokenType.AMP_AMP, "||": TokenType.PIPE_PIPE,
            "==": TokenType.EQ_EQ, "!=": TokenType.BANG_EQ, "<=": TokenType.LT_EQ,
            ">=": TokenType.GT_EQ, "+=": TokenType.PLUS_EQ, "-=": TokenType.MINUS_EQ,
            "*=": TokenType.STAR_EQ, "/=": TokenType.SLASH_EQ, "%=": TokenType.PERCENT_EQ,
            "&=": TokenType.AMP_EQ, "|=": TokenType.PIPE_EQ, "^=": TokenType.CARET_EQ,
            "<<": TokenType.SHL, ">>": TokenType.SHR,
        }
        if two in _two_map:
            self._advance()
            self._emit(_two_map[two], two, sl, sc); return

        # single-char
        _one_map = {
            "+": TokenType.PLUS, "-": TokenType.MINUS, "*": TokenType.STAR,
            "/": TokenType.SLASH, "%": TokenType.PERCENT, "&": TokenType.AMP,
            "|": TokenType.PIPE, "^": TokenType.CARET, "~": TokenType.TILDE,
            "!": TokenType.BANG, "<": TokenType.LT, ">": TokenType.GT,
            "=": TokenType.EQ, ".": TokenType.DOT, ",": TokenType.COMMA,
            ";": TokenType.SEMI, ":": TokenType.COLON, "(": TokenType.LPAREN,
            ")": TokenType.RPAREN, "{": TokenType.LBRACE, "}": TokenType.RBRACE,
            "[": TokenType.LBRACKET, "]": TokenType.RBRACKET, "#": TokenType.HASH,
            "?": TokenType.QUESTION, "@": TokenType.AT, "_": TokenType.UNDERSCORE,
        }
        if ch in _one_map:
            self._emit(_one_map[ch], ch, sl, sc); return
        # fallback – emit as ident so we don't crash
        self._emit(TokenType.IDENT, ch, sl, sc)


# ---------------------------------------------------------------------------
# AST node types
# ---------------------------------------------------------------------------

@dataclass
class TypePath:
    segments: List[str] = field(default_factory=list)
    generic_args: Optional["GenericArgs"] = None
    is_ref: bool = False
    is_mut_ref: bool = False
    lifetime: Optional[str] = None
    is_dyn: bool = False
    is_impl_trait: bool = False

    def __str__(self) -> str:
        base = "::".join(self.segments)
        if self.generic_args:
            base += str(self.generic_args)
        prefix = ""
        if self.is_ref:
            prefix = "&"
            if self.lifetime:
                prefix += self.lifetime + " "
            if self.is_mut_ref:
                prefix += "mut "
        if self.is_dyn:
            prefix += "dyn "
        if self.is_impl_trait:
            prefix += "impl "
        return prefix + base


@dataclass
class GenericArgs:
    args: List[Union[TypePath, "TypePath"]] = field(default_factory=list)
    lifetime_args: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        parts = list(self.lifetime_args) + [str(a) for a in self.args]
        return "<" + ", ".join(parts) + ">"


@dataclass
class GenericParam:
    name: str
    bounds: List[TypePath] = field(default_factory=list)


@dataclass
class WhereClause:
    predicates: List[Tuple[TypePath, List[TypePath]]] = field(default_factory=list)


# -- Patterns --

@dataclass
class LitPattern:
    value: str

@dataclass
class IdentPattern:
    name: str
    is_ref: bool = False
    is_mut: bool = False
    bound: Optional["Pattern"] = None

@dataclass
class TuplePattern:
    elements: List["Pattern"] = field(default_factory=list)

@dataclass
class StructPattern:
    path: List[str] = field(default_factory=list)
    fields: List[Tuple[str, Optional["Pattern"]]] = field(default_factory=list)
    has_rest: bool = False

@dataclass
class EnumPattern:
    path: List[str] = field(default_factory=list)
    args: List["Pattern"] = field(default_factory=list)

@dataclass
class WildcardPattern:
    pass

@dataclass
class RefPattern:
    inner: "Pattern" = None  # type: ignore[assignment]
    is_mut: bool = False

Pattern = Union[LitPattern, IdentPattern, TuplePattern, StructPattern,
                EnumPattern, WildcardPattern, RefPattern]

# -- Expressions --

@dataclass
class LitExpr:
    value: str
    kind: str = "int"  # int, float, string, char, bool

@dataclass
class PathExpr:
    segments: List[str] = field(default_factory=list)
    turbofish: Optional[GenericArgs] = None

@dataclass
class BinaryExpr:
    op: str = ""
    left: Any = None
    right: Any = None

@dataclass
class UnaryExpr:
    op: str = ""
    operand: Any = None

@dataclass
class CallExpr:
    func: Any = None
    args: List[Any] = field(default_factory=list)

@dataclass
class MethodCallExpr:
    receiver: Any = None
    method: str = ""
    turbofish: Optional[GenericArgs] = None
    args: List[Any] = field(default_factory=list)

@dataclass
class FieldAccessExpr:
    receiver: Any = None
    field_name: str = ""

@dataclass
class IndexExpr:
    receiver: Any = None
    index: Any = None

@dataclass
class RefExpr:
    expr: Any = None
    is_mut: bool = False

@dataclass
class DerefExpr:
    expr: Any = None

@dataclass
class TupleExpr:
    elements: List[Any] = field(default_factory=list)

@dataclass
class ArrayExpr:
    elements: List[Any] = field(default_factory=list)
    repeat_count: Any = None  # [expr; count] form

@dataclass
class RangeExpr:
    start: Any = None
    end: Any = None
    inclusive: bool = False

@dataclass
class ClosureExpr:
    params: List[Tuple[str, Optional[TypePath]]] = field(default_factory=list)
    body: Any = None
    is_move: bool = False

@dataclass
class ReturnExpr:
    value: Any = None

@dataclass
class BreakExpr:
    value: Any = None

@dataclass
class ContinueExpr:
    pass

Expr = Union[LitExpr, PathExpr, BinaryExpr, UnaryExpr, CallExpr,
             MethodCallExpr, FieldAccessExpr, IndexExpr, RefExpr,
             DerefExpr, TupleExpr, ArrayExpr, RangeExpr, ClosureExpr,
             ReturnExpr, BreakExpr, ContinueExpr, "IfExpr", "MatchExpr",
             "Block", "ForLoop", "WhileLoop", "Loop"]

# -- Statements / items --

@dataclass
class Block:
    stmts: List[Any] = field(default_factory=list)
    final_expr: Any = None

@dataclass
class LetBinding:
    pattern: Optional[Pattern] = None
    type_ann: Optional[TypePath] = None
    init: Any = None
    is_mut: bool = False

@dataclass
class MatchArm:
    pattern: Optional[Pattern] = None
    guard: Any = None
    body: Any = None

@dataclass
class MatchExpr:
    scrutinee: Any = None
    arms: List[MatchArm] = field(default_factory=list)

@dataclass
class IfExpr:
    condition: Any = None
    let_pattern: Optional[Pattern] = None
    then_block: Optional[Block] = None
    else_block: Any = None  # Block or IfExpr

@dataclass
class WhileLoop:
    condition: Any = None
    let_pattern: Optional[Pattern] = None
    body: Optional[Block] = None

@dataclass
class ForLoop:
    pattern: Optional[Pattern] = None
    iter_expr: Any = None
    body: Optional[Block] = None

@dataclass
class Loop:
    body: Optional[Block] = None

@dataclass
class FnParam:
    name: str = ""
    type_ann: Optional[TypePath] = None
    is_self: bool = False
    is_mut_self: bool = False
    is_ref_self: bool = False

@dataclass
class FnDef:
    name: str = ""
    generic_params: List[GenericParam] = field(default_factory=list)
    params: List[FnParam] = field(default_factory=list)
    return_type: Optional[TypePath] = None
    where_clause: Optional[WhereClause] = None
    body: Optional[Block] = None
    is_pub: bool = False
    is_async: bool = False
    is_unsafe: bool = False
    attributes: List[str] = field(default_factory=list)

@dataclass
class StructField:
    name: str = ""
    type_ann: Optional[TypePath] = None
    is_pub: bool = False

@dataclass
class StructDef:
    name: str = ""
    generic_params: List[GenericParam] = field(default_factory=list)
    fields: List[StructField] = field(default_factory=list)
    is_tuple_struct: bool = False
    is_pub: bool = False
    where_clause: Optional[WhereClause] = None
    attributes: List[str] = field(default_factory=list)

@dataclass
class EnumVariant:
    name: str = ""
    fields: List[StructField] = field(default_factory=list)
    is_tuple: bool = False
    discriminant: Any = None

@dataclass
class EnumDef:
    name: str = ""
    generic_params: List[GenericParam] = field(default_factory=list)
    variants: List[EnumVariant] = field(default_factory=list)
    is_pub: bool = False
    where_clause: Optional[WhereClause] = None
    attributes: List[str] = field(default_factory=list)

@dataclass
class ImplBlock:
    type_path: Optional[TypePath] = None
    trait_path: Optional[TypePath] = None
    generic_params: List[GenericParam] = field(default_factory=list)
    where_clause: Optional[WhereClause] = None
    items: List[Any] = field(default_factory=list)
    attributes: List[str] = field(default_factory=list)

@dataclass
class TraitDef:
    name: str = ""
    generic_params: List[GenericParam] = field(default_factory=list)
    super_traits: List[TypePath] = field(default_factory=list)
    where_clause: Optional[WhereClause] = None
    items: List[Any] = field(default_factory=list)
    is_pub: bool = False
    attributes: List[str] = field(default_factory=list)

@dataclass
class UseDecl:
    path: List[str] = field(default_factory=list)
    alias: Optional[str] = None
    is_glob: bool = False
    is_pub: bool = False

@dataclass
class ConstDef:
    name: str = ""
    type_ann: Optional[TypePath] = None
    init: Any = None
    is_pub: bool = False
    is_static: bool = False

@dataclass
class Attribute:
    content: str = ""


# ---------------------------------------------------------------------------
# RustAST – top-level container
# ---------------------------------------------------------------------------

@dataclass
class RustAST:
    functions: List[FnDef] = field(default_factory=list)
    structs: List[StructDef] = field(default_factory=list)
    enums: List[EnumDef] = field(default_factory=list)
    impls: List[ImplBlock] = field(default_factory=list)
    traits: List[TraitDef] = field(default_factory=list)
    use_decls: List[UseDecl] = field(default_factory=list)
    consts: List[ConstDef] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Recursive-descent parser
# ---------------------------------------------------------------------------

class ParseError(Exception):
    def __init__(self, msg: str, token: Optional[Token] = None) -> None:
        loc = f" at {token.line}:{token.col}" if token else ""
        super().__init__(f"{msg}{loc}")
        self.token = token


class Parser:
    """Recursive descent parser for Rust source tokens."""

    def __init__(self, tokens: List[Token]) -> None:
        self.tokens = tokens
        self.pos = 0

    # -- helpers --

    def _cur(self) -> Token:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return Token(TokenType.EOF, "", 0, 0)

    def _peek_type(self, offset: int = 0) -> TokenType:
        idx = self.pos + offset
        if idx < len(self.tokens):
            return self.tokens[idx].type
        return TokenType.EOF

    def _eat(self, tt: TokenType) -> Token:
        tok = self._cur()
        if tok.type != tt:
            raise ParseError(f"expected {tt.value!r}, got {tok.value!r}", tok)
        self.pos += 1
        return tok

    def _eat_ident(self) -> str:
        tok = self._cur()
        if tok.type == TokenType.IDENT:
            self.pos += 1
            return tok.value
        if tok.type in (TokenType.KW_SELF_LOWER, TokenType.KW_SELF_UPPER,
                        TokenType.KW_SUPER, TokenType.KW_CRATE):
            self.pos += 1
            return tok.value
        raise ParseError(f"expected identifier, got {tok.value!r}", tok)

    def _at(self, *types: TokenType) -> bool:
        return self._cur().type in types

    def _try_eat(self, tt: TokenType) -> Optional[Token]:
        if self._cur().type == tt:
            tok = self._cur()
            self.pos += 1
            return tok
        return None

    # -- top-level parsing --

    def parse(self) -> RustAST:
        ast = RustAST()
        while not self._at(TokenType.EOF):
            attrs = self._parse_outer_attrs()
            is_pub = bool(self._try_eat(TokenType.KW_PUB))
            tt = self._cur().type
            if tt == TokenType.KW_FN or (tt == TokenType.KW_ASYNC and self._peek_type(1) == TokenType.KW_FN) or \
               (tt == TokenType.KW_UNSAFE and self._peek_type(1) == TokenType.KW_FN):
                fn_def = self._parse_fn_def(is_pub, attrs)
                ast.functions.append(fn_def)
            elif tt == TokenType.KW_STRUCT:
                ast.structs.append(self._parse_struct_def(is_pub, attrs))
            elif tt == TokenType.KW_ENUM:
                ast.enums.append(self._parse_enum_def(is_pub, attrs))
            elif tt == TokenType.KW_IMPL:
                ast.impls.append(self._parse_impl_block(attrs))
            elif tt == TokenType.KW_TRAIT:
                ast.traits.append(self._parse_trait_def(is_pub, attrs))
            elif tt == TokenType.KW_USE:
                ast.use_decls.append(self._parse_use_decl(is_pub))
            elif tt in (TokenType.KW_CONST, TokenType.KW_STATIC):
                ast.consts.append(self._parse_const_def(is_pub))
            elif tt == TokenType.KW_MOD:
                self._skip_mod()
            elif tt == TokenType.KW_EXTERN:
                self._skip_extern_block()
            elif tt == TokenType.KW_TYPE:
                self._skip_type_alias()
            else:
                self.pos += 1  # skip unknown token
        return ast

    # -- attributes --

    def _parse_outer_attrs(self) -> List[str]:
        attrs: List[str] = []
        while self._at(TokenType.HASH) and self._peek_type(1) == TokenType.LBRACKET:
            self._eat(TokenType.HASH)
            self._eat(TokenType.LBRACKET)
            depth = 1
            buf: List[str] = []
            while depth > 0 and not self._at(TokenType.EOF):
                if self._at(TokenType.LBRACKET):
                    depth += 1
                elif self._at(TokenType.RBRACKET):
                    depth -= 1
                    if depth == 0:
                        break
                buf.append(self._cur().value)
                self.pos += 1
            self._eat(TokenType.RBRACKET)
            attrs.append("".join(buf))
        return attrs

    # -- use declarations --

    def _parse_use_decl(self, is_pub: bool) -> UseDecl:
        self._eat(TokenType.KW_USE)
        path: List[str] = []
        alias: Optional[str] = None
        is_glob = False
        path.append(self._eat_ident())
        while self._try_eat(TokenType.DOUBLE_COLON):
            if self._at(TokenType.STAR):
                self.pos += 1
                is_glob = True
                break
            if self._at(TokenType.LBRACE):
                # use group – simplified: skip to semicolon
                depth = 0
                while not self._at(TokenType.SEMI, TokenType.EOF):
                    if self._at(TokenType.LBRACE):
                        depth += 1
                    elif self._at(TokenType.RBRACE):
                        depth -= 1
                    self.pos += 1
                break
            path.append(self._eat_ident())
        if self._try_eat(TokenType.KW_AS):
            alias = self._eat_ident()
        self._try_eat(TokenType.SEMI)
        return UseDecl(path=path, alias=alias, is_glob=is_glob, is_pub=is_pub)

    # -- const / static --

    def _parse_const_def(self, is_pub: bool) -> ConstDef:
        is_static = bool(self._try_eat(TokenType.KW_STATIC))
        if not is_static:
            self._eat(TokenType.KW_CONST)
        self._try_eat(TokenType.KW_MUT)
        name = self._eat_ident()
        self._eat(TokenType.COLON)
        type_ann = self._parse_type()
        init = None
        if self._try_eat(TokenType.EQ):
            init = self._parse_expr()
        self._try_eat(TokenType.SEMI)
        return ConstDef(name=name, type_ann=type_ann, init=init, is_pub=is_pub, is_static=is_static)

    # -- skip helpers for less important items --

    def _skip_mod(self) -> None:
        self._eat(TokenType.KW_MOD)
        self._eat_ident()
        if self._at(TokenType.LBRACE):
            self._skip_braces()
        else:
            self._try_eat(TokenType.SEMI)

    def _skip_extern_block(self) -> None:
        self._eat(TokenType.KW_EXTERN)
        if self._at(TokenType.STRING_LIT):
            self.pos += 1
        if self._at(TokenType.LBRACE):
            self._skip_braces()
        elif self._at(TokenType.KW_FN):
            fn_def = self._parse_fn_def(False, [])
            # just discard
        else:
            self._try_eat(TokenType.SEMI)

    def _skip_type_alias(self) -> None:
        self._eat(TokenType.KW_TYPE)
        self._eat_ident()
        # skip generic params
        if self._at(TokenType.LT):
            self._skip_angle_brackets()
        if self._try_eat(TokenType.EQ):
            while not self._at(TokenType.SEMI, TokenType.EOF):
                self.pos += 1
        self._try_eat(TokenType.SEMI)

    def _skip_braces(self) -> None:
        self._eat(TokenType.LBRACE)
        depth = 1
        while depth > 0 and not self._at(TokenType.EOF):
            if self._at(TokenType.LBRACE):
                depth += 1
            elif self._at(TokenType.RBRACE):
                depth -= 1
            self.pos += 1

    def _skip_angle_brackets(self) -> None:
        self._eat(TokenType.LT)
        depth = 1
        while depth > 0 and not self._at(TokenType.EOF):
            if self._at(TokenType.LT):
                depth += 1
            elif self._at(TokenType.GT):
                depth -= 1
            self.pos += 1

    # -- generic parameters --

    def _parse_generic_params(self) -> List[GenericParam]:
        params: List[GenericParam] = []
        if not self._at(TokenType.LT):
            return params
        self._eat(TokenType.LT)
        while not self._at(TokenType.GT, TokenType.EOF):
            if self._at(TokenType.LIFETIME):
                lt = self._cur().value
                self.pos += 1
                params.append(GenericParam(name=lt))
                self._try_eat(TokenType.COMMA)
                continue
            name = self._eat_ident()
            bounds: List[TypePath] = []
            if self._try_eat(TokenType.COLON):
                bounds.append(self._parse_type())
                while self._try_eat(TokenType.PLUS):
                    bounds.append(self._parse_type())
            params.append(GenericParam(name=name, bounds=bounds))
            if not self._try_eat(TokenType.COMMA):
                break
        self._eat(TokenType.GT)
        return params

    def _parse_generic_args(self) -> GenericArgs:
        args = GenericArgs()
        self._eat(TokenType.LT)
        while not self._at(TokenType.GT, TokenType.EOF):
            if self._at(TokenType.LIFETIME):
                args.lifetime_args.append(self._cur().value)
                self.pos += 1
            else:
                args.args.append(self._parse_type())
            if not self._try_eat(TokenType.COMMA):
                break
        self._eat(TokenType.GT)
        return args

    # -- where clause --

    def _parse_where_clause(self) -> Optional[WhereClause]:
        if not self._try_eat(TokenType.KW_WHERE):
            return None
        wc = WhereClause()
        while not self._at(TokenType.LBRACE, TokenType.SEMI, TokenType.EOF):
            ty = self._parse_type()
            self._eat(TokenType.COLON)
            bounds: List[TypePath] = [self._parse_type()]
            while self._try_eat(TokenType.PLUS):
                bounds.append(self._parse_type())
            wc.predicates.append((ty, bounds))
            if not self._try_eat(TokenType.COMMA):
                break
        return wc

    # -- types --

    def _parse_type(self) -> TypePath:
        tp = TypePath()
        if self._try_eat(TokenType.AMP):
            tp.is_ref = True
            if self._at(TokenType.LIFETIME):
                tp.lifetime = self._cur().value
                self.pos += 1
            if self._try_eat(TokenType.KW_MUT):
                tp.is_mut_ref = True
        if self._try_eat(TokenType.KW_DYN):
            tp.is_dyn = True
        if self._try_eat(TokenType.KW_IMPL):
            tp.is_impl_trait = True
        if self._at(TokenType.LPAREN):
            # tuple type or unit
            self._eat(TokenType.LPAREN)
            if self._try_eat(TokenType.RPAREN):
                tp.segments = ["()"]
                return tp
            inner = [self._parse_type()]
            while self._try_eat(TokenType.COMMA):
                if self._at(TokenType.RPAREN):
                    break
                inner.append(self._parse_type())
            self._eat(TokenType.RPAREN)
            tp.segments = ["(" + ", ".join(str(t) for t in inner) + ")"]
            return tp
        if self._at(TokenType.LBRACKET):
            self._eat(TokenType.LBRACKET)
            inner_ty = self._parse_type()
            size = None
            if self._try_eat(TokenType.SEMI):
                size = self._cur().value
                self.pos += 1
            self._eat(TokenType.RBRACKET)
            if size:
                tp.segments = [f"[{inner_ty}; {size}]"]
            else:
                tp.segments = [f"[{inner_ty}]"]
            return tp
        if self._at(TokenType.KW_FN):
            self._eat(TokenType.KW_FN)
            self._eat(TokenType.LPAREN)
            param_types: List[TypePath] = []
            while not self._at(TokenType.RPAREN, TokenType.EOF):
                param_types.append(self._parse_type())
                self._try_eat(TokenType.COMMA)
            self._eat(TokenType.RPAREN)
            ret = None
            if self._try_eat(TokenType.ARROW):
                ret = self._parse_type()
            sig = "fn(" + ", ".join(str(p) for p in param_types) + ")"
            if ret:
                sig += " -> " + str(ret)
            tp.segments = [sig]
            return tp
        # path type
        tp.segments.append(self._eat_ident())
        while self._at(TokenType.DOUBLE_COLON) and self._peek_type(1) not in (TokenType.LT,):
            self._eat(TokenType.DOUBLE_COLON)
            if self._at(TokenType.IDENT) or self._cur().type in (
                    TokenType.KW_SELF_UPPER, TokenType.KW_SELF_LOWER,
                    TokenType.KW_SUPER, TokenType.KW_CRATE):
                tp.segments.append(self._eat_ident())
            else:
                break
        if self._at(TokenType.LT):
            saved = self.pos
            try:
                tp.generic_args = self._parse_generic_args()
            except ParseError:
                self.pos = saved
        if self._try_eat(TokenType.ARROW):
            ret_ty = self._parse_type()
            tp.segments[-1] += " -> " + str(ret_ty)
        return tp

    # -- fn definition --

    def _parse_fn_def(self, is_pub: bool, attrs: List[str]) -> FnDef:
        fn = FnDef(is_pub=is_pub, attributes=attrs)
        if self._try_eat(TokenType.KW_ASYNC):
            fn.is_async = True
        if self._try_eat(TokenType.KW_UNSAFE):
            fn.is_unsafe = True
        self._eat(TokenType.KW_FN)
        fn.name = self._eat_ident()
        fn.generic_params = self._parse_generic_params()
        self._eat(TokenType.LPAREN)
        fn.params = self._parse_fn_params()
        self._eat(TokenType.RPAREN)
        if self._try_eat(TokenType.ARROW):
            fn.return_type = self._parse_type()
        fn.where_clause = self._parse_where_clause()
        if self._at(TokenType.LBRACE):
            fn.body = self._parse_block()
        else:
            self._try_eat(TokenType.SEMI)
        return fn

    def _parse_fn_params(self) -> List[FnParam]:
        params: List[FnParam] = []
        while not self._at(TokenType.RPAREN, TokenType.EOF):
            p = FnParam()
            if self._at(TokenType.AMP):
                self._eat(TokenType.AMP)
                is_mut = bool(self._try_eat(TokenType.KW_MUT))
                if self._at(TokenType.KW_SELF_LOWER):
                    self._eat(TokenType.KW_SELF_LOWER)
                    p.is_self = True
                    p.is_ref_self = True
                    p.is_mut_self = is_mut
                    p.name = "self"
                    params.append(p)
                    self._try_eat(TokenType.COMMA)
                    continue
                else:
                    # not self, back up and parse normally
                    self.pos -= (2 if is_mut else 1)
            if self._at(TokenType.KW_MUT) and self._peek_type(1) == TokenType.KW_SELF_LOWER:
                self._eat(TokenType.KW_MUT)
                self._eat(TokenType.KW_SELF_LOWER)
                p.is_self = True
                p.is_mut_self = True
                p.name = "self"
                params.append(p)
                self._try_eat(TokenType.COMMA)
                continue
            if self._at(TokenType.KW_SELF_LOWER) and self._peek_type(1) in (TokenType.COMMA, TokenType.RPAREN):
                self._eat(TokenType.KW_SELF_LOWER)
                p.is_self = True
                p.name = "self"
                params.append(p)
                self._try_eat(TokenType.COMMA)
                continue
            if self._try_eat(TokenType.KW_MUT):
                pass  # mut binding
            p.name = self._eat_ident()
            self._eat(TokenType.COLON)
            p.type_ann = self._parse_type()
            params.append(p)
            if not self._try_eat(TokenType.COMMA):
                break
        return params

    # -- struct --

    def _parse_struct_def(self, is_pub: bool, attrs: List[str]) -> StructDef:
        self._eat(TokenType.KW_STRUCT)
        sd = StructDef(is_pub=is_pub, attributes=attrs)
        sd.name = self._eat_ident()
        sd.generic_params = self._parse_generic_params()
        sd.where_clause = self._parse_where_clause()
        if self._at(TokenType.LPAREN):
            # tuple struct
            sd.is_tuple_struct = True
            self._eat(TokenType.LPAREN)
            idx = 0
            while not self._at(TokenType.RPAREN, TokenType.EOF):
                fp = bool(self._try_eat(TokenType.KW_PUB))
                ty = self._parse_type()
                sd.fields.append(StructField(name=str(idx), type_ann=ty, is_pub=fp))
                idx += 1
                self._try_eat(TokenType.COMMA)
            self._eat(TokenType.RPAREN)
            self._try_eat(TokenType.SEMI)
        elif self._at(TokenType.LBRACE):
            self._eat(TokenType.LBRACE)
            while not self._at(TokenType.RBRACE, TokenType.EOF):
                self._parse_outer_attrs()  # field attrs
                fp = bool(self._try_eat(TokenType.KW_PUB))
                fname = self._eat_ident()
                self._eat(TokenType.COLON)
                fty = self._parse_type()
                sd.fields.append(StructField(name=fname, type_ann=fty, is_pub=fp))
                self._try_eat(TokenType.COMMA)
            self._eat(TokenType.RBRACE)
        else:
            self._try_eat(TokenType.SEMI)  # unit struct
        return sd

    # -- enum --

    def _parse_enum_def(self, is_pub: bool, attrs: List[str]) -> EnumDef:
        self._eat(TokenType.KW_ENUM)
        ed = EnumDef(is_pub=is_pub, attributes=attrs)
        ed.name = self._eat_ident()
        ed.generic_params = self._parse_generic_params()
        ed.where_clause = self._parse_where_clause()
        self._eat(TokenType.LBRACE)
        while not self._at(TokenType.RBRACE, TokenType.EOF):
            self._parse_outer_attrs()
            v = EnumVariant()
            v.name = self._eat_ident()
            if self._at(TokenType.LPAREN):
                v.is_tuple = True
                self._eat(TokenType.LPAREN)
                idx = 0
                while not self._at(TokenType.RPAREN, TokenType.EOF):
                    ty = self._parse_type()
                    v.fields.append(StructField(name=str(idx), type_ann=ty))
                    idx += 1
                    self._try_eat(TokenType.COMMA)
                self._eat(TokenType.RPAREN)
            elif self._at(TokenType.LBRACE):
                self._eat(TokenType.LBRACE)
                while not self._at(TokenType.RBRACE, TokenType.EOF):
                    fname = self._eat_ident()
                    self._eat(TokenType.COLON)
                    fty = self._parse_type()
                    v.fields.append(StructField(name=fname, type_ann=fty))
                    self._try_eat(TokenType.COMMA)
                self._eat(TokenType.RBRACE)
            if self._try_eat(TokenType.EQ):
                v.discriminant = self._parse_expr()
            ed.variants.append(v)
            self._try_eat(TokenType.COMMA)
        self._eat(TokenType.RBRACE)
        return ed

    # -- impl --

    def _parse_impl_block(self, attrs: List[str]) -> ImplBlock:
        self._eat(TokenType.KW_IMPL)
        ib = ImplBlock(attributes=attrs)
        ib.generic_params = self._parse_generic_params()
        first_type = self._parse_type()
        if self._try_eat(TokenType.KW_FOR):
            ib.trait_path = first_type
            ib.type_path = self._parse_type()
        else:
            ib.type_path = first_type
        ib.where_clause = self._parse_where_clause()
        self._eat(TokenType.LBRACE)
        while not self._at(TokenType.RBRACE, TokenType.EOF):
            item_attrs = self._parse_outer_attrs()
            ip = bool(self._try_eat(TokenType.KW_PUB))
            tt = self._cur().type
            if tt == TokenType.KW_FN or tt == TokenType.KW_ASYNC or tt == TokenType.KW_UNSAFE:
                ib.items.append(self._parse_fn_def(ip, item_attrs))
            elif tt == TokenType.KW_CONST:
                ib.items.append(self._parse_const_def(ip))
            elif tt == TokenType.KW_TYPE:
                self._skip_type_alias()
            else:
                self.pos += 1
        self._eat(TokenType.RBRACE)
        return ib

    # -- trait --

    def _parse_trait_def(self, is_pub: bool, attrs: List[str]) -> TraitDef:
        self._eat(TokenType.KW_TRAIT)
        td = TraitDef(is_pub=is_pub, attributes=attrs)
        td.name = self._eat_ident()
        td.generic_params = self._parse_generic_params()
        if self._try_eat(TokenType.COLON):
            td.super_traits.append(self._parse_type())
            while self._try_eat(TokenType.PLUS):
                td.super_traits.append(self._parse_type())
        td.where_clause = self._parse_where_clause()
        self._eat(TokenType.LBRACE)
        while not self._at(TokenType.RBRACE, TokenType.EOF):
            item_attrs = self._parse_outer_attrs()
            ip = bool(self._try_eat(TokenType.KW_PUB))
            tt = self._cur().type
            if tt == TokenType.KW_FN or tt == TokenType.KW_ASYNC or tt == TokenType.KW_UNSAFE:
                td.items.append(self._parse_fn_def(ip, item_attrs))
            elif tt == TokenType.KW_TYPE:
                self._skip_type_alias()
            elif tt == TokenType.KW_CONST:
                td.items.append(self._parse_const_def(ip))
            else:
                self.pos += 1
        self._eat(TokenType.RBRACE)
        return td

    # -- block --

    def _parse_block(self) -> Block:
        self._eat(TokenType.LBRACE)
        block = Block()
        while not self._at(TokenType.RBRACE, TokenType.EOF):
            stmt = self._parse_stmt()
            if stmt is not None:
                if self._at(TokenType.RBRACE) and not isinstance(stmt, (LetBinding, UseDecl, ConstDef)):
                    block.final_expr = stmt
                else:
                    block.stmts.append(stmt)
        self._eat(TokenType.RBRACE)
        return block

    def _parse_stmt(self) -> Any:
        tt = self._cur().type
        if tt == TokenType.KW_LET:
            lb = self._parse_let_binding()
            self._try_eat(TokenType.SEMI)
            return lb
        if tt == TokenType.KW_RETURN:
            self._eat(TokenType.KW_RETURN)
            val = None
            if not self._at(TokenType.SEMI, TokenType.RBRACE, TokenType.EOF):
                val = self._parse_expr()
            self._try_eat(TokenType.SEMI)
            return ReturnExpr(value=val)
        if tt == TokenType.KW_BREAK:
            self._eat(TokenType.KW_BREAK)
            val = None
            if not self._at(TokenType.SEMI, TokenType.RBRACE, TokenType.EOF):
                val = self._parse_expr()
            self._try_eat(TokenType.SEMI)
            return BreakExpr(value=val)
        if tt == TokenType.KW_CONTINUE:
            self._eat(TokenType.KW_CONTINUE)
            self._try_eat(TokenType.SEMI)
            return ContinueExpr()
        expr = self._parse_expr()
        self._try_eat(TokenType.SEMI)
        return expr

    # -- let binding --

    def _parse_let_binding(self) -> LetBinding:
        self._eat(TokenType.KW_LET)
        lb = LetBinding()
        lb.is_mut = bool(self._try_eat(TokenType.KW_MUT))
        lb.pattern = self._parse_pattern()
        if self._try_eat(TokenType.COLON):
            lb.type_ann = self._parse_type()
        if self._try_eat(TokenType.EQ):
            lb.init = self._parse_expr()
        return lb

    # -- patterns --

    def _parse_pattern(self) -> Pattern:
        if self._try_eat(TokenType.UNDERSCORE):
            return WildcardPattern()
        if self._at(TokenType.AMP):
            self._eat(TokenType.AMP)
            is_mut = bool(self._try_eat(TokenType.KW_MUT))
            inner = self._parse_pattern()
            return RefPattern(inner=inner, is_mut=is_mut)
        if self._at(TokenType.KW_REF):
            self._eat(TokenType.KW_REF)
            is_mut = bool(self._try_eat(TokenType.KW_MUT))
            inner = self._parse_pattern()
            return RefPattern(inner=inner, is_mut=is_mut)
        if self._at(TokenType.LPAREN):
            self._eat(TokenType.LPAREN)
            elems: List[Pattern] = []
            while not self._at(TokenType.RPAREN, TokenType.EOF):
                elems.append(self._parse_pattern())
                if not self._try_eat(TokenType.COMMA):
                    break
            self._eat(TokenType.RPAREN)
            return TuplePattern(elements=elems)
        if self._at(TokenType.INT_LIT, TokenType.FLOAT_LIT):
            val = self._cur().value
            self.pos += 1
            return LitPattern(value=val)
        if self._at(TokenType.MINUS) and self._peek_type(1) in (TokenType.INT_LIT, TokenType.FLOAT_LIT):
            self._eat(TokenType.MINUS)
            val = self._cur().value
            self.pos += 1
            return LitPattern(value="-" + val)
        if self._at(TokenType.STRING_LIT, TokenType.CHAR_LIT, TokenType.RAW_STRING_LIT):
            val = self._cur().value
            self.pos += 1
            return LitPattern(value=val)
        if self._at(TokenType.KW_TRUE):
            self.pos += 1
            return LitPattern(value="true")
        if self._at(TokenType.KW_FALSE):
            self.pos += 1
            return LitPattern(value="false")
        if self._at(TokenType.KW_MUT):
            self._eat(TokenType.KW_MUT)
            name = self._eat_ident()
            bound = None
            if self._try_eat(TokenType.AT):
                bound = self._parse_pattern()
            return IdentPattern(name=name, is_mut=True, bound=bound)
        # path-based patterns (ident, enum, struct)
        segments: List[str] = [self._eat_ident()]
        while self._at(TokenType.DOUBLE_COLON) and self._peek_type(1) not in (TokenType.LT,):
            self._eat(TokenType.DOUBLE_COLON)
            segments.append(self._eat_ident())
        if self._at(TokenType.LPAREN):
            self._eat(TokenType.LPAREN)
            args: List[Pattern] = []
            while not self._at(TokenType.RPAREN, TokenType.EOF):
                args.append(self._parse_pattern())
                if not self._try_eat(TokenType.COMMA):
                    break
            self._eat(TokenType.RPAREN)
            return EnumPattern(path=segments, args=args)
        if self._at(TokenType.LBRACE):
            self._eat(TokenType.LBRACE)
            fields: List[Tuple[str, Optional[Pattern]]] = []
            has_rest = False
            while not self._at(TokenType.RBRACE, TokenType.EOF):
                if self._try_eat(TokenType.DOT_DOT):
                    has_rest = True
                    break
                fname = self._eat_ident()
                fpat: Optional[Pattern] = None
                if self._try_eat(TokenType.COLON):
                    fpat = self._parse_pattern()
                fields.append((fname, fpat))
                if not self._try_eat(TokenType.COMMA):
                    break
            self._eat(TokenType.RBRACE)
            return StructPattern(path=segments, fields=fields, has_rest=has_rest)
        if len(segments) == 1:
            name = segments[0]
            bound = None
            if self._try_eat(TokenType.AT):
                bound = self._parse_pattern()
            return IdentPattern(name=name, bound=bound)
        return EnumPattern(path=segments, args=[])

    # -- expressions with precedence --

    _PRECEDENCE: Dict[TokenType, int] = {
        TokenType.EQ: 1, TokenType.PLUS_EQ: 1, TokenType.MINUS_EQ: 1,
        TokenType.STAR_EQ: 1, TokenType.SLASH_EQ: 1, TokenType.PERCENT_EQ: 1,
        TokenType.AMP_EQ: 1, TokenType.PIPE_EQ: 1, TokenType.CARET_EQ: 1,
        TokenType.SHL_EQ: 1, TokenType.SHR_EQ: 1,
        TokenType.PIPE_PIPE: 3,
        TokenType.AMP_AMP: 4,
        TokenType.EQ_EQ: 5, TokenType.BANG_EQ: 5,
        TokenType.LT: 6, TokenType.GT: 6, TokenType.LT_EQ: 6, TokenType.GT_EQ: 6,
        TokenType.PIPE: 7,
        TokenType.CARET: 8,
        TokenType.AMP: 9,
        TokenType.SHL: 10, TokenType.SHR: 10,
        TokenType.PLUS: 11, TokenType.MINUS: 11,
        TokenType.STAR: 12, TokenType.SLASH: 12, TokenType.PERCENT: 12,
        TokenType.KW_AS: 13,
    }

    def _parse_expr(self) -> Any:
        return self._parse_range_expr()

    def _parse_range_expr(self) -> Any:
        if self._at(TokenType.DOT_DOT) or self._at(TokenType.DOT_DOT_EQ):
            inclusive = self._cur().type == TokenType.DOT_DOT_EQ
            self.pos += 1
            end = None
            if not self._at(TokenType.SEMI, TokenType.COMMA, TokenType.RPAREN,
                            TokenType.RBRACE, TokenType.RBRACKET, TokenType.EOF):
                end = self._parse_prec_expr(2)
            return RangeExpr(start=None, end=end, inclusive=inclusive)
        left = self._parse_prec_expr(2)
        if self._at(TokenType.DOT_DOT) or self._at(TokenType.DOT_DOT_EQ):
            inclusive = self._cur().type == TokenType.DOT_DOT_EQ
            self.pos += 1
            end = None
            if not self._at(TokenType.SEMI, TokenType.COMMA, TokenType.RPAREN,
                            TokenType.RBRACE, TokenType.RBRACKET, TokenType.EOF):
                end = self._parse_prec_expr(2)
            return RangeExpr(start=left, end=end, inclusive=inclusive)
        return left

    def _parse_prec_expr(self, min_prec: int) -> Any:
        left = self._parse_unary_expr()
        while True:
            tt = self._cur().type
            prec = self._PRECEDENCE.get(tt)
            if prec is None or prec < min_prec:
                break
            if tt == TokenType.KW_AS:
                self.pos += 1
                cast_ty = self._parse_type()
                left = BinaryExpr(op="as", left=left, right=cast_ty)
                continue
            op_tok = self._cur()
            self.pos += 1
            # right-associative for assignment operators
            next_prec = prec + 1 if prec > 1 else prec
            right = self._parse_prec_expr(next_prec)
            left = BinaryExpr(op=op_tok.value, left=left, right=right)
        return left

    def _parse_unary_expr(self) -> Any:
        if self._at(TokenType.MINUS):
            self._eat(TokenType.MINUS)
            operand = self._parse_unary_expr()
            return UnaryExpr(op="-", operand=operand)
        if self._at(TokenType.BANG):
            self._eat(TokenType.BANG)
            operand = self._parse_unary_expr()
            return UnaryExpr(op="!", operand=operand)
        if self._at(TokenType.STAR):
            self._eat(TokenType.STAR)
            operand = self._parse_unary_expr()
            return DerefExpr(expr=operand)
        if self._at(TokenType.AMP):
            self._eat(TokenType.AMP)
            is_mut = bool(self._try_eat(TokenType.KW_MUT))
            operand = self._parse_unary_expr()
            return RefExpr(expr=operand, is_mut=is_mut)
        return self._parse_postfix_expr()

    def _parse_postfix_expr(self) -> Any:
        expr = self._parse_primary_expr()
        while True:
            if self._try_eat(TokenType.QUESTION):
                expr = UnaryExpr(op="?", operand=expr)
            elif self._at(TokenType.DOT):
                self._eat(TokenType.DOT)
                if self._at(TokenType.INT_LIT):
                    idx = self._cur().value
                    self.pos += 1
                    expr = FieldAccessExpr(receiver=expr, field_name=idx)
                elif self._try_eat(TokenType.KW_AWAIT):
                    expr = MethodCallExpr(receiver=expr, method="await", args=[])
                else:
                    method = self._eat_ident()
                    turbo: Optional[GenericArgs] = None
                    if self._at(TokenType.TURBOFISH):
                        self._eat(TokenType.TURBOFISH)
                        turbo = self._finish_generic_args()
                    if self._at(TokenType.LPAREN):
                        self._eat(TokenType.LPAREN)
                        args = self._parse_call_args()
                        self._eat(TokenType.RPAREN)
                        expr = MethodCallExpr(receiver=expr, method=method,
                                              turbofish=turbo, args=args)
                    else:
                        expr = FieldAccessExpr(receiver=expr, field_name=method)
            elif self._at(TokenType.LBRACKET):
                self._eat(TokenType.LBRACKET)
                index = self._parse_expr()
                self._eat(TokenType.RBRACKET)
                expr = IndexExpr(receiver=expr, index=index)
            elif self._at(TokenType.LPAREN) and self._is_callable(expr):
                self._eat(TokenType.LPAREN)
                args = self._parse_call_args()
                self._eat(TokenType.RPAREN)
                expr = CallExpr(func=expr, args=args)
            else:
                break
        return expr

    def _finish_generic_args(self) -> GenericArgs:
        args = GenericArgs()
        while not self._at(TokenType.GT, TokenType.EOF):
            if self._at(TokenType.LIFETIME):
                args.lifetime_args.append(self._cur().value)
                self.pos += 1
            else:
                args.args.append(self._parse_type())
            if not self._try_eat(TokenType.COMMA):
                break
        self._eat(TokenType.GT)
        return args

    def _is_callable(self, expr: Any) -> bool:
        return isinstance(expr, (PathExpr, FieldAccessExpr, CallExpr, MethodCallExpr, IndexExpr))

    def _parse_call_args(self) -> List[Any]:
        args: List[Any] = []
        while not self._at(TokenType.RPAREN, TokenType.EOF):
            args.append(self._parse_expr())
            if not self._try_eat(TokenType.COMMA):
                break
        return args

    def _parse_primary_expr(self) -> Any:
        tt = self._cur().type

        # closure: |args| body  or  move |args| body
        if tt == TokenType.KW_MOVE and self._peek_type(1) == TokenType.PIPE:
            self._eat(TokenType.KW_MOVE)
            return self._parse_closure(is_move=True)
        if tt == TokenType.PIPE:
            return self._parse_closure(is_move=False)
        if tt == TokenType.PIPE_PIPE:
            # empty closure || body
            self._eat(TokenType.PIPE_PIPE)
            body = self._parse_expr()
            return ClosureExpr(params=[], body=body, is_move=False)

        # block expression
        if tt == TokenType.LBRACE:
            return self._parse_block()
        if tt == TokenType.KW_UNSAFE and self._peek_type(1) == TokenType.LBRACE:
            self._eat(TokenType.KW_UNSAFE)
            return self._parse_block()

        # if / if let
        if tt == TokenType.KW_IF:
            return self._parse_if_expr()

        # match
        if tt == TokenType.KW_MATCH:
            return self._parse_match_expr()

        # loop
        if tt == TokenType.KW_LOOP:
            self._eat(TokenType.KW_LOOP)
            body = self._parse_block()
            return Loop(body=body)

        # while / while let
        if tt == TokenType.KW_WHILE:
            return self._parse_while_loop()

        # for
        if tt == TokenType.KW_FOR:
            return self._parse_for_loop()

        # return (expression position)
        if tt == TokenType.KW_RETURN:
            self._eat(TokenType.KW_RETURN)
            val = None
            if not self._at(TokenType.SEMI, TokenType.RBRACE, TokenType.COMMA, TokenType.EOF):
                val = self._parse_expr()
            return ReturnExpr(value=val)

        # break (expression position)
        if tt == TokenType.KW_BREAK:
            self._eat(TokenType.KW_BREAK)
            val = None
            if not self._at(TokenType.SEMI, TokenType.RBRACE, TokenType.COMMA, TokenType.EOF):
                val = self._parse_expr()
            return BreakExpr(value=val)

        if tt == TokenType.KW_CONTINUE:
            self._eat(TokenType.KW_CONTINUE)
            return ContinueExpr()

        # literals
        if tt == TokenType.INT_LIT:
            tok = self._cur(); self.pos += 1
            return LitExpr(value=tok.value, kind="int")
        if tt == TokenType.FLOAT_LIT:
            tok = self._cur(); self.pos += 1
            return LitExpr(value=tok.value, kind="float")
        if tt in (TokenType.STRING_LIT, TokenType.RAW_STRING_LIT):
            tok = self._cur(); self.pos += 1
            return LitExpr(value=tok.value, kind="string")
        if tt == TokenType.CHAR_LIT:
            tok = self._cur(); self.pos += 1
            return LitExpr(value=tok.value, kind="char")
        if tt in (TokenType.KW_TRUE, TokenType.KW_FALSE):
            tok = self._cur(); self.pos += 1
            return LitExpr(value=tok.value, kind="bool")

        # tuple / grouped expr
        if tt == TokenType.LPAREN:
            self._eat(TokenType.LPAREN)
            if self._try_eat(TokenType.RPAREN):
                return TupleExpr(elements=[])
            first = self._parse_expr()
            if self._try_eat(TokenType.COMMA):
                elems = [first]
                while not self._at(TokenType.RPAREN, TokenType.EOF):
                    elems.append(self._parse_expr())
                    if not self._try_eat(TokenType.COMMA):
                        break
                self._eat(TokenType.RPAREN)
                return TupleExpr(elements=elems)
            self._eat(TokenType.RPAREN)
            return first

        # array
        if tt == TokenType.LBRACKET:
            self._eat(TokenType.LBRACKET)
            if self._try_eat(TokenType.RBRACKET):
                return ArrayExpr(elements=[])
            first = self._parse_expr()
            if self._try_eat(TokenType.SEMI):
                count = self._parse_expr()
                self._eat(TokenType.RBRACKET)
                return ArrayExpr(elements=[first], repeat_count=count)
            elems = [first]
            while self._try_eat(TokenType.COMMA):
                if self._at(TokenType.RBRACKET):
                    break
                elems.append(self._parse_expr())
            self._eat(TokenType.RBRACKET)
            return ArrayExpr(elements=elems)

        # path / ident (including turbofish)
        if tt in (TokenType.IDENT, TokenType.KW_SELF_LOWER, TokenType.KW_SELF_UPPER,
                  TokenType.KW_SUPER, TokenType.KW_CRATE):
            return self._parse_path_expr()

        # fallback: consume token
        tok = self._cur()
        self.pos += 1
        return LitExpr(value=tok.value, kind="unknown")

    def _parse_path_expr(self) -> Any:
        segments: List[str] = [self._eat_ident()]
        turbo: Optional[GenericArgs] = None
        while self._at(TokenType.DOUBLE_COLON):
            self._eat(TokenType.DOUBLE_COLON)
            if self._at(TokenType.IDENT) or self._cur().type in (
                    TokenType.KW_SELF_LOWER, TokenType.KW_SELF_UPPER,
                    TokenType.KW_SUPER, TokenType.KW_CRATE):
                segments.append(self._eat_ident())
            elif self._at(TokenType.LT):
                # could be turbofish-like from ::< that was split
                break
            else:
                break
        if self._at(TokenType.TURBOFISH):
            self._eat(TokenType.TURBOFISH)
            turbo = self._finish_generic_args()
        # struct literal: Path { field: val, ... }
        if self._at(TokenType.LBRACE) and len(segments) >= 1 and segments[-1][0:1].isupper():
            return self._parse_struct_literal(segments)
        return PathExpr(segments=segments, turbofish=turbo)

    def _parse_struct_literal(self, segments: List[str]) -> Any:
        self._eat(TokenType.LBRACE)
        fields: List[Tuple[str, Any]] = []
        has_rest = False
        while not self._at(TokenType.RBRACE, TokenType.EOF):
            if self._at(TokenType.DOT_DOT):
                self._eat(TokenType.DOT_DOT)
                rest_expr = self._parse_expr()
                has_rest = True
                break
            if self._at(TokenType.IDENT) and self._peek_type(1) == TokenType.COLON:
                fname = self._eat_ident()
                self._eat(TokenType.COLON)
                fval = self._parse_expr()
                fields.append((fname, fval))
            elif self._at(TokenType.IDENT):
                fname = self._eat_ident()
                fields.append((fname, PathExpr(segments=[fname])))
            else:
                self.pos += 1
            self._try_eat(TokenType.COMMA)
        self._eat(TokenType.RBRACE)
        # return as a call-like node with path
        call = CallExpr(func=PathExpr(segments=segments),
                        args=[TupleExpr(elements=[v for _, v in fields])])
        return call

    def _parse_closure(self, is_move: bool) -> ClosureExpr:
        self._eat(TokenType.PIPE)
        params: List[Tuple[str, Optional[TypePath]]] = []
        while not self._at(TokenType.PIPE, TokenType.EOF):
            name = self._eat_ident()
            ty: Optional[TypePath] = None
            if self._try_eat(TokenType.COLON):
                ty = self._parse_type()
            params.append((name, ty))
            if not self._try_eat(TokenType.COMMA):
                break
        self._eat(TokenType.PIPE)
        if self._at(TokenType.LBRACE):
            body = self._parse_block()
        else:
            body = self._parse_expr()
        return ClosureExpr(params=params, body=body, is_move=is_move)

    # -- if --

    def _parse_if_expr(self) -> IfExpr:
        self._eat(TokenType.KW_IF)
        ie = IfExpr()
        if self._at(TokenType.KW_LET):
            self._eat(TokenType.KW_LET)
            ie.let_pattern = self._parse_pattern()
            self._eat(TokenType.EQ)
            ie.condition = self._parse_expr_no_struct()
        else:
            ie.condition = self._parse_expr_no_struct()
        ie.then_block = self._parse_block()
        if self._try_eat(TokenType.KW_ELSE):
            if self._at(TokenType.KW_IF):
                ie.else_block = self._parse_if_expr()
            else:
                ie.else_block = self._parse_block()
        return ie

    def _parse_expr_no_struct(self) -> Any:
        """Parse expression but stop before { to avoid ambiguity with block."""
        return self._parse_range_expr()

    # -- match --

    def _parse_match_expr(self) -> MatchExpr:
        self._eat(TokenType.KW_MATCH)
        scrutinee = self._parse_expr_no_struct()
        self._eat(TokenType.LBRACE)
        arms: List[MatchArm] = []
        while not self._at(TokenType.RBRACE, TokenType.EOF):
            arm = MatchArm()
            arm.pattern = self._parse_pattern()
            guard = None
            if self._try_eat(TokenType.KW_IF):
                guard = self._parse_expr()
            arm.guard = guard
            self._eat(TokenType.FAT_ARROW)
            arm.body = self._parse_expr()
            self._try_eat(TokenType.COMMA)
            arms.append(arm)
        self._eat(TokenType.RBRACE)
        return MatchExpr(scrutinee=scrutinee, arms=arms)

    # -- while --

    def _parse_while_loop(self) -> WhileLoop:
        self._eat(TokenType.KW_WHILE)
        wl = WhileLoop()
        if self._at(TokenType.KW_LET):
            self._eat(TokenType.KW_LET)
            wl.let_pattern = self._parse_pattern()
            self._eat(TokenType.EQ)
            wl.condition = self._parse_expr_no_struct()
        else:
            wl.condition = self._parse_expr_no_struct()
        wl.body = self._parse_block()
        return wl

    # -- for --

    def _parse_for_loop(self) -> ForLoop:
        self._eat(TokenType.KW_FOR)
        fl = ForLoop()
        fl.pattern = self._parse_pattern()
        self._eat(TokenType.KW_IN)
        fl.iter_expr = self._parse_expr_no_struct()
        fl.body = self._parse_block()
        return fl


# ---------------------------------------------------------------------------
# Visitor pattern
# ---------------------------------------------------------------------------

class ASTVisitor:
    """Base visitor – override methods to process specific node types."""

    def visit(self, node: Any) -> Any:
        method_name = "visit_" + type(node).__name__
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node: Any) -> Any:
        if isinstance(node, RustAST):
            for fn in node.functions:
                self.visit(fn)
            for st in node.structs:
                self.visit(st)
            for en in node.enums:
                self.visit(en)
            for im in node.impls:
                self.visit(im)
            for tr in node.traits:
                self.visit(tr)
            for ud in node.use_decls:
                self.visit(ud)
            for cd in node.consts:
                self.visit(cd)
        elif isinstance(node, FnDef):
            for p in node.params:
                self.visit(p)
            if node.body:
                self.visit(node.body)
        elif isinstance(node, Block):
            for s in node.stmts:
                self.visit(s)
            if node.final_expr:
                self.visit(node.final_expr)
        elif isinstance(node, LetBinding):
            if node.pattern:
                self.visit(node.pattern)
            if node.init:
                self.visit(node.init)
        elif isinstance(node, BinaryExpr):
            self.visit(node.left)
            self.visit(node.right)
        elif isinstance(node, UnaryExpr):
            self.visit(node.operand)
        elif isinstance(node, CallExpr):
            self.visit(node.func)
            for a in node.args:
                self.visit(a)
        elif isinstance(node, MethodCallExpr):
            self.visit(node.receiver)
            for a in node.args:
                self.visit(a)
        elif isinstance(node, FieldAccessExpr):
            self.visit(node.receiver)
        elif isinstance(node, IndexExpr):
            self.visit(node.receiver)
            self.visit(node.index)
        elif isinstance(node, RefExpr):
            self.visit(node.expr)
        elif isinstance(node, DerefExpr):
            self.visit(node.expr)
        elif isinstance(node, IfExpr):
            if node.condition:
                self.visit(node.condition)
            if node.then_block:
                self.visit(node.then_block)
            if node.else_block:
                self.visit(node.else_block)
        elif isinstance(node, MatchExpr):
            self.visit(node.scrutinee)
            for arm in node.arms:
                self.visit(arm)
        elif isinstance(node, MatchArm):
            if node.pattern:
                self.visit(node.pattern)
            if node.guard:
                self.visit(node.guard)
            if node.body:
                self.visit(node.body)
        elif isinstance(node, ForLoop):
            if node.pattern:
                self.visit(node.pattern)
            if node.iter_expr:
                self.visit(node.iter_expr)
            if node.body:
                self.visit(node.body)
        elif isinstance(node, WhileLoop):
            if node.condition:
                self.visit(node.condition)
            if node.body:
                self.visit(node.body)
        elif isinstance(node, Loop):
            if node.body:
                self.visit(node.body)
        elif isinstance(node, ReturnExpr):
            if node.value:
                self.visit(node.value)
        elif isinstance(node, BreakExpr):
            if node.value:
                self.visit(node.value)
        elif isinstance(node, ClosureExpr):
            if node.body:
                self.visit(node.body)
        elif isinstance(node, TupleExpr):
            for e in node.elements:
                self.visit(e)
        elif isinstance(node, ArrayExpr):
            for e in node.elements:
                self.visit(e)
            if node.repeat_count:
                self.visit(node.repeat_count)
        elif isinstance(node, RangeExpr):
            if node.start:
                self.visit(node.start)
            if node.end:
                self.visit(node.end)
        elif isinstance(node, ImplBlock):
            for item in node.items:
                self.visit(item)
        elif isinstance(node, TraitDef):
            for item in node.items:
                self.visit(item)
        elif isinstance(node, StructDef):
            for f in node.fields:
                self.visit(f)
        elif isinstance(node, EnumDef):
            for v in node.variants:
                self.visit(v)
        return node


# ---------------------------------------------------------------------------
# Pretty printer – reconstruct Rust source from AST
# ---------------------------------------------------------------------------

class ASTPrettyPrinter(ASTVisitor):
    """Reconstruct Rust-like source from AST nodes."""

    def __init__(self) -> None:
        self._indent = 0
        self._lines: List[str] = []

    def _i(self) -> str:
        return "    " * self._indent

    def _emit_line(self, text: str) -> None:
        self._lines.append(self._i() + text)

    def result(self) -> str:
        return "\n".join(self._lines)

    # -- top level --

    def visit_RustAST(self, node: RustAST) -> Any:
        for ud in node.use_decls:
            self.visit(ud)
        if node.use_decls and (node.consts or node.structs or node.enums or node.functions or node.traits or node.impls):
            self._lines.append("")
        for cd in node.consts:
            self.visit(cd)
        for st in node.structs:
            self.visit(st)
            self._lines.append("")
        for en in node.enums:
            self.visit(en)
            self._lines.append("")
        for tr in node.traits:
            self.visit(tr)
            self._lines.append("")
        for fn in node.functions:
            self.visit(fn)
            self._lines.append("")
        for im in node.impls:
            self.visit(im)
            self._lines.append("")
        return node

    def visit_UseDecl(self, node: UseDecl) -> Any:
        path = "::".join(node.path)
        if node.is_glob:
            path += "::*"
        suffix = f" as {node.alias}" if node.alias else ""
        pub = "pub " if node.is_pub else ""
        self._emit_line(f"{pub}use {path}{suffix};")
        return node

    def visit_ConstDef(self, node: ConstDef) -> Any:
        kw = "static" if node.is_static else "const"
        pub = "pub " if node.is_pub else ""
        ty = f": {node.type_ann}" if node.type_ann else ""
        init = f" = {self._expr_str(node.init)}" if node.init else ""
        self._emit_line(f"{pub}{kw} {node.name}{ty}{init};")
        return node

    # -- struct --

    def visit_StructDef(self, node: StructDef) -> Any:
        pub = "pub " if node.is_pub else ""
        generics = self._generics_str(node.generic_params)
        if node.is_tuple_struct:
            fields = ", ".join(
                f"{'pub ' if f.is_pub else ''}{f.type_ann}" for f in node.fields
            )
            self._emit_line(f"{pub}struct {node.name}{generics}({fields});")
        elif node.fields:
            self._emit_line(f"{pub}struct {node.name}{generics} {{")
            self._indent += 1
            for f in node.fields:
                fpub = "pub " if f.is_pub else ""
                self._emit_line(f"{fpub}{f.name}: {f.type_ann},")
            self._indent -= 1
            self._emit_line("}")
        else:
            self._emit_line(f"{pub}struct {node.name}{generics};")
        return node

    # -- enum --

    def visit_EnumDef(self, node: EnumDef) -> Any:
        pub = "pub " if node.is_pub else ""
        generics = self._generics_str(node.generic_params)
        self._emit_line(f"{pub}enum {node.name}{generics} {{")
        self._indent += 1
        for v in node.variants:
            if v.is_tuple and v.fields:
                types = ", ".join(str(f.type_ann) for f in v.fields)
                self._emit_line(f"{v.name}({types}),")
            elif v.fields:
                self._emit_line(f"{v.name} {{")
                self._indent += 1
                for f in v.fields:
                    self._emit_line(f"{f.name}: {f.type_ann},")
                self._indent -= 1
                self._emit_line("},")
            elif v.discriminant is not None:
                self._emit_line(f"{v.name} = {self._expr_str(v.discriminant)},")
            else:
                self._emit_line(f"{v.name},")
        self._indent -= 1
        self._emit_line("}")
        return node

    # -- fn --

    def visit_FnDef(self, node: FnDef) -> Any:
        for attr in node.attributes:
            self._emit_line(f"#[{attr}]")
        pub = "pub " if node.is_pub else ""
        async_ = "async " if node.is_async else ""
        unsafe_ = "unsafe " if node.is_unsafe else ""
        generics = self._generics_str(node.generic_params)
        params = ", ".join(self._param_str(p) for p in node.params)
        ret = f" -> {node.return_type}" if node.return_type else ""
        where = ""
        if node.where_clause and node.where_clause.predicates:
            preds = ", ".join(
                f"{ty}: {' + '.join(str(b) for b in bounds)}"
                for ty, bounds in node.where_clause.predicates
            )
            where = f" where {preds}"
        if node.body:
            self._emit_line(f"{pub}{async_}{unsafe_}fn {node.name}{generics}({params}){ret}{where} {{")
            self._indent += 1
            self._print_block_contents(node.body)
            self._indent -= 1
            self._emit_line("}")
        else:
            self._emit_line(f"{pub}{async_}{unsafe_}fn {node.name}{generics}({params}){ret}{where};")
        return node

    def _param_str(self, p: FnParam) -> str:
        if p.is_self:
            if p.is_ref_self:
                return "&mut self" if p.is_mut_self else "&self"
            return "mut self" if p.is_mut_self else "self"
        ty = f": {p.type_ann}" if p.type_ann else ""
        return f"{p.name}{ty}"

    # -- impl --

    def visit_ImplBlock(self, node: ImplBlock) -> Any:
        generics = self._generics_str(node.generic_params)
        if node.trait_path:
            self._emit_line(f"impl{generics} {node.trait_path} for {node.type_path} {{")
        else:
            self._emit_line(f"impl{generics} {node.type_path} {{")
        self._indent += 1
        for item in node.items:
            self.visit(item)
            self._lines.append("")
        self._indent -= 1
        self._emit_line("}")
        return node

    # -- trait --

    def visit_TraitDef(self, node: TraitDef) -> Any:
        pub = "pub " if node.is_pub else ""
        generics = self._generics_str(node.generic_params)
        supers = ""
        if node.super_traits:
            supers = ": " + " + ".join(str(s) for s in node.super_traits)
        self._emit_line(f"{pub}trait {node.name}{generics}{supers} {{")
        self._indent += 1
        for item in node.items:
            self.visit(item)
        self._indent -= 1
        self._emit_line("}")
        return node

    # -- block / stmts --

    def _print_block_contents(self, block: Block) -> None:
        for s in block.stmts:
            self._print_stmt(s)
        if block.final_expr:
            self._emit_line(self._expr_str(block.final_expr))

    def _print_stmt(self, s: Any) -> None:
        if isinstance(s, LetBinding):
            mut = "mut " if s.is_mut else ""
            pat = self._pattern_str(s.pattern) if s.pattern else "_"
            ty = f": {s.type_ann}" if s.type_ann else ""
            init = f" = {self._expr_str(s.init)}" if s.init else ""
            self._emit_line(f"let {mut}{pat}{ty}{init};")
        elif isinstance(s, ReturnExpr):
            val = f" {self._expr_str(s.value)}" if s.value else ""
            self._emit_line(f"return{val};")
        elif isinstance(s, BreakExpr):
            val = f" {self._expr_str(s.value)}" if s.value else ""
            self._emit_line(f"break{val};")
        elif isinstance(s, ContinueExpr):
            self._emit_line("continue;")
        else:
            self._emit_line(f"{self._expr_str(s)};")

    # -- expression to string --

    def _expr_str(self, e: Any) -> str:
        if e is None:
            return ""
        if isinstance(e, LitExpr):
            return e.value
        if isinstance(e, PathExpr):
            base = "::".join(e.segments)
            if e.turbofish:
                base += str(e.turbofish)
            return base
        if isinstance(e, BinaryExpr):
            return f"{self._expr_str(e.left)} {e.op} {self._expr_str(e.right)}"
        if isinstance(e, UnaryExpr):
            if e.op == "?":
                return f"{self._expr_str(e.operand)}?"
            return f"{e.op}{self._expr_str(e.operand)}"
        if isinstance(e, CallExpr):
            args = ", ".join(self._expr_str(a) for a in e.args)
            return f"{self._expr_str(e.func)}({args})"
        if isinstance(e, MethodCallExpr):
            args = ", ".join(self._expr_str(a) for a in e.args)
            turbo = str(e.turbofish) if e.turbofish else ""
            return f"{self._expr_str(e.receiver)}.{e.method}{turbo}({args})"
        if isinstance(e, FieldAccessExpr):
            return f"{self._expr_str(e.receiver)}.{e.field_name}"
        if isinstance(e, IndexExpr):
            return f"{self._expr_str(e.receiver)}[{self._expr_str(e.index)}]"
        if isinstance(e, RefExpr):
            mut = "mut " if e.is_mut else ""
            return f"&{mut}{self._expr_str(e.expr)}"
        if isinstance(e, DerefExpr):
            return f"*{self._expr_str(e.expr)}"
        if isinstance(e, TupleExpr):
            elems = ", ".join(self._expr_str(x) for x in e.elements)
            return f"({elems})"
        if isinstance(e, ArrayExpr):
            if e.repeat_count is not None:
                return f"[{self._expr_str(e.elements[0])}; {self._expr_str(e.repeat_count)}]"
            elems = ", ".join(self._expr_str(x) for x in e.elements)
            return f"[{elems}]"
        if isinstance(e, RangeExpr):
            s = self._expr_str(e.start) if e.start else ""
            en = self._expr_str(e.end) if e.end else ""
            op = "..=" if e.inclusive else ".."
            return f"{s}{op}{en}"
        if isinstance(e, ClosureExpr):
            params = ", ".join(
                f"{n}: {t}" if t else n for n, t in e.params
            )
            mv = "move " if e.is_move else ""
            return f"{mv}|{params}| {self._expr_str(e.body)}"
        if isinstance(e, ReturnExpr):
            val = f" {self._expr_str(e.value)}" if e.value else ""
            return f"return{val}"
        if isinstance(e, BreakExpr):
            val = f" {self._expr_str(e.value)}" if e.value else ""
            return f"break{val}"
        if isinstance(e, ContinueExpr):
            return "continue"
        if isinstance(e, IfExpr):
            return self._if_str(e)
        if isinstance(e, MatchExpr):
            return self._match_str(e)
        if isinstance(e, Block):
            parts = []
            for s in e.stmts:
                parts.append(self._expr_str(s))
            if e.final_expr:
                parts.append(self._expr_str(e.final_expr))
            return "{ " + "; ".join(parts) + " }"
        if isinstance(e, ForLoop):
            pat = self._pattern_str(e.pattern) if e.pattern else "_"
            return f"for {pat} in {self._expr_str(e.iter_expr)} {{ ... }}"
        if isinstance(e, WhileLoop):
            return f"while {self._expr_str(e.condition)} {{ ... }}"
        if isinstance(e, Loop):
            return "loop { ... }"
        if isinstance(e, TypePath):
            return str(e)
        return str(e)

    def _if_str(self, e: IfExpr) -> str:
        cond = self._expr_str(e.condition) if e.condition else ""
        if e.let_pattern:
            cond = f"let {self._pattern_str(e.let_pattern)} = {cond}"
        result = f"if {cond} {{ ... }}"
        if e.else_block:
            if isinstance(e.else_block, IfExpr):
                result += f" else {self._if_str(e.else_block)}"
            else:
                result += " else { ... }"
        return result

    def _match_str(self, e: MatchExpr) -> str:
        return f"match {self._expr_str(e.scrutinee)} {{ ... }}"

    # -- pattern to string --

    def _pattern_str(self, p: Any) -> str:
        if p is None:
            return "_"
        if isinstance(p, WildcardPattern):
            return "_"
        if isinstance(p, LitPattern):
            return p.value
        if isinstance(p, IdentPattern):
            prefix = ""
            if p.is_ref:
                prefix = "ref "
            if p.is_mut:
                prefix = "mut "
            result = f"{prefix}{p.name}"
            if p.bound:
                result += f" @ {self._pattern_str(p.bound)}"
            return result
        if isinstance(p, TuplePattern):
            elems = ", ".join(self._pattern_str(e) for e in p.elements)
            return f"({elems})"
        if isinstance(p, StructPattern):
            path = "::".join(p.path)
            fields_str = ", ".join(
                f"{n}: {self._pattern_str(v)}" if v else n
                for n, v in p.fields
            )
            rest = ", .." if p.has_rest else ""
            return f"{path} {{ {fields_str}{rest} }}"
        if isinstance(p, EnumPattern):
            path = "::".join(p.path)
            if p.args:
                args = ", ".join(self._pattern_str(a) for a in p.args)
                return f"{path}({args})"
            return path
        if isinstance(p, RefPattern):
            mut = "mut " if p.is_mut else ""
            return f"&{mut}{self._pattern_str(p.inner)}"
        return str(p)

    # -- helpers --

    def _generics_str(self, params: List[GenericParam]) -> str:
        if not params:
            return ""
        parts: List[str] = []
        for p in params:
            if p.bounds:
                bounds = " + ".join(str(b) for b in p.bounds)
                parts.append(f"{p.name}: {bounds}")
            else:
                parts.append(p.name)
        return "<" + ", ".join(parts) + ">"


# ---------------------------------------------------------------------------
# RustParser – main entry point
# ---------------------------------------------------------------------------

class RustParser:
    """Parse Rust source code into a RustAST."""

    def parse(self, source_code: str) -> RustAST:
        tokenizer = Tokenizer(source_code)
        tokens = tokenizer.tokenize()
        parser = Parser(tokens)
        return parser.parse()

    def tokenize(self, source_code: str) -> List[Token]:
        tokenizer = Tokenizer(source_code)
        return tokenizer.tokenize()

    def pretty_print(self, ast: RustAST) -> str:
        printer = ASTPrettyPrinter()
        printer.visit(ast)
        return printer.result()


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def parse_rust(source: str) -> RustAST:
    return RustParser().parse(source)


def pretty_print_rust(ast: RustAST) -> str:
    return RustParser().pretty_print(ast)
