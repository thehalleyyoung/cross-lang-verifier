"""
C Parser for Cross-Language Equivalence Verifier.

Implements a full tokenizer and recursive-descent parser for C89/C99,
producing an AST that can be pretty-printed back to valid C or walked
with a visitor.
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
    INT_LITERAL = "INT_LITERAL"
    FLOAT_LITERAL = "FLOAT_LITERAL"
    STRING_LITERAL = "STRING_LITERAL"
    CHAR_LITERAL = "CHAR_LITERAL"
    # Punctuation / operators
    LPAREN = "("
    RPAREN = ")"
    LBRACE = "{"
    RBRACE = "}"
    LBRACKET = "["
    RBRACKET = "]"
    SEMI = ";"
    COMMA = ","
    DOT = "."
    ARROW = "->"
    ELLIPSIS = "..."
    COLON = ":"
    QUESTION = "?"
    HASH = "#"
    # Arithmetic
    PLUS = "+"
    MINUS = "-"
    STAR = "*"
    SLASH = "/"
    PERCENT = "%"
    INC = "++"
    DEC = "--"
    # Bitwise
    AMP = "&"
    PIPE = "|"
    CARET = "^"
    TILDE = "~"
    LSHIFT = "<<"
    RSHIFT = ">>"
    # Logical
    AND = "&&"
    OR = "||"
    NOT = "!"
    # Comparison
    EQ = "=="
    NE = "!="
    LT = "<"
    GT = ">"
    LE = "<="
    GE = ">="
    # Assignment
    ASSIGN = "="
    PLUS_ASSIGN = "+="
    MINUS_ASSIGN = "-="
    STAR_ASSIGN = "*="
    SLASH_ASSIGN = "/="
    PERCENT_ASSIGN = "%="
    AMP_ASSIGN = "&="
    PIPE_ASSIGN = "|="
    CARET_ASSIGN = "^="
    LSHIFT_ASSIGN = "<<="
    RSHIFT_ASSIGN = ">>="
    # Special
    PREPROC = "PREPROC"
    EOF = "EOF"
    NEWLINE = "NEWLINE"


@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    col: int

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, L{self.line}:{self.col})"


# ---------------------------------------------------------------------------
# C keywords
# ---------------------------------------------------------------------------

C_KEYWORDS = {
    "auto", "break", "case", "char", "const", "continue", "default", "do",
    "double", "else", "enum", "extern", "float", "for", "goto", "if",
    "inline", "int", "long", "register", "restrict", "return", "short",
    "signed", "sizeof", "static", "struct", "switch", "typedef", "union",
    "unsigned", "void", "volatile", "while", "_Bool", "_Complex", "_Imaginary",
}

TYPE_SPECIFIERS = {
    "void", "char", "short", "int", "long", "float", "double",
    "signed", "unsigned", "_Bool", "_Complex", "_Imaginary",
}

TYPE_QUALIFIERS = {"const", "volatile", "restrict"}

STORAGE_CLASSES = {"auto", "register", "static", "extern", "typedef", "inline"}

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_MULTI_CHAR_OPS: List[Tuple[str, TokenType]] = sorted(
    [
        ("<<=", TokenType.LSHIFT_ASSIGN),
        (">>=", TokenType.RSHIFT_ASSIGN),
        ("...", TokenType.ELLIPSIS),
        ("->", TokenType.ARROW),
        ("++", TokenType.INC),
        ("--", TokenType.DEC),
        ("<<", TokenType.LSHIFT),
        (">>", TokenType.RSHIFT),
        ("&&", TokenType.AND),
        ("||", TokenType.OR),
        ("==", TokenType.EQ),
        ("!=", TokenType.NE),
        ("<=", TokenType.LE),
        (">=", TokenType.GE),
        ("+=", TokenType.PLUS_ASSIGN),
        ("-=", TokenType.MINUS_ASSIGN),
        ("*=", TokenType.STAR_ASSIGN),
        ("/=", TokenType.SLASH_ASSIGN),
        ("%=", TokenType.PERCENT_ASSIGN),
        ("&=", TokenType.AMP_ASSIGN),
        ("|=", TokenType.PIPE_ASSIGN),
        ("^=", TokenType.CARET_ASSIGN),
    ],
    key=lambda p: -len(p[0]),
)

_SINGLE_CHAR_OPS: Dict[str, TokenType] = {
    "(": TokenType.LPAREN,
    ")": TokenType.RPAREN,
    "{": TokenType.LBRACE,
    "}": TokenType.RBRACE,
    "[": TokenType.LBRACKET,
    "]": TokenType.RBRACKET,
    ";": TokenType.SEMI,
    ",": TokenType.COMMA,
    ".": TokenType.DOT,
    ":": TokenType.COLON,
    "?": TokenType.QUESTION,
    "+": TokenType.PLUS,
    "-": TokenType.MINUS,
    "*": TokenType.STAR,
    "/": TokenType.SLASH,
    "%": TokenType.PERCENT,
    "&": TokenType.AMP,
    "|": TokenType.PIPE,
    "^": TokenType.CARET,
    "~": TokenType.TILDE,
    "!": TokenType.NOT,
    "<": TokenType.LT,
    ">": TokenType.GT,
    "=": TokenType.ASSIGN,
    "#": TokenType.HASH,
}


class Tokenizer:
    """Lexer for C source code."""

    def __init__(self, source: str) -> None:
        self._src = source
        self._pos = 0
        self._line = 1
        self._col = 1
        self._tokens: List[Token] = []

    def tokenize(self) -> List[Token]:
        while self._pos < len(self._src):
            self._skip_whitespace_and_comments()
            if self._pos >= len(self._src):
                break
            ch = self._src[self._pos]

            if ch == "#" and self._at_line_start():
                self._read_preprocessor()
            elif ch == '"':
                self._read_string()
            elif ch == "'":
                self._read_char()
            elif ch.isdigit() or (ch == "." and self._peek_digit()):
                self._read_number()
            elif ch.isalpha() or ch == "_":
                self._read_ident()
            else:
                self._read_operator()

        self._tokens.append(Token(TokenType.EOF, "", self._line, self._col))
        return self._tokens

    # -- helpers --

    def _at_line_start(self) -> bool:
        i = self._pos - 1
        while i >= 0 and self._src[i] in " \t":
            i -= 1
        return i < 0 or self._src[i] == "\n"

    def _peek_digit(self) -> bool:
        nxt = self._pos + 1
        return nxt < len(self._src) and self._src[nxt].isdigit()

    def _advance(self, n: int = 1) -> str:
        chunk = self._src[self._pos: self._pos + n]
        for c in chunk:
            if c == "\n":
                self._line += 1
                self._col = 1
            else:
                self._col += 1
        self._pos += n
        return chunk

    def _skip_whitespace_and_comments(self) -> None:
        while self._pos < len(self._src):
            ch = self._src[self._pos]
            if ch in " \t\r\n":
                self._advance()
                continue
            if self._pos + 1 < len(self._src):
                two = self._src[self._pos: self._pos + 2]
                if two == "//":
                    while self._pos < len(self._src) and self._src[self._pos] != "\n":
                        self._advance()
                    continue
                if two == "/*":
                    self._advance(2)
                    while self._pos + 1 < len(self._src):
                        if self._src[self._pos: self._pos + 2] == "*/":
                            self._advance(2)
                            break
                        self._advance()
                    else:
                        if self._pos < len(self._src):
                            self._advance()
                    continue
            break

    def _read_string(self) -> None:
        line, col = self._line, self._col
        self._advance()  # opening "
        buf: List[str] = []
        while self._pos < len(self._src) and self._src[self._pos] != '"':
            if self._src[self._pos] == "\\":
                self._advance()
                if self._pos < len(self._src):
                    esc = self._advance()
                    buf.append("\\" + esc)
            else:
                buf.append(self._advance())
        if self._pos < len(self._src):
            self._advance()  # closing "
        self._tokens.append(Token(TokenType.STRING_LITERAL, "".join(buf), line, col))

    def _read_char(self) -> None:
        line, col = self._line, self._col
        self._advance()  # opening '
        buf: List[str] = []
        while self._pos < len(self._src) and self._src[self._pos] != "'":
            if self._src[self._pos] == "\\":
                self._advance()
                if self._pos < len(self._src):
                    buf.append("\\" + self._advance())
            else:
                buf.append(self._advance())
        if self._pos < len(self._src):
            self._advance()  # closing '
        self._tokens.append(Token(TokenType.CHAR_LITERAL, "".join(buf), line, col))

    def _read_number(self) -> None:
        line, col = self._line, self._col
        start = self._pos
        is_float = False
        # hex
        if self._src[self._pos] == "0" and self._pos + 1 < len(self._src) and self._src[self._pos + 1] in "xX":
            self._advance(2)
            while self._pos < len(self._src) and self._src[self._pos] in "0123456789abcdefABCDEF":
                self._advance()
        else:
            while self._pos < len(self._src) and self._src[self._pos].isdigit():
                self._advance()
            if self._pos < len(self._src) and self._src[self._pos] == ".":
                is_float = True
                self._advance()
                while self._pos < len(self._src) and self._src[self._pos].isdigit():
                    self._advance()
            if self._pos < len(self._src) and self._src[self._pos] in "eE":
                is_float = True
                self._advance()
                if self._pos < len(self._src) and self._src[self._pos] in "+-":
                    self._advance()
                while self._pos < len(self._src) and self._src[self._pos].isdigit():
                    self._advance()
        # suffix
        while self._pos < len(self._src) and self._src[self._pos] in "uUlLfF":
            self._advance()
        val = self._src[start: self._pos]
        tt = TokenType.FLOAT_LITERAL if is_float else TokenType.INT_LITERAL
        self._tokens.append(Token(tt, val, line, col))

    def _read_ident(self) -> None:
        line, col = self._line, self._col
        start = self._pos
        while self._pos < len(self._src) and (self._src[self._pos].isalnum() or self._src[self._pos] == "_"):
            self._advance()
        self._tokens.append(Token(TokenType.IDENT, self._src[start: self._pos], line, col))

    def _read_operator(self) -> None:
        line, col = self._line, self._col
        remaining = self._src[self._pos:]
        for op_str, op_type in _MULTI_CHAR_OPS:
            if remaining.startswith(op_str):
                self._advance(len(op_str))
                self._tokens.append(Token(op_type, op_str, line, col))
                return
        ch = self._src[self._pos]
        if ch in _SINGLE_CHAR_OPS:
            self._advance()
            self._tokens.append(Token(_SINGLE_CHAR_OPS[ch], ch, line, col))
        else:
            self._advance()  # skip unknown

    def _read_preprocessor(self) -> None:
        line, col = self._line, self._col
        start = self._pos
        while self._pos < len(self._src) and self._src[self._pos] != "\n":
            if self._src[self._pos] == "\\" and self._pos + 1 < len(self._src) and self._src[self._pos + 1] == "\n":
                self._advance(2)
            else:
                self._advance()
        self._tokens.append(Token(TokenType.PREPROC, self._src[start: self._pos].strip(), line, col))


# ---------------------------------------------------------------------------
# AST node types
# ---------------------------------------------------------------------------

@dataclass
class ASTNode:
    line: int = 0
    col: int = 0


# -- Expressions --

@dataclass
class IdentExpr(ASTNode):
    name: str = ""

@dataclass
class NumberLiteral(ASTNode):
    value: str = ""

@dataclass
class StringLiteral(ASTNode):
    value: str = ""

@dataclass
class CharLiteral(ASTNode):
    value: str = ""

@dataclass
class BinaryExpr(ASTNode):
    op: str = ""
    left: Optional[ASTNode] = None
    right: Optional[ASTNode] = None

@dataclass
class UnaryExpr(ASTNode):
    op: str = ""
    operand: Optional[ASTNode] = None
    postfix: bool = False

@dataclass
class CallExpr(ASTNode):
    callee: Optional[ASTNode] = None
    args: List[ASTNode] = field(default_factory=list)

@dataclass
class CastExpr(ASTNode):
    type_name: str = ""
    expr: Optional[ASTNode] = None

@dataclass
class MemberExpr(ASTNode):
    obj: Optional[ASTNode] = None
    member: str = ""
    arrow: bool = False

@dataclass
class ArraySubscriptExpr(ASTNode):
    array: Optional[ASTNode] = None
    index: Optional[ASTNode] = None

@dataclass
class SizeofExpr(ASTNode):
    operand: Optional[ASTNode] = None
    type_name: Optional[str] = None

@dataclass
class TernaryExpr(ASTNode):
    cond: Optional[ASTNode] = None
    then_expr: Optional[ASTNode] = None
    else_expr: Optional[ASTNode] = None


# -- Statements --

@dataclass
class ExprStmt(ASTNode):
    expr: Optional[ASTNode] = None

@dataclass
class CompoundStmt(ASTNode):
    stmts: List[ASTNode] = field(default_factory=list)

@dataclass
class IfStmt(ASTNode):
    cond: Optional[ASTNode] = None
    then_body: Optional[ASTNode] = None
    else_body: Optional[ASTNode] = None

@dataclass
class WhileStmt(ASTNode):
    cond: Optional[ASTNode] = None
    body: Optional[ASTNode] = None

@dataclass
class DoWhileStmt(ASTNode):
    body: Optional[ASTNode] = None
    cond: Optional[ASTNode] = None

@dataclass
class ForStmt(ASTNode):
    init: Optional[ASTNode] = None
    cond: Optional[ASTNode] = None
    update: Optional[ASTNode] = None
    body: Optional[ASTNode] = None

@dataclass
class SwitchStmt(ASTNode):
    expr: Optional[ASTNode] = None
    body: Optional[ASTNode] = None

@dataclass
class CaseLabel(ASTNode):
    expr: Optional[ASTNode] = None
    stmt: Optional[ASTNode] = None
    is_default: bool = False

@dataclass
class GotoStmt(ASTNode):
    label: str = ""

@dataclass
class LabelStmt(ASTNode):
    label: str = ""
    stmt: Optional[ASTNode] = None

@dataclass
class ReturnStmt(ASTNode):
    expr: Optional[ASTNode] = None

@dataclass
class BreakStmt(ASTNode):
    pass

@dataclass
class ContinueStmt(ASTNode):
    pass


# -- Declarations --

@dataclass
class TypeSpec(ASTNode):
    name: str = ""
    qualifiers: List[str] = field(default_factory=list)
    storage: List[str] = field(default_factory=list)
    pointer_depth: int = 0

@dataclass
class ParamDecl(ASTNode):
    type_spec: Optional[TypeSpec] = None
    name: str = ""
    is_variadic: bool = False
    array_dims: List[Optional[ASTNode]] = field(default_factory=list)

@dataclass
class VarDecl(ASTNode):
    type_spec: Optional[TypeSpec] = None
    name: str = ""
    init: Optional[ASTNode] = None
    array_dims: List[Optional[ASTNode]] = field(default_factory=list)
    is_function_ptr: bool = False
    fp_params: List[ParamDecl] = field(default_factory=list)

@dataclass
class FunctionDef(ASTNode):
    return_type: Optional[TypeSpec] = None
    name: str = ""
    params: List[ParamDecl] = field(default_factory=list)
    body: Optional[CompoundStmt] = None
    is_variadic: bool = False

@dataclass
class StructField(ASTNode):
    type_spec: Optional[TypeSpec] = None
    name: str = ""
    array_dims: List[Optional[ASTNode]] = field(default_factory=list)
    bit_width: Optional[ASTNode] = None

@dataclass
class StructDef(ASTNode):
    tag: str = ""
    fields: List[StructField] = field(default_factory=list)

@dataclass
class UnionDef(ASTNode):
    tag: str = ""
    fields: List[StructField] = field(default_factory=list)

@dataclass
class EnumValue(ASTNode):
    name: str = ""
    value: Optional[ASTNode] = None

@dataclass
class EnumDef(ASTNode):
    tag: str = ""
    values: List[EnumValue] = field(default_factory=list)

@dataclass
class TypedefDecl(ASTNode):
    original_type: Optional[TypeSpec] = None
    new_name: str = ""
    struct_def: Optional[Union[StructDef, UnionDef, EnumDef]] = None

@dataclass
class PreprocessorDirective(ASTNode):
    text: str = ""


# -- Top-level AST --

@dataclass
class CAST:
    functions: List[FunctionDef] = field(default_factory=list)
    global_vars: List[VarDecl] = field(default_factory=list)
    type_definitions: List[Union[StructDef, UnionDef, EnumDef, TypedefDecl]] = field(default_factory=list)
    includes: List[str] = field(default_factory=list)
    macros: List[str] = field(default_factory=list)
    preprocessor: List[PreprocessorDirective] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parse error
# ---------------------------------------------------------------------------

class ParseError(Exception):
    def __init__(self, msg: str, token: Optional[Token] = None) -> None:
        self.token = token
        loc = f" at L{token.line}:{token.col}" if token else ""
        super().__init__(f"{msg}{loc}")


# ---------------------------------------------------------------------------
# Recursive-descent parser
# ---------------------------------------------------------------------------

class Parser:
    """Recursive-descent parser for C source tokens."""

    def __init__(self, tokens: List[Token]) -> None:
        self._tokens = tokens
        self._pos = 0
        self._typedef_names: set = set()
        self._errors: List[str] = []

    # -- token helpers --

    def _cur(self) -> Token:
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return Token(TokenType.EOF, "", 0, 0)

    def _peek(self, offset: int = 0) -> Token:
        idx = self._pos + offset
        if idx < len(self._tokens):
            return self._tokens[idx]
        return Token(TokenType.EOF, "", 0, 0)

    def _advance(self) -> Token:
        tok = self._cur()
        if self._pos < len(self._tokens):
            self._pos += 1
        return tok

    def _expect(self, tt: TokenType) -> Token:
        tok = self._cur()
        if tok.type != tt:
            self._error(f"expected {tt.value!r}, got {tok.value!r}")
            return tok
        return self._advance()

    def _match(self, tt: TokenType) -> Optional[Token]:
        if self._cur().type == tt:
            return self._advance()
        return None

    def _match_kw(self, kw: str) -> bool:
        if self._cur().type == TokenType.IDENT and self._cur().value == kw:
            self._advance()
            return True
        return False

    def _is_kw(self, kw: str) -> bool:
        return self._cur().type == TokenType.IDENT and self._cur().value == kw

    def _error(self, msg: str) -> None:
        tok = self._cur()
        self._errors.append(f"L{tok.line}:{tok.col}: {msg}")

    def _recover_to_semi_or_brace(self) -> None:
        while self._cur().type not in (TokenType.SEMI, TokenType.RBRACE, TokenType.LBRACE, TokenType.EOF):
            self._advance()
        if self._cur().type == TokenType.SEMI:
            self._advance()

    # -- type helpers --

    def _is_type_start(self) -> bool:
        tok = self._cur()
        if tok.type != TokenType.IDENT:
            return False
        return (tok.value in TYPE_SPECIFIERS or tok.value in TYPE_QUALIFIERS
                or tok.value in STORAGE_CLASSES or tok.value in ("struct", "union", "enum")
                or tok.value in self._typedef_names)

    def _parse_type_spec(self) -> TypeSpec:
        ts = TypeSpec(line=self._cur().line, col=self._cur().col)
        quals: List[str] = []
        storage: List[str] = []
        parts: List[str] = []

        while True:
            tok = self._cur()
            if tok.type != TokenType.IDENT:
                break
            if tok.value in STORAGE_CLASSES:
                storage.append(tok.value)
                self._advance()
            elif tok.value in TYPE_QUALIFIERS:
                quals.append(tok.value)
                self._advance()
            elif tok.value in TYPE_SPECIFIERS or tok.value in self._typedef_names:
                parts.append(tok.value)
                self._advance()
            elif tok.value in ("struct", "union", "enum"):
                parts.append(tok.value)
                self._advance()
                if self._cur().type == TokenType.IDENT:
                    parts.append(self._advance().value)
                break
            else:
                break

        ts.name = " ".join(parts) if parts else "int"
        ts.qualifiers = quals
        ts.storage = storage

        while self._cur().type == TokenType.STAR:
            self._advance()
            ts.pointer_depth += 1
            while self._cur().type == TokenType.IDENT and self._cur().value in TYPE_QUALIFIERS:
                ts.qualifiers.append(self._advance().value)

        return ts

    # -- declaration parsing --

    def _parse_param_list(self) -> Tuple[List[ParamDecl], bool]:
        params: List[ParamDecl] = []
        is_variadic = False
        self._expect(TokenType.LPAREN)
        if self._cur().type == TokenType.RPAREN:
            self._advance()
            return params, False
        if self._is_kw("void") and self._peek(1).type == TokenType.RPAREN:
            self._advance()
            self._advance()
            return params, False

        while True:
            if self._cur().type == TokenType.ELLIPSIS:
                self._advance()
                is_variadic = True
                break
            if self._is_type_start():
                ts = self._parse_type_spec()
                name = ""
                if self._cur().type == TokenType.IDENT and self._cur().value not in C_KEYWORDS:
                    name = self._advance().value
                dims: List[Optional[ASTNode]] = []
                while self._cur().type == TokenType.LBRACKET:
                    self._advance()
                    dim = self._parse_expression() if self._cur().type != TokenType.RBRACKET else None
                    self._expect(TokenType.RBRACKET)
                    dims.append(dim)
                params.append(ParamDecl(type_spec=ts, name=name, array_dims=dims))
            else:
                break
            if not self._match(TokenType.COMMA):
                break

        self._expect(TokenType.RPAREN)
        return params, is_variadic

    def _parse_struct_or_union_body(self) -> List[StructField]:
        fields: List[StructField] = []
        self._expect(TokenType.LBRACE)
        while self._cur().type != TokenType.RBRACE and self._cur().type != TokenType.EOF:
            if self._is_type_start():
                ts = self._parse_type_spec()
                name = ""
                if self._cur().type == TokenType.IDENT:
                    name = self._advance().value
                dims: List[Optional[ASTNode]] = []
                while self._cur().type == TokenType.LBRACKET:
                    self._advance()
                    dim = self._parse_expression() if self._cur().type != TokenType.RBRACKET else None
                    self._expect(TokenType.RBRACKET)
                    dims.append(dim)
                bit_w: Optional[ASTNode] = None
                if self._match(TokenType.COLON):
                    bit_w = self._parse_expression()
                fields.append(StructField(type_spec=ts, name=name, array_dims=dims, bit_width=bit_w))
                self._expect(TokenType.SEMI)
            else:
                self._error("expected field declaration in struct/union")
                self._recover_to_semi_or_brace()
        self._expect(TokenType.RBRACE)
        return fields

    def _parse_enum_body(self) -> List[EnumValue]:
        values: List[EnumValue] = []
        self._expect(TokenType.LBRACE)
        while self._cur().type != TokenType.RBRACE and self._cur().type != TokenType.EOF:
            name = self._expect(TokenType.IDENT).value
            val: Optional[ASTNode] = None
            if self._match(TokenType.ASSIGN):
                val = self._parse_expression()
            values.append(EnumValue(name=name, value=val))
            if not self._match(TokenType.COMMA):
                break
        self._expect(TokenType.RBRACE)
        return values

    # -- statement parsing --

    def _parse_compound_stmt(self) -> CompoundStmt:
        cs = CompoundStmt(line=self._cur().line, col=self._cur().col)
        self._expect(TokenType.LBRACE)
        while self._cur().type != TokenType.RBRACE and self._cur().type != TokenType.EOF:
            stmt = self._parse_block_item()
            if stmt is not None:
                cs.stmts.append(stmt)
        self._expect(TokenType.RBRACE)
        return cs

    def _parse_block_item(self) -> Optional[ASTNode]:
        if self._is_type_start():
            return self._parse_local_declaration()
        return self._parse_statement()

    def _parse_local_declaration(self) -> ASTNode:
        ts = self._parse_type_spec()
        if self._cur().type == TokenType.IDENT:
            name = self._advance().value
        else:
            name = ""
        dims: List[Optional[ASTNode]] = []
        while self._cur().type == TokenType.LBRACKET:
            self._advance()
            dim = self._parse_expression() if self._cur().type != TokenType.RBRACKET else None
            self._expect(TokenType.RBRACKET)
            dims.append(dim)
        init: Optional[ASTNode] = None
        if self._match(TokenType.ASSIGN):
            if self._cur().type == TokenType.LBRACE:
                init = self._parse_initializer_list()
            else:
                init = self._parse_expression()
        vd = VarDecl(type_spec=ts, name=name, init=init, array_dims=dims,
                     line=ts.line, col=ts.col)
        self._expect(TokenType.SEMI)
        return vd

    def _parse_initializer_list(self) -> ASTNode:
        """Parse { expr, expr, ... } as a StringLiteral holding the brace-init text."""
        line, col = self._cur().line, self._cur().col
        depth = 0
        parts: List[str] = []
        while self._cur().type != TokenType.EOF:
            if self._cur().type == TokenType.LBRACE:
                depth += 1
            elif self._cur().type == TokenType.RBRACE:
                depth -= 1
                if depth == 0:
                    parts.append(self._advance().value)
                    break
            parts.append(self._cur().value)
            self._advance()
        return StringLiteral(value="".join(parts), line=line, col=col)

    def _parse_statement(self) -> Optional[ASTNode]:
        tok = self._cur()

        if tok.type == TokenType.LBRACE:
            return self._parse_compound_stmt()

        if tok.type == TokenType.SEMI:
            self._advance()
            return ExprStmt(expr=None, line=tok.line, col=tok.col)

        if tok.type == TokenType.IDENT:
            # label: stmt
            if self._peek(1).type == TokenType.COLON and tok.value not in C_KEYWORDS:
                label = self._advance().value
                self._advance()  # colon
                stmt = self._parse_statement()
                return LabelStmt(label=label, stmt=stmt, line=tok.line, col=tok.col)

            if tok.value == "if":
                return self._parse_if()
            if tok.value == "while":
                return self._parse_while()
            if tok.value == "do":
                return self._parse_do_while()
            if tok.value == "for":
                return self._parse_for()
            if tok.value == "switch":
                return self._parse_switch()
            if tok.value == "return":
                return self._parse_return()
            if tok.value == "goto":
                return self._parse_goto()
            if tok.value == "break":
                self._advance()
                self._expect(TokenType.SEMI)
                return BreakStmt(line=tok.line, col=tok.col)
            if tok.value == "continue":
                self._advance()
                self._expect(TokenType.SEMI)
                return ContinueStmt(line=tok.line, col=tok.col)
            if tok.value == "case":
                return self._parse_case()
            if tok.value == "default":
                return self._parse_default()

        # expression statement
        expr = self._parse_expression()
        self._expect(TokenType.SEMI)
        return ExprStmt(expr=expr, line=tok.line, col=tok.col)

    def _parse_if(self) -> IfStmt:
        tok = self._cur()
        self._advance()  # if
        self._expect(TokenType.LPAREN)
        cond = self._parse_expression()
        self._expect(TokenType.RPAREN)
        then_body = self._parse_statement()
        else_body: Optional[ASTNode] = None
        if self._is_kw("else"):
            self._advance()
            else_body = self._parse_statement()
        return IfStmt(cond=cond, then_body=then_body, else_body=else_body,
                      line=tok.line, col=tok.col)

    def _parse_while(self) -> WhileStmt:
        tok = self._cur()
        self._advance()
        self._expect(TokenType.LPAREN)
        cond = self._parse_expression()
        self._expect(TokenType.RPAREN)
        body = self._parse_statement()
        return WhileStmt(cond=cond, body=body, line=tok.line, col=tok.col)

    def _parse_do_while(self) -> DoWhileStmt:
        tok = self._cur()
        self._advance()  # do
        body = self._parse_statement()
        if not self._match_kw("while"):
            self._error("expected 'while' after do body")
        self._expect(TokenType.LPAREN)
        cond = self._parse_expression()
        self._expect(TokenType.RPAREN)
        self._expect(TokenType.SEMI)
        return DoWhileStmt(body=body, cond=cond, line=tok.line, col=tok.col)

    def _parse_for(self) -> ForStmt:
        tok = self._cur()
        self._advance()
        self._expect(TokenType.LPAREN)
        init: Optional[ASTNode] = None
        if self._cur().type != TokenType.SEMI:
            if self._is_type_start():
                init = self._parse_local_decl_no_semi()
            else:
                init = self._parse_expression()
        self._expect(TokenType.SEMI)
        cond: Optional[ASTNode] = None
        if self._cur().type != TokenType.SEMI:
            cond = self._parse_expression()
        self._expect(TokenType.SEMI)
        update: Optional[ASTNode] = None
        if self._cur().type != TokenType.RPAREN:
            update = self._parse_expression()
        self._expect(TokenType.RPAREN)
        body = self._parse_statement()
        return ForStmt(init=init, cond=cond, update=update, body=body,
                       line=tok.line, col=tok.col)

    def _parse_local_decl_no_semi(self) -> VarDecl:
        ts = self._parse_type_spec()
        name = self._advance().value if self._cur().type == TokenType.IDENT else ""
        init: Optional[ASTNode] = None
        if self._match(TokenType.ASSIGN):
            init = self._parse_expression()
        return VarDecl(type_spec=ts, name=name, init=init, line=ts.line, col=ts.col)

    def _parse_switch(self) -> SwitchStmt:
        tok = self._cur()
        self._advance()
        self._expect(TokenType.LPAREN)
        expr = self._parse_expression()
        self._expect(TokenType.RPAREN)
        body = self._parse_compound_stmt()
        return SwitchStmt(expr=expr, body=body, line=tok.line, col=tok.col)

    def _parse_case(self) -> CaseLabel:
        tok = self._cur()
        self._advance()  # case
        expr = self._parse_expression()
        self._expect(TokenType.COLON)
        stmt = self._parse_statement()
        return CaseLabel(expr=expr, stmt=stmt, line=tok.line, col=tok.col)

    def _parse_default(self) -> CaseLabel:
        tok = self._cur()
        self._advance()  # default
        self._expect(TokenType.COLON)
        stmt = self._parse_statement()
        return CaseLabel(expr=None, stmt=stmt, is_default=True, line=tok.line, col=tok.col)

    def _parse_return(self) -> ReturnStmt:
        tok = self._cur()
        self._advance()
        expr: Optional[ASTNode] = None
        if self._cur().type != TokenType.SEMI:
            expr = self._parse_expression()
        self._expect(TokenType.SEMI)
        return ReturnStmt(expr=expr, line=tok.line, col=tok.col)

    def _parse_goto(self) -> GotoStmt:
        tok = self._cur()
        self._advance()
        label = self._expect(TokenType.IDENT).value
        self._expect(TokenType.SEMI)
        return GotoStmt(label=label, line=tok.line, col=tok.col)

    # -- expression parsing (precedence climbing) --

    _PREC: Dict[str, int] = {
        "=": 1, "+=": 1, "-=": 1, "*=": 1, "/=": 1, "%=": 1,
        "&=": 1, "|=": 1, "^=": 1, "<<=": 1, ">>=": 1,
        "||": 3, "&&": 4, "|": 5, "^": 6, "&": 7,
        "==": 8, "!=": 8,
        "<": 9, ">": 9, "<=": 9, ">=": 9,
        "<<": 10, ">>": 10,
        "+": 11, "-": 11,
        "*": 12, "/": 12, "%": 12,
    }

    _RIGHT_ASSOC = {"=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>="}

    def _parse_expression(self, min_prec: int = 0) -> ASTNode:
        left = self._parse_unary()
        while True:
            tok = self._cur()
            op = tok.value
            if op not in self._PREC or self._PREC[op] < min_prec:
                break
            # ternary
            if tok.type == TokenType.QUESTION:
                self._advance()
                then_e = self._parse_expression(0)
                self._expect(TokenType.COLON)
                else_e = self._parse_expression(0)
                left = TernaryExpr(cond=left, then_expr=then_e, else_expr=else_e,
                                   line=tok.line, col=tok.col)
                continue
            prec = self._PREC[op]
            self._advance()
            next_min = prec if op in self._RIGHT_ASSOC else prec + 1
            right = self._parse_expression(next_min)
            left = BinaryExpr(op=op, left=left, right=right, line=tok.line, col=tok.col)
        # ternary at top level
        if self._cur().type == TokenType.QUESTION:
            tok = self._advance()
            then_e = self._parse_expression(0)
            self._expect(TokenType.COLON)
            else_e = self._parse_expression(0)
            left = TernaryExpr(cond=left, then_expr=then_e, else_expr=else_e,
                               line=tok.line, col=tok.col)
        return left

    def _parse_unary(self) -> ASTNode:
        tok = self._cur()
        # prefix operators
        if tok.type in (TokenType.INC, TokenType.DEC):
            self._advance()
            operand = self._parse_unary()
            return UnaryExpr(op=tok.value, operand=operand, postfix=False,
                             line=tok.line, col=tok.col)
        if tok.type in (TokenType.AMP, TokenType.STAR, TokenType.PLUS,
                        TokenType.MINUS, TokenType.TILDE, TokenType.NOT):
            self._advance()
            operand = self._parse_unary()
            return UnaryExpr(op=tok.value, operand=operand, postfix=False,
                             line=tok.line, col=tok.col)
        if tok.type == TokenType.IDENT and tok.value == "sizeof":
            return self._parse_sizeof()
        # cast: (type) expr
        if tok.type == TokenType.LPAREN and self._peek(1).type == TokenType.IDENT and self._peek(1).value in (
                TYPE_SPECIFIERS | TYPE_QUALIFIERS | {"struct", "union", "enum"} | self._typedef_names):
            saved = self._pos
            self._advance()  # (
            type_name_parts: List[str] = []
            while self._cur().type == TokenType.IDENT or self._cur().type == TokenType.STAR:
                type_name_parts.append(self._advance().value)
            if self._cur().type == TokenType.RPAREN:
                self._advance()
                inner = self._parse_unary()
                return CastExpr(type_name=" ".join(type_name_parts), expr=inner,
                                line=tok.line, col=tok.col)
            else:
                self._pos = saved  # backtrack

        return self._parse_postfix()

    def _parse_sizeof(self) -> SizeofExpr:
        tok = self._cur()
        self._advance()  # sizeof
        if self._cur().type == TokenType.LPAREN:
            self._advance()
            if self._is_type_start():
                parts: List[str] = []
                while self._cur().type == TokenType.IDENT or self._cur().type == TokenType.STAR:
                    parts.append(self._advance().value)
                self._expect(TokenType.RPAREN)
                return SizeofExpr(type_name=" ".join(parts), line=tok.line, col=tok.col)
            else:
                expr = self._parse_expression()
                self._expect(TokenType.RPAREN)
                return SizeofExpr(operand=expr, line=tok.line, col=tok.col)
        operand = self._parse_unary()
        return SizeofExpr(operand=operand, line=tok.line, col=tok.col)

    def _parse_postfix(self) -> ASTNode:
        node = self._parse_primary()
        while True:
            tok = self._cur()
            if tok.type == TokenType.LBRACKET:
                self._advance()
                idx = self._parse_expression()
                self._expect(TokenType.RBRACKET)
                node = ArraySubscriptExpr(array=node, index=idx, line=tok.line, col=tok.col)
            elif tok.type == TokenType.LPAREN:
                self._advance()
                args: List[ASTNode] = []
                if self._cur().type != TokenType.RPAREN:
                    args.append(self._parse_expression())
                    while self._match(TokenType.COMMA):
                        args.append(self._parse_expression())
                self._expect(TokenType.RPAREN)
                node = CallExpr(callee=node, args=args, line=tok.line, col=tok.col)
            elif tok.type == TokenType.DOT:
                self._advance()
                member = self._expect(TokenType.IDENT).value
                node = MemberExpr(obj=node, member=member, arrow=False,
                                  line=tok.line, col=tok.col)
            elif tok.type == TokenType.ARROW:
                self._advance()
                member = self._expect(TokenType.IDENT).value
                node = MemberExpr(obj=node, member=member, arrow=True,
                                  line=tok.line, col=tok.col)
            elif tok.type in (TokenType.INC, TokenType.DEC):
                self._advance()
                node = UnaryExpr(op=tok.value, operand=node, postfix=True,
                                 line=tok.line, col=tok.col)
            else:
                break
        return node

    def _parse_primary(self) -> ASTNode:
        tok = self._cur()
        if tok.type == TokenType.INT_LITERAL or tok.type == TokenType.FLOAT_LITERAL:
            self._advance()
            return NumberLiteral(value=tok.value, line=tok.line, col=tok.col)
        if tok.type == TokenType.STRING_LITERAL:
            self._advance()
            return StringLiteral(value=tok.value, line=tok.line, col=tok.col)
        if tok.type == TokenType.CHAR_LITERAL:
            self._advance()
            return CharLiteral(value=tok.value, line=tok.line, col=tok.col)
        if tok.type == TokenType.IDENT:
            self._advance()
            return IdentExpr(name=tok.value, line=tok.line, col=tok.col)
        if tok.type == TokenType.LPAREN:
            self._advance()
            expr = self._parse_expression()
            self._expect(TokenType.RPAREN)
            return expr
        self._error(f"unexpected token {tok.value!r}")
        self._advance()
        return IdentExpr(name="<error>", line=tok.line, col=tok.col)

    # -- top-level parsing --

    def parse(self) -> CAST:
        ast = CAST()
        while self._cur().type != TokenType.EOF:
            try:
                self._parse_top_level(ast)
            except ParseError as e:
                self._errors.append(str(e))
                self._recover_to_semi_or_brace()
        return ast

    def _parse_top_level(self, ast: CAST) -> None:
        tok = self._cur()

        # preprocessor
        if tok.type == TokenType.PREPROC:
            text = tok.value
            self._advance()
            directive = PreprocessorDirective(text=text, line=tok.line, col=tok.col)
            ast.preprocessor.append(directive)
            if text.startswith("#include"):
                ast.includes.append(text)
            elif text.startswith("#define"):
                ast.macros.append(text)
            return

        # struct/union/enum definition
        if self._is_kw("struct") or self._is_kw("union") or self._is_kw("enum"):
            td_node = self._parse_struct_union_enum_toplevel(ast)
            if td_node is not None:
                ast.type_definitions.append(td_node)
            return

        # typedef
        if self._is_kw("typedef"):
            td = self._parse_typedef()
            ast.type_definitions.append(td)
            return

        # function or global var
        if self._is_type_start():
            self._parse_func_or_var(ast)
            return

        # skip unknown
        self._error(f"unexpected top-level token {tok.value!r}")
        self._advance()

    def _parse_struct_union_enum_toplevel(self, ast: CAST) -> Optional[Union[StructDef, UnionDef, EnumDef]]:
        tok = self._cur()
        kind = self._advance().value  # struct/union/enum
        tag = ""
        if self._cur().type == TokenType.IDENT:
            tag = self._advance().value

        if self._cur().type == TokenType.LBRACE:
            if kind == "enum":
                values = self._parse_enum_body()
                node = EnumDef(tag=tag, values=values, line=tok.line, col=tok.col)
            else:
                fields = self._parse_struct_or_union_body()
                if kind == "struct":
                    node = StructDef(tag=tag, fields=fields, line=tok.line, col=tok.col)
                else:
                    node = UnionDef(tag=tag, fields=fields, line=tok.line, col=tok.col)
            # might be followed by variable names
            if self._cur().type == TokenType.IDENT:
                name = self._advance().value
                self._expect(TokenType.SEMI)
                vd = VarDecl(type_spec=TypeSpec(name=f"{kind} {tag}"), name=name,
                             line=tok.line, col=tok.col)
                ast.global_vars.append(vd)
            else:
                self._expect(TokenType.SEMI)
            return node
        else:
            # forward decl or variable of struct type
            if self._cur().type == TokenType.SEMI:
                self._advance()
                if kind == "struct":
                    return StructDef(tag=tag, fields=[], line=tok.line, col=tok.col)
                elif kind == "union":
                    return UnionDef(tag=tag, fields=[], line=tok.line, col=tok.col)
                else:
                    return EnumDef(tag=tag, values=[], line=tok.line, col=tok.col)
            # variable declaration with struct type
            ts = TypeSpec(name=f"{kind} {tag}", line=tok.line, col=tok.col)
            while self._cur().type == TokenType.STAR:
                self._advance()
                ts.pointer_depth += 1
            if self._cur().type == TokenType.IDENT:
                name = self._advance().value
                init: Optional[ASTNode] = None
                if self._match(TokenType.ASSIGN):
                    init = self._parse_expression()
                self._expect(TokenType.SEMI)
                ast.global_vars.append(VarDecl(type_spec=ts, name=name, init=init,
                                               line=tok.line, col=tok.col))
            return None

    def _parse_typedef(self) -> TypedefDecl:
        tok = self._cur()
        self._advance()  # typedef

        struct_def: Optional[Union[StructDef, UnionDef, EnumDef]] = None

        if self._is_kw("struct") or self._is_kw("union"):
            kind = self._advance().value
            tag = ""
            if self._cur().type == TokenType.IDENT and self._peek(1).type == TokenType.LBRACE:
                tag = self._advance().value
            elif self._cur().type == TokenType.IDENT:
                tag = self._advance().value
                new_name = ""
                # consume pointer stars
                ptr = 0
                while self._cur().type == TokenType.STAR:
                    self._advance()
                    ptr += 1
                if self._cur().type == TokenType.IDENT:
                    new_name = self._advance().value
                self._expect(TokenType.SEMI)
                ts = TypeSpec(name=f"{kind} {tag}", pointer_depth=ptr, line=tok.line, col=tok.col)
                self._typedef_names.add(new_name)
                return TypedefDecl(original_type=ts, new_name=new_name, line=tok.line, col=tok.col)

            if self._cur().type == TokenType.LBRACE:
                fields = self._parse_struct_or_union_body()
                if kind == "struct":
                    struct_def = StructDef(tag=tag, fields=fields, line=tok.line, col=tok.col)
                else:
                    struct_def = UnionDef(tag=tag, fields=fields, line=tok.line, col=tok.col)

            ptr = 0
            while self._cur().type == TokenType.STAR:
                self._advance()
                ptr += 1
            new_name = self._expect(TokenType.IDENT).value
            self._expect(TokenType.SEMI)
            ts = TypeSpec(name=f"{kind} {tag}".strip(), pointer_depth=ptr, line=tok.line, col=tok.col)
            self._typedef_names.add(new_name)
            return TypedefDecl(original_type=ts, new_name=new_name,
                               struct_def=struct_def, line=tok.line, col=tok.col)

        if self._is_kw("enum"):
            self._advance()
            tag = ""
            if self._cur().type == TokenType.IDENT and self._peek(1).type == TokenType.LBRACE:
                tag = self._advance().value
            elif self._cur().type == TokenType.IDENT and self._peek(1).type != TokenType.LBRACE:
                tag = self._advance().value
            if self._cur().type == TokenType.LBRACE:
                values = self._parse_enum_body()
                struct_def = EnumDef(tag=tag, values=values, line=tok.line, col=tok.col)
            new_name = self._expect(TokenType.IDENT).value
            self._expect(TokenType.SEMI)
            self._typedef_names.add(new_name)
            return TypedefDecl(original_type=TypeSpec(name=f"enum {tag}".strip()),
                               new_name=new_name, struct_def=struct_def,
                               line=tok.line, col=tok.col)

        ts = self._parse_type_spec()
        # function pointer typedef: typedef ret (*name)(params);
        if self._cur().type == TokenType.LPAREN and self._peek(1).type == TokenType.STAR:
            self._advance()  # (
            self._advance()  # *
            new_name = self._expect(TokenType.IDENT).value
            self._expect(TokenType.RPAREN)
            params, variadic = self._parse_param_list()
            self._expect(TokenType.SEMI)
            self._typedef_names.add(new_name)
            return TypedefDecl(original_type=ts, new_name=new_name, line=tok.line, col=tok.col)

        new_name = self._expect(TokenType.IDENT).value
        self._expect(TokenType.SEMI)
        self._typedef_names.add(new_name)
        return TypedefDecl(original_type=ts, new_name=new_name, line=tok.line, col=tok.col)

    def _parse_func_or_var(self, ast: CAST) -> None:
        ts = self._parse_type_spec()
        # function pointer global
        if self._cur().type == TokenType.LPAREN and self._peek(1).type == TokenType.STAR:
            saved = self._pos
            self._advance()  # (
            self._advance()  # *
            if self._cur().type == TokenType.IDENT:
                name = self._advance().value
                self._expect(TokenType.RPAREN)
                fp_params, _ = self._parse_param_list()
                init: Optional[ASTNode] = None
                if self._match(TokenType.ASSIGN):
                    init = self._parse_expression()
                self._expect(TokenType.SEMI)
                vd = VarDecl(type_spec=ts, name=name, init=init,
                             is_function_ptr=True, fp_params=fp_params,
                             line=ts.line, col=ts.col)
                ast.global_vars.append(vd)
                return
            self._pos = saved

        if self._cur().type != TokenType.IDENT:
            self._error("expected identifier")
            self._recover_to_semi_or_brace()
            return

        name = self._advance().value

        # function definition or declaration
        if self._cur().type == TokenType.LPAREN:
            params, is_variadic = self._parse_param_list()
            if self._cur().type == TokenType.LBRACE:
                body = self._parse_compound_stmt()
                fd = FunctionDef(return_type=ts, name=name, params=params,
                                 body=body, is_variadic=is_variadic,
                                 line=ts.line, col=ts.col)
                ast.functions.append(fd)
            else:
                self._expect(TokenType.SEMI)
                fd = FunctionDef(return_type=ts, name=name, params=params,
                                 is_variadic=is_variadic, line=ts.line, col=ts.col)
                ast.functions.append(fd)
            return

        # global variable
        dims: List[Optional[ASTNode]] = []
        while self._cur().type == TokenType.LBRACKET:
            self._advance()
            dim = self._parse_expression() if self._cur().type != TokenType.RBRACKET else None
            self._expect(TokenType.RBRACKET)
            dims.append(dim)

        init_val: Optional[ASTNode] = None
        if self._match(TokenType.ASSIGN):
            if self._cur().type == TokenType.LBRACE:
                init_val = self._parse_initializer_list()
            else:
                init_val = self._parse_expression()

        vd = VarDecl(type_spec=ts, name=name, init=init_val, array_dims=dims,
                     line=ts.line, col=ts.col)
        ast.global_vars.append(vd)

        # handle comma-separated declarations
        while self._match(TokenType.COMMA):
            ptr = 0
            while self._cur().type == TokenType.STAR:
                self._advance()
                ptr += 1
            n2 = self._expect(TokenType.IDENT).value
            ts2 = TypeSpec(name=ts.name, qualifiers=list(ts.qualifiers),
                           storage=list(ts.storage), pointer_depth=ptr)
            iv2: Optional[ASTNode] = None
            if self._match(TokenType.ASSIGN):
                iv2 = self._parse_expression()
            ast.global_vars.append(VarDecl(type_spec=ts2, name=n2, init=iv2,
                                           line=ts.line, col=ts.col))

        self._expect(TokenType.SEMI)


# ---------------------------------------------------------------------------
# AST Visitor
# ---------------------------------------------------------------------------

class ASTVisitor:
    """Base visitor – override ``visit_<NodeType>`` methods."""

    def visit(self, node: Optional[ASTNode]) -> Any:
        if node is None:
            return None
        method_name = "visit_" + type(node).__name__
        visitor_fn = getattr(self, method_name, self.generic_visit)
        return visitor_fn(node)

    def generic_visit(self, node: ASTNode) -> Any:
        for attr_name in vars(node):
            val = getattr(node, attr_name)
            if isinstance(val, ASTNode):
                self.visit(val)
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, ASTNode):
                        self.visit(item)
        return None

    def visit_FunctionDef(self, node: FunctionDef) -> Any:
        for p in node.params:
            self.visit(p)
        self.visit(node.body)
        return None

    def visit_CompoundStmt(self, node: CompoundStmt) -> Any:
        for s in node.stmts:
            self.visit(s)
        return None

    def visit_IfStmt(self, node: IfStmt) -> Any:
        self.visit(node.cond)
        self.visit(node.then_body)
        self.visit(node.else_body)
        return None

    def visit_WhileStmt(self, node: WhileStmt) -> Any:
        self.visit(node.cond)
        self.visit(node.body)
        return None

    def visit_ForStmt(self, node: ForStmt) -> Any:
        self.visit(node.init)
        self.visit(node.cond)
        self.visit(node.update)
        self.visit(node.body)
        return None

    def visit_DoWhileStmt(self, node: DoWhileStmt) -> Any:
        self.visit(node.body)
        self.visit(node.cond)
        return None

    def visit_SwitchStmt(self, node: SwitchStmt) -> Any:
        self.visit(node.expr)
        self.visit(node.body)
        return None

    def visit_CaseLabel(self, node: CaseLabel) -> Any:
        self.visit(node.expr)
        self.visit(node.stmt)
        return None

    def visit_ReturnStmt(self, node: ReturnStmt) -> Any:
        self.visit(node.expr)
        return None

    def visit_ExprStmt(self, node: ExprStmt) -> Any:
        self.visit(node.expr)
        return None

    def visit_BinaryExpr(self, node: BinaryExpr) -> Any:
        self.visit(node.left)
        self.visit(node.right)
        return None

    def visit_UnaryExpr(self, node: UnaryExpr) -> Any:
        self.visit(node.operand)
        return None

    def visit_CallExpr(self, node: CallExpr) -> Any:
        self.visit(node.callee)
        for a in node.args:
            self.visit(a)
        return None

    def visit_TernaryExpr(self, node: TernaryExpr) -> Any:
        self.visit(node.cond)
        self.visit(node.then_expr)
        self.visit(node.else_expr)
        return None

    def visit_CastExpr(self, node: CastExpr) -> Any:
        self.visit(node.expr)
        return None

    def visit_MemberExpr(self, node: MemberExpr) -> Any:
        self.visit(node.obj)
        return None

    def visit_ArraySubscriptExpr(self, node: ArraySubscriptExpr) -> Any:
        self.visit(node.array)
        self.visit(node.index)
        return None

    def visit_SizeofExpr(self, node: SizeofExpr) -> Any:
        self.visit(node.operand)
        return None

    def visit_VarDecl(self, node: VarDecl) -> Any:
        self.visit(node.init)
        return None

    def visit_GotoStmt(self, node: GotoStmt) -> Any:
        return None

    def visit_LabelStmt(self, node: LabelStmt) -> Any:
        self.visit(node.stmt)
        return None

    def visit_BreakStmt(self, node: BreakStmt) -> Any:
        return None

    def visit_ContinueStmt(self, node: ContinueStmt) -> Any:
        return None

    def visit_IdentExpr(self, node: IdentExpr) -> Any:
        return None

    def visit_NumberLiteral(self, node: NumberLiteral) -> Any:
        return None

    def visit_StringLiteral(self, node: StringLiteral) -> Any:
        return None

    def visit_CharLiteral(self, node: CharLiteral) -> Any:
        return None


# ---------------------------------------------------------------------------
# AST Pretty Printer
# ---------------------------------------------------------------------------

class ASTPrettyPrinter(ASTVisitor):
    """Reconstructs C source code from a CAST."""

    def __init__(self) -> None:
        self._indent = 0
        self._lines: List[str] = []

    def _ind(self) -> str:
        return "    " * self._indent

    def _emit(self, text: str) -> None:
        self._lines.append(text)

    def print_ast(self, ast: CAST) -> str:
        for pp in ast.preprocessor:
            self._emit(pp.text)
        if ast.preprocessor:
            self._emit("")

        for td in ast.type_definitions:
            self._print_type_def(td)
            self._emit("")

        for gv in ast.global_vars:
            self._emit(self._format_var_decl(gv) + ";")

        if ast.global_vars:
            self._emit("")

        for fn in ast.functions:
            self._print_function(fn)
            self._emit("")

        return "\n".join(self._lines)

    def _type_str(self, ts: Optional[TypeSpec]) -> str:
        if ts is None:
            return "int"
        parts: List[str] = []
        parts.extend(ts.storage)
        parts.extend(ts.qualifiers)
        parts.append(ts.name)
        base = " ".join(parts)
        return base + " " + "*" * ts.pointer_depth if ts.pointer_depth else base

    def _format_param(self, p: ParamDecl) -> str:
        if p.is_variadic:
            return "..."
        s = self._type_str(p.type_spec)
        if p.name:
            s += " " + p.name
        for dim in p.array_dims:
            s += "[" + (self._expr_str(dim) if dim else "") + "]"
        return s

    def _format_var_decl(self, v: VarDecl) -> str:
        s = self._type_str(v.type_spec) + " " + v.name
        for dim in v.array_dims:
            s += "[" + (self._expr_str(dim) if dim else "") + "]"
        if v.is_function_ptr:
            param_str = ", ".join(self._format_param(p) for p in v.fp_params)
            s = f"{self._type_str(v.type_spec)} (*{v.name})({param_str})"
        if v.init:
            s += " = " + self._expr_str(v.init)
        return s

    def _print_type_def(self, td: Union[StructDef, UnionDef, EnumDef, TypedefDecl]) -> None:
        if isinstance(td, TypedefDecl):
            if td.struct_def is not None:
                self._print_type_def_inline(td)
            else:
                ts_str = self._type_str(td.original_type)
                self._emit(f"typedef {ts_str} {td.new_name};")
        elif isinstance(td, StructDef):
            self._emit(f"struct {td.tag} {{" if td.tag else "struct {")
            self._indent += 1
            for f in td.fields:
                self._emit(self._ind() + self._format_struct_field(f) + ";")
            self._indent -= 1
            self._emit("};")
        elif isinstance(td, UnionDef):
            self._emit(f"union {td.tag} {{" if td.tag else "union {")
            self._indent += 1
            for f in td.fields:
                self._emit(self._ind() + self._format_struct_field(f) + ";")
            self._indent -= 1
            self._emit("};")
        elif isinstance(td, EnumDef):
            self._emit(f"enum {td.tag} {{" if td.tag else "enum {")
            self._indent += 1
            for i, ev in enumerate(td.values):
                line = self._ind() + ev.name
                if ev.value is not None:
                    line += " = " + self._expr_str(ev.value)
                if i < len(td.values) - 1:
                    line += ","
                self._emit(line)
            self._indent -= 1
            self._emit("};")

    def _print_type_def_inline(self, td: TypedefDecl) -> None:
        sd = td.struct_def
        if isinstance(sd, StructDef):
            header = f"typedef struct {sd.tag} {{" if sd.tag else "typedef struct {"
        elif isinstance(sd, UnionDef):
            header = f"typedef union {sd.tag} {{" if sd.tag else "typedef union {"
        elif isinstance(sd, EnumDef):
            header = f"typedef enum {sd.tag} {{" if sd.tag else "typedef enum {"
        else:
            header = "typedef {"
        self._emit(header)
        self._indent += 1
        if isinstance(sd, (StructDef, UnionDef)):
            for f in sd.fields:
                self._emit(self._ind() + self._format_struct_field(f) + ";")
        elif isinstance(sd, EnumDef):
            for i, ev in enumerate(sd.values):
                line = self._ind() + ev.name
                if ev.value is not None:
                    line += " = " + self._expr_str(ev.value)
                if i < len(sd.values) - 1:
                    line += ","
                self._emit(line)
        self._indent -= 1
        self._emit("} " + td.new_name + ";")

    def _format_struct_field(self, f: StructField) -> str:
        s = self._type_str(f.type_spec) + " " + f.name
        for dim in f.array_dims:
            s += "[" + (self._expr_str(dim) if dim else "") + "]"
        if f.bit_width:
            s += " : " + self._expr_str(f.bit_width)
        return s

    def _print_function(self, fn: FunctionDef) -> None:
        params = ", ".join(self._format_param(p) for p in fn.params)
        if fn.is_variadic:
            params += ", ..." if params else "..."
        sig = f"{self._type_str(fn.return_type)} {fn.name}({params})"
        if fn.body is None:
            self._emit(sig + ";")
            return
        self._emit(sig + " {")
        self._indent += 1
        for s in fn.body.stmts:
            self._print_stmt(s)
        self._indent -= 1
        self._emit("}")

    def _print_stmt(self, node: Optional[ASTNode]) -> None:
        if node is None:
            return
        if isinstance(node, CompoundStmt):
            self._emit(self._ind() + "{")
            self._indent += 1
            for s in node.stmts:
                self._print_stmt(s)
            self._indent -= 1
            self._emit(self._ind() + "}")
        elif isinstance(node, VarDecl):
            self._emit(self._ind() + self._format_var_decl(node) + ";")
        elif isinstance(node, ExprStmt):
            if node.expr:
                self._emit(self._ind() + self._expr_str(node.expr) + ";")
            else:
                self._emit(self._ind() + ";")
        elif isinstance(node, ReturnStmt):
            if node.expr:
                self._emit(self._ind() + "return " + self._expr_str(node.expr) + ";")
            else:
                self._emit(self._ind() + "return;")
        elif isinstance(node, IfStmt):
            self._emit(self._ind() + "if (" + self._expr_str(node.cond) + ")")
            self._print_body(node.then_body)
            if node.else_body:
                self._emit(self._ind() + "else")
                self._print_body(node.else_body)
        elif isinstance(node, WhileStmt):
            self._emit(self._ind() + "while (" + self._expr_str(node.cond) + ")")
            self._print_body(node.body)
        elif isinstance(node, DoWhileStmt):
            self._emit(self._ind() + "do")
            self._print_body(node.body)
            self._emit(self._ind() + "while (" + self._expr_str(node.cond) + ");")
        elif isinstance(node, ForStmt):
            init_s = self._expr_str(node.init) if node.init and not isinstance(node.init, VarDecl) else ""
            if isinstance(node.init, VarDecl):
                init_s = self._format_var_decl(node.init)
            cond_s = self._expr_str(node.cond) if node.cond else ""
            upd_s = self._expr_str(node.update) if node.update else ""
            self._emit(self._ind() + f"for ({init_s}; {cond_s}; {upd_s})")
            self._print_body(node.body)
        elif isinstance(node, SwitchStmt):
            self._emit(self._ind() + "switch (" + self._expr_str(node.expr) + ")")
            self._print_body(node.body)
        elif isinstance(node, CaseLabel):
            self._indent -= 1
            if node.is_default:
                self._emit(self._ind() + "default:")
            else:
                self._emit(self._ind() + "case " + self._expr_str(node.expr) + ":")
            self._indent += 1
            self._print_stmt(node.stmt)
        elif isinstance(node, GotoStmt):
            self._emit(self._ind() + "goto " + node.label + ";")
        elif isinstance(node, LabelStmt):
            self._emit(node.label + ":")
            self._print_stmt(node.stmt)
        elif isinstance(node, BreakStmt):
            self._emit(self._ind() + "break;")
        elif isinstance(node, ContinueStmt):
            self._emit(self._ind() + "continue;")
        else:
            self._emit(self._ind() + "/* unknown stmt */")

    def _print_body(self, node: Optional[ASTNode]) -> None:
        if isinstance(node, CompoundStmt):
            self._emit(self._ind() + "{")
            self._indent += 1
            for s in node.stmts:
                self._print_stmt(s)
            self._indent -= 1
            self._emit(self._ind() + "}")
        elif node is not None:
            self._indent += 1
            self._print_stmt(node)
            self._indent -= 1

    def _expr_str(self, node: Optional[ASTNode]) -> str:
        if node is None:
            return ""
        if isinstance(node, NumberLiteral):
            return node.value
        if isinstance(node, StringLiteral):
            return f'"{node.value}"'
        if isinstance(node, CharLiteral):
            return f"'{node.value}'"
        if isinstance(node, IdentExpr):
            return node.name
        if isinstance(node, BinaryExpr):
            return f"({self._expr_str(node.left)} {node.op} {self._expr_str(node.right)})"
        if isinstance(node, UnaryExpr):
            if node.postfix:
                return f"({self._expr_str(node.operand)}{node.op})"
            return f"({node.op}{self._expr_str(node.operand)})"
        if isinstance(node, CallExpr):
            args = ", ".join(self._expr_str(a) for a in node.args)
            return f"{self._expr_str(node.callee)}({args})"
        if isinstance(node, CastExpr):
            return f"({node.type_name}){self._expr_str(node.expr)}"
        if isinstance(node, MemberExpr):
            op = "->" if node.arrow else "."
            return f"{self._expr_str(node.obj)}{op}{node.member}"
        if isinstance(node, ArraySubscriptExpr):
            return f"{self._expr_str(node.array)}[{self._expr_str(node.index)}]"
        if isinstance(node, SizeofExpr):
            if node.type_name:
                return f"sizeof({node.type_name})"
            return f"sizeof({self._expr_str(node.operand)})"
        if isinstance(node, TernaryExpr):
            return (f"({self._expr_str(node.cond)} ? "
                    f"{self._expr_str(node.then_expr)} : {self._expr_str(node.else_expr)})")
        if isinstance(node, VarDecl):
            return self._format_var_decl(node)
        return "<unknown>"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class CParser:
    """
    Top-level facade.

    Usage::

        parser = CParser()
        ast = parser.parse(source_code)
        print(ast.functions)
    """

    def __init__(self) -> None:
        self._errors: List[str] = []

    def parse(self, source_code: str) -> CAST:
        tokenizer = Tokenizer(source_code)
        tokens = tokenizer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        self._errors = list(parser._errors)
        return ast

    @property
    def errors(self) -> List[str]:
        return list(self._errors)

    @staticmethod
    def pretty_print(ast: CAST) -> str:
        printer = ASTPrettyPrinter()
        return printer.print_ast(ast)

    @staticmethod
    def visit(ast: CAST, visitor: ASTVisitor) -> None:
        for fn in ast.functions:
            visitor.visit(fn)
        for gv in ast.global_vars:
            visitor.visit(gv)


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

class FunctionCollector(ASTVisitor):
    """Collects all function call names from an AST."""

    def __init__(self) -> None:
        self.calls: List[str] = []

    def visit_CallExpr(self, node: CallExpr) -> Any:
        if isinstance(node.callee, IdentExpr):
            self.calls.append(node.callee.name)
        for a in node.args:
            self.visit(a)
        return None


class VariableCollector(ASTVisitor):
    """Collects all variable declarations from an AST."""

    def __init__(self) -> None:
        self.variables: List[Tuple[str, str]] = []

    def visit_VarDecl(self, node: VarDecl) -> Any:
        type_name = node.type_spec.name if node.type_spec else "unknown"
        self.variables.append((type_name, node.name))
        self.visit(node.init)
        return None


class IdentifierCollector(ASTVisitor):
    """Collects all identifier references."""

    def __init__(self) -> None:
        self.identifiers: set = set()

    def visit_IdentExpr(self, node: IdentExpr) -> Any:
        self.identifiers.add(node.name)
        return None


# ---------------------------------------------------------------------------
# Self-test when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample = r"""
#include <stdio.h>
#include <stdlib.h>
#define MAX_SIZE 1024
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#ifdef DEBUG
#endif

typedef struct Node {
    int value;
    struct Node *next;
} Node;

typedef enum { RED, GREEN, BLUE } Color;

typedef void (*callback_t)(int, void *);

static int global_count = 0;

int add(int a, int b) {
    return a + b;
}

void process(int *arr, int n, ...) {
    int i;
    for (i = 0; i < n; i++) {
        if (arr[i] > 0) {
            arr[i] = arr[i] * 2;
        } else if (arr[i] == 0) {
            continue;
        } else {
            break;
        }
    }
}

int main(int argc, char **argv) {
    int x = 10;
    int y = 20;
    int result = add(x, y);
    printf("Result: %d\n", result);

    int *p = (int *)malloc(sizeof(int) * 10);
    if (p != NULL) {
        p[0] = 42;
        free(p);
    }

    Node node;
    node.value = 100;
    node.next = NULL;

    int z = x > y ? x : y;

    switch (x) {
        case 1:
            printf("one\n");
            break;
        case 2:
            printf("two\n");
            break;
        default:
            printf("other\n");
            break;
    }

    do {
        x--;
    } while (x > 0);

    goto done;
done:
    return 0;
}
"""

    parser = CParser()
    ast = parser.parse(sample)

    print(f"Functions: {[f.name for f in ast.functions]}")
    print(f"Global vars: {[v.name for v in ast.global_vars]}")
    print(f"Type defs: {len(ast.type_definitions)}")
    print(f"Includes: {ast.includes}")
    print(f"Macros: {ast.macros}")
    print(f"Errors: {parser.errors}")
    print()

    collector = FunctionCollector()
    CParser.visit(ast, collector)
    print(f"Function calls found: {collector.calls}")

    var_collector = VariableCollector()
    CParser.visit(ast, var_collector)
    print(f"Variables found: {var_collector.variables}")

    print("\n--- Pretty-printed ---\n")
    print(CParser.pretty_print(ast))
