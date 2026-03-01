"""
Rust Lexer for the Cross-Language Equivalence Verifier.

Tokenizes Rust source code into a stream of tokens with full source location
tracking. Handles all Rust keywords, operators, punctuation, literals
(integers with type suffixes, floats, chars with Unicode escapes, strings,
raw strings, byte strings, byte literals), lifetime tokens, and attribute tokens.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Iterator, Optional


# ---------------------------------------------------------------------------
# Source position tracking
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SourcePos:
    """A position in source code."""
    file: str = ""
    line: int = 1
    column: int = 1
    offset: int = 0

    def __str__(self) -> str:
        if self.file:
            return f"{self.file}:{self.line}:{self.column}"
        return f"{self.line}:{self.column}"


@dataclass(frozen=True)
class SourceSpan:
    """A range of source code."""
    start: SourcePos
    end: SourcePos

    def __str__(self) -> str:
        return f"{self.start}-{self.end}"

    @property
    def length(self) -> int:
        return self.end.offset - self.start.offset


# ---------------------------------------------------------------------------
# Token kinds
# ---------------------------------------------------------------------------

class TokenKind(Enum):
    """All token types for Rust."""
    EOF = auto()

    # Literals
    INT_LITERAL = auto()
    FLOAT_LITERAL = auto()
    CHAR_LITERAL = auto()
    BYTE_LITERAL = auto()
    STRING_LITERAL = auto()
    RAW_STRING_LITERAL = auto()
    BYTE_STRING_LITERAL = auto()
    RAW_BYTE_STRING_LITERAL = auto()
    BOOL_LITERAL = auto()

    # Identifier
    IDENT = auto()

    # Lifetime
    LIFETIME = auto()

    # Keywords
    KW_AS = auto()
    KW_ASYNC = auto()
    KW_AWAIT = auto()
    KW_BREAK = auto()
    KW_CONST = auto()
    KW_CONTINUE = auto()
    KW_CRATE = auto()
    KW_DYN = auto()
    KW_ELSE = auto()
    KW_ENUM = auto()
    KW_EXTERN = auto()
    KW_FALSE = auto()
    KW_FN = auto()
    KW_FOR = auto()
    KW_IF = auto()
    KW_IMPL = auto()
    KW_IN = auto()
    KW_LET = auto()
    KW_LOOP = auto()
    KW_MATCH = auto()
    KW_MOD = auto()
    KW_MOVE = auto()
    KW_MUT = auto()
    KW_PUB = auto()
    KW_REF = auto()
    KW_RETURN = auto()
    KW_SELF = auto()
    KW_SELF_TYPE = auto()  # Self (type)
    KW_STATIC = auto()
    KW_STRUCT = auto()
    KW_SUPER = auto()
    KW_TRAIT = auto()
    KW_TRUE = auto()
    KW_TYPE = auto()
    KW_UNSAFE = auto()
    KW_USE = auto()
    KW_WHERE = auto()
    KW_WHILE = auto()
    KW_YIELD = auto()

    # Reserved keywords
    KW_ABSTRACT = auto()
    KW_BECOME = auto()
    KW_BOX = auto()
    KW_DO = auto()
    KW_FINAL = auto()
    KW_MACRO = auto()
    KW_OVERRIDE = auto()
    KW_PRIV = auto()
    KW_TRY = auto()
    KW_TYPEOF = auto()
    KW_UNSIZED = auto()
    KW_VIRTUAL = auto()

    # Operators
    PLUS = auto()          # +
    MINUS = auto()         # -
    STAR = auto()          # *
    SLASH = auto()         # /
    PERCENT = auto()       # %
    AMP = auto()           # &
    PIPE = auto()          # |
    CARET = auto()         # ^
    TILDE = auto()         # ~
    BANG = auto()          # !
    ASSIGN = auto()        # =
    LT = auto()            # <
    GT = auto()            # >
    DOT = auto()           # .
    AT = auto()            # @
    UNDERSCORE = auto()    # _ (as operator/pattern)

    # Compound operators
    PLUS_ASSIGN = auto()   # +=
    MINUS_ASSIGN = auto()  # -=
    STAR_ASSIGN = auto()   # *=
    SLASH_ASSIGN = auto()  # /=
    PERCENT_ASSIGN = auto()  # %=
    AMP_ASSIGN = auto()    # &=
    PIPE_ASSIGN = auto()   # |=
    CARET_ASSIGN = auto()  # ^=
    SHL_ASSIGN = auto()    # <<=
    SHR_ASSIGN = auto()    # >>=
    EQ = auto()            # ==
    NE = auto()            # !=
    LE = auto()            # <=
    GE = auto()            # >=
    AND = auto()           # &&
    OR = auto()            # ||
    SHL = auto()           # <<
    SHR = auto()           # >>
    ARROW = auto()         # ->
    FAT_ARROW = auto()     # =>
    DOT_DOT = auto()       # ..
    DOT_DOT_DOT = auto()   # ...
    DOT_DOT_EQ = auto()    # ..=
    PATH_SEP = auto()      # ::
    QUESTION = auto()      # ?
    HASH = auto()          # #

    # Punctuation
    LPAREN = auto()        # (
    RPAREN = auto()        # )
    LBRACKET = auto()      # [
    RBRACKET = auto()      # ]
    LBRACE = auto()        # {
    RBRACE = auto()        # }
    SEMICOLON = auto()     # ;
    COMMA = auto()         # ,
    COLON = auto()         # :

    # Attribute
    HASH_BANG = auto()     # #!


# ---------------------------------------------------------------------------
# Keyword mapping
# ---------------------------------------------------------------------------

KEYWORDS: dict[str, TokenKind] = {
    "as": TokenKind.KW_AS,
    "async": TokenKind.KW_ASYNC,
    "await": TokenKind.KW_AWAIT,
    "break": TokenKind.KW_BREAK,
    "const": TokenKind.KW_CONST,
    "continue": TokenKind.KW_CONTINUE,
    "crate": TokenKind.KW_CRATE,
    "dyn": TokenKind.KW_DYN,
    "else": TokenKind.KW_ELSE,
    "enum": TokenKind.KW_ENUM,
    "extern": TokenKind.KW_EXTERN,
    "false": TokenKind.KW_FALSE,
    "fn": TokenKind.KW_FN,
    "for": TokenKind.KW_FOR,
    "if": TokenKind.KW_IF,
    "impl": TokenKind.KW_IMPL,
    "in": TokenKind.KW_IN,
    "let": TokenKind.KW_LET,
    "loop": TokenKind.KW_LOOP,
    "match": TokenKind.KW_MATCH,
    "mod": TokenKind.KW_MOD,
    "move": TokenKind.KW_MOVE,
    "mut": TokenKind.KW_MUT,
    "pub": TokenKind.KW_PUB,
    "ref": TokenKind.KW_REF,
    "return": TokenKind.KW_RETURN,
    "self": TokenKind.KW_SELF,
    "Self": TokenKind.KW_SELF_TYPE,
    "static": TokenKind.KW_STATIC,
    "struct": TokenKind.KW_STRUCT,
    "super": TokenKind.KW_SUPER,
    "trait": TokenKind.KW_TRAIT,
    "true": TokenKind.KW_TRUE,
    "type": TokenKind.KW_TYPE,
    "unsafe": TokenKind.KW_UNSAFE,
    "use": TokenKind.KW_USE,
    "where": TokenKind.KW_WHERE,
    "while": TokenKind.KW_WHILE,
    "yield": TokenKind.KW_YIELD,
    "abstract": TokenKind.KW_ABSTRACT,
    "become": TokenKind.KW_BECOME,
    "box": TokenKind.KW_BOX,
    "do": TokenKind.KW_DO,
    "final": TokenKind.KW_FINAL,
    "macro": TokenKind.KW_MACRO,
    "override": TokenKind.KW_OVERRIDE,
    "priv": TokenKind.KW_PRIV,
    "try": TokenKind.KW_TRY,
    "typeof": TokenKind.KW_TYPEOF,
    "unsized": TokenKind.KW_UNSIZED,
    "virtual": TokenKind.KW_VIRTUAL,
}

KEYWORD_KINDS = frozenset(k for k in TokenKind if k.name.startswith("KW_"))

# Integer type suffixes in Rust
INT_SUFFIXES = frozenset({
    "i8", "i16", "i32", "i64", "i128", "isize",
    "u8", "u16", "u32", "u64", "u128", "usize",
})

FLOAT_SUFFIXES = frozenset({"f32", "f64"})


# ---------------------------------------------------------------------------
# Token
# ---------------------------------------------------------------------------

@dataclass
class Token:
    """A single token from the Rust source."""
    kind: TokenKind
    text: str
    span: SourceSpan
    # Parsed values for literals
    int_value: Optional[int] = None
    float_value: Optional[float] = None
    string_value: Optional[str] = None
    char_value: Optional[int] = None
    bool_value: Optional[bool] = None
    # Type suffix
    type_suffix: str = ""  # e.g., "i32", "u64", "f32"
    # Lifetime name (without ')
    lifetime_name: str = ""

    @property
    def is_keyword(self) -> bool:
        return self.kind in KEYWORD_KINDS

    @property
    def is_literal(self) -> bool:
        return self.kind in (
            TokenKind.INT_LITERAL, TokenKind.FLOAT_LITERAL,
            TokenKind.CHAR_LITERAL, TokenKind.STRING_LITERAL,
            TokenKind.BYTE_LITERAL, TokenKind.BYTE_STRING_LITERAL,
            TokenKind.RAW_STRING_LITERAL, TokenKind.RAW_BYTE_STRING_LITERAL,
            TokenKind.BOOL_LITERAL,
        )

    @property
    def start_pos(self) -> SourcePos:
        return self.span.start

    @property
    def end_pos(self) -> SourcePos:
        return self.span.end

    def __repr__(self) -> str:
        return f"Token({self.kind.name}, {self.text!r}, {self.span.start})"


# ---------------------------------------------------------------------------
# Lexer error
# ---------------------------------------------------------------------------

class LexError(Exception):
    """Error during Rust lexing."""
    def __init__(self, message: str, pos: SourcePos):
        self.pos = pos
        super().__init__(f"{pos}: {message}")


# ---------------------------------------------------------------------------
# Escape sequence handling
# ---------------------------------------------------------------------------

_SIMPLE_ESCAPES: dict[str, str] = {
    'n': '\n', 'r': '\r', 't': '\t', '\\': '\\',
    '0': '\0', "'": "'", '"': '"',
}


def _parse_rust_escape(source: str, idx: int, end: int) -> tuple[str, int]:
    """Parse a Rust escape sequence starting after the backslash."""
    if idx >= end:
        return ('\\', idx)

    ch = source[idx]

    if ch in _SIMPLE_ESCAPES:
        return (_SIMPLE_ESCAPES[ch], idx + 1)

    # Hex escape: \xNN
    if ch == 'x':
        idx += 1
        val = 0
        for _ in range(2):
            if idx < end:
                val = val * 16 + int(source[idx], 16)
                idx += 1
        return (chr(val), idx)

    # Unicode escape: \u{XXXX}
    if ch == 'u':
        idx += 1
        if idx < end and source[idx] == '{':
            idx += 1
            val = 0
            while idx < end and source[idx] != '}':
                if source[idx] != '_':
                    val = val * 16 + int(source[idx], 16)
                idx += 1
            if idx < end:
                idx += 1  # skip }
            return (chr(val), idx)

    return (ch, idx + 1)


def _parse_rust_string_content(source: str, start: int, end: int) -> str:
    """Parse string content between quotes."""
    result: list[str] = []
    idx = start
    while idx < end:
        ch = source[idx]
        if ch == '\\':
            idx += 1
            parsed_ch, idx = _parse_rust_escape(source, idx, end)
            result.append(parsed_ch)
        else:
            result.append(ch)
            idx += 1
    return ''.join(result)


# ---------------------------------------------------------------------------
# RustLexer
# ---------------------------------------------------------------------------

class RustLexer:
    """Tokenizer for Rust source code.

    Usage::

        lexer = RustLexer(source, filename="test.rs")
        tokens = lexer.tokenize()
    """

    def __init__(self, source: str, filename: str = "<input>") -> None:
        self._source = source
        self._filename = filename
        self._pos = 0
        self._line = 1
        self._col = 1
        self._tokens: list[Token] = []
        self._errors: list[LexError] = []

    @property
    def source(self) -> str:
        return self._source

    @property
    def errors(self) -> list[LexError]:
        return list(self._errors)

    def _cur_pos(self) -> SourcePos:
        return SourcePos(self._filename, self._line, self._col, self._pos)

    def _peek(self) -> str:
        if self._pos >= len(self._source):
            return '\0'
        return self._source[self._pos]

    def _peek_at(self, offset: int) -> str:
        idx = self._pos + offset
        if idx >= len(self._source):
            return '\0'
        return self._source[idx]

    def _advance(self) -> str:
        if self._pos >= len(self._source):
            return '\0'
        ch = self._source[self._pos]
        self._pos += 1
        if ch == '\n':
            self._line += 1
            self._col = 1
        else:
            self._col += 1
        return ch

    def _advance_n(self, n: int) -> str:
        result = self._source[self._pos:self._pos + n]
        for ch in result:
            if ch == '\n':
                self._line += 1
                self._col = 1
            else:
                self._col += 1
        self._pos += n
        return result

    def _at_end(self) -> bool:
        return self._pos >= len(self._source)

    def _skip_whitespace(self) -> None:
        """Skip whitespace and comments."""
        while not self._at_end():
            ch = self._peek()
            if ch in ' \t\r\n':
                self._advance()
            elif ch == '/' and self._peek_at(1) == '/':
                self._skip_line_comment()
            elif ch == '/' and self._peek_at(1) == '*':
                self._skip_block_comment()
            else:
                break

    def _skip_line_comment(self) -> None:
        self._advance()  # /
        self._advance()  # /
        while not self._at_end() and self._peek() != '\n':
            self._advance()

    def _skip_block_comment(self) -> None:
        """Skip nested block comments."""
        start = self._cur_pos()
        self._advance()  # /
        self._advance()  # *
        depth = 1
        while not self._at_end() and depth > 0:
            if self._peek() == '/' and self._peek_at(1) == '*':
                depth += 1
                self._advance()
                self._advance()
            elif self._peek() == '*' and self._peek_at(1) == '/':
                depth -= 1
                self._advance()
                self._advance()
            else:
                self._advance()
        if depth > 0:
            self._errors.append(LexError("unterminated block comment", start))

    def _make_token(self, kind: TokenKind, start: SourcePos, text: str, **kwargs) -> Token:
        end = self._cur_pos()
        span = SourceSpan(start, end)
        return Token(kind=kind, text=text, span=span, **kwargs)

    def _lex_number(self) -> Token:
        """Lex a numeric literal (integer or float)."""
        start = self._cur_pos()
        start_idx = self._pos
        is_float = False
        base = 10

        if self._peek() == '0':
            self._advance()
            next_ch = self._peek()

            if next_ch in 'xX':
                base = 16
                self._advance()
                while not self._at_end() and (self._peek() in '0123456789abcdefABCDEF_'):
                    self._advance()
            elif next_ch in 'oO':
                base = 8
                self._advance()
                while not self._at_end() and self._peek() in '01234567_':
                    self._advance()
            elif next_ch in 'bB':
                base = 2
                self._advance()
                while not self._at_end() and self._peek() in '01_':
                    self._advance()
            else:
                while not self._at_end() and (self._peek().isdigit() or self._peek() == '_'):
                    self._advance()
                if self._peek() == '.' and self._peek_at(1) != '.' and self._peek_at(1) != ')':
                    is_float = True
                    self._advance()
                    while not self._at_end() and (self._peek().isdigit() or self._peek() == '_'):
                        self._advance()
                if self._peek() in 'eE':
                    is_float = True
                    self._advance()
                    if self._peek() in '+-':
                        self._advance()
                    while not self._at_end() and self._peek().isdigit():
                        self._advance()
        else:
            while not self._at_end() and (self._peek().isdigit() or self._peek() == '_'):
                self._advance()

            if self._peek() == '.' and self._peek_at(1) != '.' and self._peek_at(1) != ')' and not self._peek_at(1).isalpha():
                is_float = True
                self._advance()
                while not self._at_end() and (self._peek().isdigit() or self._peek() == '_'):
                    self._advance()

            if self._peek() in 'eE':
                is_float = True
                self._advance()
                if self._peek() in '+-':
                    self._advance()
                while not self._at_end() and self._peek().isdigit():
                    self._advance()

        # Check for type suffix
        suffix_start = self._pos
        type_suffix = ""
        if self._peek().isalpha() or self._peek() == '_':
            suffix_idx = self._pos
            while not self._at_end() and (self._peek().isalnum() or self._peek() == '_'):
                self._advance()
            potential_suffix = self._source[suffix_idx:self._pos]
            if potential_suffix in INT_SUFFIXES:
                type_suffix = potential_suffix
            elif potential_suffix in FLOAT_SUFFIXES:
                type_suffix = potential_suffix
                is_float = True
            else:
                # Not a valid suffix, revert
                self._pos = suffix_idx
                self._line = start.line
                self._col = start.column + (suffix_idx - start.offset)

        text = self._source[start_idx:self._pos]
        digits = text
        if type_suffix:
            digits = text[:-len(type_suffix)]
        digits = digits.replace('_', '')

        if is_float:
            try:
                value = float(digits)
            except ValueError:
                value = 0.0
            return self._make_token(
                TokenKind.FLOAT_LITERAL, start, text,
                float_value=value, type_suffix=type_suffix,
            )
        else:
            try:
                value = int(digits, base)
            except ValueError:
                value = 0
            return self._make_token(
                TokenKind.INT_LITERAL, start, text,
                int_value=value, type_suffix=type_suffix,
            )

    def _lex_string(self) -> Token:
        """Lex a regular string literal."""
        start = self._cur_pos()
        start_idx = self._pos
        self._advance()  # opening "

        content_start = self._pos
        while not self._at_end() and self._peek() != '"':
            if self._peek() == '\\':
                self._advance()
                if not self._at_end():
                    self._advance()
            else:
                self._advance()

        content_end = self._pos
        if not self._at_end():
            self._advance()  # closing "

        text = self._source[start_idx:self._pos]
        content = _parse_rust_string_content(self._source, content_start, content_end)
        return self._make_token(TokenKind.STRING_LITERAL, start, text,
                               string_value=content)

    def _lex_raw_string(self) -> Token:
        """Lex a raw string literal r#"..."#."""
        start = self._cur_pos()
        start_idx = self._pos
        self._advance()  # r

        hash_count = 0
        while not self._at_end() and self._peek() == '#':
            hash_count += 1
            self._advance()

        if not self._at_end() and self._peek() == '"':
            self._advance()  # opening "
        else:
            text = self._source[start_idx:self._pos]
            return self._make_token(TokenKind.IDENT, start, text)

        content_start = self._pos
        closing = '"' + '#' * hash_count

        while not self._at_end():
            if self._source[self._pos:self._pos + len(closing)] == closing:
                content_end = self._pos
                self._advance_n(len(closing))
                break
            self._advance()
        else:
            content_end = self._pos

        text = self._source[start_idx:self._pos]
        content = self._source[content_start:content_end]
        return self._make_token(TokenKind.RAW_STRING_LITERAL, start, text,
                               string_value=content)

    def _lex_byte_string(self) -> Token:
        """Lex a byte string literal b"..."."""
        start = self._cur_pos()
        start_idx = self._pos
        self._advance()  # b
        self._advance()  # "

        content_start = self._pos
        while not self._at_end() and self._peek() != '"':
            if self._peek() == '\\':
                self._advance()
                if not self._at_end():
                    self._advance()
            else:
                self._advance()

        content_end = self._pos
        if not self._at_end():
            self._advance()  # closing "

        text = self._source[start_idx:self._pos]
        content = _parse_rust_string_content(self._source, content_start, content_end)
        return self._make_token(TokenKind.BYTE_STRING_LITERAL, start, text,
                               string_value=content)

    def _lex_char(self) -> Token:
        """Lex a character literal."""
        start = self._cur_pos()
        start_idx = self._pos
        self._advance()  # opening '

        content_start = self._pos
        while not self._at_end() and self._peek() != "'":
            if self._peek() == '\\':
                self._advance()
                if not self._at_end():
                    self._advance()
            else:
                self._advance()

        content_end = self._pos
        if not self._at_end():
            self._advance()  # closing '

        text = self._source[start_idx:self._pos]
        content = _parse_rust_string_content(self._source, content_start, content_end)
        char_val = ord(content[0]) if content else 0
        return self._make_token(TokenKind.CHAR_LITERAL, start, text,
                               char_value=char_val)

    def _lex_byte_literal(self) -> Token:
        """Lex a byte literal b'x'."""
        start = self._cur_pos()
        start_idx = self._pos
        self._advance()  # b
        self._advance()  # '

        content_start = self._pos
        while not self._at_end() and self._peek() != "'":
            if self._peek() == '\\':
                self._advance()
                if not self._at_end():
                    self._advance()
            else:
                self._advance()

        content_end = self._pos
        if not self._at_end():
            self._advance()  # closing '

        text = self._source[start_idx:self._pos]
        content = _parse_rust_string_content(self._source, content_start, content_end)
        char_val = ord(content[0]) if content else 0
        return self._make_token(TokenKind.BYTE_LITERAL, start, text,
                               char_value=char_val)

    def _lex_lifetime_or_char(self) -> Token:
        """Lex a lifetime 'a or character literal 'x'."""
        start = self._cur_pos()
        start_idx = self._pos

        # Peek ahead: if followed by an identifier and not a closing quote,
        # it's a lifetime
        if (self._peek_at(1).isalpha() or self._peek_at(1) == '_'):
            # Check if it's a char literal (has closing ')
            saved_pos = self._pos
            self._advance()  # '
            ident_start = self._pos
            while not self._at_end() and (self._peek().isalnum() or self._peek() == '_'):
                self._advance()
            ident = self._source[ident_start:self._pos]

            if not self._at_end() and self._peek() == "'":
                # Character literal with multi-char (shouldn't happen normally)
                self._advance()
                text = self._source[start_idx:self._pos]
                char_val = ord(ident[0]) if ident else 0
                return self._make_token(TokenKind.CHAR_LITERAL, start, text,
                                       char_value=char_val)
            else:
                # Lifetime
                text = self._source[start_idx:self._pos]
                return self._make_token(TokenKind.LIFETIME, start, text,
                                       lifetime_name=ident)

        # Regular char literal
        return self._lex_char()

    def _lex_identifier_or_keyword(self) -> Token:
        """Lex an identifier or keyword."""
        start = self._cur_pos()
        start_idx = self._pos

        # Handle raw identifiers: r#ident
        is_raw = False
        if self._peek() == 'r' and self._peek_at(1) == '#' and (self._peek_at(2).isalpha() or self._peek_at(2) == '_'):
            # Check if this might be a raw string instead
            if self._peek_at(2) == '"' or (self._peek_at(2) == '#'):
                return self._lex_raw_string()
            is_raw = True
            self._advance()  # r
            self._advance()  # #

        while not self._at_end() and (self._peek().isalnum() or self._peek() == '_'):
            self._advance()

        text = self._source[start_idx:self._pos]
        ident = text[2:] if is_raw else text  # strip r# for raw idents

        if not is_raw:
            kind = KEYWORDS.get(ident, TokenKind.IDENT)

            # Handle true/false as bool literals
            if kind == TokenKind.KW_TRUE:
                return self._make_token(kind, start, text, bool_value=True)
            if kind == TokenKind.KW_FALSE:
                return self._make_token(kind, start, text, bool_value=False)
        else:
            kind = TokenKind.IDENT

        return self._make_token(kind, start, text)

    def _lex_operator_or_punct(self) -> Token:
        """Lex an operator or punctuation token."""
        start = self._cur_pos()
        ch = self._peek()
        next_ch = self._peek_at(1)
        next2_ch = self._peek_at(2)

        # Three-character operators
        if ch == '.' and next_ch == '.' and next2_ch == '=':
            self._advance_n(3)
            return self._make_token(TokenKind.DOT_DOT_EQ, start, "..=")
        if ch == '.' and next_ch == '.' and next2_ch == '.':
            self._advance_n(3)
            return self._make_token(TokenKind.DOT_DOT_DOT, start, "...")
        if ch == '<' and next_ch == '<' and next2_ch == '=':
            self._advance_n(3)
            return self._make_token(TokenKind.SHL_ASSIGN, start, "<<=")
        if ch == '>' and next_ch == '>' and next2_ch == '=':
            self._advance_n(3)
            return self._make_token(TokenKind.SHR_ASSIGN, start, ">>=")

        # Two-character operators
        two_char = ch + next_ch if next_ch != '\0' else ""
        two_char_map = {
            '+=': TokenKind.PLUS_ASSIGN,
            '-=': TokenKind.MINUS_ASSIGN,
            '*=': TokenKind.STAR_ASSIGN,
            '/=': TokenKind.SLASH_ASSIGN,
            '%=': TokenKind.PERCENT_ASSIGN,
            '&=': TokenKind.AMP_ASSIGN,
            '|=': TokenKind.PIPE_ASSIGN,
            '^=': TokenKind.CARET_ASSIGN,
            '==': TokenKind.EQ,
            '!=': TokenKind.NE,
            '<=': TokenKind.LE,
            '>=': TokenKind.GE,
            '&&': TokenKind.AND,
            '||': TokenKind.OR,
            '<<': TokenKind.SHL,
            '>>': TokenKind.SHR,
            '->': TokenKind.ARROW,
            '=>': TokenKind.FAT_ARROW,
            '..': TokenKind.DOT_DOT,
            '::': TokenKind.PATH_SEP,
            '#!': TokenKind.HASH_BANG,
        }
        if two_char in two_char_map:
            self._advance_n(2)
            return self._make_token(two_char_map[two_char], start, two_char)

        # Single-character
        one_char_map = {
            '+': TokenKind.PLUS,
            '-': TokenKind.MINUS,
            '*': TokenKind.STAR,
            '/': TokenKind.SLASH,
            '%': TokenKind.PERCENT,
            '&': TokenKind.AMP,
            '|': TokenKind.PIPE,
            '^': TokenKind.CARET,
            '~': TokenKind.TILDE,
            '!': TokenKind.BANG,
            '=': TokenKind.ASSIGN,
            '<': TokenKind.LT,
            '>': TokenKind.GT,
            '.': TokenKind.DOT,
            '@': TokenKind.AT,
            '?': TokenKind.QUESTION,
            '#': TokenKind.HASH,
            '(': TokenKind.LPAREN,
            ')': TokenKind.RPAREN,
            '[': TokenKind.LBRACKET,
            ']': TokenKind.RBRACKET,
            '{': TokenKind.LBRACE,
            '}': TokenKind.RBRACE,
            ';': TokenKind.SEMICOLON,
            ',': TokenKind.COMMA,
            ':': TokenKind.COLON,
        }

        if ch in one_char_map:
            self._advance()
            return self._make_token(one_char_map[ch], start, ch)

        self._advance()
        self._errors.append(LexError(f"unexpected character: {ch!r}", start))
        return self._make_token(TokenKind.IDENT, start, ch)

    def next_token(self) -> Token:
        """Lex and return the next token."""
        self._skip_whitespace()

        if self._at_end():
            pos = self._cur_pos()
            return self._make_token(TokenKind.EOF, pos, "")

        ch = self._peek()

        # Numbers
        if ch.isdigit():
            return self._lex_number()

        # b prefix: byte literal or byte string
        if ch == 'b':
            next_ch = self._peek_at(1)
            if next_ch == '"':
                return self._lex_byte_string()
            if next_ch == "'":
                return self._lex_byte_literal()
            if next_ch == 'r' and (self._peek_at(2) == '"' or self._peek_at(2) == '#'):
                # Raw byte string: br"..." or br#"..."#
                start = self._cur_pos()
                start_idx = self._pos
                self._advance()  # b
                tok = self._lex_raw_string()
                tok.kind = TokenKind.RAW_BYTE_STRING_LITERAL
                tok.text = 'b' + tok.text
                return tok

        # r prefix: raw string or raw identifier
        if ch == 'r' and (self._peek_at(1) == '"' or self._peek_at(1) == '#'):
            return self._lex_raw_string()

        # String literal
        if ch == '"':
            return self._lex_string()

        # Lifetime or character literal
        if ch == "'":
            return self._lex_lifetime_or_char()

        # Identifiers and keywords
        if ch.isalpha() or ch == '_':
            return self._lex_identifier_or_keyword()

        # Operators and punctuation
        return self._lex_operator_or_punct()

    def tokenize(self) -> list[Token]:
        """Tokenize the entire source."""
        if self._tokens:
            return list(self._tokens)

        tokens: list[Token] = []
        while True:
            tok = self.next_token()
            tokens.append(tok)
            if tok.kind == TokenKind.EOF:
                break

        self._tokens = tokens
        return list(tokens)

    def __iter__(self) -> Iterator[Token]:
        while True:
            tok = self.next_token()
            yield tok
            if tok.kind == TokenKind.EOF:
                break

    def reset(self) -> None:
        self._pos = 0
        self._line = 1
        self._col = 1
        self._tokens = []
        self._errors = []
