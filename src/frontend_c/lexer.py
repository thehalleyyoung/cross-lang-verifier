"""
C Lexer for the Cross-Language Equivalence Verifier.

Tokenizes C source code into a stream of tokens with full source location
tracking. Handles all C11 keywords, operators, punctuation, and literals
including integer suffixes, floating-point, character literals with escape
sequences, string literals, and preprocessor directives.
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

    def advance(self, ch: str) -> "SourcePos":
        if ch == '\n':
            return SourcePos(self.file, self.line + 1, 1, self.offset + 1)
        return SourcePos(self.file, self.line, self.column + 1, self.offset + 1)


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
    """All token types for C."""
    # End of file
    EOF = auto()

    # Literals
    INT_LITERAL = auto()
    FLOAT_LITERAL = auto()
    CHAR_LITERAL = auto()
    STRING_LITERAL = auto()

    # Identifier
    IDENT = auto()

    # Keywords
    KW_AUTO = auto()
    KW_BREAK = auto()
    KW_CASE = auto()
    KW_CHAR = auto()
    KW_CONST = auto()
    KW_CONTINUE = auto()
    KW_DEFAULT = auto()
    KW_DO = auto()
    KW_DOUBLE = auto()
    KW_ELSE = auto()
    KW_ENUM = auto()
    KW_EXTERN = auto()
    KW_FLOAT = auto()
    KW_FOR = auto()
    KW_GOTO = auto()
    KW_IF = auto()
    KW_INLINE = auto()
    KW_INT = auto()
    KW_LONG = auto()
    KW_REGISTER = auto()
    KW_RESTRICT = auto()
    KW_RETURN = auto()
    KW_SHORT = auto()
    KW_SIGNED = auto()
    KW_SIZEOF = auto()
    KW_STATIC = auto()
    KW_STRUCT = auto()
    KW_SWITCH = auto()
    KW_TYPEDEF = auto()
    KW_UNION = auto()
    KW_UNSIGNED = auto()
    KW_VOID = auto()
    KW_VOLATILE = auto()
    KW_WHILE = auto()

    # C99/C11 keywords
    KW_BOOL = auto()       # _Bool
    KW_COMPLEX = auto()    # _Complex
    KW_IMAGINARY = auto()  # _Imaginary
    KW_ALIGNAS = auto()    # _Alignas
    KW_ALIGNOF = auto()    # _Alignof
    KW_ATOMIC = auto()     # _Atomic
    KW_GENERIC = auto()    # _Generic
    KW_NORETURN = auto()   # _Noreturn
    KW_STATIC_ASSERT = auto()  # _Static_assert
    KW_THREAD_LOCAL = auto()   # _Thread_local

    # GCC/Clang extensions
    KW_ATTRIBUTE = auto()  # __attribute__
    KW_ASM = auto()        # __asm__ / asm
    KW_TYPEOF = auto()     # __typeof__ / typeof
    KW_EXTENSION = auto()  # __extension__
    KW_BUILTIN_VA_LIST = auto()  # __builtin_va_list
    KW_INT128 = auto()     # __int128
    KW_RESTRICT_GNU = auto()  # __restrict / __restrict__

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
    QUESTION = auto()      # ?
    DOT = auto()           # .

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
    INC = auto()           # ++
    DEC = auto()           # --
    ARROW = auto()         # ->
    ELLIPSIS = auto()      # ...

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
    HASH = auto()          # # (preprocessor)
    HASH_HASH = auto()     # ## (preprocessor paste)

    # Preprocessor directive (entire line)
    PP_DIRECTIVE = auto()


# ---------------------------------------------------------------------------
# Keyword mapping
# ---------------------------------------------------------------------------

KEYWORDS: dict[str, TokenKind] = {
    "auto": TokenKind.KW_AUTO,
    "break": TokenKind.KW_BREAK,
    "case": TokenKind.KW_CASE,
    "char": TokenKind.KW_CHAR,
    "const": TokenKind.KW_CONST,
    "continue": TokenKind.KW_CONTINUE,
    "default": TokenKind.KW_DEFAULT,
    "do": TokenKind.KW_DO,
    "double": TokenKind.KW_DOUBLE,
    "else": TokenKind.KW_ELSE,
    "enum": TokenKind.KW_ENUM,
    "extern": TokenKind.KW_EXTERN,
    "float": TokenKind.KW_FLOAT,
    "for": TokenKind.KW_FOR,
    "goto": TokenKind.KW_GOTO,
    "if": TokenKind.KW_IF,
    "inline": TokenKind.KW_INLINE,
    "int": TokenKind.KW_INT,
    "long": TokenKind.KW_LONG,
    "register": TokenKind.KW_REGISTER,
    "restrict": TokenKind.KW_RESTRICT,
    "return": TokenKind.KW_RETURN,
    "short": TokenKind.KW_SHORT,
    "signed": TokenKind.KW_SIGNED,
    "sizeof": TokenKind.KW_SIZEOF,
    "static": TokenKind.KW_STATIC,
    "struct": TokenKind.KW_STRUCT,
    "switch": TokenKind.KW_SWITCH,
    "typedef": TokenKind.KW_TYPEDEF,
    "union": TokenKind.KW_UNION,
    "unsigned": TokenKind.KW_UNSIGNED,
    "void": TokenKind.KW_VOID,
    "volatile": TokenKind.KW_VOLATILE,
    "while": TokenKind.KW_WHILE,
    # C99/C11
    "_Bool": TokenKind.KW_BOOL,
    "_Complex": TokenKind.KW_COMPLEX,
    "_Imaginary": TokenKind.KW_IMAGINARY,
    "_Alignas": TokenKind.KW_ALIGNAS,
    "_Alignof": TokenKind.KW_ALIGNOF,
    "_Atomic": TokenKind.KW_ATOMIC,
    "_Generic": TokenKind.KW_GENERIC,
    "_Noreturn": TokenKind.KW_NORETURN,
    "_Static_assert": TokenKind.KW_STATIC_ASSERT,
    "_Thread_local": TokenKind.KW_THREAD_LOCAL,
    # GCC/Clang extensions
    "__attribute__": TokenKind.KW_ATTRIBUTE,
    "__asm__": TokenKind.KW_ASM,
    "asm": TokenKind.KW_ASM,
    "__typeof__": TokenKind.KW_TYPEOF,
    "typeof": TokenKind.KW_TYPEOF,
    "__extension__": TokenKind.KW_EXTENSION,
    "__builtin_va_list": TokenKind.KW_BUILTIN_VA_LIST,
    "__int128": TokenKind.KW_INT128,
    "__restrict": TokenKind.KW_RESTRICT_GNU,
    "__restrict__": TokenKind.KW_RESTRICT_GNU,
}

# Set of keyword token kinds for quick checking
KEYWORD_KINDS = frozenset(k for k in TokenKind if k.name.startswith("KW_"))


# ---------------------------------------------------------------------------
# Token
# ---------------------------------------------------------------------------

@dataclass
class Token:
    """A single token from the C source."""
    kind: TokenKind
    text: str
    span: SourceSpan
    # Parsed values for literals
    int_value: Optional[int] = None
    float_value: Optional[float] = None
    string_value: Optional[str] = None
    char_value: Optional[int] = None
    # Integer suffix info
    int_unsigned: bool = False
    int_long: int = 0  # 0=none, 1=long, 2=long long
    # Float suffix info
    float_suffix: str = ""  # "", "f", "l"
    # Preprocessor info
    pp_directive_name: str = ""

    @property
    def is_keyword(self) -> bool:
        return self.kind in KEYWORD_KINDS

    @property
    def is_literal(self) -> bool:
        return self.kind in (
            TokenKind.INT_LITERAL, TokenKind.FLOAT_LITERAL,
            TokenKind.CHAR_LITERAL, TokenKind.STRING_LITERAL,
        )

    @property
    def is_type_specifier(self) -> bool:
        return self.kind in _TYPE_SPECIFIER_KINDS

    @property
    def is_type_qualifier(self) -> bool:
        return self.kind in _TYPE_QUALIFIER_KINDS

    @property
    def is_storage_class(self) -> bool:
        return self.kind in _STORAGE_CLASS_KINDS

    @property
    def start_pos(self) -> SourcePos:
        return self.span.start

    @property
    def end_pos(self) -> SourcePos:
        return self.span.end

    def __repr__(self) -> str:
        return f"Token({self.kind.name}, {self.text!r}, {self.span.start})"


_TYPE_SPECIFIER_KINDS = frozenset({
    TokenKind.KW_VOID, TokenKind.KW_CHAR, TokenKind.KW_SHORT,
    TokenKind.KW_INT, TokenKind.KW_LONG, TokenKind.KW_FLOAT,
    TokenKind.KW_DOUBLE, TokenKind.KW_SIGNED, TokenKind.KW_UNSIGNED,
    TokenKind.KW_BOOL, TokenKind.KW_COMPLEX, TokenKind.KW_STRUCT,
    TokenKind.KW_UNION, TokenKind.KW_ENUM, TokenKind.KW_INT128,
})

_TYPE_QUALIFIER_KINDS = frozenset({
    TokenKind.KW_CONST, TokenKind.KW_VOLATILE, TokenKind.KW_RESTRICT,
    TokenKind.KW_ATOMIC, TokenKind.KW_RESTRICT_GNU,
})

_STORAGE_CLASS_KINDS = frozenset({
    TokenKind.KW_TYPEDEF, TokenKind.KW_EXTERN, TokenKind.KW_STATIC,
    TokenKind.KW_AUTO, TokenKind.KW_REGISTER, TokenKind.KW_THREAD_LOCAL,
})


# ---------------------------------------------------------------------------
# Lexer error
# ---------------------------------------------------------------------------

class LexError(Exception):
    """Error during lexing."""
    def __init__(self, message: str, pos: SourcePos):
        self.pos = pos
        super().__init__(f"{pos}: {message}")


# ---------------------------------------------------------------------------
# Escape sequence handling
# ---------------------------------------------------------------------------

_SIMPLE_ESCAPES: dict[str, str] = {
    'a': '\a', 'b': '\b', 'f': '\f', 'n': '\n',
    'r': '\r', 't': '\t', 'v': '\v', '\\': '\\',
    "'": "'", '"': '"', '?': '?', '0': '\0',
}


def _parse_escape(source: str, idx: int, end: int) -> tuple[str, int]:
    """Parse a C escape sequence starting after the backslash.
    Returns (character, new_index)."""
    if idx >= end:
        return ('\\', idx)

    ch = source[idx]

    # Simple escapes
    if ch in _SIMPLE_ESCAPES:
        return (_SIMPLE_ESCAPES[ch], idx + 1)

    # Octal escape: \0nn
    if '0' <= ch <= '7':
        val = 0
        count = 0
        while idx < end and '0' <= source[idx] <= '7' and count < 3:
            val = val * 8 + (ord(source[idx]) - ord('0'))
            idx += 1
            count += 1
        return (chr(val & 0xFF), idx)

    # Hex escape: \xNN
    if ch == 'x':
        idx += 1
        val = 0
        count = 0
        while idx < end and count < 2:
            c = source[idx]
            if '0' <= c <= '9':
                val = val * 16 + ord(c) - ord('0')
            elif 'a' <= c <= 'f':
                val = val * 16 + ord(c) - ord('a') + 10
            elif 'A' <= c <= 'F':
                val = val * 16 + ord(c) - ord('A') + 10
            else:
                break
            idx += 1
            count += 1
        return (chr(val & 0xFF), idx)

    # Universal character names: \uNNNN, \UNNNNNNNN
    if ch == 'u':
        idx += 1
        val = 0
        for _ in range(4):
            if idx < end:
                val = val * 16 + int(source[idx], 16)
                idx += 1
        return (chr(val), idx)

    if ch == 'U':
        idx += 1
        val = 0
        for _ in range(8):
            if idx < end:
                val = val * 16 + int(source[idx], 16)
                idx += 1
        return (chr(val), idx)

    # Unknown escape - return as-is
    return (ch, idx + 1)


def _parse_string_content(source: str, start: int, end: int) -> str:
    """Parse the content of a string/char literal (between quotes)."""
    result: list[str] = []
    idx = start
    while idx < end:
        ch = source[idx]
        if ch == '\\':
            idx += 1
            parsed_ch, idx = _parse_escape(source, idx, end)
            result.append(parsed_ch)
        else:
            result.append(ch)
            idx += 1
    return ''.join(result)


# ---------------------------------------------------------------------------
# Integer literal parsing
# ---------------------------------------------------------------------------

_INT_SUFFIX_RE = re.compile(
    r'(u|U)?(ll|LL|l|L)?(u|U)?$'
)


def _parse_int_suffix(suffix: str) -> tuple[bool, int]:
    """Parse an integer suffix, return (is_unsigned, long_count)."""
    s = suffix.lower()
    unsigned = 'u' in s
    if 'll' in s:
        long_count = 2
    elif 'l' in s:
        long_count = 1
    else:
        long_count = 0
    return unsigned, long_count


def _parse_int_literal(text: str) -> tuple[int, bool, int]:
    """Parse an integer literal text, return (value, is_unsigned, long_count)."""
    # Separate suffix from digits
    i = 0
    base = 10
    txt = text.lower()

    if txt.startswith('0x') or txt.startswith('0X'):
        base = 16
        i = 2
    elif txt.startswith('0b') or txt.startswith('0B'):
        base = 2
        i = 2
    elif txt.startswith('0') and len(txt) > 1 and txt[1].isdigit():
        base = 8
        i = 1

    # Find end of digits
    digit_end = i
    while digit_end < len(text):
        c = text[digit_end].lower()
        if base == 16 and c in '0123456789abcdef':
            digit_end += 1
        elif base == 10 and c.isdigit():
            digit_end += 1
        elif base == 8 and '0' <= c <= '7':
            digit_end += 1
        elif base == 2 and c in '01':
            digit_end += 1
        elif c == '_':
            # C23 digit separators
            digit_end += 1
        else:
            break

    digits = text[i:digit_end].replace('_', '')
    suffix = text[digit_end:]

    value = int(digits, base) if digits else 0
    unsigned, long_count = _parse_int_suffix(suffix)

    return value, unsigned, long_count


# ---------------------------------------------------------------------------
# Float literal parsing
# ---------------------------------------------------------------------------

def _parse_float_literal(text: str) -> tuple[float, str]:
    """Parse a float literal, return (value, suffix)."""
    suffix = ""
    num_text = text

    if text.lower().endswith('f'):
        suffix = "f"
        num_text = text[:-1]
    elif text.lower().endswith('l'):
        suffix = "l"
        num_text = text[:-1]

    num_text = num_text.replace('_', '')
    value = float(num_text)
    return value, suffix


# ---------------------------------------------------------------------------
# CLexer
# ---------------------------------------------------------------------------

class CLexer:
    """Tokenizer for C source code.

    Usage::

        lexer = CLexer(source, filename="test.c")
        tokens = lexer.tokenize()
        for token in tokens:
            print(token)

    Or iterate lazily::

        lexer = CLexer(source)
        for token in lexer:
            if token.kind == TokenKind.EOF:
                break
            process(token)
    """

    def __init__(self, source: str, filename: str = "<input>") -> None:
        self._source = source
        self._filename = filename
        self._pos = 0
        self._line = 1
        self._col = 1
        self._tokens: list[Token] = []
        self._errors: list[LexError] = []
        self._typedef_names: set[str] = set()

    @property
    def source(self) -> str:
        return self._source

    @property
    def errors(self) -> list[LexError]:
        return list(self._errors)

    def add_typedef(self, name: str) -> None:
        """Register a typedef name so the lexer can distinguish it."""
        self._typedef_names.add(name)

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

    def _match(self, expected: str) -> bool:
        if self._pos < len(self._source) and self._source[self._pos] == expected:
            self._advance()
            return True
        return False

    def _at_end(self) -> bool:
        return self._pos >= len(self._source)

    def _skip_whitespace(self) -> None:
        """Skip whitespace characters (but not newlines in preprocessor mode)."""
        while not self._at_end():
            ch = self._peek()
            if ch in ' \t\r\n\f\v':
                self._advance()
            elif ch == '/' and self._peek_at(1) == '/':
                self._skip_line_comment()
            elif ch == '/' and self._peek_at(1) == '*':
                self._skip_block_comment()
            elif ch == '\\' and self._peek_at(1) == '\n':
                # Line continuation
                self._advance()
                self._advance()
            else:
                break

    def _skip_line_comment(self) -> None:
        """Skip // comment to end of line."""
        self._advance()  # /
        self._advance()  # /
        while not self._at_end() and self._peek() != '\n':
            self._advance()

    def _skip_block_comment(self) -> None:
        """Skip /* ... */ comment."""
        start = self._cur_pos()
        self._advance()  # /
        self._advance()  # *
        while not self._at_end():
            if self._peek() == '*' and self._peek_at(1) == '/':
                self._advance()  # *
                self._advance()  # /
                return
            self._advance()
        self._errors.append(LexError("unterminated block comment", start))

    def _make_token(self, kind: TokenKind, start: SourcePos, text: str, **kwargs) -> Token:
        end = self._cur_pos()
        span = SourceSpan(start, end)
        return Token(kind=kind, text=text, span=span, **kwargs)

    def _lex_number(self) -> Token:
        """Lex an integer or floating-point literal."""
        start = self._cur_pos()
        start_idx = self._pos
        is_float = False

        # Check for hex, binary, or octal prefix
        if self._peek() == '0':
            self._advance()
            next_ch = self._peek()

            if next_ch in 'xX':
                # Hex literal
                self._advance()
                while not self._at_end() and (
                    self._peek() in '0123456789abcdefABCDEF_'
                ):
                    self._advance()
                # Check for hex float (0x1.2p3)
                if self._peek() == '.':
                    is_float = True
                    self._advance()
                    while not self._at_end() and self._peek() in '0123456789abcdefABCDEF_':
                        self._advance()
                if self._peek() in 'pP':
                    is_float = True
                    self._advance()
                    if self._peek() in '+-':
                        self._advance()
                    while not self._at_end() and self._peek().isdigit():
                        self._advance()

            elif next_ch in 'bB':
                # Binary literal
                self._advance()
                while not self._at_end() and self._peek() in '01_':
                    self._advance()

            elif next_ch == '.' or next_ch in 'eE':
                # Float starting with 0
                is_float = True
                if next_ch == '.':
                    self._advance()
                    while not self._at_end() and (self._peek().isdigit() or self._peek() == '_'):
                        self._advance()
                if self._peek() in 'eE':
                    self._advance()
                    if self._peek() in '+-':
                        self._advance()
                    while not self._at_end() and self._peek().isdigit():
                        self._advance()

            else:
                # Octal or just zero
                while not self._at_end() and '0' <= self._peek() <= '7':
                    self._advance()
                # Could be float like 0.5
                if self._peek() == '.':
                    is_float = True
                    self._advance()
                    while not self._at_end() and (self._peek().isdigit() or self._peek() == '_'):
                        self._advance()
                    if self._peek() in 'eE':
                        self._advance()
                        if self._peek() in '+-':
                            self._advance()
                        while not self._at_end() and self._peek().isdigit():
                            self._advance()

        else:
            # Decimal
            while not self._at_end() and (self._peek().isdigit() or self._peek() == '_'):
                self._advance()

            if self._peek() == '.' and self._peek_at(1) != '.':
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

        # Parse suffix
        if is_float:
            if self._peek() in 'fFlL':
                self._advance()
            text = self._source[start_idx:self._pos]
            value, suffix = _parse_float_literal(text)
            return self._make_token(
                TokenKind.FLOAT_LITERAL, start, text,
                float_value=value, float_suffix=suffix,
            )
        else:
            # Integer suffix: u, l, ll, ul, ull, etc.
            suffix_start = self._pos
            while not self._at_end() and self._peek() in 'uUlL':
                self._advance()
            text = self._source[start_idx:self._pos]
            value, unsigned, long_count = _parse_int_literal(text)
            return self._make_token(
                TokenKind.INT_LITERAL, start, text,
                int_value=value, int_unsigned=unsigned, int_long=long_count,
            )

    def _lex_string(self) -> Token:
        """Lex a string literal."""
        start = self._cur_pos()
        start_idx = self._pos
        self._advance()  # opening "

        content_start = self._pos
        while not self._at_end() and self._peek() != '"':
            if self._peek() == '\\':
                self._advance()  # skip backslash
                if not self._at_end():
                    self._advance()  # skip escaped char
            elif self._peek() == '\n':
                # Unterminated string
                break
            else:
                self._advance()

        content_end = self._pos
        if not self._at_end() and self._peek() == '"':
            self._advance()  # closing "

        text = self._source[start_idx:self._pos]
        content = _parse_string_content(self._source, content_start, content_end)

        return self._make_token(
            TokenKind.STRING_LITERAL, start, text,
            string_value=content,
        )

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
        if not self._at_end() and self._peek() == "'":
            self._advance()  # closing '

        text = self._source[start_idx:self._pos]
        content = _parse_string_content(self._source, content_start, content_end)
        char_val = ord(content[0]) if content else 0

        return self._make_token(
            TokenKind.CHAR_LITERAL, start, text,
            char_value=char_val,
        )

    def _lex_identifier_or_keyword(self) -> Token:
        """Lex an identifier or keyword."""
        start = self._cur_pos()
        start_idx = self._pos

        while not self._at_end() and (
            self._peek().isalnum() or self._peek() == '_'
        ):
            self._advance()

        text = self._source[start_idx:self._pos]

        # Check for keywords
        kind = KEYWORDS.get(text, TokenKind.IDENT)

        return self._make_token(kind, start, text)

    def _lex_preprocessor(self) -> Token:
        """Lex a preprocessor directive (# at start of logical line)."""
        start = self._cur_pos()
        start_idx = self._pos
        self._advance()  # #

        # Skip whitespace between # and directive name
        while not self._at_end() and self._peek() in ' \t':
            self._advance()

        # Read directive name
        dir_name_start = self._pos
        while not self._at_end() and self._peek().isalpha():
            self._advance()
        dir_name = self._source[dir_name_start:self._pos]

        # Read until end of line (handling line continuations)
        while not self._at_end():
            if self._peek() == '\n':
                break
            if self._peek() == '\\' and self._peek_at(1) == '\n':
                self._advance()  # backslash
                self._advance()  # newline
                continue
            self._advance()

        text = self._source[start_idx:self._pos]
        return self._make_token(
            TokenKind.PP_DIRECTIVE, start, text,
            pp_directive_name=dir_name,
        )

    def _lex_operator_or_punct(self) -> Token:
        """Lex an operator or punctuation token."""
        start = self._cur_pos()
        ch = self._peek()
        next_ch = self._peek_at(1)
        next2_ch = self._peek_at(2)

        # Three-character operators
        if ch == '.' and next_ch == '.' and next2_ch == '.':
            self._advance_n(3)
            return self._make_token(TokenKind.ELLIPSIS, start, "...")
        if ch == '<' and next_ch == '<' and next2_ch == '=':
            self._advance_n(3)
            return self._make_token(TokenKind.SHL_ASSIGN, start, "<<=")
        if ch == '>' and next_ch == '>' and next2_ch == '=':
            self._advance_n(3)
            return self._make_token(TokenKind.SHR_ASSIGN, start, ">>=")
        if ch == '#' and next_ch == '#':
            self._advance_n(2)
            return self._make_token(TokenKind.HASH_HASH, start, "##")

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
            '++': TokenKind.INC,
            '--': TokenKind.DEC,
            '->': TokenKind.ARROW,
        }
        if two_char in two_char_map:
            self._advance_n(2)
            return self._make_token(two_char_map[two_char], start, two_char)

        # Single-character operators and punctuation
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
            '?': TokenKind.QUESTION,
            '.': TokenKind.DOT,
            '(': TokenKind.LPAREN,
            ')': TokenKind.RPAREN,
            '[': TokenKind.LBRACKET,
            ']': TokenKind.RBRACKET,
            '{': TokenKind.LBRACE,
            '}': TokenKind.RBRACE,
            ';': TokenKind.SEMICOLON,
            ',': TokenKind.COMMA,
            ':': TokenKind.COLON,
            '#': TokenKind.HASH,
        }

        if ch in one_char_map:
            self._advance()
            return self._make_token(one_char_map[ch], start, ch)

        # Unknown character
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

        # Preprocessor directive: # at beginning of line
        if ch == '#' and (self._col == 1 or self._peek_at(-1) == '\n'):
            # Check if this is a preprocessor directive
            next_non_ws = self._pos + 1
            while next_non_ws < len(self._source) and self._source[next_non_ws] in ' \t':
                next_non_ws += 1
            if next_non_ws < len(self._source) and self._source[next_non_ws].isalpha():
                return self._lex_preprocessor()

        # Numbers
        if ch.isdigit():
            return self._lex_number()
        if ch == '.' and self._peek_at(1).isdigit():
            return self._lex_number()

        # String literal
        if ch == '"':
            return self._lex_string()
        # L"..." wide string
        if ch == 'L' and self._peek_at(1) == '"':
            self._advance()
            tok = self._lex_string()
            tok.text = 'L' + tok.text
            return tok

        # Character literal
        if ch == "'":
            return self._lex_char()
        # L'c' wide char
        if ch == 'L' and self._peek_at(1) == "'":
            self._advance()
            tok = self._lex_char()
            tok.text = 'L' + tok.text
            return tok

        # Identifiers and keywords
        if ch.isalpha() or ch == '_':
            return self._lex_identifier_or_keyword()

        # Operators and punctuation
        return self._lex_operator_or_punct()

    def tokenize(self) -> list[Token]:
        """Tokenize the entire source, returning a list of tokens (including EOF)."""
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
        """Iterate over tokens lazily."""
        while True:
            tok = self.next_token()
            yield tok
            if tok.kind == TokenKind.EOF:
                break

    def reset(self) -> None:
        """Reset the lexer to the beginning of the source."""
        self._pos = 0
        self._line = 1
        self._col = 1
        self._tokens = []
        self._errors = []


# ---------------------------------------------------------------------------
# Utility: check if a token can start a declaration
# ---------------------------------------------------------------------------

def can_start_declaration(token: Token, typedef_names: set[str] | None = None) -> bool:
    """Return True if the token can begin a C declaration."""
    if token.is_type_specifier or token.is_type_qualifier or token.is_storage_class:
        return True
    if token.kind == TokenKind.KW_INLINE:
        return True
    if token.kind == TokenKind.KW_NORETURN:
        return True
    if token.kind == TokenKind.KW_ATTRIBUTE:
        return True
    if token.kind == TokenKind.KW_EXTENSION:
        return True
    if token.kind == TokenKind.IDENT and typedef_names and token.text in typedef_names:
        return True
    return False


def can_start_type(token: Token, typedef_names: set[str] | None = None) -> bool:
    """Return True if the token can begin a type specifier."""
    if token.is_type_specifier or token.is_type_qualifier:
        return True
    if token.kind == TokenKind.IDENT and typedef_names and token.text in typedef_names:
        return True
    return False
