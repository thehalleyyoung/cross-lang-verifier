"""
Recursive descent C parser for the Cross-Language Equivalence Verifier.

Parses C source code (targeting C2Rust output patterns) into a C AST.
Supports expression parsing with operator precedence climbing, statement
parsing, declaration parsing (variables, functions, structs, unions, enums,
typedefs), type specifier parsing with qualifiers, pointers, and arrays,
initializer parsing, and basic error recovery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from .lexer import (
    CLexer, Token, TokenKind, SourcePos, SourceSpan,
    can_start_declaration, can_start_type, KEYWORDS,
)
from .c_ast import (
    # Types
    CType, VoidCType, IntCType, FloatCType, PointerCType, ArrayCType,
    FunctionCType, StructRefCType, UnionRefCType, EnumRefCType,
    TypedefRefCType, QualifiedCType, AtomicCType, TypeofCType,
    # Qualifiers/storage
    TypeQualifier, StorageClass, FunctionSpecifier,
    # Declarations
    Decl, ParamDecl, VarDecl, FunctionDecl, TypedefDecl, FieldDecl,
    StructDecl, UnionDecl, EnumDecl, EnumeratorDecl, StaticAssertDecl,
    Attribute,
    # Statements
    Stmt, CompoundStmt, ExprStmt, IfStmt, WhileStmt, DoWhileStmt,
    ForStmt, SwitchStmt, CaseStmt, ReturnStmt, BreakStmt, ContinueStmt,
    GotoStmt, LabelStmt, NullStmt, DeclStmt, AsmStmt,
    # Expressions
    Expr, IntLiteral, FloatLiteral, CharLiteral, StringLiteral,
    IdentExpr, BinaryExpr, UnaryExpr, CastExpr, SizeofExpr,
    AlignofExpr, CallExpr, MemberExpr, ArraySubscriptExpr,
    TernaryExpr, CommaExpr, InitListExpr, Designator,
    CompoundLiteralExpr, ParenExpr, StmtExpr, BuiltinCallExpr,
    GenericExpr, VaArgExpr, OffsetofExpr, TypesCompatibleExpr,
    DesignatedInitExpr,
    BinaryOp as CASTBinaryOp, UnaryOp as CASTUnaryOp,
    # Top level
    TranslationUnit, NodeLocation,
)


# ---------------------------------------------------------------------------
# Parse error
# ---------------------------------------------------------------------------

class ParseError(Exception):
    """Error raised during parsing."""
    def __init__(self, message: str, pos: Optional[SourcePos] = None,
                 token: Optional[Token] = None):
        self.pos = pos or (token.start_pos if token else None)
        self.token = token
        loc = str(self.pos) if self.pos else "<unknown>"
        super().__init__(f"{loc}: {message}")


# ---------------------------------------------------------------------------
# Operator precedence tables
# ---------------------------------------------------------------------------

# Binary operator precedence (higher = tighter binding)
_BINOP_PRECEDENCE: dict[TokenKind, int] = {
    TokenKind.ASSIGN: 2,
    TokenKind.PLUS_ASSIGN: 2,
    TokenKind.MINUS_ASSIGN: 2,
    TokenKind.STAR_ASSIGN: 2,
    TokenKind.SLASH_ASSIGN: 2,
    TokenKind.PERCENT_ASSIGN: 2,
    TokenKind.SHL_ASSIGN: 2,
    TokenKind.SHR_ASSIGN: 2,
    TokenKind.AMP_ASSIGN: 2,
    TokenKind.PIPE_ASSIGN: 2,
    TokenKind.CARET_ASSIGN: 2,
    # Ternary ?: is precedence 3 (handled specially)
    TokenKind.OR: 4,
    TokenKind.AND: 5,
    TokenKind.PIPE: 6,
    TokenKind.CARET: 7,
    TokenKind.AMP: 8,
    TokenKind.EQ: 9,
    TokenKind.NE: 9,
    TokenKind.LT: 10,
    TokenKind.GT: 10,
    TokenKind.LE: 10,
    TokenKind.GE: 10,
    TokenKind.SHL: 11,
    TokenKind.SHR: 11,
    TokenKind.PLUS: 12,
    TokenKind.MINUS: 12,
    TokenKind.STAR: 13,
    TokenKind.SLASH: 13,
    TokenKind.PERCENT: 13,
}

_TOKEN_TO_BINOP: dict[TokenKind, CASTBinaryOp] = {
    TokenKind.PLUS: CASTBinaryOp.ADD,
    TokenKind.MINUS: CASTBinaryOp.SUB,
    TokenKind.STAR: CASTBinaryOp.MUL,
    TokenKind.SLASH: CASTBinaryOp.DIV,
    TokenKind.PERCENT: CASTBinaryOp.MOD,
    TokenKind.SHL: CASTBinaryOp.SHL,
    TokenKind.SHR: CASTBinaryOp.SHR,
    TokenKind.AMP: CASTBinaryOp.BITAND,
    TokenKind.PIPE: CASTBinaryOp.BITOR,
    TokenKind.CARET: CASTBinaryOp.BITXOR,
    TokenKind.AND: CASTBinaryOp.LOGAND,
    TokenKind.OR: CASTBinaryOp.LOGOR,
    TokenKind.EQ: CASTBinaryOp.EQ,
    TokenKind.NE: CASTBinaryOp.NE,
    TokenKind.LT: CASTBinaryOp.LT,
    TokenKind.GT: CASTBinaryOp.GT,
    TokenKind.LE: CASTBinaryOp.LE,
    TokenKind.GE: CASTBinaryOp.GE,
    TokenKind.ASSIGN: CASTBinaryOp.ASSIGN,
    TokenKind.PLUS_ASSIGN: CASTBinaryOp.ADD_ASSIGN,
    TokenKind.MINUS_ASSIGN: CASTBinaryOp.SUB_ASSIGN,
    TokenKind.STAR_ASSIGN: CASTBinaryOp.MUL_ASSIGN,
    TokenKind.SLASH_ASSIGN: CASTBinaryOp.DIV_ASSIGN,
    TokenKind.PERCENT_ASSIGN: CASTBinaryOp.MOD_ASSIGN,
    TokenKind.SHL_ASSIGN: CASTBinaryOp.SHL_ASSIGN,
    TokenKind.SHR_ASSIGN: CASTBinaryOp.SHR_ASSIGN,
    TokenKind.AMP_ASSIGN: CASTBinaryOp.AND_ASSIGN,
    TokenKind.PIPE_ASSIGN: CASTBinaryOp.OR_ASSIGN,
    TokenKind.CARET_ASSIGN: CASTBinaryOp.XOR_ASSIGN,
}

# Assignment operators are right-associative
_RIGHT_ASSOC = frozenset({
    TokenKind.ASSIGN, TokenKind.PLUS_ASSIGN, TokenKind.MINUS_ASSIGN,
    TokenKind.STAR_ASSIGN, TokenKind.SLASH_ASSIGN, TokenKind.PERCENT_ASSIGN,
    TokenKind.SHL_ASSIGN, TokenKind.SHR_ASSIGN, TokenKind.AMP_ASSIGN,
    TokenKind.PIPE_ASSIGN, TokenKind.CARET_ASSIGN,
})


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class CParser:
    """Recursive descent parser for C (C2Rust output subset).

    Usage::

        parser = CParser(source, filename="test.c")
        tu = parser.parse()
        for func in tu.function_definitions():
            print(func.name)
    """

    def __init__(self, source: str, filename: str = "<input>",
                 lenient: bool = False) -> None:
        self._lexer = CLexer(source, filename)
        self._filename = filename
        self._tokens: list[Token] = []
        self._pos = 0
        self._errors: list[ParseError] = []
        self._typedef_names: set[str] = set()
        self._in_typedef = False
        self.lenient = lenient
        # Pre-register common C2Rust typedefs
        for name in ("size_t", "ssize_t", "ptrdiff_t", "intptr_t", "uintptr_t",
                      "int8_t", "int16_t", "int32_t", "int64_t",
                      "uint8_t", "uint16_t", "uint32_t", "uint64_t",
                      "int_least8_t", "int_least16_t", "int_least32_t", "int_least64_t",
                      "uint_least8_t", "uint_least16_t", "uint_least32_t", "uint_least64_t",
                      "int_fast8_t", "int_fast16_t", "int_fast32_t", "int_fast64_t",
                      "uint_fast8_t", "uint_fast16_t", "uint_fast32_t", "uint_fast64_t",
                      "intmax_t", "uintmax_t", "wchar_t", "char16_t", "char32_t",
                      "FILE", "va_list", "__va_list_tag", "bool", "true", "false",
                      "NULL", "errno_t", "rsize_t", "max_align_t"):
            self._typedef_names.add(name)

    @property
    def errors(self) -> list[ParseError]:
        return list(self._errors)

    def _tokenize(self) -> None:
        """Tokenize the source if not already done."""
        if not self._tokens:
            self._tokens = self._lexer.tokenize()
            self._pos = 0

    def _cur(self) -> Token:
        """Return the current token."""
        if self._pos >= len(self._tokens):
            return self._tokens[-1]  # EOF
        return self._tokens[self._pos]

    def _peek(self, offset: int = 0) -> Token:
        """Peek at a token ahead."""
        idx = self._pos + offset
        if idx >= len(self._tokens):
            return self._tokens[-1]
        return self._tokens[idx]

    def _advance(self) -> Token:
        """Consume and return the current token."""
        tok = self._cur()
        if tok.kind != TokenKind.EOF:
            self._pos += 1
        return tok

    def _expect(self, kind: TokenKind, msg: str = "") -> Token:
        """Consume the next token, expecting it to be of the given kind."""
        tok = self._cur()
        if tok.kind != kind:
            if not msg:
                msg = f"expected {kind.name}, got {tok.kind.name} ({tok.text!r})"
            self._error(msg, tok)
            return tok
        return self._advance()

    def _match(self, kind: TokenKind) -> Optional[Token]:
        """If the current token matches, consume and return it; else None."""
        if self._cur().kind == kind:
            return self._advance()
        return None

    def _at(self, kind: TokenKind) -> bool:
        """Check if current token is of the given kind."""
        return self._cur().kind == kind

    def _at_any(self, *kinds: TokenKind) -> bool:
        """Check if current token is any of the given kinds."""
        return self._cur().kind in kinds

    def _error(self, msg: str, token: Optional[Token] = None) -> ParseError:
        """Record a parse error."""
        tok = token or self._cur()
        err = ParseError(msg, token=tok)
        self._errors.append(err)
        return err

    def _loc(self, start_token: Token) -> NodeLocation:
        """Create a NodeLocation from a start token to current position."""
        end = self._cur()
        span = SourceSpan(start_token.span.start, end.span.start)
        return NodeLocation(span=span)

    def _skip_to(self, *kinds: TokenKind) -> None:
        """Skip tokens until one of the given kinds (or EOF)."""
        while not self._at(TokenKind.EOF) and not self._at_any(*kinds):
            self._advance()

    def _skip_to_recovery(self) -> None:
        """Skip tokens until a likely recovery point (semicolon, brace, or EOF)."""
        while not self._at(TokenKind.EOF):
            if self._at_any(TokenKind.SEMICOLON, TokenKind.RBRACE, TokenKind.LBRACE):
                return
            self._advance()

    def _skip_balanced_parens(self) -> None:
        """Skip balanced parentheses."""
        depth = 0
        while not self._at(TokenKind.EOF):
            if self._at(TokenKind.LPAREN):
                depth += 1
            elif self._at(TokenKind.RPAREN):
                if depth <= 1:
                    self._advance()
                    return
                depth -= 1
            self._advance()

    # -------------------------------------------------------------------
    # Top-level parsing
    # -------------------------------------------------------------------

    def parse(self) -> TranslationUnit:
        """Parse a complete C translation unit."""
        self._tokenize()
        tu = TranslationUnit(filename=self._filename)
        start = self._cur()

        while not self._at(TokenKind.EOF):
            # Skip preprocessor directives
            if self._at(TokenKind.PP_DIRECTIVE):
                self._advance()
                continue

            try:
                decl = self._parse_external_declaration()
                if decl is not None:
                    if isinstance(decl, list):
                        tu.declarations.extend(decl)
                    else:
                        tu.declarations.append(decl)
            except ParseError as e:
                self._errors.append(e)
                self._skip_to(TokenKind.SEMICOLON, TokenKind.RBRACE)
                if self._at(TokenKind.SEMICOLON):
                    self._advance()

        tu.loc = self._loc(start)
        return tu

    def _parse_external_declaration(self) -> Optional[Decl | list[Decl]]:
        """Parse a top-level declaration."""
        # Handle __extension__
        if self._at(TokenKind.KW_EXTENSION):
            self._advance()

        # _Static_assert
        if self._at(TokenKind.KW_STATIC_ASSERT):
            return self._parse_static_assert()

        # __attribute__ at top level
        if self._at(TokenKind.KW_ATTRIBUTE):
            attrs = self._parse_attributes()
            # Followed by declaration
            if self._at(TokenKind.SEMICOLON):
                self._advance()
                return None

        # _Alignas at top level — skip the specifier
        if self._at(TokenKind.KW_ALIGNAS):
            self._advance()
            if self._match(TokenKind.LPAREN):
                depth = 1
                while depth > 0 and not self._at(TokenKind.EOF):
                    if self._at(TokenKind.LPAREN):
                        depth += 1
                    elif self._at(TokenKind.RPAREN):
                        depth -= 1
                        if depth == 0:
                            self._advance()
                            break
                    self._advance()

        # Parse declaration specifiers
        specs = self._parse_declaration_specifiers()
        if specs is None:
            # Unknown token - skip it
            tok = self._advance()
            self._error(f"unexpected token {tok.text!r} at file scope", tok)
            return None

        storage, qualifiers, type_spec, func_spec, is_typedef, attrs = specs

        # Check for standalone struct/union/enum definition
        if self._at(TokenKind.SEMICOLON):
            self._advance()
            if isinstance(type_spec, StructRefCType) and isinstance(type_spec, StructRefCType):
                return None  # Forward declaration
            return None

        # Parse declarator(s)
        return self._parse_init_declarator_list(
            type_spec, storage, qualifiers, func_spec, is_typedef, attrs
        )

    def _parse_init_declarator_list(
        self,
        base_type: CType,
        storage: StorageClass,
        qualifiers: list[TypeQualifier],
        func_spec: FunctionSpecifier,
        is_typedef: bool,
        attrs: list[Attribute],
    ) -> Decl | list[Decl]:
        """Parse one or more declarators after the type specifier."""
        start = self._cur()
        decls: list[Decl] = []

        first = True
        while True:
            if not first:
                if not self._match(TokenKind.COMMA):
                    break
            first = False

            if self.lenient:
                try:
                    name, decl_type = self._parse_declarator(base_type)
                except ParseError as e:
                    self._errors.append(e)
                    self._skip_to(TokenKind.SEMICOLON, TokenKind.LBRACE)
                    if self._at(TokenKind.SEMICOLON):
                        self._advance()
                    return None
            else:
                name, decl_type = self._parse_declarator(base_type)

            if is_typedef:
                td = TypedefDecl(
                    name=name,
                    underlying_type=decl_type,
                    loc=self._loc(start),
                    attributes=attrs,
                )
                self._typedef_names.add(name)
                self._lexer.add_typedef(name)
                decls.append(td)

            elif isinstance(decl_type, FunctionCType) and self._at(TokenKind.LBRACE):
                # Function definition
                body = self._parse_compound_stmt()
                fd = FunctionDecl(
                    name=name,
                    return_type=decl_type.return_type,
                    params=decl_type.params,
                    body=body,
                    storage_class=storage,
                    is_variadic=decl_type.is_variadic,
                    is_inline=func_spec == FunctionSpecifier.INLINE,
                    is_noreturn=func_spec == FunctionSpecifier.NORETURN,
                    is_definition=True,
                    loc=self._loc(start),
                    attributes=attrs,
                )
                return fd  # Function definitions are always single

            elif isinstance(decl_type, FunctionCType):
                # Function declaration (prototype)
                fd = FunctionDecl(
                    name=name,
                    return_type=decl_type.return_type,
                    params=decl_type.params,
                    storage_class=storage,
                    is_variadic=decl_type.is_variadic,
                    is_inline=func_spec == FunctionSpecifier.INLINE,
                    is_noreturn=func_spec == FunctionSpecifier.NORETURN,
                    is_definition=False,
                    loc=self._loc(start),
                    attributes=attrs,
                )
                decls.append(fd)

            else:
                # Variable declaration
                init = None
                if self._match(TokenKind.ASSIGN):
                    init = self._parse_initializer()

                vd = VarDecl(
                    name=name,
                    type_name=decl_type,
                    initializer=init,
                    storage_class=storage,
                    qualifiers=qualifiers,
                    is_global=True,
                    loc=self._loc(start),
                    attributes=attrs,
                )
                decls.append(vd)

        self._expect(TokenKind.SEMICOLON, "expected ';' after declaration")

        if len(decls) == 1:
            return decls[0]
        return decls

    # -------------------------------------------------------------------
    # Declaration specifiers
    # -------------------------------------------------------------------

    def _parse_declaration_specifiers(
        self,
    ) -> Optional[tuple[StorageClass, list[TypeQualifier], CType, FunctionSpecifier, bool, list[Attribute]]]:
        """Parse declaration specifiers: storage class, type qualifiers, type specifier."""
        storage = StorageClass.NONE
        qualifiers: list[TypeQualifier] = []
        func_spec = FunctionSpecifier.NONE
        is_typedef = False
        attrs: list[Attribute] = []

        # Collect specifiers
        has_signed = False
        has_unsigned = False
        has_short = False
        long_count = 0
        has_int = False
        has_char = False
        has_void = False
        has_float = False
        has_double = False
        has_bool = False
        has_int128 = False
        has_any_type = False
        struct_type: Optional[CType] = None
        typedef_name: Optional[str] = None

        while True:
            tok = self._cur()

            # Storage class
            if tok.kind == TokenKind.KW_TYPEDEF:
                is_typedef = True
                storage = StorageClass.TYPEDEF
                self._advance()
            elif tok.kind == TokenKind.KW_EXTERN:
                storage = StorageClass.EXTERN
                self._advance()
            elif tok.kind == TokenKind.KW_STATIC:
                storage = StorageClass.STATIC
                self._advance()
            elif tok.kind == TokenKind.KW_AUTO:
                storage = StorageClass.AUTO
                self._advance()
            elif tok.kind == TokenKind.KW_REGISTER:
                storage = StorageClass.REGISTER
                self._advance()
            elif tok.kind == TokenKind.KW_THREAD_LOCAL:
                storage = StorageClass.THREAD_LOCAL
                self._advance()

            # Type qualifiers
            elif tok.kind == TokenKind.KW_CONST:
                qualifiers.append(TypeQualifier.CONST)
                self._advance()
            elif tok.kind == TokenKind.KW_VOLATILE:
                qualifiers.append(TypeQualifier.VOLATILE)
                self._advance()
            elif tok.kind in (TokenKind.KW_RESTRICT, TokenKind.KW_RESTRICT_GNU):
                qualifiers.append(TypeQualifier.RESTRICT)
                self._advance()
            elif tok.kind == TokenKind.KW_ATOMIC:
                # _Atomic(type) is a type specifier; bare _Atomic is a qualifier
                if self._peek(1).kind == TokenKind.LPAREN:
                    self._advance()  # _Atomic
                    self._advance()  # (
                    inner_type = self._parse_type_name()
                    self._expect(TokenKind.RPAREN)
                    struct_type = AtomicCType(base=inner_type)
                    has_any_type = True
                else:
                    qualifiers.append(TypeQualifier.ATOMIC)
                    self._advance()

            # Function specifiers
            elif tok.kind == TokenKind.KW_INLINE:
                func_spec = FunctionSpecifier.INLINE
                self._advance()
            elif tok.kind == TokenKind.KW_NORETURN:
                func_spec = FunctionSpecifier.NORETURN
                self._advance()

            # Type specifiers
            elif tok.kind == TokenKind.KW_VOID:
                has_void = True
                has_any_type = True
                self._advance()
            elif tok.kind == TokenKind.KW_CHAR:
                has_char = True
                has_any_type = True
                self._advance()
            elif tok.kind == TokenKind.KW_SHORT:
                has_short = True
                has_any_type = True
                self._advance()
            elif tok.kind == TokenKind.KW_INT:
                has_int = True
                has_any_type = True
                self._advance()
            elif tok.kind == TokenKind.KW_LONG:
                long_count += 1
                has_any_type = True
                self._advance()
            elif tok.kind == TokenKind.KW_FLOAT:
                has_float = True
                has_any_type = True
                self._advance()
            elif tok.kind == TokenKind.KW_DOUBLE:
                has_double = True
                has_any_type = True
                self._advance()
            elif tok.kind == TokenKind.KW_SIGNED:
                has_signed = True
                has_any_type = True
                self._advance()
            elif tok.kind == TokenKind.KW_UNSIGNED:
                has_unsigned = True
                has_any_type = True
                self._advance()
            elif tok.kind == TokenKind.KW_BOOL:
                has_bool = True
                has_any_type = True
                self._advance()
            elif tok.kind == TokenKind.KW_INT128:
                has_int128 = True
                has_any_type = True
                self._advance()

            # struct/union/enum
            elif tok.kind == TokenKind.KW_STRUCT:
                struct_type = self._parse_struct_or_union_spec(is_struct=True)
                has_any_type = True
            elif tok.kind == TokenKind.KW_UNION:
                struct_type = self._parse_struct_or_union_spec(is_struct=False)
                has_any_type = True
            elif tok.kind == TokenKind.KW_ENUM:
                struct_type = self._parse_enum_spec()
                has_any_type = True

            # __attribute__
            elif tok.kind == TokenKind.KW_ATTRIBUTE:
                attrs.extend(self._parse_attributes())

            # __extension__
            elif tok.kind == TokenKind.KW_EXTENSION:
                self._advance()

            # typeof
            elif tok.kind == TokenKind.KW_TYPEOF:
                struct_type = self._parse_typeof()
                has_any_type = True

            # _Alignas(type_or_expr) - consume and skip
            elif tok.kind == TokenKind.KW_ALIGNAS:
                self._advance()
                if self._match(TokenKind.LPAREN):
                    # Skip balanced parens content
                    depth = 1
                    while depth > 0 and not self._at(TokenKind.EOF):
                        if self._at(TokenKind.LPAREN):
                            depth += 1
                        elif self._at(TokenKind.RPAREN):
                            depth -= 1
                            if depth == 0:
                                self._advance()
                                break
                        self._advance()

            # Typedef name
            elif (tok.kind == TokenKind.IDENT and not has_any_type
                  and tok.text in self._typedef_names):
                typedef_name = tok.text
                has_any_type = True
                self._advance()

            else:
                break

        if not has_any_type and storage == StorageClass.NONE and not qualifiers and func_spec == FunctionSpecifier.NONE:
            return None

        # Build the type
        type_spec: CType
        if struct_type is not None:
            type_spec = struct_type
        elif typedef_name is not None:
            type_spec = TypedefRefCType(name=typedef_name)
        elif has_void:
            type_spec = VoidCType()
        elif has_bool:
            type_spec = IntCType(is_bool=True)
        elif has_float:
            type_spec = FloatCType(is_float=True)
        elif has_double:
            if long_count > 0:
                type_spec = FloatCType(is_long_double=True)
            else:
                type_spec = FloatCType(is_double=True)
        elif has_int128:
            type_spec = IntCType(
                is_int128=True,
                is_signed=not has_unsigned,
                is_unsigned=has_unsigned,
            )
        else:
            type_spec = IntCType(
                is_signed=has_signed or not has_unsigned,
                is_unsigned=has_unsigned,
                is_char=has_char,
                is_short=has_short,
                is_int=has_int or (not has_char and not has_short and long_count == 0),
                is_long=long_count == 1,
                is_long_long=long_count >= 2,
            )

        if qualifiers and not isinstance(type_spec, QualifiedCType):
            type_spec = QualifiedCType(base=type_spec, qualifiers=qualifiers)

        return storage, qualifiers, type_spec, func_spec, is_typedef, attrs

    # -------------------------------------------------------------------
    # Struct / union / enum specifiers
    # -------------------------------------------------------------------

    def _parse_struct_or_union_spec(self, is_struct: bool) -> CType:
        """Parse struct/union specifier."""
        self._advance()  # struct/union keyword
        name = ""
        if self._at(TokenKind.KW_ATTRIBUTE):
            self._parse_attributes()
        if self._at(TokenKind.IDENT):
            name = self._advance().text

        if self._match(TokenKind.LBRACE):
            fields = self._parse_struct_fields()
            self._expect(TokenKind.RBRACE)
            # Parse trailing attributes
            if self._at(TokenKind.KW_ATTRIBUTE):
                self._parse_attributes()
            if is_struct:
                return StructRefCType(name=name, is_definition=True)
            else:
                return UnionRefCType(name=name, is_definition=True)

        if is_struct:
            return StructRefCType(name=name)
        return UnionRefCType(name=name)

    def _parse_struct_fields(self) -> list[FieldDecl]:
        """Parse struct/union field declarations."""
        fields: list[FieldDecl] = []
        while not self._at(TokenKind.RBRACE) and not self._at(TokenKind.EOF):
            start = self._cur()
            # _Static_assert in struct
            if self._at(TokenKind.KW_STATIC_ASSERT):
                self._parse_static_assert()
                continue

            # __extension__ in struct fields
            if self._at(TokenKind.KW_EXTENSION):
                self._advance()
                continue

            if self.lenient:
                try:
                    self._parse_struct_field_into(fields, start)
                except ParseError as e:
                    self._errors.append(e)
                    self._skip_to(TokenKind.SEMICOLON, TokenKind.RBRACE)
                    self._match(TokenKind.SEMICOLON)
            else:
                self._parse_struct_field_into(fields, start)

        return fields

    def _parse_struct_field_into(self, fields: list[FieldDecl], start: Token) -> None:
        """Parse a single struct field declaration into the fields list."""
        specs = self._parse_declaration_specifiers()
        if specs is None:
            self._advance()
            return

        _, _, type_spec, _, _, _ = specs

        # Parse field declarators
        if self._at(TokenKind.SEMICOLON):
            # Anonymous field (e.g., anonymous struct/union)
            fields.append(FieldDecl(name="", type_name=type_spec, loc=self._loc(start)))
            self._advance()
            return

        first = True
        while True:
            if not first:
                if not self._match(TokenKind.COMMA):
                    break
            first = False

            # Optional declarator (for bitfields without name)
            name = ""
            field_type = type_spec
            bitfield_width = None

            if self._at(TokenKind.COLON):
                # Unnamed bitfield
                pass
            elif not self._at(TokenKind.SEMICOLON):
                name, field_type = self._parse_declarator(type_spec)

            if self._match(TokenKind.COLON):
                bitfield_width = self._parse_assignment_expr()

            fields.append(FieldDecl(
                name=name,
                type_name=field_type,
                bitfield_width=bitfield_width,
                loc=self._loc(start),
            ))

        self._expect(TokenKind.SEMICOLON)

        return fields

    def _parse_enum_spec(self) -> CType:
        """Parse enum specifier."""
        self._advance()  # enum
        name = ""
        if self._at(TokenKind.IDENT):
            name = self._advance().text

        if self._match(TokenKind.LBRACE):
            while not self._at(TokenKind.RBRACE) and not self._at(TokenKind.EOF):
                ename = self._expect(TokenKind.IDENT).text
                val = None
                if self._match(TokenKind.ASSIGN):
                    val = self._parse_assignment_expr()
                if not self._at(TokenKind.RBRACE):
                    self._expect(TokenKind.COMMA)
            self._expect(TokenKind.RBRACE)

        return EnumRefCType(name=name)

    def _parse_typeof(self) -> CType:
        """Parse typeof(expr) or typeof(type)."""
        self._advance()  # typeof / __typeof__ / __typeof
        self._expect(TokenKind.LPAREN)
        # Check if the content looks like a type
        if self._looks_like_type_in_parens_content():
            ty = self._parse_type_name()
            self._expect(TokenKind.RPAREN)
            return ty  # typeof(type) resolves to the type itself
        expr = self._parse_expression()
        self._expect(TokenKind.RPAREN)
        return TypeofCType(expr=expr)

    def _looks_like_type_in_parens_content(self) -> bool:
        """Check if the current position starts with a type name (already inside parens)."""
        tok = self._cur()
        if tok.is_type_specifier or tok.is_type_qualifier:
            return True
        if tok.kind == TokenKind.IDENT and tok.text in self._typedef_names:
            return True
        return False

    # -------------------------------------------------------------------
    # Declarators
    # -------------------------------------------------------------------

    def _parse_declarator(self, base_type: CType) -> tuple[str, CType]:
        """Parse a declarator, returning (name, full_type)."""
        # Parse pointer qualifiers
        ptr_type = base_type
        while self._match(TokenKind.STAR):
            quals: list[TypeQualifier] = []
            while True:
                if self._at(TokenKind.KW_CONST):
                    quals.append(TypeQualifier.CONST)
                    self._advance()
                elif self._at(TokenKind.KW_VOLATILE):
                    quals.append(TypeQualifier.VOLATILE)
                    self._advance()
                elif self._at_any(TokenKind.KW_RESTRICT, TokenKind.KW_RESTRICT_GNU):
                    quals.append(TypeQualifier.RESTRICT)
                    self._advance()
                elif self._at(TokenKind.KW_ATOMIC):
                    quals.append(TypeQualifier.ATOMIC)
                    self._advance()
                else:
                    break
            ptr_type = PointerCType(pointee=ptr_type, qualifiers=quals)

        return self._parse_direct_declarator(ptr_type)

    def _parse_direct_declarator(self, base_type: CType) -> tuple[str, CType]:
        """Parse the direct part of a declarator."""
        # Handle parenthesized declarator: (*name)
        if self._at(TokenKind.LPAREN) and not self._is_param_list_start():
            self._advance()  # (
            name, inner_type = self._parse_declarator(base_type)
            self._expect(TokenKind.RPAREN)
            # Now parse array/function suffixes that modify the base type
            result_type = self._parse_declarator_suffixes(inner_type)
            return name, result_type

        # Regular name
        name = ""
        if self._at(TokenKind.IDENT):
            name = self._advance().text
        elif self._at(TokenKind.KW_ATTRIBUTE):
            self._parse_attributes()
            if self._at(TokenKind.IDENT):
                name = self._advance().text

        # Parse array/function suffixes
        result_type = self._parse_declarator_suffixes(base_type)
        return name, result_type

    def _is_param_list_start(self) -> bool:
        """Heuristic: check if ( starts a parameter list or a grouped declarator."""
        saved = self._pos
        self._advance()  # skip (
        tok = self._cur()

        result = False
        if tok.kind == TokenKind.RPAREN:
            result = True  # ()
        elif tok.kind == TokenKind.KW_VOID and self._peek(1).kind == TokenKind.RPAREN:
            result = True  # (void)
        elif tok.kind == TokenKind.ELLIPSIS:
            result = True  # (...)
        elif can_start_type(tok, self._typedef_names):
            result = True
        elif tok.kind == TokenKind.IDENT and tok.text in self._typedef_names:
            result = True

        self._pos = saved
        return result

    def _parse_declarator_suffixes(self, base_type: CType) -> CType:
        """Parse array [] and function () suffixes on a declarator."""
        result = base_type

        while True:
            if self._at(TokenKind.LBRACKET):
                # Array declarator
                self._advance()
                size_expr = None
                if not self._at(TokenKind.RBRACKET):
                    size_expr = self._parse_assignment_expr()
                self._expect(TokenKind.RBRACKET)
                result = ArrayCType(element=result, size=size_expr)

            elif self._at(TokenKind.LPAREN):
                # Function declarator
                self._advance()
                params, is_variadic = self._parse_parameter_list()
                self._expect(TokenKind.RPAREN)
                result = FunctionCType(
                    return_type=result,
                    params=params,
                    is_variadic=is_variadic,
                )

            else:
                break

        # Trailing __attribute__
        if self._at(TokenKind.KW_ATTRIBUTE):
            self._parse_attributes()

        return result

    def _parse_parameter_list(self) -> tuple[list[ParamDecl], bool]:
        """Parse a function parameter list."""
        params: list[ParamDecl] = []
        is_variadic = False

        if self._at(TokenKind.RPAREN):
            return params, False

        # (void) means no parameters
        if self._at(TokenKind.KW_VOID) and self._peek(1).kind == TokenKind.RPAREN:
            self._advance()
            return params, False

        while True:
            if self._match(TokenKind.ELLIPSIS):
                is_variadic = True
                break

            start = self._cur()
            specs = self._parse_declaration_specifiers()
            if specs is None:
                break

            _, quals, type_spec, _, _, _ = specs
            name, decl_type = self._parse_declarator(type_spec)

            params.append(ParamDecl(
                name=name,
                type_name=decl_type,
                loc=self._loc(start),
            ))

            if not self._match(TokenKind.COMMA):
                break

        return params, is_variadic

    # -------------------------------------------------------------------
    # Attributes
    # -------------------------------------------------------------------

    def _parse_attributes(self) -> list[Attribute]:
        """Parse __attribute__((name(args), ...))."""
        attrs: list[Attribute] = []
        while self._at(TokenKind.KW_ATTRIBUTE):
            start = self._cur()
            self._advance()  # __attribute__
            if not self._match(TokenKind.LPAREN):
                continue
            if not self._match(TokenKind.LPAREN):
                continue

            # Parse attribute list
            while not self._at(TokenKind.RPAREN) and not self._at(TokenKind.EOF):
                if self._at(TokenKind.IDENT) or self._at_any(*[k for k in TokenKind if k.name.startswith("KW_")]):
                    attr_name = self._advance().text
                    attr_args: list[str] = []
                    if self._match(TokenKind.LPAREN):
                        depth = 1
                        while depth > 0 and not self._at(TokenKind.EOF):
                            if self._at(TokenKind.LPAREN):
                                depth += 1
                            elif self._at(TokenKind.RPAREN):
                                depth -= 1
                                if depth == 0:
                                    self._advance()
                                    break
                            else:
                                attr_args.append(self._cur().text)
                            self._advance()
                    attrs.append(Attribute(name=attr_name, args=attr_args, loc=self._loc(start)))
                elif self._at(TokenKind.COMMA):
                    self._advance()
                else:
                    self._advance()

            self._match(TokenKind.RPAREN)
            self._match(TokenKind.RPAREN)

        return attrs

    # -------------------------------------------------------------------
    # Initializers
    # -------------------------------------------------------------------

    def _parse_initializer(self) -> Expr:
        """Parse an initializer (expression or brace-enclosed list)."""
        if self._at(TokenKind.LBRACE):
            return self._parse_init_list()
        return self._parse_assignment_expr()

    def _parse_init_list(self) -> InitListExpr:
        """Parse { expr, expr, ... } initializer list."""
        start = self._cur()
        self._expect(TokenKind.LBRACE)
        elements: list[Expr] = []
        designators: list[Optional[Designator]] = []

        while not self._at(TokenKind.RBRACE) and not self._at(TokenKind.EOF):
            # Check for designator
            desig = None
            if self._at(TokenKind.DOT) or self._at(TokenKind.LBRACKET):
                desig = self._parse_designator()
                self._expect(TokenKind.ASSIGN)

            elem = self._parse_initializer()
            elements.append(elem)
            designators.append(desig)

            if not self._at(TokenKind.RBRACE):
                self._expect(TokenKind.COMMA)
                # Allow trailing comma
                if self._at(TokenKind.RBRACE):
                    break

        self._expect(TokenKind.RBRACE)
        return InitListExpr(
            elements=elements,
            designators=designators,
            loc=self._loc(start),
        )

    def _parse_designator(self) -> Designator:
        """Parse a designator: .field or [index]."""
        start = self._cur()
        if self._match(TokenKind.DOT):
            name = self._expect(TokenKind.IDENT).text
            return Designator(field_name=name, loc=self._loc(start))
        elif self._match(TokenKind.LBRACKET):
            idx = self._parse_assignment_expr()
            self._expect(TokenKind.RBRACKET)
            return Designator(index=idx, loc=self._loc(start))
        return Designator(loc=self._loc(start))

    # -------------------------------------------------------------------
    # _Static_assert
    # -------------------------------------------------------------------

    def _parse_static_assert(self) -> StaticAssertDecl:
        """Parse _Static_assert(expr, "message");"""
        start = self._cur()
        self._advance()  # _Static_assert
        self._expect(TokenKind.LPAREN)
        cond = self._parse_assignment_expr()
        msg = ""
        if self._match(TokenKind.COMMA):
            tok = self._expect(TokenKind.STRING_LITERAL)
            msg = tok.string_value or tok.text
        self._expect(TokenKind.RPAREN)
        self._expect(TokenKind.SEMICOLON)
        return StaticAssertDecl(condition=cond, message=msg, loc=self._loc(start))

    # -------------------------------------------------------------------
    # Statements
    # -------------------------------------------------------------------

    def _parse_statement(self) -> Stmt:
        """Parse a single statement."""
        tok = self._cur()

        if tok.kind == TokenKind.LBRACE:
            return self._parse_compound_stmt()
        elif tok.kind == TokenKind.KW_IF:
            return self._parse_if_stmt()
        elif tok.kind == TokenKind.KW_WHILE:
            return self._parse_while_stmt()
        elif tok.kind == TokenKind.KW_DO:
            return self._parse_do_while_stmt()
        elif tok.kind == TokenKind.KW_FOR:
            return self._parse_for_stmt()
        elif tok.kind == TokenKind.KW_SWITCH:
            return self._parse_switch_stmt()
        elif tok.kind == TokenKind.KW_CASE:
            return self._parse_case_stmt()
        elif tok.kind == TokenKind.KW_DEFAULT:
            return self._parse_default_stmt()
        elif tok.kind == TokenKind.KW_RETURN:
            return self._parse_return_stmt()
        elif tok.kind == TokenKind.KW_BREAK:
            self._advance()
            self._expect(TokenKind.SEMICOLON)
            return BreakStmt(loc=self._loc(tok))
        elif tok.kind == TokenKind.KW_CONTINUE:
            self._advance()
            self._expect(TokenKind.SEMICOLON)
            return ContinueStmt(loc=self._loc(tok))
        elif tok.kind == TokenKind.KW_GOTO:
            return self._parse_goto_stmt()
        elif tok.kind == TokenKind.SEMICOLON:
            self._advance()
            return NullStmt(loc=self._loc(tok))
        elif tok.kind == TokenKind.KW_ASM:
            return self._parse_asm_stmt()
        elif tok.kind == TokenKind.KW_STATIC_ASSERT:
            sa = self._parse_static_assert()
            return DeclStmt(decl=sa, loc=sa.loc)
        elif tok.kind == TokenKind.KW_EXTENSION:
            self._advance()
            return self._parse_statement()

        # Check for label: ident ':'
        if tok.kind == TokenKind.IDENT and self._peek(1).kind == TokenKind.COLON:
            return self._parse_label_stmt()

        # Check if this looks like a declaration
        if can_start_declaration(tok, self._typedef_names):
            # Could be a declaration or expression statement
            if self._is_declaration():
                if self.lenient:
                    try:
                        return self._parse_declaration_stmt()
                    except ParseError as e:
                        self._errors.append(e)
                        self._skip_to_recovery()
                        self._match(TokenKind.SEMICOLON)
                        return NullStmt(loc=self._loc(tok))
                return self._parse_declaration_stmt()

        # Expression statement
        if self.lenient:
            try:
                return self._parse_expr_stmt()
            except ParseError as e:
                self._errors.append(e)
                self._skip_to_recovery()
                self._match(TokenKind.SEMICOLON)
                return NullStmt(loc=self._loc(tok))
        return self._parse_expr_stmt()

    def _is_declaration(self) -> bool:
        """Heuristic to determine if the current position starts a declaration."""
        tok = self._cur()
        if tok.is_type_specifier or tok.is_type_qualifier or tok.is_storage_class:
            return True
        if tok.kind == TokenKind.KW_INLINE or tok.kind == TokenKind.KW_NORETURN:
            return True
        if tok.kind == TokenKind.KW_ATTRIBUTE:
            return True
        if tok.kind == TokenKind.KW_EXTENSION:
            return True
        if tok.kind == TokenKind.IDENT and tok.text in self._typedef_names:
            # Check that next token looks like a declarator
            next_tok = self._peek(1)
            if next_tok.kind in (TokenKind.IDENT, TokenKind.STAR, TokenKind.LPAREN):
                return True
            if next_tok.kind == TokenKind.SEMICOLON:
                return True
        return False

    def _parse_compound_stmt(self) -> CompoundStmt:
        """Parse { items... }."""
        start = self._cur()
        self._expect(TokenKind.LBRACE)
        items: list[Stmt | Decl] = []

        while not self._at(TokenKind.RBRACE) and not self._at(TokenKind.EOF):
            try:
                stmt = self._parse_statement()
                items.append(stmt)
            except ParseError as e:
                self._errors.append(e)
                self._skip_to(TokenKind.SEMICOLON, TokenKind.RBRACE)
                if self._at(TokenKind.SEMICOLON):
                    self._advance()

        self._expect(TokenKind.RBRACE)
        return CompoundStmt(items=items, loc=self._loc(start))

    def _parse_if_stmt(self) -> IfStmt:
        start = self._cur()
        self._advance()  # if
        self._expect(TokenKind.LPAREN)
        cond = self._parse_expression()
        self._expect(TokenKind.RPAREN)
        then_body = self._parse_statement()
        else_body = None
        if self._match(TokenKind.KW_ELSE):
            else_body = self._parse_statement()
        return IfStmt(condition=cond, then_body=then_body, else_body=else_body,
                      loc=self._loc(start))

    def _parse_while_stmt(self) -> WhileStmt:
        start = self._cur()
        self._advance()  # while
        self._expect(TokenKind.LPAREN)
        cond = self._parse_expression()
        self._expect(TokenKind.RPAREN)
        body = self._parse_statement()
        return WhileStmt(condition=cond, body=body, loc=self._loc(start))

    def _parse_do_while_stmt(self) -> DoWhileStmt:
        start = self._cur()
        self._advance()  # do
        body = self._parse_statement()
        self._expect(TokenKind.KW_WHILE)
        self._expect(TokenKind.LPAREN)
        cond = self._parse_expression()
        self._expect(TokenKind.RPAREN)
        self._expect(TokenKind.SEMICOLON)
        return DoWhileStmt(body=body, condition=cond, loc=self._loc(start))

    def _parse_for_stmt(self) -> ForStmt:
        start = self._cur()
        self._advance()  # for
        self._expect(TokenKind.LPAREN)

        # Init
        init: Optional[Stmt | VarDecl] = None
        if self._at(TokenKind.SEMICOLON):
            self._advance()
        elif self._is_declaration():
            init = self._parse_declaration_stmt()
        else:
            expr = self._parse_expression()
            init = ExprStmt(expr=expr, loc=self._loc(start))
            self._expect(TokenKind.SEMICOLON)

        # Condition
        cond = None
        if not self._at(TokenKind.SEMICOLON):
            cond = self._parse_expression()
        self._expect(TokenKind.SEMICOLON)

        # Increment
        incr = None
        if not self._at(TokenKind.RPAREN):
            incr = self._parse_expression()

        self._expect(TokenKind.RPAREN)
        body = self._parse_statement()
        return ForStmt(init=init, condition=cond, increment=incr, body=body,
                       loc=self._loc(start))

    def _parse_switch_stmt(self) -> SwitchStmt:
        start = self._cur()
        self._advance()  # switch
        self._expect(TokenKind.LPAREN)
        expr = self._parse_expression()
        self._expect(TokenKind.RPAREN)
        body = self._parse_statement()
        return SwitchStmt(expr=expr, body=body, loc=self._loc(start))

    def _parse_case_stmt(self) -> CaseStmt:
        start = self._cur()
        self._advance()  # case
        expr = self._parse_assignment_expr()
        self._expect(TokenKind.COLON)
        body = self._parse_statement() if not self._at_any(
            TokenKind.KW_CASE, TokenKind.KW_DEFAULT, TokenKind.RBRACE
        ) else None
        return CaseStmt(expr=expr, body=body, loc=self._loc(start))

    def _parse_default_stmt(self) -> CaseStmt:
        start = self._cur()
        self._advance()  # default
        self._expect(TokenKind.COLON)
        body = self._parse_statement() if not self._at_any(
            TokenKind.KW_CASE, TokenKind.KW_DEFAULT, TokenKind.RBRACE
        ) else None
        return CaseStmt(expr=None, body=body, is_default=True, loc=self._loc(start))

    def _parse_return_stmt(self) -> ReturnStmt:
        start = self._cur()
        self._advance()  # return
        expr = None
        if not self._at(TokenKind.SEMICOLON):
            expr = self._parse_expression()
        self._expect(TokenKind.SEMICOLON)
        return ReturnStmt(expr=expr, loc=self._loc(start))

    def _parse_goto_stmt(self) -> GotoStmt:
        start = self._cur()
        self._advance()  # goto
        label = self._expect(TokenKind.IDENT).text
        self._expect(TokenKind.SEMICOLON)
        return GotoStmt(label=label, loc=self._loc(start))

    def _parse_label_stmt(self) -> LabelStmt:
        start = self._cur()
        label = self._advance().text
        self._advance()  # :
        body = self._parse_statement()
        return LabelStmt(label=label, body=body, loc=self._loc(start))

    def _parse_expr_stmt(self) -> ExprStmt:
        start = self._cur()
        if self.lenient:
            try:
                expr = self._parse_expression()
            except ParseError as e:
                self._errors.append(e)
                self._skip_to_recovery()
                self._match(TokenKind.SEMICOLON)
                return ExprStmt(
                    expr=IdentExpr(name="<error>", loc=self._loc(start)),
                    loc=self._loc(start),
                )
        else:
            expr = self._parse_expression()
        self._expect(TokenKind.SEMICOLON)
        return ExprStmt(expr=expr, loc=self._loc(start))

    def _parse_declaration_stmt(self) -> DeclStmt:
        """Parse a declaration as a statement."""
        start = self._cur()

        # Handle __extension__
        if self._at(TokenKind.KW_EXTENSION):
            self._advance()

        specs = self._parse_declaration_specifiers()
        if specs is None:
            self._error("expected declaration specifiers")
            self._skip_to(TokenKind.SEMICOLON)
            self._match(TokenKind.SEMICOLON)
            return DeclStmt(loc=self._loc(start))

        storage, qualifiers, type_spec, func_spec, is_typedef, attrs = specs

        # Could be a standalone struct/union/enum definition
        if self._at(TokenKind.SEMICOLON):
            self._advance()
            return DeclStmt(loc=self._loc(start))

        if self.lenient:
            try:
                name, decl_type = self._parse_declarator(type_spec)
            except ParseError as e:
                self._errors.append(e)
                self._skip_to(TokenKind.SEMICOLON)
                self._match(TokenKind.SEMICOLON)
                return DeclStmt(loc=self._loc(start))
        else:
            name, decl_type = self._parse_declarator(type_spec)

        init = None
        if self._match(TokenKind.ASSIGN):
            if self.lenient:
                try:
                    init = self._parse_initializer()
                except ParseError as e:
                    self._errors.append(e)
                    self._skip_to(TokenKind.SEMICOLON)
            else:
                init = self._parse_initializer()

        vd = VarDecl(
            name=name,
            type_name=decl_type,
            initializer=init,
            storage_class=storage,
            qualifiers=qualifiers,
            loc=self._loc(start),
            attributes=attrs,
        )

        if is_typedef:
            td = TypedefDecl(
                name=name,
                underlying_type=decl_type,
                loc=self._loc(start),
                attributes=attrs,
            )
            self._typedef_names.add(name)

        # Handle multiple declarators
        while self._match(TokenKind.COMMA):
            name2, decl_type2 = self._parse_declarator(type_spec)
            init2 = None
            if self._match(TokenKind.ASSIGN):
                init2 = self._parse_initializer()
            # We only return the first for simplicity

        self._expect(TokenKind.SEMICOLON)
        return DeclStmt(decl=vd, loc=self._loc(start))

    def _parse_asm_stmt(self) -> AsmStmt:
        """Parse __asm__ / asm statement."""
        start = self._cur()
        self._advance()  # asm
        is_volatile = bool(self._match(TokenKind.KW_VOLATILE))
        self._expect(TokenKind.LPAREN)
        # Skip the entire asm content
        depth = 1
        template = ""
        while depth > 0 and not self._at(TokenKind.EOF):
            if self._at(TokenKind.LPAREN):
                depth += 1
            elif self._at(TokenKind.RPAREN):
                depth -= 1
                if depth == 0:
                    self._advance()
                    break
            else:
                template += self._cur().text + " "
            self._advance()
        self._match(TokenKind.SEMICOLON)
        return AsmStmt(template=template.strip(), is_volatile=is_volatile,
                       loc=self._loc(start))

    # -------------------------------------------------------------------
    # Expressions with precedence climbing
    # -------------------------------------------------------------------

    def _parse_expression(self) -> Expr:
        """Parse a full expression (including comma expressions)."""
        left = self._parse_assignment_expr()
        if self._at(TokenKind.COMMA):
            exprs = [left]
            while self._match(TokenKind.COMMA):
                exprs.append(self._parse_assignment_expr())
            return CommaExpr(exprs=exprs, loc=left.loc)
        return left

    def _parse_assignment_expr(self) -> Expr:
        """Parse an assignment expression (right-associative)."""
        return self._parse_prec_expr(2)

    def _parse_prec_expr(self, min_prec: int) -> Expr:
        """Precedence climbing expression parser."""
        if self.lenient:
            try:
                left = self._parse_unary_expr()
            except ParseError as e:
                self._errors.append(e)
                return IdentExpr(name="<error>", loc=self._loc(self._cur()))
        else:
            left = self._parse_unary_expr()

        while True:
            tok = self._cur()

            # Handle ternary
            if tok.kind == TokenKind.QUESTION and min_prec <= 3:
                self._advance()
                if self.lenient:
                    try:
                        then_expr = self._parse_expression()
                        self._expect(TokenKind.COLON)
                        else_expr = self._parse_prec_expr(3)
                    except ParseError as e:
                        self._errors.append(e)
                        break
                else:
                    then_expr = self._parse_expression()
                    self._expect(TokenKind.COLON)
                    else_expr = self._parse_prec_expr(3)
                left = TernaryExpr(
                    condition=left,
                    then_expr=then_expr,
                    else_expr=else_expr,
                    loc=left.loc,
                )
                continue

            prec = _BINOP_PRECEDENCE.get(tok.kind)
            if prec is None or prec < min_prec:
                break

            op_kind = tok.kind
            binop = _TOKEN_TO_BINOP.get(op_kind)
            if binop is None:
                break

            self._advance()

            # Right-associative: use same prec; left-associative: use prec+1
            next_prec = prec if op_kind in _RIGHT_ASSOC else prec + 1
            if self.lenient:
                try:
                    right = self._parse_prec_expr(next_prec)
                except ParseError as e:
                    self._errors.append(e)
                    right = IdentExpr(name="<error>", loc=self._loc(self._cur()))
            else:
                right = self._parse_prec_expr(next_prec)

            left = BinaryExpr(op=binop, lhs=left, rhs=right, loc=left.loc)

        return left

    def _parse_unary_expr(self) -> Expr:
        """Parse a unary expression."""
        tok = self._cur()

        # sizeof
        if tok.kind == TokenKind.KW_SIZEOF:
            return self._parse_sizeof_expr()

        # _Alignof
        if tok.kind == TokenKind.KW_ALIGNOF:
            return self._parse_alignof_expr()

        # Prefix operators
        prefix_ops = {
            TokenKind.PLUS: CASTUnaryOp.PLUS,
            TokenKind.MINUS: CASTUnaryOp.MINUS,
            TokenKind.TILDE: CASTUnaryOp.BITWISE_NOT,
            TokenKind.BANG: CASTUnaryOp.LOGICAL_NOT,
            TokenKind.STAR: CASTUnaryOp.DEREF,
            TokenKind.AMP: CASTUnaryOp.ADDR,
            TokenKind.INC: CASTUnaryOp.PRE_INC,
            TokenKind.DEC: CASTUnaryOp.PRE_DEC,
        }

        if tok.kind in prefix_ops:
            self._advance()
            operand = self._parse_unary_expr()
            return UnaryExpr(
                op=prefix_ops[tok.kind],
                operand=operand,
                loc=self._loc(tok),
            )

        # Cast expression: (type)expr
        if tok.kind == TokenKind.LPAREN and self._looks_like_cast():
            return self._parse_cast_expr()

        return self._parse_postfix_expr()

    def _looks_like_cast(self) -> bool:
        """Heuristic: determine if (xxx) is a cast or parenthesized expression."""
        saved = self._pos
        self._advance()  # skip (
        tok = self._cur()

        result = False
        if can_start_type(tok, self._typedef_names) and tok.kind != TokenKind.IDENT:
            result = True
        elif tok.kind == TokenKind.IDENT and tok.text in self._typedef_names:
            # Check it's not something like (x + y)
            self._advance()
            next_tok = self._cur()
            if next_tok.kind == TokenKind.RPAREN:
                # (typedef_name) - could be cast
                # Check what follows the )
                self._advance()
                after = self._cur()
                result = (after.kind not in (
                    TokenKind.PLUS, TokenKind.MINUS, TokenKind.STAR,
                    TokenKind.SLASH, TokenKind.PERCENT, TokenKind.SEMICOLON,
                    TokenKind.RPAREN, TokenKind.COMMA, TokenKind.RBRACKET,
                    TokenKind.QUESTION, TokenKind.COLON,
                    TokenKind.EQ, TokenKind.NE, TokenKind.LT, TokenKind.GT,
                    TokenKind.LE, TokenKind.GE, TokenKind.AND, TokenKind.OR,
                    TokenKind.AMP, TokenKind.PIPE, TokenKind.CARET,
                ) or after.kind in (
                    TokenKind.IDENT, TokenKind.INT_LITERAL, TokenKind.FLOAT_LITERAL,
                    TokenKind.CHAR_LITERAL, TokenKind.STRING_LITERAL, TokenKind.LPAREN,
                    TokenKind.MINUS, TokenKind.TILDE, TokenKind.BANG, TokenKind.AMP,
                    TokenKind.STAR,
                ))
            elif next_tok.kind == TokenKind.STAR:
                result = True  # (typedef_name *)
            else:
                result = False

        self._pos = saved
        return result

    def _parse_cast_expr(self) -> CastExpr:
        """Parse (type)expr."""
        start = self._cur()
        self._advance()  # (
        cast_type = self._parse_type_name()
        self._expect(TokenKind.RPAREN)

        # Check for compound literal: (type){ init_list }
        if self._at(TokenKind.LBRACE):
            init = self._parse_init_list()
            return CompoundLiteralExpr(
                type_name=cast_type,
                init_list=init,
                loc=self._loc(start),
            )

        operand = self._parse_unary_expr()
        return CastExpr(cast_type=cast_type, operand=operand, loc=self._loc(start))

    def _parse_type_name(self) -> CType:
        """Parse a type-name (used in casts, sizeof, etc.)."""
        specs = self._parse_declaration_specifiers()
        if specs is None:
            self._error("expected type name")
            return VoidCType()
        _, _, type_spec, _, _, _ = specs

        # Parse abstract declarator (pointer/array without name)
        while self._at(TokenKind.STAR):
            self._advance()
            quals: list[TypeQualifier] = []
            while True:
                if self._at(TokenKind.KW_CONST):
                    quals.append(TypeQualifier.CONST)
                    self._advance()
                elif self._at(TokenKind.KW_VOLATILE):
                    quals.append(TypeQualifier.VOLATILE)
                    self._advance()
                elif self._at_any(TokenKind.KW_RESTRICT, TokenKind.KW_RESTRICT_GNU):
                    quals.append(TypeQualifier.RESTRICT)
                    self._advance()
                else:
                    break
            type_spec = PointerCType(pointee=type_spec, qualifiers=quals)

        # Array suffix in abstract declarator
        while self._at(TokenKind.LBRACKET):
            self._advance()
            size_expr = None
            if not self._at(TokenKind.RBRACKET):
                size_expr = self._parse_assignment_expr()
            self._expect(TokenKind.RBRACKET)
            type_spec = ArrayCType(element=type_spec, size=size_expr)

        return type_spec

    def _parse_sizeof_expr(self) -> SizeofExpr:
        """Parse sizeof(type) or sizeof expr."""
        start = self._cur()
        self._advance()  # sizeof

        if self._at(TokenKind.LPAREN) and self._looks_like_type_in_parens():
            self._advance()
            type_name = self._parse_type_name()
            self._expect(TokenKind.RPAREN)
            return SizeofExpr(operand_type=type_name, is_type=True, loc=self._loc(start))
        else:
            expr = self._parse_unary_expr()
            return SizeofExpr(operand_expr=expr, is_type=False, loc=self._loc(start))

    def _parse_alignof_expr(self) -> AlignofExpr:
        """Parse _Alignof(type)."""
        start = self._cur()
        self._advance()  # _Alignof
        self._expect(TokenKind.LPAREN)
        type_name = self._parse_type_name()
        self._expect(TokenKind.RPAREN)
        return AlignofExpr(operand_type=type_name, loc=self._loc(start))

    def _looks_like_type_in_parens(self) -> bool:
        """Check if ( is followed by a type name."""
        saved = self._pos
        self._advance()  # (
        tok = self._cur()
        result = can_start_type(tok, self._typedef_names)
        if tok.kind == TokenKind.IDENT:
            result = tok.text in self._typedef_names
        self._pos = saved
        return result

    def _parse_postfix_expr(self) -> Expr:
        """Parse postfix expressions: calls, subscripts, member access, ++/--."""
        expr = self._parse_primary_expr()

        while True:
            tok = self._cur()

            if tok.kind == TokenKind.LPAREN:
                # Function call
                self._advance()
                args: list[Expr] = []
                if not self._at(TokenKind.RPAREN):
                    args.append(self._parse_assignment_expr())
                    while self._match(TokenKind.COMMA):
                        args.append(self._parse_assignment_expr())
                self._expect(TokenKind.RPAREN)
                expr = CallExpr(callee=expr, args=args, loc=expr.loc)

            elif tok.kind == TokenKind.LBRACKET:
                # Array subscript
                self._advance()
                index = self._parse_expression()
                self._expect(TokenKind.RBRACKET)
                expr = ArraySubscriptExpr(base=expr, index=index, loc=expr.loc)

            elif tok.kind == TokenKind.DOT:
                # Member access
                self._advance()
                member = self._expect(TokenKind.IDENT).text
                expr = MemberExpr(base=expr, member=member, is_arrow=False, loc=expr.loc)

            elif tok.kind == TokenKind.ARROW:
                # Arrow member access
                self._advance()
                member = self._expect(TokenKind.IDENT).text
                expr = MemberExpr(base=expr, member=member, is_arrow=True, loc=expr.loc)

            elif tok.kind == TokenKind.INC:
                # Postfix increment
                self._advance()
                expr = UnaryExpr(op=CASTUnaryOp.POST_INC, operand=expr, loc=expr.loc)

            elif tok.kind == TokenKind.DEC:
                # Postfix decrement
                self._advance()
                expr = UnaryExpr(op=CASTUnaryOp.POST_DEC, operand=expr, loc=expr.loc)

            else:
                break

        return expr

    def _parse_primary_expr(self) -> Expr:
        """Parse primary expressions: literals, identifiers, parenthesized exprs."""
        tok = self._cur()

        if tok.kind == TokenKind.INT_LITERAL:
            self._advance()
            return IntLiteral(
                value=tok.int_value or 0,
                suffix_unsigned=tok.int_unsigned,
                suffix_long=tok.int_long,
                text=tok.text,
                loc=self._loc(tok),
            )

        if tok.kind == TokenKind.FLOAT_LITERAL:
            self._advance()
            return FloatLiteral(
                value=tok.float_value or 0.0,
                suffix=tok.float_suffix,
                text=tok.text,
                loc=self._loc(tok),
            )

        if tok.kind == TokenKind.CHAR_LITERAL:
            self._advance()
            return CharLiteral(
                value=tok.char_value or 0,
                text=tok.text,
                loc=self._loc(tok),
            )

        if tok.kind == TokenKind.STRING_LITERAL:
            self._advance()
            # Handle adjacent string literal concatenation
            value = tok.string_value or ""
            text = tok.text
            while self._at(TokenKind.STRING_LITERAL):
                next_tok = self._advance()
                value += next_tok.string_value or ""
                text += " " + next_tok.text
            return StringLiteral(
                value=value,
                text=text,
                loc=self._loc(tok),
            )

        if tok.kind == TokenKind.IDENT:
            # Handle __builtin_ calls before generic identifiers
            if tok.text.startswith("__builtin_"):
                return self._parse_builtin_call()
            self._advance()
            return IdentExpr(name=tok.text, loc=self._loc(tok))

        if tok.kind == TokenKind.LPAREN:
            self._advance()
            # GCC statement expression ({...})
            if self._at(TokenKind.LBRACE):
                body = self._parse_compound_stmt()
                self._expect(TokenKind.RPAREN)
                return StmtExpr(body=body, loc=self._loc(tok))
            expr = self._parse_expression()
            expr.parenthesized = True
            self._expect(TokenKind.RPAREN)
            return ParenExpr(inner=expr, loc=self._loc(tok))

        if tok.kind == TokenKind.KW_GENERIC:
            return self._parse_generic_expr()

        # __extension__ in expression context: skip and parse sub-expression
        if tok.kind == TokenKind.KW_EXTENSION:
            self._advance()
            return self._parse_unary_expr()

        # __builtin_va_list as identifier in expression context
        if tok.kind == TokenKind.KW_BUILTIN_VA_LIST:
            self._advance()
            return IdentExpr(name="__builtin_va_list", loc=self._loc(tok))

        # _Alignof in expression context
        if tok.kind == TokenKind.KW_ALIGNOF:
            return self._parse_alignof_expr()

        # __attribute__ in expression context — skip and parse next expression
        if tok.kind == TokenKind.KW_ATTRIBUTE:
            self._parse_attributes()
            return self._parse_unary_expr()

        # typeof in expression context — treat as a type reference  
        if tok.kind == TokenKind.KW_TYPEOF:
            ty = self._parse_typeof()
            return IdentExpr(name="typeof", loc=self._loc(tok))

        self._error(f"expected expression, got {tok.kind.name} ({tok.text!r})", tok)
        if self.lenient:
            self._advance()
            return IdentExpr(name="<error>", loc=self._loc(tok))
        self._advance()
        return IdentExpr(name="<error>", loc=self._loc(tok))

    def _parse_generic_expr(self) -> GenericExpr:
        """Parse _Generic(expr, type: expr, ...)."""
        start = self._cur()
        self._advance()  # _Generic
        self._expect(TokenKind.LPAREN)
        ctrl = self._parse_assignment_expr()
        assocs: list[tuple[Optional[CType], Expr]] = []
        while self._match(TokenKind.COMMA):
            if self._at(TokenKind.KW_DEFAULT):
                self._advance()
                self._expect(TokenKind.COLON)
                expr = self._parse_assignment_expr()
                assocs.append((None, expr))
            else:
                ty = self._parse_type_name()
                self._expect(TokenKind.COLON)
                expr = self._parse_assignment_expr()
                assocs.append((ty, expr))
        self._expect(TokenKind.RPAREN)
        return GenericExpr(controlling_expr=ctrl, associations=assocs, loc=self._loc(start))

    def _parse_builtin_call(self) -> Expr:
        """Parse __builtin_xxx(args) with special handling for type-argument builtins."""
        start = self._cur()
        name = self._advance().text

        # __builtin_offsetof(type, member)
        if name == "__builtin_offsetof":
            self._expect(TokenKind.LPAREN)
            ty = self._parse_type_name()
            self._expect(TokenKind.COMMA)
            member = self._expect(TokenKind.IDENT).text
            # Handle nested member access like a.b.c
            while self._match(TokenKind.DOT):
                member += "." + self._expect(TokenKind.IDENT).text
            self._expect(TokenKind.RPAREN)
            return OffsetofExpr(type_name=ty, member_name=member, loc=self._loc(start))

        # __builtin_types_compatible_p(type1, type2)
        if name == "__builtin_types_compatible_p":
            self._expect(TokenKind.LPAREN)
            ty1 = self._parse_type_name()
            self._expect(TokenKind.COMMA)
            ty2 = self._parse_type_name()
            self._expect(TokenKind.RPAREN)
            return TypesCompatibleExpr(type1=ty1, type2=ty2, loc=self._loc(start))

        # __builtin_va_arg is handled via VaArgExpr
        # For __builtin_expect and all other builtins, parse as normal call
        args: list[Expr] = []
        if self._match(TokenKind.LPAREN):
            if not self._at(TokenKind.RPAREN):
                args.append(self._parse_assignment_expr())
                while self._match(TokenKind.COMMA):
                    args.append(self._parse_assignment_expr())
            self._expect(TokenKind.RPAREN)
        return BuiltinCallExpr(builtin_name=name, args=args, loc=self._loc(start))
