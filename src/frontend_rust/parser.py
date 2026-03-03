"""
Recursive descent Rust parser for the Cross-Language Equivalence Verifier.

Parses Rust source code (targeting C2Rust output patterns) into a Rust AST.
Handles items, expressions, statements, patterns, and types. Supports
unsafe blocks, raw pointers, extern "C" functions, explicit casts,
and tuple structs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .lexer import (
    RustLexer, Token, TokenKind, SourcePos, SourceSpan, KEYWORDS,
)
from .rust_ast import (
    # Types
    RustType, NeverType, UnitType, PathType, ReferenceType,
    RawPointerType, ArrayType as RustArrayType, SliceType,
    TupleType, FnPointerType, InferredType, ParenType,
    # Generics
    GenericParam, Generics, WhereClause,
    # Patterns
    Pattern, IdentPattern, WildcardPattern, LiteralPattern,
    TuplePattern, StructPattern, TupleStructPattern, RefPattern,
    RangePattern, OrPattern, PathPattern, SlicePattern,
    # Items
    Item, FnItem, FnParam, StructItem, StructField as RustStructField,
    EnumItem, EnumVariant, ImplItem, UseItem, ConstItem, StaticItem,
    TypeAliasItem, TraitItem, ExternBlock, ModItem, MacroDefItem,
    UnionItem, ExternFnItem,
    Attribute, Visibility, Mutability,
    # Expressions
    Expr, LitExpr, PathExpr, BinaryExpr, UnaryExpr, CastExpr,
    AssignExpr, CallExpr, MethodCallExpr, FieldExpr, TupleFieldExpr,
    IndexExpr, RangeExpr, BlockExpr, IfExpr, MatchExpr, MatchArm,
    LoopExpr, WhileExpr, ForExpr, ReturnExpr, BreakExpr, ContinueExpr,
    ClosureExpr, TupleExpr, ArrayExpr, StructExpr, RefExpr, DerefExpr,
    UnsafeBlock, MacroInvocation, ParenExpr, TryExpr, AwaitExpr,
    AsyncBlock, IfLetExpr, WhileLetExpr, TransmuteCall, InlineAsm,
    BinaryOp as RustBinaryOp, UnaryOp as RustUnaryOp,
    # Statements
    Stmt, LetStmt, ExprStmt, ItemStmt, EmptyStmt, MacroStmt,
    LetElseStmt,
    # Top level
    Crate, NodeLocation,
)


# ---------------------------------------------------------------------------
# Parse error
# ---------------------------------------------------------------------------

class ParseError(Exception):
    """Error raised during Rust parsing."""
    def __init__(self, message: str, pos: Optional[SourcePos] = None,
                 token: Optional[Token] = None):
        self.pos = pos or (token.start_pos if token else None)
        self.token = token
        loc = str(self.pos) if self.pos else "<unknown>"
        super().__init__(f"{loc}: {message}")


# ---------------------------------------------------------------------------
# Operator precedence
# ---------------------------------------------------------------------------

_BINOP_PRECEDENCE: dict[TokenKind, int] = {
    TokenKind.OR: 3,
    TokenKind.AND: 4,
    TokenKind.EQ: 5,
    TokenKind.NE: 5,
    TokenKind.LT: 6,
    TokenKind.GT: 6,
    TokenKind.LE: 6,
    TokenKind.GE: 6,
    TokenKind.PIPE: 7,
    TokenKind.CARET: 8,
    TokenKind.AMP: 9,
    TokenKind.SHL: 10,
    TokenKind.SHR: 10,
    TokenKind.PLUS: 11,
    TokenKind.MINUS: 11,
    TokenKind.STAR: 12,
    TokenKind.SLASH: 12,
    TokenKind.PERCENT: 12,
}

_TOKEN_TO_BINOP: dict[TokenKind, RustBinaryOp] = {
    TokenKind.PLUS: RustBinaryOp.ADD,
    TokenKind.MINUS: RustBinaryOp.SUB,
    TokenKind.STAR: RustBinaryOp.MUL,
    TokenKind.SLASH: RustBinaryOp.DIV,
    TokenKind.PERCENT: RustBinaryOp.REM,
    TokenKind.AMP: RustBinaryOp.BITAND,
    TokenKind.PIPE: RustBinaryOp.BITOR,
    TokenKind.CARET: RustBinaryOp.BITXOR,
    TokenKind.SHL: RustBinaryOp.SHL,
    TokenKind.SHR: RustBinaryOp.SHR,
    TokenKind.AND: RustBinaryOp.AND,
    TokenKind.OR: RustBinaryOp.OR,
    TokenKind.EQ: RustBinaryOp.EQ,
    TokenKind.NE: RustBinaryOp.NE,
    TokenKind.LT: RustBinaryOp.LT,
    TokenKind.GT: RustBinaryOp.GT,
    TokenKind.LE: RustBinaryOp.LE,
    TokenKind.GE: RustBinaryOp.GE,
}

_ASSIGN_OPS: dict[TokenKind, Optional[RustBinaryOp]] = {
    TokenKind.ASSIGN: None,
    TokenKind.PLUS_ASSIGN: RustBinaryOp.ADD,
    TokenKind.MINUS_ASSIGN: RustBinaryOp.SUB,
    TokenKind.STAR_ASSIGN: RustBinaryOp.MUL,
    TokenKind.SLASH_ASSIGN: RustBinaryOp.DIV,
    TokenKind.PERCENT_ASSIGN: RustBinaryOp.REM,
    TokenKind.AMP_ASSIGN: RustBinaryOp.BITAND,
    TokenKind.PIPE_ASSIGN: RustBinaryOp.BITOR,
    TokenKind.CARET_ASSIGN: RustBinaryOp.BITXOR,
    TokenKind.SHL_ASSIGN: RustBinaryOp.SHL,
    TokenKind.SHR_ASSIGN: RustBinaryOp.SHR,
}


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class RustParser:
    """Recursive descent parser for Rust (C2Rust output subset).

    Usage::

        parser = RustParser(source, filename="test.rs")
        crate = parser.parse()
    """

    def __init__(self, source: str, filename: str = "<input>",
                 lenient: bool = False) -> None:
        self._lexer = RustLexer(source, filename)
        self._filename = filename
        self._tokens: list[Token] = []
        self._pos = 0
        self._errors: list[ParseError] = []
        self._no_struct_literal = False  # Disables struct literal parsing in if/while/match conditions
        self.lenient = lenient

    @property
    def errors(self) -> list[ParseError]:
        return list(self._errors)

    def _tokenize(self) -> None:
        if not self._tokens:
            self._tokens = self._lexer.tokenize()
            self._pos = 0

    def _cur(self) -> Token:
        if self._pos >= len(self._tokens):
            return self._tokens[-1]
        return self._tokens[self._pos]

    def _peek(self, offset: int = 0) -> Token:
        idx = self._pos + offset
        if idx >= len(self._tokens):
            return self._tokens[-1]
        return self._tokens[idx]

    def _advance(self) -> Token:
        tok = self._cur()
        if tok.kind != TokenKind.EOF:
            self._pos += 1
        return tok

    def _expect(self, kind: TokenKind, msg: str = "") -> Token:
        tok = self._cur()
        if tok.kind != kind:
            if not msg:
                msg = f"expected {kind.name}, got {tok.kind.name} ({tok.text!r})"
            if self.lenient:
                self._error(msg, tok)
                # Return a synthetic token so callers don't crash
                return tok
            self._error(msg, tok)
            return tok
        return self._advance()

    def _match(self, kind: TokenKind) -> Optional[Token]:
        if self._cur().kind == kind:
            return self._advance()
        return None

    def _at(self, kind: TokenKind) -> bool:
        return self._cur().kind == kind

    def _at_any(self, *kinds: TokenKind) -> bool:
        return self._cur().kind in kinds

    def _error(self, msg: str, token: Optional[Token] = None) -> ParseError:
        tok = token or self._cur()
        err = ParseError(msg, token=tok)
        self._errors.append(err)
        return err

    def _recover_to_next_boundary(self) -> None:
        """Skip tokens until we reach a statement/item boundary."""
        self._skip_to(TokenKind.SEMICOLON, TokenKind.RBRACE, TokenKind.LBRACE)
        if self._at(TokenKind.SEMICOLON):
            self._advance()

    def _loc(self, start_token: Token) -> NodeLocation:
        end = self._cur()
        span = SourceSpan(start_token.span.start, end.span.start)
        return NodeLocation(span=span)

    def _skip_to(self, *kinds: TokenKind) -> None:
        while not self._at(TokenKind.EOF) and not self._at_any(*kinds):
            self._advance()

    def _split_shr(self) -> None:
        """Split a >> (SHR) token into two > (GT) tokens.

        Used when parsing nested generics like Option<Box<i32>>.
        """
        if not self._at(TokenKind.SHR):
            return
        shr_tok = self._tokens[self._pos]
        # Replace >> with > and insert another > after it
        gt1 = Token(kind=TokenKind.GT, text=">", span=shr_tok.span)
        gt2 = Token(kind=TokenKind.GT, text=">", span=shr_tok.span)
        self._tokens[self._pos] = gt1
        self._tokens.insert(self._pos + 1, gt2)
        self._advance()  # consume first >

    # -------------------------------------------------------------------
    # Top-level parsing
    # -------------------------------------------------------------------

    def parse(self) -> Crate:
        """Parse a complete Rust crate."""
        self._tokenize()
        crate = Crate()
        start = self._cur()

        # Inner attributes
        while self._at(TokenKind.HASH_BANG):
            attr = self._parse_attribute(inner=True)
            crate.inner_attributes.append(attr)

        while not self._at(TokenKind.EOF):
            saved_pos = self._pos
            try:
                item = self._parse_item()
                if item is not None:
                    crate.items.append(item)
            except ParseError as e:
                self._errors.append(e)
                self._skip_to(TokenKind.RBRACE, TokenKind.SEMICOLON)
                if self._at(TokenKind.SEMICOLON):
                    self._advance()
                elif self._at(TokenKind.RBRACE):
                    self._advance()
            # Guard against infinite loops
            if self._pos == saved_pos:
                self._advance()

        crate.loc = self._loc(start)
        return crate

    # -------------------------------------------------------------------
    # Items
    # -------------------------------------------------------------------

    def _parse_item(self) -> Optional[Item]:
        """Parse a single item."""
        # Outer attributes
        attrs = self._parse_outer_attributes()
        vis = self._parse_visibility()
        start = self._cur()

        tok = self._cur()

        if tok.kind == TokenKind.KW_FN:
            return self._parse_fn_item(attrs, vis, start)
        if tok.kind == TokenKind.KW_UNSAFE and self._peek(1).kind == TokenKind.KW_FN:
            return self._parse_fn_item(attrs, vis, start, is_unsafe=True)
        if tok.kind == TokenKind.KW_CONST and self._peek(1).kind == TokenKind.KW_FN:
            return self._parse_fn_item(attrs, vis, start, is_const=True)
        if tok.kind == TokenKind.KW_ASYNC and self._peek(1).kind == TokenKind.KW_FN:
            return self._parse_fn_item(attrs, vis, start, is_async=True)
        if tok.kind == TokenKind.KW_UNSAFE and self._peek(1).kind == TokenKind.KW_EXTERN:
            self._advance()  # unsafe
            return self._parse_extern_block(attrs, vis, start)
        if tok.kind == TokenKind.KW_EXTERN:
            if self._peek(1).kind == TokenKind.STRING_LITERAL or self._peek(1).kind == TokenKind.KW_FN:
                return self._parse_extern_item(attrs, vis, start)
            return self._parse_extern_block(attrs, vis, start)
        if tok.kind == TokenKind.KW_STRUCT:
            return self._parse_struct_item(attrs, vis, start)
        if tok.kind == TokenKind.IDENT and tok.text == "union":
            return self._parse_union_item(attrs, vis, start)
        if tok.kind == TokenKind.KW_ENUM:
            return self._parse_enum_item(attrs, vis, start)
        if tok.kind == TokenKind.KW_IMPL:
            return self._parse_impl_item(attrs, vis, start)
        if tok.kind == TokenKind.KW_UNSAFE and self._peek(1).kind == TokenKind.KW_IMPL:
            self._advance()  # unsafe
            return self._parse_impl_item(attrs, vis, start, is_unsafe=True)
        if tok.kind == TokenKind.KW_USE:
            return self._parse_use_item(attrs, vis, start)
        if tok.kind == TokenKind.KW_CONST:
            return self._parse_const_item(attrs, vis, start)
        if tok.kind == TokenKind.KW_STATIC:
            return self._parse_static_item(attrs, vis, start)
        if tok.kind == TokenKind.KW_TYPE:
            return self._parse_type_alias_item(attrs, vis, start)
        if tok.kind == TokenKind.KW_TRAIT:
            return self._parse_trait_item(attrs, vis, start)
        if tok.kind == TokenKind.KW_UNSAFE and self._peek(1).kind == TokenKind.KW_TRAIT:
            self._advance()
            return self._parse_trait_item(attrs, vis, start, is_unsafe=True)
        if tok.kind == TokenKind.KW_MOD:
            return self._parse_mod_item(attrs, vis, start)

        # Macro invocation at item level
        if tok.kind == TokenKind.IDENT and self._peek(1).kind == TokenKind.BANG:
            return self._parse_macro_item(attrs, vis, start)

        self._error(f"expected item, got {tok.kind.name} ({tok.text!r})", tok)
        self._advance()
        return None

    def _parse_fn_item(
        self, attrs: list[Attribute], vis: Visibility,
        start: Token, is_unsafe: bool = False, is_const: bool = False,
        is_async: bool = False, abi: str = "", is_extern: bool = False,
    ) -> FnItem:
        """Parse a function definition."""
        if is_unsafe:
            self._advance()  # unsafe
        if is_const:
            self._advance()  # const
        if is_async:
            self._advance()  # async
        self._expect(TokenKind.KW_FN)
        name = self._expect(TokenKind.IDENT).text

        generics = self._parse_generics()

        self._expect(TokenKind.LPAREN)
        params = self._parse_fn_params()
        self._expect(TokenKind.RPAREN)

        ret_type = None
        if self._match(TokenKind.ARROW):
            ret_type = self._parse_type()

        # Where clause
        if self._at(TokenKind.KW_WHERE):
            self._parse_where_clause(generics)

        body = None
        if self._at(TokenKind.LBRACE):
            body = self._parse_block_expr()
        else:
            self._expect(TokenKind.SEMICOLON)

        return FnItem(
            name=name, params=params, return_type=ret_type, body=body,
            generics=generics, is_unsafe=is_unsafe, is_async=is_async,
            is_const=is_const, abi=abi, is_extern=is_extern,
            loc=self._loc(start), attributes=attrs, visibility=vis,
        )

    def _parse_fn_params(self) -> list[FnParam]:
        """Parse function parameters."""
        params: list[FnParam] = []

        while not self._at(TokenKind.RPAREN) and not self._at(TokenKind.EOF):
            start = self._cur()

            # Self parameter
            if self._at_any(TokenKind.KW_SELF, TokenKind.AMP) and self._looks_like_self_param():
                param = self._parse_self_param(start)
                params.append(param)
            else:
                pat = self._parse_pattern()
                self._expect(TokenKind.COLON)
                ty = self._parse_type()
                params.append(FnParam(
                    pattern=pat, type_ann=ty, loc=self._loc(start),
                ))

            if not self._match(TokenKind.COMMA):
                break

        return params

    def _looks_like_self_param(self) -> bool:
        """Check if current position has a self parameter."""
        if self._at(TokenKind.KW_SELF):
            return True
        if self._at(TokenKind.AMP):
            next_tok = self._peek(1)
            if next_tok.kind == TokenKind.KW_SELF:
                return True
            if next_tok.kind == TokenKind.KW_MUT and self._peek(2).kind == TokenKind.KW_SELF:
                return True
            if next_tok.kind == TokenKind.LIFETIME:
                next2 = self._peek(2)
                if next2.kind == TokenKind.KW_SELF:
                    return True
                if next2.kind == TokenKind.KW_MUT and self._peek(3).kind == TokenKind.KW_SELF:
                    return True
        return False

    def _parse_self_param(self, start: Token) -> FnParam:
        """Parse a self parameter."""
        mut = Mutability.IMMUTABLE
        if self._match(TokenKind.AMP):
            # &self or &mut self or &'a self
            if self._at(TokenKind.LIFETIME):
                self._advance()
            if self._match(TokenKind.KW_MUT):
                mut = Mutability.MUTABLE
            self._expect(TokenKind.KW_SELF)
        else:
            if self._match(TokenKind.KW_MUT):
                mut = Mutability.MUTABLE
            self._expect(TokenKind.KW_SELF)
        return FnParam(is_self=True, self_mutability=mut, loc=self._loc(start))

    def _parse_struct_item(
        self, attrs: list[Attribute], vis: Visibility, start: Token,
    ) -> StructItem:
        """Parse a struct definition."""
        self._advance()  # struct
        name = self._expect(TokenKind.IDENT).text
        generics = self._parse_generics()

        if self._at(TokenKind.KW_WHERE):
            self._parse_where_clause(generics)

        fields: list[RustStructField] = []
        is_tuple = False

        if self._at(TokenKind.LBRACE):
            # Named fields
            self._advance()
            while not self._at(TokenKind.RBRACE) and not self._at(TokenKind.EOF):
                f_attrs = self._parse_outer_attributes()
                f_vis = self._parse_visibility()
                f_name = self._expect(TokenKind.IDENT).text
                self._expect(TokenKind.COLON)
                f_type = self._parse_type()
                fields.append(RustStructField(
                    name=f_name, type_ann=f_type,
                    attributes=f_attrs, visibility=f_vis,
                ))
                if not self._match(TokenKind.COMMA):
                    break
            self._expect(TokenKind.RBRACE)
        elif self._at(TokenKind.LPAREN):
            # Tuple struct
            is_tuple = True
            self._advance()
            idx = 0
            while not self._at(TokenKind.RPAREN) and not self._at(TokenKind.EOF):
                f_attrs = self._parse_outer_attributes()
                f_vis = self._parse_visibility()
                f_type = self._parse_type()
                fields.append(RustStructField(
                    name=str(idx), type_ann=f_type,
                    attributes=f_attrs, visibility=f_vis,
                ))
                idx += 1
                if not self._match(TokenKind.COMMA):
                    break
            self._expect(TokenKind.RPAREN)
            self._expect(TokenKind.SEMICOLON)
        else:
            # Unit struct
            self._expect(TokenKind.SEMICOLON)

        return StructItem(
            name=name, fields=fields, generics=generics, is_tuple_struct=is_tuple,
            loc=self._loc(start), attributes=attrs, visibility=vis,
        )

    def _parse_union_item(
        self, attrs: list[Attribute], vis: Visibility, start: Token,
    ) -> UnionItem:
        """Parse a union definition."""
        self._advance()  # union (contextual keyword)
        name = self._expect(TokenKind.IDENT).text
        generics = self._parse_generics()

        if self._at(TokenKind.KW_WHERE):
            self._parse_where_clause(generics)

        fields: list[RustStructField] = []
        self._expect(TokenKind.LBRACE)
        while not self._at(TokenKind.RBRACE) and not self._at(TokenKind.EOF):
            f_attrs = self._parse_outer_attributes()
            f_vis = self._parse_visibility()
            f_name = self._expect(TokenKind.IDENT).text
            self._expect(TokenKind.COLON)
            f_type = self._parse_type()
            fields.append(RustStructField(
                name=f_name, type_ann=f_type,
                attributes=f_attrs, visibility=f_vis,
            ))
            if not self._match(TokenKind.COMMA):
                break
        self._expect(TokenKind.RBRACE)
        return UnionItem(
            name=name, fields=fields, generics=generics,
            loc=self._loc(start), attributes=attrs, visibility=vis,
        )

    def _parse_enum_item(
        self, attrs: list[Attribute], vis: Visibility, start: Token,
    ) -> EnumItem:
        """Parse an enum definition."""
        self._advance()  # enum
        name = self._expect(TokenKind.IDENT).text
        generics = self._parse_generics()

        # Check repr attribute
        repr_attr = ""
        for attr in attrs:
            if attr.name == "repr":
                repr_attr = attr.args

        if self._at(TokenKind.KW_WHERE):
            self._parse_where_clause(generics)

        self._expect(TokenKind.LBRACE)
        variants: list[EnumVariant] = []

        while not self._at(TokenKind.RBRACE) and not self._at(TokenKind.EOF):
            v_attrs = self._parse_outer_attributes()
            v_name = self._expect(TokenKind.IDENT).text
            v_fields: list[RustStructField] = []
            discriminant = None
            is_tuple = False
            is_unit = True

            if self._at(TokenKind.LBRACE):
                is_unit = False
                self._advance()
                while not self._at(TokenKind.RBRACE) and not self._at(TokenKind.EOF):
                    f_name = self._expect(TokenKind.IDENT).text
                    self._expect(TokenKind.COLON)
                    f_type = self._parse_type()
                    v_fields.append(RustStructField(name=f_name, type_ann=f_type))
                    if not self._match(TokenKind.COMMA):
                        break
                self._expect(TokenKind.RBRACE)
            elif self._at(TokenKind.LPAREN):
                is_unit = False
                is_tuple = True
                self._advance()
                idx = 0
                while not self._at(TokenKind.RPAREN) and not self._at(TokenKind.EOF):
                    f_type = self._parse_type()
                    v_fields.append(RustStructField(name=str(idx), type_ann=f_type))
                    idx += 1
                    if not self._match(TokenKind.COMMA):
                        break
                self._expect(TokenKind.RPAREN)

            if self._match(TokenKind.ASSIGN):
                discriminant = self._parse_expression()

            variants.append(EnumVariant(
                name=v_name, fields=v_fields, discriminant=discriminant,
                is_tuple=is_tuple, is_unit=is_unit, attributes=v_attrs,
            ))

            if not self._match(TokenKind.COMMA):
                break

        self._expect(TokenKind.RBRACE)
        return EnumItem(
            name=name, variants=variants, generics=generics, repr=repr_attr,
            loc=self._loc(start), attributes=attrs, visibility=vis,
        )

    def _parse_impl_item(
        self, attrs: list[Attribute], vis: Visibility,
        start: Token, is_unsafe: bool = False,
    ) -> ImplItem:
        """Parse an impl block."""
        self._advance()  # impl
        generics = self._parse_generics()

        # Could be `impl Trait for Type` or `impl Type`
        first_type = self._parse_type()
        trait_type = None

        if self._match(TokenKind.KW_FOR):
            trait_type = first_type
            first_type = self._parse_type()

        if self._at(TokenKind.KW_WHERE):
            self._parse_where_clause(generics)

        self._expect(TokenKind.LBRACE)
        items: list[Item] = []

        while not self._at(TokenKind.RBRACE) and not self._at(TokenKind.EOF):
            try:
                item = self._parse_item()
                if item:
                    items.append(item)
            except ParseError:
                self._skip_to(TokenKind.RBRACE, TokenKind.KW_FN)

        self._expect(TokenKind.RBRACE)

        return ImplItem(
            self_type=first_type, trait_type=trait_type, items=items,
            generics=generics, is_unsafe=is_unsafe,
            loc=self._loc(start), attributes=attrs, visibility=vis,
        )

    def _parse_use_item(
        self, attrs: list[Attribute], vis: Visibility, start: Token,
    ) -> UseItem:
        """Parse a use declaration."""
        self._advance()  # use
        path = self._parse_use_tree()
        self._expect(TokenKind.SEMICOLON)
        return UseItem(
            path=path.path, alias=path.alias, is_glob=path.is_glob,
            group=path.group,
            loc=self._loc(start), attributes=attrs, visibility=vis,
        )

    def _parse_use_tree(self) -> UseItem:
        """Parse a use tree."""
        path: list[str] = []

        while self._at(TokenKind.IDENT) or self._at_any(
            TokenKind.KW_SELF, TokenKind.KW_SUPER, TokenKind.KW_CRATE
        ):
            path.append(self._advance().text)
            if not self._match(TokenKind.PATH_SEP):
                break

        alias = ""
        if self._match(TokenKind.KW_AS):
            alias = self._expect(TokenKind.IDENT).text

        is_glob = False
        if self._match(TokenKind.STAR):
            is_glob = True

        group: list[UseItem] = []
        if self._at(TokenKind.LBRACE):
            self._advance()
            while not self._at(TokenKind.RBRACE) and not self._at(TokenKind.EOF):
                group.append(self._parse_use_tree())
                if not self._match(TokenKind.COMMA):
                    break
            self._expect(TokenKind.RBRACE)

        return UseItem(path=path, alias=alias, is_glob=is_glob, group=group)

    def _parse_const_item(
        self, attrs: list[Attribute], vis: Visibility, start: Token,
    ) -> ConstItem:
        self._advance()  # const
        name = self._expect(TokenKind.IDENT).text
        self._expect(TokenKind.COLON)
        ty = self._parse_type()
        value = None
        if self._match(TokenKind.ASSIGN):
            value = self._parse_expression()
        self._expect(TokenKind.SEMICOLON)
        return ConstItem(
            name=name, type_ann=ty, value=value,
            loc=self._loc(start), attributes=attrs, visibility=vis,
        )

    def _parse_static_item(
        self, attrs: list[Attribute], vis: Visibility, start: Token,
    ) -> StaticItem:
        self._advance()  # static
        mut = Mutability.IMMUTABLE
        if self._match(TokenKind.KW_MUT):
            mut = Mutability.MUTABLE
        name = self._expect(TokenKind.IDENT).text
        self._expect(TokenKind.COLON)
        ty = self._parse_type()
        value = None
        if self._match(TokenKind.ASSIGN):
            value = self._parse_expression()
        self._expect(TokenKind.SEMICOLON)
        return StaticItem(
            name=name, type_ann=ty, value=value, mutability=mut,
            loc=self._loc(start), attributes=attrs, visibility=vis,
        )

    def _parse_type_alias_item(
        self, attrs: list[Attribute], vis: Visibility, start: Token,
    ) -> TypeAliasItem:
        self._advance()  # type
        name = self._expect(TokenKind.IDENT).text
        generics = self._parse_generics()
        self._expect(TokenKind.ASSIGN)
        ty = self._parse_type()
        self._expect(TokenKind.SEMICOLON)
        return TypeAliasItem(
            name=name, aliased_type=ty, generics=generics,
            loc=self._loc(start), attributes=attrs, visibility=vis,
        )

    def _parse_trait_item(
        self, attrs: list[Attribute], vis: Visibility,
        start: Token, is_unsafe: bool = False,
    ) -> TraitItem:
        self._advance()  # trait
        name = self._expect(TokenKind.IDENT).text
        generics = self._parse_generics()

        supertraits: list[RustType] = []
        if self._match(TokenKind.COLON):
            supertraits.append(self._parse_type())
            while self._match(TokenKind.PLUS):
                supertraits.append(self._parse_type())

        if self._at(TokenKind.KW_WHERE):
            self._parse_where_clause(generics)

        self._expect(TokenKind.LBRACE)
        items: list[Item] = []
        while not self._at(TokenKind.RBRACE) and not self._at(TokenKind.EOF):
            try:
                item = self._parse_item()
                if item:
                    items.append(item)
            except ParseError:
                self._skip_to(TokenKind.SEMICOLON, TokenKind.RBRACE)
                self._match(TokenKind.SEMICOLON)

        self._expect(TokenKind.RBRACE)
        return TraitItem(
            name=name, items=items, generics=generics,
            supertraits=supertraits, is_unsafe=is_unsafe,
            loc=self._loc(start), attributes=attrs, visibility=vis,
        )

    def _parse_extern_item(
        self, attrs: list[Attribute], vis: Visibility, start: Token,
    ) -> Item:
        """Parse extern "C" fn ... or extern "C" { ... }."""
        self._advance()  # extern
        abi = "C"
        if self._at(TokenKind.STRING_LITERAL):
            abi = self._advance().string_value or "C"
        # After consuming ABI, check if it's a block or function
        if self._at(TokenKind.LBRACE):
            return self._parse_extern_block_body(attrs, vis, start, abi)
        return self._parse_fn_item(attrs, vis, start, abi=abi, is_extern=True)

    def _parse_extern_block_body(
        self, attrs: list[Attribute], vis: Visibility, start: Token, abi: str,
    ) -> ExternBlock:
        """Parse the body of extern "C" { ... } (LBRACE already peeked)."""
        self._advance()  # {
        items: list[Item] = []
        while not self._at(TokenKind.RBRACE) and not self._at(TokenKind.EOF):
            try:
                item = self._parse_item()
                if item:
                    items.append(item)
            except ParseError:
                self._skip_to(TokenKind.SEMICOLON, TokenKind.RBRACE)
                self._match(TokenKind.SEMICOLON)
        self._expect(TokenKind.RBRACE)
        return ExternBlock(
            abi=abi, items=items,
            loc=self._loc(start), attributes=attrs, visibility=vis,
        )

    def _parse_extern_block(
        self, attrs: list[Attribute], vis: Visibility, start: Token,
    ) -> ExternBlock:
        """Parse extern "C" { ... }."""
        self._advance()  # extern
        abi = "C"
        if self._at(TokenKind.STRING_LITERAL):
            abi = self._advance().string_value or "C"
        self._expect(TokenKind.LBRACE)
        items: list[Item] = []
        while not self._at(TokenKind.RBRACE) and not self._at(TokenKind.EOF):
            try:
                item = self._parse_item()
                if item:
                    items.append(item)
            except ParseError:
                self._skip_to(TokenKind.SEMICOLON, TokenKind.RBRACE)
                self._match(TokenKind.SEMICOLON)
        self._expect(TokenKind.RBRACE)
        return ExternBlock(
            abi=abi, items=items,
            loc=self._loc(start), attributes=attrs, visibility=vis,
        )

    def _parse_mod_item(
        self, attrs: list[Attribute], vis: Visibility, start: Token,
    ) -> ModItem:
        self._advance()  # mod
        name = self._expect(TokenKind.IDENT).text
        items: list[Item] = []
        is_inline = False
        if self._at(TokenKind.LBRACE):
            is_inline = True
            self._advance()
            while not self._at(TokenKind.RBRACE) and not self._at(TokenKind.EOF):
                item = self._parse_item()
                if item:
                    items.append(item)
            self._expect(TokenKind.RBRACE)
        else:
            self._expect(TokenKind.SEMICOLON)
        return ModItem(
            name=name, items=items, is_inline=is_inline,
            loc=self._loc(start), attributes=attrs, visibility=vis,
        )

    def _parse_macro_item(
        self, attrs: list[Attribute], vis: Visibility, start: Token,
    ) -> Optional[Item]:
        """Parse a macro invocation as item."""
        name = self._advance().text
        self._advance()  # !
        # Collect balanced delimiters
        body, _ = self._parse_macro_body()
        self._match(TokenKind.SEMICOLON)
        return MacroDefItem(name=name, body=body, loc=self._loc(start),
                           attributes=attrs, visibility=vis)

    def _parse_macro_body(self) -> tuple[str, str]:
        """Parse macro invocation body."""
        if self._at(TokenKind.LPAREN):
            return self._collect_balanced(TokenKind.LPAREN, TokenKind.RPAREN), "()"
        if self._at(TokenKind.LBRACKET):
            return self._collect_balanced(TokenKind.LBRACKET, TokenKind.RBRACKET), "[]"
        if self._at(TokenKind.LBRACE):
            return self._collect_balanced(TokenKind.LBRACE, TokenKind.RBRACE), "{}"
        return "", ""

    def _collect_balanced(self, open_kind: TokenKind, close_kind: TokenKind) -> str:
        """Collect tokens between balanced delimiters."""
        self._advance()  # opening
        depth = 1
        parts: list[str] = []
        while depth > 0 and not self._at(TokenKind.EOF):
            if self._at(open_kind):
                depth += 1
            elif self._at(close_kind):
                depth -= 1
                if depth == 0:
                    self._advance()
                    break
            parts.append(self._cur().text)
            self._advance()
        return " ".join(parts)

    # -------------------------------------------------------------------
    # Attributes
    # -------------------------------------------------------------------

    def _parse_outer_attributes(self) -> list[Attribute]:
        attrs: list[Attribute] = []
        while self._at(TokenKind.HASH) and self._peek(1).kind == TokenKind.LBRACKET:
            attrs.append(self._parse_attribute())
        return attrs

    def _parse_attribute(self, inner: bool = False) -> Attribute:
        start = self._cur()
        if inner:
            self._advance()  # #!
        else:
            self._advance()  # #
        self._expect(TokenKind.LBRACKET)

        path: list[str] = []
        if self._at(TokenKind.IDENT):
            path.append(self._advance().text)
            while self._match(TokenKind.PATH_SEP):
                if self._at(TokenKind.IDENT):
                    path.append(self._advance().text)

        args = ""
        if self._at(TokenKind.LPAREN):
            args = self._collect_balanced(TokenKind.LPAREN, TokenKind.RPAREN)
        elif self._match(TokenKind.ASSIGN):
            args = self._cur().text
            self._advance()

        self._expect(TokenKind.RBRACKET)
        return Attribute(path=path, args=args, is_inner=inner, loc=self._loc(start))

    def _parse_visibility(self) -> Visibility:
        if self._match(TokenKind.KW_PUB):
            if self._at(TokenKind.LPAREN):
                self._advance()
                if self._match(TokenKind.KW_CRATE):
                    self._expect(TokenKind.RPAREN)
                    return Visibility.PUB_CRATE
                elif self._match(TokenKind.KW_SUPER):
                    self._expect(TokenKind.RPAREN)
                    return Visibility.PUB_SUPER
                else:
                    self._skip_to(TokenKind.RPAREN)
                    self._advance()
                    return Visibility.PUB_IN
            return Visibility.PUB
        return Visibility.PRIVATE

    # -------------------------------------------------------------------
    # Generics
    # -------------------------------------------------------------------

    def _parse_generics(self) -> Generics:
        generics = Generics()
        if not self._at(TokenKind.LT):
            return generics

        self._advance()  # <
        while not self._at(TokenKind.GT) and not self._at(TokenKind.SHR) and not self._at(TokenKind.EOF):
            if self._at(TokenKind.LIFETIME):
                name = self._advance().lifetime_name
                generics.params.append(GenericParam(name=name, is_lifetime=True))
            elif self._at(TokenKind.IDENT):
                name = self._advance().text
                bounds: list[RustType] = []
                if self._match(TokenKind.COLON):
                    bounds.append(self._parse_type())
                    while self._match(TokenKind.PLUS):
                        bounds.append(self._parse_type())
                default = None
                if self._match(TokenKind.ASSIGN):
                    default = self._parse_type()
                generics.params.append(GenericParam(
                    name=name, bounds=bounds, default=default,
                ))
            elif self._at(TokenKind.KW_CONST):
                self._advance()  # const
                name = self._expect(TokenKind.IDENT).text
                self._expect(TokenKind.COLON)
                _ = self._parse_type()
                generics.params.append(GenericParam(name=name))

            if not self._match(TokenKind.COMMA):
                break

        if self._at(TokenKind.SHR):
            self._split_shr()
        else:
            self._expect(TokenKind.GT)
        return generics

    def _parse_where_clause(self, generics: Generics) -> None:
        self._advance()  # where
        while not self._at_any(TokenKind.LBRACE, TokenKind.SEMICOLON, TokenKind.EOF):
            ty = self._parse_type()
            bounds: list[RustType] = []
            if self._match(TokenKind.COLON):
                bounds.append(self._parse_type())
                while self._match(TokenKind.PLUS):
                    bounds.append(self._parse_type())
            generics.where_clauses.append(WhereClause(bounded_type=ty, bounds=bounds))
            if not self._match(TokenKind.COMMA):
                break

    # -------------------------------------------------------------------
    # Types
    # -------------------------------------------------------------------

    def _parse_type(self) -> RustType:
        """Parse a type."""
        tok = self._cur()

        if tok.kind == TokenKind.BANG:
            self._advance()
            return NeverType()

        if tok.kind == TokenKind.LPAREN:
            return self._parse_tuple_or_paren_type()

        if tok.kind == TokenKind.LBRACKET:
            return self._parse_array_or_slice_type()

        if tok.kind == TokenKind.AMP:
            return self._parse_reference_type()

        if tok.kind == TokenKind.STAR:
            return self._parse_raw_pointer_type()

        if tok.kind == TokenKind.KW_FN:
            return self._parse_fn_pointer_type()

        if tok.kind == TokenKind.KW_UNSAFE:
            return self._parse_fn_pointer_type(is_unsafe=True)

        if tok.kind == TokenKind.KW_EXTERN:
            return self._parse_fn_pointer_type(is_extern=True)

        if tok.kind == TokenKind.UNDERSCORE:
            self._advance()
            return InferredType()

        if tok.kind == TokenKind.KW_DYN:
            self._advance()
            return self._parse_type()

        # Path type
        return self._parse_path_type()

    def _parse_path_type(self) -> PathType:
        segments: list[str] = []
        generic_args: list[RustType] = []

        # Optional leading ::
        if self._match(TokenKind.PATH_SEP):
            pass

        if self._at(TokenKind.IDENT) or self._at_any(
            TokenKind.KW_SELF, TokenKind.KW_SUPER, TokenKind.KW_CRATE,
            TokenKind.KW_SELF_TYPE,
        ):
            segments.append(self._advance().text)
        else:
            self._error("expected type name")
            return PathType(segments=["?"])

        while self._match(TokenKind.PATH_SEP):
            if self._at(TokenKind.IDENT):
                segments.append(self._advance().text)
            elif self._at(TokenKind.LT):
                break
            else:
                break

        # Generic arguments
        if self._at(TokenKind.LT):
            saved = self._pos
            try:
                self._advance()  # <
                while not self._at(TokenKind.GT) and not self._at(TokenKind.SHR) and not self._at(TokenKind.EOF):
                    if self._at(TokenKind.LIFETIME):
                        self._advance()
                    else:
                        generic_args.append(self._parse_type())
                    if not self._match(TokenKind.COMMA):
                        break
                if self._at(TokenKind.SHR):
                    # >> in nested generics: consume one > and leave one
                    self._split_shr()
                else:
                    self._expect(TokenKind.GT)
            except ParseError:
                self._pos = saved
                generic_args = []

        return PathType(segments=segments, generic_args=generic_args)

    def _parse_tuple_or_paren_type(self) -> RustType:
        self._advance()  # (
        if self._at(TokenKind.RPAREN):
            self._advance()
            return UnitType()

        first = self._parse_type()
        if self._at(TokenKind.RPAREN):
            self._advance()
            return ParenType(inner=first)

        elements = [first]
        while self._match(TokenKind.COMMA):
            if self._at(TokenKind.RPAREN):
                break
            elements.append(self._parse_type())
        self._expect(TokenKind.RPAREN)
        return TupleType(elements=elements)

    def _parse_array_or_slice_type(self) -> RustType:
        self._advance()  # [
        elem = self._parse_type()
        if self._match(TokenKind.SEMICOLON):
            length = self._parse_expression()
            self._expect(TokenKind.RBRACKET)
            return RustArrayType(element=elem, length=length)
        self._expect(TokenKind.RBRACKET)
        return SliceType(element=elem)

    def _parse_reference_type(self) -> ReferenceType:
        self._advance()  # &
        lifetime = ""
        if self._at(TokenKind.LIFETIME):
            lifetime = self._advance().lifetime_name
        mut = Mutability.IMMUTABLE
        if self._match(TokenKind.KW_MUT):
            mut = Mutability.MUTABLE
        referent = self._parse_type()
        return ReferenceType(referent=referent, mutability=mut, lifetime=lifetime)

    def _parse_raw_pointer_type(self) -> RawPointerType:
        self._advance()  # *
        mut = Mutability.IMMUTABLE
        if self._match(TokenKind.KW_MUT):
            mut = Mutability.MUTABLE
        elif self._match(TokenKind.KW_CONST):
            mut = Mutability.IMMUTABLE
        pointee = self._parse_type()
        return RawPointerType(pointee=pointee, mutability=mut)

    def _parse_fn_pointer_type(
        self, is_unsafe: bool = False, is_extern: bool = False,
    ) -> FnPointerType:
        if is_unsafe:
            self._advance()  # unsafe
        abi = ""
        if self._at(TokenKind.KW_EXTERN) or is_extern:
            if self._at(TokenKind.KW_EXTERN):
                self._advance()
            if self._at(TokenKind.STRING_LITERAL):
                abi = self._advance().string_value or "C"
        self._expect(TokenKind.KW_FN)
        self._expect(TokenKind.LPAREN)
        params: list[RustType] = []
        while not self._at(TokenKind.RPAREN) and not self._at(TokenKind.EOF):
            params.append(self._parse_type())
            if not self._match(TokenKind.COMMA):
                break
        self._expect(TokenKind.RPAREN)
        ret = None
        if self._match(TokenKind.ARROW):
            ret = self._parse_type()
        return FnPointerType(params=params, return_type=ret,
                            is_unsafe=is_unsafe, abi=abi)

    # -------------------------------------------------------------------
    # Patterns
    # -------------------------------------------------------------------

    def _parse_pattern(self) -> Pattern:
        """Parse a pattern."""
        start = self._cur()
        pat = self._parse_single_pattern()

        # Or pattern
        if self._at(TokenKind.PIPE):
            alternatives = [pat]
            while self._match(TokenKind.PIPE):
                alternatives.append(self._parse_single_pattern())
            return OrPattern(alternatives=alternatives, loc=self._loc(start))

        return pat

    def _parse_single_pattern(self) -> Pattern:
        start = self._cur()

        if self._at(TokenKind.UNDERSCORE):
            self._advance()
            return WildcardPattern(loc=self._loc(start))

        if self._at(TokenKind.KW_MUT):
            self._advance()
            name = self._expect(TokenKind.IDENT).text
            return IdentPattern(name=name, mutability=Mutability.MUTABLE,
                               loc=self._loc(start))

        if self._at(TokenKind.KW_REF):
            self._advance()
            mut = Mutability.IMMUTABLE
            if self._match(TokenKind.KW_MUT):
                mut = Mutability.MUTABLE
            name = self._expect(TokenKind.IDENT).text
            return IdentPattern(name=name, is_ref=True, mutability=mut,
                               loc=self._loc(start))

        if self._at(TokenKind.AMP):
            self._advance()
            mut = Mutability.IMMUTABLE
            if self._match(TokenKind.KW_MUT):
                mut = Mutability.MUTABLE
            inner = self._parse_pattern()
            return RefPattern(inner=inner, mutability=mut, loc=self._loc(start))

        if self._at(TokenKind.LPAREN):
            self._advance()
            elements: list[Pattern] = []
            while not self._at(TokenKind.RPAREN) and not self._at(TokenKind.EOF):
                elements.append(self._parse_pattern())
                if not self._match(TokenKind.COMMA):
                    break
            self._expect(TokenKind.RPAREN)
            return TuplePattern(elements=elements, loc=self._loc(start))

        if self._at(TokenKind.LBRACKET):
            self._advance()
            elements = []
            while not self._at(TokenKind.RBRACKET) and not self._at(TokenKind.EOF):
                elements.append(self._parse_pattern())
                if not self._match(TokenKind.COMMA):
                    break
            self._expect(TokenKind.RBRACKET)
            return SlicePattern(elements=elements, loc=self._loc(start))

        if self._at_any(TokenKind.INT_LITERAL, TokenKind.FLOAT_LITERAL,
                       TokenKind.STRING_LITERAL, TokenKind.CHAR_LITERAL,
                       TokenKind.KW_TRUE, TokenKind.KW_FALSE, TokenKind.MINUS):
            expr = self._parse_primary_expr()
            return LiteralPattern(value=expr, loc=self._loc(start))

        if self._at(TokenKind.IDENT):
            name = self._advance().text
            # Check for path pattern: Name::Variant
            if self._at(TokenKind.PATH_SEP):
                path = [name]
                while self._match(TokenKind.PATH_SEP):
                    if self._at(TokenKind.IDENT):
                        path.append(self._advance().text)
                if self._at(TokenKind.LPAREN):
                    self._advance()
                    elems: list[Pattern] = []
                    while not self._at(TokenKind.RPAREN) and not self._at(TokenKind.EOF):
                        elems.append(self._parse_pattern())
                        if not self._match(TokenKind.COMMA):
                            break
                    self._expect(TokenKind.RPAREN)
                    return TupleStructPattern(path=path, elements=elems,
                                            loc=self._loc(start))
                if self._at(TokenKind.LBRACE):
                    return self._parse_struct_pattern(path, start)
                return PathPattern(path=path, loc=self._loc(start))

            # Check for struct pattern
            if self._at(TokenKind.LBRACE):
                return self._parse_struct_pattern([name], start)

            # Check for tuple struct pattern
            if self._at(TokenKind.LPAREN):
                self._advance()
                elems = []
                while not self._at(TokenKind.RPAREN) and not self._at(TokenKind.EOF):
                    elems.append(self._parse_pattern())
                    if not self._match(TokenKind.COMMA):
                        break
                self._expect(TokenKind.RPAREN)
                return TupleStructPattern(path=[name], elements=elems,
                                        loc=self._loc(start))

            # Binding pattern with possible subpattern
            subpat = None
            if self._match(TokenKind.AT):
                subpat = self._parse_pattern()
            return IdentPattern(name=name, subpattern=subpat, loc=self._loc(start))

        self._error("expected pattern")
        self._advance()
        return WildcardPattern(loc=self._loc(start))

    def _parse_struct_pattern(self, path: list[str], start: Token) -> StructPattern:
        self._advance()  # {
        fields: list[tuple[str, Pattern]] = []
        has_rest = False
        while not self._at(TokenKind.RBRACE) and not self._at(TokenKind.EOF):
            if self._at(TokenKind.DOT_DOT):
                self._advance()
                has_rest = True
                break
            fname = self._expect(TokenKind.IDENT).text
            if self._match(TokenKind.COLON):
                fpat = self._parse_pattern()
                fields.append((fname, fpat))
            else:
                fields.append((fname, IdentPattern(name=fname)))
            if not self._match(TokenKind.COMMA):
                break
        self._expect(TokenKind.RBRACE)
        return StructPattern(path=path, fields=fields, has_rest=has_rest,
                            loc=self._loc(start))

    # -------------------------------------------------------------------
    # Statements
    # -------------------------------------------------------------------

    def _parse_statement(self) -> Stmt:
        """Parse a statement within a block."""
        start = self._cur()

        # Skip outer attributes at statement level
        if self._at(TokenKind.HASH) and self._peek(1).kind == TokenKind.LBRACKET:
            self._parse_outer_attributes()
            # Fall through to parse the attributed item/expr

        if self._at(TokenKind.KW_LET):
            return self._parse_let_stmt()

        if self._at(TokenKind.SEMICOLON):
            self._advance()
            return EmptyStmt(loc=self._loc(start))

        # Check for item statement
        if self._at_any(TokenKind.KW_FN, TokenKind.KW_STRUCT, TokenKind.KW_ENUM,
                       TokenKind.KW_IMPL, TokenKind.KW_USE, TokenKind.KW_CONST,
                       TokenKind.KW_STATIC, TokenKind.KW_TYPE, TokenKind.KW_TRAIT,
                       TokenKind.KW_MOD, TokenKind.KW_EXTERN, TokenKind.KW_PUB):
            item = self._parse_item()
            return ItemStmt(item=item, loc=self._loc(start))

        if self._at(TokenKind.KW_UNSAFE) and self._peek(1).kind in (
            TokenKind.KW_FN, TokenKind.KW_IMPL, TokenKind.KW_TRAIT,
        ):
            item = self._parse_item()
            return ItemStmt(item=item, loc=self._loc(start))

        # Expression statement (with error recovery)
        try:
            expr = self._parse_expression()
        except ParseError as e:
            self._errors.append(e)
            self._skip_to(TokenKind.SEMICOLON, TokenKind.RBRACE)
            self._match(TokenKind.SEMICOLON)
            return EmptyStmt(loc=self._loc(start))
        has_semi = bool(self._match(TokenKind.SEMICOLON))
        return ExprStmt(expr=expr, has_semicolon=has_semi, loc=self._loc(start))

    def _parse_let_stmt(self) -> Stmt:
        start = self._cur()
        self._advance()  # let
        is_mut = bool(self._match(TokenKind.KW_MUT))
        try:
            pat = self._parse_pattern()
        except ParseError:
            self._skip_to(TokenKind.SEMICOLON, TokenKind.RBRACE)
            self._match(TokenKind.SEMICOLON)
            return EmptyStmt(loc=self._loc(start))
        ty = None
        if self._match(TokenKind.COLON):
            try:
                ty = self._parse_type()
            except ParseError:
                self._skip_to(TokenKind.ASSIGN, TokenKind.SEMICOLON, TokenKind.RBRACE)
        init = None
        if self._match(TokenKind.ASSIGN):
            try:
                init = self._parse_expression()
            except ParseError:
                self._skip_to(TokenKind.SEMICOLON, TokenKind.RBRACE)
        # let-else: let pat = expr else { diverging_block }
        if self._at(TokenKind.KW_ELSE):
            self._advance()
            else_block = self._parse_block_expr()
            self._match(TokenKind.SEMICOLON)
            return LetElseStmt(pattern=pat, type_ann=ty, initializer=init,
                               else_block=else_block, is_mutable=is_mut,
                               loc=self._loc(start))
        self._expect(TokenKind.SEMICOLON)
        return LetStmt(pattern=pat, type_ann=ty, initializer=init,
                       is_mutable=is_mut, loc=self._loc(start))

    # -------------------------------------------------------------------
    # Expressions
    # -------------------------------------------------------------------

    def _parse_expression(self) -> Expr:
        """Parse a full expression."""
        return self._parse_assignment_expr()

    def _parse_assignment_expr(self) -> Expr:
        """Parse assignment or binary expression."""
        left = self._parse_range_expr()

        # Assignment operators
        if self._cur().kind in _ASSIGN_OPS:
            op = _ASSIGN_OPS[self._cur().kind]
            self._advance()
            right = self._parse_expression()
            return AssignExpr(lhs=left, rhs=right, op=op, loc=left.loc)

        return left

    def _parse_range_expr(self) -> Expr:
        """Parse range expressions."""
        if self._at(TokenKind.DOT_DOT) or self._at(TokenKind.DOT_DOT_EQ):
            inclusive = self._cur().kind == TokenKind.DOT_DOT_EQ
            self._advance()
            end = self._parse_prec_expr(3) if not self._at_any(
                TokenKind.SEMICOLON, TokenKind.COMMA, TokenKind.RPAREN,
                TokenKind.RBRACKET, TokenKind.RBRACE,
            ) else None
            return RangeExpr(end=end, inclusive=inclusive)

        left = self._parse_prec_expr(3)

        if self._at_any(TokenKind.DOT_DOT, TokenKind.DOT_DOT_EQ):
            inclusive = self._cur().kind == TokenKind.DOT_DOT_EQ
            self._advance()
            end = self._parse_prec_expr(3) if not self._at_any(
                TokenKind.SEMICOLON, TokenKind.COMMA, TokenKind.RPAREN,
                TokenKind.RBRACKET, TokenKind.RBRACE,
            ) else None
            return RangeExpr(start=left, end=end, inclusive=inclusive)

        return left

    def _parse_prec_expr(self, min_prec: int) -> Expr:
        """Precedence climbing expression parser."""
        left = self._parse_prefix_expr()

        while True:
            tok = self._cur()
            prec = _BINOP_PRECEDENCE.get(tok.kind)
            if prec is None or prec < min_prec:
                break

            # Disambiguate: & as binary AND vs unary ref
            if tok.kind == TokenKind.AMP and prec == 9:
                pass  # It's a binary AND

            binop = _TOKEN_TO_BINOP.get(tok.kind)
            if binop is None:
                break

            self._advance()
            right = self._parse_prec_expr(prec + 1)
            left = BinaryExpr(op=binop, lhs=left, rhs=right, loc=left.loc)

        return left

    def _parse_prefix_expr(self) -> Expr:
        """Parse prefix/unary expressions."""
        tok = self._cur()

        if tok.kind == TokenKind.MINUS:
            self._advance()
            operand = self._parse_prefix_expr()
            return UnaryExpr(op=RustUnaryOp.NEG, operand=operand, loc=self._loc(tok))

        if tok.kind == TokenKind.BANG:
            self._advance()
            operand = self._parse_prefix_expr()
            return UnaryExpr(op=RustUnaryOp.NOT, operand=operand, loc=self._loc(tok))

        if tok.kind == TokenKind.STAR:
            self._advance()
            operand = self._parse_prefix_expr()
            return DerefExpr(operand=operand, loc=self._loc(tok))

        if tok.kind == TokenKind.AMP:
            self._advance()
            mut = Mutability.IMMUTABLE
            if self._match(TokenKind.KW_MUT):
                mut = Mutability.MUTABLE
            operand = self._parse_prefix_expr()
            return RefExpr(operand=operand, mutability=mut, loc=self._loc(tok))

        return self._parse_postfix_expr()

    def _parse_postfix_expr(self) -> Expr:
        """Parse postfix expressions: calls, field access, indexing, as casts."""
        expr = self._parse_primary_expr()

        while True:
            tok = self._cur()

            if tok.kind == TokenKind.DOT:
                self._advance()
                if self._at(TokenKind.INT_LITERAL):
                    idx = self._advance().int_value or 0
                    expr = TupleFieldExpr(base=expr, index=idx, loc=expr.loc)
                elif self._at(TokenKind.KW_AWAIT):
                    self._advance()
                    expr = AwaitExpr(operand=expr, loc=expr.loc)
                elif self._at(TokenKind.IDENT):
                    name = self._advance().text
                    if self._at(TokenKind.LPAREN):
                        # Method call
                        self._advance()
                        args: list[Expr] = []
                        while not self._at(TokenKind.RPAREN) and not self._at(TokenKind.EOF):
                            try:
                                args.append(self._parse_expression())
                            except ParseError:
                                self._skip_to(TokenKind.RPAREN, TokenKind.COMMA)
                            if not self._match(TokenKind.COMMA):
                                break
                        self._expect(TokenKind.RPAREN)
                        expr = MethodCallExpr(receiver=expr, method=name, args=args,
                                             loc=expr.loc)
                    elif self._at(TokenKind.PATH_SEP) and self._peek(1).kind == TokenKind.LT:
                        # Turbofish: .method::<T>()
                        saved = self._pos
                        try:
                            self._advance()  # ::
                            generics = self._parse_turbofish_args()
                            if self._at(TokenKind.LPAREN):
                                self._advance()
                                args = []
                                while not self._at(TokenKind.RPAREN) and not self._at(TokenKind.EOF):
                                    args.append(self._parse_expression())
                                    if not self._match(TokenKind.COMMA):
                                        break
                                self._expect(TokenKind.RPAREN)
                                expr = MethodCallExpr(receiver=expr, method=name, args=args,
                                                     generic_args=generics, loc=expr.loc)
                            else:
                                # Turbofish without call — treat as field access
                                expr = FieldExpr(base=expr, field_name=name, loc=expr.loc)
                        except ParseError:
                            self._pos = saved
                            expr = FieldExpr(base=expr, field_name=name, loc=expr.loc)
                    else:
                        expr = FieldExpr(base=expr, field_name=name, loc=expr.loc)
                else:
                    # Dangling dot — recover gracefully
                    if self.lenient:
                        break
                    break

            elif tok.kind == TokenKind.LPAREN:
                self._advance()
                args = []
                while not self._at(TokenKind.RPAREN) and not self._at(TokenKind.EOF):
                    try:
                        args.append(self._parse_expression())
                    except ParseError:
                        self._skip_to(TokenKind.RPAREN, TokenKind.COMMA)
                    if not self._match(TokenKind.COMMA):
                        break
                self._expect(TokenKind.RPAREN)
                expr = CallExpr(callee=expr, args=args, loc=expr.loc)

            elif tok.kind == TokenKind.LBRACKET:
                self._advance()
                index = self._parse_expression()
                self._expect(TokenKind.RBRACKET)
                expr = IndexExpr(base=expr, index=index, loc=expr.loc)

            elif tok.kind == TokenKind.KW_AS:
                self._advance()
                try:
                    target_type = self._parse_type()
                except ParseError:
                    target_type = PathType(segments=["?"])
                expr = CastExpr(operand=expr, target_type=target_type, loc=expr.loc)

            elif tok.kind == TokenKind.QUESTION:
                self._advance()
                expr = TryExpr(operand=expr, loc=expr.loc)

            else:
                break

        return expr

    def _parse_turbofish_args(self) -> list[RustType]:
        """Parse turbofish generic arguments: <Type, ...>."""
        generics: list[RustType] = []
        self._advance()  # <
        while not self._at(TokenKind.GT) and not self._at(TokenKind.SHR) and not self._at(TokenKind.EOF):
            generics.append(self._parse_type())
            if not self._match(TokenKind.COMMA):
                break
        if self._at(TokenKind.SHR):
            self._split_shr()
        else:
            self._expect(TokenKind.GT)
        return generics

    def _parse_primary_expr(self) -> Expr:
        """Parse primary expressions."""
        tok = self._cur()

        if tok.kind == TokenKind.INT_LITERAL:
            self._advance()
            return LitExpr(int_value=tok.int_value, type_suffix=tok.type_suffix,
                          text=tok.text, loc=self._loc(tok))

        if tok.kind == TokenKind.FLOAT_LITERAL:
            self._advance()
            return LitExpr(float_value=tok.float_value, type_suffix=tok.type_suffix,
                          text=tok.text, loc=self._loc(tok))

        if tok.kind in (TokenKind.STRING_LITERAL, TokenKind.RAW_STRING_LITERAL,
                        TokenKind.BYTE_STRING_LITERAL, TokenKind.RAW_BYTE_STRING_LITERAL):
            self._advance()
            return LitExpr(string_value=tok.string_value, text=tok.text,
                          loc=self._loc(tok))

        if tok.kind in (TokenKind.CHAR_LITERAL, TokenKind.BYTE_LITERAL):
            self._advance()
            return LitExpr(char_value=tok.char_value, text=tok.text,
                          loc=self._loc(tok))

        if tok.kind in (TokenKind.KW_TRUE, TokenKind.KW_FALSE):
            self._advance()
            return LitExpr(bool_value=tok.bool_value, text=tok.text,
                          loc=self._loc(tok))

        if tok.kind == TokenKind.IDENT:
            return self._parse_path_or_struct_expr()

        if tok.kind == TokenKind.KW_SELF:
            self._advance()
            return PathExpr(segments=["self"], loc=self._loc(tok))

        if tok.kind == TokenKind.KW_SELF_TYPE:
            self._advance()
            return PathExpr(segments=["Self"], loc=self._loc(tok))

        # Leading :: for absolute paths: ::std::mem::size_of
        if tok.kind == TokenKind.PATH_SEP:
            self._advance()
            if self._at(TokenKind.IDENT):
                return self._parse_path_or_struct_expr()
            self._error("expected identifier after ::")
            return PathExpr(segments=["<error>"], loc=self._loc(tok))

        # Labeled block: 'label: loop { ... }
        if tok.kind == TokenKind.LIFETIME:
            label = self._advance().lifetime_name
            if self._match(TokenKind.COLON):
                if self._at(TokenKind.KW_LOOP):
                    expr = self._parse_loop_expr()
                    if hasattr(expr, 'label'):
                        expr.label = label
                    return expr
                if self._at(TokenKind.KW_WHILE):
                    expr = self._parse_while_expr()
                    if hasattr(expr, 'label'):
                        expr.label = label
                    return expr
                if self._at(TokenKind.KW_FOR):
                    expr = self._parse_for_expr()
                    if hasattr(expr, 'label'):
                        expr.label = label
                    return expr
                if self._at(TokenKind.LBRACE):
                    return self._parse_block_expr()
            return PathExpr(segments=[f"'{label}"], loc=self._loc(tok))

        if tok.kind == TokenKind.LPAREN:
            return self._parse_tuple_or_paren_expr()

        if tok.kind == TokenKind.LBRACKET:
            return self._parse_array_expr()

        if tok.kind == TokenKind.LBRACE:
            return self._parse_block_expr()

        if tok.kind == TokenKind.KW_IF:
            return self._parse_if_expr()

        if tok.kind == TokenKind.KW_MATCH:
            return self._parse_match_expr()

        if tok.kind == TokenKind.KW_LOOP:
            return self._parse_loop_expr()

        if tok.kind == TokenKind.KW_WHILE:
            return self._parse_while_expr()

        if tok.kind == TokenKind.KW_FOR:
            return self._parse_for_expr()

        if tok.kind == TokenKind.KW_RETURN:
            return self._parse_return_expr()

        if tok.kind == TokenKind.KW_BREAK:
            return self._parse_break_expr()

        if tok.kind == TokenKind.KW_CONTINUE:
            self._advance()
            label = ""
            if self._at(TokenKind.LIFETIME):
                label = self._advance().lifetime_name
            return ContinueExpr(label=label, loc=self._loc(tok))

        if tok.kind == TokenKind.KW_UNSAFE:
            self._advance()
            body = self._parse_block_expr()
            body.is_unsafe = True
            return UnsafeBlock(body=body, loc=self._loc(tok))

        if tok.kind == TokenKind.KW_ASYNC:
            self._advance()
            is_move = bool(self._match(TokenKind.KW_MOVE))
            body = self._parse_block_expr()
            return AsyncBlock(body=body, is_move=is_move, loc=self._loc(tok))

        if tok.kind == TokenKind.PIPE:
            return self._parse_closure_expr()

        if tok.kind == TokenKind.OR:
            return self._parse_closure_expr()

        if tok.kind == TokenKind.KW_MOVE:
            return self._parse_closure_expr()

        if tok.kind == TokenKind.MINUS and self._peek(1).kind in (
            TokenKind.INT_LITERAL, TokenKind.FLOAT_LITERAL,
        ):
            self._advance()
            inner = self._parse_primary_expr()
            return UnaryExpr(op=RustUnaryOp.NEG, operand=inner, loc=self._loc(tok))

        self._error(f"expected expression, got {tok.kind.name} ({tok.text!r})", tok)
        self._advance()
        return PathExpr(segments=["<error>"], loc=self._loc(tok))

    def _parse_path_or_struct_expr(self) -> Expr:
        """Parse path expression, possibly with struct literal."""
        start = self._cur()
        segments: list[str] = [self._advance().text]

        while self._match(TokenKind.PATH_SEP):
            if self._at(TokenKind.IDENT):
                segments.append(self._advance().text)
            elif self._at(TokenKind.LT):
                # Turbofish on path: Path::<T>(...)
                saved = self._pos
                try:
                    generics = self._parse_turbofish_args()
                    # After turbofish, continue collecting path segments
                    while self._match(TokenKind.PATH_SEP):
                        if self._at(TokenKind.IDENT):
                            segments.append(self._advance().text)
                        else:
                            break
                    # After turbofish (+ optional path), check for function call
                    if self._at(TokenKind.LPAREN):
                        self._advance()
                        args: list[Expr] = []
                        while not self._at(TokenKind.RPAREN) and not self._at(TokenKind.EOF):
                            args.append(self._parse_expression())
                            if not self._match(TokenKind.COMMA):
                                break
                        self._expect(TokenKind.RPAREN)
                        path_expr = PathExpr(segments=segments, loc=self._loc(start))
                        return CallExpr(callee=path_expr, args=args,
                                       loc=self._loc(start))
                    # No call after turbofish — just a path
                    break
                except ParseError:
                    self._pos = saved
                    break
            else:
                break

        # Check for macro invocation: name!()
        if self._at(TokenKind.BANG):
            self._advance()
            macro_name = "::".join(segments)
            # Handle asm! as InlineAsm
            if macro_name == "asm" or macro_name == "core::arch::asm":
                body, _ = self._parse_macro_body()
                return InlineAsm(template=body, loc=self._loc(start))
            body, _ = self._parse_macro_body()
            return MacroInvocation(name=macro_name, args=body,
                                  loc=self._loc(start))

        path_expr = PathExpr(segments=segments, loc=self._loc(start))

        # Handle transmute calls: std::mem::transmute(expr) or transmute(expr)
        full_path = "::".join(segments)
        if full_path in ("std::mem::transmute", "core::mem::transmute", "transmute"):
            if self._at(TokenKind.LPAREN):
                self._advance()
                args = []
                while not self._at(TokenKind.RPAREN) and not self._at(TokenKind.EOF):
                    args.append(self._parse_expression())
                    if not self._match(TokenKind.COMMA):
                        break
                self._expect(TokenKind.RPAREN)
                operand = args[0] if args else None
                return TransmuteCall(operand=operand, loc=self._loc(start))

        # Handle transmute with turbofish: transmute::<T>(expr)
        if full_path in ("std::mem::transmute", "core::mem::transmute", "transmute"):
            if self._at(TokenKind.PATH_SEP) and self._peek(1).kind == TokenKind.LT:
                saved = self._pos
                try:
                    self._advance()  # ::
                    self._parse_turbofish_args()
                    if self._at(TokenKind.LPAREN):
                        self._advance()
                        args = []
                        while not self._at(TokenKind.RPAREN) and not self._at(TokenKind.EOF):
                            args.append(self._parse_expression())
                            if not self._match(TokenKind.COMMA):
                                break
                        self._expect(TokenKind.RPAREN)
                        operand = args[0] if args else None
                        return TransmuteCall(operand=operand, loc=self._loc(start))
                except ParseError:
                    self._pos = saved

        # Check for struct literal: Name { field: value }
        # Disabled inside if/while/match conditions to avoid ambiguity
        if self._at(TokenKind.LBRACE) and not self._no_struct_literal:
            # Heuristic: struct literal
            saved = self._pos
            try:
                return self._parse_struct_expr(segments, start)
            except ParseError:
                self._pos = saved

        return path_expr

    def _parse_struct_expr(self, path: list[str], start: Token) -> StructExpr:
        self._advance()  # {
        fields: list[tuple[str, Expr]] = []
        base = None

        while not self._at(TokenKind.RBRACE) and not self._at(TokenKind.EOF):
            if self._at(TokenKind.DOT_DOT):
                self._advance()
                base = self._parse_expression()
                break

            if self._at(TokenKind.IDENT):
                fname = self._advance().text
                if self._match(TokenKind.COLON):
                    fval = self._parse_expression()
                    fields.append((fname, fval))
                else:
                    # Shorthand: Name { x } == Name { x: x }
                    fields.append((fname, PathExpr(segments=[fname])))
            else:
                break

            if not self._match(TokenKind.COMMA):
                break

        self._expect(TokenKind.RBRACE)
        return StructExpr(path=path, fields=fields, base=base, loc=self._loc(start))

    def _parse_tuple_or_paren_expr(self) -> Expr:
        start = self._cur()
        self._advance()  # (

        if self._at(TokenKind.RPAREN):
            self._advance()
            return TupleExpr(elements=[], loc=self._loc(start))

        first = self._parse_expression()

        if self._at(TokenKind.RPAREN):
            self._advance()
            return ParenExpr(inner=first, loc=self._loc(start))

        elements = [first]
        while self._match(TokenKind.COMMA):
            if self._at(TokenKind.RPAREN):
                break
            elements.append(self._parse_expression())
        self._expect(TokenKind.RPAREN)
        return TupleExpr(elements=elements, loc=self._loc(start))

    def _parse_array_expr(self) -> ArrayExpr:
        start = self._cur()
        self._advance()  # [

        if self._at(TokenKind.RBRACKET):
            self._advance()
            return ArrayExpr(loc=self._loc(start))

        first = self._parse_expression()

        if self._match(TokenKind.SEMICOLON):
            count = self._parse_expression()
            self._expect(TokenKind.RBRACKET)
            return ArrayExpr(repeat_value=first, repeat_count=count,
                            loc=self._loc(start))

        elements = [first]
        while self._match(TokenKind.COMMA):
            if self._at(TokenKind.RBRACKET):
                break
            elements.append(self._parse_expression())
        self._expect(TokenKind.RBRACKET)
        return ArrayExpr(elements=elements, loc=self._loc(start))

    def _parse_block_expr(self) -> BlockExpr:
        start = self._cur()
        self._expect(TokenKind.LBRACE)
        stmts: list[Stmt] = []
        tail_expr: Optional[Expr] = None

        while not self._at(TokenKind.RBRACE) and not self._at(TokenKind.EOF):
            saved_pos = self._pos
            try:
                stmt = self._parse_statement()
                if isinstance(stmt, ExprStmt) and not stmt.has_semicolon:
                    # Could be tail expression
                    if self._at(TokenKind.RBRACE):
                        tail_expr = stmt.expr
                        break
                stmts.append(stmt)
            except ParseError as e:
                self._errors.append(e)
                self._skip_to(TokenKind.SEMICOLON, TokenKind.RBRACE)
                self._match(TokenKind.SEMICOLON)
            # Guard against infinite loops — must make progress
            if self._pos == saved_pos:
                self._advance()

        self._expect(TokenKind.RBRACE)
        return BlockExpr(stmts=stmts, tail_expr=tail_expr, loc=self._loc(start))

    def _parse_if_expr(self) -> Expr:
        start = self._cur()
        self._advance()  # if
        # if let pattern = expr { ... }
        if self._at(TokenKind.KW_LET):
            self._advance()  # let
            pat = self._parse_pattern()
            self._expect(TokenKind.ASSIGN)
            old_no_struct = self._no_struct_literal
            self._no_struct_literal = True
            scrutinee = self._parse_expression()
            self._no_struct_literal = old_no_struct
            then_body = self._parse_block_expr()
            else_body: Optional[Expr] = None
            if self._match(TokenKind.KW_ELSE):
                if self._at(TokenKind.KW_IF):
                    else_body = self._parse_if_expr()
                else:
                    else_body = self._parse_block_expr()
            return IfLetExpr(pattern=pat, scrutinee=scrutinee,
                             then_body=then_body, else_body=else_body,
                             loc=self._loc(start))
        # Regular if
        old_no_struct = self._no_struct_literal
        self._no_struct_literal = True
        cond = self._parse_expression()
        self._no_struct_literal = old_no_struct
        then_body = self._parse_block_expr()
        else_body = None
        if self._match(TokenKind.KW_ELSE):
            if self._at(TokenKind.KW_IF):
                else_body = self._parse_if_expr()
            else:
                else_body = self._parse_block_expr()
        return IfExpr(condition=cond, then_body=then_body, else_body=else_body,
                      loc=self._loc(start))

    def _parse_match_expr(self) -> MatchExpr:
        start = self._cur()
        self._advance()  # match
        old_no_struct = self._no_struct_literal
        self._no_struct_literal = True
        scrutinee = self._parse_expression()
        self._no_struct_literal = old_no_struct
        self._expect(TokenKind.LBRACE)
        arms: list[MatchArm] = []
        while not self._at(TokenKind.RBRACE) and not self._at(TokenKind.EOF):
            arm_start = self._cur()
            pat = self._parse_pattern()
            guard = None
            if self._match(TokenKind.KW_IF):
                guard = self._parse_expression()
            self._expect(TokenKind.FAT_ARROW)
            body = self._parse_expression()
            arms.append(MatchArm(pattern=pat, guard=guard, body=body,
                                loc=self._loc(arm_start)))
            if not self._match(TokenKind.COMMA):
                break
        self._expect(TokenKind.RBRACE)
        return MatchExpr(scrutinee=scrutinee, arms=arms, loc=self._loc(start))

    def _parse_loop_expr(self) -> LoopExpr:
        start = self._cur()
        self._advance()  # loop
        body = self._parse_block_expr()
        return LoopExpr(body=body, loc=self._loc(start))

    def _parse_while_expr(self) -> Expr:
        start = self._cur()
        self._advance()  # while
        # while let pattern = expr { ... }
        if self._at(TokenKind.KW_LET):
            self._advance()  # let
            pat = self._parse_pattern()
            self._expect(TokenKind.ASSIGN)
            old_no_struct = self._no_struct_literal
            self._no_struct_literal = True
            scrutinee = self._parse_expression()
            self._no_struct_literal = old_no_struct
            body = self._parse_block_expr()
            return WhileLetExpr(pattern=pat, scrutinee=scrutinee,
                                body=body, loc=self._loc(start))
        old_no_struct = self._no_struct_literal
        self._no_struct_literal = True
        cond = self._parse_expression()
        self._no_struct_literal = old_no_struct
        body = self._parse_block_expr()
        return WhileExpr(condition=cond, body=body, loc=self._loc(start))

    def _parse_for_expr(self) -> ForExpr:
        start = self._cur()
        self._advance()  # for
        pat = self._parse_pattern()
        self._expect(TokenKind.KW_IN)
        iter_expr = self._parse_expression()
        body = self._parse_block_expr()
        return ForExpr(pattern=pat, iterator=iter_expr, body=body,
                       loc=self._loc(start))

    def _parse_return_expr(self) -> ReturnExpr:
        start = self._cur()
        self._advance()  # return
        value = None
        if not self._at_any(TokenKind.SEMICOLON, TokenKind.RBRACE, TokenKind.EOF):
            value = self._parse_expression()
        return ReturnExpr(value=value, loc=self._loc(start))

    def _parse_break_expr(self) -> BreakExpr:
        start = self._cur()
        self._advance()  # break
        label = ""
        if self._at(TokenKind.LIFETIME):
            label = self._advance().lifetime_name
        value = None
        if not self._at_any(TokenKind.SEMICOLON, TokenKind.RBRACE, TokenKind.EOF):
            value = self._parse_expression()
        return BreakExpr(value=value, label=label, loc=self._loc(start))

    def _parse_closure_expr(self) -> ClosureExpr:
        start = self._cur()
        is_move = bool(self._match(TokenKind.KW_MOVE))

        params: list[FnParam] = []
        if self._match(TokenKind.OR):
            pass  # || — empty parameter list
        elif self._match(TokenKind.PIPE):
            # |param, ...| — parameter list between pipes
            while not self._at(TokenKind.PIPE) and not self._at(TokenKind.EOF):
                pat = self._parse_single_pattern()  # not _parse_pattern: | is delimiter here, not or-pattern
                ty = None
                if self._match(TokenKind.COLON):
                    ty = self._parse_type()
                params.append(FnParam(pattern=pat, type_ann=ty))
                if not self._match(TokenKind.COMMA):
                    break
            self._expect(TokenKind.PIPE)

        ret_type = None
        if self._match(TokenKind.ARROW):
            ret_type = self._parse_type()

        body = self._parse_expression()
        return ClosureExpr(params=params, return_type=ret_type, body=body,
                          is_move=is_move, loc=self._loc(start))
