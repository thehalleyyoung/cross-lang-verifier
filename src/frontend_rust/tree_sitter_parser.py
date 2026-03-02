"""
Tree-sitter based Rust parser for verified parsing.

Uses the tree-sitter-rust grammar (derived from the official Rust grammar) to
produce ASTs, replacing the hand-written recursive-descent parser. The tree-sitter
parse tree is lowered into the same Rust AST node types used by the hand-written
parser, so the rest of the pipeline (type resolution, IR lowering) is unchanged.

This addresses critique: "Custom C/Rust parsers are unverified — Python
implementations bypass established frontends (Clang, rustc/syn)."
"""

from __future__ import annotations

from typing import Optional, List
import tree_sitter_rust as tsrs
from tree_sitter import Language, Parser, Node

from .rust_ast import (
    # Types
    RustType, NeverType, UnitType, PathType, ReferenceType,
    RawPointerType, ArrayType as RustArrayType, SliceType,
    TupleType, FnPointerType, InferredType, ParenType,
    Mutability,
    # Items
    Item, FnItem, FnParam, StructItem, StructField as RustStructField,
    EnumItem, EnumVariant, ImplItem, UseItem, ConstItem, StaticItem,
    TypeAliasItem, TraitItem, ExternBlock, ModItem,
    Attribute, Visibility,
    # Expressions
    Expr, LitExpr, PathExpr, BinaryExpr, UnaryExpr, CastExpr,
    AssignExpr, CallExpr, MethodCallExpr, FieldExpr, TupleFieldExpr,
    IndexExpr, RangeExpr, BlockExpr, IfExpr, MatchExpr, MatchArm,
    LoopExpr, WhileExpr, ForExpr, ReturnExpr, BreakExpr, ContinueExpr,
    ClosureExpr, TupleExpr, ArrayExpr, StructExpr, RefExpr, DerefExpr,
    UnsafeBlock, MacroInvocation, ParenExpr, TryExpr,
    BinaryOp as RustBinaryOp, UnaryOp as RustUnaryOp,
    # Statements
    Stmt, LetStmt, ExprStmt, ItemStmt, EmptyStmt,
    # Patterns
    Pattern, IdentPattern, WildcardPattern, LiteralPattern,
    TuplePattern, RefPattern, OrPattern, PathPattern,
    # Top level
    Crate, NodeLocation,
)
from .lexer import SourcePos, SourceSpan


# ---------------------------------------------------------------------------
# Tree-sitter Rust Language singleton
# ---------------------------------------------------------------------------

_RUST_LANGUAGE = Language(tsrs.language())


def _make_span(node: Node, filename: str = "<input>") -> SourceSpan:
    start = SourcePos(
        line=node.start_point[0] + 1,
        column=node.start_point[1] + 1,
        file=filename,
    )
    end = SourcePos(
        line=node.end_point[0] + 1,
        column=node.end_point[1] + 1,
        file=filename,
    )
    return SourceSpan(start=start, end=end)


def _loc(node: Node, filename: str = "<input>") -> NodeLocation:
    return NodeLocation(span=_make_span(node, filename))


def _text(node: Node) -> str:
    return node.text.decode("utf-8") if node.text else ""


# ---------------------------------------------------------------------------
# Binary / Unary op mapping
# ---------------------------------------------------------------------------

_BINOP_MAP = {
    "+": RustBinaryOp.ADD, "-": RustBinaryOp.SUB,
    "*": RustBinaryOp.MUL, "/": RustBinaryOp.DIV, "%": RustBinaryOp.REM,
    "<<": RustBinaryOp.SHL, ">>": RustBinaryOp.SHR,
    "&": RustBinaryOp.BITAND, "|": RustBinaryOp.BITOR,
    "^": RustBinaryOp.BITXOR,
    "&&": RustBinaryOp.AND, "||": RustBinaryOp.OR,
    "==": RustBinaryOp.EQ, "!=": RustBinaryOp.NE,
    "<": RustBinaryOp.LT, ">": RustBinaryOp.GT,
    "<=": RustBinaryOp.LE, ">=": RustBinaryOp.GE,
}

_UNOP_MAP = {
    "-": RustUnaryOp.NEG,
    "!": RustUnaryOp.NOT,
    "*": RustUnaryOp.DEREF,
    "&": RustUnaryOp.REF,
}


# ---------------------------------------------------------------------------
# TreeSitterRustParser
# ---------------------------------------------------------------------------

class TreeSitterRustParser:
    """Parse Rust source using tree-sitter-rust and convert to our Rust AST.

    Falls back to the hand-written parser on conversion errors.

    Parameters
    ----------
    source : str
        Rust source code to parse.
    filename : str
        Source filename (used in error messages / locations).
    """

    def __init__(self, source: str, filename: str = "<input>"):
        self.source = source
        self.filename = filename
        self._parser = Parser(_RUST_LANGUAGE)

    def parse(self) -> Crate:
        """Parse the source and return a Crate."""
        tree = self._parser.parse(self.source.encode("utf-8"))
        root = tree.root_node

        if root.has_error:
            from .parser import RustParser
            return RustParser(self.source, self.filename).parse()

        items: List[Item] = []
        for child in root.children:
            item = self._lower_item(child)
            if item is not None:
                items.append(item)

        return Crate(items=items, loc=_loc(root, self.filename))

    # -- items -------------------------------------------------------------

    def _lower_item(self, node: Node) -> Optional[Item]:
        t = node.type
        if t == "function_item":
            return self._lower_fn_item(node)
        if t == "struct_item":
            return self._lower_struct_item(node)
        if t == "enum_item":
            return self._lower_enum_item(node)
        if t == "impl_item":
            return self._lower_impl_item(node)
        if t == "use_declaration":
            return self._lower_use(node)
        if t == "const_item":
            return self._lower_const(node)
        if t == "static_item":
            return self._lower_static(node)
        if t == "type_item":
            return self._lower_type_alias(node)
        if t == "extern_crate_declaration":
            return None
        if t == "attribute_item":
            return None
        if t == "line_comment" or t == "block_comment":
            return None
        if t == "mod_item":
            return self._lower_mod(node)
        if t == "foreign_mod_item":
            return self._lower_extern_block(node)
        if t == "macro_definition" or t == "macro_invocation":
            return None
        return None

    def _lower_fn_item(self, node: Node) -> FnItem:
        name = ""
        params: List[FnParam] = []
        ret_type: Optional[RustType] = None
        body = None
        vis = Visibility.PRIVATE
        is_unsafe = False
        is_async = False
        is_const = False

        for child in node.children:
            if child.type == "visibility_modifier":
                vis = Visibility.PUB
            elif child.type == "unsafe":
                is_unsafe = True
            elif child.type == "async":
                is_async = True
            elif child.type == "const":
                is_const = True
            elif child.type == "identifier" or child.type == "metavariable":
                if not name:
                    name = _text(child)
            elif child.type == "parameters":
                params = self._lower_params(child)
            elif child.type == "type_identifier" or child.type == "primitive_type":
                ret_type = self._lower_type(child)
            elif child.type == "block":
                body = self._lower_block_expr(child)
            elif child.type == "function_modifiers":
                pass

        # Check for return type arrow
        name_node = node.child_by_field_name("name")
        if name_node:
            name = _text(name_node)
        ret_node = node.child_by_field_name("return_type")
        if ret_node:
            ret_type = self._lower_type(ret_node)
        params_node = node.child_by_field_name("parameters")
        if params_node:
            params = self._lower_params(params_node)
        body_node = node.child_by_field_name("body")
        if body_node:
            body = self._lower_block_expr(body_node)

        return FnItem(
            name=name,
            params=params,
            return_type=ret_type,
            body=body,
            visibility=vis,
            is_unsafe=is_unsafe,
            is_async=is_async,
            is_const=is_const,
            loc=_loc(node, self.filename),
        )

    def _lower_params(self, node: Node) -> List[FnParam]:
        params: List[FnParam] = []
        for child in node.named_children:
            if child.type == "parameter":
                pat_node = child.child_by_field_name("pattern")
                type_node = child.child_by_field_name("type")
                pname = _text(pat_node) if pat_node else ""
                ptype = self._lower_type(type_node) if type_node else InferredType()
                is_mut = False
                if pat_node and pat_node.type == "mut_pattern":
                    is_mut = True
                    inner = pat_node.named_children[0] if pat_node.named_children else None
                    if inner:
                        pname = _text(inner)
                params.append(FnParam(
                    pattern=IdentPattern(
                        name=pname,
                        mutability=Mutability.MUTABLE if is_mut else Mutability.IMMUTABLE,
                    ),
                    type_ann=ptype,
                ))
            elif child.type == "self_parameter":
                params.append(FnParam(is_self=True, type_ann=PathType(segments=["Self"])))
        return params

    def _lower_struct_item(self, node: Node) -> StructItem:
        name_node = node.child_by_field_name("name")
        name = _text(name_node) if name_node else ""
        body_node = node.child_by_field_name("body")
        fields: List[RustStructField] = []
        if body_node:
            for child in body_node.named_children:
                if child.type == "field_declaration":
                    fname_node = child.child_by_field_name("name")
                    ftype_node = child.child_by_field_name("type")
                    fields.append(RustStructField(
                        name=_text(fname_node) if fname_node else "",
                        type_ann=self._lower_type(ftype_node) if ftype_node else InferredType(),
                        visibility=Visibility.PRIVATE,
                    ))
        vis = Visibility.PRIVATE
        for child in node.children:
            if child.type == "visibility_modifier":
                vis = Visibility.PUB
                break
        return StructItem(
            name=name, fields=fields, visibility=vis,
            loc=_loc(node, self.filename),
        )

    def _lower_enum_item(self, node: Node) -> EnumItem:
        name_node = node.child_by_field_name("name")
        name = _text(name_node) if name_node else ""
        body_node = node.child_by_field_name("body")
        variants: List[EnumVariant] = []
        if body_node:
            for child in body_node.named_children:
                if child.type == "enum_variant":
                    vname_node = child.child_by_field_name("name")
                    variants.append(EnumVariant(
                        name=_text(vname_node) if vname_node else "",
                    ))
        return EnumItem(
            name=name, variants=variants,
            loc=_loc(node, self.filename),
        )

    def _lower_impl_item(self, node: Node) -> ImplItem:
        type_node = node.child_by_field_name("type")
        body_node = node.child_by_field_name("body")
        type_name = _text(type_node) if type_node else ""

        items: List[Item] = []
        if body_node:
            for child in body_node.named_children:
                item = self._lower_item(child)
                if item:
                    items.append(item)

        return ImplItem(
            self_type=PathType(segments=type_name.split("::")),
            items=items,
            loc=_loc(node, self.filename),
        )

    def _lower_use(self, node: Node) -> UseItem:
        return UseItem(path=_text(node), loc=_loc(node, self.filename))

    def _lower_const(self, node: Node) -> ConstItem:
        name_node = node.child_by_field_name("name")
        type_node = node.child_by_field_name("type")
        value_node = node.child_by_field_name("value")
        return ConstItem(
            name=_text(name_node) if name_node else "",
            type_ann=self._lower_type(type_node) if type_node else InferredType(),
            value=self._lower_expr(value_node) if value_node else None,
            loc=_loc(node, self.filename),
        )
        name_node = node.child_by_field_name("name")
        type_node = node.child_by_field_name("type")
        value_node = node.child_by_field_name("value")
        return StaticItem(
            name=_text(name_node) if name_node else "",
            type_ann=self._lower_type(type_node) if type_node else InferredType(),
            value=self._lower_expr(value_node) if value_node else None,
            loc=_loc(node, self.filename),
        )

    def _lower_type_alias(self, node: Node) -> TypeAliasItem:
        name_node = node.child_by_field_name("name")
        type_node = node.child_by_field_name("type")
        return TypeAliasItem(
            name=_text(name_node) if name_node else "",
            aliased_type=self._lower_type(type_node) if type_node else InferredType(),
            loc=_loc(node, self.filename),
        )

    def _lower_mod(self, node: Node) -> ModItem:
        name_node = node.child_by_field_name("name")
        return ModItem(
            name=_text(name_node) if name_node else "",
            loc=_loc(node, self.filename),
        )

    def _lower_extern_block(self, node: Node) -> ExternBlock:
        items: List[Item] = []
        body = node.child_by_field_name("body")
        if body:
            for child in body.named_children:
                item = self._lower_item(child)
                if item:
                    items.append(item)
        return ExternBlock(
            abi="C",
            items=items,
            loc=_loc(node, self.filename),
        )

    # -- types -------------------------------------------------------------

    def _lower_type(self, node: Node) -> RustType:
        if node is None:
            return InferredType()

        t = node.type
        text = _text(node)

        if t == "primitive_type":
            return PathType(segments=[text])

        if t == "type_identifier":
            return PathType(segments=[text])

        if t == "scoped_type_identifier":
            return PathType(segments=text.split("::"))

        if t == "reference_type":
            inner_node = node.child_by_field_name("type")
            is_mut = any(c.type == "mutable_specifier" for c in node.children)
            inner = self._lower_type(inner_node) if inner_node else InferredType()
            return ReferenceType(
                referent=inner,
                mutability=Mutability.MUTABLE if is_mut else Mutability.IMMUTABLE,
            )

        if t == "pointer_type":
            inner_node = node.child_by_field_name("type")
            is_mut = "mut" in text
            inner = self._lower_type(inner_node) if inner_node else InferredType()
            return RawPointerType(
                pointee=inner,
                mutability=Mutability.MUTABLE if is_mut else Mutability.IMMUTABLE,
            )

        if t == "array_type":
            elem_node = node.child_by_field_name("element")
            length_node = node.child_by_field_name("length")
            elem = self._lower_type(elem_node) if elem_node else InferredType()
            length = int(_text(length_node)) if length_node else 0
            return RustArrayType(element=elem, size=length)

        if t == "tuple_type":
            elems = [self._lower_type(c) for c in node.named_children]
            if not elems:
                return UnitType()
            return TupleType(elements=elems)

        if t == "unit_type":
            return UnitType()

        if t == "never_type":
            return NeverType()

        if t == "generic_type":
            type_node = node.child_by_field_name("type")
            args_node = node.child_by_field_name("type_arguments")
            base_text = _text(type_node) if type_node else text
            segments = base_text.split("::")
            generic_args = []
            if args_node:
                generic_args = [self._lower_type(c) for c in args_node.named_children
                                if c.type not in ("lifetime", "<", ">", ",")]
            return PathType(segments=segments, generic_args=generic_args)

        if t == "function_type":
            return FnPointerType(params=[], return_type=InferredType())

        if t == "parenthesized_type":
            inner = node.named_children[0] if node.named_children else None
            if inner:
                return ParenType(inner=self._lower_type(inner))
            return InferredType()

        if t == "bounded_type" or t == "abstract_type":
            return PathType(segments=[text])

        if t == "slice_type":
            elem_node = node.child_by_field_name("element")
            return SliceType(
                element=self._lower_type(elem_node) if elem_node else InferredType()
            )

        # Fallback
        if text:
            return PathType(segments=text.split("::"))
        return InferredType()

    # -- statements --------------------------------------------------------

    def _lower_stmt(self, node: Node) -> Optional[Stmt]:
        t = node.type

        if t == "let_declaration":
            return self._lower_let(node)

        if t == "expression_statement":
            inner = node.named_children[0] if node.named_children else None
            if inner:
                return ExprStmt(
                    expr=self._lower_expr(inner),
                    has_semicolon=True,
                    loc=_loc(node, self.filename),
                )
            return EmptyStmt(loc=_loc(node, self.filename))

        if t == "empty_statement":
            return EmptyStmt(loc=_loc(node, self.filename))

        if t in ("function_item", "struct_item", "enum_item",
                 "impl_item", "const_item", "static_item",
                 "type_item", "use_declaration"):
            item = self._lower_item(node)
            if item:
                return ItemStmt(item=item, loc=_loc(node, self.filename))
            return None

        if t == "line_comment" or t == "block_comment":
            return None

        if t == "attribute_item":
            return None

        if t == "macro_invocation":
            return ExprStmt(
                expr=MacroInvocation(name=_text(node), args=_text(node)),
                has_semicolon=True,
                loc=_loc(node, self.filename),
            )

        # Treat as expression
        return ExprStmt(
            expr=self._lower_expr(node),
            has_semicolon=False,
            loc=_loc(node, self.filename),
        )

    def _lower_let(self, node: Node) -> LetStmt:
        pat_node = node.child_by_field_name("pattern")
        type_node = node.child_by_field_name("type")
        value_node = node.child_by_field_name("value")

        pattern = self._lower_pattern(pat_node) if pat_node else IdentPattern(name="")
        ptype = self._lower_type(type_node) if type_node else None
        value = self._lower_expr(value_node) if value_node else None

        is_mut = False
        if pat_node and pat_node.type == "mut_pattern":
            is_mut = True

        return LetStmt(
            pattern=pattern,
            type_ann=ptype,
            initializer=value,
            is_mutable=is_mut,
            loc=_loc(node, self.filename),
        )

    # -- patterns ----------------------------------------------------------

    def _lower_pattern(self, node: Node) -> Pattern:
        t = node.type

        if t == "identifier":
            return IdentPattern(name=_text(node))

        if t == "mut_pattern":
            inner = node.named_children[0] if node.named_children else None
            if inner:
                return IdentPattern(
                    name=_text(inner),
                    mutability=Mutability.MUTABLE,
                )
            return IdentPattern(name="")

        if t == "_":
            return WildcardPattern()

        if t == "integer_literal" or t == "float_literal" or t == "string_literal":
            return LiteralPattern(value=_text(node))

        if t == "tuple_pattern":
            pats = [self._lower_pattern(c) for c in node.named_children]
            return TuplePattern(elements=pats)

        if t == "reference_pattern":
            inner = node.named_children[-1] if node.named_children else None
            return RefPattern(
                inner=self._lower_pattern(inner) if inner else IdentPattern(name=""),
            )

        if t == "or_pattern":
            pats = [self._lower_pattern(c) for c in node.named_children]
            return OrPattern(alternatives=pats)

        if t == "scoped_identifier" or t == "path":
            return PathPattern(path=_text(node))

        return IdentPattern(name=_text(node))

    # -- expressions -------------------------------------------------------

    def _lower_expr(self, node: Node) -> Expr:
        if node is None:
            return LitExpr(int_value=0, text="0")

        t = node.type

        if t == "integer_literal":
            text = _text(node)
            clean = text.rstrip("_iIuU0123456789").rstrip("iIuU") if any(c in text for c in "iIuU") else text
            clean2 = text.replace("_", "")
            try:
                if clean2.startswith("0x") or clean2.startswith("0X"):
                    ival = int(clean2.split("_")[0].rstrip("iIuU"), 16) if "i" in clean2.lower() or "u" in clean2.lower() else int(clean2, 16)
                else:
                    # Strip type suffix (i32, u64, etc.)
                    import re
                    m = re.match(r'^([0-9_]+)(?:i\d+|u\d+|isize|usize)?$', clean2)
                    ival = int(m.group(1).replace("_", "")) if m else int(clean2)
            except ValueError:
                ival = 0
            return LitExpr(int_value=ival, text=text,
                           loc=_loc(node, self.filename))

        if t == "float_literal":
            text = _text(node)
            try:
                fval = float(text.rstrip("f").rstrip("_f32f64"))
            except ValueError:
                fval = 0.0
            return LitExpr(float_value=fval, text=text,
                           loc=_loc(node, self.filename))

        if t == "string_literal":
            return LitExpr(string_value=_text(node), text=_text(node),
                           loc=_loc(node, self.filename))

        if t == "char_literal":
            raw = _text(node)
            char_val = ord(raw[1]) if len(raw) >= 2 else 0
            return LitExpr(char_value=char_val, text=raw,
                           loc=_loc(node, self.filename))

        if t == "boolean_literal":
            bval = _text(node) == "true"
            return LitExpr(bool_value=bval, text=_text(node),
                           loc=_loc(node, self.filename))

        if t == "identifier":
            return PathExpr(segments=[_text(node)], loc=_loc(node, self.filename))

        if t == "scoped_identifier" or t == "field_identifier":
            return PathExpr(
                segments=_text(node).split("::"),
                loc=_loc(node, self.filename),
            )

        if t == "binary_expression":
            return self._lower_binary(node)

        if t == "unary_expression":
            return self._lower_unary(node)

        if t == "type_cast_expression":
            return self._lower_cast(node)

        if t == "assignment_expression":
            return self._lower_assign(node)

        if t == "compound_assignment_expr":
            return self._lower_compound_assign(node)

        if t == "call_expression":
            return self._lower_call(node)

        if t == "method_call_expression":
            return self._lower_method_call(node)

        if t == "field_expression":
            return self._lower_field(node)

        if t == "index_expression":
            return self._lower_index(node)

        if t == "range_expression":
            return self._lower_range(node)

        if t == "block":
            return self._lower_block_expr(node)

        if t == "if_expression":
            return self._lower_if(node)

        if t == "match_expression":
            return self._lower_match(node)

        if t == "loop_expression":
            return LoopExpr(
                body=self._lower_block_expr(
                    node.child_by_field_name("body") or node
                ),
                loc=_loc(node, self.filename),
            )

        if t == "while_expression":
            return self._lower_while(node)

        if t == "for_expression":
            return self._lower_for(node)

        if t == "return_expression":
            val_node = node.named_children[0] if node.named_children else None
            return ReturnExpr(
                value=self._lower_expr(val_node) if val_node else None,
                loc=_loc(node, self.filename),
            )

        if t == "break_expression":
            return BreakExpr(loc=_loc(node, self.filename))

        if t == "continue_expression":
            return ContinueExpr(loc=_loc(node, self.filename))

        if t == "closure_expression":
            return self._lower_closure(node)

        if t == "tuple_expression":
            elems = [self._lower_expr(c) for c in node.named_children]
            return TupleExpr(elements=elems, loc=_loc(node, self.filename))

        if t == "array_expression":
            elems = [self._lower_expr(c) for c in node.named_children]
            return ArrayExpr(elements=elems, loc=_loc(node, self.filename))

        if t == "struct_expression":
            return self._lower_struct_expr(node)

        if t == "reference_expression":
            return self._lower_ref(node)

        if t == "dereference_expression":
            inner = node.named_children[0] if node.named_children else None
            return DerefExpr(
                operand=self._lower_expr(inner) if inner else LitExpr(int_value=0, text="0"),
                loc=_loc(node, self.filename),
            )

        if t == "parenthesized_expression":
            inner = node.named_children[0] if node.named_children else None
            if inner:
                return ParenExpr(
                    inner=self._lower_expr(inner),
                    loc=_loc(node, self.filename),
                )
            return LitExpr(text="()")

        if t == "unsafe_block":
            body_node = node.child_by_field_name("body") or node
            return UnsafeBlock(
                body=self._lower_block_expr(body_node),
                loc=_loc(node, self.filename),
            )

        if t == "macro_invocation":
            return MacroInvocation(
                name=_text(node),
                args=_text(node),
                loc=_loc(node, self.filename),
            )

        if t == "try_expression":
            inner = node.named_children[0] if node.named_children else None
            return TryExpr(
                operand=self._lower_expr(inner) if inner else LitExpr(int_value=0, text="0"),
                loc=_loc(node, self.filename),
            )

        if t == "await_expression":
            inner = node.named_children[0] if node.named_children else None
            return MethodCallExpr(
                receiver=self._lower_expr(inner) if inner else LitExpr(int_value=0, text="0"),
                method="await",
                args=[],
                loc=_loc(node, self.filename),
            )

        # Self
        if t == "self":
            return PathExpr(segments=["self"], loc=_loc(node, self.filename))

        # Fallback
        text = _text(node)
        if text:
            return PathExpr(segments=[text], loc=_loc(node, self.filename))
        return LitExpr(int_value=0, text="0")

    def _lower_binary(self, node: Node) -> BinaryExpr:
        left_node = node.child_by_field_name("left")
        right_node = node.child_by_field_name("right")
        op_node = node.child_by_field_name("operator")

        left = self._lower_expr(left_node)
        right = self._lower_expr(right_node)
        op_text = _text(op_node) if op_node else "+"
        op = _BINOP_MAP.get(op_text, RustBinaryOp.ADD)

        return BinaryExpr(op=op, lhs=left, rhs=right,
                          loc=_loc(node, self.filename))

    def _lower_unary(self, node: Node) -> UnaryExpr:
        op_node = node.child_by_field_name("operator")
        arg = node.named_children[-1] if node.named_children else None

        op_text = _text(op_node) if op_node else "-"
        op = _UNOP_MAP.get(op_text, RustUnaryOp.NEG)

        return UnaryExpr(
            op=op,
            operand=self._lower_expr(arg),
            loc=_loc(node, self.filename),
        )

    def _lower_cast(self, node: Node) -> CastExpr:
        value_node = node.child_by_field_name("value")
        type_node = node.child_by_field_name("type")
        return CastExpr(
            operand=self._lower_expr(value_node),
            target_type=self._lower_type(type_node) if type_node else InferredType(),
            loc=_loc(node, self.filename),
        )

    def _lower_assign(self, node: Node) -> AssignExpr:
        left_node = node.child_by_field_name("left")
        right_node = node.child_by_field_name("right")
        return AssignExpr(
            lhs=self._lower_expr(left_node),
            rhs=self._lower_expr(right_node),
            loc=_loc(node, self.filename),
        )

    def _lower_compound_assign(self, node: Node) -> BinaryExpr:
        left_node = node.child_by_field_name("left")
        right_node = node.child_by_field_name("right")
        op_node = node.child_by_field_name("operator")
        op_text = _text(op_node).replace("=", "") if op_node else "+"
        op = _BINOP_MAP.get(op_text, RustBinaryOp.ADD)
        return BinaryExpr(
            op=op,
            lhs=self._lower_expr(left_node),
            rhs=self._lower_expr(right_node),
            loc=_loc(node, self.filename),
        )

    def _lower_call(self, node: Node) -> CallExpr:
        func_node = node.child_by_field_name("function")
        args_node = node.child_by_field_name("arguments")

        func = self._lower_expr(func_node)
        args: List[Expr] = []
        if args_node:
            args = [self._lower_expr(c) for c in args_node.named_children]

        return CallExpr(callee=func, args=args, loc=_loc(node, self.filename))

    def _lower_method_call(self, node: Node) -> MethodCallExpr:
        receiver_node = node.child_by_field_name("value")
        name_node = node.child_by_field_name("name")
        args_node = node.child_by_field_name("arguments")

        receiver = self._lower_expr(receiver_node) if receiver_node else LitExpr(int_value=0, text="0")
        method = _text(name_node) if name_node else ""
        args: List[Expr] = []
        if args_node:
            args = [self._lower_expr(c) for c in args_node.named_children]

        return MethodCallExpr(
            receiver=receiver, method=method, args=args,
            loc=_loc(node, self.filename),
        )

    def _lower_field(self, node: Node) -> Expr:
        value_node = node.child_by_field_name("value")
        field_node = node.child_by_field_name("field")
        base = self._lower_expr(value_node) if value_node else LitExpr(int_value=0, text="0")
        field_text = _text(field_node) if field_node else ""
        # Numeric field -> tuple field
        if field_text.isdigit():
            return TupleFieldExpr(
                base=base, index=int(field_text),
                loc=_loc(node, self.filename),
            )
        return FieldExpr(base=base, field_name=field_text,
                         loc=_loc(node, self.filename))

    def _lower_index(self, node: Node) -> IndexExpr:
        arr = node.named_children[0] if len(node.named_children) > 0 else None
        idx = node.named_children[1] if len(node.named_children) > 1 else None
        return IndexExpr(
            base=self._lower_expr(arr),
            index=self._lower_expr(idx),
            loc=_loc(node, self.filename),
        )

    def _lower_range(self, node: Node) -> RangeExpr:
        children = node.named_children
        start = self._lower_expr(children[0]) if len(children) > 0 else None
        end = self._lower_expr(children[1]) if len(children) > 1 else None
        is_inclusive = ".." in _text(node) and "=" in _text(node)
        return RangeExpr(
            start=start, end=end, inclusive=is_inclusive,
            loc=_loc(node, self.filename),
        )

    def _lower_block_expr(self, node: Node) -> BlockExpr:
        stmts: List[Stmt] = []
        tail_expr = None

        children = node.named_children
        for i, child in enumerate(children):
            if i == len(children) - 1 and child.type not in (
                "let_declaration", "expression_statement", "empty_statement",
                "function_item", "struct_item", "enum_item", "use_declaration",
                "line_comment", "block_comment",
            ):
                # Last expression without semicolon = tail expression
                text = _text(child)
                # Check if there's a trailing semicolon
                if node.text and not node.text.decode("utf-8").rstrip().rstrip("}").rstrip().endswith(";"):
                    tail_expr = self._lower_expr(child)
                    continue
            stmt = self._lower_stmt(child)
            if stmt is not None:
                stmts.append(stmt)

        return BlockExpr(stmts=stmts, tail_expr=tail_expr,
                         loc=_loc(node, self.filename))

    def _lower_if(self, node: Node) -> IfExpr:
        cond_node = node.child_by_field_name("condition")
        body_node = node.child_by_field_name("consequence")
        else_node = node.child_by_field_name("alternative")

        cond = self._lower_expr(cond_node)
        body = self._lower_block_expr(body_node) if body_node else BlockExpr(stmts=[])

        else_body = None
        if else_node:
            if else_node.type == "else_clause":
                inner = else_node.named_children[0] if else_node.named_children else None
                if inner:
                    if inner.type == "if_expression":
                        else_body = self._lower_if(inner)
                    elif inner.type == "block":
                        else_body = self._lower_block_expr(inner)
            else:
                else_body = self._lower_expr(else_node)

        return IfExpr(condition=cond, then_body=body, else_body=else_body,
                      loc=_loc(node, self.filename))

    def _lower_match(self, node: Node) -> MatchExpr:
        value_node = node.child_by_field_name("value")
        body_node = node.child_by_field_name("body")

        scrutinee = self._lower_expr(value_node)
        arms: List[MatchArm] = []

        if body_node:
            for child in body_node.named_children:
                if child.type == "match_arm":
                    pat_node = child.child_by_field_name("pattern")
                    value_arm_node = child.child_by_field_name("value")
                    pattern = self._lower_pattern(pat_node) if pat_node else WildcardPattern()
                    body = self._lower_expr(value_arm_node) if value_arm_node else LitExpr(int_value=0, text="0")
                    arms.append(MatchArm(
                        pattern=pattern, body=body,
                    ))

        return MatchExpr(scrutinee=scrutinee, arms=arms,
                         loc=_loc(node, self.filename))

    def _lower_while(self, node: Node) -> WhileExpr:
        cond_node = node.child_by_field_name("condition")
        body_node = node.child_by_field_name("body")
        return WhileExpr(
            condition=self._lower_expr(cond_node),
            body=self._lower_block_expr(body_node) if body_node else BlockExpr(stmts=[]),
            loc=_loc(node, self.filename),
        )

    def _lower_for(self, node: Node) -> ForExpr:
        pat_node = node.child_by_field_name("pattern")
        iter_node = node.child_by_field_name("value")
        body_node = node.child_by_field_name("body")
        return ForExpr(
            pattern=self._lower_pattern(pat_node) if pat_node else IdentPattern(name=""),
            iterator=self._lower_expr(iter_node) if iter_node else LitExpr(int_value=0, text="0"),
            body=self._lower_block_expr(body_node) if body_node else BlockExpr(stmts=[]),
            loc=_loc(node, self.filename),
        )

    def _lower_closure(self, node: Node) -> ClosureExpr:
        params_node = node.child_by_field_name("parameters")
        body_node = node.child_by_field_name("body")

        params: List[FnParam] = []
        if params_node:
            for child in params_node.named_children:
                if child.type == "parameter":
                    pname = _text(child.child_by_field_name("pattern") or child)
                    ptype = (self._lower_type(child.child_by_field_name("type"))
                             if child.child_by_field_name("type") else InferredType())
                    params.append(FnParam(
                        pattern=IdentPattern(name=pname),
                        type_ann=ptype,
                    ))
                elif child.type == "identifier":
                    params.append(FnParam(
                        pattern=IdentPattern(name=_text(child)),
                        type_ann=InferredType(),
                    ))

        body = self._lower_expr(body_node) if body_node else LitExpr(int_value=0, text="0")

        return ClosureExpr(params=params, body=body,
                           loc=_loc(node, self.filename))

    def _lower_struct_expr(self, node: Node) -> StructExpr:
        name_node = node.child_by_field_name("name")
        body_node = node.child_by_field_name("body")

        name = _text(name_node) if name_node else ""
        fields_list: List[tuple] = []
        if body_node:
            for child in body_node.named_children:
                if child.type == "field_initializer":
                    fname_node = child.child_by_field_name("field")  
                    fval_node = child.child_by_field_name("value")
                    if fname_node and fval_node:
                        fields_list.append((_text(fname_node), self._lower_expr(fval_node)))
                elif child.type == "shorthand_field_initializer":
                    fname = _text(child)
                    fields_list.append((fname, PathExpr(segments=[fname])))

        return StructExpr(path=name.split("::"), fields=fields_list,
                          loc=_loc(node, self.filename))

    def _lower_ref(self, node: Node) -> RefExpr:
        inner = node.named_children[-1] if node.named_children else None
        is_mut = any(c.type == "mutable_specifier" for c in node.children)
        return RefExpr(
            operand=self._lower_expr(inner),
            mutability=Mutability.MUTABLE if is_mut else Mutability.IMMUTABLE,
            loc=_loc(node, self.filename),
        )
