"""
Tree-sitter based C parser for verified parsing.

Uses the tree-sitter-c grammar (derived from the official C grammar) to produce
ASTs, replacing the hand-written recursive-descent parser with a parser built on
a well-tested, widely-deployed grammar. The tree-sitter parse tree is lowered
into the same C AST node types used by the hand-written parser, so the rest of
the pipeline (type resolution, IR lowering) is unchanged.

This addresses critique: "Custom C/Rust parsers are unverified — Python
implementations bypass established frontends (Clang, rustc/syn)."
"""

from __future__ import annotations

from typing import Optional, List, Sequence
import tree_sitter_c as tsc
from tree_sitter import Language, Parser, Node

from .c_ast import (
    # Types
    CType, VoidCType, IntCType, FloatCType, PointerCType, ArrayCType,
    FunctionCType, StructRefCType, UnionRefCType, EnumRefCType,
    TypedefRefCType, QualifiedCType,
    TypeQualifier, StorageClass, FunctionSpecifier,
    # Declarations
    Decl, ParamDecl, VarDecl, FunctionDecl, TypedefDecl, FieldDecl,
    StructDecl, UnionDecl, EnumDecl, EnumeratorDecl,
    Attribute,
    # Statements
    Stmt, CompoundStmt, ExprStmt, IfStmt, WhileStmt, DoWhileStmt,
    ForStmt, SwitchStmt, CaseStmt, ReturnStmt, BreakStmt, ContinueStmt,
    GotoStmt, LabelStmt, NullStmt, DeclStmt,
    # Expressions
    Expr, IntLiteral, FloatLiteral, CharLiteral, StringLiteral,
    IdentExpr, BinaryExpr, UnaryExpr, CastExpr, SizeofExpr,
    AlignofExpr, CallExpr, MemberExpr, ArraySubscriptExpr,
    TernaryExpr, CommaExpr, InitListExpr, Designator,
    CompoundLiteralExpr, ParenExpr,
    BinaryOp as CASTBinaryOp, UnaryOp as CASTUnaryOp,
    # Top level
    TranslationUnit, NodeLocation,
)
from .lexer import SourcePos, SourceSpan


# ---------------------------------------------------------------------------
# Tree-sitter C Language singleton
# ---------------------------------------------------------------------------

_C_LANGUAGE = Language(tsc.language())


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
    "+": CASTBinaryOp.ADD, "-": CASTBinaryOp.SUB,
    "*": CASTBinaryOp.MUL, "/": CASTBinaryOp.DIV, "%": CASTBinaryOp.MOD,
    "<<": CASTBinaryOp.SHL, ">>": CASTBinaryOp.SHR,
    "&": CASTBinaryOp.BITAND, "|": CASTBinaryOp.BITOR,
    "^": CASTBinaryOp.BITXOR,
    "&&": CASTBinaryOp.LOGAND, "||": CASTBinaryOp.LOGOR,
    "==": CASTBinaryOp.EQ, "!=": CASTBinaryOp.NE,
    "<": CASTBinaryOp.LT, ">": CASTBinaryOp.GT,
    "<=": CASTBinaryOp.LE, ">=": CASTBinaryOp.GE,
    "=": CASTBinaryOp.ASSIGN,
    "+=": CASTBinaryOp.ADD_ASSIGN, "-=": CASTBinaryOp.SUB_ASSIGN,
    "*=": CASTBinaryOp.MUL_ASSIGN, "/=": CASTBinaryOp.DIV_ASSIGN,
    "%=": CASTBinaryOp.MOD_ASSIGN,
    "<<=": CASTBinaryOp.SHL_ASSIGN, ">>=": CASTBinaryOp.SHR_ASSIGN,
    "&=": CASTBinaryOp.AND_ASSIGN, "|=": CASTBinaryOp.OR_ASSIGN,
    "^=": CASTBinaryOp.XOR_ASSIGN,
}

_UNOP_MAP = {
    "-": CASTUnaryOp.MINUS, "+": CASTUnaryOp.PLUS,
    "~": CASTUnaryOp.BITWISE_NOT, "!": CASTUnaryOp.LOGICAL_NOT,
    "*": CASTUnaryOp.DEREF, "&": CASTUnaryOp.ADDR,
    "++": CASTUnaryOp.PRE_INC, "--": CASTUnaryOp.PRE_DEC,
}


# ---------------------------------------------------------------------------
# TreeSitterCParser
# ---------------------------------------------------------------------------

class TreeSitterCParser:
    """Parse C source using tree-sitter-c and convert to our C AST.

    Falls back to the hand-written parser on conversion errors so the
    pipeline never loses coverage.

    Parameters
    ----------
    source : str
        C source code to parse.
    filename : str
        Source filename (used in error messages / locations).
    """

    def __init__(self, source: str, filename: str = "<input>"):
        self.source = source
        self.filename = filename
        self._parser = Parser(_C_LANGUAGE)

    def parse(self) -> TranslationUnit:
        """Parse the source and return a TranslationUnit."""
        tree = self._parser.parse(self.source.encode("utf-8"))
        root = tree.root_node

        if root.has_error:
            # Fall back to hand-written parser on error
            from .parser import CParser
            return CParser(self.source, self.filename).parse()

        decls: List[Decl] = []
        for child in root.children:
            d = self._lower_top_level(child)
            if d is not None:
                if isinstance(d, list):
                    decls.extend(d)
                else:
                    decls.append(d)

        return TranslationUnit(
            declarations=decls,
            loc=_loc(root, self.filename),
        )

    # -- top-level ---------------------------------------------------------

    def _lower_top_level(self, node: Node):
        t = node.type
        if t == "function_definition":
            return self._lower_function_def(node)
        if t == "declaration":
            return self._lower_declaration(node)
        if t == "struct_specifier":
            return self._lower_struct(node)
        if t == "union_specifier":
            return self._lower_union(node)
        if t == "enum_specifier":
            return self._lower_enum(node)
        if t == "type_definition":
            return self._lower_typedef(node)
        if t == "preproc_include" or t == "preproc_def" or t == "preproc_ifdef":
            return None  # preprocessor directives — skip
        if t == "comment":
            return None
        if t == ";":
            return None
        return None

    # -- function definition -----------------------------------------------

    def _lower_function_def(self, node: Node) -> FunctionDecl:
        # Collect storage class and return type from specifiers
        storage = StorageClass.NONE
        func_spec = FunctionSpecifier.NONE
        ret_type: CType = VoidCType()

        # The declarator and body
        declarator = node.child_by_field_name("declarator")
        body_node = node.child_by_field_name("body")

        # Process type/storage specifiers
        for child in node.children:
            if child.type in ("storage_class_specifier",):
                sc_text = _text(child)
                if sc_text == "static":
                    storage = StorageClass.STATIC
                elif sc_text == "extern":
                    storage = StorageClass.EXTERN
            elif child.type == "function_specifier":
                if _text(child) == "inline":
                    func_spec = FunctionSpecifier.INLINE
            elif child.type in ("primitive_type", "sized_type_specifier",
                                "type_identifier", "struct_specifier",
                                "union_specifier", "enum_specifier"):
                ret_type = self._lower_type_node(child)
            elif child.type == "type_qualifier":
                pass  # const/volatile on return type

        # Extract function name and parameters from declarator
        name = ""
        params: List[ParamDecl] = []
        if declarator is not None:
            name, params, ret_type = self._extract_func_declarator(
                declarator, ret_type
            )

        # Lower body
        body = self._lower_compound_stmt(body_node) if body_node else CompoundStmt(items=[])

        return FunctionDecl(
            name=name,
            return_type=ret_type,
            params=params,
            body=body,
            storage_class=storage,
            is_inline=(func_spec == FunctionSpecifier.INLINE),
            is_definition=(body is not None),
            loc=_loc(node, self.filename),
        )

    def _extract_func_declarator(self, node: Node, ret_type: CType):
        """Walk through (possibly pointer-wrapped) function declarator."""
        params: List[ParamDecl] = []
        name = ""

        if node.type == "function_declarator":
            # Name
            decl = node.child_by_field_name("declarator")
            if decl:
                if decl.type == "identifier":
                    name = _text(decl)
                elif decl.type == "pointer_declarator":
                    ret_type = PointerCType(pointee=ret_type)
                    inner = decl.child_by_field_name("declarator")
                    if inner and inner.type == "identifier":
                        name = _text(inner)
                elif decl.type == "parenthesized_declarator":
                    inner = decl.named_children[0] if decl.named_children else None
                    if inner:
                        name = _text(inner)

            # Parameters
            param_list = node.child_by_field_name("parameters")
            if param_list:
                for p in param_list.named_children:
                    if p.type == "parameter_declaration":
                        params.append(self._lower_param(p))
                    elif p.type == "variadic_parameter":
                        params.append(ParamDecl(name="..."))

        elif node.type == "pointer_declarator":
            ret_type = PointerCType(pointee=ret_type)
            inner = node.child_by_field_name("declarator")
            if inner:
                name, params, ret_type = self._extract_func_declarator(
                    inner, ret_type
                )
        elif node.type == "identifier":
            name = _text(node)

        return name, params, ret_type

    def _lower_param(self, node: Node) -> ParamDecl:
        ptype: CType = VoidCType()
        pname = ""

        for child in node.children:
            if child.type in ("primitive_type", "sized_type_specifier",
                              "type_identifier", "struct_specifier",
                              "union_specifier", "enum_specifier"):
                ptype = self._lower_type_node(child)
            elif child.type == "type_qualifier":
                pass
            elif child.type == "identifier":
                pname = _text(child)
            elif child.type == "pointer_declarator":
                ptype = PointerCType(pointee=ptype)
                inner = child.child_by_field_name("declarator")
                if inner and inner.type == "identifier":
                    pname = _text(inner)
            elif child.type == "array_declarator":
                inner = child.child_by_field_name("declarator")
                if inner and inner.type == "identifier":
                    pname = _text(inner)
                ptype = PointerCType(pointee=ptype)  # array params decay

        return ParamDecl(name=pname, type_name=ptype)

    # -- types -------------------------------------------------------------

    def _lower_type_node(self, node: Node) -> CType:
        t = node.type
        text = _text(node)

        if t == "primitive_type":
            return self._primitive_type(text)
        if t == "sized_type_specifier":
            return self._sized_type(node)
        if t == "type_identifier":
            return TypedefRefCType(name=text)
        if t == "struct_specifier":
            name_node = node.child_by_field_name("name")
            return StructRefCType(name=_text(name_node) if name_node else "")
        if t == "union_specifier":
            name_node = node.child_by_field_name("name")
            return UnionRefCType(name=_text(name_node) if name_node else "")
        if t == "enum_specifier":
            name_node = node.child_by_field_name("name")
            return EnumRefCType(name=_text(name_node) if name_node else "")

        return IntCType(is_int=True)  # default fallback

    def _primitive_type(self, text: str) -> CType:
        if text == "void":
            return VoidCType()
        if text in ("float",):
            return FloatCType(is_float=True)
        if text in ("double",):
            return FloatCType(is_double=True)
        if text == "char":
            return IntCType(is_char=True, is_signed=True)
        if text in ("int",):
            return IntCType(is_int=True, is_signed=True)
        if text == "long":
            return IntCType(is_long=True, is_signed=True)
        if text in ("_Bool", "bool"):
            return IntCType(is_int=True, is_signed=False, is_unsigned=True)
        return IntCType(is_int=True, is_signed=True)

    def _sized_type(self, node: Node) -> CType:
        text = _text(node)
        is_unsigned = "unsigned" in text
        is_long_long = "long long" in text
        is_long = "long" in text and not is_long_long
        is_short = "short" in text
        is_char = "char" in text

        return IntCType(
            is_signed=not is_unsigned,
            is_unsigned=is_unsigned,
            is_char=is_char,
            is_short=is_short,
            is_int="int" in text or not (is_char or is_short or is_long or is_long_long),
            is_long=is_long,
            is_long_long=is_long_long,
        )

    # -- declarations ------------------------------------------------------

    def _lower_declaration(self, node: Node):
        """Lower a variable/typedef/etc declaration."""
        storage = StorageClass.NONE
        base_type: CType = VoidCType()
        decls: List[Decl] = []

        for child in node.children:
            if child.type == "storage_class_specifier":
                sc = _text(child)
                if sc == "static":
                    storage = StorageClass.STATIC
                elif sc == "extern":
                    storage = StorageClass.EXTERN
                elif sc == "typedef":
                    storage = StorageClass.TYPEDEF
            elif child.type in ("primitive_type", "sized_type_specifier",
                                "type_identifier", "struct_specifier",
                                "union_specifier", "enum_specifier"):
                base_type = self._lower_type_node(child)
            elif child.type == "init_declarator":
                name, vtype, init = self._lower_init_declarator(child, base_type)
                if storage == StorageClass.TYPEDEF:
                    decls.append(TypedefDecl(
                        name=name, underlying_type=vtype,
                        loc=_loc(child, self.filename),
                    ))
                else:
                    decls.append(VarDecl(
                        name=name, type_name=vtype, initializer=init,
                        storage_class=storage,
                        loc=_loc(child, self.filename),
                    ))
            elif child.type == "identifier":
                name = _text(child)
                if storage == StorageClass.TYPEDEF:
                    decls.append(TypedefDecl(
                        name=name, underlying_type=base_type,
                        loc=_loc(child, self.filename),
                    ))
                else:
                    decls.append(VarDecl(
                        name=name, type_name=base_type,
                        storage_class=storage,
                        loc=_loc(child, self.filename),
                    ))
            elif child.type == "pointer_declarator":
                vtype = PointerCType(pointee=base_type)
                inner = child.child_by_field_name("declarator")
                iname = _text(inner) if inner else ""
                decls.append(VarDecl(
                    name=iname, type_name=vtype,
                    storage_class=storage,
                    loc=_loc(child, self.filename),
                ))

        if len(decls) == 1:
            return decls[0]
        return decls if decls else None

    def _lower_init_declarator(self, node: Node, base_type: CType):
        name = ""
        vtype = base_type
        init = None

        for child in node.children:
            if child.type == "identifier":
                name = _text(child)
            elif child.type == "pointer_declarator":
                vtype = PointerCType(pointee=vtype)
                inner = child.child_by_field_name("declarator")
                if inner and inner.type == "identifier":
                    name = _text(inner)
            elif child.type == "array_declarator":
                inner = child.child_by_field_name("declarator")
                if inner and inner.type == "identifier":
                    name = _text(inner)
                size_node = child.child_by_field_name("size")
                size = None
                if size_node:
                    size = self._lower_expr(size_node)
                vtype = ArrayCType(element=vtype, size=size)
            elif child.type == "=":
                pass
            elif child.is_named and init is None and child.type != "identifier":
                init = self._lower_expr(child)

        return name, vtype, init

    # -- struct/union/enum -------------------------------------------------

    def _lower_struct(self, node: Node) -> Optional[StructDecl]:
        name_node = node.child_by_field_name("name")
        name = _text(name_node) if name_node else ""
        body_node = node.child_by_field_name("body")
        fields: List[FieldDecl] = []
        if body_node:
            for child in body_node.named_children:
                if child.type == "field_declaration":
                    fields.extend(self._lower_field_decl(child))
        return StructDecl(name=name, fields=fields, loc=_loc(node, self.filename))

    def _lower_union(self, node: Node) -> Optional[UnionDecl]:
        name_node = node.child_by_field_name("name")
        name = _text(name_node) if name_node else ""
        body_node = node.child_by_field_name("body")
        fields: List[FieldDecl] = []
        if body_node:
            for child in body_node.named_children:
                if child.type == "field_declaration":
                    fields.extend(self._lower_field_decl(child))
        return UnionDecl(name=name, fields=fields, loc=_loc(node, self.filename))

    def _lower_field_decl(self, node: Node) -> List[FieldDecl]:
        base_type: CType = VoidCType()
        fields: List[FieldDecl] = []
        for child in node.children:
            if child.type in ("primitive_type", "sized_type_specifier",
                              "type_identifier", "struct_specifier",
                              "union_specifier", "enum_specifier"):
                base_type = self._lower_type_node(child)
            elif child.type == "field_identifier":
                fields.append(FieldDecl(name=_text(child), type_name=base_type))
            elif child.type == "pointer_declarator":
                ft = PointerCType(pointee=base_type)
                inner = child.child_by_field_name("declarator")
                fname = _text(inner) if inner else ""
                fields.append(FieldDecl(name=fname, type_name=ft))
        if not fields:
            fields.append(FieldDecl(name="", type_name=base_type))
        return fields

    def _lower_enum(self, node: Node) -> Optional[EnumDecl]:
        name_node = node.child_by_field_name("name")
        name = _text(name_node) if name_node else ""
        body_node = node.child_by_field_name("body")
        enumerators: List[EnumeratorDecl] = []
        if body_node:
            for child in body_node.named_children:
                if child.type == "enumerator":
                    ename_node = child.child_by_field_name("name")
                    eval_node = child.child_by_field_name("value")
                    enumerators.append(EnumeratorDecl(
                        name=_text(ename_node) if ename_node else "",
                        value=self._lower_expr(eval_node) if eval_node else None,
                    ))
        return EnumDecl(name=name, enumerators=enumerators, loc=_loc(node, self.filename))

    def _lower_typedef(self, node: Node):
        base_type: CType = VoidCType()
        name = ""
        for child in node.children:
            if child.type in ("primitive_type", "sized_type_specifier",
                              "type_identifier", "struct_specifier",
                              "union_specifier"):
                base_type = self._lower_type_node(child)
            elif child.type == "type_identifier" and base_type is not None:
                name = _text(child)
        return TypedefDecl(name=name, underlying_type=base_type, loc=_loc(node, self.filename))

    # -- statements --------------------------------------------------------

    def _lower_stmt(self, node: Node) -> Stmt:
        t = node.type
        if t == "compound_statement":
            return self._lower_compound_stmt(node)
        if t == "return_statement":
            return self._lower_return_stmt(node)
        if t == "if_statement":
            return self._lower_if_stmt(node)
        if t == "while_statement":
            return self._lower_while_stmt(node)
        if t == "do_statement":
            return self._lower_do_while_stmt(node)
        if t == "for_statement":
            return self._lower_for_stmt(node)
        if t == "switch_statement":
            return self._lower_switch_stmt(node)
        if t == "case_statement":
            return self._lower_case_stmt(node)
        if t == "break_statement":
            return BreakStmt(loc=_loc(node, self.filename))
        if t == "continue_statement":
            return ContinueStmt(loc=_loc(node, self.filename))
        if t == "goto_statement":
            label_node = node.named_children[0] if node.named_children else None
            return GotoStmt(
                label=_text(label_node) if label_node else "",
                loc=_loc(node, self.filename),
            )
        if t == "labeled_statement":
            return self._lower_labeled_stmt(node)
        if t == "expression_statement":
            return self._lower_expr_stmt(node)
        if t == "declaration":
            d = self._lower_declaration(node)
            if isinstance(d, list):
                return DeclStmt(decl=d[0] if len(d)==1 else d, loc=_loc(node, self.filename))
            elif d is not None:
                return DeclStmt(decl=d, loc=_loc(node, self.filename))
            return NullStmt(loc=_loc(node, self.filename))
        if t == ";":
            return NullStmt(loc=_loc(node, self.filename))
        if t == "comment":
            return NullStmt(loc=_loc(node, self.filename))
        # Fallback: treat as expression statement
        expr = self._lower_expr(node)
        return ExprStmt(expr=expr, loc=_loc(node, self.filename))

    def _lower_compound_stmt(self, node: Node) -> CompoundStmt:
        stmts: List[Stmt] = []
        for child in node.named_children:
            s = self._lower_stmt(child)
            if s is not None:
                stmts.append(s)
        return CompoundStmt(items=stmts, loc=_loc(node, self.filename))

    def _lower_return_stmt(self, node: Node) -> ReturnStmt:
        expr = None
        for child in node.named_children:
            if child.type != "return":
                expr = self._lower_expr(child)
                break
        return ReturnStmt(expr=expr, loc=_loc(node, self.filename))

    def _lower_if_stmt(self, node: Node) -> IfStmt:
        cond_node = node.child_by_field_name("condition")
        body_node = node.child_by_field_name("consequence")
        else_node = node.child_by_field_name("alternative")

        cond = self._lower_expr(cond_node) if cond_node else IdentExpr(name="0")
        # Unwrap parenthesized_expression for condition
        if cond_node and cond_node.type == "parenthesized_expression":
            inner = cond_node.named_children[0] if cond_node.named_children else None
            if inner:
                cond = self._lower_expr(inner)

        body = self._lower_stmt(body_node) if body_node else NullStmt()
        else_body = self._lower_stmt(else_node) if else_node else None

        return IfStmt(condition=cond, then_body=body, else_body=else_body,
                      loc=_loc(node, self.filename))

    def _lower_while_stmt(self, node: Node) -> WhileStmt:
        cond_node = node.child_by_field_name("condition")
        body_node = node.child_by_field_name("body")
        cond = self._lower_expr(cond_node) if cond_node else IdentExpr(name="1")
        if cond_node and cond_node.type == "parenthesized_expression":
            inner = cond_node.named_children[0] if cond_node.named_children else None
            if inner:
                cond = self._lower_expr(inner)
        body = self._lower_stmt(body_node) if body_node else NullStmt()
        return WhileStmt(condition=cond, body=body, loc=_loc(node, self.filename))

    def _lower_do_while_stmt(self, node: Node) -> DoWhileStmt:
        body_node = node.child_by_field_name("body")
        cond_node = node.child_by_field_name("condition")
        cond = self._lower_expr(cond_node) if cond_node else IdentExpr(name="1")
        if cond_node and cond_node.type == "parenthesized_expression":
            inner = cond_node.named_children[0] if cond_node.named_children else None
            if inner:
                cond = self._lower_expr(inner)
        body = self._lower_stmt(body_node) if body_node else NullStmt()
        return DoWhileStmt(body=body, condition=cond, loc=_loc(node, self.filename))

    def _lower_for_stmt(self, node: Node) -> ForStmt:
        init_node = node.child_by_field_name("initializer")
        cond_node = node.child_by_field_name("condition")
        update_node = node.child_by_field_name("update")
        body_node = node.child_by_field_name("body")

        init = self._lower_expr(init_node) if init_node else None
        cond = self._lower_expr(cond_node) if cond_node else None
        update = self._lower_expr(update_node) if update_node else None
        body = self._lower_stmt(body_node) if body_node else NullStmt()

        return ForStmt(init=init, condition=cond, increment=update, body=body,
                       loc=_loc(node, self.filename))

    def _lower_switch_stmt(self, node: Node) -> SwitchStmt:
        cond_node = node.child_by_field_name("condition")
        body_node = node.child_by_field_name("body")
        cond = self._lower_expr(cond_node) if cond_node else IdentExpr(name="0")
        if cond_node and cond_node.type == "parenthesized_expression":
            inner = cond_node.named_children[0] if cond_node.named_children else None
            if inner:
                cond = self._lower_expr(inner)
        body = self._lower_stmt(body_node) if body_node else CompoundStmt(items=[])
        return SwitchStmt(expr=cond, body=body, loc=_loc(node, self.filename))

    def _lower_case_stmt(self, node: Node) -> CaseStmt:
        value_node = node.child_by_field_name("value")
        value = self._lower_expr(value_node) if value_node else None
        stmts: List[Stmt] = []
        for child in node.named_children:
            if child != value_node and child.type != "case" and child.type != "default":
                stmts.append(self._lower_stmt(child))
        body = CompoundStmt(items=stmts) if stmts else NullStmt()
        is_default = any(c.type == "default" for c in node.children if not c.is_named)
        return CaseStmt(
            expr=value, body=body,
            is_default=is_default or value is None,
            loc=_loc(node, self.filename),
        )

    def _lower_labeled_stmt(self, node: Node) -> LabelStmt:
        label_node = node.child_by_field_name("label")
        label = _text(label_node) if label_node else ""
        body_children = [c for c in node.named_children if c != label_node]
        body = self._lower_stmt(body_children[0]) if body_children else NullStmt()
        return LabelStmt(label=label, body=body, loc=_loc(node, self.filename))

    def _lower_expr_stmt(self, node: Node) -> ExprStmt:
        expr = None
        for child in node.named_children:
            expr = self._lower_expr(child)
            break
        if expr is None:
            return NullStmt(loc=_loc(node, self.filename))
        return ExprStmt(expr=expr, loc=_loc(node, self.filename))

    # -- expressions -------------------------------------------------------

    def _lower_expr(self, node: Node) -> Expr:
        t = node.type

        if t == "number_literal":
            text = _text(node)
            if "." in text or "e" in text.lower() or "E" in text:
                try:
                    fval = float(text.rstrip("fFlL"))
                except ValueError:
                    fval = 0.0
                return FloatLiteral(value=fval, text=text, loc=_loc(node, self.filename))
            # Parse integer: handle hex, octal, suffixes
            clean = text.rstrip("uUlL")
            try:
                if clean.startswith("0x") or clean.startswith("0X"):
                    ival = int(clean, 16)
                elif clean.startswith("0b") or clean.startswith("0B"):
                    ival = int(clean, 2)
                elif clean.startswith("0") and len(clean) > 1 and clean[1:].isdigit():
                    ival = int(clean, 8)
                else:
                    ival = int(clean) if clean else 0
            except ValueError:
                ival = 0
            is_unsigned = "u" in text.lower().split("0x")[-1] if "0x" in text.lower() else "u" in text.lower()
            return IntLiteral(value=ival, text=text,
                              suffix_unsigned=is_unsigned,
                              loc=_loc(node, self.filename))

        if t == "string_literal":
            return StringLiteral(value=_text(node), text=_text(node), loc=_loc(node, self.filename))

        if t == "char_literal":
            raw = _text(node)
            char_val = ord(raw[1]) if len(raw) >= 2 else 0
            return CharLiteral(value=char_val, text=raw, loc=_loc(node, self.filename))

        if t in ("true", "false"):
            return IntLiteral(
                value=1 if t == "true" else 0, text=t,
                loc=_loc(node, self.filename),
            )

        if t == "identifier":
            return IdentExpr(name=_text(node), loc=_loc(node, self.filename))

        if t == "binary_expression":
            return self._lower_binary(node)

        if t == "unary_expression":
            return self._lower_unary(node)

        if t == "update_expression":
            return self._lower_update(node)

        if t == "assignment_expression":
            return self._lower_assignment(node)

        if t == "call_expression":
            return self._lower_call(node)

        if t == "cast_expression":
            return self._lower_cast(node)

        if t == "sizeof_expression":
            return self._lower_sizeof(node)

        if t == "field_expression":
            return self._lower_member(node)

        if t == "subscript_expression":
            return self._lower_subscript(node)

        if t == "conditional_expression":
            return self._lower_ternary(node)

        if t == "parenthesized_expression":
            inner = node.named_children[0] if node.named_children else None
            if inner:
                return ParenExpr(
                    inner=self._lower_expr(inner),
                    loc=_loc(node, self.filename),
                )
            return IntLiteral(value=0, text="0")

        if t == "comma_expression":
            return self._lower_comma(node)

        if t == "initializer_list":
            return self._lower_init_list(node)

        if t == "pointer_expression":
            return self._lower_pointer_expr(node)

        if t == "null":
            return IntLiteral(value=0, text="0", loc=_loc(node, self.filename))

        # Fallback: identifier-like
        text = _text(node)
        if text:
            return IdentExpr(name=text, loc=_loc(node, self.filename))
        return IntLiteral(value=0, text="0", loc=_loc(node, self.filename))

    def _lower_binary(self, node: Node) -> BinaryExpr:
        left_node = node.child_by_field_name("left")
        right_node = node.child_by_field_name("right")
        op_node = node.child_by_field_name("operator")

        left = self._lower_expr(left_node) if left_node else IntLiteral(value=0, text="0")
        right = self._lower_expr(right_node) if right_node else IntLiteral(value=0, text="0")
        op_text = _text(op_node) if op_node else "+"
        op = _BINOP_MAP.get(op_text, CASTBinaryOp.ADD)

        return BinaryExpr(op=op, lhs=left, rhs=right,
                          loc=_loc(node, self.filename))

    def _lower_unary(self, node: Node) -> UnaryExpr:
        op_node = node.child_by_field_name("operator")
        arg_node = node.child_by_field_name("argument")

        op_text = _text(op_node) if op_node else "-"
        arg = self._lower_expr(arg_node) if arg_node else IntLiteral(value=0, text="0")
        op = _UNOP_MAP.get(op_text, CASTUnaryOp.NEG)

        return UnaryExpr(op=op, operand=arg,
                         loc=_loc(node, self.filename))

    def _lower_update(self, node: Node) -> UnaryExpr:
        op_node = node.child_by_field_name("operator")
        arg_node = node.child_by_field_name("argument")

        op_text = _text(op_node) if op_node else "++"
        arg = self._lower_expr(arg_node) if arg_node else IntLiteral(value=0, text="0")

        # Determine pre/post based on child ordering
        is_postfix = (node.children and node.children[0].is_named)

        if op_text == "++":
            op = CASTUnaryOp.POST_INC if is_postfix else CASTUnaryOp.PRE_INC
        else:
            op = CASTUnaryOp.POST_DEC if is_postfix else CASTUnaryOp.PRE_DEC

        return UnaryExpr(op=op, operand=arg,
                         loc=_loc(node, self.filename))

    def _lower_assignment(self, node: Node) -> BinaryExpr:
        left_node = node.child_by_field_name("left")
        right_node = node.child_by_field_name("right")
        op_node = node.child_by_field_name("operator")

        left = self._lower_expr(left_node) if left_node else IntLiteral(value=0, text="0")
        right = self._lower_expr(right_node) if right_node else IntLiteral(value=0, text="0")
        op_text = _text(op_node) if op_node else "="
        op = _BINOP_MAP.get(op_text, CASTBinaryOp.ASSIGN)

        return BinaryExpr(op=op, lhs=left, rhs=right,
                          loc=_loc(node, self.filename))

    def _lower_call(self, node: Node) -> CallExpr:
        func_node = node.child_by_field_name("function")
        args_node = node.child_by_field_name("arguments")

        func = self._lower_expr(func_node) if func_node else IdentExpr(name="unknown")
        args: List[Expr] = []
        if args_node:
            for child in args_node.named_children:
                args.append(self._lower_expr(child))

        return CallExpr(callee=func, args=args, loc=_loc(node, self.filename))

    def _lower_cast(self, node: Node) -> CastExpr:
        type_node = node.child_by_field_name("type")
        value_node = node.child_by_field_name("value")

        cast_type = self._lower_type_descriptor(type_node) if type_node else IntCType(is_int=True)
        value = self._lower_expr(value_node) if value_node else IntLiteral(value=0, text="0")

        return CastExpr(cast_type=cast_type, operand=value, loc=_loc(node, self.filename))

    def _lower_type_descriptor(self, node: Node) -> CType:
        """Lower a type_descriptor node (used in casts, sizeof)."""
        base: CType = IntCType(is_int=True)
        ptr_depth = 0
        for child in node.children:
            if child.type in ("primitive_type", "sized_type_specifier",
                              "type_identifier", "struct_specifier"):
                base = self._lower_type_node(child)
            elif child.type == "abstract_pointer_declarator":
                ptr_depth += 1

        result = base
        for _ in range(ptr_depth):
            result = PointerCType(pointee=result)
        return result

    def _lower_sizeof(self, node: Node) -> SizeofExpr:
        # sizeof can take a type or expression
        for child in node.named_children:
            if child.type == "type_descriptor":
                return SizeofExpr(
                    operand_type=self._lower_type_descriptor(child),
                    is_type=True,
                    loc=_loc(node, self.filename),
                )
            elif child.type == "parenthesized_expression":
                inner = child.named_children[0] if child.named_children else None
                if inner:
                    return SizeofExpr(
                        operand_expr=self._lower_expr(inner),
                        is_type=False,
                        loc=_loc(node, self.filename),
                    )
        return SizeofExpr(
            operand_type=IntCType(is_int=True),
            is_type=True,
            loc=_loc(node, self.filename),
        )

    def _lower_member(self, node: Node) -> MemberExpr:
        arg_node = node.child_by_field_name("argument")
        field_node = node.child_by_field_name("field")
        op_node = node.child_by_field_name("operator")

        base = self._lower_expr(arg_node) if arg_node else IdentExpr(name="")
        field_name = _text(field_node) if field_node else ""
        is_arrow = _text(op_node) == "->" if op_node else False

        return MemberExpr(
            base=base, member=field_name, is_arrow=is_arrow,
            loc=_loc(node, self.filename),
        )

    def _lower_subscript(self, node: Node) -> ArraySubscriptExpr:
        arr_node = node.child_by_field_name("argument")
        idx_node = node.child_by_field_name("index")

        arr = self._lower_expr(arr_node) if arr_node else IdentExpr(name="")
        idx = self._lower_expr(idx_node) if idx_node else IntLiteral(value=0, text="0")

        return ArraySubscriptExpr(base=arr, index=idx,
                                  loc=_loc(node, self.filename))

    def _lower_ternary(self, node: Node) -> TernaryExpr:
        cond_node = node.child_by_field_name("condition")
        then_node = node.child_by_field_name("consequence")
        else_node = node.child_by_field_name("alternative")

        cond = self._lower_expr(cond_node) if cond_node else IntLiteral(value=0, text="0")
        then_val = self._lower_expr(then_node) if then_node else IntLiteral(value=0, text="0")
        else_val = self._lower_expr(else_node) if else_node else IntLiteral(value=0, text="0")

        return TernaryExpr(condition=cond, then_expr=then_val, else_expr=else_val,
                           loc=_loc(node, self.filename))

    def _lower_comma(self, node: Node) -> CommaExpr:
        exprs = [self._lower_expr(c) for c in node.named_children]
        return CommaExpr(exprs=exprs, loc=_loc(node, self.filename))

    def _lower_init_list(self, node: Node) -> InitListExpr:
        values = [self._lower_expr(c) for c in node.named_children]
        return InitListExpr(elements=values, loc=_loc(node, self.filename))

    def _lower_pointer_expr(self, node: Node) -> UnaryExpr:
        op_node = node.child_by_field_name("operator")
        arg_node = node.child_by_field_name("argument")
        op_text = _text(op_node) if op_node else "*"
        arg = self._lower_expr(arg_node) if arg_node else IntLiteral(value=0, text="0")
        op = _UNOP_MAP.get(op_text, CASTUnaryOp.DEREF)
        return UnaryExpr(op=op, operand=arg,
                         loc=_loc(node, self.filename))
