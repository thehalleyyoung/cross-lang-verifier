"""
C AST node types for the Cross-Language Equivalence Verifier.

Defines a complete C abstract syntax tree covering declarations (functions,
variables, typedefs, structs, unions, enums), statements, expressions, and
type representations. Each node carries source location information and
optional type annotations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Sequence, Union

from .lexer import SourcePos, SourceSpan


# ---------------------------------------------------------------------------
# Source location mixin
# ---------------------------------------------------------------------------

@dataclass
class NodeLocation:
    """Source location attached to every AST node."""
    span: Optional[SourceSpan] = None

    @property
    def start(self) -> Optional[SourcePos]:
        return self.span.start if self.span else None

    @property
    def end(self) -> Optional[SourcePos]:
        return self.span.end if self.span else None

    @property
    def line(self) -> int:
        return self.span.start.line if self.span else 0

    @property
    def column(self) -> int:
        return self.span.start.column if self.span else 0

    @property
    def file(self) -> str:
        return self.span.start.file if self.span else ""


# ---------------------------------------------------------------------------
# Type qualifiers and storage classes
# ---------------------------------------------------------------------------

class TypeQualifier(Enum):
    CONST = auto()
    VOLATILE = auto()
    RESTRICT = auto()
    ATOMIC = auto()


class StorageClass(Enum):
    NONE = auto()
    TYPEDEF = auto()
    EXTERN = auto()
    STATIC = auto()
    AUTO = auto()
    REGISTER = auto()
    THREAD_LOCAL = auto()


class FunctionSpecifier(Enum):
    NONE = auto()
    INLINE = auto()
    NORETURN = auto()


# ---------------------------------------------------------------------------
# Type nodes
# ---------------------------------------------------------------------------

class CType:
    """Base class for all C type representations."""
    pass


@dataclass
class VoidCType(CType):
    """void type."""
    pass


@dataclass
class IntCType(CType):
    """Integer type (char, short, int, long, long long, _Bool, __int128)."""
    is_signed: bool = True
    is_unsigned: bool = False
    is_char: bool = False
    is_short: bool = False
    is_int: bool = False
    is_long: bool = False
    is_long_long: bool = False
    is_bool: bool = False
    is_int128: bool = False

    @property
    def width_bits(self) -> int:
        if self.is_bool:
            return 1
        if self.is_char:
            return 8
        if self.is_short:
            return 16
        if self.is_long_long or self.is_int128:
            return 128 if self.is_int128 else 64
        if self.is_long:
            return 64
        return 32  # int

    def __str__(self) -> str:
        parts = []
        if self.is_unsigned:
            parts.append("unsigned")
        elif self.is_signed and not self.is_char and not self.is_bool:
            parts.append("signed")
        if self.is_bool:
            parts.append("_Bool")
        elif self.is_int128:
            parts.append("__int128")
        elif self.is_char:
            parts.append("char")
        elif self.is_short:
            parts.append("short")
        elif self.is_long_long:
            parts.append("long long")
        elif self.is_long:
            parts.append("long")
        else:
            parts.append("int")
        return " ".join(parts)


@dataclass
class FloatCType(CType):
    """Floating-point type (float, double, long double)."""
    is_float: bool = False
    is_double: bool = False
    is_long_double: bool = False

    @property
    def width_bits(self) -> int:
        if self.is_float:
            return 32
        if self.is_long_double:
            return 80  # platform-dependent
        return 64  # double

    def __str__(self) -> str:
        if self.is_float:
            return "float"
        if self.is_long_double:
            return "long double"
        return "double"


@dataclass
class PointerCType(CType):
    """Pointer type."""
    pointee: CType
    qualifiers: list[TypeQualifier] = field(default_factory=list)

    def __str__(self) -> str:
        quals = " ".join(q.name.lower() for q in self.qualifiers)
        if quals:
            return f"{self.pointee} * {quals}"
        return f"{self.pointee} *"


@dataclass
class ArrayCType(CType):
    """Array type."""
    element: CType
    size: Optional["Expr"] = None  # None = unsized (e.g., int[])
    is_static: bool = False
    qualifiers: list[TypeQualifier] = field(default_factory=list)

    def __str__(self) -> str:
        if self.size:
            return f"{self.element}[...]"
        return f"{self.element}[]"


@dataclass
class FunctionCType(CType):
    """Function type."""
    return_type: CType
    params: list["ParamDecl"] = field(default_factory=list)
    is_variadic: bool = False
    is_old_style: bool = False

    def __str__(self) -> str:
        params_str = ", ".join(str(p.type_name) for p in self.params)
        if self.is_variadic:
            params_str += ", ..."
        return f"{self.return_type}({params_str})"


@dataclass
class StructRefCType(CType):
    """Reference to a struct type by name."""
    name: str
    is_definition: bool = False

    def __str__(self) -> str:
        return f"struct {self.name}"


@dataclass
class UnionRefCType(CType):
    """Reference to a union type by name."""
    name: str
    is_definition: bool = False

    def __str__(self) -> str:
        return f"union {self.name}"


@dataclass
class EnumRefCType(CType):
    """Reference to an enum type by name."""
    name: str

    def __str__(self) -> str:
        return f"enum {self.name}"


@dataclass
class TypedefRefCType(CType):
    """Reference to a typedef name."""
    name: str

    def __str__(self) -> str:
        return self.name


@dataclass
class QualifiedCType(CType):
    """A type with qualifiers."""
    base: CType
    qualifiers: list[TypeQualifier] = field(default_factory=list)

    def __str__(self) -> str:
        quals = " ".join(q.name.lower() for q in self.qualifiers)
        return f"{quals} {self.base}" if quals else str(self.base)


@dataclass
class TypeofCType(CType):
    """typeof(expr) type."""
    expr: "Expr"

    def __str__(self) -> str:
        return f"typeof(...)"


@dataclass
class AtomicCType(CType):
    """_Atomic(type)."""
    base: CType

    def __str__(self) -> str:
        return f"_Atomic({self.base})"


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------

@dataclass
class Attribute:
    """A __attribute__((name(args...))) annotation."""
    name: str
    args: list[str] = field(default_factory=list)
    loc: NodeLocation = field(default_factory=NodeLocation)


@dataclass
class Decl:
    """Base class for all declarations."""
    loc: NodeLocation = field(default_factory=NodeLocation)
    attributes: list[Attribute] = field(default_factory=list)


@dataclass
class ParamDecl(Decl):
    """Function parameter declaration."""
    name: str = ""
    type_name: Optional[CType] = None
    storage_class: StorageClass = StorageClass.NONE
    is_register: bool = False

    def __str__(self) -> str:
        t = str(self.type_name) if self.type_name else "?"
        if self.name:
            return f"{t} {self.name}"
        return t


@dataclass
class VarDecl(Decl):
    """Variable declaration (local or global)."""
    name: str = ""
    type_name: Optional[CType] = None
    initializer: Optional["Expr"] = None
    storage_class: StorageClass = StorageClass.NONE
    qualifiers: list[TypeQualifier] = field(default_factory=list)
    is_global: bool = False
    bitfield_width: Optional[int] = None


@dataclass
class FunctionDecl(Decl):
    """Function declaration or definition."""
    name: str = ""
    return_type: Optional[CType] = None
    params: list[ParamDecl] = field(default_factory=list)
    body: Optional["CompoundStmt"] = None
    storage_class: StorageClass = StorageClass.NONE
    is_variadic: bool = False
    is_inline: bool = False
    is_noreturn: bool = False
    is_definition: bool = False
    old_style_params: list[VarDecl] = field(default_factory=list)

    @property
    def is_prototype(self) -> bool:
        return self.body is None


@dataclass
class TypedefDecl(Decl):
    """typedef declaration."""
    name: str = ""
    underlying_type: Optional[CType] = None


@dataclass
class FieldDecl(Decl):
    """Struct/union field declaration."""
    name: str = ""
    type_name: Optional[CType] = None
    bitfield_width: Optional["Expr"] = None


@dataclass
class StructDecl(Decl):
    """Struct declaration/definition."""
    name: str = ""
    fields: list[FieldDecl] = field(default_factory=list)
    is_definition: bool = False
    is_packed: bool = False
    alignment: Optional[int] = None


@dataclass
class UnionDecl(Decl):
    """Union declaration/definition."""
    name: str = ""
    fields: list[FieldDecl] = field(default_factory=list)
    is_definition: bool = False


@dataclass
class EnumeratorDecl:
    """A single enum constant."""
    name: str = ""
    value: Optional["Expr"] = None
    loc: NodeLocation = field(default_factory=NodeLocation)


@dataclass
class EnumDecl(Decl):
    """Enum declaration/definition."""
    name: str = ""
    enumerators: list[EnumeratorDecl] = field(default_factory=list)
    is_definition: bool = False


@dataclass
class StaticAssertDecl(Decl):
    """_Static_assert(expr, message)."""
    condition: Optional["Expr"] = None
    message: str = ""


# ---------------------------------------------------------------------------
# Statements
# ---------------------------------------------------------------------------

@dataclass
class Stmt:
    """Base class for all statements."""
    loc: NodeLocation = field(default_factory=NodeLocation)


@dataclass
class CompoundStmt(Stmt):
    """Compound statement (block): { ... }."""
    items: list[Union[Stmt, Decl]] = field(default_factory=list)


@dataclass
class ExprStmt(Stmt):
    """Expression statement: expr;"""
    expr: Optional["Expr"] = None


@dataclass
class IfStmt(Stmt):
    """if (cond) then_stmt else else_stmt."""
    condition: Optional["Expr"] = None
    then_body: Optional[Stmt] = None
    else_body: Optional[Stmt] = None


@dataclass
class WhileStmt(Stmt):
    """while (cond) body."""
    condition: Optional["Expr"] = None
    body: Optional[Stmt] = None


@dataclass
class DoWhileStmt(Stmt):
    """do body while (cond);"""
    body: Optional[Stmt] = None
    condition: Optional["Expr"] = None


@dataclass
class ForStmt(Stmt):
    """for (init; cond; incr) body."""
    init: Optional[Union[Stmt, VarDecl]] = None
    condition: Optional["Expr"] = None
    increment: Optional["Expr"] = None
    body: Optional[Stmt] = None


@dataclass
class SwitchStmt(Stmt):
    """switch (expr) body."""
    expr: Optional["Expr"] = None
    body: Optional[Stmt] = None


@dataclass
class CaseStmt(Stmt):
    """case expr: stmt."""
    expr: Optional["Expr"] = None
    body: Optional[Stmt] = None
    is_default: bool = False


@dataclass
class ReturnStmt(Stmt):
    """return expr;"""
    expr: Optional["Expr"] = None


@dataclass
class BreakStmt(Stmt):
    """break;"""
    pass


@dataclass
class ContinueStmt(Stmt):
    """continue;"""
    pass


@dataclass
class GotoStmt(Stmt):
    """goto label;"""
    label: str = ""


@dataclass
class LabelStmt(Stmt):
    """label: stmt."""
    label: str = ""
    body: Optional[Stmt] = None


@dataclass
class NullStmt(Stmt):
    """Empty statement ;"""
    pass


@dataclass
class DeclStmt(Stmt):
    """Declaration used as a statement (in compound statement)."""
    decl: Optional[Decl] = None


@dataclass
class AsmStmt(Stmt):
    """Inline assembly statement."""
    template: str = ""
    outputs: list[str] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    clobbers: list[str] = field(default_factory=list)
    is_volatile: bool = False


# ---------------------------------------------------------------------------
# Expressions
# ---------------------------------------------------------------------------

class UnaryOp(Enum):
    """Unary operator kinds."""
    PLUS = "+"
    MINUS = "-"
    BITWISE_NOT = "~"
    LOGICAL_NOT = "!"
    DEREF = "*"
    ADDR = "&"
    PRE_INC = "++"
    PRE_DEC = "--"
    POST_INC = "p++"
    POST_DEC = "p--"


class BinaryOp(Enum):
    """Binary operator kinds."""
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    MOD = "%"
    SHL = "<<"
    SHR = ">>"
    BITAND = "&"
    BITOR = "|"
    BITXOR = "^"
    LOGAND = "&&"
    LOGOR = "||"
    EQ = "=="
    NE = "!="
    LT = "<"
    GT = ">"
    LE = "<="
    GE = ">="
    ASSIGN = "="
    ADD_ASSIGN = "+="
    SUB_ASSIGN = "-="
    MUL_ASSIGN = "*="
    DIV_ASSIGN = "/="
    MOD_ASSIGN = "%="
    SHL_ASSIGN = "<<="
    SHR_ASSIGN = ">>="
    AND_ASSIGN = "&="
    OR_ASSIGN = "|="
    XOR_ASSIGN = "^="
    COMMA = ","


@dataclass
class Expr:
    """Base class for all expressions."""
    loc: NodeLocation = field(default_factory=NodeLocation)
    type_annotation: Optional[CType] = None
    is_lvalue: bool = False
    parenthesized: bool = False


@dataclass
class IntLiteral(Expr):
    """Integer literal."""
    value: int = 0
    suffix_unsigned: bool = False
    suffix_long: int = 0  # 0, 1 (L), 2 (LL)
    text: str = ""

    def __str__(self) -> str:
        return self.text or str(self.value)


@dataclass
class FloatLiteral(Expr):
    """Floating-point literal."""
    value: float = 0.0
    suffix: str = ""  # "", "f", "l"
    text: str = ""

    def __str__(self) -> str:
        return self.text or str(self.value)


@dataclass
class CharLiteral(Expr):
    """Character literal."""
    value: int = 0
    text: str = ""
    is_wide: bool = False

    def __str__(self) -> str:
        return self.text or f"'{chr(self.value)}'"


@dataclass
class StringLiteral(Expr):
    """String literal."""
    value: str = ""
    text: str = ""
    is_wide: bool = False

    def __str__(self) -> str:
        return self.text or f'"{self.value}"'


@dataclass
class IdentExpr(Expr):
    """Identifier reference."""
    name: str = ""

    def __str__(self) -> str:
        return self.name


@dataclass
class BinaryExpr(Expr):
    """Binary expression: lhs op rhs."""
    op: BinaryOp = BinaryOp.ADD
    lhs: Optional[Expr] = None
    rhs: Optional[Expr] = None

    def __str__(self) -> str:
        return f"({self.lhs} {self.op.value} {self.rhs})"

    @property
    def is_assignment(self) -> bool:
        return self.op in (
            BinaryOp.ASSIGN, BinaryOp.ADD_ASSIGN, BinaryOp.SUB_ASSIGN,
            BinaryOp.MUL_ASSIGN, BinaryOp.DIV_ASSIGN, BinaryOp.MOD_ASSIGN,
            BinaryOp.SHL_ASSIGN, BinaryOp.SHR_ASSIGN, BinaryOp.AND_ASSIGN,
            BinaryOp.OR_ASSIGN, BinaryOp.XOR_ASSIGN,
        )

    @property
    def is_comparison(self) -> bool:
        return self.op in (
            BinaryOp.EQ, BinaryOp.NE, BinaryOp.LT, BinaryOp.GT,
            BinaryOp.LE, BinaryOp.GE,
        )

    @property
    def is_logical(self) -> bool:
        return self.op in (BinaryOp.LOGAND, BinaryOp.LOGOR)


@dataclass
class UnaryExpr(Expr):
    """Unary expression: op expr."""
    op: UnaryOp = UnaryOp.PLUS
    operand: Optional[Expr] = None

    def __str__(self) -> str:
        if self.op in (UnaryOp.POST_INC, UnaryOp.POST_DEC):
            return f"({self.operand}{self.op.value[1:]})"
        return f"({self.op.value}{self.operand})"


@dataclass
class CastExpr(Expr):
    """Cast expression: (type)expr."""
    cast_type: Optional[CType] = None
    operand: Optional[Expr] = None

    def __str__(self) -> str:
        return f"({self.cast_type}){self.operand}"


@dataclass
class SizeofExpr(Expr):
    """sizeof expression: sizeof(type) or sizeof expr."""
    operand_type: Optional[CType] = None
    operand_expr: Optional[Expr] = None
    is_type: bool = True

    def __str__(self) -> str:
        if self.is_type:
            return f"sizeof({self.operand_type})"
        return f"sizeof({self.operand_expr})"


@dataclass
class AlignofExpr(Expr):
    """_Alignof expression."""
    operand_type: Optional[CType] = None

    def __str__(self) -> str:
        return f"_Alignof({self.operand_type})"


@dataclass
class CallExpr(Expr):
    """Function call: callee(args...)."""
    callee: Optional[Expr] = None
    args: list[Expr] = field(default_factory=list)

    def __str__(self) -> str:
        args_str = ", ".join(str(a) for a in self.args)
        return f"{self.callee}({args_str})"


@dataclass
class MemberExpr(Expr):
    """Member access: expr.member or expr->member."""
    base: Optional[Expr] = None
    member: str = ""
    is_arrow: bool = False

    def __str__(self) -> str:
        op = "->" if self.is_arrow else "."
        return f"{self.base}{op}{self.member}"


@dataclass
class ArraySubscriptExpr(Expr):
    """Array subscript: base[index]."""
    base: Optional[Expr] = None
    index: Optional[Expr] = None

    def __str__(self) -> str:
        return f"{self.base}[{self.index}]"


@dataclass
class TernaryExpr(Expr):
    """Ternary (conditional) expression: cond ? then : else."""
    condition: Optional[Expr] = None
    then_expr: Optional[Expr] = None
    else_expr: Optional[Expr] = None

    def __str__(self) -> str:
        return f"({self.condition} ? {self.then_expr} : {self.else_expr})"


@dataclass
class CommaExpr(Expr):
    """Comma expression: expr1, expr2."""
    exprs: list[Expr] = field(default_factory=list)

    def __str__(self) -> str:
        return ", ".join(str(e) for e in self.exprs)


@dataclass
class InitListExpr(Expr):
    """Initializer list: { expr1, expr2, ... }."""
    elements: list[Expr] = field(default_factory=list)
    designators: list[Optional["Designator"]] = field(default_factory=list)

    def __str__(self) -> str:
        return "{ " + ", ".join(str(e) for e in self.elements) + " }"


@dataclass
class Designator:
    """Designated initializer component."""
    field_name: Optional[str] = None  # .field
    index: Optional[Expr] = None      # [index]
    loc: NodeLocation = field(default_factory=NodeLocation)


@dataclass
class CompoundLiteralExpr(Expr):
    """Compound literal: (type){ init_list }."""
    type_name: Optional[CType] = None
    init_list: Optional[InitListExpr] = None

    def __str__(self) -> str:
        return f"({self.type_name}){self.init_list}"


@dataclass
class GenericExpr(Expr):
    """_Generic expression (C11)."""
    controlling_expr: Optional[Expr] = None
    associations: list[tuple[Optional[CType], Expr]] = field(default_factory=list)


@dataclass
class StmtExpr(Expr):
    """GCC statement expression: ({ stmt; expr; })."""
    body: Optional[CompoundStmt] = None

    def __str__(self) -> str:
        return "({...})"


@dataclass
class BuiltinCallExpr(Expr):
    """Compiler builtin call: __builtin_xxx(args)."""
    builtin_name: str = ""
    args: list[Expr] = field(default_factory=list)

    def __str__(self) -> str:
        args_str = ", ".join(str(a) for a in self.args)
        return f"{self.builtin_name}({args_str})"


@dataclass
class ImplicitCastExpr(Expr):
    """Implicit type conversion (inserted during type checking)."""
    cast_kind: str = ""  # e.g., "IntegralPromotion", "LValueToRValue"
    operand: Optional[Expr] = None
    target_type: Optional[CType] = None

    def __str__(self) -> str:
        return f"<{self.cast_kind}>({self.operand})"


@dataclass
class ParenExpr(Expr):
    """Parenthesized expression - preserved for exact source fidelity."""
    inner: Optional[Expr] = None

    def __str__(self) -> str:
        return f"({self.inner})"


@dataclass
class OffsetofExpr(Expr):
    """__builtin_offsetof(type, member)."""
    type_name: Optional[CType] = None
    member_name: str = ""

    def __str__(self) -> str:
        return f"__builtin_offsetof({self.type_name}, {self.member_name})"


@dataclass
class VaArgExpr(Expr):
    """va_arg(ap, type)."""
    ap_expr: Optional[Expr] = None
    arg_type: Optional[CType] = None

    def __str__(self) -> str:
        return f"va_arg({self.ap_expr}, {self.arg_type})"


# ---------------------------------------------------------------------------
# Translation unit (top-level)
# ---------------------------------------------------------------------------

@dataclass
class TranslationUnit:
    """Top-level AST node: a complete C source file."""
    declarations: list[Decl] = field(default_factory=list)
    loc: NodeLocation = field(default_factory=NodeLocation)
    filename: str = ""

    def functions(self) -> list[FunctionDecl]:
        """Return all function declarations/definitions."""
        return [d for d in self.declarations if isinstance(d, FunctionDecl)]

    def function_definitions(self) -> list[FunctionDecl]:
        """Return only function definitions (with bodies)."""
        return [d for d in self.declarations
                if isinstance(d, FunctionDecl) and d.is_definition]

    def global_variables(self) -> list[VarDecl]:
        """Return all global variable declarations."""
        return [d for d in self.declarations if isinstance(d, VarDecl)]

    def typedefs(self) -> list[TypedefDecl]:
        """Return all typedef declarations."""
        return [d for d in self.declarations if isinstance(d, TypedefDecl)]

    def structs(self) -> list[StructDecl]:
        """Return all struct declarations/definitions."""
        return [d for d in self.declarations if isinstance(d, StructDecl)]

    def unions(self) -> list[UnionDecl]:
        """Return all union declarations/definitions."""
        return [d for d in self.declarations if isinstance(d, UnionDecl)]

    def enums(self) -> list[EnumDecl]:
        """Return all enum declarations/definitions."""
        return [d for d in self.declarations if isinstance(d, EnumDecl)]


# ---------------------------------------------------------------------------
# AST Visitor
# ---------------------------------------------------------------------------

class CASTVisitor:
    """Visitor pattern for C AST nodes."""

    def visit_translation_unit(self, node: TranslationUnit) -> None:
        for decl in node.declarations:
            self.visit_decl(decl)

    def visit_decl(self, node: Decl) -> None:
        if isinstance(node, FunctionDecl):
            self.visit_function_decl(node)
        elif isinstance(node, VarDecl):
            self.visit_var_decl(node)
        elif isinstance(node, TypedefDecl):
            self.visit_typedef_decl(node)
        elif isinstance(node, StructDecl):
            self.visit_struct_decl(node)
        elif isinstance(node, UnionDecl):
            self.visit_union_decl(node)
        elif isinstance(node, EnumDecl):
            self.visit_enum_decl(node)
        elif isinstance(node, StaticAssertDecl):
            self.visit_static_assert(node)

    def visit_function_decl(self, node: FunctionDecl) -> None:
        if node.body:
            self.visit_stmt(node.body)

    def visit_var_decl(self, node: VarDecl) -> None:
        if node.initializer:
            self.visit_expr(node.initializer)

    def visit_typedef_decl(self, node: TypedefDecl) -> None:
        pass

    def visit_struct_decl(self, node: StructDecl) -> None:
        pass

    def visit_union_decl(self, node: UnionDecl) -> None:
        pass

    def visit_enum_decl(self, node: EnumDecl) -> None:
        pass

    def visit_static_assert(self, node: StaticAssertDecl) -> None:
        pass

    def visit_stmt(self, node: Stmt) -> None:
        if isinstance(node, CompoundStmt):
            self.visit_compound_stmt(node)
        elif isinstance(node, ExprStmt):
            self.visit_expr_stmt(node)
        elif isinstance(node, IfStmt):
            self.visit_if_stmt(node)
        elif isinstance(node, WhileStmt):
            self.visit_while_stmt(node)
        elif isinstance(node, DoWhileStmt):
            self.visit_do_while_stmt(node)
        elif isinstance(node, ForStmt):
            self.visit_for_stmt(node)
        elif isinstance(node, SwitchStmt):
            self.visit_switch_stmt(node)
        elif isinstance(node, CaseStmt):
            self.visit_case_stmt(node)
        elif isinstance(node, ReturnStmt):
            self.visit_return_stmt(node)
        elif isinstance(node, BreakStmt):
            self.visit_break_stmt(node)
        elif isinstance(node, ContinueStmt):
            self.visit_continue_stmt(node)
        elif isinstance(node, GotoStmt):
            self.visit_goto_stmt(node)
        elif isinstance(node, LabelStmt):
            self.visit_label_stmt(node)
        elif isinstance(node, DeclStmt):
            if node.decl:
                self.visit_decl(node.decl)

    def visit_compound_stmt(self, node: CompoundStmt) -> None:
        for item in node.items:
            if isinstance(item, Stmt):
                self.visit_stmt(item)
            elif isinstance(item, Decl):
                self.visit_decl(item)

    def visit_expr_stmt(self, node: ExprStmt) -> None:
        if node.expr:
            self.visit_expr(node.expr)

    def visit_if_stmt(self, node: IfStmt) -> None:
        if node.condition:
            self.visit_expr(node.condition)
        if node.then_body:
            self.visit_stmt(node.then_body)
        if node.else_body:
            self.visit_stmt(node.else_body)

    def visit_while_stmt(self, node: WhileStmt) -> None:
        if node.condition:
            self.visit_expr(node.condition)
        if node.body:
            self.visit_stmt(node.body)

    def visit_do_while_stmt(self, node: DoWhileStmt) -> None:
        if node.body:
            self.visit_stmt(node.body)
        if node.condition:
            self.visit_expr(node.condition)

    def visit_for_stmt(self, node: ForStmt) -> None:
        if node.init:
            if isinstance(node.init, Stmt):
                self.visit_stmt(node.init)
            elif isinstance(node.init, VarDecl):
                self.visit_var_decl(node.init)
        if node.condition:
            self.visit_expr(node.condition)
        if node.increment:
            self.visit_expr(node.increment)
        if node.body:
            self.visit_stmt(node.body)

    def visit_switch_stmt(self, node: SwitchStmt) -> None:
        if node.expr:
            self.visit_expr(node.expr)
        if node.body:
            self.visit_stmt(node.body)

    def visit_case_stmt(self, node: CaseStmt) -> None:
        if node.expr:
            self.visit_expr(node.expr)
        if node.body:
            self.visit_stmt(node.body)

    def visit_return_stmt(self, node: ReturnStmt) -> None:
        if node.expr:
            self.visit_expr(node.expr)

    def visit_break_stmt(self, node: BreakStmt) -> None:
        pass

    def visit_continue_stmt(self, node: ContinueStmt) -> None:
        pass

    def visit_goto_stmt(self, node: GotoStmt) -> None:
        pass

    def visit_label_stmt(self, node: LabelStmt) -> None:
        if node.body:
            self.visit_stmt(node.body)

    def visit_expr(self, node: Expr) -> None:
        if isinstance(node, BinaryExpr):
            self.visit_binary_expr(node)
        elif isinstance(node, UnaryExpr):
            self.visit_unary_expr(node)
        elif isinstance(node, CastExpr):
            self.visit_cast_expr(node)
        elif isinstance(node, CallExpr):
            self.visit_call_expr(node)
        elif isinstance(node, MemberExpr):
            self.visit_member_expr(node)
        elif isinstance(node, ArraySubscriptExpr):
            self.visit_array_subscript_expr(node)
        elif isinstance(node, TernaryExpr):
            self.visit_ternary_expr(node)
        elif isinstance(node, SizeofExpr):
            self.visit_sizeof_expr(node)
        elif isinstance(node, CommaExpr):
            self.visit_comma_expr(node)
        elif isinstance(node, InitListExpr):
            self.visit_init_list_expr(node)
        elif isinstance(node, CompoundLiteralExpr):
            self.visit_compound_literal_expr(node)
        elif isinstance(node, IdentExpr):
            self.visit_ident_expr(node)
        elif isinstance(node, IntLiteral):
            self.visit_int_literal(node)
        elif isinstance(node, FloatLiteral):
            self.visit_float_literal(node)
        elif isinstance(node, StringLiteral):
            self.visit_string_literal(node)
        elif isinstance(node, CharLiteral):
            self.visit_char_literal(node)
        elif isinstance(node, ParenExpr):
            if node.inner:
                self.visit_expr(node.inner)
        elif isinstance(node, ImplicitCastExpr):
            if node.operand:
                self.visit_expr(node.operand)

    def visit_binary_expr(self, node: BinaryExpr) -> None:
        if node.lhs:
            self.visit_expr(node.lhs)
        if node.rhs:
            self.visit_expr(node.rhs)

    def visit_unary_expr(self, node: UnaryExpr) -> None:
        if node.operand:
            self.visit_expr(node.operand)

    def visit_cast_expr(self, node: CastExpr) -> None:
        if node.operand:
            self.visit_expr(node.operand)

    def visit_call_expr(self, node: CallExpr) -> None:
        if node.callee:
            self.visit_expr(node.callee)
        for arg in node.args:
            self.visit_expr(arg)

    def visit_member_expr(self, node: MemberExpr) -> None:
        if node.base:
            self.visit_expr(node.base)

    def visit_array_subscript_expr(self, node: ArraySubscriptExpr) -> None:
        if node.base:
            self.visit_expr(node.base)
        if node.index:
            self.visit_expr(node.index)

    def visit_ternary_expr(self, node: TernaryExpr) -> None:
        if node.condition:
            self.visit_expr(node.condition)
        if node.then_expr:
            self.visit_expr(node.then_expr)
        if node.else_expr:
            self.visit_expr(node.else_expr)

    def visit_sizeof_expr(self, node: SizeofExpr) -> None:
        if node.operand_expr:
            self.visit_expr(node.operand_expr)

    def visit_comma_expr(self, node: CommaExpr) -> None:
        for e in node.exprs:
            self.visit_expr(e)

    def visit_init_list_expr(self, node: InitListExpr) -> None:
        for e in node.elements:
            self.visit_expr(e)

    def visit_compound_literal_expr(self, node: CompoundLiteralExpr) -> None:
        if node.init_list:
            self.visit_expr(node.init_list)

    def visit_ident_expr(self, node: IdentExpr) -> None:
        pass

    def visit_int_literal(self, node: IntLiteral) -> None:
        pass

    def visit_float_literal(self, node: FloatLiteral) -> None:
        pass

    def visit_string_literal(self, node: StringLiteral) -> None:
        pass

    def visit_char_literal(self, node: CharLiteral) -> None:
        pass
