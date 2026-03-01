"""
Rust AST node types for the Cross-Language Equivalence Verifier.

Defines a complete Rust abstract syntax tree covering items (functions,
structs, enums, impls, use, const, static, type aliases, traits, extern blocks),
expressions, statements, patterns, types, and generics. Focused on the subset
of Rust produced by C2Rust transpilation.
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
    def line(self) -> int:
        return self.span.start.line if self.span else 0

    @property
    def column(self) -> int:
        return self.span.start.column if self.span else 0

    @property
    def file(self) -> str:
        return self.span.start.file if self.span else ""


# ---------------------------------------------------------------------------
# Visibility
# ---------------------------------------------------------------------------

class Visibility(Enum):
    PRIVATE = auto()
    PUB = auto()
    PUB_CRATE = auto()
    PUB_SUPER = auto()
    PUB_IN = auto()


# ---------------------------------------------------------------------------
# Mutability
# ---------------------------------------------------------------------------

class Mutability(Enum):
    IMMUTABLE = auto()
    MUTABLE = auto()


# ---------------------------------------------------------------------------
# Type nodes
# ---------------------------------------------------------------------------

class RustType:
    """Base class for all Rust type representations."""
    pass


@dataclass
class NeverType(RustType):
    """The never type: !"""
    def __str__(self) -> str:
        return "!"


@dataclass
class UnitType(RustType):
    """The unit type: ()"""
    def __str__(self) -> str:
        return "()"


@dataclass
class PathType(RustType):
    """A named type path: std::io::Result, i32, MyStruct, etc."""
    segments: list[str] = field(default_factory=list)
    generic_args: list[RustType] = field(default_factory=list)

    def __str__(self) -> str:
        path = "::".join(self.segments)
        if self.generic_args:
            args = ", ".join(str(a) for a in self.generic_args)
            return f"{path}<{args}>"
        return path

    @property
    def name(self) -> str:
        return self.segments[-1] if self.segments else ""

    @property
    def is_primitive(self) -> bool:
        return self.name in (
            "i8", "i16", "i32", "i64", "i128", "isize",
            "u8", "u16", "u32", "u64", "u128", "usize",
            "f32", "f64", "bool", "char", "str",
        )


@dataclass
class ReferenceType(RustType):
    """Reference type: &T or &mut T"""
    referent: RustType
    mutability: Mutability = Mutability.IMMUTABLE
    lifetime: str = ""

    def __str__(self) -> str:
        lt = f"'{self.lifetime} " if self.lifetime else ""
        mut = "mut " if self.mutability == Mutability.MUTABLE else ""
        return f"&{lt}{mut}{self.referent}"


@dataclass
class RawPointerType(RustType):
    """Raw pointer type: *const T or *mut T"""
    pointee: RustType
    mutability: Mutability = Mutability.IMMUTABLE

    def __str__(self) -> str:
        kind = "mut" if self.mutability == Mutability.MUTABLE else "const"
        return f"*{kind} {self.pointee}"


@dataclass
class ArrayType(RustType):
    """Array type: [T; N]"""
    element: RustType
    length: Optional["Expr"] = None

    def __str__(self) -> str:
        if self.length:
            return f"[{self.element}; {self.length}]"
        return f"[{self.element}]"


@dataclass
class SliceType(RustType):
    """Slice type: [T]"""
    element: RustType

    def __str__(self) -> str:
        return f"[{self.element}]"


@dataclass
class TupleType(RustType):
    """Tuple type: (T1, T2, ...)"""
    elements: list[RustType] = field(default_factory=list)

    def __str__(self) -> str:
        elems = ", ".join(str(e) for e in self.elements)
        return f"({elems})"


@dataclass
class FnPointerType(RustType):
    """Function pointer type: fn(T1, T2) -> T3"""
    params: list[RustType] = field(default_factory=list)
    return_type: Optional[RustType] = None
    is_unsafe: bool = False
    abi: str = ""

    def __str__(self) -> str:
        unsafe_prefix = "unsafe " if self.is_unsafe else ""
        abi = f'extern "{self.abi}" ' if self.abi else ""
        params = ", ".join(str(p) for p in self.params)
        ret = f" -> {self.return_type}" if self.return_type else ""
        return f"{unsafe_prefix}{abi}fn({params}){ret}"


@dataclass
class InferredType(RustType):
    """Inferred type: _"""
    def __str__(self) -> str:
        return "_"


@dataclass
class OptionType(RustType):
    """Option<T> type (common in C2Rust output)."""
    inner: RustType

    def __str__(self) -> str:
        return f"Option<{self.inner}>"


@dataclass
class ResultType(RustType):
    """Result<T, E> type."""
    ok_type: RustType
    err_type: RustType

    def __str__(self) -> str:
        return f"Result<{self.ok_type}, {self.err_type}>"


@dataclass
class BoxType(RustType):
    """Box<T> type."""
    inner: RustType

    def __str__(self) -> str:
        return f"Box<{self.inner}>"


@dataclass
class ParenType(RustType):
    """Parenthesized type for disambiguation."""
    inner: RustType

    def __str__(self) -> str:
        return f"({self.inner})"


# ---------------------------------------------------------------------------
# Generics
# ---------------------------------------------------------------------------

@dataclass
class GenericParam:
    """A generic parameter."""
    name: str = ""
    bounds: list[RustType] = field(default_factory=list)
    default: Optional[RustType] = None
    is_lifetime: bool = False
    loc: NodeLocation = field(default_factory=NodeLocation)


@dataclass
class Generics:
    """Generic parameters and where clauses."""
    params: list[GenericParam] = field(default_factory=list)
    where_clauses: list["WhereClause"] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.params and not self.where_clauses


@dataclass
class WhereClause:
    """A where clause predicate."""
    bounded_type: Optional[RustType] = None
    bounds: list[RustType] = field(default_factory=list)
    loc: NodeLocation = field(default_factory=NodeLocation)


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

class Pattern:
    """Base class for all patterns."""
    loc: NodeLocation = NodeLocation()


@dataclass
class IdentPattern(Pattern):
    """Identifier pattern: name, mut name, ref name."""
    name: str = ""
    mutability: Mutability = Mutability.IMMUTABLE
    is_ref: bool = False
    subpattern: Optional[Pattern] = None
    loc: NodeLocation = field(default_factory=NodeLocation)

    def __str__(self) -> str:
        parts = []
        if self.is_ref:
            parts.append("ref")
        if self.mutability == Mutability.MUTABLE:
            parts.append("mut")
        parts.append(self.name)
        return " ".join(parts)


@dataclass
class WildcardPattern(Pattern):
    """Wildcard pattern: _"""
    loc: NodeLocation = field(default_factory=NodeLocation)

    def __str__(self) -> str:
        return "_"


@dataclass
class LiteralPattern(Pattern):
    """Literal pattern: 42, "hello", true."""
    value: Optional["Expr"] = None
    loc: NodeLocation = field(default_factory=NodeLocation)


@dataclass
class TuplePattern(Pattern):
    """Tuple pattern: (a, b, c)."""
    elements: list[Pattern] = field(default_factory=list)
    loc: NodeLocation = field(default_factory=NodeLocation)


@dataclass
class StructPattern(Pattern):
    """Struct pattern: MyStruct { field: pattern, .. }."""
    path: list[str] = field(default_factory=list)
    fields: list[tuple[str, Pattern]] = field(default_factory=list)
    has_rest: bool = False
    loc: NodeLocation = field(default_factory=NodeLocation)


@dataclass
class TupleStructPattern(Pattern):
    """Tuple struct pattern: MyStruct(a, b)."""
    path: list[str] = field(default_factory=list)
    elements: list[Pattern] = field(default_factory=list)
    loc: NodeLocation = field(default_factory=NodeLocation)


@dataclass
class RefPattern(Pattern):
    """Reference pattern: &pattern, &mut pattern."""
    inner: Optional[Pattern] = None
    mutability: Mutability = Mutability.IMMUTABLE
    loc: NodeLocation = field(default_factory=NodeLocation)


@dataclass
class RangePattern(Pattern):
    """Range pattern: 1..=10."""
    start: Optional["Expr"] = None
    end: Optional["Expr"] = None
    inclusive: bool = True
    loc: NodeLocation = field(default_factory=NodeLocation)


@dataclass
class OrPattern(Pattern):
    """Or pattern: pat1 | pat2."""
    alternatives: list[Pattern] = field(default_factory=list)
    loc: NodeLocation = field(default_factory=NodeLocation)


@dataclass
class PathPattern(Pattern):
    """Path pattern: Some(x), None, MyEnum::Variant."""
    path: list[str] = field(default_factory=list)
    loc: NodeLocation = field(default_factory=NodeLocation)


@dataclass
class SlicePattern(Pattern):
    """Slice pattern: [a, b, .., c]."""
    elements: list[Pattern] = field(default_factory=list)
    loc: NodeLocation = field(default_factory=NodeLocation)


# ---------------------------------------------------------------------------
# Attributes
# ---------------------------------------------------------------------------

@dataclass
class Attribute:
    """An attribute: #[name(args)] or #![name(args)]."""
    path: list[str] = field(default_factory=list)
    args: str = ""
    is_inner: bool = False
    loc: NodeLocation = field(default_factory=NodeLocation)

    @property
    def name(self) -> str:
        return "::".join(self.path)


# ---------------------------------------------------------------------------
# Items (top-level declarations)
# ---------------------------------------------------------------------------

@dataclass
class Item:
    """Base class for all items."""
    loc: NodeLocation = field(default_factory=NodeLocation)
    attributes: list[Attribute] = field(default_factory=list)
    visibility: Visibility = Visibility.PRIVATE


@dataclass
class FnParam:
    """Function parameter."""
    pattern: Optional[Pattern] = None
    type_ann: Optional[RustType] = None
    is_self: bool = False
    self_mutability: Mutability = Mutability.IMMUTABLE
    loc: NodeLocation = field(default_factory=NodeLocation)

    @property
    def name(self) -> str:
        if isinstance(self.pattern, IdentPattern):
            return self.pattern.name
        return ""


@dataclass
class FnItem(Item):
    """Function definition."""
    name: str = ""
    params: list[FnParam] = field(default_factory=list)
    return_type: Optional[RustType] = None
    body: Optional["BlockExpr"] = None
    generics: Generics = field(default_factory=Generics)
    is_unsafe: bool = False
    is_async: bool = False
    is_const: bool = False
    abi: str = ""  # e.g., "C" for extern "C"
    is_extern: bool = False

    @property
    def is_definition(self) -> bool:
        return self.body is not None


@dataclass
class StructField(Item):
    """A struct field."""
    name: str = ""
    type_ann: Optional[RustType] = None


@dataclass
class StructItem(Item):
    """Struct definition."""
    name: str = ""
    fields: list[StructField] = field(default_factory=list)
    generics: Generics = field(default_factory=Generics)
    is_tuple_struct: bool = False

    @property
    def is_unit_struct(self) -> bool:
        return not self.fields


@dataclass
class EnumVariant:
    """An enum variant."""
    name: str = ""
    fields: list[StructField] = field(default_factory=list)
    discriminant: Optional["Expr"] = None
    is_tuple: bool = False
    is_unit: bool = True
    loc: NodeLocation = field(default_factory=NodeLocation)
    attributes: list[Attribute] = field(default_factory=list)


@dataclass
class EnumItem(Item):
    """Enum definition."""
    name: str = ""
    variants: list[EnumVariant] = field(default_factory=list)
    generics: Generics = field(default_factory=Generics)
    repr: str = ""  # #[repr(C)], #[repr(u32)], etc.


@dataclass
class ImplItem(Item):
    """Impl block."""
    self_type: Optional[RustType] = None
    trait_type: Optional[RustType] = None
    items: list[Item] = field(default_factory=list)
    generics: Generics = field(default_factory=Generics)
    is_unsafe: bool = False

    @property
    def is_trait_impl(self) -> bool:
        return self.trait_type is not None


@dataclass
class UseItem(Item):
    """Use declaration."""
    path: list[str] = field(default_factory=list)
    alias: str = ""
    is_glob: bool = False
    group: list["UseItem"] = field(default_factory=list)

    @property
    def full_path(self) -> str:
        return "::".join(self.path)


@dataclass
class ConstItem(Item):
    """Const declaration."""
    name: str = ""
    type_ann: Optional[RustType] = None
    value: Optional["Expr"] = None


@dataclass
class StaticItem(Item):
    """Static declaration."""
    name: str = ""
    type_ann: Optional[RustType] = None
    value: Optional["Expr"] = None
    mutability: Mutability = Mutability.IMMUTABLE


@dataclass
class TypeAliasItem(Item):
    """Type alias: type Name = Type;"""
    name: str = ""
    aliased_type: Optional[RustType] = None
    generics: Generics = field(default_factory=Generics)


@dataclass
class TraitItem(Item):
    """Trait definition."""
    name: str = ""
    items: list[Item] = field(default_factory=list)
    generics: Generics = field(default_factory=Generics)
    supertraits: list[RustType] = field(default_factory=list)
    is_unsafe: bool = False
    is_auto: bool = False


@dataclass
class ExternBlock(Item):
    """Extern block: extern "C" { ... }"""
    abi: str = "C"
    items: list[Item] = field(default_factory=list)


@dataclass
class ModItem(Item):
    """Module declaration."""
    name: str = ""
    items: list[Item] = field(default_factory=list)
    is_inline: bool = False


@dataclass
class MacroDefItem(Item):
    """Macro definition."""
    name: str = ""
    body: str = ""


# ---------------------------------------------------------------------------
# Expressions
# ---------------------------------------------------------------------------

class BinaryOp(Enum):
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    REM = "%"
    BITAND = "&"
    BITOR = "|"
    BITXOR = "^"
    SHL = "<<"
    SHR = ">>"
    AND = "&&"
    OR = "||"
    EQ = "=="
    NE = "!="
    LT = "<"
    GT = ">"
    LE = "<="
    GE = ">="


class UnaryOp(Enum):
    NEG = "-"
    NOT = "!"
    DEREF = "*"
    REF = "&"
    REF_MUT = "&mut"


@dataclass
class Expr:
    """Base class for all expressions."""
    loc: NodeLocation = field(default_factory=NodeLocation)
    type_annotation: Optional[RustType] = None


@dataclass
class LitExpr(Expr):
    """Literal expression: 42, 3.14, "hello", true, 'a'."""
    int_value: Optional[int] = None
    float_value: Optional[float] = None
    string_value: Optional[str] = None
    char_value: Optional[int] = None
    bool_value: Optional[bool] = None
    type_suffix: str = ""
    text: str = ""

    @property
    def is_int(self) -> bool:
        return self.int_value is not None

    @property
    def is_float(self) -> bool:
        return self.float_value is not None

    @property
    def is_string(self) -> bool:
        return self.string_value is not None

    @property
    def is_bool(self) -> bool:
        return self.bool_value is not None


@dataclass
class PathExpr(Expr):
    """Path expression: x, std::io::stdin, MyEnum::Variant."""
    segments: list[str] = field(default_factory=list)
    generic_args: list[RustType] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.segments[-1] if self.segments else ""

    @property
    def is_simple(self) -> bool:
        return len(self.segments) == 1

    def __str__(self) -> str:
        return "::".join(self.segments)


@dataclass
class BinaryExpr(Expr):
    """Binary expression: lhs op rhs."""
    op: BinaryOp = BinaryOp.ADD
    lhs: Optional[Expr] = None
    rhs: Optional[Expr] = None


@dataclass
class UnaryExpr(Expr):
    """Unary expression: op expr."""
    op: UnaryOp = UnaryOp.NEG
    operand: Optional[Expr] = None


@dataclass
class CastExpr(Expr):
    """Cast expression: expr as Type."""
    operand: Optional[Expr] = None
    target_type: Optional[RustType] = None


@dataclass
class AssignExpr(Expr):
    """Assignment: lhs = rhs."""
    lhs: Optional[Expr] = None
    rhs: Optional[Expr] = None
    op: Optional[BinaryOp] = None  # For compound assignment (+=, etc.)


@dataclass
class CallExpr(Expr):
    """Function call: callee(args...)."""
    callee: Optional[Expr] = None
    args: list[Expr] = field(default_factory=list)


@dataclass
class MethodCallExpr(Expr):
    """Method call: receiver.method(args...)."""
    receiver: Optional[Expr] = None
    method: str = ""
    args: list[Expr] = field(default_factory=list)
    generic_args: list[RustType] = field(default_factory=list)


@dataclass
class FieldExpr(Expr):
    """Field access: expr.field."""
    base: Optional[Expr] = None
    field_name: str = ""


@dataclass
class TupleFieldExpr(Expr):
    """Tuple field access: expr.0, expr.1."""
    base: Optional[Expr] = None
    index: int = 0


@dataclass
class IndexExpr(Expr):
    """Index expression: expr[index]."""
    base: Optional[Expr] = None
    index: Optional[Expr] = None


@dataclass
class RangeExpr(Expr):
    """Range expression: start..end, start..=end, ..end, start.."""
    start: Optional[Expr] = None
    end: Optional[Expr] = None
    inclusive: bool = False


@dataclass
class BlockExpr(Expr):
    """Block expression: { stmts; expr }."""
    stmts: list["Stmt"] = field(default_factory=list)
    tail_expr: Optional[Expr] = None
    is_unsafe: bool = False
    label: str = ""


@dataclass
class IfExpr(Expr):
    """If expression: if cond { ... } else { ... }."""
    condition: Optional[Expr] = None
    then_body: Optional[BlockExpr] = None
    else_body: Optional[Expr] = None  # Can be another IfExpr or BlockExpr


@dataclass
class MatchArm:
    """A single match arm."""
    pattern: Optional[Pattern] = None
    guard: Optional[Expr] = None
    body: Optional[Expr] = None
    loc: NodeLocation = field(default_factory=NodeLocation)


@dataclass
class MatchExpr(Expr):
    """Match expression: match expr { arms }."""
    scrutinee: Optional[Expr] = None
    arms: list[MatchArm] = field(default_factory=list)


@dataclass
class LoopExpr(Expr):
    """Loop expression: loop { ... }."""
    body: Optional[BlockExpr] = None
    label: str = ""


@dataclass
class WhileExpr(Expr):
    """While expression: while cond { ... }."""
    condition: Optional[Expr] = None
    body: Optional[BlockExpr] = None
    label: str = ""


@dataclass
class ForExpr(Expr):
    """For expression: for pat in iter { ... }."""
    pattern: Optional[Pattern] = None
    iterator: Optional[Expr] = None
    body: Optional[BlockExpr] = None
    label: str = ""


@dataclass
class ReturnExpr(Expr):
    """Return expression: return expr."""
    value: Optional[Expr] = None


@dataclass
class BreakExpr(Expr):
    """Break expression: break 'label expr."""
    value: Optional[Expr] = None
    label: str = ""


@dataclass
class ContinueExpr(Expr):
    """Continue expression: continue 'label."""
    label: str = ""


@dataclass
class ClosureExpr(Expr):
    """Closure expression: |params| -> ret { body } or |params| expr."""
    params: list[FnParam] = field(default_factory=list)
    return_type: Optional[RustType] = None
    body: Optional[Expr] = None
    is_move: bool = False
    is_async: bool = False


@dataclass
class TupleExpr(Expr):
    """Tuple expression: (a, b, c)."""
    elements: list[Expr] = field(default_factory=list)


@dataclass
class ArrayExpr(Expr):
    """Array expression: [a, b, c] or [val; count]."""
    elements: list[Expr] = field(default_factory=list)
    repeat_value: Optional[Expr] = None
    repeat_count: Optional[Expr] = None

    @property
    def is_repeat(self) -> bool:
        return self.repeat_value is not None


@dataclass
class StructExpr(Expr):
    """Struct expression: MyStruct { field: value, .. }."""
    path: list[str] = field(default_factory=list)
    fields: list[tuple[str, Expr]] = field(default_factory=list)
    base: Optional[Expr] = None  # .. base
    generic_args: list[RustType] = field(default_factory=list)


@dataclass
class RefExpr(Expr):
    """Reference expression: &expr or &mut expr."""
    operand: Optional[Expr] = None
    mutability: Mutability = Mutability.IMMUTABLE


@dataclass
class DerefExpr(Expr):
    """Dereference expression: *expr."""
    operand: Optional[Expr] = None


@dataclass
class UnsafeBlock(Expr):
    """Unsafe block expression: unsafe { ... }."""
    body: Optional[BlockExpr] = None


@dataclass
class MacroInvocation(Expr):
    """Macro invocation: name!(args) or name![args] or name!{args}."""
    name: str = ""
    args: str = ""
    delimiter: str = ""  # "(", "[", "{"


@dataclass
class TypeAscription(Expr):
    """Type ascription: expr : Type (nightly feature)."""
    operand: Optional[Expr] = None
    ascribed_type: Optional[RustType] = None


@dataclass
class ParenExpr(Expr):
    """Parenthesized expression."""
    inner: Optional[Expr] = None


@dataclass
class AwaitExpr(Expr):
    """Await expression: expr.await."""
    operand: Optional[Expr] = None


@dataclass
class TryExpr(Expr):
    """Try expression: expr?"""
    operand: Optional[Expr] = None


# ---------------------------------------------------------------------------
# Statements
# ---------------------------------------------------------------------------

@dataclass
class Stmt:
    """Base class for all statements."""
    loc: NodeLocation = field(default_factory=NodeLocation)


@dataclass
class LetStmt(Stmt):
    """Let binding: let pat: Type = expr;"""
    pattern: Optional[Pattern] = None
    type_ann: Optional[RustType] = None
    initializer: Optional[Expr] = None
    is_mutable: bool = False


@dataclass
class ExprStmt(Stmt):
    """Expression statement: expr; or expr (without semicolon for tail)."""
    expr: Optional[Expr] = None
    has_semicolon: bool = True


@dataclass
class ItemStmt(Stmt):
    """An item used as a statement (e.g., fn inside fn, struct inside fn)."""
    item: Optional[Item] = None


@dataclass
class EmptyStmt(Stmt):
    """Empty statement: ;"""
    pass


@dataclass
class MacroStmt(Stmt):
    """Macro invocation as statement: name!(args);"""
    invocation: Optional[MacroInvocation] = None


# ---------------------------------------------------------------------------
# Crate (top-level)
# ---------------------------------------------------------------------------

@dataclass
class Crate:
    """Top-level AST node: a Rust crate."""
    items: list[Item] = field(default_factory=list)
    inner_attributes: list[Attribute] = field(default_factory=list)
    loc: NodeLocation = field(default_factory=NodeLocation)

    def functions(self) -> list[FnItem]:
        return [i for i in self.items if isinstance(i, FnItem)]

    def structs(self) -> list[StructItem]:
        return [i for i in self.items if isinstance(i, StructItem)]

    def enums(self) -> list[EnumItem]:
        return [i for i in self.items if isinstance(i, EnumItem)]

    def impls(self) -> list[ImplItem]:
        return [i for i in self.items if isinstance(i, ImplItem)]

    def uses(self) -> list[UseItem]:
        return [i for i in self.items if isinstance(i, UseItem)]

    def type_aliases(self) -> list[TypeAliasItem]:
        return [i for i in self.items if isinstance(i, TypeAliasItem)]

    def consts(self) -> list[ConstItem]:
        return [i for i in self.items if isinstance(i, ConstItem)]

    def statics(self) -> list[StaticItem]:
        return [i for i in self.items if isinstance(i, StaticItem)]

    def extern_blocks(self) -> list[ExternBlock]:
        return [i for i in self.items if isinstance(i, ExternBlock)]


# ---------------------------------------------------------------------------
# AST Visitor
# ---------------------------------------------------------------------------

class RustASTVisitor:
    """Visitor pattern for Rust AST nodes."""

    def visit_crate(self, node: Crate) -> None:
        for item in node.items:
            self.visit_item(item)

    def visit_item(self, node: Item) -> None:
        if isinstance(node, FnItem):
            self.visit_fn_item(node)
        elif isinstance(node, StructItem):
            self.visit_struct_item(node)
        elif isinstance(node, EnumItem):
            self.visit_enum_item(node)
        elif isinstance(node, ImplItem):
            self.visit_impl_item(node)
        elif isinstance(node, UseItem):
            self.visit_use_item(node)
        elif isinstance(node, ConstItem):
            self.visit_const_item(node)
        elif isinstance(node, StaticItem):
            self.visit_static_item(node)
        elif isinstance(node, TypeAliasItem):
            self.visit_type_alias_item(node)
        elif isinstance(node, TraitItem):
            self.visit_trait_item(node)
        elif isinstance(node, ExternBlock):
            self.visit_extern_block(node)
        elif isinstance(node, ModItem):
            self.visit_mod_item(node)

    def visit_fn_item(self, node: FnItem) -> None:
        if node.body:
            self.visit_expr(node.body)

    def visit_struct_item(self, node: StructItem) -> None:
        pass

    def visit_enum_item(self, node: EnumItem) -> None:
        pass

    def visit_impl_item(self, node: ImplItem) -> None:
        for item in node.items:
            self.visit_item(item)

    def visit_use_item(self, node: UseItem) -> None:
        pass

    def visit_const_item(self, node: ConstItem) -> None:
        if node.value:
            self.visit_expr(node.value)

    def visit_static_item(self, node: StaticItem) -> None:
        if node.value:
            self.visit_expr(node.value)

    def visit_type_alias_item(self, node: TypeAliasItem) -> None:
        pass

    def visit_trait_item(self, node: TraitItem) -> None:
        for item in node.items:
            self.visit_item(item)

    def visit_extern_block(self, node: ExternBlock) -> None:
        for item in node.items:
            self.visit_item(item)

    def visit_mod_item(self, node: ModItem) -> None:
        for item in node.items:
            self.visit_item(item)

    def visit_stmt(self, node: Stmt) -> None:
        if isinstance(node, LetStmt):
            self.visit_let_stmt(node)
        elif isinstance(node, ExprStmt):
            if node.expr:
                self.visit_expr(node.expr)
        elif isinstance(node, ItemStmt):
            if node.item:
                self.visit_item(node.item)

    def visit_let_stmt(self, node: LetStmt) -> None:
        if node.initializer:
            self.visit_expr(node.initializer)

    def visit_expr(self, node: Expr) -> None:
        if isinstance(node, BinaryExpr):
            if node.lhs: self.visit_expr(node.lhs)
            if node.rhs: self.visit_expr(node.rhs)
        elif isinstance(node, UnaryExpr):
            if node.operand: self.visit_expr(node.operand)
        elif isinstance(node, CastExpr):
            if node.operand: self.visit_expr(node.operand)
        elif isinstance(node, CallExpr):
            if node.callee: self.visit_expr(node.callee)
            for arg in node.args: self.visit_expr(arg)
        elif isinstance(node, MethodCallExpr):
            if node.receiver: self.visit_expr(node.receiver)
            for arg in node.args: self.visit_expr(arg)
        elif isinstance(node, FieldExpr):
            if node.base: self.visit_expr(node.base)
        elif isinstance(node, IndexExpr):
            if node.base: self.visit_expr(node.base)
            if node.index: self.visit_expr(node.index)
        elif isinstance(node, BlockExpr):
            for stmt in node.stmts: self.visit_stmt(stmt)
            if node.tail_expr: self.visit_expr(node.tail_expr)
        elif isinstance(node, IfExpr):
            if node.condition: self.visit_expr(node.condition)
            if node.then_body: self.visit_expr(node.then_body)
            if node.else_body: self.visit_expr(node.else_body)
        elif isinstance(node, MatchExpr):
            if node.scrutinee: self.visit_expr(node.scrutinee)
            for arm in node.arms:
                if arm.body: self.visit_expr(arm.body)
        elif isinstance(node, LoopExpr):
            if node.body: self.visit_expr(node.body)
        elif isinstance(node, WhileExpr):
            if node.condition: self.visit_expr(node.condition)
            if node.body: self.visit_expr(node.body)
        elif isinstance(node, ForExpr):
            if node.iterator: self.visit_expr(node.iterator)
            if node.body: self.visit_expr(node.body)
        elif isinstance(node, ReturnExpr):
            if node.value: self.visit_expr(node.value)
        elif isinstance(node, AssignExpr):
            if node.lhs: self.visit_expr(node.lhs)
            if node.rhs: self.visit_expr(node.rhs)
        elif isinstance(node, ClosureExpr):
            if node.body: self.visit_expr(node.body)
        elif isinstance(node, RefExpr):
            if node.operand: self.visit_expr(node.operand)
        elif isinstance(node, DerefExpr):
            if node.operand: self.visit_expr(node.operand)
        elif isinstance(node, UnsafeBlock):
            if node.body: self.visit_expr(node.body)
        elif isinstance(node, ParenExpr):
            if node.inner: self.visit_expr(node.inner)
        elif isinstance(node, TupleExpr):
            for e in node.elements: self.visit_expr(e)
        elif isinstance(node, ArrayExpr):
            for e in node.elements: self.visit_expr(e)
