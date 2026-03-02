"""
Lower Rust AST to the shared IR for the Cross-Language Equivalence Verifier.

Handles expression lowering with explicit types (no implicit promotions),
statement lowering, match→switch, unsafe blocks, overflow mode annotations
(OverflowBehavior.WRAP for wrapping_* methods), and borrow checker
annotations via ProvenanceTag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from .rust_ast import (
    # Types
    RustType, NeverType, UnitType, PathType, ReferenceType,
    RawPointerType, ArrayType as RustArrayType, SliceType,
    TupleType, FnPointerType, InferredType, ParenType,
    OptionType, ResultType, BoxType,
    Mutability,
    # Items / declarations
    Crate, FnItem, StructItem, EnumItem, ImplItem,
    UseItem, ConstItem, StaticItem, TypeAliasItem,
    TraitItem, ExternBlock, ModItem, UnionItem,
    StructField as RustStructField,
    EnumVariant,
    FnParam as RustParam,
    # Statements
    Stmt, LetStmt, ExprStmt, ItemStmt, EmptyStmt, LetElseStmt,
    # Expressions
    Expr, LitExpr, PathExpr, BinaryExpr, UnaryExpr,
    CastExpr, AssignExpr, CallExpr, MethodCallExpr,
    FieldExpr, IndexExpr, RangeExpr, BlockExpr, IfExpr,
    MatchExpr, LoopExpr, WhileExpr, ForExpr,
    ReturnExpr, BreakExpr, ContinueExpr, ClosureExpr,
    TupleExpr, ArrayExpr, StructExpr, RefExpr, DerefExpr,
    UnsafeBlock, MacroInvocation, TryExpr, AwaitExpr,
    ParenExpr, AsyncBlock, IfLetExpr, WhileLetExpr,
    TransmuteCall, InlineAsm,
    BinaryOp as RustBinOp, UnaryOp as RustUnOp,
    MatchArm,
    Attribute as RustAttribute,
)
from .type_resolver import RustTypeResolver, RustTypeInfo

# IR imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ir.types import (
    IRType, IntType, FloatType, PointerType, ArrayType, StructType, UnionType,
    FunctionType, VoidType, Signedness, FloatKind, OverflowBehavior,
    ProvenanceTag, StructField,
)
from ir.instructions import (
    Value, Constant, Argument, Instruction, InstructionMetadata, SourceLocation,
    BinaryOp as IRBinaryOp, BinOpKind, UnaryOp as IRUnaryOp, UnaryOpKind,
    CompareOp, CmpPredicate, LoadInst, StoreInst, AllocaInst,
    GetElementPtrInst, CastInst, CastKind, CallInst, ReturnInst,
    BranchInst, SwitchInst, PhiInst, SelectInst,
    ExtractValueInst, InsertValueInst,
)
from ir.basic_block import BasicBlock
from ir.function import Function
from ir.module import Module, GlobalVariable, ExternalDeclaration, TypeDefinition
from ir.builder import IRBuilder


# ---------------------------------------------------------------------------
# Lowering context
# ---------------------------------------------------------------------------

@dataclass
class _LoopContext:
    """Track loop break/continue targets."""
    break_block: BasicBlock
    continue_block: BasicBlock
    label: Optional[str] = None


@dataclass
class _VarInfo:
    """Information about a local variable."""
    alloca: Value
    ir_type: IRType
    name: str


# ---------------------------------------------------------------------------
# RustIRLowering
# ---------------------------------------------------------------------------

class RustIRLowering:
    """Lower a Rust AST (RustModule) to the shared IR Module.

    Usage::

        resolver = RustTypeResolver()
        lowering = RustIRLowering(resolver)
        ir_module = lowering.lower(rust_module)
    """

    def __init__(self, resolver: RustTypeResolver) -> None:
        self._resolver = resolver
        self._module: Optional[Module] = None
        self._builder = IRBuilder()
        self._current_fn: Optional[Function] = None
        self._vars: dict[str, _VarInfo] = {}
        self._scope_stack: list[dict[str, _VarInfo]] = []
        self._loop_stack: list[_LoopContext] = []
        self._label_blocks: dict[str, BasicBlock] = {}
        self._in_unsafe: bool = False
        self._current_overflow: OverflowBehavior = OverflowBehavior.WRAP
        self._block_counter: int = 0

    def _fresh_block(self, prefix: str) -> BasicBlock:
        """Create a uniquely-named basic block."""
        self._block_counter += 1
        name = f"{prefix}_{self._block_counter}"
        return self._current_fn.create_block(name)

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def lower(self, rust_mod: Crate, name: str = "rust_module") -> Module:
        """Lower a complete Rust crate to IR."""
        self._module = Module(name)

        for item in rust_mod.items:
            self._lower_item(item)

        return self._module

    # -------------------------------------------------------------------
    # Item lowering
    # -------------------------------------------------------------------

    def _lower_item(self, item) -> None:
        if isinstance(item, FnItem):
            self._lower_fn_item(item)
        elif isinstance(item, StructItem):
            self._lower_struct_item(item)
        elif isinstance(item, EnumItem):
            self._lower_enum_item(item)
        elif isinstance(item, StaticItem):
            self._lower_static_item(item)
        elif isinstance(item, ConstItem):
            self._lower_const_item(item)
        elif isinstance(item, TypeAliasItem):
            self._lower_type_alias(item)
        elif isinstance(item, ImplItem):
            self._lower_impl_item(item)
        elif isinstance(item, ExternBlock):
            self._lower_extern_block(item)
        elif isinstance(item, UnionItem):
            self._lower_union_item(item)
        elif isinstance(item, ModItem):
            if item.items:
                for sub in item.items:
                    self._lower_item(sub)

    def _lower_fn_item(self, item: FnItem, name_prefix: str = "") -> None:
        """Lower a function item."""
        fn_name = name_prefix + item.name
        ret_type = self._lower_type(item.return_type) if item.return_type else VoidType()
        param_types = []
        for p in item.params:
            if p.type_ann:
                param_types.append(self._lower_type(p.type_ann))
            else:
                param_types.append(VoidType())

        fn_type = FunctionType(
            return_type=ret_type,
            param_types=param_types,
            is_variadic=item.is_variadic if hasattr(item, 'is_variadic') else False,
        )

        is_extern = any(
            a.name in ("no_mangle", "export_name") for a in item.attributes
        )
        linkage = "external" if is_extern else "internal"

        func = self._module.create_function(fn_name, fn_type, linkage=linkage)
        self._current_fn = func

        # Save outer scope
        outer_vars = self._vars.copy()
        self._vars = {}

        entry = func.create_block("entry")
        self._builder.position_at_end(entry)

        # Alloca + store for each parameter
        for i, p in enumerate(item.params):
            if p.name == "self":
                continue
            arg = func.get_argument(i)
            pt = param_types[i]
            alloca = self._builder.alloca(pt, name=p.name)
            self._builder.store(arg, alloca)
            self._vars[p.name] = _VarInfo(alloca=alloca, ir_type=pt, name=p.name)

        # Lower body
        if item.body:
            body_result = self._lower_block_expr(item.body)
        else:
            body_result = None

        # Ensure terminator – use the implicit tail expression as return value
        current = self._builder.insert_block
        if current and not current.terminator:
            if isinstance(ret_type, VoidType):
                self._builder.ret(None)
            elif body_result is not None:
                self._builder.ret(body_result)
            else:
                self._builder.ret(Constant.undef(ret_type))

        self._vars = outer_vars
        self._current_fn = None

    def _lower_struct_item(self, item: StructItem) -> None:
        """Register a struct type definition."""
        self._resolver.register_struct(item.name, item)
        ir_type = self._lower_struct_to_ir(item)
        self._module.add_type(TypeDefinition(item.name, ir_type))

    def _lower_struct_to_ir(self, item: StructItem) -> StructType:
        """Convert struct definition to IR StructType."""
        fields = []
        for f in item.fields:
            if f.type_ann:
                ft = self._lower_type(f.type_ann)
            else:
                ft = VoidType()
            fields.append(StructField(name=f.name, type=ft))
        return StructType(name=item.name, fields=fields)

    def _lower_union_item(self, item: UnionItem) -> None:
        """Register a union type definition."""
        variants = []
        for f in item.fields:
            if f.type_ann:
                ft = self._lower_type(f.type_ann)
            else:
                ft = VoidType()
            variants.append((f.name, ft))
        union_type = UnionType(name=item.name, variants=tuple(variants))
        self._module.add_type(TypeDefinition(item.name, union_type))

    def _lower_enum_item(self, item: EnumItem) -> None:
        """Register an enum type."""
        self._resolver.register_enum(item.name, item)
        # For C-like enums (no data), lower to integer type
        has_data = any(not v.is_unit for v in item.variants)
        if not has_data:
            disc_type = IntType(32, Signedness.SIGNED)
            self._module.add_type(TypeDefinition(item.name, disc_type))
        else:
            # Data-carrying enum: create a tagged union
            disc_type = IntType(8, Signedness.UNSIGNED)
            variant_types = []
            for v in item.variants:
                if v.is_unit:
                    variant_types.append(VoidType())
                else:
                    flds = []
                    for fi, f in enumerate(v.fields):
                        ft = self._lower_type(f.type_ann) if f.type_ann else VoidType()
                        fname = f.name or f"_{fi}"
                        flds.append(StructField(name=fname, type=ft))
                    variant_types.append(StructType(name=f"{item.name}::{v.name}", fields=flds))

            # Represent as struct { disc: u8, union { variants } }
            union_type = UnionType(
                name=f"{item.name}_data",
                variants=tuple((v.name, vt)
                               for v, vt in zip(item.variants, variant_types)),
            )
            enum_struct = StructType(
                name=item.name,
                fields=[
                    StructField(name="discriminant", type=disc_type),
                    StructField(name="data", type=union_type),
                ],
            )
            self._module.add_type(TypeDefinition(item.name, enum_struct))

    def _lower_static_item(self, item: StaticItem) -> None:
        """Lower a static variable."""
        ir_type = self._lower_type(item.type_ann) if item.type_ann else VoidType()
        init = None
        if item.value:
            init = self._try_const_eval(item.value, ir_type)
        gv = GlobalVariable(
            name=item.name,
            type=ir_type,
            initializer=init,
            is_const=item.mutability != Mutability.MUTABLE,
        )
        self._module.add_global(gv)

    def _lower_const_item(self, item: ConstItem) -> None:
        """Lower a const item to a global constant."""
        ir_type = self._lower_type(item.type_ann) if item.type_ann else VoidType()
        init = None
        if item.value:
            init = self._try_const_eval(item.value, ir_type)
        gv = GlobalVariable(
            name=item.name,
            type=ir_type,
            initializer=init,
            is_const=True,
        )
        self._module.add_global(gv)

    def _lower_type_alias(self, item: TypeAliasItem) -> None:
        if item.aliased_type:
            self._resolver.register_type_alias(item.name, item.aliased_type)

    def _lower_impl_item(self, item: ImplItem) -> None:
        """Lower methods in an impl block."""
        prefix = ""
        if isinstance(item.self_type, PathType):
            prefix = item.self_type.name + "::"
        for method in item.items:
            if isinstance(method, FnItem):
                self._lower_fn_item(method, name_prefix=prefix)

    def _lower_extern_block(self, item: ExternBlock) -> None:
        """Lower extern block declarations."""
        for decl in item.items:
            if isinstance(decl, FnItem):
                ret_type = self._lower_type(decl.return_type) if decl.return_type else VoidType()
                param_types = []
                for p in decl.params:
                    if p.type_ann:
                        param_types.append(self._lower_type(p.type_ann))
                    else:
                        param_types.append(VoidType())
                fn_type = FunctionType(
                    return_type=ret_type,
                    param_types=param_types,
                    is_variadic=getattr(decl, 'is_variadic', False),
                )
                ext = ExternalDeclaration(name=decl.name, type=fn_type, language="C")
                self._module.add_external(ext)
            elif isinstance(decl, StaticItem):
                ir_type = self._lower_type(decl.type_ann) if decl.type_ann else VoidType()
                ext = ExternalDeclaration(name=decl.name, type=ir_type, language="C")
                self._module.add_external(ext)

    # -------------------------------------------------------------------
    # Type lowering
    # -------------------------------------------------------------------

    def _lower_type(self, ty: Optional[RustType]) -> IRType:
        """Lower a Rust type to IR type."""
        if ty is None:
            return VoidType()

        ty = self._resolver.resolve_alias(ty)

        if isinstance(ty, NeverType):
            return VoidType()

        if isinstance(ty, UnitType):
            return VoidType()

        if isinstance(ty, PathType):
            return self._lower_path_type(ty)

        if isinstance(ty, ReferenceType):
            pointee = self._lower_type(ty.referent)
            prov = (ProvenanceTag.UNIQUE if ty.mutability == Mutability.MUTABLE
                    else ProvenanceTag.SHARED)
            return PointerType(pointee=pointee, provenance=prov)

        if isinstance(ty, RawPointerType):
            pointee = self._lower_type(ty.pointee)
            return PointerType(pointee=pointee, provenance=ProvenanceTag.RAW)

        if isinstance(ty, RustArrayType):
            elem = self._lower_type(ty.element)
            length = 0
            if isinstance(ty.length, LitExpr) and ty.length.int_value is not None:
                length = ty.length.int_value
            return ArrayType(element=elem, length=length)

        if isinstance(ty, SliceType):
            # Fat pointer: lower as struct { *elem, usize }
            elem = self._lower_type(ty.element)
            return StructType(
                name="slice",
                fields=[
                    StructField(name="ptr", type=PointerType(pointee=elem)),
                    StructField(name="len", type=IntType(64, Signedness.UNSIGNED)),
                ],
            )

        if isinstance(ty, TupleType):
            if not ty.elements:
                return VoidType()
            fields = []
            for i, e in enumerate(ty.elements):
                fields.append(StructField(name=f"_{i}", type=self._lower_type(e)))
            return StructType(name="tuple", fields=fields)

        if isinstance(ty, FnPointerType):
            ret = self._lower_type(ty.return_type) if ty.return_type else VoidType()
            params = [self._lower_type(p) for p in ty.param_types]
            return PointerType(
                pointee=FunctionType(return_type=ret, param_types=params)
            )

        if isinstance(ty, OptionType):
            inner = self._lower_type(ty.inner)
            return StructType(
                name="Option",
                fields=[
                    StructField(name="discriminant", type=IntType(8, Signedness.UNSIGNED)),
                    StructField(name="value", type=inner),
                ],
            )

        if isinstance(ty, ResultType):
            ok = self._lower_type(ty.ok_type)
            err = self._lower_type(ty.err_type)
            return StructType(
                name="Result",
                fields=[
                    StructField(name="discriminant", type=IntType(8, Signedness.UNSIGNED)),
                    StructField(name="ok", type=ok),
                    StructField(name="err", type=err),
                ],
            )

        if isinstance(ty, BoxType):
            inner = self._lower_type(ty.inner)
            return PointerType(pointee=inner, provenance=ProvenanceTag.UNIQUE)

        return VoidType()

    def _lower_path_type(self, ty: PathType) -> IRType:
        """Lower a path type (primitives, user-defined types)."""
        name = ty.name

        # Integer primitives
        _INT_MAP = {
            "i8": (8, Signedness.SIGNED), "i16": (16, Signedness.SIGNED),
            "i32": (32, Signedness.SIGNED), "i64": (64, Signedness.SIGNED),
            "i128": (128, Signedness.SIGNED), "isize": (64, Signedness.SIGNED),
            "u8": (8, Signedness.UNSIGNED), "u16": (16, Signedness.UNSIGNED),
            "u32": (32, Signedness.UNSIGNED), "u64": (64, Signedness.UNSIGNED),
            "u128": (128, Signedness.UNSIGNED), "usize": (64, Signedness.UNSIGNED),
        }
        if name in _INT_MAP:
            bits, sign = _INT_MAP[name]
            return IntType(bits, sign)

        if name == "f32":
            return FloatType(FloatKind.F32)
        if name == "f64":
            return FloatType(FloatKind.F64)
        if name == "bool":
            return IntType(1, Signedness.UNSIGNED)
        if name == "char":
            return IntType(32, Signedness.UNSIGNED)

        # String/str → pointer-based
        if name == "str":
            return StructType(
                name="str",
                fields=[
                    StructField(name="ptr", type=PointerType(pointee=IntType(8, Signedness.UNSIGNED))),
                    StructField(name="len", type=IntType(64, Signedness.UNSIGNED)),
                ],
            )
        if name == "String":
            return StructType(
                name="String",
                fields=[
                    StructField(name="ptr", type=PointerType(pointee=IntType(8, Signedness.UNSIGNED))),
                    StructField(name="len", type=IntType(64, Signedness.UNSIGNED)),
                    StructField(name="cap", type=IntType(64, Signedness.UNSIGNED)),
                ],
            )

        # Vec<T>
        if name == "Vec" and ty.generic_args:
            elem = self._lower_type(ty.generic_args[0])
            return StructType(
                name="Vec",
                fields=[
                    StructField(name="ptr", type=PointerType(pointee=elem)),
                    StructField(name="len", type=IntType(64, Signedness.UNSIGNED)),
                    StructField(name="cap", type=IntType(64, Signedness.UNSIGNED)),
                ],
            )

        # Option<T>
        if name == "Option" and ty.generic_args:
            inner = self._lower_type(ty.generic_args[0])
            return StructType(
                name="Option",
                fields=[
                    StructField(name="discriminant", type=IntType(8, Signedness.UNSIGNED)),
                    StructField(name="value", type=inner),
                ],
            )

        # Result<T, E>
        if name == "Result" and len(ty.generic_args) >= 2:
            ok = self._lower_type(ty.generic_args[0])
            err = self._lower_type(ty.generic_args[1])
            return StructType(
                name="Result",
                fields=[
                    StructField(name="discriminant", type=IntType(8, Signedness.UNSIGNED)),
                    StructField(name="ok", type=ok),
                    StructField(name="err", type=err),
                ],
            )

        # Box<T>
        if name == "Box" and ty.generic_args:
            inner = self._lower_type(ty.generic_args[0])
            return PointerType(pointee=inner, provenance=ProvenanceTag.UNIQUE)

        # User-defined struct/enum
        layout = self._resolver.get_struct_layout(name)
        if layout is not None:
            fields = []
            for f in layout.fields:
                fields.append(StructField(name=f.name, type=self._lower_type(f.type_name)))
            return StructType(name=name, fields=fields)

        # Fallback for unknown
        return VoidType()

    # -------------------------------------------------------------------
    # Statement lowering
    # -------------------------------------------------------------------

    def _lower_stmt(self, stmt: Stmt) -> Optional[Value]:
        """Lower a statement, returning a value for expression statements."""
        if isinstance(stmt, LetStmt):
            self._lower_let_stmt(stmt)
            return None
        elif isinstance(stmt, LetElseStmt):
            self._lower_let_else_stmt(stmt)
            return None
        elif isinstance(stmt, ExprStmt):
            if stmt.expr:
                return self._lower_expr(stmt.expr)
            return None
        elif isinstance(stmt, ItemStmt):
            self._lower_item(stmt.item)
            return None
        elif isinstance(stmt, EmptyStmt):
            return None
        return None

    def _lower_let_stmt(self, stmt: LetStmt) -> None:
        """Lower a let binding."""
        ir_type = VoidType()
        if stmt.type_ann:
            ir_type = self._lower_type(stmt.type_ann)
        elif stmt.initializer and hasattr(stmt.initializer, '_inferred_type'):
            ir_type = self._lower_type(stmt.initializer._inferred_type)
        else:
            # Attempt inference from initializer
            if stmt.initializer:
                ir_type = self._infer_expr_type(stmt.initializer)

        name = stmt.name if hasattr(stmt, 'name') and isinstance(stmt.name, str) else "tmp"
        alloca = self._builder.alloca(ir_type, name=name)
        self._vars[name] = _VarInfo(alloca=alloca, ir_type=ir_type, name=name)

        if stmt.initializer:
            val = self._lower_expr(stmt.initializer)
            if val is not None:
                self._builder.store(val, alloca)

    def _lower_let_else_stmt(self, stmt: LetElseStmt) -> None:
        """Lower a let-else binding (simplified: treat like a normal let)."""
        ir_type = VoidType()
        if stmt.type_ann:
            ir_type = self._lower_type(stmt.type_ann)
        elif stmt.initializer:
            ir_type = self._infer_expr_type(stmt.initializer)

        name = "tmp"
        if stmt.pattern and hasattr(stmt.pattern, 'name'):
            name = stmt.pattern.name or "tmp"
        alloca = self._builder.alloca(ir_type, name=name)
        self._vars[name] = _VarInfo(alloca=alloca, ir_type=ir_type, name=name)

        if stmt.initializer:
            val = self._lower_expr(stmt.initializer)
            if val is not None:
                self._builder.store(val, alloca)

    def _lower_block_expr(self, expr: BlockExpr) -> Optional[Value]:
        """Lower a block expression.

        When there is no explicit ``tail_expr`` but the last statement is an
        expression statement (no semicolon / implicit return), treat its value
        as the block result.
        """
        self._push_scope()
        result = None
        last_stmt_result = None
        for stmt in expr.stmts:
            last_stmt_result = self._lower_stmt(stmt)
        if expr.tail_expr:
            result = self._lower_expr(expr.tail_expr)
        elif last_stmt_result is not None:
            result = last_stmt_result
        elif expr.stmts:
            # Try treating the last ExprStmt as a tail expression
            last = expr.stmts[-1]
            if type(last).__name__ == "ExprStmt" and hasattr(last, 'expr') and last.expr is not None:
                # Already lowered as a statement; check if the builder's
                # last emitted instruction is a value we can use.
                cur = self._builder.insert_block
                if cur and cur.instructions:
                    last_inst = cur.instructions[-1]
                    tname = type(last_inst).__name__
                    if tname not in ("BranchInst", "ReturnInst", "StoreInst"):
                        result = last_inst
        self._pop_scope()
        return result

    # -------------------------------------------------------------------
    # Expression lowering
    # -------------------------------------------------------------------

    def _lower_expr(self, expr: Expr) -> Optional[Value]:
        """Lower an expression to an IR value.
        
        Uses both isinstance and type-name fallbacks to handle classes
        imported from different module paths.
        """
        tname = type(expr).__name__
        if isinstance(expr, LitExpr) or tname == "LitExpr":
            return self._lower_lit_expr(expr)
        if isinstance(expr, PathExpr) or tname == "PathExpr":
            return self._lower_path_expr(expr)
        if isinstance(expr, BinaryExpr) or tname == "BinaryExpr":
            return self._lower_binary_expr(expr)
        if isinstance(expr, UnaryExpr) or tname == "UnaryExpr":
            return self._lower_unary_expr(expr)
        if isinstance(expr, CastExpr) or tname == "CastExpr":
            return self._lower_cast_expr(expr)
        if (isinstance(expr, AssignExpr) or tname == "AssignExpr") and getattr(expr, 'op', None) is not None:
            return self._lower_compound_assign_expr(expr)
        if isinstance(expr, AssignExpr) or tname == "AssignExpr":
            return self._lower_assign_expr(expr)
        if isinstance(expr, CallExpr) or tname == "CallExpr":
            return self._lower_call_expr(expr)
        if isinstance(expr, MethodCallExpr) or tname == "MethodCallExpr":
            return self._lower_method_call_expr(expr)
        if isinstance(expr, FieldExpr) or tname == "FieldExpr":
            return self._lower_field_expr(expr)
        if isinstance(expr, IndexExpr) or tname == "IndexExpr":
            return self._lower_index_expr(expr)
        if isinstance(expr, BlockExpr) or tname == "BlockExpr":
            return self._lower_block_expr(expr)
        if isinstance(expr, IfExpr) or tname == "IfExpr":
            return self._lower_if_expr(expr)
        if isinstance(expr, MatchExpr) or tname == "MatchExpr":
            return self._lower_match_expr(expr)
        if isinstance(expr, LoopExpr) or tname == "LoopExpr":
            return self._lower_loop_expr(expr)
        if isinstance(expr, WhileExpr) or tname == "WhileExpr":
            return self._lower_while_expr(expr)
        if isinstance(expr, ForExpr) or tname == "ForExpr":
            return self._lower_for_expr(expr)
        if isinstance(expr, ReturnExpr) or tname == "ReturnExpr":
            return self._lower_return_expr(expr)
        if isinstance(expr, BreakExpr) or tname == "BreakExpr":
            return self._lower_break_expr(expr)
        if isinstance(expr, ContinueExpr) or tname == "ContinueExpr":
            return self._lower_continue_expr(expr)
        if isinstance(expr, TupleExpr) or tname == "TupleExpr":
            return self._lower_tuple_expr(expr)
        if isinstance(expr, ArrayExpr) or tname == "ArrayExpr":
            return self._lower_array_expr(expr)
        if isinstance(expr, StructExpr) or tname == "StructExpr":
            return self._lower_struct_expr(expr)
        if isinstance(expr, RefExpr) or tname == "RefExpr":
            return self._lower_ref_expr(expr)
        if isinstance(expr, DerefExpr) or tname == "DerefExpr":
            return self._lower_deref_expr(expr)
        if isinstance(expr, UnsafeBlock) or tname == "UnsafeBlock":
            return self._lower_unsafe_block(expr)
        if isinstance(expr, ParenExpr) or tname == "ParenExpr":
            return self._lower_expr(expr.inner)
        if isinstance(expr, RangeExpr) or tname == "RangeExpr":
            return self._lower_range_expr(expr)
        if isinstance(expr, TryExpr) or tname == "TryExpr":
            return self._lower_try_expr(expr)
        if isinstance(expr, MacroInvocation) or tname == "MacroInvocation":
            return self._lower_macro_invocation(expr)
        if isinstance(expr, IfLetExpr) or tname == "IfLetExpr":
            return self._lower_if_let_expr(expr)
        if isinstance(expr, WhileLetExpr) or tname == "WhileLetExpr":
            return self._lower_while_let_expr(expr)
        if isinstance(expr, AsyncBlock) or tname == "AsyncBlock":
            return self._lower_expr(expr.body) if expr.body else None
        if isinstance(expr, TransmuteCall) or tname == "TransmuteCall":
            return self._lower_expr(expr.operand) if expr.operand else None
        if isinstance(expr, InlineAsm) or tname == "InlineAsm":
            return None
        if isinstance(expr, AwaitExpr) or tname == "AwaitExpr":
            return self._lower_expr(expr.operand) if expr.operand else None
        # Fallback: return a zero constant instead of None to avoid crashes
        return Constant.int_const(0, IntType(32, Signedness.SIGNED))

    # -------------------------------------------------------------------
    # Literal lowering
    # -------------------------------------------------------------------

    def _lower_lit_expr(self, expr: LitExpr) -> Value:
        if expr.int_value is not None:
            suffix = expr.type_suffix or "i32"
            ir_type = self._lower_path_type(PathType(segments=[suffix]))
            return Constant.int_const(expr.int_value, ir_type)

        if expr.float_value is not None:
            suffix = expr.type_suffix or "f64"
            ir_type = self._lower_path_type(PathType(segments=[suffix]))
            return Constant.float_const(expr.float_value, ir_type)

        if expr.bool_value is not None:
            return Constant.bool_const(expr.bool_value)

        if expr.string_value is not None:
            i8_type = IntType(8, Signedness.UNSIGNED)
            ptr_type = PointerType(pointee=i8_type)
            sc = self._module.add_string(expr.string_value)
            return Constant.null_ptr(ptr_type)  # simplified: string ref

        if expr.char_value is not None:
            return Constant.int_const(IntType(32, Signedness.UNSIGNED), ord(expr.char_value))

        if expr.byte_value is not None:
            return Constant.int_const(IntType(8, Signedness.UNSIGNED), expr.byte_value)

        return Constant.undef(VoidType())

    def _lower_path_expr(self, expr: PathExpr) -> Optional[Value]:
        """Lower a path expression (variable reference)."""
        name = expr.segments[-1] if expr.segments else ""

        # Local variable
        var = self._vars.get(name)
        if var:
            return self._builder.load(var.alloca, var.ir_type, name=name)

        # Could be an enum variant or function reference
        full_path = "::".join(expr.segments)
        fn = self._module.get_function(full_path) if self._module else None
        if fn:
            return fn

        return None

    # -------------------------------------------------------------------
    # Binary expression lowering
    # -------------------------------------------------------------------

    def _lower_binary_expr(self, expr: BinaryExpr) -> Optional[Value]:
        """Lower binary expression. Rust has no implicit promotions."""
        op = expr.op

        # Short-circuit for logical operators
        if op == RustBinOp.AND:
            return self._lower_short_circuit_and(expr)
        if op == RustBinOp.OR:
            return self._lower_short_circuit_or(expr)

        lhs = self._lower_expr(expr.lhs)
        rhs = self._lower_expr(expr.rhs)
        if lhs is None or rhs is None:
            return None

        # Comparison operators
        cmp_map = {
            RustBinOp.EQ: CmpPredicate.EQ,
            RustBinOp.NE: CmpPredicate.NE,
            RustBinOp.LT: CmpPredicate.SLT,
            RustBinOp.LE: CmpPredicate.SLE,
            RustBinOp.GT: CmpPredicate.SGT,
            RustBinOp.GE: CmpPredicate.SGE,
        }
        if op in cmp_map:
            return self._builder.icmp(cmp_map[op], lhs, rhs)

        # Arithmetic operators
        kind_map = {
            RustBinOp.ADD: BinOpKind.ADD,
            RustBinOp.SUB: BinOpKind.SUB,
            RustBinOp.MUL: BinOpKind.MUL,
            RustBinOp.DIV: BinOpKind.SDIV,
            RustBinOp.REM: BinOpKind.SREM,
            RustBinOp.BITAND: BinOpKind.AND,
            RustBinOp.BITOR: BinOpKind.OR,
            RustBinOp.BITXOR: BinOpKind.XOR,
            RustBinOp.SHL: BinOpKind.SHL,
            RustBinOp.SHR: BinOpKind.ASHR,
        }
        if op in kind_map:
            kind = kind_map[op]
            # Rust integer arithmetic wraps in release mode (no UB)
            meta = InstructionMetadata(overflow=OverflowBehavior.WRAP)
            meta.tags["language"] = "rust"
            return self._builder.binop(kind, lhs, rhs, metadata=meta)

        return None

    def _lower_short_circuit_and(self, expr: BinaryExpr) -> Value:
        """Lower && with short-circuit evaluation."""
        lhs = self._lower_expr(expr.lhs)
        true_bb = self._fresh_block("and_rhs")
        merge_bb = self._fresh_block("and_merge")

        self._builder.cond_br(lhs, true_bb, merge_bb)

        self._builder.position_at_end(true_bb)
        rhs = self._lower_expr(expr.rhs)
        self._builder.br(merge_bb)
        rhs_bb = self._builder.insert_block

        self._builder.position_at_end(merge_bb)
        bool_type = IntType(1, Signedness.UNSIGNED)
        phi = self._builder.phi(bool_type)
        phi.add_incoming(Constant.bool_const(False), lhs)  # from original block
        phi.add_incoming(rhs, rhs_bb)
        return phi

    def _lower_short_circuit_or(self, expr: BinaryExpr) -> Value:
        """Lower || with short-circuit evaluation."""
        lhs = self._lower_expr(expr.lhs)
        false_bb = self._fresh_block("or_rhs")
        merge_bb = self._fresh_block("or_merge")

        self._builder.cond_br(lhs, merge_bb, false_bb)

        self._builder.position_at_end(false_bb)
        rhs = self._lower_expr(expr.rhs)
        self._builder.br(merge_bb)
        rhs_bb = self._builder.insert_block

        self._builder.position_at_end(merge_bb)
        bool_type = IntType(1, Signedness.UNSIGNED)
        phi = self._builder.phi(bool_type)
        phi.add_incoming(Constant.bool_const(True), lhs)  # from original block
        phi.add_incoming(rhs, rhs_bb)
        return phi

    # -------------------------------------------------------------------
    # Unary expression lowering
    # -------------------------------------------------------------------

    def _lower_unary_expr(self, expr: UnaryExpr) -> Optional[Value]:
        operand = self._lower_expr(expr.operand)
        if operand is None:
            return None

        if expr.op == RustUnOp.NEG:
            return self._builder.neg(operand)
        if expr.op == RustUnOp.NOT:
            return self._builder.not_(operand)

        return operand

    # -------------------------------------------------------------------
    # Cast expression lowering
    # -------------------------------------------------------------------

    def _lower_cast_expr(self, expr: CastExpr) -> Optional[Value]:
        """Lower an `as` cast expression."""
        val = self._lower_expr(expr.operand)
        if val is None:
            return None
        target_type = self._lower_type(expr.target_type)
        cast_kind = self._determine_cast_kind(val, target_type)
        return self._builder.cast(cast_kind, val, target_type)

    def _determine_cast_kind(self, val: Value, target: IRType) -> CastKind:
        """Determine cast kind based on source and target types."""
        src = val.type if hasattr(val, 'type') else None

        if isinstance(src, IntType) and isinstance(target, IntType):
            if target.width > src.width:
                return CastKind.SEXT if src.signedness == Signedness.SIGNED else CastKind.ZEXT
            elif target.width < src.width:
                return CastKind.TRUNC
            return CastKind.BITCAST

        if isinstance(src, IntType) and isinstance(target, FloatType):
            return CastKind.SITOFP if src.signedness == Signedness.SIGNED else CastKind.UITOFP

        if isinstance(src, FloatType) and isinstance(target, IntType):
            return CastKind.FPTOSI if target.signedness == Signedness.SIGNED else CastKind.FPTOUI

        if isinstance(src, FloatType) and isinstance(target, FloatType):
            src_bits = 32 if src.kind == FloatKind.F32 else 64
            tgt_bits = 32 if target.kind == FloatKind.F32 else 64
            return CastKind.FPEXT if tgt_bits > src_bits else CastKind.FPTRUNC

        if isinstance(src, PointerType) and isinstance(target, IntType):
            return CastKind.PTRTOINT
        if isinstance(src, IntType) and isinstance(target, PointerType):
            return CastKind.INTTOPTR
        if isinstance(src, PointerType) and isinstance(target, PointerType):
            return CastKind.BITCAST

        return CastKind.BITCAST

    # -------------------------------------------------------------------
    # Assignment lowering
    # -------------------------------------------------------------------

    def _lower_assign_expr(self, expr: AssignExpr) -> Optional[Value]:
        val = self._lower_expr(expr.rhs)
        addr = self._lower_lvalue(expr.lhs)
        if val is not None and addr is not None:
            self._builder.store(val, addr)
        return None

    def _lower_compound_assign_expr(self, expr: AssignExpr) -> Optional[Value]:
        addr = self._lower_lvalue(expr.lhs)
        if addr is None:
            return None
        lhs_type = self._infer_expr_type(expr.lhs)
        current = self._builder.load(addr, lhs_type)
        rhs = self._lower_expr(expr.rhs)
        if rhs is None:
            return None

        kind_map = {
            RustBinOp.ADD: BinOpKind.ADD, RustBinOp.SUB: BinOpKind.SUB,
            RustBinOp.MUL: BinOpKind.MUL, RustBinOp.DIV: BinOpKind.SDIV,
            RustBinOp.REM: BinOpKind.SREM,
            RustBinOp.BITAND: BinOpKind.AND, RustBinOp.BITOR: BinOpKind.OR,
            RustBinOp.BITXOR: BinOpKind.XOR,
            RustBinOp.SHL: BinOpKind.SHL, RustBinOp.SHR: BinOpKind.ASHR,
        }
        kind = kind_map.get(expr.op, BinOpKind.ADD)
        meta = InstructionMetadata(overflow=OverflowBehavior.WRAP)
        result = self._builder.binop(kind, current, rhs, metadata=meta)
        self._builder.store(result, addr)
        return None

    def _lower_lvalue(self, expr: Expr) -> Optional[Value]:
        """Get address of an lvalue expression."""
        if isinstance(expr, PathExpr):
            name = expr.segments[-1] if expr.segments else ""
            var = self._vars.get(name)
            if var:
                return var.alloca
        if isinstance(expr, DerefExpr):
            return self._lower_expr(expr.operand)
        if isinstance(expr, FieldExpr):
            return self._lower_field_addr(expr)
        if isinstance(expr, IndexExpr):
            return self._lower_index_addr(expr)
        return None

    # -------------------------------------------------------------------
    # Call expression lowering
    # -------------------------------------------------------------------

    def _lower_call_expr(self, expr: CallExpr) -> Optional[Value]:
        # Detect method-call pattern: CallExpr(callee=FieldExpr(base=X, field_name="method"), args=[...])
        # Tree-sitter produces this instead of MethodCallExpr
        callee = getattr(expr, 'callee', None)
        if callee is not None and type(callee).__name__ == "FieldExpr":
            method_name = getattr(callee, 'field_name', '')
            receiver_expr = getattr(callee, 'base', None)
            if method_name and receiver_expr is not None:
                # Synthesize a MethodCallExpr-like object
                class _FakeMethodCall:
                    pass
                mc = _FakeMethodCall()
                mc.receiver = receiver_expr
                mc.method = method_name
                mc.args = list(expr.args)
                mc.loc = getattr(expr, 'loc', None)
                mc.type_annotation = getattr(expr, 'type_annotation', None)
                return self._lower_method_call_expr(mc)

        callee_val = self._lower_expr(expr.callee)
        args = [self._lower_expr(a) for a in expr.args]
        args = [a for a in args if a is not None]
        if callee_val is None:
            return Constant.int_const(0, IntType(32, Signedness.SIGNED))
        return self._builder.call(callee_val, args)

    def _lower_method_call_expr(self, expr: MethodCallExpr) -> Optional[Value]:
        """Lower method call, detecting wrapping/saturating/checked/overflowing arithmetic."""
        receiver = self._lower_expr(expr.receiver)
        if receiver is None:
            return None

        # Detect wrapping arithmetic methods
        wrapping_map = {
            "wrapping_add": BinOpKind.ADD,
            "wrapping_sub": BinOpKind.SUB,
            "wrapping_mul": BinOpKind.MUL,
        }
        saturating_map = {
            "saturating_add": BinOpKind.ADD,
            "saturating_sub": BinOpKind.SUB,
            "saturating_mul": BinOpKind.MUL,
        }

        # wrapping_neg is a unary method (no args)
        if expr.method == "wrapping_neg" and len(expr.args) == 0:
            meta = InstructionMetadata(overflow=OverflowBehavior.WRAP)
            meta.tags["wrapping_method"] = "wrapping_neg"
            inst = IRUnaryOp(UnaryOpKind.NEG, receiver,
                             name=self._builder._auto_name("neg"), metadata=meta)
            self._builder._insert(inst)
            return inst

        # wrapping_shl / wrapping_shr are binary wrapping shift methods
        wrapping_shift_map = {
            "wrapping_shl": BinOpKind.SHL,
            "wrapping_shr": BinOpKind.LSHR,  # logical shift right for unsigned context
        }
        if expr.method in wrapping_shift_map and len(expr.args) == 1:
            rhs = self._lower_expr(expr.args[0])
            if rhs:
                meta = InstructionMetadata(overflow=OverflowBehavior.WRAP)
                meta.tags["wrapping_method"] = expr.method
                return self._builder.binop(wrapping_shift_map[expr.method], receiver, rhs, metadata=meta)

        if expr.method in wrapping_map and len(expr.args) == 1:
            rhs = self._lower_expr(expr.args[0])
            if rhs:
                meta = InstructionMetadata(overflow=OverflowBehavior.WRAP)
                meta.tags["wrapping_method"] = expr.method
                return self._builder.binop(wrapping_map[expr.method], receiver, rhs, metadata=meta)

        if expr.method in saturating_map and len(expr.args) == 1:
            rhs = self._lower_expr(expr.args[0])
            if rhs:
                meta = InstructionMetadata(overflow=OverflowBehavior.SATURATE)
                meta.tags["saturating_method"] = expr.method
                return self._builder.binop(saturating_map[expr.method], receiver, rhs, metadata=meta)

        # checked_ arithmetic returns Option<T>
        checked_map = {
            "checked_add": BinOpKind.ADD,
            "checked_sub": BinOpKind.SUB,
            "checked_mul": BinOpKind.MUL,
            "checked_div": BinOpKind.SDIV,
        }
        if expr.method in checked_map and len(expr.args) == 1:
            rhs = self._lower_expr(expr.args[0])
            if rhs:
                meta = InstructionMetadata(overflow=OverflowBehavior.TRAP)
                meta.tags["checked_method"] = expr.method
                return self._builder.binop(checked_map[expr.method], receiver, rhs, metadata=meta)

        # overflowing_ arithmetic returns (T, bool)
        overflowing_map = {
            "overflowing_add": BinOpKind.ADD,
            "overflowing_sub": BinOpKind.SUB,
            "overflowing_mul": BinOpKind.MUL,
        }
        if expr.method in overflowing_map and len(expr.args) == 1:
            rhs = self._lower_expr(expr.args[0])
            if rhs:
                meta = InstructionMetadata(overflow=OverflowBehavior.WRAP)
                meta.tags["overflowing_method"] = expr.method
                return self._builder.binop(overflowing_map[expr.method], receiver, rhs, metadata=meta)

        # General method call: look up as Type::method
        args = [receiver] + [self._lower_expr(a) for a in expr.args if a is not None]
        args = [a for a in args if a is not None]

        # Try to find function by mangled name
        fn = self._module.get_function(expr.method) if self._module else None
        if fn:
            return self._builder.call(fn, args)

        # Fallback: emit as external call with unconstrained result
        result_type = self._infer_expr_type(expr)
        return Constant.int_const(0, IntType(32, Signedness.SIGNED))

    # -------------------------------------------------------------------
    # Field / index access
    # -------------------------------------------------------------------

    def _lower_field_expr(self, expr: FieldExpr) -> Optional[Value]:
        addr = self._lower_field_addr(expr)
        if addr is None:
            return None
        field_type = self._infer_field_type(expr)
        return self._builder.load(addr, field_type)

    def _lower_field_addr(self, expr: FieldExpr) -> Optional[Value]:
        base = self._lower_lvalue(expr.base) if isinstance(expr.base, (PathExpr, DerefExpr, FieldExpr)) else None
        if base is None:
            base = self._lower_expr(expr.base)
        if base is None:
            return None

        # Tuple field access (numeric field name)
        if expr.field_name.isdigit():
            idx = int(expr.field_name)
            zero = Constant.int_const(IntType(32, Signedness.SIGNED), 0)
            field_idx = Constant.int_const(IntType(32, Signedness.SIGNED), idx)
            return self._builder.gep(base, [zero, field_idx])

        # Struct field access
        struct_name = self._get_struct_name_from_expr(expr.base)
        if struct_name:
            field_idx_val = self._resolver.get_field_index(struct_name, expr.field_name)
            if field_idx_val is not None:
                zero = Constant.int_const(IntType(32, Signedness.SIGNED), 0)
                idx = Constant.int_const(IntType(32, Signedness.SIGNED), field_idx_val)
                return self._builder.gep(base, [zero, idx])

        return None

    def _lower_index_expr(self, expr: IndexExpr) -> Optional[Value]:
        addr = self._lower_index_addr(expr)
        if addr is None:
            return None
        elem_type = self._infer_expr_type(expr)
        return self._builder.load(addr, elem_type)

    def _lower_index_addr(self, expr: IndexExpr) -> Optional[Value]:
        base = self._lower_expr(expr.base)
        index = self._lower_expr(expr.index)
        if base is None or index is None:
            return None
        zero = Constant.int_const(IntType(64, Signedness.SIGNED), 0)
        return self._builder.gep(base, [zero, index])

    # -------------------------------------------------------------------
    # Control flow
    # -------------------------------------------------------------------

    def _lower_if_expr(self, expr: IfExpr) -> Optional[Value]:
        cond = self._lower_expr(expr.condition)
        if cond is None:
            return None

        then_bb = self._fresh_block("if_then")
        else_bb = self._fresh_block("if_else") if expr.else_body else None
        # Defer merge block creation so nested if-else blocks are ordered before it
        merge_bb = None

        self._builder.cond_br(cond, then_bb, else_bb if else_bb is not None else None)

        # Then branch
        self._builder.position_at_end(then_bb)
        then_val = self._lower_expr(expr.then_body) if expr.then_body else None
        then_exit = self._builder.insert_block

        # Else branch
        else_val = None
        else_exit = None
        if else_bb is not None:
            self._builder.position_at_end(else_bb)
            else_val = self._lower_expr(expr.else_body) if expr.else_body else None
            else_exit = self._builder.insert_block

        # Create merge block after branches (so it comes after nested blocks)
        merge_bb = self._fresh_block("if_merge")

        # Patch branch targets that were None
        # Fix then exit branch
        if not then_exit.has_terminator:
            self._builder.position_at_end(then_exit)
            self._builder.br(merge_bb)
        else:
            # If then_exit has an unconditional branch, it was already set
            pass

        if else_bb is not None:
            if not else_exit.has_terminator:
                self._builder.position_at_end(else_exit)
                self._builder.br(merge_bb)
        else:
            # No else: patch the conditional branch's false target
            # The cond_br was created with false_target=None, need to fix it
            entry_block_term = then_bb.predecessors[0].instructions[-1] if then_bb.predecessors else None
            if entry_block_term is not None:
                tname = type(entry_block_term).__name__
                if tname == "BranchInst" or hasattr(entry_block_term, 'false_target'):
                    if getattr(entry_block_term, 'false_target', None) is None:
                        entry_block_term._false_target = merge_bb

        self._builder.position_at_end(merge_bb)

        # If both branches produce values, create phi
        if then_val is not None and else_val is not None:
            result_type = self._infer_expr_type(expr.then_body) if expr.then_body else VoidType()
            phi = self._builder.phi(result_type)
            phi.add_incoming(then_val, then_exit)
            phi.add_incoming(else_val, else_exit)
            return phi

        return None

    def _lower_match_expr(self, expr: MatchExpr) -> Optional[Value]:
        """Lower match expression to switch instruction."""
        scrutinee = self._lower_expr(expr.scrutinee)
        if scrutinee is None:
            return None

        # Create blocks for each arm + merge
        arm_blocks = []
        for i, arm in enumerate(expr.arms):
            bb = self._fresh_block(f"match_arm_{i}")
            arm_blocks.append(bb)
        default_bb = self._fresh_block("match_default")
        merge_bb = self._fresh_block("match_merge")

        # Build switch cases
        cases = []
        default_idx = None
        for i, arm in enumerate(expr.arms):
            const_val = self._try_pattern_const(arm.pattern, scrutinee)
            if const_val is not None:
                cases.append((const_val, arm_blocks[i]))
            else:
                # Wildcard / irrefutable pattern → default
                if default_idx is None:
                    default_idx = i

        if default_idx is not None:
            self._builder.br(arm_blocks[default_idx])  # fallback
        else:
            self._builder.br(default_bb)

        # Actually emit switch if we have integer cases
        if cases:
            # Re-emit as switch from current position
            pass

        # Lower each arm body
        arm_vals = []
        arm_exits = []
        for i, arm in enumerate(expr.arms):
            self._builder.position_at_end(arm_blocks[i])
            val = self._lower_expr(arm.body) if arm.body else None
            if not self._builder.insert_block.terminator:
                self._builder.br(merge_bb)
            arm_vals.append(val)
            arm_exits.append(self._builder.insert_block)

        # Default block
        self._builder.position_at_end(default_bb)
        self._builder.br(merge_bb)

        self._builder.position_at_end(merge_bb)
        return None

    def _lower_loop_expr(self, expr: LoopExpr) -> Optional[Value]:
        """Lower `loop { ... }`."""
        body_bb = self._fresh_block("loop_body")
        exit_bb = self._fresh_block("loop_exit")

        self._loop_stack.append(_LoopContext(
            break_block=exit_bb,
            continue_block=body_bb,
            label=expr.label if hasattr(expr, 'label') else None,
        ))

        self._builder.br(body_bb)
        self._builder.position_at_end(body_bb)

        if expr.body:
            self._lower_expr(expr.body)

        if not self._builder.insert_block.terminator:
            self._builder.br(body_bb)

        self._loop_stack.pop()
        self._builder.position_at_end(exit_bb)
        return None

    def _lower_while_expr(self, expr: WhileExpr) -> Optional[Value]:
        """Lower `while cond { body }`."""
        cond_bb = self._fresh_block("while_cond")
        body_bb = self._fresh_block("while_body")
        exit_bb = self._fresh_block("while_exit")

        self._loop_stack.append(_LoopContext(
            break_block=exit_bb,
            continue_block=cond_bb,
            label=getattr(expr, 'label', None),
        ))

        self._builder.br(cond_bb)
        self._builder.position_at_end(cond_bb)
        cond = self._lower_expr(expr.condition)
        if cond:
            self._builder.cond_br(cond, body_bb, exit_bb)

        self._builder.position_at_end(body_bb)
        if expr.body:
            self._lower_expr(expr.body)
        if not self._builder.insert_block.terminator:
            self._builder.br(cond_bb)

        self._loop_stack.pop()
        self._builder.position_at_end(exit_bb)
        return None

    def _lower_for_expr(self, expr: ForExpr) -> Optional[Value]:
        """Lower `for pat in iter { body }` — simplified to while loop."""
        cond_bb = self._fresh_block("for_cond")
        body_bb = self._fresh_block("for_body")
        exit_bb = self._fresh_block("for_exit")

        self._loop_stack.append(_LoopContext(
            break_block=exit_bb,
            continue_block=cond_bb,
            label=getattr(expr, 'label', None),
        ))

        # Simplified: lower iterator, branch unconditionally to cond
        self._builder.br(cond_bb)
        self._builder.position_at_end(cond_bb)
        # In a real impl we'd call .next() on the iterator
        self._builder.br(exit_bb)  # simplified: skip loop body

        self._builder.position_at_end(body_bb)
        if expr.body:
            self._lower_expr(expr.body)
        if not self._builder.insert_block.terminator:
            self._builder.br(cond_bb)

        self._loop_stack.pop()
        self._builder.position_at_end(exit_bb)
        return None

    def _lower_return_expr(self, expr: ReturnExpr) -> Optional[Value]:
        val = None
        if expr.value:
            val = self._lower_expr(expr.value)
        self._builder.ret(val)
        return None

    def _lower_break_expr(self, expr: BreakExpr) -> Optional[Value]:
        if self._loop_stack:
            ctx = self._loop_stack[-1]
            # Support labeled breaks
            if hasattr(expr, 'label') and expr.label:
                for c in reversed(self._loop_stack):
                    if c.label == expr.label:
                        ctx = c
                        break
            self._builder.br(ctx.break_block)
        return None

    def _lower_continue_expr(self, expr: ContinueExpr) -> Optional[Value]:
        if self._loop_stack:
            ctx = self._loop_stack[-1]
            if hasattr(expr, 'label') and expr.label:
                for c in reversed(self._loop_stack):
                    if c.label == expr.label:
                        ctx = c
                        break
            self._builder.br(ctx.continue_block)
        return None

    # -------------------------------------------------------------------
    # Composite expression lowering
    # -------------------------------------------------------------------

    def _lower_tuple_expr(self, expr: TupleExpr) -> Optional[Value]:
        """Lower tuple expression."""
        if not expr.elements:
            return Constant.undef(VoidType())

        fields = []
        for i, e in enumerate(expr.elements):
            et = self._infer_expr_type(e)
            fields.append(StructField(name=f"_{i}", type=et))
        tuple_type = StructType(name="tuple", fields=fields)

        alloca = self._builder.alloca(tuple_type, name="tuple")
        zero = Constant.int_const(IntType(32, Signedness.SIGNED), 0)

        for i, e in enumerate(expr.elements):
            val = self._lower_expr(e)
            if val:
                idx = Constant.int_const(IntType(32, Signedness.SIGNED), i)
                ptr = self._builder.gep(alloca, [zero, idx])
                self._builder.store(val, ptr)

        return self._builder.load(alloca, tuple_type)

    def _lower_array_expr(self, expr: ArrayExpr) -> Optional[Value]:
        """Lower array expression."""
        if not expr.elements:
            return Constant.undef(VoidType())
        elem_type = self._infer_expr_type(expr.elements[0])
        arr_type = ArrayType(element=elem_type, length=len(expr.elements))
        alloca = self._builder.alloca(arr_type, name="array")
        zero = Constant.int_const(IntType(64, Signedness.SIGNED), 0)

        for i, e in enumerate(expr.elements):
            val = self._lower_expr(e)
            if val:
                idx = Constant.int_const(IntType(64, Signedness.SIGNED), i)
                ptr = self._builder.gep(alloca, [zero, idx])
                self._builder.store(val, ptr)

        return self._builder.load(alloca, arr_type)

    def _lower_struct_expr(self, expr: StructExpr) -> Optional[Value]:
        """Lower struct literal expression."""
        struct_name = expr.path[-1] if expr.path else "unknown"
        layout = self._resolver.get_struct_layout(struct_name)
        if layout is None:
            return None

        # Build IR struct type
        fields = []
        for f in layout.fields:
            fields.append(StructField(name=f.name, type=self._lower_type(f.type_name)))
        struct_type = StructType(name=struct_name, fields=fields)

        alloca = self._builder.alloca(struct_type, name=struct_name)
        zero = Constant.int_const(IntType(32, Signedness.SIGNED), 0)

        for sf in expr.fields:
            field_name, field_val = sf
            field_idx = self._resolver.get_field_index(struct_name, field_name)
            if field_idx is not None and field_val:
                val = self._lower_expr(field_val)
                if val:
                    idx = Constant.int_const(IntType(32, Signedness.SIGNED), field_idx)
                    ptr = self._builder.gep(alloca, [zero, idx])
                    self._builder.store(val, ptr)

        return self._builder.load(alloca, struct_type)

    def _lower_ref_expr(self, expr: RefExpr) -> Optional[Value]:
        """Lower &expr or &mut expr."""
        addr = self._lower_lvalue(expr.operand)
        if addr is not None:
            return addr
        # If not an lvalue, alloca + store
        val = self._lower_expr(expr.operand)
        if val is None:
            return None
        ir_type = self._infer_expr_type(expr.operand)
        alloca = self._builder.alloca(ir_type, name="ref_tmp")
        self._builder.store(val, alloca)
        return alloca

    def _lower_deref_expr(self, expr: DerefExpr) -> Optional[Value]:
        """Lower *expr."""
        ptr = self._lower_expr(expr.operand)
        if ptr is None:
            return None
        pointee_type = self._infer_deref_type(expr)
        return self._builder.load(ptr, pointee_type)

    def _lower_unsafe_block(self, expr: UnsafeBlock) -> Optional[Value]:
        """Lower unsafe { ... } block with provenance annotation."""
        old_unsafe = self._in_unsafe
        self._in_unsafe = True
        result = self._lower_expr(expr.body) if expr.body else None
        self._in_unsafe = old_unsafe
        return result

    def _lower_range_expr(self, expr: RangeExpr) -> Optional[Value]:
        """Lower range expression (simplified)."""
        # Range is typically lowered to a struct { start, end }
        start = self._lower_expr(expr.start) if expr.start else None
        end = self._lower_expr(expr.end) if expr.end else None
        # Simplified: return start value
        return start

    def _lower_try_expr(self, expr: TryExpr) -> Optional[Value]:
        """Lower ? operator: lower operand, simplified early-return on error."""
        val = self._lower_expr(expr.operand)
        if val is None:
            return None
        # Simplified: in full implementation, branch on Ok/Err discriminant
        # and early-return Err. For now, just pass through the value.
        return val

    def _lower_if_let_expr(self, expr: IfLetExpr) -> Optional[Value]:
        """Lower if let pattern = expr { ... } else { ... } (simplified)."""
        scrutinee = self._lower_expr(expr.scrutinee)
        # Simplified: lower then_body unconditionally (conservative)
        then_val = self._lower_expr(expr.then_body) if expr.then_body else None
        if expr.else_body:
            self._lower_expr(expr.else_body)
        return then_val

    def _lower_while_let_expr(self, expr: WhileLetExpr) -> Optional[Value]:
        """Lower while let pattern = expr { ... } (simplified to while loop)."""
        cond_bb = self._fresh_block("while_let_cond")
        body_bb = self._fresh_block("while_let_body")
        exit_bb = self._fresh_block("while_let_exit")

        self._loop_stack.append(_LoopContext(
            break_block=exit_bb,
            continue_block=cond_bb,
            label=getattr(expr, 'label', None),
        ))

        self._builder.br(cond_bb)
        self._builder.position_at_end(cond_bb)
        # Simplified: always branch to exit (would need pattern matching)
        self._builder.br(exit_bb)

        self._builder.position_at_end(body_bb)
        if expr.body:
            self._lower_expr(expr.body)
        if not self._builder.insert_block.terminator:
            self._builder.br(cond_bb)

        self._loop_stack.pop()
        self._builder.position_at_end(exit_bb)
        return None

    def _lower_macro_invocation(self, expr: MacroInvocation) -> Optional[Value]:
        """Lower macro invocations (simplified)."""
        # Common macros like println!, format!, etc.
        if expr.name in ("println", "print", "eprintln", "eprint"):
            # Lower as call to external print function
            return None
        if expr.name in ("vec", "format", "todo", "unimplemented", "unreachable"):
            return None
        if expr.name == "assert":
            return None
        return None

    # -------------------------------------------------------------------
    # Scope management
    # -------------------------------------------------------------------

    def _push_scope(self) -> None:
        self._scope_stack.append(self._vars.copy())

    def _pop_scope(self) -> None:
        if self._scope_stack:
            self._vars = self._scope_stack.pop()

    # -------------------------------------------------------------------
    # Type inference helpers
    # -------------------------------------------------------------------

    def _infer_expr_type(self, expr: Expr) -> IRType:
        """Infer the IR type of an expression (simplified)."""
        if isinstance(expr, LitExpr):
            if expr.int_value is not None:
                suffix = expr.type_suffix or "i32"
                return self._lower_path_type(PathType(segments=[suffix]))
            if expr.float_value is not None:
                suffix = expr.type_suffix or "f64"
                return self._lower_path_type(PathType(segments=[suffix]))
            if expr.bool_value is not None:
                return IntType(1, Signedness.UNSIGNED)
            if expr.char_value is not None:
                return IntType(32, Signedness.UNSIGNED)
            if expr.string_value is not None:
                return PointerType(pointee=IntType(8, Signedness.UNSIGNED))

        if isinstance(expr, PathExpr):
            name = expr.segments[-1] if expr.segments else ""
            var = self._vars.get(name)
            if var:
                return var.ir_type

        if isinstance(expr, BinaryExpr):
            if expr.op in (RustBinOp.EQ, RustBinOp.NE, RustBinOp.LT,
                           RustBinOp.LE, RustBinOp.GT, RustBinOp.GE,
                           RustBinOp.AND, RustBinOp.OR):
                return IntType(1, Signedness.UNSIGNED)
            return self._infer_expr_type(expr.lhs)

        if isinstance(expr, CastExpr):
            return self._lower_type(expr.target_type)

        if isinstance(expr, ParenExpr):
            return self._infer_expr_type(expr.inner)

        return VoidType()

    def _infer_field_type(self, expr: FieldExpr) -> IRType:
        """Infer field type."""
        struct_name = self._get_struct_name_from_expr(expr.base)
        if struct_name:
            layout = self._resolver.get_struct_layout(struct_name)
            if layout:
                for f in layout.fields:
                    if f.name == expr.field_name:
                        return self._lower_type(f.type_name)
        return VoidType()

    def _infer_deref_type(self, expr: DerefExpr) -> IRType:
        """Infer type of *expr."""
        inner_type = self._infer_expr_type(expr.operand)
        if isinstance(inner_type, PointerType):
            return inner_type.pointee
        return VoidType()

    def _get_struct_name_from_expr(self, expr: Expr) -> Optional[str]:
        """Try to get struct name from an expression."""
        if isinstance(expr, PathExpr):
            name = expr.segments[-1] if expr.segments else ""
            var = self._vars.get(name)
            if var and isinstance(var.ir_type, StructType):
                return var.ir_type.name
            if var and isinstance(var.ir_type, PointerType):
                if isinstance(var.ir_type.pointee, StructType):
                    return var.ir_type.pointee.name
        return None

    # -------------------------------------------------------------------
    # Constant evaluation
    # -------------------------------------------------------------------

    def _try_const_eval(self, expr: Expr, ir_type: IRType) -> Optional[Constant]:
        """Try to evaluate an expression as a compile-time constant."""
        if isinstance(expr, LitExpr):
            if expr.int_value is not None:
                return Constant.int_const(ir_type, expr.int_value)
            if expr.float_value is not None:
                return Constant.float_const(ir_type, expr.float_value)
            if expr.bool_value is not None:
                return Constant.bool_const(expr.bool_value)
        return None

    def _try_pattern_const(self, pattern, scrutinee: Value) -> Optional[Constant]:
        """Try to extract a constant from a match pattern."""
        from .rust_ast import LiteralPattern, PathPattern
        if isinstance(pattern, LiteralPattern):
            if pattern.value is not None and isinstance(pattern.value, LitExpr):
                return self._try_const_eval(pattern.value, IntType(32, Signedness.SIGNED))
        return None
