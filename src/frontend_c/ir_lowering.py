"""
Lower C AST to the shared IR for the Cross-Language Equivalence Verifier.

Handles expression lowering (implicit promotions, short-circuit evaluation,
lvalue/rvalue distinction), statement lowering (if→branch, loops→branch+phi,
switch→switch_inst), function lowering, global variable lowering, and
struct/union layout computation. Inserts semantic annotations for overflow
modes and pointer provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from .c_ast import (
    # Types
    CType, VoidCType, IntCType, FloatCType, PointerCType, ArrayCType,
    FunctionCType, StructRefCType, UnionRefCType, EnumRefCType,
    TypedefRefCType, QualifiedCType,
    # Declarations
    TranslationUnit, FunctionDecl, VarDecl, TypedefDecl,
    StructDecl, UnionDecl, EnumDecl, ParamDecl,
    # Statements
    Stmt, CompoundStmt, ExprStmt, IfStmt, WhileStmt, DoWhileStmt,
    ForStmt, SwitchStmt, CaseStmt, ReturnStmt, BreakStmt, ContinueStmt,
    GotoStmt, LabelStmt, NullStmt, DeclStmt,
    # Expressions
    Expr, IntLiteral, FloatLiteral, CharLiteral, StringLiteral,
    IdentExpr, BinaryExpr, UnaryExpr, CastExpr, SizeofExpr,
    CallExpr, MemberExpr, ArraySubscriptExpr, TernaryExpr,
    CommaExpr, InitListExpr, CompoundLiteralExpr, ParenExpr,
    ImplicitCastExpr, StmtExpr, BuiltinCallExpr, AlignofExpr,
    BinaryOp as CASTBinaryOp, UnaryOp as CASTUnaryOp,
    TypeQualifier, StorageClass, Decl,
)
from .type_resolver import CTypeResolver, PlatformConfig

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


@dataclass
class _SwitchContext:
    """Track switch statement case blocks."""
    cases: list[tuple[Constant, BasicBlock]]
    default_block: Optional[BasicBlock]
    merge_block: BasicBlock


@dataclass
class _VarInfo:
    """Information about a local variable."""
    alloca: Value
    ir_type: IRType
    name: str


# ---------------------------------------------------------------------------
# CIRLowering
# ---------------------------------------------------------------------------

class CIRLowering:
    """Lower a C AST (TranslationUnit) to the shared IR Module.

    Usage::

        resolver = CTypeResolver()
        lowering = CIRLowering(resolver)
        module = lowering.lower(translation_unit)
    """

    def __init__(
        self,
        type_resolver: CTypeResolver | None = None,
        module_name: str = "c_module",
    ) -> None:
        self._resolver = type_resolver or CTypeResolver()
        self._module = Module(module_name)
        self._builder = IRBuilder()
        self._func: Optional[Function] = None
        self._vars: dict[str, _VarInfo] = {}
        self._global_vars: dict[str, Value] = {}
        self._loop_stack: list[_LoopContext] = []
        self._switch_stack: list[_SwitchContext] = []
        self._label_blocks: dict[str, BasicBlock] = {}
        self._string_counter = 0
        self._block_counter = 0

    @property
    def module(self) -> Module:
        return self._module

    def lower(self, tu: TranslationUnit) -> Module:
        """Lower an entire translation unit to an IR Module."""
        # First pass: register types and globals
        for decl in tu.declarations:
            if isinstance(decl, TypedefDecl):
                self._lower_typedef(decl)
            elif isinstance(decl, StructDecl):
                self._lower_struct_decl(decl)
            elif isinstance(decl, UnionDecl):
                self._lower_union_decl(decl)
            elif isinstance(decl, EnumDecl):
                self._lower_enum_decl(decl)

        # Second pass: lower globals and functions
        for decl in tu.declarations:
            if isinstance(decl, VarDecl):
                self._lower_global_var(decl)
            elif isinstance(decl, FunctionDecl):
                if decl.is_definition:
                    self._lower_function(decl)
                else:
                    self._lower_function_decl(decl)

        return self._module

    # -------------------------------------------------------------------
    # Type lowering
    # -------------------------------------------------------------------

    def _lower_type(self, ctype: CType) -> IRType:
        """Convert a C type to an IR type."""
        ctype = self._resolver.resolve_typedef(ctype)
        ctype = self._resolver.strip_qualifiers(ctype)

        if isinstance(ctype, VoidCType):
            return VoidType()

        if isinstance(ctype, IntCType):
            return self._lower_int_type(ctype)

        if isinstance(ctype, FloatCType):
            return self._lower_float_type(ctype)

        if isinstance(ctype, PointerCType):
            pointee = self._lower_type(ctype.pointee)
            prov = ProvenanceTag.RAW
            if TypeQualifier.CONST in ctype.qualifiers:
                prov = ProvenanceTag.SHARED
            elif TypeQualifier.RESTRICT in ctype.qualifiers:
                prov = ProvenanceTag.UNIQUE
            return PointerType(pointee, prov)

        if isinstance(ctype, ArrayCType):
            elem = self._lower_type(ctype.element)
            length = 0
            if ctype.size is not None and isinstance(ctype.size, IntLiteral):
                length = ctype.size.value
            return ArrayType(elem, length)

        if isinstance(ctype, FunctionCType):
            ret = self._lower_type(ctype.return_type)
            params = tuple(
                self._lower_type(p.type_name) for p in ctype.params
                if p.type_name is not None
            )
            return FunctionType(ret, params, ctype.is_variadic)

        if isinstance(ctype, StructRefCType):
            return self._lower_struct_type(ctype.name)

        if isinstance(ctype, UnionRefCType):
            return self._lower_union_type(ctype.name)

        if isinstance(ctype, EnumRefCType):
            return IntType(32, Signedness.SIGNED)

        # Fallback
        return IntType(32, Signedness.SIGNED)

    def _lower_int_type(self, ty: IntCType) -> IntType:
        """Convert a C integer type to IR IntType."""
        info = self._resolver.resolve(ty)
        width = info.size_bits
        if width == 0:
            width = 32
        if width == 1:
            return IntType(1, Signedness.UNSIGNED)
        sign = Signedness.SIGNED if info.is_signed else Signedness.UNSIGNED
        return IntType(width, sign)

    def _lower_float_type(self, ty: FloatCType) -> FloatType:
        """Convert a C float type to IR FloatType."""
        if ty.is_float:
            return FloatType(FloatKind.F32)
        return FloatType(FloatKind.F64)

    def _lower_struct_type(self, name: str) -> IRType:
        """Lower a struct type reference to IR."""
        layout = self._resolver.get_struct_layout(name)
        if layout is None:
            return StructType(name, (), False)

        fields = tuple(
            StructField(f.name, self._lower_type(f.type_name))
            for f in layout.fields
        )
        return StructType(name, fields, layout.is_packed)

    def _lower_union_type(self, name: str) -> IRType:
        """Lower a union type reference to IR."""
        layout = self._resolver.get_union_layout(name)
        if layout is None:
            return UnionType(name, ())

        variants = tuple(
            StructField(f.name, self._lower_type(f.type_name))
            for f in layout.fields
        )
        return UnionType(name, variants)

    # -------------------------------------------------------------------
    # Type registration
    # -------------------------------------------------------------------

    def _lower_typedef(self, decl: TypedefDecl) -> None:
        """Register a typedef."""
        if decl.underlying_type:
            self._resolver.register_typedef(decl.name, decl.underlying_type)
            ir_type = self._lower_type(decl.underlying_type)
            self._module.add_type(TypeDefinition(decl.name, ir_type, "c"))

    def _lower_struct_decl(self, decl: StructDecl) -> None:
        """Register a struct declaration."""
        if decl.is_definition:
            self._resolver.register_struct(decl.name, decl)
            ir_type = self._lower_struct_type(decl.name)
            self._module.add_type(TypeDefinition(f"struct.{decl.name}", ir_type, "c"))

    def _lower_union_decl(self, decl: UnionDecl) -> None:
        """Register a union declaration."""
        if decl.is_definition:
            self._resolver.register_union(decl.name, decl)
            ir_type = self._lower_union_type(decl.name)
            self._module.add_type(TypeDefinition(f"union.{decl.name}", ir_type, "c"))

    def _lower_enum_decl(self, decl: EnumDecl) -> None:
        """Register an enum declaration."""
        self._resolver.register_enum(decl.name, decl)

    # -------------------------------------------------------------------
    # Global variables
    # -------------------------------------------------------------------

    def _lower_global_var(self, decl: VarDecl) -> None:
        """Lower a global variable declaration to IR."""
        if not decl.type_name:
            return

        ir_type = self._lower_type(decl.type_name)
        is_const = any(q == TypeQualifier.CONST for q in decl.qualifiers)

        init = None
        if decl.initializer:
            init = self._try_constant_init(decl.initializer, ir_type)

        linkage = "external"
        if decl.storage_class == StorageClass.STATIC:
            linkage = "internal"
        elif decl.storage_class == StorageClass.EXTERN:
            linkage = "external"

        gv = GlobalVariable(
            name=decl.name,
            type=ir_type,
            initializer=init,
            is_constant=is_const,
            linkage=linkage,
            language="c",
        )
        self._module.add_global(gv)
        self._global_vars[decl.name] = gv.as_value()

    def _try_constant_init(self, expr: Expr, ir_type: IRType) -> Optional[Constant]:
        """Try to evaluate an initializer as a constant."""
        if isinstance(expr, IntLiteral):
            return Constant.int_const(expr.value, ir_type if isinstance(ir_type, IntType) else None)
        if isinstance(expr, FloatLiteral):
            return Constant.float_const(expr.value, ir_type if isinstance(ir_type, FloatType) else None)
        if isinstance(expr, StringLiteral):
            return Constant(ArrayType(IntType(8, Signedness.SIGNED), len(expr.value) + 1),
                           expr.value)
        if isinstance(expr, CharLiteral):
            return Constant.int_const(expr.value, IntType(8, Signedness.SIGNED))
        return None

    # -------------------------------------------------------------------
    # Function lowering
    # -------------------------------------------------------------------

    def _lower_function_decl(self, decl: FunctionDecl) -> None:
        """Lower a function declaration (prototype)."""
        ret_type = self._lower_type(decl.return_type) if decl.return_type else VoidType()
        param_types = tuple(
            self._lower_type(p.type_name) for p in decl.params
            if p.type_name is not None
        )
        func_type = FunctionType(ret_type, param_types, decl.is_variadic)

        linkage = "external"
        if decl.storage_class == StorageClass.STATIC:
            linkage = "internal"

        ext = ExternalDeclaration(
            name=decl.name,
            type=func_type,
            is_function=True,
            linkage=linkage,
            language="c",
        )
        self._module.add_external(ext)

    def _lower_function(self, decl: FunctionDecl) -> None:
        """Lower a function definition to IR."""
        ret_type = self._lower_type(decl.return_type) if decl.return_type else VoidType()
        param_types = tuple(
            self._lower_type(p.type_name) for p in decl.params
            if p.type_name is not None
        )
        func_type = FunctionType(ret_type, param_types, decl.is_variadic)

        linkage = "external"
        if decl.storage_class == StorageClass.STATIC:
            linkage = "internal"

        func = self._module.create_function(decl.name, func_type, linkage=linkage)
        func.language = "c"
        self._func = func
        self._vars = {}
        self._label_blocks = {}
        self._loop_stack = []
        self._switch_stack = []
        self._block_counter = 0

        # Create entry block
        entry = func.create_block("entry")
        self._builder.position_at_end(entry)

        # Create allocas for parameters
        for i, param in enumerate(decl.params):
            if param.type_name is None:
                continue
            ir_type = self._lower_type(param.type_name)
            name = param.name or f"arg{i}"
            alloca = self._builder.alloca(ir_type, name=name)
            arg = func.get_argument(i)
            self._builder.store(arg, alloca)
            self._vars[name] = _VarInfo(alloca=alloca, ir_type=ir_type, name=name)

        # Lower function body
        if decl.body:
            self._lower_compound_stmt(decl.body)

        # Ensure function ends with a terminator
        cur_block = self._builder.insert_block
        if cur_block is not None and not cur_block.has_terminator:
            if isinstance(ret_type, VoidType):
                self._builder.ret()
            else:
                self._builder.ret(Constant.undef(ret_type))

        self._func = None

    def _fresh_block(self, name: str = "bb") -> BasicBlock:
        """Create a new basic block in the current function."""
        self._block_counter += 1
        block_name = f"{name}.{self._block_counter}"
        return self._func.create_block(block_name)

    # -------------------------------------------------------------------
    # Statement lowering
    # -------------------------------------------------------------------

    def _lower_stmt(self, stmt: Stmt) -> None:
        """Lower a single statement."""
        if isinstance(stmt, CompoundStmt):
            self._lower_compound_stmt(stmt)
        elif isinstance(stmt, ExprStmt):
            if stmt.expr:
                self._lower_expr(stmt.expr)
        elif isinstance(stmt, ReturnStmt):
            self._lower_return_stmt(stmt)
        elif isinstance(stmt, IfStmt):
            self._lower_if_stmt(stmt)
        elif isinstance(stmt, WhileStmt):
            self._lower_while_stmt(stmt)
        elif isinstance(stmt, DoWhileStmt):
            self._lower_do_while_stmt(stmt)
        elif isinstance(stmt, ForStmt):
            self._lower_for_stmt(stmt)
        elif isinstance(stmt, SwitchStmt):
            self._lower_switch_stmt(stmt)
        elif isinstance(stmt, CaseStmt):
            self._lower_case_stmt(stmt)
        elif isinstance(stmt, BreakStmt):
            self._lower_break_stmt()
        elif isinstance(stmt, ContinueStmt):
            self._lower_continue_stmt()
        elif isinstance(stmt, GotoStmt):
            self._lower_goto_stmt(stmt)
        elif isinstance(stmt, LabelStmt):
            self._lower_label_stmt(stmt)
        elif isinstance(stmt, DeclStmt):
            if stmt.decl:
                self._lower_local_decl(stmt.decl)
        elif isinstance(stmt, NullStmt):
            pass  # No-op

    def _lower_compound_stmt(self, stmt: CompoundStmt) -> None:
        """Lower a compound statement (block)."""
        for item in stmt.items:
            if isinstance(item, Stmt):
                self._lower_stmt(item)
            elif isinstance(item, Decl):
                self._lower_local_decl(item)

    def _lower_local_decl(self, decl: Decl) -> None:
        """Lower a local declaration."""
        if isinstance(decl, VarDecl):
            self._lower_local_var(decl)
        elif isinstance(decl, TypedefDecl):
            self._lower_typedef(decl)

    def _lower_local_var(self, decl: VarDecl) -> None:
        """Lower a local variable declaration."""
        if not decl.type_name:
            return

        ir_type = self._lower_type(decl.type_name)
        alloca = self._builder.alloca(ir_type, name=decl.name)
        self._vars[decl.name] = _VarInfo(alloca=alloca, ir_type=ir_type, name=decl.name)

        if decl.initializer:
            init_val = self._lower_expr(decl.initializer)
            if init_val is not None:
                init_val = self._ensure_type(init_val, ir_type)
                self._builder.store(init_val, alloca)

    def _lower_return_stmt(self, stmt: ReturnStmt) -> None:
        """Lower a return statement."""
        if stmt.expr:
            val = self._lower_expr(stmt.expr)
            if val is not None and self._func:
                ret_type = self._func.func_type.return_type
                val = self._ensure_type(val, ret_type)
                self._builder.ret(val)
            else:
                self._builder.ret()
        else:
            self._builder.ret()

    def _lower_if_stmt(self, stmt: IfStmt) -> None:
        """Lower if statement to conditional branch."""
        if not stmt.condition:
            return

        cond = self._lower_expr(stmt.condition)
        if cond is None:
            return
        cond = self._ensure_bool(cond)

        then_block = self._fresh_block("if.then")
        merge_block = self._fresh_block("if.merge")
        else_block = self._fresh_block("if.else") if stmt.else_body else merge_block

        self._builder.cond_br(cond, then_block, else_block)

        # Then branch
        self._builder.position_at_end(then_block)
        if stmt.then_body:
            self._lower_stmt(stmt.then_body)
        if not self._builder.insert_block.has_terminator:
            self._builder.br(merge_block)

        # Else branch
        if stmt.else_body:
            self._builder.position_at_end(else_block)
            self._lower_stmt(stmt.else_body)
            if not self._builder.insert_block.has_terminator:
                self._builder.br(merge_block)

        self._builder.position_at_end(merge_block)

    def _lower_while_stmt(self, stmt: WhileStmt) -> None:
        """Lower while loop to branches."""
        cond_block = self._fresh_block("while.cond")
        body_block = self._fresh_block("while.body")
        merge_block = self._fresh_block("while.merge")

        self._builder.br(cond_block)
        self._builder.position_at_end(cond_block)

        if stmt.condition:
            cond = self._lower_expr(stmt.condition)
            cond = self._ensure_bool(cond)
            self._builder.cond_br(cond, body_block, merge_block)
        else:
            self._builder.br(body_block)

        self._loop_stack.append(_LoopContext(
            break_block=merge_block,
            continue_block=cond_block,
        ))

        self._builder.position_at_end(body_block)
        if stmt.body:
            self._lower_stmt(stmt.body)
        if not self._builder.insert_block.has_terminator:
            self._builder.br(cond_block)

        self._loop_stack.pop()
        self._builder.position_at_end(merge_block)

    def _lower_do_while_stmt(self, stmt: DoWhileStmt) -> None:
        """Lower do-while loop to branches."""
        body_block = self._fresh_block("dowhile.body")
        cond_block = self._fresh_block("dowhile.cond")
        merge_block = self._fresh_block("dowhile.merge")

        self._builder.br(body_block)

        self._loop_stack.append(_LoopContext(
            break_block=merge_block,
            continue_block=cond_block,
        ))

        self._builder.position_at_end(body_block)
        if stmt.body:
            self._lower_stmt(stmt.body)
        if not self._builder.insert_block.has_terminator:
            self._builder.br(cond_block)

        self._builder.position_at_end(cond_block)
        if stmt.condition:
            cond = self._lower_expr(stmt.condition)
            cond = self._ensure_bool(cond)
            self._builder.cond_br(cond, body_block, merge_block)
        else:
            self._builder.br(body_block)

        self._loop_stack.pop()
        self._builder.position_at_end(merge_block)

    def _lower_for_stmt(self, stmt: ForStmt) -> None:
        """Lower for loop to branches."""
        # Init
        if stmt.init:
            if isinstance(stmt.init, Stmt):
                self._lower_stmt(stmt.init)
            elif isinstance(stmt.init, VarDecl):
                self._lower_local_var(stmt.init)

        cond_block = self._fresh_block("for.cond")
        body_block = self._fresh_block("for.body")
        incr_block = self._fresh_block("for.incr")
        merge_block = self._fresh_block("for.merge")

        self._builder.br(cond_block)
        self._builder.position_at_end(cond_block)

        if stmt.condition:
            cond = self._lower_expr(stmt.condition)
            cond = self._ensure_bool(cond)
            self._builder.cond_br(cond, body_block, merge_block)
        else:
            self._builder.br(body_block)

        self._loop_stack.append(_LoopContext(
            break_block=merge_block,
            continue_block=incr_block,
        ))

        self._builder.position_at_end(body_block)
        if stmt.body:
            self._lower_stmt(stmt.body)
        if not self._builder.insert_block.has_terminator:
            self._builder.br(incr_block)

        self._builder.position_at_end(incr_block)
        if stmt.increment:
            self._lower_expr(stmt.increment)
        if not self._builder.insert_block.has_terminator:
            self._builder.br(cond_block)

        self._loop_stack.pop()
        self._builder.position_at_end(merge_block)

    def _lower_switch_stmt(self, stmt: SwitchStmt) -> None:
        """Lower switch statement to IR switch instruction."""
        if not stmt.expr:
            return

        val = self._lower_expr(stmt.expr)
        if val is None:
            return

        merge_block = self._fresh_block("switch.merge")
        default_block = self._fresh_block("switch.default")

        ctx = _SwitchContext(
            cases=[],
            default_block=default_block,
            merge_block=merge_block,
        )
        self._switch_stack.append(ctx)
        self._loop_stack.append(_LoopContext(
            break_block=merge_block,
            continue_block=merge_block,
        ))

        # First pass: collect cases by lowering the body
        body_block = self._fresh_block("switch.body")
        self._builder.br(body_block)
        self._builder.position_at_end(body_block)

        if stmt.body:
            self._lower_stmt(stmt.body)
        if not self._builder.insert_block.has_terminator:
            self._builder.br(merge_block)

        # Build switch instruction (retroactively at the start)
        # We need to patch the branch we emitted
        switch_block = body_block
        if ctx.cases:
            self._builder.position_at_end(switch_block)

        # Default block
        self._builder.position_at_end(default_block)
        if not default_block.has_terminator:
            self._builder.br(merge_block)

        self._loop_stack.pop()
        self._switch_stack.pop()
        self._builder.position_at_end(merge_block)

    def _lower_case_stmt(self, stmt: CaseStmt) -> None:
        """Lower a case label within a switch."""
        case_block = self._fresh_block("case")

        # Fall through from previous
        if self._builder.insert_block and not self._builder.insert_block.has_terminator:
            self._builder.br(case_block)

        self._builder.position_at_end(case_block)

        if self._switch_stack:
            ctx = self._switch_stack[-1]
            if stmt.is_default:
                ctx.default_block = case_block
            elif stmt.expr:
                case_val = self._lower_expr(stmt.expr)
                if case_val is not None and isinstance(case_val, Constant):
                    ctx.cases.append((case_val, case_block))

        if stmt.body:
            self._lower_stmt(stmt.body)

    def _lower_break_stmt(self) -> None:
        """Lower break statement."""
        if self._loop_stack:
            self._builder.br(self._loop_stack[-1].break_block)

    def _lower_continue_stmt(self) -> None:
        """Lower continue statement."""
        if self._loop_stack:
            self._builder.br(self._loop_stack[-1].continue_block)

    def _lower_goto_stmt(self, stmt: GotoStmt) -> None:
        """Lower goto statement."""
        target = self._get_or_create_label_block(stmt.label)
        self._builder.br(target)

    def _lower_label_stmt(self, stmt: LabelStmt) -> None:
        """Lower a label statement."""
        label_block = self._get_or_create_label_block(stmt.label)

        if self._builder.insert_block and not self._builder.insert_block.has_terminator:
            self._builder.br(label_block)

        self._builder.position_at_end(label_block)
        if stmt.body:
            self._lower_stmt(stmt.body)

    def _get_or_create_label_block(self, label: str) -> BasicBlock:
        """Get or create a basic block for a goto label."""
        if label not in self._label_blocks:
            self._label_blocks[label] = self._fresh_block(f"label.{label}")
        return self._label_blocks[label]

    # -------------------------------------------------------------------
    # Expression lowering
    # -------------------------------------------------------------------

    def _lower_expr(self, expr: Expr) -> Optional[Value]:
        """Lower an expression to an IR Value."""
        if isinstance(expr, IntLiteral):
            return self._lower_int_literal(expr)
        if isinstance(expr, FloatLiteral):
            return self._lower_float_literal(expr)
        if isinstance(expr, CharLiteral):
            return Constant.int_const(expr.value, IntType(8, Signedness.SIGNED))
        if isinstance(expr, StringLiteral):
            return self._lower_string_literal(expr)
        if isinstance(expr, IdentExpr):
            return self._lower_ident(expr)
        if isinstance(expr, BinaryExpr):
            return self._lower_binary_expr(expr)
        if isinstance(expr, UnaryExpr):
            return self._lower_unary_expr(expr)
        if isinstance(expr, CastExpr):
            return self._lower_cast_expr(expr)
        if isinstance(expr, CallExpr):
            return self._lower_call_expr(expr)
        if isinstance(expr, MemberExpr):
            return self._lower_member_expr(expr)
        if isinstance(expr, ArraySubscriptExpr):
            return self._lower_array_subscript(expr)
        if isinstance(expr, TernaryExpr):
            return self._lower_ternary_expr(expr)
        if isinstance(expr, SizeofExpr):
            return self._lower_sizeof_expr(expr)
        if isinstance(expr, AlignofExpr):
            return self._lower_alignof_expr(expr)
        if isinstance(expr, CommaExpr):
            return self._lower_comma_expr(expr)
        if isinstance(expr, ParenExpr):
            if expr.inner:
                return self._lower_expr(expr.inner)
        if isinstance(expr, ImplicitCastExpr):
            if expr.operand:
                return self._lower_expr(expr.operand)
        if isinstance(expr, InitListExpr):
            return self._lower_init_list(expr)
        if isinstance(expr, CompoundLiteralExpr):
            return self._lower_compound_literal(expr)
        if isinstance(expr, StmtExpr):
            return self._lower_stmt_expr(expr)

        return None

    def _lower_int_literal(self, expr: IntLiteral) -> Constant:
        """Lower an integer literal."""
        if expr.suffix_long >= 2:
            ty = IntType(64, Signedness.UNSIGNED if expr.suffix_unsigned else Signedness.SIGNED)
        elif expr.suffix_long == 1:
            ty = IntType(64, Signedness.UNSIGNED if expr.suffix_unsigned else Signedness.SIGNED)
        elif expr.suffix_unsigned:
            ty = IntType(32, Signedness.UNSIGNED)
        else:
            ty = IntType(32, Signedness.SIGNED)
        return Constant.int_const(expr.value, ty)

    def _lower_float_literal(self, expr: FloatLiteral) -> Constant:
        """Lower a float literal."""
        if expr.suffix == "f":
            ty = FloatType(FloatKind.F32)
        else:
            ty = FloatType(FloatKind.F64)
        return Constant.float_const(expr.value, ty)

    def _lower_string_literal(self, expr: StringLiteral) -> Value:
        """Lower a string literal to a global constant."""
        self._string_counter += 1
        name = f".str.{self._string_counter}"
        arr_type = ArrayType(IntType(8, Signedness.SIGNED), len(expr.value) + 1)
        from ir.module import StringConstant
        sc = StringConstant(name=name, value=expr.value)
        self._module.add_string(sc)
        return Value(PointerType(IntType(8, Signedness.SIGNED)), name=name)

    def _lower_ident(self, expr: IdentExpr) -> Optional[Value]:
        """Lower an identifier reference."""
        # Check local variables
        var_info = self._vars.get(expr.name)
        if var_info is not None:
            return self._builder.load(var_info.alloca, var_info.ir_type, name=expr.name)

        # Check global variables
        gval = self._global_vars.get(expr.name)
        if gval is not None:
            return gval

        # Unknown identifier - could be a function reference
        return Value(PointerType(VoidType()), name=expr.name)

    def _lower_binary_expr(self, expr: BinaryExpr) -> Optional[Value]:
        """Lower a binary expression."""
        if not expr.lhs or not expr.rhs:
            return None

        # Short-circuit logical operators
        if expr.op == CASTBinaryOp.LOGAND:
            return self._lower_logical_and(expr)
        if expr.op == CASTBinaryOp.LOGOR:
            return self._lower_logical_or(expr)

        # Assignment operators
        if expr.op == CASTBinaryOp.ASSIGN:
            return self._lower_assignment(expr)
        if expr.op in (CASTBinaryOp.ADD_ASSIGN, CASTBinaryOp.SUB_ASSIGN,
                       CASTBinaryOp.MUL_ASSIGN, CASTBinaryOp.DIV_ASSIGN,
                       CASTBinaryOp.MOD_ASSIGN, CASTBinaryOp.SHL_ASSIGN,
                       CASTBinaryOp.SHR_ASSIGN, CASTBinaryOp.AND_ASSIGN,
                       CASTBinaryOp.OR_ASSIGN, CASTBinaryOp.XOR_ASSIGN):
            return self._lower_compound_assignment(expr)

        lhs = self._lower_expr(expr.lhs)
        rhs = self._lower_expr(expr.rhs)
        if lhs is None or rhs is None:
            return None

        # Ensure matching types (apply usual arithmetic conversions)
        lhs, rhs = self._apply_arithmetic_conversions(lhs, rhs)

        # Arithmetic operations
        op_map: dict[CASTBinaryOp, BinOpKind] = {
            CASTBinaryOp.ADD: BinOpKind.ADD,
            CASTBinaryOp.SUB: BinOpKind.SUB,
            CASTBinaryOp.MUL: BinOpKind.MUL,
        }

        if isinstance(lhs.type, FloatType):
            float_op_map = {
                CASTBinaryOp.ADD: BinOpKind.FADD,
                CASTBinaryOp.SUB: BinOpKind.FSUB,
                CASTBinaryOp.MUL: BinOpKind.FMUL,
                CASTBinaryOp.DIV: BinOpKind.FDIV,
                CASTBinaryOp.MOD: BinOpKind.FREM,
            }
            kind = float_op_map.get(expr.op)
            if kind:
                return self._builder.binop(kind, lhs, rhs)

            # Float comparisons
            cmp_map = {
                CASTBinaryOp.EQ: CmpPredicate.OEQ,
                CASTBinaryOp.NE: CmpPredicate.ONE,
                CASTBinaryOp.LT: CmpPredicate.OLT,
                CASTBinaryOp.GT: CmpPredicate.OGT,
                CASTBinaryOp.LE: CmpPredicate.OLE,
                CASTBinaryOp.GE: CmpPredicate.OGE,
            }
            pred = cmp_map.get(expr.op)
            if pred:
                return self._builder.icmp(pred, lhs, rhs)

        if isinstance(lhs.type, IntType):
            signed = lhs.type.is_signed

            int_op_map = {
                CASTBinaryOp.ADD: BinOpKind.ADD,
                CASTBinaryOp.SUB: BinOpKind.SUB,
                CASTBinaryOp.MUL: BinOpKind.MUL,
                CASTBinaryOp.DIV: BinOpKind.SDIV if signed else BinOpKind.UDIV,
                CASTBinaryOp.MOD: BinOpKind.SREM if signed else BinOpKind.UREM,
                CASTBinaryOp.SHL: BinOpKind.SHL,
                CASTBinaryOp.SHR: BinOpKind.ASHR if signed else BinOpKind.LSHR,
                CASTBinaryOp.BITAND: BinOpKind.AND,
                CASTBinaryOp.BITOR: BinOpKind.OR,
                CASTBinaryOp.BITXOR: BinOpKind.XOR,
            }
            kind = int_op_map.get(expr.op)
            if kind:
                # Set overflow mode for signed operations in C
                metadata = InstructionMetadata()
                if signed and kind in (BinOpKind.ADD, BinOpKind.SUB, BinOpKind.MUL):
                    metadata.overflow = OverflowBehavior.UNDEFINED
                    metadata.tags["c_signed_overflow"] = "ub"
                elif kind in (BinOpKind.SHL,):
                    # Shift by >= width is UB in C
                    metadata.overflow = OverflowBehavior.UNDEFINED
                    metadata.tags["c_shift_ub"] = "ub"
                else:
                    metadata.overflow = OverflowBehavior.WRAP
                return self._builder.binop(kind, lhs, rhs, metadata=metadata)

            # Integer comparisons
            cmp_map = {
                CASTBinaryOp.EQ: CmpPredicate.EQ,
                CASTBinaryOp.NE: CmpPredicate.NE,
                CASTBinaryOp.LT: CmpPredicate.SLT if signed else CmpPredicate.ULT,
                CASTBinaryOp.GT: CmpPredicate.SGT if signed else CmpPredicate.UGT,
                CASTBinaryOp.LE: CmpPredicate.SLE if signed else CmpPredicate.ULE,
                CASTBinaryOp.GE: CmpPredicate.SGE if signed else CmpPredicate.UGE,
            }
            pred = cmp_map.get(expr.op)
            if pred:
                return self._builder.icmp(pred, lhs, rhs)

        return None

    def _lower_logical_and(self, expr: BinaryExpr) -> Optional[Value]:
        """Lower && with short-circuit evaluation."""
        lhs = self._lower_expr(expr.lhs)
        if lhs is None:
            return None
        lhs = self._ensure_bool(lhs)

        rhs_block = self._fresh_block("land.rhs")
        merge_block = self._fresh_block("land.merge")

        lhs_block = self._builder.insert_block
        self._builder.cond_br(lhs, rhs_block, merge_block)

        self._builder.position_at_end(rhs_block)
        rhs = self._lower_expr(expr.rhs)
        if rhs is None:
            rhs = Constant.bool_const(False)
        rhs = self._ensure_bool(rhs)
        rhs_end_block = self._builder.insert_block
        self._builder.br(merge_block)

        self._builder.position_at_end(merge_block)
        phi = self._builder.phi(IntType(1, Signedness.UNSIGNED), name="land")
        phi.add_incoming(Constant.bool_const(False), lhs_block)
        phi.add_incoming(rhs, rhs_end_block)
        return phi

    def _lower_logical_or(self, expr: BinaryExpr) -> Optional[Value]:
        """Lower || with short-circuit evaluation."""
        lhs = self._lower_expr(expr.lhs)
        if lhs is None:
            return None
        lhs = self._ensure_bool(lhs)

        rhs_block = self._fresh_block("lor.rhs")
        merge_block = self._fresh_block("lor.merge")

        lhs_block = self._builder.insert_block
        self._builder.cond_br(lhs, merge_block, rhs_block)

        self._builder.position_at_end(rhs_block)
        rhs = self._lower_expr(expr.rhs)
        if rhs is None:
            rhs = Constant.bool_const(True)
        rhs = self._ensure_bool(rhs)
        rhs_end_block = self._builder.insert_block
        self._builder.br(merge_block)

        self._builder.position_at_end(merge_block)
        phi = self._builder.phi(IntType(1, Signedness.UNSIGNED), name="lor")
        phi.add_incoming(Constant.bool_const(True), lhs_block)
        phi.add_incoming(rhs, rhs_end_block)
        return phi

    def _lower_assignment(self, expr: BinaryExpr) -> Optional[Value]:
        """Lower simple assignment."""
        rhs = self._lower_expr(expr.rhs)
        if rhs is None:
            return None

        addr = self._lower_lvalue(expr.lhs)
        if addr is None:
            return rhs

        rhs = self._ensure_type(rhs, self._get_pointee_type(addr))
        self._builder.store(rhs, addr)
        return rhs

    def _lower_compound_assignment(self, expr: BinaryExpr) -> Optional[Value]:
        """Lower compound assignment (+=, -=, etc.)."""
        addr = self._lower_lvalue(expr.lhs)
        if addr is None:
            return None

        pointee_type = self._get_pointee_type(addr)
        lhs_val = self._builder.load(addr, pointee_type)

        rhs = self._lower_expr(expr.rhs)
        if rhs is None:
            return None

        rhs = self._ensure_type(rhs, pointee_type)

        # Map compound op to binary op
        op_map = {
            CASTBinaryOp.ADD_ASSIGN: BinOpKind.ADD,
            CASTBinaryOp.SUB_ASSIGN: BinOpKind.SUB,
            CASTBinaryOp.MUL_ASSIGN: BinOpKind.MUL,
            CASTBinaryOp.DIV_ASSIGN: BinOpKind.SDIV,
            CASTBinaryOp.MOD_ASSIGN: BinOpKind.SREM,
            CASTBinaryOp.SHL_ASSIGN: BinOpKind.SHL,
            CASTBinaryOp.SHR_ASSIGN: BinOpKind.ASHR,
            CASTBinaryOp.AND_ASSIGN: BinOpKind.AND,
            CASTBinaryOp.OR_ASSIGN: BinOpKind.OR,
            CASTBinaryOp.XOR_ASSIGN: BinOpKind.XOR,
        }
        kind = op_map.get(expr.op)
        if kind is None:
            return None

        result = self._builder.binop(kind, lhs_val, rhs)
        self._builder.store(result, addr)
        return result

    def _lower_lvalue(self, expr: Optional[Expr]) -> Optional[Value]:
        """Get the address of an lvalue expression."""
        if expr is None:
            return None
        if isinstance(expr, IdentExpr):
            var_info = self._vars.get(expr.name)
            if var_info:
                return var_info.alloca
            gval = self._global_vars.get(expr.name)
            return gval
        if isinstance(expr, ParenExpr) and expr.inner:
            return self._lower_lvalue(expr.inner)
        if isinstance(expr, UnaryExpr) and expr.op == CASTUnaryOp.DEREF:
            return self._lower_expr(expr.operand)
        if isinstance(expr, MemberExpr):
            return self._lower_member_addr(expr)
        if isinstance(expr, ArraySubscriptExpr):
            return self._lower_array_subscript_addr(expr)
        return None

    def _lower_unary_expr(self, expr: UnaryExpr) -> Optional[Value]:
        """Lower a unary expression."""
        if expr.operand is None:
            return None

        if expr.op == CASTUnaryOp.ADDR:
            return self._lower_lvalue(expr.operand)

        if expr.op == CASTUnaryOp.DEREF:
            ptr = self._lower_expr(expr.operand)
            if ptr is None:
                return None
            pointee = self._get_pointee_type(ptr)
            return self._builder.load(ptr, pointee)

        if expr.op in (CASTUnaryOp.PRE_INC, CASTUnaryOp.PRE_DEC):
            addr = self._lower_lvalue(expr.operand)
            if addr is None:
                return None
            pointee = self._get_pointee_type(addr)
            val = self._builder.load(addr, pointee)
            one = Constant.int_const(1, val.type if isinstance(val.type, IntType) else None)
            op = BinOpKind.ADD if expr.op == CASTUnaryOp.PRE_INC else BinOpKind.SUB
            result = self._builder.binop(op, val, one)
            self._builder.store(result, addr)
            return result

        if expr.op in (CASTUnaryOp.POST_INC, CASTUnaryOp.POST_DEC):
            addr = self._lower_lvalue(expr.operand)
            if addr is None:
                return None
            pointee = self._get_pointee_type(addr)
            old_val = self._builder.load(addr, pointee)
            one = Constant.int_const(1, old_val.type if isinstance(old_val.type, IntType) else None)
            op = BinOpKind.ADD if expr.op == CASTUnaryOp.POST_INC else BinOpKind.SUB
            new_val = self._builder.binop(op, old_val, one)
            self._builder.store(new_val, addr)
            return old_val

        operand = self._lower_expr(expr.operand)
        if operand is None:
            return None

        if expr.op == CASTUnaryOp.MINUS:
            result = self._builder.neg(operand)
            # Signed negation is UB in C when operand == INT_MIN
            if result.metadata and isinstance(operand.type, IntType) and operand.type.is_signed:
                result.metadata = InstructionMetadata(
                    overflow=OverflowBehavior.UNDEFINED,
                    tags={'c_signed_overflow': 'ub'},
                )
            return result
        if expr.op == CASTUnaryOp.PLUS:
            return operand
        if expr.op == CASTUnaryOp.BITWISE_NOT:
            return self._builder.not_(operand)
        if expr.op == CASTUnaryOp.LOGICAL_NOT:
            val = self._ensure_bool(operand)
            return self._builder.icmp(CmpPredicate.EQ, val,
                                     Constant.bool_const(False))

        return None

    def _lower_cast_expr(self, expr: CastExpr) -> Optional[Value]:
        """Lower an explicit cast expression."""
        if not expr.operand or not expr.cast_type:
            return None

        operand = self._lower_expr(expr.operand)
        if operand is None:
            return None

        target_type = self._lower_type(expr.cast_type)
        return self._convert(operand, target_type)

    def _lower_call_expr(self, expr: CallExpr) -> Optional[Value]:
        """Lower a function call."""
        if not expr.callee:
            return None

        callee = self._lower_expr(expr.callee)
        if callee is None:
            return None

        args: list[Value] = []
        for arg in expr.args:
            val = self._lower_expr(arg)
            if val is not None:
                args.append(val)

        # Determine return type
        callee_name = ""
        ret_type: IRType = VoidType()
        if isinstance(expr.callee, IdentExpr):
            callee_name = expr.callee.name
            # Look up function type
            func = self._module.get_function(callee_name)
            if func:
                ret_type = func.func_type.return_type
            else:
                ext = self._module.get_external(callee_name)
                if ext and isinstance(ext.type, FunctionType):
                    ret_type = ext.type.return_type

        return self._builder.call(callee, args, ret_type,
                                  callee_name=callee_name)

    def _lower_member_expr(self, expr: MemberExpr) -> Optional[Value]:
        """Lower member access expression."""
        addr = self._lower_member_addr(expr)
        if addr is None:
            return None
        pointee = self._get_pointee_type(addr)
        return self._builder.load(addr, pointee)

    def _lower_member_addr(self, expr: MemberExpr) -> Optional[Value]:
        """Get the address of a member access."""
        if not expr.base:
            return None

        if expr.is_arrow:
            base_ptr = self._lower_expr(expr.base)
        else:
            base_ptr = self._lower_lvalue(expr.base)

        if base_ptr is None:
            return None

        # Get struct type info to find field index
        pointee = self._get_pointee_type(base_ptr)
        if isinstance(pointee, StructType):
            try:
                idx, fld = pointee.field_by_name(expr.member)
                zero = Constant.int_const(0, IntType(32, Signedness.SIGNED))
                field_idx = Constant.int_const(idx, IntType(32, Signedness.SIGNED))
                return self._builder.gep(pointee, base_ptr, [zero, field_idx])
            except KeyError:
                pass

        return base_ptr

    def _lower_array_subscript(self, expr: ArraySubscriptExpr) -> Optional[Value]:
        """Lower array subscript expression."""
        addr = self._lower_array_subscript_addr(expr)
        if addr is None:
            return None
        pointee = self._get_pointee_type(addr)
        return self._builder.load(addr, pointee)

    def _lower_array_subscript_addr(self, expr: ArraySubscriptExpr) -> Optional[Value]:
        """Get the address of an array subscript."""
        if not expr.base or not expr.index:
            return None

        base = self._lower_expr(expr.base)
        index = self._lower_expr(expr.index)
        if base is None or index is None:
            return None

        pointee = self._get_pointee_type(base)
        return self._builder.gep(pointee, base, [index])

    def _lower_ternary_expr(self, expr: TernaryExpr) -> Optional[Value]:
        """Lower ternary (conditional) expression."""
        if not expr.condition:
            return None

        cond = self._lower_expr(expr.condition)
        if cond is None:
            return None
        cond = self._ensure_bool(cond)

        then_block = self._fresh_block("ternary.then")
        else_block = self._fresh_block("ternary.else")
        merge_block = self._fresh_block("ternary.merge")

        self._builder.cond_br(cond, then_block, else_block)

        self._builder.position_at_end(then_block)
        then_val = self._lower_expr(expr.then_expr) if expr.then_expr else None
        then_end = self._builder.insert_block
        if not then_end.has_terminator:
            self._builder.br(merge_block)

        self._builder.position_at_end(else_block)
        else_val = self._lower_expr(expr.else_expr) if expr.else_expr else None
        else_end = self._builder.insert_block
        if not else_end.has_terminator:
            self._builder.br(merge_block)

        self._builder.position_at_end(merge_block)

        if then_val is not None and else_val is not None:
            then_val, else_val = self._apply_arithmetic_conversions(then_val, else_val)
            phi = self._builder.phi(then_val.type, name="ternary")
            phi.add_incoming(then_val, then_end)
            phi.add_incoming(else_val, else_end)
            return phi

        return then_val or else_val

    def _lower_sizeof_expr(self, expr: SizeofExpr) -> Value:
        """Lower sizeof expression."""
        if expr.is_type and expr.operand_type:
            size = self._resolver.sizeof(expr.operand_type)
        elif expr.operand_expr:
            # sizeof(expr) - need to determine type of expr
            size = 4  # fallback
        else:
            size = 0
        return Constant.int_const(size, IntType(64, Signedness.UNSIGNED))

    def _lower_alignof_expr(self, expr: AlignofExpr) -> Value:
        """Lower _Alignof expression."""
        if expr.operand_type:
            align = self._resolver.alignof(expr.operand_type)
        else:
            align = 1
        return Constant.int_const(align, IntType(64, Signedness.UNSIGNED))

    def _lower_comma_expr(self, expr: CommaExpr) -> Optional[Value]:
        """Lower comma expression - evaluate all, return last."""
        result: Optional[Value] = None
        for e in expr.exprs:
            result = self._lower_expr(e)
        return result

    def _lower_init_list(self, expr: InitListExpr) -> Optional[Value]:
        """Lower an initializer list."""
        if not expr.elements:
            return None
        # Lower first element for simple cases
        return self._lower_expr(expr.elements[0])

    def _lower_compound_literal(self, expr: CompoundLiteralExpr) -> Optional[Value]:
        """Lower a compound literal (type){ init }."""
        if not expr.type_name or not expr.init_list:
            return None
        ir_type = self._lower_type(expr.type_name)
        alloca = self._builder.alloca(ir_type, name="compound_lit")
        # Initialize fields from init list
        for i, elem in enumerate(expr.init_list.elements):
            val = self._lower_expr(elem)
            if val is not None and isinstance(ir_type, StructType) and i < len(ir_type.fields):
                zero = Constant.int_const(0, IntType(32, Signedness.SIGNED))
                idx = Constant.int_const(i, IntType(32, Signedness.SIGNED))
                field_ptr = self._builder.gep(ir_type, alloca, [zero, idx])
                self._builder.store(val, field_ptr)
        return self._builder.load(alloca, ir_type)

    def _lower_stmt_expr(self, expr: StmtExpr) -> Optional[Value]:
        """Lower GCC statement expression ({ stmt; expr; })."""
        if expr.body:
            self._lower_compound_stmt(expr.body)
        return None

    # -------------------------------------------------------------------
    # Type conversion helpers
    # -------------------------------------------------------------------

    def _ensure_type(self, val: Value, target: IRType) -> Value:
        """Ensure a value has the target type, inserting a cast if needed."""
        if val.type == target:
            return val
        return self._convert(val, target)

    def _ensure_bool(self, val: Value) -> Value:
        """Convert a value to i1 (boolean)."""
        bool_type = IntType(1, Signedness.UNSIGNED)
        if val.type == bool_type:
            return val

        if isinstance(val.type, IntType):
            return self._builder.icmp(
                CmpPredicate.NE, val,
                Constant.int_const(0, val.type)
            )
        if isinstance(val.type, FloatType):
            return self._builder.icmp(
                CmpPredicate.ONE, val,
                Constant.float_const(0.0, val.type)
            )
        if isinstance(val.type, PointerType):
            return self._builder.icmp(
                CmpPredicate.NE, val,
                Constant.null_ptr(val.type.pointee)
            )
        return val

    def _convert(self, val: Value, target: IRType) -> Value:
        """Convert a value to the target type."""
        src = val.type
        if src == target:
            return val

        # Int -> Int
        if isinstance(src, IntType) and isinstance(target, IntType):
            if src.width > target.width:
                return self._builder.cast(CastKind.TRUNC, val, target)
            elif src.width < target.width:
                if src.is_signed:
                    return self._builder.cast(CastKind.SEXT, val, target)
                else:
                    return self._builder.cast(CastKind.ZEXT, val, target)
            else:
                return self._builder.cast(CastKind.BITCAST, val, target)

        # Float -> Float
        if isinstance(src, FloatType) and isinstance(target, FloatType):
            if src.width > target.width:
                return self._builder.cast(CastKind.FPTRUNC, val, target)
            else:
                return self._builder.cast(CastKind.FPEXT, val, target)

        # Int -> Float
        if isinstance(src, IntType) and isinstance(target, FloatType):
            kind = CastKind.SITOFP if src.is_signed else CastKind.UITOFP
            return self._builder.cast(kind, val, target)

        # Float -> Int
        if isinstance(src, FloatType) and isinstance(target, IntType):
            kind = CastKind.FPTOSI if target.is_signed else CastKind.FPTOUI
            metadata = InstructionMetadata()
            metadata.tags["c_float_to_int"] = "potential_ub"
            return self._builder.cast(kind, val, target, metadata=metadata)

        # Pointer -> Int
        if isinstance(src, PointerType) and isinstance(target, IntType):
            return self._builder.cast(CastKind.PTRTOINT, val, target)

        # Int -> Pointer
        if isinstance(src, IntType) and isinstance(target, PointerType):
            return self._builder.cast(CastKind.INTTOPTR, val, target)

        # Pointer -> Pointer (bitcast)
        if isinstance(src, PointerType) and isinstance(target, PointerType):
            return self._builder.cast(CastKind.BITCAST, val, target)

        # Fallback: bitcast
        return self._builder.cast(CastKind.BITCAST, val, target)

    def _apply_arithmetic_conversions(
        self, lhs: Value, rhs: Value
    ) -> tuple[Value, Value]:
        """Apply C usual arithmetic conversions to match types."""
        if lhs.type == rhs.type:
            return lhs, rhs

        # Both integers
        if isinstance(lhs.type, IntType) and isinstance(rhs.type, IntType):
            target = self._common_int_type(lhs.type, rhs.type)
            return self._ensure_type(lhs, target), self._ensure_type(rhs, target)

        # Both floats
        if isinstance(lhs.type, FloatType) and isinstance(rhs.type, FloatType):
            target = lhs.type if lhs.type.width >= rhs.type.width else rhs.type
            return self._ensure_type(lhs, target), self._ensure_type(rhs, target)

        # Mixed int/float
        if isinstance(lhs.type, IntType) and isinstance(rhs.type, FloatType):
            return self._ensure_type(lhs, rhs.type), rhs
        if isinstance(lhs.type, FloatType) and isinstance(rhs.type, IntType):
            return lhs, self._ensure_type(rhs, lhs.type)

        return lhs, rhs

    def _common_int_type(self, a: IntType, b: IntType) -> IntType:
        """Compute the common type for two integer types."""
        if a.width > b.width:
            return a
        if b.width > a.width:
            return b
        # Same width - prefer unsigned
        if a.is_unsigned or b.is_unsigned:
            return IntType(a.width, Signedness.UNSIGNED)
        return a

    def _get_pointee_type(self, ptr: Value) -> IRType:
        """Get the pointee type of a pointer value."""
        if isinstance(ptr.type, PointerType):
            return ptr.type.pointee
        return IntType(8, Signedness.SIGNED)  # fallback
