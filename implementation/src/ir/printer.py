"""
Pretty printer for Cross-Language Equivalence Verifier IR.

Produces a human-readable text format inspired by LLVM IR syntax,
suitable for debugging, diffing, and logging.
"""

from __future__ import annotations

import io
from typing import TextIO

from .basic_block import BasicBlock
from .function import Function
from .module import Module, GlobalVariable, TypeDefinition, ExternalDeclaration, StringConstant
from .instructions import (
    AllocaInst,
    AtomicCmpXchgInst,
    AtomicRMWInst,
    BinaryOp,
    BranchInst,
    CallInst,
    CastInst,
    CompareOp,
    Constant,
    ExtractValueInst,
    FenceInst,
    GetElementPtrInst,
    InsertValueInst,
    Instruction,
    InstructionVisitor,
    LoadInst,
    MemcpyInst,
    MemsetInst,
    PhiInst,
    ReturnInst,
    SelectInst,
    StoreInst,
    SwitchInst,
    UnaryOp,
    Value,
)
from .types import (
    ArrayType,
    FloatType,
    FunctionType,
    IRType,
    IntType,
    PointerType,
    StructType,
    UnionType,
    VoidType,
)


class IRPrinter(InstructionVisitor):
    """Pretty-prints IR modules, functions, blocks, and instructions.

    Usage::

        printer = IRPrinter()
        text = printer.print_module(module)
        print(text)
    """

    def __init__(self, indent: str = "  ", show_metadata: bool = True) -> None:
        self._indent = indent
        self._show_metadata = show_metadata
        self._out: TextIO = io.StringIO()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def print_module(self, module: Module) -> str:
        """Print an entire module to string."""
        self._out = io.StringIO()
        self._emit_module(module)
        return self._out.getvalue()

    def print_function(self, func: Function) -> str:
        self._out = io.StringIO()
        self._emit_function(func)
        return self._out.getvalue()

    def print_block(self, block: BasicBlock) -> str:
        self._out = io.StringIO()
        self._emit_block(block)
        return self._out.getvalue()

    def print_instruction(self, inst: Instruction) -> str:
        self._out = io.StringIO()
        self._emit_instruction(inst)
        return self._out.getvalue().rstrip("\n")

    def print_type(self, ty: IRType) -> str:
        return self._format_type(ty)

    def print_value(self, val: Value) -> str:
        return self._format_value(val)

    # ------------------------------------------------------------------
    # Internal: writing helpers
    # ------------------------------------------------------------------

    def _write(self, text: str) -> None:
        self._out.write(text)

    def _writeln(self, text: str = "") -> None:
        self._out.write(text + "\n")

    def _write_indent(self, level: int = 1) -> None:
        self._out.write(self._indent * level)

    # ------------------------------------------------------------------
    # Type formatting
    # ------------------------------------------------------------------

    def _format_type(self, ty: IRType) -> str:
        match ty:
            case VoidType():
                return "void"
            case IntType(width=w, signedness=s):
                prefix = "i" if s.name == "SIGNED" else "u"
                return f"{prefix}{w}"
            case FloatType(kind=k):
                return str(k)
            case PointerType(pointee=pt):
                return f"ptr<{self._format_type(pt)}>"
            case ArrayType(element=el, length=n):
                return f"[{n} x {self._format_type(el)}]"
            case StructType(name=name, fields=fields, packed=packed):
                parts = ", ".join(
                    f"{f.name}: {self._format_type(f.type)}" for f in fields
                )
                pack = "<packed> " if packed else ""
                nm = f"%{name} " if name else ""
                return f"{nm}{{{pack}{parts}}}"
            case UnionType(name=name, variants=variants):
                parts = " | ".join(
                    f"{n}: {self._format_type(t)}" for n, t in variants
                )
                nm = f"%{name} " if name else ""
                return f"{nm}union{{{parts}}}"
            case FunctionType(return_type=ret, param_types=params, is_variadic=va):
                ps = ", ".join(self._format_type(p) for p in params)
                if va:
                    ps += ", ..."
                return f"({ps}) -> {self._format_type(ret)}"
            case _:
                return str(ty)

    # ------------------------------------------------------------------
    # Value formatting
    # ------------------------------------------------------------------

    def _format_value(self, val: Value) -> str:
        if isinstance(val, Constant):
            return self._format_constant(val)
        return val.display_name

    def _format_constant(self, c: Constant) -> str:
        if c.is_undef:
            return f"undef"
        if c.is_null:
            return f"null"
        ty = self._format_type(c.type)
        return f"{ty} {c.value}"

    def _format_typed_value(self, val: Value) -> str:
        """Format as 'type value'."""
        ty = self._format_type(val.type)
        if isinstance(val, Constant):
            if val.is_undef:
                return f"{ty} undef"
            if val.is_null:
                return f"{ty} null"
            return f"{ty} {val.value}"
        return f"{ty} {val.display_name}"

    # ------------------------------------------------------------------
    # Module
    # ------------------------------------------------------------------

    def _emit_module(self, mod: Module) -> None:
        self._writeln(f"; Module: {mod.name}")
        if mod.source_filename:
            self._writeln(f'source_filename = "{mod.source_filename}"')
        if mod.target_triple:
            self._writeln(f'target triple = "{mod.target_triple}"')
        if mod.data_layout:
            self._writeln(f'target datalayout = "{mod.data_layout}"')
        self._writeln()

        # Type definitions
        for td in mod.types.values():
            self._writeln(f"%{td.name} = type {self._format_type(td.type)}")
        if mod.types:
            self._writeln()

        # String constants
        for sc in mod.strings.values():
            escaped = sc.value.replace("\\", "\\\\").replace('"', '\\"')
            self._writeln(
                f'@{sc.name} = private constant [{sc.byte_length} x u8] '
                f'c"{escaped}\\00"'
            )
        if mod.strings:
            self._writeln()

        # Globals
        for gv in mod.globals.values():
            self._emit_global(gv)
        if mod.globals:
            self._writeln()

        # External declarations
        for ext in mod.externals.values():
            self._emit_external(ext)
        if mod.externals:
            self._writeln()

        # Functions
        for func in mod.functions.values():
            self._emit_function(func)
            self._writeln()

    def _emit_global(self, gv: GlobalVariable) -> None:
        const = "constant" if gv.is_constant else "global"
        ty = self._format_type(gv.type)
        init = ""
        if gv.initializer is not None:
            init = f" {self._format_constant(gv.initializer)}"
        else:
            init = " zeroinitializer"
        align = ""
        if gv.alignment > 0:
            align = f", align {gv.alignment}"
        self._writeln(f"@{gv.name} = {gv.linkage} {const} {ty}{init}{align}")

    def _emit_external(self, ext: ExternalDeclaration) -> None:
        if ext.is_function and isinstance(ext.type, FunctionType):
            ft = ext.type
            params = ", ".join(self._format_type(p) for p in ft.param_types)
            if ft.is_variadic:
                params += ", ..."
            ret = self._format_type(ft.return_type)
            self._writeln(f"declare {ret} @{ext.name}({params})")
        else:
            self._writeln(f"@{ext.name} = external global {self._format_type(ext.type)}")

    # ------------------------------------------------------------------
    # Function
    # ------------------------------------------------------------------

    def _emit_function(self, func: Function) -> None:
        params = ", ".join(
            f"{self._format_type(a.type)} {a.display_name}" for a in func.arguments
        )
        ret = self._format_type(func.return_type)
        linkage = f"{func.linkage} " if func.linkage else ""
        self._writeln(f"define {linkage}{ret} @{func.name}({params}) {{")

        for block in func.blocks:
            self._emit_block(block)

        self._writeln("}")

    # ------------------------------------------------------------------
    # Basic block
    # ------------------------------------------------------------------

    def _emit_block(self, block: BasicBlock) -> None:
        preds = ", ".join(f"%{b.name}" for b in block.predecessors)
        pred_comment = f"  ; preds = {preds}" if preds else ""
        self._writeln(f"{block.name}:{pred_comment}")

        for inst in block:
            self._write_indent()
            self._emit_instruction(inst)

    # ------------------------------------------------------------------
    # Instructions (visitor-based dispatch)
    # ------------------------------------------------------------------

    def _emit_instruction(self, inst: Instruction) -> None:
        """Dispatch to the appropriate visit_* method."""
        inst.accept(self)
        if self._show_metadata and inst.metadata.source_loc:
            self._write(f"  ; {inst.metadata.source_loc}")
        self._writeln()

    def visit_binary_op(self, inst: BinaryOp) -> None:
        lhs = self._format_typed_value(inst.lhs)
        rhs = self._format_value(inst.rhs)
        self._write(f"{inst.display_name} = {inst.opcode_name()} {lhs}, {rhs}")

    def visit_unary_op(self, inst: UnaryOp) -> None:
        op = self._format_typed_value(inst.operand)
        self._write(f"{inst.display_name} = {inst.opcode_name()} {op}")

    def visit_compare_op(self, inst: CompareOp) -> None:
        lhs = self._format_typed_value(inst.lhs)
        rhs = self._format_value(inst.rhs)
        self._write(f"{inst.display_name} = {inst.opcode_name()} {lhs}, {rhs}")

    def visit_load(self, inst: LoadInst) -> None:
        ty = self._format_type(inst.type)
        addr = self._format_typed_value(inst.address)
        vol = ", volatile" if inst.volatile else ""
        align = f", align {inst.alignment}" if inst.alignment else ""
        self._write(f"{inst.display_name} = {inst.opcode_name()} {ty}, {addr}{vol}{align}")

    def visit_store(self, inst: StoreInst) -> None:
        val = self._format_typed_value(inst.value)
        addr = self._format_typed_value(inst.address)
        vol = ", volatile" if inst.volatile else ""
        align = f", align {inst.alignment}" if inst.alignment else ""
        self._write(f"store {val}, {addr}{vol}{align}")

    def visit_alloca(self, inst: AllocaInst) -> None:
        ty = self._format_type(inst.alloc_type)
        num = ""
        if inst.num_elements > 1:
            num = f", i64 {inst.num_elements}"
        align = f", align {inst.alignment}" if inst.alignment else ""
        self._write(f"{inst.display_name} = alloca {ty}{num}{align}")

    def visit_gep(self, inst: GetElementPtrInst) -> None:
        src_ty = self._format_type(inst.source_element_type)
        base = self._format_typed_value(inst.base)
        indices = ", ".join(self._format_typed_value(i) for i in inst.indices)
        ib = "inbounds " if inst.inbounds else ""
        self._write(
            f"{inst.display_name} = getelementptr {ib}{src_ty}, {base}, {indices}"
        )

    def visit_cast(self, inst: CastInst) -> None:
        src = self._format_typed_value(inst.operand)
        dst_ty = self._format_type(inst.dest_type)
        self._write(f"{inst.display_name} = {inst.opcode_name()} {src} to {dst_ty}")

    def visit_call(self, inst: CallInst) -> None:
        ret_ty = self._format_type(inst.type)
        callee = self._format_value(inst.callee)
        args = ", ".join(self._format_typed_value(a) for a in inst.args)
        prefix = "tail " if inst.is_tail_call else ""
        if inst.type.is_void():
            self._write(f"{prefix}call {ret_ty} {callee}({args})")
        else:
            self._write(f"{inst.display_name} = {prefix}call {ret_ty} {callee}({args})")

    def visit_return(self, inst: ReturnInst) -> None:
        if inst.is_void_return:
            self._write("ret void")
        else:
            rv = inst.return_value
            assert rv is not None
            self._write(f"ret {self._format_typed_value(rv)}")

    def visit_branch(self, inst: BranchInst) -> None:
        if inst.is_conditional:
            cond = self._format_typed_value(inst.condition)
            self._write(
                f"br {cond}, label %{inst.true_target.name}, "
                f"label %{inst.false_target.name}"
            )
        else:
            self._write(f"br label %{inst.true_target.name}")

    def visit_switch(self, inst: SwitchInst) -> None:
        cond = self._format_typed_value(inst.condition)
        self._write(f"switch {cond}, label %{inst.default_target.name} [")
        for val, target in inst.cases:
            cv = self._format_typed_value(val)
            self._write(f" {cv}, label %{target.name}")
        self._write(" ]")

    def visit_phi(self, inst: PhiInst) -> None:
        ty = self._format_type(inst.type)
        entries = ", ".join(
            f"[ {self._format_value(v)}, %{b.name} ]"
            for v, b in inst.incoming
        )
        self._write(f"{inst.display_name} = phi {ty} {entries}")

    def visit_select(self, inst: SelectInst) -> None:
        cond = self._format_typed_value(inst.condition)
        tv = self._format_typed_value(inst.true_value)
        fv = self._format_typed_value(inst.false_value)
        self._write(f"{inst.display_name} = select {cond}, {tv}, {fv}")

    def visit_extract_value(self, inst: ExtractValueInst) -> None:
        agg = self._format_typed_value(inst.aggregate)
        idxs = ", ".join(str(i) for i in inst.indices)
        self._write(f"{inst.display_name} = extractvalue {agg}, {idxs}")

    def visit_insert_value(self, inst: InsertValueInst) -> None:
        agg = self._format_typed_value(inst.aggregate)
        val = self._format_typed_value(inst.inserted_value)
        idxs = ", ".join(str(i) for i in inst.indices)
        self._write(f"{inst.display_name} = insertvalue {agg}, {val}, {idxs}")

    def visit_memcpy(self, inst: MemcpyInst) -> None:
        dst = self._format_typed_value(inst.dest)
        src = self._format_typed_value(inst.src)
        ln = self._format_typed_value(inst.length)
        vol = ", volatile" if inst.is_volatile else ""
        self._write(f"call void @llvm.memcpy({dst}, {src}, {ln}{vol})")

    def visit_memset(self, inst: MemsetInst) -> None:
        dst = self._format_typed_value(inst.dest)
        val = self._format_typed_value(inst.fill_value)
        ln = self._format_typed_value(inst.length)
        vol = ", volatile" if inst.is_volatile else ""
        self._write(f"call void @llvm.memset({dst}, {val}, {ln}{vol})")

    def visit_fence(self, inst: FenceInst) -> None:
        self._write(f"fence {inst.ordering.name.lower()}")

    def visit_atomic_rmw(self, inst: AtomicRMWInst) -> None:
        addr = self._format_typed_value(inst.address)
        val = self._format_typed_value(inst.value)
        order = inst.ordering.name.lower()
        self._write(
            f"{inst.display_name} = atomicrmw {inst.rmw_op.value} {addr}, {val} {order}"
        )

    def visit_atomic_cmpxchg(self, inst: AtomicCmpXchgInst) -> None:
        addr = self._format_typed_value(inst.address)
        exp = self._format_typed_value(inst.expected)
        des = self._format_typed_value(inst.desired)
        so = inst.success_ordering.name.lower()
        fo = inst.failure_ordering.name.lower()
        self._write(
            f"{inst.display_name} = cmpxchg {addr}, {exp}, {des} {so} {fo}"
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def print_module(module: Module, show_metadata: bool = True) -> str:
    """Print a module to a string."""
    return IRPrinter(show_metadata=show_metadata).print_module(module)


def print_function(func: Function, show_metadata: bool = True) -> str:
    """Print a function to a string."""
    return IRPrinter(show_metadata=show_metadata).print_function(func)
