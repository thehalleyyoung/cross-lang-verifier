"""
Module for the Cross-Language Equivalence Verifier IR.

A Module is the top-level container holding functions, global variables,
type definitions, string constants, and external declarations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional, Sequence

from .basic_block import BasicBlock
from .instructions import Constant, Value
from .function import Function
from .types import (
    IRType,
    IntType,
    FloatType,
    PointerType,
    ArrayType,
    StructType,
    FunctionType,
    VoidType,
    Signedness,
)


@dataclass
class GlobalVariable:
    """A module-level global variable.

    Attributes:
        name: symbol name.
        type: the type of the stored value.
        initializer: optional constant initializer.
        is_constant: True if the global is immutable.
        linkage: linkage kind.
        alignment: override alignment (0 = natural).
        section: optional section name.
    """
    name: str
    type: IRType
    initializer: Constant | None = None
    is_constant: bool = False
    linkage: str = "external"
    alignment: int = 0
    section: str = ""
    language: str = ""

    @property
    def pointer_type(self) -> PointerType:
        return PointerType(self.type)

    def as_value(self) -> Value:
        """Return a Value representing the address of this global."""
        return Value(self.pointer_type, name=self.name)

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.name:
            errors.append("Global variable has no name")
        if self.initializer is not None:
            if self.initializer.type != self.type:
                errors.append(
                    f"Global '{self.name}': initializer type {self.initializer.type} "
                    f"!= declared type {self.type}"
                )
        return errors


@dataclass
class TypeDefinition:
    """A named type alias / definition."""
    name: str
    type: IRType
    language: str = ""  # "c" or "rust"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TypeDefinition):
            return NotImplemented
        return self.name == other.name and self.type == other.type

    def __hash__(self) -> int:
        return hash((self.name, self.type))


@dataclass
class ExternalDeclaration:
    """An external function or variable declaration (no body / initializer)."""
    name: str
    type: IRType
    is_function: bool = True
    linkage: str = "external"
    language: str = ""

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.name:
            errors.append("External declaration has no name")
        if self.is_function and not isinstance(self.type, FunctionType):
            errors.append(f"External function '{self.name}' has non-function type {self.type}")
        return errors


@dataclass
class StringConstant:
    """A string literal stored in the module's constant pool."""
    name: str
    value: str
    encoding: str = "utf-8"
    null_terminated: bool = True

    @property
    def byte_length(self) -> int:
        raw = self.value.encode(self.encoding)
        return len(raw) + (1 if self.null_terminated else 0)

    @property
    def ir_type(self) -> ArrayType:
        return ArrayType(IntType(8, Signedness.UNSIGNED), self.byte_length)


class Module:
    """Top-level IR module.

    Contains functions, global variables, type definitions, string constants,
    and external declarations.

    Attributes:
        name: module name / identifier.
        source_filename: original source file path.
        target_triple: target platform triple (e.g. "x86_64-unknown-linux-gnu").
        data_layout: data layout string.
        language: primary source language.
    """
    __slots__ = (
        "name", "source_filename", "target_triple", "data_layout",
        "language", "_functions", "_globals", "_types",
        "_strings", "_externals", "_metadata",
    )

    def __init__(
        self,
        name: str = "",
        source_filename: str = "",
        target_triple: str = "",
        data_layout: str = "",
        language: str = "",
    ) -> None:
        self.name = name
        self.source_filename = source_filename
        self.target_triple = target_triple
        self.data_layout = data_layout
        self.language = language

        self._functions: dict[str, Function] = {}
        self._globals: dict[str, GlobalVariable] = {}
        self._types: dict[str, TypeDefinition] = {}
        self._strings: dict[str, StringConstant] = {}
        self._externals: dict[str, ExternalDeclaration] = {}
        self._metadata: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Functions
    # ------------------------------------------------------------------

    @property
    def functions(self) -> dict[str, Function]:
        return dict(self._functions)

    def add_function(self, func: Function) -> Function:
        if func.name in self._functions:
            raise ValueError(f"Function '{func.name}' already exists in module")
        self._functions[func.name] = func
        return func

    def create_function(
        self, name: str, func_type: FunctionType, linkage: str = "external"
    ) -> Function:
        func = Function(name, func_type, linkage, self.language)
        return self.add_function(func)

    def get_function(self, name: str) -> Function | None:
        return self._functions.get(name)

    def remove_function(self, name: str) -> None:
        del self._functions[name]

    def iter_functions(self) -> Iterator[Function]:
        return iter(self._functions.values())

    @property
    def num_functions(self) -> int:
        return len(self._functions)

    # ------------------------------------------------------------------
    # Global variables
    # ------------------------------------------------------------------

    @property
    def globals(self) -> dict[str, GlobalVariable]:
        return dict(self._globals)

    def add_global(self, gv: GlobalVariable) -> GlobalVariable:
        if gv.name in self._globals:
            raise ValueError(f"Global '{gv.name}' already exists in module")
        self._globals[gv.name] = gv
        return gv

    def create_global(
        self,
        name: str,
        ty: IRType,
        initializer: Constant | None = None,
        is_constant: bool = False,
        linkage: str = "external",
    ) -> GlobalVariable:
        gv = GlobalVariable(name, ty, initializer, is_constant, linkage)
        return self.add_global(gv)

    def get_global(self, name: str) -> GlobalVariable | None:
        return self._globals.get(name)

    def remove_global(self, name: str) -> None:
        del self._globals[name]

    def iter_globals(self) -> Iterator[GlobalVariable]:
        return iter(self._globals.values())

    # ------------------------------------------------------------------
    # Type definitions
    # ------------------------------------------------------------------

    @property
    def types(self) -> dict[str, TypeDefinition]:
        return dict(self._types)

    def add_type(self, typedef: TypeDefinition) -> TypeDefinition:
        if typedef.name in self._types:
            raise ValueError(f"Type '{typedef.name}' already defined in module")
        self._types[typedef.name] = typedef
        return typedef

    def define_type(self, name: str, ty: IRType, language: str = "") -> TypeDefinition:
        td = TypeDefinition(name, ty, language)
        return self.add_type(td)

    def get_type(self, name: str) -> TypeDefinition | None:
        return self._types.get(name)

    def resolve_type(self, name: str) -> IRType | None:
        td = self._types.get(name)
        return td.type if td else None

    # ------------------------------------------------------------------
    # String constants
    # ------------------------------------------------------------------

    @property
    def strings(self) -> dict[str, StringConstant]:
        return dict(self._strings)

    def add_string(self, sc: StringConstant) -> StringConstant:
        self._strings[sc.name] = sc
        return sc

    def create_string(
        self, name: str, value: str, null_terminated: bool = True
    ) -> StringConstant:
        sc = StringConstant(name, value, null_terminated=null_terminated)
        return self.add_string(sc)

    def get_string(self, name: str) -> StringConstant | None:
        return self._strings.get(name)

    # ------------------------------------------------------------------
    # External declarations
    # ------------------------------------------------------------------

    @property
    def externals(self) -> dict[str, ExternalDeclaration]:
        return dict(self._externals)

    def add_external(self, ext: ExternalDeclaration) -> ExternalDeclaration:
        self._externals[ext.name] = ext
        return ext

    def declare_function(
        self, name: str, func_type: FunctionType, linkage: str = "external"
    ) -> ExternalDeclaration:
        ext = ExternalDeclaration(name, func_type, is_function=True, linkage=linkage)
        return self.add_external(ext)

    def declare_variable(
        self, name: str, ty: IRType, linkage: str = "external"
    ) -> ExternalDeclaration:
        ext = ExternalDeclaration(name, ty, is_function=False, linkage=linkage)
        return self.add_external(ext)

    def get_external(self, name: str) -> ExternalDeclaration | None:
        return self._externals.get(name)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def set_metadata(self, key: str, value: str) -> None:
        self._metadata[key] = value

    def get_metadata(self, key: str) -> str | None:
        return self._metadata.get(key)

    # ------------------------------------------------------------------
    # Symbol resolution
    # ------------------------------------------------------------------

    def resolve_symbol(self, name: str) -> Function | GlobalVariable | ExternalDeclaration | None:
        """Look up a symbol by name across all namespaces."""
        if name in self._functions:
            return self._functions[name]
        if name in self._globals:
            return self._globals[name]
        if name in self._externals:
            return self._externals[name]
        return None

    def all_symbol_names(self) -> set[str]:
        names: set[str] = set()
        names.update(self._functions.keys())
        names.update(self._globals.keys())
        names.update(self._externals.keys())
        return names

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """Validate the entire module."""
        errors: list[str] = []

        # Validate globals
        for gv in self._globals.values():
            errors.extend(gv.validate())

        # Validate externals
        for ext in self._externals.values():
            errors.extend(ext.validate())

        # Validate functions
        for func in self._functions.values():
            func_errors = func.validate()
            for e in func_errors:
                errors.append(f"Module '{self.name}': {e}")

        # Check for symbol conflicts between functions and globals
        func_names = set(self._functions.keys())
        global_names = set(self._globals.keys())
        conflicts = func_names & global_names
        for c in conflicts:
            errors.append(f"Module '{self.name}': symbol conflict: '{c}' is both a function and global")

        return errors

    # ------------------------------------------------------------------
    # Linking (simple merge)
    # ------------------------------------------------------------------

    def link(self, other: "Module") -> list[str]:
        """Merge *other* module into this module.

        Returns a list of warnings/errors encountered during linking.
        """
        warnings: list[str] = []

        # Merge type definitions
        for name, td in other._types.items():
            existing = self._types.get(name)
            if existing is not None:
                if existing.type != td.type:
                    warnings.append(f"Type conflict for '{name}': {existing.type} vs {td.type}")
            else:
                self._types[name] = td

        # Merge globals
        for name, gv in other._globals.items():
            if name in self._globals:
                existing_gv = self._globals[name]
                if existing_gv.type != gv.type:
                    warnings.append(f"Global type conflict for '{name}'")
                elif gv.initializer is not None and existing_gv.initializer is None:
                    self._globals[name] = gv
            else:
                self._globals[name] = gv

        # Merge functions
        for name, func in other._functions.items():
            if name in self._functions:
                existing_func = self._functions[name]
                if existing_func.func_type != func.func_type:
                    warnings.append(f"Function type conflict for '{name}'")
                elif existing_func.num_blocks == 0 and func.num_blocks > 0:
                    self._functions[name] = func
            else:
                self._functions[name] = func

        # Merge externals
        for name, ext in other._externals.items():
            if name not in self._externals and name not in self._functions:
                self._externals[name] = ext

        # Merge strings
        for name, sc in other._strings.items():
            if name not in self._strings:
                self._strings[name] = sc

        # Remove externals that are now defined
        resolved = [n for n in self._externals if n in self._functions or n in self._globals]
        for n in resolved:
            del self._externals[n]

        return warnings

    # ------------------------------------------------------------------
    # Printing
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        return f"Module({self.name})"

    def __repr__(self) -> str:
        return (
            f"Module({self.name}, {self.num_functions} functions, "
            f"{len(self._globals)} globals)"
        )

    def dump(self) -> str:
        """Return a multi-line text representation of the entire module."""
        lines: list[str] = []
        lines.append(f'; Module: {self.name}')
        if self.source_filename:
            lines.append(f'source_filename = "{self.source_filename}"')
        if self.target_triple:
            lines.append(f'target triple = "{self.target_triple}"')
        if self.data_layout:
            lines.append(f'target datalayout = "{self.data_layout}"')
        lines.append("")

        # Type definitions
        for td in self._types.values():
            lines.append(f"%{td.name} = type {td.type}")
        if self._types:
            lines.append("")

        # String constants
        for sc in self._strings.values():
            escaped = sc.value.replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'@{sc.name} = private constant [{sc.byte_length} x i8] c"{escaped}\\00"')
        if self._strings:
            lines.append("")

        # Globals
        for gv in self._globals.values():
            const_str = "constant" if gv.is_constant else "global"
            init_str = ""
            if gv.initializer is not None:
                init_str = f" {gv.initializer.value}"
            else:
                init_str = " zeroinitializer"
            lines.append(f"@{gv.name} = {gv.linkage} {const_str} {gv.type}{init_str}")
        if self._globals:
            lines.append("")

        # External declarations
        for ext in self._externals.values():
            if ext.is_function and isinstance(ext.type, FunctionType):
                ft = ext.type
                params = ", ".join(str(p) for p in ft.param_types)
                if ft.is_variadic:
                    params += ", ..."
                lines.append(f"declare {ft.return_type} @{ext.name}({params})")
            else:
                lines.append(f"@{ext.name} = external global {ext.type}")
        if self._externals:
            lines.append("")

        # Functions
        for func in self._functions.values():
            lines.append(func.dump())
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, int]:
        """Return summary statistics for this module."""
        total_blocks = sum(f.num_blocks for f in self._functions.values())
        total_insts = sum(f.instruction_count for f in self._functions.values())
        return {
            "functions": self.num_functions,
            "globals": len(self._globals),
            "types": len(self._types),
            "strings": len(self._strings),
            "externals": len(self._externals),
            "total_blocks": total_blocks,
            "total_instructions": total_insts,
        }
