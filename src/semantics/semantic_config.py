"""
Semantic configuration for the Cross-Language Equivalence Verifier IR.

Provides dataclass-based configuration that parameterises how the IR is
executed / interpreted.  Different configurations correspond to different
source languages (C11, Rust-debug, Rust-release) and allow the evaluator
and verifier to model behavioural differences precisely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ── Configuration enums ──────────────────────────────────────────────────

class OverflowMode(Enum):
    """How integer overflow is handled at runtime."""
    Wrap = auto()      # Two's complement wrap-around
    Panic = auto()     # Runtime panic / trap
    UB = auto()        # Undefined behavior (anything may happen)
    Saturate = auto()  # Clamp to min/max representable value

    def __str__(self) -> str:
        return self.name

    @property
    def is_defined(self) -> bool:
        return self is not OverflowMode.UB

    @property
    def is_trapping(self) -> bool:
        return self is OverflowMode.Panic


class FloatModel(Enum):
    """Floating-point evaluation model."""
    IEEE754Strict = auto()   # Strict IEEE 754 compliance
    FastMath = auto()        # Allow reassociation, FMA, etc.
    StrictFinite = auto()    # No NaN/Inf results allowed

    def __str__(self) -> str:
        return self.name

    @property
    def allows_reassociation(self) -> bool:
        return self is FloatModel.FastMath

    @property
    def allows_nan(self) -> bool:
        return self is not FloatModel.StrictFinite

    @property
    def allows_fma(self) -> bool:
        return self is FloatModel.FastMath


class ErrorModel(Enum):
    """How runtime errors (division by zero, OOB, etc.) are handled."""
    Panic = auto()     # Rust-style: unwind / abort
    Return = auto()    # Return error code (C-style errno)
    UB = auto()        # Undefined behavior

    def __str__(self) -> str:
        return self.name

    @property
    def is_defined(self) -> bool:
        return self is not ErrorModel.UB


class PointerModel(Enum):
    """Memory / pointer model."""
    Flat = auto()          # Flat address space (C-style)
    Provenance = auto()    # Pointer provenance tracking (Rust / CHERI-style)

    def __str__(self) -> str:
        return self.name


class IntegerPromotionModel(Enum):
    """Integer promotion rules."""
    CStyle = auto()    # Implicit integer promotion per C standard
    RustStyle = auto() # No implicit promotion; all casts are explicit

    def __str__(self) -> str:
        return self.name


class ShiftModel(Enum):
    """Shift operation semantics."""
    UB_on_overshift = auto()   # C: UB if shift >= width or negative
    Mask = auto()              # Rust release: mask shift amount
    Panic_on_overshift = auto()# Rust debug: panic on overshift

    def __str__(self) -> str:
        return self.name


class DivisionModel(Enum):
    """Division-by-zero handling."""
    UB = auto()        # C: undefined behavior
    Panic = auto()     # Rust: always panic
    Trap = auto()      # Hardware trap

    def __str__(self) -> str:
        return self.name


class LayoutModel(Enum):
    """Struct/union layout model."""
    CCompat = auto()    # C-compatible layout (repr(C))
    RustDefault = auto() # Rust default (may reorder fields)
    Packed = auto()     # Packed layout (no padding)

    def __str__(self) -> str:
        return self.name


class FloatToIntModel(Enum):
    """Float-to-integer cast behaviour for out-of-range values."""
    UB = auto()         # C: undefined behavior
    Saturate = auto()   # Rust 1.45+: saturate + NaN→0

    def __str__(self) -> str:
        return self.name


class ArrayBoundsModel(Enum):
    """Array bounds checking policy."""
    NoCheck = auto()    # C: no bounds checking
    AlwaysCheck = auto()# Rust: always check in safe code
    DebugOnly = auto()  # Check only in debug builds

    def __str__(self) -> str:
        return self.name


class UnwindModel(Enum):
    """Stack unwinding behavior on panic."""
    NoUnwind = auto()   # C: no structured unwinding (setjmp/longjmp aside)
    Unwind = auto()     # Rust default: unwind on panic
    Abort = auto()      # Rust panic=abort: abort on panic

    def __str__(self) -> str:
        return self.name


# ── Semantic configuration dataclass ─────────────────────────────────────

@dataclass
class SemanticConfig:
    """Complete parameterisation of IR execution semantics.

    Each field controls one axis of semantic variation between C and Rust.
    Preset configurations for common language modes are provided as class
    methods.
    """

    # Core arithmetic
    signed_overflow: OverflowMode = OverflowMode.UB
    unsigned_overflow: OverflowMode = OverflowMode.Wrap
    float_model: FloatModel = FloatModel.IEEE754Strict
    integer_promotion: IntegerPromotionModel = IntegerPromotionModel.CStyle

    # Error handling
    error_model: ErrorModel = ErrorModel.UB
    division_model: DivisionModel = DivisionModel.UB
    float_to_int: FloatToIntModel = FloatToIntModel.UB

    # Memory / pointers
    pointer_model: PointerModel = PointerModel.Flat
    array_bounds: ArrayBoundsModel = ArrayBoundsModel.NoCheck
    null_pointer_valid: bool = False  # Whether null is a valid address

    # Shifts
    shift_model: ShiftModel = ShiftModel.UB_on_overshift

    # Unwinding
    unwind_model: UnwindModel = UnwindModel.NoUnwind

    # Struct layout
    layout_model: LayoutModel = LayoutModel.CCompat

    # Platform / ABI
    pointer_size: int = 64       # bits
    int_size: int = 32           # bits  (C `int` width)
    long_size: int = 64          # bits
    char_signed: bool = True     # whether `char` is signed
    wchar_size: int = 32         # bits
    endianness: str = "little"

    # Language tag
    language: str = ""           # "c", "rust", or empty

    # Optimization level
    opt_level: int = 0           # 0=debug, 1=O1, 2=O2, 3=O3

    # Additional flags
    strict_aliasing: bool = True
    volatile_is_observable: bool = True
    allow_type_punning: bool = True  # C-style union type punning

    # Metadata
    name: str = ""
    description: str = ""

    # ── Preset constructors ──────────────────────────────────────────────

    @classmethod
    def c11(cls, pointer_size: int = 64) -> "SemanticConfig":
        """Standard C11 semantics."""
        return cls(
            signed_overflow=OverflowMode.UB,
            unsigned_overflow=OverflowMode.Wrap,
            float_model=FloatModel.IEEE754Strict,
            integer_promotion=IntegerPromotionModel.CStyle,
            error_model=ErrorModel.UB,
            division_model=DivisionModel.UB,
            float_to_int=FloatToIntModel.UB,
            pointer_model=PointerModel.Flat,
            array_bounds=ArrayBoundsModel.NoCheck,
            null_pointer_valid=False,
            shift_model=ShiftModel.UB_on_overshift,
            unwind_model=UnwindModel.NoUnwind,
            pointer_size=pointer_size,
            int_size=32,
            long_size=64 if pointer_size == 64 else 32,
            char_signed=True,
            endianness="little",
            language="c",
            opt_level=0,
            strict_aliasing=True,
            volatile_is_observable=True,
            allow_type_punning=True,
            name="C11",
            description="Standard C11 semantics",
        )

    @classmethod
    def c11_optimized(cls, pointer_size: int = 64) -> "SemanticConfig":
        """C11 with aggressive optimizations (more UB exploitation)."""
        cfg = cls.c11(pointer_size)
        cfg.opt_level = 2
        cfg.name = "C11-O2"
        cfg.description = "C11 with -O2 optimizations"
        return cfg

    @classmethod
    def rust_debug(cls, pointer_size: int = 64) -> "SemanticConfig":
        """Rust debug-mode semantics (overflow panics, bounds checks)."""
        return cls(
            signed_overflow=OverflowMode.Panic,
            unsigned_overflow=OverflowMode.Panic,
            float_model=FloatModel.IEEE754Strict,
            integer_promotion=IntegerPromotionModel.RustStyle,
            error_model=ErrorModel.Panic,
            division_model=DivisionModel.Panic,
            float_to_int=FloatToIntModel.Saturate,
            pointer_model=PointerModel.Provenance,
            array_bounds=ArrayBoundsModel.AlwaysCheck,
            null_pointer_valid=False,
            shift_model=ShiftModel.Panic_on_overshift,
            unwind_model=UnwindModel.Unwind,
            pointer_size=pointer_size,
            int_size=32,
            long_size=64,
            char_signed=True,
            endianness="little",
            language="rust",
            opt_level=0,
            strict_aliasing=True,
            volatile_is_observable=True,
            allow_type_punning=False,
            name="Rust-debug",
            description="Rust debug mode (overflow checks enabled)",
        )

    @classmethod
    def rust_release(cls, pointer_size: int = 64) -> "SemanticConfig":
        """Rust release-mode semantics (overflow wraps, bounds still checked)."""
        return cls(
            signed_overflow=OverflowMode.Wrap,
            unsigned_overflow=OverflowMode.Wrap,
            float_model=FloatModel.IEEE754Strict,
            integer_promotion=IntegerPromotionModel.RustStyle,
            error_model=ErrorModel.Panic,
            division_model=DivisionModel.Panic,
            float_to_int=FloatToIntModel.Saturate,
            pointer_model=PointerModel.Provenance,
            array_bounds=ArrayBoundsModel.AlwaysCheck,
            null_pointer_valid=False,
            shift_model=ShiftModel.Mask,
            unwind_model=UnwindModel.Unwind,
            pointer_size=pointer_size,
            int_size=32,
            long_size=64,
            char_signed=True,
            endianness="little",
            language="rust",
            opt_level=2,
            strict_aliasing=True,
            volatile_is_observable=True,
            allow_type_punning=False,
            name="Rust-release",
            description="Rust release mode (overflow wraps)",
        )

    @classmethod
    def rust_release_abort(cls, pointer_size: int = 64) -> "SemanticConfig":
        """Rust release with panic=abort."""
        cfg = cls.rust_release(pointer_size)
        cfg.unwind_model = UnwindModel.Abort
        cfg.name = "Rust-release-abort"
        cfg.description = "Rust release mode with panic=abort"
        return cfg

    # ── Query helpers ────────────────────────────────────────────────────

    @property
    def is_c(self) -> bool:
        return self.language == "c"

    @property
    def is_rust(self) -> bool:
        return self.language == "rust"

    @property
    def is_debug(self) -> bool:
        return self.opt_level == 0

    @property
    def is_release(self) -> bool:
        return self.opt_level >= 1

    @property
    def has_overflow_checks(self) -> bool:
        return (
            self.signed_overflow is OverflowMode.Panic
            or self.unsigned_overflow is OverflowMode.Panic
        )

    @property
    def has_bounds_checks(self) -> bool:
        return self.array_bounds is not ArrayBoundsModel.NoCheck

    @property
    def has_ub(self) -> bool:
        """Return True if this config contains any UB-on-error behavior."""
        return (
            self.signed_overflow is OverflowMode.UB
            or self.error_model is ErrorModel.UB
            or self.division_model is DivisionModel.UB
            or self.float_to_int is FloatToIntModel.UB
            or self.shift_model is ShiftModel.UB_on_overshift
        )

    def int_max(self, width: int, signed: bool) -> int:
        if signed:
            return (1 << (width - 1)) - 1
        return (1 << width) - 1

    def int_min(self, width: int, signed: bool) -> int:
        if signed:
            return -(1 << (width - 1))
        return 0

    def get_c_int_width(self) -> int:
        """Return the width of C `int` for this config."""
        return self.int_size

    def get_overflow_mode(self, signed: bool) -> OverflowMode:
        """Return the overflow mode for the given signedness."""
        return self.signed_overflow if signed else self.unsigned_overflow

    # ── Comparison and diff ──────────────────────────────────────────────

    def diff(self, other: "SemanticConfig") -> "ConfigDiff":
        """Compare this config with another, returning all differences."""
        return ConfigDiff.compute(self, other)

    def is_compatible_with(self, other: "SemanticConfig") -> bool:
        """Return True if both configs would produce identical behaviour
        for all well-defined programs."""
        d = self.diff(other)
        return len(d.critical_diffs) == 0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SemanticConfig):
            return NotImplemented
        return (
            self.signed_overflow == other.signed_overflow
            and self.unsigned_overflow == other.unsigned_overflow
            and self.float_model == other.float_model
            and self.integer_promotion == other.integer_promotion
            and self.error_model == other.error_model
            and self.division_model == other.division_model
            and self.float_to_int == other.float_to_int
            and self.pointer_model == other.pointer_model
            and self.array_bounds == other.array_bounds
            and self.null_pointer_valid == other.null_pointer_valid
            and self.shift_model == other.shift_model
            and self.unwind_model == other.unwind_model
            and self.pointer_size == other.pointer_size
            and self.int_size == other.int_size
            and self.char_signed == other.char_signed
            and self.strict_aliasing == other.strict_aliasing
            and self.allow_type_punning == other.allow_type_punning
        )

    def __hash__(self) -> int:
        return hash((
            self.signed_overflow, self.unsigned_overflow,
            self.float_model, self.integer_promotion,
            self.error_model, self.division_model,
            self.float_to_int, self.pointer_model,
            self.array_bounds, self.shift_model,
        ))

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "signed_overflow": self.signed_overflow.name,
            "unsigned_overflow": self.unsigned_overflow.name,
            "float_model": self.float_model.name,
            "integer_promotion": self.integer_promotion.name,
            "error_model": self.error_model.name,
            "division_model": self.division_model.name,
            "float_to_int": self.float_to_int.name,
            "pointer_model": self.pointer_model.name,
            "array_bounds": self.array_bounds.name,
            "null_pointer_valid": self.null_pointer_valid,
            "shift_model": self.shift_model.name,
            "unwind_model": self.unwind_model.name,
            "pointer_size": self.pointer_size,
            "int_size": self.int_size,
            "long_size": self.long_size,
            "char_signed": self.char_signed,
            "endianness": self.endianness,
            "language": self.language,
            "opt_level": self.opt_level,
            "strict_aliasing": self.strict_aliasing,
            "volatile_is_observable": self.volatile_is_observable,
            "allow_type_punning": self.allow_type_punning,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SemanticConfig":
        """Deserialise from a dictionary."""
        _enum_map: dict[str, type] = {
            "signed_overflow": OverflowMode,
            "unsigned_overflow": OverflowMode,
            "float_model": FloatModel,
            "integer_promotion": IntegerPromotionModel,
            "error_model": ErrorModel,
            "division_model": DivisionModel,
            "float_to_int": FloatToIntModel,
            "pointer_model": PointerModel,
            "array_bounds": ArrayBoundsModel,
            "shift_model": ShiftModel,
            "unwind_model": UnwindModel,
        }
        kwargs: dict[str, Any] = {}
        for key, val in d.items():
            if key in _enum_map:
                kwargs[key] = _enum_map[key][val]
            else:
                kwargs[key] = val
        return cls(**kwargs)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"SemanticConfig: {self.name or '(unnamed)'}",
            f"  Language:            {self.language or 'unspecified'}",
            f"  Opt level:           {self.opt_level}",
            f"  Signed overflow:     {self.signed_overflow}",
            f"  Unsigned overflow:   {self.unsigned_overflow}",
            f"  Float model:         {self.float_model}",
            f"  Integer promotion:   {self.integer_promotion}",
            f"  Error model:         {self.error_model}",
            f"  Division model:      {self.division_model}",
            f"  Float→int model:     {self.float_to_int}",
            f"  Pointer model:       {self.pointer_model}",
            f"  Array bounds:        {self.array_bounds}",
            f"  Shift model:         {self.shift_model}",
            f"  Unwind model:        {self.unwind_model}",
            f"  Pointer size:        {self.pointer_size} bits",
            f"  C int size:          {self.int_size} bits",
            f"  char signed:         {self.char_signed}",
            f"  Strict aliasing:     {self.strict_aliasing}",
            f"  Type punning:        {self.allow_type_punning}",
        ]
        return "\n".join(lines)


# ── Config diff ──────────────────────────────────────────────────────────

@dataclass
class ConfigDiffEntry:
    """A single difference between two configs."""
    field_name: str
    left_value: Any
    right_value: Any
    severity: str = "info"  # "critical", "moderate", "info"
    description: str = ""

    def __str__(self) -> str:
        return (
            f"[{self.severity:8s}] {self.field_name}: "
            f"{self.left_value} → {self.right_value}"
            + (f"  ({self.description})" if self.description else "")
        )


@dataclass
class ConfigDiff:
    """Complete diff between two SemanticConfigs."""
    left: SemanticConfig
    right: SemanticConfig
    entries: list[ConfigDiffEntry] = field(default_factory=list)

    # Severity classification of field names
    _CRITICAL_FIELDS: frozenset[str] = frozenset({
        "signed_overflow", "unsigned_overflow", "division_model",
        "float_to_int", "shift_model", "integer_promotion",
    })
    _MODERATE_FIELDS: frozenset[str] = frozenset({
        "error_model", "pointer_model", "array_bounds",
        "float_model", "strict_aliasing", "allow_type_punning",
        "unwind_model",
    })

    @property
    def critical_diffs(self) -> list[ConfigDiffEntry]:
        return [e for e in self.entries if e.severity == "critical"]

    @property
    def moderate_diffs(self) -> list[ConfigDiffEntry]:
        return [e for e in self.entries if e.severity == "moderate"]

    @property
    def info_diffs(self) -> list[ConfigDiffEntry]:
        return [e for e in self.entries if e.severity == "info"]

    @property
    def has_differences(self) -> bool:
        return len(self.entries) > 0

    @property
    def has_critical(self) -> bool:
        return len(self.critical_diffs) > 0

    @classmethod
    def compute(cls, left: SemanticConfig, right: SemanticConfig) -> "ConfigDiff":
        """Compute the diff between two configs."""
        diff = cls(left=left, right=right)

        _DESCRIPTIONS: dict[str, str] = {
            "signed_overflow": "Signed overflow handling differs",
            "unsigned_overflow": "Unsigned overflow handling differs",
            "float_model": "Floating-point evaluation model differs",
            "integer_promotion": "Integer promotion rules differ",
            "error_model": "Error handling paradigm differs",
            "division_model": "Division-by-zero handling differs",
            "float_to_int": "Float-to-int out-of-range handling differs",
            "pointer_model": "Pointer/memory model differs",
            "array_bounds": "Array bounds checking policy differs",
            "null_pointer_valid": "Null pointer validity differs",
            "shift_model": "Shift overflow handling differs",
            "unwind_model": "Unwinding behavior differs",
            "pointer_size": "Pointer size differs",
            "int_size": "C int width differs",
            "long_size": "C long width differs",
            "char_signed": "char signedness differs",
            "endianness": "Byte order differs",
            "strict_aliasing": "Strict aliasing assumption differs",
            "allow_type_punning": "Type punning allowance differs",
            "volatile_is_observable": "Volatile observability differs",
        }

        _fields_to_compare = [
            "signed_overflow", "unsigned_overflow", "float_model",
            "integer_promotion", "error_model", "division_model",
            "float_to_int", "pointer_model", "array_bounds",
            "null_pointer_valid", "shift_model", "unwind_model",
            "pointer_size", "int_size", "long_size",
            "char_signed", "endianness",
            "strict_aliasing", "allow_type_punning",
            "volatile_is_observable",
        ]

        for fname in _fields_to_compare:
            lv = getattr(left, fname)
            rv = getattr(right, fname)
            if lv != rv:
                if fname in cls._CRITICAL_FIELDS:
                    sev = "critical"
                elif fname in cls._MODERATE_FIELDS:
                    sev = "moderate"
                else:
                    sev = "info"
                diff.entries.append(ConfigDiffEntry(
                    field_name=fname,
                    left_value=lv,
                    right_value=rv,
                    severity=sev,
                    description=_DESCRIPTIONS.get(fname, ""),
                ))

        return diff

    def summary(self) -> str:
        lines = [
            f"Config diff: {self.left.name or 'left'} vs {self.right.name or 'right'}",
            f"  Total differences: {len(self.entries)}",
            f"  Critical:          {len(self.critical_diffs)}",
            f"  Moderate:          {len(self.moderate_diffs)}",
            f"  Info:              {len(self.info_diffs)}",
        ]
        if self.entries:
            lines.append("  Details:")
            for e in self.entries:
                lines.append(f"    {e}")
        return "\n".join(lines)


# ── Convenience ──────────────────────────────────────────────────────────

def compare_c_rust() -> ConfigDiff:
    """Compare default C11 and Rust-release configs."""
    return SemanticConfig.c11().diff(SemanticConfig.rust_release())


def compare_rust_modes() -> ConfigDiff:
    """Compare Rust debug and release configs."""
    return SemanticConfig.rust_debug().diff(SemanticConfig.rust_release())


def all_presets() -> list[SemanticConfig]:
    """Return all built-in preset configurations."""
    return [
        SemanticConfig.c11(),
        SemanticConfig.c11_optimized(),
        SemanticConfig.rust_debug(),
        SemanticConfig.rust_release(),
        SemanticConfig.rust_release_abort(),
    ]
