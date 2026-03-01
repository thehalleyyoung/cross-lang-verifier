"""
I/O function models: C stdio ↔ Rust std::io / std::fs.

Models printf format strings, file operations (fopen↔File::open),
error code↔Result mapping, errno↔io::Error conversion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict, Any, Tuple

import z3

from .memory import DivergenceLevel, FunctionEquivalence, ModelResult


# ---------------------------------------------------------------------------
# Error code mapping
# ---------------------------------------------------------------------------

class CErrorCode(Enum):
    """Common C errno values."""
    SUCCESS = 0
    EPERM = 1
    ENOENT = 2
    EIO = 5
    ENOMEM = 12
    EACCES = 13
    EEXIST = 17
    ENOTDIR = 20
    EISDIR = 21
    EINVAL = 22
    EMFILE = 24
    ENFILE = 23
    ENOSPC = 28
    EROFS = 30
    ENAMETOOLONG = 36
    ENOTEMPTY = 39
    EWOULDBLOCK = 11
    EAGAIN = 11
    EBADF = 9

    def to_rust_error_kind(self) -> str:
        """Map C errno to Rust io::ErrorKind."""
        mapping = {
            CErrorCode.EPERM: "PermissionDenied",
            CErrorCode.ENOENT: "NotFound",
            CErrorCode.EIO: "Other",
            CErrorCode.ENOMEM: "OutOfMemory",
            CErrorCode.EACCES: "PermissionDenied",
            CErrorCode.EEXIST: "AlreadyExists",
            CErrorCode.ENOTDIR: "NotADirectory",
            CErrorCode.EISDIR: "IsADirectory",
            CErrorCode.EINVAL: "InvalidInput",
            CErrorCode.EMFILE: "Other",
            CErrorCode.ENFILE: "Other",
            CErrorCode.ENOSPC: "StorageFull",
            CErrorCode.EROFS: "ReadOnlyFilesystem",
            CErrorCode.ENAMETOOLONG: "InvalidInput",
            CErrorCode.ENOTEMPTY: "DirectoryNotEmpty",
            CErrorCode.EWOULDBLOCK: "WouldBlock",
            CErrorCode.EBADF: "Other",
        }
        return mapping.get(self, "Other")


@dataclass
class ErrorMapping:
    """Maps C error patterns to Rust Result patterns."""
    c_pattern: str              # How C signals error
    rust_pattern: str           # How Rust signals error
    conversion_notes: str

    @staticmethod
    def common_mappings() -> List[ErrorMapping]:
        return [
            ErrorMapping(
                "NULL return + errno",
                "Err(io::Error::last_os_error())",
                "C fopen returns NULL, Rust File::open returns Err",
            ),
            ErrorMapping(
                "EOF / -1 return",
                "Ok(0) / Err",
                "C fread returns 0/short, Rust read returns Ok(0) for EOF",
            ),
            ErrorMapping(
                "negative return + errno",
                "Err(io::Error)",
                "C write returns -1, Rust write returns Err",
            ),
            ErrorMapping(
                "ferror() check",
                "Result::is_err()",
                "C requires checking ferror after operations",
            ),
        ]


# ---------------------------------------------------------------------------
# printf model
# ---------------------------------------------------------------------------

@dataclass
class FormatSpec:
    """Parsed format specifier from a printf-style format string."""
    flags: str = ""             # -, +, 0, space, #
    width: Optional[int] = None
    precision: Optional[int] = None
    length_modifier: str = ""   # h, hh, l, ll, z, j, t
    conversion: str = ""        # d, i, u, o, x, X, f, e, g, s, c, p, n, %

    @property
    def c_type(self) -> str:
        """Expected C argument type."""
        type_map = {
            ("d", ""): "int", ("d", "l"): "long", ("d", "ll"): "long long",
            ("d", "h"): "short", ("d", "hh"): "char",
            ("i", ""): "int", ("i", "l"): "long", ("i", "ll"): "long long",
            ("u", ""): "unsigned", ("u", "l"): "unsigned long",
            ("u", "ll"): "unsigned long long",
            ("x", ""): "unsigned", ("X", ""): "unsigned",
            ("o", ""): "unsigned",
            ("f", ""): "double", ("f", "l"): "long double",
            ("e", ""): "double", ("e", "l"): "long double",
            ("g", ""): "double", ("g", "l"): "long double",
            ("s", ""): "char*", ("c", ""): "int",
            ("p", ""): "void*",
            ("n", ""): "int*",
        }
        return type_map.get((self.conversion, self.length_modifier), "unknown")

    @property
    def rust_format(self) -> str:
        """Equivalent Rust format specifier."""
        conv_map = {
            "d": "{}", "i": "{}", "u": "{}",
            "x": "{:x}", "X": "{:X}", "o": "{:o}",
            "f": "{}", "e": "{:e}", "g": "{}",
            "s": "{}", "c": "{}", "p": "{:p}",
        }
        base = conv_map.get(self.conversion, "{}")
        return base


class PrintfModel:
    """
    Model for printf/fprintf/sprintf ↔ print!/write!/format!.
    
    Major differences: format string syntax, type safety, buffer safety.
    """

    equivalence = FunctionEquivalence(
        c_function="printf / fprintf / sprintf / snprintf",
        rust_equivalent="print! / write! / format! / writeln!",
        divergence_level=DivergenceLevel.HIGH,
        preconditions=["Format string matches argument types"],
        divergence_points=[
            "Type safety: C has no compile-time checking, Rust checks at compile time",
            "Buffer overflow: sprintf can overflow, format! allocates",
            "%n specifier: C writes count (security risk), Rust has no equivalent",
            "Locale: C can be locale-dependent, Rust is not by default",
            "Return value: C returns char count, Rust returns fmt::Result",
            "Null format string: C is UB, Rust would panic",
        ],
    )

    DANGEROUS_SPECIFIERS = {"%n"}  # Can write to memory in C

    @staticmethod
    def apply(
        dst: Optional[z3.BitVecRef],
        format_str_len: z3.BitVecRef,
        num_args: int,
        has_n_specifier: bool = False,
        addr_width: int = 64,
    ) -> ModelResult:
        result = ModelResult()

        ret = z3.BitVec("printf_result", 32)

        if has_n_specifier:
            result.divergence_condition = z3.BoolVal(True)

        if dst is not None:
            null = z3.BitVecVal(0, addr_width)
            result.error_condition = dst == null

        result.constraints.append(z3.Implies(
            format_str_len == z3.BitVecVal(0, format_str_len.size()),
            ret == z3.BitVecVal(0, 32),
        ))

        result.return_value = ret
        return result


# ---------------------------------------------------------------------------
# fopen model
# ---------------------------------------------------------------------------

class FopenModel:
    """
    Model for fopen ↔ File::open / File::create / OpenOptions.
    
    C: Returns FILE* or NULL, sets errno.
    Rust: Returns io::Result<File>.
    """

    equivalence = FunctionEquivalence(
        c_function="fopen / freopen",
        rust_equivalent="File::open / File::create / OpenOptions::open",
        divergence_level=DivergenceLevel.MODERATE,
        preconditions=[],
        divergence_points=[
            "Error handling: C returns NULL + errno, Rust returns Result::Err",
            "Mode string: C uses 'r','w','a','+','b'; Rust uses OpenOptions builder",
            "Binary mode: C requires 'b' flag, Rust is always binary",
            "Text mode: C does newline translation on Windows, Rust doesn't",
            "Resource cleanup: C requires fclose, Rust has Drop trait",
        ],
    )

    MODE_MAP = {
        "r":  {"read": True, "write": False, "create": False, "truncate": False, "append": False},
        "w":  {"read": False, "write": True, "create": True, "truncate": True, "append": False},
        "a":  {"read": False, "write": True, "create": True, "truncate": False, "append": True},
        "r+": {"read": True, "write": True, "create": False, "truncate": False, "append": False},
        "w+": {"read": True, "write": True, "create": True, "truncate": True, "append": False},
        "a+": {"read": True, "write": True, "create": True, "truncate": False, "append": True},
    }

    @staticmethod
    def apply(
        path_ptr: z3.BitVecRef,
        mode_is_read: z3.BoolRef,
        mode_is_write: z3.BoolRef,
        addr_width: int = 64,
    ) -> ModelResult:
        result = ModelResult()
        null = z3.BitVecVal(0, addr_width)

        result.error_condition = path_ptr == null

        file_handle = z3.BitVec("fopen_result", addr_width)

        # C: returns NULL on failure
        success = z3.Bool("fopen_success")
        result.constraints.append(z3.Implies(z3.Not(success), file_handle == null))
        result.constraints.append(z3.Implies(success, file_handle != null))

        result.return_value = file_handle
        return result


# ---------------------------------------------------------------------------
# fclose model
# ---------------------------------------------------------------------------

class FcloseModel:
    """
    Model for fclose ↔ Drop / File::sync_all.
    
    C: fclose flushes and closes. Returns EOF on error.
    Rust: Drop flushes (ignoring errors), explicit sync_all for checking.
    """

    equivalence = FunctionEquivalence(
        c_function="fclose",
        rust_equivalent="Drop(File) / File::sync_all",
        divergence_level=DivergenceLevel.MODERATE,
        preconditions=["File handle is valid"],
        divergence_points=[
            "Error on close: C returns EOF and sets errno, Rust Drop ignores flush errors",
            "Double close: C is UB, Rust prevents via ownership",
            "Use after close: C is UB, Rust prevents at compile time",
            "Flush: C fclose flushes, Rust Drop flushes but ignores error",
        ],
    )

    @staticmethod
    def apply(
        file_handle: z3.BitVecRef,
        addr_width: int = 64,
    ) -> ModelResult:
        result = ModelResult()
        null = z3.BitVecVal(0, addr_width)

        # NULL file handle is UB in C
        result.error_condition = file_handle == null

        ret = z3.BitVec("fclose_result", 32)
        eof = z3.BitVecVal(0xFFFFFFFF, 32)  # EOF is typically -1

        success = z3.Bool("fclose_success")
        result.constraints.append(z3.Implies(success, ret == z3.BitVecVal(0, 32)))
        result.constraints.append(z3.Implies(z3.Not(success), ret == eof))

        # Divergence: Rust Drop silently ignores flush errors
        result.divergence_condition = z3.Not(success)

        result.return_value = ret
        return result


# ---------------------------------------------------------------------------
# fread / fwrite model
# ---------------------------------------------------------------------------

class FreadModel:
    """
    Model for fread/fwrite ↔ Read::read / Write::write.
    
    C: Returns number of items read (not bytes). Short read on EOF/error.
    Rust: Returns Result<usize> with byte count.
    """

    equivalence = FunctionEquivalence(
        c_function="fread / fwrite",
        rust_equivalent="Read::read / Write::write / read_exact / write_all",
        divergence_level=DivergenceLevel.MODERATE,
        preconditions=["Buffer is large enough", "File handle is valid"],
        divergence_points=[
            "Return value: C returns item count, Rust returns byte count",
            "Short read: C requires ferror/feof check, Rust returns Ok(n) where n < requested",
            "Error: C returns 0 + ferror(), Rust returns Err",
            "read_exact: Rust variant that ensures full read or returns Err",
            "Partial write: C may write less, Rust write_all ensures all written",
        ],
    )

    @staticmethod
    def apply(
        buffer: z3.BitVecRef,
        elem_size: z3.BitVecRef,
        count: z3.BitVecRef,
        file_handle: z3.BitVecRef,
        is_write: bool = False,
        addr_width: int = 64,
    ) -> ModelResult:
        result = ModelResult()
        null = z3.BitVecVal(0, addr_width)

        result.error_condition = z3.Or(
            buffer == null,
            file_handle == null,
        )

        # Return value: number of items (C) vs bytes (Rust)
        c_ret = z3.BitVec("fread_items", 64)
        rust_ret = z3.BitVec("read_bytes", 64)

        # C returns items <= count
        result.constraints.append(z3.ULE(c_ret, z3.ZeroExt(64 - count.size(), count)))

        # Rust returns bytes = items * elem_size
        result.constraints.append(
            rust_ret == c_ret * z3.ZeroExt(64 - elem_size.size(), elem_size)
        )

        result.return_value = c_ret
        return result


# ---------------------------------------------------------------------------
# errno model
# ---------------------------------------------------------------------------

class ErrnoModel:
    """
    Model for errno ↔ io::Error / Result.
    
    C: errno is a thread-local int, set by many functions.
    Rust: io::Error wraps OS error codes, returned via Result.
    """

    equivalence = FunctionEquivalence(
        c_function="errno (global)",
        rust_equivalent="io::Error / io::Result",
        divergence_level=DivergenceLevel.HIGH,
        preconditions=[],
        divergence_points=[
            "Global state: C errno is thread-local global, Rust errors are local values",
            "Clobbering: C errno can be overwritten by next call, Rust Result is consumed",
            "Checking: C requires manual errno check, Rust forces handling via Result",
            "Zero errno: C 0 means no error, Rust has no equivalent",
            "Thread safety: C errno is thread-local, Rust Result is Send+Sync",
        ],
    )

    @staticmethod
    def apply(
        errno_val: z3.BitVecRef,
    ) -> ModelResult:
        result = ModelResult()

        is_error = errno_val != z3.BitVecVal(0, errno_val.size())
        rust_is_err = z3.Bool("rust_result_is_err")

        # When errno != 0, Rust would have Err
        result.constraints.append(z3.Implies(is_error, rust_is_err))
        result.constraints.append(z3.Implies(z3.Not(is_error), z3.Not(rust_is_err)))

        # Divergence: errno can be silently ignored in C
        result.divergence_condition = is_error

        return result


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

class IOFunctionModels:
    """Registry of all I/O function models."""

    models = {
        "printf": PrintfModel,
        "fprintf": PrintfModel,
        "sprintf": PrintfModel,
        "snprintf": PrintfModel,
        "fopen": FopenModel,
        "freopen": FopenModel,
        "fclose": FcloseModel,
        "fread": FreadModel,
        "fwrite": FreadModel,
    }

    error_model = ErrnoModel

    @classmethod
    def get_model(cls, func_name: str) -> Optional[type]:
        return cls.models.get(func_name)

    @classmethod
    def get_equivalence(cls, func_name: str) -> Optional[FunctionEquivalence]:
        model = cls.get_model(func_name)
        if model and hasattr(model, 'equivalence'):
            return model.equivalence
        return None

    @classmethod
    def all_equivalences(cls) -> List[FunctionEquivalence]:
        seen = set()
        result = []
        for model_cls in cls.models.values():
            if id(model_cls) not in seen and hasattr(model_cls, 'equivalence'):
                seen.add(id(model_cls))
                result.append(model_cls.equivalence)
        # Also include errno
        result.append(ErrnoModel.equivalence)
        return result

    @classmethod
    def get_error_mapping(cls, errno_code: int) -> Optional[str]:
        """Map C errno to Rust io::ErrorKind."""
        try:
            c_err = CErrorCode(errno_code)
            return c_err.to_rust_error_kind()
        except ValueError:
            return None

    @classmethod
    def summary(cls) -> str:
        lines = ["I/O Function Models:"]
        for eq in cls.all_equivalences():
            lines.append(f"  {eq.summary()}")
            for dp in eq.divergence_points:
                lines.append(f"    ⚠ {dp}")
        return "\n".join(lines)
