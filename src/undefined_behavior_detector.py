"""
Undefined behavior detector for C code.
Detects signed integer overflow, null pointer dereference, out-of-bounds access,
use-after-free, double-free, uninitialized reads, strict aliasing violations,
sequence point violations, data races, and shift overflow.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from enum import Enum, auto
import copy


# ---------------------------------------------------------------------------
# UB types per C standard
# ---------------------------------------------------------------------------

class UBType(Enum):
    SIGNED_OVERFLOW = "signed-integer-overflow"
    NULL_DEREF = "null-pointer-dereference"
    OOB_ACCESS = "out-of-bounds-access"
    USE_AFTER_FREE = "use-after-free"
    DOUBLE_FREE = "double-free"
    UNINIT_READ = "uninitialized-variable-read"
    STRICT_ALIASING = "strict-aliasing-violation"
    SEQUENCE_POINT = "sequence-point-violation"
    DATA_RACE = "data-race"
    SHIFT_OVERFLOW = "shift-overflow"
    DIVISION_BY_ZERO = "division-by-zero"
    INVALID_CAST = "invalid-cast"
    SIGNED_OVERFLOW_NEGATION = "signed-overflow-negation"
    INFINITE_LOOP = "potentially-infinite-loop"
    STACK_OVERFLOW = "stack-overflow"
    FORMAT_STRING = "format-string-mismatch"
    MISSING_RETURN = "missing-return-value"
    OVERLAPPING_MEMCPY = "overlapping-memcpy"
    ALIGNMENT_VIOLATION = "alignment-violation"
    VLA_OVERFLOW = "vla-size-overflow"


C_STANDARD_SECTIONS: Dict[UBType, str] = {
    UBType.SIGNED_OVERFLOW: "C11 §6.5/5: If an exceptional condition occurs during evaluation of an expression (that is, if the result is not mathematically defined or not in the range of representable values for its type), the behavior is undefined.",
    UBType.NULL_DEREF: "C11 §6.5.3.2/4: If an invalid value has been assigned to the pointer, the behavior of the unary * operator is undefined.",
    UBType.OOB_ACCESS: "C11 §6.5.6/8: If the result [of pointer arithmetic] points one past the last element of the array object, it shall not be used as the operand of a unary * operator.",
    UBType.USE_AFTER_FREE: "C11 §6.2.4/2: The value of a pointer becomes indeterminate when the object it points to reaches the end of its lifetime.",
    UBType.DOUBLE_FREE: "C11 §7.22.3.3/2: If the argument [to free] does not match a pointer earlier returned by a memory management function, or if the space has been deallocated by a call to free or realloc, the behavior is undefined.",
    UBType.UNINIT_READ: "C11 §6.3.2.1/2: If the lvalue designates an object of automatic storage duration that could have been declared with the register storage class (never had its address taken), and that object is uninitialized, the behavior is undefined.",
    UBType.STRICT_ALIASING: "C11 §6.5/7: An object shall have its stored value accessed only by an lvalue expression that has one of the compatible types listed.",
    UBType.SEQUENCE_POINT: "C11 §6.5/2: If a side effect on a scalar object is unsequenced relative to either a different side effect on the same scalar object or a value computation using the value of the same scalar object, the behavior is undefined.",
    UBType.DATA_RACE: "C11 §5.1.2.4/25: Two expression evaluations conflict if one of them modifies a memory location and the other one reads or modifies the same memory location.",
    UBType.SHIFT_OVERFLOW: "C11 §6.5.7/3-4: If the value of the right operand is negative or is greater than or equal to the width of the promoted left operand, the behavior is undefined.",
}


# ---------------------------------------------------------------------------
# UB violation
# ---------------------------------------------------------------------------

@dataclass
class UBViolation:
    type: UBType
    location: str
    severity: str = "error"
    explanation: str = ""
    c_standard_section: str = ""
    code_snippet: str = ""
    variable: str = ""
    confidence: float = 1.0

    def __post_init__(self):
        if not self.c_standard_section:
            self.c_standard_section = C_STANDARD_SECTIONS.get(self.type, "")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "location": self.location,
            "severity": self.severity,
            "explanation": self.explanation,
            "c_standard_section": self.c_standard_section,
            "code_snippet": self.code_snippet,
            "variable": self.variable,
            "confidence": self.confidence,
        }


# ---------------------------------------------------------------------------
# Internal tracking state
# ---------------------------------------------------------------------------

class AllocState(Enum):
    ALLOCATED = auto()
    FREED = auto()
    STACK = auto()
    UNKNOWN = auto()


@dataclass
class VarInfo:
    name: str
    type_name: str = "int"
    initialized: bool = False
    may_be_null: bool = False
    alloc_state: AllocState = AllocState.UNKNOWN
    points_to: Set[str] = field(default_factory=set)
    buffer_size: Optional[int] = None
    value_range: Optional[Tuple[float, float]] = None
    modified_count: int = 0
    last_modified_at: str = ""
    is_pointer: bool = False
    is_shared: bool = False
    thread_id: Optional[int] = None
    declared_type: str = ""
    cast_history: List[str] = field(default_factory=list)

    def copy(self) -> "VarInfo":
        v = VarInfo(
            name=self.name, type_name=self.type_name,
            initialized=self.initialized, may_be_null=self.may_be_null,
            alloc_state=self.alloc_state, points_to=set(self.points_to),
            buffer_size=self.buffer_size,
            value_range=self.value_range,
            modified_count=self.modified_count,
            last_modified_at=self.last_modified_at,
            is_pointer=self.is_pointer, is_shared=self.is_shared,
            thread_id=self.thread_id, declared_type=self.declared_type,
            cast_history=list(self.cast_history),
        )
        return v


@dataclass
class AnalysisState:
    variables: Dict[str, VarInfo] = field(default_factory=dict)
    scope_stack: List[Set[str]] = field(default_factory=list)
    violations: List[UBViolation] = field(default_factory=list)
    current_thread: int = 0
    in_sequence: bool = False
    modifications_in_sequence: Dict[str, int] = field(default_factory=dict)

    def copy(self) -> "AnalysisState":
        return AnalysisState(
            variables={k: v.copy() for k, v in self.variables.items()},
            scope_stack=[set(s) for s in self.scope_stack],
            violations=list(self.violations),
            current_thread=self.current_thread,
            in_sequence=self.in_sequence,
            modifications_in_sequence=dict(self.modifications_in_sequence),
        )

    def push_scope(self) -> None:
        self.scope_stack.append(set())

    def pop_scope(self) -> None:
        if self.scope_stack:
            leaving = self.scope_stack.pop()
            for var in leaving:
                self.variables.pop(var, None)

    def add_local(self, name: str) -> None:
        if self.scope_stack:
            self.scope_stack[-1].add(name)

    def join(self, other: "AnalysisState") -> "AnalysisState":
        result = self.copy()
        for k, v in other.variables.items():
            if k in result.variables:
                existing = result.variables[k]
                if v.may_be_null and not existing.may_be_null:
                    existing.may_be_null = True
                if not v.initialized:
                    existing.initialized = False
                if v.alloc_state != existing.alloc_state:
                    existing.alloc_state = AllocState.UNKNOWN
                existing.points_to |= v.points_to
            else:
                result.variables[k] = v.copy()
        result.violations.extend(other.violations)
        return result


# ---------------------------------------------------------------------------
# Type compatibility for strict aliasing
# ---------------------------------------------------------------------------

COMPATIBLE_TYPE_GROUPS = [
    {"char", "signed char", "unsigned char"},
    {"short", "unsigned short", "int16_t", "uint16_t"},
    {"int", "unsigned int", "int32_t", "uint32_t"},
    {"long", "unsigned long"},
    {"long long", "unsigned long long", "int64_t", "uint64_t"},
    {"float"},
    {"double"},
]

def types_compatible_for_aliasing(type_a: str, type_b: str) -> bool:
    if type_a == type_b:
        return True
    if "char" in type_a or "char" in type_b:
        return True
    if type_a == "void*" or type_b == "void*":
        return True
    for group in COMPATIBLE_TYPE_GROUPS:
        if type_a in group and type_b in group:
            return True
    a_base = type_a.rstrip("*").strip()
    b_base = type_b.rstrip("*").strip()
    if a_base == b_base:
        return True
    return False


INT_MIN = -2147483648
INT_MAX = 2147483647
UINT_MAX = 4294967295
TYPE_WIDTHS: Dict[str, int] = {
    "char": 8, "short": 16, "int": 32, "long": 64,
    "long long": 64, "int8_t": 8, "int16_t": 16,
    "int32_t": 32, "int64_t": 64,
    "uint8_t": 8, "uint16_t": 16, "uint32_t": 32, "uint64_t": 64,
}

TYPE_RANGES: Dict[str, Tuple[int, int]] = {
    "char": (-128, 127),
    "unsigned char": (0, 255),
    "short": (-32768, 32767),
    "unsigned short": (0, 65535),
    "int": (INT_MIN, INT_MAX),
    "unsigned int": (0, UINT_MAX),
    "long": (-9223372036854775808, 9223372036854775807),
    "unsigned long": (0, 18446744073709551615),
}


# ---------------------------------------------------------------------------
# UB Detector
# ---------------------------------------------------------------------------

class UBDetector:
    """Detect undefined behavior in C code representations."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._state = AnalysisState()
        self._violations: List[UBViolation] = []
        self._function_calls: Dict[str, int] = {}

    def _reset(self) -> None:
        self._state = AnalysisState()
        self._violations = []
        self._function_calls = {}

    # --- signed integer overflow ---
    def _check_signed_overflow(self, op: str, left_range: Optional[Tuple[float, float]],
                               right_range: Optional[Tuple[float, float]],
                               result_type: str, loc: str) -> None:
        if result_type.startswith("unsigned"):
            return
        type_range = TYPE_RANGES.get(result_type, (INT_MIN, INT_MAX))
        lo, hi = type_range

        if left_range is None or right_range is None:
            return

        l_lo, l_hi = left_range
        r_lo, r_hi = right_range

        may_overflow = False
        if op == "+":
            if l_hi + r_hi > hi or l_lo + r_lo < lo:
                may_overflow = True
        elif op == "-":
            if l_hi - r_lo > hi or l_lo - r_hi < lo:
                may_overflow = True
        elif op == "*":
            products = [l_lo * r_lo, l_lo * r_hi, l_hi * r_lo, l_hi * r_hi]
            if max(products) > hi or min(products) < lo:
                may_overflow = True
        elif op == "/":
            if 0 in (r_lo, r_hi) or (r_lo <= 0 <= r_hi):
                self._add_violation(UBType.DIVISION_BY_ZERO, loc,
                                    "Division by zero is undefined behavior",
                                    confidence=0.8)
            if l_lo <= lo and (r_lo <= -1 <= r_hi):
                may_overflow = True
        elif op == "-" and right_range is None:
            if l_lo == lo:
                may_overflow = True

        if may_overflow:
            self._add_violation(
                UBType.SIGNED_OVERFLOW, loc,
                f"Signed integer overflow possible in `{op}` operation on type `{result_type}`: "
                f"left range [{l_lo}, {l_hi}], right range [{r_lo}, {r_hi}]",
                confidence=0.7,
            )

    # --- null pointer dereference ---
    def _check_null_deref(self, ptr_name: str, loc: str) -> None:
        info = self._state.variables.get(ptr_name)
        if not info:
            return
        if info.may_be_null:
            self._add_violation(
                UBType.NULL_DEREF, loc,
                f"Pointer `{ptr_name}` may be null when dereferenced",
                variable=ptr_name,
            )

    # --- out-of-bounds ---
    def _check_oob_access(self, base: str, index_range: Optional[Tuple[float, float]],
                          loc: str) -> None:
        info = self._state.variables.get(base)
        if not info or info.buffer_size is None:
            return
        if index_range is None:
            return
        idx_lo, idx_hi = index_range
        if idx_lo < 0:
            self._add_violation(
                UBType.OOB_ACCESS, loc,
                f"Negative array index on `{base}`: index may be {idx_lo}",
                variable=base,
            )
        if idx_hi >= info.buffer_size:
            self._add_violation(
                UBType.OOB_ACCESS, loc,
                f"Array index on `{base}` may exceed bounds: index up to {idx_hi}, size {info.buffer_size}",
                variable=base,
            )

    # --- use-after-free ---
    def _check_use_after_free(self, ptr_name: str, loc: str) -> None:
        info = self._state.variables.get(ptr_name)
        if not info:
            return
        if info.alloc_state == AllocState.FREED:
            self._add_violation(
                UBType.USE_AFTER_FREE, loc,
                f"Use of pointer `{ptr_name}` after it has been freed",
                variable=ptr_name,
            )
        for target in info.points_to:
            target_info = self._state.variables.get(target)
            if target_info and target_info.alloc_state == AllocState.FREED:
                self._add_violation(
                    UBType.USE_AFTER_FREE, loc,
                    f"Pointer `{ptr_name}` references freed memory `{target}`",
                    variable=ptr_name,
                )

    # --- double free ---
    def _check_double_free(self, ptr_name: str, loc: str) -> None:
        info = self._state.variables.get(ptr_name)
        if not info:
            return
        if info.alloc_state == AllocState.FREED:
            self._add_violation(
                UBType.DOUBLE_FREE, loc,
                f"Double free of pointer `{ptr_name}`",
                variable=ptr_name,
            )

    # --- uninitialized read ---
    def _check_uninit_read(self, var_name: str, loc: str) -> None:
        info = self._state.variables.get(var_name)
        if not info:
            return
        if not info.initialized:
            self._add_violation(
                UBType.UNINIT_READ, loc,
                f"Read of potentially uninitialized variable `{var_name}`",
                variable=var_name,
            )

    # --- strict aliasing ---
    def _check_strict_aliasing(self, ptr_name: str, cast_to: str,
                               loc: str) -> None:
        info = self._state.variables.get(ptr_name)
        if not info:
            return

        original_type = info.declared_type or info.type_name
        original_base = original_type.rstrip("*").strip()
        cast_base = cast_to.rstrip("*").strip()

        if not types_compatible_for_aliasing(original_base, cast_base):
            self._add_violation(
                UBType.STRICT_ALIASING, loc,
                f"Strict aliasing violation: casting `{ptr_name}` from `{original_type}` to `{cast_to}*`. "
                f"Types `{original_base}` and `{cast_base}` are not compatible for aliasing.",
                variable=ptr_name,
            )

        info.cast_history.append(cast_to)
        if len(info.cast_history) > 2:
            self._add_violation(
                UBType.STRICT_ALIASING, loc,
                f"Pointer `{ptr_name}` has been cast through multiple types: {info.cast_history}. "
                f"This pattern often indicates type punning.",
                variable=ptr_name,
                severity="warning",
                confidence=0.6,
            )

    # --- sequence point violations ---
    def _check_sequence_point(self, var_name: str, is_modification: bool,
                              loc: str) -> None:
        if not self._state.in_sequence:
            return

        count = self._state.modifications_in_sequence.get(var_name, 0)
        if is_modification:
            if count > 0:
                self._add_violation(
                    UBType.SEQUENCE_POINT, loc,
                    f"Variable `{var_name}` modified multiple times between sequence points",
                    variable=var_name,
                )
            self._state.modifications_in_sequence[var_name] = count + 1
        else:
            if count > 0:
                self._add_violation(
                    UBType.SEQUENCE_POINT, loc,
                    f"Variable `{var_name}` read and modified between the same sequence points",
                    variable=var_name,
                    severity="warning",
                    confidence=0.8,
                )

    def _enter_expression(self) -> None:
        self._state.in_sequence = True
        self._state.modifications_in_sequence = {}

    def _exit_expression(self) -> None:
        self._state.in_sequence = False
        self._state.modifications_in_sequence = {}

    # --- data races ---
    def _check_data_race(self, var_name: str, is_write: bool,
                         thread_id: int, loc: str) -> None:
        info = self._state.variables.get(var_name)
        if not info:
            return
        if not info.is_shared:
            return
        if info.thread_id is not None and info.thread_id != thread_id:
            if is_write or info.modified_count > 0:
                self._add_violation(
                    UBType.DATA_RACE, loc,
                    f"Data race on `{var_name}`: accessed from thread {thread_id} "
                    f"while previously {'modified' if info.modified_count > 0 else 'accessed'} "
                    f"from thread {info.thread_id} without synchronization",
                    variable=var_name,
                )

    # --- shift overflow ---
    def _check_shift_overflow(self, left_type: str,
                              shift_range: Optional[Tuple[float, float]],
                              loc: str) -> None:
        width = TYPE_WIDTHS.get(left_type, 32)
        if shift_range is None:
            return
        s_lo, s_hi = shift_range
        if s_lo < 0:
            self._add_violation(
                UBType.SHIFT_OVERFLOW, loc,
                f"Shift amount may be negative ({s_lo}), which is undefined behavior",
            )
        if s_hi >= width:
            self._add_violation(
                UBType.SHIFT_OVERFLOW, loc,
                f"Shift amount ({s_hi}) may be >= type width ({width} bits for `{left_type}`)",
            )

    # --- helper ---
    def _add_violation(self, ub_type: UBType, location: str,
                       explanation: str, variable: str = "",
                       severity: str = "error",
                       confidence: float = 1.0,
                       code_snippet: str = "") -> None:
        v = UBViolation(
            type=ub_type, location=location, severity=severity,
            explanation=explanation, variable=variable,
            confidence=confidence, code_snippet=code_snippet,
        )
        self._violations.append(v)
        self._state.violations.append(v)

    # --- expression analysis ---
    def _analyze_expr(self, expr: Any, loc: str) -> Optional[Tuple[float, float]]:
        if expr is None:
            return None
        if isinstance(expr, (int, float)):
            return (expr, expr)
        if isinstance(expr, str):
            self._check_uninit_read(expr, loc)
            info = self._state.variables.get(expr)
            if info:
                return info.value_range
            return None
        if not isinstance(expr, dict):
            return None

        kind = expr.get("kind", "")

        if kind == "const":
            v = expr.get("value", 0)
            return (v, v)

        if kind == "var":
            name = expr.get("name", "")
            self._check_uninit_read(name, loc)
            self._check_sequence_point(name, False, loc)
            info = self._state.variables.get(name)
            if info:
                return info.value_range
            return None

        if kind == "binop":
            op = expr.get("op", "+")
            left = self._analyze_expr(expr.get("left"), loc)
            right = self._analyze_expr(expr.get("right"), loc)

            result_type = expr.get("type", "int")
            self._check_signed_overflow(op, left, right, result_type, loc)

            if op in ("<<", ">>"):
                left_type = expr.get("left_type", "int")
                self._check_shift_overflow(left_type, right, loc)

            if op in ("/", "%") and right:
                r_lo, r_hi = right
                if r_lo <= 0 <= r_hi:
                    self._add_violation(UBType.DIVISION_BY_ZERO, loc,
                                        f"Possible division/modulo by zero")

            if left is not None and right is not None:
                return self._compute_range(op, left, right)
            return None

        if kind == "unop":
            op = expr.get("op", "")
            operand = self._analyze_expr(expr.get("operand"), loc)
            if op in ("++", "--"):
                var = expr.get("operand", {})
                if isinstance(var, dict) and var.get("kind") == "var":
                    var_name = var.get("name", "")
                    self._check_sequence_point(var_name, True, loc)
                    if operand:
                        if op == "++":
                            self._check_signed_overflow("+", operand, (1, 1),
                                                        expr.get("type", "int"), loc)
                        else:
                            self._check_signed_overflow("-", operand, (1, 1),
                                                        expr.get("type", "int"), loc)
            if op == "-" and operand:
                if operand[0] == INT_MIN:
                    self._add_violation(UBType.SIGNED_OVERFLOW_NEGATION, loc,
                                        "Negation of INT_MIN is undefined behavior")
            return operand

        if kind == "deref":
            ptr_expr = expr.get("operand")
            if isinstance(ptr_expr, dict) and ptr_expr.get("kind") == "var":
                ptr_name = ptr_expr["name"]
                self._check_null_deref(ptr_name, loc)
                self._check_use_after_free(ptr_name, loc)
            return None

        if kind == "index":
            base_expr = expr.get("base")
            index_expr = expr.get("index")
            idx_range = self._analyze_expr(index_expr, loc)
            if isinstance(base_expr, dict) and base_expr.get("kind") == "var":
                base_name = base_expr["name"]
                self._check_oob_access(base_name, idx_range, loc)
            return None

        if kind == "cast":
            inner = self._analyze_expr(expr.get("operand"), loc)
            target_type = expr.get("type", "")
            if isinstance(expr.get("operand"), dict) and expr["operand"].get("kind") == "var":
                ptr_name = expr["operand"]["name"]
                if target_type.endswith("*"):
                    self._check_strict_aliasing(ptr_name, target_type, loc)
            return inner

        if kind == "addr":
            return None

        if kind == "call":
            func = expr.get("func", "")
            args = expr.get("args", [])
            for arg in args:
                self._analyze_expr(arg, loc)
            if func == "free":
                if args and isinstance(args[0], dict) and args[0].get("kind") == "var":
                    ptr_name = args[0]["name"]
                    self._check_double_free(ptr_name, loc)
                    info = self._state.variables.get(ptr_name)
                    if info:
                        info.alloc_state = AllocState.FREED
            elif func in ("malloc", "calloc", "realloc"):
                pass
            elif func == "memcpy":
                if len(args) >= 2:
                    for a in args[:2]:
                        if isinstance(a, dict) and a.get("kind") == "var":
                            pass
            return None

        if kind == "sizeof":
            return None

        if kind == "ternary":
            self._analyze_expr(expr.get("cond"), loc)
            t = self._analyze_expr(expr.get("true_expr"), loc)
            f = self._analyze_expr(expr.get("false_expr"), loc)
            if t and f:
                return (min(t[0], f[0]), max(t[1], f[1]))
            return t or f

        if kind == "comma":
            exprs = expr.get("exprs", [])
            result = None
            for e in exprs:
                result = self._analyze_expr(e, loc)
            return result

        return None

    def _compute_range(self, op: str, left: Tuple[float, float],
                       right: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        l_lo, l_hi = left
        r_lo, r_hi = right

        if op == "+":
            return (l_lo + r_lo, l_hi + r_hi)
        if op == "-":
            return (l_lo - r_hi, l_hi - r_lo)
        if op == "*":
            products = [l_lo * r_lo, l_lo * r_hi, l_hi * r_lo, l_hi * r_hi]
            return (min(products), max(products))
        if op in ("/", "%"):
            if r_lo == 0 and r_hi == 0:
                return None
            return None
        if op in ("==", "!=", "<", "<=", ">", ">=", "&&", "||"):
            return (0, 1)
        if op in ("<<", ">>"):
            return None
        if op in ("&", "|", "^"):
            return None
        return None

    # --- statement analysis ---
    def _analyze_stmt(self, stmt: Dict[str, Any]) -> None:
        kind = stmt.get("kind", "")
        loc = stmt.get("location", "unknown")

        if kind == "decl":
            name = stmt.get("name", "")
            var_type = stmt.get("type", "int")
            info = VarInfo(
                name=name, type_name=var_type,
                is_pointer=var_type.endswith("*"),
                declared_type=var_type,
            )
            if "init" in stmt:
                self._enter_expression()
                val_range = self._analyze_expr(stmt["init"], loc)
                self._exit_expression()
                info.initialized = True
                info.value_range = val_range
                init_expr = stmt["init"]
                if isinstance(init_expr, dict):
                    if init_expr.get("kind") == "const" and init_expr.get("value") == 0 and info.is_pointer:
                        info.may_be_null = True
                    elif init_expr.get("kind") == "call":
                        func = init_expr.get("func", "")
                        if func in ("malloc", "calloc", "realloc"):
                            info.alloc_state = AllocState.ALLOCATED
                            info.may_be_null = True
                            size_arg = init_expr.get("args", [None])[0] if init_expr.get("args") else None
                            if size_arg is not None:
                                size_range = self._analyze_expr(size_arg, loc)
                                if size_range:
                                    info.buffer_size = int(size_range[1])
            self._state.variables[name] = info
            self._state.add_local(name)

        elif kind == "assign":
            target = stmt.get("target", "")
            self._enter_expression()
            val_range = self._analyze_expr(stmt.get("value"), loc)
            self._exit_expression()

            info = self._state.variables.get(target)
            if info:
                info.initialized = True
                info.value_range = val_range
                info.modified_count += 1
                info.last_modified_at = loc

                value_expr = stmt.get("value")
                if isinstance(value_expr, dict):
                    if value_expr.get("kind") == "const" and value_expr.get("value") == 0 and info.is_pointer:
                        info.may_be_null = True
                    elif value_expr.get("kind") == "call" and value_expr.get("func") in ("malloc", "calloc"):
                        info.alloc_state = AllocState.ALLOCATED
                        info.may_be_null = True
                    elif value_expr.get("kind") == "var":
                        src_name = value_expr["name"]
                        src = self._state.variables.get(src_name)
                        if src:
                            info.points_to = set(src.points_to)
                            info.may_be_null = src.may_be_null
                            info.alloc_state = src.alloc_state

        elif kind == "expr":
            self._enter_expression()
            self._analyze_expr(stmt.get("expr"), loc)
            self._exit_expression()

        elif kind == "call":
            func_name = stmt.get("func", "")
            args = stmt.get("args", [])
            self._enter_expression()
            for arg in args:
                self._analyze_expr(arg, loc)
            self._exit_expression()

            if func_name == "free":
                if args and isinstance(args[0], dict) and args[0].get("kind") == "var":
                    ptr_name = args[0]["name"]
                    self._check_double_free(ptr_name, loc)
                    info = self._state.variables.get(ptr_name)
                    if info:
                        info.alloc_state = AllocState.FREED

            self._function_calls[func_name] = self._function_calls.get(func_name, 0) + 1

        elif kind == "return":
            if "value" in stmt:
                self._enter_expression()
                self._analyze_expr(stmt["value"], loc)
                self._exit_expression()

        elif kind == "if":
            self._enter_expression()
            self._analyze_expr(stmt.get("cond"), loc)
            self._exit_expression()

            saved = self._state.copy()
            self._apply_condition(stmt.get("cond"), True)
            for s in stmt.get("then", []):
                self._analyze_stmt(s)
            then_state = self._state

            self._state = saved
            self._apply_condition(stmt.get("cond"), False)
            for s in stmt.get("else", []):
                self._analyze_stmt(s)
            else_state = self._state

            self._state = then_state.join(else_state)

        elif kind in ("while", "for"):
            if kind == "for" and stmt.get("init"):
                self._analyze_stmt(stmt["init"])

            for _ in range(3):
                self._enter_expression()
                self._analyze_expr(stmt.get("cond"), loc)
                self._exit_expression()
                for s in stmt.get("body", []):
                    self._analyze_stmt(s)
                if kind == "for" and stmt.get("update"):
                    self._analyze_stmt(stmt["update"])

        elif kind == "block":
            self._state.push_scope()
            for s in stmt.get("body", []):
                self._analyze_stmt(s)
            self._state.pop_scope()

        elif kind == "switch":
            self._enter_expression()
            self._analyze_expr(stmt.get("expr"), loc)
            self._exit_expression()
            for case in stmt.get("cases", []):
                for s in case.get("body", []):
                    self._analyze_stmt(s)

        elif kind == "thread_create":
            thread_id = stmt.get("thread_id", 1)
            shared_vars = stmt.get("shared_vars", [])
            for var in shared_vars:
                info = self._state.variables.get(var)
                if info:
                    info.is_shared = True

        elif kind == "thread_access":
            var = stmt.get("variable", "")
            is_write = stmt.get("is_write", False)
            thread_id = stmt.get("thread_id", 0)
            self._check_data_race(var, is_write, thread_id, loc)
            info = self._state.variables.get(var)
            if info:
                info.thread_id = thread_id
                if is_write:
                    info.modified_count += 1

        elif kind == "sequence_expr":
            self._enter_expression()
            for sub_expr in stmt.get("exprs", []):
                self._analyze_expr(sub_expr, loc)
            self._exit_expression()

        elif kind == "assert":
            self._enter_expression()
            self._analyze_expr(stmt.get("cond"), loc)
            self._exit_expression()

    def _apply_condition(self, cond: Any, truth: bool) -> None:
        if not isinstance(cond, dict):
            return
        kind = cond.get("kind", "")
        if kind == "binop":
            op = cond.get("op", "")
            left = cond.get("left")
            right = cond.get("right")
            if isinstance(left, dict) and left.get("kind") == "var":
                var_name = left["name"]
                info = self._state.variables.get(var_name)
                if info and isinstance(right, dict) and right.get("kind") == "const":
                    val = right["value"]
                    if (op == "!=" and truth) or (op == "==" and not truth):
                        if val == 0 and info.is_pointer:
                            info.may_be_null = False
                    elif (op == "==" and truth) or (op == "!=" and not truth):
                        if val == 0 and info.is_pointer:
                            info.may_be_null = True
                        info.value_range = (val, val)

    # --- main entry ---
    def detect(self, c_ast: Any) -> List[UBViolation]:
        self._reset()

        if isinstance(c_ast, dict):
            stmts = c_ast.get("body", [])
            params = c_ast.get("params", [])
        elif isinstance(c_ast, list):
            stmts = c_ast
            params = []
        else:
            return []

        self._state.push_scope()

        for param in params:
            name = param.get("name", "")
            ptype = param.get("type", "int")
            info = VarInfo(
                name=name, type_name=ptype, initialized=True,
                is_pointer=ptype.endswith("*"),
                declared_type=ptype,
            )
            if ptype.endswith("*"):
                info.may_be_null = True
            else:
                info.value_range = TYPE_RANGES.get(ptype)
            self._state.variables[name] = info
            self._state.add_local(name)

        for stmt in stmts:
            self._analyze_stmt(stmt)

        self._state.pop_scope()

        seen = set()
        deduped: List[UBViolation] = []
        for v in self._violations:
            key = (v.type, v.location, v.explanation)
            if key not in seen:
                seen.add(key)
                deduped.append(v)

        return deduped

    def detect_in_function(self, func_ast: Dict[str, Any]) -> List[UBViolation]:
        return self.detect(func_ast)

    def detect_in_program(self, program: Dict[str, Any]) -> Dict[str, List[UBViolation]]:
        results: Dict[str, List[UBViolation]] = {}
        for func in program.get("functions", []):
            name = func.get("name", "<anonymous>")
            results[name] = self.detect_in_function(func)
        return results

    def get_summary(self) -> Dict[str, Any]:
        by_type: Dict[str, int] = {}
        for v in self._violations:
            t = v.type.value
            by_type[t] = by_type.get(t, 0) + 1
        return {
            "total_violations": len(self._violations),
            "by_type": by_type,
            "violations": [v.to_dict() for v in self._violations],
        }
