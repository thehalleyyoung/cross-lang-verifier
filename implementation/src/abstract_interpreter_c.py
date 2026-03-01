"""
Abstract interpretation for C code.
Implements interval domain, pointer analysis, null pointer analysis,
buffer size tracking, integer overflow detection, use-after-free detection,
and uninitialized variable detection.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from enum import Enum, auto
import copy
import math


# ---------------------------------------------------------------------------
# Interval domain
# ---------------------------------------------------------------------------

class Interval:
    """Numeric interval [lo, hi].  lo/hi may be -inf / +inf."""

    NEG_INF = float("-inf")
    POS_INF = float("inf")

    def __init__(self, lo: float = NEG_INF, hi: float = POS_INF):
        self.lo = lo
        self.hi = hi

    # --- factories ---
    @classmethod
    def top(cls) -> "Interval":
        return cls(cls.NEG_INF, cls.POS_INF)

    @classmethod
    def bottom(cls) -> "Interval":
        return cls(cls.POS_INF, cls.NEG_INF)

    @classmethod
    def const(cls, v: float) -> "Interval":
        return cls(v, v)

    @classmethod
    def from_c_type(cls, ctype: str) -> "Interval":
        type_ranges = {
            "char": (-128, 127),
            "unsigned char": (0, 255),
            "short": (-32768, 32767),
            "unsigned short": (0, 65535),
            "int": (-2147483648, 2147483647),
            "unsigned int": (0, 4294967295),
            "long": (-9223372036854775808, 9223372036854775807),
            "unsigned long": (0, 18446744073709551615),
            "size_t": (0, 18446744073709551615),
        }
        if ctype in type_ranges:
            lo, hi = type_ranges[ctype]
            return cls(lo, hi)
        return cls.top()

    # --- predicates ---
    def is_bottom(self) -> bool:
        return self.lo > self.hi

    def is_top(self) -> bool:
        return self.lo == self.NEG_INF and self.hi == self.POS_INF

    def contains(self, v: float) -> bool:
        return self.lo <= v <= self.hi

    def __contains__(self, other: "Interval") -> bool:
        if other.is_bottom():
            return True
        return self.lo <= other.lo and other.hi <= self.hi

    # --- lattice ops ---
    def join(self, other: "Interval") -> "Interval":
        if self.is_bottom():
            return other
        if other.is_bottom():
            return self
        return Interval(min(self.lo, other.lo), max(self.hi, other.hi))

    def meet(self, other: "Interval") -> "Interval":
        return Interval(max(self.lo, other.lo), min(self.hi, other.hi))

    def widen(self, other: "Interval") -> "Interval":
        lo = self.lo if other.lo >= self.lo else self.NEG_INF
        hi = self.hi if other.hi <= self.hi else self.POS_INF
        return Interval(lo, hi)

    def narrow(self, other: "Interval") -> "Interval":
        lo = other.lo if self.lo == self.NEG_INF else self.lo
        hi = other.hi if self.hi == self.POS_INF else self.hi
        return Interval(lo, hi)

    # --- arithmetic ---
    def __add__(self, other: "Interval") -> "Interval":
        if self.is_bottom() or other.is_bottom():
            return Interval.bottom()
        return Interval(self.lo + other.lo, self.hi + other.hi)

    def __sub__(self, other: "Interval") -> "Interval":
        if self.is_bottom() or other.is_bottom():
            return Interval.bottom()
        return Interval(self.lo - other.hi, self.hi - other.lo)

    def __mul__(self, other: "Interval") -> "Interval":
        if self.is_bottom() or other.is_bottom():
            return Interval.bottom()
        products = [
            self.lo * other.lo,
            self.lo * other.hi,
            self.hi * other.lo,
            self.hi * other.hi,
        ]
        finite = [p for p in products if math.isfinite(p)]
        if not finite:
            return Interval.top()
        return Interval(min(finite), max(finite))

    def __truediv__(self, other: "Interval") -> "Interval":
        if self.is_bottom() or other.is_bottom():
            return Interval.bottom()
        if other.contains(0):
            return Interval.top()
        quotients = []
        for n in (self.lo, self.hi):
            for d in (other.lo, other.hi):
                if d != 0 and math.isfinite(n) and math.isfinite(d):
                    quotients.append(n / d)
        if not quotients:
            return Interval.top()
        return Interval(min(quotients), max(quotients))

    def __neg__(self) -> "Interval":
        return Interval(-self.hi, -self.lo)

    def __mod__(self, other: "Interval") -> "Interval":
        if self.is_bottom() or other.is_bottom():
            return Interval.bottom()
        if other.contains(0):
            return Interval.top()
        abs_max = max(abs(other.lo), abs(other.hi))
        return Interval(-(abs_max - 1), abs_max - 1)

    def __repr__(self) -> str:
        if self.is_bottom():
            return "⊥"
        return f"[{self.lo}, {self.hi}]"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Interval):
            return NotImplemented
        return self.lo == other.lo and self.hi == other.hi

    def __hash__(self) -> int:
        return hash((self.lo, self.hi))

    def shift_left(self, other: "Interval") -> "Interval":
        if self.is_bottom() or other.is_bottom():
            return Interval.bottom()
        if other.lo < 0 or other.hi >= 64:
            return Interval.top()
        lo = int(self.lo) << int(other.lo)
        hi = int(self.hi) << int(other.hi)
        return Interval(min(lo, hi), max(lo, hi))

    def shift_right(self, other: "Interval") -> "Interval":
        if self.is_bottom() or other.is_bottom():
            return Interval.bottom()
        if other.lo < 0 or other.hi >= 64:
            return Interval.top()
        lo = int(self.lo) >> int(other.hi)
        hi = int(self.hi) >> int(other.lo)
        return Interval(min(lo, hi), max(lo, hi))

    def bitwise_and(self, other: "Interval") -> "Interval":
        if self.is_bottom() or other.is_bottom():
            return Interval.bottom()
        if self.lo >= 0 and other.lo >= 0:
            return Interval(0, min(self.hi, other.hi))
        return Interval.top()


# ---------------------------------------------------------------------------
# Allocation state for heap objects
# ---------------------------------------------------------------------------

class AllocState(Enum):
    ALLOCATED = auto()
    FREED = auto()
    UNKNOWN = auto()


# ---------------------------------------------------------------------------
# Bug / warning descriptors
# ---------------------------------------------------------------------------

class BugKind(Enum):
    NULL_DEREF = "null-pointer-dereference"
    BUFFER_OVERFLOW = "buffer-overflow"
    INTEGER_OVERFLOW = "integer-overflow"
    USE_AFTER_FREE = "use-after-free"
    DOUBLE_FREE = "double-free"
    UNINIT_READ = "uninitialized-variable-read"
    MEMORY_LEAK = "memory-leak"
    DIVISION_BY_ZERO = "division-by-zero"
    SHIFT_OVERFLOW = "shift-overflow"
    DANGLING_POINTER = "dangling-pointer"


@dataclass
class BugReport:
    kind: BugKind
    location: str
    variable: str
    message: str
    severity: str = "warning"


# ---------------------------------------------------------------------------
# Pointer info
# ---------------------------------------------------------------------------

@dataclass
class PointerInfo:
    targets: Set[str] = field(default_factory=set)
    may_be_null: bool = True
    alloc_state: AllocState = AllocState.UNKNOWN
    buffer_size: Optional[Interval] = None


# ---------------------------------------------------------------------------
# AbstractState
# ---------------------------------------------------------------------------

@dataclass
class AbstractState:
    """The abstract state at a program point."""
    variable_ranges: Dict[str, Interval] = field(default_factory=dict)
    pointer_targets: Dict[str, PointerInfo] = field(default_factory=dict)
    null_pointers: Set[str] = field(default_factory=set)
    buffer_sizes: Dict[str, Interval] = field(default_factory=dict)
    initialized: Set[str] = field(default_factory=set)
    alloc_states: Dict[str, AllocState] = field(default_factory=dict)
    bugs_found: List[BugReport] = field(default_factory=list)
    type_info: Dict[str, str] = field(default_factory=dict)
    scope_stack: List[Set[str]] = field(default_factory=list)

    def copy(self) -> "AbstractState":
        return AbstractState(
            variable_ranges=dict(self.variable_ranges),
            pointer_targets={k: PointerInfo(
                targets=set(v.targets), may_be_null=v.may_be_null,
                alloc_state=v.alloc_state,
                buffer_size=copy.copy(v.buffer_size),
            ) for k, v in self.pointer_targets.items()},
            null_pointers=set(self.null_pointers),
            buffer_sizes=dict(self.buffer_sizes),
            initialized=set(self.initialized),
            alloc_states=dict(self.alloc_states),
            bugs_found=list(self.bugs_found),
            type_info=dict(self.type_info),
            scope_stack=[set(s) for s in self.scope_stack],
        )

    def join(self, other: "AbstractState") -> "AbstractState":
        result = AbstractState()
        all_vars = set(self.variable_ranges) | set(other.variable_ranges)
        for v in all_vars:
            a = self.variable_ranges.get(v, Interval.bottom())
            b = other.variable_ranges.get(v, Interval.bottom())
            result.variable_ranges[v] = a.join(b)

        all_ptrs = set(self.pointer_targets) | set(other.pointer_targets)
        for p in all_ptrs:
            a = self.pointer_targets.get(p, PointerInfo())
            b = other.pointer_targets.get(p, PointerInfo())
            result.pointer_targets[p] = PointerInfo(
                targets=a.targets | b.targets,
                may_be_null=a.may_be_null or b.may_be_null,
                alloc_state=a.alloc_state if a.alloc_state == b.alloc_state else AllocState.UNKNOWN,
                buffer_size=(a.buffer_size.join(b.buffer_size)
                             if a.buffer_size and b.buffer_size else None),
            )

        result.null_pointers = self.null_pointers | other.null_pointers
        all_bufs = set(self.buffer_sizes) | set(other.buffer_sizes)
        for b in all_bufs:
            sa = self.buffer_sizes.get(b, Interval.bottom())
            sb = other.buffer_sizes.get(b, Interval.bottom())
            result.buffer_sizes[b] = sa.join(sb)

        result.initialized = self.initialized & other.initialized
        result.alloc_states = {}
        for k in set(self.alloc_states) | set(other.alloc_states):
            a = self.alloc_states.get(k, AllocState.UNKNOWN)
            b = other.alloc_states.get(k, AllocState.UNKNOWN)
            result.alloc_states[k] = a if a == b else AllocState.UNKNOWN

        result.bugs_found = list(self.bugs_found) + list(other.bugs_found)
        result.type_info = {**self.type_info, **other.type_info}
        return result

    def widen(self, other: "AbstractState") -> "AbstractState":
        result = self.copy()
        for v in other.variable_ranges:
            if v in result.variable_ranges:
                result.variable_ranges[v] = result.variable_ranges[v].widen(
                    other.variable_ranges[v])
            else:
                result.variable_ranges[v] = other.variable_ranges[v]
        return result

    def is_subset(self, other: "AbstractState") -> bool:
        for v, iv in self.variable_ranges.items():
            ov = other.variable_ranges.get(v, Interval.bottom())
            if iv not in ov:
                return False
        return True

    def push_scope(self) -> None:
        self.scope_stack.append(set())

    def pop_scope(self) -> None:
        if self.scope_stack:
            leaving = self.scope_stack.pop()
            for var in leaving:
                self.variable_ranges.pop(var, None)
                self.pointer_targets.pop(var, None)
                self.initialized.discard(var)

    def add_local(self, name: str) -> None:
        if self.scope_stack:
            self.scope_stack[-1].add(name)


# ---------------------------------------------------------------------------
# CFG representation
# ---------------------------------------------------------------------------

@dataclass
class CFGNode:
    node_id: int
    label: str = ""
    statements: List[Dict[str, Any]] = field(default_factory=list)
    successors: List[int] = field(default_factory=list)
    predecessors: List[int] = field(default_factory=list)
    is_entry: bool = False
    is_exit: bool = False


@dataclass
class CFG:
    nodes: Dict[int, CFGNode] = field(default_factory=dict)
    entry: int = 0
    exit_nodes: List[int] = field(default_factory=list)

    def add_node(self, node: CFGNode) -> None:
        self.nodes[node.node_id] = node

    def add_edge(self, src: int, dst: int) -> None:
        if src in self.nodes and dst in self.nodes:
            if dst not in self.nodes[src].successors:
                self.nodes[src].successors.append(dst)
            if src not in self.nodes[dst].predecessors:
                self.nodes[dst].predecessors.append(src)


# ---------------------------------------------------------------------------
# C Abstract Interpreter
# ---------------------------------------------------------------------------

class CAbstractInterpreter:
    """Abstract interpreter for C programs using interval + pointer domain."""

    INT_MIN = -2147483648
    INT_MAX = 2147483647
    UINT_MAX = 4294967295
    MAX_ITERATIONS = 100

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.warnings: List[BugReport] = []
        self.function_summaries: Dict[str, Dict[str, Any]] = {}
        self._node_counter = 0
        self._builtin_models: Dict[str, Any] = self._init_builtin_models()

    def _init_builtin_models(self) -> Dict[str, Any]:
        return {
            "malloc": self._model_malloc,
            "calloc": self._model_calloc,
            "realloc": self._model_realloc,
            "free": self._model_free,
            "strlen": self._model_strlen,
            "memcpy": self._model_memcpy,
            "memset": self._model_memset,
            "printf": self._model_printf,
            "sizeof": self._model_sizeof,
            "abs": self._model_abs,
        }

    # --- builtin models ---
    def _model_malloc(self, state: AbstractState, args: List[Any],
                      loc: str) -> Tuple[AbstractState, Optional[Interval]]:
        state = state.copy()
        alloc_name = f"heap_{self._node_counter}"
        self._node_counter += 1
        if args:
            size_iv = self._eval_expr(state, args[0], loc)
            state.buffer_sizes[alloc_name] = size_iv
        else:
            state.buffer_sizes[alloc_name] = Interval.top()
        state.alloc_states[alloc_name] = AllocState.ALLOCATED
        ptr_info = PointerInfo(
            targets={alloc_name}, may_be_null=True,
            alloc_state=AllocState.ALLOCATED,
            buffer_size=state.buffer_sizes[alloc_name],
        )
        state.pointer_targets[alloc_name] = ptr_info
        return state, Interval.const(0)

    def _model_calloc(self, state: AbstractState, args: List[Any],
                      loc: str) -> Tuple[AbstractState, Optional[Interval]]:
        state = state.copy()
        alloc_name = f"heap_{self._node_counter}"
        self._node_counter += 1
        if len(args) >= 2:
            count_iv = self._eval_expr(state, args[0], loc)
            size_iv = self._eval_expr(state, args[1], loc)
            total = count_iv * size_iv
            state.buffer_sizes[alloc_name] = total
        else:
            state.buffer_sizes[alloc_name] = Interval.top()
        state.alloc_states[alloc_name] = AllocState.ALLOCATED
        return state, Interval.const(0)

    def _model_realloc(self, state: AbstractState, args: List[Any],
                       loc: str) -> Tuple[AbstractState, Optional[Interval]]:
        state = state.copy()
        alloc_name = f"heap_{self._node_counter}"
        self._node_counter += 1
        if len(args) >= 2:
            size_iv = self._eval_expr(state, args[1], loc)
            state.buffer_sizes[alloc_name] = size_iv
        state.alloc_states[alloc_name] = AllocState.ALLOCATED
        return state, Interval.const(0)

    def _model_free(self, state: AbstractState, args: List[Any],
                    loc: str) -> Tuple[AbstractState, Optional[Interval]]:
        state = state.copy()
        if args and isinstance(args[0], dict) and args[0].get("kind") == "var":
            var_name = args[0]["name"]
            ptr_info = state.pointer_targets.get(var_name)
            if ptr_info:
                for target in ptr_info.targets:
                    prev = state.alloc_states.get(target, AllocState.UNKNOWN)
                    if prev == AllocState.FREED:
                        state.bugs_found.append(BugReport(
                            kind=BugKind.DOUBLE_FREE, location=loc,
                            variable=var_name,
                            message=f"Double free of {var_name} (target {target})",
                            severity="error",
                        ))
                    state.alloc_states[target] = AllocState.FREED
                ptr_info.alloc_state = AllocState.FREED
            elif var_name in state.alloc_states:
                if state.alloc_states[var_name] == AllocState.FREED:
                    state.bugs_found.append(BugReport(
                        kind=BugKind.DOUBLE_FREE, location=loc,
                        variable=var_name,
                        message=f"Double free of {var_name}",
                        severity="error",
                    ))
                state.alloc_states[var_name] = AllocState.FREED
        return state, None

    def _model_strlen(self, state: AbstractState, args: List[Any],
                      loc: str) -> Tuple[AbstractState, Optional[Interval]]:
        return state, Interval(0, self.INT_MAX)

    def _model_memcpy(self, state: AbstractState, args: List[Any],
                      loc: str) -> Tuple[AbstractState, Optional[Interval]]:
        if len(args) >= 3:
            size_iv = self._eval_expr(state, args[2], loc)
            if isinstance(args[0], dict) and args[0].get("kind") == "var":
                dst = args[0]["name"]
                buf = state.buffer_sizes.get(dst)
                if buf and not size_iv.is_bottom() and not buf.is_bottom():
                    if size_iv.hi > buf.hi:
                        state.bugs_found.append(BugReport(
                            kind=BugKind.BUFFER_OVERFLOW, location=loc,
                            variable=dst,
                            message=f"memcpy may overflow {dst}: copying up to {size_iv.hi} bytes into buffer of size {buf.hi}",
                            severity="error",
                        ))
        return state, None

    def _model_memset(self, state: AbstractState, args: List[Any],
                      loc: str) -> Tuple[AbstractState, Optional[Interval]]:
        return state, None

    def _model_printf(self, state: AbstractState, args: List[Any],
                      loc: str) -> Tuple[AbstractState, Optional[Interval]]:
        return state, Interval(0, self.INT_MAX)

    def _model_sizeof(self, state: AbstractState, args: List[Any],
                      loc: str) -> Tuple[AbstractState, Optional[Interval]]:
        type_sizes = {"char": 1, "short": 2, "int": 4, "long": 8,
                      "float": 4, "double": 8, "void*": 8}
        if args and isinstance(args[0], str) and args[0] in type_sizes:
            return state, Interval.const(type_sizes[args[0]])
        return state, Interval(1, 1024)

    def _model_abs(self, state: AbstractState, args: List[Any],
                   loc: str) -> Tuple[AbstractState, Optional[Interval]]:
        if args:
            iv = self._eval_expr(state, args[0], loc)
            if iv.lo == self.INT_MIN:
                state.bugs_found.append(BugReport(
                    kind=BugKind.INTEGER_OVERFLOW, location=loc,
                    variable="abs_arg",
                    message="abs(INT_MIN) is undefined behavior",
                    severity="error",
                ))
            lo = 0
            hi = max(abs(iv.lo) if math.isfinite(iv.lo) else float("inf"),
                     abs(iv.hi) if math.isfinite(iv.hi) else float("inf"))
            return state, Interval(lo, hi)
        return state, Interval.top()

    # --- expression evaluator ---
    def _eval_expr(self, state: AbstractState, expr: Any,
                   loc: str) -> Interval:
        if expr is None:
            return Interval.top()
        if isinstance(expr, (int, float)):
            return Interval.const(expr)
        if isinstance(expr, str):
            return state.variable_ranges.get(expr, Interval.top())
        if not isinstance(expr, dict):
            return Interval.top()

        kind = expr.get("kind", "")

        if kind == "const":
            return Interval.const(expr.get("value", 0))

        if kind == "var":
            name = expr.get("name", "")
            if name not in state.initialized:
                state.bugs_found.append(BugReport(
                    kind=BugKind.UNINIT_READ, location=loc,
                    variable=name,
                    message=f"Read of potentially uninitialized variable '{name}'",
                    severity="warning",
                ))
            return state.variable_ranges.get(name, Interval.top())

        if kind == "binop":
            op = expr.get("op", "+")
            lhs = self._eval_expr(state, expr.get("left"), loc)
            rhs = self._eval_expr(state, expr.get("right"), loc)
            return self._eval_binop(state, op, lhs, rhs, loc)

        if kind == "unop":
            op = expr.get("op", "-")
            operand = self._eval_expr(state, expr.get("operand"), loc)
            return self._eval_unop(state, op, operand, loc)

        if kind == "call":
            func_name = expr.get("func", "")
            args = expr.get("args", [])
            if func_name in self._builtin_models:
                new_state, result = self._builtin_models[func_name](
                    state, args, loc)
                state.variable_ranges.update(new_state.variable_ranges)
                state.pointer_targets.update(new_state.pointer_targets)
                state.alloc_states.update(new_state.alloc_states)
                state.buffer_sizes.update(new_state.buffer_sizes)
                state.bugs_found.extend(new_state.bugs_found)
                return result if result is not None else Interval.top()
            return Interval.top()

        if kind == "deref":
            ptr_expr = expr.get("operand")
            if isinstance(ptr_expr, dict) and ptr_expr.get("kind") == "var":
                ptr_name = ptr_expr["name"]
                self._check_deref_safety(state, ptr_name, loc)
            return Interval.top()

        if kind == "addr":
            return Interval.top()

        if kind == "index":
            base = expr.get("base")
            index_expr = expr.get("index")
            index_iv = self._eval_expr(state, index_expr, loc)
            if isinstance(base, dict) and base.get("kind") == "var":
                base_name = base["name"]
                buf_size = state.buffer_sizes.get(base_name)
                if buf_size and not index_iv.is_bottom():
                    if index_iv.lo < 0:
                        state.bugs_found.append(BugReport(
                            kind=BugKind.BUFFER_OVERFLOW, location=loc,
                            variable=base_name,
                            message=f"Negative index into {base_name}: index may be {index_iv.lo}",
                            severity="error",
                        ))
                    if not buf_size.is_bottom() and index_iv.hi >= buf_size.hi:
                        state.bugs_found.append(BugReport(
                            kind=BugKind.BUFFER_OVERFLOW, location=loc,
                            variable=base_name,
                            message=f"Index {index_iv.hi} may exceed buffer size {buf_size.hi} for {base_name}",
                            severity="error",
                        ))
            return Interval.top()

        if kind == "cast":
            target_type = expr.get("type", "int")
            inner = self._eval_expr(state, expr.get("operand"), loc)
            type_range = Interval.from_c_type(target_type)
            return inner.meet(type_range) if not type_range.is_top() else inner

        if kind == "sizeof":
            return self._model_sizeof(state, [expr.get("type")], loc)[1] or Interval.top()

        if kind == "ternary":
            true_val = self._eval_expr(state, expr.get("true_expr"), loc)
            false_val = self._eval_expr(state, expr.get("false_expr"), loc)
            return true_val.join(false_val)

        return Interval.top()

    def _eval_binop(self, state: AbstractState, op: str,
                    lhs: Interval, rhs: Interval, loc: str) -> Interval:
        if op == "+":
            result = lhs + rhs
            self._check_overflow(state, result, loc, f"{lhs} + {rhs}")
            return result
        if op == "-":
            result = lhs - rhs
            self._check_overflow(state, result, loc, f"{lhs} - {rhs}")
            return result
        if op == "*":
            result = lhs * rhs
            self._check_overflow(state, result, loc, f"{lhs} * {rhs}")
            return result
        if op == "/":
            if rhs.contains(0):
                state.bugs_found.append(BugReport(
                    kind=BugKind.DIVISION_BY_ZERO, location=loc,
                    variable="", message="Possible division by zero",
                    severity="error",
                ))
            return lhs / rhs
        if op == "%":
            if rhs.contains(0):
                state.bugs_found.append(BugReport(
                    kind=BugKind.DIVISION_BY_ZERO, location=loc,
                    variable="", message="Possible modulo by zero",
                    severity="error",
                ))
            return lhs % rhs
        if op == "<<":
            if rhs.hi >= 32 or rhs.lo < 0:
                state.bugs_found.append(BugReport(
                    kind=BugKind.SHIFT_OVERFLOW, location=loc,
                    variable="",
                    message=f"Shift amount {rhs} may be out of range",
                    severity="warning",
                ))
            return lhs.shift_left(rhs)
        if op == ">>":
            return lhs.shift_right(rhs)
        if op == "&":
            return lhs.bitwise_and(rhs)
        if op in ("==", "!=", "<", "<=", ">", ">="):
            return Interval(0, 1)
        if op in ("&&", "||"):
            return Interval(0, 1)
        return Interval.top()

    def _eval_unop(self, state: AbstractState, op: str,
                   operand: Interval, loc: str) -> Interval:
        if op == "-":
            return -operand
        if op == "!":
            return Interval(0, 1)
        if op == "~":
            return Interval.top()
        if op == "++":
            result = operand + Interval.const(1)
            self._check_overflow(state, result, loc, f"++({operand})")
            return result
        if op == "--":
            result = operand - Interval.const(1)
            self._check_overflow(state, result, loc, f"--({operand})")
            return result
        return Interval.top()

    def _check_overflow(self, state: AbstractState, iv: Interval,
                        loc: str, expr_desc: str) -> None:
        if iv.is_bottom() or iv.is_top():
            return
        if iv.hi > self.INT_MAX or iv.lo < self.INT_MIN:
            state.bugs_found.append(BugReport(
                kind=BugKind.INTEGER_OVERFLOW, location=loc,
                variable="",
                message=f"Possible integer overflow in {expr_desc}: result range {iv}",
                severity="warning",
            ))

    def _check_deref_safety(self, state: AbstractState, ptr_name: str,
                            loc: str) -> None:
        ptr_info = state.pointer_targets.get(ptr_name)
        if ptr_info:
            if ptr_info.may_be_null or ptr_name in state.null_pointers:
                state.bugs_found.append(BugReport(
                    kind=BugKind.NULL_DEREF, location=loc,
                    variable=ptr_name,
                    message=f"Pointer '{ptr_name}' may be null when dereferenced",
                    severity="error",
                ))
            if ptr_info.alloc_state == AllocState.FREED:
                state.bugs_found.append(BugReport(
                    kind=BugKind.USE_AFTER_FREE, location=loc,
                    variable=ptr_name,
                    message=f"Pointer '{ptr_name}' used after free",
                    severity="error",
                ))
            for target in ptr_info.targets:
                if state.alloc_states.get(target) == AllocState.FREED:
                    state.bugs_found.append(BugReport(
                        kind=BugKind.USE_AFTER_FREE, location=loc,
                        variable=ptr_name,
                        message=f"Pointer '{ptr_name}' target '{target}' has been freed",
                        severity="error",
                    ))
        else:
            if ptr_name in state.null_pointers:
                state.bugs_found.append(BugReport(
                    kind=BugKind.NULL_DEREF, location=loc,
                    variable=ptr_name,
                    message=f"Null pointer '{ptr_name}' dereferenced",
                    severity="error",
                ))

    # --- statement interpreter ---
    def _interpret_stmt(self, state: AbstractState, stmt: Dict[str, Any],
                        loc: str) -> AbstractState:
        state = state.copy()
        kind = stmt.get("kind", "")

        if kind == "decl":
            var_name = stmt.get("name", "")
            var_type = stmt.get("type", "int")
            state.type_info[var_name] = var_type
            state.add_local(var_name)
            if "init" in stmt:
                val = self._eval_expr(state, stmt["init"], loc)
                state.variable_ranges[var_name] = val
                state.initialized.add(var_name)
                if var_type.endswith("*"):
                    init_expr = stmt["init"]
                    if isinstance(init_expr, dict) and init_expr.get("kind") == "const" and init_expr.get("value") == 0:
                        state.null_pointers.add(var_name)
                        state.pointer_targets[var_name] = PointerInfo(
                            may_be_null=True, alloc_state=AllocState.UNKNOWN)
            else:
                state.variable_ranges[var_name] = Interval.from_c_type(var_type)
                if var_type.endswith("*"):
                    state.pointer_targets[var_name] = PointerInfo(may_be_null=True)

        elif kind == "assign":
            var_name = stmt.get("target", "")
            val = self._eval_expr(state, stmt.get("value"), loc)
            state.variable_ranges[var_name] = val
            state.initialized.add(var_name)
            value_expr = stmt.get("value")
            if isinstance(value_expr, dict):
                if value_expr.get("kind") == "call" and value_expr.get("func") in ("malloc", "calloc", "realloc"):
                    alloc_name = f"heap_{self._node_counter - 1}"
                    state.pointer_targets[var_name] = PointerInfo(
                        targets={alloc_name}, may_be_null=True,
                        alloc_state=AllocState.ALLOCATED,
                        buffer_size=state.buffer_sizes.get(alloc_name),
                    )
                    if alloc_name in state.buffer_sizes:
                        state.buffer_sizes[var_name] = state.buffer_sizes[alloc_name]
                elif value_expr.get("kind") == "const" and value_expr.get("value") == 0:
                    if var_name in state.pointer_targets or (var_name in state.type_info and state.type_info[var_name].endswith("*")):
                        state.null_pointers.add(var_name)
                elif value_expr.get("kind") == "addr":
                    target = value_expr.get("operand", {}).get("name", "")
                    if target:
                        state.pointer_targets[var_name] = PointerInfo(
                            targets={target}, may_be_null=False,
                            alloc_state=AllocState.ALLOCATED)
                        state.null_pointers.discard(var_name)
                elif value_expr.get("kind") == "var":
                    src = value_expr["name"]
                    if src in state.pointer_targets:
                        state.pointer_targets[var_name] = PointerInfo(
                            targets=set(state.pointer_targets[src].targets),
                            may_be_null=state.pointer_targets[src].may_be_null,
                            alloc_state=state.pointer_targets[src].alloc_state,
                            buffer_size=state.pointer_targets[src].buffer_size)

        elif kind == "call":
            func_name = stmt.get("func", "")
            args = stmt.get("args", [])
            if func_name in self._builtin_models:
                new_state, _ = self._builtin_models[func_name](state, args, loc)
                state = new_state
            result_var = stmt.get("result")
            if result_var:
                state.variable_ranges[result_var] = Interval.top()
                state.initialized.add(result_var)

        elif kind == "return":
            if "value" in stmt:
                self._eval_expr(state, stmt["value"], loc)

        elif kind == "if":
            cond_val = self._eval_expr(state, stmt.get("cond"), loc)
            true_state = state.copy()
            false_state = state.copy()
            self._apply_condition(true_state, stmt.get("cond"), True)
            self._apply_condition(false_state, stmt.get("cond"), False)
            for s in stmt.get("then", []):
                true_state = self._interpret_stmt(true_state, s, loc)
            for s in stmt.get("else", []):
                false_state = self._interpret_stmt(false_state, s, loc)
            state = true_state.join(false_state)

        elif kind == "while" or kind == "for":
            state = self._interpret_loop(state, stmt, loc)

        elif kind == "block":
            state.push_scope()
            for s in stmt.get("body", []):
                state = self._interpret_stmt(state, s, loc)
            state.pop_scope()

        elif kind == "expr":
            self._eval_expr(state, stmt.get("expr"), loc)

        elif kind == "assert":
            cond = stmt.get("cond")
            self._eval_expr(state, cond, loc)

        return state

    def _apply_condition(self, state: AbstractState, cond: Any,
                         truth: bool) -> None:
        if not isinstance(cond, dict):
            return
        kind = cond.get("kind", "")
        if kind == "binop":
            op = cond.get("op", "")
            left = cond.get("left")
            right = cond.get("right")
            if isinstance(left, dict) and left.get("kind") == "var" and isinstance(right, dict) and right.get("kind") == "const":
                var_name = left["name"]
                val = right["value"]
                cur = state.variable_ranges.get(var_name, Interval.top())
                if (op == "==" and truth) or (op == "!=" and not truth):
                    state.variable_ranges[var_name] = cur.meet(Interval.const(val))
                    if val == 0 and var_name in state.pointer_targets:
                        state.null_pointers.add(var_name)
                elif (op == "!=" and truth) or (op == "==" and not truth):
                    if val == 0 and var_name in state.pointer_targets:
                        state.null_pointers.discard(var_name)
                        state.pointer_targets[var_name].may_be_null = False
                elif (op == "<" and truth) or (op == ">=" and not truth):
                    state.variable_ranges[var_name] = cur.meet(Interval(cur.lo, val - 1))
                elif (op == "<=" and truth) or (op == ">" and not truth):
                    state.variable_ranges[var_name] = cur.meet(Interval(cur.lo, val))
                elif (op == ">" and truth) or (op == "<=" and not truth):
                    state.variable_ranges[var_name] = cur.meet(Interval(val + 1, cur.hi))
                elif (op == ">=" and truth) or (op == "<" and not truth):
                    state.variable_ranges[var_name] = cur.meet(Interval(val, cur.hi))

    def _interpret_loop(self, state: AbstractState, stmt: Dict[str, Any],
                        loc: str) -> AbstractState:
        if stmt.get("kind") == "for":
            init = stmt.get("init")
            if init:
                state = self._interpret_stmt(state, init, loc)

        body = stmt.get("body", [])
        cond = stmt.get("cond")

        prev_state = state.copy()
        for iteration in range(self.MAX_ITERATIONS):
            loop_state = prev_state.copy()
            if cond:
                self._apply_condition(loop_state, cond, True)
            for s in body:
                loop_state = self._interpret_stmt(loop_state, s, loc)
            if stmt.get("kind") == "for" and stmt.get("update"):
                loop_state = self._interpret_stmt(loop_state, stmt["update"], loc)

            merged = prev_state.join(loop_state)
            if iteration > 3:
                merged = prev_state.widen(merged)
            if merged.is_subset(prev_state):
                break
            prev_state = merged

        exit_state = prev_state.copy()
        if cond:
            self._apply_condition(exit_state, cond, False)
        return exit_state

    # --- CFG-based analysis ---
    def _build_cfg(self, stmts: List[Dict[str, Any]]) -> CFG:
        cfg = CFG()
        entry = CFGNode(node_id=0, label="entry", is_entry=True)
        cfg.add_node(entry)
        cfg.entry = 0
        current_id = 0
        self._node_counter = 1

        for stmt in stmts:
            new_id = self._node_counter
            self._node_counter += 1
            node = CFGNode(node_id=new_id, statements=[stmt])
            cfg.add_node(node)
            cfg.add_edge(current_id, new_id)
            current_id = new_id

        exit_id = self._node_counter
        self._node_counter += 1
        exit_node = CFGNode(node_id=exit_id, label="exit", is_exit=True)
        cfg.add_node(exit_node)
        cfg.add_edge(current_id, exit_id)
        cfg.exit_nodes = [exit_id]
        return cfg

    def _analyze_cfg(self, cfg: CFG, init_state: AbstractState) -> Dict[int, AbstractState]:
        states: Dict[int, AbstractState] = {}
        states[cfg.entry] = init_state.copy()
        worklist = [cfg.entry]
        iterations = 0

        while worklist and iterations < self.MAX_ITERATIONS * len(cfg.nodes):
            iterations += 1
            node_id = worklist.pop(0)
            node = cfg.nodes[node_id]
            current_state = states.get(node_id, AbstractState())

            for stmt in node.statements:
                current_state = self._interpret_stmt(
                    current_state, stmt, f"node_{node_id}")

            for succ_id in node.successors:
                old_state = states.get(succ_id)
                if old_state is None:
                    states[succ_id] = current_state.copy()
                    worklist.append(succ_id)
                else:
                    merged = old_state.join(current_state)
                    if iterations > len(cfg.nodes) * 3:
                        merged = old_state.widen(merged)
                    if not merged.is_subset(old_state):
                        states[succ_id] = merged
                        if succ_id not in worklist:
                            worklist.append(succ_id)

        return states

    def _check_memory_leaks(self, state: AbstractState) -> None:
        for name, alloc_state in state.alloc_states.items():
            if alloc_state == AllocState.ALLOCATED:
                reachable = False
                for ptr_info in state.pointer_targets.values():
                    if name in ptr_info.targets and ptr_info.alloc_state != AllocState.FREED:
                        reachable = True
                        break
                if not reachable:
                    state.bugs_found.append(BugReport(
                        kind=BugKind.MEMORY_LEAK, location="end-of-scope",
                        variable=name,
                        message=f"Heap allocation '{name}' may leak: allocated but not freed and not reachable",
                        severity="warning",
                    ))

    # --- main entry point ---
    def analyze(self, c_ast: Any) -> AbstractState:
        if isinstance(c_ast, dict):
            stmts = c_ast.get("body", [])
            params = c_ast.get("params", [])
        elif isinstance(c_ast, list):
            stmts = c_ast
            params = []
        else:
            return AbstractState()

        init_state = AbstractState()
        init_state.push_scope()

        for param in params:
            name = param.get("name", "")
            ptype = param.get("type", "int")
            init_state.type_info[name] = ptype
            init_state.initialized.add(name)
            init_state.add_local(name)
            if ptype.endswith("*"):
                init_state.pointer_targets[name] = PointerInfo(may_be_null=True)
                init_state.variable_ranges[name] = Interval.top()
            else:
                init_state.variable_ranges[name] = Interval.from_c_type(ptype)

        cfg = self._build_cfg(stmts)
        node_states = self._analyze_cfg(cfg, init_state)

        final_state = AbstractState()
        for exit_id in cfg.exit_nodes:
            if exit_id in node_states:
                final_state = final_state.join(node_states[exit_id])

        all_bugs: List[BugReport] = []
        for ns in node_states.values():
            all_bugs.extend(ns.bugs_found)
        seen = set()
        deduped: List[BugReport] = []
        for b in all_bugs:
            key = (b.kind, b.location, b.variable, b.message)
            if key not in seen:
                seen.add(key)
                deduped.append(b)
        final_state.bugs_found = deduped

        self._check_memory_leaks(final_state)
        self.warnings = list(final_state.bugs_found)
        return final_state

    def analyze_function(self, func_ast: Dict[str, Any]) -> AbstractState:
        name = func_ast.get("name", "<anonymous>")
        result = self.analyze(func_ast)
        self.function_summaries[name] = {
            "ranges": dict(result.variable_ranges),
            "bugs": len(result.bugs_found),
            "bug_kinds": [b.kind.value for b in result.bugs_found],
        }
        return result

    def analyze_program(self, program: Dict[str, Any]) -> Dict[str, AbstractState]:
        results: Dict[str, AbstractState] = {}
        functions = program.get("functions", [])
        for func in functions:
            name = func.get("name", "<anonymous>")
            results[name] = self.analyze_function(func)
        return results

    def get_summary(self) -> Dict[str, Any]:
        total_bugs = sum(s.get("bugs", 0) for s in self.function_summaries.values())
        bug_kinds: Dict[str, int] = {}
        for s in self.function_summaries.values():
            for k in s.get("bug_kinds", []):
                bug_kinds[k] = bug_kinds.get(k, 0) + 1
        return {
            "functions_analyzed": len(self.function_summaries),
            "total_bugs": total_bugs,
            "bug_kinds": bug_kinds,
            "function_summaries": self.function_summaries,
        }
