"""
Symbolic execution engine for verifying C and Rust code equivalence.

Performs symbolic execution on function ASTs from either C or Rust parsers,
collecting path constraints, detecting bugs, and producing execution trees
that can be compared across languages.
"""

from __future__ import annotations

import copy
import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union


# ---------------------------------------------------------------------------
# Symbolic value hierarchy
# ---------------------------------------------------------------------------

class SymValue:
    """Base class for all symbolic values."""

    _counter: int = 0

    def __init__(self, name: Optional[str] = None):
        if name is None:
            SymValue._counter += 1
            name = f"sym_{SymValue._counter}"
        self.name = name

    def eval(self, env: Dict[str, Any] | None = None) -> Any:
        """Evaluate the symbolic value under a concrete environment.

        If *env* is ``None`` or does not contain a binding for this value,
        return ``self`` (remain symbolic).
        """
        if env is not None and self.name in env:
            return env[self.name]
        return self

    def is_concrete(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f"SymValue({self.name})"


class SymInt(SymValue):
    """Symbolic integer with optional concrete value, bit width, and signedness."""

    def __init__(
        self,
        value: Optional[int] = None,
        bits: int = 32,
        signed: bool = True,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.bits = bits
        self.signed = signed
        self._value = value
        if value is not None:
            self._value = self._wrap(value)

    def _mask(self) -> int:
        return (1 << self.bits) - 1

    def _wrap(self, v: int) -> int:
        """Wrap *v* into the range defined by bit-width and signedness."""
        v = v & self._mask()
        if self.signed and v >= (1 << (self.bits - 1)):
            v -= 1 << self.bits
        return v

    def is_concrete(self) -> bool:
        return self._value is not None

    def concrete(self) -> int:
        if self._value is None:
            raise ValueError(f"SymInt {self.name} has no concrete value")
        return self._value

    def eval(self, env: Dict[str, Any] | None = None) -> Any:
        if self._value is not None:
            return self._value
        if env is not None and self.name in env:
            return self._wrap(env[self.name])
        return self

    def __repr__(self) -> str:
        if self._value is not None:
            return f"SymInt({self._value}, {self.bits}b, {'s' if self.signed else 'u'})"
        return f"SymInt({self.name}, {self.bits}b, {'s' if self.signed else 'u'})"


class SymBool(SymValue):
    """Symbolic boolean value."""

    def __init__(self, value: Optional[bool] = None, name: Optional[str] = None):
        super().__init__(name)
        self._value = value

    def is_concrete(self) -> bool:
        return self._value is not None

    def concrete(self) -> bool:
        if self._value is None:
            raise ValueError(f"SymBool {self.name} has no concrete value")
        return self._value

    def eval(self, env: Dict[str, Any] | None = None) -> Any:
        if self._value is not None:
            return self._value
        if env is not None and self.name in env:
            return bool(env[self.name])
        return self

    def negate(self) -> SymBool:
        if self._value is not None:
            return SymBool(value=not self._value)
        return SymUnaryOp("not", self)

    def __repr__(self) -> str:
        if self._value is not None:
            return f"SymBool({self._value})"
        return f"SymBool({self.name})"


class SymPointer(SymValue):
    """Symbolic pointer with a base address and an offset (both possibly symbolic)."""

    def __init__(
        self,
        base: Union[int, SymValue],
        offset: Union[int, SymValue, None] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.base = base
        self.offset = offset if offset is not None else SymInt(value=0)

    def is_concrete(self) -> bool:
        base_conc = isinstance(self.base, int) or (
            isinstance(self.base, SymValue) and self.base.is_concrete()
        )
        off_conc = isinstance(self.offset, int) or (
            isinstance(self.offset, SymValue) and self.offset.is_concrete()
        )
        return base_conc and off_conc

    def eval(self, env: Dict[str, Any] | None = None) -> Any:
        b = self.base if isinstance(self.base, int) else self.base.eval(env)
        o = self.offset if isinstance(self.offset, int) else self.offset.eval(env)
        if isinstance(b, int) and isinstance(o, int):
            return b + o
        return self

    def is_null(self) -> bool:
        if isinstance(self.base, int) and self.base == 0:
            if isinstance(self.offset, int) and self.offset == 0:
                return True
            if isinstance(self.offset, SymInt) and self.offset.is_concrete() and self.offset.concrete() == 0:
                return True
        if isinstance(self.base, SymInt) and self.base.is_concrete() and self.base.concrete() == 0:
            if isinstance(self.offset, SymInt) and self.offset.is_concrete() and self.offset.concrete() == 0:
                return True
        return False

    def __repr__(self) -> str:
        return f"SymPointer(base={self.base}, off={self.offset})"


class SymArray(SymValue):
    """Symbolic array that maps symbolic indices to symbolic values."""

    def __init__(
        self,
        size: Union[int, SymInt, None] = None,
        element_type: str = "int",
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.size = size
        self.element_type = element_type
        self._store: Dict[int, SymValue] = {}
        self._default: Optional[SymValue] = None

    def read(self, index: Union[int, SymInt]) -> SymValue:
        if isinstance(index, SymInt) and index.is_concrete():
            index = index.concrete()
        if isinstance(index, int):
            if index in self._store:
                return self._store[index]
            if self._default is not None:
                return self._default
            return SymInt(name=f"{self.name}[{index}]")
        return SymInt(name=f"{self.name}[{index.name}]")

    def write(self, index: Union[int, SymInt], value: SymValue) -> SymArray:
        new_arr = SymArray(size=self.size, element_type=self.element_type, name=self.name)
        new_arr._store = dict(self._store)
        new_arr._default = self._default
        if isinstance(index, SymInt) and index.is_concrete():
            index = index.concrete()
        if isinstance(index, int):
            new_arr._store[index] = value
        else:
            new_arr._store[hash(index.name)] = value
        return new_arr

    def is_concrete(self) -> bool:
        return all(v.is_concrete() for v in self._store.values())

    def eval(self, env: Dict[str, Any] | None = None) -> Any:
        return self

    def __repr__(self) -> str:
        return f"SymArray({self.name}, size={self.size})"


class SymStruct(SymValue):
    """Symbolic struct mapping field names to symbolic values."""

    def __init__(self, type_name: str = "", fields: Optional[Dict[str, SymValue]] = None, name: Optional[str] = None):
        super().__init__(name)
        self.type_name = type_name
        self.fields: Dict[str, SymValue] = fields if fields is not None else {}

    def get_field(self, field_name: str) -> SymValue:
        if field_name in self.fields:
            return self.fields[field_name]
        fresh = SymInt(name=f"{self.name}.{field_name}")
        self.fields[field_name] = fresh
        return fresh

    def set_field(self, field_name: str, value: SymValue) -> SymStruct:
        new_struct = SymStruct(type_name=self.type_name, name=self.name)
        new_struct.fields = dict(self.fields)
        new_struct.fields[field_name] = value
        return new_struct

    def is_concrete(self) -> bool:
        return all(v.is_concrete() for v in self.fields.values())

    def eval(self, env: Dict[str, Any] | None = None) -> Any:
        return self

    def __repr__(self) -> str:
        return f"SymStruct({self.type_name}, fields={list(self.fields.keys())})"


# -- Compound symbolic expressions ------------------------------------------

class SymBinOp(SymValue):
    """Symbolic binary operation (lazy / deferred evaluation)."""

    ARITH_OPS = {"+", "-", "*", "/", "%", "<<", ">>", "&", "|", "^"}
    CMP_OPS = {"==", "!=", "<", "<=", ">", ">="}
    LOGIC_OPS = {"&&", "||"}

    def __init__(self, op: str, left: SymValue, right: SymValue, name: Optional[str] = None):
        super().__init__(name)
        self.op = op
        self.left = left
        self.right = right

    def _eval_arith(self, l: int, r: int, bits: int = 32, signed: bool = True) -> int:
        ops = {
            "+": lambda a, b: a + b,
            "-": lambda a, b: a - b,
            "*": lambda a, b: a * b,
            "/": lambda a, b: a // b if b != 0 else 0,
            "%": lambda a, b: a % b if b != 0 else 0,
            "<<": lambda a, b: a << min(b, bits) if b >= 0 else 0,
            ">>": lambda a, b: a >> min(b, bits) if b >= 0 else 0,
            "&": lambda a, b: a & b,
            "|": lambda a, b: a | b,
            "^": lambda a, b: a ^ b,
        }
        if self.op in ops:
            result = ops[self.op](l, r)
            mask = (1 << bits) - 1
            result = result & mask
            if signed and result >= (1 << (bits - 1)):
                result -= 1 << bits
            return result
        return 0

    def _eval_cmp(self, l: Any, r: Any) -> bool:
        ops = {
            "==": lambda a, b: a == b,
            "!=": lambda a, b: a != b,
            "<": lambda a, b: a < b,
            "<=": lambda a, b: a <= b,
            ">": lambda a, b: a > b,
            ">=": lambda a, b: a >= b,
        }
        if self.op in ops:
            return ops[self.op](l, r)
        return False

    def is_concrete(self) -> bool:
        return self.left.is_concrete() and self.right.is_concrete()

    def eval(self, env: Dict[str, Any] | None = None) -> Any:
        l = self.left.eval(env)
        r = self.right.eval(env)
        if isinstance(l, (int, float)) and isinstance(r, (int, float)):
            if self.op in self.CMP_OPS:
                return self._eval_cmp(l, r)
            bits = 32
            signed = True
            if isinstance(self.left, SymInt):
                bits = self.left.bits
                signed = self.left.signed
            return self._eval_arith(int(l), int(r), bits, signed)
        if self.op in self.LOGIC_OPS:
            if isinstance(l, bool) and isinstance(r, bool):
                if self.op == "&&":
                    return l and r
                return l or r
        return self

    def __repr__(self) -> str:
        return f"({self.left} {self.op} {self.right})"


class SymUnaryOp(SymValue):
    """Symbolic unary operation."""

    def __init__(self, op: str, operand: SymValue, name: Optional[str] = None):
        super().__init__(name)
        self.op = op
        self.operand = operand

    def is_concrete(self) -> bool:
        return self.operand.is_concrete()

    def eval(self, env: Dict[str, Any] | None = None) -> Any:
        v = self.operand.eval(env)
        if self.op == "-" and isinstance(v, int):
            return -v
        if self.op == "~" and isinstance(v, int):
            return ~v
        if self.op == "not" and isinstance(v, bool):
            return not v
        if self.op == "!" and isinstance(v, (bool, int)):
            return not v
        return self

    def negate(self) -> SymValue:
        if self.op == "not" or self.op == "!":
            return self.operand
        return SymUnaryOp("not", self)

    def __repr__(self) -> str:
        return f"({self.op} {self.operand})"


class SymIte(SymValue):
    """Symbolic if-then-else (for phi nodes at merge points)."""

    def __init__(self, condition: SymValue, then_val: SymValue, else_val: SymValue, name: Optional[str] = None):
        super().__init__(name)
        self.condition = condition
        self.then_val = then_val
        self.else_val = else_val

    def is_concrete(self) -> bool:
        return self.condition.is_concrete()

    def eval(self, env: Dict[str, Any] | None = None) -> Any:
        c = self.condition.eval(env)
        if isinstance(c, bool):
            if c:
                return self.then_val.eval(env)
            return self.else_val.eval(env)
        if isinstance(c, int):
            if c != 0:
                return self.then_val.eval(env)
            return self.else_val.eval(env)
        return self

    def __repr__(self) -> str:
        return f"ITE({self.condition}, {self.then_val}, {self.else_val})"


# ---------------------------------------------------------------------------
# Constraint simplification
# ---------------------------------------------------------------------------

class ConstraintSimplifier:
    """Simplifies symbolic constraints via constant folding, identity
    elimination, dead constraint removal, and boolean simplification."""

    def simplify(self, expr: SymValue) -> SymValue:
        if isinstance(expr, SymBinOp):
            return self._simplify_binop(expr)
        if isinstance(expr, SymUnaryOp):
            return self._simplify_unary(expr)
        if isinstance(expr, SymIte):
            return self._simplify_ite(expr)
        return expr

    def _simplify_binop(self, expr: SymBinOp) -> SymValue:
        left = self.simplify(expr.left)
        right = self.simplify(expr.right)

        # Constant folding
        if left.is_concrete() and right.is_concrete():
            result = SymBinOp(expr.op, left, right).eval(None)
            if isinstance(result, bool):
                return SymBool(value=result)
            if isinstance(result, int):
                bits = left.bits if isinstance(left, SymInt) else 32
                signed = left.signed if isinstance(left, SymInt) else True
                return SymInt(value=result, bits=bits, signed=signed)

        # Identity elimination for arithmetic
        if expr.op == "+" and isinstance(right, SymInt) and right.is_concrete() and right.concrete() == 0:
            return left
        if expr.op == "+" and isinstance(left, SymInt) and left.is_concrete() and left.concrete() == 0:
            return right
        if expr.op == "-" and isinstance(right, SymInt) and right.is_concrete() and right.concrete() == 0:
            return left
        if expr.op == "*" and isinstance(right, SymInt) and right.is_concrete() and right.concrete() == 1:
            return left
        if expr.op == "*" and isinstance(left, SymInt) and left.is_concrete() and left.concrete() == 1:
            return right
        if expr.op == "*" and isinstance(right, SymInt) and right.is_concrete() and right.concrete() == 0:
            return SymInt(value=0)
        if expr.op == "*" and isinstance(left, SymInt) and left.is_concrete() and left.concrete() == 0:
            return SymInt(value=0)

        # Bitwise identity: x & 0xFFFFFFFF = x for 32-bit
        if expr.op == "&" and isinstance(right, SymInt) and right.is_concrete():
            if isinstance(left, SymInt):
                mask = (1 << left.bits) - 1
                if right.concrete() & mask == mask:
                    return left

        # Boolean simplification
        if expr.op == "&&":
            if isinstance(left, SymBool) and left.is_concrete():
                return right if left.concrete() else SymBool(value=False)
            if isinstance(right, SymBool) and right.is_concrete():
                return left if right.concrete() else SymBool(value=False)
        if expr.op == "||":
            if isinstance(left, SymBool) and left.is_concrete():
                return SymBool(value=True) if left.concrete() else right
            if isinstance(right, SymBool) and right.is_concrete():
                return SymBool(value=True) if right.concrete() else left

        # x - x = 0
        if expr.op == "-" and isinstance(left, SymValue) and isinstance(right, SymValue):
            if left.name == right.name and left.name is not None:
                return SymInt(value=0)

        # x == x => True
        if expr.op == "==" and isinstance(left, SymValue) and isinstance(right, SymValue):
            if left.name == right.name and left.name is not None:
                return SymBool(value=True)

        return SymBinOp(expr.op, left, right)

    def _simplify_unary(self, expr: SymUnaryOp) -> SymValue:
        inner = self.simplify(expr.operand)
        if inner.is_concrete():
            result = SymUnaryOp(expr.op, inner).eval(None)
            if isinstance(result, bool):
                return SymBool(value=result)
            if isinstance(result, int):
                return SymInt(value=result)
        # Double negation elimination
        if expr.op in ("not", "!") and isinstance(inner, SymUnaryOp) and inner.op in ("not", "!"):
            return inner.operand
        return SymUnaryOp(expr.op, inner)

    def _simplify_ite(self, expr: SymIte) -> SymValue:
        cond = self.simplify(expr.condition)
        t = self.simplify(expr.then_val)
        e = self.simplify(expr.else_val)
        if isinstance(cond, SymBool) and cond.is_concrete():
            return t if cond.concrete() else e
        if isinstance(cond, SymInt) and cond.is_concrete():
            return t if cond.concrete() != 0 else e
        # ITE(c, x, x) => x
        if isinstance(t, SymValue) and isinstance(e, SymValue) and t.name == e.name:
            return t
        return SymIte(cond, t, e)

    def eliminate_dead_constraints(
        self, constraints: List[SymValue], live_vars: Set[str]
    ) -> List[SymValue]:
        """Remove constraints that only reference variables not in *live_vars*."""
        result: List[SymValue] = []
        for c in constraints:
            refs = self._collect_refs(c)
            if not refs or refs & live_vars:
                result.append(c)
        return result

    def _collect_refs(self, expr: SymValue) -> Set[str]:
        if isinstance(expr, (SymInt, SymBool)):
            if expr.is_concrete():
                return set()
            return {expr.name}
        if isinstance(expr, SymBinOp):
            return self._collect_refs(expr.left) | self._collect_refs(expr.right)
        if isinstance(expr, SymUnaryOp):
            return self._collect_refs(expr.operand)
        if isinstance(expr, SymIte):
            return (
                self._collect_refs(expr.condition)
                | self._collect_refs(expr.then_val)
                | self._collect_refs(expr.else_val)
            )
        if isinstance(expr, SymPointer):
            refs: Set[str] = set()
            if isinstance(expr.base, SymValue):
                refs |= self._collect_refs(expr.base)
            if isinstance(expr.offset, SymValue):
                refs |= self._collect_refs(expr.offset)
            return refs
        return {expr.name} if not expr.is_concrete() else set()


# ---------------------------------------------------------------------------
# Memory model
# ---------------------------------------------------------------------------

@dataclass
class AllocationBlock:
    address: int
    size: Union[int, SymInt]
    freed: bool = False


class SymHeap:
    """Symbolic heap: allocate, read, write with metadata tracking."""

    def __init__(self) -> None:
        self._next_addr: int = 0x1000
        self._store: Dict[int, SymValue] = {}
        self.blocks: Dict[int, AllocationBlock] = {}

    def allocate(self, size: Union[int, SymInt]) -> SymPointer:
        addr = self._next_addr
        int_size = size.concrete() if isinstance(size, SymInt) and size.is_concrete() else 64
        if isinstance(size, int):
            int_size = size
        self._next_addr += int_size + 16  # padding
        self.blocks[addr] = AllocationBlock(address=addr, size=size)
        return SymPointer(base=addr, offset=SymInt(value=0))

    def read(self, ptr: SymPointer) -> SymValue:
        addr = self._resolve(ptr)
        if addr is not None and addr in self._store:
            return self._store[addr]
        return SymInt(name=f"mem@{ptr}")

    def write(self, ptr: SymPointer, value: SymValue) -> None:
        addr = self._resolve(ptr)
        if addr is not None:
            self._store[addr] = value

    def free(self, ptr: SymPointer) -> Optional[str]:
        """Free a pointer. Returns an error string if double-free detected."""
        addr = self._resolve_base(ptr)
        if addr is not None:
            if addr in self.blocks:
                if self.blocks[addr].freed:
                    return "double-free"
                self.blocks[addr].freed = True
                return None
        return "invalid-free"

    def is_freed(self, ptr: SymPointer) -> bool:
        addr = self._resolve_base(ptr)
        if addr is not None and addr in self.blocks:
            return self.blocks[addr].freed
        return False

    def check_bounds(self, ptr: SymPointer) -> Optional[str]:
        base_addr = self._resolve_base(ptr)
        if base_addr is None:
            return None  # can't check symbolic addresses
        if base_addr not in self.blocks:
            return "invalid-pointer"
        blk = self.blocks[base_addr]
        offset = self._resolve_offset(ptr)
        if offset is None:
            return None  # symbolic offset, can't check statically
        blk_size = blk.size if isinstance(blk.size, int) else (
            blk.size.concrete() if isinstance(blk.size, SymInt) and blk.size.is_concrete() else None
        )
        if blk_size is not None and (offset < 0 or offset >= blk_size):
            return "buffer-overflow"
        return None

    def _resolve(self, ptr: SymPointer) -> Optional[int]:
        b = self._resolve_base(ptr)
        o = self._resolve_offset(ptr)
        if b is not None and o is not None:
            return b + o
        return None

    def _resolve_base(self, ptr: SymPointer) -> Optional[int]:
        if isinstance(ptr.base, int):
            return ptr.base
        if isinstance(ptr.base, SymInt) and ptr.base.is_concrete():
            return ptr.base.concrete()
        return None

    def _resolve_offset(self, ptr: SymPointer) -> Optional[int]:
        if isinstance(ptr.offset, int):
            return ptr.offset
        if isinstance(ptr.offset, SymInt) and ptr.offset.is_concrete():
            return ptr.offset.concrete()
        return None

    def clone(self) -> SymHeap:
        h = SymHeap()
        h._next_addr = self._next_addr
        h._store = dict(self._store)
        h.blocks = {k: AllocationBlock(v.address, v.size, v.freed) for k, v in self.blocks.items()}
        return h


# ---------------------------------------------------------------------------
# Symbolic state
# ---------------------------------------------------------------------------

class SymState:
    """Complete symbolic execution state for one path."""

    def __init__(self) -> None:
        self.variables: Dict[str, SymValue] = {}
        self.memory: SymHeap = SymHeap()
        self.path_condition: List[SymValue] = []
        self.allocated_blocks: Dict[int, AllocationBlock] = {}
        self.freed_blocks: Set[int] = set()
        self.effects: List[Effect] = []
        self.bugs: List[Bug] = []
        self.return_value: Optional[SymValue] = None
        self.is_terminated: bool = False

    def add_constraint(self, constraint: SymValue) -> None:
        simplified = _simplifier.simplify(constraint)
        if isinstance(simplified, SymBool) and simplified.is_concrete():
            if not simplified.concrete():
                self.is_terminated = True
            return
        self.path_condition.append(simplified)

    def is_feasible(self) -> bool:
        """Quick feasibility check via constant propagation."""
        if self.is_terminated:
            return False
        for c in self.path_condition:
            if isinstance(c, SymBool) and c.is_concrete() and not c.concrete():
                return False
            evaluated = c.eval(None)
            if isinstance(evaluated, bool) and not evaluated:
                return False
        return not self._has_contradiction()

    def _has_contradiction(self) -> bool:
        """Detect simple contradictions: x == a AND x == b with a != b."""
        equalities: Dict[str, Any] = {}
        for c in self.path_condition:
            if isinstance(c, SymBinOp) and c.op == "==":
                if isinstance(c.left, (SymInt, SymBool)) and not c.left.is_concrete():
                    if c.right.is_concrete():
                        key = c.left.name
                        val = c.right.eval(None)
                        if key in equalities and equalities[key] != val:
                            return True
                        equalities[key] = val
                if isinstance(c.right, (SymInt, SymBool)) and not c.right.is_concrete():
                    if c.left.is_concrete():
                        key = c.right.name
                        val = c.left.eval(None)
                        if key in equalities and equalities[key] != val:
                            return True
                        equalities[key] = val
        return False

    def lookup(self, name: str) -> SymValue:
        if name in self.variables:
            return self.variables[name]
        fresh = SymInt(name=name)
        self.variables[name] = fresh
        return fresh

    def assign(self, name: str, value: SymValue) -> None:
        self.variables[name] = value

    def clone(self) -> SymState:
        s = SymState()
        s.variables = dict(self.variables)
        s.memory = self.memory.clone()
        s.path_condition = list(self.path_condition)
        s.allocated_blocks = dict(self.allocated_blocks)
        s.freed_blocks = set(self.freed_blocks)
        s.effects = list(self.effects)
        s.bugs = list(self.bugs)
        s.return_value = self.return_value
        s.is_terminated = self.is_terminated
        return s


# Singleton simplifier
_simplifier = ConstraintSimplifier()


# ---------------------------------------------------------------------------
# Bug, Effect, ExecutionPath, ExecutionTree, FunctionSummary
# ---------------------------------------------------------------------------

class Severity(enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Bug:
    kind: str
    location: Optional[str]
    description: str
    severity: str
    symbolic_input: Optional[Dict[str, Any]] = None

    def __repr__(self) -> str:
        return f"Bug({self.kind}, {self.severity}, {self.description[:60]})"


@dataclass
class Effect:
    kind: str  # 'write', 'alloc', 'free', 'call'
    details: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Effect({self.kind}, {self.details})"


@dataclass
class ExecutionPath:
    constraints: List[SymValue] = field(default_factory=list)
    effects: List[Effect] = field(default_factory=list)
    return_value: Optional[SymValue] = None
    bugs_found: List[Bug] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"ExecutionPath(constraints={len(self.constraints)}, "
            f"effects={len(self.effects)}, bugs={len(self.bugs_found)})"
        )


@dataclass
class ExecutionTree:
    paths: List[ExecutionPath] = field(default_factory=list)
    function_name: str = ""
    language: str = "c"

    def path_count(self) -> int:
        return len(self.paths)

    def all_bugs(self) -> List[Bug]:
        bugs: List[Bug] = []
        for p in self.paths:
            bugs.extend(p.bugs_found)
        return bugs

    def __repr__(self) -> str:
        return f"ExecutionTree({self.function_name}, {self.language}, paths={len(self.paths)})"


@dataclass
class FunctionSummary:
    name: str
    preconditions: List[SymValue] = field(default_factory=list)
    postconditions: List[SymValue] = field(default_factory=list)
    effects: List[Effect] = field(default_factory=list)
    bugs: List[Bug] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"FunctionSummary({self.name}, pre={len(self.preconditions)}, post={len(self.postconditions)})"


# ---------------------------------------------------------------------------
# Path merger
# ---------------------------------------------------------------------------

class PathMerger:
    """Merge two execution paths that diverged at an if/else into one."""

    def merge(
        self, cond: SymValue, true_state: SymState, false_state: SymState
    ) -> SymState:
        merged = SymState()
        all_vars = set(true_state.variables.keys()) | set(false_state.variables.keys())
        for var in all_vars:
            t_val = true_state.variables.get(var, SymInt(name=var))
            f_val = false_state.variables.get(var, SymInt(name=var))
            if self._values_equal(t_val, f_val):
                merged.variables[var] = t_val
            else:
                merged.variables[var] = SymIte(cond, t_val, f_val)

        merged.memory = self._merge_heaps(cond, true_state.memory, false_state.memory)
        merged.path_condition = self._merge_constraints(
            cond, true_state.path_condition, false_state.path_condition
        )
        merged.effects = true_state.effects + false_state.effects
        merged.bugs = true_state.bugs + false_state.bugs
        merged.allocated_blocks = {**true_state.allocated_blocks, **false_state.allocated_blocks}
        merged.freed_blocks = true_state.freed_blocks | false_state.freed_blocks

        if true_state.return_value is not None and false_state.return_value is not None:
            if self._values_equal(true_state.return_value, false_state.return_value):
                merged.return_value = true_state.return_value
            else:
                merged.return_value = SymIte(cond, true_state.return_value, false_state.return_value)
        elif true_state.return_value is not None:
            merged.return_value = true_state.return_value
        elif false_state.return_value is not None:
            merged.return_value = false_state.return_value

        return merged

    def _values_equal(self, a: SymValue, b: SymValue) -> bool:
        if a is b:
            return True
        if type(a) is type(b):
            if isinstance(a, SymInt) and a.is_concrete() and b.is_concrete():
                return a.concrete() == b.concrete()
            if isinstance(a, SymBool) and a.is_concrete() and b.is_concrete():
                return a.concrete() == b.concrete()
            if a.name == b.name:
                return True
        return False

    def _merge_heaps(self, cond: SymValue, h1: SymHeap, h2: SymHeap) -> SymHeap:
        merged = h1.clone()
        for addr, val in h2._store.items():
            if addr in merged._store:
                if not self._values_equal(merged._store[addr], val):
                    merged._store[addr] = SymIte(cond, merged._store[addr], val)
            else:
                merged._store[addr] = val
        for addr, blk in h2.blocks.items():
            if addr not in merged.blocks:
                merged.blocks[addr] = blk
        return merged

    def _merge_constraints(
        self, cond: SymValue, c1: List[SymValue], c2: List[SymValue]
    ) -> List[SymValue]:
        common: List[SymValue] = []
        s1 = {id(c) for c in c1}
        s2 = {id(c) for c in c2}
        # Constraints present in both branches are kept unconditionally
        name_map_1 = {c.name: c for c in c1 if hasattr(c, "name")}
        name_map_2 = {c.name: c for c in c2 if hasattr(c, "name")}
        shared_names = set(name_map_1.keys()) & set(name_map_2.keys())
        for n in shared_names:
            common.append(name_map_1[n])
        # Branch-specific constraints become conditional
        for c in c1:
            if hasattr(c, "name") and c.name in shared_names:
                continue
            common.append(SymBinOp("||", SymUnaryOp("not", cond), c))
        for c in c2:
            if hasattr(c, "name") and c.name in shared_names:
                continue
            common.append(SymBinOp("||", cond, c))
        return common


# ---------------------------------------------------------------------------
# Loop handler
# ---------------------------------------------------------------------------

class LoopHandler:
    """Bounded loop unrolling with simple invariant detection and widening."""

    def __init__(self, max_depth: int = 10):
        self.max_depth = max_depth

    def unroll(
        self,
        body_fn,
        cond_fn,
        state: SymState,
        executor: "SymbolicExecutor",
    ) -> List[SymState]:
        """Unroll a loop up to *max_depth* times.

        *cond_fn(state)* returns a SymValue for the loop condition.
        *body_fn(state, executor)* executes the loop body and returns a new state.
        """
        result_states: List[SymState] = []
        current = state.clone()

        for iteration in range(self.max_depth):
            cond = cond_fn(current)
            cond_eval = cond.eval(None)

            # Condition is concretely false -> loop exits
            if isinstance(cond_eval, bool) and not cond_eval:
                result_states.append(current)
                return result_states
            if isinstance(cond_eval, int) and cond_eval == 0:
                result_states.append(current)
                return result_states

            # Fork: one state exits, one continues
            exit_state = current.clone()
            neg = self._negate(cond)
            exit_state.add_constraint(neg)
            if exit_state.is_feasible():
                result_states.append(exit_state)

            # Continue state
            cont = current.clone()
            cont.add_constraint(cond)
            if not cont.is_feasible():
                break

            # Execute body
            new_state = body_fn(cont, executor)
            if new_state.is_terminated:
                break

            # Simple invariant detection: check if a counter variable changed
            invariant = self._detect_invariant(current, new_state)
            if invariant is not None and iteration >= 2:
                widened = self._widen(new_state, invariant)
                result_states.append(widened)
                return result_states

            current = new_state

        # Reached max unrolling depth
        result_states.append(current)
        return result_states

    def _negate(self, cond: SymValue) -> SymValue:
        if isinstance(cond, SymBool):
            return cond.negate()
        if isinstance(cond, SymUnaryOp) and cond.op in ("not", "!"):
            return cond.operand
        return SymUnaryOp("not", cond)

    def _detect_invariant(
        self, prev: SymState, curr: SymState
    ) -> Optional[Tuple[str, str]]:
        """Detect a simple counter invariant: a variable that increments or
        decrements by a constant each iteration.  Returns (var_name, direction)."""
        for var in prev.variables:
            if var not in curr.variables:
                continue
            old = prev.variables[var]
            new = curr.variables[var]
            if isinstance(old, SymInt) and old.is_concrete() and isinstance(new, SymInt) and new.is_concrete():
                diff = new.concrete() - old.concrete()
                if diff == 1:
                    return (var, "increment")
                if diff == -1:
                    return (var, "decrement")
            if isinstance(new, SymBinOp) and new.op == "+":
                if isinstance(new.right, SymInt) and new.right.is_concrete() and new.right.concrete() == 1:
                    if isinstance(new.left, SymValue) and new.left.name == var:
                        return (var, "increment")
        return None

    def _widen(self, state: SymState, invariant: Tuple[str, str]) -> SymState:
        """Apply widening: replace the loop counter with an unconstrained
        symbolic value bounded by the loop guard."""
        var_name, direction = invariant
        widened = state.clone()
        fresh = SymInt(name=f"{var_name}_widened")
        widened.variables[var_name] = fresh
        if direction == "increment":
            lb = state.variables.get(var_name)
            if lb is not None:
                widened.add_constraint(SymBinOp(">=", fresh, lb))
        else:
            ub = state.variables.get(var_name)
            if ub is not None:
                widened.add_constraint(SymBinOp("<=", fresh, ub))
        return widened


# ---------------------------------------------------------------------------
# Bug detectors
# ---------------------------------------------------------------------------

class BugDetector:
    """Detect various bug classes during symbolic execution."""

    def __init__(self, language: str = "c"):
        self.language = language

    def check_division(self, divisor: SymValue, location: Optional[str] = None) -> Optional[Bug]:
        if isinstance(divisor, SymInt) and divisor.is_concrete() and divisor.concrete() == 0:
            return Bug(
                kind="division-by-zero",
                location=location,
                description="Division by a concretely zero divisor",
                severity="critical",
            )
        if not divisor.is_concrete():
            return Bug(
                kind="possible-division-by-zero",
                location=location,
                description=f"Divisor {divisor} may be zero",
                severity="medium",
            )
        return None

    def check_overflow(
        self,
        op: str,
        left: SymValue,
        right: SymValue,
        result: SymValue,
        bits: int = 32,
        signed: bool = True,
        location: Optional[str] = None,
    ) -> Optional[Bug]:
        if not (isinstance(left, SymInt) and left.is_concrete() and isinstance(right, SymInt) and right.is_concrete()):
            return None
        l, r = left.concrete(), right.concrete()
        raw_ops = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b}
        if op not in raw_ops:
            return None
        raw = raw_ops[op](l, r)
        if signed:
            lo = -(1 << (bits - 1))
            hi = (1 << (bits - 1)) - 1
        else:
            lo = 0
            hi = (1 << bits) - 1
        if raw < lo or raw > hi:
            severity = "high" if self.language == "c" else "medium"
            desc = (
                f"Integer overflow: {l} {op} {r} = {raw} exceeds "
                f"{'signed' if signed else 'unsigned'} {bits}-bit range [{lo}, {hi}]"
            )
            if self.language == "rust":
                desc += " (Rust panics in debug mode)"
            else:
                desc += " (undefined behavior in C)"
            return Bug(
                kind="integer-overflow",
                location=location,
                description=desc,
                severity=severity,
            )
        return None

    def check_null_deref(self, ptr: SymPointer, location: Optional[str] = None) -> Optional[Bug]:
        if ptr.is_null():
            return Bug(
                kind="null-dereference",
                location=location,
                description="Dereference of a null pointer",
                severity="critical",
            )
        return None

    def check_buffer_overflow(
        self, ptr: SymPointer, heap: SymHeap, location: Optional[str] = None
    ) -> Optional[Bug]:
        err = heap.check_bounds(ptr)
        if err == "buffer-overflow":
            return Bug(
                kind="buffer-overflow",
                location=location,
                description=f"Array access out of bounds at {ptr}",
                severity="critical",
            )
        if err == "invalid-pointer":
            return Bug(
                kind="invalid-pointer",
                location=location,
                description=f"Access through invalid pointer {ptr}",
                severity="high",
            )
        return None

    def check_use_after_free(
        self, ptr: SymPointer, heap: SymHeap, location: Optional[str] = None
    ) -> Optional[Bug]:
        if heap.is_freed(ptr):
            return Bug(
                kind="use-after-free",
                location=location,
                description=f"Memory access after free at {ptr}",
                severity="critical",
            )
        return None

    def check_double_free(self, err: Optional[str], location: Optional[str] = None) -> Optional[Bug]:
        if err == "double-free":
            return Bug(
                kind="double-free",
                location=location,
                description="Memory freed more than once",
                severity="critical",
            )
        return None


# ---------------------------------------------------------------------------
# Symbolic Executor
# ---------------------------------------------------------------------------

class SymbolicExecutor:
    """Main symbolic execution engine.

    Executes a function AST symbolically, forking on branches, tracking
    constraints, detecting bugs, and producing an ``ExecutionTree``.
    """

    def __init__(self, max_loop_depth: int = 10, max_paths: int = 256):
        self.max_loop_depth = max_loop_depth
        self.max_paths = max_paths
        self.simplifier = ConstraintSimplifier()
        self.merger = PathMerger()
        self.loop_handler = LoopHandler(max_depth=max_loop_depth)
        self.summaries: Dict[str, FunctionSummary] = {}
        self._bug_detector: Optional[BugDetector] = None

    def execute(self, function_ast: Dict[str, Any], language: str = "c") -> ExecutionTree:
        """Symbolically execute *function_ast* under C or Rust semantics."""
        self._bug_detector = BugDetector(language=language)
        func_name = function_ast.get("name", "<anonymous>")
        params = function_ast.get("params", [])
        body = function_ast.get("body", [])

        # Initial state with symbolic parameters
        init_state = SymState()
        for p in params:
            pname = p.get("name", p) if isinstance(p, dict) else str(p)
            ptype = p.get("type", "int") if isinstance(p, dict) else "int"
            sym = self._make_symbolic_param(pname, ptype)
            init_state.assign(pname, sym)

        # Execute the body, collecting all completed paths
        completed = self._exec_block(body, init_state, language)

        # Build execution tree
        tree = ExecutionTree(function_name=func_name, language=language)
        for st in completed:
            path = ExecutionPath(
                constraints=st.path_condition,
                effects=st.effects,
                return_value=st.return_value,
                bugs_found=st.bugs,
            )
            tree.paths.append(path)

        # Compute and cache function summary
        summary = self._compute_summary(func_name, tree)
        self.summaries[func_name] = summary
        return tree

    def _make_symbolic_param(self, name: str, typ: str) -> SymValue:
        if typ in ("int", "i32", "i64", "u32", "u64", "short", "long"):
            bits = {"int": 32, "i32": 32, "i64": 64, "u32": 32, "u64": 64, "short": 16, "long": 64}.get(typ, 32)
            signed = typ not in ("u32", "u64", "unsigned")
            return SymInt(bits=bits, signed=signed, name=name)
        if typ in ("bool",):
            return SymBool(name=name)
        if typ.endswith("*") or typ == "pointer":
            return SymPointer(base=SymInt(name=f"{name}_base"), offset=SymInt(value=0), name=name)
        return SymInt(name=name)

    # -- Statement execution ------------------------------------------------

    def _exec_block(
        self, stmts: List[Dict[str, Any]], state: SymState, lang: str
    ) -> List[SymState]:
        """Execute a list of statements, returning all resulting states."""
        active: List[SymState] = [state]
        for stmt in stmts:
            next_active: List[SymState] = []
            for s in active:
                if s.is_terminated:
                    next_active.append(s)
                    continue
                results = self._exec_stmt(stmt, s, lang)
                next_active.extend(results)
                if len(next_active) > self.max_paths:
                    next_active = next_active[: self.max_paths]
            active = next_active
        return active

    def _exec_stmt(
        self, stmt: Dict[str, Any], state: SymState, lang: str
    ) -> List[SymState]:
        kind = stmt.get("kind", "")
        if kind == "assign":
            return self._exec_assign(stmt, state, lang)
        if kind == "return":
            return self._exec_return(stmt, state, lang)
        if kind == "if":
            return self._exec_if(stmt, state, lang)
        if kind == "while" or kind == "for" or kind == "loop":
            return self._exec_loop(stmt, state, lang)
        if kind == "call":
            return self._exec_call(stmt, state, lang)
        if kind == "alloc":
            return self._exec_alloc(stmt, state, lang)
        if kind == "free":
            return self._exec_free(stmt, state, lang)
        if kind == "deref_write":
            return self._exec_deref_write(stmt, state, lang)
        if kind == "deref_read":
            return self._exec_deref_read(stmt, state, lang)
        if kind == "array_write":
            return self._exec_array_write(stmt, state, lang)
        if kind == "array_read":
            return self._exec_array_read(stmt, state, lang)
        if kind == "field_write":
            return self._exec_field_write(stmt, state, lang)
        if kind == "field_read":
            return self._exec_field_read(stmt, state, lang)
        if kind == "block":
            return self._exec_block(stmt.get("body", []), state, lang)
        if kind == "break":
            state.is_terminated = True
            return [state]
        if kind == "expr":
            self._eval_expr(stmt.get("expr", {}), state, lang)
            return [state]
        # Unknown statement: treat as no-op
        return [state]

    def _exec_assign(self, stmt: Dict, state: SymState, lang: str) -> List[SymState]:
        target = stmt.get("target", "")
        expr = stmt.get("value", {})
        val = self._eval_expr(expr, state, lang)
        state.assign(target, val)
        state.effects.append(Effect(kind="write", details={"var": target}))
        return [state]

    def _exec_return(self, stmt: Dict, state: SymState, lang: str) -> List[SymState]:
        expr = stmt.get("value", None)
        if expr is not None:
            val = self._eval_expr(expr, state, lang)
            state.return_value = val
        else:
            state.return_value = SymInt(value=0)
        state.is_terminated = True
        return [state]

    def _exec_if(self, stmt: Dict, state: SymState, lang: str) -> List[SymState]:
        cond_expr = stmt.get("condition", {})
        cond = self._eval_expr(cond_expr, state, lang)
        cond_sym = self._to_sym_bool(cond)

        then_body = stmt.get("then", [])
        else_body = stmt.get("else", [])

        # Concrete condition -> take one branch
        if isinstance(cond_sym, SymBool) and cond_sym.is_concrete():
            if cond_sym.concrete():
                return self._exec_block(then_body, state, lang)
            return self._exec_block(else_body, state, lang) if else_body else [state]

        results: List[SymState] = []

        # True branch
        true_state = state.clone()
        true_state.add_constraint(cond_sym)
        if true_state.is_feasible():
            true_results = self._exec_block(then_body, true_state, lang)
            results.extend(true_results)

        # False branch
        false_state = state.clone()
        neg = self._negate_sym(cond_sym)
        false_state.add_constraint(neg)
        if false_state.is_feasible():
            if else_body:
                false_results = self._exec_block(else_body, false_state, lang)
                results.extend(false_results)
            else:
                results.append(false_state)

        # Try path merging if both branches completed with one state each
        if len(results) == 2 and not results[0].is_terminated and not results[1].is_terminated:
            merged = self.merger.merge(cond_sym, results[0], results[1])
            return [merged]

        if not results:
            state.is_terminated = True
            return [state]
        return results

    def _exec_loop(self, stmt: Dict, state: SymState, lang: str) -> List[SymState]:
        cond_ast = stmt.get("condition", {})
        body_ast = stmt.get("body", [])

        def cond_fn(s: SymState) -> SymValue:
            return self._eval_expr(cond_ast, s, lang)

        def body_fn(s: SymState, _executor: SymbolicExecutor) -> SymState:
            results = self._exec_block(body_ast, s, lang)
            if results:
                return results[0]
            return s

        return self.loop_handler.unroll(body_fn, cond_fn, state, self)

    def _exec_call(self, stmt: Dict, state: SymState, lang: str) -> List[SymState]:
        func_name = stmt.get("function", "")
        args_ast = stmt.get("args", [])
        target = stmt.get("target", None)

        args = [self._eval_expr(a, state, lang) for a in args_ast]

        # Check if we have a cached summary
        if func_name in self.summaries:
            summary = self.summaries[func_name]
            ret_val = SymInt(name=f"{func_name}_ret")
            for post in summary.postconditions:
                state.add_constraint(post)
            state.bugs.extend(summary.bugs)
            if target:
                state.assign(target, ret_val)
            state.effects.append(Effect(kind="call", details={"function": func_name, "args": len(args)}))
            return [state]

        # No summary: return fresh symbolic value
        ret_val = SymInt(name=f"{func_name}_ret")
        if target:
            state.assign(target, ret_val)
        state.effects.append(Effect(kind="call", details={"function": func_name, "args": len(args)}))
        return [state]

    def _exec_alloc(self, stmt: Dict, state: SymState, lang: str) -> List[SymState]:
        target = stmt.get("target", "")
        size_expr = stmt.get("size", {"kind": "literal", "value": 1})
        size = self._eval_expr(size_expr, state, lang)
        if not isinstance(size, SymInt):
            size = SymInt(value=1)

        ptr = state.memory.allocate(size)
        state.assign(target, ptr)
        state.effects.append(Effect(kind="alloc", details={"target": target, "size": size}))
        return [state]

    def _exec_free(self, stmt: Dict, state: SymState, lang: str) -> List[SymState]:
        ptr_name = stmt.get("pointer", "")
        ptr_val = state.lookup(ptr_name)
        location = stmt.get("location", None)

        if isinstance(ptr_val, SymPointer):
            err = state.memory.free(ptr_val)
            bug = self._bug_detector.check_double_free(err, location)
            if bug is not None:
                state.bugs.append(bug)
            state.effects.append(Effect(kind="free", details={"pointer": ptr_name}))
        else:
            state.bugs.append(Bug(
                kind="invalid-free",
                location=location,
                description=f"Free called on non-pointer value {ptr_name}",
                severity="high",
            ))
        return [state]

    def _exec_deref_write(self, stmt: Dict, state: SymState, lang: str) -> List[SymState]:
        ptr_name = stmt.get("pointer", "")
        val_expr = stmt.get("value", {})
        location = stmt.get("location", None)

        ptr_val = state.lookup(ptr_name)
        value = self._eval_expr(val_expr, state, lang)

        if isinstance(ptr_val, SymPointer):
            null_bug = self._bug_detector.check_null_deref(ptr_val, location)
            if null_bug is not None:
                state.bugs.append(null_bug)
            uaf_bug = self._bug_detector.check_use_after_free(ptr_val, state.memory, location)
            if uaf_bug is not None:
                state.bugs.append(uaf_bug)
            oob_bug = self._bug_detector.check_buffer_overflow(ptr_val, state.memory, location)
            if oob_bug is not None:
                state.bugs.append(oob_bug)
            state.memory.write(ptr_val, value)
            state.effects.append(Effect(kind="write", details={"pointer": ptr_name}))
        return [state]

    def _exec_deref_read(self, stmt: Dict, state: SymState, lang: str) -> List[SymState]:
        target = stmt.get("target", "")
        ptr_name = stmt.get("pointer", "")
        location = stmt.get("location", None)

        ptr_val = state.lookup(ptr_name)
        if isinstance(ptr_val, SymPointer):
            null_bug = self._bug_detector.check_null_deref(ptr_val, location)
            if null_bug is not None:
                state.bugs.append(null_bug)
            uaf_bug = self._bug_detector.check_use_after_free(ptr_val, state.memory, location)
            if uaf_bug is not None:
                state.bugs.append(uaf_bug)
            val = state.memory.read(ptr_val)
            state.assign(target, val)
        else:
            state.assign(target, SymInt(name=f"deref_{ptr_name}"))
        return [state]

    def _exec_array_write(self, stmt: Dict, state: SymState, lang: str) -> List[SymState]:
        arr_name = stmt.get("array", "")
        idx_expr = stmt.get("index", {})
        val_expr = stmt.get("value", {})
        location = stmt.get("location", None)

        arr = state.lookup(arr_name)
        idx = self._eval_expr(idx_expr, state, lang)
        val = self._eval_expr(val_expr, state, lang)

        if isinstance(arr, SymArray):
            # Bounds check
            if arr.size is not None and isinstance(idx, SymInt) and idx.is_concrete():
                arr_size = arr.size if isinstance(arr.size, int) else (
                    arr.size.concrete() if isinstance(arr.size, SymInt) and arr.size.is_concrete() else None
                )
                if arr_size is not None and idx.concrete() >= arr_size:
                    state.bugs.append(Bug(
                        kind="buffer-overflow",
                        location=location,
                        description=f"Array index {idx.concrete()} >= size {arr_size}",
                        severity="critical",
                    ))
            new_arr = arr.write(idx, val)
            state.assign(arr_name, new_arr)
        else:
            fresh_arr = SymArray(name=arr_name)
            fresh_arr = fresh_arr.write(idx, val)
            state.assign(arr_name, fresh_arr)
        return [state]

    def _exec_array_read(self, stmt: Dict, state: SymState, lang: str) -> List[SymState]:
        target = stmt.get("target", "")
        arr_name = stmt.get("array", "")
        idx_expr = stmt.get("index", {})
        location = stmt.get("location", None)

        arr = state.lookup(arr_name)
        idx = self._eval_expr(idx_expr, state, lang)

        if isinstance(arr, SymArray):
            if arr.size is not None and isinstance(idx, SymInt) and idx.is_concrete():
                arr_size = arr.size if isinstance(arr.size, int) else (
                    arr.size.concrete() if isinstance(arr.size, SymInt) and arr.size.is_concrete() else None
                )
                if arr_size is not None and idx.concrete() >= arr_size:
                    state.bugs.append(Bug(
                        kind="buffer-overflow",
                        location=location,
                        description=f"Array read index {idx.concrete()} >= size {arr_size}",
                        severity="critical",
                    ))
            val = arr.read(idx)
            state.assign(target, val)
        else:
            state.assign(target, SymInt(name=f"{arr_name}[{idx}]"))
        return [state]

    def _exec_field_write(self, stmt: Dict, state: SymState, lang: str) -> List[SymState]:
        struct_name = stmt.get("struct", "")
        field_name = stmt.get("field", "")
        val_expr = stmt.get("value", {})

        obj = state.lookup(struct_name)
        val = self._eval_expr(val_expr, state, lang)

        if isinstance(obj, SymStruct):
            new_obj = obj.set_field(field_name, val)
            state.assign(struct_name, new_obj)
        else:
            s = SymStruct(name=struct_name)
            s = s.set_field(field_name, val)
            state.assign(struct_name, s)
        return [state]

    def _exec_field_read(self, stmt: Dict, state: SymState, lang: str) -> List[SymState]:
        target = stmt.get("target", "")
        struct_name = stmt.get("struct", "")
        field_name = stmt.get("field", "")

        obj = state.lookup(struct_name)
        if isinstance(obj, SymStruct):
            val = obj.get_field(field_name)
        else:
            val = SymInt(name=f"{struct_name}.{field_name}")
        state.assign(target, val)
        return [state]

    # -- Expression evaluation ----------------------------------------------

    def _eval_expr(self, expr: Any, state: SymState, lang: str) -> SymValue:
        if not isinstance(expr, dict):
            if isinstance(expr, int):
                return SymInt(value=expr)
            if isinstance(expr, bool):
                return SymBool(value=expr)
            if isinstance(expr, str):
                return state.lookup(expr)
            return SymInt(value=0)

        kind = expr.get("kind", "")

        if kind == "literal":
            val = expr.get("value", 0)
            if isinstance(val, bool):
                return SymBool(value=val)
            if isinstance(val, int):
                bits = expr.get("bits", 32)
                signed = expr.get("signed", True)
                return SymInt(value=val, bits=bits, signed=signed)
            return SymInt(value=0)

        if kind == "var":
            name = expr.get("name", "")
            return state.lookup(name)

        if kind == "binop":
            op = expr.get("op", "+")
            left = self._eval_expr(expr.get("left", {}), state, lang)
            right = self._eval_expr(expr.get("right", {}), state, lang)
            location = expr.get("location", None)
            return self._eval_binop(op, left, right, lang, state, location)

        if kind == "unaryop":
            op = expr.get("op", "-")
            operand = self._eval_expr(expr.get("operand", {}), state, lang)
            return self._eval_unaryop(op, operand)

        if kind == "call":
            func_name = expr.get("function", "")
            args = [self._eval_expr(a, state, lang) for a in expr.get("args", [])]
            return self._eval_call_expr(func_name, args, state)

        if kind == "deref":
            ptr_name = expr.get("pointer", "")
            ptr_val = state.lookup(ptr_name)
            if isinstance(ptr_val, SymPointer):
                return state.memory.read(ptr_val)
            return SymInt(name=f"*{ptr_name}")

        if kind == "addr":
            var_name = expr.get("var", "")
            return SymPointer(
                base=SymInt(name=f"&{var_name}_base"),
                offset=SymInt(value=0),
                name=f"&{var_name}",
            )

        if kind == "index":
            arr_name = expr.get("array", "")
            idx = self._eval_expr(expr.get("index", {}), state, lang)
            arr = state.lookup(arr_name)
            if isinstance(arr, SymArray):
                return arr.read(idx)
            return SymInt(name=f"{arr_name}[{idx}]")

        if kind == "field":
            struct_name = expr.get("struct", "")
            field_name = expr.get("field", "")
            obj = state.lookup(struct_name)
            if isinstance(obj, SymStruct):
                return obj.get_field(field_name)
            return SymInt(name=f"{struct_name}.{field_name}")

        if kind == "cast":
            inner = self._eval_expr(expr.get("expr", {}), state, lang)
            target_type = expr.get("target_type", "int")
            return self._eval_cast(inner, target_type)

        if kind == "ternary":
            cond = self._eval_expr(expr.get("condition", {}), state, lang)
            then_v = self._eval_expr(expr.get("then", {}), state, lang)
            else_v = self._eval_expr(expr.get("else", {}), state, lang)
            return SymIte(self._to_sym_bool(cond), then_v, else_v)

        if kind == "sizeof":
            type_name = expr.get("type", "int")
            sizes = {"char": 1, "short": 2, "int": 4, "long": 8, "float": 4, "double": 8, "pointer": 8}
            return SymInt(value=sizes.get(type_name, 4))

        # Default: fresh symbolic
        return SymInt(name=f"expr_{SymValue._counter}")

    def _eval_binop(
        self,
        op: str,
        left: SymValue,
        right: SymValue,
        lang: str,
        state: SymState,
        location: Optional[str] = None,
    ) -> SymValue:
        # Division-by-zero check
        if op in ("/", "%"):
            bug = self._bug_detector.check_division(right, location)
            if bug is not None:
                state.bugs.append(bug)
                if isinstance(right, SymInt) and right.is_concrete() and right.concrete() == 0:
                    return SymInt(value=0)

        # Attempt constant folding
        sym_expr = SymBinOp(op, left, right)
        simplified = self.simplifier.simplify(sym_expr)

        # Overflow check for concrete arithmetic
        if op in ("+", "-", "*") and isinstance(left, SymInt) and isinstance(right, SymInt):
            bits = left.bits
            signed = left.signed
            overflow_bug = self._bug_detector.check_overflow(op, left, right, simplified, bits, signed, location)
            if overflow_bug is not None:
                state.bugs.append(overflow_bug)
                if lang == "rust":
                    # Rust panics on overflow in debug mode; we model this as termination
                    state.is_terminated = True

        return simplified

    def _eval_unaryop(self, op: str, operand: SymValue) -> SymValue:
        expr = SymUnaryOp(op, operand)
        return self.simplifier.simplify(expr)

    def _eval_call_expr(
        self, func_name: str, args: List[SymValue], state: SymState
    ) -> SymValue:
        if func_name in self.summaries:
            summary = self.summaries[func_name]
            for post in summary.postconditions:
                state.add_constraint(post)
            state.effects.append(Effect(kind="call", details={"function": func_name}))
            return SymInt(name=f"{func_name}_ret")
        state.effects.append(Effect(kind="call", details={"function": func_name}))
        return SymInt(name=f"{func_name}_ret")

    def _eval_cast(self, value: SymValue, target_type: str) -> SymValue:
        type_info = {
            "int": (32, True), "i32": (32, True), "i64": (64, True),
            "u32": (32, False), "u64": (64, False), "i16": (16, True),
            "u16": (16, False), "i8": (8, True), "u8": (8, False),
            "short": (16, True), "long": (64, True), "char": (8, True),
            "bool": (1, False),
        }
        if target_type == "bool":
            if isinstance(value, SymInt) and value.is_concrete():
                return SymBool(value=value.concrete() != 0)
            return SymBool(name=f"cast_bool_{value.name}")

        if target_type in type_info:
            bits, signed = type_info[target_type]
            if isinstance(value, SymInt) and value.is_concrete():
                return SymInt(value=value.concrete(), bits=bits, signed=signed)
            return SymInt(bits=bits, signed=signed, name=f"cast_{target_type}_{value.name}")
        return value

    # -- Helpers ------------------------------------------------------------

    def _to_sym_bool(self, value: SymValue) -> SymValue:
        if isinstance(value, SymBool):
            return value
        if isinstance(value, SymInt) and value.is_concrete():
            return SymBool(value=value.concrete() != 0)
        if isinstance(value, SymBinOp) and value.op in SymBinOp.CMP_OPS:
            return value
        # Treat non-zero as true: value != 0
        return SymBinOp("!=", value, SymInt(value=0))

    def _negate_sym(self, cond: SymValue) -> SymValue:
        if isinstance(cond, SymBool):
            return cond.negate()
        if isinstance(cond, SymUnaryOp) and cond.op in ("not", "!"):
            return cond.operand
        if isinstance(cond, SymBinOp) and cond.op in SymBinOp.CMP_OPS:
            neg_map = {"==": "!=", "!=": "==", "<": ">=", ">=": "<", ">": "<=", "<=": ">"}
            return SymBinOp(neg_map[cond.op], cond.left, cond.right)
        return SymUnaryOp("not", cond)

    def _compute_summary(self, name: str, tree: ExecutionTree) -> FunctionSummary:
        """Compute a function summary from the completed execution tree."""
        all_pre: List[SymValue] = []
        all_post: List[SymValue] = []
        all_effects: List[Effect] = []
        all_bugs: List[Bug] = []

        for path in tree.paths:
            # Preconditions: constraints that involve only input variables
            for c in path.constraints:
                refs = _simplifier._collect_refs(c)
                # Heuristic: constraints on original param names are preconditions
                is_pre = all(not r.startswith("sym_") and "ret" not in r for r in refs) if refs else False
                if is_pre:
                    all_pre.append(c)
                else:
                    all_post.append(c)

            all_effects.extend(path.effects)
            all_bugs.extend(path.bugs_found)

        # Deduplicate effects by kind
        seen_effects: Set[str] = set()
        unique_effects: List[Effect] = []
        for e in all_effects:
            key = f"{e.kind}:{e.details}"
            if key not in seen_effects:
                seen_effects.add(key)
                unique_effects.append(e)

        # Deduplicate bugs by kind+description
        seen_bugs: Set[str] = set()
        unique_bugs: List[Bug] = []
        for b in all_bugs:
            key = f"{b.kind}:{b.description}"
            if key not in seen_bugs:
                seen_bugs.add(key)
                unique_bugs.append(b)

        return FunctionSummary(
            name=name,
            preconditions=all_pre,
            postconditions=all_post,
            effects=unique_effects,
            bugs=unique_bugs,
        )
