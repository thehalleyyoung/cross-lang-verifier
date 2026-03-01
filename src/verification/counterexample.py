"""
Counterexample generation and validation for the Cross-Language Equivalence Verifier.

Extracts counterexamples from SMT models, converts to concrete inputs,
validates by executing both programs, minimizes, and formats as test cases.

Provides:
- CounterexampleGenerator: extract counterexamples from SMT models
- Counterexample: a concrete counterexample to equivalence
- ConcreteInput: a concrete input value
- CounterexampleValidator: validate counterexamples by execution
- CounterexampleMinimizer: minimize counterexamples
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple

from ..ir.function import Function
from ..ir.basic_block import BasicBlock
from ..ir.instructions import (
    Instruction, Value, Constant, Argument,
    BinaryOp, BinOpKind, UnaryOp,
    CompareOp, CmpPredicate,
    LoadInst, StoreInst, AllocaInst,
    CastInst, CastKind,
    CallInst, ReturnInst, BranchInst,
    PhiInst, SelectInst,
)
from ..ir.types import IRType, IntType, FloatType, Signedness, VoidType

logger = logging.getLogger(__name__)


# ─── Concrete Input ─────────────────────────────────────────────────

@dataclass
class ConcreteInput:
    """A concrete input value for a function parameter."""
    name: str
    ir_type: IRType
    value: Any
    bit_width: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.ir_type, IntType):
            self.bit_width = self.ir_type.width
        elif isinstance(self.ir_type, FloatType):
            self.bit_width = 32 if self.ir_type.kind.name == "F32" else 64

    def as_hex(self) -> str:
        if isinstance(self.value, int):
            return f"0x{self.value & ((1 << self.bit_width) - 1):x}"
        return str(self.value)

    def as_c_literal(self) -> str:
        if isinstance(self.value, int):
            if isinstance(self.ir_type, IntType):
                if self.ir_type.signedness == Signedness.SIGNED:
                    return str(self.value)
                return f"{self.value}u"
        elif isinstance(self.value, float):
            return f"{self.value}"
        return str(self.value)

    def __str__(self) -> str:
        return f"{self.name}: {self.ir_type} = {self.value}"


# ─── Counterexample ────────────────────────────────────────────────

class CounterexampleKind(Enum):
    """Kind of counterexample."""
    RETURN_VALUE_MISMATCH = auto()
    MEMORY_STATE_MISMATCH = auto()
    SIDE_EFFECT_MISMATCH = auto()
    EXCEPTION_MISMATCH = auto()


@dataclass
class ExecutionTrace:
    """Trace of execution for one side of the counterexample."""
    function_name: str
    blocks_visited: List[str] = field(default_factory=list)
    return_value: Optional[Any] = None
    memory_writes: List[Tuple[int, Any]] = field(default_factory=list)
    calls_made: List[Tuple[str, List[Any]]] = field(default_factory=list)
    exception: Optional[str] = None

    def __str__(self) -> str:
        lines = [f"Execution of {self.function_name}:"]
        if self.blocks_visited:
            lines.append(f"  Path: {' → '.join(self.blocks_visited)}")
        if self.return_value is not None:
            lines.append(f"  Return: {self.return_value}")
        if self.memory_writes:
            lines.append(f"  Memory writes: {len(self.memory_writes)}")
        if self.exception:
            lines.append(f"  Exception: {self.exception}")
        return "\n".join(lines)


@dataclass
class Counterexample:
    """A counterexample demonstrating non-equivalence of two functions."""
    inputs: List[ConcreteInput]
    kind: CounterexampleKind
    left_trace: Optional[ExecutionTrace] = None
    right_trace: Optional[ExecutionTrace] = None
    left_output: Optional[Any] = None
    right_output: Optional[Any] = None
    description: str = ""
    is_validated: bool = False
    is_minimized: bool = False
    confidence: float = 1.0

    @property
    def input_dict(self) -> Dict[str, Any]:
        return {inp.name: inp.value for inp in self.inputs}

    def format_as_test(self, language: str = "c") -> str:
        """Format counterexample as a test case."""
        lines = [f"// Counterexample: {self.description}"]
        lines.append(f"// Kind: {self.kind.name}")

        if language == "c":
            return self._format_c_test(lines)
        elif language == "rust":
            return self._format_rust_test(lines)
        return self._format_generic(lines)

    def _format_c_test(self, lines: List[str]) -> str:
        lines.append("void test_counterexample() {")
        for inp in self.inputs:
            c_type = self._ir_type_to_c(inp.ir_type)
            lines.append(f"    {c_type} {inp.name} = {inp.as_c_literal()};")
        lines.append("")
        if self.left_output is not None and self.right_output is not None:
            lines.append(f"    // Left function returns: {self.left_output}")
            lines.append(f"    // Right function returns: {self.right_output}")
        lines.append("}")
        return "\n".join(lines)

    def _format_rust_test(self, lines: List[str]) -> str:
        lines.append("#[test]")
        lines.append("fn test_counterexample() {")
        for inp in self.inputs:
            rust_type = self._ir_type_to_rust(inp.ir_type)
            lines.append(f"    let {inp.name}: {rust_type} = {inp.value};")
        lines.append("}")
        return "\n".join(lines)

    def _format_generic(self, lines: List[str]) -> str:
        lines.append("Inputs:")
        for inp in self.inputs:
            lines.append(f"  {inp}")
        if self.left_output is not None:
            lines.append(f"Left output: {self.left_output}")
        if self.right_output is not None:
            lines.append(f"Right output: {self.right_output}")
        return "\n".join(lines)

    def _ir_type_to_c(self, typ: IRType) -> str:
        if isinstance(typ, IntType):
            widths = {8: "int8_t", 16: "int16_t", 32: "int32_t", 64: "int64_t"}
            if typ.signedness == Signedness.UNSIGNED:
                widths = {8: "uint8_t", 16: "uint16_t", 32: "uint32_t", 64: "uint64_t"}
            return widths.get(typ.width, f"int{typ.width}_t")
        if isinstance(typ, FloatType):
            return "float" if typ.kind.name == "F32" else "double"
        return "void*"

    def _ir_type_to_rust(self, typ: IRType) -> str:
        if isinstance(typ, IntType):
            prefix = "i" if typ.signedness == Signedness.SIGNED else "u"
            return f"{prefix}{typ.width}"
        if isinstance(typ, FloatType):
            return "f32" if typ.kind.name == "F32" else "f64"
        return "&()"

    def __str__(self) -> str:
        lines = [f"Counterexample ({self.kind.name}):"]
        lines.append(f"  {self.description}")
        lines.append("  Inputs:")
        for inp in self.inputs:
            lines.append(f"    {inp}")
        if self.left_output is not None:
            lines.append(f"  Left output: {self.left_output}")
        if self.right_output is not None:
            lines.append(f"  Right output: {self.right_output}")
        if self.is_validated:
            lines.append("  [Validated ✓]")
        if self.is_minimized:
            lines.append("  [Minimized ✓]")
        return "\n".join(lines)


# ─── Counterexample Generator ─────────────────────────────────────

class CounterexampleGenerator:
    """Extract counterexamples from SMT models.

    Given an SMT model that satisfies the negation of equivalence,
    extract concrete input values that demonstrate the non-equivalence.
    """

    def __init__(self) -> None:
        self._generated = 0

    @property
    def num_generated(self) -> int:
        return self._generated

    def from_smt_model(self, model: Any, left: Function, right: Function,
                        input_prefix: str = "shared_input_") -> Optional[Counterexample]:
        """Extract a counterexample from an SMT model."""
        inputs: List[ConcreteInput] = []

        for i, arg in enumerate(left.arguments):
            var_name = f"{input_prefix}{i}"
            value = self._extract_value(model, var_name, arg.ir_type)
            inputs.append(ConcreteInput(
                name=arg.name or f"arg{i}",
                ir_type=arg.ir_type,
                value=value,
            ))

        # Extract outputs
        left_output = self._extract_value(model, "L_return", left.return_type)
        right_output = self._extract_value(model, "R_return", right.return_type)

        cex = Counterexample(
            inputs=inputs,
            kind=CounterexampleKind.RETURN_VALUE_MISMATCH,
            left_output=left_output,
            right_output=right_output,
            description=f"Inputs that cause {left.name} and {right.name} to differ",
        )

        self._generated += 1
        return cex

    def from_concrete_values(self, values: Dict[str, Any],
                              left: Function, right: Function) -> Counterexample:
        """Create a counterexample from concrete input values."""
        inputs: List[ConcreteInput] = []
        for i, arg in enumerate(left.arguments):
            name = arg.name or f"arg{i}"
            value = values.get(name, 0)
            inputs.append(ConcreteInput(
                name=name, ir_type=arg.ir_type, value=value,
            ))

        cex = Counterexample(
            inputs=inputs,
            kind=CounterexampleKind.RETURN_VALUE_MISMATCH,
            description=f"Concrete inputs for {left.name} vs {right.name}",
        )
        self._generated += 1
        return cex

    def _extract_value(self, model: Any, name: str, typ: IRType) -> Any:
        """Extract a value from the SMT model."""
        if model is None:
            return self._default_value(typ)

        try:
            # Try z3 model extraction
            import z3
            if isinstance(model, z3.ModelRef):
                for decl in model.decls():
                    if str(decl) == name:
                        val = model[decl]
                        return self._z3_to_python(val, typ)
        except ImportError:
            pass

        # Try dict-like model
        if isinstance(model, dict):
            return model.get(name, self._default_value(typ))

        return self._default_value(typ)

    def _z3_to_python(self, z3_val: Any, typ: IRType) -> Any:
        """Convert a Z3 value to a Python value."""
        try:
            import z3
            if z3.is_bv_value(z3_val):
                return z3_val.as_long()
            if z3.is_int_value(z3_val):
                return z3_val.as_long()
            if z3.is_true(z3_val):
                return 1
            if z3.is_false(z3_val):
                return 0
            if isinstance(z3_val, z3.FPNumRef):
                return float(z3_val.as_decimal(10))
        except (ImportError, Exception):
            pass
        return 0

    def _default_value(self, typ: IRType) -> Any:
        if isinstance(typ, IntType):
            return 0
        if isinstance(typ, FloatType):
            return 0.0
        return None


# ─── Counterexample Validator ─────────────────────────────────────

class CounterexampleValidator:
    """Validate counterexamples by executing both programs on concrete inputs.

    Performs concrete interpretation of both functions on the
    counterexample inputs to verify the outputs actually differ.
    """

    def __init__(self) -> None:
        self._validated = 0
        self._spurious = 0

    @property
    def num_validated(self) -> int:
        return self._validated

    @property
    def num_spurious(self) -> int:
        return self._spurious

    def validate(self, cex: Counterexample, left: Function,
                  right: Function) -> bool:
        """Validate a counterexample. Returns True if genuine."""
        left_trace = self._execute(left, cex.inputs)
        right_trace = self._execute(right, cex.inputs)

        cex.left_trace = left_trace
        cex.right_trace = right_trace

        if left_trace.exception is not None or right_trace.exception is not None:
            # Exception during execution: might be spurious
            if left_trace.exception != right_trace.exception:
                cex.kind = CounterexampleKind.EXCEPTION_MISMATCH
                cex.is_validated = True
                self._validated += 1
                return True
            self._spurious += 1
            return False

        # Check return values
        if left_trace.return_value != right_trace.return_value:
            cex.left_output = left_trace.return_value
            cex.right_output = right_trace.return_value
            cex.is_validated = True
            self._validated += 1
            return True

        # Check memory state
        if left_trace.memory_writes != right_trace.memory_writes:
            cex.kind = CounterexampleKind.MEMORY_STATE_MISMATCH
            cex.is_validated = True
            self._validated += 1
            return True

        # Counterexample is spurious
        self._spurious += 1
        cex.is_validated = False
        return False

    def _execute(self, func: Function, inputs: List[ConcreteInput]) -> ExecutionTrace:
        """Execute a function on concrete inputs via interpretation."""
        trace = ExecutionTrace(function_name=func.name)

        try:
            env: Dict[int, Any] = {}

            # Bind inputs to arguments
            for inp, arg in zip(inputs, func.arguments):
                env[arg.id] = inp.value

            # Interpret basic blocks
            entry = func.entry_block
            if entry is None:
                return trace

            current_block = entry
            max_steps = 10000
            steps = 0

            while current_block is not None and steps < max_steps:
                steps += 1
                trace.blocks_visited.append(current_block.name)
                next_block = None

                for inst in current_block.instructions:
                    result = self._interpret_instruction(inst, env, trace)
                    if isinstance(inst, ReturnInst):
                        trace.return_value = result
                        return trace
                    elif isinstance(inst, BranchInst):
                        if inst.is_conditional:
                            cond = env.get(inst.condition.id if hasattr(inst.condition, 'id') else id(inst.condition), 0)
                            next_block = inst.true_block if cond else inst.false_block
                        else:
                            next_block = inst.target if hasattr(inst, 'target') else inst.true_block
                        break

                current_block = next_block

        except Exception as e:
            trace.exception = str(e)

        return trace

    def _interpret_instruction(self, inst: Instruction, env: Dict[int, Any],
                                trace: ExecutionTrace) -> Any:
        """Interpret a single instruction."""
        def get_val(v: Value) -> Any:
            if isinstance(v, Constant):
                return v.value if hasattr(v, 'value') else 0
            return env.get(v.id if hasattr(v, 'id') else id(v), 0)

        result = None

        if isinstance(inst, BinaryOp):
            l, r = get_val(inst.left), get_val(inst.right)
            if isinstance(l, (int, bool)) and isinstance(r, (int, bool)):
                l, r = int(l), int(r)
                op_map = {
                    BinOpKind.ADD: lambda: l + r,
                    BinOpKind.SUB: lambda: l - r,
                    BinOpKind.MUL: lambda: l * r,
                    BinOpKind.SDIV: lambda: l // r if r != 0 else 0,
                    BinOpKind.UDIV: lambda: l // r if r != 0 else 0,
                    BinOpKind.SREM: lambda: l % r if r != 0 else 0,
                    BinOpKind.UREM: lambda: l % r if r != 0 else 0,
                    BinOpKind.SHL: lambda: l << r,
                    BinOpKind.LSHR: lambda: l >> r,
                    BinOpKind.ASHR: lambda: l >> r,
                    BinOpKind.AND: lambda: l & r,
                    BinOpKind.OR: lambda: l | r,
                    BinOpKind.XOR: lambda: l ^ r,
                }
                compute = op_map.get(inst.op)
                if compute:
                    try:
                        result = compute()
                    except (ZeroDivisionError, OverflowError):
                        result = 0

        elif isinstance(inst, CompareOp):
            l, r = get_val(inst.left), get_val(inst.right)
            cmp_map = {
                CmpPredicate.EQ: l == r,
                CmpPredicate.NE: l != r,
                CmpPredicate.SLT: l < r,
                CmpPredicate.SLE: l <= r,
                CmpPredicate.SGT: l > r,
                CmpPredicate.SGE: l >= r,
            }
            result = 1 if cmp_map.get(inst.predicate, False) else 0

        elif isinstance(inst, SelectInst):
            cond = get_val(inst.condition)
            result = get_val(inst.true_value) if cond else get_val(inst.false_value)

        elif isinstance(inst, CastInst):
            result = get_val(inst.operand)
            if isinstance(inst.ir_type, IntType):
                mask = (1 << inst.ir_type.width) - 1
                result = int(result) & mask if isinstance(result, (int, float)) else 0

        elif isinstance(inst, PhiInst):
            # In concrete execution, phi should already be resolved
            if inst.incoming:
                result = get_val(inst.incoming[0][0])

        elif isinstance(inst, ReturnInst):
            result = get_val(inst.value) if inst.value is not None else None

        elif isinstance(inst, StoreInst):
            addr = get_val(inst.address)
            val = get_val(inst.value)
            trace.memory_writes.append((addr, val))
            result = None

        elif isinstance(inst, LoadInst):
            result = 0  # Default loaded value

        elif isinstance(inst, CallInst):
            args = [get_val(a) for a in inst.arguments]
            callee_name = getattr(inst, 'callee_name', 'unknown')
            trace.calls_made.append((callee_name, args))
            result = 0

        if result is not None and hasattr(inst, 'id'):
            env[inst.id] = result

        return result


# ─── Counterexample Minimizer ────────────────────────────────────

class CounterexampleMinimizer:
    """Minimize counterexamples by reducing input values.

    Uses delta debugging to find the smallest input values
    that still demonstrate the non-equivalence.
    """

    def __init__(self, max_attempts: int = 100) -> None:
        self._max_attempts = max_attempts
        self._minimized = 0

    @property
    def num_minimized(self) -> int:
        return self._minimized

    def minimize(self, cex: Counterexample, left: Function,
                  right: Function) -> Counterexample:
        """Minimize a counterexample. Returns minimized version."""
        validator = CounterexampleValidator()

        best = cex
        best_size = self._size_metric(cex)

        for attempt in range(self._max_attempts):
            candidate = self._generate_smaller(best)
            if candidate is None:
                break

            candidate_size = self._size_metric(candidate)
            if candidate_size >= best_size:
                continue

            if validator.validate(candidate, left, right):
                best = candidate
                best_size = candidate_size

        best.is_minimized = True
        self._minimized += 1
        return best

    def _size_metric(self, cex: Counterexample) -> int:
        """Compute a size metric for a counterexample."""
        total = 0
        for inp in cex.inputs:
            if isinstance(inp.value, int):
                total += abs(inp.value)
            elif isinstance(inp.value, float):
                total += int(abs(inp.value))
        return total

    def _generate_smaller(self, cex: Counterexample) -> Optional[Counterexample]:
        """Generate a smaller version of the counterexample."""
        if not cex.inputs:
            return None

        new_inputs = list(cex.inputs)
        idx = random.randint(0, len(new_inputs) - 1)
        inp = new_inputs[idx]

        if isinstance(inp.value, int):
            if inp.value == 0:
                return None
            new_val = inp.value // 2
            new_inputs[idx] = ConcreteInput(
                name=inp.name, ir_type=inp.ir_type, value=new_val,
            )
        elif isinstance(inp.value, float):
            if abs(inp.value) < 1e-10:
                return None
            new_val = inp.value / 2.0
            new_inputs[idx] = ConcreteInput(
                name=inp.name, ir_type=inp.ir_type, value=new_val,
            )
        else:
            return None

        return Counterexample(
            inputs=new_inputs,
            kind=cex.kind,
            description=f"Minimized: {cex.description}",
        )
