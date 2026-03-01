"""
Model decoder: converts Z3 models to concrete values and counterexamples.

Given a Z3 model (satisfying assignment), decodes back to concrete values,
generates human-readable counterexamples, formats as test cases, and
validates by concrete execution.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict, Tuple, Any, Union

import z3

from ..ir.types import IRType, IntType, FloatType, PointerType, VoidType, Signedness, FloatKind


# ---------------------------------------------------------------------------
# Concrete values
# ---------------------------------------------------------------------------

class ValueKind(Enum):
    INTEGER = auto()
    FLOAT = auto()
    POINTER = auto()
    BOOLEAN = auto()
    UNKNOWN = auto()


@dataclass
class ConcreteValue:
    """A decoded concrete value from a Z3 model."""
    name: str
    kind: ValueKind
    raw_value: Any
    bit_width: int = 0
    signed: bool = True
    ir_type: Optional[IRType] = None

    @property
    def as_int(self) -> int:
        if isinstance(self.raw_value, int):
            return self.raw_value
        return 0

    @property
    def as_signed_int(self) -> int:
        val = self.as_int
        if self.signed and self.bit_width > 0:
            if val >= (1 << (self.bit_width - 1)):
                val -= (1 << self.bit_width)
        return val

    @property
    def as_unsigned_int(self) -> int:
        val = self.as_int
        if self.bit_width > 0:
            val &= (1 << self.bit_width) - 1
        return val

    @property
    def as_float(self) -> float:
        if isinstance(self.raw_value, float):
            return self.raw_value
        return 0.0

    @property
    def as_hex(self) -> str:
        if self.kind == ValueKind.INTEGER or self.kind == ValueKind.POINTER:
            hex_digits = max(self.bit_width // 4, 1) if self.bit_width else 8
            return f"0x{self.as_unsigned_int:0{hex_digits}x}"
        return str(self.raw_value)

    @property
    def c_literal(self) -> str:
        """Format as a C literal."""
        if self.kind == ValueKind.INTEGER:
            val = self.as_signed_int
            if self.bit_width <= 32:
                return str(val)
            return f"{val}LL"
        if self.kind == ValueKind.FLOAT:
            f = self.as_float
            if self.bit_width == 32:
                return f"{f}f"
            return str(f)
        if self.kind == ValueKind.POINTER:
            return f"(void*){self.as_hex}"
        if self.kind == ValueKind.BOOLEAN:
            return "1" if self.raw_value else "0"
        return str(self.raw_value)

    @property
    def rust_literal(self) -> str:
        """Format as a Rust literal."""
        if self.kind == ValueKind.INTEGER:
            val = self.as_signed_int
            if self.signed:
                return f"{val}i{self.bit_width}"
            return f"{self.as_unsigned_int}u{self.bit_width}"
        if self.kind == ValueKind.FLOAT:
            f = self.as_float
            if self.bit_width == 32:
                return f"{f}f32"
            return f"{f}f64"
        if self.kind == ValueKind.POINTER:
            return f"{self.as_hex} as *const _"
        if self.kind == ValueKind.BOOLEAN:
            return "true" if self.raw_value else "false"
        return str(self.raw_value)

    def __repr__(self) -> str:
        return f"ConcreteValue({self.name}={self.raw_value}, {self.kind.name}, {self.bit_width}b)"


# ---------------------------------------------------------------------------
# Counterexample
# ---------------------------------------------------------------------------

@dataclass
class Counterexample:
    """A counterexample showing divergent behavior."""
    inputs: List[ConcreteValue] = field(default_factory=list)
    c_output: Optional[ConcreteValue] = None
    rust_output: Optional[ConcreteValue] = None
    divergence_description: str = ""
    divergence_class: str = ""
    is_validated: bool = False

    @property
    def has_divergence(self) -> bool:
        if self.c_output is None or self.rust_output is None:
            return False
        return self.c_output.raw_value != self.rust_output.raw_value

    def summary(self) -> str:
        lines = [f"Counterexample ({self.divergence_class}):"]
        lines.append("  Inputs:")
        for inp in self.inputs:
            lines.append(f"    {inp.name} = {inp.raw_value} ({inp.as_hex})")
        if self.c_output:
            lines.append(f"  C output:    {self.c_output.raw_value} ({self.c_output.as_hex})")
        if self.rust_output:
            lines.append(f"  Rust output: {self.rust_output.raw_value} ({self.rust_output.as_hex})")
        if self.divergence_description:
            lines.append(f"  Description: {self.divergence_description}")
        lines.append(f"  Validated: {self.is_validated}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Test case
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    """A test case generated from a counterexample."""
    name: str
    inputs: Dict[str, ConcreteValue] = field(default_factory=dict)
    expected_c_output: Optional[ConcreteValue] = None
    expected_rust_output: Optional[ConcreteValue] = None
    function_name: str = ""
    divergence_class: str = ""

    def to_c_code(self) -> str:
        """Generate C test code."""
        lines = [
            f"// Test: {self.name}",
            f"// Divergence: {self.divergence_class}",
            f"void test_{self.name}(void) {{",
        ]

        # Declare inputs
        for name, val in self.inputs.items():
            c_type = self._c_type(val)
            lines.append(f"    {c_type} {name} = {val.c_literal};")

        # Call function
        args = ", ".join(self.inputs.keys())
        if self.expected_c_output:
            c_ret_type = self._c_type(self.expected_c_output)
            lines.append(f"    {c_ret_type} result = {self.function_name}({args});")
            lines.append(f"    // Expected: {self.expected_c_output.c_literal}")
        else:
            lines.append(f"    {self.function_name}({args});")

        lines.append("}")
        return "\n".join(lines)

    def to_rust_code(self) -> str:
        """Generate Rust test code."""
        lines = [
            f"// Test: {self.name}",
            f"// Divergence: {self.divergence_class}",
            "#[test]",
            f"fn test_{self.name}() {{",
        ]

        for name, val in self.inputs.items():
            rust_type = self._rust_type(val)
            lines.append(f"    let {name}: {rust_type} = {val.rust_literal};")

        args = ", ".join(self.inputs.keys())
        if self.expected_rust_output:
            lines.append(f"    let result = {self.function_name}({args});")
            lines.append(f"    // Expected: {self.expected_rust_output.rust_literal}")
        else:
            lines.append(f"    {self.function_name}({args});")

        lines.append("}")
        return "\n".join(lines)

    def _c_type(self, val: ConcreteValue) -> str:
        if val.kind == ValueKind.INTEGER:
            if val.bit_width <= 8:
                return "int8_t" if val.signed else "uint8_t"
            if val.bit_width <= 16:
                return "int16_t" if val.signed else "uint16_t"
            if val.bit_width <= 32:
                return "int32_t" if val.signed else "uint32_t"
            return "int64_t" if val.signed else "uint64_t"
        if val.kind == ValueKind.FLOAT:
            return "float" if val.bit_width <= 32 else "double"
        if val.kind == ValueKind.POINTER:
            return "void*"
        return "int"

    def _rust_type(self, val: ConcreteValue) -> str:
        if val.kind == ValueKind.INTEGER:
            prefix = "i" if val.signed else "u"
            return f"{prefix}{val.bit_width}"
        if val.kind == ValueKind.FLOAT:
            return "f32" if val.bit_width <= 32 else "f64"
        if val.kind == ValueKind.POINTER:
            return "*const u8"
        return "i32"


# ---------------------------------------------------------------------------
# Model decoder
# ---------------------------------------------------------------------------

class ModelDecoder:
    """
    Decodes Z3 models into concrete values, counterexamples, and test cases.
    """

    def __init__(self, pointer_width: int = 64):
        self.pointer_width = pointer_width

    def decode_model(
        self,
        model: z3.ModelRef,
        variables: Optional[List[z3.ExprRef]] = None,
        type_hints: Optional[Dict[str, IRType]] = None,
    ) -> Dict[str, ConcreteValue]:
        """Decode all variables in a model to concrete values."""
        type_hints = type_hints or {}
        result: Dict[str, ConcreteValue] = {}

        if variables is not None:
            for v in variables:
                name = str(v)
                val = model.evaluate(v, model_completion=True)
                ir_type = type_hints.get(name)
                concrete = self._decode_value(name, val, ir_type)
                result[name] = concrete
        else:
            for decl in model.decls():
                name = decl.name()
                val = model[decl]
                ir_type = type_hints.get(name)
                concrete = self._decode_value(name, val, ir_type)
                result[name] = concrete

        return result

    def _decode_value(
        self,
        name: str,
        z3_val: z3.ExprRef,
        ir_type: Optional[IRType] = None,
    ) -> ConcreteValue:
        """Decode a single Z3 value to a ConcreteValue."""
        if z3.is_bv_value(z3_val):
            return self._decode_bv(name, z3_val, ir_type)
        if z3.is_fp_value(z3_val):
            return self._decode_fp(name, z3_val, ir_type)
        if z3.is_true(z3_val) or z3.is_false(z3_val):
            return ConcreteValue(
                name=name,
                kind=ValueKind.BOOLEAN,
                raw_value=z3.is_true(z3_val),
                bit_width=1,
            )
        if z3.is_int_value(z3_val):
            return ConcreteValue(
                name=name,
                kind=ValueKind.INTEGER,
                raw_value=z3_val.as_long(),
                bit_width=64,
                signed=True,
            )

        # Try to extract value for complex expressions
        try:
            if z3.is_bv(z3_val):
                return self._decode_bv(name, z3_val, ir_type)
            if z3.is_fp(z3_val):
                return self._decode_fp(name, z3_val, ir_type)
        except (z3.Z3Exception, ValueError):
            pass

        return ConcreteValue(
            name=name,
            kind=ValueKind.UNKNOWN,
            raw_value=str(z3_val),
        )

    def _decode_bv(
        self,
        name: str,
        z3_val: z3.ExprRef,
        ir_type: Optional[IRType] = None,
    ) -> ConcreteValue:
        """Decode a bitvector value."""
        try:
            int_val = z3_val.as_long()
        except (AttributeError, z3.Z3Exception):
            return ConcreteValue(name=name, kind=ValueKind.INTEGER, raw_value=0, bit_width=32)

        width = z3_val.size() if z3.is_bv(z3_val) else 32

        # Determine if this is a pointer
        if isinstance(ir_type, PointerType) or width == self.pointer_width and int_val > 0x10000:
            return ConcreteValue(
                name=name,
                kind=ValueKind.POINTER,
                raw_value=int_val,
                bit_width=width,
                signed=False,
                ir_type=ir_type,
            )

        # Determine signedness
        signed = True
        if isinstance(ir_type, IntType):
            signed = ir_type.signed
            width = ir_type.width

        return ConcreteValue(
            name=name,
            kind=ValueKind.INTEGER,
            raw_value=int_val,
            bit_width=width,
            signed=signed,
            ir_type=ir_type,
        )

    def _decode_fp(
        self,
        name: str,
        z3_val: z3.ExprRef,
        ir_type: Optional[IRType] = None,
    ) -> ConcreteValue:
        """Decode a floating-point value."""
        try:
            # Try to extract the float value
            if z3.is_fp_value(z3_val):
                # Check for special values
                if z3_val.isNaN():
                    return ConcreteValue(
                        name=name, kind=ValueKind.FLOAT,
                        raw_value=float('nan'), bit_width=64, ir_type=ir_type,
                    )
                if z3_val.isInf():
                    sign = z3_val.params()[0]
                    val = float('-inf') if sign else float('inf')
                    return ConcreteValue(
                        name=name, kind=ValueKind.FLOAT,
                        raw_value=val, bit_width=64, ir_type=ir_type,
                    )

                # Extract sign, exponent, significand
                sign = z3_val.params()[0]
                exp = z3_val.params()[1]
                sig = z3_val.params()[2]

                # Try string conversion
                fp_str = str(z3_val)
                try:
                    float_val = float(fp_str)
                except ValueError:
                    float_val = 0.0

                width = 64
                if isinstance(ir_type, FloatType) and ir_type.kind == FloatKind.F32:
                    width = 32

                return ConcreteValue(
                    name=name, kind=ValueKind.FLOAT,
                    raw_value=float_val, bit_width=width, ir_type=ir_type,
                )
        except (z3.Z3Exception, ValueError, IndexError):
            pass

        return ConcreteValue(
            name=name, kind=ValueKind.FLOAT,
            raw_value=0.0, bit_width=64, ir_type=ir_type,
        )

    # -- Counterexample generation --

    def extract_counterexample(
        self,
        model: z3.ModelRef,
        input_vars: List[z3.ExprRef],
        c_output_var: Optional[z3.ExprRef] = None,
        rust_output_var: Optional[z3.ExprRef] = None,
        type_hints: Optional[Dict[str, IRType]] = None,
        divergence_class: str = "",
    ) -> Counterexample:
        """Extract a counterexample from a satisfying model."""
        type_hints = type_hints or {}

        inputs: List[ConcreteValue] = []
        for v in input_vars:
            name = str(v)
            val = model.evaluate(v, model_completion=True)
            concrete = self._decode_value(name, val, type_hints.get(name))
            inputs.append(concrete)

        c_output = None
        if c_output_var is not None:
            c_val = model.evaluate(c_output_var, model_completion=True)
            c_output = self._decode_value("c_output", c_val, type_hints.get("c_output"))

        rust_output = None
        if rust_output_var is not None:
            r_val = model.evaluate(rust_output_var, model_completion=True)
            rust_output = self._decode_value("rust_output", r_val, type_hints.get("rust_output"))

        desc = ""
        if c_output and rust_output:
            if c_output.raw_value != rust_output.raw_value:
                desc = (
                    f"C returns {c_output.raw_value} ({c_output.as_hex}), "
                    f"Rust returns {rust_output.raw_value} ({rust_output.as_hex})"
                )

        return Counterexample(
            inputs=inputs,
            c_output=c_output,
            rust_output=rust_output,
            divergence_description=desc,
            divergence_class=divergence_class,
        )

    # -- Test case generation --

    def counterexample_to_test(
        self,
        cex: Counterexample,
        function_name: str,
        test_name: Optional[str] = None,
    ) -> TestCase:
        """Convert a counterexample to a test case."""
        if test_name is None:
            test_name = f"{function_name}_{cex.divergence_class}"

        inputs: Dict[str, ConcreteValue] = {}
        for i, inp in enumerate(cex.inputs):
            name = inp.name if inp.name and inp.name != f"v_{i}" else f"arg_{i}"
            inputs[name] = inp

        return TestCase(
            name=test_name,
            inputs=inputs,
            expected_c_output=cex.c_output,
            expected_rust_output=cex.rust_output,
            function_name=function_name,
            divergence_class=cex.divergence_class,
        )

    # -- Validation --

    def validate_counterexample(
        self,
        cex: Counterexample,
        c_function: Optional[Any] = None,
        rust_function: Optional[Any] = None,
    ) -> bool:
        """
        Validate a counterexample by concrete execution.
        
        If callable functions are provided, execute them with the
        counterexample inputs and check if the outputs diverge.
        """
        if c_function is None or rust_function is None:
            return False

        try:
            # Extract concrete input values
            args = [inp.raw_value for inp in cex.inputs]

            # Execute C function
            c_result = c_function(*args)

            # Execute Rust function
            rust_result = rust_function(*args)

            # Check divergence
            cex.is_validated = True
            if c_result != rust_result:
                cex.c_output = ConcreteValue(
                    name="c_output", kind=ValueKind.INTEGER,
                    raw_value=c_result, bit_width=32,
                )
                cex.rust_output = ConcreteValue(
                    name="rust_output", kind=ValueKind.INTEGER,
                    raw_value=rust_result, bit_width=32,
                )
                return True
            return False
        except Exception:
            return False

    # -- Batch decoding --

    def decode_multiple_models(
        self,
        models: List[z3.ModelRef],
        variables: List[z3.ExprRef],
        type_hints: Optional[Dict[str, IRType]] = None,
    ) -> List[Dict[str, ConcreteValue]]:
        """Decode multiple models."""
        return [self.decode_model(m, variables, type_hints) for m in models]

    # -- Formatting --

    def format_counterexample(self, cex: Counterexample) -> str:
        """Format a counterexample as human-readable text."""
        return cex.summary()

    def format_test_case(self, tc: TestCase, language: str = "both") -> str:
        """Format a test case in the specified language."""
        parts: List[str] = []
        if language in ("c", "both"):
            parts.append("// C Test:")
            parts.append(tc.to_c_code())
        if language in ("rust", "both"):
            parts.append("// Rust Test:")
            parts.append(tc.to_rust_code())
        return "\n\n".join(parts)
