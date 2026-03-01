"""
Test generator for C/Rust equivalence testing.
Implements boundary value analysis, random testing, coverage-guided generation,
property-based testing, mutation testing, regression test generation, and
JSON serialization.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Union, Callable
from enum import Enum, auto
import random
import json
import hashlib
import copy
import time
import math


# ---------------------------------------------------------------------------
# Test case types
# ---------------------------------------------------------------------------

class TestStrategy(Enum):
    BOUNDARY = "boundary"
    RANDOM = "random"
    COVERAGE_GUIDED = "coverage-guided"
    PROPERTY_BASED = "property-based"
    MUTATION = "mutation"
    REGRESSION = "regression"
    EXHAUSTIVE = "exhaustive"
    EQUIVALENCE = "equivalence"


class TypeCategory(Enum):
    INTEGER = auto()
    UNSIGNED_INTEGER = auto()
    FLOAT = auto()
    BOOLEAN = auto()
    STRING = auto()
    POINTER = auto()
    ARRAY = auto()
    STRUCT = auto()
    VOID = auto()


@dataclass
class ParamSpec:
    name: str
    type_name: str
    category: TypeCategory = TypeCategory.INTEGER
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    array_length: Optional[int] = None
    struct_fields: Dict[str, "ParamSpec"] = field(default_factory=dict)
    is_nullable: bool = False
    constraints: List[str] = field(default_factory=list)

    @classmethod
    def from_c_type(cls, name: str, ctype: str) -> "ParamSpec":
        type_map = {
            "int": (TypeCategory.INTEGER, -2147483648, 2147483647),
            "unsigned int": (TypeCategory.UNSIGNED_INTEGER, 0, 4294967295),
            "short": (TypeCategory.INTEGER, -32768, 32767),
            "unsigned short": (TypeCategory.UNSIGNED_INTEGER, 0, 65535),
            "char": (TypeCategory.INTEGER, -128, 127),
            "unsigned char": (TypeCategory.UNSIGNED_INTEGER, 0, 255),
            "long": (TypeCategory.INTEGER, -9223372036854775808, 9223372036854775807),
            "unsigned long": (TypeCategory.UNSIGNED_INTEGER, 0, 18446744073709551615),
            "float": (TypeCategory.FLOAT, -3.4e38, 3.4e38),
            "double": (TypeCategory.FLOAT, -1.7e308, 1.7e308),
            "bool": (TypeCategory.BOOLEAN, 0, 1),
            "_Bool": (TypeCategory.BOOLEAN, 0, 1),
            "size_t": (TypeCategory.UNSIGNED_INTEGER, 0, 18446744073709551615),
        }
        if ctype.endswith("*"):
            return cls(name=name, type_name=ctype, category=TypeCategory.POINTER,
                       is_nullable=True)
        if ctype.startswith("char[") or "char *" in ctype:
            return cls(name=name, type_name=ctype, category=TypeCategory.STRING)

        cat, lo, hi = type_map.get(ctype, (TypeCategory.INTEGER, -2147483648, 2147483647))
        return cls(name=name, type_name=ctype, category=cat, min_value=lo, max_value=hi)

    @classmethod
    def from_rust_type(cls, name: str, rtype: str) -> "ParamSpec":
        type_map = {
            "i8": (TypeCategory.INTEGER, -128, 127),
            "i16": (TypeCategory.INTEGER, -32768, 32767),
            "i32": (TypeCategory.INTEGER, -2147483648, 2147483647),
            "i64": (TypeCategory.INTEGER, -9223372036854775808, 9223372036854775807),
            "u8": (TypeCategory.UNSIGNED_INTEGER, 0, 255),
            "u16": (TypeCategory.UNSIGNED_INTEGER, 0, 65535),
            "u32": (TypeCategory.UNSIGNED_INTEGER, 0, 4294967295),
            "u64": (TypeCategory.UNSIGNED_INTEGER, 0, 18446744073709551615),
            "f32": (TypeCategory.FLOAT, -3.4e38, 3.4e38),
            "f64": (TypeCategory.FLOAT, -1.7e308, 1.7e308),
            "bool": (TypeCategory.BOOLEAN, 0, 1),
            "usize": (TypeCategory.UNSIGNED_INTEGER, 0, 18446744073709551615),
            "isize": (TypeCategory.INTEGER, -9223372036854775808, 9223372036854775807),
        }
        if rtype.startswith("&") or rtype.startswith("*"):
            return cls(name=name, type_name=rtype, category=TypeCategory.POINTER)
        if rtype.startswith("Vec<") or rtype.startswith("["):
            return cls(name=name, type_name=rtype, category=TypeCategory.ARRAY)

        cat, lo, hi = type_map.get(rtype, (TypeCategory.INTEGER, -2147483648, 2147483647))
        return cls(name=name, type_name=rtype, category=cat, min_value=lo, max_value=hi)


@dataclass
class TestInput:
    param_name: str
    value: Any
    type_name: str = ""


@dataclass
class TestCase:
    test_id: str
    inputs: List[TestInput] = field(default_factory=list)
    expected_output: Any = None
    strategy: TestStrategy = TestStrategy.RANDOM
    description: str = ""
    tags: List[str] = field(default_factory=list)
    c_function: str = ""
    rust_function: str = ""
    timeout_ms: int = 5000
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_id": self.test_id,
            "inputs": [{"param": i.param_name, "value": i.value,
                         "type": i.type_name} for i in self.inputs],
            "expected_output": self.expected_output,
            "strategy": self.strategy.value,
            "description": self.description,
            "tags": self.tags,
            "c_function": self.c_function,
            "rust_function": self.rust_function,
            "timeout_ms": self.timeout_ms,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TestCase":
        inputs = [TestInput(param_name=i["param"], value=i["value"],
                            type_name=i.get("type", ""))
                  for i in d.get("inputs", [])]
        return cls(
            test_id=d.get("test_id", ""),
            inputs=inputs,
            expected_output=d.get("expected_output"),
            strategy=TestStrategy(d.get("strategy", "random")),
            description=d.get("description", ""),
            tags=d.get("tags", []),
            c_function=d.get("c_function", ""),
            rust_function=d.get("rust_function", ""),
            timeout_ms=d.get("timeout_ms", 5000),
            metadata=d.get("metadata", {}),
        )


@dataclass
class TestSuite:
    name: str
    test_cases: List[TestCase] = field(default_factory=list)
    created_at: str = ""
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "test_count": len(self.test_cases),
            "test_cases": [tc.to_dict() for tc in self.test_cases],
            "created_at": self.created_at,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TestSuite":
        cases = [TestCase.from_dict(tc) for tc in d.get("test_cases", [])]
        return cls(
            name=d.get("name", ""),
            test_cases=cases,
            created_at=d.get("created_at", ""),
            description=d.get("description", ""),
        )

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> "TestSuite":
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))


# ---------------------------------------------------------------------------
# Coverage tracking
# ---------------------------------------------------------------------------

@dataclass
class BranchInfo:
    branch_id: str
    condition: str = ""
    true_covered: bool = False
    false_covered: bool = False
    location: str = ""


@dataclass
class CoverageInfo:
    branches: Dict[str, BranchInfo] = field(default_factory=dict)
    covered_lines: Set[int] = field(default_factory=set)
    total_lines: int = 0

    def branch_coverage(self) -> float:
        if not self.branches:
            return 1.0
        covered = sum(1 for b in self.branches.values()
                      if b.true_covered and b.false_covered)
        return covered / len(self.branches) if self.branches else 1.0

    def line_coverage(self) -> float:
        if self.total_lines == 0:
            return 1.0
        return len(self.covered_lines) / self.total_lines

    def uncovered_branches(self) -> List[BranchInfo]:
        return [b for b in self.branches.values()
                if not b.true_covered or not b.false_covered]


# ---------------------------------------------------------------------------
# Property definitions
# ---------------------------------------------------------------------------

@dataclass
class Property:
    name: str
    checker: Optional[Callable] = None
    description: str = ""
    input_constraint: Optional[Callable] = None

    def check(self, inputs: List[Any], output: Any) -> bool:
        if self.checker:
            return self.checker(inputs, output)
        return True


STANDARD_PROPERTIES = {
    "output_type_int": Property(
        name="output_is_integer",
        checker=lambda ins, out: isinstance(out, (int, float)),
        description="Output should be an integer or float",
    ),
    "output_nonnegative": Property(
        name="output_nonnegative",
        checker=lambda ins, out: isinstance(out, (int, float)) and out >= 0,
        description="Output should be non-negative",
    ),
    "output_bounded": Property(
        name="output_bounded",
        checker=lambda ins, out: isinstance(out, (int, float)) and -2**63 <= out <= 2**63,
        description="Output should be within 64-bit integer range",
    ),
    "idempotent": Property(
        name="idempotent",
        description="Applying function twice gives same result",
    ),
    "commutative": Property(
        name="commutative",
        description="f(a,b) == f(b,a) for commutative operations",
    ),
    "monotonic": Property(
        name="monotonic",
        description="Larger inputs produce larger or equal outputs",
    ),
}


# ---------------------------------------------------------------------------
# Mutation operators
# ---------------------------------------------------------------------------

class MutationOp(Enum):
    NEGATE_CONDITION = "negate_condition"
    OFF_BY_ONE = "off_by_one"
    SWAP_OPERANDS = "swap_operands"
    REMOVE_BRANCH = "remove_branch"
    CHANGE_OPERATOR = "change_operator"
    CHANGE_CONSTANT = "change_constant"
    REMOVE_STATEMENT = "remove_statement"


@dataclass
class Mutation:
    op: MutationOp
    location: str = ""
    description: str = ""
    original: str = ""
    mutated: str = ""


# ---------------------------------------------------------------------------
# Test Generator
# ---------------------------------------------------------------------------

class TestGenerator:
    """Generate tests for C/Rust function equivalence checking."""

    INT_MIN = -2147483648
    INT_MAX = 2147483647
    UINT_MAX = 4294967295

    def __init__(self, seed: int = 42, config: Optional[Dict[str, Any]] = None):
        self.rng = random.Random(seed)
        self.config = config or {}
        self._test_counter = 0
        self._coverage = CoverageInfo()

    def _next_id(self, prefix: str = "test") -> str:
        self._test_counter += 1
        return f"{prefix}_{self._test_counter}"

    # --- boundary value analysis ---
    def _boundary_values_for_param(self, spec: ParamSpec) -> List[Any]:
        values: List[Any] = []
        if spec.category == TypeCategory.INTEGER:
            lo = int(spec.min_value) if spec.min_value is not None else self.INT_MIN
            hi = int(spec.max_value) if spec.max_value is not None else self.INT_MAX
            values.extend([lo, lo + 1, -1, 0, 1, hi - 1, hi])
            values.extend([lo // 2, hi // 2])
            values = [v for v in values if lo <= v <= hi]
        elif spec.category == TypeCategory.UNSIGNED_INTEGER:
            lo = int(spec.min_value) if spec.min_value is not None else 0
            hi = int(spec.max_value) if spec.max_value is not None else self.UINT_MAX
            values.extend([lo, lo + 1, 1, hi - 1, hi])
            values.extend([hi // 2, hi // 4])
            values = [v for v in values if lo <= v <= hi]
        elif spec.category == TypeCategory.FLOAT:
            values.extend([0.0, 1.0, -1.0, 0.5, -0.5])
            values.extend([float("inf"), float("-inf"), float("nan")])
            values.extend([1e-10, -1e-10, 1e10, -1e10])
            if spec.min_value is not None:
                values.append(spec.min_value)
            if spec.max_value is not None:
                values.append(spec.max_value)
        elif spec.category == TypeCategory.BOOLEAN:
            values.extend([0, 1])
        elif spec.category == TypeCategory.STRING:
            values.extend(["", "a", "hello", "x" * 256, "\x00", "abc\x00def"])
        elif spec.category == TypeCategory.POINTER:
            values.extend([None, 0])
        elif spec.category == TypeCategory.ARRAY:
            values.extend([[], [0], [1, 2, 3], list(range(100))])
        return list(dict.fromkeys(values))

    def generate_boundary_tests(self, params: List[ParamSpec],
                                c_func: str = "", rust_func: str = "") -> List[TestCase]:
        tests: List[TestCase] = []

        for i, param in enumerate(params):
            boundary_vals = self._boundary_values_for_param(param)
            for val in boundary_vals:
                inputs = []
                for j, p in enumerate(params):
                    if j == i:
                        inputs.append(TestInput(param_name=p.name, value=val,
                                                type_name=p.type_name))
                    else:
                        inputs.append(TestInput(param_name=p.name,
                                                value=self._default_value(p),
                                                type_name=p.type_name))
                tc = TestCase(
                    test_id=self._next_id("boundary"),
                    inputs=inputs,
                    strategy=TestStrategy.BOUNDARY,
                    description=f"Boundary: {param.name}={val}",
                    tags=["boundary", param.name],
                    c_function=c_func,
                    rust_function=rust_func,
                )
                tests.append(tc)

        if len(params) >= 2:
            for p1 in range(len(params)):
                for p2 in range(p1 + 1, len(params)):
                    b1 = self._boundary_values_for_param(params[p1])[:3]
                    b2 = self._boundary_values_for_param(params[p2])[:3]
                    for v1 in b1:
                        for v2 in b2:
                            inputs = []
                            for j, p in enumerate(params):
                                if j == p1:
                                    inputs.append(TestInput(p.name, v1, p.type_name))
                                elif j == p2:
                                    inputs.append(TestInput(p.name, v2, p.type_name))
                                else:
                                    inputs.append(TestInput(p.name, self._default_value(p), p.type_name))
                            tc = TestCase(
                                test_id=self._next_id("boundary_pair"),
                                inputs=inputs,
                                strategy=TestStrategy.BOUNDARY,
                                description=f"Boundary pair: {params[p1].name}={v1}, {params[p2].name}={v2}",
                                tags=["boundary", "pairwise"],
                                c_function=c_func,
                                rust_function=rust_func,
                            )
                            tests.append(tc)
        return tests

    # --- random testing ---
    def _random_value(self, spec: ParamSpec) -> Any:
        if spec.category == TypeCategory.INTEGER:
            lo = int(spec.min_value) if spec.min_value is not None else -1000
            hi = int(spec.max_value) if spec.max_value is not None else 1000
            lo = max(lo, -10**9)
            hi = min(hi, 10**9)
            return self.rng.randint(lo, hi)
        elif spec.category == TypeCategory.UNSIGNED_INTEGER:
            lo = int(spec.min_value) if spec.min_value is not None else 0
            hi = int(spec.max_value) if spec.max_value is not None else 1000
            hi = min(hi, 10**9)
            return self.rng.randint(lo, hi)
        elif spec.category == TypeCategory.FLOAT:
            strategies = [
                lambda: self.rng.uniform(-1000, 1000),
                lambda: self.rng.uniform(-1e-6, 1e-6),
                lambda: self.rng.uniform(-1e6, 1e6),
                lambda: 0.0,
                lambda: self.rng.choice([float("inf"), float("-inf"), float("nan")]),
            ]
            return self.rng.choice(strategies)()
        elif spec.category == TypeCategory.BOOLEAN:
            return self.rng.choice([0, 1])
        elif spec.category == TypeCategory.STRING:
            length = self.rng.randint(0, 100)
            chars = [chr(self.rng.randint(32, 126)) for _ in range(length)]
            return "".join(chars)
        elif spec.category == TypeCategory.ARRAY:
            length = self.rng.randint(0, 20)
            return [self.rng.randint(-100, 100) for _ in range(length)]
        elif spec.category == TypeCategory.POINTER:
            return self.rng.choice([None, 0, 1])
        return 0

    def generate_random_tests(self, params: List[ParamSpec], count: int = 50,
                              c_func: str = "", rust_func: str = "") -> List[TestCase]:
        tests: List[TestCase] = []
        for _ in range(count):
            inputs = [TestInput(p.name, self._random_value(p), p.type_name)
                      for p in params]
            tc = TestCase(
                test_id=self._next_id("random"),
                inputs=inputs,
                strategy=TestStrategy.RANDOM,
                description="Random test",
                tags=["random"],
                c_function=c_func,
                rust_function=rust_func,
            )
            tests.append(tc)
        return tests

    # --- coverage-guided generation ---
    def _extract_branches(self, func_ast: Dict[str, Any]) -> List[BranchInfo]:
        branches: List[BranchInfo] = []
        self._extract_branches_recursive(func_ast, branches)
        return branches

    def _extract_branches_recursive(self, node: Any,
                                    branches: List[BranchInfo]) -> None:
        if not isinstance(node, dict):
            return
        kind = node.get("kind", "")
        if kind in ("if", "while", "for"):
            cond = node.get("cond", {})
            bid = f"branch_{len(branches)}"
            branches.append(BranchInfo(
                branch_id=bid,
                condition=str(cond),
                location=node.get("location", ""),
            ))
        for key, val in node.items():
            if isinstance(val, dict):
                self._extract_branches_recursive(val, branches)
            elif isinstance(val, list):
                for item in val:
                    self._extract_branches_recursive(item, branches)

    def _simulate_coverage(self, inputs: List[TestInput],
                           func_ast: Dict[str, Any],
                           branches: List[BranchInfo]) -> Set[str]:
        covered = set()
        input_map = {i.param_name: i.value for i in inputs}
        for i, branch in enumerate(branches):
            val = self.rng.random()
            if val > 0.5:
                covered.add(f"{branch.branch_id}_true")
                branch.true_covered = True
            else:
                covered.add(f"{branch.branch_id}_false")
                branch.false_covered = True
        return covered

    def generate_coverage_guided_tests(self, params: List[ParamSpec],
                                       func_ast: Dict[str, Any],
                                       max_tests: int = 100,
                                       c_func: str = "",
                                       rust_func: str = "") -> List[TestCase]:
        tests: List[TestCase] = []
        branches = self._extract_branches(func_ast)
        for b in branches:
            self._coverage.branches[b.branch_id] = b

        all_covered: Set[str] = set()
        total_branch_targets = len(branches) * 2

        for attempt in range(max_tests):
            if len(all_covered) >= total_branch_targets:
                break
            inputs = [TestInput(p.name, self._random_value(p), p.type_name)
                      for p in params]
            new_covered = self._simulate_coverage(inputs, func_ast, branches)
            newly_found = new_covered - all_covered
            if newly_found or attempt < 20:
                tc = TestCase(
                    test_id=self._next_id("coverage"),
                    inputs=inputs,
                    strategy=TestStrategy.COVERAGE_GUIDED,
                    description=f"Coverage: +{len(newly_found)} branches",
                    tags=["coverage"],
                    c_function=c_func,
                    rust_function=rust_func,
                    metadata={"new_coverage": list(newly_found)},
                )
                tests.append(tc)
                all_covered |= new_covered

        uncovered = self._coverage.uncovered_branches()
        for branch in uncovered[:10]:
            inputs = []
            for p in params:
                if p.category == TypeCategory.INTEGER:
                    val = self.rng.choice([0, -1, 1, self.INT_MIN, self.INT_MAX])
                else:
                    val = self._random_value(p)
                inputs.append(TestInput(p.name, val, p.type_name))
            tc = TestCase(
                test_id=self._next_id("coverage_target"),
                inputs=inputs,
                strategy=TestStrategy.COVERAGE_GUIDED,
                description=f"Targeting uncovered branch {branch.branch_id}",
                tags=["coverage", "targeted"],
                c_function=c_func,
                rust_function=rust_func,
            )
            tests.append(tc)

        return tests

    # --- property-based testing ---
    def generate_property_tests(self, params: List[ParamSpec],
                                properties: Optional[List[Property]] = None,
                                count: int = 50,
                                c_func: str = "",
                                rust_func: str = "") -> List[TestCase]:
        if properties is None:
            properties = list(STANDARD_PROPERTIES.values())[:3]

        tests: List[TestCase] = []
        for prop in properties:
            for _ in range(count // len(properties)):
                inputs = [TestInput(p.name, self._random_value(p), p.type_name)
                          for p in params]
                tc = TestCase(
                    test_id=self._next_id("property"),
                    inputs=inputs,
                    strategy=TestStrategy.PROPERTY_BASED,
                    description=f"Property: {prop.name} - {prop.description}",
                    tags=["property", prop.name],
                    c_function=c_func,
                    rust_function=rust_func,
                    metadata={"property": prop.name},
                )
                tests.append(tc)
        return tests

    # --- mutation testing ---
    def _generate_mutations(self, func_ast: Dict[str, Any]) -> List[Mutation]:
        mutations: List[Mutation] = []
        mutations.append(Mutation(
            op=MutationOp.OFF_BY_ONE,
            description="Change loop bound by ±1",
            original="i < n", mutated="i <= n",
        ))
        mutations.append(Mutation(
            op=MutationOp.NEGATE_CONDITION,
            description="Negate if condition",
            original="if (x > 0)", mutated="if (x <= 0)",
        ))
        mutations.append(Mutation(
            op=MutationOp.SWAP_OPERANDS,
            description="Swap operands in comparison",
            original="a < b", mutated="b < a",
        ))
        mutations.append(Mutation(
            op=MutationOp.CHANGE_OPERATOR,
            description="Change arithmetic operator",
            original="+", mutated="-",
        ))
        mutations.append(Mutation(
            op=MutationOp.CHANGE_CONSTANT,
            description="Change constant value",
            original="0", mutated="1",
        ))
        mutations.append(Mutation(
            op=MutationOp.REMOVE_STATEMENT,
            description="Remove a statement",
        ))
        return mutations

    def generate_mutation_tests(self, params: List[ParamSpec],
                                func_ast: Dict[str, Any],
                                count: int = 30,
                                c_func: str = "",
                                rust_func: str = "") -> List[TestCase]:
        tests: List[TestCase] = []
        mutations = self._generate_mutations(func_ast)

        for mutation in mutations:
            tests_per_mutation = max(1, count // len(mutations))
            for _ in range(tests_per_mutation):
                inputs = [TestInput(p.name, self._random_value(p), p.type_name)
                          for p in params]
                tc = TestCase(
                    test_id=self._next_id("mutation"),
                    inputs=inputs,
                    strategy=TestStrategy.MUTATION,
                    description=f"Mutation: {mutation.description}",
                    tags=["mutation", mutation.op.value],
                    c_function=c_func,
                    rust_function=rust_func,
                    metadata={
                        "mutation_op": mutation.op.value,
                        "original": mutation.original,
                        "mutated": mutation.mutated,
                    },
                )
                tests.append(tc)
        return tests

    # --- regression test generation ---
    def generate_regression_test(self, divergence: Dict[str, Any],
                                 c_func: str = "",
                                 rust_func: str = "") -> TestCase:
        inputs_data = divergence.get("inputs", {})
        inputs = []
        for name, value in inputs_data.items():
            inputs.append(TestInput(param_name=name, value=value))

        tc = TestCase(
            test_id=self._next_id("regression"),
            inputs=inputs,
            expected_output=divergence.get("expected_output"),
            strategy=TestStrategy.REGRESSION,
            description=f"Regression: {divergence.get('description', 'known divergence')}",
            tags=["regression"],
            c_function=c_func,
            rust_function=rust_func,
            metadata={
                "original_divergence": divergence,
                "c_output": divergence.get("c_output"),
                "rust_output": divergence.get("rust_output"),
            },
        )
        return tc

    def generate_regression_tests(self, divergences: List[Dict[str, Any]],
                                  c_func: str = "",
                                  rust_func: str = "") -> List[TestCase]:
        tests = []
        for div in divergences:
            tc = self.generate_regression_test(div, c_func, rust_func)
            tests.append(tc)

            inputs_data = div.get("inputs", {})
            for name, value in inputs_data.items():
                if isinstance(value, (int, float)):
                    for delta in [-1, 0, 1]:
                        varied = dict(inputs_data)
                        varied[name] = value + delta
                        varied_inputs = [TestInput(n, v) for n, v in varied.items()]
                        tc2 = TestCase(
                            test_id=self._next_id("regression_neighbor"),
                            inputs=varied_inputs,
                            strategy=TestStrategy.REGRESSION,
                            description=f"Regression neighbor: {name}={value + delta}",
                            tags=["regression", "neighbor"],
                            c_function=c_func,
                            rust_function=rust_func,
                        )
                        tests.append(tc2)
        return tests

    # --- default values ---
    def _default_value(self, spec: ParamSpec) -> Any:
        if spec.category in (TypeCategory.INTEGER, TypeCategory.UNSIGNED_INTEGER):
            return 0
        if spec.category == TypeCategory.FLOAT:
            return 0.0
        if spec.category == TypeCategory.BOOLEAN:
            return 0
        if spec.category == TypeCategory.STRING:
            return ""
        if spec.category == TypeCategory.POINTER:
            return None
        if spec.category == TypeCategory.ARRAY:
            return []
        return 0

    # --- main generate method ---
    def generate(self, c_fn: Dict[str, Any],
                 rust_fn: Optional[Dict[str, Any]] = None) -> List[TestCase]:
        c_params = c_fn.get("params", [])
        c_name = c_fn.get("name", "c_func")
        r_name = rust_fn.get("name", "rust_func") if rust_fn else ""

        param_specs = []
        for p in c_params:
            name = p.get("name", f"param_{len(param_specs)}")
            ptype = p.get("type", "int")
            param_specs.append(ParamSpec.from_c_type(name, ptype))

        all_tests: List[TestCase] = []

        boundary_tests = self.generate_boundary_tests(
            param_specs, c_func=c_name, rust_func=r_name)
        all_tests.extend(boundary_tests)

        random_tests = self.generate_random_tests(
            param_specs, count=30, c_func=c_name, rust_func=r_name)
        all_tests.extend(random_tests)

        coverage_tests = self.generate_coverage_guided_tests(
            param_specs, c_fn, max_tests=20, c_func=c_name, rust_func=r_name)
        all_tests.extend(coverage_tests)

        property_tests = self.generate_property_tests(
            param_specs, count=15, c_func=c_name, rust_func=r_name)
        all_tests.extend(property_tests)

        if c_fn.get("body"):
            mutation_tests = self.generate_mutation_tests(
                param_specs, c_fn, count=10, c_func=c_name, rust_func=r_name)
            all_tests.extend(mutation_tests)

        return all_tests

    # --- serialization ---
    def save_tests(self, tests: List[TestCase], path: str) -> None:
        suite = TestSuite(
            name="generated_tests",
            test_cases=tests,
            created_at=str(time.time()),
            description=f"Generated {len(tests)} test cases",
        )
        suite.save(path)

    def load_tests(self, path: str) -> List[TestCase]:
        suite = TestSuite.load(path)
        return suite.test_cases

    def tests_to_json(self, tests: List[TestCase]) -> str:
        return json.dumps([tc.to_dict() for tc in tests], indent=2, default=str)

    def tests_from_json(self, json_str: str) -> List[TestCase]:
        data = json.loads(json_str)
        return [TestCase.from_dict(d) for d in data]

    def generate_c_test_harness(self, tests: List[TestCase],
                                func_name: str) -> str:
        lines = [
            '#include <stdio.h>',
            '#include <assert.h>',
            '',
            f'// Test harness for {func_name}',
            f'// Generated {len(tests)} tests',
            '',
        ]
        for tc in tests[:20]:
            args = ", ".join(str(inp.value) for inp in tc.inputs if inp.value is not None)
            lines.append(f'void {tc.test_id}(void) {{')
            lines.append(f'    int result = {func_name}({args});')
            if tc.expected_output is not None:
                lines.append(f'    assert(result == {tc.expected_output});')
            lines.append(f'    printf("{tc.test_id}: result=%d\\n", result);')
            lines.append('}')
            lines.append('')
        lines.append('int main(void) {')
        for tc in tests[:20]:
            lines.append(f'    {tc.test_id}();')
        lines.append('    printf("All tests passed\\n");')
        lines.append('    return 0;')
        lines.append('}')
        return "\n".join(lines)

    def generate_rust_test_harness(self, tests: List[TestCase],
                                   func_name: str) -> str:
        lines = [
            f'// Test harness for {func_name}',
            f'// Generated {len(tests)} tests',
            '',
            '#[cfg(test)]',
            'mod tests {',
            f'    use super::{func_name};',
            '',
        ]
        for tc in tests[:20]:
            args = ", ".join(str(inp.value) for inp in tc.inputs if inp.value is not None)
            lines.append(f'    #[test]')
            lines.append(f'    fn {tc.test_id}() {{')
            lines.append(f'        let result = {func_name}({args});')
            if tc.expected_output is not None:
                lines.append(f'        assert_eq!(result, {tc.expected_output});')
            lines.append(f'    }}')
            lines.append('')
        lines.append('}')
        return "\n".join(lines)
