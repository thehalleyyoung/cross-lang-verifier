"""
Incremental migration engine for gradual C→Rust migration.
Implements migration ordering, FFI wrapper generation, build system integration,
integration testing, rollback support, progress tracking, and risk assessment.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from enum import Enum, auto
import json
import copy
import math
import time


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class MigrationPriority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class MigrationState(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    MIGRATED = "migrated"
    TESTED = "tested"
    VERIFIED = "verified"
    ROLLED_BACK = "rolled_back"


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class OrderingStrategy(Enum):
    LEAF_FIRST = "leaf_first"
    MOST_CALLED = "most_called"
    MOST_BUGGY = "most_buggy"
    SMALLEST_FIRST = "smallest_first"
    DEPENDENCY_ORDER = "dependency_order"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FunctionInfo:
    name: str
    params: List[Dict[str, Any]] = field(default_factory=list)
    return_type: str = "void"
    body_size: int = 0
    calls: List[str] = field(default_factory=list)
    called_by: List[str] = field(default_factory=list)
    bug_count: int = 0
    complexity: int = 1
    state: MigrationState = MigrationState.PENDING
    risk_level: RiskLevel = RiskLevel.MEDIUM
    test_coverage: float = 0.0
    notes: List[str] = field(default_factory=list)

    def is_leaf(self) -> bool:
        return len(self.calls) == 0

    def call_count(self) -> int:
        return len(self.called_by)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "params": self.params,
            "return_type": self.return_type,
            "body_size": self.body_size,
            "calls": self.calls,
            "called_by": self.called_by,
            "bug_count": self.bug_count,
            "complexity": self.complexity,
            "state": self.state.value,
            "risk_level": self.risk_level.value,
            "test_coverage": self.test_coverage,
        }


@dataclass
class NextIncrement:
    functions_to_migrate: List[str] = field(default_factory=list)
    priority: MigrationPriority = MigrationPriority.MEDIUM
    estimated_effort_hours: float = 0.0
    risk_assessment: Dict[str, Any] = field(default_factory=dict)
    ffi_wrappers_needed: List[str] = field(default_factory=list)
    build_changes: List[str] = field(default_factory=list)
    tests_to_create: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "functions_to_migrate": self.functions_to_migrate,
            "priority": self.priority.value,
            "estimated_effort_hours": self.estimated_effort_hours,
            "risk_assessment": self.risk_assessment,
            "ffi_wrappers_needed": self.ffi_wrappers_needed,
            "build_changes": self.build_changes,
            "tests_to_create": self.tests_to_create,
            "dependencies": self.dependencies,
            "rationale": self.rationale,
        }


@dataclass
class FFIWrapper:
    c_function: str
    rust_function: str
    wrapper_code: str = ""
    header_declaration: str = ""
    calling_convention: str = "C"
    params: List[Dict[str, str]] = field(default_factory=list)
    return_type: str = ""


@dataclass
class BuildRule:
    target: str
    rule_type: str = ""
    content: str = ""
    language: str = ""


@dataclass
class IntegrationTest:
    name: str
    c_function: str
    rust_function: str
    test_code: str = ""
    expected_behavior: str = ""


@dataclass
class RollbackInfo:
    function_name: str
    original_state: MigrationState = MigrationState.PENDING
    rolled_back_at: str = ""
    reason: str = ""
    files_to_restore: List[str] = field(default_factory=list)


@dataclass
class ProgressReport:
    total_functions: int = 0
    migrated_functions: int = 0
    migrated_loc: int = 0
    total_loc: int = 0
    test_coverage: float = 0.0
    functions_by_state: Dict[str, int] = field(default_factory=dict)
    migration_percentage_by_loc: float = 0.0
    migration_percentage_by_function: float = 0.0
    migration_percentage_by_test: float = 0.0
    estimated_remaining_hours: float = 0.0
    risk_profile: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_functions": self.total_functions,
            "migrated_functions": self.migrated_functions,
            "migrated_loc": self.migrated_loc,
            "total_loc": self.total_loc,
            "migration_by_loc": f"{self.migration_percentage_by_loc:.1f}%",
            "migration_by_function": f"{self.migration_percentage_by_function:.1f}%",
            "migration_by_test": f"{self.migration_percentage_by_test:.1f}%",
            "test_coverage": f"{self.test_coverage:.1f}%",
            "estimated_remaining_hours": self.estimated_remaining_hours,
            "functions_by_state": self.functions_by_state,
            "risk_profile": self.risk_profile,
        }


@dataclass
class RiskAssessment:
    function_name: str
    risk_level: RiskLevel = RiskLevel.MEDIUM
    risk_score: float = 0.5
    factors: List[str] = field(default_factory=list)
    mitigations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "function_name": self.function_name,
            "risk_level": self.risk_level.value,
            "risk_score": self.risk_score,
            "factors": self.factors,
            "mitigations": self.mitigations,
        }


# ---------------------------------------------------------------------------
# Type mapping
# ---------------------------------------------------------------------------

C_TO_RUST_TYPE: Dict[str, str] = {
    "int": "c_int", "unsigned int": "c_uint",
    "short": "c_short", "unsigned short": "c_ushort",
    "long": "c_long", "unsigned long": "c_ulong",
    "long long": "c_longlong", "unsigned long long": "c_ulonglong",
    "char": "c_char", "unsigned char": "c_uchar",
    "float": "c_float", "double": "c_double",
    "void": "()", "void*": "*mut c_void",
    "size_t": "usize", "ssize_t": "isize",
    "bool": "bool", "_Bool": "bool",
    "int8_t": "i8", "uint8_t": "u8",
    "int16_t": "i16", "uint16_t": "u16",
    "int32_t": "i32", "uint32_t": "u32",
    "int64_t": "i64", "uint64_t": "u64",
}


# ---------------------------------------------------------------------------
# Incremental Migrator
# ---------------------------------------------------------------------------

class IncrementalMigrator:
    """Support gradual C→Rust migration with ordering, FFI, and tracking."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._functions: Dict[str, FunctionInfo] = {}
        self._call_graph: Dict[str, Set[str]] = {}
        self._reverse_call_graph: Dict[str, Set[str]] = {}
        self._rollback_history: List[RollbackInfo] = []
        self._migration_log: List[Dict[str, Any]] = []

    def _reset(self) -> None:
        self._functions = {}
        self._call_graph = {}
        self._reverse_call_graph = {}
        self._rollback_history = []
        self._migration_log = []

    # --- project analysis ---
    def _analyze_project(self, c_project: Dict[str, Any]) -> None:
        functions = c_project.get("functions", [])
        for func in functions:
            name = func.get("name", "")
            calls = func.get("calls", [])
            body = func.get("body", [])
            body_size = len(body) if isinstance(body, list) else 1

            info = FunctionInfo(
                name=name,
                params=func.get("params", []),
                return_type=func.get("return_type", "void"),
                body_size=body_size,
                calls=calls,
                bug_count=func.get("bug_count", 0),
                complexity=func.get("complexity", 1),
                test_coverage=func.get("test_coverage", 0.0),
            )

            self._functions[name] = info
            self._call_graph[name] = set(calls)

        for name, calls in self._call_graph.items():
            for callee in calls:
                if callee not in self._reverse_call_graph:
                    self._reverse_call_graph[callee] = set()
                self._reverse_call_graph[callee].add(name)

        for name, info in self._functions.items():
            info.called_by = list(self._reverse_call_graph.get(name, set()))

    def _update_migrated(self, migrated_so_far: List[str]) -> None:
        for name in migrated_so_far:
            if name in self._functions:
                self._functions[name].state = MigrationState.MIGRATED

    # --- ordering strategies ---
    def _order_leaf_first(self) -> List[str]:
        pending = [n for n, f in self._functions.items()
                   if f.state == MigrationState.PENDING]

        ordered = []
        remaining = set(pending)

        while remaining:
            leaves = []
            for name in remaining:
                calls = self._call_graph.get(name, set())
                unmigrated_deps = calls & remaining
                if not unmigrated_deps:
                    leaves.append(name)

            if not leaves:
                leaves = [min(remaining, key=lambda n: self._functions[n].complexity)]

            leaves.sort(key=lambda n: self._functions[n].complexity)
            ordered.extend(leaves)
            remaining -= set(leaves)

        return ordered

    def _order_most_called(self) -> List[str]:
        pending = [n for n, f in self._functions.items()
                   if f.state == MigrationState.PENDING]
        return sorted(pending,
                      key=lambda n: self._functions[n].call_count(),
                      reverse=True)

    def _order_most_buggy(self) -> List[str]:
        pending = [n for n, f in self._functions.items()
                   if f.state == MigrationState.PENDING]
        return sorted(pending,
                      key=lambda n: self._functions[n].bug_count,
                      reverse=True)

    def _order_smallest_first(self) -> List[str]:
        pending = [n for n, f in self._functions.items()
                   if f.state == MigrationState.PENDING]
        return sorted(pending,
                      key=lambda n: self._functions[n].body_size)

    def _get_ordering(self, strategy: OrderingStrategy) -> List[str]:
        strategies = {
            OrderingStrategy.LEAF_FIRST: self._order_leaf_first,
            OrderingStrategy.MOST_CALLED: self._order_most_called,
            OrderingStrategy.MOST_BUGGY: self._order_most_buggy,
            OrderingStrategy.SMALLEST_FIRST: self._order_smallest_first,
            OrderingStrategy.DEPENDENCY_ORDER: self._order_leaf_first,
        }
        return strategies.get(strategy, self._order_leaf_first)()

    # --- FFI wrapper generation ---
    def _generate_ffi_wrapper(self, func_info: FunctionInfo) -> FFIWrapper:
        params = []
        param_strs_c = []
        param_strs_rust = []
        call_args = []

        for p in func_info.params:
            pname = p.get("name", "arg")
            ptype = p.get("type", "int")
            rtype = C_TO_RUST_TYPE.get(ptype, "c_int")

            if ptype.endswith("*"):
                base = ptype[:-1].strip()
                rtype = f"*mut {C_TO_RUST_TYPE.get(base, 'c_void')}"

            params.append({"name": pname, "c_type": ptype, "rust_type": rtype})
            param_strs_c.append(f"{ptype} {pname}")
            param_strs_rust.append(f"{pname}: {rtype}")
            call_args.append(pname)

        ret_type = func_info.return_type
        rust_ret = C_TO_RUST_TYPE.get(ret_type, "c_int")

        rust_wrapper = (
            f'#[no_mangle]\n'
            f'pub extern "C" fn {func_info.name}({", ".join(param_strs_rust)}) -> {rust_ret} {{\n'
            f'    // Rust implementation\n'
            f'    rust_{func_info.name}({", ".join(call_args)})\n'
            f'}}\n'
        )

        c_header = f'{ret_type} {func_info.name}({", ".join(param_strs_c)});'

        return FFIWrapper(
            c_function=func_info.name,
            rust_function=f"rust_{func_info.name}",
            wrapper_code=rust_wrapper,
            header_declaration=c_header,
            params=params,
            return_type=rust_ret,
        )

    def _generate_all_ffi_wrappers(self, functions: List[str]) -> List[FFIWrapper]:
        wrappers = []
        for name in functions:
            info = self._functions.get(name)
            if info:
                wrappers.append(self._generate_ffi_wrapper(info))
        return wrappers

    # --- build system integration ---
    def _generate_cmake_rules(self, migrated: List[str],
                              remaining_c: List[str]) -> str:
        lines = [
            "# Auto-generated CMake rules for mixed C/Rust build",
            "cmake_minimum_required(VERSION 3.15)",
            "project(mixed_c_rust)",
            "",
            "# C sources (not yet migrated)",
        ]

        c_sources = [f"{name}.c" for name in remaining_c]
        lines.append(f'set(C_SOURCES {" ".join(c_sources)})')
        lines.append("")

        lines.append("# Rust library (migrated functions)")
        lines.append("add_custom_command(")
        lines.append("    OUTPUT ${CMAKE_BINARY_DIR}/librust_impl.a")
        lines.append("    COMMAND cargo build --release --manifest-path ${CMAKE_SOURCE_DIR}/rust_impl/Cargo.toml")
        lines.append("    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/rust_impl")
        lines.append(")")
        lines.append("")

        lines.append("add_custom_target(rust_lib DEPENDS ${CMAKE_BINARY_DIR}/librust_impl.a)")
        lines.append("")

        lines.append("# Main executable")
        lines.append("add_executable(main ${C_SOURCES})")
        lines.append("add_dependencies(main rust_lib)")
        lines.append("target_link_libraries(main ${CMAKE_BINARY_DIR}/librust_impl.a)")
        lines.append("")

        lines.append("# Integration tests")
        lines.append("enable_testing()")
        lines.append('add_test(NAME integration_test COMMAND ${CMAKE_BINARY_DIR}/main --test)')
        lines.append("")

        return "\n".join(lines)

    def _generate_makefile_rules(self, migrated: List[str],
                                 remaining_c: List[str]) -> str:
        lines = [
            "# Auto-generated Makefile for mixed C/Rust build",
            "CC = gcc",
            "CFLAGS = -Wall -Wextra -O2",
            "RUST_LIB = rust_impl/target/release/librust_impl.a",
            "",
            "C_SOURCES = " + " ".join(f"{n}.c" for n in remaining_c),
            "C_OBJECTS = $(C_SOURCES:.c=.o)",
            "",
            "all: main",
            "",
            "$(RUST_LIB):",
            "\tcd rust_impl && cargo build --release",
            "",
            "%.o: %.c",
            "\t$(CC) $(CFLAGS) -c $< -o $@",
            "",
            "main: $(C_OBJECTS) $(RUST_LIB)",
            "\t$(CC) $(CFLAGS) $^ -o $@ -lpthread -ldl",
            "",
            "test: main",
            "\t./main --test",
            "",
            "clean:",
            "\trm -f $(C_OBJECTS) main",
            "\tcd rust_impl && cargo clean",
            "",
        ]
        return "\n".join(lines)

    def _generate_cargo_toml(self, migrated: List[str]) -> str:
        lines = [
            '[package]',
            'name = "rust_impl"',
            'version = "0.1.0"',
            'edition = "2021"',
            '',
            '[lib]',
            'crate-type = ["staticlib", "cdylib"]',
            '',
            '[dependencies]',
            'libc = "0.2"',
            '',
            f'# Migrated functions: {", ".join(migrated)}',
        ]
        return "\n".join(lines)

    # --- integration tests ---
    def _generate_integration_tests(self, functions: List[str]) -> List[IntegrationTest]:
        tests = []
        for name in functions:
            info = self._functions.get(name)
            if not info:
                continue

            param_inits = []
            call_args = []
            for p in info.params:
                pname = p.get("name", "arg")
                ptype = p.get("type", "int")
                if ptype == "int":
                    param_inits.append(f"    {ptype} {pname} = 42;")
                elif ptype.endswith("*"):
                    param_inits.append(f"    {ptype} {pname} = NULL;")
                else:
                    param_inits.append(f"    {ptype} {pname} = 0;")
                call_args.append(pname)

            test_code = (
                f'void test_{name}_integration(void) {{\n'
                f'{"".join(p + chr(10) for p in param_inits)}'
                f'    {info.return_type} c_result = c_{name}({", ".join(call_args)});\n'
                f'    {info.return_type} rust_result = {name}({", ".join(call_args)});\n'
                f'    assert(c_result == rust_result);\n'
                f'    printf("PASS: {name} integration test\\n");\n'
                f'}}\n'
            )

            tests.append(IntegrationTest(
                name=f"test_{name}_integration",
                c_function=f"c_{name}",
                rust_function=name,
                test_code=test_code,
                expected_behavior=f"C and Rust versions of {name} produce same output",
            ))

        return tests

    # --- rollback ---
    def rollback(self, function_name: str, reason: str = "") -> RollbackInfo:
        info = self._functions.get(function_name)
        if not info:
            return RollbackInfo(
                function_name=function_name,
                reason=f"Function {function_name} not found",
            )

        original = info.state
        info.state = MigrationState.ROLLED_BACK

        rollback = RollbackInfo(
            function_name=function_name,
            original_state=original,
            rolled_back_at=str(time.time()),
            reason=reason or "Manual rollback",
            files_to_restore=[
                f"{function_name}.c",
                f"rust_impl/src/{function_name}.rs",
                f"ffi/{function_name}_wrapper.rs",
            ],
        )

        self._rollback_history.append(rollback)
        self._migration_log.append({
            "action": "rollback",
            "function": function_name,
            "reason": reason,
            "timestamp": rollback.rolled_back_at,
        })

        return rollback

    def rollback_increment(self, functions: List[str],
                           reason: str = "") -> List[RollbackInfo]:
        return [self.rollback(f, reason) for f in functions]

    # --- progress tracking ---
    def get_progress(self) -> ProgressReport:
        report = ProgressReport()
        report.total_functions = len(self._functions)

        state_counts: Dict[str, int] = {}
        migrated_loc = 0
        total_loc = 0
        tested_count = 0
        total_coverage = 0.0

        for info in self._functions.values():
            state = info.state.value
            state_counts[state] = state_counts.get(state, 0) + 1
            total_loc += info.body_size

            if info.state in (MigrationState.MIGRATED, MigrationState.TESTED,
                              MigrationState.VERIFIED):
                migrated_loc += info.body_size
                report.migrated_functions += 1

            total_coverage += info.test_coverage

        report.migrated_loc = migrated_loc
        report.total_loc = total_loc
        report.functions_by_state = state_counts
        report.test_coverage = (total_coverage / len(self._functions)
                                if self._functions else 0.0)

        report.migration_percentage_by_loc = (
            (migrated_loc / total_loc * 100) if total_loc > 0 else 0.0
        )
        report.migration_percentage_by_function = (
            (report.migrated_functions / report.total_functions * 100)
            if report.total_functions > 0 else 0.0
        )

        pending = [f for f in self._functions.values()
                   if f.state == MigrationState.PENDING]
        hours_per_loc = self.config.get("hours_per_loc", 0.1)
        report.estimated_remaining_hours = sum(
            f.body_size * hours_per_loc * (1 + f.complexity * 0.2)
            for f in pending
        )

        risk_counts: Dict[str, int] = {}
        for f in pending:
            rl = f.risk_level.value
            risk_counts[rl] = risk_counts.get(rl, 0) + 1
        report.risk_profile = risk_counts

        return report

    # --- risk assessment ---
    def _assess_risk(self, func_name: str) -> RiskAssessment:
        info = self._functions.get(func_name)
        if not info:
            return RiskAssessment(function_name=func_name)

        risk_score = 0.0
        factors = []
        mitigations = []

        if info.complexity > 10:
            risk_score += 0.3
            factors.append(f"High complexity: {info.complexity}")
            mitigations.append("Break into smaller functions before migrating")
        elif info.complexity > 5:
            risk_score += 0.15
            factors.append(f"Moderate complexity: {info.complexity}")

        if info.bug_count > 0:
            risk_score += min(0.3, info.bug_count * 0.1)
            factors.append(f"Known bugs: {info.bug_count}")
            mitigations.append("Fix known bugs before or during migration")

        if info.call_count() > 5:
            risk_score += 0.15
            factors.append(f"Widely called: {info.call_count()} callers")
            mitigations.append("Ensure comprehensive integration tests")

        if len(info.calls) > 5:
            risk_score += 0.1
            factors.append(f"Many dependencies: {len(info.calls)} callees")
            mitigations.append("Migrate callees first or create FFI wrappers")

        unmigrated_deps = [c for c in info.calls
                           if c in self._functions and
                           self._functions[c].state == MigrationState.PENDING]
        if unmigrated_deps:
            risk_score += len(unmigrated_deps) * 0.05
            factors.append(f"Unmigrated dependencies: {unmigrated_deps}")
            mitigations.append("Use FFI wrappers for unmigrated dependencies")

        if info.body_size > 100:
            risk_score += 0.15
            factors.append(f"Large function: {info.body_size} LOC")
            mitigations.append("Consider decomposing before migration")

        if info.test_coverage < 0.5:
            risk_score += 0.2
            factors.append(f"Low test coverage: {info.test_coverage:.0%}")
            mitigations.append("Write more tests before migrating")

        has_pointers = any("*" in p.get("type", "") for p in info.params)
        if has_pointers:
            risk_score += 0.15
            factors.append("Uses pointer parameters")
            mitigations.append("Pay special attention to null safety and ownership")

        risk_score = min(1.0, risk_score)

        if risk_score > 0.7:
            risk_level = RiskLevel.CRITICAL
        elif risk_score > 0.5:
            risk_level = RiskLevel.HIGH
        elif risk_score > 0.3:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW

        info.risk_level = risk_level

        return RiskAssessment(
            function_name=func_name,
            risk_level=risk_level,
            risk_score=risk_score,
            factors=factors,
            mitigations=mitigations,
        )

    # --- main entry ---
    def plan_increment(self, c_project: Dict[str, Any],
                       migrated_so_far: Optional[List[str]] = None,
                       strategy: OrderingStrategy = OrderingStrategy.LEAF_FIRST,
                       batch_size: int = 5) -> NextIncrement:
        self._reset()
        self._analyze_project(c_project)

        if migrated_so_far:
            self._update_migrated(migrated_so_far)

        ordering = self._get_ordering(strategy)
        batch = ordering[:batch_size]

        if not batch:
            return NextIncrement(
                rationale="All functions have been migrated!",
                risk_assessment={"overall_risk": "none"},
            )

        risk_assessments = {name: self._assess_risk(name) for name in batch}
        overall_risk = sum(ra.risk_score for ra in risk_assessments.values()) / len(batch)

        deps = set()
        for name in batch:
            info = self._functions.get(name)
            if info:
                for call in info.calls:
                    if call not in batch and call in self._functions:
                        if self._functions[call].state == MigrationState.PENDING:
                            deps.add(call)

        ffi_needed = list(deps)

        effort = sum(
            self._functions[n].body_size * 0.1 * (1 + self._functions[n].complexity * 0.2)
            for n in batch if n in self._functions
        )

        migrated = migrated_so_far or []
        remaining_c = [n for n in self._functions if n not in migrated and n not in batch]

        increment = NextIncrement(
            functions_to_migrate=batch,
            priority=MigrationPriority.HIGH if overall_risk > 0.5 else MigrationPriority.MEDIUM,
            estimated_effort_hours=effort,
            risk_assessment={
                "overall_risk_score": overall_risk,
                "per_function": {n: ra.to_dict() for n, ra in risk_assessments.items()},
            },
            ffi_wrappers_needed=ffi_needed,
            build_changes=[
                "Update CMakeLists.txt to include new Rust library",
                "Add FFI wrapper source files",
                "Update linker flags",
            ],
            tests_to_create=[f"test_{n}_integration" for n in batch],
            dependencies=ffi_needed,
            rationale=f"Selected {len(batch)} functions using {strategy.value} strategy. "
                      f"Overall risk: {overall_risk:.2f}. "
                      f"Estimated effort: {effort:.1f} hours.",
        )

        return increment

    def generate_increment_artifacts(self, increment: NextIncrement,
                                     c_project: Dict[str, Any],
                                     migrated_so_far: Optional[List[str]] = None) -> Dict[str, Any]:
        if not self._functions:
            self._analyze_project(c_project)
            if migrated_so_far:
                self._update_migrated(migrated_so_far)

        ffi_wrappers = self._generate_all_ffi_wrappers(increment.functions_to_migrate)
        integration_tests = self._generate_integration_tests(increment.functions_to_migrate)

        migrated = (migrated_so_far or []) + increment.functions_to_migrate
        remaining = [n for n in self._functions if n not in migrated]

        cmake = self._generate_cmake_rules(migrated, remaining)
        makefile = self._generate_makefile_rules(migrated, remaining)
        cargo_toml = self._generate_cargo_toml(migrated)

        return {
            "ffi_wrappers": [{"function": w.c_function,
                              "wrapper_code": w.wrapper_code,
                              "header": w.header_declaration}
                             for w in ffi_wrappers],
            "integration_tests": [{"name": t.name,
                                   "test_code": t.test_code}
                                  for t in integration_tests],
            "cmake": cmake,
            "makefile": makefile,
            "cargo_toml": cargo_toml,
        }

    def mark_migrated(self, function_name: str) -> None:
        info = self._functions.get(function_name)
        if info:
            info.state = MigrationState.MIGRATED
            self._migration_log.append({
                "action": "migrate",
                "function": function_name,
                "timestamp": str(time.time()),
            })

    def mark_tested(self, function_name: str) -> None:
        info = self._functions.get(function_name)
        if info:
            info.state = MigrationState.TESTED

    def mark_verified(self, function_name: str) -> None:
        info = self._functions.get(function_name)
        if info:
            info.state = MigrationState.VERIFIED

    def get_migration_log(self) -> List[Dict[str, Any]]:
        return list(self._migration_log)
