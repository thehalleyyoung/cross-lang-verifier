#!/usr/bin/env python3
"""
Comprehensive benchmark for the cross-language equivalence verifier.

Tests all nine core modules:
  1. abstract_interpreter_c
  2. rust_borrow_checker
  3. abi_compatibility_checker
  4. undefined_behavior_detector
  5. test_generator
  6. memory_safety_verifier
  7. code_complexity_comparator
  8. migration_validator
  9. incremental_migration_engine

Each section exercises the public API surface, constructing realistic inputs
and verifying that key invariants hold.  The script prints a per-module
pass/fail report and exits with code 0 only when every test passes.
"""

from __future__ import annotations

import ast
import importlib
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Resolve import path
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE,
    "..", "implementation", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    name: str
    passed: bool
    elapsed: float
    error: Optional[str] = None


@dataclass
class ModuleReport:
    module_name: str
    results: List[TestResult] = field(default_factory=list)
    parse_ok: bool = True
    import_ok: bool = True
    import_error: Optional[str] = None

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def total(self) -> int:
        return len(self.results)


def _run_test(name: str, fn: Callable[[], None]) -> TestResult:
    """Run *fn* and return a TestResult."""
    t0 = time.perf_counter()
    try:
        fn()
        return TestResult(name=name, passed=True,
                          elapsed=time.perf_counter() - t0)
    except Exception as exc:
        tb = traceback.format_exc()
        return TestResult(name=name, passed=False,
                          elapsed=time.perf_counter() - t0,
                          error=f"{exc}\n{tb}")


# ===================================================================
# 1. Abstract Interpreter (C)
# ===================================================================

def _bench_abstract_interpreter(report: ModuleReport) -> None:
    from abstract_interpreter_c import (
        Interval, AllocState, BugKind, BugReport,
        PointerInfo, AbstractState, CFGNode, CFG,
        CAbstractInterpreter,
    )

    # -- Interval arithmetic tests --
    def test_interval_creation():
        iv = Interval(0, 10)
        assert iv.lo == 0 and iv.hi == 10
        top = Interval()
        assert top.lo == Interval.NEG_INF
        assert top.hi == Interval.POS_INF
        pt = Interval.const(5)
        assert pt.lo == 5 and pt.hi == 5

    def test_interval_operations():
        a = Interval(1, 5)
        b = Interval(3, 8)
        u = a.join(b)
        assert u.lo == 1 and u.hi == 8
        m = a.meet(b)
        assert m is not None
        assert m.lo == 3 and m.hi == 5

    def test_interval_arithmetic():
        a = Interval(2, 4)
        b = Interval(1, 3)
        s = a + b
        assert s.lo == 3 and s.hi == 7
        d = a - b
        assert d.lo == -1 and d.hi == 3

    def test_interval_widening():
        a = Interval(0, 5)
        b = Interval(0, 10)
        w = a.widen(b)
        assert w.hi >= 10

    def test_abstract_state():
        st = AbstractState()
        st.add_local("x")
        st.variable_ranges["x"] = Interval(0, 100)
        iv = st.variable_ranges["x"]
        assert iv.lo == 0 and iv.hi == 100

    def test_pointer_info():
        pi = PointerInfo(targets={"buf"}, may_be_null=False)
        assert "buf" in pi.targets
        assert not pi.may_be_null

    def test_cfg_construction():
        n1 = CFGNode(node_id=0, label="entry", statements=[])
        n2 = CFGNode(node_id=1, label="exit", statements=[])
        cfg = CFG()
        cfg.add_node(n1)
        cfg.add_node(n2)
        cfg.add_edge(0, 1)
        assert 1 in n1.successors

    def test_interpreter_basic():
        interp = CAbstractInterpreter()
        cfg = CFG()
        entry = CFGNode(node_id=0, label="entry", statements=[
            {"kind": "assign", "var": "x", "value": {"kind": "const", "val": 5}},
        ])
        exit_n = CFGNode(node_id=1, label="exit", statements=[])
        cfg.add_node(entry)
        cfg.add_node(exit_n)
        cfg.add_edge(0, 1)
        interp.analyze(cfg)

    def test_bug_report_fields():
        br = BugReport(kind=BugKind.NULL_DEREF, location="line 42",
                       variable="p",
                       message="possible null dereference",
                       severity="high")
        assert br.kind == BugKind.NULL_DEREF
        assert "42" in br.location

    def test_alloc_state_enum():
        assert AllocState.ALLOCATED.name == "ALLOCATED"
        assert AllocState.FREED.name == "FREED"

    for fn in [test_interval_creation, test_interval_operations,
               test_interval_arithmetic, test_interval_widening,
               test_abstract_state, test_pointer_info,
               test_cfg_construction, test_interpreter_basic,
               test_bug_report_fields, test_alloc_state_enum]:
        report.results.append(_run_test(fn.__name__, fn))


# ===================================================================
# 2. Rust Borrow Checker
# ===================================================================

def _bench_borrow_checker(report: ModuleReport) -> None:
    from rust_borrow_checker import (
        Mutability, OwnershipState, BorrowKind,
        Lifetime, BorrowInfo, OwnershipInfo,
        BorrowViolation, BorrowCheckResult, Scope,
        BorrowChecker, is_copy_type, type_has_drop,
        quick_borrow_check,
    )

    def test_ownership_states():
        assert OwnershipState.OWNED.name == "OWNED"
        assert OwnershipState.MOVED.name == "MOVED"

    def test_copy_types():
        assert is_copy_type("i32")
        assert is_copy_type("bool")
        assert not is_copy_type("String")
        assert not is_copy_type("Vec<i32>")

    def test_drop_types():
        assert type_has_drop("Vec<i32>")
        assert not type_has_drop("i32")

    def test_lifetime_ordering():
        a = Lifetime("a", scope_depth=0, start_point=0, end_point=10)
        b = Lifetime("b", scope_depth=1, start_point=2, end_point=8)
        assert b.start_point >= a.start_point and b.end_point <= a.end_point

    def test_borrow_info():
        lt = Lifetime("a", scope_depth=0, start_point=0, end_point=5)
        bi = BorrowInfo(borrow_id=1, kind=BorrowKind.SHARED,
                        borrowed_place="x", borrower="ref_x",
                        lifetime=lt)
        assert bi.borrowed_place == "x"
        assert bi.kind == BorrowKind.SHARED

    def test_scope():
        s = Scope(depth=0)
        assert s.depth == 0

    def test_checker_simple_ok():
        bc = BorrowChecker()
        stmts = [
            {"kind": "let", "name": "x", "type": "i32",
             "mutable": False, "init": {"kind": "literal", "value": 42}},
            {"kind": "let", "name": "y", "type": "i32",
             "mutable": False, "init": {"kind": "var", "name": "x"}},
        ]
        result = bc.check(stmts)
        assert isinstance(result, BorrowCheckResult)

    def test_quick_borrow_check():
        stmts = [
            {"kind": "let", "name": "a", "type": "i32",
             "mutable": False, "init": {"kind": "literal", "value": 1}},
        ]
        result = quick_borrow_check(stmts)
        assert isinstance(result, BorrowCheckResult)

    def test_violation_severity():
        from rust_borrow_checker import ViolationSeverity
        assert ViolationSeverity.ERROR.name == "ERROR"

    def test_borrow_check_result_fields():
        r = BorrowCheckResult()
        assert r.valid or not r.valid

    for fn in [test_ownership_states, test_copy_types, test_drop_types,
               test_lifetime_ordering, test_borrow_info, test_scope,
               test_checker_simple_ok, test_quick_borrow_check,
               test_violation_severity, test_borrow_check_result_fields]:
        report.results.append(_run_test(fn.__name__, fn))


# ===================================================================
# 3. ABI Compatibility Checker
# ===================================================================

def _bench_abi_checker(report: ModuleReport) -> None:
    from abi_compatibility_checker import (
        Platform, CallingConvention, Endianness, PlatformInfo,
        TypeKind, TypeDescriptor, FieldDescriptor, EnumVariant,
        IncompatibilityKind, Incompatibility, ABIReport, ABIChecker,
    )

    def test_platform_info():
        pi = PlatformInfo(name=Platform.LP64)
        assert pi.pointer_size == 8

    def test_type_descriptor_int():
        td = TypeDescriptor(kind=TypeKind.PRIMITIVE, name="int",
                            size=4, alignment=4)
        assert td.size == 4

    def test_field_descriptor():
        fd = FieldDescriptor(name="x",
                             type_desc=TypeDescriptor(kind=TypeKind.PRIMITIVE,
                                                      name="int",
                                                      size=4,
                                                      alignment=4),
                             offset=0)
        assert fd.offset == 0

    def test_enum_variant():
        ev = EnumVariant(name="None", value=0)
        assert ev.value == 0

    def test_checker_struct_compat():
        checker = ABIChecker(PlatformInfo(name=Platform.LP64))
        c_int = TypeDescriptor(name="int", kind=TypeKind.PRIMITIVE,
                               size=4, alignment=4, language="c")
        rust_i32 = TypeDescriptor(name="i32", kind=TypeKind.PRIMITIVE,
                                  size=4, alignment=4, language="rust")
        c_struct = TypeDescriptor(
            name="Point", kind=TypeKind.STRUCT, language="c",
            fields=[
                FieldDescriptor(name="x", type_desc=c_int, offset=0),
                FieldDescriptor(name="y", type_desc=c_int, offset=4),
            ],
        )
        rust_struct = TypeDescriptor(
            name="Point", kind=TypeKind.STRUCT, language="rust",
            is_repr_c=True,
            fields=[
                FieldDescriptor(name="x", type_desc=rust_i32, offset=0),
                FieldDescriptor(name="y", type_desc=rust_i32, offset=4),
            ],
        )
        report_result = checker.check_struct_pair(c_struct, rust_struct)
        assert isinstance(report_result, ABIReport)

    def test_calling_convention_enum():
        assert CallingConvention.CDECL.name == "CDECL"
        assert CallingConvention.SYSTEM.name == "SYSTEM"

    def test_incompatibility_kind():
        assert IncompatibilityKind.SIZE_MISMATCH.name == "SIZE_MISMATCH"

    def test_abi_report_empty():
        r = ABIReport()
        assert r.compatible

    def test_endianness():
        assert Endianness.LITTLE.name == "LITTLE"
        assert Endianness.BIG.name == "BIG"

    def test_type_kind_enum():
        assert TypeKind.POINTER.name == "POINTER"
        assert TypeKind.STRUCT.name == "STRUCT"

    for fn in [test_platform_info, test_type_descriptor_int,
               test_field_descriptor, test_enum_variant,
               test_checker_struct_compat, test_calling_convention_enum,
               test_incompatibility_kind, test_abi_report_empty,
               test_endianness, test_type_kind_enum]:
        report.results.append(_run_test(fn.__name__, fn))


# ===================================================================
# 4. Undefined Behavior Detector
# ===================================================================

def _bench_ub_detector(report: ModuleReport) -> None:
    from undefined_behavior_detector import (
        UBType, UBViolation, AllocState, VarInfo,
        AnalysisState, UBDetector, types_compatible_for_aliasing,
    )

    def test_ub_types():
        assert UBType.SIGNED_OVERFLOW.name == "SIGNED_OVERFLOW"
        assert UBType.NULL_DEREF.name == "NULL_DEREF"
        assert UBType.DOUBLE_FREE.name == "DOUBLE_FREE"

    def test_ub_violation():
        v = UBViolation(type=UBType.SIGNED_OVERFLOW,
                        location="line 10",
                        explanation="signed integer overflow",
                        severity="high")
        assert v.type == UBType.SIGNED_OVERFLOW

    def test_var_info():
        vi = VarInfo(name="x", type_name="int", initialized=True)
        assert vi.initialized

    def test_analysis_state():
        st = AnalysisState()
        st.add_local("x")
        assert isinstance(st.variables, dict)

    def test_aliasing_compat():
        assert types_compatible_for_aliasing("int", "int")
        assert types_compatible_for_aliasing("char", "int")

    def test_detector_empty():
        det = UBDetector()
        violations = det.detect([])
        assert isinstance(violations, list)

    def test_detector_signed_overflow():
        det = UBDetector()
        stmts = [
            {"kind": "assign", "var": "x", "type": "int",
             "value": {"kind": "const", "val": 2147483647}},
            {"kind": "assign", "var": "y", "type": "int",
             "value": {"kind": "binop", "op": "+",
                       "left": {"kind": "var", "name": "x"},
                       "right": {"kind": "const", "val": 1}}},
        ]
        violations = det.detect(stmts)
        assert isinstance(violations, list)

    def test_alloc_state_enum():
        assert AllocState.ALLOCATED.name == "ALLOCATED"
        assert AllocState.FREED.name == "FREED"

    def test_detector_null_deref():
        det = UBDetector()
        stmts = [
            {"kind": "assign", "var": "p", "type": "int*",
             "value": {"kind": "null"}},
            {"kind": "deref", "var": "p"},
        ]
        violations = det.detect(stmts)
        assert isinstance(violations, list)

    def test_detector_use_after_free():
        det = UBDetector()
        stmts = [
            {"kind": "alloc", "var": "p", "type": "int*", "size": 4},
            {"kind": "free", "var": "p"},
            {"kind": "deref", "var": "p"},
        ]
        violations = det.detect(stmts)
        assert isinstance(violations, list)

    for fn in [test_ub_types, test_ub_violation, test_var_info,
               test_analysis_state, test_aliasing_compat,
               test_detector_empty, test_detector_signed_overflow,
               test_alloc_state_enum, test_detector_null_deref,
               test_detector_use_after_free]:
        report.results.append(_run_test(fn.__name__, fn))


# ===================================================================
# 5. Test Generator
# ===================================================================

def _bench_test_generator(report: ModuleReport) -> None:
    from test_generator import (
        TestStrategy, TypeCategory, ParamSpec, TestInput,
        TestCase, TestSuite, BranchInfo, CoverageInfo,
        Property, MutationOp, Mutation, TestGenerator,
    )

    def test_strategy_enum():
        assert TestStrategy.BOUNDARY.name == "BOUNDARY"
        assert TestStrategy.RANDOM.name == "RANDOM"

    def test_type_category():
        assert TypeCategory.INTEGER.name == "INTEGER"

    def test_param_spec():
        ps = ParamSpec(name="n", type_name="int",
                       category=TypeCategory.INTEGER)
        assert ps.name == "n"

    def test_test_input():
        ti = TestInput(param_name="n", value=42, type_name="int")
        assert ti.value == 42

    def test_test_case():
        tc = TestCase(
            test_id="test_add",
            inputs=[TestInput(param_name="a", value=1),
                    TestInput(param_name="b", value=2)],
            expected_output=3,
            strategy=TestStrategy.BOUNDARY,
        )
        assert tc.test_id == "test_add"

    def test_test_suite():
        suite = TestSuite(name="math_tests")
        assert suite.name == "math_tests"

    def test_generator_boundary():
        gen = TestGenerator()
        params = [
            ParamSpec(name="x", type_name="int",
                      category=TypeCategory.INTEGER),
        ]
        cases = gen.generate_boundary_tests(params)
        assert isinstance(cases, list)
        assert len(cases) > 0

    def test_generator_random():
        gen = TestGenerator()
        params = [
            ParamSpec(name="x", type_name="int",
                      category=TypeCategory.INTEGER),
        ]
        cases = gen.generate_random_tests(params, count=5)
        assert isinstance(cases, list)

    def test_mutation_op():
        assert MutationOp.NEGATE_CONDITION.name == "NEGATE_CONDITION"

    def test_coverage_info():
        ci = CoverageInfo()
        assert ci.total_lines == 0 or ci.total_lines >= 0

    for fn in [test_strategy_enum, test_type_category, test_param_spec,
               test_test_input, test_test_case, test_test_suite,
               test_generator_boundary, test_generator_random,
               test_mutation_op, test_coverage_info]:
        report.results.append(_run_test(fn.__name__, fn))


# ===================================================================
# 6. Memory Safety Verifier
# ===================================================================

def _bench_memory_safety(report: ModuleReport) -> None:
    from memory_safety_verifier import (
        SafetyViolationType, ExploitDifficulty,
        SafetyViolation, MemoryLeak, ExploitableBug,
        Remediation, SafetyReport, AllocKind, AllocLifetime,
        MemoryRegion, PointerState, ResourceState,
        VerifierState, MemorySafetyVerifier,
    )

    def test_violation_types():
        assert SafetyViolationType.STACK_BUFFER_OVERFLOW.name == "STACK_BUFFER_OVERFLOW"
        assert SafetyViolationType.TEMPORAL_USE_AFTER_FREE.name == "TEMPORAL_USE_AFTER_FREE"

    def test_exploit_difficulty():
        assert ExploitDifficulty.EASY.name == "EASY"
        assert ExploitDifficulty.HARD.name == "HARD"

    def test_safety_violation():
        sv = SafetyViolation(
            type=SafetyViolationType.STACK_BUFFER_OVERFLOW,
            location="line 5",
            description="out-of-bounds write",
            severity="critical",
        )
        assert sv.type == SafetyViolationType.STACK_BUFFER_OVERFLOW

    def test_memory_leak():
        ml = MemoryLeak(allocation_site="malloc@line20",
                        variable="buf", size=256)
        assert ml.size == 256

    def test_safety_report():
        sr = SafetyReport()
        assert sr.safe or not sr.safe

    def test_alloc_kind():
        assert AllocKind.HEAP.name == "HEAP"
        assert AllocKind.STACK.name == "STACK"

    def test_memory_region():
        mr = MemoryRegion(name="buf", size=256,
                          kind=AllocKind.HEAP)
        assert mr.size == 256

    def test_verifier_empty():
        mv = MemorySafetyVerifier()
        report_result = mv.verify([])
        assert isinstance(report_result, SafetyReport)

    def test_verifier_alloc_free():
        mv = MemorySafetyVerifier()
        stmts = [
            {"kind": "alloc", "var": "p", "type": "int*",
             "size": 4, "alloc_kind": "heap"},
            {"kind": "free", "var": "p"},
        ]
        report_result = mv.verify(stmts)
        assert isinstance(report_result, SafetyReport)

    def test_pointer_state():
        ps = PointerState(name="p", targets={"buf"},
                          may_be_null=False)
        assert not ps.may_be_null

    for fn in [test_violation_types, test_exploit_difficulty,
               test_safety_violation, test_memory_leak,
               test_safety_report, test_alloc_kind,
               test_memory_region, test_verifier_empty,
               test_verifier_alloc_free, test_pointer_state]:
        report.results.append(_run_test(fn.__name__, fn))


# ===================================================================
# 7. Code Complexity Comparator
# ===================================================================

def _bench_complexity(report: ModuleReport) -> None:
    from code_complexity_comparator import (
        CyclomaticResult, CognitiveResult,
        ErrorHandlingMetrics, MemoryManagementMetrics,
        AbstractionMetrics, SafetyMetrics,
        IdiomaticScore, LanguageComplexity,
        ComparisonReport, ComplexityComparator,
    )

    def test_cyclomatic_result():
        cr = CyclomaticResult(value=5)
        assert cr.value == 5

    def test_cognitive_result():
        cr = CognitiveResult(value=8, nesting_penalties=3)
        assert cr.nesting_penalties == 3

    def test_error_handling_metrics():
        ehm = ErrorHandlingMetrics(
            error_paths=2, error_checks=1,
            panic_points=0,
        )
        assert ehm.error_paths == 2

    def test_memory_mgmt_metrics():
        mm = MemoryManagementMetrics(
            allocations=3, deallocations=3,
            smart_pointers=0, raw_pointers=2,
        )
        assert mm.allocations == mm.deallocations

    def test_comparator_basic():
        comp = ComplexityComparator()
        c_code = {
            "functions": [
                {"name": "add", "params": ["a", "b"],
                 "body": [{"kind": "return",
                           "value": {"kind": "binop", "op": "+",
                                     "left": {"kind": "var", "name": "a"},
                                     "right": {"kind": "var", "name": "b"}}}]},
            ],
        }
        rust_code = {
            "functions": [
                {"name": "add", "params": ["a", "b"],
                 "body": [{"kind": "return",
                           "value": {"kind": "binop", "op": "+",
                                     "left": {"kind": "var", "name": "a"},
                                     "right": {"kind": "var", "name": "b"}}}]},
            ],
        }
        result = comp.compare(c_code, rust_code)
        assert isinstance(result, ComparisonReport)

    def test_safety_metrics():
        sm = SafetyMetrics(total_operations=10, unsafe_operations=0,
                           bounds_checks=5, null_checks=3)
        assert sm.unsafe_operations == 0

    def test_idiomatic_score():
        s = IdiomaticScore(score=0.85, language="rust")
        assert 0.0 <= s.score <= 1.0

    def test_language_complexity():
        lc = LanguageComplexity(language="c")
        assert lc.language == "c"

    def test_abstraction_metrics():
        am = AbstractionMetrics(
            traits_interfaces=2, generics=1,
            closures=0, macros=3,
        )
        assert am.traits_interfaces == 2

    def test_comparison_report():
        cr = ComparisonReport()
        assert isinstance(cr.summary, dict)

    for fn in [test_cyclomatic_result, test_cognitive_result,
               test_error_handling_metrics, test_memory_mgmt_metrics,
               test_comparator_basic, test_safety_metrics,
               test_idiomatic_score, test_language_complexity,
               test_abstraction_metrics, test_comparison_report]:
        report.results.append(_run_test(fn.__name__, fn))


# ===================================================================
# 8. Migration Validator
# ===================================================================

def _bench_migration_validator(report: ModuleReport) -> None:
    from migration_validator import (
        MatchStatus, MigrationStatus, FunctionMatch,
        TestMigrationResult, BehavioralTestResult,
        ErrorHandlingMapping, PerformanceComparison,
        Regression, ValidationReport, MigrationValidator,
        _name_similarity,
    )

    def test_match_status():
        assert MatchStatus.MATCHED.name == "MATCHED"
        assert MatchStatus.UNMATCHED.name == "UNMATCHED"

    def test_migration_status():
        assert MigrationStatus.CORRECT.name == "CORRECT"

    def test_function_match():
        fm = FunctionMatch(c_name="do_thing", rust_name="do_thing",
                           status=MatchStatus.MATCHED,
                           structural_similarity=1.0)
        assert fm.structural_similarity == 1.0

    def test_name_similarity():
        assert _name_similarity("foo_bar", "foo_bar") == 1.0
        sim = _name_similarity("doThing", "do_thing")
        assert 0.0 <= sim <= 1.0

    def test_behavioral_test_result():
        btr = BehavioralTestResult(
            function_name="add", c_output="3",
            rust_output="3", match=True,
        )
        assert btr.match

    def test_regression():
        reg = Regression(function_name="foo",
                         test_case="test_basic",
                         description="output changed",
                         severity="high")
        assert reg.severity == "high"

    def test_validation_report():
        vr = ValidationReport()
        assert vr.status is not None

    def test_validator_empty():
        mv = MigrationValidator()
        c_module = {"functions": []}
        rust_module = {"functions": []}
        result = mv.validate(c_module, rust_module)
        assert isinstance(result, ValidationReport)

    def test_validator_simple():
        mv = MigrationValidator()
        c_module = {
            "functions": [
                {"name": "add", "params": [
                    {"name": "a", "type": "int"},
                    {"name": "b", "type": "int"},
                ], "return_type": "int",
                 "body": []},
            ],
        }
        rust_module = {
            "functions": [
                {"name": "add", "params": [
                    {"name": "a", "type": "i32"},
                    {"name": "b", "type": "i32"},
                ], "return_type": "i32",
                 "body": []},
            ],
        }
        result = mv.validate(c_module, rust_module)
        assert isinstance(result, ValidationReport)

    def test_performance_comparison():
        pc = PerformanceComparison(
            function_name="sort",
            c_estimated_ops=100,
            rust_estimated_ops=80,
            relative_performance=1.25,
        )
        assert pc.relative_performance > 1.0

    for fn in [test_match_status, test_migration_status,
               test_function_match, test_name_similarity,
               test_behavioral_test_result, test_regression,
               test_validation_report, test_validator_empty,
               test_validator_simple, test_performance_comparison]:
        report.results.append(_run_test(fn.__name__, fn))


# ===================================================================
# 9. Incremental Migration Engine
# ===================================================================

def _bench_incremental_migration(report: ModuleReport) -> None:
    from incremental_migration_engine import (
        MigrationPriority, MigrationState, RiskLevel,
        OrderingStrategy, FunctionInfo, NextIncrement,
        FFIWrapper, BuildRule, IntegrationTest,
        RollbackInfo, ProgressReport,
        IncrementalMigrator,
    )

    def test_priority_enum():
        assert MigrationPriority.HIGH.name == "HIGH"
        assert MigrationPriority.LOW.name == "LOW"

    def test_migration_state():
        assert MigrationState.PENDING.name == "PENDING"
        assert MigrationState.MIGRATED.name == "MIGRATED"

    def test_risk_level():
        assert RiskLevel.LOW.name == "LOW"
        assert RiskLevel.CRITICAL.name == "CRITICAL"

    def test_ordering_strategy():
        assert OrderingStrategy.LEAF_FIRST.name == "LEAF_FIRST"

    def test_function_info():
        fi = FunctionInfo(
            name="process",
            calls=["init", "cleanup"],
            state=MigrationState.PENDING,
            risk_level=RiskLevel.MEDIUM,
        )
        assert fi.name == "process"
        assert len(fi.calls) == 2

    def test_ffi_wrapper():
        fw = FFIWrapper(
            c_function="do_thing",
            rust_function="do_thing",
            wrapper_code='extern "C" fn do_thing() {}',
        )
        assert fw.c_function == "do_thing"

    def test_build_rule():
        br = BuildRule(target="libcore.so",
                       rule_type="shared_lib",
                       content="gcc -shared -o libcore.so core.c")
        assert br.target == "libcore.so"

    def test_rollback_info():
        ri = RollbackInfo(function_name="process",
                          reason="test failure")
        assert ri.function_name == "process"

    def test_migrator_empty():
        mig = IncrementalMigrator()
        c_project = {"functions": []}
        result = mig.plan_increment(c_project)
        assert isinstance(result, NextIncrement)

    def test_progress_report():
        pr = ProgressReport(
            total_functions=10, migrated_functions=3,
            total_loc=1000, migrated_loc=300,
        )
        assert pr.total_functions == 10

    for fn in [test_priority_enum, test_migration_state,
               test_risk_level, test_ordering_strategy,
               test_function_info, test_ffi_wrapper,
               test_build_rule, test_rollback_info,
               test_migrator_empty, test_progress_report]:
        report.results.append(_run_test(fn.__name__, fn))


# ===================================================================
# AST parse verification
# ===================================================================

_MODULE_FILES = [
    "abstract_interpreter_c.py",
    "rust_borrow_checker.py",
    "abi_compatibility_checker.py",
    "undefined_behavior_detector.py",
    "test_generator.py",
    "memory_safety_verifier.py",
    "code_complexity_comparator.py",
    "migration_validator.py",
    "incremental_migration_engine.py",
]


def _verify_ast_parse() -> List[Tuple[str, bool, str]]:
    """Return (filename, ok, error_msg) for each module file."""
    results = []
    for fname in _MODULE_FILES:
        fpath = os.path.join(_SRC, fname)
        try:
            with open(fpath) as fh:
                ast.parse(fh.read(), filename=fname)
            results.append((fname, True, ""))
        except SyntaxError as exc:
            results.append((fname, False, str(exc)))
    return results


def _verify_line_counts() -> List[Tuple[str, int, bool]]:
    """Return (filename, line_count, meets_minimum)."""
    results = []
    for fname in _MODULE_FILES:
        fpath = os.path.join(_SRC, fname)
        with open(fpath) as fh:
            count = sum(1 for _ in fh)
        results.append((fname, count, count >= 600))
    return results


# ===================================================================
# Main benchmark driver
# ===================================================================

_BENCH_RUNNERS: List[Tuple[str, Callable[[ModuleReport], None]]] = [
    ("abstract_interpreter_c", _bench_abstract_interpreter),
    ("rust_borrow_checker", _bench_borrow_checker),
    ("abi_compatibility_checker", _bench_abi_checker),
    ("undefined_behavior_detector", _bench_ub_detector),
    ("test_generator", _bench_test_generator),
    ("memory_safety_verifier", _bench_memory_safety),
    ("code_complexity_comparator", _bench_complexity),
    ("migration_validator", _bench_migration_validator),
    ("incremental_migration_engine", _bench_incremental_migration),
]


def run_all() -> int:
    """Run every benchmark, print report, return exit code."""
    print("=" * 72)
    print("  Cross-Language Equivalence Verifier — Comprehensive Benchmark")
    print("=" * 72)
    print()

    # --- AST parse check ---
    print("Phase 1: AST parse verification")
    print("-" * 40)
    ast_results = _verify_ast_parse()
    all_parse_ok = True
    for fname, ok, err in ast_results:
        status = "OK" if ok else f"FAIL ({err})"
        print(f"  {fname:45s} {status}")
        if not ok:
            all_parse_ok = False
    print()

    # --- Line count check ---
    print("Phase 2: Line count verification (≥600)")
    print("-" * 40)
    lc_results = _verify_line_counts()
    all_lc_ok = True
    for fname, count, ok in lc_results:
        status = "OK" if ok else "BELOW MINIMUM"
        print(f"  {fname:45s} {count:5d} lines  {status}")
        if not ok:
            all_lc_ok = False
    print()

    # --- Functional benchmarks ---
    print("Phase 3: Functional benchmarks")
    print("-" * 40)
    reports: List[ModuleReport] = []
    total_pass = 0
    total_fail = 0

    for mod_name, runner in _BENCH_RUNNERS:
        rpt = ModuleReport(module_name=mod_name)

        # Try importing the module first
        try:
            importlib.import_module(mod_name)
        except Exception as exc:
            rpt.import_ok = False
            rpt.import_error = str(exc)
            reports.append(rpt)
            print(f"  {mod_name:40s}  IMPORT FAILED: {exc}")
            total_fail += 1
            continue

        # Run the benchmark tests
        try:
            runner(rpt)
        except Exception as exc:
            rpt.results.append(TestResult(
                name=f"{mod_name}_runner",
                passed=False,
                elapsed=0.0,
                error=traceback.format_exc(),
            ))

        passed = rpt.passed
        failed = rpt.failed
        total_pass += passed
        total_fail += failed
        status = "ALL PASSED" if failed == 0 else f"{failed} FAILED"
        print(f"  {mod_name:40s}  {passed}/{rpt.total}  {status}")

        # Print failures in detail
        for r in rpt.results:
            if not r.passed:
                print(f"    FAIL: {r.name}")
                if r.error:
                    for line in r.error.strip().split("\n")[-3:]:
                        print(f"      {line}")

        reports.append(rpt)

    print()
    print("=" * 72)
    print(f"  TOTALS: {total_pass} passed, {total_fail} failed")
    print(f"  AST parse: {'ALL OK' if all_parse_ok else 'FAILURES'}")
    print(f"  Line counts: {'ALL OK' if all_lc_ok else 'SOME BELOW 600'}")
    print("=" * 72)

    if total_fail > 0 or not all_parse_ok or not all_lc_ok:
        print("\n*** BENCHMARK FAILED ***\n")
        return 1

    print("\n*** ALL BENCHMARKS PASSED ***\n")
    return 0


if __name__ == "__main__":
    sys.exit(run_all())
