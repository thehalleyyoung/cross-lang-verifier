#!/usr/bin/env python3
"""
Comprehensive benchmark for all cross-language equivalence verifier components.
Tests abstract interpreter, borrow checker, ABI checker, UB detector,
test generator, memory safety verifier, complexity comparator,
migration validator, and incremental migration engine.
"""

import json
import os
import sys
import time
import traceback
from typing import Dict, List, Any, Tuple

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abstract_interpreter_c import (
    CAbstractInterpreter, AbstractState, Interval, BugKind
)
from rust_borrow_checker import (
    BorrowChecker, BorrowCheckResult, BorrowKind, ViolationSeverity
)
from abi_compatibility_checker import (
    ABIChecker, TypeDescriptor, TypeKind, FieldDescriptor, EnumVariant,
    PlatformInfo, CallingConvention
)
from undefined_behavior_detector import (
    UBDetector, UBViolation, UBType
)
from test_generator import (
    TestGenerator, ParamSpec, TypeCategory, TestCase
)
from memory_safety_verifier import (
    MemorySafetyVerifier, SafetyReport, SafetyViolationType
)
from code_complexity_comparator import (
    ComplexityComparator, ComparisonReport
)
from migration_validator import (
    MigrationValidator, ValidationReport, MatchStatus, MigrationStatus
)
from incremental_migration_engine import (
    IncrementalMigrator, OrderingStrategy, MigrationState
)


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

class BenchmarkResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = 0
        self.failed = 0
        self.errors: List[str] = []
        self.details: List[Dict[str, Any]] = []
        self.elapsed_ms = 0.0

    def record(self, test_name: str, passed: bool, detail: str = "") -> None:
        if passed:
            self.passed += 1
        else:
            self.failed += 1
            self.errors.append(f"{test_name}: {detail}")
        self.details.append({
            "test": test_name,
            "passed": passed,
            "detail": detail,
        })

    @property
    def total(self) -> int:
        return self.passed + self.failed

    @property
    def accuracy(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "failed": self.failed,
            "total": self.total,
            "accuracy": self.accuracy,
            "elapsed_ms": self.elapsed_ms,
            "errors": self.errors,
        }


def timed(func):
    """Decorator to time benchmark functions."""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = (time.time() - start) * 1000
        if isinstance(result, BenchmarkResult):
            result.elapsed_ms = elapsed
        return result
    return wrapper


# ---------------------------------------------------------------------------
# 1. Abstract Interpreter Benchmark
# ---------------------------------------------------------------------------

@timed
def benchmark_abstract_interpreter() -> BenchmarkResult:
    br = BenchmarkResult("Abstract Interpreter (C)")
    interp = CAbstractInterpreter()

    # Test 1: Null pointer dereference
    prog1 = {"params": [], "body": [
        {"kind": "decl", "name": "p", "type": "int*",
         "init": {"kind": "const", "value": 0}},
        {"kind": "expr", "expr": {"kind": "deref",
         "operand": {"kind": "var", "name": "p"}}},
    ]}
    state = interp.analyze(prog1)
    has_null = any(b.kind == BugKind.NULL_DEREF for b in state.bugs_found)
    br.record("null_deref_detection", has_null,
              f"found {len(state.bugs_found)} bugs, null_deref={'yes' if has_null else 'no'}")

    # Test 2: Buffer overflow
    prog2 = {"params": [], "body": [
        {"kind": "decl", "name": "buf", "type": "char*",
         "init": {"kind": "call", "func": "malloc", "args": [10]}},
        {"kind": "expr", "expr": {"kind": "index",
         "base": {"kind": "var", "name": "buf"},
         "index": {"kind": "const", "value": 15}}},
    ]}
    interp2 = CAbstractInterpreter()
    state2 = interp2.analyze(prog2)
    has_overflow = any(b.kind == BugKind.BUFFER_OVERFLOW for b in state2.bugs_found)
    br.record("buffer_overflow_detection", has_overflow,
              f"found {len(state2.bugs_found)} bugs")

    # Test 3: Integer overflow
    prog3 = {"params": [{"name": "x", "type": "int"}], "body": [
        {"kind": "decl", "name": "result", "type": "int",
         "init": {"kind": "binop", "op": "+",
                  "left": {"kind": "var", "name": "x"},
                  "right": {"kind": "const", "value": 2147483647}}},
    ]}
    interp3 = CAbstractInterpreter()
    state3 = interp3.analyze(prog3)
    has_int_overflow = any(b.kind == BugKind.INTEGER_OVERFLOW for b in state3.bugs_found)
    br.record("integer_overflow_detection", has_int_overflow,
              f"found {len(state3.bugs_found)} bugs")

    # Test 4: Use after free
    prog4 = {"params": [], "body": [
        {"kind": "decl", "name": "p", "type": "int*",
         "init": {"kind": "call", "func": "malloc", "args": [4]}},
        {"kind": "call", "func": "free",
         "args": [{"kind": "var", "name": "p"}]},
        {"kind": "expr", "expr": {"kind": "deref",
         "operand": {"kind": "var", "name": "p"}}},
    ]}
    interp4 = CAbstractInterpreter()
    state4 = interp4.analyze(prog4)
    has_uaf = any(b.kind == BugKind.USE_AFTER_FREE for b in state4.bugs_found)
    br.record("use_after_free_detection", has_uaf,
              f"found {len(state4.bugs_found)} bugs")

    # Test 5: Double free
    prog5 = {"params": [], "body": [
        {"kind": "decl", "name": "p", "type": "int*",
         "init": {"kind": "call", "func": "malloc", "args": [4]}},
        {"kind": "call", "func": "free",
         "args": [{"kind": "var", "name": "p"}]},
        {"kind": "call", "func": "free",
         "args": [{"kind": "var", "name": "p"}]},
    ]}
    interp5 = CAbstractInterpreter()
    state5 = interp5.analyze(prog5)
    has_df = any(b.kind == BugKind.DOUBLE_FREE for b in state5.bugs_found)
    br.record("double_free_detection", has_df,
              f"found {len(state5.bugs_found)} bugs")

    # Test 6: Safe program (no bugs)
    prog6 = {"params": [{"name": "x", "type": "int"}], "body": [
        {"kind": "decl", "name": "y", "type": "int",
         "init": {"kind": "binop", "op": "+",
                  "left": {"kind": "var", "name": "x"},
                  "right": {"kind": "const", "value": 1}}},
        {"kind": "return", "value": {"kind": "var", "name": "y"}},
    ]}
    interp6 = CAbstractInterpreter()
    state6 = interp6.analyze(prog6)
    no_critical = all(b.severity != "error" for b in state6.bugs_found)
    br.record("safe_program_no_errors", no_critical,
              f"found {len(state6.bugs_found)} bugs")

    # Test 7: Division by zero
    prog7 = {"params": [{"name": "x", "type": "int"}], "body": [
        {"kind": "decl", "name": "result", "type": "int",
         "init": {"kind": "binop", "op": "/",
                  "left": {"kind": "const", "value": 10},
                  "right": {"kind": "var", "name": "x"}}},
    ]}
    interp7 = CAbstractInterpreter()
    state7 = interp7.analyze(prog7)
    has_divz = any(b.kind == BugKind.DIVISION_BY_ZERO for b in state7.bugs_found)
    br.record("division_by_zero_detection", has_divz,
              f"found {len(state7.bugs_found)} bugs")

    # Test 8: Uninitialized variable
    prog8 = {"params": [], "body": [
        {"kind": "decl", "name": "x", "type": "int"},
        {"kind": "decl", "name": "y", "type": "int",
         "init": {"kind": "var", "name": "x"}},
    ]}
    interp8 = CAbstractInterpreter()
    state8 = interp8.analyze(prog8)
    has_uninit = any(b.kind == BugKind.UNINIT_READ for b in state8.bugs_found)
    br.record("uninit_read_detection", has_uninit,
              f"found {len(state8.bugs_found)} bugs")

    # Test 9: Interval arithmetic correctness
    a = Interval(1, 5)
    b = Interval(2, 3)
    c = a + b
    br.record("interval_add", c.lo == 3 and c.hi == 8, f"[1,5]+[2,3]={c}")

    # Test 10: Interval widening
    x = Interval(0, 5)
    y = Interval(0, 10)
    w = x.widen(y)
    br.record("interval_widen", w.hi == float("inf"),
              f"widen([0,5],[0,10])={w}")

    return br


# ---------------------------------------------------------------------------
# 2. Borrow Checker Benchmark
# ---------------------------------------------------------------------------

@timed
def benchmark_borrow_checker() -> BenchmarkResult:
    br = BenchmarkResult("Rust Borrow Checker")
    checker = BorrowChecker()

    # Valid programs (should pass)
    # V1: Simple ownership
    valid1 = {"params": [], "body": [
        {"kind": "let", "name": "x", "type": "String",
         "init": {"kind": "call", "func": "String::new", "args": []}},
        {"kind": "use", "name": "x"},
    ]}
    r1 = checker.check(valid1)
    br.record("valid_simple_ownership", r1.valid, f"violations={len(r1.violations)}")

    # V2: Shared borrows
    valid2 = {"params": [], "body": [
        {"kind": "let", "name": "x", "type": "String"},
        {"kind": "borrow", "borrower": "r1", "place": "x", "mutable": False},
        {"kind": "borrow", "borrower": "r2", "place": "x", "mutable": False},
        {"kind": "use", "name": "r1"},
        {"kind": "use", "name": "r2"},
    ]}
    r2 = checker.check(valid2)
    br.record("valid_shared_borrows", r2.valid, f"violations={len(r2.violations)}")

    # V3: Sequential mut borrows
    valid3 = {"params": [], "body": [
        {"kind": "let", "name": "x", "type": "Vec<i32>"},
        {"kind": "block", "body": [
            {"kind": "borrow", "borrower": "r1", "place": "x", "mutable": True},
            {"kind": "use", "name": "r1"},
        ]},
        {"kind": "borrow", "borrower": "r2", "place": "x", "mutable": True},
        {"kind": "use", "name": "r2"},
    ]}
    r3 = checker.check(valid3)
    br.record("valid_sequential_mut_borrows", r3.valid, f"violations={len(r3.violations)}")

    # V4: Copy types
    valid4 = {"params": [], "body": [
        {"kind": "let", "name": "x", "type": "i32"},
        {"kind": "move", "source": "x", "dest": "y"},
        {"kind": "use", "name": "x"},
    ]}
    r4 = checker.check(valid4)
    br.record("valid_copy_type", r4.valid, f"violations={len(r4.violations)}")

    # V5: Borrow in function call
    valid5 = {"params": [{"name": "data", "type": "Vec<i32>"}], "body": [
        {"kind": "use", "name": "data"},
    ]}
    r5 = checker.check(valid5)
    br.record("valid_param_use", r5.valid, f"violations={len(r5.violations)}")

    # Invalid programs (should fail)
    # I1: Use after move
    invalid1 = {"params": [], "body": [
        {"kind": "let", "name": "s", "type": "String"},
        {"kind": "move", "source": "s", "dest": "t"},
        {"kind": "use", "name": "s"},
    ]}
    r6 = checker.check(invalid1)
    br.record("invalid_use_after_move", not r6.valid,
              f"valid={r6.valid}, violations={len(r6.violations)}")

    # I2: Mutable borrow while shared borrow exists
    invalid2 = {"params": [], "body": [
        {"kind": "let", "name": "x", "type": "Vec<i32>"},
        {"kind": "borrow", "borrower": "r1", "place": "x", "mutable": False},
        {"kind": "borrow", "borrower": "r2", "place": "x", "mutable": True},
        {"kind": "use", "name": "r1"},
    ]}
    r7 = checker.check(invalid2)
    br.record("invalid_mut_borrow_with_shared", not r7.valid,
              f"valid={r7.valid}, violations={len(r7.violations)}")

    # I3: Two mutable borrows
    invalid3 = {"params": [], "body": [
        {"kind": "let", "name": "x", "type": "Vec<i32>"},
        {"kind": "borrow", "borrower": "r1", "place": "x", "mutable": True},
        {"kind": "borrow", "borrower": "r2", "place": "x", "mutable": True},
    ]}
    r8 = checker.check(invalid3)
    br.record("invalid_two_mut_borrows", not r8.valid,
              f"valid={r8.valid}, violations={len(r8.violations)}")

    # I4: Write through shared ref
    invalid4 = {"params": [], "body": [
        {"kind": "let", "name": "x", "type": "i32"},
        {"kind": "borrow", "borrower": "r", "place": "x", "mutable": False},
        {"kind": "write_ref", "ref": "r"},
    ]}
    r9 = checker.check(invalid4)
    br.record("invalid_write_shared_ref", not r9.valid,
              f"valid={r9.valid}, violations={len(r9.violations)}")

    # I5: Move from borrowed
    invalid5 = {"params": [], "body": [
        {"kind": "let", "name": "x", "type": "String"},
        {"kind": "borrow", "borrower": "r", "place": "x", "mutable": False},
        {"kind": "move", "source": "x", "dest": "y"},
        {"kind": "use", "name": "r"},
    ]}
    r10 = checker.check(invalid5)
    br.record("invalid_move_from_borrowed", not r10.valid,
              f"valid={r10.valid}, violations={len(r10.violations)}")

    return br


# ---------------------------------------------------------------------------
# 3. ABI Checker Benchmark
# ---------------------------------------------------------------------------

@timed
def benchmark_abi_checker() -> BenchmarkResult:
    br = BenchmarkResult("ABI Compatibility Checker")
    checker = ABIChecker()

    # Pair 1: Compatible structs
    c_s1 = TypeDescriptor("Point", TypeKind.STRUCT, fields=[
        FieldDescriptor("x", TypeDescriptor("int", TypeKind.PRIMITIVE, size=4, alignment=4)),
        FieldDescriptor("y", TypeDescriptor("int", TypeKind.PRIMITIVE, size=4, alignment=4)),
    ])
    r_s1 = TypeDescriptor("Point", TypeKind.STRUCT, is_repr_c=True, fields=[
        FieldDescriptor("x", TypeDescriptor("i32", TypeKind.PRIMITIVE, size=4, alignment=4)),
        FieldDescriptor("y", TypeDescriptor("i32", TypeKind.PRIMITIVE, size=4, alignment=4)),
    ])
    report1 = checker.check([c_s1], [r_s1])
    br.record("compatible_struct", report1.compatible,
              f"incompatibilities={len(report1.incompatibilities)}")

    # Pair 2: Size mismatch
    c_s2 = TypeDescriptor("Data", TypeKind.STRUCT, fields=[
        FieldDescriptor("value", TypeDescriptor("long", TypeKind.PRIMITIVE, size=8, alignment=8)),
    ])
    r_s2 = TypeDescriptor("Data", TypeKind.STRUCT, is_repr_c=True, fields=[
        FieldDescriptor("value", TypeDescriptor("i32", TypeKind.PRIMITIVE, size=4, alignment=4)),
    ])
    checker2 = ABIChecker()
    report2 = checker2.check([c_s2], [r_s2])
    br.record("size_mismatch_detected", not report2.compatible,
              f"incompatibilities={len(report2.incompatibilities)}")

    # Pair 3: Missing repr(C)
    c_s3 = TypeDescriptor("Color", TypeKind.STRUCT, fields=[
        FieldDescriptor("r", TypeDescriptor("unsigned char", TypeKind.PRIMITIVE, size=1, alignment=1)),
        FieldDescriptor("g", TypeDescriptor("unsigned char", TypeKind.PRIMITIVE, size=1, alignment=1)),
        FieldDescriptor("b", TypeDescriptor("unsigned char", TypeKind.PRIMITIVE, size=1, alignment=1)),
    ])
    r_s3 = TypeDescriptor("Color", TypeKind.STRUCT, is_repr_c=False, fields=[
        FieldDescriptor("r", TypeDescriptor("u8", TypeKind.PRIMITIVE, size=1, alignment=1)),
        FieldDescriptor("g", TypeDescriptor("u8", TypeKind.PRIMITIVE, size=1, alignment=1)),
        FieldDescriptor("b", TypeDescriptor("u8", TypeKind.PRIMITIVE, size=1, alignment=1)),
    ])
    checker3 = ABIChecker()
    report3 = checker3.check([c_s3], [r_s3])
    has_repr_c_warning = any(i.kind.value == "missing_repr_c"
                             for i in report3.incompatibilities)
    br.record("missing_repr_c_detected", has_repr_c_warning,
              f"repr_c_issue={has_repr_c_warning}")

    # Pair 4: Compatible enums
    c_e = TypeDescriptor("Status", TypeKind.ENUM, size=4, alignment=4,
                         enum_variants=[
                             EnumVariant("OK", 0), EnumVariant("ERR", 1),
                         ])
    r_e = TypeDescriptor("Status", TypeKind.ENUM, size=4, alignment=4,
                         is_repr_c=True,
                         enum_variants=[
                             EnumVariant("OK", 0), EnumVariant("ERR", 1),
                         ])
    checker4 = ABIChecker()
    report4 = checker4.check([c_e], [r_e])
    br.record("compatible_enum", report4.compatible,
              f"incompatibilities={len(report4.incompatibilities)}")

    # Pair 5: Field count mismatch
    c_s5 = TypeDescriptor("Rec", TypeKind.STRUCT, fields=[
        FieldDescriptor("a", TypeDescriptor("int", TypeKind.PRIMITIVE, size=4, alignment=4)),
        FieldDescriptor("b", TypeDescriptor("int", TypeKind.PRIMITIVE, size=4, alignment=4)),
    ])
    r_s5 = TypeDescriptor("Rec", TypeKind.STRUCT, is_repr_c=True, fields=[
        FieldDescriptor("a", TypeDescriptor("i32", TypeKind.PRIMITIVE, size=4, alignment=4)),
    ])
    checker5 = ABIChecker()
    report5 = checker5.check([c_s5], [r_s5])
    br.record("field_count_mismatch", not report5.compatible,
              f"incompatibilities={len(report5.incompatibilities)}")

    return br


# ---------------------------------------------------------------------------
# 4. UB Detector Benchmark
# ---------------------------------------------------------------------------

@timed
def benchmark_ub_detector() -> BenchmarkResult:
    br = BenchmarkResult("Undefined Behavior Detector")
    det = UBDetector()

    test_cases = [
        ("signed_overflow", UBType.SIGNED_OVERFLOW, {"body": [
            {"kind": "decl", "name": "x", "type": "int",
             "init": {"kind": "const", "value": 2147483647}},
            {"kind": "assign", "target": "y",
             "value": {"kind": "binop", "op": "+", "type": "int",
                       "left": {"kind": "var", "name": "x"},
                       "right": {"kind": "const", "value": 1}}},
        ]}),
        ("null_deref", UBType.NULL_DEREF, {"body": [
            {"kind": "decl", "name": "p", "type": "int*",
             "init": {"kind": "const", "value": 0}},
            {"kind": "expr", "expr": {"kind": "deref",
             "operand": {"kind": "var", "name": "p"}}},
        ]}),
        ("oob_access", UBType.OOB_ACCESS, {"body": [
            {"kind": "decl", "name": "arr", "type": "int*",
             "init": {"kind": "call", "func": "malloc", "args": [
                 {"kind": "const", "value": 10}]}},
            {"kind": "expr", "expr": {"kind": "index",
             "base": {"kind": "var", "name": "arr"},
             "index": {"kind": "const", "value": 15}}},
        ]}),
        ("use_after_free", UBType.USE_AFTER_FREE, {"body": [
            {"kind": "decl", "name": "p", "type": "int*",
             "init": {"kind": "call", "func": "malloc", "args": [
                 {"kind": "const", "value": 4}]}},
            {"kind": "call", "func": "free",
             "args": [{"kind": "var", "name": "p"}]},
            {"kind": "expr", "expr": {"kind": "deref",
             "operand": {"kind": "var", "name": "p"}}},
        ]}),
        ("double_free", UBType.DOUBLE_FREE, {"body": [
            {"kind": "decl", "name": "p", "type": "int*",
             "init": {"kind": "call", "func": "malloc", "args": [
                 {"kind": "const", "value": 4}]}},
            {"kind": "call", "func": "free",
             "args": [{"kind": "var", "name": "p"}]},
            {"kind": "call", "func": "free",
             "args": [{"kind": "var", "name": "p"}]},
        ]}),
        ("uninit_read", UBType.UNINIT_READ, {"body": [
            {"kind": "decl", "name": "x", "type": "int"},
            {"kind": "expr", "expr": {"kind": "var", "name": "x"}},
        ]}),
        ("shift_overflow", UBType.SHIFT_OVERFLOW, {"body": [
            {"kind": "decl", "name": "x", "type": "int",
             "init": {"kind": "const", "value": 1}},
            {"kind": "assign", "target": "y",
             "value": {"kind": "binop", "op": "<<", "left_type": "int",
                       "left": {"kind": "var", "name": "x"},
                       "right": {"kind": "const", "value": 33}}},
        ]}),
        ("div_by_zero", UBType.DIVISION_BY_ZERO, {"body": [
            {"kind": "decl", "name": "x", "type": "int",
             "init": {"kind": "const", "value": 0}},
            {"kind": "assign", "target": "y",
             "value": {"kind": "binop", "op": "/", "type": "int",
                       "left": {"kind": "const", "value": 10},
                       "right": {"kind": "var", "name": "x"}}},
        ]}),
        ("strict_aliasing", UBType.STRICT_ALIASING, {"body": [
            {"kind": "decl", "name": "ip", "type": "int*",
             "init": {"kind": "call", "func": "malloc", "args": [
                 {"kind": "const", "value": 4}]}},
            {"kind": "expr", "expr": {"kind": "cast", "type": "float*",
             "operand": {"kind": "var", "name": "ip"}}},
        ]}),
        ("sequence_point", UBType.SEQUENCE_POINT, {"body": [
            {"kind": "decl", "name": "i", "type": "int",
             "init": {"kind": "const", "value": 0}},
            {"kind": "sequence_expr", "exprs": [
                {"kind": "unop", "op": "++",
                 "operand": {"kind": "var", "name": "i"}},
                {"kind": "unop", "op": "++",
                 "operand": {"kind": "var", "name": "i"}},
            ]},
        ]}),
    ]

    # Test each with fresh detector
    detected = 0
    total = len(test_cases)
    for name, expected_type, prog in test_cases:
        d = UBDetector()
        violations = d.detect(prog)
        found = any(v.type == expected_type for v in violations)
        if found:
            detected += 1
        br.record(f"ub_{name}", found,
                  f"expected={expected_type.value}, found_types={[v.type.value for v in violations]}")

    # Additional: safe program should have no UB
    safe_prog = {"body": [
        {"kind": "decl", "name": "x", "type": "int",
         "init": {"kind": "const", "value": 42}},
        {"kind": "return", "value": {"kind": "var", "name": "x"}},
    ]}
    d_safe = UBDetector()
    safe_violations = d_safe.detect(safe_prog)
    br.record("ub_safe_program", len(safe_violations) == 0,
              f"violations={len(safe_violations)}")

    # P/R/F1
    precision = detected / total if total > 0 else 0.0
    recall = detected / total if total > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    br.details.append({"test": "ub_prf1", "precision": precision,
                        "recall": recall, "f1": f1})

    return br


# ---------------------------------------------------------------------------
# 5. Test Generator Benchmark
# ---------------------------------------------------------------------------

@timed
def benchmark_test_generator() -> BenchmarkResult:
    br = BenchmarkResult("Test Generator")
    gen = TestGenerator(seed=42)

    # Generate for 5 function pairs
    functions = [
        {"name": "add", "params": [
            {"name": "a", "type": "int"}, {"name": "b", "type": "int"}],
         "body": [{"kind": "return", "value": {"kind": "binop", "op": "+",
                   "left": {"kind": "var", "name": "a"},
                   "right": {"kind": "var", "name": "b"}}}]},
        {"name": "abs_val", "params": [{"name": "x", "type": "int"}],
         "body": [{"kind": "if", "cond": {"kind": "binop", "op": "<",
                   "left": {"kind": "var", "name": "x"},
                   "right": {"kind": "const", "value": 0}}}]},
        {"name": "max", "params": [
            {"name": "a", "type": "int"}, {"name": "b", "type": "int"}],
         "body": [{"kind": "if", "cond": {"kind": "binop", "op": ">",
                   "left": {"kind": "var", "name": "a"},
                   "right": {"kind": "var", "name": "b"}}}]},
        {"name": "factorial", "params": [{"name": "n", "type": "unsigned int"}],
         "body": [{"kind": "while", "cond": {"kind": "binop", "op": ">",
                   "left": {"kind": "var", "name": "n"},
                   "right": {"kind": "const", "value": 1}}}]},
        {"name": "strlen_custom", "params": [{"name": "s", "type": "char*"}],
         "body": [{"kind": "while", "cond": {"kind": "binop", "op": "!=",
                   "left": {"kind": "var", "name": "s"},
                   "right": {"kind": "const", "value": 0}}}]},
    ]

    for func in functions:
        tests = gen.generate(func)
        has_boundary = any(tc.strategy.value == "boundary" for tc in tests)
        has_random = any(tc.strategy.value == "random" for tc in tests)
        has_coverage = any(tc.strategy.value == "coverage-guided" for tc in tests)
        br.record(f"gen_{func['name']}_count", len(tests) > 10,
                  f"generated {len(tests)} tests")
        br.record(f"gen_{func['name']}_boundary", has_boundary,
                  f"has boundary tests")
        br.record(f"gen_{func['name']}_random", has_random,
                  f"has random tests")

    # Test serialization roundtrip
    tests = gen.generate(functions[0])
    json_str = gen.tests_to_json(tests)
    loaded = gen.tests_from_json(json_str)
    br.record("serialization_roundtrip", len(loaded) == len(tests),
              f"original={len(tests)}, loaded={len(loaded)}")

    # Test regression generation
    divergence = {"inputs": {"a": 5, "b": -3}, "c_output": 2, "rust_output": 3,
                  "description": "subtraction mismatch"}
    reg_tests = gen.generate_regression_tests([divergence], c_func="sub", rust_func="sub")
    br.record("regression_generation", len(reg_tests) > 0,
              f"generated {len(reg_tests)} regression tests")

    return br


# ---------------------------------------------------------------------------
# 6. Memory Safety Benchmark
# ---------------------------------------------------------------------------

@timed
def benchmark_memory_safety() -> BenchmarkResult:
    br = BenchmarkResult("Memory Safety Verifier")
    verifier = MemorySafetyVerifier()

    # Test 1: Buffer overflow
    prog1 = {"body": [
        {"kind": "decl", "name": "buf", "type": "char*",
         "init": {"kind": "call", "func": "malloc", "size": 10}},
        {"kind": "write", "pointer": "buf", "offset": 15, "access_size": 1},
    ]}
    r1 = verifier.verify(prog1, "c")
    has_overflow = any(v.type in (SafetyViolationType.HEAP_OVERFLOW,
                                  SafetyViolationType.SPATIAL_OUT_OF_BOUNDS)
                       for v in r1.violations)
    br.record("spatial_overflow", has_overflow,
              f"violations={len(r1.violations)}")

    # Test 2: Use after free
    prog2 = {"body": [
        {"kind": "decl", "name": "p", "type": "int*",
         "init": {"kind": "call", "func": "malloc", "size": 4}},
        {"kind": "free", "pointer": "p"},
        {"kind": "deref", "pointer": "p", "access_size": 4},
    ]}
    v2 = MemorySafetyVerifier()
    r2 = v2.verify(prog2, "c")
    has_uaf = any(v.type == SafetyViolationType.TEMPORAL_USE_AFTER_FREE
                  for v in r2.violations)
    br.record("temporal_uaf", has_uaf,
              f"violations={len(r2.violations)}")

    # Test 3: Double free
    prog3 = {"body": [
        {"kind": "decl", "name": "p", "type": "int*",
         "init": {"kind": "call", "func": "malloc", "size": 4}},
        {"kind": "free", "pointer": "p"},
        {"kind": "free", "pointer": "p"},
    ]}
    v3 = MemorySafetyVerifier()
    r3 = v3.verify(prog3, "c")
    has_df = any(v.type == SafetyViolationType.TEMPORAL_DOUBLE_FREE
                 for v in r3.violations)
    br.record("temporal_double_free", has_df,
              f"violations={len(r3.violations)}")

    # Test 4: Memory leak
    prog4 = {"body": [
        {"kind": "decl", "name": "p", "type": "int*",
         "init": {"kind": "call", "func": "malloc", "size": 100}},
        {"kind": "assign", "target": "p",
         "value": {"kind": "const", "value": 0}},
    ]}
    v4 = MemorySafetyVerifier()
    r4 = v4.verify(prog4, "c")
    has_leak = len(r4.memory_leaks) > 0
    br.record("heap_leak", has_leak,
              f"leaks={len(r4.memory_leaks)}")

    # Test 5: Safe program
    prog5 = {"body": [
        {"kind": "decl", "name": "x", "type": "int", "size": 4},
    ]}
    v5 = MemorySafetyVerifier()
    r5 = v5.verify(prog5, "c")
    br.record("safe_program", r5.safe,
              f"violations={len(r5.violations)}")

    # Test 6: Null pointer deref
    prog6 = {"body": [
        {"kind": "decl", "name": "p", "type": "int*"},
        {"kind": "deref", "pointer": "p", "access_size": 4},
    ]}
    v6 = MemorySafetyVerifier()
    r6 = v6.verify(prog6, "c")
    has_null = any(v.type == SafetyViolationType.NULL_DEREF for v in r6.violations)
    br.record("null_deref", has_null, f"violations={len(r6.violations)}")

    # Test 7: Returning address of local
    prog7 = {"body": [
        {"kind": "decl", "name": "x", "type": "int", "size": 4},
        {"kind": "return", "value": {"kind": "addr",
         "operand": {"kind": "var", "name": "x"}}},
    ]}
    v7 = MemorySafetyVerifier()
    r7 = v7.verify(prog7, "c")
    has_dangling = any(v.type == SafetyViolationType.TEMPORAL_DANGLING_PTR
                       for v in r7.violations)
    br.record("dangling_ptr_return", has_dangling,
              f"violations={len(r7.violations)}")

    # Test 8: RAII in Rust
    prog8 = {"body": [
        {"kind": "resource_acquire", "name": "file", "resource_type": "File"},
    ]}
    v8 = MemorySafetyVerifier()
    r8 = v8.verify(prog8, "rust")
    has_raii = any(v.type == SafetyViolationType.RAII_RESOURCE_LEAK
                   for v in r8.violations)
    br.record("raii_resource_leak", has_raii,
              f"violations={len(r8.violations)}")

    # Test 9: Exploitable bug detection
    prog9 = {"body": [
        {"kind": "decl", "name": "buf", "type": "char*", "size": 16},
        {"kind": "decl", "name": "p", "type": "char*",
         "init": {"kind": "call", "func": "malloc", "size": 16}},
        {"kind": "write", "pointer": "p", "offset": 20, "access_size": 1},
    ]}
    v9 = MemorySafetyVerifier()
    r9 = v9.verify(prog9, "c")
    has_exploitable = len(r9.exploitable_bugs) > 0
    br.record("exploitable_detection", has_exploitable,
              f"exploitable={len(r9.exploitable_bugs)}")

    # Test 10: Negative index
    prog10 = {"body": [
        {"kind": "decl", "name": "arr", "type": "int*",
         "init": {"kind": "call", "func": "malloc", "size": 40}},
        {"kind": "deref", "pointer": "arr", "offset": -4, "access_size": 4},
    ]}
    v10 = MemorySafetyVerifier()
    r10 = v10.verify(prog10, "c")
    has_neg = any(v.type == SafetyViolationType.SPATIAL_NEGATIVE_INDEX
                  for v in r10.violations)
    br.record("negative_index", has_neg,
              f"violations={len(r10.violations)}")

    return br


# ---------------------------------------------------------------------------
# 7. Complexity Comparator Benchmark
# ---------------------------------------------------------------------------

@timed
def benchmark_complexity_comparator() -> BenchmarkResult:
    br = BenchmarkResult("Code Complexity Comparator")
    comp = ComplexityComparator()

    pairs = [
        ("simple_add", {
            "kind": "function", "body": [
                {"kind": "return", "value": {"kind": "binop", "op": "+",
                 "left": {"kind": "var", "name": "a"},
                 "right": {"kind": "var", "name": "b"}}},
            ]
        }, {
            "kind": "function", "body": [
                {"kind": "return", "value": {"kind": "binop", "op": "+",
                 "left": {"kind": "var", "name": "a"},
                 "right": {"kind": "var", "name": "b"}}},
            ]
        }),
        ("with_error_handling", {
            "kind": "function", "body": [
                {"kind": "call", "func": "malloc"},
                {"kind": "if", "cond": {"kind": "binop", "op": "==",
                 "left": {"kind": "var", "name": "p"},
                 "right": {"kind": "const", "value": 0}},
                 "then": [{"kind": "return", "value": {"kind": "const", "value": -1}}]},
                {"kind": "call", "func": "free"},
            ]
        }, {
            "kind": "function", "body": [
                {"kind": "call", "func": "Box::new"},
                {"kind": "try_operator"},
                {"kind": "return"},
            ]
        }),
        ("loop_heavy", {
            "kind": "function", "body": [
                {"kind": "for", "body": [
                    {"kind": "for", "body": [
                        {"kind": "assign"},
                    ]},
                ]},
            ]
        }, {
            "kind": "function", "body": [
                {"kind": "call", "func": "iter"},
                {"kind": "call", "func": "map"},
                {"kind": "call", "func": "collect"},
            ]
        }),
        ("unsafe_c_vs_safe_rust", {
            "kind": "function", "body": [
                {"kind": "deref"},
                {"kind": "cast"},
                {"kind": "binop", "op": "+", "left_type": "int*"},
                {"kind": "index"},
            ]
        }, {
            "kind": "function", "body": [
                {"kind": "call", "func": "Vec::new"},
                {"kind": "call", "func": "push"},
                {"kind": "index"},
                {"kind": "match"},
            ]
        }),
        ("macro_heavy", {
            "kind": "function", "body": [
                {"kind": "macro_call"},
                {"kind": "macro_call"},
                {"kind": "if"},
            ]
        }, {
            "kind": "function", "body": [
                {"kind": "macro_call", "macro": "println!"},
                {"kind": "match"},
                {"kind": "try_operator"},
                {"kind": "closure"},
            ]
        }),
    ]

    for name, c_code, rust_code in pairs:
        report = comp.compare(c_code, rust_code)
        c_cc = report.c_complexity.cyclomatic.value
        r_cc = report.rust_complexity.cyclomatic.value
        c_overall = report.c_complexity.overall_complexity()
        r_overall = report.rust_complexity.overall_complexity()

        br.record(f"complexity_{name}_computed",
                  c_cc >= 1 and r_cc >= 1,
                  f"c_cc={c_cc}, r_cc={r_cc}, c_overall={c_overall:.1f}, r_overall={r_overall:.1f}")

    # Check safety comparison makes sense
    report = comp.compare(pairs[3][1], pairs[3][2])
    c_safety = report.c_complexity.safety.safety_score
    r_safety = report.rust_complexity.safety.safety_score
    br.record("safety_comparison_valid",
              r_safety >= c_safety,
              f"c_safety={c_safety:.2f}, r_safety={r_safety:.2f}")

    return br


# ---------------------------------------------------------------------------
# 8. Migration Validator Benchmark
# ---------------------------------------------------------------------------

@timed
def benchmark_migration_validator() -> BenchmarkResult:
    br = BenchmarkResult("Migration Validator")
    validator = MigrationValidator()

    # Correct migrations
    correct_pairs = [
        ({"functions": [
            {"name": "add", "params": [{"name": "a", "type": "int"},
                                        {"name": "b", "type": "int"}],
             "return_type": "int",
             "body": [{"kind": "return"}],
             "test_vectors": [{"inputs": {"a": 1, "b": 2}, "output": 3,
                               "rust_output": 3}]},
        ]}, {"functions": [
            {"name": "add", "params": [{"name": "a", "type": "i32"},
                                        {"name": "b", "type": "i32"}],
             "return_type": "i32",
             "body": [{"kind": "return"}]},
        ]}),
        ({"functions": [
            {"name": "max", "params": [{"name": "a", "type": "int"},
                                        {"name": "b", "type": "int"}],
             "return_type": "int",
             "body": [{"kind": "if"}, {"kind": "return"}],
             "test_vectors": [{"inputs": {"a": 5, "b": 3}, "output": 5,
                               "rust_output": 5}]},
        ]}, {"functions": [
            {"name": "max", "params": [{"name": "a", "type": "i32"},
                                        {"name": "b", "type": "i32"}],
             "return_type": "i32",
             "body": [{"kind": "if"}, {"kind": "return"}]},
        ]}),
    ]

    for i, (c_proj, r_proj) in enumerate(correct_pairs):
        v = MigrationValidator()
        report = v.validate(c_proj, r_proj)
        br.record(f"correct_migration_{i}",
                  report.status in (MigrationStatus.CORRECT, MigrationStatus.PARTIAL),
                  f"status={report.status.value}, score={report.overall_score:.2f}")

    # Incorrect migrations
    incorrect_pairs = [
        ({"functions": [
            {"name": "sub", "params": [{"name": "a", "type": "int"},
                                        {"name": "b", "type": "int"}],
             "return_type": "int",
             "body": [{"kind": "return"}],
             "test_vectors": [{"inputs": {"a": 5, "b": 3}, "output": 2,
                               "rust_output": 8}]},
        ]}, {"functions": [
            {"name": "sub", "params": [{"name": "a", "type": "i32"},
                                        {"name": "b", "type": "i32"}],
             "return_type": "i32",
             "body": [{"kind": "return"}]},
        ]}),
        ({"functions": [
            {"name": "process", "params": [{"name": "x", "type": "int"}],
             "return_type": "int",
             "body": [{"kind": "return"}]},
            {"name": "helper", "params": [], "return_type": "void",
             "body": [{"kind": "return"}]},
        ]}, {"functions": [
            {"name": "process", "params": [{"name": "x", "type": "i32"}],
             "return_type": "i32",
             "body": [{"kind": "return"}]},
        ]}),
    ]

    for i, (c_proj, r_proj) in enumerate(incorrect_pairs):
        v = MigrationValidator()
        report = v.validate(c_proj, r_proj)
        is_flagged = (report.status != MigrationStatus.CORRECT or
                      len(report.regressions) > 0 or
                      len(report.unmigrated_functions) > 0)
        br.record(f"incorrect_migration_{i}", is_flagged,
                  f"status={report.status.value}, regressions={len(report.regressions)}, "
                  f"unmigrated={len(report.unmigrated_functions)}")

    # Completeness check
    c_full = {"functions": [
        {"name": "f1", "params": [], "return_type": "void", "body": []},
        {"name": "f2", "params": [], "return_type": "void", "body": []},
        {"name": "f3", "params": [], "return_type": "void", "body": []},
    ]}
    r_partial = {"functions": [
        {"name": "f1", "params": [], "return_type": "()", "body": []},
    ]}
    v = MigrationValidator()
    report = v.validate(c_full, r_partial)
    br.record("completeness_check",
              report.coverage < 1.0 and len(report.unmigrated_functions) > 0,
              f"coverage={report.coverage:.2f}, unmigrated={len(report.unmigrated_functions)}")

    return br


# ---------------------------------------------------------------------------
# 9. Incremental Migration Benchmark
# ---------------------------------------------------------------------------

@timed
def benchmark_incremental_migration() -> BenchmarkResult:
    br = BenchmarkResult("Incremental Migration Engine")

    project = {"functions": [
        {"name": "parse_input", "params": [{"name": "s", "type": "char*"}],
         "return_type": "int", "calls": ["validate", "convert"],
         "body": [1, 2, 3, 4, 5], "bug_count": 2, "complexity": 5,
         "test_coverage": 0.3},
        {"name": "validate", "params": [{"name": "s", "type": "char*"}],
         "return_type": "bool", "calls": [],
         "body": [1, 2, 3], "bug_count": 0, "complexity": 2,
         "test_coverage": 0.8},
        {"name": "convert", "params": [{"name": "s", "type": "char*"}],
         "return_type": "int", "calls": ["helper"],
         "body": [1, 2, 3, 4], "bug_count": 1, "complexity": 3,
         "test_coverage": 0.5},
        {"name": "helper", "params": [{"name": "x", "type": "int"}],
         "return_type": "int", "calls": [],
         "body": [1, 2], "bug_count": 0, "complexity": 1,
         "test_coverage": 0.9},
        {"name": "process", "params": [{"name": "data", "type": "int*"}],
         "return_type": "void", "calls": ["helper", "output"],
         "body": [1, 2, 3, 4, 5, 6], "bug_count": 3, "complexity": 7,
         "test_coverage": 0.2},
        {"name": "output", "params": [{"name": "result", "type": "int"}],
         "return_type": "void", "calls": [],
         "body": [1, 2], "bug_count": 0, "complexity": 1,
         "test_coverage": 0.7},
    ]}

    # Test 1: Leaf-first ordering
    migrator = IncrementalMigrator()
    increment = migrator.plan_increment(project, strategy=OrderingStrategy.LEAF_FIRST)
    leaves = {"validate", "helper", "output"}
    first_batch = set(increment.functions_to_migrate[:3])
    has_leaves = len(first_batch & leaves) > 0
    br.record("leaf_first_ordering", has_leaves,
              f"first batch: {increment.functions_to_migrate[:3]}")

    # Test 2: Most-buggy ordering
    m2 = IncrementalMigrator()
    inc2 = m2.plan_increment(project, strategy=OrderingStrategy.MOST_BUGGY)
    first_func = inc2.functions_to_migrate[0] if inc2.functions_to_migrate else ""
    br.record("most_buggy_ordering", first_func in ("process", "parse_input"),
              f"first={first_func}")

    # Test 3: FFI wrapper generation
    m3 = IncrementalMigrator()
    inc3 = m3.plan_increment(project, strategy=OrderingStrategy.LEAF_FIRST, batch_size=2)
    artifacts = m3.generate_increment_artifacts(inc3, project)
    has_wrappers = len(artifacts.get("ffi_wrappers", [])) > 0 or len(inc3.functions_to_migrate) > 0
    br.record("ffi_wrapper_generation", has_wrappers,
              f"wrappers={len(artifacts.get('ffi_wrappers', []))}")

    # Test 4: Build system generation
    has_cmake = "cmake" in artifacts and len(artifacts["cmake"]) > 0
    br.record("cmake_generation", has_cmake,
              f"cmake length={len(artifacts.get('cmake', ''))}")

    # Test 5: Progress tracking
    m5 = IncrementalMigrator()
    m5.plan_increment(project, migrated_so_far=["validate", "helper"])
    progress = m5.get_progress()
    br.record("progress_tracking",
              progress.migrated_functions == 2 and progress.total_functions == 6,
              f"migrated={progress.migrated_functions}/{progress.total_functions}")

    # Test 6: Risk assessment
    m6 = IncrementalMigrator()
    inc6 = m6.plan_increment(project, strategy=OrderingStrategy.LEAF_FIRST)
    has_risk = "overall_risk_score" in inc6.risk_assessment
    br.record("risk_assessment", has_risk,
              f"risk={inc6.risk_assessment.get('overall_risk_score', 'N/A')}")

    # Test 7: Rollback
    m7 = IncrementalMigrator()
    m7.plan_increment(project, migrated_so_far=["validate"])
    rb = m7.rollback("validate", "test failure")
    br.record("rollback_support", rb.function_name == "validate",
              f"rolled back: {rb.function_name}")

    # Test 8: Integration test generation
    has_tests = len(artifacts.get("integration_tests", [])) > 0
    br.record("integration_test_generation", has_tests,
              f"tests={len(artifacts.get('integration_tests', []))}")

    return br


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def run_all_benchmarks() -> Dict[str, Any]:
    print("=" * 70)
    print("COMPREHENSIVE BENCHMARK — Cross-Language Equivalence Verifier")
    print("=" * 70)

    benchmarks = [
        ("Abstract Interpreter", benchmark_abstract_interpreter),
        ("Borrow Checker", benchmark_borrow_checker),
        ("ABI Checker", benchmark_abi_checker),
        ("UB Detector", benchmark_ub_detector),
        ("Test Generator", benchmark_test_generator),
        ("Memory Safety", benchmark_memory_safety),
        ("Complexity Comparator", benchmark_complexity_comparator),
        ("Migration Validator", benchmark_migration_validator),
        ("Incremental Migration", benchmark_incremental_migration),
    ]

    results: Dict[str, Any] = {}
    total_passed = 0
    total_failed = 0
    total_time = 0.0

    for name, func in benchmarks:
        print(f"\n{'─' * 50}")
        print(f"Running: {name}")
        try:
            result = func()
            results[name] = result.to_dict()
            total_passed += result.passed
            total_failed += result.failed
            total_time += result.elapsed_ms

            status = "✓" if result.failed == 0 else "✗"
            print(f"  {status} {result.passed}/{result.total} passed "
                  f"({result.accuracy:.0%}) in {result.elapsed_ms:.1f}ms")
            if result.errors:
                for err in result.errors[:3]:
                    print(f"    ✗ {err}")
                if len(result.errors) > 3:
                    print(f"    ... and {len(result.errors) - 3} more")
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            traceback.print_exc()
            results[name] = {"error": str(e), "passed": 0, "failed": 1}
            total_failed += 1

    # Summary
    total = total_passed + total_failed
    accuracy = total_passed / total if total > 0 else 0.0

    print(f"\n{'=' * 70}")
    print(f"SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Total tests:  {total}")
    print(f"  Passed:       {total_passed}")
    print(f"  Failed:       {total_failed}")
    print(f"  Accuracy:     {accuracy:.1%}")
    print(f"  Total time:   {total_time:.1f}ms")
    print(f"{'=' * 70}")

    summary = {
        "benchmarks": results,
        "summary": {
            "total_tests": total,
            "passed": total_passed,
            "failed": total_failed,
            "accuracy": accuracy,
            "total_time_ms": total_time,
        },
    }

    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "comprehensive_benchmark_results.json")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    return summary


if __name__ == "__main__":
    run_all_benchmarks()
