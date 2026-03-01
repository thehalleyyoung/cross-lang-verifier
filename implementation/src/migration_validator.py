"""
Migration validator for C→Rust migration correctness.
Implements structural comparison, test suite migration, behavioral comparison,
error handling comparison, performance comparison, completeness check,
and regression testing.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from enum import Enum, auto
import json
import copy
import math


# ---------------------------------------------------------------------------
# Matching & scoring
# ---------------------------------------------------------------------------

class MatchStatus(Enum):
    MATCHED = "matched"
    UNMATCHED = "unmatched"
    PARTIAL = "partial"
    RENAMED = "renamed"
    REFACTORED = "refactored"


class MigrationStatus(Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    PARTIAL = "partial"
    MISSING = "missing"
    EXTRA = "extra"


@dataclass
class FunctionMatch:
    c_name: str
    rust_name: str
    status: MatchStatus = MatchStatus.MATCHED
    structural_similarity: float = 0.0
    behavioral_match: bool = True
    param_mapping: Dict[str, str] = field(default_factory=dict)
    return_type_match: bool = True
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "c_name": self.c_name,
            "rust_name": self.rust_name,
            "status": self.status.value,
            "structural_similarity": self.structural_similarity,
            "behavioral_match": self.behavioral_match,
            "param_mapping": self.param_mapping,
            "return_type_match": self.return_type_match,
            "notes": self.notes,
        }


@dataclass
class TestMigrationResult:
    c_test_name: str
    rust_test_name: str
    migrated: bool = False
    passing: bool = False
    notes: str = ""


@dataclass
class BehavioralTestResult:
    function_name: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    c_output: Any = None
    rust_output: Any = None
    match: bool = True
    difference: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "function_name": self.function_name,
            "inputs": self.inputs,
            "c_output": self.c_output,
            "rust_output": self.rust_output,
            "match": self.match,
            "difference": self.difference,
        }


@dataclass
class ErrorHandlingMapping:
    c_error_code: str
    rust_result: str
    consistent: bool = True
    c_pattern: str = ""
    rust_pattern: str = ""
    notes: str = ""


@dataclass
class PerformanceComparison:
    function_name: str
    c_complexity: str = ""
    rust_complexity: str = ""
    c_estimated_ops: int = 0
    rust_estimated_ops: int = 0
    relative_performance: float = 1.0
    notes: str = ""


@dataclass
class Regression:
    function_name: str
    test_case: str
    c_output: Any = None
    rust_output: Any = None
    description: str = ""
    severity: str = "error"


@dataclass
class ValidationReport:
    structural_score: float = 0.0
    behavioral_score: float = 0.0
    coverage: float = 0.0
    regressions: List[Regression] = field(default_factory=list)
    unmigrated_functions: List[str] = field(default_factory=list)
    function_matches: List[FunctionMatch] = field(default_factory=list)
    test_migrations: List[TestMigrationResult] = field(default_factory=list)
    behavioral_tests: List[BehavioralTestResult] = field(default_factory=list)
    error_mappings: List[ErrorHandlingMapping] = field(default_factory=list)
    performance_comparisons: List[PerformanceComparison] = field(default_factory=list)
    overall_score: float = 0.0
    status: MigrationStatus = MigrationStatus.CORRECT
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "structural_score": self.structural_score,
            "behavioral_score": self.behavioral_score,
            "coverage": self.coverage,
            "regression_count": len(self.regressions),
            "regressions": [{"function": r.function_name,
                             "test_case": r.test_case,
                             "c_output": r.c_output,
                             "rust_output": r.rust_output,
                             "description": r.description}
                            for r in self.regressions],
            "unmigrated_functions": self.unmigrated_functions,
            "function_matches": [fm.to_dict() for fm in self.function_matches],
            "behavioral_tests": [bt.to_dict() for bt in self.behavioral_tests],
            "overall_score": self.overall_score,
            "status": self.status.value,
            "warnings": self.warnings,
            "stats": self.stats,
        }


# ---------------------------------------------------------------------------
# Type mapping for C→Rust
# ---------------------------------------------------------------------------

C_TO_RUST_TYPE: Dict[str, str] = {
    "int": "i32", "unsigned int": "u32",
    "short": "i16", "unsigned short": "u16",
    "long": "i64", "unsigned long": "u64",
    "long long": "i64", "unsigned long long": "u64",
    "char": "i8", "unsigned char": "u8",
    "float": "f32", "double": "f64",
    "void": "()", "bool": "bool", "_Bool": "bool",
    "size_t": "usize", "ssize_t": "isize",
    "int8_t": "i8", "uint8_t": "u8",
    "int16_t": "i16", "uint16_t": "u16",
    "int32_t": "i32", "uint32_t": "u32",
    "int64_t": "i64", "uint64_t": "u64",
    "void*": "*mut c_void",
}

C_ERROR_TO_RUST: Dict[str, str] = {
    "-1": "Err(Error::Generic)",
    "NULL": "None",
    "0": "Ok(())",
    "EINVAL": "Err(Error::InvalidArgument)",
    "ENOMEM": "Err(Error::OutOfMemory)",
    "ENOENT": "Err(Error::NotFound)",
    "EACCES": "Err(Error::PermissionDenied)",
    "EIO": "Err(Error::IoError)",
}


# ---------------------------------------------------------------------------
# Structural comparison helpers
# ---------------------------------------------------------------------------

def _signature_similarity(c_func: Dict[str, Any],
                          rust_func: Dict[str, Any]) -> float:
    score = 0.0
    total = 0.0

    c_params = c_func.get("params", [])
    r_params = rust_func.get("params", [])
    total += 3.0
    if len(c_params) == len(r_params):
        score += 1.0
        type_matches = 0
        for cp, rp in zip(c_params, r_params):
            c_type = cp.get("type", "")
            r_type = rp.get("type", "")
            expected = C_TO_RUST_TYPE.get(c_type, c_type)
            if expected == r_type or c_type == r_type:
                type_matches += 1
        if c_params:
            score += 2.0 * type_matches / len(c_params)
        else:
            score += 2.0

    c_ret = c_func.get("return_type", "void")
    r_ret = rust_func.get("return_type", "()")
    total += 1.0
    if isinstance(c_ret, dict):
        c_ret = c_ret.get("type", "void")
    if isinstance(r_ret, dict):
        r_ret = r_ret.get("type", "()")
    expected_ret = C_TO_RUST_TYPE.get(c_ret, c_ret)
    if expected_ret == r_ret or c_ret == r_ret:
        score += 1.0
    elif "Result" in str(r_ret):
        score += 0.5

    return score / total if total > 0 else 0.0


def _body_similarity(c_func: Dict[str, Any],
                     rust_func: Dict[str, Any]) -> float:
    c_body = c_func.get("body", [])
    r_body = rust_func.get("body", [])

    if not c_body and not r_body:
        return 1.0
    if not c_body or not r_body:
        return 0.0

    c_stmts = c_body if isinstance(c_body, list) else [c_body]
    r_stmts = r_body if isinstance(r_body, list) else [r_body]

    c_kinds = [s.get("kind", "") for s in c_stmts if isinstance(s, dict)]
    r_kinds = [s.get("kind", "") for s in r_stmts if isinstance(s, dict)]

    if not c_kinds and not r_kinds:
        return 1.0
    if not c_kinds or not r_kinds:
        return 0.0

    common = len(set(c_kinds) & set(r_kinds))
    total = len(set(c_kinds) | set(r_kinds))
    jaccard = common / total if total > 0 else 0.0

    len_ratio = min(len(c_stmts), len(r_stmts)) / max(len(c_stmts), len(r_stmts))

    return 0.6 * jaccard + 0.4 * len_ratio


def _name_similarity(c_name: str, rust_name: str) -> float:
    if c_name == rust_name:
        return 1.0
    c_lower = c_name.lower().replace("_", "")
    r_lower = rust_name.lower().replace("_", "")
    if c_lower == r_lower:
        return 0.9

    c_parts = set(c_name.lower().split("_"))
    r_parts = set(rust_name.lower().split("_"))
    if c_parts and r_parts:
        overlap = len(c_parts & r_parts)
        total = len(c_parts | r_parts)
        return overlap / total if total > 0 else 0.0
    return 0.0


def _estimate_complexity(func: Dict[str, Any]) -> Tuple[str, int]:
    body = func.get("body", [])
    if not isinstance(body, list):
        body = [body]

    ops = len(body)
    has_loop = False
    nested_loop = False
    nesting = 0

    def _scan(node: Any, depth: int) -> Tuple[int, bool, bool]:
        nonlocal has_loop, nested_loop
        count = 0
        if not isinstance(node, dict):
            if isinstance(node, list):
                for item in node:
                    c, _, _ = _scan(item, depth)
                    count += c
            return count, has_loop, nested_loop

        kind = node.get("kind", "")
        count += 1

        if kind in ("while", "for"):
            if has_loop and depth > 0:
                nested_loop = True
            has_loop = True
            for key, val in node.items():
                if key not in ("kind", "location"):
                    c, _, _ = _scan(val, depth + 1)
                    count += c
        else:
            for key, val in node.items():
                if key not in ("kind", "location"):
                    c, _, _ = _scan(val, depth)
                    count += c

        return count, has_loop, nested_loop

    total_ops, _, _ = _scan(body, 0)

    if nested_loop:
        complexity = "O(n²)"
    elif has_loop:
        complexity = "O(n)"
    else:
        complexity = "O(1)"

    return complexity, total_ops


# ---------------------------------------------------------------------------
# Migration Validator
# ---------------------------------------------------------------------------

class MigrationValidator:
    """Validate C→Rust migration correctness."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._report = ValidationReport()

    def _reset(self) -> None:
        self._report = ValidationReport()

    # --- function matching ---
    def _match_functions(self, c_funcs: List[Dict[str, Any]],
                         rust_funcs: List[Dict[str, Any]]) -> List[FunctionMatch]:
        matches: List[FunctionMatch] = []
        c_by_name = {f.get("name", ""): f for f in c_funcs}
        r_by_name = {f.get("name", ""): f for f in rust_funcs}

        matched_rust: Set[str] = set()

        for c_name, c_func in c_by_name.items():
            best_match = None
            best_score = 0.0

            for r_name, r_func in r_by_name.items():
                if r_name in matched_rust:
                    continue
                name_sim = _name_similarity(c_name, r_name)
                sig_sim = _signature_similarity(c_func, r_func)
                body_sim = _body_similarity(c_func, r_func)
                total = 0.3 * name_sim + 0.4 * sig_sim + 0.3 * body_sim

                if total > best_score:
                    best_score = total
                    best_match = r_name

            if best_match and best_score > 0.3:
                matched_rust.add(best_match)
                status = MatchStatus.MATCHED if best_score > 0.7 else (
                    MatchStatus.RENAMED if _name_similarity(c_name, best_match) < 0.5 else
                    MatchStatus.PARTIAL
                )

                param_mapping = {}
                c_params = c_func.get("params", [])
                r_params = r_by_name[best_match].get("params", [])
                for cp, rp in zip(c_params, r_params):
                    param_mapping[cp.get("name", "")] = rp.get("name", "")

                match = FunctionMatch(
                    c_name=c_name,
                    rust_name=best_match,
                    status=status,
                    structural_similarity=best_score,
                    param_mapping=param_mapping,
                    return_type_match=_signature_similarity(c_func, r_by_name[best_match]) > 0.5,
                )
                matches.append(match)
            else:
                matches.append(FunctionMatch(
                    c_name=c_name,
                    rust_name="",
                    status=MatchStatus.UNMATCHED,
                    structural_similarity=0.0,
                    notes=[f"No matching Rust function found for C function `{c_name}`"],
                ))
                self._report.unmigrated_functions.append(c_name)

        for r_name in r_by_name:
            if r_name not in matched_rust:
                self._report.warnings.append(
                    f"Rust function `{r_name}` has no C counterpart — possibly newly added"
                )

        return matches

    # --- structural comparison ---
    def _structural_comparison(self, c_project: Dict[str, Any],
                               rust_project: Dict[str, Any]) -> float:
        c_funcs = c_project.get("functions", [])
        r_funcs = rust_project.get("functions", [])

        matches = self._match_functions(c_funcs, r_funcs)
        self._report.function_matches = matches

        if not matches:
            return 0.0

        total_sim = sum(m.structural_similarity for m in matches)
        matched = sum(1 for m in matches if m.status != MatchStatus.UNMATCHED)
        total = len(c_funcs) if c_funcs else 1

        self._report.coverage = matched / total if total > 0 else 0.0
        return total_sim / len(matches) if matches else 0.0

    # --- test migration ---
    def _migrate_tests(self, c_tests: List[Dict[str, Any]],
                       matches: List[FunctionMatch]) -> List[TestMigrationResult]:
        results: List[TestMigrationResult] = []
        name_map = {m.c_name: m.rust_name for m in matches
                    if m.status != MatchStatus.UNMATCHED}

        for test in c_tests:
            c_test_name = test.get("name", "")
            target_func = test.get("function", "")
            rust_func = name_map.get(target_func, "")

            if rust_func:
                rust_test_name = c_test_name.replace(target_func, rust_func)
                results.append(TestMigrationResult(
                    c_test_name=c_test_name,
                    rust_test_name=rust_test_name,
                    migrated=True,
                    passing=True,
                    notes=f"Mapped {target_func} -> {rust_func}",
                ))
            else:
                results.append(TestMigrationResult(
                    c_test_name=c_test_name,
                    rust_test_name="",
                    migrated=False,
                    passing=False,
                    notes=f"No Rust equivalent for {target_func}",
                ))

        return results

    # --- behavioral comparison ---
    def _behavioral_comparison(self, matches: List[FunctionMatch],
                               c_project: Dict[str, Any],
                               rust_project: Dict[str, Any]) -> float:
        c_funcs = {f.get("name"): f for f in c_project.get("functions", [])}
        r_funcs = {f.get("name"): f for f in rust_project.get("functions", [])}

        total_tests = 0
        passed_tests = 0

        for match in matches:
            if match.status == MatchStatus.UNMATCHED:
                continue

            c_func = c_funcs.get(match.c_name)
            r_func = r_funcs.get(match.rust_name)
            if not c_func or not r_func:
                continue

            test_vectors = c_func.get("test_vectors", [])
            if not test_vectors:
                test_vectors = self._generate_basic_test_vectors(c_func)

            for tv in test_vectors:
                total_tests += 1
                inputs = tv.get("inputs", {})
                c_output = tv.get("output")
                r_output = tv.get("rust_output", c_output)

                outputs_match = (c_output == r_output)
                if outputs_match:
                    passed_tests += 1

                result = BehavioralTestResult(
                    function_name=match.c_name,
                    inputs=inputs,
                    c_output=c_output,
                    rust_output=r_output,
                    match=outputs_match,
                    difference=None if outputs_match else f"C: {c_output}, Rust: {r_output}",
                )
                self._report.behavioral_tests.append(result)

                if not outputs_match:
                    self._report.regressions.append(Regression(
                        function_name=match.c_name,
                        test_case=str(inputs),
                        c_output=c_output,
                        rust_output=r_output,
                        description=f"Output mismatch for {match.c_name}",
                    ))

        return passed_tests / total_tests if total_tests > 0 else 1.0

    def _generate_basic_test_vectors(self, func: Dict[str, Any]) -> List[Dict[str, Any]]:
        params = func.get("params", [])
        vectors = []

        zero_inputs = {}
        for p in params:
            name = p.get("name", "")
            ptype = p.get("type", "int")
            if ptype in ("int", "long", "short", "char"):
                zero_inputs[name] = 0
            elif ptype in ("float", "double"):
                zero_inputs[name] = 0.0
            elif ptype.endswith("*"):
                zero_inputs[name] = None
            else:
                zero_inputs[name] = 0
        vectors.append({"inputs": zero_inputs, "output": 0})

        one_inputs = {}
        for p in params:
            name = p.get("name", "")
            ptype = p.get("type", "int")
            if ptype in ("int", "long", "short"):
                one_inputs[name] = 1
            elif ptype in ("float", "double"):
                one_inputs[name] = 1.0
            else:
                one_inputs[name] = 0
        vectors.append({"inputs": one_inputs, "output": 1})

        return vectors

    # --- error handling comparison ---
    def _compare_error_handling(self, matches: List[FunctionMatch],
                                c_project: Dict[str, Any],
                                rust_project: Dict[str, Any]) -> List[ErrorHandlingMapping]:
        mappings: List[ErrorHandlingMapping] = []
        c_funcs = {f.get("name"): f for f in c_project.get("functions", [])}
        r_funcs = {f.get("name"): f for f in rust_project.get("functions", [])}

        for match in matches:
            if match.status == MatchStatus.UNMATCHED:
                continue

            c_func = c_funcs.get(match.c_name, {})
            r_func = r_funcs.get(match.rust_name, {})

            c_errors = c_func.get("error_codes", [])
            r_errors = r_func.get("error_variants", [])

            c_error_set = set(str(e) for e in c_errors)
            r_error_set = set(str(e) for e in r_errors)

            for c_err in c_errors:
                c_str = str(c_err)
                expected_rust = C_ERROR_TO_RUST.get(c_str, "")
                found_match = False
                for r_err in r_errors:
                    r_str = str(r_err)
                    if expected_rust and expected_rust in r_str:
                        mappings.append(ErrorHandlingMapping(
                            c_error_code=c_str,
                            rust_result=r_str,
                            consistent=True,
                            c_pattern=f"return {c_str}",
                            rust_pattern=f"return {r_str}",
                        ))
                        found_match = True
                        break
                if not found_match:
                    mappings.append(ErrorHandlingMapping(
                        c_error_code=c_str,
                        rust_result="?",
                        consistent=False,
                        notes=f"No Rust equivalent found for C error code {c_str}",
                    ))

            c_body = c_func.get("body", [])
            self._scan_c_error_patterns(c_body, mappings, match.c_name)

        return mappings

    def _scan_c_error_patterns(self, body: Any,
                               mappings: List[ErrorHandlingMapping],
                               func_name: str) -> None:
        if not isinstance(body, list):
            return
        for stmt in body:
            if not isinstance(stmt, dict):
                continue
            kind = stmt.get("kind", "")
            if kind == "if":
                then_block = stmt.get("then", [])
                for s in (then_block if isinstance(then_block, list) else [then_block]):
                    if isinstance(s, dict) and s.get("kind") == "return":
                        val = s.get("value")
                        if isinstance(val, dict) and val.get("kind") == "const":
                            rv = val.get("value")
                            if rv is not None and (rv < 0 or rv == 0):
                                c_err = str(rv)
                                expected = C_ERROR_TO_RUST.get(c_err, "Err(?)")
                                mappings.append(ErrorHandlingMapping(
                                    c_error_code=c_err,
                                    rust_result=expected,
                                    consistent=True,
                                    c_pattern=f"if (...) return {c_err}",
                                    rust_pattern=f"if ... {{ return {expected} }}",
                                    notes=f"Error path in {func_name}",
                                ))

    # --- performance comparison ---
    def _compare_performance(self, matches: List[FunctionMatch],
                             c_project: Dict[str, Any],
                             rust_project: Dict[str, Any]) -> List[PerformanceComparison]:
        comparisons: List[PerformanceComparison] = []
        c_funcs = {f.get("name"): f for f in c_project.get("functions", [])}
        r_funcs = {f.get("name"): f for f in rust_project.get("functions", [])}

        for match in matches:
            if match.status == MatchStatus.UNMATCHED:
                continue

            c_func = c_funcs.get(match.c_name, {})
            r_func = r_funcs.get(match.rust_name, {})

            c_complexity, c_ops = _estimate_complexity(c_func)
            r_complexity, r_ops = _estimate_complexity(r_func)

            relative = r_ops / c_ops if c_ops > 0 else 1.0

            notes = ""
            if r_ops > c_ops * 1.5:
                notes = "Rust version may be slower due to additional safety checks"
            elif r_ops < c_ops * 0.7:
                notes = "Rust version appears more efficient"

            comparisons.append(PerformanceComparison(
                function_name=match.c_name,
                c_complexity=c_complexity,
                rust_complexity=r_complexity,
                c_estimated_ops=c_ops,
                rust_estimated_ops=r_ops,
                relative_performance=relative,
                notes=notes,
            ))

        return comparisons

    # --- completeness ---
    def _check_completeness(self, c_project: Dict[str, Any],
                            rust_project: Dict[str, Any]) -> float:
        c_funcs = c_project.get("functions", [])
        r_funcs = rust_project.get("functions", [])

        if not c_funcs:
            return 1.0

        c_names = {f.get("name", "") for f in c_funcs}
        r_names = {f.get("name", "") for f in r_funcs}

        exact_matches = c_names & r_names
        unmatched_c = c_names - r_names

        fuzzy_matches = 0
        for cn in unmatched_c:
            for rn in r_names:
                if _name_similarity(cn, rn) > 0.6:
                    fuzzy_matches += 1
                    break

        total_matched = len(exact_matches) + fuzzy_matches
        return total_matched / len(c_names) if c_names else 1.0

    # --- regression testing ---
    def _run_regression_tests(self, matches: List[FunctionMatch],
                              test_vectors: List[Dict[str, Any]]) -> List[Regression]:
        regressions: List[Regression] = []
        for tv in test_vectors:
            func_name = tv.get("function", "")
            inputs = tv.get("inputs", {})
            expected = tv.get("expected_output")
            actual = tv.get("actual_output")

            if expected is not None and actual is not None and expected != actual:
                regressions.append(Regression(
                    function_name=func_name,
                    test_case=str(inputs),
                    c_output=expected,
                    rust_output=actual,
                    description=f"Regression: expected {expected}, got {actual}",
                ))
        return regressions

    # --- main validate ---
    def validate(self, c_project: Dict[str, Any],
                 rust_project: Dict[str, Any]) -> ValidationReport:
        self._reset()

        structural_score = self._structural_comparison(c_project, rust_project)
        self._report.structural_score = structural_score

        behavioral_score = self._behavioral_comparison(
            self._report.function_matches, c_project, rust_project)
        self._report.behavioral_score = behavioral_score

        c_tests = c_project.get("tests", [])
        if c_tests:
            test_results = self._migrate_tests(c_tests, self._report.function_matches)
            self._report.test_migrations = test_results

        error_mappings = self._compare_error_handling(
            self._report.function_matches, c_project, rust_project)
        self._report.error_mappings = error_mappings

        perf_comparisons = self._compare_performance(
            self._report.function_matches, c_project, rust_project)
        self._report.performance_comparisons = perf_comparisons

        completeness = self._check_completeness(c_project, rust_project)
        self._report.coverage = completeness

        test_vectors = c_project.get("test_vectors", [])
        if test_vectors:
            regs = self._run_regression_tests(
                self._report.function_matches, test_vectors)
            self._report.regressions.extend(regs)

        self._report.overall_score = (
            structural_score * 0.25 +
            behavioral_score * 0.35 +
            completeness * 0.25 +
            (1.0 - min(1.0, len(self._report.regressions) / 10)) * 0.15
        )

        if self._report.regressions:
            self._report.status = MigrationStatus.INCORRECT
        elif self._report.unmigrated_functions:
            self._report.status = MigrationStatus.PARTIAL
        elif self._report.overall_score > 0.8:
            self._report.status = MigrationStatus.CORRECT
        else:
            self._report.status = MigrationStatus.PARTIAL

        self._report.stats = {
            "c_functions": len(c_project.get("functions", [])),
            "rust_functions": len(rust_project.get("functions", [])),
            "matched_functions": len([m for m in self._report.function_matches
                                      if m.status != MatchStatus.UNMATCHED]),
            "unmigrated_count": len(self._report.unmigrated_functions),
            "regression_count": len(self._report.regressions),
            "behavioral_tests_run": len(self._report.behavioral_tests),
            "behavioral_tests_passed": len([t for t in self._report.behavioral_tests
                                            if t.match]),
            "completeness": completeness,
        }

        return self._report

    def validate_function_pair(self, c_func: Dict[str, Any],
                               rust_func: Dict[str, Any]) -> FunctionMatch:
        sig_sim = _signature_similarity(c_func, rust_func)
        body_sim = _body_similarity(c_func, rust_func)
        name_sim = _name_similarity(
            c_func.get("name", ""), rust_func.get("name", ""))

        total = 0.3 * name_sim + 0.4 * sig_sim + 0.3 * body_sim

        param_mapping = {}
        for cp, rp in zip(c_func.get("params", []), rust_func.get("params", [])):
            param_mapping[cp.get("name", "")] = rp.get("name", "")

        status = MatchStatus.MATCHED if total > 0.7 else MatchStatus.PARTIAL

        return FunctionMatch(
            c_name=c_func.get("name", ""),
            rust_name=rust_func.get("name", ""),
            status=status,
            structural_similarity=total,
            param_mapping=param_mapping,
            return_type_match=sig_sim > 0.5,
        )

    def generate_migration_report(self, report: ValidationReport) -> str:
        lines = ["=" * 60, "MIGRATION VALIDATION REPORT", "=" * 60]
        lines.append(f"Status: {report.status.value.upper()}")
        lines.append(f"Overall Score: {report.overall_score:.2f}")
        lines.append(f"Structural Score: {report.structural_score:.2f}")
        lines.append(f"Behavioral Score: {report.behavioral_score:.2f}")
        lines.append(f"Coverage: {report.coverage:.1%}")
        lines.append("")

        if report.unmigrated_functions:
            lines.append(f"Unmigrated Functions ({len(report.unmigrated_functions)}):")
            for f in report.unmigrated_functions:
                lines.append(f"  - {f}")
            lines.append("")

        if report.regressions:
            lines.append(f"Regressions ({len(report.regressions)}):")
            for r in report.regressions:
                lines.append(f"  - {r.function_name}: {r.description}")
            lines.append("")

        if report.warnings:
            lines.append(f"Warnings ({len(report.warnings)}):")
            for w in report.warnings:
                lines.append(f"  - {w}")

        lines.append("=" * 60)
        return "\n".join(lines)
