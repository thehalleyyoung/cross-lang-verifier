"""
Code complexity comparator for C and Rust.
Implements cyclomatic complexity, cognitive complexity, error handling complexity,
memory management complexity, abstraction level, safety metric, and idiomatic score.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from enum import Enum, auto
import math


# ---------------------------------------------------------------------------
# Complexity metrics
# ---------------------------------------------------------------------------

@dataclass
class CyclomaticResult:
    value: int = 1
    decision_points: List[str] = field(default_factory=list)
    per_function: Dict[str, int] = field(default_factory=dict)


@dataclass
class CognitiveResult:
    value: int = 0
    nesting_penalties: int = 0
    structural_increments: int = 0
    fundamental_increments: int = 0
    hybrid_increments: int = 0
    detail: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ErrorHandlingMetrics:
    error_paths: int = 0
    error_checks: int = 0
    propagation_depth: int = 0
    error_handling_style: str = ""
    error_codes_used: List[str] = field(default_factory=list)
    panic_points: int = 0
    complexity_score: float = 0.0


@dataclass
class MemoryManagementMetrics:
    allocations: int = 0
    deallocations: int = 0
    unmatched_allocs: int = 0
    manual_management_points: int = 0
    raii_resources: int = 0
    smart_pointers: int = 0
    raw_pointers: int = 0
    unsafe_blocks: int = 0
    complexity_score: float = 0.0


@dataclass
class AbstractionMetrics:
    indirections: int = 0
    function_pointers: int = 0
    generics: int = 0
    traits_interfaces: int = 0
    macros: int = 0
    type_parameters: int = 0
    closures: int = 0
    higher_order_functions: int = 0
    abstraction_level: float = 0.0


@dataclass
class SafetyMetrics:
    total_operations: int = 0
    unsafe_operations: int = 0
    safe_operations: int = 0
    unsafe_ratio: float = 0.0
    pointer_arithmetic: int = 0
    unchecked_casts: int = 0
    raw_memory_access: int = 0
    bounds_checks: int = 0
    null_checks: int = 0
    safety_score: float = 0.0


@dataclass
class IdiomaticScore:
    score: float = 0.0
    max_score: float = 100.0
    issues: List[str] = field(default_factory=list)
    good_practices: List[str] = field(default_factory=list)
    language: str = ""


@dataclass
class LanguageComplexity:
    language: str = ""
    lines_of_code: int = 0
    cyclomatic: CyclomaticResult = field(default_factory=CyclomaticResult)
    cognitive: CognitiveResult = field(default_factory=CognitiveResult)
    error_handling: ErrorHandlingMetrics = field(default_factory=ErrorHandlingMetrics)
    memory_management: MemoryManagementMetrics = field(default_factory=MemoryManagementMetrics)
    abstraction: AbstractionMetrics = field(default_factory=AbstractionMetrics)
    safety: SafetyMetrics = field(default_factory=SafetyMetrics)
    idiomatic: IdiomaticScore = field(default_factory=IdiomaticScore)

    def overall_complexity(self) -> float:
        return (
            self.cyclomatic.value * 1.0 +
            self.cognitive.value * 0.5 +
            self.error_handling.complexity_score * 0.8 +
            self.memory_management.complexity_score * 1.2 +
            (1.0 - self.safety.safety_score) * 10.0
        )


@dataclass
class ComparisonReport:
    c_complexity: LanguageComplexity = field(default_factory=lambda: LanguageComplexity(language="c"))
    rust_complexity: LanguageComplexity = field(default_factory=lambda: LanguageComplexity(language="rust"))
    safety_comparison: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "c_complexity": {
                "cyclomatic": self.c_complexity.cyclomatic.value,
                "cognitive": self.c_complexity.cognitive.value,
                "error_handling": self.c_complexity.error_handling.complexity_score,
                "memory_management": self.c_complexity.memory_management.complexity_score,
                "safety_score": self.c_complexity.safety.safety_score,
                "idiomatic_score": self.c_complexity.idiomatic.score,
                "overall": self.c_complexity.overall_complexity(),
            },
            "rust_complexity": {
                "cyclomatic": self.rust_complexity.cyclomatic.value,
                "cognitive": self.rust_complexity.cognitive.value,
                "error_handling": self.rust_complexity.error_handling.complexity_score,
                "memory_management": self.rust_complexity.memory_management.complexity_score,
                "safety_score": self.rust_complexity.safety.safety_score,
                "idiomatic_score": self.rust_complexity.idiomatic.score,
                "overall": self.rust_complexity.overall_complexity(),
            },
            "safety_comparison": self.safety_comparison,
            "recommendations": self.recommendations,
            "summary": self.summary,
        }


# ---------------------------------------------------------------------------
# Complexity Comparator
# ---------------------------------------------------------------------------

class ComplexityComparator:
    """Compare code complexity between C and Rust implementations."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    # --- cyclomatic complexity ---
    def _compute_cyclomatic(self, ast_node: Any, language: str) -> CyclomaticResult:
        result = CyclomaticResult(value=1)
        self._count_decisions(ast_node, result, language)
        return result

    def _count_decisions(self, node: Any, result: CyclomaticResult,
                         language: str) -> None:
        if not isinstance(node, dict):
            if isinstance(node, list):
                for item in node:
                    self._count_decisions(item, result, language)
            return

        kind = node.get("kind", "")

        decision_kinds = {"if", "while", "for", "case", "catch",
                          "&&", "||", "ternary"}

        if kind in decision_kinds:
            result.value += 1
            result.decision_points.append(
                f"{kind} at {node.get('location', '?')}")

        if kind == "match" and language == "rust":
            arms = node.get("arms", [])
            result.value += max(0, len(arms) - 1)
            result.decision_points.append(
                f"match with {len(arms)} arms at {node.get('location', '?')}")

        if kind == "switch" and language == "c":
            cases = node.get("cases", [])
            result.value += len(cases)
            result.decision_points.append(
                f"switch with {len(cases)} cases at {node.get('location', '?')}")

        if kind == "binop" and node.get("op") in ("&&", "||"):
            result.value += 1
            result.decision_points.append(
                f"{node.get('op')} at {node.get('location', '?')}")

        if kind == "if" and node.get("else_if"):
            result.value += 1

        for key, val in node.items():
            if key in ("kind", "location", "op", "name", "type", "value"):
                continue
            self._count_decisions(val, result, language)

    # --- cognitive complexity ---
    def _compute_cognitive(self, ast_node: Any, language: str) -> CognitiveResult:
        result = CognitiveResult()
        self._assess_cognitive(ast_node, 0, result, language)
        return result

    def _assess_cognitive(self, node: Any, nesting: int,
                          result: CognitiveResult, language: str) -> None:
        if not isinstance(node, dict):
            if isinstance(node, list):
                for item in node:
                    self._assess_cognitive(item, nesting, result, language)
            return

        kind = node.get("kind", "")

        # Structural increments + nesting penalty
        if kind in ("if", "while", "for", "switch", "match", "try", "catch"):
            increment = 1 + nesting
            result.value += increment
            result.structural_increments += 1
            result.nesting_penalties += nesting
            result.detail.append({
                "kind": kind,
                "nesting": nesting,
                "increment": increment,
                "location": node.get("location", "?"),
            })

        nesting_delta = 0
        if kind in ("if", "while", "for", "switch", "match", "try", "catch",
                     "closure", "lambda", "nested_func"):
            nesting_delta = 1

        # Fundamental increments (no nesting penalty)
        if kind == "binop" and node.get("op") in ("&&", "||"):
            result.value += 1
            result.fundamental_increments += 1

        if kind == "goto":
            result.value += 1
            result.fundamental_increments += 1

        if kind == "break" and node.get("label"):
            result.value += 1
            result.fundamental_increments += 1

        if kind == "continue" and node.get("label"):
            result.value += 1
            result.fundamental_increments += 1

        if kind in ("else", "else_if"):
            result.value += 1
            result.hybrid_increments += 1

        if kind == "recursion":
            result.value += 1
            result.fundamental_increments += 1

        for key, val in node.items():
            if key in ("kind", "location", "op", "name", "type", "value"):
                continue
            self._assess_cognitive(val, nesting + nesting_delta, result, language)

    # --- error handling complexity ---
    def _analyze_error_handling_c(self, ast_node: Any) -> ErrorHandlingMetrics:
        metrics = ErrorHandlingMetrics(error_handling_style="error_codes")
        self._scan_c_error_handling(ast_node, metrics)
        metrics.complexity_score = (
            metrics.error_paths * 1.5 +
            metrics.error_checks * 0.5 +
            metrics.propagation_depth * 2.0
        )
        return metrics

    def _scan_c_error_handling(self, node: Any, metrics: ErrorHandlingMetrics) -> None:
        if not isinstance(node, dict):
            if isinstance(node, list):
                for item in node:
                    self._scan_c_error_handling(item, metrics)
            return

        kind = node.get("kind", "")

        if kind == "if":
            cond = node.get("cond", {})
            if isinstance(cond, dict):
                if cond.get("kind") == "binop" and cond.get("op") in ("==", "!=", "<"):
                    right = cond.get("right", {})
                    if isinstance(right, dict) and right.get("kind") == "const":
                        val = right.get("value")
                        if val in (0, -1, None):
                            metrics.error_checks += 1
                            then_block = node.get("then", [])
                            for s in (then_block if isinstance(then_block, list) else [then_block]):
                                if isinstance(s, dict) and s.get("kind") == "return":
                                    metrics.error_paths += 1

        if kind == "return":
            val = node.get("value")
            if isinstance(val, dict) and val.get("kind") == "const":
                rv = val.get("value")
                if rv is not None and rv < 0:
                    metrics.error_paths += 1
                    metrics.error_codes_used.append(str(rv))

        if kind == "call" and node.get("func") in ("perror", "fprintf", "exit", "abort"):
            metrics.error_paths += 1

        if kind == "goto":
            label = node.get("label", "")
            if "err" in label.lower() or "error" in label.lower() or "cleanup" in label.lower():
                metrics.error_paths += 1
                metrics.propagation_depth += 1

        for key, val in node.items():
            if key in ("kind", "location", "op", "name", "type", "value"):
                continue
            self._scan_c_error_handling(val, metrics)

    def _analyze_error_handling_rust(self, ast_node: Any) -> ErrorHandlingMetrics:
        metrics = ErrorHandlingMetrics(error_handling_style="Result/Option")
        self._scan_rust_error_handling(ast_node, metrics)
        metrics.complexity_score = (
            metrics.error_paths * 0.5 +
            metrics.error_checks * 0.3 +
            metrics.panic_points * 3.0
        )
        return metrics

    def _scan_rust_error_handling(self, node: Any,
                                  metrics: ErrorHandlingMetrics) -> None:
        if not isinstance(node, dict):
            if isinstance(node, list):
                for item in node:
                    self._scan_rust_error_handling(item, metrics)
            return

        kind = node.get("kind", "")

        if kind == "try_operator" or node.get("op") == "?":
            metrics.error_checks += 1
            metrics.propagation_depth += 1

        if kind == "match":
            scrutinee_type = node.get("scrutinee_type", "")
            if "Result" in scrutinee_type or "Option" in scrutinee_type:
                metrics.error_checks += 1
                arms = node.get("arms", [])
                for arm in arms:
                    pattern = arm.get("pattern", "")
                    if isinstance(pattern, str) and pattern in ("Err", "None"):
                        metrics.error_paths += 1

        if kind == "call":
            func = node.get("func", "")
            if func in ("unwrap", "expect"):
                metrics.panic_points += 1
            if func in ("unwrap_or", "unwrap_or_else", "unwrap_or_default"):
                metrics.error_checks += 1
            if func in ("map_err", "and_then", "or_else"):
                metrics.error_checks += 1

        if kind == "macro_call" and node.get("macro") in ("panic!", "todo!", "unimplemented!"):
            metrics.panic_points += 1

        for key, val in node.items():
            if key in ("kind", "location", "op", "name", "type", "value"):
                continue
            self._scan_rust_error_handling(val, metrics)

    # --- memory management complexity ---
    def _analyze_memory_c(self, ast_node: Any) -> MemoryManagementMetrics:
        metrics = MemoryManagementMetrics()
        self._scan_c_memory(ast_node, metrics)
        metrics.unmatched_allocs = abs(metrics.allocations - metrics.deallocations)
        metrics.manual_management_points = metrics.allocations + metrics.deallocations
        metrics.complexity_score = (
            metrics.allocations * 2.0 +
            metrics.deallocations * 1.5 +
            metrics.unmatched_allocs * 5.0 +
            metrics.raw_pointers * 1.0
        )
        return metrics

    def _scan_c_memory(self, node: Any, metrics: MemoryManagementMetrics) -> None:
        if not isinstance(node, dict):
            if isinstance(node, list):
                for item in node:
                    self._scan_c_memory(item, metrics)
            return

        kind = node.get("kind", "")

        if kind == "call":
            func = node.get("func", "")
            if func in ("malloc", "calloc", "realloc", "aligned_alloc"):
                metrics.allocations += 1
            elif func in ("free",):
                metrics.deallocations += 1

        if kind == "decl":
            var_type = node.get("type", "")
            if "*" in var_type:
                metrics.raw_pointers += 1

        if kind == "binop" and node.get("op") in ("+", "-"):
            left = node.get("left", {})
            if isinstance(left, dict) and left.get("type", "").endswith("*"):
                metrics.raw_pointers += 1

        for key, val in node.items():
            if key in ("kind", "location", "op", "name", "type", "value"):
                continue
            self._scan_c_memory(val, metrics)

    def _analyze_memory_rust(self, ast_node: Any) -> MemoryManagementMetrics:
        metrics = MemoryManagementMetrics()
        self._scan_rust_memory(ast_node, metrics)
        metrics.complexity_score = (
            metrics.unsafe_blocks * 5.0 +
            metrics.raw_pointers * 3.0 +
            metrics.smart_pointers * 0.5 +
            metrics.raii_resources * 0.2
        )
        return metrics

    def _scan_rust_memory(self, node: Any,
                          metrics: MemoryManagementMetrics) -> None:
        if not isinstance(node, dict):
            if isinstance(node, list):
                for item in node:
                    self._scan_rust_memory(item, metrics)
            return

        kind = node.get("kind", "")

        if kind == "unsafe_block":
            metrics.unsafe_blocks += 1

        if kind == "call":
            func = node.get("func", "")
            if func in ("Box::new", "Rc::new", "Arc::new", "Vec::new"):
                metrics.smart_pointers += 1
                metrics.raii_resources += 1
            if "alloc" in func.lower():
                metrics.allocations += 1

        if kind == "decl":
            var_type = node.get("type", "")
            if var_type.startswith("*mut") or var_type.startswith("*const"):
                metrics.raw_pointers += 1
            if any(t in var_type for t in ("Box<", "Rc<", "Arc<", "Vec<", "String")):
                metrics.smart_pointers += 1
                metrics.raii_resources += 1

        for key, val in node.items():
            if key in ("kind", "location", "op", "name", "type", "value"):
                continue
            self._scan_rust_memory(val, metrics)

    # --- abstraction level ---
    def _analyze_abstraction(self, ast_node: Any,
                             language: str) -> AbstractionMetrics:
        metrics = AbstractionMetrics()
        self._scan_abstraction(ast_node, metrics, language)
        metrics.abstraction_level = (
            metrics.generics * 2.0 +
            metrics.traits_interfaces * 2.0 +
            metrics.closures * 1.5 +
            metrics.higher_order_functions * 1.5 +
            metrics.function_pointers * 1.0 +
            metrics.macros * 1.0
        )
        return metrics

    def _scan_abstraction(self, node: Any, metrics: AbstractionMetrics,
                          language: str) -> None:
        if not isinstance(node, dict):
            if isinstance(node, list):
                for item in node:
                    self._scan_abstraction(item, metrics, language)
            return

        kind = node.get("kind", "")

        if kind == "function_pointer" or (kind == "decl" and "(*" in node.get("type", "")):
            metrics.function_pointers += 1
            metrics.indirections += 1

        if kind == "generic" or node.get("type_params"):
            metrics.generics += 1
            tp = node.get("type_params", [])
            metrics.type_parameters += len(tp) if isinstance(tp, list) else 1

        if kind in ("trait_def", "trait_impl", "impl"):
            metrics.traits_interfaces += 1

        if kind in ("closure", "lambda"):
            metrics.closures += 1

        if kind == "call" and node.get("is_higher_order"):
            metrics.higher_order_functions += 1

        if kind == "call":
            func = node.get("func", "")
            if func in ("map", "filter", "fold", "reduce", "for_each",
                         "flat_map", "collect", "iter", "into_iter"):
                metrics.higher_order_functions += 1

        if kind == "macro_call" or kind == "macro_def":
            metrics.macros += 1

        if kind == "deref" or kind == "addr":
            metrics.indirections += 1

        for key, val in node.items():
            if key in ("kind", "location", "op", "name", "type", "value"):
                continue
            self._scan_abstraction(val, metrics, language)

    # --- safety metric ---
    def _analyze_safety(self, ast_node: Any, language: str) -> SafetyMetrics:
        metrics = SafetyMetrics()
        self._scan_safety(ast_node, metrics, language)
        if metrics.total_operations > 0:
            metrics.unsafe_ratio = metrics.unsafe_operations / metrics.total_operations
            metrics.safety_score = 1.0 - metrics.unsafe_ratio
        else:
            metrics.safety_score = 1.0
        return metrics

    def _scan_safety(self, node: Any, metrics: SafetyMetrics,
                     language: str) -> None:
        if not isinstance(node, dict):
            if isinstance(node, list):
                for item in node:
                    self._scan_safety(item, metrics, language)
            return

        kind = node.get("kind", "")
        metrics.total_operations += 1

        if language == "c":
            metrics.unsafe_operations += 1

            if kind in ("deref", "index"):
                metrics.raw_memory_access += 1
            if kind == "cast":
                metrics.unchecked_casts += 1
            if kind == "binop" and node.get("op") in ("+", "-"):
                left_type = node.get("left_type", "")
                if "*" in left_type:
                    metrics.pointer_arithmetic += 1

            if kind == "if":
                cond = node.get("cond", {})
                if isinstance(cond, dict):
                    if cond.get("kind") == "binop" and cond.get("op") in ("!=", "=="):
                        right = cond.get("right", {})
                        if isinstance(right, dict) and right.get("kind") == "const":
                            if right.get("value") == 0:
                                metrics.null_checks += 1
                                metrics.bounds_checks += 1

        elif language == "rust":
            if kind == "unsafe_block":
                metrics.unsafe_operations += 1
            else:
                metrics.safe_operations += 1

            if kind == "call":
                func = node.get("func", "")
                if func.endswith("_unchecked"):
                    metrics.unsafe_operations += 1
                if "unsafe" in node.get("attributes", []):
                    metrics.unsafe_operations += 1

            if kind == "deref" and node.get("raw_pointer"):
                metrics.unsafe_operations += 1
                metrics.raw_memory_access += 1

            if kind == "index":
                metrics.bounds_checks += 1

        for key, val in node.items():
            if key in ("kind", "location", "op", "name", "type", "value"):
                continue
            self._scan_safety(val, metrics, language)

    # --- idiomatic score ---
    def _compute_idiomatic_c(self, ast_node: Any) -> IdiomaticScore:
        score = IdiomaticScore(score=70.0, language="c")
        self._check_c_idioms(ast_node, score)
        score.score = max(0.0, min(100.0, score.score))
        return score

    def _check_c_idioms(self, node: Any, score: IdiomaticScore) -> None:
        if not isinstance(node, dict):
            if isinstance(node, list):
                for item in node:
                    self._check_c_idioms(item, score)
            return

        kind = node.get("kind", "")

        if kind == "goto":
            label = node.get("label", "")
            if "err" in label.lower() or "cleanup" in label.lower():
                score.good_practices.append("goto for error cleanup (idiomatic C)")
                score.score += 2
            else:
                score.issues.append("goto used for non-error control flow")
                score.score -= 5

        if kind == "call":
            func = node.get("func", "")
            if func in ("strcpy", "strcat", "sprintf", "gets"):
                score.issues.append(f"Unsafe function `{func}` used; prefer bounded version")
                score.score -= 10
            if func in ("strncpy", "snprintf", "strncat", "fgets"):
                score.good_practices.append(f"Bounded function `{func}` used")
                score.score += 3

        if kind == "decl":
            var_type = node.get("type", "")
            if var_type.startswith("const"):
                score.good_practices.append("const correctness")
                score.score += 1

        for key, val in node.items():
            if key in ("kind", "location", "op", "name", "type", "value"):
                continue
            self._check_c_idioms(val, score)

    def _compute_idiomatic_rust(self, ast_node: Any) -> IdiomaticScore:
        score = IdiomaticScore(score=70.0, language="rust")
        self._check_rust_idioms(ast_node, score)
        score.score = max(0.0, min(100.0, score.score))
        return score

    def _check_rust_idioms(self, node: Any, score: IdiomaticScore) -> None:
        if not isinstance(node, dict):
            if isinstance(node, list):
                for item in node:
                    self._check_rust_idioms(item, score)
            return

        kind = node.get("kind", "")

        if kind == "call":
            func = node.get("func", "")
            if func == "unwrap":
                score.issues.append("unwrap() used — consider proper error handling")
                score.score -= 5
            if func in ("map", "filter", "fold", "collect", "iter"):
                score.good_practices.append(f"Iterator method `{func}` used")
                score.score += 2
            if func in ("unwrap_or_else", "map_err"):
                score.good_practices.append(f"Proper error handling with `{func}`")
                score.score += 3

        if kind == "match":
            score.good_practices.append("Pattern matching used")
            score.score += 2

        if kind == "try_operator" or node.get("op") == "?":
            score.good_practices.append("? operator for error propagation")
            score.score += 3

        if kind == "unsafe_block":
            score.issues.append("unsafe block used")
            score.score -= 3

        if kind == "decl":
            if node.get("mutable"):
                pass
            else:
                score.good_practices.append("Immutable binding (default)")
                score.score += 0.5

        if kind == "clone":
            score.issues.append("Explicit clone — consider borrowing instead")
            score.score -= 2

        for key, val in node.items():
            if key in ("kind", "location", "op", "name", "type", "value"):
                continue
            self._check_rust_idioms(val, score)

    # --- line counting ---
    def _count_lines(self, ast_node: Any) -> int:
        if isinstance(ast_node, dict):
            loc = ast_node.get("loc", 0)
            if loc:
                return loc
            body = ast_node.get("body", [])
            if isinstance(body, list):
                return max(len(body) * 2, 1)
            return 1
        if isinstance(ast_node, list):
            return len(ast_node)
        return 0

    # --- main compare ---
    def compare(self, c_code: Any, rust_code: Any) -> ComparisonReport:
        report = ComparisonReport()

        # C analysis
        c = report.c_complexity
        c.language = "c"
        c.lines_of_code = self._count_lines(c_code)
        c.cyclomatic = self._compute_cyclomatic(c_code, "c")
        c.cognitive = self._compute_cognitive(c_code, "c")
        c.error_handling = self._analyze_error_handling_c(c_code)
        c.memory_management = self._analyze_memory_c(c_code)
        c.abstraction = self._analyze_abstraction(c_code, "c")
        c.safety = self._analyze_safety(c_code, "c")
        c.idiomatic = self._compute_idiomatic_c(c_code)

        # Rust analysis
        r = report.rust_complexity
        r.language = "rust"
        r.lines_of_code = self._count_lines(rust_code)
        r.cyclomatic = self._compute_cyclomatic(rust_code, "rust")
        r.cognitive = self._compute_cognitive(rust_code, "rust")
        r.error_handling = self._analyze_error_handling_rust(rust_code)
        r.memory_management = self._analyze_memory_rust(rust_code)
        r.abstraction = self._analyze_abstraction(rust_code, "rust")
        r.safety = self._analyze_safety(rust_code, "rust")
        r.idiomatic = self._compute_idiomatic_rust(rust_code)

        # Safety comparison
        report.safety_comparison = {
            "c_safety_score": c.safety.safety_score,
            "rust_safety_score": r.safety.safety_score,
            "c_unsafe_operations": c.safety.unsafe_operations,
            "rust_unsafe_operations": r.safety.unsafe_operations,
            "safety_improvement": r.safety.safety_score - c.safety.safety_score,
            "c_pointer_arithmetic": c.safety.pointer_arithmetic,
            "rust_pointer_arithmetic": r.safety.pointer_arithmetic,
        }

        # Recommendations
        if c.cyclomatic.value > r.cyclomatic.value:
            report.recommendations.append(
                "Rust version has lower cyclomatic complexity — simpler control flow"
            )
        elif c.cyclomatic.value < r.cyclomatic.value:
            report.recommendations.append(
                "C version has lower cyclomatic complexity — consider simplifying Rust"
            )

        if r.safety.safety_score > c.safety.safety_score:
            report.recommendations.append(
                f"Rust version is safer: safety score {r.safety.safety_score:.2f} vs C {c.safety.safety_score:.2f}"
            )

        if r.error_handling.panic_points > 0:
            report.recommendations.append(
                f"Rust code has {r.error_handling.panic_points} panic points — consider using Result instead"
            )

        if c.memory_management.unmatched_allocs > 0:
            report.recommendations.append(
                f"C code has {c.memory_management.unmatched_allocs} unmatched allocations — potential memory leaks"
            )

        if r.idiomatic.score < 50:
            report.recommendations.append(
                "Rust code has low idiomatic score — consider refactoring to use more Rust idioms"
            )

        report.summary = {
            "c_overall": c.overall_complexity(),
            "rust_overall": r.overall_complexity(),
            "complexity_ratio": (r.overall_complexity() / c.overall_complexity()
                                 if c.overall_complexity() > 0 else 0),
            "safety_improvement": r.safety.safety_score - c.safety.safety_score,
            "recommendation_count": len(report.recommendations),
        }

        return report

    def analyze_c(self, c_code: Any) -> LanguageComplexity:
        lc = LanguageComplexity(language="c")
        lc.lines_of_code = self._count_lines(c_code)
        lc.cyclomatic = self._compute_cyclomatic(c_code, "c")
        lc.cognitive = self._compute_cognitive(c_code, "c")
        lc.error_handling = self._analyze_error_handling_c(c_code)
        lc.memory_management = self._analyze_memory_c(c_code)
        lc.abstraction = self._analyze_abstraction(c_code, "c")
        lc.safety = self._analyze_safety(c_code, "c")
        lc.idiomatic = self._compute_idiomatic_c(c_code)
        return lc

    def analyze_rust(self, rust_code: Any) -> LanguageComplexity:
        lc = LanguageComplexity(language="rust")
        lc.lines_of_code = self._count_lines(rust_code)
        lc.cyclomatic = self._compute_cyclomatic(rust_code, "rust")
        lc.cognitive = self._compute_cognitive(rust_code, "rust")
        lc.error_handling = self._analyze_error_handling_rust(rust_code)
        lc.memory_management = self._analyze_memory_rust(rust_code)
        lc.abstraction = self._analyze_abstraction(rust_code, "rust")
        lc.safety = self._analyze_safety(rust_code, "rust")
        lc.idiomatic = self._compute_idiomatic_rust(rust_code)
        return lc
