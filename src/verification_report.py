"""
Verification report generation for cross-language equivalence checking.
Produces human-readable reports in Markdown, HTML, and JSON formats.
"""

import json
import html as html_mod
from datetime import datetime, timezone
from typing import Any

TOOL_VERSION = "0.3.0"

SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}
SEVERITY_EMOJI = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
STATUS_EMOJI = {"equivalent": "✅", "partial": "⚠️", "not_equivalent": "❌"}


class DivergenceExplainer:
    """Provides human-readable explanations and fix suggestions for divergence kinds."""

    _EXPLANATIONS = {
        "type_mismatch": (
            "The C type does not directly correspond to the Rust type used in the "
            "translation. C's type system is looser and allows implicit conversions "
            "that Rust forbids. For example, C's `int` is platform-dependent (typically "
            "32 bits) while Rust's `i32` is always 32 bits.",
            "Use the exact-width Rust integer type that matches the C type's semantics "
            "(e.g., `i32` for `int`, `u8` for `unsigned char`). Add explicit casts with "
            "`as` where narrowing is intentional."
        ),
        "control_flow": (
            "The control flow structure differs between the C and Rust versions. This "
            "may be due to C's use of goto, fall-through switches, or multi-exit loops "
            "that don't translate directly into Rust's structured control flow.",
            "Refactor into Rust idioms: replace goto with labeled loops/breaks, convert "
            "switch fall-through to explicit match arms with combined patterns, and use "
            "loop { ... break value; } for multi-exit loops."
        ),
        "overflow_behavior": (
            "C has undefined behavior on signed integer overflow, while Rust panics in "
            "debug mode and wraps in release mode. This means the same arithmetic can "
            "produce different results or different runtime behavior.",
            "Use `wrapping_add`, `wrapping_mul` etc. to match C release-mode behavior, "
            "or `checked_add`, `checked_mul` to detect overflow explicitly. Consider "
            "`Wrapping<i32>` for variables that should always wrap."
        ),
        "null_handling": (
            "C uses NULL pointers that can be dereferenced (causing UB), while Rust "
            "represents nullable values with `Option<T>`. The Rust version may differ "
            "in how it handles the absent-value case.",
            "Replace raw pointer checks with `Option::map`, `Option::unwrap_or`, or "
            "`Option::unwrap_or_else`. Use `if let Some(v) = x` instead of null checks."
        ),
        "memory_management": (
            "C uses manual `malloc`/`free` for heap allocation, while Rust uses "
            "ownership and RAII. The Rust translation may have different allocation "
            "lifetimes or missing frees (which Rust handles automatically).",
            "Use `Box<T>` for single heap objects, `Vec<T>` for dynamic arrays, and "
            "`Rc<T>` or `Arc<T>` for shared ownership. Remove explicit free calls and "
            "let Rust's drop semantics handle deallocation."
        ),
        "error_handling": (
            "C typically returns error codes (negative values or NULL), while Rust "
            "uses `Result<T, E>`. The translation may not preserve all error paths "
            "or may collapse distinct error codes.",
            "Map each C error code to a Rust enum variant inside `Result::Err`. Use "
            "the `?` operator for propagation and implement `From` traits for error "
            "conversion between layers."
        ),
        "string_handling": (
            "C strings are null-terminated `char*` arrays with manual length tracking, "
            "while Rust uses `String` (owned, UTF-8) and `&str` (borrowed). Buffer "
            "overflows possible in C are prevented in Rust.",
            "Use `CStr::from_ptr` at FFI boundaries, then convert to `&str` with "
            "`.to_str()`. For internal code, use `String` and `&str` exclusively. "
            "Replace `strncpy`/`snprintf` with `format!` or `String::push_str`."
        ),
    }

    def explain(self, divergence: dict) -> dict:
        """Return a dict with 'explanation' and 'suggestion' for a divergence."""
        kind = divergence.get("kind", "")
        if kind in self._EXPLANATIONS:
            base_explanation, base_suggestion = self._EXPLANATIONS[kind]
        else:
            base_explanation = (
                f"A divergence of kind '{kind}' was detected between the C and Rust "
                "implementations. The two versions may behave differently for some inputs."
            )
            base_suggestion = (
                "Manually review both implementations and add tests covering the "
                "divergent behavior to ensure correctness."
            )

        detail = divergence.get("description", "")
        if detail:
            base_explanation = f"{base_explanation} Specifically: {detail}"

        c_loc = divergence.get("c_location", {})
        rust_loc = divergence.get("rust_location", {})
        if c_loc or rust_loc:
            c_ref = f"{c_loc.get('file', '?')}:{c_loc.get('line', '?')}" if c_loc else "N/A"
            rust_ref = f"{rust_loc.get('file', '?')}:{rust_loc.get('line', '?')}" if rust_loc else "N/A"
            base_explanation += f" (C: {c_ref}, Rust: {rust_ref})"

        return {"explanation": base_explanation, "suggestion": base_suggestion}


class FixSuggestionGenerator:
    """Generates concrete fix suggestions for each divergence type."""

    _TYPE_MAP = {
        "int": "i32", "unsigned int": "u32", "long": "i64", "unsigned long": "u64",
        "short": "i16", "unsigned short": "u16", "char": "i8", "unsigned char": "u8",
        "float": "f32", "double": "f64", "size_t": "usize", "ssize_t": "isize",
        "int8_t": "i8", "uint8_t": "u8", "int16_t": "i16", "uint16_t": "u16",
        "int32_t": "i32", "uint32_t": "u32", "int64_t": "i64", "uint64_t": "u64",
        "bool": "bool", "_Bool": "bool", "void*": "*mut std::ffi::c_void",
    }

    def suggest_fix(self, divergence: dict) -> str:
        kind = divergence.get("kind", "")
        c_type = divergence.get("c_type", "")
        func_name = divergence.get("function", "unknown")

        if kind == "type_mismatch":
            rust_type = self._TYPE_MAP.get(c_type, "i32")
            return (
                f"In function `{func_name}`, change the Rust type to `{rust_type}` "
                f"to match the C type `{c_type}`. Add `as {rust_type}` casts at "
                "conversion boundaries."
            )
        if kind == "control_flow":
            return (
                f"In function `{func_name}`, restructure the Rust control flow to use "
                "a labeled loop with `'outer: loop {{ ... break 'outer value; }}` to "
                "replicate the C goto/multi-exit pattern."
            )
        if kind == "overflow_behavior":
            op = divergence.get("operation", "add")
            wrapping_fn = f"wrapping_{op}" if op in ("add", "sub", "mul") else "wrapping_add"
            return (
                f"In function `{func_name}`, replace the arithmetic operator with "
                f"`.{wrapping_fn}()` to match C's wrapping behavior, or use "
                f"`.checked_{op}()` if overflow should be detected."
            )
        if kind == "null_handling":
            return (
                f"In function `{func_name}`, wrap the nullable value in `Option<T>` "
                "and use `.map(|v| ...)` for transformation or `.unwrap_or(default)` "
                "for fallback values instead of raw pointer null checks."
            )
        if kind == "memory_management":
            alloc = divergence.get("c_allocator", "malloc")
            if alloc in ("malloc", "calloc"):
                return (
                    f"In function `{func_name}`, replace `{alloc}`/`free` with "
                    "`Box::new(value)` for single objects or `vec![0; n]` for arrays. "
                    "Remove the corresponding `free` call."
                )
            return (
                f"In function `{func_name}`, replace manual memory management with "
                "`Rc<T>` for shared ownership or `Arc<T>` for thread-safe sharing."
            )
        if kind == "error_handling":
            return (
                f"In function `{func_name}`, define an error enum and return "
                "`Result<T, MyError>` instead of error codes. Use the `?` operator "
                "for propagation."
            )
        if kind == "string_handling":
            return (
                f"In function `{func_name}`, use `std::ffi::CStr` at the FFI boundary "
                "and convert to `&str` with `.to_str().unwrap_or_default()`. Use "
                "`String` for owned string data internally."
            )
        return (
            f"In function `{func_name}`, manually review the divergence of kind "
            f"'{kind}' and add tests to verify equivalent behavior."
        )


class SummaryGenerator:
    """Computes overall equivalence status and summary statistics."""

    def generate_summary(self, results: dict) -> dict:
        equiv = results.get("equivalence_result", {})
        functions = equiv.get("functions", [])
        test_result = results.get("test_result", {})

        total = max(len(functions), 1)
        eq_count = sum(1 for f in functions if f.get("status") == "equivalent")
        partial_count = sum(1 for f in functions if f.get("status") == "partial")
        ne_count = total - eq_count - partial_count

        if ne_count > 0:
            overall = "not_equivalent"
        elif partial_count > 0:
            overall = "partial"
        else:
            overall = "equivalent"

        weights = {"equivalent": 1.0, "partial": 0.5, "not_equivalent": 0.0}
        confidence_sum = sum(
            weights.get(f.get("status", "not_equivalent"), 0.0)
            * f.get("confidence", 0.5)
            for f in functions
        )
        confidence = confidence_sum / total if total else 0.0

        divergences = equiv.get("divergences", [])
        sev_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for d in divergences:
            sev = d.get("severity", "medium")
            sev_counts[sev] = sev_counts.get(sev, 0) + 1

        tests_passed = test_result.get("passed", 0)
        tests_failed = test_result.get("failed", 0)
        tests_errors = test_result.get("errors", 0)
        tests_total = tests_passed + tests_failed + tests_errors

        if overall == "equivalent":
            para = (
                f"All {eq_count} functions are verified as equivalent between the C "
                f"and Rust implementations with {confidence:.0%} confidence. "
                f"{tests_passed} of {tests_total} tests passed successfully."
            )
        elif overall == "partial":
            para = (
                f"Of {total} functions analyzed, {eq_count} are fully equivalent, "
                f"{partial_count} are partially equivalent, and {ne_count} have "
                f"divergences. Overall confidence is {confidence:.0%}. "
                f"{sev_counts['critical']} critical and {sev_counts['high']} high "
                f"severity issues require attention."
            )
        else:
            para = (
                f"Significant divergences found: {ne_count} of {total} functions are "
                f"NOT equivalent. {sev_counts['critical']} critical issues detected. "
                f"Overall confidence is {confidence:.0%}. Immediate review recommended."
            )

        return {
            "overall_status": overall,
            "confidence": confidence,
            "functions_total": total,
            "functions_equivalent": eq_count,
            "functions_partial": partial_count,
            "functions_not_equivalent": ne_count,
            "severity_counts": sev_counts,
            "tests_passed": tests_passed,
            "tests_failed": tests_failed,
            "tests_errors": tests_errors,
            "tests_total": tests_total,
            "executive_summary": para,
        }


class RiskMatrix:
    """Builds a likelihood × impact risk matrix from divergences."""

    _SEVERITY_TO_IMPACT = {"critical": 5, "high": 4, "medium": 3, "low": 1}
    _KIND_LIKELIHOOD = {
        "overflow_behavior": 4, "memory_management": 4, "null_handling": 3,
        "type_mismatch": 3, "error_handling": 3, "control_flow": 2,
        "string_handling": 2,
    }

    def build_matrix(self, divergences: list) -> dict:
        cells: dict[tuple[int, int], list] = {}
        for i in range(1, 6):
            for j in range(1, 6):
                cells[(i, j)] = []

        for div in divergences:
            sev = div.get("severity", "medium")
            kind = div.get("kind", "")
            impact = self._SEVERITY_TO_IMPACT.get(sev, 3)
            likelihood = self._KIND_LIKELIHOOD.get(kind, 2)
            cells[(likelihood, impact)].append(div.get("id", div.get("kind", "?")))

        total_score = 0
        max_possible = max(len(divergences) * 25, 1)
        for (lik, imp), items in cells.items():
            total_score += lik * imp * len(items)

        normalized = total_score / max_possible
        if normalized >= 0.6:
            risk_level = "critical"
        elif normalized >= 0.4:
            risk_level = "high"
        elif normalized >= 0.2:
            risk_level = "medium"
        else:
            risk_level = "low"

        serializable_cells = {}
        for (lik, imp), items in cells.items():
            if items:
                serializable_cells[f"{lik},{imp}"] = items

        return {
            "cells": serializable_cells,
            "overall_score": round(normalized, 3),
            "risk_level": risk_level,
            "total_raw_score": total_score,
        }


class MarkdownGenerator:
    """Generates a full Markdown verification report."""

    def __init__(self):
        self._explainer = DivergenceExplainer()
        self._summary_gen = SummaryGenerator()
        self._risk_matrix = RiskMatrix()

    def generate(self, results: dict) -> str:
        summary = self._summary_gen.generate_summary(results)
        sections = [
            self._title_section(summary),
            self._summary_section(summary),
            self._confidence_bar(summary),
            self._function_table(results),
            self._type_mapping_table(results),
            self._divergence_details(results),
            self._test_results_section(results, summary),
            self._risk_matrix_section(results),
        ]
        return "\n\n".join(sections) + "\n"

    def _title_section(self, summary: dict) -> str:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        status_str = STATUS_EMOJI.get(summary["overall_status"], "❓")
        label = summary["overall_status"].upper().replace("_", " ")
        return (
            f"# Cross-Language Equivalence Verification Report\n\n"
            f"**Date:** {now}  \n"
            f"**Tool Version:** {TOOL_VERSION}  \n"
            f"**Overall Status:** {status_str} {label}"
        )

    def _summary_section(self, summary: dict) -> str:
        return (
            f"## Executive Summary\n\n"
            f"{summary['executive_summary']}\n\n"
            f"| Metric | Value |\n"
            f"|--------|-------|\n"
            f"| Functions Analyzed | {summary['functions_total']} |\n"
            f"| Fully Equivalent | {summary['functions_equivalent']} |\n"
            f"| Partially Equivalent | {summary['functions_partial']} |\n"
            f"| Not Equivalent | {summary['functions_not_equivalent']} |\n"
            f"| Tests Passed | {summary['tests_passed']}/{summary['tests_total']} |\n"
            f"| Critical Issues | {summary['severity_counts']['critical']} |\n"
            f"| High Issues | {summary['severity_counts']['high']} |"
        )

    def _confidence_bar(self, summary: dict) -> str:
        pct = summary["confidence"]
        filled = int(pct * 10)
        empty = 10 - filled
        bar = "█" * filled + "░" * empty
        return f"## Confidence Level\n\n`[{bar}]` {pct:.0%}"

    def _function_table(self, results: dict) -> str:
        functions = results.get("equivalence_result", {}).get("functions", [])
        if not functions:
            return "## Per-Function Results\n\nNo functions analyzed."
        lines = [
            "## Per-Function Results\n",
            "| Function | Status | Divergences | Test Results |",
            "|----------|--------|-------------|--------------|",
        ]
        for fn in functions:
            name = fn.get("name", "unknown")
            status = fn.get("status", "unknown")
            emoji = STATUS_EMOJI.get(status, "❓")
            div_count = fn.get("divergence_count", 0)
            test_pass = fn.get("tests_passed", 0)
            test_total = fn.get("tests_total", 0)
            lines.append(
                f"| `{name}` | {emoji} {status} | {div_count} | "
                f"{test_pass}/{test_total} passed |"
            )
        return "\n".join(lines)

    def _type_mapping_table(self, results: dict) -> str:
        mappings = results.get("equivalence_result", {}).get("type_mappings", [])
        if not mappings:
            return "## Type Mappings\n\nNo type mappings recorded."
        lines = [
            "## Type Mappings\n",
            "| C Type | Rust Type | Confidence |",
            "|--------|-----------|------------|",
        ]
        for m in mappings:
            c_type = m.get("c_type", "?")
            rust_type = m.get("rust_type", "?")
            conf = m.get("confidence", 0.0)
            lines.append(f"| `{c_type}` | `{rust_type}` | {conf:.0%} |")
        return "\n".join(lines)

    def _divergence_details(self, results: dict) -> str:
        divergences = results.get("equivalence_result", {}).get("divergences", [])
        if not divergences:
            return "## Divergence Details\n\nNo divergences found. 🎉"
        parts = ["## Divergence Details\n"]
        for i, div in enumerate(divergences, 1):
            sev = div.get("severity", "medium")
            emoji = SEVERITY_EMOJI.get(sev, "⚪")
            explained = self._explainer.explain(div)
            c_loc = div.get("c_location", {})
            rust_loc = div.get("rust_location", {})
            c_where = f"{c_loc.get('file', '?')}:{c_loc.get('line', '?')}" if c_loc else "N/A"
            rust_where = f"{rust_loc.get('file', '?')}:{rust_loc.get('line', '?')}" if rust_loc else "N/A"
            parts.append(
                f"### {i}. {div.get('kind', 'unknown')} ({emoji} {sev})\n\n"
                f"- **What:** {div.get('description', 'No description')}\n"
                f"- **Where:** C: `{c_where}` → Rust: `{rust_where}`\n"
                f"- **Why:** {explained['explanation']}\n"
                f"- **Severity:** {emoji} {sev}\n"
                f"- **Suggestion:** {explained['suggestion']}"
            )
        return "\n\n".join(parts)

    def _test_results_section(self, results: dict, summary: dict) -> str:
        test_result = results.get("test_result", {})
        parts = [
            "## Test Results\n",
            f"- **Passed:** {summary['tests_passed']}",
            f"- **Failed:** {summary['tests_failed']}",
            f"- **Errors:** {summary['tests_errors']}",
        ]
        failures = test_result.get("failures", [])
        if failures:
            parts.append("\n### Failures\n")
            for f in failures:
                name = f.get("test_name", "unknown")
                msg = f.get("message", "no message")
                c_out = f.get("c_output", "")
                r_out = f.get("rust_output", "")
                parts.append(
                    f"#### `{name}`\n\n"
                    f"**Message:** {msg}\n\n"
                    f"```\nC output:    {c_out}\nRust output: {r_out}\n```"
                )
        return "\n".join(parts)

    def _risk_matrix_section(self, results: dict) -> str:
        divergences = results.get("equivalence_result", {}).get("divergences", [])
        matrix = self._risk_matrix.build_matrix(divergences)
        header = (
            f"## Risk Matrix\n\n"
            f"**Overall Risk Score:** {matrix['overall_score']:.1%} "
            f"({matrix['risk_level'].upper()})\n"
        )
        lines = [
            "| | Impact 1 | Impact 2 | Impact 3 | Impact 4 | Impact 5 |",
            "|---|---|---|---|---|---|",
        ]
        for lik in range(5, 0, -1):
            row = [f"**Likelihood {lik}**"]
            for imp in range(1, 6):
                key = f"{lik},{imp}"
                items = matrix["cells"].get(key, [])
                row.append(", ".join(items) if items else "-")
            lines.append("| " + " | ".join(row) + " |")
        return header + "\n".join(lines)


class HTMLGenerator:
    """Generates a full HTML5 verification report with embedded CSS."""

    def __init__(self):
        self._summary_gen = SummaryGenerator()
        self._explainer = DivergenceExplainer()
        self._risk_matrix = RiskMatrix()

    def generate(self, results: dict) -> str:
        summary = self._summary_gen.generate_summary(results)
        return (
            self._head(summary)
            + self._sidebar(results, summary)
            + '<div class="main">'
            + self._dashboard(summary)
            + self._functions_section(results)
            + self._type_mappings_section(results)
            + self._divergences_section(results)
            + self._tests_section(results, summary)
            + self._risk_section(results)
            + "</div></body></html>"
        )

    def _head(self, summary: dict) -> str:
        status = summary["overall_status"].upper().replace("_", " ")
        css = """
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               display: flex; min-height: 100vh; color: #333; background: #f5f5f5; }
        .sidebar { width: 240px; background: #1a1a2e; color: #eee; padding: 20px;
                    position: fixed; height: 100vh; overflow-y: auto; }
        .sidebar h2 { font-size: 14px; text-transform: uppercase; color: #888;
                       margin: 20px 0 8px; }
        .sidebar a { display: block; color: #ccc; text-decoration: none;
                      padding: 6px 0; font-size: 14px; }
        .sidebar a:hover { color: #fff; }
        .main { margin-left: 240px; padding: 30px; flex: 1; }
        .dashboard { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                     gap: 16px; margin-bottom: 30px; }
        .card { background: #fff; border-radius: 8px; padding: 20px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        .card .label { font-size: 12px; color: #888; text-transform: uppercase; }
        .card .value { font-size: 28px; font-weight: 700; margin-top: 4px; }
        .card.pass { border-left: 4px solid #22c55e; }
        .card.fail { border-left: 4px solid #ef4444; }
        .card.warn { border-left: 4px solid #f59e0b; }
        table { width: 100%; border-collapse: collapse; margin: 16px 0; background: #fff;
                border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
        th { background: #f0f0f0; text-align: left; padding: 10px 14px; font-size: 13px;
             text-transform: uppercase; color: #555; }
        td { padding: 10px 14px; border-top: 1px solid #eee; font-size: 14px; }
        tr:nth-child(even) td { background: #fafafa; }
        details { background: #fff; border-radius: 8px; margin: 12px 0; padding: 16px;
                  box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
        summary { cursor: pointer; font-weight: 600; font-size: 15px; }
        summary:hover { color: #2563eb; }
        pre { background: #1e1e1e; color: #d4d4d4; padding: 14px; border-radius: 6px;
              overflow-x: auto; margin: 10px 0; font-size: 13px; }
        code { font-family: 'Fira Code', 'Consolas', monospace; }
        h2 { margin: 30px 0 14px; font-size: 22px; }
        .bar-chart { display: flex; align-items: flex-end; gap: 6px; height: 120px;
                     margin: 16px 0; padding: 10px; background: #fff; border-radius: 8px; }
        .bar { display: flex; flex-direction: column; align-items: center; flex: 1; }
        .bar-fill { width: 100%; border-radius: 4px 4px 0 0; transition: height 0.3s; }
        .bar-label { font-size: 11px; margin-top: 4px; color: #666; }
        .sev-critical { color: #dc2626; } .sev-high { color: #ea580c; }
        .sev-medium { color: #ca8a04; } .sev-low { color: #16a34a; }
        .confidence-bar { height: 20px; background: #e5e7eb; border-radius: 10px;
                          overflow: hidden; margin: 8px 0; max-width: 400px; }
        .confidence-fill { height: 100%; background: #3b82f6; border-radius: 10px; }
        .risk-cell { text-align: center; min-width: 60px; }
        .risk-high { background: #fecaca !important; }
        .risk-med { background: #fef08a !important; }
        .risk-low { background: #bbf7d0 !important; }
        @media (max-width: 768px) {
            .sidebar { display: none; }
            .main { margin-left: 0; }
        }
        """
        return (
            "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
            "<meta charset=\"UTF-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
            f"<title>Equivalence Report – {status}</title>\n"
            f"<style>{css}</style>\n"
            "</head>\n<body>\n"
        )

    def _sidebar(self, results: dict, summary: dict) -> str:
        nav_items = [
            ("dashboard", "Dashboard"),
            ("functions", "Functions"),
            ("types", "Type Mappings"),
            ("divergences", "Divergences"),
            ("tests", "Test Results"),
            ("risk", "Risk Matrix"),
        ]
        links = "\n".join(
            f'<a href="#{id_}">{label}</a>' for id_, label in nav_items
        )
        emoji = STATUS_EMOJI.get(summary["overall_status"], "❓")
        return (
            '<div class="sidebar">\n'
            f"<h1 style=\"font-size:18px;margin-bottom:4px\">{emoji} Verification</h1>\n"
            f"<p style=\"font-size:12px;color:#888\">{TOOL_VERSION}</p>\n"
            "<h2>Navigation</h2>\n"
            f"{links}\n"
            "</div>\n"
        )

    def _dashboard(self, summary: dict) -> str:
        pct = summary["confidence"]
        status = summary["overall_status"].upper().replace("_", " ")
        status_class = {
            "equivalent": "pass", "partial": "warn", "not_equivalent": "fail"
        }.get(summary["overall_status"], "warn")
        return (
            '<div id="dashboard">\n<h2>Dashboard</h2>\n<div class="dashboard">\n'
            f'<div class="card {status_class}"><div class="label">Status</div>'
            f'<div class="value">{status}</div></div>\n'
            f'<div class="card pass"><div class="label">Confidence</div>'
            f'<div class="value">{pct:.0%}</div></div>\n'
            f'<div class="card pass"><div class="label">Functions OK</div>'
            f'<div class="value">{summary["functions_equivalent"]}/{summary["functions_total"]}</div></div>\n'
            f'<div class="card {"pass" if summary["tests_failed"]==0 else "fail"}">'
            f'<div class="label">Tests</div>'
            f'<div class="value">{summary["tests_passed"]}/{summary["tests_total"]}</div></div>\n'
            f'<div class="card {"fail" if summary["severity_counts"]["critical"] else "pass"}">'
            f'<div class="label">Critical</div>'
            f'<div class="value">{summary["severity_counts"]["critical"]}</div></div>\n'
            "</div>\n"
            f'<div class="confidence-bar"><div class="confidence-fill" '
            f'style="width:{pct:.0%}"></div></div>\n'
            f"<p>{html_mod.escape(summary['executive_summary'])}</p>\n</div>\n"
        )

    def _functions_section(self, results: dict) -> str:
        functions = results.get("equivalence_result", {}).get("functions", [])
        rows = []
        for fn in functions:
            status = fn.get("status", "unknown")
            emoji = STATUS_EMOJI.get(status, "❓")
            rows.append(
                f"<tr><td><code>{html_mod.escape(fn.get('name', '?'))}</code></td>"
                f"<td>{emoji} {html_mod.escape(status)}</td>"
                f"<td>{fn.get('divergence_count', 0)}</td>"
                f"<td>{fn.get('tests_passed', 0)}/{fn.get('tests_total', 0)}</td></tr>"
            )
        return (
            '<div id="functions"><h2>Per-Function Results</h2>\n<table>\n'
            "<tr><th>Function</th><th>Status</th><th>Divergences</th><th>Tests</th></tr>\n"
            + "\n".join(rows)
            + "\n</table></div>\n"
        )

    def _type_mappings_section(self, results: dict) -> str:
        mappings = results.get("equivalence_result", {}).get("type_mappings", [])
        rows = []
        for m in mappings:
            conf = m.get("confidence", 0)
            rows.append(
                f"<tr><td><code>{html_mod.escape(m.get('c_type', '?'))}</code></td>"
                f"<td><code>{html_mod.escape(m.get('rust_type', '?'))}</code></td>"
                f"<td>{conf:.0%}</td></tr>"
            )
        return (
            '<div id="types"><h2>Type Mappings</h2>\n<table>\n'
            "<tr><th>C Type</th><th>Rust Type</th><th>Confidence</th></tr>\n"
            + "\n".join(rows)
            + "\n</table></div>\n"
        )

    def _divergences_section(self, results: dict) -> str:
        divergences = results.get("equivalence_result", {}).get("divergences", [])
        if not divergences:
            return '<div id="divergences"><h2>Divergences</h2><p>None found. 🎉</p></div>\n'
        items = []
        for i, div in enumerate(divergences, 1):
            sev = div.get("severity", "medium")
            emoji = SEVERITY_EMOJI.get(sev, "⚪")
            explained = self._explainer.explain(div)
            c_loc = div.get("c_location", {})
            rust_loc = div.get("rust_location", {})
            c_ref = f"{c_loc.get('file','?')}:{c_loc.get('line','?')}" if c_loc else "N/A"
            r_ref = f"{rust_loc.get('file','?')}:{rust_loc.get('line','?')}" if rust_loc else "N/A"
            c_snippet = div.get("c_snippet", "")
            rust_snippet = div.get("rust_snippet", "")
            code_block = ""
            if c_snippet or rust_snippet:
                code_block = (
                    f'<pre><code class="lang-c">// C\n{html_mod.escape(c_snippet)}'
                    f'</code></pre>\n<pre><code class="lang-rust">// Rust\n'
                    f'{html_mod.escape(rust_snippet)}</code></pre>'
                )
            items.append(
                f"<details>\n<summary>{emoji} #{i}: "
                f"{html_mod.escape(div.get('kind','?'))} "
                f'<span class="sev-{sev}">({sev})</span></summary>\n'
                f"<p><strong>What:</strong> {html_mod.escape(div.get('description',''))}</p>\n"
                f"<p><strong>Where:</strong> C: <code>{html_mod.escape(c_ref)}</code> → "
                f"Rust: <code>{html_mod.escape(r_ref)}</code></p>\n"
                f"<p><strong>Why:</strong> {html_mod.escape(explained['explanation'])}</p>\n"
                f"<p><strong>Suggestion:</strong> {html_mod.escape(explained['suggestion'])}</p>\n"
                f"{code_block}\n</details>"
            )
        return (
            '<div id="divergences"><h2>Divergences</h2>\n'
            + "\n".join(items)
            + "\n</div>\n"
        )

    def _tests_section(self, results: dict, summary: dict) -> str:
        test_result = results.get("test_result", {})
        max_val = max(summary["tests_passed"], summary["tests_failed"],
                      summary["tests_errors"], 1)
        bars = []
        for label, val, color in [
            ("Passed", summary["tests_passed"], "#22c55e"),
            ("Failed", summary["tests_failed"], "#ef4444"),
            ("Errors", summary["tests_errors"], "#f59e0b"),
        ]:
            h = int(val / max_val * 100) if max_val else 0
            bars.append(
                f'<div class="bar"><div class="bar-fill" '
                f'style="height:{h}px;background:{color}"></div>'
                f'<div class="bar-label">{label}<br>{val}</div></div>'
            )
        failures_html = ""
        for f in test_result.get("failures", []):
            failures_html += (
                f"<details><summary>❌ {html_mod.escape(f.get('test_name','?'))}"
                f"</summary>\n<p>{html_mod.escape(f.get('message',''))}</p>\n"
                f"<pre><code>C output:    {html_mod.escape(f.get('c_output',''))}\n"
                f"Rust output: {html_mod.escape(f.get('rust_output',''))}</code></pre>\n"
                f"</details>\n"
            )
        return (
            f'<div id="tests"><h2>Test Results</h2>\n'
            f'<div class="bar-chart">{"".join(bars)}</div>\n'
            f"{failures_html}</div>\n"
        )

    def _risk_section(self, results: dict) -> str:
        divergences = results.get("equivalence_result", {}).get("divergences", [])
        matrix = self._risk_matrix.build_matrix(divergences)
        rows = []
        for lik in range(5, 0, -1):
            cells = [f"<td><strong>L{lik}</strong></td>"]
            for imp in range(1, 6):
                key = f"{lik},{imp}"
                items = matrix["cells"].get(key, [])
                risk_score = lik * imp
                css = "risk-high" if risk_score >= 15 else ("risk-med" if risk_score >= 8 else "risk-low")
                content = ", ".join(html_mod.escape(str(x)) for x in items) if items else "&ndash;"
                cells.append(f'<td class="risk-cell {css}">{content}</td>')
            rows.append("<tr>" + "".join(cells) + "</tr>")
        header_cells = "<th></th>" + "".join(f"<th>I{i}</th>" for i in range(1, 6))
        return (
            f'<div id="risk"><h2>Risk Matrix</h2>\n'
            f"<p>Overall risk: <strong>{matrix['risk_level'].upper()}</strong> "
            f"(score: {matrix['overall_score']:.1%})</p>\n"
            f"<table><tr>{header_cells}</tr>\n"
            + "\n".join(rows)
            + "\n</table></div>\n"
        )


class JSONGenerator:
    """Generates a structured JSON verification report."""

    def __init__(self):
        self._summary_gen = SummaryGenerator()
        self._risk_matrix = RiskMatrix()
        self._explainer = DivergenceExplainer()

    def generate(self, results: dict) -> str:
        summary = self._summary_gen.generate_summary(results)
        equiv = results.get("equivalence_result", {})
        divergences = equiv.get("divergences", [])
        matrix = self._risk_matrix.build_matrix(divergences)

        enriched_divs = []
        for div in divergences:
            explained = self._explainer.explain(div)
            enriched = dict(div)
            enriched["explanation"] = explained["explanation"]
            enriched["suggestion"] = explained["suggestion"]
            enriched_divs.append(enriched)

        report = {
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "tool_version": TOOL_VERSION,
                "input_files": results.get("input_files", []),
            },
            "summary": summary,
            "functions": equiv.get("functions", []),
            "divergences": enriched_divs,
            "type_mappings": equiv.get("type_mappings", []),
            "test_results": {
                "passed": summary["tests_passed"],
                "failed": summary["tests_failed"],
                "errors": summary["tests_errors"],
                "total": summary["tests_total"],
                "failures": results.get("test_result", {}).get("failures", []),
            },
            "risk_matrix": matrix,
            "symbolic_results": results.get("symbolic_results", {}),
        }
        return json.dumps(report, indent=2, default=str)


class CIAnnotationGenerator:
    """Generates CI-compatible annotations and SARIF output."""

    _SARIF_SEVERITY = {
        "critical": "error", "high": "error", "medium": "warning", "low": "note",
    }

    def generate_annotations(self, divergences: list) -> str:
        lines = []
        for div in divergences:
            sev = div.get("severity", "medium")
            level = "error" if sev in ("critical", "high") else "warning"
            c_loc = div.get("c_location", {})
            rust_loc = div.get("rust_location", {})
            loc = rust_loc if rust_loc else c_loc
            file_ = loc.get("file", "unknown")
            line = loc.get("line", 1)
            kind = div.get("kind", "divergence")
            desc = div.get("description", "Divergence detected")
            lines.append(f"::{level} file={file_},line={line}::[{kind}] {desc}")
        return "\n".join(lines)

    def generate_sarif(self, divergences: list) -> dict:
        results = []
        rules = {}
        for i, div in enumerate(divergences):
            kind = div.get("kind", "unknown")
            rule_id = f"cross-lang/{kind}"
            if rule_id not in rules:
                rules[rule_id] = {
                    "id": rule_id,
                    "shortDescription": {"text": f"Cross-language divergence: {kind}"},
                    "defaultConfiguration": {
                        "level": self._SARIF_SEVERITY.get(
                            div.get("severity", "medium"), "warning"
                        )
                    },
                }

            rust_loc = div.get("rust_location", {})
            c_loc = div.get("c_location", {})
            locations = []
            for loc_data, role in [(rust_loc, "resultFile"), (c_loc, "relatedFile")]:
                if loc_data:
                    locations.append({
                        "physicalLocation": {
                            "artifactLocation": {"uri": loc_data.get("file", "unknown")},
                            "region": {
                                "startLine": loc_data.get("line", 1),
                                "startColumn": loc_data.get("column", 1),
                            },
                        },
                    })

            results.append({
                "ruleId": rule_id,
                "level": self._SARIF_SEVERITY.get(div.get("severity", "medium"), "warning"),
                "message": {"text": div.get("description", "Divergence detected")},
                "locations": locations[:1],
                "relatedLocations": [
                    {
                        "id": idx,
                        "physicalLocation": loc["physicalLocation"],
                        "message": {"text": "Related location"},
                    }
                    for idx, loc in enumerate(locations[1:], 1)
                ],
            })

        return {
            "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/main/sarif-2.1/schema/sarif-schema-2.1.0.json",
            "version": "2.1.0",
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": "cross-language-equivalence-verifier",
                            "version": TOOL_VERSION,
                            "rules": list(rules.values()),
                        }
                    },
                    "results": results,
                }
            ],
        }


class VerificationReporter:
    """Top-level dispatcher that generates reports in multiple formats."""

    def __init__(self):
        self._generators = {
            "markdown": MarkdownGenerator(),
            "html": HTMLGenerator(),
            "json": JSONGenerator(),
        }

    def generate(self, results: dict, format: str = "markdown") -> str:
        fmt = format.lower().strip()
        if fmt not in self._generators:
            available = ", ".join(sorted(self._generators.keys()))
            return json.dumps({
                "error": f"Unknown format '{fmt}'. Available: {available}",
                "available_formats": sorted(self._generators.keys()),
            })
        return self._generators[fmt].generate(results)
