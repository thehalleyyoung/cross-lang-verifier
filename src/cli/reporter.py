"""Result reporting for the Cross-Language Equivalence Verifier.

Provides EquivalenceVerdict, VerificationReport, and output formatters
for JSON, HTML, and terminal display.
"""

from __future__ import annotations

import json
import time
import html as html_mod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence


class VerdictKind(Enum):
    """Top-level equivalence verdict."""
    EQUIVALENT = "equivalent"
    DIVERGENT = "divergent"
    UNKNOWN = "unknown"

    @property
    def symbol(self) -> str:
        return {"equivalent": "✓", "divergent": "✗", "unknown": "?"}[self.value]

    @property
    def color_code(self) -> str:
        return {
            "equivalent": "\033[1;32m",
            "divergent": "\033[1;31m",
            "unknown": "\033[1;33m",
        }[self.value]


class DivergenceCategory(Enum):
    """Category of divergence."""
    INTEGER_OVERFLOW = "integer_overflow"
    SIGNED_UNSIGNED_MISMATCH = "signed_unsigned_mismatch"
    FLOAT_PRECISION = "float_precision"
    POINTER_SEMANTICS = "pointer_semantics"
    ARRAY_BOUNDS = "array_bounds"
    NULL_HANDLING = "null_handling"
    UNDEFINED_BEHAVIOR = "undefined_behavior"
    MEMORY_LAYOUT = "memory_layout"
    STRING_HANDLING = "string_handling"
    DIVISION_BY_ZERO = "division_by_zero"
    SHIFT_OVERFLOW = "shift_overflow"
    CAST_TRUNCATION = "cast_truncation"
    ENUM_REPRESENTATION = "enum_representation"
    OTHER = "other"

    @property
    def description(self) -> str:
        descriptions = {
            "integer_overflow": "Integer overflow behavior differs (C wraps/UB, Rust panics)",
            "signed_unsigned_mismatch": "Signed/unsigned interpretation mismatch",
            "float_precision": "Floating-point precision or rounding difference",
            "pointer_semantics": "Pointer aliasing or provenance difference",
            "array_bounds": "Array bounds checking difference",
            "null_handling": "Null pointer handling difference",
            "undefined_behavior": "C undefined behavior vs Rust defined behavior",
            "memory_layout": "Memory layout or alignment difference",
            "string_handling": "String encoding or null-termination difference",
            "division_by_zero": "Division by zero handling difference",
            "shift_overflow": "Bit shift amount handling difference",
            "cast_truncation": "Type cast truncation behavior difference",
            "enum_representation": "Enum representation or variant difference",
            "other": "Other divergence",
        }
        return descriptions.get(self.value, "Unknown divergence category")


@dataclass
class ConcreteValue:
    """A concrete input/output value for counterexample display."""
    name: str
    c_value: Any
    rust_value: Any
    ir_type: str = ""
    bit_width: int = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "c_value": self._serialize(self.c_value),
            "rust_value": self._serialize(self.rust_value),
            "ir_type": self.ir_type,
            "bit_width": self.bit_width,
        }

    @staticmethod
    def _serialize(val: Any) -> Any:
        if isinstance(val, (int, float, str, bool, type(None))):
            return val
        return str(val)


@dataclass
class Counterexample:
    """A concrete counterexample showing divergent behavior."""
    inputs: List[ConcreteValue] = field(default_factory=list)
    c_output: Any = None
    rust_output: Any = None
    c_trace: List[str] = field(default_factory=list)
    rust_trace: List[str] = field(default_factory=list)
    category: DivergenceCategory = DivergenceCategory.OTHER
    description: str = ""
    source_c: str = ""
    source_rust: str = ""
    minimized: bool = False

    def to_dict(self) -> dict:
        return {
            "inputs": [v.to_dict() for v in self.inputs],
            "c_output": ConcreteValue._serialize(self.c_output),
            "rust_output": ConcreteValue._serialize(self.rust_output),
            "c_trace": self.c_trace,
            "rust_trace": self.rust_trace,
            "category": self.category.value,
            "description": self.description,
            "source_c": self.source_c,
            "source_rust": self.source_rust,
            "minimized": self.minimized,
        }

    def format_terminal(self) -> str:
        lines = [f"  Category: {self.category.value}"]
        if self.description:
            lines.append(f"  Description: {self.description}")
        lines.append("  Inputs:")
        for inp in self.inputs:
            lines.append(f"    {inp.name} = {inp.c_value} ({inp.ir_type})")
        lines.append(f"  C output:    {self.c_output}")
        lines.append(f"  Rust output: {self.rust_output}")
        if self.c_trace:
            lines.append("  C trace:")
            for step in self.c_trace[:5]:
                lines.append(f"    {step}")
        if self.rust_trace:
            lines.append("  Rust trace:")
            for step in self.rust_trace[:5]:
                lines.append(f"    {step}")
        return "\n".join(lines)


@dataclass
class CoverageSummary:
    """Coverage statistics from verification."""
    total_paths: int = 0
    explored_paths: int = 0
    verified_paths: int = 0
    timed_out_paths: int = 0
    fuzzed_paths: int = 0
    block_coverage_c: float = 0.0
    block_coverage_rust: float = 0.0
    instruction_coverage: float = 0.0

    @property
    def path_coverage(self) -> float:
        if self.total_paths == 0:
            return 0.0
        return self.explored_paths / self.total_paths

    def to_dict(self) -> dict:
        return {
            "total_paths": self.total_paths,
            "explored_paths": self.explored_paths,
            "verified_paths": self.verified_paths,
            "timed_out_paths": self.timed_out_paths,
            "fuzzed_paths": self.fuzzed_paths,
            "block_coverage_c": self.block_coverage_c,
            "block_coverage_rust": self.block_coverage_rust,
            "instruction_coverage": self.instruction_coverage,
            "path_coverage": self.path_coverage,
        }

    def format_terminal(self) -> str:
        lines = [
            f"  Paths: {self.explored_paths}/{self.total_paths} explored, "
            f"{self.verified_paths} verified, {self.timed_out_paths} timed out",
            f"  Block coverage: C {self.block_coverage_c:.1%}, Rust {self.block_coverage_rust:.1%}",
            f"  Instruction coverage: {self.instruction_coverage:.1%}",
        ]
        if self.fuzzed_paths > 0:
            lines.append(f"  Fuzzed paths: {self.fuzzed_paths}")
        return "\n".join(lines)


@dataclass
class DivergenceSummary:
    """Summary of divergences found, grouped by category."""
    categories: Dict[str, int] = field(default_factory=dict)
    total: int = 0

    def add(self, category: DivergenceCategory) -> None:
        key = category.value
        self.categories[key] = self.categories.get(key, 0) + 1
        self.total += 1

    def to_dict(self) -> dict:
        return {"categories": self.categories, "total": self.total}

    def format_terminal(self) -> str:
        if self.total == 0:
            return "  No divergences found."
        lines = [f"  Total divergences: {self.total}"]
        for cat, count in sorted(self.categories.items(), key=lambda x: -x[1]):
            try:
                desc = DivergenceCategory(cat).description
            except ValueError:
                desc = cat
            lines.append(f"    {cat}: {count} ({desc})")
        return "\n".join(lines)


@dataclass
class TimingInfo:
    """Timing information for each phase."""
    total_seconds: float = 0.0
    parse_seconds: float = 0.0
    analysis_seconds: float = 0.0
    product_seconds: float = 0.0
    symbolic_seconds: float = 0.0
    smt_seconds: float = 0.0
    fuzz_seconds: float = 0.0
    report_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "total_seconds": round(self.total_seconds, 3),
            "parse_seconds": round(self.parse_seconds, 3),
            "analysis_seconds": round(self.analysis_seconds, 3),
            "product_seconds": round(self.product_seconds, 3),
            "symbolic_seconds": round(self.symbolic_seconds, 3),
            "smt_seconds": round(self.smt_seconds, 3),
            "fuzz_seconds": round(self.fuzz_seconds, 3),
            "report_seconds": round(self.report_seconds, 3),
        }

    def format_terminal(self) -> str:
        return (
            f"  Total: {self.total_seconds:.2f}s "
            f"(parse: {self.parse_seconds:.2f}s, analysis: {self.analysis_seconds:.2f}s, "
            f"product: {self.product_seconds:.2f}s, symbolic: {self.symbolic_seconds:.2f}s, "
            f"SMT: {self.smt_seconds:.2f}s, fuzz: {self.fuzz_seconds:.2f}s)"
        )


@dataclass
class EquivalenceVerdict:
    """The final verdict of equivalence verification."""
    kind: VerdictKind
    confidence: float = 1.0
    reason: str = ""
    conditions: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "verdict": self.kind.value,
            "confidence": self.confidence,
            "reason": self.reason,
            "conditions": self.conditions,
        }

    def format_terminal(self) -> str:
        reset = "\033[0m"
        color = self.kind.color_code
        symbol = self.kind.symbol
        result = f"{color}{symbol} {self.kind.value.upper()}{reset}"
        if self.confidence < 1.0:
            result += f" (confidence: {self.confidence:.0%})"
        if self.reason:
            result += f"\n  Reason: {self.reason}"
        for cond in self.conditions:
            result += f"\n  Condition: {cond}"
        return result


@dataclass
class VerificationReport:
    """Complete verification report."""
    verdict: EquivalenceVerdict
    c_source: str = ""
    rust_source: str = ""
    c_function: str = ""
    rust_function: str = ""
    counterexamples: List[Counterexample] = field(default_factory=list)
    coverage: CoverageSummary = field(default_factory=CoverageSummary)
    divergence_summary: DivergenceSummary = field(default_factory=DivergenceSummary)
    timing: TimingInfo = field(default_factory=TimingInfo)
    warnings: List[str] = field(default_factory=list)
    ir_stats: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict.to_dict(),
            "c_source": self.c_source,
            "rust_source": self.rust_source,
            "c_function": self.c_function,
            "rust_function": self.rust_function,
            "counterexamples": [ce.to_dict() for ce in self.counterexamples],
            "coverage": self.coverage.to_dict(),
            "divergence_summary": self.divergence_summary.to_dict(),
            "timing": self.timing.to_dict(),
            "warnings": self.warnings,
            "ir_stats": self.ir_stats,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def format_terminal(self) -> str:
        """Format report for terminal display."""
        lines: List[str] = []
        lines.append("=" * 60)
        lines.append("Cross-Language Equivalence Verification Report")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"C function:    {self.c_function or self.c_source}")
        lines.append(f"Rust function: {self.rust_function or self.rust_source}")
        lines.append("")
        lines.append("Verdict:")
        lines.append(self.verdict.format_terminal())
        lines.append("")

        if self.counterexamples:
            lines.append(f"Counterexamples ({len(self.counterexamples)}):")
            for i, ce in enumerate(self.counterexamples[:10]):
                lines.append(f"\n  [{i + 1}]")
                lines.append(ce.format_terminal())
            if len(self.counterexamples) > 10:
                lines.append(f"\n  ... and {len(self.counterexamples) - 10} more")
            lines.append("")

        lines.append("Divergence Summary:")
        lines.append(self.divergence_summary.format_terminal())
        lines.append("")

        lines.append("Coverage:")
        lines.append(self.coverage.format_terminal())
        lines.append("")

        lines.append("Timing:")
        lines.append(self.timing.format_terminal())
        lines.append("")

        if self.warnings:
            lines.append(f"Warnings ({len(self.warnings)}):")
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    def format_html(self) -> str:
        """Generate an HTML report."""
        esc = html_mod.escape
        verdict_color = {
            VerdictKind.EQUIVALENT: "#28a745",
            VerdictKind.DIVERGENT: "#dc3545",
            VerdictKind.UNKNOWN: "#ffc107",
        }[self.verdict.kind]

        parts: List[str] = []
        parts.append("<!DOCTYPE html><html><head>")
        parts.append("<meta charset='utf-8'>")
        parts.append("<title>Equivalence Verification Report</title>")
        parts.append("<style>")
        parts.append("body { font-family: 'Segoe UI', sans-serif; margin: 2em; }")
        parts.append("h1 { color: #333; }")
        parts.append(".verdict { padding: 1em; border-radius: 8px; color: white; font-size: 1.5em; }")
        parts.append("table { border-collapse: collapse; width: 100%; margin: 1em 0; }")
        parts.append("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
        parts.append("th { background: #f5f5f5; }")
        parts.append("pre { background: #f8f8f8; padding: 1em; border-radius: 4px; overflow-x: auto; }")
        parts.append(".ce { background: #fff3f3; border: 1px solid #fcc; padding: 1em; margin: 0.5em 0; border-radius: 4px; }")
        parts.append(".warning { color: #856404; background: #fff3cd; padding: 0.5em; border-radius: 4px; margin: 0.2em 0; }")
        parts.append("</style></head><body>")

        parts.append("<h1>Cross-Language Equivalence Verification Report</h1>")
        parts.append(f"<p>Generated: {esc(self.timestamp)}</p>")

        parts.append(f"<div class='verdict' style='background:{verdict_color}'>")
        parts.append(f"{self.verdict.kind.symbol} {esc(self.verdict.kind.value.upper())}")
        if self.verdict.confidence < 1.0:
            parts.append(f" (confidence: {self.verdict.confidence:.0%})")
        parts.append("</div>")

        if self.verdict.reason:
            parts.append(f"<p><strong>Reason:</strong> {esc(self.verdict.reason)}</p>")

        # Functions
        parts.append("<h2>Functions</h2>")
        parts.append("<table>")
        parts.append(f"<tr><th>C</th><td>{esc(self.c_function or self.c_source)}</td></tr>")
        parts.append(f"<tr><th>Rust</th><td>{esc(self.rust_function or self.rust_source)}</td></tr>")
        parts.append("</table>")

        # Counterexamples
        if self.counterexamples:
            parts.append(f"<h2>Counterexamples ({len(self.counterexamples)})</h2>")
            for i, ce in enumerate(self.counterexamples[:20]):
                parts.append(f"<div class='ce'><h3>#{i + 1}: {esc(ce.category.value)}</h3>")
                if ce.description:
                    parts.append(f"<p>{esc(ce.description)}</p>")
                parts.append("<table><tr><th>Input</th><th>Value</th><th>Type</th></tr>")
                for inp in ce.inputs:
                    parts.append(f"<tr><td>{esc(inp.name)}</td><td>{esc(str(inp.c_value))}</td>")
                    parts.append(f"<td>{esc(inp.ir_type)}</td></tr>")
                parts.append("</table>")
                parts.append(f"<p>C output: <code>{esc(str(ce.c_output))}</code></p>")
                parts.append(f"<p>Rust output: <code>{esc(str(ce.rust_output))}</code></p>")
                parts.append("</div>")

        # Divergence summary
        parts.append("<h2>Divergence Summary</h2>")
        if self.divergence_summary.total > 0:
            parts.append("<table><tr><th>Category</th><th>Count</th><th>Description</th></tr>")
            for cat, count in sorted(self.divergence_summary.categories.items(), key=lambda x: -x[1]):
                try:
                    desc = DivergenceCategory(cat).description
                except ValueError:
                    desc = ""
                parts.append(f"<tr><td>{esc(cat)}</td><td>{count}</td><td>{esc(desc)}</td></tr>")
            parts.append("</table>")
        else:
            parts.append("<p>No divergences found.</p>")

        # Coverage
        parts.append("<h2>Coverage</h2>")
        parts.append("<table>")
        parts.append(f"<tr><th>Paths</th><td>{self.coverage.explored_paths}/{self.coverage.total_paths}</td></tr>")
        parts.append(f"<tr><th>Verified</th><td>{self.coverage.verified_paths}</td></tr>")
        parts.append(f"<tr><th>C block coverage</th><td>{self.coverage.block_coverage_c:.1%}</td></tr>")
        parts.append(f"<tr><th>Rust block coverage</th><td>{self.coverage.block_coverage_rust:.1%}</td></tr>")
        parts.append("</table>")

        # Timing
        parts.append("<h2>Timing</h2>")
        parts.append("<table>")
        for phase_name, val in self.timing.to_dict().items():
            parts.append(f"<tr><td>{esc(phase_name)}</td><td>{val:.3f}s</td></tr>")
        parts.append("</table>")

        # Warnings
        if self.warnings:
            parts.append(f"<h2>Warnings ({len(self.warnings)})</h2>")
            for w in self.warnings:
                parts.append(f"<div class='warning'>⚠ {esc(w)}</div>")

        parts.append("</body></html>")
        return "\n".join(parts)

    @staticmethod
    def from_dict(d: dict) -> VerificationReport:
        verdict_data = d.get("verdict", {})
        verdict = EquivalenceVerdict(
            kind=VerdictKind(verdict_data.get("verdict", "unknown")),
            confidence=verdict_data.get("confidence", 1.0),
            reason=verdict_data.get("reason", ""),
            conditions=verdict_data.get("conditions", []),
        )
        report = VerificationReport(verdict=verdict)
        report.c_source = d.get("c_source", "")
        report.rust_source = d.get("rust_source", "")
        report.c_function = d.get("c_function", "")
        report.rust_function = d.get("rust_function", "")
        report.warnings = d.get("warnings", [])
        report.ir_stats = d.get("ir_stats", {})
        report.metadata = d.get("metadata", {})
        report.timestamp = d.get("timestamp", "")

        cov = d.get("coverage", {})
        report.coverage = CoverageSummary(
            total_paths=cov.get("total_paths", 0),
            explored_paths=cov.get("explored_paths", 0),
            verified_paths=cov.get("verified_paths", 0),
            timed_out_paths=cov.get("timed_out_paths", 0),
            fuzzed_paths=cov.get("fuzzed_paths", 0),
            block_coverage_c=cov.get("block_coverage_c", 0.0),
            block_coverage_rust=cov.get("block_coverage_rust", 0.0),
            instruction_coverage=cov.get("instruction_coverage", 0.0),
        )

        ds = d.get("divergence_summary", {})
        report.divergence_summary = DivergenceSummary(
            categories=ds.get("categories", {}),
            total=ds.get("total", 0),
        )

        ti = d.get("timing", {})
        report.timing = TimingInfo(**{k: v for k, v in ti.items() if k in TimingInfo.__dataclass_fields__})

        return report


class ReportWriter:
    """Writes verification reports to files or stdout."""

    @staticmethod
    def write_json(report: VerificationReport, path: str) -> None:
        with open(path, "w") as f:
            f.write(report.to_json())

    @staticmethod
    def write_html(report: VerificationReport, path: str) -> None:
        with open(path, "w") as f:
            f.write(report.format_html())

    @staticmethod
    def write_text(report: VerificationReport, path: str) -> None:
        with open(path, "w") as f:
            f.write(report.format_terminal())

    @staticmethod
    def write(report: VerificationReport, path: str, fmt: str = "json") -> None:
        writers = {
            "json": ReportWriter.write_json,
            "html": ReportWriter.write_html,
            "text": ReportWriter.write_text,
        }
        writer = writers.get(fmt)
        if writer is None:
            raise ValueError(f"Unknown format: {fmt}")
        writer(report, path)
