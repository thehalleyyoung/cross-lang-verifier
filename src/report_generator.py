"""Report generation for XEquiv verification results.

Produces HTML reports with divergence visualization, SARIF for IDE
integration, GitHub Actions annotations, and Markdown for PR comments.
"""

import json
import time
import html as html_mod
from dataclasses import dataclass
from typing import List, Optional, Dict

from .api import VerificationResult, Divergence, Counterexample
from .project_scanner import (
    FunctionVerification, ProjectVerificationResult, MigrationStatus,
)


# ---------------------------------------------------------------------------
# SARIF output
# ---------------------------------------------------------------------------

def to_sarif(pvr: ProjectVerificationResult,
             tool_version: str = "0.2.0") -> dict:
    """Generate SARIF v2.1.0 log from project verification results."""
    rule_ids: Dict[str, dict] = {}
    results: List[dict] = []

    for fv in pvr.verifications:
        if fv.status != "failed" or fv.result is None:
            continue
        for div in fv.result.divergences:
            rule_id = f"xequiv/{div.category}"
            if rule_id not in rule_ids:
                rule_ids[rule_id] = {
                    "id": rule_id,
                    "shortDescription": {"text": div.category.replace("_", " ").title()},
                    "helpUri": "https://github.com/xequiv/xequiv#divergence-categories",
                    "properties": {"tags": ["correctness", "migration"]},
                }
            sarif_result = {
                "ruleId": rule_id,
                "level": "error" if div.severity == "critical" else "warning",
                "message": {
                    "text": f"{div.description}\n  C: {div.c_behavior}\n  Rust: {div.rust_behavior}",
                },
                "locations": [{
                    "physicalLocation": {
                        "artifactLocation": {"uri": fv.match.c_function.file_path},
                        "region": {
                            "startLine": fv.match.c_function.line_number,
                            "message": {"text": f"C function: {fv.match.c_function.name}"},
                        },
                    },
                }],
                "relatedLocations": [{
                    "id": 1,
                    "physicalLocation": {
                        "artifactLocation": {"uri": fv.match.rust_function.file_path},
                        "region": {
                            "startLine": fv.match.rust_function.line_number,
                        },
                    },
                    "message": {"text": f"Rust function: {fv.match.rust_function.name}"},
                }],
            }
            if fv.result.counterexamples:
                cx = fv.result.counterexamples[0]
                sarif_result["properties"] = {
                    "counterexample": {
                        "inputs": cx.inputs,
                        "c_output": cx.c_output,
                        "rust_output": cx.rust_output,
                    }
                }
            results.append(sarif_result)

    return {
        "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/main/sarif-2.1/schema/sarif-schema-2.1.0.json",
        "version": "2.1.0",
        "runs": [{
            "tool": {
                "driver": {
                    "name": "XEquiv",
                    "version": tool_version,
                    "informationUri": "https://github.com/xequiv/xequiv",
                    "rules": list(rule_ids.values()),
                },
            },
            "results": results,
            "invocations": [{
                "executionSuccessful": True,
                "properties": {
                    "totalPairsChecked": pvr.status.matched,
                    "equivalentPairs": pvr.status.verified_equivalent,
                    "divergentPairs": pvr.status.verified_divergent,
                    "durationMs": pvr.duration_ms,
                },
            }],
        }],
    }


def write_sarif(pvr: ProjectVerificationResult, output_path: str) -> None:
    sarif = to_sarif(pvr)
    with open(output_path, "w") as f:
        json.dump(sarif, f, indent=2)


# ---------------------------------------------------------------------------
# GitHub Actions annotations
# ---------------------------------------------------------------------------

def to_github_annotations(pvr: ProjectVerificationResult) -> str:
    """Generate GitHub Actions annotation commands for workflow output."""
    lines: List[str] = []
    for fv in pvr.verifications:
        if fv.status != "failed" or fv.result is None:
            continue
        for div in fv.result.divergences:
            level = "error" if div.severity == "critical" else "warning"
            file_path = fv.match.c_function.file_path
            line = fv.match.c_function.line_number
            msg = (f"XEquiv: {div.category} divergence in "
                   f"{fv.match.c_function.name} ↔ {fv.match.rust_function.name}: "
                   f"{div.description}")
            lines.append(f"::{level} file={file_path},line={line}::{msg}")

    if not lines:
        lines.append("::notice ::XEquiv: All verified function pairs are equivalent")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Markdown summary (for PR comments)
# ---------------------------------------------------------------------------

def to_markdown(pvr: ProjectVerificationResult,
                include_details: bool = True,
                max_divergences: int = 20) -> str:
    """Generate Markdown summary suitable for PR comments."""
    s = pvr.status
    pct = (s.verified_equivalent / s.matched * 100) if s.matched > 0 else 0

    icon = "✅" if s.verified_divergent == 0 else "❌"
    lines = [
        f"## {icon} XEquiv Verification Report\n",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Functions scanned (C) | {s.total_c_functions} |",
        f"| Matched pairs | {s.matched} |",
        f"| Verified equivalent | {s.verified_equivalent} |",
        f"| **Divergences found** | **{s.verified_divergent}** |",
        f"| Unverified | {s.unverified} |",
        f"| Unmatched C functions | {s.unmatched} |",
        f"| Verification time | {pvr.duration_ms:.0f}ms |",
        f"\n**Equivalence rate: {pct:.1f}%**\n",
    ]

    if include_details and s.verified_divergent > 0:
        lines.append("### Divergences\n")
        shown = 0
        for fv in pvr.verifications:
            if fv.status != "failed" or fv.result is None:
                continue
            if shown >= max_divergences:
                remaining = s.verified_divergent - shown
                lines.append(f"\n*... and {remaining} more divergence(s)*\n")
                break
            lines.append(
                f"#### `{fv.match.c_function.name}` ↔ `{fv.match.rust_function.name}`\n"
            )
            for div in fv.result.divergences:
                sev_icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(div.severity, "⚪")
                lines.append(f"- {sev_icon} **{div.category}**: {div.description}")
                lines.append(f"  - C: {div.c_behavior}")
                lines.append(f"  - Rust: {div.rust_behavior}")
            for cx in fv.result.counterexamples[:2]:
                lines.append(f"- 🧪 Counterexample: inputs={cx.inputs} → C={cx.c_output}, Rust={cx.rust_output}")
            lines.append("")
            shown += 1

    if include_details:
        equiv_list = [
            fv for fv in pvr.verifications if fv.status == "passed"
        ]
        if equiv_list:
            lines.append("<details><summary>✅ Equivalent pairs</summary>\n")
            for fv in equiv_list:
                lines.append(f"- `{fv.match.c_function.name}` ↔ `{fv.match.rust_function.name}`")
            lines.append("\n</details>\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML report with divergence visualization
# ---------------------------------------------------------------------------

def _severity_color(severity: str) -> str:
    return {"critical": "#e53935", "warning": "#fb8c00", "info": "#1e88e5"}.get(severity, "#757575")


def _escape(text: str) -> str:
    return html_mod.escape(text)


def to_html(pvr: ProjectVerificationResult) -> str:
    """Generate standalone HTML report with divergence visualization."""
    s = pvr.status
    pct = (s.verified_equivalent / s.matched * 100) if s.matched > 0 else 0
    bar_color = "#4caf50" if pct > 80 else "#ff9800" if pct > 50 else "#f44336"

    divergence_rows = []
    for fv in pvr.verifications:
        if fv.status != "failed" or fv.result is None:
            continue
        for div in fv.result.divergences:
            color = _severity_color(div.severity)
            divergence_rows.append(f"""
            <tr>
              <td><code>{_escape(fv.match.c_function.name)}</code></td>
              <td><code>{_escape(fv.match.rust_function.name)}</code></td>
              <td><span style="color:{color};font-weight:bold">{_escape(div.severity)}</span></td>
              <td>{_escape(div.category)}</td>
              <td>{_escape(div.description)}</td>
              <td><small>C: {_escape(div.c_behavior)}<br>Rust: {_escape(div.rust_behavior)}</small></td>
            </tr>""")

    equiv_rows = []
    for fv in pvr.verifications:
        if fv.status != "passed":
            continue
        equiv_rows.append(f"""
            <tr>
              <td><code>{_escape(fv.match.c_function.name)}</code></td>
              <td><code>{_escape(fv.match.rust_function.name)}</code></td>
              <td>{fv.match.confidence:.0%}</td>
              <td>{fv.result.duration_ms:.1f}ms</td>
            </tr>""")

    # Category breakdown for chart
    cat_counts: Dict[str, int] = {}
    for fv in pvr.verifications:
        if fv.result:
            for div in fv.result.divergences:
                cat_counts[div.category] = cat_counts.get(div.category, 0) + 1

    chart_bars = ""
    if cat_counts:
        max_count = max(cat_counts.values())
        for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
            width = int(count / max_count * 100)
            chart_bars += (
                f'<div style="display:flex;align-items:center;margin:4px 0">'
                f'<span style="width:180px;font-size:13px">{cat}</span>'
                f'<div style="background:#e53935;height:20px;width:{width}%;'
                f'border-radius:3px;min-width:20px"></div>'
                f'<span style="margin-left:8px;font-size:13px">{count}</span></div>'
            )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>XEquiv Verification Report</title>
<style>
* {{ box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       max-width: 1100px; margin: 0 auto; padding: 24px; color: #212121; }}
h1 {{ border-bottom: 2px solid #1565c0; padding-bottom: 8px; }}
h2 {{ color: #1565c0; margin-top: 32px; }}
table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
th {{ background: #f5f5f5; text-align: left; }}
th, td {{ border: 1px solid #e0e0e0; padding: 8px 12px; font-size: 14px; }}
tr:hover {{ background: #fafafa; }}
.metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin: 16px 0; }}
.metric {{ background: #f5f5f5; border-radius: 8px; padding: 16px; text-align: center; }}
.metric .value {{ font-size: 28px; font-weight: bold; }}
.metric .label {{ font-size: 12px; color: #757575; margin-top: 4px; }}
.bar-outer {{ background: #e0e0e0; border-radius: 6px; overflow: hidden; height: 28px; margin: 16px 0; }}
.bar-inner {{ height: 100%; border-radius: 6px; transition: width 0.5s; }}
code {{ background: #f5f5f5; padding: 2px 6px; border-radius: 3px; font-size: 13px; }}
</style></head><body>
<h1>XEquiv Verification Report</h1>
<p>Generated at {time.strftime('%Y-%m-%d %H:%M:%S')}</p>

<div class="bar-outer">
  <div class="bar-inner" style="width:{pct:.0f}%;background:{bar_color}"></div>
</div>
<p style="text-align:center;font-size:18px"><strong>{pct:.1f}%</strong> equivalent</p>

<div class="metric-grid">
  <div class="metric"><div class="value">{s.total_c_functions}</div><div class="label">C Functions</div></div>
  <div class="metric"><div class="value">{s.matched}</div><div class="label">Matched Pairs</div></div>
  <div class="metric"><div class="value" style="color:#4caf50">{s.verified_equivalent}</div><div class="label">Equivalent</div></div>
  <div class="metric"><div class="value" style="color:#e53935">{s.verified_divergent}</div><div class="label">Divergent</div></div>
  <div class="metric"><div class="value" style="color:#ff9800">{s.unverified}</div><div class="label">Unverified</div></div>
  <div class="metric"><div class="value">{pvr.duration_ms:.0f}ms</div><div class="label">Total Time</div></div>
</div>

{"<h2>Divergence Categories</h2>" + chart_bars if chart_bars else ""}

<h2>Divergences ({s.verified_divergent})</h2>
{"<p>No divergences found! All matched pairs are equivalent.</p>" if not divergence_rows else ""}
<table>
<tr><th>C Function</th><th>Rust Function</th><th>Severity</th><th>Category</th><th>Description</th><th>Behaviors</th></tr>
{"".join(divergence_rows)}
</table>

<h2>Equivalent Pairs ({s.verified_equivalent})</h2>
<table>
<tr><th>C Function</th><th>Rust Function</th><th>Match Confidence</th><th>Verify Time</th></tr>
{"".join(equiv_rows)}
</table>
</body></html>"""


def write_html(pvr: ProjectVerificationResult, output_path: str) -> None:
    with open(output_path, "w") as f:
        f.write(to_html(pvr))


# ---------------------------------------------------------------------------
# Unified report generator
# ---------------------------------------------------------------------------

def generate_report(pvr: ProjectVerificationResult,
                    format: str = "markdown",
                    output_path: Optional[str] = None) -> str:
    """Generate a report in the specified format.

    Args:
        pvr: Project verification result
        format: "html", "sarif", "github", "markdown"
        output_path: If given, write to file

    Returns:
        Report content as string
    """
    if format == "html":
        content = to_html(pvr)
    elif format == "sarif":
        content = json.dumps(to_sarif(pvr), indent=2)
    elif format == "github":
        content = to_github_annotations(pvr)
    else:
        content = to_markdown(pvr)

    if output_path:
        with open(output_path, "w") as f:
            f.write(content)

    return content
