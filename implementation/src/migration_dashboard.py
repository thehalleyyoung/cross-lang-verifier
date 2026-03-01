"""Web dashboard for C→Rust migration progress.

Generates HTML reports, risk heatmaps, effort timelines, and team
assignment plans for managing large-scale C→Rust migrations.
"""

import re
import os
import math
import time
import textwrap
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from enum import Enum


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FileRisk:
    """Risk assessment for a single file."""
    file_path: str
    risk_level: RiskLevel
    risk_score: float  # 0.0 – 1.0
    loc: int
    complexity: int
    unsafe_patterns: int
    dependencies: int
    reasons: List[str] = field(default_factory=list)

    @property
    def color(self) -> str:
        colors = {
            RiskLevel.LOW: "#27ae60",
            RiskLevel.MEDIUM: "#f39c12",
            RiskLevel.HIGH: "#e67e22",
            RiskLevel.CRITICAL: "#e74c3c",
        }
        return colors[self.risk_level]


@dataclass
class MigrationFile:
    """Migration status of one file."""
    c_path: str
    rust_path: Optional[str]
    status: str  # "pending", "in_progress", "migrated", "verified", "skipped"
    functions_total: int = 0
    functions_migrated: int = 0
    functions_verified: int = 0
    loc: int = 0
    risk: RiskLevel = RiskLevel.MEDIUM

    @property
    def progress(self) -> float:
        if self.functions_total == 0:
            return 0.0
        return self.functions_migrated / self.functions_total


@dataclass
class Report:
    """Migration progress report."""
    total_c_files: int = 0
    total_rust_files: int = 0
    total_c_loc: int = 0
    total_rust_loc: int = 0
    total_functions: int = 0
    migrated_functions: int = 0
    verified_functions: int = 0
    files: List[MigrationFile] = field(default_factory=list)
    overall_progress: float = 0.0
    safety_score: float = 0.0
    estimated_remaining_hours: float = 0.0

    @property
    def verification_rate(self) -> float:
        if self.migrated_functions == 0:
            return 0.0
        return self.verified_functions / self.migrated_functions


@dataclass
class TimelinePhase:
    """A phase in the migration timeline."""
    name: str
    files: List[str]
    estimated_hours: float
    dependencies: List[str] = field(default_factory=list)
    risk: RiskLevel = RiskLevel.MEDIUM

    @property
    def estimated_days(self) -> float:
        return self.estimated_hours / 8.0


@dataclass
class Timeline:
    """Estimated migration timeline."""
    phases: List[TimelinePhase] = field(default_factory=list)
    total_hours: float = 0.0
    total_weeks: float = 0.0
    critical_path_hours: float = 0.0

    @property
    def total_days(self) -> float:
        return self.total_hours / 8.0


@dataclass
class Assignment:
    """Developer work assignment."""
    developer_id: int
    files: List[str] = field(default_factory=list)
    estimated_hours: float = 0.0
    risk_level: RiskLevel = RiskLevel.MEDIUM
    specialization: str = ""

    @property
    def estimated_days(self) -> float:
        return self.estimated_hours / 8.0


# ---------------------------------------------------------------------------
# File scanning helpers
# ---------------------------------------------------------------------------

def _scan_c_dir(c_dir: str) -> List[Dict]:
    """Scan a C directory and gather file metadata."""
    root = Path(c_dir)
    files = []
    for c_file in sorted(root.rglob("*.c")):
        try:
            content = c_file.read_text(encoding="utf-8", errors="replace")
        except Exception:
            content = ""
        loc = content.count('\n') + 1
        fn_count = len(re.findall(
            r'^\w[\w\s\*]+\s+\w+\s*\([^)]*\)\s*\{', content, re.MULTILINE
        ))
        complexity = _estimate_complexity(content)
        unsafe_count = _count_unsafe_patterns(content)
        deps = _count_dependencies(content)

        files.append({
            "path": str(c_file.relative_to(root)),
            "abs_path": str(c_file),
            "loc": loc,
            "functions": fn_count,
            "complexity": complexity,
            "unsafe_patterns": unsafe_count,
            "dependencies": deps,
        })
    return files


def _scan_rust_dir(rust_dir: str) -> Dict[str, Dict]:
    """Scan Rust directory for migrated files."""
    root = Path(rust_dir)
    files = {}
    for rs_file in sorted(root.rglob("*.rs")):
        try:
            content = rs_file.read_text(encoding="utf-8", errors="replace")
        except Exception:
            content = ""
        loc = content.count('\n') + 1
        fn_count = len(re.findall(r'\bfn\s+\w+', content))
        verified = len(re.findall(r'// verified|#\[verified\]', content))

        rel = str(rs_file.relative_to(root))
        files[rel] = {
            "path": rel,
            "abs_path": str(rs_file),
            "loc": loc,
            "functions": fn_count,
            "verified": verified,
        }
    return files


def _estimate_complexity(c_code: str) -> int:
    """Estimate cyclomatic complexity of C code."""
    decision_points = 0
    keywords = ['if', 'else', 'for', 'while', 'switch', 'case',
                '&&', '||', '?']
    for kw in keywords:
        decision_points += c_code.count(kw)
    return decision_points


def _count_unsafe_patterns(c_code: str) -> int:
    """Count potentially unsafe patterns in C code."""
    patterns = [
        r'\bstrcpy\b', r'\bstrcat\b', r'\bgets\b', r'\bsprintf\b',
        r'\bfree\b', r'\bmalloc\b', r'\brealloc\b',
        r'\bscanf\s*\(\s*"[^"]*%s',
        r'printf\s*\(\s*\w+\s*\)',
        r'\bgoto\b',
    ]
    count = 0
    for p in patterns:
        count += len(re.findall(p, c_code))
    return count


def _count_dependencies(c_code: str) -> int:
    """Count #include directives as a proxy for dependencies."""
    return len(re.findall(r'#include', c_code))


def _assess_risk(file_info: Dict) -> FileRisk:
    """Assess migration risk for a single file."""
    score = 0.0
    reasons: List[str] = []

    # LOC factor
    loc = file_info["loc"]
    if loc > 1000:
        score += 0.2
        reasons.append(f"Large file ({loc} LOC)")
    elif loc > 500:
        score += 0.1

    # Complexity factor
    complexity = file_info["complexity"]
    if complexity > 50:
        score += 0.25
        reasons.append(f"High complexity ({complexity})")
    elif complexity > 20:
        score += 0.1

    # Unsafe patterns
    unsafe = file_info["unsafe_patterns"]
    if unsafe > 10:
        score += 0.3
        reasons.append(f"Many unsafe patterns ({unsafe})")
    elif unsafe > 5:
        score += 0.15
        reasons.append(f"Some unsafe patterns ({unsafe})")

    # Dependencies
    deps = file_info["dependencies"]
    if deps > 10:
        score += 0.15
        reasons.append(f"Many dependencies ({deps})")
    elif deps > 5:
        score += 0.05

    score = min(1.0, score)

    if score >= 0.7:
        level = RiskLevel.CRITICAL
    elif score >= 0.45:
        level = RiskLevel.HIGH
    elif score >= 0.2:
        level = RiskLevel.MEDIUM
    else:
        level = RiskLevel.LOW

    return FileRisk(
        file_path=file_info["path"],
        risk_level=level,
        risk_score=score,
        loc=loc,
        complexity=complexity,
        unsafe_patterns=unsafe,
        dependencies=deps,
        reasons=reasons,
    )


def _estimate_effort_hours(file_info: Dict) -> float:
    """Estimate migration effort in hours for a file."""
    base = file_info["loc"] / 50.0  # ~50 LOC/hour baseline
    complexity_factor = 1.0 + file_info["complexity"] / 100.0
    unsafe_factor = 1.0 + file_info["unsafe_patterns"] * 0.1
    return base * complexity_factor * unsafe_factor


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def migration_progress_report(c_dir: str, rust_dir: str) -> Report:
    """Generate a migration progress report.

    Args:
        c_dir: Root directory of the C project
        rust_dir: Root directory of the Rust translation

    Returns:
        Report with overall and per-file progress
    """
    c_files = _scan_c_dir(c_dir)
    r_files = _scan_rust_dir(rust_dir)

    report = Report(
        total_c_files=len(c_files),
        total_rust_files=len(r_files),
    )

    for cf in c_files:
        rust_name = cf["path"].replace(".c", ".rs")
        rust_info = r_files.get(rust_name) or r_files.get(
            "src/" + rust_name
        )

        risk = _assess_risk(cf)
        report.total_c_loc += cf["loc"]
        report.total_functions += cf["functions"]

        if rust_info:
            report.total_rust_loc += rust_info["loc"]
            migrated = rust_info["functions"]
            verified = rust_info.get("verified", 0)
            report.migrated_functions += migrated
            report.verified_functions += verified
            status = "verified" if verified > 0 else "migrated"
        else:
            migrated = 0
            verified = 0
            status = "pending"

        mf = MigrationFile(
            c_path=cf["path"],
            rust_path=rust_name if rust_info else None,
            status=status,
            functions_total=cf["functions"],
            functions_migrated=migrated,
            functions_verified=verified,
            loc=cf["loc"],
            risk=risk.risk_level,
        )
        report.files.append(mf)

    if report.total_functions > 0:
        report.overall_progress = (
            report.migrated_functions / report.total_functions
        )

    # Estimate remaining effort
    pending_files = [f for f in c_files
                     if f["path"].replace(".c", ".rs") not in r_files
                     and "src/" + f["path"].replace(".c", ".rs") not in r_files]
    report.estimated_remaining_hours = sum(
        _estimate_effort_hours(f) for f in pending_files
    )

    return report


def risk_heatmap(c_dir: str) -> str:
    """Generate an HTML risk heatmap for all C files.

    Args:
        c_dir: Root directory of the C project

    Returns:
        HTML string containing the interactive heatmap
    """
    c_files = _scan_c_dir(c_dir)
    risks = [_assess_risk(f) for f in c_files]
    risks.sort(key=lambda r: -r.risk_score)

    rows = []
    for risk in risks:
        reasons_html = "<br>".join(risk.reasons) if risk.reasons else "—"
        rows.append(f"""
            <tr style="background-color: {risk.color}22;">
                <td style="padding:8px;border:1px solid #ddd;">
                    <span style="color:{risk.color};">●</span> {risk.file_path}
                </td>
                <td style="padding:8px;border:1px solid #ddd;text-align:center;">
                    <strong style="color:{risk.color};">{risk.risk_level.value.upper()}</strong>
                </td>
                <td style="padding:8px;border:1px solid #ddd;text-align:center;">
                    {risk.risk_score:.2f}
                </td>
                <td style="padding:8px;border:1px solid #ddd;text-align:right;">
                    {risk.loc}
                </td>
                <td style="padding:8px;border:1px solid #ddd;text-align:right;">
                    {risk.unsafe_patterns}
                </td>
                <td style="padding:8px;border:1px solid #ddd;">
                    {reasons_html}
                </td>
            </tr>
        """)

    summary_counts = {level: 0 for level in RiskLevel}
    for r in risks:
        summary_counts[r.risk_level] += 1

    return _html_page("Risk Heatmap", f"""
        <h2>Risk Summary</h2>
        <div style="display:flex;gap:20px;margin-bottom:20px;">
            <div style="padding:15px;background:#e74c3c22;border-radius:8px;flex:1;">
                <strong style="color:#e74c3c;">CRITICAL</strong>
                <div style="font-size:2em;">{summary_counts[RiskLevel.CRITICAL]}</div>
            </div>
            <div style="padding:15px;background:#e67e2222;border-radius:8px;flex:1;">
                <strong style="color:#e67e22;">HIGH</strong>
                <div style="font-size:2em;">{summary_counts[RiskLevel.HIGH]}</div>
            </div>
            <div style="padding:15px;background:#f39c1222;border-radius:8px;flex:1;">
                <strong style="color:#f39c12;">MEDIUM</strong>
                <div style="font-size:2em;">{summary_counts[RiskLevel.MEDIUM]}</div>
            </div>
            <div style="padding:15px;background:#27ae6022;border-radius:8px;flex:1;">
                <strong style="color:#27ae60;">LOW</strong>
                <div style="font-size:2em;">{summary_counts[RiskLevel.LOW]}</div>
            </div>
        </div>

        <h2>File Risk Details</h2>
        <table style="width:100%;border-collapse:collapse;">
            <thead>
                <tr style="background:#2c3e50;color:white;">
                    <th style="padding:10px;text-align:left;">File</th>
                    <th style="padding:10px;">Risk Level</th>
                    <th style="padding:10px;">Score</th>
                    <th style="padding:10px;">LOC</th>
                    <th style="padding:10px;">Unsafe</th>
                    <th style="padding:10px;text-align:left;">Reasons</th>
                </tr>
            </thead>
            <tbody>
                {"".join(rows)}
            </tbody>
        </table>
    """)


def effort_timeline(c_dir: str) -> Timeline:
    """Estimate a phased migration timeline.

    Analyzes the C project to determine dependency order, groups files
    into phases, and estimates hours per phase.

    Args:
        c_dir: Root directory of the C project

    Returns:
        Timeline with ordered phases and effort estimates
    """
    c_files = _scan_c_dir(c_dir)

    # Sort by dependencies (fewer deps first) then by risk
    risks = {f["path"]: _assess_risk(f) for f in c_files}
    efforts = {f["path"]: _estimate_effort_hours(f) for f in c_files}

    # Phase 1: Leaf files (few dependencies, low risk)
    # Phase 2: Utility/internal files
    # Phase 3: Core files (high dependency, high risk)
    # Phase 4: Integration & verification

    leaves = [f for f in c_files if f["dependencies"] <= 3]
    internals = [f for f in c_files
                 if 3 < f["dependencies"] <= 8
                 and f not in leaves]
    core = [f for f in c_files if f not in leaves and f not in internals]

    phases = []

    if leaves:
        phases.append(TimelinePhase(
            name="Phase 1: Leaf Modules",
            files=[f["path"] for f in leaves],
            estimated_hours=sum(efforts[f["path"]] for f in leaves),
            risk=RiskLevel.LOW,
        ))

    if internals:
        phases.append(TimelinePhase(
            name="Phase 2: Internal Modules",
            files=[f["path"] for f in internals],
            estimated_hours=sum(efforts[f["path"]] for f in internals),
            dependencies=["Phase 1: Leaf Modules"] if leaves else [],
            risk=RiskLevel.MEDIUM,
        ))

    if core:
        phases.append(TimelinePhase(
            name="Phase 3: Core Modules",
            files=[f["path"] for f in core],
            estimated_hours=sum(efforts[f["path"]] for f in core),
            dependencies=["Phase 2: Internal Modules"] if internals else [],
            risk=RiskLevel.HIGH,
        ))

    # Always add verification phase
    verify_hours = sum(efforts.values()) * 0.3  # 30% of total for verification
    phases.append(TimelinePhase(
        name="Phase 4: Integration & Verification",
        files=["all"],
        estimated_hours=verify_hours,
        dependencies=[p.name for p in phases],
        risk=RiskLevel.MEDIUM,
    ))

    total_hours = sum(p.estimated_hours for p in phases)
    # Critical path = sequential phases
    critical_hours = sum(p.estimated_hours for p in phases)

    return Timeline(
        phases=phases,
        total_hours=total_hours,
        total_weeks=total_hours / 40.0,
        critical_path_hours=critical_hours,
    )


def team_assignment(c_dir: str,
                    n_developers: int = 3) -> List[Assignment]:
    """Assign migration work to developers.

    Uses a greedy load-balancing algorithm to distribute files across
    developers, considering risk levels and specialization.

    Args:
        c_dir: Root directory of the C project
        n_developers: Number of developers available

    Returns:
        List of Assignment objects, one per developer
    """
    c_files = _scan_c_dir(c_dir)
    risks = {f["path"]: _assess_risk(f) for f in c_files}
    efforts = {f["path"]: _estimate_effort_hours(f) for f in c_files}

    # Sort files by effort (descending) for greedy assignment
    sorted_files = sorted(c_files, key=lambda f: -efforts[f["path"]])

    assignments = [
        Assignment(
            developer_id=i + 1,
            specialization=_developer_specialization(i),
        )
        for i in range(n_developers)
    ]

    # Greedy: assign each file to developer with least current load
    for f in sorted_files:
        # Find developer with minimum current hours
        min_dev = min(assignments, key=lambda a: a.estimated_hours)
        min_dev.files.append(f["path"])
        min_dev.estimated_hours += efforts[f["path"]]

        # Track highest risk
        file_risk = risks[f["path"]].risk_level
        risk_order = {
            RiskLevel.LOW: 0, RiskLevel.MEDIUM: 1,
            RiskLevel.HIGH: 2, RiskLevel.CRITICAL: 3,
        }
        if risk_order[file_risk] > risk_order[min_dev.risk_level]:
            min_dev.risk_level = file_risk

    return assignments


def _developer_specialization(idx: int) -> str:
    """Assign a specialization label based on developer index."""
    specializations = [
        "systems/unsafe",
        "algorithms/data-structures",
        "integration/ffi",
        "testing/verification",
        "concurrency/async",
    ]
    return specializations[idx % len(specializations)]


def generate_dashboard(c_dir: str, rust_dir: str) -> str:
    """Generate a complete HTML migration dashboard.

    Args:
        c_dir: Root directory of the C project
        rust_dir: Root directory of the Rust translation

    Returns:
        HTML string for the full dashboard
    """
    report = migration_progress_report(c_dir, rust_dir)
    tl = effort_timeline(c_dir)
    assignments = team_assignment(c_dir)

    # Progress bar
    pct = report.overall_progress * 100
    bar_color = "#27ae60" if pct > 75 else "#f39c12" if pct > 40 else "#e74c3c"

    # File status rows
    file_rows = []
    for mf in report.files:
        status_color = {
            "pending": "#e74c3c",
            "in_progress": "#f39c12",
            "migrated": "#3498db",
            "verified": "#27ae60",
            "skipped": "#95a5a6",
        }.get(mf.status, "#95a5a6")

        risk_color = {
            RiskLevel.LOW: "#27ae60",
            RiskLevel.MEDIUM: "#f39c12",
            RiskLevel.HIGH: "#e67e22",
            RiskLevel.CRITICAL: "#e74c3c",
        }[mf.risk]

        fn_pct = mf.progress * 100

        file_rows.append(f"""
            <tr>
                <td style="padding:6px 10px;border-bottom:1px solid #eee;">
                    {mf.c_path}
                </td>
                <td style="padding:6px 10px;border-bottom:1px solid #eee;text-align:center;">
                    <span style="color:{status_color};font-weight:bold;">
                        {mf.status.upper()}
                    </span>
                </td>
                <td style="padding:6px 10px;border-bottom:1px solid #eee;text-align:center;">
                    <span style="color:{risk_color};">
                        {mf.risk.value.upper()}
                    </span>
                </td>
                <td style="padding:6px 10px;border-bottom:1px solid #eee;text-align:center;">
                    {mf.functions_migrated}/{mf.functions_total}
                </td>
                <td style="padding:6px 10px;border-bottom:1px solid #eee;">
                    <div style="background:#eee;border-radius:4px;overflow:hidden;">
                        <div style="width:{fn_pct:.0f}%;background:{bar_color};
                                    height:16px;border-radius:4px;"></div>
                    </div>
                </td>
            </tr>
        """)

    # Timeline rows
    timeline_rows = []
    for phase in tl.phases:
        timeline_rows.append(f"""
            <tr>
                <td style="padding:8px;border-bottom:1px solid #eee;font-weight:bold;">
                    {phase.name}
                </td>
                <td style="padding:8px;border-bottom:1px solid #eee;text-align:center;">
                    {len(phase.files)} files
                </td>
                <td style="padding:8px;border-bottom:1px solid #eee;text-align:center;">
                    {phase.estimated_hours:.0f}h ({phase.estimated_days:.1f}d)
                </td>
                <td style="padding:8px;border-bottom:1px solid #eee;">
                    {phase.risk.value.upper()}
                </td>
            </tr>
        """)

    # Assignment rows
    assign_rows = []
    for a in assignments:
        assign_rows.append(f"""
            <tr>
                <td style="padding:8px;border-bottom:1px solid #eee;">
                    Developer {a.developer_id}
                </td>
                <td style="padding:8px;border-bottom:1px solid #eee;">
                    {a.specialization}
                </td>
                <td style="padding:8px;border-bottom:1px solid #eee;text-align:center;">
                    {len(a.files)}
                </td>
                <td style="padding:8px;border-bottom:1px solid #eee;text-align:center;">
                    {a.estimated_hours:.0f}h ({a.estimated_days:.1f}d)
                </td>
                <td style="padding:8px;border-bottom:1px solid #eee;">
                    {a.risk_level.value.upper()}
                </td>
            </tr>
        """)

    return _html_page("Migration Dashboard", f"""
        <h2>Overall Progress</h2>
        <div style="display:flex;gap:20px;margin-bottom:30px;flex-wrap:wrap;">
            <div style="padding:20px;background:#f8f9fa;border-radius:10px;flex:1;
                        min-width:150px;text-align:center;">
                <div style="color:#666;font-size:0.9em;">Progress</div>
                <div style="font-size:2.5em;font-weight:bold;color:{bar_color};">
                    {pct:.0f}%
                </div>
            </div>
            <div style="padding:20px;background:#f8f9fa;border-radius:10px;flex:1;
                        min-width:150px;text-align:center;">
                <div style="color:#666;font-size:0.9em;">Files</div>
                <div style="font-size:2.5em;font-weight:bold;">
                    {report.total_rust_files}/{report.total_c_files}
                </div>
            </div>
            <div style="padding:20px;background:#f8f9fa;border-radius:10px;flex:1;
                        min-width:150px;text-align:center;">
                <div style="color:#666;font-size:0.9em;">Functions</div>
                <div style="font-size:2.5em;font-weight:bold;">
                    {report.migrated_functions}/{report.total_functions}
                </div>
            </div>
            <div style="padding:20px;background:#f8f9fa;border-radius:10px;flex:1;
                        min-width:150px;text-align:center;">
                <div style="color:#666;font-size:0.9em;">Verified</div>
                <div style="font-size:2.5em;font-weight:bold;color:#27ae60;">
                    {report.verified_functions}
                </div>
            </div>
            <div style="padding:20px;background:#f8f9fa;border-radius:10px;flex:1;
                        min-width:150px;text-align:center;">
                <div style="color:#666;font-size:0.9em;">Est. Remaining</div>
                <div style="font-size:2.5em;font-weight:bold;color:#e67e22;">
                    {report.estimated_remaining_hours:.0f}h
                </div>
            </div>
        </div>

        <div style="background:#eee;border-radius:8px;overflow:hidden;height:30px;
                    margin-bottom:30px;">
            <div style="width:{pct:.0f}%;background:{bar_color};height:100%;
                        border-radius:8px;transition:width 0.5s;"></div>
        </div>

        <h2>File Status</h2>
        <table style="width:100%;border-collapse:collapse;margin-bottom:30px;">
            <thead>
                <tr style="background:#2c3e50;color:white;">
                    <th style="padding:10px;text-align:left;">File</th>
                    <th style="padding:10px;">Status</th>
                    <th style="padding:10px;">Risk</th>
                    <th style="padding:10px;">Functions</th>
                    <th style="padding:10px;width:200px;">Progress</th>
                </tr>
            </thead>
            <tbody>
                {"".join(file_rows)}
            </tbody>
        </table>

        <h2>Migration Timeline</h2>
        <div style="margin-bottom:10px;color:#666;">
            Total estimate: <strong>{tl.total_hours:.0f} hours</strong>
            ({tl.total_weeks:.1f} weeks)
        </div>
        <table style="width:100%;border-collapse:collapse;margin-bottom:30px;">
            <thead>
                <tr style="background:#2c3e50;color:white;">
                    <th style="padding:10px;text-align:left;">Phase</th>
                    <th style="padding:10px;">Files</th>
                    <th style="padding:10px;">Effort</th>
                    <th style="padding:10px;">Risk</th>
                </tr>
            </thead>
            <tbody>
                {"".join(timeline_rows)}
            </tbody>
        </table>

        <h2>Team Assignments</h2>
        <table style="width:100%;border-collapse:collapse;">
            <thead>
                <tr style="background:#2c3e50;color:white;">
                    <th style="padding:10px;text-align:left;">Developer</th>
                    <th style="padding:10px;text-align:left;">Specialization</th>
                    <th style="padding:10px;">Files</th>
                    <th style="padding:10px;">Effort</th>
                    <th style="padding:10px;">Max Risk</th>
                </tr>
            </thead>
            <tbody>
                {"".join(assign_rows)}
            </tbody>
        </table>
    """)


def _html_page(title: str, body: str) -> str:
    """Wrap content in a complete HTML page."""
    return textwrap.dedent(f"""\
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>XEquiv — {title}</title>
            <style>
                * {{ box-sizing: border-box; margin: 0; padding: 0; }}
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI',
                                 Roboto, Oxygen, sans-serif;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px 30px;
                    background: #fff;
                    color: #2c3e50;
                }}
                h1 {{
                    font-size: 1.8em;
                    margin-bottom: 5px;
                    color: #2c3e50;
                }}
                h2 {{
                    font-size: 1.3em;
                    margin: 25px 0 12px;
                    color: #34495e;
                    border-bottom: 2px solid #ecf0f1;
                    padding-bottom: 5px;
                }}
                table {{ font-size: 0.9em; }}
                th {{ text-align: left; }}
                .subtitle {{
                    color: #7f8c8d;
                    margin-bottom: 20px;
                    font-size: 0.95em;
                }}
            </style>
        </head>
        <body>
            <h1>XEquiv — {title}</h1>
            <p class="subtitle">
                Generated by XEquiv Cross-Language Migration Platform
            </p>
            {body}
            <footer style="margin-top:40px;padding-top:15px;border-top:1px solid #ecf0f1;
                           color:#95a5a6;font-size:0.85em;">
                XEquiv Cross-Language Equivalence Verifier &mdash;
                Report generated at <script>document.write(new Date().toLocaleString())</script>
            </footer>
        </body>
        </html>
    """)
