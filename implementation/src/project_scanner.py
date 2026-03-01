"""Scan C and Rust projects to find and verify translated function pairs.

Extends XEquiv from single-pair verification to whole-project migration
verification. Walks directory trees, extracts function signatures, matches
C→Rust pairs by name similarity and type signatures, and orchestrates
parallel verification across all matched pairs.
"""

import os
import re
import json
import time
import hashlib
import fnmatch
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, Callable
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from difflib import SequenceMatcher

from .api import verify_equivalence, verify_files, VerificationResult, Divergence


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CFunction:
    name: str
    return_type: str
    params: List[Tuple[str, str]]  # (type, name) pairs
    body: str
    file_path: str
    line_number: int
    raw_source: str

    @property
    def signature(self) -> str:
        param_str = ", ".join(f"{t} {n}" for t, n in self.params)
        return f"{self.return_type} {self.name}({param_str})"


@dataclass
class RustFunction:
    name: str
    return_type: str
    params: List[Tuple[str, str]]  # (name, type) pairs — Rust ordering
    body: str
    file_path: str
    line_number: int
    raw_source: str
    is_unsafe: bool = False
    is_pub: bool = False

    @property
    def signature(self) -> str:
        param_str = ", ".join(f"{n}: {t}" for n, t in self.params)
        ret = f" -> {self.return_type}" if self.return_type != "()" else ""
        return f"fn {self.name}({param_str}){ret}"


@dataclass
class FunctionMatch:
    c_function: CFunction
    rust_function: RustFunction
    confidence: float  # 0..1
    match_reason: str  # "exact_name", "fuzzy_name", "signature", "hybrid"


@dataclass
class FunctionVerification:
    match: FunctionMatch
    result: Optional[VerificationResult] = None
    status: str = "pending"  # pending | running | passed | failed | error
    error_message: str = ""


@dataclass
class ProjectScanResult:
    c_functions: List[CFunction]
    rust_functions: List[RustFunction]
    matches: List[FunctionMatch]
    unmatched_c: List[CFunction]
    unmatched_rust: List[RustFunction]
    scan_duration_ms: float = 0.0

    @property
    def match_rate(self) -> float:
        total = len(self.c_functions)
        return len(self.matches) / total if total > 0 else 0.0


@dataclass
class MigrationStatus:
    total_c_functions: int
    matched: int
    verified_equivalent: int
    verified_divergent: int
    unverified: int
    unmatched: int
    duration_ms: float = 0.0


@dataclass
class ProjectVerificationResult:
    scan: ProjectScanResult
    verifications: List[FunctionVerification]
    status: MigrationStatus
    sarif: Optional[dict] = None
    duration_ms: float = 0.0


# ---------------------------------------------------------------------------
# Parsers — lightweight regex-based extraction
# ---------------------------------------------------------------------------

_C_FUNC_RE = re.compile(
    r"(?P<ret>(?:unsigned\s+|signed\s+|const\s+|static\s+|inline\s+)*"
    r"(?:void|int|long|short|char|float|double|size_t|ssize_t|uint\d+_t|int\d+_t|bool)\s*\*?\s*)"
    r"(?P<name>[a-zA-Z_]\w*)\s*"
    r"\((?P<params>[^)]*)\)\s*\{",
    re.MULTILINE,
)

_RUST_FUNC_RE = re.compile(
    r"(?P<prefix>(?:pub\s+)?(?:unsafe\s+)?)"
    r"fn\s+(?P<name>[a-zA-Z_]\w*)\s*"
    r"\((?P<params>[^)]*)\)"
    r"(?:\s*->\s*(?P<ret>[^{]+?))?\s*\{",
    re.MULTILINE,
)


def _parse_c_params(raw: str) -> List[Tuple[str, str]]:
    raw = raw.strip()
    if not raw or raw == "void":
        return []
    params = []
    for part in raw.split(","):
        part = part.strip()
        tokens = part.rsplit(None, 1)
        if len(tokens) == 2:
            params.append((tokens[0].strip(), tokens[1].strip("* ")))
        elif tokens:
            params.append((tokens[0], ""))
    return params


def _parse_rust_params(raw: str) -> List[Tuple[str, str]]:
    raw = raw.strip()
    if not raw or raw == "&self" or raw == "&mut self":
        return [(raw, "self")] if raw else []
    params = []
    for part in raw.split(","):
        part = part.strip()
        if ":" in part:
            name, typ = part.split(":", 1)
            params.append((name.strip(), typ.strip()))
        elif part:
            params.append((part, ""))
    return params


def _extract_brace_block(source: str, start: int) -> str:
    """Return body text from opening brace at *start* to its matching close."""
    depth = 0
    i = start
    while i < len(source):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[start : i + 1]
        i += 1
    return source[start:]


def extract_c_functions(source: str, file_path: str = "<inline>") -> List[CFunction]:
    functions: List[CFunction] = []
    for m in _C_FUNC_RE.finditer(source):
        body = _extract_brace_block(source, m.start())
        line = source[: m.start()].count("\n") + 1
        functions.append(CFunction(
            name=m.group("name"),
            return_type=m.group("ret").strip(),
            params=_parse_c_params(m.group("params")),
            body=body,
            file_path=file_path,
            line_number=line,
            raw_source=body,
        ))
    return functions


def extract_rust_functions(source: str, file_path: str = "<inline>") -> List[RustFunction]:
    functions: List[RustFunction] = []
    for m in _RUST_FUNC_RE.finditer(source):
        body = _extract_brace_block(source, m.end() - 1)
        line = source[: m.start()].count("\n") + 1
        prefix = m.group("prefix") or ""
        ret = (m.group("ret") or "()").strip()
        functions.append(RustFunction(
            name=m.group("name"),
            return_type=ret,
            params=_parse_rust_params(m.group("params")),
            body=body,
            file_path=file_path,
            line_number=line,
            raw_source=m.group() + body,
            is_unsafe="unsafe" in prefix,
            is_pub="pub" in prefix,
        ))
    return functions


# ---------------------------------------------------------------------------
# Directory scanning
# ---------------------------------------------------------------------------

def _collect_files(directory: str, extensions: Set[str],
                   ignore_patterns: Optional[List[str]] = None) -> List[str]:
    ignore_patterns = ignore_patterns or ["build", "target", ".git", "node_modules"]
    collected: List[str] = []
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if not any(fnmatch.fnmatch(d, p) for p in ignore_patterns)]
        for f in files:
            if any(f.endswith(ext) for ext in extensions):
                collected.append(os.path.join(root, f))
    return sorted(collected)


def scan_project(c_dir: str, rust_dir: str,
                 ignore_patterns: Optional[List[str]] = None,
                 match_strategy: str = "hybrid") -> ProjectScanResult:
    """Scan two project directories and auto-match function pairs by name/signature."""
    start = time.time()

    c_files = _collect_files(c_dir, {".c", ".h"}, ignore_patterns)
    rust_files = _collect_files(rust_dir, {".rs"}, ignore_patterns)

    c_functions: List[CFunction] = []
    for fp in c_files:
        with open(fp) as fh:
            c_functions.extend(extract_c_functions(fh.read(), fp))

    rust_functions: List[RustFunction] = []
    for fp in rust_files:
        with open(fp) as fh:
            rust_functions.extend(extract_rust_functions(fh.read(), fp))

    matches = match_functions(c_functions, rust_functions, strategy=match_strategy)

    matched_c = {id(m.c_function) for m in matches}
    matched_rust = {id(m.rust_function) for m in matches}

    return ProjectScanResult(
        c_functions=c_functions,
        rust_functions=rust_functions,
        matches=matches,
        unmatched_c=[f for f in c_functions if id(f) not in matched_c],
        unmatched_rust=[f for f in rust_functions if id(f) not in matched_rust],
        scan_duration_ms=(time.time() - start) * 1000,
    )


# ---------------------------------------------------------------------------
# Function matching
# ---------------------------------------------------------------------------

_C_TO_RUST_TYPES: Dict[str, str] = {
    "int": "i32", "unsigned int": "u32", "unsigned": "u32",
    "long": "i64", "unsigned long": "u64",
    "short": "i16", "unsigned short": "u16",
    "char": "i8", "unsigned char": "u8",
    "float": "f32", "double": "f64",
    "void": "()", "bool": "bool", "_Bool": "bool",
    "size_t": "usize", "ssize_t": "isize",
    "int8_t": "i8", "int16_t": "i16", "int32_t": "i32", "int64_t": "i64",
    "uint8_t": "u8", "uint16_t": "u16", "uint32_t": "u32", "uint64_t": "u64",
}


def _normalise_c_name(name: str) -> str:
    """Strip common C prefixes/suffixes for fuzzy comparison."""
    name = name.lower()
    for prefix in ("my_", "c_", "lib_", "__"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name


def _name_similarity(c_name: str, rust_name: str) -> float:
    c_norm = _normalise_c_name(c_name)
    r_norm = rust_name.lower().replace("_", "")
    c_clean = c_norm.replace("_", "")
    if c_clean == r_norm:
        return 1.0
    return SequenceMatcher(None, c_clean, r_norm).ratio()


def _type_compatible(c_type: str, rust_type: str) -> bool:
    c_type = c_type.strip().replace("const ", "").replace("*", "").strip()
    rust_type = rust_type.strip().lstrip("&").replace("mut ", "").strip()
    mapped = _C_TO_RUST_TYPES.get(c_type)
    if mapped and mapped == rust_type:
        return True
    if c_type == rust_type:
        return True
    return False


def _signature_score(c_fn: CFunction, r_fn: RustFunction) -> float:
    if not _type_compatible(c_fn.return_type, r_fn.return_type):
        return 0.0
    if len(c_fn.params) != len(r_fn.params):
        return 0.0
    if len(c_fn.params) == 0:
        return 0.6
    type_matches = sum(
        1 for (ct, _), (_, rt) in zip(c_fn.params, r_fn.params)
        if _type_compatible(ct, rt)
    )
    return type_matches / len(c_fn.params)


def match_functions(c_functions: List[CFunction],
                    rust_functions: List[RustFunction],
                    strategy: str = "hybrid",
                    threshold: float = 0.5) -> List[FunctionMatch]:
    """Match C functions to Rust equivalents using name similarity + type signatures."""
    matches: List[FunctionMatch] = []
    used_rust: Set[int] = set()

    scored_pairs: List[Tuple[float, str, int, int]] = []

    for ci, cf in enumerate(c_functions):
        for ri, rf in enumerate(rust_functions):
            name_sim = _name_similarity(cf.name, rf.name)
            sig_score = _signature_score(cf, rf)

            if strategy == "name":
                score = name_sim
                reason = "name"
            elif strategy == "signature":
                score = sig_score
                reason = "signature"
            else:  # hybrid
                score = 0.6 * name_sim + 0.4 * sig_score
                reason = "hybrid"
                if name_sim == 1.0:
                    score = max(score, 0.9)
                    reason = "exact_name"

            if score >= threshold:
                scored_pairs.append((score, reason, ci, ri))

    scored_pairs.sort(key=lambda t: t[0], reverse=True)

    used_c: Set[int] = set()
    for score, reason, ci, ri in scored_pairs:
        if ci in used_c or ri in used_rust:
            continue
        matches.append(FunctionMatch(
            c_function=c_functions[ci],
            rust_function=rust_functions[ri],
            confidence=score,
            match_reason=reason,
        ))
        used_c.add(ci)
        used_rust.add(ri)

    return matches


# ---------------------------------------------------------------------------
# Verification orchestration
# ---------------------------------------------------------------------------

def _verify_single(match: FunctionMatch, timeout_s: float,
                   method: str) -> FunctionVerification:
    fv = FunctionVerification(match=match, status="running")
    try:
        result = verify_equivalence(
            match.c_function.raw_source,
            match.rust_function.raw_source,
            timeout_s=timeout_s,
            method=method,
        )
        fv.result = result
        fv.status = "passed" if result.equivalent else "failed"
    except Exception as exc:
        fv.status = "error"
        fv.error_message = str(exc)
    return fv


def _build_sarif(verifications: List[FunctionVerification]) -> dict:
    """Build a SARIF v2.1.0 log from verification results."""
    results_list = []
    for fv in verifications:
        if fv.status != "failed" or fv.result is None:
            continue
        for div in fv.result.divergences:
            results_list.append({
                "ruleId": f"xequiv/{div.category}",
                "level": "error" if div.severity == "critical" else "warning",
                "message": {"text": div.description},
                "locations": [{
                    "physicalLocation": {
                        "artifactLocation": {
                            "uri": fv.match.c_function.file_path,
                        },
                        "region": {
                            "startLine": fv.match.c_function.line_number,
                        },
                    }
                }],
                "relatedLocations": [{
                    "physicalLocation": {
                        "artifactLocation": {
                            "uri": fv.match.rust_function.file_path,
                        },
                        "region": {
                            "startLine": fv.match.rust_function.line_number,
                        },
                    },
                    "message": {"text": f"Rust: {div.rust_behavior}"},
                }],
            })

    return {
        "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/main/sarif-2.1/schema/sarif-schema-2.1.0.json",
        "version": "2.1.0",
        "runs": [{
            "tool": {
                "driver": {
                    "name": "XEquiv",
                    "version": "0.2.0",
                    "informationUri": "https://github.com/xequiv/xequiv",
                    "rules": [
                        {"id": f"xequiv/{cat}", "shortDescription": {"text": cat.replace("_", " ").title()}}
                        for cat in {d.category for fv in verifications
                                    if fv.result for d in fv.result.divergences}
                    ],
                }
            },
            "results": results_list,
        }],
    }


def verify_project(c_dir: str, rust_dir: str,
                   parallel: int = 4,
                   timeout_s: float = 120.0,
                   method: str = "hybrid",
                   output_format: str = "sarif",
                   match_strategy: str = "hybrid",
                   match_threshold: float = 0.5) -> ProjectVerificationResult:
    """Full project verification with parallel execution and SARIF output."""
    start = time.time()

    scan = scan_project(c_dir, rust_dir, match_strategy=match_strategy)

    verifications: List[FunctionVerification] = []

    if parallel <= 1:
        for m in scan.matches:
            verifications.append(_verify_single(m, timeout_s, method))
    else:
        with ProcessPoolExecutor(max_workers=parallel) as pool:
            futures = {
                pool.submit(_verify_single, m, timeout_s, method): m
                for m in scan.matches
            }
            for future in as_completed(futures):
                verifications.append(future.result())

    equiv_count = sum(1 for fv in verifications if fv.status == "passed")
    divergent_count = sum(1 for fv in verifications if fv.status == "failed")
    error_count = sum(1 for fv in verifications if fv.status == "error")

    status = MigrationStatus(
        total_c_functions=len(scan.c_functions),
        matched=len(scan.matches),
        verified_equivalent=equiv_count,
        verified_divergent=divergent_count,
        unverified=error_count,
        unmatched=len(scan.unmatched_c),
        duration_ms=(time.time() - start) * 1000,
    )

    sarif = _build_sarif(verifications) if output_format == "sarif" else None

    return ProjectVerificationResult(
        scan=scan,
        verifications=verifications,
        status=status,
        sarif=sarif,
        duration_ms=(time.time() - start) * 1000,
    )


# ---------------------------------------------------------------------------
# Migration tracker
# ---------------------------------------------------------------------------

class MigrationTracker:
    """Track C→Rust migration progress with verification status per function."""

    def __init__(self, c_dir: str, rust_dir: str,
                 state_file: Optional[str] = None):
        self.c_dir = c_dir
        self.rust_dir = rust_dir
        self.state_file = state_file or os.path.join(
            rust_dir, ".xequiv-migration.json"
        )
        self._scan: Optional[ProjectScanResult] = None
        self._verifications: Dict[str, FunctionVerification] = {}
        self._load_state()

    def _load_state(self) -> None:
        if os.path.exists(self.state_file):
            with open(self.state_file) as f:
                data = json.load(f)
            for key, entry in data.get("verifications", {}).items():
                self._verifications[key] = FunctionVerification(
                    match=FunctionMatch(
                        c_function=CFunction(
                            name=entry["c_name"], return_type="", params=[],
                            body="", file_path=entry.get("c_file", ""),
                            line_number=0, raw_source="",
                        ),
                        rust_function=RustFunction(
                            name=entry["rust_name"], return_type="", params=[],
                            body="", file_path=entry.get("rust_file", ""),
                            line_number=0, raw_source="",
                        ),
                        confidence=entry.get("confidence", 0.0),
                        match_reason=entry.get("match_reason", ""),
                    ),
                    status=entry.get("status", "pending"),
                )

    def _save_state(self) -> None:
        data = {"verifications": {}}
        for key, fv in self._verifications.items():
            data["verifications"][key] = {
                "c_name": fv.match.c_function.name,
                "rust_name": fv.match.rust_function.name,
                "c_file": fv.match.c_function.file_path,
                "rust_file": fv.match.rust_function.file_path,
                "confidence": fv.match.confidence,
                "match_reason": fv.match.match_reason,
                "status": fv.status,
            }
        os.makedirs(os.path.dirname(self.state_file) or ".", exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(data, f, indent=2)

    def _match_key(self, m: FunctionMatch) -> str:
        return f"{m.c_function.file_path}::{m.c_function.name}|{m.rust_function.file_path}::{m.rust_function.name}"

    def rescan(self) -> ProjectScanResult:
        self._scan = scan_project(self.c_dir, self.rust_dir)
        for m in self._scan.matches:
            key = self._match_key(m)
            if key not in self._verifications:
                self._verifications[key] = FunctionVerification(
                    match=m, status="pending"
                )
        self._save_state()
        return self._scan

    def verify_all(self, parallel: int = 4, method: str = "hybrid",
                   timeout_s: float = 120.0) -> None:
        if self._scan is None:
            self.rescan()
        pending = [
            (key, fv) for key, fv in self._verifications.items()
            if fv.status in ("pending", "error")
        ]
        for key, fv in pending:
            fv.status = "running"
            try:
                result = verify_equivalence(
                    fv.match.c_function.raw_source,
                    fv.match.rust_function.raw_source,
                    timeout_s=timeout_s,
                    method=method,
                )
                fv.result = result
                fv.status = "passed" if result.equivalent else "failed"
            except Exception as exc:
                fv.status = "error"
                fv.error_message = str(exc)
        self._save_state()

    def status(self) -> MigrationStatus:
        if self._scan is None:
            self.rescan()
        total = len(self._scan.c_functions) if self._scan else 0
        matched = len(self._verifications)
        equiv = sum(1 for fv in self._verifications.values() if fv.status == "passed")
        div = sum(1 for fv in self._verifications.values() if fv.status == "failed")
        pending = sum(1 for fv in self._verifications.values() if fv.status in ("pending", "running", "error"))
        unmatched = total - matched
        return MigrationStatus(
            total_c_functions=total,
            matched=matched,
            verified_equivalent=equiv,
            verified_divergent=div,
            unverified=pending,
            unmatched=max(unmatched, 0),
        )

    def verified_percentage(self) -> float:
        s = self.status()
        total = s.total_c_functions
        return (s.verified_equivalent / total * 100.0) if total > 0 else 0.0

    def unverified_functions(self) -> List[str]:
        return [
            fv.match.c_function.name
            for fv in self._verifications.values()
            if fv.status not in ("passed",)
        ]

    def generate_report(self, format: str = "html") -> str:
        s = self.status()
        if format == "html":
            return self._html_report(s)
        elif format == "markdown":
            return self._markdown_report(s)
        return self._text_report(s)

    def _text_report(self, s: MigrationStatus) -> str:
        lines = [
            "XEquiv Migration Report",
            "=" * 40,
            f"Total C functions:       {s.total_c_functions}",
            f"Matched to Rust:         {s.matched}",
            f"Verified equivalent:     {s.verified_equivalent}",
            f"Divergences found:       {s.verified_divergent}",
            f"Unverified:              {s.unverified}",
            f"Unmatched:               {s.unmatched}",
            f"Migration verified:      {self.verified_percentage():.1f}%",
        ]
        return "\n".join(lines)

    def _markdown_report(self, s: MigrationStatus) -> str:
        lines = [
            "# XEquiv Migration Report\n",
            f"| Metric | Count |",
            f"|--------|-------|",
            f"| Total C functions | {s.total_c_functions} |",
            f"| Matched to Rust | {s.matched} |",
            f"| Verified equivalent | {s.verified_equivalent} |",
            f"| Divergences found | {s.verified_divergent} |",
            f"| Unverified | {s.unverified} |",
            f"| Unmatched | {s.unmatched} |",
            f"\n**Migration verified: {self.verified_percentage():.1f}%**\n",
            "## Per-function status\n",
            "| C Function | Rust Function | Status |",
            "|------------|---------------|--------|",
        ]
        for fv in self._verifications.values():
            icon = {"passed": "✅", "failed": "❌", "error": "⚠️"}.get(fv.status, "⏳")
            lines.append(
                f"| `{fv.match.c_function.name}` | `{fv.match.rust_function.name}` | {icon} {fv.status} |"
            )
        return "\n".join(lines)

    def _html_report(self, s: MigrationStatus) -> str:
        pct = self.verified_percentage()
        bar_color = "#4caf50" if pct > 80 else "#ff9800" if pct > 50 else "#f44336"
        rows = []
        for fv in self._verifications.values():
            color = {"passed": "#4caf50", "failed": "#f44336", "error": "#ff9800"}.get(fv.status, "#9e9e9e")
            rows.append(
                f"<tr><td>{fv.match.c_function.name}</td>"
                f"<td>{fv.match.rust_function.name}</td>"
                f"<td style='color:{color}'>{fv.status}</td>"
                f"<td>{fv.match.confidence:.0%}</td></tr>"
            )
        return f"""<!DOCTYPE html>
<html><head><title>XEquiv Migration Report</title>
<style>
body {{ font-family: -apple-system, sans-serif; max-width: 900px; margin: 40px auto; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
th {{ background: #f5f5f5; }}
.bar {{ height: 24px; border-radius: 4px; }}
</style></head><body>
<h1>XEquiv Migration Report</h1>
<div style="background:#eee;border-radius:4px;overflow:hidden;margin:16px 0">
  <div class="bar" style="width:{pct:.0f}%;background:{bar_color}"></div>
</div>
<p><strong>{pct:.1f}%</strong> of C functions verified equivalent in Rust</p>
<table>
<tr><th>Metric</th><th>Count</th></tr>
<tr><td>Total C functions</td><td>{s.total_c_functions}</td></tr>
<tr><td>Matched to Rust</td><td>{s.matched}</td></tr>
<tr><td>Verified equivalent</td><td>{s.verified_equivalent}</td></tr>
<tr><td>Divergences found</td><td>{s.verified_divergent}</td></tr>
<tr><td>Unverified</td><td>{s.unverified}</td></tr>
<tr><td>Unmatched</td><td>{s.unmatched}</td></tr>
</table>
<h2>Function Details</h2>
<table>
<tr><th>C Function</th><th>Rust Function</th><th>Status</th><th>Match Confidence</th></tr>
{"".join(rows)}
</table>
</body></html>"""
