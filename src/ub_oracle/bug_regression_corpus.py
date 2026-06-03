"""Frozen regression corpus for confirmed extraction-unit divergence findings.

Step 163 turns each responsibly recorded finding into a regression test case:
the minimized source/target units, witness inputs, source hashes, and existing
reproduction bundle are pinned in a byte-stable manifest.  The corpus deliberately
inherits the conservative evidence tier from :mod:`divergence_findings`: these are
confirmed extraction-unit divergences, not upstream CVEs or repository-defect
claims until an upstream source audit says otherwise.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from . import divergence_findings as findings
from .reexec import ReexecHarness, toolchain_available

SCHEMA_VERSION = "bug-regression-corpus/v1"

_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = _ROOT / "experiments" / "bug_regressions"
RESULTS_PATH = EXPERIMENT_DIR / "bug_regression_corpus.json"
DOC_PATH = _ROOT / "docs" / "bug_regression_corpus.md"

REGRESSION_CONTRACT = (
    "finding id appears in divergence_findings.finding_records()",
    "checked C/Rust extraction-unit source hashes match this frozen manifest",
    "symbolic witness and safe input remain attached to the finding record",
    "existing docs/repro reproduction bundle is present and shell-syntax valid",
    "when clang/UBSan and the target compiler are available, witness traps on C "
    "and is defined on the target while the safe control stays silent",
)


@dataclass(frozen=True)
class BugRegressionConfirmation:
    finding_id: str
    available: bool
    confirmed: bool
    ub_reachable: bool
    target_defined: bool
    safe_silent: bool
    bundle_path: str
    detail: str = ""


@dataclass(frozen=True)
class BugRegressionReport:
    available: bool
    ok: bool
    n_regressions: int
    n_confirmed: int
    manifest_valid: bool
    frozen_sources_valid: bool
    bundles_valid: bool
    confirmations: Tuple[BugRegressionConfirmation, ...] = field(default_factory=tuple)
    detail: str = ""


def _canonical_bytes(obj: object) -> bytes:
    return json.dumps(
        obj, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _bundle_path(rec: findings.FindingRecord) -> str:
    return f"docs/repro/{rec.finding_id}.sh"


def _regression_json(rec: findings.FindingRecord) -> Dict[str, object]:
    return {
        "regression_id": f"BUG-REG-{rec.finding_id}",
        "finding_id": rec.finding_id,
        "sample_id": rec.sample_id,
        "candidate_repo": rec.candidate_repo,
        "candidate_url": rec.candidate_url,
        "source_family": rec.source_family,
        "source_lang": "c",
        "target_lang": rec.target_lang,
        "divergence_class": rec.divergence_class,
        "evidence_tier": rec.evidence_tier,
        "upstream_status": rec.upstream_status,
        "witness_input": list(rec.witness_input),
        "safe_input": list(rec.safe_input),
        "c_file": rec.c_file,
        "rust_file": rec.rust_file,
        "c_sha256": rec.c_sha256,
        "rust_sha256": rec.rust_sha256,
        "bundle_path": _bundle_path(rec),
        "reproduction_command": ["bash", _bundle_path(rec)],
        "regression_contract": list(REGRESSION_CONTRACT),
        "remediation": rec.remediation,
    }


def regression_entries(
    records: Optional[Sequence[findings.FindingRecord]] = None,
) -> Tuple[Dict[str, object], ...]:
    records = findings.finding_records() if records is None else records
    return tuple(_regression_json(rec) for rec in records)


def content_hash(entries: Optional[Sequence[Mapping[str, object]]] = None) -> str:
    entries = regression_entries() if entries is None else entries
    stable = [
        {
            "regression_id": entry["regression_id"],
            "finding_id": entry["finding_id"],
            "divergence_class": entry["divergence_class"],
            "target_lang": entry["target_lang"],
            "witness_input": entry["witness_input"],
            "safe_input": entry["safe_input"],
            "c_file": entry["c_file"],
            "rust_file": entry["rust_file"],
            "c_sha256": entry["c_sha256"],
            "rust_sha256": entry["rust_sha256"],
            "bundle_path": entry["bundle_path"],
        }
        for entry in entries
    ]
    return hashlib.sha256(_canonical_bytes(stable)).hexdigest()


def results_document() -> Dict[str, object]:
    entries = regression_entries()
    by_class: Dict[str, int] = {}
    by_target: Dict[str, int] = {}
    for entry in entries:
        cls = str(entry["divergence_class"])
        target = str(entry["target_lang"])
        by_class[cls] = by_class.get(cls, 0) + 1
        by_target[target] = by_target.get(target, 0) + 1
    return {
        "schema": SCHEMA_VERSION,
        "content_hash": content_hash(entries),
        "n_regressions": len(entries),
        "n_source_families": len({str(e["source_family"]) for e in entries}),
        "by_divergence_class": by_class,
        "by_target_lang": by_target,
        "evidence_policy": (
            "Frozen regressions are confirmed extraction-unit divergences derived "
            "from the responsible findings lane; they are not upstream CVEs or "
            "repository-defect claims without a separate upstream source audit."
        ),
        "regressions": list(entries),
    }


def load_results(path: Path = RESULTS_PATH) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_results(path: Path = RESULTS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(results_document(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _entries(doc: Mapping[str, object]) -> Sequence[Mapping[str, object]]:
    raw = doc.get("regressions")
    if not isinstance(raw, list):
        raise AssertionError("regressions must be a list")
    return raw


def validate_manifest(doc: Optional[Mapping[str, object]] = None) -> Tuple[bool, str]:
    doc = load_results() if doc is None else doc
    if doc.get("schema") != SCHEMA_VERSION:
        return False, f"unexpected schema {doc.get('schema')!r}"
    entries = _entries(doc)
    record_ids = {rec.finding_id for rec in findings.finding_records()}
    entry_ids = {str(entry.get("finding_id")) for entry in entries}
    if entry_ids != record_ids:
        missing = sorted(record_ids - entry_ids)
        extra = sorted(entry_ids - record_ids)
        return False, f"finding id drift: missing={missing} extra={extra}"
    if doc.get("n_regressions") != len(entries):
        return False, "n_regressions does not match regressions length"
    expected_hash = content_hash(entries)
    if doc.get("content_hash") != expected_hash:
        return False, "content_hash does not match regression entries"
    for entry in entries:
        if not entry.get("witness_input") or not entry.get("safe_input"):
            return False, f"{entry.get('finding_id')}: missing witness or safe input"
        if str(entry.get("bundle_path", "")).startswith("/"):
            return False, f"{entry.get('finding_id')}: bundle path must be relative"
    return True, "OK"


def validate_frozen_sources(
    doc: Optional[Mapping[str, object]] = None,
) -> Tuple[bool, str]:
    doc = load_results() if doc is None else doc
    for entry in _entries(doc):
        fid = entry.get("finding_id")
        c_path = _ROOT / str(entry.get("c_file"))
        rust_path = _ROOT / str(entry.get("rust_file"))
        if not c_path.exists():
            return False, f"{fid}: missing C source {c_path}"
        if not rust_path.exists():
            return False, f"{fid}: missing Rust source {rust_path}"
        if _sha256_text(_read(c_path)) != entry.get("c_sha256"):
            return False, f"{fid}: C source hash drifted"
        if _sha256_text(_read(rust_path)) != entry.get("rust_sha256"):
            return False, f"{fid}: Rust source hash drifted"
    return True, "OK"


def validate_bundles(doc: Optional[Mapping[str, object]] = None) -> Tuple[bool, str]:
    doc = load_results() if doc is None else doc
    for entry in _entries(doc):
        fid = entry.get("finding_id")
        bundle = _ROOT / str(entry.get("bundle_path"))
        if not bundle.exists():
            return False, f"{fid}: missing reproduction bundle {bundle}"
        text = _read(bundle)
        if str(fid) not in text:
            return False, f"{fid}: bundle does not name its finding id"
        syntax = subprocess.run(
            ["bash", "-n", str(bundle)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if syntax.returncode != 0:
            return False, f"{fid}: bundle syntax failed: {syntax.stderr}"
    return True, "OK"


def check_results(path: Path = RESULTS_PATH) -> Tuple[bool, str]:
    if not path.exists():
        return False, f"{path} is missing"
    regenerated = json.dumps(results_document(), indent=2, sort_keys=True) + "\n"
    on_disk = path.read_text(encoding="utf-8")
    if on_disk != regenerated:
        return False, f"{path} does not match regenerated bug-regression corpus"
    doc = json.loads(on_disk)
    for validator in (validate_manifest, validate_frozen_sources, validate_bundles):
        ok, detail = validator(doc)
        if not ok:
            return ok, detail
    return True, "OK"


def confirm_regressions(path: Path = RESULTS_PATH) -> BugRegressionReport:
    doc = load_results(path)
    manifest_valid, manifest_detail = validate_manifest(doc)
    frozen_valid, frozen_detail = validate_frozen_sources(doc)
    bundles_valid, bundles_detail = validate_bundles(doc)
    records = findings.finding_records()
    n_regressions = len(_entries(doc))
    status = toolchain_available()
    if not all(status.full_for(rec.target_lang) for rec in records):
        ok = manifest_valid and frozen_valid and bundles_valid
        return BugRegressionReport(
            available=False,
            ok=ok,
            n_regressions=n_regressions,
            n_confirmed=0,
            manifest_valid=manifest_valid,
            frozen_sources_valid=frozen_valid,
            bundles_valid=bundles_valid,
            detail=(
                "target toolchain(s) absent; structural regression corpus checks "
                "passed"
                if ok
                else f"manifest={manifest_detail}; sources={frozen_detail}; bundles={bundles_detail}"
            ),
        )

    bundle_by_id = {
        str(entry["finding_id"]): str(entry["bundle_path"]) for entry in _entries(doc)
    }
    harness = ReexecHarness(status)
    confirmations: List[BugRegressionConfirmation] = []
    for rec in records:
        conf = findings.reproduce_finding(rec, harness=harness, write_bundle=False)
        confirmations.append(
            BugRegressionConfirmation(
                finding_id=rec.finding_id,
                available=conf.available,
                confirmed=conf.confirmed,
                ub_reachable=conf.ub_reachable,
                target_defined=conf.target_defined,
                safe_silent=conf.safe_silent,
                bundle_path=bundle_by_id[rec.finding_id],
                detail=conf.detail,
            )
        )
    n_confirmed = sum(1 for conf in confirmations if conf.confirmed)
    live_ok = n_confirmed == len(records)
    ok = manifest_valid and frozen_valid and bundles_valid and live_ok
    return BugRegressionReport(
        available=True,
        ok=ok,
        n_regressions=n_regressions,
        n_confirmed=n_confirmed,
        manifest_valid=manifest_valid,
        frozen_sources_valid=frozen_valid,
        bundles_valid=bundles_valid,
        confirmations=tuple(confirmations),
        detail=(
            "every frozen bug regression replays live without rewriting bundles"
            if ok
            else f"manifest={manifest_detail}; sources={frozen_detail}; bundles={bundles_detail}; live={n_confirmed}/{len(records)}"
        ),
    )


def markdown_document(doc: Optional[Mapping[str, object]] = None) -> str:
    doc = results_document() if doc is None else doc
    lines = [
        "# Frozen bug-regression corpus",
        "",
        "This page is generated by `ub_oracle.bug_regression_corpus`. It freezes "
        "each responsibly recorded divergence finding into a regression case with "
        "pinned source hashes, witness inputs, a safe control, and a checked "
        "reproduction bundle.",
        "",
        "The corpus inherits the evidence policy from `docs/divergence_findings.md`: "
        "these are confirmed extraction-unit divergences, not upstream CVEs or "
        "repository-defect claims without a separate upstream source audit.",
        "",
        f"- **Schema:** `{doc['schema']}`",
        f"- **Regressions:** {doc['n_regressions']}",
        f"- **Content hash:** `{doc['content_hash']}`",
        "",
    ]
    for entry in _entries(doc):
        lines += [
            f"## {entry['finding_id']} — `{entry['sample_id']}`",
            "",
            f"- **Pair / class:** C -> {entry['target_lang']} / `{entry['divergence_class']}`",
            f"- **Repository family:** [{entry['candidate_repo']}]({entry['candidate_url']})",
            f"- **Witness / safe input:** `{' '.join(entry['witness_input'])}` / "
            f"`{' '.join(entry['safe_input'])}`",
            f"- **Frozen sources:** `{entry['c_file']}` (`{str(entry['c_sha256'])[:12]}`), "
            f"`{entry['rust_file']}` (`{str(entry['rust_sha256'])[:12]}`)",
            f"- **Reproduction bundle:** `{entry['bundle_path']}`",
            f"- **Upstream status:** {entry['upstream_status']}",
            "",
            "**Regression contract.** The focused gate recomputes source hashes, "
            "checks that this finding still appears in `finding_records()`, lints "
            "the existing bundle, and, when `clang`/UBSan plus the target compiler "
            "are available, replays the witness and safe control on real code.",
            "",
        ]
    lines.append("Run `make bug-regression-check` to validate the frozen corpus.")
    lines.append("")
    return "\n".join(lines)


def write_docs(path: Path = DOC_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown_document(load_results()), encoding="utf-8")


def check_docs(path: Path = DOC_PATH) -> Tuple[bool, str]:
    if not path.exists():
        return False, f"{path} is missing"
    expected = markdown_document(load_results())
    if path.read_text(encoding="utf-8") != expected:
        return False, f"{path} does not match regenerated bug-regression docs"
    return True, "OK"


def write_artifacts() -> Tuple[Path, Path]:
    write_results()
    write_docs()
    return RESULTS_PATH, DOC_PATH


if __name__ == "__main__":  # pragma: no cover
    results, docs = write_artifacts()
    ok, detail = check_results(results)
    doc_ok, doc_detail = check_docs(docs)
    rep = confirm_regressions(results)
    print(
        "bug-regression-corpus "
        f"manifest={ok} docs={doc_ok} live_available={rep.available} "
        f"confirmed={rep.n_confirmed}/{rep.n_regressions}"
    )
    if not ok:
        print(detail)
    if not doc_ok:
        print(doc_detail)
