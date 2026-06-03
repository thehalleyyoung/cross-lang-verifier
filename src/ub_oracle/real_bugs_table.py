"""Evidence-tiered "real bugs found" table generated from the frozen corpus."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

from . import bug_regression_corpus as regressions

SCHEMA_VERSION = "real-bugs-table/v1"

_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = _ROOT / "experiments" / "real_bugs"
RESULTS_PATH = EXPERIMENT_DIR / "real_bugs_found.json"
DOC_PATH = _ROOT / "docs" / "real_bugs_found.md"


def _canonical_bytes(obj: object) -> bytes:
    return json.dumps(
        obj, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _entries(doc: Mapping[str, object]) -> Sequence[Mapping[str, object]]:
    raw = doc.get("regressions")
    if not isinstance(raw, list):
        raise AssertionError("regressions must be a list")
    return raw


def _row(entry: Mapping[str, object]) -> Dict[str, object]:
    command = list(entry.get("reproduction_command", []))
    return {
        "table_id": f"REAL-BUG-{entry['finding_id']}",
        "finding_id": entry["finding_id"],
        "regression_id": entry["regression_id"],
        "candidate_repo": entry["candidate_repo"],
        "candidate_url": entry["candidate_url"],
        "source_family": entry["source_family"],
        "pair": f"C -> {entry['target_lang']}",
        "divergence_class": entry["divergence_class"],
        "evidence_tier": entry["evidence_tier"],
        "witness_input": list(entry["witness_input"]),
        "safe_input": list(entry["safe_input"]),
        "c_file": entry["c_file"],
        "rust_file": entry["rust_file"],
        "c_sha256": entry["c_sha256"],
        "rust_sha256": entry["rust_sha256"],
        "live_reproduction_link": entry["bundle_path"],
        "live_reproduction_command": command,
        "upstream_defect_claim": False,
        "upstream_status": entry["upstream_status"],
        "remediation": entry["remediation"],
    }


def table_rows(
    regression_doc: Optional[Mapping[str, object]] = None,
) -> Tuple[Dict[str, object], ...]:
    regression_doc = regressions.load_results() if regression_doc is None else regression_doc
    return tuple(_row(entry) for entry in _entries(regression_doc))


def content_hash(rows: Optional[Sequence[Mapping[str, object]]] = None) -> str:
    rows = table_rows() if rows is None else rows
    stable = [
        {
            "finding_id": row["finding_id"],
            "regression_id": row["regression_id"],
            "divergence_class": row["divergence_class"],
            "pair": row["pair"],
            "evidence_tier": row["evidence_tier"],
            "witness_input": row["witness_input"],
            "safe_input": row["safe_input"],
            "live_reproduction_link": row["live_reproduction_link"],
            "upstream_defect_claim": row["upstream_defect_claim"],
        }
        for row in rows
    ]
    return hashlib.sha256(_canonical_bytes(stable)).hexdigest()


def results_document(
    regression_doc: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    regression_doc = regressions.load_results() if regression_doc is None else regression_doc
    rows = table_rows(regression_doc)
    by_class: Dict[str, int] = {}
    by_evidence_tier: Dict[str, int] = {}
    for row in rows:
        cls = str(row["divergence_class"])
        tier = str(row["evidence_tier"])
        by_class[cls] = by_class.get(cls, 0) + 1
        by_evidence_tier[tier] = by_evidence_tier.get(tier, 0) + 1
    return {
        "schema": SCHEMA_VERSION,
        "content_hash": content_hash(rows),
        "source_manifest": str(regressions.RESULTS_PATH.relative_to(_ROOT)),
        "source_manifest_hash": regression_doc["content_hash"],
        "n_rows": len(rows),
        "n_upstream_defect_claims": sum(
            1 for row in rows if bool(row["upstream_defect_claim"])
        ),
        "by_divergence_class": by_class,
        "by_evidence_tier": by_evidence_tier,
        "evidence_policy": (
            "Rows are verified extraction-unit divergences from the frozen "
            "bug-regression corpus. They are reviewer-ready real-bug evidence "
            "with live reproduction links, but not upstream CVEs or repository "
            "defect claims unless upstream_status says so."
        ),
        "rows": list(rows),
    }


def load_results(path: Path = RESULTS_PATH) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_results(path: Path = RESULTS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(results_document(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def check_results(path: Path = RESULTS_PATH) -> Tuple[bool, str]:
    if not path.exists():
        return False, f"{path} is missing"
    ok, detail = regressions.check_results()
    if not ok:
        return False, f"source bug-regression corpus invalid: {detail}"
    expected = json.dumps(results_document(), indent=2, sort_keys=True) + "\n"
    if path.read_text(encoding="utf-8") != expected:
        return False, f"{path} does not match regenerated real-bugs table"
    doc = json.loads(expected)
    if doc.get("schema") != SCHEMA_VERSION:
        return False, f"unexpected schema {doc.get('schema')!r}"
    if doc.get("n_upstream_defect_claims") != 0:
        return False, "table must not promote extraction-unit evidence to upstream claims"
    corpus_ids = {str(entry["finding_id"]) for entry in _entries(regressions.load_results())}
    row_ids = {str(row["finding_id"]) for row in doc.get("rows", [])}
    if row_ids != corpus_ids:
        return False, f"row id drift: corpus={sorted(corpus_ids)} rows={sorted(row_ids)}"
    return True, "OK"


def markdown_document(doc: Optional[Mapping[str, object]] = None) -> str:
    doc = results_document() if doc is None else doc
    lines = [
        "# Real bugs found — evidence-tiered table",
        "",
        "This page is generated by `ub_oracle.real_bugs_table` from the frozen "
        "bug-regression corpus. It is intentionally evidence-tiered: every row "
        "has a live reproduction command, but extraction-unit findings are not "
        "promoted to upstream CVEs or repository-defect claims without source audit.",
        "",
        f"- **Rows:** {doc['n_rows']}",
        f"- **Upstream defect claims:** {doc['n_upstream_defect_claims']}",
        f"- **Source manifest:** `{doc['source_manifest']}` "
        f"(`{str(doc['source_manifest_hash'])[:16]}`)",
        f"- **Content hash:** `{doc['content_hash']}`",
        "",
        "| Finding | Pair / class | Evidence tier | Live reproduction | Upstream status |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in doc["rows"]:
        command = " ".join(str(part) for part in row["live_reproduction_command"])
        lines.append(
            f"| `{row['finding_id']}` | {row['pair']} / `{row['divergence_class']}` | "
            f"{row['evidence_tier']} | `{command}` | {row['upstream_status']} |"
        )
    lines += [
        "",
        "## Reproduction contract",
        "",
        "Each row links to a checked shell bundle under `docs/repro/`. The focused "
        "gate recomputes the frozen source hashes, checks the bundle syntax, and, "
        "when `clang`/UBSan plus the target compiler are available, proves the "
        "witness still traps on the C side while the target remains defined and "
        "the safe control stays silent.",
        "",
        "Run `make real-bugs-table-check` to regenerate and validate this table.",
        "",
    ]
    return "\n".join(lines)


def write_docs(path: Path = DOC_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown_document(load_results()), encoding="utf-8")


def check_docs(path: Path = DOC_PATH) -> Tuple[bool, str]:
    if not path.exists():
        return False, f"{path} is missing"
    expected = markdown_document(load_results())
    if path.read_text(encoding="utf-8") != expected:
        return False, f"{path} does not match regenerated real-bugs table docs"
    return True, "OK"


def write_artifacts() -> Tuple[Path, Path]:
    write_results()
    write_docs()
    return RESULTS_PATH, DOC_PATH


if __name__ == "__main__":  # pragma: no cover
    results, docs = write_artifacts()
    ok, detail = check_results(results)
    doc_ok, doc_detail = check_docs(docs)
    print(f"real-bugs-table manifest={ok} docs={doc_ok} rows={results_document()['n_rows']}")
    if not ok:
        print(detail)
    if not doc_ok:
        print(doc_detail)
    print(f"wrote {results} and {docs}")
