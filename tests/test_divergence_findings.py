from __future__ import annotations

import json
import subprocess

import pytest

from src.ub_oracle import divergence_findings as findings
from src.ub_oracle.reexec import toolchain_available


def test_finding_record_is_evidence_tiered_not_upstream_cve():
    records = findings.finding_records()
    assert len(records) == 1
    rec = records[0]
    assert rec.finding_id.startswith("CLV-DIV-")
    assert rec.candidate_repo == "uutils/coreutils"
    assert rec.evidence_tier == "confirmed extraction-unit finding"
    assert "upstream-instance source audit pending" in rec.upstream_status
    assert "CVE" in rec.upstream_status
    assert rec.maintenance_snapshot["archived"] is False
    assert rec.maintenance_snapshot["pushed_at"] >= "2026-06-03"
    assert rec.witness_input == ("2147479552",)
    assert rec.safe_input == ("0",)


def test_divergence_findings_results_json_is_byte_reproducible():
    findings.write_results()
    ok, detail = findings.check_results()
    assert ok, detail
    on_disk = json.loads(findings.RESULTS_PATH.read_text(encoding="utf-8"))
    regenerated = findings.results_document()
    assert on_disk == regenerated
    assert on_disk["content_hash"] == findings.content_hash()
    assert "not upstream CVEs" in on_disk["evidence_policy"]


def test_divergence_findings_docs_and_bundle_generate():
    doc, bundle, rep = findings.generate_findings()
    assert doc.exists()
    assert bundle.exists()
    assert rep.ok, rep.detail
    text = doc.read_text(encoding="utf-8")
    assert "Divergence findings" in text
    assert "Confirmed extraction-unit finding" in text
    assert "Upstream defect claim" in text
    assert "intentionally **not** made here" in text
    assert "`docs/repro/CLV-DIV-0001.sh`" in text
    assert findings.FINDING_ID in text
    syntax = subprocess.run(["bash", "-n", str(bundle)], capture_output=True, text=True)
    assert syntax.returncode == 0, syntax.stderr


_TC = toolchain_available()


@pytest.mark.skipif(
    not _TC.full_for("rust"),
    reason=f"needs clang+UBSan+rustc toolchain ({_TC})",
)
def test_divergence_finding_confirms_checked_in_sources_against_real_compilers():
    rep = findings.confirm_findings()
    assert rep.available and rep.ok, rep.detail
    assert rep.n_confirmed == rep.n_records == 1
    conf = rep.confirmations[0]
    assert conf.confirmed
    assert conf.ub_reachable
    assert conf.target_defined
    assert conf.safe_silent
    assert conf.bundle_path == "docs/repro/CLV-DIV-0001.sh"


@pytest.mark.skipif(
    not _TC.full_for("rust"),
    reason=f"needs clang+UBSan+rustc toolchain ({_TC})",
)
def test_divergence_finding_bundle_runs_end_to_end():
    records = findings.finding_records()
    conf = findings.reproduce_finding(records[0], write_bundle=True)
    assert conf.confirmed
    out = subprocess.run(
        ["bash", str(findings._ROOT / conf.bundle_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    combined = out.stdout + out.stderr
    assert out.returncode == 0, combined
    assert "signed integer overflow" in combined
    assert "== Safe input control ==" in combined
