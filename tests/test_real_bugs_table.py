from __future__ import annotations

import json

from src.ub_oracle import bug_regression_corpus as regressions
from src.ub_oracle import divergence_findings as findings
from src.ub_oracle import real_bugs_table as table


def test_real_bugs_table_is_generated_from_frozen_bug_corpus():
    table.write_artifacts()
    ok, detail = table.check_results()
    assert ok, detail
    ok, detail = table.check_docs()
    assert ok, detail

    doc = json.loads(table.RESULTS_PATH.read_text(encoding="utf-8"))
    corpus = regressions.load_results()
    assert doc == table.results_document()
    assert doc["source_manifest_hash"] == corpus["content_hash"]
    assert doc["n_rows"] == corpus["n_regressions"] == 1
    assert doc["n_upstream_defect_claims"] == 0
    assert {row["finding_id"] for row in doc["rows"]} == {
        entry["finding_id"] for entry in corpus["regressions"]
    }


def test_real_bugs_table_preserves_conservative_evidence_tier_and_live_links():
    doc = table.load_results()
    assert "not upstream CVEs" in doc["evidence_policy"]
    row = doc["rows"][0]
    rec = findings.finding_records()[0]
    assert row["finding_id"] == rec.finding_id
    assert row["evidence_tier"] == "confirmed extraction-unit finding"
    assert row["upstream_defect_claim"] is False
    assert "upstream-instance source audit pending" in row["upstream_status"]
    assert row["live_reproduction_link"] == f"docs/repro/{rec.finding_id}.sh"
    assert row["live_reproduction_command"] == ["bash", f"docs/repro/{rec.finding_id}.sh"]
    assert row["witness_input"] == list(rec.witness_input)
    assert row["safe_input"] == list(rec.safe_input)


def test_real_bugs_table_docs_are_reviewer_ready_but_not_overclaimed():
    text = table.DOC_PATH.read_text(encoding="utf-8")
    assert "Real bugs found" in text
    assert "evidence-tiered" in text
    assert "not promoted to upstream CVEs" in text
    assert "`make real-bugs-table-check`" in text
    assert findings.FINDING_ID in text
