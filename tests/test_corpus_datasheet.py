from __future__ import annotations

import json

from src.ub_oracle import adversarial_corpus
from src.ub_oracle import bug_regression_corpus
from src.ub_oracle import c2rust_corpus
from src.ub_oracle import corpus_datasheet as datasheet
from src.ub_oracle import cve_corpus
from src.ub_oracle import divergence_findings
from src.ub_oracle import divergence_zoo
from src.ub_oracle import existing_tools_study
from src.ub_oracle import github_port_miner
from src.ub_oracle import idiomatic_corpus
from src.ub_oracle import large_scale_study
from src.ub_oracle import llm_scale_study
from src.ub_oracle import multipair_corpus
from src.ub_oracle import negative_corpus


EXPECTED_IDS = {
    "adversarial-near-misses",
    "bug-regressions",
    "c2rust-output",
    "divergence-zoo",
    "existing-tools-head-to-head",
    "github-mined-ports",
    "historical-cve-replays",
    "idiomatic-ports",
    "large-scale-generated",
    "llm-translations",
    "multi-pair-translations",
    "negative-true-equivalence",
    "responsible-findings",
}


def _record(doc, corpus_id):
    return next(record for record in doc["records"] if record["corpus_id"] == corpus_id)


def test_corpus_datasheet_covers_every_corpus_surface_with_live_counts():
    doc = datasheet.results_document()

    assert doc["schema"] == datasheet.SCHEMA_VERSION
    assert {record["corpus_id"] for record in doc["records"]} == EXPECTED_IDS
    assert doc["n_records"] == len(EXPECTED_IDS)
    assert doc["records"] == sorted(doc["records"], key=lambda record: record["corpus_id"])
    assert {"c->rust", "c->go", "c->swift"} <= set(doc["language_pairs"])

    assert _record(doc, "c2rust-output")["population_size"] == len(c2rust_corpus.CORPUS)
    assert _record(doc, "github-mined-ports")["population_size"] == len(github_port_miner.SAMPLES)
    assert _record(doc, "responsible-findings")["population_size"] == len(
        divergence_findings.finding_records()
    )
    assert _record(doc, "bug-regressions")["population_size"] == len(
        bug_regression_corpus.regression_entries()
    )
    assert _record(doc, "idiomatic-ports")["population_size"] == sum(
        len(item.targets) for item in idiomatic_corpus.CORPUS
    )
    assert _record(doc, "llm-translations")["population_size"] == len(
        llm_scale_study.generate_corpus()
    )
    assert _record(doc, "historical-cve-replays")["population_size"] == len(
        cve_corpus.historical_cve_cases()
    )
    assert _record(doc, "historical-cve-replays")["population_size"] >= 50
    assert _record(doc, "negative-true-equivalence")["population_size"] == len(
        negative_corpus.generate_corpus()
    )
    assert _record(doc, "adversarial-near-misses")["population_size"] == len(
        adversarial_corpus.corpus()
    )
    assert _record(doc, "multi-pair-translations")["population_size"] == sum(
        len(item.targets) for item in multipair_corpus.CORPUS
    )
    assert _record(doc, "divergence-zoo")["population_size"] == len(divergence_zoo.EXHIBITS)
    assert _record(doc, "existing-tools-head-to-head")["population_size"] == len(
        existing_tools_study.build_subjects()
    )

    large = _record(doc, "large-scale-generated")
    live_large_census = large_scale_study.corpus_census(large_scale_study.generate_corpus())
    assert large["population_size"] == live_large_census["n_items"]
    assert f"total_loc={live_large_census['total_loc']}" in large["notes"]
    assert live_large_census["total_loc"] >= large_scale_study.MIN_TOTAL_LOC


def test_corpus_datasheet_records_have_datasheet_fields_and_honest_scope():
    doc = datasheet.results_document()

    for record in doc["records"]:
        assert record["provenance"].strip(), record["corpus_id"]
        assert record["construction"].strip(), record["corpus_id"]
        assert record["real_code_evidence"].strip(), record["corpus_id"]
        assert record["validation_commands"], record["corpus_id"]
        assert record["limitations"], record["corpus_id"]
        assert record["content_hash"], record["corpus_id"]
        assert record["language_pairs"], record["corpus_id"]
        assert record["label_balance"], record["corpus_id"]

    existing = _record(doc, "existing-tools-head-to-head")
    assert existing["kind"] == "benchmark over c2rust-output corpus"
    assert "avoid double-counting" in existing["provenance"]

    cve = _record(doc, "historical-cve-replays")
    assert "not original vulnerable vendor source" in cve["limitations"][0]

    negative = _record(doc, "negative-true-equivalence")
    assert negative["label_balance"] == {"go": 500, "rust": 500}

    bug = _record(doc, "bug-regressions")
    findings = _record(doc, "responsible-findings")
    assert bug["population_size"] == findings["population_size"]


def test_corpus_datasheet_artifacts_are_byte_fresh():
    ok, detail = datasheet.check_all()
    assert ok, detail

    on_disk = json.loads(datasheet.RESULTS_PATH.read_text(encoding="utf-8"))
    regenerated = datasheet.results_document()
    assert on_disk == regenerated
    assert datasheet.DOC_PATH.read_text(encoding="utf-8") == datasheet.markdown_document()
    assert len(on_disk["content_hash"]) == 64
