from __future__ import annotations

import pytest

from src.ub_oracle import scale_measure as sm
from src.ub_oracle.result_store import (
    ResultStoreError,
    build_result_store_doc,
    migrate_scale_measure_v1,
    prove_verdict_hash_stability,
    validate_result_store_doc,
)


def _verdict(label: str = "equivalent"):
    return {
        "item_id": "a",
        "lang": "rust",
        "klass": "safe_add",
        "declared_label": label,
        "observed_label": label,
        "decided": True,
        "abstained": False,
    }


def test_result_store_hashes_ignore_measurement_metadata():
    left = build_result_store_doc(
        artifact_kind="unit-test",
        producer="tests",
        verdicts=[_verdict()],
        measurements={"wall_ms": 1.0},
        metadata={"note": "first"},
    )
    right = build_result_store_doc(
        artifact_kind="unit-test",
        producer="tests",
        verdicts=[_verdict()],
        measurements={"wall_ms": 999.0},
        metadata={"note": "second"},
    )

    validate_result_store_doc(left)
    validate_result_store_doc(right)
    lemma = prove_verdict_hash_stability(left, right)
    assert lemma.ok
    assert left["hashes"] == right["hashes"]


def test_store_hash_binds_toolchain_fingerprint_separately_from_verdict_hash():
    base = build_result_store_doc(
        artifact_kind="unit-test",
        producer="tests",
        verdicts=[_verdict()],
    )
    with_toolchain = build_result_store_doc(
        artifact_kind="unit-test",
        producer="tests",
        verdicts=[_verdict()],
        toolchain_provenance={
            "fingerprint": {
                "cc": "clang version test",
                "rust": "rustc version test",
            }
        },
    )

    assert base["hashes"]["verdict_hash"] == with_toolchain["hashes"]["verdict_hash"]
    assert base["hashes"]["store_hash"] != with_toolchain["hashes"]["store_hash"]


def test_migrate_scale_measure_v1_preserves_legacy_content_hash():
    report = sm.MeasurementReport(
        available=True,
        schema=sm.SCHEMA_VERSION,
        langs=("rust",),
        total_items=1,
        measured=[
            sm.ItemMeasurement(
                item_id="a",
                lang="rust",
                klass="safe_add",
                declared_label="equivalent",
                observed_label="equivalent",
                decided=True,
                abstained=False,
                wall_ms=12.0,
                peak_rss_kb=34,
            )
        ],
        total_wall_ms=12.0,
        peak_rss_kb=34,
    )
    v1 = sm.results_document(report)

    migrated = migrate_scale_measure_v1(v1)

    assert migrated["schema"] == "result-store/v2"
    assert migrated["hashes"]["verdict_hash"] == v1["content_hash"]
    assert migrated["metadata"]["legacy_content_hash"] == v1["content_hash"]
    validate_result_store_doc(migrated)


def test_validation_rejects_stale_hashes():
    doc = build_result_store_doc(
        artifact_kind="unit-test",
        producer="tests",
        verdicts=[_verdict()],
    )
    doc["verdicts"][0]["observed_label"] = "divergent"

    with pytest.raises(ResultStoreError, match="verdict hash mismatch"):
        validate_result_store_doc(doc)


def test_migration_rejects_stale_legacy_hash():
    report = sm.MeasurementReport(
        available=True,
        schema=sm.SCHEMA_VERSION,
        langs=("rust",),
        total_items=1,
        measured=[],
    )
    v1 = sm.results_document(report)
    v1["content_hash"] = "wrong"

    with pytest.raises(ResultStoreError, match="legacy content_hash"):
        migrate_scale_measure_v1(v1)
