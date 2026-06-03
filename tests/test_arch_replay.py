from __future__ import annotations

import pytest

from src.ub_oracle import arch_replay as ar
from src.ub_oracle.ground_truth import enumerate_corpus
from src.ub_oracle.reexec import toolchain_available


def _verdict(item_id: str, observed: str):
    return {
        "item_id": item_id,
        "lang": "rust",
        "klass": "safe_add",
        "declared_label": "equivalent",
        "observed_label": observed,
        "decided": observed in {"equivalent", "divergent"},
        "abstained": observed not in {"equivalent", "divergent"},
    }


def test_arch_normalization_is_total_for_common_names():
    assert ar.normalize_arch("AMD64") == "x86_64"
    assert ar.normalize_arch("x64") == "x86_64"
    assert ar.normalize_arch("aarch64") == "arm64"
    assert ar.normalize_arch("arm64") == "arm64"
    assert ar.normalize_arch("riscv64") == "riscv64"
    assert ar.normalize_arch("") == "unknown"


def test_synthetic_detector_marks_identical_arch_verdicts_as_stable():
    report = ar.synthetic_arch_report({
        "arm64": [_verdict("a", "equivalent")],
        "x86_64": [_verdict("a", "equivalent")],
    })

    assert report.synthetic
    assert report.ok
    assert not report.arch_dependency_detected
    assert len({r.content_hash for r in report.results}) == 1
    assert "synthetic" in report.summary()


def test_synthetic_detector_finds_arch_dependent_verdict_change():
    report = ar.synthetic_arch_report({
        "arm64": [_verdict("a", "equivalent")],
        "x86_64": [_verdict("a", "divergent")],
    })

    assert report.synthetic
    assert report.arch_dependency_detected
    assert report.arch_dependency_witnesses == ({
        "item_id": "a",
        "lang": "rust",
        "klass": "safe_add",
        "observed_by_arch": {
            "arm64": "equivalent",
            "x86_64": "divergent",
        },
    },)


def test_native_report_is_honest_when_no_toolchain_is_available():
    status = ar.ToolchainStatus(cc=None, ubsan=False, targets=())
    report = ar.confirm_cross_architecture_replay(
        requested_arches=(ar.normalize_arch(), "x86_64", "arm64"),
        status=status,
    )

    assert not report.synthetic
    assert report.ok
    assert not report.available
    assert report.unavailable_arches
    assert not report.arch_dependency_detected


_TC = toolchain_available()


@pytest.mark.skipif(
    not _TC.full_for("rust"),
    reason=f"needs clang+UBSan+rustc toolchain ({_TC})",
)
def test_native_arch_replay_executes_real_witness_and_control():
    rust_items = [item for item in enumerate_corpus(("rust",)) if item.lang == "rust"]
    sample = [
        next(item for item in rust_items if item.declared_label == "divergent"),
        next(item for item in rust_items if item.declared_label == "equivalent"),
    ]

    report = ar.confirm_cross_architecture_replay(
        items=sample,
        requested_arches=(ar.normalize_arch(),),
        status=_TC,
    )

    assert report.available
    assert report.ok
    assert not report.synthetic
    assert report.available_arches == (ar.normalize_arch(),)
    assert report.results[0].executed == 2
    assert {v["observed_label"] for v in report.results[0].verdicts} == {
        "divergent",
        "equivalent",
    }
