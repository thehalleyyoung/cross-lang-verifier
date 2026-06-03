from __future__ import annotations

import itertools

import pytest

from src.ub_oracle.flaky_toolchain import detect_flaky_toolchain, run_evidence_once
from src.ub_oracle.ground_truth import GTItem, LabelEvidence
from src.ub_oracle.reexec import ToolchainStatus


def _status() -> ToolchainStatus:
    return ToolchainStatus(cc=None, ubsan=False, targets=())


def _item(item_id: str = "stable") -> GTItem:
    return GTItem(
        item_id=item_id,
        lang="rust",
        klass="safe_add",
        cwe="",
        declared_label="equivalent",
        c_src=f"/* {item_id} C */",
        target_src=f"// {item_id} Rust",
        inputs=(),
    )


def stable_label(_h, item: GTItem) -> LabelEvidence:
    return LabelEvidence(
        observed_label=item.declared_label,
        ub_trapped=False,
        c_out="7\n",
        target_out="7\n",
        detail="deterministic compiler evidence",
    )


_OUTPUT_COUNTER = itertools.count()


def evidence_flaky_label(_h, item: GTItem) -> LabelEvidence:
    n = next(_OUTPUT_COUNTER)
    return LabelEvidence(
        observed_label=item.declared_label,
        ub_trapped=False,
        c_out="7\n",
        target_out=f"7 #{n}\n",
        detail="same verdict, different target output",
    )


_VERDICT_COUNTER = itertools.count()


def verdict_flaky_label(_h, _item: GTItem) -> LabelEvidence:
    n = next(_VERDICT_COUNTER)
    observed = "equivalent" if n % 2 == 0 else "divergent"
    return LabelEvidence(
        observed_label=observed,
        ub_trapped=observed == "divergent",
        c_out="7\n",
        target_out="8\n" if observed == "divergent" else "7\n",
        detail=f"flipped verdict {n}",
    )


def test_stable_evidence_passes_across_reruns():
    report = detect_flaky_toolchain(
        [_item()],
        runs=3,
        workers=1,
        status=_status(),
        label_fn=stable_label,
    )

    assert report.stable
    assert report.verdict_stable
    assert report.evidence_stable
    assert report.quarantined == []
    assert len(set(report.run_evidence_hashes)) == 1


def test_same_verdict_different_output_is_quarantined_as_evidence_flaky():
    report = detect_flaky_toolchain(
        [_item("output-flaky")],
        runs=2,
        workers=1,
        status=_status(),
        label_fn=evidence_flaky_label,
    )

    assert not report.stable
    assert report.verdict_stable
    assert not report.evidence_stable
    assert len(report.quarantined) == 1
    assert report.quarantined[0].evidence_unstable
    assert not report.quarantined[0].verdict_unstable


def test_verdict_flip_is_quarantined_loudly():
    report = detect_flaky_toolchain(
        [_item("verdict-flaky")],
        runs=2,
        workers=1,
        status=_status(),
        label_fn=verdict_flaky_label,
    )

    assert not report.stable
    assert not report.verdict_stable
    assert len(report.quarantined) == 1
    assert report.quarantined[0].verdict_unstable


def test_duplicate_items_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        run_evidence_once(
            [_item("dup"), _item("dup")],
            status=_status(),
            label_fn=stable_label,
        )
