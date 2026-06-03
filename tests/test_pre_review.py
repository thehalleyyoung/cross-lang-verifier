from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from src.ub_oracle import pre_review as pr


def test_pre_review_packet_passes_and_is_hash_stable():
    first = pr.confirm_pre_review_packet()
    second = pr.confirm_pre_review_packet()

    assert first.ok, first.detail
    assert second.ok, second.detail
    assert first.packet_hash == second.packet_hash
    assert first.packet.soundness_registered == 60
    assert len(first.packet.reviewer_lanes) >= 4
    assert all(item.passed for item in first.packet.release_checklist)


def test_pre_review_packet_references_real_review_evidence():
    check = pr.confirm_pre_review_packet()
    packet = check.packet
    evidence = {item.id: item for item in packet.evidence}

    assert {"paper-source", "artifact-appendix", "traceability",
            "soundness-compendium", "reproduction-kit"} <= set(evidence)
    for lane in packet.reviewer_lanes:
        assert len(lane.questions) >= 3
        assert set(lane.evidence_ids) <= set(evidence)


def test_pre_review_negative_control_rejects_missing_evidence_path():
    packet = pr.build_pre_review_packet()
    bad_items = tuple(
        replace(item, path="docs/DOES_NOT_EXIST.md")
        if item.id == "paper-source" else item
        for item in packet.evidence
    )

    check = pr.validate_packet(replace(packet, evidence=bad_items))

    assert not check.ok
    assert any("missing evidence path for paper-source" in p for p in check.problems)


def test_pre_review_negative_control_rejects_unknown_lane_evidence():
    packet = pr.build_pre_review_packet()
    bad_lanes = (
        replace(packet.reviewer_lanes[0], evidence_ids=("unknown-evidence",) * 3),
        *packet.reviewer_lanes[1:],
    )

    check = pr.validate_packet(replace(packet, reviewer_lanes=bad_lanes))

    assert not check.ok
    assert any("references unknown evidence" in p for p in check.problems)


def test_pre_review_doc_is_deterministic_and_fresh(tmp_path: Path):
    packet = pr.build_pre_review_packet()
    rendered = pr.render_pre_review_markdown(packet)
    out = tmp_path / "PRE_REVIEW.md"

    pr.write_pre_review_doc(out)
    first = out.read_text()
    pr.write_pre_review_doc(out)
    second = out.read_text()

    assert first == second == rendered
    assert packet.content_hash in rendered
    assert "simulated" not in rendered.lower()

    tracked = Path("docs/PRE_REVIEW.md")
    assert tracked.read_text() == rendered
