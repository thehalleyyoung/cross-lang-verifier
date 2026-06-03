from __future__ import annotations

import pytest

from src.ub_oracle import mechanized_soundness as ms
from src.ub_oracle.replay import (
    CERTIFICATE_ISSUER,
    CERTIFICATE_THEOREM,
    ProofCertificate,
    canonical_verdict_preimage,
)


def _payload(**overrides):
    payload = {
        "schema_version": 1,
        "verdict": "divergent",
        "observation": {
            "ub_reached": True,
            "target_defined": True,
            "consequence": False,
        },
        "kernel_theorem": CERTIFICATE_THEOREM,
        "checker_scope": "scope",
        "issuer": CERTIFICATE_ISSUER,
        "counterexample_hash": "sha256:abc",
    }
    payload.update(overrides)
    return payload


def test_verdict_canonical_preimage_is_length_prefixed_and_stable():
    assert canonical_verdict_preimage(_payload()) == (
        b"clv-verdict-v1;"
        b"s:14:schema_version;i:1;"
        b"s:7:verdict;s:9:divergent;"
        b"s:22:observation.ub_reached;b:1;"
        b"s:26:observation.target_defined;b:1;"
        b"s:23:observation.consequence;b:0;"
        b"s:14:kernel_theorem;s:12:oracle_sound;"
        b"s:13:checker_scope;s:5:scope;"
        b"s:6:issuer;s:19:cross-lang-verifier;"
        b"s:19:counterexample_hash;s:10:sha256:abc;"
    )


def test_verdict_preimage_distinguishes_boundary_ambiguous_strings():
    left = _payload(verdict="ab", kernel_theorem="c")
    right = _payload(verdict="a", kernel_theorem="bc")

    assert canonical_verdict_preimage(left) != canonical_verdict_preimage(right)


def test_verdict_preimage_rejects_bool_int_confusion():
    bool_as_int = _payload(schema_version=True)
    with pytest.raises(ValueError, match="JSON integer"):
        canonical_verdict_preimage(bool_as_int)

    int_as_bool = _payload(observation={
        "ub_reached": 1,
        "target_defined": True,
        "consequence": False,
    })
    with pytest.raises(ValueError, match="JSON booleans"):
        canonical_verdict_preimage(int_as_bool)


def test_proof_certificate_hash_uses_verdict_canonicalization():
    cert = ProofCertificate(
        verdict="divergent",
        ub_reached=True,
        target_defined=True,
        consequence=True,
        counterexample_hash="sha256:abc",
    )
    baseline = cert.recompute_hash()

    cert.consequence = False
    changed_observation = cert.recompute_hash()
    cert.consequence = True
    cert.counterexample_hash = "sha256:def"
    changed_binding = cert.recompute_hash()

    assert baseline != changed_observation
    assert baseline != changed_binding
    assert baseline.startswith("sha256:")


def test_hash_stability_declares_and_checks_required_theorems():
    report = ms.confirm_hash_stability()

    assert report.source_present
    assert report.lakefile_present
    assert report.source_hash and len(report.source_hash) == 64
    assert set(report.theorems_present) == set(ms.REQUIRED_HASH_STABILITY_THEOREMS)
    assert not report.theorems_missing, report.theorems_missing
    assert report.ok, ms.render_hash_stability(report)
    if report.available:
        assert report.kernel_accepted is True, report.stderr_tail
        assert report.fully_checked


def test_hash_stability_scope_does_not_overclaim_sha_collision_freedom():
    source = open(ms.HASH_STABILITY_SOURCE, encoding="utf-8").read()

    assert "does not pretend SHA-256 itself is collision-free" in source
    assert "verdict_canonicalization_injective" in source
