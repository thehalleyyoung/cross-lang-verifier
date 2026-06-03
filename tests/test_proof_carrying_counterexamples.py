from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.ub_oracle import cache as _cache
from src.ub_oracle import cli as _cli
from src.ub_oracle import mechanized_soundness as ms
from src.ub_oracle.plugin import OracleResult, OracleVerdict
from src.ub_oracle.reexec import ReexecResult, RunOutcome
from src.ub_oracle.replay import Counterexample, verify_certificate
from src.ub_oracle.verify import VerifyReport, VerifyVerdict
from src.ub_oracle import get_oracle


def _certified_counterexample() -> Counterexample:
    ce = Counterexample(
        divergence_class="signed_overflow",
        source_lang="c",
        target_lang="rust",
        inputs={"x": 2147483647},
        source_snippet="int main(void){return 0;}\n",
        target_snippet="fn main() {}\n",
        source_definedness="undefined",
        divergence_witness="C signed overflow is reachable",
        definedness_witness="x is a valid i32 input",
        source_observed={"O0": "0", "O2": "1", "san": ""},
        target_observed="0",
        confirmed=True,
    )
    ce.attach_proof_certificate({
        "ub_reached": True,
        "target_defined": True,
        "consequence": True,
    })
    return ce


def _report_with(ce: Counterexample) -> VerifyReport:
    return VerifyReport(
        VerifyVerdict.DIVERGENT,
        {"name": "overflow", "source_lang": "c", "target_lang": "rust"},
        divergence=OracleResult(
            OracleVerdict.DIVERGENT,
            ce.divergence_class,
            counterexample=ce,
        ),
        detail="confirmed by fake harness",
    )


def test_proof_certificate_roundtrips_and_binds_to_counterexample():
    ce = _certified_counterexample()

    back = Counterexample.from_json(ce.to_json())
    cert = verify_certificate(back)

    assert cert.observation == {
        "ub_reached": True,
        "target_defined": True,
        "consequence": True,
    }
    assert back.to_dict() == ce.to_dict()


def test_proof_certificate_rejects_rebinding_to_changed_payload():
    ce = Counterexample.from_json(_certified_counterexample().to_json())
    ce.inputs["x"] = 0

    with pytest.raises(ValueError, match="not bound"):
        verify_certificate(ce)


def test_proof_certificate_rejects_broken_hash_and_nonviolating_observation():
    ce = Counterexample.from_json(_certified_counterexample().to_json())
    assert ce.proof_certificate is not None
    ce.proof_certificate.certificate_hash = "sha256:broken"
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_certificate(ce)

    ce = Counterexample.from_json(_certified_counterexample().to_json())
    assert ce.proof_certificate is not None
    ce.proof_certificate.consequence = False
    ce.proof_certificate.certificate_hash = ce.proof_certificate.recompute_hash()
    with pytest.raises(ValueError, match="does not violate"):
        verify_certificate(ce)


def test_proof_certificate_requires_real_json_booleans():
    payload = _certified_counterexample().to_dict()
    payload["proof_certificate"]["observation"]["ub_reached"] = "true"

    with pytest.raises(ValueError, match="JSON boolean"):
        Counterexample.from_dict(payload)


def test_oracle_confirmation_mints_certificate_from_raw_reexec_facts():
    class FakeHarness:
        def confirm_ub_divergence(self, *args, **kwargs):
            return ReexecResult(
                available=True,
                divergence_class="signed_overflow",
                inputs={"arg0": "2147483647"},
                c_runs={
                    "O0": RunOutcome(0, "0", ""),
                    "O2": RunOutcome(0, "1", ""),
                    "san": RunOutcome(1, "", "runtime error: signed overflow"),
                },
                rust_run=RunOutcome(0, "0", ""),
                ub_reachable=True,
                ub_consequential=True,
                rust_defined=True,
                confirmed=True,
                reason="fake confirmed",
            )

    oracle = get_oracle("signed_overflow")
    result = oracle.find_divergence({
        "kind": "binop_const",
        "op": "add",
        "const": 1,
        "width": 32,
        "var": "x",
        "signed": True,
    })

    confirmed = oracle.confirm(result, FakeHarness())

    assert confirmed.counterexample is not None
    assert confirmed.counterexample.proof_certificate is not None
    verify_certificate(confirmed.counterexample)


def test_symbolic_candidate_has_no_certificate_before_confirmation():
    result = get_oracle("signed_overflow").find_divergence({
        "kind": "binop_const",
        "op": "add",
        "const": 1,
        "width": 32,
        "var": "x",
        "signed": True,
    })

    assert result.counterexample is not None
    assert result.counterexample.proof_certificate is None


def test_cli_verified_check_consumes_certificate(monkeypatch):
    calls = []

    monkeypatch.setattr(
        ms,
        "build_verified_checker",
        lambda: SimpleNamespace(ok=True, source_hash="checker-hash"),
    )

    def fake_run(verdict, ub_reached, target_defined, consequence, *, build, timeout=300):
        calls.append((verdict, ub_reached, target_defined, consequence, build))
        return SimpleNamespace(ok=True, exit_code=0, stdout='{"accepted":true}', stderr="")

    monkeypatch.setattr(ms, "run_verified_checker", fake_run)

    summary, err = _cli._run_verified_checks([_report_with(_certified_counterexample())])

    assert err is None
    assert summary["status"] == "accepted"
    assert summary["checked"] == 1
    assert calls == [("divergent", True, True, True, False)]


def test_cli_verified_check_rejects_missing_certificate_before_lean(monkeypatch):
    monkeypatch.setattr(
        ms,
        "build_verified_checker",
        lambda: pytest.fail("Lean checker should not build for an invalid certificate"),
    )
    ce = _certified_counterexample()
    ce.proof_certificate = None

    summary, err = _cli._run_verified_checks([_report_with(ce)])

    assert summary is None
    assert err is not None
    assert "invalid proof certificate" in err


def test_cache_hit_rehydrates_certificate_bearing_counterexample(monkeypatch):
    unit = {"name": "overflow", "source_lang": "c", "target_lang": "rust"}
    report = _report_with(_certified_counterexample())
    cache = _cache.VerificationCache(fingerprint={"cc": "clang", "ubsan": "yes", "rust": "rustc"})

    def fake_verify(unit_arg, **kwargs):
        assert unit_arg == unit
        return report

    monkeypatch.setattr(_cache, "verify_unit", fake_verify)

    cold = _cache.verify_incremental([unit], cache)
    warm = _cache.verify_incremental([unit], cache)

    assert cold.misses == 1 and cold.stored == 1
    assert warm.hits == 1 and warm.misses == 0
    rehydrated = warm.reports[0].divergence.counterexample
    assert rehydrated is not None
    verify_certificate(rehydrated)
