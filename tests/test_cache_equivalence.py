"""Focused tests for Step 142's cache equivalence proof."""

from __future__ import annotations

from collections import Counter

import pytest

from src.ub_oracle import cache as cache_mod
from src.ub_oracle.cache import (
    cache_key,
    prove_cache_equivalence,
    report_signature_hash,
    unit_content_hash,
)
from src.ub_oracle.plugin import OracleResult, OracleVerdict
from src.ub_oracle.reexec import ToolchainStatus, toolchain_available
from src.ub_oracle.replay import Counterexample
from src.ub_oracle.verify import VerifyReport, VerifyVerdict


def _status() -> ToolchainStatus:
    return ToolchainStatus(cc=None, ubsan=False, targets=())


def _fp(**kw):
    base = {"cc": "clang X", "ubsan": "yes", "rust": "rustc Y"}
    base.update(kw)
    return base


def _counterexample(name: str) -> Counterexample:
    ce = Counterexample(
        divergence_class="signed_overflow",
        source_lang="c",
        target_lang="rust",
        inputs={"x": 2_147_483_647},
        source_snippet="int f(int x){return x+1;}",
        target_snippet="fn f(x:i32)->i32{x.wrapping_add(1)}",
        divergence_witness=f"{name}: C signed overflow, Rust wraps",
        definedness_witness="x is a valid i32",
        source_observed={"O0": "-2147483648", "O2": "2147483647"},
        target_observed="-2147483648",
        confirmed=True,
    )
    ce.attach_proof_certificate({
        "ub_reached": True,
        "target_defined": True,
        "consequence": True,
    })
    return ce


def _report(unit):
    name = unit["name"]
    if name.startswith("div"):
        res = OracleResult(
            OracleVerdict.DIVERGENT,
            "signed_overflow",
            counterexample=_counterexample(name),
        )
        return VerifyReport(
            VerifyVerdict.DIVERGENT,
            unit,
            oracle_results=[res],
            divergence=res,
            detail="confirmed by fake ground truth",
        )
    if name.startswith("candidate"):
        return VerifyReport(
            VerifyVerdict.CANDIDATE,
            unit,
            detail="symbolic witness intentionally not cacheable",
        )
    return VerifyReport(
        VerifyVerdict.NO_DIVERGENCE_FOUND,
        unit,
        detail="covered classes checked",
    )


def test_unit_content_hash_is_order_independent_and_toolchain_bound():
    a = {"name": "same", "kind": "binop_const", "op": "add", "const": 1}
    b = {"const": 1, "op": "add", "kind": "binop_const", "name": "same"}
    assert unit_content_hash(a) == unit_content_hash(b)
    assert cache_key(a, _fp()) == cache_key(b, _fp())
    assert cache_key(a, _fp(cc="clang newer")) != cache_key(a, _fp())


def test_cache_equivalence_proof_replays_cold_verdict_layer(monkeypatch):
    calls = []

    def fake_verify(unit, **kwargs):
        calls.append(unit["name"])
        return _report(unit)

    monkeypatch.setattr(cache_mod, "verify_unit", fake_verify)
    units = [{"name": "div-overflow"}, {"name": "safe-add"}]

    proof = prove_cache_equivalence(
        units, fingerprint=_fp(), status=_status(), confirm=True)

    assert proof.ok, proof.mismatches
    assert proof.total == 2
    assert proof.cacheable == 2
    assert proof.cold_hits == 0 and proof.cold_misses == 2
    assert proof.warm_hits == 2 and proof.warm_misses == 0
    assert proof.cold_signature_hash == proof.warm_signature_hash
    assert Counter(calls) == Counter({"div-overflow": 1, "safe-add": 1})


def test_cache_equivalence_allows_noncacheable_warm_misses(monkeypatch):
    calls = []

    def fake_verify(unit, **kwargs):
        calls.append(unit["name"])
        return _report(unit)

    monkeypatch.setattr(cache_mod, "verify_unit", fake_verify)
    units = [{"name": "div-overflow"}, {"name": "candidate-oob"}]

    proof = prove_cache_equivalence(
        units, fingerprint=_fp(), status=_status(), confirm=True)

    assert proof.ok, proof.mismatches
    assert proof.cacheable == 1
    assert proof.warm_hits == 1 and proof.warm_misses == 1
    assert Counter(calls) == Counter({"div-overflow": 1, "candidate-oob": 2})


_TC = toolchain_available()


@pytest.mark.skipif(
    not _TC.full,
    reason=f"needs C+UBSan+rustc toolchain ({_TC})",
)
def test_cache_equivalence_proof_against_real_compilers():
    units = [
        {"name": "add1_w32", "kind": "binop_const", "op": "add", "const": 1,
         "width": 32, "var": "x", "signed": True, "probe": "signed_overflow",
         "source_lang": "c", "target_lang": "rust"},
        {"name": "noovf", "kind": "binop_const", "op": "add", "const": 0,
         "width": 32, "var": "x", "signed": True, "probe": "signed_overflow",
         "source_lang": "c", "target_lang": "rust"},
    ]

    proof = prove_cache_equivalence(units)

    assert proof.ok, proof.to_dict()
    assert proof.cacheable == 2
    assert proof.warm_hits == 2


def test_report_signature_hash_roundtrips_counterexample_without_cached_suffix():
    unit = {"name": "div-overflow"}
    cold = _report(unit)
    warm = VerifyReport(
        cold.verdict,
        unit,
        divergence=cold.divergence,
        detail=cold.detail + "  [cached]",
    )
    assert report_signature_hash([unit], [cold]) == report_signature_hash([unit], [warm])
