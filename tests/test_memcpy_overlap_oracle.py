from __future__ import annotations

import pytest

from src.ub_oracle import CATALOGUE, ReexecHarness, toolchain_available, verify_unit
from src.ub_oracle import VerifyVerdict
from src.ub_oracle import libc_model as libc
from src.ub_oracle import oracles as _oracles  # noqa: F401  (registers plugins)
from src.ub_oracle import plugin
from src.ub_oracle.plugin import OracleVerdict
from src.ub_oracle import divergence_zoo


_TC = toolchain_available()


def test_memcpy_overlap_catalogue_registry_and_zoo_index():
    rust = plugin.get_oracle_for("memcpy_overlap", "c", "rust")
    go = plugin.get_oracle_for("memcpy_overlap", "c", "go")
    assert rust.confirmation_mode == "libc_contract_trap_vs_defined"
    assert go.confirmation_mode == "libc_contract_trap_vs_defined"
    assert "memcpy_overlap" in CATALOGUE
    assert CATALOGUE["memcpy_overlap"].is_ub_rooted()

    idx = divergence_zoo.index_by_class_and_pair()
    assert "idio:memcpy-overlap:rust" in idx["memcpy_overlap"]["c->rust"]
    assert "idio:memcpy-overlap:go" in idx["memcpy_overlap"]["c->go"]


def test_memcpy_overlap_witness_is_runtime_sized_and_honors_safe_ranges():
    orc = plugin.get_oracle_for("memcpy_overlap", "c", "rust")
    res = orc.find_divergence({"kind": "memcpy_overlap", "buffer_len": 16})
    assert res.verdict is OracleVerdict.DIVERGENT
    ce = res.counterexample
    assert ce.inputs == {"dst": 1, "src": 0, "n": 4}
    assert "strtol(argv[3]" in ce.source_snippet
    assert "memcpy(buf + dst, buf + src, n)" in ce.source_snippet
    assert "copy_within" in ce.target_snippet

    safe = orc.find_divergence(
        {"kind": "memcpy_overlap", "buffer_len": 16, "dst": 8, "src": 0, "n": 4}
    )
    assert safe.verdict is OracleVerdict.NO_DIVERGENCE_FOUND
    assert orc.find_divergence({"kind": "div", "width": 32}).verdict is (
        OracleVerdict.NOT_APPLICABLE
    )


def test_libc_model_distinguishes_memcpy_overlap_from_memmove():
    assert libc.ranges_overlap(1, 0, 4)
    assert not libc.ranges_overlap(8, 0, 4)
    with pytest.raises(ValueError):
        libc.model_memcpy_into(b"ABCDEFGH", 1, 0, 4)
    assert libc.model_memmove(b"ABCDEFGH", 1, 0, 4) == b"AABCDFGH"
    assert libc.model_memcpy_into(b"ABCDEFGH", 4, 0, 4) == b"ABCDABCD"


def test_verify_unit_reports_candidate_without_reexecution():
    rep = verify_unit({"kind": "memcpy_overlap", "probe": "memcpy_overlap"},
                      confirm=False, status=_TC)
    assert rep.verdict is VerifyVerdict.CANDIDATE
    assert rep.divergence is None
    assert rep.oracle_results[-1].counterexample.inputs["n"] == 4


@pytest.mark.skipif(not _TC.full_libc_contract_for("rust"),
                    reason=f"needs C+rustc toolchain ({_TC})")
def test_memcpy_overlap_rust_confirmed_against_real_compilers():
    orc = plugin.get_oracle_for("memcpy_overlap", "c", "rust")
    res = orc.confirm(orc.find_divergence({"kind": "memcpy_overlap"}),
                      ReexecHarness(_TC))
    rr = res.reexec
    assert rr.available, rr.reason
    assert rr.mode == "libc_contract_trap_vs_defined"
    assert rr.ub_reachable
    assert "memcpy-param-overlap" in rr.c_runs["contract"].stderr
    assert rr.rust_defined
    assert rr.confirmed and res.counterexample.confirmed

    safe = ReexecHarness(_TC).confirm_libc_contract_trap_vs_defined(
        res.counterexample.source_snippet,
        res.counterexample.target_snippet,
        ["8", "0", "4"],
        "memcpy_overlap",
        "rust",
    )
    assert safe.available
    assert not safe.ub_reachable
    assert safe.rust_defined
    assert not safe.confirmed


@pytest.mark.skipif(not _TC.full_libc_contract_for("go"),
                    reason=f"needs C+go toolchain ({_TC})")
def test_memcpy_overlap_go_confirmed_against_real_compilers():
    orc = plugin.get_oracle_for("memcpy_overlap", "c", "go")
    res = orc.confirm(orc.find_divergence({"kind": "memcpy_overlap"}),
                      ReexecHarness(_TC))
    rr = res.reexec
    assert rr.available, rr.reason
    assert rr.mode == "libc_contract_trap_vs_defined"
    assert rr.ub_reachable
    assert rr.rust_defined
    assert rr.confirmed and res.counterexample.confirmed
