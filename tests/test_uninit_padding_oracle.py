from __future__ import annotations

import pytest

from src.ub_oracle import CATALOGUE, Definedness, ReexecHarness, VerifyVerdict
from src.ub_oracle import divergence_zoo, plugin, toolchain_available, verify_unit
from src.ub_oracle.plugin import OracleVerdict
from src.ub_oracle import oracles as _oracles  # noqa: F401  (registers plugins)


_TC = toolchain_available()


def test_uninit_padding_catalogue_registry_and_zoo_index():
    rust = plugin.get_oracle_for("uninit_padding", "c", "rust")
    go = plugin.get_oracle_for("uninit_padding", "c", "go")
    assert rust.confirmation_mode == "uninit_padding"
    assert go.confirmation_mode == "uninit_padding"

    entry = CATALOGUE["uninit_padding"]
    assert entry.source_definedness is Definedness.UNSPECIFIED
    assert not entry.is_ub_rooted()
    assert "6.2.6.1" in entry.c_standard_ref

    idx = divergence_zoo.index_by_class_and_pair()
    assert "idio:uninit-padding:rust" in idx["uninit_padding"]["c->rust"]
    assert "idio:uninit-padding:go" in idx["uninit_padding"]["c->go"]


def test_uninit_padding_oracle_finds_compiler_asserted_padding_witness():
    orc = plugin.get_oracle_for("uninit_padding", "c", "rust")
    res = orc.find_divergence({"kind": "uninit_padding"})
    assert res.verdict is OracleVerdict.DIVERGENT
    ce = res.counterexample
    assert ce.inputs == {"tag": 7, "value": 0x01020304, "expose_padding": 1}
    assert ce.source_definedness == Definedness.UNSPECIFIED.value
    assert "_Static_assert(sizeof(struct P) == 8" in ce.source_snippet
    assert "_Static_assert(offsetof(struct P, value) == 4" in ce.source_snippet
    assert "if (!expose_padding) memset(&p, 0, sizeof p);" in ce.source_snippet
    assert "memcpy(bytes, &p, sizeof p)" in ce.source_snippet
    assert "let mut bytes = [0u8; 8]" in ce.target_snippet
    assert "zero-initialized byte buffer" in ce.divergence_witness


def test_uninit_padding_negative_controls_do_not_fabricate_padding():
    orc = plugin.get_oracle_for("uninit_padding", "c", "rust")
    no_padding = orc.find_divergence(
        {
            "kind": "uninit_padding",
            "fields": [
                {"name": "a", "type": "u32", "value": 1},
                {"name": "b", "type": "u32", "value": 2},
            ],
        }
    )
    assert no_padding.verdict is OracleVerdict.NO_DIVERGENCE_FOUND

    packed = orc.find_divergence({"kind": "uninit_padding", "packed": True})
    assert packed.verdict is OracleVerdict.NO_DIVERGENCE_FOUND
    assert orc.find_divergence({"kind": "div", "width": 32}).verdict is (
        OracleVerdict.NOT_APPLICABLE
    )


def test_verify_unit_reports_unconfirmed_padding_candidate():
    rep = verify_unit({"kind": "uninit_padding", "probe": "uninit_padding"},
                      confirm=False, status=_TC)
    assert rep.verdict is VerifyVerdict.CANDIDATE
    assert rep.oracle_results[-1].divergence_class == "uninit_padding"


@pytest.mark.skipif(
    not _TC.full_uninit_padding_for("rust"),
    reason=f"needs C+rustc plus MSan or clang auto-var-init ({_TC})",
)
def test_uninit_padding_rust_confirmed_against_real_compilers():
    orc = plugin.get_oracle_for("uninit_padding", "c", "rust")
    found = orc.find_divergence({"kind": "uninit_padding"})
    res = orc.confirm(found, ReexecHarness(_TC))
    rr = res.reexec
    assert rr.available, rr.reason
    assert rr.mode == "uninit_padding"
    assert rr.ub_reachable, rr.reason
    assert rr.rust_defined, rr.reason
    assert rr.confirmed and res.counterexample.confirmed
    if _TC.auto_var_init:
        assert rr.c_runs["pattern"].stdout != rr.c_runs["zero"].stdout
        assert rr.c_runs["zero"].stdout == rr.c_runs["zero_padding"].stdout
    if _TC.msan:
        assert rr.c_runs["msan"].msan_trapped
        assert not rr.c_runs["msan_clean"].msan_trapped

    safe_args = [str(v) for v in found.counterexample.inputs.values()]
    safe_args[-1] = "0"
    safe = ReexecHarness(_TC).confirm_uninit_padding_vs_defined(
        found.counterexample.source_snippet,
        found.counterexample.target_snippet,
        safe_args,
        "uninit_padding",
        "rust",
    )
    assert safe.available
    assert safe.rust_defined
    assert not safe.confirmed


@pytest.mark.skipif(
    not _TC.full_uninit_padding_for("go"),
    reason=f"needs C+go plus MSan or clang auto-var-init ({_TC})",
)
def test_uninit_padding_go_confirmed_against_real_compilers():
    orc = plugin.get_oracle_for("uninit_padding", "c", "go")
    res = orc.confirm(orc.find_divergence({"kind": "uninit_padding"}), ReexecHarness(_TC))
    rr = res.reexec
    assert rr.available, rr.reason
    assert rr.mode == "uninit_padding"
    assert rr.confirmed and res.counterexample.confirmed
