from __future__ import annotations

import pytest

from src.ub_oracle import CATALOGUE, Definedness, ReexecHarness, toolchain_available, verify_unit
from src.ub_oracle import VerifyVerdict
from src.ub_oracle import oracles as _oracles  # noqa: F401  (registers plugins)
from src.ub_oracle import plugin
from src.ub_oracle.oracles.atomic_ordering import (
    _seq_cst_outcomes,
    _source_relaxed_outcomes,
    store_buffering_gap,
)
from src.ub_oracle.plugin import OracleVerdict
from src.ub_oracle.reexec import RunOutcome, ToolchainStatus


_TC = toolchain_available()


def test_atomic_ordering_catalogue_and_registry():
    rust = plugin.get_oracle_for("atomic_ordering", "c", "rust")
    go = plugin.get_oracle_for("atomic_ordering", "c", "go")
    assert rust.confirmation_mode == "model_level_divergence"
    assert go.confirmation_mode == "model_level_divergence"

    entry = CATALOGUE["atomic_ordering"]
    assert entry.source_definedness is Definedness.DEFINED
    assert not entry.is_ub_rooted()
    assert "7.17" in entry.c_standard_ref


def test_store_buffering_enumerator_finds_allowed_set_gap():
    source = _source_relaxed_outcomes()
    target = _seq_cst_outcomes()
    assert (0, 0) in source
    assert (0, 0) not in target
    assert {(0, 1), (1, 0), (1, 1)} <= set(target)

    gap = store_buffering_gap()
    assert gap is not None
    assert gap.outcome == (0, 0)
    assert "initial y=0" in gap.source_trace
    assert "cycle" in gap.target_reason


def test_atomic_ordering_oracle_finds_rust_and_go_witnesses():
    for target in ("rust", "go"):
        orc = plugin.get_oracle_for("atomic_ordering", "c", target)
        res = orc.find_divergence(
            {
                "kind": "atomic_litmus",
                "pattern": "store_buffering",
                "source_order": "relaxed",
                "target_order": "seq_cst",
                "target_lang": target,
            }
        )
        assert res.verdict is OracleVerdict.DIVERGENT
        ce = res.counterexample
        assert ce.source_definedness == Definedness.DEFINED.value
        assert ce.inputs == {"r0": 0, "r1": 0}
        assert "memory_order_relaxed" in ce.source_snippet
        assert "target_seq_cst_all_zero" in ce.target_snippet
        assert "allowed-execution-set" in ce.definedness_witness


def test_atomic_ordering_negative_controls_do_not_claim_gap():
    orc = plugin.get_oracle_for("atomic_ordering", "c", "rust")
    no_target_gap = orc.find_divergence(
        {
            "kind": "atomic_litmus",
            "pattern": "store_buffering",
            "source_order": "relaxed",
            "target_order": "relaxed",
        }
    )
    assert no_target_gap.verdict is OracleVerdict.NO_DIVERGENCE_FOUND

    no_source_gap = orc.find_divergence(
        {
            "kind": "atomic_litmus",
            "pattern": "store_buffering",
            "source_order": "seq_cst",
            "target_order": "seq_cst",
        }
    )
    assert no_source_gap.verdict is OracleVerdict.NO_DIVERGENCE_FOUND


def test_verify_unit_reports_unconfirmed_atomic_candidate():
    rep = verify_unit(
        {
            "kind": "atomic_litmus",
            "pattern": "store_buffering",
            "source_order": "relaxed",
            "target_order": "seq_cst",
            "probe": "atomic_ordering",
        },
        confirm=False,
        status=_TC,
    )
    assert rep.verdict is VerifyVerdict.CANDIDATE
    assert rep.oracle_results[-1].divergence_class == "atomic_ordering"


def test_model_level_confirmation_rejects_panicking_target_checker():
    class PanicAfterTokenHarness(ReexecHarness):
        def __init__(self):
            super().__init__(
                ToolchainStatus(cc="fake-cc", ubsan=False, targets=(("rust", "fake-rustc"),))
            )

        def _compile_c(self, src, args, workdir, name):  # noqa: D401
            return "source-model"

        def _compile_target(self, src, target_lang, workdir, name):  # noqa: D401
            return "target-model"

        def _run(self, argv, env=None):  # noqa: D401
            if argv[0] == "source-model":
                return RunOutcome(0, "source_relaxed_all_zero=allowed", "")
            return RunOutcome(101, "target_seq_cst_all_zero=forbidden", "panic")

    rr = PanicAfterTokenHarness().confirm_model_level_divergence(
        "int main(void){return 0;}",
        "fn main() {}",
        [],
        target_lang="rust",
    )
    assert rr.available
    assert rr.ub_reachable
    assert not rr.rust_defined
    assert not rr.confirmed
    assert "target_forbidden=False" in rr.reason


@pytest.mark.skipif(
    not (_TC.c_available and _TC.target_available("rust")),
    reason=f"needs C+rustc toolchain ({_TC})",
)
def test_atomic_ordering_rust_confirmed_by_model_level_real_code():
    orc = plugin.get_oracle_for("atomic_ordering", "c", "rust")
    res = orc.confirm(
        orc.find_divergence({"kind": "atomic_litmus", "probe": "atomic_ordering"}),
        ReexecHarness(_TC),
    )
    rr = res.reexec
    assert rr.available, rr.reason
    assert rr.mode == "model_level_divergence"
    assert rr.ub_reachable
    assert rr.rust_defined
    assert rr.confirmed and res.counterexample.confirmed
    assert "source_relaxed_all_zero=allowed" in rr.c_runs["source_model"].stdout
    assert "target_seq_cst_all_zero=forbidden" in rr.rust_run.stdout


@pytest.mark.skipif(
    not (_TC.c_available and _TC.target_available("go")),
    reason=f"needs C+go toolchain ({_TC})",
)
def test_atomic_ordering_go_confirmed_by_model_level_real_code():
    orc = plugin.get_oracle_for("atomic_ordering", "c", "go")
    res = orc.confirm(
        orc.find_divergence(
            {"kind": "atomic_litmus", "probe": "atomic_ordering", "target_lang": "go"}
        ),
        ReexecHarness(_TC),
    )
    rr = res.reexec
    assert rr.available, rr.reason
    assert rr.mode == "model_level_divergence"
    assert rr.confirmed and res.counterexample.confirmed
