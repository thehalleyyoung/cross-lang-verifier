from __future__ import annotations

import math

import pytest

from src.ub_oracle import ReexecHarness, get_oracle, toolchain_available
from src.ub_oracle import oracles as _oracles  # noqa: F401  (register plugins)
from src.ub_oracle import smt_float
from src.ub_oracle.plugin import OracleVerdict

_TC = toolchain_available()


def test_smt_float_finds_bit_precise_fp_contraction_witness():
    witness = smt_float.solve_fp_contraction()

    assert witness is not None
    assert witness.fused_bits != witness.unfused_bits
    assert witness.fused != witness.unfused
    assert witness.operand_bits.keys() == {"a", "b", "c"}
    assert witness.result_bits.keys() == {"fused", "unfused"}
    assert math.fma(witness.a, witness.b, witness.c) != (
        witness.a * witness.b
    ) + witness.c


def test_smt_float_replays_witness_bits_through_z3_certificate():
    witness = smt_float.solve_fp_contraction()
    assert witness is not None

    proof = smt_float.prove_fp_contraction_witness(witness)

    assert proof.class_key == "fp_contraction"
    assert proof.width == 64
    assert proof.proved, proof.detail


def test_smt_float_oracle_uses_the_bit_precise_artifact():
    oracle = get_oracle("fp_contraction")
    result = oracle.find_divergence({"kind": "fp_fma", "probe": "fp_contraction"})

    assert result.verdict is OracleVerdict.DIVERGENT
    assert result.counterexample is not None
    assert set(result.counterexample.inputs) == {"a", "b", "c"}
    assert "fused=0x" in result.detail
    assert "unfused=0x" in result.detail


@pytest.mark.skipif(not _TC.full_for("rust"), reason="needs C+UBSan+rustc toolchain")
def test_smt_float_fp_contraction_witness_confirms_on_real_compilers():
    oracle = get_oracle("fp_contraction")
    result = oracle.confirm(
        oracle.find_divergence({"kind": "fp_fma", "probe": "fp_contraction"}),
        ReexecHarness(_TC),
    )
    reexec = result.reexec

    assert reexec is not None and reexec.available
    assert reexec.mode == "optimizer_exploited"
    assert reexec.c_runs["A"].stdout != reexec.c_runs["B"].stdout
    assert reexec.rust_defined
    assert reexec.confirmed and result.counterexample is not None
    assert result.counterexample.confirmed
