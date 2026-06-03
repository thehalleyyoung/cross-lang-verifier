from __future__ import annotations

import pytest

from src.ub_oracle import ReexecHarness, toolchain_available
from src.ub_oracle import mechanized_soundness as ms
from src.ub_oracle import oracles as _oracles  # noqa: F401  (register plugins)
from src.ub_oracle.minimizer import (
    MinimizationResult,
    MinimizationStep,
    minimization_certificate,
    minimize_counterexample,
    verify_minimization_certificate,
)
from src.ub_oracle.plugin import ALL_ORACLES
from src.ub_oracle.regression_matrix import canonical_unit_for


_STATUS = toolchain_available()


def _anchor_oracle(class_key):
    return next(
        o for o in ALL_ORACLES
        if o.divergence_class == class_key
        and o.source_lang == "c"
        and o.target_lang == "rust"
    )


def test_mechanized_contract_requires_minimizer_soundness_theorems():
    report = ms.confirm_mechanized_soundness()
    required = {
        "minimizer_preserves_divergence",
        "minimizer_reduction_steps_preserve_divergence",
        "minimizer_certificate_sound",
    }

    assert required <= set(ms.REQUIRED_THEOREMS)
    assert required <= set(report.theorems_present)
    assert not report.theorems_missing, report.theorems_missing
    assert report.ok, report.stderr_tail


def test_minimization_certificate_checks_chain_category_and_simplification():
    result = MinimizationResult(
        divergence_class="signed_overflow",
        source_lang="c",
        target_lang="rust",
        original_inputs={"x": 1073741824},
        minimized_inputs={"x": 1},
        fields_reduced=["x"],
        probes=3,
        confirmed=True,
        certified_locally_minimal=True,
        ub_category="signed integer overflow",
        accepted_reductions=[
            MinimizationStep(
                field="x",
                before_inputs={"x": 1073741824},
                after_inputs={"x": 1},
                ub_category="signed integer overflow",
            ),
        ],
    )
    cert = minimization_certificate(result)

    assert verify_minimization_certificate(cert) == (True, "ok")

    cert.reductions[0].ub_category = "division by zero"
    ok, detail = verify_minimization_certificate(cert)
    assert not ok
    assert "UB category drifted" in detail


def test_minimization_certificate_rejects_unchained_or_unsimplifying_steps():
    result = MinimizationResult(
        divergence_class="signed_overflow",
        source_lang="c",
        target_lang="rust",
        original_inputs={"x": 10},
        minimized_inputs={"x": 11},
        confirmed=True,
        certified_locally_minimal=True,
        accepted_reductions=[
            MinimizationStep(
                field="x",
                before_inputs={"x": 10},
                after_inputs={"x": 11},
            ),
        ],
    )

    ok, detail = verify_minimization_certificate(minimization_certificate(result))
    assert not ok
    assert "does not simplify" in detail


@pytest.mark.skipif(
    not (
        _STATUS.c_available
        and _STATUS.target_available("rust")
        and _STATUS.target_runnable("rust")
    ),
    reason="needs C compiler + rustc toolchain",
)
def test_signed_overflow_minimizer_certificate_is_backed_by_real_compilers():
    oracle = _anchor_oracle("signed_overflow")
    result = oracle.find_divergence(canonical_unit_for(oracle))
    minimized = minimize_counterexample(
        oracle,
        result,
        ReexecHarness(_STATUS),
        max_probes=400,
    )
    cert = minimization_certificate(minimized)

    assert minimized.confirmed
    assert minimized.minimized_inputs == {"x": 1}
    assert minimized.accepted_reductions
    assert verify_minimization_certificate(cert) == (True, "ok")
