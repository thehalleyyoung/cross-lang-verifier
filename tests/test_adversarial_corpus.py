from __future__ import annotations

import pytest

from src.ub_oracle import adversarial_corpus as ac
from src.ub_oracle.reexec import ReexecHarness, toolchain_available


def test_adversarial_corpus_shape_and_validation():
    cases = ac.corpus()
    valid, detail = ac.validate_cases(cases)
    census = ac.corpus_census(cases)

    assert valid, detail
    assert len(cases) == census["n_cases"] == 23
    assert len({case.case_id for case in cases}) == len(cases)
    assert census["by_policy"] == {
        ac.POLICY_ABSTAIN: 4,
        ac.POLICY_DIVERGENT: 10,
        ac.POLICY_SAFE: 9,
    }
    assert census["by_pair"]["c->rust"] >= 18
    assert census["by_pair"]["c->go"] == 3
    assert census["by_pair"]["c->c"] == 1
    assert census["by_pair"]["rust->go"] == 1
    assert {"signed_overflow", "division_by_zero", "intmin_div_neg1", "shift_oob"} <= {
        case.family for case in cases
    }


def test_adversarial_static_verdicts_and_abstentions_are_exact():
    report = ac.run_static_corpus()

    assert report.ok, [failure.to_dict() for failure in report.failures]
    assert report.verdict_counts() == {
        "candidate": 10,
        "no_divergence_found": 9,
        "not_covered": 4,
    }
    assert not report.breaches
    outcomes = {outcome.case_id: outcome for outcome in report.outcomes}
    assert outcomes["division-zero-window"].prepass_pruned == ("intmin_div_neg1",)
    assert outcomes["intmin-div-neg1-boundary"].prepass_pruned == ("div_by_zero",)
    assert outcomes["overflow-unit-wrong-probe-abstains"].applicable_classes == ()
    for outcome in report.outcomes:
        if outcome.policy == ac.POLICY_SAFE:
            assert outcome.prepass_pruned, outcome
            assert outcome.verdict == "no_divergence_found"
        if outcome.policy == ac.POLICY_ABSTAIN:
            assert outcome.verdict == "not_covered"
            assert outcome.applicable_classes == ()


def test_adversarial_manifest_is_byte_fresh():
    ok, detail = ac.check_results()

    assert ok, detail
    doc = ac.load_results()
    assert doc == ac.results_document()
    assert doc["static_verdicts"]["ok"]
    assert doc["static_verdicts"]["n_breaches"] == 0
    assert doc["census"]["n_cases"] == 23
    assert len(doc["content_hash"]) == 64


_TC = toolchain_available()


@pytest.mark.skipif(
    not (_TC.full_for("rust") or _TC.full_for("go")),
    reason=f"needs clang+UBSan plus a supported target compiler ({_TC})",
)
def test_adversarial_divergent_controls_stay_sound_under_real_compilers():
    report = ac.confirm_divergent_controls(status=_TC, harness=ReexecHarness(_TC))

    assert report.available
    assert report.ok, report.detail
    assert report.n_controls == 10
    assert report.n_confirmed > 0
    assert not report.breaches, [outcome.to_dict() for outcome in report.breaches]
    for outcome in report.outcomes:
        assert outcome.verdict not in {"no_divergence_found", "not_covered"}
