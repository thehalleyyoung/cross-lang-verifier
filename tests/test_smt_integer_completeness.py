from __future__ import annotations

from src.ub_oracle import smt_integer as smt


def test_smt_integer_decision_path_agrees_with_enumeration_on_10k_cases():
    report = smt.differential_test_smt_vs_enumeration(total_cases=10_000)

    assert report.total_cases == 10_000
    assert report.ok, report.mismatches
    assert {(g.class_key, g.width) for g in report.groups} == {
        (class_key, width)
        for class_key in smt.CLASSES
        for width in smt.SHIPPED_WIDTHS
    }
    assert all(g.cases > 0 for g in report.groups)

