from __future__ import annotations

import pytest

from src.ub_oracle import negative_corpus as nc
from src.ub_oracle.reexec import toolchain_available


def test_negative_corpus_has_1000_distinct_true_equivalence_ports():
    items = nc.generate_corpus()
    census = nc.corpus_census(items)

    assert len(items) == census["n_items"] == nc.MIN_NEGATIVE_ITEMS
    assert census["n_langs"] == 2
    assert census["n_distinct_ports"] == len(items)
    assert census["declared_label"] == "equivalent"
    assert census["by_target_lang"] == {"go": 500, "rust": 500}
    assert census["by_family"] == {
        "safe_add_const": 200,
        "safe_div": 200,
        "safe_rem": 200,
        "safe_shift": 200,
        "safe_sub_const": 200,
    }
    assert len({item.content_hash for item in items}) == len(items)
    assert {item.declared_label for item in items} == {"equivalent"}


def test_negative_corpus_proves_zero_false_positive_flags_on_full_corpus():
    report = nc.prove_zero_false_positives()

    assert report.ok
    assert report.total_items == nc.MIN_NEGATIVE_ITEMS
    assert report.n_false_positive_flags == 0
    assert report.n_not_covered == 0
    assert report.n_covered == report.total_items
    assert report.n_fully_pruned == report.total_items
    for outcome in report.outcomes:
        assert outcome.applicable_classes
        assert outcome.covered
        assert outcome.fully_pruned
        assert not outcome.false_positive_flag
        assert outcome.verdict == "no_divergence_found"


def test_negative_corpus_bounded_equivalence_and_manifest_are_fresh():
    bounded = nc.bounded_equivalence_results()
    assert len(bounded) == nc.MIN_NEGATIVE_ITEMS
    assert sum(result.checked_inputs for result in bounded) == 3000
    assert all(result.ok for result in bounded)

    ok, detail = nc.check_results()
    assert ok, detail
    doc = nc.load_results()
    assert doc == nc.results_document()
    assert doc["false_positive_bound"]["false_positive_flags"] == 0
    assert doc["false_positive_bound"]["not_covered_items"] == 0
    assert doc["false_positive_bound"]["fully_range_pruned_items"] == nc.MIN_NEGATIVE_ITEMS
    assert doc["bounded_equivalence"]["checked_inputs"] == 3000
    assert doc["bounded_equivalence"]["failures"] == []


_TC = toolchain_available()


@pytest.mark.skipif(
    not (_TC.full_for("rust") and _TC.full_for("go")),
    reason=f"needs clang+UBSan+rustc+go toolchains ({_TC})",
)
def test_negative_corpus_live_sample_is_compiler_verified_when_available():
    conf = nc.confirm_negative_corpus(sample_size=10, seed=164)

    assert conf.available
    assert conf.ok, conf.detail
    assert conf.false_positive_report.ok
    assert conf.corpus_size == nc.MIN_NEGATIVE_ITEMS
    assert conf.n_distinct_ports == conf.corpus_size
    assert len(conf.live_results) == 10
    assert {r.target_lang for r in conf.live_results} == {"rust", "go"}
    assert {r.family for r in conf.live_results} == set(nc.FAMILIES)
    for result in conf.live_results:
        assert result.observed_label == "equivalent", result
        assert result.agrees, result
