"""Focused tests for the Step 141 million-LOC migration-scale corpus."""

from __future__ import annotations

import pytest

from src.ub_oracle import large_scale_study as ls
from src.ub_oracle.ground_truth import label_item
from src.ub_oracle.reexec import ReexecHarness, toolchain_available


@pytest.fixture(scope="module")
def corpus():
    return ls.generate_corpus()


def test_million_loc_census_is_exact_distinct_and_balanced(corpus):
    census = ls.corpus_census(corpus)
    assert census["n_items"] == 60_000
    assert census["n_distinct_programs"] == 60_000
    assert census["total_loc"] == 1_044_000
    assert census["total_loc"] >= ls.MIN_TOTAL_LOC == 1_000_000
    assert census["by_pair"] == {"go": 28_000, "rust": 32_000}
    assert census["by_label"] == {"divergent": 29_600, "equivalent": 30_400}
    assert census["label_balance_delta"] == 800
    assert census["label_balance_delta"] / census["n_items"] < 0.02


def test_million_loc_census_preserves_class_breadth(corpus):
    census = ls.corpus_census(corpus)
    assert census["by_class"] == {
        "div_by_zero": 9_600,
        "oob_read": 9_600,
        "oversized_shift": 6_400,
        "safe_add": 9_600,
        "safe_mod": 9_600,
        "safe_mul": 4_800,
        "safe_shift": 6_400,
        "signed_overflow": 4_000,
    }


def test_signed_overflow_family_inputs_are_all_real_overflows():
    overflow_items = [
        item for item in ls.generate_corpus(("rust",))
        if item.klass == "signed_overflow"
    ]
    assert len(overflow_items) == 4_000
    for item in (overflow_items[0], overflow_items[len(overflow_items) // 2], overflow_items[-1]):
        a = int(item.item_id.rsplit("-", 1)[1])
        b = int(item.inputs[0])
        assert a <= 2_147_483_647
        assert a + b > 2_147_483_647


_TC = toolchain_available()


@pytest.mark.skipif(
    not _TC.full_for("rust"),
    reason=f"needs clang+UBSan+rustc toolchain ({_TC})",
)
def test_signed_overflow_family_confirms_against_real_compilers():
    item = next(
        item for item in ls.generate_corpus(("rust",))
        if item.klass == "signed_overflow"
    )
    evidence = label_item(ReexecHarness(_TC), item)
    assert evidence.observed_label == "divergent"
    assert evidence.ub_trapped is True
