from __future__ import annotations

import pytest

from src.ub_oracle import llm_scale_study as llm
from src.ub_oracle.reexec import toolchain_available
from src.ub_oracle.statistical_rigor import wilson_interval


def test_llm_scale_corpus_has_200_plus_frozen_translations():
    items = llm.generate_corpus()
    census = llm.corpus_census(items)

    assert len(items) == census["n_items"] >= llm.MIN_TRANSLATED_FUNCTIONS
    assert census["n_distinct_programs"] == len(items)
    assert census["n_source_libraries"] == 12
    assert census["translator"]["translator_kind"] == "llm"
    assert census["translator"]["model_id"] == "gpt-5.5"
    assert census["by_label"]["divergent"] > census["by_label"]["equivalent"] > 0
    assert {"signed_overflow", "div_by_zero", "shift_oob", "array_oob"} <= set(
        census["by_class"]
    )


def test_llm_scale_items_are_ground_truth_compatible():
    for item in llm.generate_corpus()[:20]:
        gt = item.to_gt_item()
        assert gt.lang == "rust"
        assert gt.item_id == item.item_id
        assert gt.c_src.startswith("#include <stdio.h>")
        assert "fn main()" in gt.target_src
        assert gt.declared_label in {"divergent", "equivalent"}
        assert item.prompt_hash
        assert item.content_hash


def test_llm_scale_corpus_hashes_are_stable_and_unique():
    first = llm.generate_corpus()
    second = llm.generate_corpus()

    assert [it.content_hash for it in first] == [it.content_hash for it in second]
    assert len({it.content_hash for it in first}) == len(first)
    assert llm.corpus_census(first) == llm.corpus_census(second)


def test_llm_scale_metric_intervals_use_wilson_scores():
    outcomes = [
        llm.LlmStudyOutcome("tp1", "lib", "signed_overflow", "divergent", "divergent", True, True, True),
        llm.LlmStudyOutcome("tp2", "lib", "shift_oob", "divergent", "divergent", True, True, True),
        llm.LlmStudyOutcome("fn", "lib", "array_oob", "divergent", "divergent", False, True, True),
        llm.LlmStudyOutcome("tn1", "lib", "safe_control", "equivalent", "equivalent", False, False, True),
        llm.LlmStudyOutcome("tn2", "lib", "safe_control", "equivalent", "equivalent", False, False, True),
    ]

    metrics = llm._metrics(outcomes)

    precision = metrics["precision_divergence"]
    recall = metrics["recall_divergence"]
    fpr = metrics["false_positive_rate"]
    assert (precision.successes, precision.trials) == (2, 2)
    assert (recall.successes, recall.trials) == (2, 3)
    assert (fpr.successes, fpr.trials) == (0, 2)
    assert (precision.ci_lo, precision.ci_hi) == wilson_interval(2, 2)
    assert (recall.ci_lo, recall.ci_hi) == wilson_interval(2, 3)
    assert (fpr.ci_lo, fpr.ci_hi) == wilson_interval(0, 2)


_TC = toolchain_available()


@pytest.mark.skipif(
    not _TC.full_for("rust"),
    reason=f"needs clang+UBSan+rustc toolchain ({_TC})",
)
def test_llm_scale_live_sample_confirms_against_real_compilers():
    report1 = llm.confirm_llm_scale_study(sample_size=8, seed=159)
    report2 = llm.confirm_llm_scale_study(sample_size=8, seed=159)

    assert report1.available
    assert report1.ok, report1.summary()
    assert report1.sample_size == 8
    assert report1.content_hash == report2.content_hash
    assert report1.metrics["precision_divergence"].successes > 0
    assert report1.metrics["recall_divergence"].successes > 0
    assert report1.metrics["false_positive_rate"].successes == 0
