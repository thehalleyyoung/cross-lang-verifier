from __future__ import annotations

import pytest

from src.ub_oracle import ground_truth as gt
from src.ub_oracle import leaderboard as lb
from src.ub_oracle.reexec import ReexecHarness, toolchain_available


def test_leaderboard_split_is_balanced_public_and_label_hidden():
    cases = lb.build_cases()
    public = lb.public_cases_document(cases)
    key = lb.answer_key_document(cases)

    assert len(cases) == public["n_cases"] == key["n_cases"] == 120
    assert key["by_label"] == {"divergent": 60, "equivalent": 60}
    assert key["by_pair"] == {"c->go": 60, "c->rust": 60}
    assert len({case.case_id for case in cases}) == len(cases)
    assert len({case.content_hash for case in cases}) == len(cases)
    assert public["content_hash"] == key["public_cases_hash"]

    skipped = set(lb._calibration_ids(gt.enumerate_corpus(lb.TARGET_LANGS)))
    assert not skipped.intersection(
        str(answer["source_item_id"]) for answer in key["answers"]
    )

    for entry in public["cases"]:
        assert "label" not in entry
        assert "source_item_id" not in entry
        assert "family" not in entry
        assert entry["source"].startswith("#include")
        assert entry["target"]
        assert entry["inputs"]


def test_leaderboard_artifacts_are_byte_fresh():
    ok, detail = lb.check_all()

    assert ok, detail
    assert lb.load_public_cases() == lb.public_cases_document()
    assert lb.load_answer_key() == lb.answer_key_document()


def test_leaderboard_scores_perfect_abstaining_and_partial_submissions():
    key = lb.answer_key_document()

    perfect = lb.score_submission(lb.perfect_submission_document(), key)
    assert perfect.valid
    assert perfect.total_cases == 120
    assert perfect.correct_cases == perfect.total_cases
    assert perfect.coverage == 1.0
    assert perfect.macro_f1 == 1.0
    assert perfect.primary_score == 100.0

    abstain = lb.score_submission(lb.sample_submission_document(), key)
    assert abstain.valid
    assert abstain.answered_cases == 0
    assert abstain.correct_cases == 0
    assert abstain.coverage == 0.0
    assert abstain.primary_score == 0.0

    first_answer = key["answers"][0]
    partial = {
        "schema": lb.SUBMISSION_SCHEMA_VERSION,
        "benchmark_id": lb.BENCHMARK_ID,
        "submission_id": "one-correct-rest-missing",
        "predictions": [
            {
                "case_id": first_answer["case_id"],
                "prediction": first_answer["label"],
            }
        ],
    }
    partial_score = lb.score_submission(partial, key)
    assert partial_score.valid
    assert partial_score.answered_cases == 1
    assert partial_score.correct_cases == 1
    assert len(partial_score.missing_case_ids) == 119
    assert 0.0 < partial_score.primary_score < 100.0


def test_leaderboard_rejects_bad_submission_schema_without_metric_crashes():
    key = lb.answer_key_document()
    case_id = str(key["answers"][0]["case_id"])
    bad = {
        "submission_id": "bad",
        "predictions": [
            {"case_id": case_id, "prediction": "divergent"},
            {"case_id": case_id, "prediction": "equivalent"},
            {"case_id": "not-a-case", "prediction": "divergent"},
            {"case_id": str(key["answers"][1]["case_id"]), "prediction": "maybe"},
        ],
    }

    score = lb.score_submission(bad, key)

    assert not score.valid
    assert any("duplicate" in err for err in score.errors)
    assert any("unknown case_id" in err for err in score.errors)
    assert any("invalid prediction" in err for err in score.errors)
    assert score.total_cases == 120
    assert 0.0 <= score.primary_score <= 100.0


_TC = toolchain_available()


@pytest.mark.skipif(
    not any(_TC.full_for(lang) for lang in lb.TARGET_LANGS),
    reason=f"needs clang+UBSan plus at least one target compiler ({_TC})",
)
def test_leaderboard_sample_labels_match_real_compilers_when_available():
    report = lb.confirm_leaderboard_sample(
        sample_size=4,
        seed=168,
        status=_TC,
        harness=ReexecHarness(_TC),
    )

    assert report.available
    assert report.ok, report.to_dict()
    assert report.sample_size == 4
    assert report.n_confirmed == 4
    assert {result.expected_label for result in report.results} <= {
        "divergent",
        "equivalent",
    }
