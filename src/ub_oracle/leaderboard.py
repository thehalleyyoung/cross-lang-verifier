"""Step 168 -- public leaderboard and auto-scoring harness.

The leaderboard is deliberately split into:

* a public benchmark manifest with runnable C/target programs and argv inputs;
* an answer key that a benchmark host can keep server-side; and
* a pure-Python scorer for third-party submissions.

The committed answer key makes the artifact reproducible and CI-checkable.  The
scorer never recompiles during normal scoring: labels are the ground-truth split's
frozen answers, while a separate optional confirmation path recompiles a sample
with real clang/UBSan plus the target compiler to prove the split remains grounded
in executable code.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .ground_truth import GTItem, LabelEvidence, label_item
from . import ground_truth as gt
from .reexec import ReexecHarness, ToolchainStatus, toolchain_available

SCHEMA_VERSION = "leaderboard/v1"
PUBLIC_CASES_SCHEMA_VERSION = "leaderboard-public-cases/v1"
ANSWER_KEY_SCHEMA_VERSION = "leaderboard-answer-key/v1"
SUBMISSION_SCHEMA_VERSION = "leaderboard-submission/v1"

BENCHMARK_ID = "xlev-semrec-public-leaderboard-2026-06"
SPLIT_SEED = "cross-lang-verifier-step-168-public-leaderboard-v1"
TARGET_LANGS: Tuple[str, ...] = ("rust", "go")
LABELS: Tuple[str, ...] = ("divergent", "equivalent")
PREDICTIONS: Tuple[str, ...] = ("divergent", "equivalent", "abstain")
CASES_PER_LABEL_PER_LANG = 30
CALIBRATION_SKIP_PER_CLASS = 2

_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = _ROOT / "experiments" / "leaderboard"
PUBLIC_CASES_PATH = EXPERIMENT_DIR / "public_cases.json"
ANSWER_KEY_PATH = EXPERIMENT_DIR / "answer_key.json"
SAMPLE_SUBMISSION_PATH = EXPERIMENT_DIR / "sample_submission.json"
DOC_PATH = _ROOT / "docs" / "leaderboard.md"

_ABSTAIN_ALIASES = {"abstain", "unknown", "not-covered", "not_covered", "unsupported"}


def _canonical_bytes(obj: object) -> bytes:
    return json.dumps(
        obj, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_obj(obj: object) -> str:
    return hashlib.sha256(_canonical_bytes(obj)).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _counts(values: Iterable[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items()))


def _score_rank(item: GTItem) -> str:
    payload = f"{SPLIT_SEED}:{item.item_id}:{item.content_hash}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _calibration_ids(items: Sequence[GTItem]) -> Tuple[str, ...]:
    by_bucket: Dict[Tuple[str, str], List[GTItem]] = {}
    for item in items:
        by_bucket.setdefault((item.lang, item.klass), []).append(item)
    skipped: List[str] = []
    for bucket in by_bucket.values():
        skipped.extend(
            item.item_id for item in bucket[:CALIBRATION_SKIP_PER_CLASS]
        )
    return tuple(sorted(skipped))


@dataclass(frozen=True)
class LeaderboardCase:
    case_id: str
    source_item_id: str
    target_lang: str
    family: str
    cwe: str
    label: str
    c_src: str
    target_src: str
    inputs: Tuple[str, ...]

    @property
    def source_lang(self) -> str:
        return "c"

    @property
    def pair(self) -> str:
        return f"c->{self.target_lang}"

    @property
    def source_sha256(self) -> str:
        return _sha256_text(self.c_src)

    @property
    def target_sha256(self) -> str:
        return _sha256_text(self.target_src)

    @property
    def content_hash(self) -> str:
        stable = {
            "case_id": self.case_id,
            "pair": self.pair,
            "source_sha256": self.source_sha256,
            "target_sha256": self.target_sha256,
            "inputs": list(self.inputs),
        }
        return _sha256_obj(stable)

    def to_gt_item(self) -> GTItem:
        return GTItem(
            item_id=self.case_id,
            lang=self.target_lang,
            klass=self.family,
            cwe=self.cwe,
            declared_label=self.label,
            c_src=self.c_src,
            target_src=self.target_src,
            inputs=self.inputs,
        )

    def public_entry(self) -> Dict[str, object]:
        return {
            "case_id": self.case_id,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "pair": self.pair,
            "inputs": list(self.inputs),
            "source_sha256": self.source_sha256,
            "target_sha256": self.target_sha256,
            "content_hash": self.content_hash,
            "source": self.c_src,
            "target": self.target_src,
        }

    def answer_entry(self) -> Dict[str, object]:
        return {
            "case_id": self.case_id,
            "source_item_id": self.source_item_id,
            "target_lang": self.target_lang,
            "pair": self.pair,
            "family": self.family,
            "cwe": self.cwe,
            "label": self.label,
            "inputs": list(self.inputs),
            "source_sha256": self.source_sha256,
            "target_sha256": self.target_sha256,
            "content_hash": self.content_hash,
        }


def build_cases() -> Tuple[LeaderboardCase, ...]:
    """Select the deterministic public leaderboard split.

    The split is balanced by target language and label, and it excludes the first
    few per-family calibration examples used by other smoke checks.  In an open
    repository this is not an anti-cheat mechanism; it simply keeps the benchmark
    split distinct from tutorial/control examples while preserving full
    reproducibility.
    """

    all_items = gt.enumerate_corpus(TARGET_LANGS)
    skipped = set(_calibration_ids(all_items))
    pool = [item for item in all_items if item.item_id not in skipped]

    selected: List[GTItem] = []
    for lang in TARGET_LANGS:
        for label in LABELS:
            bucket = [
                item
                for item in pool
                if item.lang == lang and item.declared_label == label
            ]
            bucket = sorted(bucket, key=_score_rank)
            if len(bucket) < CASES_PER_LABEL_PER_LANG:
                raise AssertionError(
                    f"not enough {lang}/{label} items for leaderboard split"
                )
            selected.extend(bucket[:CASES_PER_LABEL_PER_LANG])

    ordered = sorted(selected, key=lambda item: (_score_rank(item), item.lang, item.item_id))
    cases: List[LeaderboardCase] = []
    for index, item in enumerate(ordered, 1):
        cases.append(
            LeaderboardCase(
                case_id=f"lb-{index:03d}",
                source_item_id=item.item_id,
                target_lang=item.lang,
                family=item.klass,
                cwe=item.cwe,
                label=item.declared_label,
                c_src=item.c_src,
                target_src=item.target_src,
                inputs=tuple(item.inputs),
            )
        )
    return tuple(cases)


def public_cases_document(
    cases: Optional[Sequence[LeaderboardCase]] = None,
) -> Dict[str, object]:
    cases = build_cases() if cases is None else tuple(cases)
    stable = {
        "schema": PUBLIC_CASES_SCHEMA_VERSION,
        "benchmark_id": BENCHMARK_ID,
        "split_seed_sha256": _sha256_text(SPLIT_SEED),
        "n_cases": len(cases),
        "n_labels_hidden_from_manifest": len(LABELS),
        "target_langs": list(TARGET_LANGS),
        "prediction_labels": list(PREDICTIONS),
        "submission_schema": SUBMISSION_SCHEMA_VERSION,
        "case_selection": {
            "cases_per_label_per_lang": CASES_PER_LABEL_PER_LANG,
            "calibration_skip_per_class": CALIBRATION_SKIP_PER_CLASS,
            "selection_policy": (
                "hash-ranked deterministic split from ub_oracle.ground_truth "
                "excluding calibration examples"
            ),
        },
        "cases": [case.public_entry() for case in cases],
    }
    return {
        **stable,
        "content_hash": _sha256_obj(stable),
    }


def answer_key_document(
    cases: Optional[Sequence[LeaderboardCase]] = None,
) -> Dict[str, object]:
    cases = build_cases() if cases is None else tuple(cases)
    answers = [case.answer_entry() for case in cases]
    stable = {
        "schema": ANSWER_KEY_SCHEMA_VERSION,
        "benchmark_id": BENCHMARK_ID,
        "public_cases_hash": public_cases_document(cases)["content_hash"],
        "n_cases": len(cases),
        "by_label": _counts(case.label for case in cases),
        "by_pair": _counts(case.pair for case in cases),
        "by_family": _counts(case.family for case in cases),
        "ground_truth_policy": (
            "Labels come from the Step-45 sanitizer-backed ground-truth corpus. "
            "Scoring uses this frozen key without compiling; "
            "confirm_leaderboard_sample recompiles a deterministic sample with "
            "real clang/UBSan and target compilers as an evidence check."
        ),
        "answers": answers,
    }
    return {
        **stable,
        "content_hash": _sha256_obj(stable),
    }


def sample_submission_document(
    cases: Optional[Sequence[LeaderboardCase]] = None,
) -> Dict[str, object]:
    cases = build_cases() if cases is None else tuple(cases)
    stable = {
        "schema": SUBMISSION_SCHEMA_VERSION,
        "benchmark_id": BENCHMARK_ID,
        "submission_id": "sample-all-abstain",
        "tool": "example-abstaining-baseline",
        "tool_version": "0",
        "prediction_semantics": (
            "Each prediction is one of divergent, equivalent, or abstain. "
            "Missing cases are scored as abstain."
        ),
        "predictions": [
            {"case_id": case.case_id, "prediction": "abstain"} for case in cases
        ],
    }
    return {
        **stable,
        "content_hash": _sha256_obj(stable),
    }


def perfect_submission_document(
    cases: Optional[Sequence[LeaderboardCase]] = None,
) -> Dict[str, object]:
    cases = build_cases() if cases is None else tuple(cases)
    stable = {
        "schema": SUBMISSION_SCHEMA_VERSION,
        "benchmark_id": BENCHMARK_ID,
        "submission_id": "oracle-perfect-reference",
        "tool": "answer-key-reference",
        "tool_version": "0",
        "predictions": [
            {"case_id": case.case_id, "prediction": case.label} for case in cases
        ],
    }
    return {
        **stable,
        "content_hash": _sha256_obj(stable),
    }


def _json_line(obj: Mapping[str, object]) -> str:
    return json.dumps(obj, indent=2, sort_keys=True) + "\n"


def write_artifacts() -> None:
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    PUBLIC_CASES_PATH.write_text(_json_line(public_cases_document()), encoding="utf-8")
    ANSWER_KEY_PATH.write_text(_json_line(answer_key_document()), encoding="utf-8")
    SAMPLE_SUBMISSION_PATH.write_text(
        _json_line(sample_submission_document()), encoding="utf-8"
    )


def load_public_cases(path: Path = PUBLIC_CASES_PATH) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_answer_key(path: Path = ANSWER_KEY_PATH) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def check_artifacts() -> Tuple[bool, str]:
    expected = (
        (PUBLIC_CASES_PATH, _json_line(public_cases_document()), "public cases"),
        (ANSWER_KEY_PATH, _json_line(answer_key_document()), "answer key"),
        (SAMPLE_SUBMISSION_PATH, _json_line(sample_submission_document()), "sample submission"),
    )
    for path, regenerated, label in expected:
        if not path.exists():
            return False, f"{path} is missing"
        if path.read_text(encoding="utf-8") != regenerated:
            return False, f"{path} does not match regenerated {label}"
    key = load_answer_key()
    if key.get("public_cases_hash") != load_public_cases().get("content_hash"):
        return False, "answer key does not bind to the public cases hash"
    return True, "OK"


def _normalize_prediction(raw: object) -> Optional[str]:
    if not isinstance(raw, str):
        return None
    value = raw.strip().lower()
    if value in LABELS:
        return value
    if value in _ABSTAIN_ALIASES:
        return "abstain"
    return None


def _answers_by_case(answer_key: Mapping[str, object]) -> Dict[str, str]:
    answers = answer_key.get("answers")
    if not isinstance(answers, list):
        raise ValueError("answer key must contain an answers array")
    out: Dict[str, str] = {}
    for entry in answers:
        if not isinstance(entry, Mapping):
            raise ValueError("answer key entries must be objects")
        case_id = str(entry.get("case_id", ""))
        label = str(entry.get("label", ""))
        if not case_id or label not in LABELS:
            raise ValueError(f"invalid answer entry for case_id={case_id!r}")
        out[case_id] = label
    return out


@dataclass(frozen=True)
class LeaderboardScore:
    valid: bool
    submission_id: str
    benchmark_id: str
    total_cases: int
    answered_cases: int
    abstained_cases: int
    correct_cases: int
    accuracy: float
    answered_accuracy: float
    coverage: float
    divergent_precision: float
    divergent_recall: float
    divergent_f1: float
    equivalent_precision: float
    equivalent_recall: float
    equivalent_f1: float
    macro_f1: float
    primary_score: float
    confusion: Mapping[str, Mapping[str, int]]
    missing_case_ids: Tuple[str, ...] = ()
    errors: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, object]:
        return {
            "valid": self.valid,
            "submission_id": self.submission_id,
            "benchmark_id": self.benchmark_id,
            "total_cases": self.total_cases,
            "answered_cases": self.answered_cases,
            "abstained_cases": self.abstained_cases,
            "correct_cases": self.correct_cases,
            "accuracy": self.accuracy,
            "answered_accuracy": self.answered_accuracy,
            "coverage": self.coverage,
            "divergent_precision": self.divergent_precision,
            "divergent_recall": self.divergent_recall,
            "divergent_f1": self.divergent_f1,
            "equivalent_precision": self.equivalent_precision,
            "equivalent_recall": self.equivalent_recall,
            "equivalent_f1": self.equivalent_f1,
            "macro_f1": self.macro_f1,
            "primary_score": self.primary_score,
            "confusion": {
                truth: dict(preds) for truth, preds in self.confusion.items()
            },
            "missing_case_ids": list(self.missing_case_ids),
            "errors": list(self.errors),
        }


def _ratio(num: int, den: int) -> float:
    return 0.0 if den == 0 else num / den


def _f1(precision: float, recall: float) -> float:
    den = precision + recall
    return 0.0 if den == 0.0 else 2.0 * precision * recall / den


def score_submission(
    submission: Mapping[str, object],
    answer_key: Optional[Mapping[str, object]] = None,
) -> LeaderboardScore:
    """Score a leaderboard submission against the frozen answer key.

    Unknown, duplicate, or malformed case IDs make the submission invalid.
    Missing cases are allowed but scored as abstentions.
    """

    answer_key = load_answer_key() if answer_key is None else answer_key
    answers = _answers_by_case(answer_key)
    expected_cases = set(answers)
    submission_id = str(submission.get("submission_id") or "<unnamed>")
    benchmark_id = str(submission.get("benchmark_id") or answer_key.get("benchmark_id", ""))
    predictions = submission.get("predictions", [])
    errors: List[str] = []
    pred_by_case: Dict[str, str] = {}
    seen: set = set()

    if not isinstance(predictions, list):
        errors.append("predictions must be an array")
        predictions = []

    for idx, pred in enumerate(predictions):
        if not isinstance(pred, Mapping):
            errors.append(f"prediction[{idx}] must be an object")
            continue
        case_id = str(pred.get("case_id") or "")
        if not case_id:
            errors.append(f"prediction[{idx}] is missing case_id")
            continue
        if case_id in seen:
            errors.append(f"duplicate prediction for {case_id}")
            continue
        seen.add(case_id)
        if case_id not in expected_cases:
            errors.append(f"unknown case_id {case_id}")
            continue
        verdict = _normalize_prediction(pred.get("prediction", pred.get("verdict")))
        if verdict is None:
            errors.append(f"{case_id}: invalid prediction {pred.get('prediction')!r}")
            continue
        pred_by_case[case_id] = verdict

    missing = tuple(sorted(expected_cases - set(pred_by_case)))
    for case_id in missing:
        pred_by_case[case_id] = "abstain"

    confusion: Dict[str, Dict[str, int]] = {
        truth: {pred: 0 for pred in PREDICTIONS} for truth in LABELS
    }
    correct = 0
    answered = 0
    for case_id in sorted(expected_cases):
        truth = answers[case_id]
        pred = pred_by_case[case_id]
        if pred not in PREDICTIONS:
            pred = "abstain"
        confusion[truth][pred] += 1
        if pred != "abstain":
            answered += 1
        if pred == truth:
            correct += 1

    total = len(expected_cases)
    abstained = total - answered

    def class_metrics(label: str) -> Tuple[float, float, float]:
        tp = confusion[label][label]
        fp = sum(
            confusion[truth][label] for truth in LABELS if truth != label
        )
        fn = sum(
            count
            for pred, count in confusion[label].items()
            if pred != label
        )
        precision = _ratio(tp, tp + fp)
        recall = _ratio(tp, tp + fn)
        return precision, recall, _f1(precision, recall)

    div_p, div_r, div_f = class_metrics("divergent")
    eq_p, eq_r, eq_f = class_metrics("equivalent")
    macro_f1 = (div_f + eq_f) / 2.0
    coverage = _ratio(answered, total)
    return LeaderboardScore(
        valid=not errors,
        submission_id=submission_id,
        benchmark_id=benchmark_id,
        total_cases=total,
        answered_cases=answered,
        abstained_cases=abstained,
        correct_cases=correct,
        accuracy=_ratio(correct, total),
        answered_accuracy=_ratio(correct, answered),
        coverage=coverage,
        divergent_precision=div_p,
        divergent_recall=div_r,
        divergent_f1=div_f,
        equivalent_precision=eq_p,
        equivalent_recall=eq_r,
        equivalent_f1=eq_f,
        macro_f1=macro_f1,
        primary_score=round(100.0 * macro_f1 * coverage, 2),
        confusion=confusion,
        missing_case_ids=missing,
        errors=tuple(errors),
    )


@dataclass(frozen=True)
class LeaderboardConfirmationResult:
    case_id: str
    target_lang: str
    family: str
    expected_label: str
    observed_label: str
    agrees: bool
    detail: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "case_id": self.case_id,
            "target_lang": self.target_lang,
            "family": self.family,
            "expected_label": self.expected_label,
            "observed_label": self.observed_label,
            "agrees": self.agrees,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class LeaderboardConfirmationReport:
    available: bool
    ok: bool
    sample_size: int
    n_confirmed: int
    results: Tuple[LeaderboardConfirmationResult, ...] = field(default_factory=tuple)
    detail: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "available": self.available,
            "ok": self.ok,
            "sample_size": self.sample_size,
            "n_confirmed": self.n_confirmed,
            "results": [result.to_dict() for result in self.results],
            "detail": self.detail,
        }


def _sample_cases(
    cases: Sequence[LeaderboardCase],
    *,
    sample_size: int,
    seed: int,
) -> Tuple[LeaderboardCase, ...]:
    rng = random.Random(seed)
    ordered = sorted(cases, key=lambda case: case.case_id)
    if sample_size >= len(ordered):
        return tuple(ordered)
    return tuple(sorted(rng.sample(ordered, sample_size), key=lambda case: case.case_id))


def confirm_leaderboard_sample(
    *,
    sample_size: int = 8,
    seed: int = 168,
    status: Optional[ToolchainStatus] = None,
    harness: Optional[ReexecHarness] = None,
) -> LeaderboardConfirmationReport:
    status = status or toolchain_available()
    cases = [
        case for case in build_cases()
        if status.full_for(case.target_lang)
    ]
    if not cases:
        return LeaderboardConfirmationReport(
            available=False,
            ok=True,
            sample_size=0,
            n_confirmed=0,
            detail="no full C+UBSan+target toolchain available for leaderboard targets",
        )
    harness = harness or ReexecHarness(status)
    chosen = _sample_cases(cases, sample_size=sample_size, seed=seed)
    results: List[LeaderboardConfirmationResult] = []
    for case in chosen:
        evidence: LabelEvidence = label_item(harness, case.to_gt_item())
        agrees = evidence.observed_label == case.label
        results.append(
            LeaderboardConfirmationResult(
                case_id=case.case_id,
                target_lang=case.target_lang,
                family=case.family,
                expected_label=case.label,
                observed_label=evidence.observed_label,
                agrees=agrees,
                detail=evidence.detail,
            )
        )
    ok = all(result.agrees for result in results)
    return LeaderboardConfirmationReport(
        available=True,
        ok=ok,
        sample_size=len(chosen),
        n_confirmed=sum(1 for result in results if result.agrees),
        results=tuple(results),
        detail=(
            "sample labels agree with real compiler-backed ground truth"
            if ok
            else "at least one leaderboard sample label disagreed with live ground truth"
        ),
    )


def results_document() -> Dict[str, object]:
    cases = build_cases()
    key = answer_key_document(cases)
    sample_score = score_submission(sample_submission_document(cases), key)
    perfect_score = score_submission(perfect_submission_document(cases), key)
    stable = {
        "schema": SCHEMA_VERSION,
        "benchmark_id": BENCHMARK_ID,
        "public_cases_hash": public_cases_document(cases)["content_hash"],
        "answer_key_hash": key["content_hash"],
        "n_cases": len(cases),
        "by_label": key["by_label"],
        "by_pair": key["by_pair"],
        "by_family": key["by_family"],
        "sample_submission_score": sample_score.to_dict(),
        "perfect_reference_score": perfect_score.to_dict(),
    }
    return {
        **stable,
        "content_hash": _sha256_obj(stable),
    }


def markdown_document() -> str:
    doc = results_document()
    lines = [
        "# Public leaderboard",
        "",
        "Generated by `ub_oracle.leaderboard`; run `make leaderboard-check` to "
        "assert that the public cases, answer key, sample submission, and scorer "
        "remain byte-reproducible.",
        "",
        "The benchmark is a deterministic C→Rust/C→Go split of executable "
        "ground-truth cases. The public manifest contains source code and argv "
        "inputs but no labels. The committed answer key is present so the artifact "
        "can be reproduced and audited; a hosted leaderboard can keep the same "
        "key server-side and accept only submission JSON.",
        "",
        f"**Benchmark:** `{doc['benchmark_id']}`",
        f"**Cases:** {doc['n_cases']}",
        f"**Public cases hash:** `{doc['public_cases_hash']}`",
        f"**Answer key hash:** `{doc['answer_key_hash']}`",
        f"**Leaderboard hash:** `{doc['content_hash']}`",
        "",
        "## Submit",
        "",
        "```bash",
        "python3 -m experiments.leaderboard.run --score my_submission.json --json",
        "```",
        "",
        "A submission is:",
        "",
        "```json",
        "{",
        f'  "schema": "{SUBMISSION_SCHEMA_VERSION}",',
        f'  "benchmark_id": "{BENCHMARK_ID}",',
        '  "submission_id": "my-tool-2026-06-03",',
        '  "tool": "my-tool",',
        '  "predictions": [',
        '    {"case_id": "lb-001", "prediction": "divergent"}',
        "  ]",
        "}",
        "```",
        "",
        "`prediction` is `divergent`, `equivalent`, or `abstain`; missing cases "
        "are scored as abstentions. The primary score is "
        "`100 * macro_f1 * coverage`, so abstaining is safe but not free.",
        "",
        "## Split balance",
        "",
        "| dimension | counts |",
        "|---|---:|",
        f"| labels | {_fmt_counts(doc['by_label'])} |",
        f"| pairs | {_fmt_counts(doc['by_pair'])} |",
        f"| families | {_fmt_counts(doc['by_family'])} |",
        "",
        "## Reference scores",
        "",
        "| submission | valid | coverage | macro-F1 | primary score |",
        "|---|---:|---:|---:|---:|",
        _score_row("sample all-abstain", doc["sample_submission_score"]),
        _score_row("perfect reference", doc["perfect_reference_score"]),
        "",
        "## Evidence boundary",
        "",
        "Normal scoring is toolchain-free and compares predictions to the frozen "
        "answer key. `confirm_leaderboard_sample()` is a separate evidence check "
        "that recompiles a deterministic sample with real clang/UBSan plus the "
        "target compiler and requires the observed sanitizer label to match the "
        "key. The public, committed key makes this repository reproducible; it is "
        "not an anti-cheat secret in an open-source checkout.",
    ]
    return "\n".join(lines).rstrip() + "\n"


def _fmt_counts(mapping: Mapping[str, object]) -> str:
    return ", ".join(f"`{key}`={mapping[key]}" for key in sorted(mapping))


def _score_row(name: str, score: Mapping[str, object]) -> str:
    return (
        f"| {name} | {score['valid']} | {float(score['coverage']):.3f} | "
        f"{float(score['macro_f1']):.3f} | {float(score['primary_score']):.2f} |"
    )


def write_markdown(path: Path = DOC_PATH) -> None:
    path.write_text(markdown_document(), encoding="utf-8")


def check_markdown(path: Path = DOC_PATH) -> Tuple[bool, str]:
    if not path.exists():
        return False, f"{path} is missing"
    if path.read_text(encoding="utf-8") != markdown_document():
        return False, f"{path} does not match regenerated leaderboard docs"
    return True, "OK"


def write_all() -> None:
    write_artifacts()
    write_markdown()


def check_all() -> Tuple[bool, str]:
    ok, detail = check_artifacts()
    if not ok:
        return ok, detail
    return check_markdown()


__all__ = [
    "SCHEMA_VERSION",
    "PUBLIC_CASES_SCHEMA_VERSION",
    "ANSWER_KEY_SCHEMA_VERSION",
    "SUBMISSION_SCHEMA_VERSION",
    "BENCHMARK_ID",
    "PUBLIC_CASES_PATH",
    "ANSWER_KEY_PATH",
    "SAMPLE_SUBMISSION_PATH",
    "DOC_PATH",
    "LeaderboardCase",
    "LeaderboardScore",
    "LeaderboardConfirmationReport",
    "build_cases",
    "public_cases_document",
    "answer_key_document",
    "sample_submission_document",
    "perfect_submission_document",
    "score_submission",
    "confirm_leaderboard_sample",
    "results_document",
    "write_artifacts",
    "check_artifacts",
    "markdown_document",
    "write_all",
    "check_all",
]
