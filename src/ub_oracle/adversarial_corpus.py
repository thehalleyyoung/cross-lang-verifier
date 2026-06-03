"""Step 165 -- hand-crafted adversarial near-miss corpus.

The internal red-team already mutates genuinely divergent witnesses across the
whole oracle registry.  This module is deliberately different: it is a small,
persistent corpus of *near misses* at the verifier's decision boundaries.  Each
case is one of:

* a divergent control that must be reported as a symbolic ``CANDIDATE`` on the
  deterministic no-toolchain path, never as ``NO_DIVERGENCE_FOUND``;
* a safe twin one edit away from divergence that must be discharged by the
  interval pre-pass as ``NO_DIVERGENCE_FOUND``; or
* an unsupported/mis-probed unit that must abstain loudly as ``NOT_COVERED``.

The committed manifest is generated only from the deterministic no-toolchain
path.  Real-compiler confirmation is kept as a separate optional check so the
byte-stable artifact never depends on host toolchains.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

from .ir import validate_unit
from .reexec import ReexecHarness, ToolchainStatus, toolchain_available
from .verify import VerifyVerdict, applicable_oracles, verify_unit

SCHEMA_VERSION = "adversarial-corpus/v1"

_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = _ROOT / "experiments" / "adversarial_corpus"
RESULTS_PATH = EXPERIMENT_DIR / "adversarial_corpus.json"

INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1

POLICY_DIVERGENT = "divergent_control"
POLICY_SAFE = "safe_near_miss"
POLICY_ABSTAIN = "loud_abstention"

_NO_TOOLCHAIN = ToolchainStatus(cc=None, ubsan=False, targets=(), runners=())
_BREACH_FOR_DIVERGENT = {
    VerifyVerdict.NO_DIVERGENCE_FOUND.value,
    VerifyVerdict.NOT_COVERED.value,
}


def _canonical_bytes(obj: object) -> bytes:
    return json.dumps(
        obj, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_obj(obj: object) -> str:
    return hashlib.sha256(_canonical_bytes(obj)).hexdigest()


@dataclass(frozen=True)
class AdversarialCase:
    case_id: str
    family: str
    policy: str
    unit: Mapping[str, object]
    expected_static_verdict: str
    rationale: str
    required_prepass_pruned: Tuple[str, ...] = ()

    @property
    def source_lang(self) -> str:
        return str(self.unit.get("source_lang") or "c")

    @property
    def target_lang(self) -> str:
        return str(self.unit.get("target_lang") or "rust")

    @property
    def pair(self) -> str:
        return f"{self.source_lang}->{self.target_lang}"

    @property
    def unit_hash(self) -> str:
        return _sha256_obj(dict(self.unit))[:24]

    @property
    def content_hash(self) -> str:
        payload = {
            "case_id": self.case_id,
            "family": self.family,
            "policy": self.policy,
            "unit": dict(self.unit),
            "expected_static_verdict": self.expected_static_verdict,
            "required_prepass_pruned": self.required_prepass_pruned,
            "rationale": self.rationale,
        }
        return _sha256_obj(payload)[:24]

    def to_manifest_entry(self) -> Dict[str, object]:
        return {
            "case_id": self.case_id,
            "family": self.family,
            "policy": self.policy,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "pair": self.pair,
            "unit": dict(self.unit),
            "unit_hash": self.unit_hash,
            "content_hash": self.content_hash,
            "expected_static_verdict": self.expected_static_verdict,
            "required_prepass_pruned": list(self.required_prepass_pruned),
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class AdversarialOutcome:
    case_id: str
    family: str
    policy: str
    expected_static_verdict: str
    verdict: str
    applicable_classes: Tuple[str, ...]
    prepass_pruned: Tuple[str, ...]
    passed: bool
    breach: bool
    detail: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "case_id": self.case_id,
            "family": self.family,
            "policy": self.policy,
            "expected_static_verdict": self.expected_static_verdict,
            "verdict": self.verdict,
            "applicable_classes": list(self.applicable_classes),
            "prepass_pruned": list(self.prepass_pruned),
            "passed": self.passed,
            "breach": self.breach,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class AdversarialReport:
    outcomes: Tuple[AdversarialOutcome, ...] = field(default_factory=tuple)

    @property
    def n_cases(self) -> int:
        return len(self.outcomes)

    @property
    def failures(self) -> Tuple[AdversarialOutcome, ...]:
        return tuple(o for o in self.outcomes if not o.passed)

    @property
    def breaches(self) -> Tuple[AdversarialOutcome, ...]:
        return tuple(o for o in self.outcomes if o.breach)

    @property
    def ok(self) -> bool:
        return not self.failures and not self.breaches

    def verdict_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for outcome in self.outcomes:
            counts[outcome.verdict] = counts.get(outcome.verdict, 0) + 1
        return dict(sorted(counts.items()))

    def policy_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for outcome in self.outcomes:
            counts[outcome.policy] = counts.get(outcome.policy, 0) + 1
        return dict(sorted(counts.items()))


@dataclass(frozen=True)
class LiveAdversarialReport:
    available: bool
    ok: bool
    n_controls: int
    n_confirmed: int
    outcomes: Tuple[AdversarialOutcome, ...]
    detail: str = ""

    @property
    def breaches(self) -> Tuple[AdversarialOutcome, ...]:
        return tuple(o for o in self.outcomes if o.breach)


def _case(
    case_id: str,
    family: str,
    policy: str,
    unit: Mapping[str, object],
    expected: str,
    rationale: str,
    pruned: Sequence[str] = (),
) -> AdversarialCase:
    return AdversarialCase(
        case_id=case_id,
        family=family,
        policy=policy,
        unit=dict(unit),
        expected_static_verdict=expected,
        required_prepass_pruned=tuple(pruned),
        rationale=rationale,
    )


def corpus() -> Tuple[AdversarialCase, ...]:
    """The deterministic hand-crafted Step-165 corpus."""
    return (
        _case(
            "signed-overflow-upper-boundary",
            "signed_overflow",
            POLICY_DIVERGENT,
            {
                "kind": "binop_const",
                "op": "add",
                "const": 1,
                "width": 32,
                "signed": True,
                "x_range": [INT32_MAX, INT32_MAX],
                "source_lang": "c",
                "target_lang": "rust",
            },
            VerifyVerdict.CANDIDATE.value,
            "The range contains exactly INT_MAX, so x+1 is reachable C signed-overflow UB.",
        ),
        _case(
            "signed-overflow-safe-one-below",
            "signed_overflow",
            POLICY_SAFE,
            {
                "kind": "binop_const",
                "op": "add",
                "const": 1,
                "width": 32,
                "signed": True,
                "x_range": [INT32_MAX - 2, INT32_MAX - 1],
                "source_lang": "c",
                "target_lang": "rust",
            },
            VerifyVerdict.NO_DIVERGENCE_FOUND.value,
            "One-off safe twin: the declared range ends below the overflowing value.",
            ("signed_overflow",),
        ),
        _case(
            "signed-underflow-lower-boundary",
            "signed_overflow",
            POLICY_DIVERGENT,
            {
                "kind": "binop_const",
                "op": "sub",
                "const": 1,
                "width": 32,
                "signed": True,
                "x_range": [INT32_MIN, INT32_MIN],
                "source_lang": "c",
                "target_lang": "rust",
            },
            VerifyVerdict.CANDIDATE.value,
            "The lower boundary reaches INT_MIN-1 under subtraction, the signed-underflow twin of overflow.",
        ),
        _case(
            "signed-underflow-safe-one-above",
            "signed_overflow",
            POLICY_SAFE,
            {
                "kind": "binop_const",
                "op": "sub",
                "const": 1,
                "width": 32,
                "signed": True,
                "x_range": [INT32_MIN + 1, INT32_MIN + 2],
                "source_lang": "c",
                "target_lang": "rust",
            },
            VerifyVerdict.NO_DIVERGENCE_FOUND.value,
            "A near miss just above INT_MIN must be pruned rather than flagged.",
            ("signed_overflow",),
        ),
        _case(
            "signed-overflow-zero-constant-probe",
            "signed_overflow",
            POLICY_SAFE,
            {
                "kind": "binop_const",
                "op": "add",
                "const": 0,
                "width": 32,
                "signed": True,
                "probe": "signed_overflow",
                "source_lang": "c",
                "target_lang": "rust",
            },
            VerifyVerdict.NO_DIVERGENCE_FOUND.value,
            "An explicit signed-overflow probe with a zero constant should discharge across the full type.",
            ("signed_overflow",),
        ),
        _case(
            "overflow-unit-wrong-probe-abstains",
            "probe_routing",
            POLICY_ABSTAIN,
            {
                "kind": "binop_const",
                "op": "add",
                "const": 1,
                "width": 32,
                "signed": True,
                "x_range": [INT32_MAX, INT32_MAX],
                "probe": "div_by_zero",
                "source_lang": "c",
                "target_lang": "rust",
            },
            VerifyVerdict.NOT_COVERED.value,
            "A mismatched explicit probe must not silently route to the signed-overflow oracle.",
        ),
        _case(
            "go-overflow-upper-boundary",
            "signed_overflow",
            POLICY_DIVERGENT,
            {
                "kind": "binop_const",
                "op": "add",
                "const": 1,
                "width": 32,
                "signed": True,
                "x_range": [INT32_MAX, INT32_MAX],
                "source_lang": "c",
                "target_lang": "go",
            },
            VerifyVerdict.CANDIDATE.value,
            "The same UB boundary should transfer to a non-Rust target pack.",
        ),
        _case(
            "go-overflow-safe-one-below",
            "signed_overflow",
            POLICY_SAFE,
            {
                "kind": "binop_const",
                "op": "add",
                "const": 1,
                "width": 32,
                "signed": True,
                "x_range": [INT32_MAX - 2, INT32_MAX - 1],
                "source_lang": "c",
                "target_lang": "go",
            },
            VerifyVerdict.NO_DIVERGENCE_FOUND.value,
            "The range pre-pass must prune the safe twin independently of target language.",
            ("signed_overflow",),
        ),
        _case(
            "division-zero-window",
            "division_by_zero",
            POLICY_DIVERGENT,
            {
                "kind": "div",
                "width": 32,
                "signed": True,
                "a_range": [0, 1000],
                "b_range": [-1, 1],
                "source_lang": "c",
                "target_lang": "rust",
            },
            VerifyVerdict.CANDIDATE.value,
            "The divisor interval includes zero but the dividend excludes INT_MIN, isolating div-by-zero.",
            ("intmin_div_neg1",),
        ),
        _case(
            "division-nonzero-safe-window",
            "division_by_zero",
            POLICY_SAFE,
            {
                "kind": "div",
                "width": 32,
                "signed": True,
                "a_range": [0, 1000],
                "b_range": [1, 3],
                "source_lang": "c",
                "target_lang": "rust",
            },
            VerifyVerdict.NO_DIVERGENCE_FOUND.value,
            "The safe twin excludes both zero and the INT_MIN/-1 pair.",
            ("div_by_zero", "intmin_div_neg1"),
        ),
        _case(
            "remainder-zero-window",
            "division_by_zero",
            POLICY_DIVERGENT,
            {
                "kind": "rem",
                "width": 32,
                "signed": True,
                "a_range": [0, 1000],
                "b_range": [-1, 1],
                "source_lang": "c",
                "target_lang": "rust",
            },
            VerifyVerdict.CANDIDATE.value,
            "Remainder by zero shares the same divisor precondition as division.",
            ("intmin_div_neg1",),
        ),
        _case(
            "intmin-div-neg1-boundary",
            "intmin_div_neg1",
            POLICY_DIVERGENT,
            {
                "kind": "div",
                "width": 32,
                "signed": True,
                "a_range": [INT32_MIN, INT32_MIN],
                "b_range": [-1, -1],
                "source_lang": "c",
                "target_lang": "rust",
            },
            VerifyVerdict.CANDIDATE.value,
            "The unique signed-division overflow pair is reachable while zero is excluded.",
            ("div_by_zero",),
        ),
        _case(
            "intmin-div-neg1-safe-positive-divisor",
            "intmin_div_neg1",
            POLICY_SAFE,
            {
                "kind": "div",
                "width": 32,
                "signed": True,
                "a_range": [INT32_MIN, INT32_MIN],
                "b_range": [1, 1],
                "source_lang": "c",
                "target_lang": "rust",
            },
            VerifyVerdict.NO_DIVERGENCE_FOUND.value,
            "The most-negative dividend is safe when the divisor cannot be -1 or zero.",
            ("div_by_zero", "intmin_div_neg1"),
        ),
        _case(
            "unsigned-intmin-neg1-near-miss",
            "intmin_div_neg1",
            POLICY_SAFE,
            {
                "kind": "div",
                "width": 32,
                "signed": False,
                "a_range": [INT32_MIN, INT32_MIN],
                "b_range": [-1, -1],
                "source_lang": "c",
                "target_lang": "rust",
            },
            VerifyVerdict.NO_DIVERGENCE_FOUND.value,
            "The signed-overflow division oracle must not apply when the unit declares unsigned arithmetic.",
            ("div_by_zero",),
        ),
        _case(
            "shift-width-boundary",
            "shift_oob",
            POLICY_DIVERGENT,
            {
                "kind": "shift",
                "width": 32,
                "value": 1,
                "shift_range": [32, 32],
                "source_lang": "c",
                "target_lang": "rust",
            },
            VerifyVerdict.CANDIDATE.value,
            "The shift amount is exactly the bit width, the smallest out-of-range count.",
        ),
        _case(
            "shift-safe-max-count",
            "shift_oob",
            POLICY_SAFE,
            {
                "kind": "shift",
                "width": 32,
                "value": 1,
                "shift_range": [0, 31],
                "source_lang": "c",
                "target_lang": "rust",
            },
            VerifyVerdict.NO_DIVERGENCE_FOUND.value,
            "The safe twin reaches 31 but never the C UB threshold of 32.",
            ("shift_oob",),
        ),
        _case(
            "shift-negative-count",
            "shift_oob",
            POLICY_DIVERGENT,
            {
                "kind": "shift",
                "width": 32,
                "value": 1,
                "shift_range": [-1, -1],
                "source_lang": "c",
                "target_lang": "rust",
            },
            VerifyVerdict.CANDIDATE.value,
            "A negative count is also C shift UB and must not be pruned as safe.",
        ),
        _case(
            "go-shift-safe-max-count",
            "shift_oob",
            POLICY_SAFE,
            {
                "kind": "shift",
                "width": 32,
                "value": 1,
                "shift_range": [0, 31],
                "source_lang": "c",
                "target_lang": "go",
            },
            VerifyVerdict.NO_DIVERGENCE_FOUND.value,
            "The target-pack-generated Go shift oracle must respect the same pre-pass proof.",
            ("shift_oob",),
        ),
        _case(
            "array-oob-control",
            "array_oob",
            POLICY_DIVERGENT,
            {
                "kind": "array_index",
                "length": 4,
                "source_lang": "c",
                "target_lang": "rust",
            },
            VerifyVerdict.CANDIDATE.value,
            "A memory-shape divergent control keeps the corpus from being only arithmetic.",
        ),
        _case(
            "array-oob-wrong-probe-abstains",
            "probe_routing",
            POLICY_ABSTAIN,
            {
                "kind": "array_index",
                "length": 4,
                "probe": "signed_overflow",
                "source_lang": "c",
                "target_lang": "rust",
            },
            VerifyVerdict.NOT_COVERED.value,
            "A wrong probe on an otherwise covered memory unit must abstain loudly.",
        ),
        _case(
            "strict-aliasing-control",
            "strict_aliasing",
            POLICY_DIVERGENT,
            {
                "kind": "type_pun",
                "source_lang": "c",
                "target_lang": "rust",
            },
            VerifyVerdict.CANDIDATE.value,
            "Optimizer-exploited type-punning gives the corpus a non-sanitizer UB control.",
        ),
        _case(
            "c-to-c-overflow-unsupported-pair",
            "unsupported_pair",
            POLICY_ABSTAIN,
            {
                "kind": "binop_const",
                "op": "add",
                "const": 1,
                "width": 32,
                "signed": True,
                "x_range": [INT32_MAX, INT32_MAX],
                "source_lang": "c",
                "target_lang": "c",
            },
            VerifyVerdict.NOT_COVERED.value,
            "A well-formed but unsupported C-to-C pair must not be silently treated as C-to-Rust.",
        ),
        _case(
            "rust-to-go-unsupported-pair",
            "unsupported_pair",
            POLICY_ABSTAIN,
            {
                "kind": "binop_const",
                "op": "add",
                "const": 1,
                "width": 32,
                "signed": True,
                "x_range": [INT32_MAX, INT32_MAX],
                "source_lang": "rust",
                "target_lang": "go",
            },
            VerifyVerdict.NOT_COVERED.value,
            "A known-language but unsupported direction should abstain rather than guess.",
        ),
    )


def validate_cases(cases: Optional[Sequence[AdversarialCase]] = None) -> Tuple[bool, str]:
    items = tuple(cases if cases is not None else corpus())
    ids = [case.case_id for case in items]
    if len(ids) != len(set(ids)):
        return False, "duplicate case_id in adversarial corpus"
    allowed_policies = {POLICY_DIVERGENT, POLICY_SAFE, POLICY_ABSTAIN}
    for case in items:
        if case.policy not in allowed_policies:
            return False, f"{case.case_id}: unknown policy {case.policy!r}"
        errors = validate_unit(dict(case.unit))
        if errors:
            return False, f"{case.case_id}: invalid IR unit: {errors}"
        if case.policy == POLICY_SAFE and not case.required_prepass_pruned:
            return False, f"{case.case_id}: safe near-miss must name a pre-pass proof"
        if case.expected_static_verdict not in {v.value for v in VerifyVerdict}:
            return False, f"{case.case_id}: invalid expected verdict"
    return True, "OK"


def run_static_corpus(
    cases: Optional[Sequence[AdversarialCase]] = None,
) -> AdversarialReport:
    """Run the deterministic no-toolchain corpus and check exact static verdicts."""
    items = tuple(cases if cases is not None else corpus())
    outcomes = []
    for case in items:
        unit = dict(case.unit)
        applicable = tuple(
            sorted({oracle.divergence_class for oracle in applicable_oracles(unit)})
        )
        report = verify_unit(
            unit, confirm=False, status=_NO_TOOLCHAIN, prepass=True
        )
        pruned = tuple(sorted(set(report.prepass_pruned)))
        missing_pruned = tuple(
            cls for cls in case.required_prepass_pruned if cls not in pruned
        )
        verdict = report.verdict.value
        passed = verdict == case.expected_static_verdict and not missing_pruned
        breach = case.policy == POLICY_DIVERGENT and verdict in _BREACH_FOR_DIVERGENT
        detail = report.detail
        if missing_pruned:
            detail += f"; missing required pre-pass proof(s): {missing_pruned}"
        outcomes.append(
            AdversarialOutcome(
                case_id=case.case_id,
                family=case.family,
                policy=case.policy,
                expected_static_verdict=case.expected_static_verdict,
                verdict=verdict,
                applicable_classes=applicable,
                prepass_pruned=pruned,
                passed=passed,
                breach=breach,
                detail=detail,
            )
        )
    return AdversarialReport(tuple(outcomes))


def confirm_divergent_controls(
    cases: Optional[Sequence[AdversarialCase]] = None,
    *,
    status: Optional[ToolchainStatus] = None,
    harness: Optional[ReexecHarness] = None,
) -> LiveAdversarialReport:
    """Optionally replay divergent controls with real compilers.

    The live acceptance criterion is intentionally soundness-oriented: a
    divergent control may remain ``CANDIDATE`` if its target toolchain is absent,
    but it may never become ``NO_DIVERGENCE_FOUND`` or ``NOT_COVERED``.
    """
    status = status or toolchain_available()
    harness = harness or ReexecHarness(status)
    items = tuple(cases if cases is not None else corpus())
    controls = tuple(case for case in items if case.policy == POLICY_DIVERGENT)
    outcomes = []
    for case in controls:
        unit = dict(case.unit)
        applicable = tuple(
            sorted({oracle.divergence_class for oracle in applicable_oracles(unit)})
        )
        report = verify_unit(unit, harness=harness, status=status, confirm=True)
        verdict = report.verdict.value
        breach = verdict in _BREACH_FOR_DIVERGENT
        outcomes.append(
            AdversarialOutcome(
                case_id=case.case_id,
                family=case.family,
                policy=case.policy,
                expected_static_verdict=case.expected_static_verdict,
                verdict=verdict,
                applicable_classes=applicable,
                prepass_pruned=tuple(sorted(set(report.prepass_pruned))),
                passed=not breach,
                breach=breach,
                detail=report.detail,
            )
        )
    n_confirmed = sum(1 for outcome in outcomes if outcome.verdict == VerifyVerdict.DIVERGENT.value)
    available = any(
        status.full_for(case.target_lang)
        for case in controls
        if case.source_lang == "c"
    )
    ok = not any(outcome.breach for outcome in outcomes)
    if available:
        ok = ok and n_confirmed > 0
    detail = "OK" if ok else "divergent control escaped the sound verdict set"
    return LiveAdversarialReport(
        available=available,
        ok=ok,
        n_controls=len(controls),
        n_confirmed=n_confirmed,
        outcomes=tuple(outcomes),
        detail=detail,
    )


def corpus_census(
    cases: Optional[Sequence[AdversarialCase]] = None,
) -> Dict[str, object]:
    items = tuple(cases if cases is not None else corpus())
    by_policy: Dict[str, int] = {}
    by_family: Dict[str, int] = {}
    by_pair: Dict[str, int] = {}
    for case in items:
        by_policy[case.policy] = by_policy.get(case.policy, 0) + 1
        by_family[case.family] = by_family.get(case.family, 0) + 1
        by_pair[case.pair] = by_pair.get(case.pair, 0) + 1
    return {
        "schema": SCHEMA_VERSION,
        "n_cases": len(items),
        "n_families": len(by_family),
        "n_pairs": len(by_pair),
        "by_policy": dict(sorted(by_policy.items())),
        "by_family": dict(sorted(by_family.items())),
        "by_pair": dict(sorted(by_pair.items())),
    }


def manifest_entries(
    cases: Optional[Sequence[AdversarialCase]] = None,
) -> Tuple[Dict[str, object], ...]:
    items = tuple(cases if cases is not None else corpus())
    return tuple(case.to_manifest_entry() for case in items)


def content_hash(
    entries: Optional[Sequence[Mapping[str, object]]] = None,
    outcomes: Optional[Sequence[AdversarialOutcome]] = None,
) -> str:
    entries = manifest_entries() if entries is None else tuple(entries)
    outcomes = run_static_corpus().outcomes if outcomes is None else tuple(outcomes)
    stable = {
        "cases": [
            {
                "case_id": entry["case_id"],
                "policy": entry["policy"],
                "family": entry["family"],
                "pair": entry["pair"],
                "unit_hash": entry["unit_hash"],
                "content_hash": entry["content_hash"],
                "expected_static_verdict": entry["expected_static_verdict"],
                "required_prepass_pruned": entry["required_prepass_pruned"],
            }
            for entry in entries
        ],
        "outcomes": [
            {
                "case_id": outcome.case_id,
                "verdict": outcome.verdict,
                "prepass_pruned": list(outcome.prepass_pruned),
                "applicable_classes": list(outcome.applicable_classes),
            }
            for outcome in outcomes
        ],
    }
    return _sha256_obj(stable)


def results_document() -> Dict[str, object]:
    cases = corpus()
    valid, detail = validate_cases(cases)
    static = run_static_corpus(cases)
    entries = manifest_entries(cases)
    return {
        "schema": SCHEMA_VERSION,
        "content_hash": content_hash(entries, static.outcomes),
        "valid_cases": valid,
        "validation_detail": detail,
        "census": corpus_census(cases),
        "acceptance_policy": {
            POLICY_DIVERGENT: (
                "deterministic no-toolchain path must return CANDIDATE, and "
                "the optional live path may not return NO_DIVERGENCE_FOUND or NOT_COVERED"
            ),
            POLICY_SAFE: (
                "safe near-miss cases must return NO_DIVERGENCE_FOUND and carry "
                "the named abstract-interpretation pre-pass proof"
            ),
            POLICY_ABSTAIN: (
                "unsupported pairs or mismatched probes must return NOT_COVERED "
                "rather than silently routing to an unrelated oracle"
            ),
        },
        "static_verdicts": {
            "ok": static.ok,
            "n_cases": static.n_cases,
            "n_failures": len(static.failures),
            "n_breaches": len(static.breaches),
            "by_verdict": static.verdict_counts(),
            "by_policy": static.policy_counts(),
        },
        "cases": list(entries),
        "outcomes": [outcome.to_dict() for outcome in static.outcomes],
    }


def load_results(path: Path = RESULTS_PATH) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_results(path: Path = RESULTS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(results_document(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def check_results(path: Path = RESULTS_PATH) -> Tuple[bool, str]:
    if not path.exists():
        return False, f"{path} is missing"
    regenerated = json.dumps(results_document(), indent=2, sort_keys=True) + "\n"
    on_disk = path.read_text(encoding="utf-8")
    if on_disk != regenerated:
        return False, f"{path} does not match regenerated adversarial corpus"
    doc = json.loads(on_disk)
    if doc.get("schema") != SCHEMA_VERSION:
        return False, f"unexpected schema {doc.get('schema')!r}"
    if not doc.get("valid_cases"):
        return False, str(doc.get("validation_detail"))
    if not doc.get("static_verdicts", {}).get("ok"):
        return False, "static adversarial verdicts are not all accepted"
    return True, "OK"
