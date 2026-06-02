"""
Per-class precision / recall harness (100_STEPS step 23).

A divergence oracle is only credible if it (a) *fires on the divergences it
claims to cover* (recall) and (b) *does not fire spuriously* on inputs that are
either out of its scope or genuinely non-divergent (precision). This module
pins down a labelled benchmark of translation units and measures both, per
divergence class, at two levels of rigour:

* **symbolic** — purely the oracle's solver-level decision (``find_divergence``
  returns ``DIVERGENT`` or not). Deterministic, needs no toolchain, and is wired
  into the reproducible ``results.json``.
* **confirmed** — the same decision *after* ground-truth re-execution of real
  compiled C and Rust. Needs the toolchain; exercised by the test-suite.

Each oracle is scored against the *entire* labelled set, so firing on another
class's case (or on a non-divergent case) counts against its precision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .plugin import REGISTRY, OracleVerdict
from .reexec import ReexecHarness


@dataclass(frozen=True)
class LabeledCase:
    name: str
    unit: Dict
    #: the divergence class this case *is* an instance of, or None if it is a
    #: deliberately non-divergent ("defined") case.
    truth_class: Optional[str]

    @property
    def is_divergent(self) -> bool:
        return self.truth_class is not None


# ── the labelled benchmark ───────────────────────────────────────────────────
# Positives: a genuine divergence of the named class lives in the unit.
POSITIVE_CASES: List[LabeledCase] = [
    LabeledCase("ovf_add1_w32",
                {"kind": "binop_const", "op": "add", "const": 1, "width": 32,
                 "var": "x", "signed": True, "probe": "signed_overflow"},
                "signed_overflow"),
    LabeledCase("ovf_sub1_w32",
                {"kind": "binop_const", "op": "sub", "const": 1, "width": 32,
                 "var": "x", "signed": True, "probe": "signed_overflow"},
                "signed_overflow"),
    LabeledCase("ovf_add1_w64",
                {"kind": "binop_const", "op": "add", "const": 1, "width": 64,
                 "var": "x", "signed": True, "probe": "signed_overflow"},
                "signed_overflow"),
    LabeledCase("shift_w32",
                {"kind": "shift", "width": 32, "probe": "shift_oob"}, "shift_oob"),
    LabeledCase("shift_w64",
                {"kind": "shift", "width": 64, "probe": "shift_oob"}, "shift_oob"),
    LabeledCase("divzero_w32",
                {"kind": "div", "width": 32, "probe": "div_by_zero"}, "div_by_zero"),
    LabeledCase("remzero_w32",
                {"kind": "rem", "width": 32, "probe": "div_by_zero"}, "div_by_zero"),
    LabeledCase("intmin_div_w32",
                {"kind": "div", "width": 32, "signed": True,
                 "probe": "intmin_div_neg1"}, "intmin_div_neg1"),
    LabeledCase("oob_len4",
                {"kind": "array_index", "length": 4, "probe": "array_oob"},
                "array_oob"),
    LabeledCase("alias_pun",
                {"kind": "type_pun", "probe": "strict_aliasing"},
                "strict_aliasing"),
    LabeledCase("fp_fma",
                {"kind": "fp_fma", "probe": "fp_contraction"},
                "fp_contraction"),
    LabeledCase("uninit_struct_b",
                {"kind": "uninit_read",
                 "storage": {"kind": "struct", "fields": ["a", "b"]},
                 "writes": [{"slot": "a"}], "read": "b",
                 "probe": "uninit_read"},
                "uninit_read"),
    LabeledCase("vla_neg_w32",
                {"kind": "vla", "width": 32, "var": "n", "probe": "vla_bound"},
                "vla_bound"),
    LabeledCase("fcast_oob_w32",
                {"kind": "float_cast", "width": 32, "var": "x",
                 "probe": "float_cast_overflow"},
                "float_cast_overflow"),
    LabeledCase("fast_math_reassoc",
                {"kind": "fp_reassoc", "probe": "fast_math_reassoc"},
                "fast_math_reassoc"),
    LabeledCase("restrict_alias",
                {"kind": "restrict_pair", "probe": "restrict_violation"},
                "restrict_violation"),
    LabeledCase("pointer_provenance",
                {"kind": "pointer_offset", "width": 32, "var": "n",
                 "probe": "pointer_provenance"},
                "pointer_provenance"),
]

# Negatives: applicable-looking units with NO divergence of that class, plus
# out-of-scope units. A correct oracle declines all of these.
NEGATIVE_CASES: List[LabeledCase] = [
    # add of 0 never signed-overflows: applicable to the overflow oracle, but no
    # witness exists -> must return NO_DIVERGENCE_FOUND.
    LabeledCase("noovf_add0_w32",
                {"kind": "binop_const", "op": "add", "const": 0, "width": 32,
                 "var": "x", "signed": True, "probe": "signed_overflow"}, None),
    # an unsigned unit: the signed-overflow oracle must not apply.
    LabeledCase("unsigned_add1_w32",
                {"kind": "binop_const", "op": "add", "const": 1, "width": 32,
                 "var": "x", "signed": False}, None),
    # an opaque unit no integer oracle understands.
    LabeledCase("opaque_unit", {"kind": "string_concat", "width": 32}, None),
    # a fully-initialized read: the uninit_read oracle must decline it.
    LabeledCase("uninit_all_written",
                {"kind": "uninit_read",
                 "storage": {"kind": "array", "length": 2},
                 "writes": [{"slot": 0}, {"slot": 1}], "read": 1,
                 "probe": "uninit_read"}, None),
]

ALL_CASES: List[LabeledCase] = POSITIVE_CASES + NEGATIVE_CASES


# ── scoring ──────────────────────────────────────────────────────────────────
@dataclass
class ClassScore:
    divergence_class: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return 1.0 if denom == 0 else self.tp / denom

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return 1.0 if denom == 0 else self.tp / denom

    def as_dict(self) -> Dict:
        return {
            "divergence_class": self.divergence_class,
            "tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn,
            "precision": self.precision, "recall": self.recall,
        }


def _predict_symbolic(oracle, case: LabeledCase) -> bool:
    """Whether ``oracle`` claims a divergence on ``case`` at the solver level."""
    if not oracle.applies_to(case.unit):
        return False
    return oracle.find_divergence(case.unit).verdict is OracleVerdict.DIVERGENT


def _predict_confirmed(oracle, case: LabeledCase, harness: ReexecHarness) -> bool:
    """Whether ``oracle`` claims a divergence that *re-execution confirms*."""
    if not oracle.applies_to(case.unit):
        return False
    res = oracle.find_divergence(case.unit)
    if res.verdict is not OracleVerdict.DIVERGENT:
        return False
    res = oracle.confirm(res, harness)
    return bool(res.reexec and res.reexec.available and res.reexec.confirmed)


def _score(predict) -> Dict:
    scores: Dict[str, ClassScore] = {
        key: ClassScore(key) for key in sorted(REGISTRY)
    }
    for key, oracle in REGISTRY.items():
        sc = scores[key]
        for case in ALL_CASES:
            predicted = predict(oracle, case)
            truth = case.truth_class == key
            if predicted and truth:
                sc.tp += 1
            elif predicted and not truth:
                sc.fp += 1
            elif (not predicted) and truth:
                sc.fn += 1
            else:
                sc.tn += 1
    overall = ClassScore("__overall__")
    for sc in scores.values():
        overall.tp += sc.tp
        overall.fp += sc.fp
        overall.fn += sc.fn
        overall.tn += sc.tn
    return {
        "per_class": {k: v.as_dict() for k, v in sorted(scores.items())},
        "overall": overall.as_dict(),
        "num_cases": len(ALL_CASES),
        "num_positive": len(POSITIVE_CASES),
        "num_negative": len(NEGATIVE_CASES),
    }


def evaluate_symbolic() -> Dict:
    """Deterministic, toolchain-free precision/recall over the labelled set."""
    return _score(_predict_symbolic)


def evaluate_confirmed(harness: Optional[ReexecHarness] = None) -> Dict:
    """Precision/recall where every positive prediction is re-executed."""
    harness = harness or ReexecHarness()
    return _score(lambda o, c: _predict_confirmed(o, c, harness))
