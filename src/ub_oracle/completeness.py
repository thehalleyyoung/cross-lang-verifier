"""
Per-class completeness characterization (100_STEPS step 81).

Soundness (only ever claim a *confirmed* divergence) is the headline guarantee
of this tool, and it is necessarily one-sided: outside a covered class we say
nothing. But for the integer divergence classes the symbolic search is not just
sound — it is **complete on a precisely-stated fragment**. This module makes that
completeness claim *executable*: for each class it

1. defines the **decidable fragment** — the exact family of translation units on
   which we assert completeness (operator family, supported widths, and a finite
   integer operating range per operand);
2. enumerates a representative grid of units inside that fragment;
3. computes **ground truth by brute-force enumeration** of every input in the
   declared range against the concrete C undefined-behavior predicate (no solver,
   no formula — the actual per-point check); and
4. asserts the oracle's symbolic (Z3) search agrees *exactly*: it returns a
   ``DIVERGENT`` witness **iff** a divergence-triggering input exists in the
   range, and any witness it returns is in range and genuinely triggers the UB.

That two-sided agreement (∃-witness ⇔ oracle-finds-witness) over brute-forced
ground truth is the completeness evidence. The characterization is honest about
its limits: completeness is asserted only over the bounded ranges of the
fragment, and only for the integer classes whose UB condition is a decidable
predicate over the operands (signed overflow, out-of-range shift, division by
zero, INT_MIN/-1). Memory-shape and floating-point classes are explicitly
*outside* the completeness fragment and are documented as such.

The fragment is pair-agnostic: the C->Rust, C->Go and C->Swift oracles share the
same symbolic search and only differ in the emitted target program, so a
completeness result established on the search transfers across pairs (the
``check_pair_completeness`` helper re-runs the grid for any registered pair).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from .catalogue import (
    SIGNED_OVERFLOW,
    SHIFT_OUT_OF_RANGE,
    DIVISION_BY_ZERO,
    INT_MIN_DIV_NEG1,
)
from .plugin import OracleVerdict, get_oracle_for

_SUPPORTED_WIDTHS = (32, 64)


def _smin(width: int) -> int:
    return -(1 << (width - 1))


def _smax(width: int) -> int:
    return (1 << (width - 1)) - 1


# --- concrete per-point UB predicates (ground truth) -------------------------
#
# Each returns True iff the *concrete* input triggers the class's C undefined
# behavior. These are deliberately the simplest possible direct encodings of the
# C standard rule so they are obviously correct and independent of the solver.


def _ub_signed_overflow(op: str, c: int, width: int, x: int) -> bool:
    r = x + c if op == "add" else x - c
    return not (_smin(width) <= r <= _smax(width))


def _ub_shift(width: int, s: int) -> bool:
    return s < 0 or s >= width


def _ub_div_by_zero(b: int) -> bool:
    return b == 0


def _ub_intmin_div_neg1(width: int, a: int, b: int) -> bool:
    return b != 0 and a == _smin(width) and b == -1


@dataclass(frozen=True)
class FragmentPoint:
    """One unit in a class's completeness fragment + its brute-forced truth."""

    unit: Dict
    has_divergence: bool
    #: a concrete witnessing input tuple (the first found), if any.
    witness: Optional[Tuple[int, ...]] = None


@dataclass(frozen=True)
class Fragment:
    """The decidable fragment for one divergence class."""

    divergence_class: str
    description: str
    #: yields every (unit, ground-truth) point in the (bounded) fragment grid.
    enumerate_points: Callable[[], List[FragmentPoint]]


# --- fragment generators -----------------------------------------------------
#
# Ranges are deliberately small *windows positioned around the interesting
# boundaries* (INT_MAX, 0, INT_MIN, the width) so brute force is cheap yet covers
# both the "divergence exists" and "no divergence" sides of every boundary.


def _enum_signed_overflow() -> List[FragmentPoint]:
    pts: List[FragmentPoint] = []
    for width in _SUPPORTED_WIDTHS:
        smax, smin = _smax(width), _smin(width)
        for op in ("add", "sub"):
            for c in (0, 1, 2, 7, 100):
                # windows: safely interior; straddling the high boundary; the
                # low boundary; and a degenerate single point.
                windows = [
                    (0, 10),
                    (smax - 5, smax),
                    (smin, smin + 5),
                    (smax, smax),
                    (smin, smin),
                ]
                for lo, hi in windows:
                    unit = {
                        "kind": "binop_const", "op": op, "const": c,
                        "width": width, "signed": True, "x_range": [lo, hi],
                    }
                    witness = None
                    for x in range(lo, hi + 1):
                        if _ub_signed_overflow(op, c, width, x):
                            witness = (x,)
                            break
                    pts.append(FragmentPoint(unit, witness is not None, witness))
    return pts


def _enum_shift() -> List[FragmentPoint]:
    pts: List[FragmentPoint] = []
    for width in _SUPPORTED_WIDTHS:
        # The characterized fragment is NON-NEGATIVE shift amounts: the oracle's
        # witness search + re-execution model the unsigned shift-amount regime
        # (the emitted Rust harness takes the amount as ``u32``), so negative
        # shift amounts — also C-UB — are explicitly OUTSIDE this fragment.
        windows = [
            (0, width - 1),      # all in-range  -> no divergence
            (0, width),          # straddles the width boundary -> divergence
            (width, width + 4),  # all over-shift -> divergence
            (1, width - 1),      # interior, all safe -> no divergence
            (width - 1, width),  # the exact boundary pair -> divergence
        ]
        for lo, hi in windows:
            unit = {"kind": "shift", "width": width, "shift_range": [lo, hi]}
            witness = None
            for s in range(lo, hi + 1):
                if _ub_shift(width, s):
                    witness = (s,)
                    break
            pts.append(FragmentPoint(unit, witness is not None, witness))
    return pts


def _enum_div_by_zero() -> List[FragmentPoint]:
    pts: List[FragmentPoint] = []
    for width in _SUPPORTED_WIDTHS:
        for kind in ("div", "rem"):
            windows = [
                (1, 9),     # excludes 0 -> no divergence
                (-9, -1),   # excludes 0 -> no divergence
                (-3, 3),    # includes 0 -> divergence
                (0, 0),     # exactly 0 -> divergence
            ]
            for lo, hi in windows:
                unit = {"kind": kind, "width": width, "signed": True,
                        "b_range": [lo, hi], "a_range": [1, 5]}
                witness = None
                for b in range(lo, hi + 1):
                    if _ub_div_by_zero(b):
                        witness = (1, b)
                        break
                pts.append(FragmentPoint(unit, witness is not None, witness))
    return pts


def _enum_intmin_div_neg1() -> List[FragmentPoint]:
    pts: List[FragmentPoint] = []
    for width in _SUPPORTED_WIDTHS:
        smin = _smin(width)
        # a small enumerable b range; a_range either includes or excludes INT_MIN.
        configs = [
            ([smin, smin + 2], [-2, -1], True),    # a can be INT_MIN, b can be -1
            ([smin + 1, smin + 3], [-2, -1], False),  # a excludes INT_MIN
            ([smin, smin + 2], [1, 3], False),     # b excludes -1 (and 0)
            ([smin, smin], [-1, -1], True),        # exact overflow point only
        ]
        for a_rng, b_rng, _expect in configs:
            unit = {"kind": "div", "width": width, "signed": True,
                    "a_range": a_rng, "b_range": b_rng, "probe": INT_MIN_DIV_NEG1.key}
            witness = None
            for a in range(a_rng[0], a_rng[1] + 1):
                for b in range(b_rng[0], b_rng[1] + 1):
                    if _ub_intmin_div_neg1(width, a, b):
                        witness = (a, b)
                        break
                if witness:
                    break
            pts.append(FragmentPoint(unit, witness is not None, witness))
    return pts


#: the registry of completeness fragments, keyed by divergence class.
FRAGMENTS: Dict[str, Fragment] = {
    SIGNED_OVERFLOW.key: Fragment(
        SIGNED_OVERFLOW.key,
        "f(x) = x {+,-} c over signed width in {32,64}, x ranging over a finite "
        "interval: complete — a witness is reported iff some x in the interval "
        "makes the exact-integer result leave [INT_MIN, INT_MAX].",
        _enum_signed_overflow),
    SHIFT_OUT_OF_RANGE.key: Fragment(
        SHIFT_OUT_OF_RANGE.key,
        "f(x,s) = x << s over signed width in {32,64}, s ranging over a finite "
        "NON-NEGATIVE interval: complete — a witness is reported iff the interval "
        "contains a shift amount >= width. (Negative shift amounts, also C-UB, "
        "are outside this fragment: the search/re-exec model the unsigned regime.)",
        _enum_shift),
    DIVISION_BY_ZERO.key: Fragment(
        DIVISION_BY_ZERO.key,
        "f(a,b) = a {/,%} b over signed width in {32,64}, b ranging over a finite "
        "interval: complete — a witness is reported iff 0 is in the interval.",
        _enum_div_by_zero),
    INT_MIN_DIV_NEG1.key: Fragment(
        INT_MIN_DIV_NEG1.key,
        "f(a,b) = a {/,%} b over signed width in {32,64}, a and b ranging over "
        "finite intervals: complete — a witness is reported iff INT_MIN is in a's "
        "interval and -1 is in b's interval.",
        _enum_intmin_div_neg1),
}

#: classes explicitly OUTSIDE the completeness fragment (documented honestly).
OUT_OF_FRAGMENT = (
    "array_oob",        # length/index modeled, but heap shape is not exhaustive
    "strict_aliasing",  # type-punning UB is detected, not enumerated to completeness
    "fp_contraction",   # FP search is sound but not characterized complete here
)


@dataclass
class ClassCompleteness:
    divergence_class: str
    source_lang: str
    target_lang: str
    n_points: int
    n_with_divergence: int
    #: points where the oracle's verdict disagreed with brute-forced ground truth.
    mismatches: List[str] = field(default_factory=list)
    #: points where a reported witness was out-of-range or didn't trigger UB.
    bad_witnesses: List[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.mismatches and not self.bad_witnesses


def _label(point: FragmentPoint) -> str:
    u = point.unit
    keys = [k for k in ("op", "const", "width", "x_range", "shift_range",
                        "a_range", "b_range", "kind") if k in u]
    return ", ".join(f"{k}={u[k]}" for k in keys)


def check_class_completeness(divergence_class: str, *, source_lang: str = "c",
                             target_lang: str = "rust") -> ClassCompleteness:
    """Run the brute-force-vs-oracle agreement check for one class.

    For every point in the fragment grid: the oracle must report ``DIVERGENT``
    exactly when brute-force enumeration found a triggering input, and any
    reported witness must lie in the declared range and actually trigger the UB.
    """
    frag = FRAGMENTS[divergence_class]
    oracle = get_oracle_for(divergence_class, source_lang, target_lang)
    points = frag.enumerate_points()

    res = ClassCompleteness(divergence_class, source_lang, target_lang,
                            n_points=len(points),
                            n_with_divergence=sum(p.has_divergence for p in points))

    for p in points:
        out = oracle.find_divergence(p.unit)
        found = out.verdict is OracleVerdict.DIVERGENT
        if found != p.has_divergence:
            res.mismatches.append(
                f"{_label(p)}: oracle found={found} but ground-truth "
                f"has_divergence={p.has_divergence}")
            continue
        if found:
            if not _witness_is_valid(divergence_class, p.unit, out):
                res.bad_witnesses.append(
                    f"{_label(p)}: reported witness {_witness_inputs(out)} is "
                    f"out-of-range or does not trigger UB")
    return res


def _witness_inputs(out) -> Dict:
    return out.counterexample.inputs if out.counterexample else {}


def _in_range(unit: Dict, key: str, v: int) -> bool:
    rng = unit.get(key)
    if rng is None:
        return True
    return rng[0] <= v <= rng[1]


def _witness_is_valid(divergence_class: str, unit: Dict, out) -> bool:
    """Re-check (without the solver) that the reported witness is real."""
    ce = out.counterexample
    if ce is None:
        return False
    inp = ce.inputs
    width = unit.get("width", 32)
    if divergence_class == SIGNED_OVERFLOW.key:
        x = inp.get(unit.get("var", "x"))
        if x is None or not _in_range(unit, "x_range", x):
            return False
        return _ub_signed_overflow(unit["op"], int(unit["const"]), width, x)
    if divergence_class == SHIFT_OUT_OF_RANGE.key:
        s = inp.get(unit.get("shift_var", "s"))
        if s is None or not _in_range(unit, "shift_range", s):
            return False
        return _ub_shift(width, s)
    if divergence_class == DIVISION_BY_ZERO.key:
        b = inp.get(unit.get("b", "b"))
        if b is None or not _in_range(unit, "b_range", b):
            return False
        return _ub_div_by_zero(b)
    if divergence_class == INT_MIN_DIV_NEG1.key:
        a = inp.get(unit.get("a", "a"))
        b = inp.get(unit.get("b", "b"))
        if a is None or b is None:
            return False
        if not (_in_range(unit, "a_range", a) and _in_range(unit, "b_range", b)):
            return False
        return _ub_intmin_div_neg1(width, a, b)
    return False  # pragma: no cover - guarded by FRAGMENTS keys


def check_all_completeness(*, source_lang: str = "c",
                           target_lang: str = "rust") -> List[ClassCompleteness]:
    """Run the completeness check for every characterized class on one pair."""
    return [check_class_completeness(k, source_lang=source_lang,
                                     target_lang=target_lang)
            for k in FRAGMENTS]


def check_pair_completeness(pairs: Optional[Iterable[Tuple[str, str]]] = None
                            ) -> Dict[Tuple[str, str], List[ClassCompleteness]]:
    """Run completeness for each registered language pair that has the classes.

    Pairs default to every registered pair; classes a pair does not implement
    are skipped (so a pair-pack that lacks, say, INT_MIN/-1 is not a failure).
    """
    from .plugin import language_pairs, oracles_for

    out: Dict[Tuple[str, str], List[ClassCompleteness]] = {}
    selected = list(pairs) if pairs is not None else language_pairs()
    for (src, tgt) in selected:
        results: List[ClassCompleteness] = []
        for k in FRAGMENTS:
            if not oracles_for(src, tgt, k):
                continue
            results.append(check_class_completeness(k, source_lang=src,
                                                    target_lang=tgt))
        out[(src, tgt)] = results
    return out
