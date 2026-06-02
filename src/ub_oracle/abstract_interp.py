"""
Interval-domain abstract-interpretation pre-pass (100_STEPS step 78).

Before the (comparatively expensive) per-class Z3 search runs, this module
performs a cheap, **sound** abstract interpretation of the source-language
fragment over the classic *interval* domain and tries to *prove* that the
undefined-behavior region for a given divergence class is **unreachable** under
the unit's declared operating range. When it succeeds, the divergence is
discharged without ever invoking the solver — pruning "obviously-equivalent"
fragments to improve scale (no SMT call) and precision (a unit that declares a
safe operating range is not flagged for a divergence that range forbids).

Soundness contract (one-sided, like all of this tool):

* The analysis **over-approximates** the set of reachable inputs (an interval is
  a superset of the concrete value set for each variable).
* It returns :data:`PrePassVerdict.NO_UB_REACHABLE` for a class only when the
  UB-triggering region is *disjoint* from that over-approximation — i.e. **no**
  input in the declared range can trigger the class's UB. Pruning therefore can
  never hide a real divergence: if any input could diverge, the analysis must
  fall through (``MAYBE``) to the full oracle + ground-truth re-execution.
* It **never** asserts a divergence. Confirming divergence remains the exclusive
  job of the SMT search + real re-execution (``verify_unit``).

The domain is pair-agnostic: it reasons purely about the *source* (C) UB
conditions, so the same pre-pass prunes uniformly for C->Rust, C->Go and
C->Swift units.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


def signed_min(width: int) -> int:
    return -(1 << (width - 1))


def signed_max(width: int) -> int:
    return (1 << (width - 1)) - 1


@dataclass(frozen=True)
class Interval:
    """A closed integer interval ``[lo, hi]`` over the mathematical integers.

    ``lo > hi`` denotes bottom (the empty set). Transfer functions compute over
    *unbounded* integers so that an out-of-range result is observable as the
    interval escaping the machine-representable range (which is exactly the
    signed-overflow UB condition).
    """

    lo: int
    hi: int

    @property
    def is_bottom(self) -> bool:
        return self.lo > self.hi

    def contains(self, v: int) -> bool:
        return self.lo <= v <= self.hi

    def intersects(self, other: "Interval") -> bool:
        if self.is_bottom or other.is_bottom:
            return False
        return not (self.hi < other.lo or other.hi < self.lo)

    def meet(self, other: "Interval") -> "Interval":
        return Interval(max(self.lo, other.lo), min(self.hi, other.hi))

    def add_const(self, c: int) -> "Interval":
        if self.is_bottom:
            return self
        return Interval(self.lo + c, self.hi + c)

    def sub_const(self, c: int) -> "Interval":
        if self.is_bottom:
            return self
        return Interval(self.lo - c, self.hi - c)

    def subset_of(self, other: "Interval") -> bool:
        if self.is_bottom:
            return True
        return other.lo <= self.lo and self.hi <= other.hi

    def __str__(self) -> str:  # pragma: no cover - trivial
        if self.is_bottom:
            return "[bottom]"
        return f"[{self.lo}, {self.hi}]"


def repr_interval(width: int) -> Interval:
    """The interval of machine-representable signed values at ``width`` bits."""
    return Interval(signed_min(width), signed_max(width))


_BOTTOM = Interval(1, 0)


def parse_range(unit: Dict, key: str, width: int) -> Optional[Interval]:
    """Parse a declared ``[lo, hi]`` operating range, clamped to the type range.

    Returns ``None`` when the unit declares no range for ``key`` (meaning the
    variable is unconstrained over the whole signed type — the legacy default).
    A declared range is *intersected* with the representable interval so a unit
    can never claim its variable ranges outside its own type.
    """
    raw = unit.get(key)
    if raw is None:
        return None
    if (not isinstance(raw, (list, tuple))) or len(raw) != 2:
        raise ValueError(f"{key} must be a [lo, hi] pair, got {raw!r}")
    lo, hi = int(raw[0]), int(raw[1])
    if lo > hi:
        raise ValueError(f"{key} has lo > hi: {raw!r}")
    return Interval(lo, hi).meet(repr_interval(width))


def domain_for(unit: Dict, key: str, width: int) -> Interval:
    """The abstract domain for a variable: its declared range, else the full type."""
    declared = parse_range(unit, key, width)
    return declared if declared is not None else repr_interval(width)


class PrePassVerdict(Enum):
    #: proved that no input in the declared range triggers this class's UB.
    NO_UB_REACHABLE = "no_ub_reachable"
    #: the interval analysis cannot rule out the UB region — defer to the oracle.
    MAYBE = "maybe"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


@dataclass
class PrePassResult:
    divergence_class: str
    verdict: PrePassVerdict
    reason: str = ""

    @property
    def prunable(self) -> bool:
        return self.verdict is PrePassVerdict.NO_UB_REACHABLE


# --- per-class analyses ------------------------------------------------------
#
# Each returns NO_UB_REACHABLE only when the UB region is provably disjoint from
# the over-approximated reachable inputs. Anything it does not fully understand
# falls through to MAYBE so the oracle (and re-execution) make the real call.


def _analyze_signed_overflow(unit: Dict) -> PrePassResult:
    from .catalogue import SIGNED_OVERFLOW

    cls = SIGNED_OVERFLOW.key
    op = unit.get("op")
    width = unit.get("width", 32)
    if op not in ("add", "sub"):
        return PrePassResult(cls, PrePassVerdict.MAYBE, "unsupported op")
    c = int(unit["const"])
    x = domain_for(unit, "x_range", width)
    result = x.add_const(c) if op == "add" else x.sub_const(c)
    rep = repr_interval(width)
    if result.subset_of(rep):
        return PrePassResult(
            cls, PrePassVerdict.NO_UB_REACHABLE,
            f"x in {x}, so x {('+' if op == 'add' else '-')} {c} in {result} "
            f"subset of representable {rep}: signed overflow is unreachable")
    return PrePassResult(
        cls, PrePassVerdict.MAYBE,
        f"x {('+' if op == 'add' else '-')} {c} in {result} can escape {rep}")


def _analyze_shift(unit: Dict) -> PrePassResult:
    from .catalogue import SHIFT_OUT_OF_RANGE

    cls = SHIFT_OUT_OF_RANGE.key
    width = unit.get("width", 32)
    # the shift amount is the UB-bearing operand; default unconstrained.
    declared = parse_range(unit, "shift_range", 32)
    if declared is None:
        return PrePassResult(cls, PrePassVerdict.MAYBE,
                             "shift amount unconstrained")
    ub_region = Interval(width, signed_max(32))  # s >= width (and non-negative)
    neg = Interval(signed_min(32), -1)            # s < 0 is also UB
    if declared.intersects(ub_region) or declared.intersects(neg):
        return PrePassResult(cls, PrePassVerdict.MAYBE,
                             f"shift range {declared} can reach >= width {width} "
                             f"or be negative")
    return PrePassResult(
        cls, PrePassVerdict.NO_UB_REACHABLE,
        f"shift range {declared} stays in [0, {width - 1}]: out-of-range shift "
        f"is unreachable")


def _analyze_div_by_zero(unit: Dict) -> PrePassResult:
    from .catalogue import DIVISION_BY_ZERO

    cls = DIVISION_BY_ZERO.key
    width = unit.get("width", 32)
    b = domain_for(unit, "b_range", width)
    if b.contains(0):
        return PrePassResult(cls, PrePassVerdict.MAYBE,
                             f"divisor range {b} includes 0")
    return PrePassResult(cls, PrePassVerdict.NO_UB_REACHABLE,
                         f"divisor range {b} excludes 0: division-by-zero "
                         f"is unreachable")


def _analyze_intmin_div_neg1(unit: Dict) -> PrePassResult:
    from .catalogue import INT_MIN_DIV_NEG1

    cls = INT_MIN_DIV_NEG1.key
    width = unit.get("width", 32)
    a = domain_for(unit, "a_range", width)
    b = domain_for(unit, "b_range", width)
    # UB iff a == INT_MIN AND b == -1 (both must be reachable).
    if a.contains(signed_min(width)) and b.contains(-1):
        return PrePassResult(cls, PrePassVerdict.MAYBE,
                             f"a range {a} can be INT_MIN and b range {b} can be -1")
    reason = (f"INT_MIN ({signed_min(width)}) not in a range {a}"
              if not a.contains(signed_min(width))
              else f"-1 not in b range {b}")
    return PrePassResult(cls, PrePassVerdict.NO_UB_REACHABLE,
                         f"{reason}: signed-division overflow is unreachable")


def analyze_unit(unit: Dict) -> Dict[str, PrePassResult]:
    """Run every applicable per-class analysis and key results by class.

    A single unit can be in scope for several classes (e.g. a ``div`` unit is
    checked by both division-by-zero and INT_MIN/-1); each gets its own verdict.
    Classes the pre-pass does not model are simply absent from the result, so
    the caller defers to the oracle for them.
    """
    from .catalogue import (
        SIGNED_OVERFLOW, SHIFT_OUT_OF_RANGE, DIVISION_BY_ZERO, INT_MIN_DIV_NEG1,
    )

    kind = unit.get("kind")
    probe = unit.get("probe")
    out: Dict[str, PrePassResult] = {}

    def _wanted(cls_key: str) -> bool:
        return probe in (None, cls_key)

    if kind == "binop_const" and unit.get("signed", True) and _wanted(SIGNED_OVERFLOW.key):
        out[SIGNED_OVERFLOW.key] = _analyze_signed_overflow(unit)
    if kind == "shift" and _wanted(SHIFT_OUT_OF_RANGE.key):
        out[SHIFT_OUT_OF_RANGE.key] = _analyze_shift(unit)
    if kind in ("div", "rem"):
        if _wanted(DIVISION_BY_ZERO.key):
            out[DIVISION_BY_ZERO.key] = _analyze_div_by_zero(unit)
        if unit.get("signed", True) and _wanted(INT_MIN_DIV_NEG1.key):
            out[INT_MIN_DIV_NEG1.key] = _analyze_intmin_div_neg1(unit)
    return out


def prunable_classes(unit: Dict) -> Dict[str, PrePassResult]:
    """Subset of :func:`analyze_unit` whose verdict is ``NO_UB_REACHABLE``."""
    return {k: v for k, v in analyze_unit(unit).items() if v.prunable}
