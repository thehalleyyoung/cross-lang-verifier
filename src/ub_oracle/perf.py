"""
Performance / scalability measurement (100_STEPS step 50).

"Does it scale?" is the question every reviewer asks.  This module measures the
**real** cost of the symbolic divergence search — the Z3 solve inside each
oracle's ``find_divergence`` — and characterises how it grows, so the answer is
backed by numbers rather than hand-waving.

Three complementary measurements, all timing *actual* work (no simulated data):

* :func:`class_pair_profile` — time every registered oracle on its canonical unit
  across **every language pair**, giving a robustness table (which classes/pairs
  are cheap, which are not) with the deterministic verdict alongside each timing.

* :func:`width_scaling_curve` — for the integer oracles that support multiple bit
  widths, time the search at each width to expose any width sensitivity.

* :func:`smt_scaling_curve` — drive the *underlying* bitvector-overflow solving
  the oracles rely on across a wide range of bit widths (8 … 512) and fit the
  growth, returning a ``growth_ratio`` and a ``pathological`` flag.  This is the
  headline scalability curve: it shows the SMT backbone stays well-behaved.

Timings are wall-clock medians over repeats and are therefore environment
dependent; the *grid* (classes, pairs, widths, verdicts) is deterministic.  The
companion experiment driver writes the deterministic grid to a version-controlled
JSON and the measured timings to a gitignored artifact.
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import z3

from . import oracles as _oracles  # noqa: F401  (registers all pairs)
from .plugin import ALL_ORACLES, OracleVerdict
from .regression_matrix import CANONICAL_UNITS, canonical_unit_for, _sorted_oracles


def z3_version() -> str:
    """Human-readable Z3 version used by environment-dependent timing runs."""
    return getattr(z3, "get_version_string", lambda: "unknown")()


def _time_samples(fn: Callable[[], object], repeats: int) -> Tuple[List[float], object]:
    """Wall-clock samples for ``fn`` over ``repeats`` real executions (>=1)."""
    repeats = max(1, repeats)
    samples: List[float] = []
    last = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        last = fn()
        samples.append(time.perf_counter() - t0)
    return samples, last


def _median_time(fn: Callable[[], object], repeats: int) -> Tuple[float, object]:
    """Median wall-clock seconds of ``fn`` over ``repeats`` runs (>=1).

    Returns ``(median_seconds, last_return_value)``. Uses ``perf_counter`` and
    discards the return values except the last so the measured work is real.
    """
    samples, last = _time_samples(fn, repeats)
    return statistics.median(samples), last


def percentile(samples: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile for small perf samples.

    The perf gate intentionally uses a conservative nearest-rank statistic rather
    than interpolation: one slow sample should count against the p95 budget.
    """
    if not samples:
        return 0.0
    if pct <= 0:
        return min(samples)
    if pct >= 100:
        return max(samples)
    ordered = sorted(samples)
    idx = max(0, math.ceil((pct / 100.0) * len(ordered)) - 1)
    return ordered[min(idx, len(ordered) - 1)]


@dataclass
class TimingRow:
    label: str
    seconds: float
    verdict: str
    extra: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        d = {"label": self.label, "seconds": self.seconds, "verdict": self.verdict}
        d.update(self.extra)
        return d


@dataclass(frozen=True)
class LatencyBudget:
    """P50/P95 latency budget for one measured perf item."""

    p50_seconds: float
    p95_seconds: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "p50_seconds": self.p50_seconds,
            "p95_seconds": self.p95_seconds,
        }


@dataclass(frozen=True)
class PerfBudgets:
    """Latency budgets for the CI perf gate.

    Most oracle searches are sub-millisecond to low-millisecond bit-vector or
    dataflow checks, so they use a tight default. FP-contraction is intentionally
    granted a larger class-specific budget because it drives Z3's IEEE-754 theory
    and is the known, real slow path the paper calls out.
    """

    default: LatencyBudget = LatencyBudget(0.25, 0.50)
    by_class: Mapping[str, LatencyBudget] = field(default_factory=lambda: {
        "fp_contraction": LatencyBudget(15.0, 20.0),
        "fast_math_reassoc": LatencyBudget(0.75, 1.0),
        "strict_aliasing": LatencyBudget(0.25, 0.50),
        "bitfield_layout": LatencyBudget(0.25, 0.50),
    })
    width_scaling: LatencyBudget = LatencyBudget(0.25, 0.50)
    smt_scaling: LatencyBudget = LatencyBudget(0.05, 0.10)

    def for_class(self, divergence_class: str) -> LatencyBudget:
        return self.by_class.get(divergence_class, self.default)

    def to_dict(self) -> Dict[str, object]:
        return {
            "default": self.default.to_dict(),
            "by_class": {k: v.to_dict() for k, v in sorted(self.by_class.items())},
            "width_scaling": self.width_scaling.to_dict(),
            "smt_scaling": self.smt_scaling.to_dict(),
        }


DEFAULT_BUDGETS = PerfBudgets()


@dataclass
class LatencyBudgetRow:
    label: str
    samples: List[float]
    verdict: str
    budget: LatencyBudget
    extra: Dict[str, object] = field(default_factory=dict)

    @property
    def p50_seconds(self) -> float:
        return percentile(self.samples, 50)

    @property
    def p95_seconds(self) -> float:
        return percentile(self.samples, 95)

    @property
    def max_seconds(self) -> float:
        return max(self.samples) if self.samples else 0.0

    @property
    def ok(self) -> bool:
        return (
            self.p50_seconds <= self.budget.p50_seconds
            and self.p95_seconds <= self.budget.p95_seconds
        )

    def failures(self) -> List[str]:
        out = []
        if self.p50_seconds > self.budget.p50_seconds:
            out.append(
                f"{self.label} p50 {self.p50_seconds:.6f}s "
                f"> budget {self.budget.p50_seconds:.6f}s"
            )
        if self.p95_seconds > self.budget.p95_seconds:
            out.append(
                f"{self.label} p95 {self.p95_seconds:.6f}s "
                f"> budget {self.budget.p95_seconds:.6f}s"
            )
        return out

    def to_dict(self) -> Dict:
        d = {
            "label": self.label,
            "samples": self.samples,
            "p50_seconds": self.p50_seconds,
            "p95_seconds": self.p95_seconds,
            "max_seconds": self.max_seconds,
            "budget_p50_seconds": self.budget.p50_seconds,
            "budget_p95_seconds": self.budget.p95_seconds,
            "ok": self.ok,
            "verdict": self.verdict,
        }
        d.update(self.extra)
        return d


def _measure_latency_row(label: str,
                         fn: Callable[[], object],
                         repeats: int,
                         budget: LatencyBudget,
                         extra: Optional[Dict[str, object]] = None
                         ) -> LatencyBudgetRow:
    samples, res = _time_samples(fn, repeats)
    verdict = getattr(getattr(res, "verdict", None), "value", str(res))
    return LatencyBudgetRow(
        label=label,
        samples=samples,
        verdict=verdict,
        budget=budget,
        extra=extra or {},
    )


def class_pair_profile(repeats: int = 5) -> List[TimingRow]:
    """Time every oracle's canonical search, across every language pair."""
    rows: List[TimingRow] = []
    for oracle in _sorted_oracles():
        if oracle.divergence_class not in CANONICAL_UNITS:
            continue
        unit = canonical_unit_for(oracle)
        secs, res = _median_time(lambda: oracle.find_divergence(unit), repeats)
        rows.append(TimingRow(
            label=f"{oracle.source_lang}->{oracle.target_lang}:"
                  f"{oracle.divergence_class}",
            seconds=secs,
            verdict=res.verdict.value,
            extra={"source_lang": oracle.source_lang,
                   "target_lang": oracle.target_lang,
                   "divergence_class": oracle.divergence_class},
        ))
    return rows


#: integer oracles whose canonical unit accepts a ``width`` field.
_WIDTH_SCALABLE_CLASSES = ("signed_overflow", "shift_oob", "div_by_zero",
                           "intmin_div_neg1")
_SUPPORTED_WIDTHS = (32, 64)


def width_scaling_curve(repeats: int = 5,
                        widths: Tuple[int, ...] = _SUPPORTED_WIDTHS
                        ) -> List[TimingRow]:
    """Time the anchor integer searches at each supported bit width."""
    rows: List[TimingRow] = []
    by_class = {o.divergence_class: o for o in ALL_ORACLES
                if o.source_lang == "c" and o.target_lang == "rust"}
    for cls in _WIDTH_SCALABLE_CLASSES:
        oracle = by_class.get(cls)
        if oracle is None or cls not in CANONICAL_UNITS:
            continue
        for w in widths:
            unit = dict(CANONICAL_UNITS[cls], width=w,
                        source_lang="c", target_lang="rust")
            secs, res = _median_time(lambda: oracle.find_divergence(unit), repeats)
            rows.append(TimingRow(
                label=f"{cls}@{w}", seconds=secs, verdict=res.verdict.value,
                extra={"divergence_class": cls, "width": w}))
    return rows


def _solve_overflow_at_width(width: int) -> str:
    """Solve the canonical 'find a signed-add overflow' constraint at ``width``.

    This is exactly the shape of constraint the signed-overflow oracle issues —
    a pure bitvector-overflow query — scaled to an arbitrary width so we can
    characterise the SMT backbone's growth independent of source generation.
    Returns the Z3 result string (``"sat"``/``"unsat"``/``"unknown"``).
    """
    x = z3.BitVec("x", width)
    c = z3.BitVecVal(1, width)
    overflows = z3.Not(z3.And(z3.BVAddNoOverflow(x, c, True),
                              z3.BVAddNoUnderflow(x, c)))
    s = z3.Solver()
    s.add(overflows)
    return str(s.check())


@dataclass
class ScalingCurve:
    sizes: List[int]
    seconds: List[float]
    verdicts: List[str]
    growth_ratio: float
    per_size_growth: List[float]
    pathological: bool
    threshold: float

    def to_dict(self) -> Dict:
        return {
            "sizes": self.sizes,
            "seconds": self.seconds,
            "verdicts": self.verdicts,
            "growth_ratio": self.growth_ratio,
            "per_size_growth": self.per_size_growth,
            "pathological": self.pathological,
            "threshold": self.threshold,
        }


def smt_scaling_curve(sizes: Tuple[int, ...] = (8, 16, 32, 64, 128, 256, 512),
                      repeats: int = 5,
                      pathology_threshold: float = 8.0) -> ScalingCurve:
    """Measure SMT solve time vs. bit width and characterise the growth.

    ``growth_ratio`` is the per-step time ratio when the problem size *doubles*,
    geometric-mean averaged over the doublings.  A value near 1 means flat
    (excellent); the search is flagged ``pathological`` only if this exceeds
    ``pathology_threshold`` (i.e. worse than ~cubic per doubling), which would
    indicate a real blow-up worth fixing.
    """
    sizes = tuple(sizes)
    seconds: List[float] = []
    verdicts: List[str] = []
    for w in sizes:
        secs, res = _median_time(lambda: _solve_overflow_at_width(w), repeats)
        seconds.append(secs)
        verdicts.append(str(res))

    # per-doubling growth ratios, guarding against zero/sub-resolution timings.
    eps = 1e-6
    per_size_growth: List[float] = []
    doubling_ratios: List[float] = []
    for i in range(1, len(sizes)):
        prev, cur = max(seconds[i - 1], eps), max(seconds[i], eps)
        ratio = cur / prev
        per_size_growth.append(ratio)
        if sizes[i] == 2 * sizes[i - 1]:
            doubling_ratios.append(ratio)

    if doubling_ratios:
        # geometric mean of the doubling ratios.
        prod = 1.0
        for r in doubling_ratios:
            prod *= r
        growth_ratio = prod ** (1.0 / len(doubling_ratios))
    else:
        growth_ratio = 1.0

    return ScalingCurve(
        sizes=list(sizes), seconds=seconds, verdicts=verdicts,
        growth_ratio=growth_ratio, per_size_growth=per_size_growth,
        pathological=growth_ratio > pathology_threshold,
        threshold=pathology_threshold,
    )


def latency_budget_report(repeats: int = 3,
                          budgets: PerfBudgets = DEFAULT_BUDGETS,
                          smt_sizes: Tuple[int, ...] = (8, 16, 32, 64, 128, 256, 512)
                          ) -> Dict[str, object]:
    """Measure real perf items and enforce p50/p95 latency budgets.

    This is the Step-145 CI gate: every registered class/pair search, each
    supported integer-width search, and the SMT backbone scaling probes are timed
    on real code paths. A row fails if either its nearest-rank p50 or p95 exceeds
    the declared budget. The default budgets are deliberately generous enough for
    ordinary CI hosts yet tight enough to catch accidental quadratic/exponential
    regressions outside the known FP-theory slow path.
    """
    repeats = max(1, repeats)
    profile: List[LatencyBudgetRow] = []
    for oracle in _sorted_oracles():
        if oracle.divergence_class not in CANONICAL_UNITS:
            continue
        unit = canonical_unit_for(oracle)
        label = f"{oracle.source_lang}->{oracle.target_lang}:{oracle.divergence_class}"
        profile.append(_measure_latency_row(
            label,
            lambda oracle=oracle, unit=unit: oracle.find_divergence(unit),
            repeats,
            budgets.for_class(oracle.divergence_class),
            {
                "source_lang": oracle.source_lang,
                "target_lang": oracle.target_lang,
                "divergence_class": oracle.divergence_class,
            },
        ))

    width_rows: List[LatencyBudgetRow] = []
    by_class = {o.divergence_class: o for o in ALL_ORACLES
                if o.source_lang == "c" and o.target_lang == "rust"}
    for cls in _WIDTH_SCALABLE_CLASSES:
        oracle = by_class.get(cls)
        if oracle is None or cls not in CANONICAL_UNITS:
            continue
        for w in _SUPPORTED_WIDTHS:
            unit = dict(CANONICAL_UNITS[cls], width=w,
                        source_lang="c", target_lang="rust")
            width_rows.append(_measure_latency_row(
                f"{cls}@{w}",
                lambda oracle=oracle, unit=unit: oracle.find_divergence(unit),
                repeats,
                budgets.width_scaling,
                {"divergence_class": cls, "width": w},
            ))

    smt_rows: List[LatencyBudgetRow] = []
    for w in smt_sizes:
        smt_rows.append(_measure_latency_row(
            f"smt_overflow@{w}",
            lambda w=w: _solve_overflow_at_width(w),
            repeats,
            budgets.smt_scaling,
            {"width": w},
        ))

    sections = {
        "class_pair_profile": [r.to_dict() for r in profile],
        "width_scaling": [r.to_dict() for r in width_rows],
        "smt_scaling": [r.to_dict() for r in smt_rows],
    }
    failures = [msg for rows in (profile, width_rows, smt_rows)
                for row in rows for msg in row.failures()]
    return {
        "schema": "perf-budget-report/v1",
        "repeats": repeats,
        "z3_version": z3_version(),
        "budgets": budgets.to_dict(),
        **sections,
        "ok": not failures,
        "failures": failures,
    }


def deterministic_grid() -> Dict:
    """The size/class/pair grid + verdicts, with NO timings (reproducible).

    This is the version-controllable evidence that the perf study covers the full
    matrix; the measured seconds live in a separate, environment-dependent file.
    """
    profile = []
    for oracle in _sorted_oracles():
        if oracle.divergence_class not in CANONICAL_UNITS:
            continue
        unit = canonical_unit_for(oracle)
        res = oracle.find_divergence(unit)
        profile.append({
            "label": f"{oracle.source_lang}->{oracle.target_lang}:"
                     f"{oracle.divergence_class}",
            "source_lang": oracle.source_lang,
            "target_lang": oracle.target_lang,
            "divergence_class": oracle.divergence_class,
            "verdict": res.verdict.value,
        })

    widths = []
    by_class = {o.divergence_class: o for o in ALL_ORACLES
                if o.source_lang == "c" and o.target_lang == "rust"}
    for cls in _WIDTH_SCALABLE_CLASSES:
        oracle = by_class.get(cls)
        if oracle is None:
            continue
        for w in _SUPPORTED_WIDTHS:
            unit = dict(CANONICAL_UNITS[cls], width=w,
                        source_lang="c", target_lang="rust")
            res = oracle.find_divergence(unit)
            widths.append({"divergence_class": cls, "width": w,
                           "verdict": res.verdict.value})

    smt_sizes = [8, 16, 32, 64, 128, 256, 512]
    smt = [{"width": w, "result": _solve_overflow_at_width(w)} for w in smt_sizes]

    return {
        "class_pair_profile": profile,
        "width_scaling": widths,
        "smt_scaling": smt,
    }
