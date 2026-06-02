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

import statistics
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import z3

from . import oracles as _oracles  # noqa: F401  (registers all pairs)
from .plugin import ALL_ORACLES, OracleVerdict
from .regression_matrix import CANONICAL_UNITS, canonical_unit_for, _sorted_oracles


def _median_time(fn: Callable[[], object], repeats: int) -> Tuple[float, object]:
    """Median wall-clock seconds of ``fn`` over ``repeats`` runs (>=1).

    Returns ``(median_seconds, last_return_value)``.  Uses ``perf_counter`` and
    discards the return values except the last so the measured work is real.
    """
    repeats = max(1, repeats)
    samples: List[float] = []
    last = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        last = fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples), last


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
