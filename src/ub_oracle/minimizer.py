"""
Counterexample minimizer + quality assessment (100_STEPS step 52).

A symbolic witness from Z3 is *correct* but often not *minimal*: e.g. the signed
-overflow oracle may report ``x = 1073741824`` when ``x = 1`` already triggers the
exact same confirmed divergence.  A minimal, canonical witness is a real
usability differentiator — it is what a human reads, files in a bug report, and
turns into a regression test.

This module shrinks a confirmed counterexample to a locally-minimal,
canonical witness **and proves every reduction against the real ground-truth
harness** (the same compile-and-run confirmation the oracle uses).  Nothing here
trusts a heuristic: a candidate witness is accepted only if the real compilers
agree it still diverges.

Minimality is defined precisely and modestly: a witness is *1-minimal w.r.t. the
canonical reduction ladder* if, for every integer input field, no strictly
simpler value drawn from a canonical neighbour set ({0, ±1, |v|-1, ⌊v/2⌋, …})
preserves the confirmed divergence.  ``certified_locally_minimal`` reports
exactly that — never an unprovable "globally minimal" claim.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .plugin import DivergenceOracle, OracleResult, OracleVerdict
from .reexec import ReexecHarness
from .replay import Counterexample


def simplicity_cost(value: int) -> Tuple[int, int]:
    """Lexicographic simplicity cost of an integer (smaller == simpler).

    Primary key is magnitude; ties prefer non-negative values, so 0 is the
    simplest possible witness, then 1, then -1, then 2, ...
    """
    return (abs(value), 0 if value >= 0 else 1)


def _is_simpler(candidate: int, current: int) -> bool:
    return simplicity_cost(candidate) < simplicity_cost(current)


def _reduction_ladder(v: int) -> List[int]:
    """Strictly-simpler candidate values for ``v``, simplest first, de-duplicated.

    The ladder mixes canonical anchors (0, ±1) with magnitude-halving steps and
    the immediate neighbours ``v±1``, which between them cover both "collapse to a
    boundary" and "nudge toward zero" minimizations.
    """
    cands: List[int] = [0, 1, -1, v - 1, v + 1]
    m = abs(v)
    sign = -1 if v < 0 else 1
    while m > 1:
        m //= 2
        cands.append(sign * m)
        cands.append(-sign * m)
    # keep only strictly-simpler, unique, ordered by ascending simplicity cost
    seen: set = set()
    out: List[int] = []
    for c in sorted((c for c in cands if _is_simpler(c, v)), key=simplicity_cost):
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


@dataclass
class MinimizationStep:
    """One accepted reduction in a real-harness-backed minimization trace."""

    field: str
    before_inputs: Dict[str, Any]
    after_inputs: Dict[str, Any]
    ub_category: str = ""
    confirmed: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "before_inputs": self.before_inputs,
            "after_inputs": self.after_inputs,
            "ub_category": self.ub_category,
            "confirmed": self.confirmed,
        }


@dataclass
class MinimizationResult:
    divergence_class: str
    source_lang: str
    target_lang: str
    original_inputs: Dict[str, Any]
    minimized_inputs: Dict[str, Any]
    fields_reduced: List[str] = field(default_factory=list)
    probes: int = 0
    confirmed: bool = False
    certified_locally_minimal: bool = False
    already_minimal: bool = False
    ub_category: str = ""
    accepted_reductions: List[MinimizationStep] = field(default_factory=list)

    @property
    def reduced(self) -> bool:
        return self.minimized_inputs != self.original_inputs

    def original_cost(self) -> int:
        return sum(abs(v) for v in self.original_inputs.values() if isinstance(v, int))

    def minimized_cost(self) -> int:
        return sum(abs(v) for v in self.minimized_inputs.values() if isinstance(v, int))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "divergence_class": self.divergence_class,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "original_inputs": self.original_inputs,
            "minimized_inputs": self.minimized_inputs,
            "fields_reduced": self.fields_reduced,
            "original_cost": self.original_cost(),
            "minimized_cost": self.minimized_cost(),
            "reduced": self.reduced,
            "probes": self.probes,
            "confirmed": self.confirmed,
            "certified_locally_minimal": self.certified_locally_minimal,
            "already_minimal": self.already_minimal,
        }


@dataclass
class MinimizationCertificate:
    """Offline certificate for the scoped minimizer guarantee.

    The certificate does not replace compiler re-execution.  It records the
    accepted reduction chain produced by :func:`minimize_counterexample`, whose
    reductions were already admitted only after the real harness confirmed the
    same divergence class (and the same UBSan category when one exists).
    """

    divergence_class: str
    source_lang: str
    target_lang: str
    original_inputs: Dict[str, Any]
    minimized_inputs: Dict[str, Any]
    ub_category: str
    reductions: List[MinimizationStep] = field(default_factory=list)
    confirmed: bool = False
    certified_locally_minimal: bool = False
    theorem: str = "minimizer_certificate_sound"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "divergence_class": self.divergence_class,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "original_inputs": self.original_inputs,
            "minimized_inputs": self.minimized_inputs,
            "ub_category": self.ub_category,
            "reductions": [r.to_dict() for r in self.reductions],
            "confirmed": self.confirmed,
            "certified_locally_minimal": self.certified_locally_minimal,
            "theorem": self.theorem,
        }


def minimization_certificate(result: MinimizationResult) -> MinimizationCertificate:
    """Build the deterministic offline certificate for ``result``."""

    return MinimizationCertificate(
        divergence_class=result.divergence_class,
        source_lang=result.source_lang,
        target_lang=result.target_lang,
        original_inputs=dict(result.original_inputs),
        minimized_inputs=dict(result.minimized_inputs),
        ub_category=result.ub_category,
        reductions=list(result.accepted_reductions),
        confirmed=result.confirmed,
        certified_locally_minimal=result.certified_locally_minimal,
    )


def _step_decreases_cost(step: MinimizationStep) -> bool:
    before = step.before_inputs.get(step.field)
    after = step.after_inputs.get(step.field)
    if not isinstance(before, int) or not isinstance(after, int):
        return False
    return _is_simpler(after, before)


def verify_minimization_certificate(cert: MinimizationCertificate) -> Tuple[bool, str]:
    """Check the offline shape of a minimization certificate.

    This is intentionally deterministic and toolchain-free: the expensive fact
    that each step was real-harness-confirmed is recorded as ``confirmed=True``
    by the minimizer at the point the compiler-backed probe accepted the step.
    """

    if cert.theorem != "minimizer_certificate_sound":
        return False, "unexpected minimizer theorem"
    if not cert.divergence_class or not cert.source_lang or not cert.target_lang:
        return False, "missing class or language pair"
    if not cert.confirmed:
        return False, "minimized witness was not confirmed"
    if not cert.certified_locally_minimal:
        return False, "local minimality was not certified"

    current = dict(cert.original_inputs)
    for step in cert.reductions:
        if not step.confirmed:
            return False, f"unconfirmed reduction for field {step.field!r}"
        if step.before_inputs != current:
            return False, f"reduction for {step.field!r} is not chained"
        if cert.ub_category and step.ub_category != cert.ub_category:
            return False, f"UB category drifted at field {step.field!r}"
        if not _step_decreases_cost(step):
            return False, f"reduction for {step.field!r} does not simplify"
        current = dict(step.after_inputs)

    if current != cert.minimized_inputs:
        return False, "reduction chain does not end at minimized inputs"
    if not cert.reductions and cert.original_inputs != cert.minimized_inputs:
        return False, "changed inputs without recorded reductions"
    return True, "ok"


class _Probe:
    """Counts real ground-truth confirmations of trial witnesses.

    Each probe records not just *whether* the trial still diverges, but the
    UBSan diagnostic category that fired, so the minimizer can require a
    simplified witness to trigger the SAME undefined behavior as the original
    (never silently drifting into a different UB class).
    """

    def __init__(self, oracle: DivergenceOracle, base_ce: Counterexample,
                 harness: ReexecHarness):
        self.oracle = oracle
        self.base_ce = base_ce
        self.harness = harness
        self.count = 0

    def evaluate(self, inputs: Dict[str, Any]) -> Tuple[bool, str]:
        """Return (confirmed, ub_category) for the trial ``inputs``."""
        self.count += 1
        ce = copy.deepcopy(self.base_ce)
        ce.inputs = dict(inputs)
        ce.confirmed = False
        trial = OracleResult(OracleVerdict.DIVERGENT, self.oracle.divergence_class,
                             counterexample=ce)
        rr = self.oracle.confirm(trial, self.harness).reexec
        if rr is None or not rr.available or not rr.confirmed:
            return (False, "")
        san = rr.c_runs.get("san")
        return (True, san.ub_category if san is not None else "")


def minimize_counterexample(oracle: DivergenceOracle, result: OracleResult,
                            harness: ReexecHarness,
                            max_probes: int = 400) -> MinimizationResult:
    """Shrink a confirmed counterexample to a locally-minimal canonical witness.

    Every accepted reduction is verified by really compiling and running the
    source and target via ``harness``.  Returns a :class:`MinimizationResult`
    describing the original and minimized witnesses, the number of real probes
    used, and whether the result is certified 1-minimal w.r.t. the canonical
    ladder.
    """
    if result.counterexample is None:
        raise ValueError("cannot minimize a result without a counterexample")
    base = result.counterexample
    orig = dict(base.inputs)
    int_fields = [k for k, v in orig.items() if isinstance(v, int)]

    probe = _Probe(oracle, base, harness)
    current = dict(orig)

    # Sanity: the original witness must really confirm, or minimization is moot.
    confirmed, orig_category = probe.evaluate(current)
    if not confirmed:
        return MinimizationResult(
            divergence_class=base.divergence_class,
            source_lang=base.source_lang, target_lang=base.target_lang,
            original_inputs=orig, minimized_inputs=current,
            probes=probe.count, confirmed=False)

    accepted_reductions: List[MinimizationStep] = []

    def _accepts(inputs: Dict[str, Any]) -> Tuple[bool, str]:
        """A trial is acceptable only if it still confirms AND triggers the
        SAME undefined behavior category as the original witness.  This is what
        keeps minimization *faithful*: without it, magnitude reduction silently
        drifts a witness into a different UB (e.g. INT_MIN/-1 collapsing to a
        plain division-by-zero, or a too-large shift becoming a negative shift).
        When the original carries no UBSan category (non-trapping/optimizer-
        exploited classes) we fall back to confirmation alone.
        """
        ok, cat = probe.evaluate(inputs)
        if not ok:
            return False, cat
        if orig_category != "" and cat != orig_category:
            return False, cat
        return True, cat

    # Greedy 1-minimization to a fixpoint: repeatedly try to reduce each field to
    # the simplest confirming value, accepting the first (simplest) that holds.
    changed = True
    while changed and probe.count < max_probes:
        changed = False
        for fld in int_fields:
            v = current[fld]
            for cand in _reduction_ladder(v):
                if probe.count >= max_probes:
                    break
                trial = dict(current)
                trial[fld] = cand
                ok, cat = _accepts(trial)
                if ok:
                    accepted_reductions.append(MinimizationStep(
                        field=fld,
                        before_inputs=dict(current),
                        after_inputs=dict(trial),
                        ub_category=cat,
                    ))
                    current[fld] = cand
                    changed = True
                    break  # restart this field from the newly-simpler value

    fields_reduced = [k for k in int_fields if current[k] != orig[k]]

    # Certify local minimality: for each field, no strictly-simpler canonical
    # candidate may preserve the confirmed divergence (in the same UB category).
    locally_minimal = True
    for fld in int_fields:
        v = current[fld]
        for cand in _reduction_ladder(v):
            if probe.count >= max_probes:
                locally_minimal = False  # budget exhausted; cannot certify
                break
            trial = dict(current)
            trial[fld] = cand
            ok, _cat = _accepts(trial)
            if ok:
                locally_minimal = False
                break
        if not locally_minimal:
            break

    return MinimizationResult(
        divergence_class=base.divergence_class,
        source_lang=base.source_lang, target_lang=base.target_lang,
        original_inputs=orig, minimized_inputs=current,
        fields_reduced=fields_reduced, probes=probe.count, confirmed=True,
        certified_locally_minimal=locally_minimal,
        already_minimal=(not fields_reduced) and locally_minimal,
        ub_category=orig_category,
        accepted_reductions=accepted_reductions)
