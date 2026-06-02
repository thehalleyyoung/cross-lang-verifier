"""Formal divergence semantics (Step 80).

This module pins down, as an *executable* predicate, exactly what this tool
means by a cross-language **divergence rooted in source undefined behavior**.
The prose definition is in ``docs/SEMANTICS.md``; here we give the same
definition as code so that:

  * the formal predicate can be applied to a structured *observation* of a
    concrete re-execution, and
  * a property test can prove the formal predicate **coincides** with the
    decision the re-execution harness (:class:`reexec.ReexecHarness`) actually
    makes — i.e. the math and the implementation are the same object.

The definition is parameterized over the *consequence mode* (``exploited`` vs
``trap_vs_defined``); both modes share the same source-undefinedness premise and
the same target-definedness obligation, differing only in what counts as the
observable *consequence* of the undefinedness.  It is otherwise agnostic to the
language pair: the target observation is the only pair-specific input, exactly
as in the harness (the pair changes the emitted target program, not the rule).

--------------------------------------------------------------------------------
Definition (Divergence modulo source-undefinedness).

Fix a language pair ``(S, T)``, a translation pair ``(P_S, P_T)`` of one unit,
and a concrete input ``i`` drawn from the unit's declared operating domain
``D``.  Let the *source observation* approximate the set of behaviors a
conforming S-implementation may exhibit on ``i`` by three measurements:

  * ``o0``  — the observable of ``P_S`` built without optimization,
  * ``o2``  — the observable of ``P_S`` built with optimization,
  * ``san`` — whether a sanitizer build of ``P_S`` *traps* on ``i`` (a witness
              that ``i`` actually reaches undefined behavior in S).

Let the *target observation* of ``P_T`` on ``i`` record whether the target
outcome is **defined** (a normal value, or a clean deterministic trap/panic
that the target language guarantees) and, where relevant, **deterministic**
(identical across repeated runs).

The pair **diverges at ``i`` (modulo source-undefinedness)** iff:

  (P) *Premise — source undefinedness is reached*:  ``san`` traps; and
  (T) *Target is defined*:  the target observation is defined
      (and, in the ``trap_vs_defined`` mode, deterministic); and
  (C) *Consequence*, by mode:
        - ``exploited``        : ``o0`` and ``o2`` are both defined values that
                                 **disagree** — the optimizer demonstrably
                                 exploited the UB to change the observable; or
        - ``trap_vs_defined``  : the consequence *is* the definedness gap
                                 itself — S is undefined here while T is
                                 defined — so (C) reduces to (P) ∧ (T).

Divergence is therefore a *witnessed failure* of "observational equivalence on
defined inputs": equivalence would require, for every defined ``i``, that the
defined target observable coincide with a single stable source observable.  We
never claim equivalence (the property is one-sided); we only ever *witness*
divergence, and only when (P) holds — so a reported divergence is always rooted
in genuine source UB, never in a mere value mismatch.

--------------------------------------------------------------------------------
Soundness note.  ``is_divergence`` returns ``True`` only when (P) holds, so by
construction every positive verdict is rooted in observed source UB.  The
companion theorem :func:`coincides_with_harness` (exercised by the test suite on
**real** compiled programs) states that, on any observation extracted from a
harness run, ``is_divergence`` equals ``ReexecResult.confirmed`` — the formal
predicate and the operational check are one and the same decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

__all__ = [
    "EXPLOITED",
    "TRAP_VS_DEFINED",
    "MODES",
    "Outcome",
    "SourceObservation",
    "TargetObservation",
    "Observation",
    "DivergenceJudgment",
    "is_divergence",
    "judge",
    "observation_from_reexec",
    "coincides_with_harness",
]

EXPLOITED = "exploited"
TRAP_VS_DEFINED = "trap_vs_defined"
MODES = (EXPLOITED, TRAP_VS_DEFINED)


@dataclass(frozen=True)
class Outcome:
    """A single observable outcome of one compiled program on one input.

    ``returncode`` follows the harness convention (0 = normal exit printing
    ``value``; a non-zero / negative code denotes a trap/panic/abort).  ``value``
    is the program's stdout (only meaningful when ``returncode == 0``).
    """

    returncode: int
    value: str = ""

    @property
    def is_value(self) -> bool:
        return self.returncode == 0

    @property
    def is_trap(self) -> bool:
        return self.returncode != 0


@dataclass(frozen=True)
class SourceObservation:
    """The source side of one observation: O0, O2 and the sanitizer verdict."""

    o0: Outcome
    o2: Outcome
    san_trapped: bool

    @property
    def ub_reached(self) -> bool:
        """(P): a sanitizer build witnessed that the input reaches UB."""
        return self.san_trapped

    @property
    def opt_levels_disagree(self) -> bool:
        """The optimizer changed the observable: both build defined, values differ."""
        return self.o0.is_value and self.o2.is_value and self.o0.value != self.o2.value


@dataclass(frozen=True)
class TargetObservation:
    """The target side of one observation."""

    defined: bool
    deterministic: bool = True


@dataclass(frozen=True)
class Observation:
    """A complete, mode-tagged observation of one re-execution at one input."""

    source: SourceObservation
    target: TargetObservation
    mode: str = EXPLOITED

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(
                f"unknown consequence mode {self.mode!r}; expected one of {MODES}"
            )


@dataclass(frozen=True)
class DivergenceJudgment:
    """Structured result of applying the formal definition to an observation."""

    diverges: bool
    premise_ub_reached: bool
    target_defined: bool
    consequence_met: bool
    mode: str
    reason: str

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return self.diverges


def _consequence(obs: Observation) -> bool:
    """Clause (C) of the definition, by mode."""
    if obs.mode == EXPLOITED:
        return obs.source.opt_levels_disagree
    # TRAP_VS_DEFINED: the consequence is the definedness gap itself, i.e. (P)∧(T),
    # plus determinism of the defined target outcome.
    return obs.source.ub_reached and obs.target.defined and obs.target.deterministic


def judge(obs: Observation) -> DivergenceJudgment:
    """Apply the formal definition; return a structured judgment with reasons."""
    premise = obs.source.ub_reached
    target_defined = obs.target.defined and (
        obs.target.deterministic or obs.mode == EXPLOITED
    )
    consequence = _consequence(obs)
    diverges = premise and target_defined and consequence

    if diverges:
        if obs.mode == EXPLOITED:
            reason = (
                "rooted in source UB (sanitizer trapped); optimizer exploited it "
                f"(O0={obs.source.o0.value!r} != O2={obs.source.o2.value!r}); "
                "target defined"
            )
        else:
            reason = (
                "rooted in source UB (sanitizer trapped); target defined and "
                "deterministic while source is undefined"
            )
    elif not premise:
        reason = "no divergence: clause (P) fails — source UB not reached on this input"
    elif not target_defined:
        reason = "no divergence: clause (T) fails — target outcome not defined/deterministic"
    else:
        reason = "no divergence: clause (C) fails — undefinedness had no observed consequence"

    return DivergenceJudgment(
        diverges=diverges,
        premise_ub_reached=premise,
        target_defined=target_defined,
        consequence_met=consequence,
        mode=obs.mode,
        reason=reason,
    )


def is_divergence(obs: Observation) -> bool:
    """The formal divergence predicate (boolean form of :func:`judge`)."""
    return judge(obs).diverges


def observation_from_reexec(result) -> Optional[Observation]:
    """Extract a formal :class:`Observation` from a :class:`reexec.ReexecResult`.

    Returns ``None`` when the harness run was unavailable / did not produce the
    measurements the definition needs (e.g. the toolchain was missing), so that
    the coincidence theorem is only asserted on genuinely-observed runs.
    """
    if not getattr(result, "available", False):
        return None
    c_runs = getattr(result, "c_runs", {}) or {}
    if "O0" not in c_runs:
        return None

    o0_run = c_runs["O0"]
    o2_run = c_runs.get("O2", o0_run)
    o0 = Outcome(o0_run.returncode, o0_run.stdout)
    o2 = Outcome(o2_run.returncode, o2_run.stdout)
    src = SourceObservation(o0=o0, o2=o2, san_trapped=bool(result.ub_reachable))

    # The harness already collapses target definedness (incl. the determinism
    # re-run in trap_vs_defined mode) into ``rust_defined``; reuse it verbatim so
    # the formal predicate consumes exactly the harness's own measurement.
    tgt = TargetObservation(defined=bool(result.rust_defined), deterministic=True)

    mode = getattr(result, "mode", EXPLOITED)
    if mode not in MODES:
        mode = EXPLOITED
    return Observation(source=src, target=tgt, mode=mode)


def coincides_with_harness(result) -> bool:
    """Theorem (checkable per-run): the formal predicate equals the harness verdict.

    For any harness result that yielded a usable observation,
    ``is_divergence(observation_from_reexec(result)) == result.confirmed``.
    Returns ``True`` vacuously when no observation could be extracted (the run
    was unavailable), so callers can fold it over a batch of runs.
    """
    obs = observation_from_reexec(result)
    if obs is None:
        return True
    return is_divergence(obs) == bool(result.confirmed)
