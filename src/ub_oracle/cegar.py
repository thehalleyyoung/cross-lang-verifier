"""
Counterexample-guided abstraction refinement (100_STEPS step 73).

The interval pre-pass (:mod:`abstract_interp`, step 78) is a *non-relational*
abstraction: it reasons about each variable's range in isolation and therefore
loses any **path condition** (``assume``) that correlates the input with the UB
region. A unit guarded by ``assume(x % 2 == 0)`` whose only overflowing input is
the odd value ``INT_MAX`` is *equivalent* (no reachable overflow), but the
interval domain — which cannot represent "even" — has to fall through to the
full solver.

This module closes that gap with a genuine **lazy predicate-abstraction CEGAR
loop**. It decides the guarded reachability question

        ∃ x . UB(x)  ∧  ⋀_i assume_i(x)

("is there an input that both triggers the class's undefined behaviour *and*
satisfies every path condition?") by starting from the coarsest abstraction
(``UB(x)`` with **no** assumes) and refining **one assume at a time**, driven by
spurious counterexamples:

* **Abstract check.** Ask the solver for a model of ``UB(x)`` conjoined with only
  the *currently active* assumes. Dropping the inactive assumes is a sound
  over-approximation of the reachable-and-undefined set, so:
  * **UNSAT ⟹ proved equivalent.** If even this over-approximation has no model,
    no input can be both reachable (under the full guard) and undefined. The
    fragment is discharged *without ever enumerating the input space* — exactly
    the soundness contract of the interval pre-pass, but now path-sensitive.
* **Concretize.** Otherwise take the model ``x = v`` and test it against the
  assumes that are *not yet* active.
  * **All hold ⟹ genuine witness.** ``v`` satisfies ``UB`` and every guard, so it
    is a real divergence-triggering input (we hand it to the ground-truth
    harness to confirm the C ``-O0``/``-O2`` builds actually disagree).
  * **Some assume violated ⟹ spurious.** The abstract counterexample is an
    artefact of having dropped that guard. **Refine** by activating exactly that
    predicate and loop. Each refinement strictly grows the active set, so the
    loop runs at most ``len(assumes) + 1`` solver calls.

The loop reports its statistics (solver calls, refinement count, the predicates
it had to learn) so the refinement behaviour is *measured*, not asserted. The
whole construction is source-language (C-UB) reasoning, hence pair-agnostic: the
same refinement discharges or witnesses a guarded fragment for C→Rust, C→Go and
C→Swift alike.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import z3

from .abstract_interp import signed_max, signed_min


# --- guarded queries ---------------------------------------------------------


@dataclass(frozen=True)
class Predicate:
    """A path-condition atom over the single input variable ``x``.

    ``z3_of`` lowers the predicate to a Z3 constraint (for the abstract check)
    and ``holds_at`` evaluates it concretely on a signed integer (for the
    spuriousness test). Keeping both in one object guarantees the symbolic and
    concrete readings of every guard stay in lock-step.
    """

    text: str
    z3_of: Callable[[z3.ExprRef], z3.BoolRef]
    holds_at: Callable[[int], bool]


def even() -> Predicate:
    return Predicate("x % 2 == 0",
                     lambda x: x % 2 == 0,
                     lambda v: v % 2 == 0)


def odd() -> Predicate:
    return Predicate("x % 2 == 1",
                     lambda x: x % 2 == 1,
                     lambda v: v % 2 == 1)


def at_most(bound: int) -> Predicate:
    return Predicate(f"x <= {bound}",
                     lambda x: x <= bound,
                     lambda v: v <= bound)


def at_least(bound: int) -> Predicate:
    return Predicate(f"x >= {bound}",
                     lambda x: x >= bound,
                     lambda v: v >= bound)


def not_equal(value: int) -> Predicate:
    return Predicate(f"x != {value}",
                     lambda x: x != value,
                     lambda v: v != value)


def multiple_of(k: int) -> Predicate:
    return Predicate(f"x % {k} == 0",
                     lambda x: x % k == 0,
                     lambda v: v % k == 0)


@dataclass(frozen=True)
class GuardedQuery:
    """``f(x) = x <op> const`` at ``width`` bits, guarded by ``assumes``.

    The UB condition is the signed-overflow of the single arithmetic op (the
    anchor's signed-overflow class), so a genuine witness is directly
    confirmable against real compilers via the existing signed-overflow oracle.
    """

    op: str            # "add" | "sub"
    const: int
    width: int = 32
    var: str = "x"
    assumes: Tuple[Predicate, ...] = ()

    def __post_init__(self) -> None:
        if self.op not in ("add", "sub"):
            raise ValueError(f"unsupported op {self.op!r}")
        if self.width not in (32, 64):
            raise ValueError(f"unsupported width {self.width!r}")


# --- the UB condition --------------------------------------------------------


def _overflow_z3(x: z3.ExprRef, op: str, const: int, width: int) -> z3.BoolRef:
    c = z3.BitVecVal(const, width)
    if op == "add":
        no_of = z3.And(z3.BVAddNoOverflow(x, c, True), z3.BVAddNoUnderflow(x, c))
    else:
        no_of = z3.And(z3.BVSubNoOverflow(x, c), z3.BVSubNoUnderflow(x, c, True))
    return z3.Not(no_of)


def _overflows_concretely(v: int, op: str, const: int, width: int) -> bool:
    lo, hi = signed_min(width), signed_max(width)
    r = v + const if op == "add" else v - const
    return r < lo or r > hi


# --- the CEGAR loop ----------------------------------------------------------


class CegarVerdict(Enum):
    #: proved no input is both reachable (under the guards) and undefined.
    EQUIVALENT = "equivalent"
    #: found a concrete input that triggers UB and satisfies every guard.
    DIVERGENT = "divergent"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


@dataclass
class CegarResult:
    verdict: CegarVerdict
    refinements: int
    solver_calls: int
    learned_predicates: List[str] = field(default_factory=list)
    witness: Optional[int] = None
    trace: List[str] = field(default_factory=list)

    @property
    def proved_equivalent(self) -> bool:
        return self.verdict is CegarVerdict.EQUIVALENT


def _signed(raw: int, width: int) -> int:
    hi = signed_max(width)
    return raw if raw <= hi else raw - (1 << width)


def run_cegar(query: GuardedQuery) -> CegarResult:
    """Decide ``∃x. UB(x) ∧ ⋀ assume(x)`` by lazy predicate refinement."""
    width = query.width
    x = z3.BitVec(query.var, width)
    ub = _overflow_z3(x, query.op, query.const, width)

    active: List[int] = []         # indices into query.assumes
    refinements = 0
    solver_calls = 0
    trace: List[str] = []

    while True:
        solver = z3.Solver()
        solver.add(ub)
        for i in active:
            solver.add(query.assumes[i].z3_of(x))
        solver_calls += 1
        status = solver.check()

        if status != z3.sat:
            # Over-approximation already excludes UB ⟹ provably equivalent.
            trace.append(
                f"abstract check #{solver_calls} with {len(active)} active "
                f"predicate(s): UNSAT — UB unreachable under the guards")
            return CegarResult(
                CegarVerdict.EQUIVALENT, refinements, solver_calls,
                learned_predicates=[query.assumes[i].text for i in active],
                trace=trace)

        v = _signed(solver.model()[x].as_long(), width)
        # Concretize: find an *inactive* assume that this model violates.
        spurious_idx: Optional[int] = None
        for i, pred in enumerate(query.assumes):
            if i in active:
                continue
            if not pred.holds_at(v):
                spurious_idx = i
                break

        if spurious_idx is None:
            trace.append(
                f"abstract check #{solver_calls}: model x={v} satisfies UB and "
                f"every guard — genuine witness")
            return CegarResult(
                CegarVerdict.DIVERGENT, refinements, solver_calls,
                learned_predicates=[query.assumes[i].text for i in active],
                witness=v, trace=trace)

        # Spurious: refine by learning the violated guard.
        active.append(spurious_idx)
        refinements += 1
        trace.append(
            f"abstract check #{solver_calls}: model x={v} violates guard "
            f"'{query.assumes[spurious_idx].text}' — refine (learn it)")


# --- ground truth for validation --------------------------------------------


def brute_force_witness(query: GuardedQuery,
                        scan: int = 4) -> Optional[int]:
    """Exact decision by enumerating the (tiny) concrete UB region.

    Signed ``add``/``sub`` overflow of ``x <op> c`` happens only on a narrow band
    of ``x`` at the extreme of the type, so the genuinely-undefined inputs number
    ``|c|`` — enumerable exactly regardless of the 32/64-bit width. We widen the
    band by ``scan`` for safety and return the first UB input that satisfies
    every guard, or ``None`` if there is none. This is the ground truth that
    :func:`run_cegar` is checked against in the tests.
    """
    width = query.width
    lo, hi = signed_min(width), signed_max(width)
    band = abs(query.const) + scan
    if query.op == "add":
        candidates = range(hi - band, hi + 1)            # large positive x
    else:
        candidates = range(lo, lo + band + 1)            # large negative x
    for v in candidates:
        if v < lo or v > hi:
            continue
        if not _overflows_concretely(v, query.op, query.const, width):
            continue
        if all(p.holds_at(v) for p in query.assumes):
            return v
    return None


def to_signed_overflow_unit(query: GuardedQuery, witness: int) -> Dict:
    """A ``binop_const`` unit pinned to ``witness`` for real-compiler confirmation.

    Pinning the operating range to ``[witness, witness]`` makes the existing
    signed-overflow oracle's Z3 search return exactly the CEGAR witness, so the
    ground-truth harness then confirms the real C ``-O0``/``-O2`` builds disagree
    on that input — closing the loop from CEGAR's symbolic decision to actual
    compiled behaviour.
    """
    return {
        "kind": "binop_const",
        "op": query.op,
        "const": query.const,
        "width": query.width,
        "var": query.var,
        "signed": True,
        "x_range": [witness, witness],
        "probe": "signed_overflow",
    }
