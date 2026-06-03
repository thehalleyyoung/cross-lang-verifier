"""
SMT-backed oracle for the integer-UB family (100_STEPS step 133).

The per-class oracles in :mod:`ub_oracle.oracles.integer_ub` already use Z3 to
*find* a witnessing input.  This module goes one level deeper and makes the
*encoding itself* a first-class, machine-checked artifact:

  * For every integer-UB class we give two **independent** Z3 bit-vector
    encodings of "this concrete C operation is undefined":

        - ``encode_operational`` — the formula an oracle actually solves
          (e.g. ``Not(BVSDivNoOverflow(a, b))`` for INT_MIN / -1), and
        - ``encode_spec``        — a human-readable specification of the same
          condition (e.g. ``a == INT_MIN && b == -1``).

  * :func:`prove_equisatisfiable` discharges, *for the real bit width*, the
    validity of ``operational <-> spec`` (Z3 proves the iff for **all** inputs,
    not a sample).  This is the "prove the encoding equisatisfiable" obligation:
    the SMT path and the specification accept exactly the same witnesses.

  * :func:`enumerate_witness` is a brute-force reference that scans the input
    space in order; :func:`smt_witness` asks Z3.  :func:`benchmark` times the
    two and reports the speed-up.  For the rare classes (a single witness in
    ``2**width`` inputs) enumeration blows its probe budget while the SMT path
    answers in milliseconds — the quantitative case for the SMT encoding.

Everything here is pure Z3/Python: no compilers, so it is fast to test and is a
clean, mechanized companion to the operational oracles.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import z3

# The integer-UB family this module mechanizes.
CLASSES: Tuple[str, ...] = (
    "signed_overflow",
    "shift_oob",
    "div_by_zero",
    "intmin_div_neg1",
)

# Bit widths the operational oracles ship for.
SHIPPED_WIDTHS: Tuple[int, ...] = (32, 64)


def _bounds(width: int) -> Tuple[int, int]:
    return -(1 << (width - 1)), (1 << (width - 1)) - 1


# --------------------------------------------------------------------------- #
# Encodings.  Each returns (variables, operational_expr, spec_expr) over fresh
# Z3 bit-vectors of the requested width.
# --------------------------------------------------------------------------- #


def _enc_signed_overflow(width: int):
    x = z3.BitVec("x", width)
    c = z3.BitVec("c", width)
    # Operational: the no-overflow/no-underflow builtins the oracle relies on.
    operational = z3.Or(
        z3.Not(z3.BVAddNoOverflow(x, c, True)),
        z3.Not(z3.BVAddNoUnderflow(x, c)),
    )
    # Spec: compute x + c in one extra bit and check it leaves the signed range.
    lo, hi = _bounds(width)
    wide = z3.SignExt(1, x) + z3.SignExt(1, c)
    spec = z3.Or(
        wide > z3.BitVecVal(hi, width + 1),
        wide < z3.BitVecVal(lo, width + 1),
    )
    return (x, c), operational, spec


def _enc_shift_oob(width: int):
    # Shift amount modelled as a signed value of the same width.
    s = z3.BitVec("s", width)
    operational = z3.Or(s < z3.BitVecVal(0, width),
                        s >= z3.BitVecVal(width, width))
    # Spec: NOT in the in-range band [0, width-1].
    spec = z3.Not(z3.And(s >= z3.BitVecVal(0, width),
                         s <= z3.BitVecVal(width - 1, width)))
    return (s,), operational, spec


def _enc_div_by_zero(width: int):
    b = z3.BitVec("b", width)
    operational = b == z3.BitVecVal(0, width)
    spec = z3.Not(b != z3.BitVecVal(0, width))
    return (b,), operational, spec


def _enc_intmin_div_neg1(width: int):
    a = z3.BitVec("a", width)
    b = z3.BitVec("b", width)
    lo, _ = _bounds(width)
    # Operational: the signed-division-overflow builtin, excluding b == 0.
    operational = z3.And(b != z3.BitVecVal(0, width),
                        z3.Not(z3.BVSDivNoOverflow(a, b)))
    # Spec: the unique overflowing pair.
    spec = z3.And(a == z3.BitVecVal(lo, width),
                 b == z3.BitVecVal(-1, width))
    return (a, b), operational, spec


_ENCODERS: Dict[str, Callable[[int], tuple]] = {
    "signed_overflow": _enc_signed_overflow,
    "shift_oob": _enc_shift_oob,
    "div_by_zero": _enc_div_by_zero,
    "intmin_div_neg1": _enc_intmin_div_neg1,
}


# --------------------------------------------------------------------------- #
# Reference (enumeration) semantics — concrete Python predicates over signed
# integers.  These define UB independently of Z3 so the brute-force path is a
# genuine cross-check of the symbolic encoding.
# --------------------------------------------------------------------------- #


def _pred_signed_overflow(width: int, vals: Dict[str, int]) -> bool:
    lo, hi = _bounds(width)
    s = vals["x"] + vals["c"]
    return s > hi or s < lo


def _pred_shift_oob(width: int, vals: Dict[str, int]) -> bool:
    s = vals["s"]
    return s < 0 or s >= width


def _pred_div_by_zero(width: int, vals: Dict[str, int]) -> bool:
    return vals["b"] == 0


def _pred_intmin_div_neg1(width: int, vals: Dict[str, int]) -> bool:
    lo, _ = _bounds(width)
    return vals["a"] == lo and vals["b"] == -1


_PREDICATES: Dict[str, Callable[[int, Dict[str, int]], bool]] = {
    "signed_overflow": _pred_signed_overflow,
    "shift_oob": _pred_shift_oob,
    "div_by_zero": _pred_div_by_zero,
    "intmin_div_neg1": _pred_intmin_div_neg1,
}

# Variable names per class (the order enumeration iterates).
_VARS: Dict[str, Tuple[str, ...]] = {
    "signed_overflow": ("x", "c"),
    "shift_oob": ("s",),
    "div_by_zero": ("b",),
    "intmin_div_neg1": ("a", "b"),
}


def variables_of(class_key: str) -> Tuple[str, ...]:
    return _VARS[class_key]


# --------------------------------------------------------------------------- #
# Equisatisfiability proof.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EquisatProof:
    class_key: str
    width: int
    proved: bool
    detail: str


def prove_equisatisfiable(class_key: str, width: int) -> EquisatProof:
    """Z3-prove ``operational <-> spec`` is valid for *every* input of `width`.

    Returns a proof object whose ``proved`` flag is True iff the negation of the
    bi-implication is unsatisfiable (i.e. the two encodings accept exactly the
    same witnesses, hence are equisatisfiable).
    """
    if class_key not in _ENCODERS:
        raise KeyError(f"unknown class {class_key!r}")
    _vars, operational, spec = _ENCODERS[class_key](width)
    solver = z3.Solver()
    solver.add(operational != spec)  # a model here is a disagreement
    res = solver.check()
    if res == z3.unsat:
        return EquisatProof(class_key, width, True,
                            "operational <-> spec is valid (no disagreeing input)")
    if res == z3.sat:
        return EquisatProof(class_key, width, False,
                            f"disagreement witness: {solver.model()}")
    return EquisatProof(class_key, width, False, f"z3 returned {res}")


def prove_all_equisatisfiable(widths: Tuple[int, ...] = SHIPPED_WIDTHS
                              ) -> List[EquisatProof]:
    return [prove_equisatisfiable(c, w) for c in CLASSES for w in widths]


# --------------------------------------------------------------------------- #
# Witness search: SMT vs enumeration.
# --------------------------------------------------------------------------- #


def smt_witness(class_key: str, width: int) -> Optional[Dict[str, int]]:
    """Solve the operational encoding; return a signed-int witness or None."""
    vars_, operational, _spec = _ENCODERS[class_key](width)
    solver = z3.Solver()
    solver.add(operational)
    if solver.check() != z3.sat:
        return None
    model = solver.model()
    out: Dict[str, int] = {}
    for v in vars_:
        raw = model[v]
        # An unconstrained variable can be absent from the model; default to 0.
        value = 0 if raw is None else _signed(raw.as_long(), width)
        out[str(v)] = value
    return out


def _signed(raw: int, width: int) -> int:
    hi = (1 << (width - 1)) - 1
    return raw if raw <= hi else raw - (1 << width)


def _unsigned(value: int, width: int) -> int:
    return value & ((1 << width) - 1)


@dataclass(frozen=True)
class EnumResult:
    found: Optional[Dict[str, int]]
    probes: int
    exhausted: bool  # True iff the whole space (within budget) was scanned


def enumerate_witness(class_key: str, width: int,
                      probe_budget: int = 1 << 22) -> EnumResult:
    """Brute-force the input space in unsigned order for a witness.

    Scans up to ``probe_budget`` points.  For multi-variable classes the budget
    is spread across a square grid.  Returns the first witness found (decoded to
    signed ints) plus the probe count — the cost the SMT path avoids.
    """
    pred = _PREDICATES[class_key]
    names = _VARS[class_key]
    lo, hi = _bounds(width)
    span = 1 << width
    probes = 0

    if len(names) == 1:
        n = names[0]
        limit = min(span, probe_budget)
        for u in range(limit):
            val = _signed(u, width)
            probes += 1
            if pred(width, {n: val}):
                return EnumResult({n: val}, probes, limit >= span)
        return EnumResult(None, probes, limit >= span)

    # Two-variable: scan a side x side grid in unsigned order.
    import math
    side = min(span, max(1, int(math.isqrt(probe_budget))))
    a_name, b_name = names
    for ua in range(side):
        for ub in range(side):
            va, vb = _signed(ua, width), _signed(ub, width)
            probes += 1
            if pred(width, {a_name: va, b_name: vb}):
                return EnumResult({a_name: va, b_name: vb}, probes,
                                  side >= span)
    return EnumResult(None, probes, side >= span)


@dataclass(frozen=True)
class Benchmark:
    class_key: str
    width: int
    smt_seconds: float
    enum_seconds: float
    smt_found: bool
    enum_found: bool
    enum_probes: int
    enum_exhausted: bool

    @property
    def speedup(self) -> float:
        if self.smt_seconds <= 0:
            return float("inf")
        return self.enum_seconds / self.smt_seconds


def benchmark(class_key: str, width: int,
              probe_budget: int = 1 << 22) -> Benchmark:
    t0 = time.perf_counter()
    smt = smt_witness(class_key, width)
    t1 = time.perf_counter()
    enum = enumerate_witness(class_key, width, probe_budget=probe_budget)
    t2 = time.perf_counter()
    return Benchmark(
        class_key=class_key,
        width=width,
        smt_seconds=t1 - t0,
        enum_seconds=t2 - t1,
        smt_found=smt is not None,
        enum_found=enum.found is not None,
        enum_probes=enum.probes,
        enum_exhausted=enum.exhausted,
    )


def smt_path_agrees_with_predicate(class_key: str, width: int) -> bool:
    """The SMT witness, decoded, must satisfy the independent Python predicate."""
    w = smt_witness(class_key, width)
    if w is None:
        return False
    return _PREDICATES[class_key](width, w)


@dataclass(frozen=True)
class AgreementGroup:
    class_key: str
    width: int
    cases: int
    agreed: bool
    detail: str


@dataclass(frozen=True)
class AgreementReport:
    total_cases: int
    seed: int
    groups: Tuple[AgreementGroup, ...]

    @property
    def ok(self) -> bool:
        return (
            bool(self.groups)
            and sum(g.cases for g in self.groups) == self.total_cases
            and all(g.agreed for g in self.groups)
        )

    @property
    def mismatches(self) -> Tuple[str, ...]:
        return tuple(g.detail for g in self.groups if not g.agreed)


def _interesting_values(width: int) -> Tuple[int, ...]:
    lo, hi = _bounds(width)
    raw = {
        lo, lo + 1, -width - 1, -width, -1, 0, 1,
        width - 1, width, width + 1, hi - 1, hi,
    }
    return tuple(sorted(v for v in raw if lo <= v <= hi))


def _sample_assignment(
    rng: random.Random,
    class_key: str,
    width: int,
) -> Dict[str, int]:
    vals: Dict[str, int] = {}
    interesting = _interesting_values(width)
    for name in _VARS[class_key]:
        if rng.random() < 0.45:
            vals[name] = rng.choice(interesting)
        else:
            vals[name] = _signed(rng.randrange(1 << width), width)
    return vals


def _agreement_group(
    class_key: str,
    width: int,
    samples: List[Dict[str, int]],
) -> AgreementGroup:
    vars_, operational, _spec = _ENCODERS[class_key](width)
    mismatch_terms = []
    for sample in samples:
        expected = _PREDICATES[class_key](width, sample)
        constraints = [
            v == z3.BitVecVal(_unsigned(sample[str(v)], width), width)
            for v in vars_
        ]
        mismatch_terms.append(
            z3.And(*(constraints + [operational != z3.BoolVal(expected)])))

    solver = z3.Solver()
    solver.add(z3.Or(*mismatch_terms) if mismatch_terms else z3.BoolVal(False))
    result = solver.check()
    if result == z3.unsat:
        return AgreementGroup(class_key, width, len(samples), True, "all sampled cases agree")
    if result == z3.unknown:
        return AgreementGroup(class_key, width, len(samples), False, f"z3 returned {result}")

    model = solver.model()
    model_vals = {
        str(v): _signed(model.eval(v, model_completion=True).as_long(), width)
        for v in vars_
    }
    smt_value = z3.is_true(model.eval(operational, model_completion=True))
    enum_value = _PREDICATES[class_key](width, model_vals)
    return AgreementGroup(
        class_key,
        width,
        len(samples),
        False,
        f"{class_key}@{width}: model={model_vals} smt={smt_value} enum={enum_value}",
    )


def differential_test_smt_vs_enumeration(
    total_cases: int = 10_000,
    seed: int = 0xC1055EED,
    widths: Tuple[int, ...] = SHIPPED_WIDTHS,
) -> AgreementReport:
    """Differential-test the SMT decision path against enumeration semantics.

    The independent Python predicates are the same concrete semantics used by
    the brute-force enumeration path.  For each generated concrete assignment we
    ask Z3 whether the operational encoding's truth value differs from that
    predicate.  A group passes only when the disjunction of *all* sampled
    disagreements is UNSAT, so the 10k-case check is one solver proof per
    (class, width), not 10k uninspected booleans.
    """
    if total_cases <= 0:
        raise ValueError("total_cases must be positive")
    if not widths:
        raise ValueError("at least one width is required")

    rng = random.Random(seed)
    grouped: Dict[Tuple[str, int], List[Dict[str, int]]] = {}
    for i in range(total_cases):
        class_key = CLASSES[i % len(CLASSES)]
        width = widths[(i // len(CLASSES)) % len(widths)]
        grouped.setdefault((class_key, width), []).append(
            _sample_assignment(rng, class_key, width))

    groups = tuple(
        _agreement_group(class_key, width, samples)
        for (class_key, width), samples in sorted(grouped.items())
    )
    return AgreementReport(total_cases=total_cases, seed=seed, groups=groups)
