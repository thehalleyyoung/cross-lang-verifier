"""
k-induction for loops beyond bounded unrolling (100_STEPS step 74).

Every other decision procedure in this project is *bounded*: the SMT search, the
completeness fragments and the CEGAR loop (step 73) all reason about a single
straight-line evaluation. A real translated program, however, contains **loops**,
and a divergence (or a *proof of its absence*) may only manifest after an
unbounded number of iterations. Plain bounded model checking can never *prove*
the safe case — it can only fail to find a counterexample up to its unrolling
depth.

This module closes that gap with textbook **k-induction** over a small
transition-system IR, so a no-divergence claim is no longer limited to "we
unrolled N times and saw nothing":

* **Base case (depth k).** ``init(s0) ∧ trans(s0,s1) ∧ … ∧ trans(s_{k-1},s_k)``
  together with ``¬prop`` at some step. Because it *starts from init*, any model
  is a **genuine reachable counterexample** — a concrete iteration at which the
  loop's undefined behaviour fires. ⟹ ``DIVERGENT`` with a real witness.
* **Inductive step (depth k).** Assume ``prop`` holds for *k* consecutive
  (otherwise arbitrary) states and the transition relation links them; show
  ``prop`` necessarily holds at the ``(k+1)``-th. If this is UNSAT the property
  is **k-inductive** and therefore holds on *every* reachable state, for an
  **unbounded** iteration count. ⟹ ``SAFE`` (no divergence, ever).
* If the base case is UNSAT (no counterexample within *k*) but the step is SAT
  (a *k*-induction counterexample-to-induction exists), the depth is too small:
  **increase k** and retry.

A property that is not inductive at the chosen depth can be *strengthened* with
an auxiliary invariant; the engine first checks each supplied lemma is itself
inductive (init ⟹ lemma, and lemma is preserved by ``trans``) before using it,
so strengthening can never smuggle in an unsound assumption.

The reasoning is over the *source*-language (C-UB) transition relation, hence
pair-agnostic: the same SAFE proof discharges a looping fragment for C→Rust,
C→Go and C→Swift, and a DIVERGENT witness is a concrete loop-trip count that the
ground-truth harness compiles and runs to confirm the real C ``-O0``/``-O2``
builds disagree while the target language stays defined.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import z3

INT32_MAX = (1 << 31) - 1
INT32_MIN = -(1 << 31)


# --- transition-system IR ----------------------------------------------------


State = Dict[str, z3.ArithRef]


@dataclass
class TransitionSystem:
    """A loop as ``(init, trans, prop)`` over integer state variables.

    State is modelled over the mathematical integers (LIA) and the safety
    property pins the machine-representable range explicitly, so an
    out-of-range value is *observable* as a property violation rather than
    silently wrapping — exactly the C signed-overflow UB condition.
    """

    state_vars: Tuple[str, ...]
    init: Callable[[State], z3.BoolRef]
    trans: Callable[[State, State], z3.BoolRef]
    prop: Callable[[State], z3.BoolRef]
    #: human-readable description (for traces / reports).
    label: str = ""
    #: concrete one-step simulator for ground-truth cross-checking. Maps a
    #: concrete state dict to its successor (independent of Z3).
    step_concrete: Optional[Callable[[Dict[str, int]], Dict[str, int]]] = None
    #: concrete initial state for simulation.
    init_concrete: Optional[Dict[str, int]] = None
    #: concrete property for simulation (good == True).
    prop_concrete: Optional[Callable[[Dict[str, int]], bool]] = None

    def fresh(self, k: int) -> State:
        return {v: z3.Int(f"{v}@{k}") for v in self.state_vars}


# --- results -----------------------------------------------------------------


class KIndVerdict(Enum):
    SAFE = "safe"            # property proved for all reachable states (k-inductive)
    DIVERGENT = "divergent"  # a concrete reachable state violates the property
    UNKNOWN = "unknown"      # neither closed within the depth budget

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


@dataclass
class KIndResult:
    verdict: KIndVerdict
    k: int                              # depth at which the verdict was reached
    base_checks: int = 0
    step_checks: int = 0
    #: for DIVERGENT: the trip count (number of loop iterations) to the violation
    #: and the concrete state trace up to and including the violating state.
    witness_depth: Optional[int] = None
    witness_trace: List[Dict[str, int]] = field(default_factory=list)
    aux_invariants_used: List[str] = field(default_factory=list)
    trace: List[str] = field(default_factory=list)

    @property
    def proved_safe(self) -> bool:
        return self.verdict is KIndVerdict.SAFE


# --- the engine --------------------------------------------------------------


def _model_int(m: z3.ModelRef, v: z3.ArithRef) -> int:
    val = m.eval(v, model_completion=True)
    return val.as_long()


def _check_aux_inductive(ts: TransitionSystem,
                        aux: List[Tuple[str, Callable[[State], z3.BoolRef]]]
                        ) -> List[str]:
    """Reject any auxiliary lemma that is not itself 1-inductive.

    A lemma may only be assumed if ``init ⟹ lemma`` and ``lemma(s) ∧ trans(s,s')
    ⟹ lemma(s')``; otherwise conjoining it could mask a real divergence. Returns
    the labels of the lemmas that pass.
    """
    good: List[str] = []
    for name, lemma in aux:
        s0 = ts.fresh(0)
        s1 = ts.fresh(1)
        # init ⟹ lemma
        sol = z3.Solver()
        sol.add(ts.init(s0))
        sol.add(z3.Not(lemma(s0)))
        if sol.check() != z3.unsat:
            continue
        # lemma preserved by trans
        sol = z3.Solver()
        sol.add(lemma(s0))
        sol.add(ts.trans(s0, s1))
        sol.add(z3.Not(lemma(s1)))
        if sol.check() != z3.unsat:
            continue
        good.append(name)
    return good


def prove(ts: TransitionSystem, max_k: int = 12,
          aux: Optional[List[Tuple[str, Callable[[State], z3.BoolRef]]]] = None
          ) -> KIndResult:
    """Decide the safety of ``ts`` by k-induction up to depth ``max_k``."""
    aux = aux or []
    good_aux = _check_aux_inductive(ts, aux)
    aux_by_name = {n: f for n, f in aux if n in good_aux}

    def strengthened_prop(s: State) -> z3.BoolRef:
        clauses = [ts.prop(s)] + [f(s) for f in aux_by_name.values()]
        return z3.And(*clauses) if len(clauses) > 1 else clauses[0]

    base_checks = 0
    step_checks = 0
    trace: List[str] = []

    for k in range(1, max_k + 1):
        states = [ts.fresh(i) for i in range(k + 1)]

        # ── base case: any reachable violation within k steps is genuine ──
        base = z3.Solver()
        base.add(ts.init(states[0]))
        for i in range(k):
            base.add(ts.trans(states[i], states[i + 1]))
        base.add(z3.Or(*[z3.Not(ts.prop(states[i])) for i in range(k + 1)]))
        base_checks += 1
        if base.check() == z3.sat:
            m = base.model()
            depth = next(i for i in range(k + 1)
                         if not _is_true(m, ts.prop(states[i])))
            wtrace = [{v: _model_int(m, states[i][v]) for v in ts.state_vars}
                      for i in range(depth + 1)]
            trace.append(
                f"base@k={k}: reachable counterexample at iteration {depth} "
                f"(state {wtrace[-1]}) — genuine divergence")
            return KIndResult(
                KIndVerdict.DIVERGENT, k, base_checks, step_checks,
                witness_depth=depth, witness_trace=wtrace, trace=trace)

        # ── inductive step: prop preserved across k+1 arbitrary states ──
        step = z3.Solver()
        for i in range(k + 1):
            step.add(strengthened_prop(states[i]))
        for i in range(k + 1):
            step.add(ts.trans(states[i], states[i + 1]) if i < k else True)
        # one more transition to the (k+1)-th state
        s_next = ts.fresh(k + 1)
        step.add(ts.trans(states[k], s_next))
        step.add(z3.Not(strengthened_prop(s_next)))
        step_checks += 1
        if step.check() == z3.unsat:
            trace.append(
                f"step@k={k}: property is {k}-inductive"
                + (f" (with aux {good_aux})" if good_aux else "")
                + " — SAFE for unbounded iterations")
            return KIndResult(
                KIndVerdict.SAFE, k, base_checks, step_checks,
                aux_invariants_used=list(good_aux), trace=trace)

        trace.append(
            f"step@k={k}: counterexample-to-induction exists — deepen to k={k + 1}")

    return KIndResult(KIndVerdict.UNKNOWN, max_k, base_checks, step_checks,
                      aux_invariants_used=list(good_aux), trace=trace)


def _is_true(m: z3.ModelRef, expr: z3.BoolRef) -> bool:
    return z3.is_true(m.eval(expr, model_completion=True))


# --- ground-truth concrete simulation ----------------------------------------


def simulate(ts: TransitionSystem, steps: int
             ) -> Tuple[bool, Optional[int], List[Dict[str, int]]]:
    """Run the transition system concretely for ``steps`` iterations.

    Returns ``(safe, first_violation_depth, trace)``. ``safe`` is False iff the
    concrete property is violated within the horizon. This is the independent
    ground truth that :func:`prove` is checked against in the tests.
    """
    assert ts.step_concrete and ts.init_concrete is not None and ts.prop_concrete, \
        "transition system lacks a concrete simulator"
    s = dict(ts.init_concrete)
    trace = [dict(s)]
    if not ts.prop_concrete(s):
        return False, 0, trace
    for d in range(1, steps + 1):
        s = ts.step_concrete(s)
        trace.append(dict(s))
        if not ts.prop_concrete(s):
            return False, d, trace
    return True, None, trace


# --- concrete loop fragments (instantiations) --------------------------------


def saturating_counter(modulus: int = 1000) -> Tuple[
        TransitionSystem, List[Tuple[str, Callable[[State], z3.BoolRef]]]]:
    """``i = 0; loop { i = (i+1 == M) ? 0 : i+1; }`` — provably overflow-free.

    The no-overflow property alone is *not* inductive (a spurious unreachable
    state ``i = INT_MAX-1`` breaks the step); supplying the range invariant
    ``0 <= i < M`` (itself 1-inductive) makes it 1-inductive — the canonical
    "strengthen to close induction" pattern.
    """
    M = modulus

    def init(s: State) -> z3.BoolRef:
        return s["i"] == 0

    def trans(s: State, s2: State) -> z3.BoolRef:
        nxt = s["i"] + 1
        return s2["i"] == z3.If(nxt == M, z3.IntVal(0), nxt)

    def prop(s: State) -> z3.BoolRef:
        # the loop body computes i+1 in machine int: it must not overflow.
        return s["i"] + 1 <= INT32_MAX

    def invariant(s: State) -> z3.BoolRef:
        return z3.And(s["i"] >= 0, s["i"] < M)

    ts = TransitionSystem(
        ("i",), init, trans, prop, label=f"saturating_counter(M={M})",
        step_concrete=lambda s: {"i": 0 if s["i"] + 1 == M else s["i"] + 1},
        init_concrete={"i": 0},
        prop_concrete=lambda s: s["i"] + 1 <= INT32_MAX,
    )
    return ts, [("0<=i<M", invariant)]


def accumulator_overflow(start: int = INT32_MAX - 2, step: int = 1
                         ) -> TransitionSystem:
    """``acc = start; loop { acc = acc + step; }`` — overflows after a few trips.

    With ``start = INT_MAX-2, step = 1`` the accumulator reaches ``INT_MAX`` (the
    last safe value) and the *next* update overflows, so the base case finds a
    genuine reachable counterexample at a small depth — a concrete loop-trip
    count the ground-truth harness can compile and run.
    """
    def init(s: State) -> z3.BoolRef:
        return s["acc"] == start

    def trans(s: State, s2: State) -> z3.BoolRef:
        return s2["acc"] == s["acc"] + step

    def prop(s: State) -> z3.BoolRef:
        return s["acc"] + step <= INT32_MAX

    return TransitionSystem(
        ("acc",), init, trans, prop,
        label=f"accumulator_overflow(start={start}, step={step})",
        step_concrete=lambda s: {"acc": s["acc"] + step},
        init_concrete={"acc": start},
        prop_concrete=lambda s: s["acc"] + step <= INT32_MAX,
    )


# --- real-compiler confirmation sources --------------------------------------


def accumulator_c_source(start: int, step: int) -> str:
    """A C loop whose trip count comes from argv; it signed-overflows once the
    accumulator passes INT_MAX (undefined behaviour the optimizer may exploit)."""
    return (
        "#include <stdio.h>\n"
        "#include <stdlib.h>\n"
        "#include <limits.h>\n"
        f"int f(int bound){{ int acc = {start}; "
        f"for (int k = 0; k < bound; k++) {{ acc = acc + {step}; }} return acc; }}\n"
        "int main(int argc, char**argv){\n"
        "    if (argc < 2) return 2;\n"
        "    int bound = (int)strtol(argv[1], 0, 10);\n"
        "    printf(\"%d\\n\", f(bound));\n"
        "    return 0;\n"
        "}\n"
    )


def accumulator_rust_source(start: int, step: int) -> str:
    """The idiomatic Rust translation: wrapping arithmetic, fully defined."""
    return (
        f"fn f(bound:i32)->i32 {{ let mut acc:i32 = {start}; "
        f"let mut k=0i32; while k < bound {{ acc = acc.wrapping_add({step}); "
        f"k += 1; }} acc }}\n"
        "fn main(){\n"
        "    let bound:i32 = std::env::args().nth(1).unwrap().parse().unwrap();\n"
        "    println!(\"{}\", f(bound));\n"
        "}\n"
    )


def saturating_c_source(modulus: int) -> str:
    """A C modular counter — defined and deterministic for *any* trip count."""
    return (
        "#include <stdio.h>\n"
        "#include <stdlib.h>\n"
        f"int f(int bound){{ int i = 0; "
        f"for (int k = 0; k < bound; k++) {{ int n = i + 1; i = (n == {modulus}) ? 0 : n; }} "
        "return i; }\n"
        "int main(int argc, char**argv){\n"
        "    if (argc < 2) return 2;\n"
        "    int bound = (int)strtol(argv[1], 0, 10);\n"
        "    printf(\"%d\\n\", f(bound));\n"
        "    return 0;\n"
        "}\n"
    )


def saturating_rust_source(modulus: int) -> str:
    return (
        f"fn f(bound:i32)->i32 {{ let mut i:i32 = 0; let mut k=0i32; "
        f"while k < bound {{ let n = i + 1; i = if n == {modulus} {{ 0 }} else {{ n }}; "
        f"k += 1; }} i }}\n"
        "fn main(){\n"
        "    let bound:i32 = std::env::args().nth(1).unwrap().parse().unwrap();\n"
        "    println!(\"{}\", f(bound));\n"
        "}\n"
    )


def saturating_go_source(modulus: int) -> str:
    return (
        "package main\n"
        "import (\n\t\"fmt\"\n\t\"os\"\n\t\"strconv\"\n)\n"
        "func f(bound int32) int32 {\n"
        "\tvar i int32 = 0\n"
        "\tfor k := int32(0); k < bound; k++ {\n"
        f"\t\tn := i + 1\n\t\tif n == {modulus} {{ i = 0 }} else {{ i = n }}\n"
        "\t}\n\treturn i\n}\n"
        "func main() {\n"
        "\tbound := int32(0)\n"
        "\tif len(os.Args) > 1 { v, _ := strconv.ParseInt(os.Args[1], 10, 32); bound = int32(v) }\n"
        "\tfmt.Println(f(bound))\n"
        "}\n"
    )


def saturating_swift_source(modulus: int) -> str:
    return (
        "import Foundation\n"
        "func f(_ bound: Int32) -> Int32 {\n"
        "    var i: Int32 = 0\n"
        "    var k: Int32 = 0\n"
        "    while k < bound {\n"
        f"        let n = i + 1\n        i = (n == {modulus}) ? 0 : n\n"
        "        k += 1\n    }\n    return i\n}\n"
        "let bound = CommandLine.arguments.count > 1 ? (Int32(CommandLine.arguments[1]) ?? 0) : 0\n"
        "print(f(bound))\n"
    )


_SATURATING_EMITTERS = {
    "rust": saturating_rust_source,
    "go": saturating_go_source,
    "swift": saturating_swift_source,
}


def trips_to_overflow(start: int, step: int) -> int:
    """The smallest loop-trip count at which the C accumulator first overflows."""
    acc = start
    trips = 0
    while INT32_MIN <= acc + step <= INT32_MAX:
        acc += step
        trips += 1
    return trips + 1
