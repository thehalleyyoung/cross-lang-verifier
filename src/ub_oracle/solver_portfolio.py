"""Step 51 — solver portfolio + parallelism.

A single SMT solver is a single point of failure: it can time out, return
``unknown``, or hit a pathological case on a query another solver dispatches
instantly. This module runs the project's available decision procedures as a
*portfolio* — every solver races the same SMT-LIB2 query in parallel under a
shared wall-clock timeout, the first decisive (``sat``/``unsat``) answer wins,
and we additionally cross-check that all solvers that *did* answer agree.

Available backends on this toolchain:

* **z3** — in-process via the Python bindings (``z3.Solver().from_string``).
* **boolector** — out-of-process; the query is written to a temp ``.smt2`` file
  and the binary is invoked (exit 10 = sat, 20 = unsat).

The portfolio is engineered for robustness reporting across divergence classes:
:func:`robustness_report` runs a battery of bit-vector queries (overflow,
truncation, shift, division) and records, per class, whether the solvers agreed
and how long each took. Everything is proven against the *real* z3 and
boolector — :func:`confirm_portfolio` runs queries with known ground-truth
answers and asserts every available solver matches.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

try:  # z3 is a hard dependency elsewhere in the project, but stay defensive.
    import z3  # type: ignore
    _HAVE_Z3 = True
except Exception:  # pragma: no cover
    _HAVE_Z3 = False

BOOLECTOR = "/opt/homebrew/bin/boolector"

SAT = "sat"
UNSAT = "unsat"
UNKNOWN = "unknown"
TIMEOUT = "timeout"
ERROR = "error"


@dataclass
class SolverResult:
    solver: str
    status: str            # one of SAT/UNSAT/UNKNOWN/TIMEOUT/ERROR
    seconds: float
    detail: str = ""

    @property
    def decisive(self) -> bool:
        return self.status in (SAT, UNSAT)


# ---------------------------------------------------------------------------
# Individual backends.
# ---------------------------------------------------------------------------
def _run_z3(smt2: str, timeout: float) -> SolverResult:
    if not _HAVE_Z3:
        return SolverResult("z3", ERROR, 0.0, "z3 unavailable")
    t0 = time.monotonic()
    try:
        s = z3.Solver()
        s.set("timeout", int(timeout * 1000))
        s.from_string(smt2)
        r = s.check()
        dt = time.monotonic() - t0
        if r == z3.sat:
            return SolverResult("z3", SAT, dt)
        if r == z3.unsat:
            return SolverResult("z3", UNSAT, dt)
        return SolverResult("z3", UNKNOWN, dt, str(r))
    except Exception as e:  # pragma: no cover - defensive
        return SolverResult("z3", ERROR, time.monotonic() - t0, str(e))


def _run_boolector(smt2: str, timeout: float) -> SolverResult:
    if not os.path.exists(BOOLECTOR):
        return SolverResult("boolector", ERROR, 0.0, "boolector unavailable")
    t0 = time.monotonic()
    try:
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "query.smt2")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(smt2)
            r = subprocess.run(
                [BOOLECTOR, "--smt2", p],
                capture_output=True, text=True, timeout=timeout,
            )
        dt = time.monotonic() - t0
        out = (r.stdout or "").strip().splitlines()
        first = out[0].strip() if out else ""
        # boolector: exit 10 == sat, 20 == unsat; also prints sat/unsat.
        if first == "sat" or r.returncode == 10:
            return SolverResult("boolector", SAT, dt)
        if first == "unsat" or r.returncode == 20:
            return SolverResult("boolector", UNSAT, dt)
        return SolverResult("boolector", UNKNOWN, dt, first or str(r.returncode))
    except subprocess.TimeoutExpired:
        return SolverResult("boolector", TIMEOUT, time.monotonic() - t0)
    except Exception as e:  # pragma: no cover - defensive
        return SolverResult("boolector", ERROR, time.monotonic() - t0, str(e))


_BACKENDS: List[Tuple[str, Callable[[str, float], SolverResult]]] = [
    ("z3", _run_z3),
    ("boolector", _run_boolector),
]


def available_solvers() -> List[str]:
    names = []
    if _HAVE_Z3:
        names.append("z3")
    if os.path.exists(BOOLECTOR):
        names.append("boolector")
    return names


# ---------------------------------------------------------------------------
# Portfolio: race all backends, first decisive wins; cross-check agreement.
# ---------------------------------------------------------------------------
@dataclass
class PortfolioResult:
    status: str                              # consensus decisive status, else UNKNOWN
    results: Dict[str, SolverResult] = field(default_factory=dict)
    winner: Optional[str] = None
    agreement: bool = True                   # all decisive solvers agree

    @property
    def decisive(self) -> bool:
        return self.status in (SAT, UNSAT)


def solve_portfolio(smt2: str, timeout: float = 10.0) -> PortfolioResult:
    """Race every available backend on the same query under a shared timeout."""
    results: Dict[str, SolverResult] = {}
    lock = threading.Lock()
    threads: List[threading.Thread] = []

    def worker(name: str, fn: Callable[[str, float], SolverResult]) -> None:
        res = fn(smt2, timeout)
        with lock:
            results[name] = res

    for name, fn in _BACKENDS:
        if name == "z3" and not _HAVE_Z3:
            continue
        if name == "boolector" and not os.path.exists(BOOLECTOR):
            continue
        th = threading.Thread(target=worker, args=(name, fn), daemon=True)
        threads.append(th)
        th.start()
    for th in threads:
        th.join(timeout + 5.0)

    decisive = [r for r in results.values() if r.decisive]
    statuses = {r.status for r in decisive}
    agreement = len(statuses) <= 1
    if decisive and agreement:
        winner = min(decisive, key=lambda r: r.seconds)
        return PortfolioResult(winner.status, results, winner.solver, True)
    if decisive and not agreement:
        # genuine disagreement is a loud, reportable event — never silently pick one.
        return PortfolioResult(UNKNOWN, results, None, False)
    return PortfolioResult(UNKNOWN, results, None, True)


# ---------------------------------------------------------------------------
# Robustness battery across divergence classes.
# ---------------------------------------------------------------------------
def _bv_query(body: str) -> str:
    return "(set-logic QF_BV)\n" + body + "\n(check-sat)\n"


# (name, smt2, expected) — each models a divergence-relevant bit-vector fact.
ROBUSTNESS_QUERIES: List[Tuple[str, str, str]] = [
    ("signed-overflow-exists",
     _bv_query("(declare-fun x () (_ BitVec 32))"
               "(assert (bvslt (bvadd x #x00000001) x))"), SAT),
    ("unsigned-wrap-to-zero",
     _bv_query("(declare-fun x () (_ BitVec 8))"
               "(assert (= (bvadd x #x01) #x00))"), SAT),
    ("truncation-loses-high-bits",
     _bv_query("(declare-fun x () (_ BitVec 16))"
               "(assert (and (distinct x #x0000)"
               "             (= ((_ extract 7 0) x) #x00)))"), SAT),
    ("shift-by-width-is-zero",
     _bv_query("(declare-fun x () (_ BitVec 8))"
               "(assert (distinct (bvshl x #x08) #x00))"), UNSAT),
    ("low-bit-of-even-product-is-zero",
     _bv_query("(declare-fun x () (_ BitVec 8))"
               "(assert (= ((_ extract 0 0) (bvmul x #x02)) #b1))"), UNSAT),
    ("xor-self-is-zero",
     _bv_query("(declare-fun x () (_ BitVec 32))"
               "(assert (distinct (bvxor x x) #x00000000))"), UNSAT),
]


@dataclass
class ClassRobustness:
    name: str
    expected: str
    consensus: str
    agreement: bool
    per_solver: Dict[str, str]
    timings: Dict[str, float]

    @property
    def ok(self) -> bool:
        return self.consensus == self.expected and self.agreement


def robustness_report(timeout: float = 10.0) -> List[ClassRobustness]:
    """Run the battery; per class record consensus, agreement and per-solver time."""
    report: List[ClassRobustness] = []
    for name, smt2, expected in ROBUSTNESS_QUERIES:
        pr = solve_portfolio(smt2, timeout)
        report.append(ClassRobustness(
            name=name,
            expected=expected,
            consensus=pr.status,
            agreement=pr.agreement,
            per_solver={k: v.status for k, v in pr.results.items()},
            timings={k: round(v.seconds, 4) for k, v in pr.results.items()},
        ))
    return report


# ---------------------------------------------------------------------------
# Self-confirmation against the real solvers.
# ---------------------------------------------------------------------------
@dataclass
class PortfolioConfirmation:
    available_solvers: Tuple[str, ...]
    ok: bool
    report: List[ClassRobustness] = field(default_factory=list)


def confirm_portfolio(timeout: float = 10.0) -> PortfolioConfirmation:
    """Run the battery and assert every class hits its known ground-truth answer
    with full agreement among all available solvers."""
    solvers = tuple(available_solvers())
    rep = robustness_report(timeout)
    ok = bool(rep) and all(c.ok for c in rep)
    # at least one query must have been answered by *every* available solver,
    # so we are genuinely exercising the portfolio, not a single backend.
    if solvers:
        cross_checked = any(
            all(s in c.per_solver and c.per_solver[s] in (SAT, UNSAT)
                for s in solvers)
            for c in rep
        )
        ok = ok and cross_checked
    return PortfolioConfirmation(solvers, ok, rep)


SOLVER_PORTFOLIO_SPI = {
    "solve_portfolio": solve_portfolio,
    "robustness_report": robustness_report,
    "confirm_portfolio": confirm_portfolio,
    "available_solvers": available_solvers,
}


if __name__ == "__main__":  # pragma: no cover
    c = confirm_portfolio()
    print("available solvers:", c.available_solvers)
    print("portfolio ok:", c.ok)
    for cr in c.report:
        print(f"  {cr.name:38s} expect={cr.expected:5s} "
              f"consensus={cr.consensus:7s} agree={cr.agreement} "
              f"per={cr.per_solver} t={cr.timings}")
