"""
Independent re-execution / ground-truth harness (100_STEPS step 10).

This module is deliberately *separate* from the SMT oracle: its only job is to
take a candidate counterexample and find out what real compilers and real
hardware actually do.  No verdict in this project is allowed to ship unless this
harness independently confirms it.

For the C->Rust UB anchor it:

* compiles the C source THREE ways and runs each on the witness input:
    - ``O0``  : ``clang -O0``                         (naive, UB usually "benign")
    - ``O2``  : ``clang -O2``                          (optimizer may exploit UB)
    - ``san`` : ``clang -O1 -fsanitize=undefined -fno-sanitize-recover=all``
                (traps -> proves the UB is actually *reachable* on this input)
* compiles the Rust source with ``rustc -O`` and runs it on the same input.

A signed-overflow divergence is *confirmed* when, on a fully-defined input:
  1. the sanitizer build traps  (UB is reachable), AND
  2. the O0 and O2 builds disagree  (the UB is *consequential*, not benign), AND
  3. the Rust build produces a single, defined value.

The harness shells out to the system toolchain; if a compiler is missing it
reports ``available=False`` so callers/tests can skip gracefully.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ToolchainStatus:
    cc: Optional[str]
    rustc: Optional[str]
    ubsan: bool

    @property
    def c_available(self) -> bool:
        return self.cc is not None

    @property
    def rust_available(self) -> bool:
        return self.rustc is not None

    @property
    def full(self) -> bool:
        return self.c_available and self.rust_available and self.ubsan


def _find_cc() -> Optional[str]:
    for cand in ("clang", "cc", "gcc"):
        path = shutil.which(cand)
        if path:
            return path
    return None


def _check_ubsan(cc: str) -> bool:
    """Verify the C compiler actually supports -fsanitize=undefined."""
    src = "int main(void){int x=0; return x;}\n"
    with tempfile.TemporaryDirectory() as d:
        cpath = os.path.join(d, "t.c")
        opath = os.path.join(d, "t.out")
        with open(cpath, "w") as f:
            f.write(src)
        try:
            r = subprocess.run(
                [cc, "-fsanitize=undefined", "-fno-sanitize-recover=all", "-o", opath, cpath],
                capture_output=True, timeout=60,
            )
            return r.returncode == 0
        except (subprocess.SubprocessError, OSError):
            return False


def toolchain_available() -> ToolchainStatus:
    cc = _find_cc()
    rustc = shutil.which("rustc")
    ubsan = _check_ubsan(cc) if cc else False
    return ToolchainStatus(cc=cc, rustc=rustc, ubsan=ubsan)


@dataclass
class RunOutcome:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def ub_trapped(self) -> bool:
        """A UBSan-instrumented run that aborted on UB."""
        if "runtime error:" in self.stderr and "UndefinedBehaviorSanitizer" in self.stderr:
            return True
        # SIGABRT from -fno-sanitize-recover shows up as 134 (128+6) or -6.
        return self.returncode in (134, -6) and "runtime error:" in self.stderr

    @property
    def rust_outcome_defined(self) -> bool:
        """A Rust run with a *defined* outcome: a normal exit (0) or a clean
        unwinding panic (101). Both are defined behavior in safe Rust; only a
        timeout or an unexpected signal would be non-defined here."""
        if self.timed_out:
            return False
        return self.returncode in (0, 101)


@dataclass
class ReexecResult:
    available: bool
    divergence_class: str
    inputs: Dict[str, object]
    # mode: "exploited" (UB changes the observable value across opt levels) or
    # "trap_vs_defined" (C is UB on a defined input while Rust is defined).
    mode: str = "exploited"
    c_runs: Dict[str, RunOutcome] = field(default_factory=dict)
    rust_run: Optional[RunOutcome] = None
    ub_reachable: bool = False
    ub_consequential: bool = False
    rust_defined: bool = False
    confirmed: bool = False
    reason: str = ""

    def summary(self) -> str:
        if not self.available:
            return f"[skipped] {self.reason}"
        bits = [
            f"ub_reachable={self.ub_reachable}",
            f"ub_consequential={self.ub_consequential}",
            f"rust_defined={self.rust_defined}",
            f"confirmed={self.confirmed}",
        ]
        return " ".join(bits)


class ReexecHarness:
    """Compiles & runs real C and Rust to confirm a counterexample."""

    def __init__(self, status: Optional[ToolchainStatus] = None, timeout: int = 60):
        self.status = status or toolchain_available()
        self.timeout = timeout

    # ── low-level runners ────────────────────────────────────────────────
    def _run(self, argv: List[str]) -> RunOutcome:
        try:
            r = subprocess.run(argv, capture_output=True, timeout=self.timeout, text=True)
            return RunOutcome(r.returncode, r.stdout.strip(), r.stderr.strip())
        except subprocess.TimeoutExpired:
            return RunOutcome(-1, "", "timeout", timed_out=True)
        except OSError as e:  # pragma: no cover - environment dependent
            return RunOutcome(-1, "", f"oserror: {e}")

    def _compile_c(self, src: str, args: List[str], workdir: str, name: str) -> Optional[str]:
        cpath = os.path.join(workdir, f"{name}.c")
        opath = os.path.join(workdir, f"{name}.out")
        with open(cpath, "w") as f:
            f.write(src)
        r = subprocess.run([self.status.cc, *args, "-o", opath, cpath],
                           capture_output=True, text=True, timeout=self.timeout)
        if r.returncode != 0:
            return None
        return opath

    def _compile_rust(self, src: str, workdir: str, name: str) -> Optional[str]:
        rpath = os.path.join(workdir, f"{name}.rs")
        opath = os.path.join(workdir, f"{name}.out")
        with open(rpath, "w") as f:
            f.write(src)
        r = subprocess.run([self.status.rustc, "-O", "-o", opath, rpath],
                           capture_output=True, text=True, timeout=self.timeout)
        if r.returncode != 0:
            return None
        return opath

    # ── the anchor confirmation: signed-overflow style UB divergence ─────
    def confirm_ub_divergence(
        self,
        c_src: str,
        rust_src: str,
        argv_inputs: List[str],
        divergence_class: str = "signed_overflow",
    ) -> ReexecResult:
        """
        ``c_src`` and ``rust_src`` must each define a program that reads its
        integer arguments from ``argv`` and prints a single integer result.

        Returns a fully-populated :class:`ReexecResult`.
        """
        res = ReexecResult(available=self.status.full,
                           divergence_class=divergence_class,
                           inputs={f"arg{i}": v for i, v in enumerate(argv_inputs)})
        if not self.status.full:
            missing = []
            if not self.status.c_available:
                missing.append("C compiler")
            if not self.status.ubsan:
                missing.append("UBSan")
            if not self.status.rust_available:
                missing.append("rustc")
            res.reason = "toolchain unavailable: " + ", ".join(missing)
            return res

        with tempfile.TemporaryDirectory() as d:
            o0 = self._compile_c(c_src, ["-O0"], d, "c_o0")
            o2 = self._compile_c(c_src, ["-O2"], d, "c_o2")
            san = self._compile_c(
                c_src,
                ["-O1", "-fsanitize=undefined", "-fno-sanitize-recover=all"],
                d, "c_san",
            )
            rs = self._compile_rust(rust_src, d, "rs")
            if not all((o0, o2, san, rs)):
                res.reason = "compilation failed (o0=%s o2=%s san=%s rs=%s)" % (
                    bool(o0), bool(o2), bool(san), bool(rs))
                res.available = False
                return res

            res.c_runs["O0"] = self._run([o0, *argv_inputs])
            res.c_runs["O2"] = self._run([o2, *argv_inputs])
            res.c_runs["san"] = self._run([san, *argv_inputs])
            res.rust_run = self._run([rs, *argv_inputs])

        san_run = res.c_runs["san"]
        o0_run = res.c_runs["O0"]
        o2_run = res.c_runs["O2"]

        res.ub_reachable = san_run.ub_trapped
        # consequential iff the two non-sanitized builds disagree on stdout
        res.ub_consequential = (
            o0_run.returncode == 0 and o2_run.returncode == 0
            and o0_run.stdout != o2_run.stdout
        )
        res.rust_defined = res.rust_run is not None and res.rust_run.returncode == 0
        res.confirmed = res.ub_reachable and res.ub_consequential and res.rust_defined
        if res.confirmed:
            res.reason = (
                f"UB reachable (sanitizer trapped); O0={o0_run.stdout!r} vs "
                f"O2={o2_run.stdout!r} differ; Rust defined={res.rust_run.stdout!r}"
            )
        else:
            res.reason = "not all confirmation conditions met"
        return res

    # ── definedness confirmation: trap-in-C vs defined-in-Rust ───────────
    def confirm_trap_vs_defined(
        self,
        c_src: str,
        rust_src: str,
        argv_inputs: List[str],
        divergence_class: str = "division_by_zero",
    ) -> ReexecResult:
        """
        Confirm a *definedness* divergence: on the same concrete input, the C
        program executes undefined behaviour (the UBSan build traps) while the
        Rust program has a fully defined outcome (a normal exit or a clean,
        deterministic panic).

        Unlike :meth:`confirm_ub_divergence`, this does **not** require the two
        optimisation levels to disagree on stdout — many UB classes (division
        by zero, out-of-range shift, ``INT_MIN / -1``) crash rather than
        silently producing a different value, so the consequential signal is
        the trap itself, not an observable value flip. Rust definedness is
        established by running the Rust binary twice and requiring identical,
        defined outcomes.
        """
        res = ReexecResult(available=self.status.full,
                           divergence_class=divergence_class,
                           mode="trap_vs_defined",
                           inputs={f"arg{i}": v for i, v in enumerate(argv_inputs)})
        if not self.status.full:
            missing = []
            if not self.status.c_available:
                missing.append("C compiler")
            if not self.status.ubsan:
                missing.append("UBSan")
            if not self.status.rust_available:
                missing.append("rustc")
            res.reason = "toolchain unavailable: " + ", ".join(missing)
            return res

        with tempfile.TemporaryDirectory() as d:
            san = self._compile_c(
                c_src,
                ["-O1", "-fsanitize=undefined", "-fno-sanitize-recover=all"],
                d, "c_san",
            )
            o0 = self._compile_c(c_src, ["-O0"], d, "c_o0")
            rs = self._compile_rust(rust_src, d, "rs")
            if not all((san, o0, rs)):
                res.reason = "compilation failed (san=%s o0=%s rs=%s)" % (
                    bool(san), bool(o0), bool(rs))
                res.available = False
                return res

            res.c_runs["san"] = self._run([san, *argv_inputs])
            res.c_runs["O0"] = self._run([o0, *argv_inputs])
            # Determinism check: run the Rust binary twice.
            rust_a = self._run([rs, *argv_inputs])
            rust_b = self._run([rs, *argv_inputs])
            res.rust_run = rust_a

        san_run = res.c_runs["san"]

        res.ub_reachable = san_run.ub_trapped
        rust_deterministic = (
            rust_a.returncode == rust_b.returncode
            and rust_a.stdout == rust_b.stdout
        )
        res.rust_defined = (
            rust_a.rust_outcome_defined and rust_b.rust_outcome_defined
            and rust_deterministic
        )
        # The "consequence" of the divergence is precisely the definedness gap:
        # C is undefined here while Rust is defined.
        res.ub_consequential = res.ub_reachable and res.rust_defined
        res.confirmed = res.ub_reachable and res.rust_defined
        if res.confirmed:
            kind = "value" if rust_a.returncode == 0 else "panic"
            res.reason = (
                f"UB reachable (sanitizer trapped: {san_run.stderr.splitlines()[0] if san_run.stderr else 'trap'!r}); "
                f"Rust defined & deterministic ({kind}, rc={rust_a.returncode}, out={rust_a.stdout!r})"
            )
        else:
            res.reason = (
                f"not confirmed: ub_reachable={res.ub_reachable}, "
                f"rust_defined={res.rust_defined} (deterministic={rust_deterministic})"
            )
        return res
