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

from .target_semantics import PACKS, get_pack


@dataclass(frozen=True)
class ToolchainStatus:
    """Availability of the C toolchain + each registered target compiler.

    Target compilers are stored as an immutable tuple of ``(name, path)`` pairs
    discovered from the :data:`~src.ub_oracle.target_semantics.PACKS` registry,
    so adding a target language never changes this dataclass — it is pure data.
    """

    cc: Optional[str]
    ubsan: bool
    targets: Tuple[Tuple[str, Optional[str]], ...] = ()

    def target_path(self, name: str = "rust") -> Optional[str]:
        for n, p in self.targets:
            if n == name:
                return p
        return None

    @property
    def c_available(self) -> bool:
        return self.cc is not None

    def target_available(self, name: str = "rust") -> bool:
        return self.target_path(name) is not None

    # back-compat accessors for the anchor pair.
    @property
    def rust_available(self) -> bool:
        return self.target_available("rust")

    @property
    def go_available(self) -> bool:
        return self.target_available("go")

    @property
    def rustc(self) -> Optional[str]:
        return self.target_path("rust")

    @property
    def go(self) -> Optional[str]:
        return self.target_path("go")

    def full_for(self, name: str = "rust") -> bool:
        """Whether the full pipeline (C + UBSan + the requested target) is
        available. ``full`` is preserved as the Rust-anchor shorthand."""
        return self.c_available and self.ubsan and self.target_available(name)

    @property
    def full(self) -> bool:
        return self.full_for("rust")


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


def _resolve_compiler(pack) -> Optional[str]:
    for cand in pack.compiler_candidates:
        path = shutil.which(cand)
        if path:
            return path
    return None


def toolchain_available() -> ToolchainStatus:
    cc = _find_cc()
    ubsan = _check_ubsan(cc) if cc else False
    targets = tuple((name, _resolve_compiler(pack)) for name, pack in PACKS.items())
    return ToolchainStatus(cc=cc, ubsan=ubsan, targets=targets)


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
        """Back-compat shorthand for the Rust anchor's definedness predicate."""
        return self.target_outcome_defined("rust")

    def target_outcome_defined(self, target_lang: str = "rust") -> bool:
        """Whether this run is a *defined* outcome for the given target
        language, per that target's semantics pack. Each safe target declares
        the set of process return codes (as observed by Python's subprocess)
        that correspond to language-defined behaviour — a value or a guaranteed,
        deterministic abort:

        * Rust : 0 (value) or 101 (clean unwinding panic).
        * Go   : 0 (value) or 2 (runtime panic, e.g. divide-by-zero / OOB).
        * Swift: 0 (value) or -5 (SIGTRAP runtime trap).

        Anything else (timeout, unexpected signal) is treated as non-defined.
        Unknown target names raise loudly rather than defaulting to value-only."""
        if self.timed_out:
            return False
        return self.returncode in get_pack(target_lang).defined_returncodes


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

    def _compile_target(self, src: str, target_lang: str,
                        workdir: str, name: str) -> Optional[str]:
        """Compile a single-file target program using its semantics pack.

        Entirely data-driven: the pack supplies the source suffix, the compiler
        argv and any hermetic build environment, so a new target language needs
        no change here."""
        pack = get_pack(target_lang)
        compiler = self.status.target_path(target_lang)
        if compiler is None:
            return None
        spath = os.path.join(workdir, f"{name}{pack.source_suffix}")
        opath = os.path.join(workdir, f"{name}.out")
        with open(spath, "w") as f:
            f.write(src)
        env = dict(os.environ)
        env.update(pack.compile_env(workdir))
        r = subprocess.run(pack.compile_argv(compiler, spath, opath),
                           capture_output=True, text=True, timeout=self.timeout,
                           env=env)
        if r.returncode != 0:
            return None
        return opath

    @staticmethod
    def _missing_for(status: "ToolchainStatus", target_lang: str) -> List[str]:
        missing = []
        if not status.c_available:
            missing.append("C compiler")
        if not status.ubsan:
            missing.append("UBSan")
        if not status.target_available(target_lang):
            pack = PACKS.get(target_lang)
            missing.append(pack.compiler_candidates[0] if pack else target_lang)
        return missing

    # ── the anchor confirmation: signed-overflow style UB divergence ─────
    def confirm_ub_divergence(
        self,
        c_src: str,
        rust_src: str,
        argv_inputs: List[str],
        divergence_class: str = "signed_overflow",
        target_lang: str = "rust",
    ) -> ReexecResult:
        """
        ``c_src`` and ``rust_src`` must each define a program that reads its
        integer arguments from ``argv`` and prints a single integer result.
        ``rust_src`` is the *target* source; ``target_lang`` selects which
        compiler (``rust`` or ``go``) builds and runs it.

        Returns a fully-populated :class:`ReexecResult`.
        """
        res = ReexecResult(available=self.status.full_for(target_lang),
                           divergence_class=divergence_class,
                           inputs={f"arg{i}": v for i, v in enumerate(argv_inputs)})
        if not self.status.full_for(target_lang):
            res.reason = "toolchain unavailable: " + ", ".join(
                self._missing_for(self.status, target_lang))
            return res

        with tempfile.TemporaryDirectory() as d:
            o0 = self._compile_c(c_src, ["-O0"], d, "c_o0")
            o2 = self._compile_c(c_src, ["-O2"], d, "c_o2")
            san = self._compile_c(
                c_src,
                ["-O1", "-fsanitize=undefined", "-fno-sanitize-recover=all"],
                d, "c_san",
            )
            rs = self._compile_target(rust_src, target_lang, d, "tgt")
            if not all((o0, o2, san, rs)):
                res.reason = "compilation failed (o0=%s o2=%s san=%s tgt=%s)" % (
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
        target_lang: str = "rust",
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
        res = ReexecResult(available=self.status.full_for(target_lang),
                           divergence_class=divergence_class,
                           mode="trap_vs_defined",
                           inputs={f"arg{i}": v for i, v in enumerate(argv_inputs)})
        if not self.status.full_for(target_lang):
            res.reason = "toolchain unavailable: " + ", ".join(
                self._missing_for(self.status, target_lang))
            return res

        with tempfile.TemporaryDirectory() as d:
            san = self._compile_c(
                c_src,
                ["-O1", "-fsanitize=undefined", "-fno-sanitize-recover=all"],
                d, "c_san",
            )
            o0 = self._compile_c(c_src, ["-O0"], d, "c_o0")
            rs = self._compile_target(rust_src, target_lang, d, "tgt")
            if not all((san, o0, rs)):
                res.reason = "compilation failed (san=%s o0=%s tgt=%s)" % (
                    bool(san), bool(o0), bool(rs))
                res.available = False
                return res

            res.c_runs["san"] = self._run([san, *argv_inputs])
            res.c_runs["O0"] = self._run([o0, *argv_inputs])
            # Determinism check: run the target binary twice.
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
            rust_a.target_outcome_defined(target_lang)
            and rust_b.target_outcome_defined(target_lang)
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

    # ── exploited-without-trap: O0 != O2 vs defined Rust ─────────────────
    def confirm_optimizer_exploited(
        self,
        c_src: str,
        rust_src: str,
        argv_inputs: List[str],
        divergence_class: str = "strict_aliasing",
        c_flags_a: Optional[List[str]] = None,
        c_flags_b: Optional[List[str]] = None,
        target_lang: str = "rust",
    ) -> ReexecResult:
        """
        Confirm a divergence for a class that **no sanitizer can trap** (e.g.
        strict-aliasing violations, floating-point contraction, unspecified
        evaluation order): the evidence of source-side under-determinedness is
        that the *same* C source, on the *same* input, produces **different
        observable output under two standard-conforming compilations**
        (``c_flags_a`` vs ``c_flags_b``).

        Two builds of one deterministic program disagreeing is itself a proof
        that the program's result is not fixed by the language (it depends on
        UB/unspecified behaviour the compiler is free to resolve either way),
        whereas the Rust translation runs to a single, deterministic, defined
        result. That gap is the cross-language divergence — established here
        without relying on a sanitizer.

        ``c_flags_a``/``c_flags_b`` default to ``-O0`` vs
        ``-O2 -fstrict-aliasing`` (the optimisation-exploitation pair); an oracle
        whose class is resolved by a different licence (e.g. FP contraction) can
        pass an apt flag pair such as ``-ffp-contract=off`` vs
        ``-ffp-contract=fast``.
        """
        flags_a = c_flags_a if c_flags_a is not None else ["-O0"]
        flags_b = c_flags_b if c_flags_b is not None else ["-O2", "-fstrict-aliasing"]
        res = ReexecResult(available=(self.status.c_available
                                      and self.status.target_available(target_lang)),
                           divergence_class=divergence_class,
                           mode="optimizer_exploited",
                           inputs={f"arg{i}": v for i, v in enumerate(argv_inputs)})
        if not (self.status.c_available and self.status.target_available(target_lang)):
            res.available = False
            pack = PACKS.get(target_lang)
            res.reason = ("toolchain unavailable: needs a C compiler and "
                          + (pack.compiler_candidates[0] if pack else target_lang))
            return res

        with tempfile.TemporaryDirectory() as d:
            o0 = self._compile_c(c_src, flags_a, d, "c_a")
            o2 = self._compile_c(c_src, flags_b, d, "c_b")
            rs = self._compile_target(rust_src, target_lang, d, "tgt")
            if not all((o0, o2, rs)):
                res.reason = "compilation failed (a=%s b=%s rs=%s)" % (
                    bool(o0), bool(o2), bool(rs))
                res.available = False
                return res

            res.c_runs["A"] = self._run([o0, *argv_inputs])
            res.c_runs["B"] = self._run([o2, *argv_inputs])
            rust_a = self._run([rs, *argv_inputs])
            rust_b = self._run([rs, *argv_inputs])
            res.rust_run = rust_a

        o0_run = res.c_runs["A"]
        o2_run = res.c_runs["B"]

        # consequential iff the two builds disagree on stdout (both ran cleanly)
        res.ub_consequential = (
            o0_run.returncode == 0 and o2_run.returncode == 0
            and o0_run.stdout != o2_run.stdout
        )
        # the under-determinedness *is* the reachable source-side divergence
        res.ub_reachable = res.ub_consequential
        rust_deterministic = (
            rust_a.returncode == rust_b.returncode
            and rust_a.stdout == rust_b.stdout
        )
        res.rust_defined = (
            rust_a.target_outcome_defined(target_lang)
            and rust_b.target_outcome_defined(target_lang)
            and rust_deterministic
        )
        res.confirmed = res.ub_consequential and res.rust_defined
        if res.confirmed:
            res.reason = (
                f"same C source under-determined: build_A({' '.join(flags_a)})="
                f"{o0_run.stdout!r} vs build_B({' '.join(flags_b)})="
                f"{o2_run.stdout!r}; Rust defined & deterministic={rust_a.stdout!r}"
            )
        else:
            res.reason = (
                f"not confirmed: A!=B={res.ub_consequential}, "
                f"rust_defined={res.rust_defined} (deterministic={rust_deterministic})"
            )
        return res
