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
    Libc-contract classes (e.g. overlapping ``memcpy``) and static-diagnostic
    classes (e.g. unsequenced side effects) use separate confirmation modes
    because UBSan does not instrument those preconditions.
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
import re
import signal
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .target_semantics import PACKS, get_pack

NO_TOOLCHAIN_ENV = "XLEV_NO_TOOLCHAIN"
_DISABLE_TOOLCHAIN_VALUES = {"1", "true", "yes", "on"}


def toolchain_discovery_disabled() -> bool:
    """Whether tests should pretend no external compiler toolchains exist.

    The cold-start CI gate uses this to run the full toolchain-independent pytest
    phase deterministically on fresh runners that may or may not already have
    clang/rustc/go/swift/etc. installed, then unsets it for one real compiled
    representative sample.
    """
    return os.environ.get(NO_TOOLCHAIN_ENV, "").strip().lower() in _DISABLE_TOOLCHAIN_VALUES


@dataclass(frozen=True)
class ToolchainStatus:
    """Availability of the C toolchain + each registered target compiler.

    Target compilers are stored as an immutable tuple of ``(name, path)`` pairs
    discovered from the :data:`~src.ub_oracle.target_semantics.PACKS` registry,
    so adding a target language never changes this dataclass — it is pure data.
    """

    cc: Optional[str]
    ubsan: bool
    asan: bool = False
    msan: bool = False
    auto_var_init: bool = False
    targets: Tuple[Tuple[str, Optional[str]], ...] = ()
    runners: Tuple[Tuple[str, Optional[str]], ...] = ()

    def target_path(self, name: str = "rust") -> Optional[str]:
        for n, p in self.targets:
            if n == name:
                return p
        return None

    def target_runner_path(self, name: str = "rust") -> Optional[str]:
        for n, p in self.runners:
            if n == name:
                return p
        return None

    @property
    def c_available(self) -> bool:
        return self.cc is not None

    def target_available(self, name: str = "rust") -> bool:
        return self.target_path(name) is not None

    def target_runnable(self, name: str = "rust") -> bool:
        pack = PACKS.get(name)
        if pack is None or not pack.runner_candidates:
            return True
        return self.target_runner_path(name) is not None

    def can_compile(self, lang: str) -> bool:
        """Whether a program in ``lang`` can be built on this host. ``"c"`` maps
        to the C compiler; every other language maps to its semantics-pack
        compiler. This is the single availability predicate used to gate the
        non-C source pairs (e.g. Go->Rust) whose confirmation needs neither the
        C compiler nor UBSan."""
        if lang == "c":
            return self.c_available
        return self.target_available(lang) and self.target_runnable(lang)

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
        return (
            self.c_available
            and self.ubsan
            and self.target_available(name)
            and self.target_runnable(name)
        )

    def full_asan_for(self, name: str = "rust") -> bool:
        """Whether the ASan-backed C + target confirmation path is available."""
        return (
            self.c_available
            and self.asan
            and self.target_available(name)
            and self.target_runnable(name)
        )

    def full_libc_contract_for(self, name: str = "rust") -> bool:
        """Whether the C-libc-contract + target confirmation path is available."""
        return (
            self.c_available
            and self.target_available(name)
            and self.target_runnable(name)
        )

    def full_uninit_padding_for(self, name: str = "rust") -> bool:
        """Whether uninitialized-padding confirmation can run for ``name``.

        MemorySanitizer is the strongest proof object for this class, but it is
        unavailable on common host targets (notably arm64 Darwin).  The fallback
        uses clang's real automatic-variable-initialization modes to prove that
        the same C source's observable hash depends specifically on padding
        bytes while the target serialization is deterministic.
        """
        return (
            self.c_available
            and self.target_available(name)
            and self.target_runnable(name)
            and (self.msan or self.auto_var_init)
        )

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


def _check_asan(cc: str) -> bool:
    """Verify the C compiler actually supports AddressSanitizer."""
    src = "int main(void){return 0;}\n"
    with tempfile.TemporaryDirectory() as d:
        cpath = os.path.join(d, "t.c")
        opath = os.path.join(d, "t.out")
        with open(cpath, "w") as f:
            f.write(src)
        try:
            r = subprocess.run(
                [cc, "-fsanitize=address", "-o", opath, cpath],
                capture_output=True, timeout=60,
            )
            return r.returncode == 0
        except (subprocess.SubprocessError, OSError):
            return False


def _check_msan(cc: str) -> bool:
    """Verify MemorySanitizer both links and traps on a real uninitialized read."""
    src = (
        "#include <stdio.h>\n"
        "__attribute__((noinline)) static int f(void){ int x; return x; }\n"
        "int main(void){ volatile int y = f(); printf(\"%d\\n\", y); return 0; }\n"
    )
    with tempfile.TemporaryDirectory() as d:
        cpath = os.path.join(d, "t.c")
        opath = os.path.join(d, "t.out")
        with open(cpath, "w") as f:
            f.write(src)
        try:
            c = subprocess.run(
                [cc, "-O1", "-g", "-fsanitize=memory",
                 "-fno-sanitize-recover=all", "-o", opath, cpath],
                capture_output=True, text=True, timeout=60,
            )
            if c.returncode != 0:
                return False
            r = subprocess.run([opath], capture_output=True, text=True, timeout=60)
            return (
                r.returncode != 0
                and "MemorySanitizer:" in r.stderr
                and "use-of-uninitialized-value" in r.stderr
            )
        except (subprocess.SubprocessError, OSError):
            return False


def _check_auto_var_init(cc: str) -> bool:
    """Whether clang accepts both automatic-variable-initialization modes."""
    src = "int main(void){return 0;}\n"
    with tempfile.TemporaryDirectory() as d:
        cpath = os.path.join(d, "t.c")
        with open(cpath, "w") as f:
            f.write(src)
        try:
            for mode in ("pattern", "zero"):
                opath = os.path.join(d, f"t_{mode}.out")
                r = subprocess.run(
                    [cc, "-ftrivial-auto-var-init=" + mode, "-o", opath, cpath],
                    capture_output=True, timeout=60,
                )
                if r.returncode != 0:
                    return False
            return True
        except (subprocess.SubprocessError, OSError):
            return False


def _resolve_compiler(pack) -> Optional[str]:
    for cand in pack.compiler_candidates:
        path = shutil.which(cand)
        if path:
            return path
    return None


def _resolve_runner(pack) -> Optional[str]:
    if not pack.runner_candidates:
        return None
    for cand in pack.runner_candidates:
        path = shutil.which(cand)
        if path:
            return path
    return None


def toolchain_available() -> ToolchainStatus:
    if toolchain_discovery_disabled():
        return ToolchainStatus(
            cc=None,
            ubsan=False,
            asan=False,
            msan=False,
            auto_var_init=False,
            targets=tuple((name, None) for name in PACKS),
            runners=tuple((name, None) for name in PACKS),
        )
    cc = _find_cc()
    ubsan = _check_ubsan(cc) if cc else False
    asan = _check_asan(cc) if cc else False
    msan = _check_msan(cc) if cc else False
    auto_var_init = _check_auto_var_init(cc) if cc else False
    targets = tuple((name, _resolve_compiler(pack)) for name, pack in PACKS.items())
    runners = tuple((name, _resolve_runner(pack)) for name, pack in PACKS.items())
    return ToolchainStatus(
        cc=cc,
        ubsan=ubsan,
        asan=asan,
        msan=msan,
        auto_var_init=auto_var_init,
        targets=targets,
        runners=runners,
    )


@dataclass
class RunOutcome:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    resource_exhausted: bool = False
    peak_rss_kb: int = 0

    @property
    def ub_trapped(self) -> bool:
        """A UBSan-instrumented run that aborted on UB."""
        if self.resource_exhausted:
            return False
        if "runtime error:" in self.stderr and "UndefinedBehaviorSanitizer" in self.stderr:
            return True
        # SIGABRT from -fno-sanitize-recover shows up as 134 (128+6) or -6.
        return self.returncode in (134, -6) and "runtime error:" in self.stderr

    @property
    def asan_trapped(self) -> bool:
        """An AddressSanitizer-instrumented run that aborted on a memory fault."""
        if self.resource_exhausted:
            return False
        if "AddressSanitizer:" in self.stderr:
            return True
        # Some libc interceptors print only the diagnostic kind on compact builds.
        if "memcpy-param-overlap" in self.stderr:
            return True
        return False

    @property
    def msan_trapped(self) -> bool:
        """A MemorySanitizer run that aborted on an uninitialized-value use."""
        if self.resource_exhausted:
            return False
        return (
            "MemorySanitizer:" in self.stderr
            and "use-of-uninitialized-value" in self.stderr
        )

    @property
    def libc_contract_trapped(self) -> bool:
        """A checked-libc-contract run that aborted on a modeled C UB precondition."""
        return self.contract_trapped("memcpy-param-overlap")

    def contract_trapped(self, token: str = "clv-contract:") -> bool:
        """A checked-contract run aborted on a modeled source-language precondition."""
        if self.resource_exhausted:
            return False
        return "runtime error:" in self.stderr and token in self.stderr

    @property
    def rust_outcome_defined(self) -> bool:
        """Back-compat shorthand for the Rust anchor's definedness predicate."""
        return self.target_outcome_defined("rust")

    @property
    def ub_category(self) -> str:
        """A normalized UndefinedBehaviorSanitizer diagnostic category.

        UBSan prints ``... runtime error: <message>`` where ``<message>``
        identifies *which* undefined behavior fired (signed overflow, shift too
        large, shift negative, division by zero, ``INT_MIN/-1``, index out of
        bounds, ...).  We strip the operand-specific numbers and quoted types so
        the residual phrase is a stable identity for the UB *kind* — letting the
        counterexample minimizer guarantee that a simplified witness triggers the
        SAME undefined behavior, not merely some other confirmed divergence.

        Returns ``""`` when no UBSan diagnostic is present.
        """
        if "runtime error:" not in self.stderr:
            return ""
        for line in self.stderr.splitlines():
            m = re.search(r"runtime error:\s*(.+)", line)
            if not m:
                continue
            msg = m.group(1)
            msg = re.sub(r"'[^']*'", "", msg)   # drop quoted 'type' names
            msg = re.sub(r"-?\d+", "", msg)      # drop operand-specific numbers
            msg = re.sub(r"\s+", " ", msg).strip()
            return msg
        return ""

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
        if self.timed_out or self.resource_exhausted:
            return False
        if target_lang == "c":
            # C is normally the *source* (UB) side, but the impl-defined classes
            # (bit-field layout, out-of-range enum) are language-*defined* given
            # the implementation: a clean exit (rc 0) is a defined outcome. This
            # lets the defined-vs-different harness compare a defined C program
            # against a defined target program without registering a full pack.
            return self.returncode == 0
        return get_pack(target_lang).is_defined_outcome(
            self.returncode, self.stdout, self.stderr, self.timed_out)


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

    def __init__(
        self,
        status: Optional[ToolchainStatus] = None,
        timeout: int = 60,
        max_rss_mb: Optional[int] = None,
        rss_poll_interval: float = 0.02,
    ):
        if max_rss_mb is not None:
            if max_rss_mb <= 0:
                raise ValueError("max_rss_mb must be positive")
            if shutil.which("ps") is None:
                raise RuntimeError("memory-bounded mode requires the POSIX 'ps' tool")
        if rss_poll_interval <= 0:
            raise ValueError("rss_poll_interval must be positive")
        self.status = status or toolchain_available()
        self.timeout = timeout
        self.max_rss_mb = max_rss_mb
        self.max_rss_kb = max_rss_mb * 1024 if max_rss_mb is not None else None
        self.rss_poll_interval = rss_poll_interval
        self.peak_rss_kb = 0
        self.resource_exhausted = False
        self.resource_exhaustions: List[str] = []

    # ── low-level runners ────────────────────────────────────────────────
    @staticmethod
    def _process_snapshot() -> Tuple[Dict[int, List[int]], Dict[int, int]]:
        try:
            r = subprocess.run(
                ["ps", "-eo", "pid=,ppid=,rss="],
                capture_output=True,
                text=True,
                timeout=2,
            )
        except (OSError, subprocess.SubprocessError):
            return {}, {}
        if r.returncode != 0:
            return {}, {}
        children: Dict[int, List[int]] = {}
        rss: Dict[int, int] = {}
        for line in r.stdout.splitlines():
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                pid = int(parts[0])
                ppid = int(parts[1])
                rss_kb = int(parts[2])
            except ValueError:
                continue
            rss[pid] = max(rss_kb, 0)
            children.setdefault(ppid, []).append(pid)
        return children, rss

    @classmethod
    def _process_tree_pids(cls, root_pid: int) -> List[int]:
        children, _rss = cls._process_snapshot()
        out: List[int] = []
        stack = [root_pid]
        seen = set()
        while stack:
            pid = stack.pop()
            if pid in seen:
                continue
            seen.add(pid)
            out.append(pid)
            stack.extend(children.get(pid, ()))
        return out

    @classmethod
    def _process_tree_rss_kb(cls, root_pid: int) -> int:
        children, rss = cls._process_snapshot()
        total = 0
        stack = [root_pid]
        seen = set()
        while stack:
            pid = stack.pop()
            if pid in seen:
                continue
            seen.add(pid)
            total += rss.get(pid, 0)
            stack.extend(children.get(pid, ()))
        return total

    @classmethod
    def _kill_process_tree(cls, root_pid: int) -> None:
        for pid in reversed(cls._process_tree_pids(root_pid)):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def _subprocess_run(
        self,
        argv: List[str],
        *,
        capture_output: bool,
        timeout: int,
        text: bool,
        env: Optional[Dict[str, str]] = None,
    ) -> subprocess.CompletedProcess:
        if self.max_rss_kb is None:
            r = subprocess.run(
                argv,
                capture_output=capture_output,
                timeout=timeout,
                text=text,
                env=env,
            )
            setattr(r, "resource_exhausted", False)
            setattr(r, "peak_rss_kb", 0)
            return r
        if not capture_output or not text:
            raise ValueError("RSS-bounded runner requires capture_output=True and text=True")

        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        start = time.monotonic()
        peak = self._process_tree_rss_kb(proc.pid)
        exhausted = False
        stdout = ""
        stderr = ""
        while True:
            peak = max(peak, self._process_tree_rss_kb(proc.pid))
            if self.max_rss_kb is not None and peak > self.max_rss_kb:
                exhausted = True
                self._kill_process_tree(proc.pid)
                stdout, stderr = proc.communicate()
                break
            remaining = timeout - (time.monotonic() - start)
            if remaining <= 0:
                self._kill_process_tree(proc.pid)
                stdout, stderr = proc.communicate()
                self.peak_rss_kb = max(self.peak_rss_kb, peak)
                raise subprocess.TimeoutExpired(
                    argv, timeout, output=stdout, stderr=stderr)
            try:
                stdout, stderr = proc.communicate(
                    timeout=min(self.rss_poll_interval, remaining))
                peak = max(peak, self._process_tree_rss_kb(proc.pid))
                break
            except subprocess.TimeoutExpired:
                continue

        self.peak_rss_kb = max(self.peak_rss_kb, peak)
        returncode = proc.returncode
        if exhausted:
            self.resource_exhausted = True
            msg = (
                "resource limit exceeded: observed process-tree RSS "
                f"{peak} KiB > cap {self.max_rss_kb} KiB")
            self.resource_exhaustions.append(f"{argv[0]}: {msg}")
            stderr = ((stderr.rstrip() + "\n") if stderr else "") + msg
            returncode = -signal.SIGKILL

        r = subprocess.CompletedProcess(argv, returncode, stdout or "", stderr or "")
        setattr(r, "resource_exhausted", exhausted)
        setattr(r, "peak_rss_kb", peak)
        return r

    def _run(self, argv: List[str], env: Optional[Dict[str, str]] = None) -> RunOutcome:
        try:
            r = self._subprocess_run(
                argv, capture_output=True, timeout=self.timeout, text=True, env=env)
            return RunOutcome(
                r.returncode,
                r.stdout.strip(),
                r.stderr.strip(),
                resource_exhausted=bool(getattr(r, "resource_exhausted", False)),
                peak_rss_kb=int(getattr(r, "peak_rss_kb", 0)),
            )
        except subprocess.TimeoutExpired:
            return RunOutcome(-1, "", "timeout", timed_out=True)
        except OSError as e:  # pragma: no cover - environment dependent
            return RunOutcome(-1, "", f"oserror: {e}")

    def _compile_c(self, src: str, args: List[str], workdir: str, name: str) -> Optional[str]:
        cpath = os.path.join(workdir, f"{name}.c")
        opath = os.path.join(workdir, f"{name}.out")
        with open(cpath, "w") as f:
            f.write(src)
        r = self._subprocess_run(
            [self.status.cc, *args, "-o", opath, cpath],
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        if r.returncode != 0:
            return None
        return opath

    def _run_target(
        self,
        artifact: str,
        target_lang: str,
        argv_inputs: List[str],
        env: Optional[Dict[str, str]] = None,
    ) -> RunOutcome:
        pack = get_pack(target_lang)
        runner = self.status.target_runner_path(target_lang)
        if pack.runner_candidates:
            if runner is None:
                return RunOutcome(-1, "", f"runtime unavailable for {target_lang}")
            return self._run(pack.run_argv(runner, artifact, argv_inputs), env=env)
        return self._run([artifact, *argv_inputs], env=env)

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
        spath = os.path.join(workdir, f"{name}_source{pack.source_suffix}")
        opath = os.path.join(workdir, f"{name}{pack.artifact_suffix}")
        with open(spath, "w") as f:
            f.write(src)
        env = dict(os.environ)
        env.update(pack.compile_env(workdir))
        r = self._subprocess_run(
            pack.compile_argv(compiler, spath, opath),
            capture_output=True,
            text=True,
            timeout=self.timeout,
            env=env,
        )
        if r.returncode != 0:
            return None
        return opath

    def _compile_lang(self, src: str, lang: str, workdir: str,
                      name: str) -> Optional[str]:
        """Compile ``src`` written in ``lang`` (``"c"`` or any pack language).

        C is built at ``-O2`` (a single, defined optimisation level); every other
        language is built through its semantics pack. Used by the
        ``defined_divergence`` confirmation, where each side is a fully-defined
        program in its own language."""
        if lang == "c":
            return self._compile_c(src, ["-O2"], workdir, name)
        return self._compile_target(src, lang, workdir, name)

    def _run_lang(self, artifact: str, lang: str,
                  argv_inputs: List[str]) -> RunOutcome:
        if lang == "c":
            return self._run([artifact, *argv_inputs])
        return self._run_target(artifact, lang, argv_inputs)

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
        if not status.target_runnable(target_lang):
            pack = PACKS.get(target_lang)
            missing.append(pack.runner_candidates[0] if pack else f"{target_lang} runtime")
        return missing

    @staticmethod
    def _missing_for_asan(status: "ToolchainStatus", target_lang: str) -> List[str]:
        missing = []
        if not status.c_available:
            missing.append("C compiler")
        if not status.asan:
            missing.append("AddressSanitizer")
        if not status.target_available(target_lang):
            pack = PACKS.get(target_lang)
            missing.append(pack.compiler_candidates[0] if pack else target_lang)
        if not status.target_runnable(target_lang):
            pack = PACKS.get(target_lang)
            missing.append(pack.runner_candidates[0] if pack else f"{target_lang} runtime")
        return missing

    @staticmethod
    def _missing_for_libc_contract(status: "ToolchainStatus", target_lang: str) -> List[str]:
        missing = []
        if not status.c_available:
            missing.append("C compiler")
        if not status.target_available(target_lang):
            pack = PACKS.get(target_lang)
            missing.append(pack.compiler_candidates[0] if pack else target_lang)
        if not status.target_runnable(target_lang):
            pack = PACKS.get(target_lang)
            missing.append(pack.runner_candidates[0] if pack else f"{target_lang} runtime")
        return missing

    @staticmethod
    def _missing_for_uninit_padding(status: "ToolchainStatus", target_lang: str) -> List[str]:
        missing = []
        if not status.c_available:
            missing.append("C compiler")
        if not (status.msan or status.auto_var_init):
            missing.append("MemorySanitizer or clang -ftrivial-auto-var-init")
        if not status.target_available(target_lang):
            pack = PACKS.get(target_lang)
            missing.append(pack.compiler_candidates[0] if pack else target_lang)
        if not status.target_runnable(target_lang):
            pack = PACKS.get(target_lang)
            missing.append(pack.runner_candidates[0] if pack else f"{target_lang} runtime")
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
            res.rust_run = self._run_target(rs, target_lang, argv_inputs)

        san_run = res.c_runs["san"]
        o0_run = res.c_runs["O0"]
        o2_run = res.c_runs["O2"]

        res.ub_reachable = san_run.ub_trapped
        # consequential iff the two non-sanitized builds disagree on stdout
        res.ub_consequential = (
            o0_run.returncode == 0 and o2_run.returncode == 0
            and o0_run.stdout != o2_run.stdout
        )
        res.rust_defined = (
            res.rust_run is not None
            and res.rust_run.target_outcome_defined(target_lang)
        )
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
            rust_a = self._run_target(rs, target_lang, argv_inputs)
            rust_b = self._run_target(rs, target_lang, argv_inputs)
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

    # ── ASan-definedness confirmation: C memory UB vs defined target ───────
    def confirm_libc_contract_trap_vs_defined(
        self,
        c_src: str,
        rust_src: str,
        argv_inputs: List[str],
        divergence_class: str = "memcpy_overlap",
        target_lang: str = "rust",
        contract_macro: str = "CLV_CHECK_MEMCPY",
        contract_token: str = "memcpy-param-overlap",
        use_asan: bool = True,
    ) -> ReexecResult:
        """Confirm a C library-precondition divergence that UBSan does not cover.

        Overlapping ``memcpy`` is undefined by the C library contract, but UBSan
        does not instrument that precondition and Apple clang's ASan runtime does
        not reliably report ``memcpy-param-overlap``. This mode therefore runs
        two real C binaries on the witness: an ASan build when available, and a
        contract-sanitized build enabled by ``contract_macro``. The latter is an
        explicit executable check of a C standard precondition (the default is the
        C17 ``memcpy`` non-overlap rule), while the target still runs as an
        ordinary compiled program.
        """
        res = ReexecResult(available=self.status.full_libc_contract_for(target_lang),
                           divergence_class=divergence_class,
                           mode="libc_contract_trap_vs_defined",
                           inputs={f"arg{i}": v for i, v in enumerate(argv_inputs)})
        if not self.status.full_libc_contract_for(target_lang):
            res.reason = "toolchain unavailable: " + ", ".join(
                self._missing_for_libc_contract(self.status, target_lang))
            return res

        with tempfile.TemporaryDirectory() as d:
            asan = None
            if use_asan and self.status.asan:
                asan = self._compile_c(
                    c_src,
                    ["-O1", "-g", "-fsanitize=address", "-fno-omit-frame-pointer",
                     "-fno-builtin-memcpy"],
                    d, "c_asan",
                )
            contract = self._compile_c(
                c_src, ["-O1", f"-D{contract_macro}"], d, "c_contract")
            o0 = self._compile_c(c_src, ["-O0"], d, "c_o0")
            rs = self._compile_target(rust_src, target_lang, d, "tgt")
            if not all((contract, o0, rs)):
                res.reason = "compilation failed (contract=%s o0=%s tgt=%s)" % (
                    bool(contract), bool(o0), bool(rs))
                res.available = False
                return res

            if asan is not None:
                asan_env = dict(os.environ)
                asan_env["ASAN_OPTIONS"] = (
                    "abort_on_error=1:"
                    "detect_stack_use_after_return=0:"
                    + asan_env.get("ASAN_OPTIONS", "")
                )
                res.c_runs["asan"] = self._run([asan, *argv_inputs], env=asan_env)
            res.c_runs["contract"] = self._run([contract, *argv_inputs])
            res.c_runs["O0"] = self._run([o0, *argv_inputs])
            target_a = self._run_target(rs, target_lang, argv_inputs)
            target_b = self._run_target(rs, target_lang, argv_inputs)
            res.rust_run = target_a

        asan_run = res.c_runs.get("asan")
        contract_run = res.c_runs["contract"]
        res.ub_reachable = bool(
            (asan_run is not None and asan_run.asan_trapped)
            or contract_run.contract_trapped(contract_token)
        )
        target_deterministic = (
            target_a.returncode == target_b.returncode
            and target_a.stdout == target_b.stdout
        )
        res.rust_defined = (
            target_a.target_outcome_defined(target_lang)
            and target_b.target_outcome_defined(target_lang)
            and target_deterministic
        )
        res.ub_consequential = res.ub_reachable and res.rust_defined
        res.confirmed = res.ub_reachable and res.rust_defined
        if res.confirmed:
            trapping = asan_run if (asan_run is not None and asan_run.asan_trapped) else contract_run
            first = trapping.stderr.splitlines()[0] if trapping.stderr else "contract trap"
            kind = "value" if target_a.returncode == 0 else "panic"
            res.reason = (
                f"UB reachable (libc contract trapped: {first!r}); "
                f"{target_lang} defined & deterministic "
                f"({kind}, rc={target_a.returncode}, out={target_a.stdout!r})"
            )
        else:
            res.reason = (
                f"not confirmed: contract_reachable={res.ub_reachable}, "
                f"target_defined={res.rust_defined} "
                f"(deterministic={target_deterministic})"
            )
        return res

    def confirm_asan_trap_vs_defined(
        self,
        c_src: str,
        rust_src: str,
        argv_inputs: List[str],
        divergence_class: str = "memcpy_overlap",
        target_lang: str = "rust",
    ) -> ReexecResult:
        """Backward-compatible alias for the libc-contract memory-UB path."""
        return self.confirm_libc_contract_trap_vs_defined(
            c_src, rust_src, argv_inputs, divergence_class, target_lang)

    # ── padding-definedness confirmation: indeterminate C padding vs target ──
    def confirm_uninit_padding_vs_defined(
        self,
        c_src: str,
        target_src: str,
        argv_inputs: List[str],
        divergence_class: str = "uninit_padding",
        target_lang: str = "rust",
    ) -> ReexecResult:
        """Confirm a struct-padding read/serialization divergence.

        The preferred proof object is MemorySanitizer: a field-assigned C struct
        whose padding is copied into a hash must report
        ``use-of-uninitialized-value``, while the same source with explicit
        padding zeroing must not.  On hosts where MSan is unavailable, a real
        clang fallback compiles the same source twice with
        ``-ftrivial-auto-var-init=pattern`` and ``=zero``; a value delta, plus a
        zero-padding control that matches the zero build, proves the observable
        depends specifically on indeterminate padding bytes.  In both cases the
        target program must compile, run twice, and produce one defined,
        deterministic outcome.
        """
        res = ReexecResult(
            available=self.status.full_uninit_padding_for(target_lang),
            divergence_class=divergence_class,
            mode="uninit_padding",
            inputs={f"arg{i}": v for i, v in enumerate(argv_inputs)},
        )
        if not self.status.full_uninit_padding_for(target_lang):
            res.reason = "toolchain unavailable: " + ", ".join(
                self._missing_for_uninit_padding(self.status, target_lang))
            return res

        with tempfile.TemporaryDirectory() as d:
            tgt = self._compile_target(target_src, target_lang, d, "tgt")
            if tgt is None:
                res.available = False
                res.reason = "compilation failed (target=False)"
                return res

            msan_signal = False
            auto_signal = False
            if self.status.msan:
                msan = self._compile_c(
                    c_src,
                    ["-O1", "-g", "-fsanitize=memory",
                     "-fno-sanitize-recover=all"],
                    d,
                    "c_msan",
                )
                msan_clean = self._compile_c(
                    c_src,
                    ["-O1", "-g", "-fsanitize=memory",
                     "-fno-sanitize-recover=all", "-DCLV_ZERO_PADDING"],
                    d,
                    "c_msan_clean",
                )
                if msan is not None and msan_clean is not None:
                    res.c_runs["msan"] = self._run([msan, *argv_inputs])
                    res.c_runs["msan_clean"] = self._run([msan_clean, *argv_inputs])
                    msan_signal = (
                        res.c_runs["msan"].msan_trapped
                        and not res.c_runs["msan_clean"].msan_trapped
                        and res.c_runs["msan_clean"].returncode == 0
                    )

            if self.status.auto_var_init:
                pattern = self._compile_c(
                    c_src,
                    ["-O1", "-ftrivial-auto-var-init=pattern"],
                    d,
                    "c_pattern",
                )
                zero = self._compile_c(
                    c_src,
                    ["-O1", "-ftrivial-auto-var-init=zero"],
                    d,
                    "c_zero",
                )
                clean = self._compile_c(
                    c_src,
                    ["-O1", "-ftrivial-auto-var-init=pattern",
                     "-DCLV_ZERO_PADDING"],
                    d,
                    "c_clean",
                )
                if pattern is not None and zero is not None and clean is not None:
                    res.c_runs["pattern"] = self._run([pattern, *argv_inputs])
                    res.c_runs["zero"] = self._run([zero, *argv_inputs])
                    res.c_runs["zero_padding"] = self._run([clean, *argv_inputs])
                    p = res.c_runs["pattern"]
                    z = res.c_runs["zero"]
                    c = res.c_runs["zero_padding"]
                    auto_signal = (
                        p.returncode == 0
                        and z.returncode == 0
                        and c.returncode == 0
                        and p.stdout != z.stdout
                        and z.stdout == c.stdout
                    )

            target_a = self._run_target(tgt, target_lang, argv_inputs)
            target_b = self._run_target(tgt, target_lang, argv_inputs)
            res.rust_run = target_a

        target_deterministic = (
            target_a.returncode == target_b.returncode
            and target_a.stdout == target_b.stdout
        )
        res.rust_defined = (
            target_a.target_outcome_defined(target_lang)
            and target_b.target_outcome_defined(target_lang)
            and target_deterministic
        )
        res.ub_reachable = msan_signal or auto_signal
        res.ub_consequential = res.ub_reachable and res.rust_defined
        res.confirmed = res.ub_reachable and res.rust_defined
        if res.confirmed:
            signals = []
            if msan_signal:
                signals.append("MSan use-of-uninitialized-value")
            if auto_signal:
                p = res.c_runs["pattern"].stdout
                z = res.c_runs["zero"].stdout
                signals.append(f"auto-init delta pattern={p!r} zero={z!r}")
            res.reason = (
                "uninitialized padding confirmed ("
                + "; ".join(signals)
                + f"); {target_lang} defined & deterministic "
                f"(rc={target_a.returncode}, out={target_a.stdout!r})"
            )
        else:
            res.reason = (
                f"not confirmed: padding_signal={res.ub_reachable} "
                f"(msan={msan_signal}, auto_init={auto_signal}), "
                f"target_defined={res.rust_defined} "
                f"(deterministic={target_deterministic})"
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
        target_ok = (
            self.status.target_available(target_lang)
            and self.status.target_runnable(target_lang)
        )
        res = ReexecResult(available=(self.status.c_available and target_ok),
                           divergence_class=divergence_class,
                           mode="optimizer_exploited",
                           inputs={f"arg{i}": v for i, v in enumerate(argv_inputs)})
        if not (self.status.c_available and target_ok):
            res.available = False
            res.reason = "toolchain unavailable: " + ", ".join(
                self._missing_for_libc_contract(self.status, target_lang))
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
            rust_a = self._run_target(rs, target_lang, argv_inputs)
            rust_b = self._run_target(rs, target_lang, argv_inputs)
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

    # ── defined-but-different: two *safe* languages disagree (e.g. Go->Rust) ──
    def confirm_defined_divergence(
        self,
        src_a: str,
        lang_a: str,
        src_b: str,
        lang_b: str,
        argv_inputs: List[str],
        divergence_class: str = "intmin_div_neg1",
    ) -> ReexecResult:
        """Confirm a *defined-but-different* divergence between two languages that
        are each fully defined on the witnessing input (the safe<->safe case, e.g.
        **Go -> Rust**).

        Unlike the UB modes there is no sanitizer to trap and no optimisation
        level to disagree: *both* programs are language-defined. The evidence of a
        cross-language divergence is that, on the *same* input, the two defined
        programs produce **observably different** behaviour — a different printed
        value, or one returns a value while the other takes a guaranteed,
        deterministic abort (e.g. Go wraps ``INT_MIN/-1`` to a value while Rust
        panics). A program faithfully "translated" from ``lang_a`` to ``lang_b``
        therefore changes meaning.

        Both sides are run **twice** and required to be deterministic before they
        are compared, so a nondeterministic program can never be mistaken for a
        divergence. Only ``stdout`` and the return code are compared; panic
        diagnostics on ``stderr`` (paths, versions, backtraces) are ignored.
        """
        avail = (self.status.can_compile(lang_a)
                 and self.status.can_compile(lang_b))
        res = ReexecResult(available=avail, divergence_class=divergence_class,
                           mode="defined_divergence",
                           inputs={f"arg{i}": v for i, v in enumerate(argv_inputs)})
        if not avail:
            res.reason = (f"toolchain unavailable: needs {lang_a} and {lang_b}")
            return res

        with tempfile.TemporaryDirectory() as d:
            ea = self._compile_lang(src_a, lang_a, d, "a")
            eb = self._compile_lang(src_b, lang_b, d, "b")
            if not (ea and eb):
                res.available = False
                res.reason = "compilation failed (a=%s b=%s)" % (bool(ea), bool(eb))
                return res
            a1 = self._run_lang(ea, lang_a, argv_inputs)
            a2 = self._run_lang(ea, lang_a, argv_inputs)
            b1 = self._run_lang(eb, lang_b, argv_inputs)
            b2 = self._run_lang(eb, lang_b, argv_inputs)

        res.c_runs["A"] = a1
        res.c_runs["B"] = b1
        res.rust_run = b1

        a_det = (a1.returncode == a2.returncode and a1.stdout == a2.stdout)
        b_det = (b1.returncode == b2.returncode and b1.stdout == b2.stdout)
        a_defined = a_det and a1.target_outcome_defined(lang_a)
        b_defined = b_det and b1.target_outcome_defined(lang_b)
        # observable difference: a different value, or value-vs-abort.
        differs = (a1.stdout != b1.stdout) or ((a1.returncode == 0) != (b1.returncode == 0))

        res.rust_defined = a_defined and b_defined           # both sides defined
        res.ub_reachable = differs
        res.ub_consequential = differs
        res.confirmed = a_defined and b_defined and differs
        if res.confirmed:
            res.reason = (
                f"both defined & deterministic but observably differ: "
                f"{lang_a}=(rc={a1.returncode}, out={a1.stdout!r}) vs "
                f"{lang_b}=(rc={b1.returncode}, out={b1.stdout!r})"
            )
        else:
            res.reason = (
                f"not confirmed: {lang_a}_defined={a_defined} "
                f"{lang_b}_defined={b_defined} differs={differs} "
                f"(a_det={a_det}, b_det={b_det})"
            )
        return res

    # ── reverse direction: defined source vs UB-introducing C target ─────────
    def confirm_source_defined_target_ub(
        self,
        source_src: str,
        source_lang: str,
        target_c_src: str,
        argv_inputs: List[str],
        divergence_class: str = "intmin_div_neg1",
    ) -> ReexecResult:
        """Confirm a reverse-pair divergence where a defined source-language
        behaviour becomes undefined behaviour in a C lowering.

        Rust -> C is the motivating case: ``INT_MIN / -1`` is a guaranteed Rust
        panic (a language-defined abort), but the same operation in C is UB. The
        source program must therefore run deterministically to a defined outcome,
        while the translated C target must trap under UBSan on the same input.
        """
        source_ok = self.status.can_compile(source_lang)
        c_ubsan_ok = bool(self.status.c_available and self.status.ubsan)
        res = ReexecResult(
            available=source_ok and c_ubsan_ok,
            divergence_class=divergence_class,
            mode="source_defined_target_ub",
            inputs={f"arg{i}": v for i, v in enumerate(argv_inputs)},
        )
        if not res.available:
            missing = []
            if not source_ok:
                missing.append(source_lang)
            if not self.status.c_available:
                missing.append("C compiler")
            if not self.status.ubsan:
                missing.append("UBSan")
            res.reason = "toolchain unavailable: " + ", ".join(missing)
            return res

        with tempfile.TemporaryDirectory() as d:
            source_exe = self._compile_lang(source_src, source_lang, d, "source")
            target_san = self._compile_c(
                target_c_src,
                ["-O1", "-fsanitize=undefined", "-fno-sanitize-recover=all"],
                d,
                "target_san",
            )
            if not (source_exe and target_san):
                res.available = False
                res.reason = "compilation failed (source=%s target_san=%s)" % (
                    bool(source_exe), bool(target_san))
                return res
            source_a = self._run_lang(source_exe, source_lang, argv_inputs)
            source_b = self._run_lang(source_exe, source_lang, argv_inputs)
            target = self._run([target_san, *argv_inputs])

        res.c_runs["source"] = source_a
        res.c_runs["target_san"] = target
        # Legacy field name; here it records the target-side observed run.
        res.rust_run = target

        source_det = (
            source_a.returncode == source_b.returncode
            and source_a.stdout == source_b.stdout
        )
        source_defined = (
            source_a.target_outcome_defined(source_lang)
            and source_b.target_outcome_defined(source_lang)
            and source_det
        )
        target_ub = target.ub_trapped
        res.rust_defined = source_defined
        res.ub_reachable = target_ub
        res.ub_consequential = source_defined and target_ub
        res.confirmed = source_defined and target_ub
        if res.confirmed:
            first = target.stderr.splitlines()[0] if target.stderr else "UBSan trap"
            res.reason = (
                f"{source_lang} source defined & deterministic "
                f"(rc={source_a.returncode}, out={source_a.stdout!r}); "
                f"C target UB reachable under UBSan ({first!r})"
            )
        else:
            res.reason = (
                f"not confirmed: {source_lang}_defined={source_defined} "
                f"target_ub={target_ub} (source_det={source_det})"
            )
        return res

    # ── static source-UB diagnostic vs defined target ─────────────────────
    def confirm_static_ub_vs_defined(
        self,
        c_src: str,
        target_src: str,
        argv_inputs: List[str],
        divergence_class: str = "eval_order",
        target_lang: str = "rust",
    ) -> ReexecResult:
        """Confirm a C UB class whose proof is a real compiler diagnostic.

        Some C undefined behaviours are compile-time sequencing facts rather than
        runtime sanitizer traps.  In particular, clang diagnoses unsequenced
        modification/access with ``-Wunsequenced`` but UBSan does not emit a
        runtime check.  This mode therefore proves source-side UB by running the
        real C compiler in syntax-check mode and requiring a concrete
        ``-Wunsequenced`` diagnostic, then compiles and runs the target twice to
        prove the translation has one deterministic, language-defined outcome.
        """
        avail = (
            self.status.c_available
            and self.status.target_available(target_lang)
            and self.status.target_runnable(target_lang)
        )
        res = ReexecResult(available=avail, divergence_class=divergence_class,
                           mode="static_ub_vs_defined",
                           inputs={f"arg{i}": v for i, v in enumerate(argv_inputs)})
        if not avail:
            res.reason = "toolchain unavailable: " + ", ".join(
                self._missing_for_libc_contract(self.status, target_lang))
            return res

        with tempfile.TemporaryDirectory() as d:
            cpath = os.path.join(d, "source.c")
            with open(cpath, "w", encoding="utf-8") as f:
                f.write(c_src)
            static = self._run(
                [self.status.cc, "-fsyntax-only", "-Wunsequenced", cpath])
            tgt = self._compile_target(target_src, target_lang, d, "tgt")
            if tgt is None:
                res.available = False
                res.reason = "compilation failed (target=False)"
                return res
            target_a = self._run_target(tgt, target_lang, argv_inputs)
            target_b = self._run_target(tgt, target_lang, argv_inputs)
            res.c_runs["static"] = static
            res.rust_run = target_a

        diag_lines = [
            line for line in static.stderr.splitlines()
            if "warning:" in line.lower()
            and "unsequenced" in line.lower()
            and ("modification" in line.lower() or "access" in line.lower())
        ]
        target_deterministic = (
            target_a.returncode == target_b.returncode
            and target_a.stdout == target_b.stdout
        )
        res.ub_reachable = bool(diag_lines)
        res.rust_defined = (
            target_a.target_outcome_defined(target_lang)
            and target_b.target_outcome_defined(target_lang)
            and target_deterministic
        )
        res.ub_consequential = res.ub_reachable and res.rust_defined
        res.confirmed = res.ub_reachable and res.rust_defined
        if res.confirmed:
            first = diag_lines[0].strip()
            kind = "value" if target_a.returncode == 0 else "panic"
            res.reason = (
                f"C static UB diagnostic observed ({first!r}); "
                f"{target_lang} defined & deterministic "
                f"({kind}, rc={target_a.returncode}, out={target_a.stdout!r})"
            )
        else:
            res.reason = (
                f"not confirmed: static_unsequenced={res.ub_reachable}, "
                f"target_defined={res.rust_defined} "
                f"(deterministic={target_deterministic})"
            )
        return res

    # ── model-level allowed-execution-set gap (atomics litmus tests) ──────
    def confirm_model_level_divergence(
        self,
        c_src: str,
        target_src: str,
        argv_inputs: List[str],
        divergence_class: str = "atomic_ordering",
        target_lang: str = "rust",
    ) -> ReexecResult:
        """Confirm a divergence whose proof object is a bounded model check.

        Some concurrency divergences are differences in *allowed execution sets*,
        not deterministic single-run outputs.  A real relaxed-atomics runtime test
        would be scheduler-dependent and unsuitable for the normal re-execution
        harness.  This mode therefore compiles and runs deterministic model
        checkers written in the source and target languages.  They must use the
        real atomics APIs and agree with the expected tokens:

        * C relaxed model: ``source_relaxed_all_zero=allowed``
        * target SeqCst model: ``target_seq_cst_all_zero=forbidden``

        The result is explicitly marked ``model_level_divergence``; callers must
        not interpret it as a sanitizer trap or a runtime interleaving observed on
        a particular hardware execution.
        """
        avail = (
            self.status.c_available
            and self.status.target_available(target_lang)
            and self.status.target_runnable(target_lang)
        )
        res = ReexecResult(
            available=avail,
            divergence_class=divergence_class,
            mode="model_level_divergence",
            inputs={f"arg{i}": v for i, v in enumerate(argv_inputs)},
        )
        if not avail:
            res.reason = "toolchain unavailable: " + ", ".join(
                self._missing_for_libc_contract(self.status, target_lang))
            return res

        with tempfile.TemporaryDirectory() as d:
            c_exe = self._compile_c(c_src, ["-std=c11", "-O2"], d, "source_model")
            tgt_exe = self._compile_target(target_src, target_lang, d, "target_model")
            if not (c_exe and tgt_exe):
                res.available = False
                res.reason = "compilation failed (source=%s target=%s)" % (
                    bool(c_exe), bool(tgt_exe))
                return res
            c_a = self._run([c_exe, *argv_inputs])
            c_b = self._run([c_exe, *argv_inputs])
            t_a = self._run_target(tgt_exe, target_lang, argv_inputs)
            t_b = self._run_target(tgt_exe, target_lang, argv_inputs)
            res.c_runs["source_model"] = c_a
            res.rust_run = t_a

        c_det = c_a.returncode == c_b.returncode and c_a.stdout == c_b.stdout
        t_det = t_a.returncode == t_b.returncode and t_a.stdout == t_b.stdout
        source_allowed = (
            c_det and c_a.returncode == 0
            and "source_relaxed_all_zero=allowed" in c_a.stdout
        )
        target_forbidden = (
            t_det and t_a.returncode == 0
            and "target_seq_cst_all_zero=forbidden" in t_a.stdout
        )
        res.ub_reachable = source_allowed
        res.ub_consequential = source_allowed and target_forbidden
        res.rust_defined = target_forbidden
        res.confirmed = source_allowed and target_forbidden
        if res.confirmed:
            res.reason = (
                "model-level allowed-set gap confirmed by real compiled snippets: "
                f"C relaxed={c_a.stdout!r}; {target_lang} SeqCst={t_a.stdout!r}"
            )
        else:
            res.reason = (
                f"not confirmed: source_allowed={source_allowed} "
                f"target_forbidden={target_forbidden} "
                f"(source_det={c_det}, target_det={t_det})"
            )
        return res

    # ── public single-shot build+run helpers (used by the N-language
    #    consistency oracle, step 125) ─────────────────────────────────────
    def build_and_run_c(self, c_src: str, flags: List[str],
                        argv_inputs: List[str]) -> Optional[RunOutcome]:
        """Compile ``c_src`` with ``flags`` and run it on ``argv_inputs``.

        Returns the :class:`RunOutcome`, or ``None`` if the C toolchain is
        absent or compilation fails."""
        if not self.status.c_available:
            return None
        with tempfile.TemporaryDirectory() as d:
            exe = self._compile_c(c_src, flags, d, "c_unit")
            if exe is None:
                return None
            return self._run_target(exe, target_lang, argv_inputs)

    def build_and_run_target(self, target_src: str, target_lang: str,
                             argv_inputs: List[str]) -> Optional[RunOutcome]:
        """Compile ``target_src`` for ``target_lang`` and run it on
        ``argv_inputs``.

        Returns the :class:`RunOutcome`, or ``None`` if that target compiler is
        absent or compilation fails."""
        if not self.status.can_compile(target_lang):
            return None
        with tempfile.TemporaryDirectory() as d:
            exe = self._compile_target(target_src, target_lang, d, "tgt")
            if exe is None:
                return None
            return self._run([exe, *argv_inputs])
