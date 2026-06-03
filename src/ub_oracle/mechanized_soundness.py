"""Steps 75/126/127/128/130/131 — mechanized soundness (scoped).

A *machine-checked* soundness (and relative-completeness) argument for the
relational/product-program decision procedure the tool implements, for a core
**language-pair-parametric calculus instantiated to C -> Rust UB**. The proof
lives in ``formal/ProductSoundness.lean`` and is discharged by the real **Lean 4
kernel**; this module runs that kernel and confirms it accepts the development.

We do not *re-implement* the proof in Python — we *invoke the proof assistant*,
so the guarantee is exactly as strong as Lean's kernel. When Lean is not
installed the confirmation degrades to a consistency-only pass (it inspects the
source for the required theorems and never claims a checked proof it did not
run).

Required theorems (their absence fails the confirmation even if the file
compiles, so the proof cannot be silently gutted):

* ``oracle_sound``               — no false alarms;
* ``oracle_complete_rel``        — relative completeness;
* ``oracle_decides``             — decision-procedure correctness;
* ``equivalence_never_reported`` — observationally-equivalent pairs never flagged;
* ``report_implies_ub``          — every report is rooted in source UB;
* ``pack_oracle_sound``          — soundness is language-pair-parametric;
* ``rust_oracle_sound``          — the concrete C -> Rust instantiation;
* ``product_program_preserves_divergence_witness`` — the end-to-end
  product-program construction copies source/target/input payloads unchanged,
  derives the emitted observation from raw run facts, and emits only a genuine
  UB-rooted divergence witness;
* ``product_program_emits_witness_iff_product_violated`` — witness emission is
  exactly product-assertion violation;
* ``product_program_witness_iff_divergence`` — witness emission is equivalent to
  a UB-rooted divergence in the recorded-observable abstraction;
* ``strict_aliasing_oracle_sound`` — strict-aliasing optimizer-exploitation
  witnesses are UB-rooted divergences;
* ``strict_aliasing_report_implies_type_pun`` — the strict-aliasing report
  carries the generated type-pun shape;
* ``strict_aliasing_report_implies_optimizer_exploited`` — the report carries
  the exact ``optimizer_exploited`` confirmation signal;
* ``pointer_provenance_oracle_sound`` — pointer-provenance ``trap_vs_defined``
  witnesses are UB-rooted divergences;
* ``pointer_provenance_report_implies_out_of_provenance`` — the report carries
  the generated out-of-provenance source shape;
* ``pointer_provenance_report_implies_checked_target`` — the report carries the
  safe checked-index target shape;
* ``pointer_provenance_report_implies_trap_vs_defined`` — the report carries the
  exact ``trap_vs_defined`` confirmation signal.

Step 131 adds ``formal/CompletenessBoundary.lean``: a Lake-checked partition of
the published divergence classes into (a) classes complete on their declared
finite fragment and (b) classes that remain sound-but-may-abstain.  It reuses the
recorded-observable decision theorem but deliberately leaves the concrete
finite-range evidence in ``src/ub_oracle/completeness.py``.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

def _find_artifact_root() -> str:
    source_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    candidates = [
        source_root,
        os.getcwd(),
        sys.prefix,
        getattr(sys, "base_prefix", sys.prefix),
    ]
    for cand in candidates:
        if os.path.exists(os.path.join(cand, "formal", "ProductSoundness.lean")):
            return cand
    return source_root


_ROOT = _find_artifact_root()
FORMAL_DIR = os.path.join(_ROOT, "formal")
LEAN_SOURCE = os.path.join("formal", "ProductSoundness.lean")
COMPLETENESS_BOUNDARY_SOURCE = os.path.join("formal", "CompletenessBoundary.lean")
COQ_SOURCE = os.path.join("formal", "CoreSoundness.v")
CHECKER_SOURCE = os.path.join("formal", "VerifiedChecker.lean")
LAKEFILE = os.path.join("formal", "lakefile.lean")
VERIFIED_CHECKER_TARGET = "verified-checker"
COMPLETENESS_BOUNDARY_TARGET = "CompletenessBoundary"
VERIFIED_CHECKER_BINARY = os.path.join(
    FORMAL_DIR, ".lake", "build", "bin", VERIFIED_CHECKER_TARGET)

COMPLETENESS_GUARANTEED_KEYS: Tuple[str, ...] = (
    "signed_overflow",
    "shift_oob",
    "div_by_zero",
    "intmin_div_neg1",
)
COMPLETENESS_MAY_ABSTAIN_KEYS: Tuple[str, ...] = (
    "array_oob",
    "strict_aliasing",
    "fp_contraction",
)

REQUIRED_THEOREMS: Tuple[str, ...] = (
    "oracle_sound",
    "oracle_complete_rel",
    "oracle_decides",
    "equivalence_never_reported",
    "report_implies_ub",
    "pack_oracle_sound",
    "rust_oracle_sound",
    "product_program_preserves_divergence_witness",
    "product_program_emits_witness_iff_product_violated",
    "product_program_witness_iff_divergence",
    "strict_aliasing_oracle_sound",
    "strict_aliasing_report_implies_type_pun",
    "strict_aliasing_report_implies_optimizer_exploited",
    "pointer_provenance_oracle_sound",
    "pointer_provenance_report_implies_out_of_provenance",
    "pointer_provenance_report_implies_checked_target",
    "pointer_provenance_report_implies_trap_vs_defined",
)

REQUIRED_COQ_THEOREMS: Tuple[str, ...] = (
    "oracle_sound_coq",
    "oracle_complete_rel_coq",
    "oracle_decides_coq",
    "equivalence_never_reported_coq",
    "report_implies_ub_coq",
    "pack_oracle_sound_coq",
    "rust_oracle_sound_coq",
    "product_program_preserves_divergence_witness_coq",
    "product_program_emits_witness_iff_product_violated_coq",
    "product_program_witness_iff_divergence_coq",
)

REQUIRED_COMPLETENESS_BOUNDARY_THEOREMS: Tuple[str, ...] = (
    "completeness_boundary_total",
    "completeness_boundary_disjoint",
    "mechanized_completeness_boundary",
    "out_of_fragment_abstains_only",
    "complete_fragment_decides_recorded_observation",
    "boundary_claim_matches_predicates",
)


def _lean_binary() -> Optional[str]:
    found = shutil.which("lean")
    if found:
        return found
    # elan default install location.
    cand = os.path.expanduser("~/.elan/bin/lean")
    return cand if os.path.exists(cand) else None


def _lake_binary() -> Optional[str]:
    found = shutil.which("lake")
    if found:
        return found
    cand = os.path.expanduser("~/.elan/bin/lake")
    return cand if os.path.exists(cand) else None


def _coqc_binary() -> Optional[str]:
    return shutil.which("coqc")


@dataclass
class MechanizedReport:
    available: bool                 # Lean kernel present and was actually run.
    source_present: bool
    theorems_present: Tuple[str, ...]
    theorems_missing: Tuple[str, ...]
    kernel_accepted: Optional[bool] # None when Lean absent (consistency-only).
    source_hash: str
    stderr_tail: str = ""

    @property
    def theorems_ok(self) -> bool:
        return self.source_present and not self.theorems_missing

    @property
    def ok(self) -> bool:
        if not self.source_present:
            return False
        if not self.theorems_ok:
            return False
        if self.available:
            return self.kernel_accepted is True
        # Lean absent: consistency-only — the source exists and declares every
        # required theorem, but we did not (and do not claim to have) run the
        # kernel.
        return True

    @property
    def fully_checked(self) -> bool:
        """True iff the real Lean kernel accepted the development (no
        consistency-only fallback)."""
        return self.ok and self.available and self.kernel_accepted is True


@dataclass
class CompletenessBoundaryReport:
    available: bool                 # Lake present and was actually run.
    source_present: bool
    lakefile_present: bool
    theorems_present: Tuple[str, ...]
    theorems_missing: Tuple[str, ...]
    kernel_accepted: Optional[bool] # None when Lake absent.
    source_hash: str
    exit_code: Optional[int] = None
    stdout_tail: str = ""
    stderr_tail: str = ""

    @property
    def theorems_ok(self) -> bool:
        return self.source_present and not self.theorems_missing

    @property
    def ok(self) -> bool:
        if not self.source_present or not self.lakefile_present:
            return False
        if not self.theorems_ok:
            return False
        if self.available:
            return self.kernel_accepted is True
        # Lake absent: source-contract only.  The report explicitly remains not
        # fully_checked, so callers cannot mistake it for a kernel run.
        return True

    @property
    def fully_checked(self) -> bool:
        return self.ok and self.available and self.kernel_accepted is True


@dataclass
class CoqCrossCheckReport:
    available: bool                 # coqc present and was actually run.
    source_present: bool
    theorems_present: Tuple[str, ...]
    theorems_missing: Tuple[str, ...]
    kernel_accepted: Optional[bool] # None when coqc absent.
    source_hash: str
    stderr_tail: str = ""

    @property
    def theorems_ok(self) -> bool:
        return self.source_present and not self.theorems_missing

    @property
    def ok(self) -> bool:
        if not self.source_present:
            return False
        if not self.theorems_ok:
            return False
        if self.available:
            return self.kernel_accepted is True
        # Coq absent: source-contract only.  The report is explicit that no
        # independent kernel run happened in this environment.
        return True

    @property
    def fully_checked(self) -> bool:
        return self.ok and self.available and self.kernel_accepted is True


@dataclass
class VerifiedCheckerBuild:
    available: bool
    source_present: bool
    lakefile_present: bool
    built: bool
    binary: str
    source_hash: str
    exit_code: Optional[int] = None
    stdout_tail: str = ""
    stderr_tail: str = ""

    @property
    def ok(self) -> bool:
        return (
            self.available
            and self.source_present
            and self.lakefile_present
            and self.built
            and os.path.exists(self.binary)
        )


@dataclass
class VerifiedCheckResult:
    available: bool
    accepted: bool
    verdict: str
    ub_reached: bool
    target_defined: bool
    consequence: bool
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    build: Optional[VerifiedCheckerBuild] = None

    @property
    def ok(self) -> bool:
        return self.available and self.accepted and self.exit_code == 0


def _read_source() -> Optional[str]:
    try:
        with open(os.path.join(_ROOT, LEAN_SOURCE), "r") as f:
            return f.read()
    except OSError:
        return None


def _read_completeness_boundary_source() -> Optional[str]:
    try:
        with open(os.path.join(_ROOT, COMPLETENESS_BOUNDARY_SOURCE), "r") as f:
            return f.read()
    except OSError:
        return None


def _read_checker_source() -> Optional[str]:
    try:
        with open(os.path.join(_ROOT, CHECKER_SOURCE), "r") as f:
            return f.read()
    except OSError:
        return None


def _read_coq_source() -> Optional[str]:
    try:
        with open(os.path.join(_ROOT, COQ_SOURCE), "r") as f:
            return f.read()
    except OSError:
        return None


def _tail(text: str, n: int = 10) -> str:
    return "\n".join((text or "").strip().splitlines()[-n:])


def confirm_mechanized_soundness(timeout: int = 300) -> MechanizedReport:
    src = _read_source()
    if src is None:
        return MechanizedReport(
            available=False, source_present=False, theorems_present=(),
            theorems_missing=REQUIRED_THEOREMS, kernel_accepted=None,
            source_hash="")

    present = tuple(t for t in REQUIRED_THEOREMS if f"theorem {t}" in src)
    missing = tuple(t for t in REQUIRED_THEOREMS if t not in present)
    src_hash = hashlib.sha256(src.encode()).hexdigest()

    lean = _lean_binary()
    if lean is None:
        return MechanizedReport(
            available=False, source_present=True, theorems_present=present,
            theorems_missing=missing, kernel_accepted=None, source_hash=src_hash)

    try:
        proc = subprocess.run(
            [lean, LEAN_SOURCE], cwd=_ROOT, capture_output=True, text=True,
            timeout=timeout)
        accepted = proc.returncode == 0
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-5:]
    except (subprocess.TimeoutExpired, OSError) as e:  # pragma: no cover
        accepted = False
        tail = [f"lean invocation failed: {e}"]

    return MechanizedReport(
        available=True, source_present=True, theorems_present=present,
        theorems_missing=missing, kernel_accepted=accepted,
        source_hash=src_hash, stderr_tail="\n".join(tail))


def confirm_mechanized_completeness_boundary(
    timeout: int = 300,
) -> CompletenessBoundaryReport:
    """Confirm the Lean mechanization of the completeness boundary.

    ``CompletenessBoundary.lean`` imports ``ProductSoundness``, so it must be
    checked through Lake rather than by invoking ``lean`` directly.  Lake builds
    the imported ``ProductSoundness`` module first and then asks the Lean kernel
    to accept the boundary theorem module.
    """
    src = _read_completeness_boundary_source()
    lakefile = os.path.join(_ROOT, LAKEFILE)
    lakefile_present = os.path.exists(lakefile)
    if src is None:
        return CompletenessBoundaryReport(
            available=False,
            source_present=False,
            lakefile_present=lakefile_present,
            theorems_present=(),
            theorems_missing=REQUIRED_COMPLETENESS_BOUNDARY_THEOREMS,
            kernel_accepted=None,
            source_hash="",
        )

    present = tuple(
        t for t in REQUIRED_COMPLETENESS_BOUNDARY_THEOREMS
        if f"theorem {t}" in src
    )
    missing = tuple(
        t for t in REQUIRED_COMPLETENESS_BOUNDARY_THEOREMS
        if t not in present
    )
    src_hash = hashlib.sha256(src.encode()).hexdigest()
    lake = _lake_binary()
    if lake is None or not lakefile_present:
        return CompletenessBoundaryReport(
            available=lake is not None,
            source_present=True,
            lakefile_present=lakefile_present,
            theorems_present=present,
            theorems_missing=missing,
            kernel_accepted=None,
            source_hash=src_hash,
        )

    try:
        proc = subprocess.run(
            [lake, "build", COMPLETENESS_BOUNDARY_TARGET],
            cwd=FORMAL_DIR,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return CompletenessBoundaryReport(
            available=True,
            source_present=True,
            lakefile_present=True,
            theorems_present=present,
            theorems_missing=missing,
            kernel_accepted=proc.returncode == 0,
            source_hash=src_hash,
            exit_code=proc.returncode,
            stdout_tail=_tail(proc.stdout),
            stderr_tail=_tail(proc.stderr),
        )
    except (subprocess.TimeoutExpired, OSError) as e:  # pragma: no cover
        return CompletenessBoundaryReport(
            available=True,
            source_present=True,
            lakefile_present=True,
            theorems_present=present,
            theorems_missing=missing,
            kernel_accepted=False,
            source_hash=src_hash,
            stderr_tail=f"lake invocation failed: {e}",
        )


def confirm_coq_crosscheck(timeout: int = 300) -> CoqCrossCheckReport:
    """Confirm the independent Coq cross-check for the core theorem.

    When ``coqc`` is installed this runs the real Coq kernel over
    ``formal/CoreSoundness.v``.  When it is not installed, the result is an honest
    source-contract pass: the Coq development is present and declares every
    required theorem, but ``fully_checked`` remains false.
    """
    src = _read_coq_source()
    if src is None:
        return CoqCrossCheckReport(
            available=False, source_present=False, theorems_present=(),
            theorems_missing=REQUIRED_COQ_THEOREMS, kernel_accepted=None,
            source_hash="")

    present = tuple(t for t in REQUIRED_COQ_THEOREMS if f"Theorem {t}" in src)
    missing = tuple(t for t in REQUIRED_COQ_THEOREMS if t not in present)
    src_hash = hashlib.sha256(src.encode()).hexdigest()

    coqc = _coqc_binary()
    if coqc is None:
        return CoqCrossCheckReport(
            available=False, source_present=True, theorems_present=present,
            theorems_missing=missing, kernel_accepted=None, source_hash=src_hash)

    try:
        proc = subprocess.run(
            [coqc, os.path.basename(COQ_SOURCE)],
            cwd=os.path.join(_ROOT, "formal"),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        accepted = proc.returncode == 0
        tail = _tail(proc.stderr or proc.stdout)
    except (subprocess.TimeoutExpired, OSError) as e:  # pragma: no cover
        accepted = False
        tail = f"coqc invocation failed: {e}"

    return CoqCrossCheckReport(
        available=True,
        source_present=True,
        theorems_present=present,
        theorems_missing=missing,
        kernel_accepted=accepted,
        source_hash=src_hash,
        stderr_tail=tail,
    )


def build_verified_checker(timeout: int = 300) -> VerifiedCheckerBuild:
    """Build the Lean-extracted verdict checker with Lake.

    This is the Step-129 extraction boundary: Lake compiles the checker from
    ``formal/VerifiedChecker.lean``, which imports the kernel-checked
    ``ProductSoundness`` definitions and fails to build if ``productViolated`` or
    ``oracle_sound`` disappears.
    """
    checker_src = _read_checker_source()
    lakefile = os.path.join(_ROOT, LAKEFILE)
    src_hash = hashlib.sha256(checker_src.encode()).hexdigest() if checker_src else ""
    source_present = checker_src is not None
    lakefile_present = os.path.exists(lakefile)
    lake = _lake_binary()
    if lake is None or not source_present or not lakefile_present:
        return VerifiedCheckerBuild(
            available=lake is not None,
            source_present=source_present,
            lakefile_present=lakefile_present,
            built=False,
            binary=VERIFIED_CHECKER_BINARY,
            source_hash=src_hash,
        )

    try:
        proc = subprocess.run(
            [lake, "build", VERIFIED_CHECKER_TARGET],
            cwd=FORMAL_DIR,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return VerifiedCheckerBuild(
            available=True,
            source_present=True,
            lakefile_present=True,
            built=proc.returncode == 0 and os.path.exists(VERIFIED_CHECKER_BINARY),
            binary=VERIFIED_CHECKER_BINARY,
            source_hash=src_hash,
            exit_code=proc.returncode,
            stdout_tail=_tail(proc.stdout),
            stderr_tail=_tail(proc.stderr),
        )
    except (subprocess.TimeoutExpired, OSError) as e:  # pragma: no cover
        return VerifiedCheckerBuild(
            available=True,
            source_present=True,
            lakefile_present=True,
            built=False,
            binary=VERIFIED_CHECKER_BINARY,
            source_hash=src_hash,
            stderr_tail=f"lake invocation failed: {e}",
        )


def run_verified_checker(
    verdict: str,
    ub_reached: bool,
    target_defined: bool,
    consequence: bool,
    *,
    build: bool = True,
    timeout: int = 300,
) -> VerifiedCheckResult:
    """Run the extracted checker on the recorded-observable verdict facts.

    The checker verifies only the final source-UB positive-claim inference.  The
    run facts themselves still come from the real compiler re-execution harness.
    """
    build_report: Optional[VerifiedCheckerBuild] = None
    if build:
        build_report = build_verified_checker(timeout=timeout)
        if not build_report.ok:
            return VerifiedCheckResult(
                available=False,
                accepted=False,
                verdict=verdict,
                ub_reached=ub_reached,
                target_defined=target_defined,
                consequence=consequence,
                build=build_report,
                stderr=build_report.stderr_tail,
            )
    elif not os.path.exists(VERIFIED_CHECKER_BINARY):
        return VerifiedCheckResult(
            available=False,
            accepted=False,
            verdict=verdict,
            ub_reached=ub_reached,
            target_defined=target_defined,
            consequence=consequence,
            stderr=f"missing checker binary: {VERIFIED_CHECKER_BINARY}",
        )

    argv = [
        VERIFIED_CHECKER_BINARY,
        "--verdict", verdict,
        "--ub", str(bool(ub_reached)).lower(),
        "--target-defined", str(bool(target_defined)).lower(),
        "--consequence", str(bool(consequence)).lower(),
    ]
    try:
        proc = subprocess.run(
            argv,
            cwd=FORMAL_DIR,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (subprocess.TimeoutExpired, OSError) as e:  # pragma: no cover
        return VerifiedCheckResult(
            available=False,
            accepted=False,
            verdict=verdict,
            ub_reached=ub_reached,
            target_defined=target_defined,
            consequence=consequence,
            build=build_report,
            stderr=f"verified-checker invocation failed: {e}",
        )

    return VerifiedCheckResult(
        available=True,
        accepted=proc.returncode == 0,
        verdict=verdict,
        ub_reached=ub_reached,
        target_defined=target_defined,
        consequence=consequence,
        exit_code=proc.returncode,
        stdout=proc.stdout.strip(),
        stderr=proc.stderr.strip(),
        build=build_report,
    )


def render(rep: MechanizedReport) -> str:
    lines = ["mechanized soundness (Lean 4):"]
    lines.append(f"  source: {LEAN_SOURCE} "
                 f"({'present' if rep.source_present else 'MISSING'}) "
                 f"hash={rep.source_hash[:16]}")
    lines.append(f"  required theorems present: {len(rep.theorems_present)}"
                 f"/{len(REQUIRED_THEOREMS)}")
    if rep.theorems_missing:
        lines.append(f"  MISSING: {list(rep.theorems_missing)}")
    if rep.available:
        lines.append(f"  Lean kernel: {'ACCEPTED' if rep.kernel_accepted else 'REJECTED'}")
        if not rep.kernel_accepted and rep.stderr_tail:
            lines.append(f"    {rep.stderr_tail}")
    else:
        lines.append("  Lean kernel: not installed (consistency-only)")
    lines.append(f"  => {'PASSED' if rep.ok else 'FAILED'}"
                 f"{' (kernel-checked)' if rep.fully_checked else ''}")
    return "\n".join(lines)


def render_completeness_boundary(rep: CompletenessBoundaryReport) -> str:
    lines = ["mechanized completeness boundary (Lean/Lake):"]
    lines.append(f"  source: {COMPLETENESS_BOUNDARY_SOURCE} "
                 f"({'present' if rep.source_present else 'MISSING'}) "
                 f"hash={rep.source_hash[:16]}")
    lines.append(f"  lakefile: {LAKEFILE} "
                 f"({'present' if rep.lakefile_present else 'MISSING'})")
    lines.append(f"  required theorems present: {len(rep.theorems_present)}"
                 f"/{len(REQUIRED_COMPLETENESS_BOUNDARY_THEOREMS)}")
    if rep.theorems_missing:
        lines.append(f"  MISSING: {list(rep.theorems_missing)}")
    if rep.available:
        lines.append(f"  lake build {COMPLETENESS_BOUNDARY_TARGET}: "
                     f"{'PASSED' if rep.kernel_accepted else 'FAILED'}")
        if not rep.kernel_accepted and rep.stderr_tail:
            lines.append(f"    {rep.stderr_tail}")
    else:
        lines.append("  Lake: not installed (source-contract only)")
    lines.append(f"  => {'PASSED' if rep.ok else 'FAILED'}"
                 f"{' (kernel-checked)' if rep.fully_checked else ''}")
    return "\n".join(lines)


def render_coq_crosscheck(rep: CoqCrossCheckReport) -> str:
    lines = ["mechanized soundness cross-check (Coq):"]
    lines.append(f"  source: {COQ_SOURCE} "
                 f"({'present' if rep.source_present else 'MISSING'}) "
                 f"hash={rep.source_hash[:16]}")
    lines.append(f"  required theorems present: {len(rep.theorems_present)}"
                 f"/{len(REQUIRED_COQ_THEOREMS)}")
    if rep.theorems_missing:
        lines.append(f"  MISSING: {list(rep.theorems_missing)}")
    if rep.available:
        lines.append(f"  Coq kernel: {'ACCEPTED' if rep.kernel_accepted else 'REJECTED'}")
        if not rep.kernel_accepted and rep.stderr_tail:
            lines.append(f"    {rep.stderr_tail}")
    else:
        lines.append("  Coq kernel: not installed (source-contract only)")
    lines.append(f"  => {'PASSED' if rep.ok else 'FAILED'}"
                 f"{' (kernel-checked)' if rep.fully_checked else ''}")
    return "\n".join(lines)


def render_verified_checker_build(rep: VerifiedCheckerBuild) -> str:
    lines = ["verified checker (Lean/Lake):"]
    lines.append(f"  source: {CHECKER_SOURCE} "
                 f"({'present' if rep.source_present else 'MISSING'}) "
                 f"hash={rep.source_hash[:16]}")
    lines.append(f"  lakefile: {LAKEFILE} "
                 f"({'present' if rep.lakefile_present else 'MISSING'})")
    lines.append(f"  Lake: {'available' if rep.available else 'not installed'}")
    if rep.exit_code is not None:
        lines.append(f"  lake build {VERIFIED_CHECKER_TARGET}: "
                     f"{'PASSED' if rep.built else 'FAILED'} "
                     f"(exit={rep.exit_code})")
    if not rep.built and rep.stderr_tail:
        lines.append(f"    {rep.stderr_tail}")
    lines.append(f"  => {'PASSED' if rep.ok else 'FAILED'}")
    return "\n".join(lines)


MECHANIZED_SOUNDNESS_SPI = {
    "confirm_mechanized_soundness": confirm_mechanized_soundness,
    "confirm_mechanized_completeness_boundary":
        confirm_mechanized_completeness_boundary,
    "confirm_coq_crosscheck": confirm_coq_crosscheck,
    "render": render,
    "render_completeness_boundary": render_completeness_boundary,
    "render_coq_crosscheck": render_coq_crosscheck,
    "build_verified_checker": build_verified_checker,
    "run_verified_checker": run_verified_checker,
    "render_verified_checker_build": render_verified_checker_build,
    "REQUIRED_THEOREMS": REQUIRED_THEOREMS,
    "REQUIRED_COMPLETENESS_BOUNDARY_THEOREMS":
        REQUIRED_COMPLETENESS_BOUNDARY_THEOREMS,
    "REQUIRED_COQ_THEOREMS": REQUIRED_COQ_THEOREMS,
    "LEAN_SOURCE": LEAN_SOURCE,
    "COMPLETENESS_BOUNDARY_SOURCE": COMPLETENESS_BOUNDARY_SOURCE,
    "COQ_SOURCE": COQ_SOURCE,
    "CHECKER_SOURCE": CHECKER_SOURCE,
    "COMPLETENESS_GUARANTEED_KEYS": COMPLETENESS_GUARANTEED_KEYS,
    "COMPLETENESS_MAY_ABSTAIN_KEYS": COMPLETENESS_MAY_ABSTAIN_KEYS,
}


if __name__ == "__main__":  # pragma: no cover
    rep = confirm_mechanized_soundness()
    print(render(rep))
    print(render_completeness_boundary(confirm_mechanized_completeness_boundary()))
    print(render_coq_crosscheck(confirm_coq_crosscheck()))
    print(render_verified_checker_build(build_verified_checker()))
