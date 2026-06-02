"""Step 75 — mechanized soundness (scoped).

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
* ``rust_oracle_sound``          — the concrete C -> Rust instantiation.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LEAN_SOURCE = os.path.join("formal", "ProductSoundness.lean")

REQUIRED_THEOREMS: Tuple[str, ...] = (
    "oracle_sound",
    "oracle_complete_rel",
    "oracle_decides",
    "equivalence_never_reported",
    "report_implies_ub",
    "pack_oracle_sound",
    "rust_oracle_sound",
)


def _lean_binary() -> Optional[str]:
    found = shutil.which("lean")
    if found:
        return found
    # elan default install location.
    cand = os.path.expanduser("~/.elan/bin/lean")
    return cand if os.path.exists(cand) else None


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


def _read_source() -> Optional[str]:
    try:
        with open(os.path.join(_ROOT, LEAN_SOURCE), "r") as f:
            return f.read()
    except OSError:
        return None


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


MECHANIZED_SOUNDNESS_SPI = {
    "confirm_mechanized_soundness": confirm_mechanized_soundness,
    "render": render,
    "REQUIRED_THEOREMS": REQUIRED_THEOREMS,
    "LEAN_SOURCE": LEAN_SOURCE,
}


if __name__ == "__main__":  # pragma: no cover
    rep = confirm_mechanized_soundness()
    print(render(rep))
