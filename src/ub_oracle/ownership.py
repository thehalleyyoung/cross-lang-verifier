"""
Ownership / borrow facts from the real Rust borrow checker (100_STEPS step 76).

C and Rust disagree most sharply not about arithmetic but about *aliasing*. C
freely permits two mutable pointers to the same object; Rust's ownership
discipline forbids it (aliasing **xor** mutability). That disagreement is a
machine-checked fact: the Rust borrow checker *rejects* the naive translation of
a mutably-aliasing C function, with a specific diagnostic. A translator must
therefore either restructure the code so the borrows are disjoint/sequential, or
drop to ``unsafe`` raw pointers — which re-admits the C aliasing semantics (and
the divergence risk the rest of this tool hunts for).

This module turns those ownership facts into ground truth. Each
:class:`OwnershipPattern` pairs a C aliasing/move idiom with the *idiomatic safe*
Rust translation and the borrow-checker outcome it should provoke:

* ``two_mut_borrows`` — two ``&mut`` to one value → **rejected**, ``E0499``;
* ``mut_while_shared`` — ``&mut`` taken while a ``&`` is live → **rejected**, ``E0502``;
* ``use_after_move`` — read a value after it was moved → **rejected**, ``E0382``;
* ``sequential_borrows`` — disjoint/sequential borrows → **accepted**;
* ``raw_ptr_aliasing`` — aliasing re-expressed via ``unsafe`` raw pointers →
  **accepted** (the safety obligation moves to the programmer).

:func:`confirm_ownership` compiles the Rust with the real ``rustc`` and reads
back whether the borrow checker accepted it and, on rejection, the exact error
code — so the predicted ownership fact is never asserted, it is *observed*. The
general interface (:data:`OWNERSHIP_INTERFACE`) states how to feed such facts
into equivalence reasoning and how to retarget it to another safety model.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class OwnershipPattern:
    name: str
    #: the C aliasing/move idiom this pattern stands in for.
    c_gloss: str
    #: the idiomatic *safe* Rust translation.
    rust_src: str
    #: True if the borrow checker should accept the safe translation.
    accepts: bool
    #: expected rustc error code on rejection (e.g. "E0499"), or "" if accepted.
    error_code: str
    #: one-line explanation of the ownership consequence.
    consequence: str


PATTERNS: Dict[str, OwnershipPattern] = {
    "two_mut_borrows": OwnershipPattern(
        "two_mut_borrows",
        "C: `int *a = &v; int *b = &v; *a = ...; *b = ...;` — two mutable "
        "pointers alias the same object.",
        "fn main(){ let mut v = vec![1,2,3]; let a = &mut v; let b = &mut v; "
        "a.push(4); b.push(5); }",
        accepts=False, error_code="E0499",
        consequence="aliasing xor mutability: a second &mut is rejected, so the "
                    "translator must sequentialize or use raw pointers."),
    "mut_while_shared": OwnershipPattern(
        "mut_while_shared",
        "C: read through one pointer while writing through another to the same "
        "object.",
        "fn main(){ let mut x = 5; let r = &x; let m = &mut x; *m += 1; "
        "println!(\"{}\", r); }",
        accepts=False, error_code="E0502",
        consequence="a &mut may not coexist with a live &: the read/write "
                    "overlap C allows is rejected."),
    "use_after_move": OwnershipPattern(
        "use_after_move",
        "C: keep using a struct after its contents were logically transferred "
        "(shallow copy + continued use).",
        "fn main(){ let s = String::from(\"hi\"); let t = s; "
        "println!(\"{}\", s); let _ = t; }",
        accepts=False, error_code="E0382",
        consequence="move semantics invalidate the source binding; the "
                    "C-style continued use is rejected."),
    "sequential_borrows": OwnershipPattern(
        "sequential_borrows",
        "C: mutate then read, with non-overlapping lifetimes.",
        "fn main(){ let mut x = 5; { let m = &mut x; *m += 1; } let r = &x; "
        "println!(\"{}\", r); }",
        accepts=True, error_code="",
        consequence="disjoint borrow lifetimes are accepted: a faithful, safe "
                    "translation exists."),
    "raw_ptr_aliasing": OwnershipPattern(
        "raw_ptr_aliasing",
        "C: genuine mutable aliasing that cannot be sequentialized.",
        "fn main(){ let mut x = 5i32; let p = &mut x as *mut i32; "
        "unsafe { *p += 1; *p += 1; } println!(\"{}\", x); }",
        accepts=True, error_code="",
        consequence="re-expressing aliasing via unsafe raw pointers compiles, "
                    "but moves the safety obligation onto the programmer and "
                    "re-admits C aliasing semantics."),
}


def pattern(name: str) -> OwnershipPattern:
    if name not in PATTERNS:
        raise KeyError(f"unknown ownership pattern {name!r}; known: {sorted(PATTERNS)}")
    return PATTERNS[name]


# --- the documented general ownership interface ------------------------------

OWNERSHIP_INTERFACE = {
    "ownership_fact_is_a_checker_verdict":
        "An ownership fact is the target safety-checker's accept/reject verdict "
        "(plus diagnostic) on a candidate translation — ground truth, not a guess.",
    "rejection_forces_a_translation_choice":
        "A rejected safe translation means the source's aliasing/move pattern has "
        "no direct safe analogue; the translator must restructure or use an "
        "escape hatch (unsafe), which equivalence reasoning must then treat as "
        "re-admitting the source's aliasing semantics.",
    "acceptance_licenses_alias_assumptions":
        "An accepted safe translation licenses the target's aliasing guarantees "
        "(e.g. &mut is unique), which the equivalence checker may assume.",
    "retargetable":
        "The interface is parameterized by the target's safety checker; another "
        "target model (e.g. an ownership analysis for a different language) plugs "
        "in by supplying its own accept/reject oracle and diagnostic vocabulary.",
}


# --- real-compiler (rustc borrow-checker) confirmation -----------------------


@dataclass
class OwnershipConfirmation:
    available: bool
    accepted: Optional[bool] = None
    error_code: str = ""
    reason: str = ""
    stderr: str = ""

    def matches(self, expected: OwnershipPattern) -> bool:
        """True iff the observed verdict matches the pattern's prediction."""
        if self.accepted is None:
            return False
        if self.accepted != expected.accepts:
            return False
        if not expected.accepts:
            return self.error_code == expected.error_code
        return True


_ERR_RE = re.compile(r"\b(E\d{4})\b")


def _extract_error_code(stderr: str) -> str:
    m = _ERR_RE.search(stderr)
    return m.group(1) if m else ""


def confirm_ownership(name: str, rustc: str) -> OwnershipConfirmation:
    """Compile a pattern's safe Rust translation and read back the borrow-check
    verdict (accept / reject + error code)."""
    pat = pattern(name)
    with tempfile.TemporaryDirectory() as d:
        rp = os.path.join(d, "o.rs")
        out = os.path.join(d, "o.out")
        with open(rp, "w") as fh:
            fh.write(pat.rust_src + "\n")
        r = subprocess.run([rustc, "--edition", "2021", "-o", out, rp],
                           capture_output=True, text=True, timeout=120)
        accepted = (r.returncode == 0)
        code = "" if accepted else _extract_error_code(r.stderr)
        return OwnershipConfirmation(True, accepted=accepted, error_code=code,
                                     stderr=r.stderr[:400])
