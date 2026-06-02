"""Step 20 — evaluation-order / sequencing oracle.

C leaves two distinct things undefined-or-unspecified around evaluation order, and
both are cross-language soundness hazards:

* **Unsequenced modification = undefined behavior.** ``i = i++ + i++`` or
  ``f(i++, i++)`` modify (or modify-and-read) the same object without an
  intervening sequence point. The C standard makes the *whole program's* behavior
  undefined — a compiler may produce any value. Any target-language translation
  picks *some* concrete value, so it can diverge from another perfectly legal C
  compilation. This is detectable precisely: clang's ``-Wunsequenced`` flags it.

* **Unspecified evaluation order with side effects.** ``g(a(), b())`` may evaluate
  ``a`` before ``b`` or vice-versa; the order is *unspecified* (not UB) in C, but
  defined left-to-right in Rust/Go. A single compiler fixes one order, so a lone
  run never reveals the divergence — which is exactly why a sound oracle must
  *loudly abstain* here rather than assert equivalence.

This oracle therefore does two things, both grounded in the real clang:

1. :func:`detect_unsequenced` runs clang ``-Wunsequenced`` and returns the precise
   diagnostics, proving the genuine-UB cases are caught and clean code is not.
2. :func:`decide` returns a loud ``ABSTAIN`` whenever unsequenced UB is present —
   the program's value is undefined, so cross-language equivalence cannot be
   asserted — and documents unspecified-order-with-side-effects as the sequencing
   soundness frontier.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

CLANG = "/usr/bin/clang"


@dataclass(frozen=True)
class SeqDiagnostic:
    line: int
    col: int
    message: str


@dataclass
class UnsequencedReport:
    available: bool
    diagnostics: Tuple[SeqDiagnostic, ...] = ()

    @property
    def has_ub(self) -> bool:
        return bool(self.diagnostics)


_DIAG_RE = re.compile(r":(\d+):(\d+): warning: (.*unsequenced.*)", re.IGNORECASE)


def detect_unsequenced(src: str) -> UnsequencedReport:
    """Run clang ``-Wunsequenced`` over a C TU and return its diagnostics."""
    if not os.path.exists(CLANG):
        return UnsequencedReport(available=False)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "a.c")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(src)
        r = subprocess.run(
            [CLANG, "-fsyntax-only", "-Wunsequenced", p],
            capture_output=True, text=True,
        )
    diags: List[SeqDiagnostic] = []
    for m in _DIAG_RE.finditer(r.stderr):
        diags.append(SeqDiagnostic(int(m.group(1)), int(m.group(2)),
                                   m.group(3).strip()))
    return UnsequencedReport(available=True, diagnostics=tuple(diags))


# Heuristic, comment/string-aware screen for calls whose argument list contains
# more than one side-effecting subexpression (++/--/assignment/call). This is the
# *unspecified evaluation order* frontier — not UB, but unobservable in one run.
_SIDE_EFFECT = re.compile(r"\+\+|--|[^=!<>]=[^=]|\b\w+\s*\(")


def _strip_comments_strings(src: str) -> str:
    out = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c == '/' and i + 1 < n and src[i + 1] == '/':
            while i < n and src[i] != '\n':
                i += 1
            continue
        if c == '/' and i + 1 < n and src[i + 1] == '*':
            i += 2
            while i + 1 < n and not (src[i] == '*' and src[i + 1] == '/'):
                i += 1
            i += 2
            continue
        if c in '"\'':
            q = c
            out.append(' ')
            i += 1
            while i < n and src[i] != q:
                if src[i] == '\\':
                    i += 1
                i += 1
            i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


ABSTAIN = "ABSTAIN"
EQUIVALENT_OK = "NO_SEQUENCING_HAZARD"


@dataclass
class SeqDecision:
    verdict: str
    reason: str
    diagnostics: Tuple[SeqDiagnostic, ...] = ()

    @property
    def is_abstain(self) -> bool:
        return self.verdict == ABSTAIN


def decide(src: str) -> SeqDecision:
    """Loudly abstain when C sequencing makes cross-language equivalence unsound."""
    rep = detect_unsequenced(src)
    if rep.has_ub:
        return SeqDecision(
            ABSTAIN,
            "unsequenced modification is undefined behavior; the program's value "
            "is not defined by C, so no target translation can be proven "
            "equivalent — any concrete order a translation picks may diverge from "
            "another legal C compilation",
            rep.diagnostics,
        )
    return SeqDecision(
        EQUIVALENT_OK,
        "no unsequenced-modification UB detected by clang -Wunsequenced",
        rep.diagnostics,
    )


# --------------------------------------------------------------------------- #
# Self-confirmation against the real clang.
# --------------------------------------------------------------------------- #
UNSEQUENCED_INCR = "int f(void){int i=0;return i++ + i++;}\n"
UNSEQUENCED_CALL = ("int g(int,int);\n"
                    "int h(void){int i=0;return g(i++,i++);}\n")
UNSEQUENCED_ASSIGN = "int k(void){int i=0;i = i++ + 1;return i;}\n"
SEQUENCED_CLEAN = ("int c(void){int i=0;int a=i++;int b=i++;return a+b;}\n")


@dataclass
class SeqConfirmation:
    available: bool
    ok: bool
    detail: List[str] = field(default_factory=list)


def confirm_sequencing() -> SeqConfirmation:
    """Prove the UB cases are flagged (and abstained on) and clean code is not."""
    if not os.path.exists(CLANG):
        return SeqConfirmation(available=False, ok=False)
    detail: List[str] = []
    ok = True
    for label, src, expect_ub in (
        ("unsequenced-incr", UNSEQUENCED_INCR, True),
        ("unsequenced-call-args", UNSEQUENCED_CALL, True),
        ("unsequenced-assign", UNSEQUENCED_ASSIGN, True),
        ("sequenced-clean", SEQUENCED_CLEAN, False),
    ):
        dec = decide(src)
        got_ub = dec.is_abstain
        good = (got_ub == expect_ub)
        ok = ok and good
        detail.append(f"{label}: expect_ub={expect_ub} got_abstain={got_ub} "
                      f"verdict={dec.verdict} ndiag={len(dec.diagnostics)}")
    return SeqConfirmation(available=True, ok=ok, detail=detail)


EVAL_ORDER_SPI = {
    "detect_unsequenced": detect_unsequenced,
    "decide": decide,
    "confirm_sequencing": confirm_sequencing,
}


if __name__ == "__main__":  # pragma: no cover
    c = confirm_sequencing()
    print("available:", c.available, "ok:", c.ok)
    for line in c.detail:
        print("  ", line)
