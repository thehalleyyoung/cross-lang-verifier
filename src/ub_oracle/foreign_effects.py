"""Foreign-effect / soundness-frontier detector (Step 35).

The divergence oracles reason about a *pure*, well-defined fragment of C: integer
and memory semantics whose observable behaviour is a function of the program's
inputs.  Several C constructs step **outside** that fragment — their observable
behaviour depends on state the model does not (and cannot soundly) track:

  * ``volatile`` accesses model memory-mapped I/O / hardware registers: every read
    may yield a different value and every access is an observable side effect, so
    the optimizer is forbidden from coalescing them.  A pure model that folds
    ``*p + *p`` into ``2 * *p`` is simply **wrong** here.
  * inline assembly (``__asm__``) is an opaque blob the compiler treats as a black
    box with programmer-declared clobbers; its semantics are not in the IR.
  * calls to ``extern`` functions with no visible definition (true FFI) have an
    unknown body — the foreign side can do anything permitted by the ABI.
  * atomics / ``_Atomic`` carry a memory-ordering contract about *other threads*
    that a single-thread semantics cannot witness.
  * ``setjmp``/``longjmp`` and signal handlers introduce non-local control flow and
    asynchronous interruption that the straight-line model does not represent.

The right engineering answer is not to guess but to **abstain loudly**: detect
these boundaries, refuse to issue a divergence verdict for the affected unit, and
name exactly which construct (and where) forced the abstention.  This module is
that detector plus a set of *real-compiler* confirmations that each construct is
genuinely opaque to a pure model — e.g. that ``clang`` really does keep four
separate ``load volatile`` where the non-volatile version coalesces to one.
"""

from __future__ import annotations

import enum
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

CC = "/usr/bin/clang"


class ForeignKind(enum.Enum):
    VOLATILE = "volatile"
    INLINE_ASM = "inline_asm"
    FOREIGN_CALL = "foreign_call"
    ATOMIC = "atomic"
    NONLOCAL_JUMP = "nonlocal_jump"
    SIGNAL = "signal"


# Library functions whose effect is foreign / non-local even when declared in a
# standard header (their *bodies* are outside the analysable fragment).
_NONLOCAL_FUNCS = {"setjmp", "sigsetjmp", "longjmp", "siglongjmp", "_setjmp",
                   "_longjmp"}
_SIGNAL_FUNCS = {"signal", "sigaction", "raise", "kill", "pthread_kill"}
# A small allowlist of pure libc functions that must NOT be treated as foreign
# calls just because they are external symbols.
_PURE_LIBC = {
    "abs", "labs", "llabs", "memcpy", "memmove", "memset", "strlen", "strcmp",
    "strncmp", "strcpy", "strncpy", "strcat", "strncat", "memcmp", "snprintf",
    "printf", "puts", "putchar", "malloc", "calloc", "realloc", "free",
    "abort", "exit",
}


@dataclass(frozen=True)
class ForeignSite:
    kind: ForeignKind
    line: int
    snippet: str

    def describe(self) -> str:
        return f"line {self.line}: {self.kind.value} — {self.snippet.strip()!r}"


@dataclass
class FrontierVerdict:
    """Loud abstention record.  ``clear`` is True iff no foreign effect was found.

    When not clear, ``status`` is ``ABSTAIN`` and ``reasons`` enumerates every
    boundary so the caller (and a human) sees *why* no divergence verdict is
    issued — silence here would be unsound.
    """
    clear: bool
    sites: List[ForeignSite] = field(default_factory=list)

    @property
    def status(self) -> str:
        return "CLEAR" if self.clear else "ABSTAIN"

    @property
    def kinds(self) -> Tuple[ForeignKind, ...]:
        # de-duplicated, in first-seen order.
        seen: List[ForeignKind] = []
        for s in self.sites:
            if s.kind not in seen:
                seen.append(s.kind)
        return tuple(seen)

    @property
    def reasons(self) -> List[str]:
        return [s.describe() for s in self.sites]

    def loud_message(self) -> str:
        if self.clear:
            return "CLEAR: no foreign-effect boundary detected; oracle may proceed."
        head = ("ABSTAIN: refusing to issue a divergence verdict — this unit "
                "leaves the analysable pure fragment via:")
        return head + "".join("\n  * " + r for r in self.reasons)


def _strip_comments_and_strings(src: str) -> str:
    """Blank out comments and string/char literals (preserving newlines) so the
    lexical scanner never fires on the word 'volatile' inside a comment."""
    out: List[str] = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        two = src[i:i + 2]
        if two == "//":
            while i < n and src[i] != "\n":
                i += 1
            continue
        if two == "/*":
            i += 2
            while i < n and src[i:i + 2] != "*/":
                out.append("\n" if src[i] == "\n" else " ")
                i += 1
            i += 2
            continue
        if c in "\"'":
            quote = c
            out.append(" ")
            i += 1
            while i < n and src[i] != quote:
                if src[i] == "\\" and i + 1 < n:
                    out.append("  ")
                    i += 2
                    continue
                out.append("\n" if src[i] == "\n" else " ")
                i += 1
            out.append(" ")
            i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


_RE_VOLATILE = re.compile(r"\bvolatile\b")
_RE_ASM = re.compile(r"\b(?:__asm__|__asm|asm)\s*(?:volatile|__volatile__|goto)?\s*\(")
_RE_ATOMIC = re.compile(r"\b_Atomic\b|\batomic_[a-z_]+\b|\bstdatomic\.h\b")
_RE_CALL = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
_RE_DECL_KEYWORDS = re.compile(
    r"\b(?:if|for|while|switch|return|sizeof|_Alignof|alignof|defined)\b")


def scan_c_source(src: str) -> List[ForeignSite]:
    """Lexically detect foreign-effect boundaries in C source.

    Comments and string/char literals are removed first.  Detection is
    intentionally conservative: it would rather flag (abstain) than miss a
    genuine boundary, but it must not fire on the pure fragment, so plain libc
    calls and language keywords are excluded from the foreign-call rule.
    """
    clean = _strip_comments_and_strings(src)
    sites: List[ForeignSite] = []
    lines = clean.splitlines()
    raw_lines = src.splitlines()

    def snip(idx: int) -> str:
        return raw_lines[idx] if idx < len(raw_lines) else lines[idx]

    for idx, line in enumerate(lines, start=0):
        ln = idx + 1
        if _RE_ASM.search(line):
            sites.append(ForeignSite(ForeignKind.INLINE_ASM, ln, snip(idx)))
        if _RE_VOLATILE.search(line):
            sites.append(ForeignSite(ForeignKind.VOLATILE, ln, snip(idx)))
        if _RE_ATOMIC.search(line):
            sites.append(ForeignSite(ForeignKind.ATOMIC, ln, snip(idx)))
        for m in _RE_CALL.finditer(line):
            name = m.group(1)
            # skip control keywords that look like calls.
            if _RE_DECL_KEYWORDS.match(name):
                continue
            if name in _NONLOCAL_FUNCS:
                sites.append(ForeignSite(ForeignKind.NONLOCAL_JUMP, ln, snip(idx)))
            elif name in _SIGNAL_FUNCS:
                sites.append(ForeignSite(ForeignKind.SIGNAL, ln, snip(idx)))

    # FFI: an `extern` function declaration (not a definition) signals a body the
    # analyser cannot see.  Detect `extern <type> name(...)` ending in `;`.
    for idx, line in enumerate(lines, start=0):
        if re.search(r"\bextern\b", line) and "{" not in line:
            m = re.search(r"\bextern\b[^;{]*?\b([A-Za-z_]\w*)\s*\([^;{]*\)\s*;",
                          line)
            if m and m.group(1) not in _PURE_LIBC:
                sites.append(ForeignSite(ForeignKind.FOREIGN_CALL, idx + 1,
                                         raw_lines[idx] if idx < len(raw_lines)
                                         else line))
    sites.sort(key=lambda s: (s.line, s.kind.value))
    return sites


def decide(src: str) -> FrontierVerdict:
    """Top-level entry: scan and produce a loud CLEAR/ABSTAIN verdict."""
    sites = scan_c_source(src)
    return FrontierVerdict(clear=not sites, sites=sites)


# --------------------------------------------------------------------------- #
# Real-compiler confirmations that each construct is genuinely opaque to a pure
# model.  These compile the construct to LLVM IR with clang -O2 and check the
# observable opacity fact, *and* that the lexical scanner flagged it.
# --------------------------------------------------------------------------- #

def _emit_ir(src: str) -> Optional[str]:
    if not os.path.exists(CC):
        return None
    with tempfile.TemporaryDirectory() as d:
        cpath = os.path.join(d, "a.c")
        with open(cpath, "w") as f:
            f.write(src)
        r = subprocess.run(
            [CC, "-O2", "-S", "-emit-llvm", "-o", "-", cpath],
            capture_output=True, text=True)
        if r.returncode != 0:
            return None
        return r.stdout


@dataclass
class OpacityConfirmation:
    kind: ForeignKind
    detected: bool          # did the lexical scanner flag it?
    opaque_in_ir: bool      # did real clang IR confirm it is opaque to a pure model?
    detail: str

    @property
    def ok(self) -> bool:
        return self.detected and self.opaque_in_ir


def confirm_volatile_opaque() -> Optional[OpacityConfirmation]:
    """A pure model would fold four reads of ``*p`` into one load.  With
    ``volatile`` the real compiler keeps all four — proving the fold is unsound."""
    vol = "int f(volatile int*p){int s=0;for(int i=0;i<4;i++)s+=*p;return s;}"
    pure = "int f(int*p){int s=0;for(int i=0;i<4;i++)s+=*p;return s;}"
    vir, pir = _emit_ir(vol), _emit_ir(pure)
    if vir is None or pir is None:
        return None
    vcount = vir.count("load volatile")
    pcount = pir.count(" load ")
    opaque = vcount >= 4 and pcount < vcount
    detected = any(s.kind is ForeignKind.VOLATILE for s in scan_c_source(vol))
    return OpacityConfirmation(
        ForeignKind.VOLATILE, detected, opaque,
        f"volatile loads kept={vcount}, pure loads={pcount}")


def confirm_inline_asm_opaque() -> Optional[OpacityConfirmation]:
    src = 'int f(int x){int y;__asm__("mov %1,%0":"=r"(y):"r"(x));return y;}'
    ir = _emit_ir(src)
    if ir is None:
        return None
    opaque = bool(re.search(r"call\s+i32\s+asm", ir))
    detected = any(s.kind is ForeignKind.INLINE_ASM for s in scan_c_source(src))
    return OpacityConfirmation(
        ForeignKind.INLINE_ASM, detected, opaque,
        "inline asm appears as opaque `call ... asm` in IR")


def confirm_foreign_call_opaque() -> Optional[OpacityConfirmation]:
    src = "extern int g(int);\nint f(int x){return g(x)+1;}"
    ir = _emit_ir(src)
    if ir is None:
        return None
    opaque = any(l.startswith("declare") and "@g(" in l for l in ir.splitlines())
    detected = any(s.kind is ForeignKind.FOREIGN_CALL for s in scan_c_source(src))
    return OpacityConfirmation(
        ForeignKind.FOREIGN_CALL, detected, opaque,
        "extern callee is an undefined `declare` (no visible body)")


def confirm_atomic_opaque() -> Optional[OpacityConfirmation]:
    src = "#include <stdatomic.h>\nint f(_Atomic int*p){return atomic_load(p);}"
    ir = _emit_ir(src)
    if ir is None:
        return None
    opaque = "load atomic" in ir
    detected = any(s.kind is ForeignKind.ATOMIC for s in scan_c_source(src))
    return OpacityConfirmation(
        ForeignKind.ATOMIC, detected, opaque,
        "atomic access lowers to `load atomic` carrying a cross-thread contract")


def confirm_all() -> List[OpacityConfirmation]:
    out: List[OpacityConfirmation] = []
    for fn in (confirm_volatile_opaque, confirm_inline_asm_opaque,
               confirm_foreign_call_opaque, confirm_atomic_opaque):
        c = fn()
        if c is not None:
            out.append(c)
    return out


# The documented soundness frontier per language pair: which foreign effects each
# target *can* express (so a faithful translation must preserve them) and why the
# oracle abstains rather than reasons about them.
FOREIGN_FRONTIER: Dict[str, Dict[str, str]] = {
    "volatile": {
        "c": "qualifier `volatile`; every access is an observable side effect",
        "rust": "`core::ptr::read_volatile`/`write_volatile`",
        "go": "no direct equivalent (use sync/atomic or unsafe + compiler "
              "barriers); a faithful port must not constant-fold the accesses",
        "frontier": "pure value semantics is unsound; abstain",
    },
    "inline_asm": {
        "c": "`__asm__`/`asm`",
        "rust": "`core::arch::asm!`",
        "go": "Plan9 assembly in `.s` files",
        "frontier": "opaque to IR-level reasoning; abstain",
    },
    "foreign_call": {
        "c": "`extern` declaration, body in another TU/library",
        "rust": "`extern \"C\"` block / FFI",
        "go": "cgo / `//go:linkname`",
        "frontier": "callee body unknown; abstain unless both sides bind the "
                    "same symbol with a matching ABI",
    },
    "atomic": {
        "c": "`_Atomic`/`<stdatomic.h>`",
        "rust": "`core::sync::atomic`",
        "go": "`sync/atomic`",
        "frontier": "single-thread semantics cannot witness the ordering "
                    "contract; abstain",
    },
    "nonlocal_jump": {
        "c": "`setjmp`/`longjmp`",
        "rust": "no safe equivalent (unwinding differs)",
        "go": "`panic`/`recover` (different semantics)",
        "frontier": "non-local control flow not modelled; abstain",
    },
    "signal": {
        "c": "`signal`/`sigaction`",
        "rust": "`signal-hook` / `nix`",
        "go": "`os/signal`",
        "frontier": "asynchronous interruption not modelled; abstain",
    },
}
