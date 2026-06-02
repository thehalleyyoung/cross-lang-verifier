"""Step 36 — fuzzer-guided frontend hardening.

A frontend that silently drops a function, mis-reads a signature, or crashes on
an unusual-but-legal construct is a correctness hole that no downstream oracle can
recover from. This module points a *differential fuzzer* at the C frontend
(:func:`ir_ingest.ingest_clang`) and uses the real clang as the oracle:

1. A seeded generator emits random — but always well-typed and compilable — C
   translation units: random function counts, return types drawn from the C
   integer/pointer/void zoo, random arity, random parameter types and names, and
   a random storage class (``static`` or external linkage). Bodies are synthesised
   to be valid for the chosen return type.
2. Each program is first **compiled** by clang (``-fsyntax-only``); any program
   that does not compile is discarded so we never blame the frontend for invalid
   input.
3. The program is ingested and the recovered :class:`ir_ingest.IRFunction` set is
   compared against the *generator's own ground truth*: every defined function
   must be recovered with the exact name, arity, return type, parameter types and
   storage class. A single mismatch is a parse-divergence.

The frontend must also be **crash-proof on garbage**: a corpus of malformed /
non-C inputs must make the ingester return ``None`` rather than raise.

:func:`fuzz_clang_frontend` runs the whole loop for a fixed seed and reports every
divergence and crash; :func:`confirm_fuzz` asserts a sizeable run survives with
zero divergences and zero crashes.
"""

from __future__ import annotations

import os
import random
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import ir_ingest as _iri

CLANG = _iri.CLANG

# Return/parameter types and a zero-value literal usable to build a valid body.
_INT_TYPES: List[Tuple[str, str]] = [
    ("int", "0"),
    ("unsigned int", "0u"),
    ("long", "0L"),
    ("unsigned long", "0uL"),
    ("short", "0"),
    ("char", "0"),
    ("unsigned char", "0"),
    ("long long", "0LL"),
]
_PTR_TYPES: List[Tuple[str, str]] = [
    ("int *", "0"),
    ("char *", "0"),
    ("const char *", "0"),
    ("void *", "0"),
]
_PARAM_TYPES: List[str] = [t for t, _ in _INT_TYPES] + [t for t, _ in _PTR_TYPES]


@dataclass(frozen=True)
class GenParam:
    name: str
    type: str


@dataclass(frozen=True)
class GenFunction:
    name: str
    ret_type: str
    params: Tuple[GenParam, ...]
    storage: str          # "static" or ""


@dataclass
class GenProgram:
    functions: Tuple[GenFunction, ...]
    source: str


def _normalize_type(t: str) -> str:
    """Canonicalise spacing so generator/clang spellings compare equal."""
    return " ".join(t.replace("*", " * ").split()).replace(" *", " *")


def generate_program(rng: random.Random, idx: int) -> GenProgram:
    nfns = rng.randint(1, 5)
    fns: List[GenFunction] = []
    used = set()
    for fi in range(nfns):
        name = f"f{idx}_{fi}_{rng.randint(0, 9999)}"
        while name in used:
            name = f"f{idx}_{fi}_{rng.randint(0, 9999)}"
        used.add(name)
        is_ptr = rng.random() < 0.35
        if is_ptr:
            ret_type, zero = rng.choice(_PTR_TYPES)
        else:
            ret_type, zero = rng.choice(_INT_TYPES)
        nparams = rng.randint(0, 4)
        params: List[GenParam] = []
        for pi in range(nparams):
            ptype = rng.choice(_PARAM_TYPES)
            params.append(GenParam(name=f"p{pi}", type=ptype))
        storage = "static" if rng.random() < 0.4 else ""
        fns.append(GenFunction(name, ret_type, tuple(params), storage))

    lines: List[str] = []
    for f in fns:
        plist = ", ".join(f"{p.type} {p.name}" for p in f.params) or "void"
        head = f"{f.storage + ' ' if f.storage else ''}{f.ret_type} {f.name}({plist})"
        # a body valid for the return type (pointers -> NULL via 0 cast, ints -> 0).
        if f.ret_type.endswith("*"):
            body = f"return ({f.ret_type})0;"
        else:
            body = "return 0;"
        lines.append(f"{head} {{ {body} }}")
    return GenProgram(tuple(fns), "\n".join(lines) + "\n")


def _compiles(src: str) -> bool:
    if not os.path.exists(CLANG):
        return False
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "a.c")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(src)
        r = subprocess.run(
            [CLANG, "-fsyntax-only", "-Wno-everything", p],
            capture_output=True, text=True,
        )
    return r.returncode == 0


@dataclass
class Divergence:
    function: str
    field: str
    expected: str
    got: str


@dataclass
class FuzzReport:
    iterations: int
    compiled: int
    divergences: List[Divergence] = field(default_factory=list)
    crashes: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.divergences and not self.crashes


def _check_program(prog: GenProgram) -> Tuple[List[Divergence], Optional[str]]:
    """Ingest one program and diff against ground truth. Returns (divergences, crash)."""
    try:
        mod = _iri.ingest_clang(prog.source)
    except Exception as e:  # the frontend must never raise on valid input
        return [], f"ingest raised: {e!r}"
    if mod is None:
        return [], "ingest returned None on a compiling program"
    divs: List[Divergence] = []
    for gf in prog.functions:
        rf = mod.functions.get(gf.name)
        if rf is None:
            divs.append(Divergence(gf.name, "presence", "recovered", "missing"))
            continue
        if _normalize_type(rf.ret_type) != _normalize_type(gf.ret_type):
            divs.append(Divergence(gf.name, "ret_type", gf.ret_type, rf.ret_type))
        if rf.arity != len(gf.params):
            divs.append(Divergence(gf.name, "arity",
                                   str(len(gf.params)), str(rf.arity)))
        else:
            for gp, rp in zip(gf.params, rf.params):
                if _normalize_type(rp.type) != _normalize_type(gp.type):
                    divs.append(Divergence(gf.name, "param_type",
                                           gp.type, rp.type))
        exp_storage = gf.storage
        if rf.storage != exp_storage:
            divs.append(Divergence(gf.name, "storage",
                                   exp_storage or "(none)", rf.storage or "(none)"))
    return divs, None


# Malformed / hostile inputs the frontend must survive without raising.
GARBAGE_INPUTS: List[str] = [
    "",
    "this is not C at all )(*&^%$#@",
    "int f( {",                       # unbalanced
    "int main() { return",            # truncated
    "#include <nonexistent_xyz.h>\nint g(void){return 0;}",
    "struct { int a; ;;; } ",
    "\x00\x01\x02 garbage bytes",
    "int " * 1000,                    # pathological repetition
]


def fuzz_clang_frontend(iterations: int = 60, seed: int = 0xC0FFEE) -> FuzzReport:
    """Differential-fuzz the C frontend against clang for a fixed seed."""
    rng = random.Random(seed)
    rep = FuzzReport(iterations=iterations, compiled=0)
    if not os.path.exists(CLANG):
        return rep
    for i in range(iterations):
        prog = generate_program(rng, i)
        if not _compiles(prog.source):
            continue
        rep.compiled += 1
        divs, crash = _check_program(prog)
        rep.divergences.extend(divs)
        if crash:
            rep.crashes.append(crash)
    # garbage must never crash the frontend.
    for g in GARBAGE_INPUTS:
        try:
            _iri.ingest_clang(g)
        except Exception as e:
            rep.crashes.append(f"garbage crash {g[:24]!r}: {e!r}")
    return rep


@dataclass
class FuzzConfirmation:
    available: bool
    ok: bool
    report: Optional[FuzzReport] = None


def confirm_fuzz(iterations: int = 60, seed: int = 0xC0FFEE) -> FuzzConfirmation:
    if not os.path.exists(CLANG):
        return FuzzConfirmation(available=False, ok=False)
    rep = fuzz_clang_frontend(iterations=iterations, seed=seed)
    # require a meaningful number of programs to have actually compiled+ingested.
    ok = rep.ok and rep.compiled >= max(1, iterations // 2)
    return FuzzConfirmation(available=True, ok=ok, report=rep)


FRONTEND_FUZZ_SPI = {
    "generate_program": generate_program,
    "fuzz_clang_frontend": fuzz_clang_frontend,
    "confirm_fuzz": confirm_fuzz,
}


if __name__ == "__main__":  # pragma: no cover
    c = confirm_fuzz()
    print("available:", c.available, "ok:", c.ok)
    if c.report:
        print(f"iterations={c.report.iterations} compiled={c.report.compiled} "
              f"divergences={len(c.report.divergences)} crashes={len(c.report.crashes)}")
        for d in c.report.divergences[:10]:
            print("  DIV", d)
        for cr in c.report.crashes[:10]:
            print("  CRASH", cr)
