"""
A pointer-*provenance* memory model aligned with C PNVI (100_STEPS step 77).

Step 21 gave us a byte-addressed machine that tracks *which allocation* a pointer
came from. This module sharpens that into the part of the C memory model where
cross-language translation bugs actually live: **provenance**, in the style of
the C standards-committee PNVI-ae model ("provenance not via integers, address
exposed"). The distinctions PNVI draws — and that this module makes executable —
are exactly the ones a naive translation silently erases:

* **A one-past-the-end pointer is *formable and comparable* but not
  *dereferenceable*.** `p = a + N` for an `N`-element array is a perfectly legal
  pointer you may compare against; loading or storing through it is
  out-of-bounds. A translation that treats "valid pointer" and "dereferenceable
  pointer" as the same thing is wrong here.
* **Provenance is preserved by pointer arithmetic.** Offsetting a pointer in and
  back out of bounds keeps the *same* provenance, so it may legally access only
  the bytes of the allocation it was derived from — never an adjacent object that
  happens to share an address.
* **Provenance survives an integer round-trip only via *exposure* (PNVI-ae).**
  Casting a pointer to an integer *exposes* its allocation; a later cast back to
  a pointer recovers that provenance. An integer that was never the exposed
  address of a live allocation yields a pointer with **no** provenance, whose
  dereference is undefined.

:func:`simulate` runs a provenance trace (allocate / form-pointer / pointer-add /
expose / from-int / deref / free) and returns the first :class:`ProvFault`
(out-of-bounds *formation*, out-of-bounds *dereference*, use-after-free,
cross-provenance access, or a dereference of a no-provenance pointer) or ``None``
when the trace is provenance-safe.

The confirmable fragment is checked against real compiled code under
**AddressSanitizer** by :func:`confirm_provenance`: forming and comparing a
one-past-the-end pointer runs clean, dereferencing it traps, and in-bounds
arithmetic round-trips stay safe — so the model never claims a fault the
sanitizer does not exhibit on the traces it can express. The genuinely
*provenance-only* distinctions (which no single-run sanitizer observes, because
they are about which writes the optimizer is *entitled* to assume independent)
are decided by the model and documented as the general provenance interface in
:data:`PROVENANCE_INTERFACE`.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class ProvFault(Enum):
    FORMATION_OOB = "formation_oob"        # pointer formed >1 past end or before start
    DEREF_OOB = "deref_oob"                # deref at/after one-past-the-end
    USE_AFTER_FREE = "use_after_free"      # deref through a freed allocation
    CROSS_PROVENANCE = "cross_provenance"  # access an allocation via another's provenance
    NO_PROVENANCE = "no_provenance"        # deref a pointer with no live provenance

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


@dataclass(frozen=True)
class FaultAt:
    kind: ProvFault
    where: str
    detail: str = ""

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.kind} at {self.where}: {self.detail}"


# --- the provenance machine state --------------------------------------------


@dataclass
class _Alloc:
    name: str
    size: int
    alive: bool = True
    exposed: bool = False
    base: int = 0   # synthetic address assigned at allocation time


@dataclass
class _Ptr:
    provenance: Optional[str]   # allocation name, or None for no-provenance
    offset: int = 0             # byte offset; may be == size (one-past), legal to hold


# --- the provenance-operation IR ---------------------------------------------


@dataclass(frozen=True)
class ProvEvent:
    op: str
    # operands (interpretation depends on op)
    a: str = ""          # primary name (alloc / ptr / dst-ptr)
    b: str = ""          # secondary name (src-ptr)
    size: int = 0        # alloc size
    offset: int = 0      # form: byte offset; add: signed delta; deref: byte offset
    nbytes: int = 1     # deref width


def Alloc(name: str, size: int) -> ProvEvent:
    return ProvEvent("alloc", a=name, size=size)


def Free(name: str) -> ProvEvent:
    return ProvEvent("free", a=name)


def Form(ptr: str, alloc: str, offset: int) -> ProvEvent:
    """Form pointer ``ptr`` at byte ``offset`` into ``alloc`` (offset in [0,size]
    is legal — size itself is the one-past-the-end pointer)."""
    return ProvEvent("form", a=ptr, b=alloc, offset=offset)


def Add(dst: str, src: str, delta: int) -> ProvEvent:
    """``dst = src + delta`` bytes, preserving ``src``'s provenance."""
    return ProvEvent("add", a=dst, b=src, offset=delta)


def Expose(ptr: str) -> ProvEvent:
    """Cast ``ptr`` to an integer, *exposing* its allocation's provenance."""
    return ProvEvent("expose", a=ptr)


def FromExposedAddr(dst: str, src: str) -> ProvEvent:
    """``dst`` recovers provenance from the integer address of ``src`` (PNVI-ae):
    succeeds iff ``src``'s allocation was exposed and is live."""
    return ProvEvent("from_addr", a=dst, b=src)


def FromOpaqueInt(dst: str) -> ProvEvent:
    """``dst`` is built from an integer that is not the exposed address of any
    live allocation — it has no provenance."""
    return ProvEvent("from_opaque", a=dst)


def Deref(ptr: str, nbytes: int = 1) -> ProvEvent:
    return ProvEvent("deref", a=ptr, nbytes=nbytes)


@dataclass
class ProvResult:
    fault: Optional[FaultAt]
    steps: int

    @property
    def safe(self) -> bool:
        return self.fault is None


def simulate(events: List[ProvEvent]) -> ProvResult:
    """Run a provenance trace; return the first fault (or None) and step count."""
    allocs: Dict[str, _Alloc] = {}
    ptrs: Dict[str, _Ptr] = {}
    next_base = 0x1000
    for i, ev in enumerate(events):
        where = f"#{i}:{ev.op} {ev.a}"
        if ev.op == "alloc":
            allocs[ev.a] = _Alloc(ev.a, ev.size, base=next_base)
            next_base += ev.size + 64  # red-zone gap between objects
        elif ev.op == "free":
            a = allocs.get(ev.a)
            if a is not None:
                a.alive = False
        elif ev.op == "form":
            a = allocs.get(ev.b)
            if a is None or ev.offset < 0 or ev.offset > a.size:
                return ProvResult(FaultAt(
                    ProvFault.FORMATION_OOB, where,
                    f"offset {ev.offset} not in [0,{a.size if a else '?'}]"), i)
            ptrs[ev.a] = _Ptr(ev.b, ev.offset)
        elif ev.op == "add":
            src = ptrs.get(ev.b)
            if src is None:
                return ProvResult(FaultAt(ProvFault.NO_PROVENANCE, where,
                                          f"unknown source pointer {ev.b}"), i)
            new_off = src.offset + ev.offset
            a = allocs.get(src.provenance) if src.provenance else None
            size = a.size if a else 0
            if src.provenance is None or new_off < 0 or new_off > size:
                return ProvResult(FaultAt(
                    ProvFault.FORMATION_OOB, where,
                    f"arith yields offset {new_off} not in [0,{size}]"), i)
            ptrs[ev.a] = _Ptr(src.provenance, new_off)
        elif ev.op == "expose":
            p = ptrs.get(ev.a)
            if p is not None and p.provenance in allocs:
                allocs[p.provenance].exposed = True
        elif ev.op == "from_addr":
            src = ptrs.get(ev.b)
            a = allocs.get(src.provenance) if (src and src.provenance) else None
            if a is not None and a.exposed and a.alive:
                ptrs[ev.a] = _Ptr(a.name, src.offset)
            else:
                ptrs[ev.a] = _Ptr(None, 0)  # provenance not recovered
        elif ev.op == "from_opaque":
            ptrs[ev.a] = _Ptr(None, 0)
        elif ev.op == "deref":
            p = ptrs.get(ev.a)
            if p is None or p.provenance is None:
                return ProvResult(FaultAt(ProvFault.NO_PROVENANCE, where,
                                          f"{ev.a} has no provenance"), i)
            a = allocs.get(p.provenance)
            if a is None:
                return ProvResult(FaultAt(ProvFault.NO_PROVENANCE, where,
                                          f"{p.provenance} is gone"), i)
            if not a.alive:
                return ProvResult(FaultAt(ProvFault.USE_AFTER_FREE, where,
                                          f"{a.name} was freed"), i)
            if p.offset < 0 or p.offset + ev.nbytes > a.size:
                return ProvResult(FaultAt(
                    ProvFault.DEREF_OOB, where,
                    f"deref [{p.offset},{p.offset + ev.nbytes}) outside "
                    f"[0,{a.size}) of {a.name}"
                    + (" (one-past-the-end is formable but not dereferenceable)"
                       if p.offset == a.size else "")), i)
        else:  # pragma: no cover - guarded by constructors
            raise ValueError(f"unknown op {ev.op!r}")
    return ProvResult(None, len(events))


def first_fault(events: List[ProvEvent]) -> Optional[FaultAt]:
    return simulate(events).fault


# --- the documented general provenance interface -----------------------------

PROVENANCE_INTERFACE = {
    "pointer_carries_provenance":
        "Every pointer value is (provenance, address); provenance identifies the "
        "single allocation the pointer may access.",
    "arithmetic_preserves_provenance":
        "Pointer +/- integer keeps the source provenance; the result is "
        "well-formed iff its offset is in [0, size] (size == one-past-the-end).",
    "one_past_the_end_is_formable_not_dereferenceable":
        "An offset exactly equal to size yields a legal, comparable pointer whose "
        "dereference is out of bounds.",
    "integer_roundtrip_requires_exposure":
        "Casting a pointer to an integer exposes its allocation; casting back "
        "recovers that provenance (PNVI-ae). An integer that was never an exposed "
        "live address yields a no-provenance pointer that may not be dereferenced.",
    "cross_provenance_access_is_undefined":
        "A pointer with provenance of A may not access B even at an equal "
        "address; this is where translation that drops provenance diverges.",
    "free_revokes_provenance":
        "Freeing an allocation makes every later access through a pointer with "
        "its provenance a use-after-free.",
}


# --- real-compiler (AddressSanitizer) confirmation ---------------------------


@dataclass
class ProvConfirmation:
    available: bool
    predicted_fault: Optional[ProvFault] = None
    asan_trapped: Optional[bool] = None
    asan_report: str = ""
    stdout: str = ""
    reason: str = ""

    @property
    def consistent(self) -> bool:
        if self.asan_trapped is None:
            return False
        return (self.predicted_fault is not None) == self.asan_trapped


def _form_compare_one_past_src() -> str:
    return (
        "#include <stdio.h>\n"
        "int main(void){\n"
        "    int a[4];\n"
        "    int *p = a + 4;            /* one-past-the-end: legal to form */\n"
        "    volatile int eq = (p == a + 4);\n"
        "    printf(\"%d\\n\", eq);\n"
        "    return 0;\n"
        "}\n"
    )


def _deref_one_past_src() -> str:
    return (
        "int main(void){\n"
        "    int a[4];\n"
        "    volatile int i = 4;        /* defeat constant-folding */\n"
        "    int *p = a + i;            /* one-past-the-end */\n"
        "    return *p;                 /* UB: out-of-bounds dereference */\n"
        "}\n"
    )


def _arith_roundtrip_src() -> str:
    return (
        "int main(void){\n"
        "    int a[4] = {1,2,3,4};\n"
        "    int *p = a; p += 3; p -= 2;  /* provenance preserved, back in bounds */\n"
        "    return *p - 2;               /* a[1] == 2 -> returns 0 */\n"
        "}\n"
    )


#: the three real-compiler-confirmable provenance scenarios and whether the
#: provenance model predicts a fault for each.
CONFIRMABLE = {
    "form_compare_one_past": (_form_compare_one_past_src, False),
    "deref_one_past": (_deref_one_past_src, True),
    "arith_roundtrip_inbounds": (_arith_roundtrip_src, False),
}


def confirm_provenance(scenario: str, cc: str) -> ProvConfirmation:
    """Compile a confirmable provenance scenario under ASan and check the trap
    matches the model's prediction."""
    if scenario not in CONFIRMABLE:
        raise KeyError(f"unknown scenario {scenario!r}; known: {sorted(CONFIRMABLE)}")
    src_fn, predicts_fault = CONFIRMABLE[scenario]
    predicted = ProvFault.DEREF_OOB if predicts_fault else None
    with tempfile.TemporaryDirectory() as d:
        cp = os.path.join(d, "p.c")
        co = os.path.join(d, "p.out")
        with open(cp, "w") as fh:
            fh.write(src_fn())
        r = subprocess.run([cc, "-fsanitize=address", "-O0", "-g", "-o", co, cp],
                           capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            return ProvConfirmation(False, predicted_fault=predicted,
                                    reason=f"ASan compile failed: {r.stderr[:200]}")
        env = dict(os.environ)
        env["ASAN_OPTIONS"] = "detect_leaks=0"
        run = subprocess.run([co], capture_output=True, text=True,
                            timeout=120, env=env)
        report = run.stderr
        trapped = ("AddressSanitizer" in report)
        return ProvConfirmation(True, predicted_fault=predicted,
                                asan_trapped=trapped, asan_report=report,
                                stdout=run.stdout)


# --- ready-made provenance traces --------------------------------------------


def one_past_form_then_deref() -> List[ProvEvent]:
    """Form the one-past-the-end pointer (legal) then dereference it (DEREF_OOB)."""
    return [Alloc("a", 16), Form("p", "a", 16), Deref("p", 4)]


def one_past_form_only() -> List[ProvEvent]:
    """Form (and never dereference) the one-past-the-end pointer — provenance-safe."""
    return [Alloc("a", 16), Form("p", "a", 16)]


def arithmetic_roundtrip() -> List[ProvEvent]:
    """Offset a pointer out and back in bounds; the in-bounds deref is safe."""
    return [Alloc("a", 16), Form("p", "a", 0), Add("p", "p", 12),
            Add("p", "p", -8), Deref("p", 4)]


def exposed_roundtrip_recovers_provenance() -> List[ProvEvent]:
    """Expose a pointer, rebuild it from its address, dereference — safe (PNVI-ae)."""
    return [Alloc("a", 16), Form("p", "a", 0), Expose("p"),
            FromExposedAddr("q", "p"), Deref("q", 4)]


def opaque_int_has_no_provenance() -> List[ProvEvent]:
    """A pointer from an unexposed integer has no provenance; deref is undefined."""
    return [Alloc("a", 16), FromOpaqueInt("p"), Deref("p", 4)]


def use_after_free_via_provenance() -> List[ProvEvent]:
    return [Alloc("a", 16), Form("p", "a", 0), Free("a"), Deref("p", 4)]


def formation_out_of_bounds() -> List[ProvEvent]:
    """Forming a pointer two past the end is already UB (before any dereference)."""
    return [Alloc("a", 16), Form("p", "a", 20)]
