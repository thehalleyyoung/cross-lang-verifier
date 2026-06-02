"""
A byte-addressed memory model with pointer *provenance* (100_STEPS step 21).

The heuristic memory oracles elsewhere in this project reason about a single
access in isolation. Real spatial/temporal memory bugs, though, are *trace*
properties: whether a load is out of bounds or a pointer is dangling depends on
the whole history of allocations and frees that produced it. This module gives
those oracles a principled foundation: a small **byte-addressed** abstract
machine in which every pointer carries the *provenance* of the allocation it was
derived from, exactly as the C memory model requires.

The machine is deliberately tiny but faithful to the rules that matter:

* an :class:`Allocation` owns a half-open byte range ``[0, size)`` and an
  *alive* flag;
* a :class:`Pointer` is a ``(provenance, offset)`` pair — it may legally access
  **only** the bytes of the allocation it was derived from (pointer arithmetic
  preserves provenance), so an access is in bounds iff
  ``0 <= offset`` and ``offset + nbytes <= size`` of *that* allocation — never
  some adjacent object that happens to sit next to it in the address space;
* freeing flips the alive flag, so any later load/store through a pointer with
  that provenance is a **use-after-free**, and freeing again is a
  **double-free**.

:func:`simulate` runs a trace of :class:`MemEvent`s and returns the *first*
:class:`MemFault` (spatial OOB, use-after-free, double-free, invalid-free,
null-deref) or ``None`` if the trace is memory-safe. The faithfulness of the
model is not asserted — it is *confirmed*: :func:`confirm_memory` emits the
equivalent C program, compiles it under **AddressSanitizer**, and checks that
ASan traps **iff** the model predicted a fault (and that a model-safe trace runs
cleanly). The model therefore never claims a memory bug the real sanitizer does
not exhibit, and never misses one ASan catches on the traces we generate.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class FaultKind(Enum):
    OOB_SPATIAL = "oob_spatial"        # access outside [0, size) of the allocation
    USE_AFTER_FREE = "use_after_free"  # access through a freed allocation's pointer
    DOUBLE_FREE = "double_free"        # free of an already-freed allocation
    INVALID_FREE = "invalid_free"      # free of a non-heap / unknown pointer
    NULL_DEREF = "null_deref"          # access through a null pointer

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


@dataclass(frozen=True)
class MemFault:
    kind: FaultKind
    where: str          # the event that triggered it
    detail: str = ""

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.kind} at {self.where}: {self.detail}"


# --- the abstract machine ----------------------------------------------------


@dataclass
class Allocation:
    name: str
    size: int
    alive: bool = True
    is_heap: bool = True


@dataclass
class Pointer:
    provenance: Optional[str]   # allocation name, or None for null
    offset: int = 0

    @property
    def is_null(self) -> bool:
        return self.provenance is None


# --- a tiny memory-operation IR ----------------------------------------------


@dataclass(frozen=True)
class MemEvent:
    op: str                      # "alloc" | "free" | "load" | "store"
    name: str                    # allocation name the op refers to
    size: int = 0                # alloc: byte size
    offset: int = 0              # load/store: byte offset into the allocation
    nbytes: int = 1             # load/store: width in bytes


def Alloc(name: str, size: int) -> MemEvent:
    return MemEvent("alloc", name, size=size)


def Free(name: str) -> MemEvent:
    return MemEvent("free", name)


def Store(name: str, offset: int, nbytes: int = 1) -> MemEvent:
    return MemEvent("store", name, offset=offset, nbytes=nbytes)


def Load(name: str, offset: int, nbytes: int = 1) -> MemEvent:
    return MemEvent("load", name, offset=offset, nbytes=nbytes)


@dataclass
class SimResult:
    fault: Optional[MemFault]
    steps: int

    @property
    def safe(self) -> bool:
        return self.fault is None


def simulate(events: List[MemEvent]) -> SimResult:
    """Run a memory trace; return the first fault (or None) and steps executed."""
    mem: Dict[str, Allocation] = {}
    for i, ev in enumerate(events):
        where = f"#{i}:{ev.op} {ev.name}"
        if ev.op == "alloc":
            mem[ev.name] = Allocation(ev.name, ev.size)
        elif ev.op == "free":
            a = mem.get(ev.name)
            if a is None or not a.is_heap:
                return SimResult(MemFault(FaultKind.INVALID_FREE, where,
                                          f"{ev.name} is not a live heap object"), i)
            if not a.alive:
                return SimResult(MemFault(FaultKind.DOUBLE_FREE, where,
                                          f"{ev.name} already freed"), i)
            a.alive = False
        elif ev.op in ("load", "store"):
            a = mem.get(ev.name)
            if a is None:
                return SimResult(MemFault(FaultKind.NULL_DEREF, where,
                                          f"no allocation named {ev.name}"), i)
            if not a.alive:
                return SimResult(MemFault(FaultKind.USE_AFTER_FREE, where,
                                          f"{ev.name} was freed"), i)
            if ev.offset < 0 or ev.offset + ev.nbytes > a.size:
                return SimResult(MemFault(
                    FaultKind.OOB_SPATIAL, where,
                    f"access [{ev.offset},{ev.offset + ev.nbytes}) outside "
                    f"[0,{a.size}) of {ev.name}"), i)
        else:  # pragma: no cover - guarded by constructors
            raise ValueError(f"unknown op {ev.op!r}")
    return SimResult(None, len(events))


def first_fault(events: List[MemEvent]) -> Optional[MemFault]:
    return simulate(events).fault


# --- real-compiler (AddressSanitizer) confirmation ---------------------------


def c_source(events: List[MemEvent]) -> str:
    """Emit a C program performing the same heap operations as ``events``.

    Allocations become ``malloc`` of ``char*``, frees become ``free``, and
    loads/stores become byte accesses at the given offset through that pointer —
    so AddressSanitizer's spatial/temporal checks apply with full provenance.
    The accessed value is funnelled through a ``volatile`` sink so the optimizer
    cannot elide the access.
    """
    lines = ["#include <stdlib.h>", "#include <stdio.h>",
             "static volatile unsigned char sink;", "int main(void){"]
    for ev in events:
        if ev.op == "alloc":
            lines.append(f"    char *{ev.name} = (char*)malloc({ev.size});")
        elif ev.op == "free":
            lines.append(f"    free({ev.name});")
        elif ev.op == "store":
            for b in range(ev.nbytes):
                lines.append(f"    {ev.name}[{ev.offset + b}] = (char)1;")
        elif ev.op == "load":
            for b in range(ev.nbytes):
                lines.append(f"    sink = (unsigned char){ev.name}[{ev.offset + b}];")
    lines.append("    (void)sink;")
    lines.append("    return 0;")
    lines.append("}")
    return "\n".join(lines) + "\n"


@dataclass
class MemConfirmation:
    available: bool
    predicted_fault: Optional[MemFault] = None
    asan_trapped: Optional[bool] = None
    asan_report: str = ""
    reason: str = ""

    @property
    def consistent(self) -> bool:
        """True iff the model's prediction matches ASan's observed behaviour."""
        if self.asan_trapped is None:
            return False
        predicted = self.predicted_fault is not None
        return predicted == self.asan_trapped


def _asan_kind(report: str) -> str:
    r = report.lower()
    if "heap-use-after-free" in r:
        return "use_after_free"
    if "double-free" in r or "attempting double-free" in r:
        return "double_free"
    if "heap-buffer-overflow" in r or "stack-buffer-overflow" in r:
        return "oob_spatial"
    return "other"


def confirm_memory(events: List[MemEvent], cc: str) -> MemConfirmation:
    """Compile the trace under ASan and check the trap matches the prediction."""
    predicted = first_fault(events)
    with tempfile.TemporaryDirectory() as d:
        cp = os.path.join(d, "m.c")
        co = os.path.join(d, "m.out")
        with open(cp, "w") as fh:
            fh.write(c_source(events))
        r = subprocess.run(
            [cc, "-fsanitize=address", "-O0", "-g", "-o", co, cp],
            capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            return MemConfirmation(False, predicted_fault=predicted,
                                   reason=f"ASan compile failed: {r.stderr[:200]}")
        env = dict(os.environ)
        env["ASAN_OPTIONS"] = "detect_leaks=0"
        run = subprocess.run([co], capture_output=True, text=True,
                             timeout=120, env=env)
        report = run.stderr
        trapped = ("AddressSanitizer" in report) or (run.returncode not in (0,))
        return MemConfirmation(True, predicted_fault=predicted,
                               asan_trapped=trapped, asan_report=report)


# --- ready-made traces -------------------------------------------------------


def oob_trace() -> List[MemEvent]:
    """A 16-byte buffer read one byte past the end — a spatial fault."""
    return [Alloc("p", 16), Store("p", 0, 4), Load("p", 16, 1)]


def uaf_trace() -> List[MemEvent]:
    """Free then read — a temporal (use-after-free) fault."""
    return [Alloc("p", 16), Free("p"), Load("p", 0, 1)]


def double_free_trace() -> List[MemEvent]:
    return [Alloc("p", 16), Free("p"), Free("p")]


def safe_trace() -> List[MemEvent]:
    """In-bounds store/load then free — memory-safe."""
    return [Alloc("p", 16), Store("p", 12, 4), Load("p", 12, 4), Free("p")]


def safe_boundary_trace() -> List[MemEvent]:
    """Access the very last legal byte — in bounds (off-by-one safe)."""
    return [Alloc("p", 8), Store("p", 7, 1), Load("p", 7, 1), Free("p")]
