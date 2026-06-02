"""Uninitialized-read / definedness divergence oracle (100_STEPS step 17).

Reading an object with automatic storage duration before it is initialized is
**undefined behavior** in C (C11 6.3.2.1, 6.7.9): the value is indeterminate and
the optimizer is free to materialize *any* value — including different values at
different optimization levels.  The idiomatic translation into Rust / Go / Swift,
by contrast, either **forbids** the read (Rust's type system rejects use of a
possibly-uninitialized binding) or **zero/default-initializes** the storage (Go's
zero values; an explicit Swift initializer), so the translated program has a
single, defined, deterministic result.

This oracle is built around a small but real **definedness-lattice dataflow
analysis** (`analyze_definedness`): each storage slot is tracked through the
unit's writes with the three-point lattice

    UNINIT  ⊏  MAYBE  ⊏  DEFINED

— an *unconditional* write makes a slot ``DEFINED``; a *guarded* write can only
raise an ``UNINIT`` slot to ``MAYBE`` (must-analysis: it is not defined on every
path); a slot never written stays ``UNINIT``.  A read of a slot whose state is
``UNINIT`` or ``MAYBE`` is the undefined read this class is about; a read of a
``DEFINED`` slot is *not* flagged (the oracle honestly returns
``NO_DIVERGENCE_FOUND``).

Because no sanitizer traps an uninitialized read at runtime, the divergence is
confirmed with the ``optimizer_exploited`` harness mode: the **same** C source,
on the **same** input, produces different observable output under two
standard-conforming compilations (``-O0`` vs ``-O2``) — proof that the C result
is not fixed by the language — while the target program is defined and
deterministic.  To make the ``-O0`` build reliably read *non-zero* leftover
storage (so the two builds genuinely disagree), the emitted C first calls a
``noinline`` helper that fills an equally-sized stack frame with an
input-derived, always-odd bit pattern; the optimizer elides or constant-folds
the uninitialized read at ``-O2``, yielding a different value.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..catalogue import UNINIT_READ, Definedness
from ..plugin import DivergenceOracle, OracleResult, OracleVerdict, register
from ..replay import Counterexample

# Definedness lattice points (string-valued for easy serialization/inspection).
UNINIT = "uninit"
MAYBE = "maybe"
DEFINED = "defined"
_ORDER = {UNINIT: 0, MAYBE: 1, DEFINED: 2}

# Size of the stack-poison frame; large enough to dominate the read's slot.
_POISON_N = 192


# ── the definedness-lattice dataflow analysis ────────────────────────────────

def _slots_of(storage: Dict) -> List[object]:
    kind = storage.get("kind")
    if kind == "scalar":
        return [None]
    if kind == "array":
        return list(range(int(storage["length"])))
    if kind == "struct":
        return list(storage["fields"])
    raise ValueError(f"unknown storage kind {kind!r}")


def analyze_definedness(unit: Dict) -> Dict[object, str]:
    """Propagate the three-point definedness lattice through the unit's writes.

    Returns a map ``slot -> {UNINIT, MAYBE, DEFINED}``.  This is a *must*-analysis:
    a slot reaches ``DEFINED`` only via a write that executes on every path (an
    unguarded write); a guarded write can at most raise ``UNINIT`` to ``MAYBE``.
    """
    storage = unit["storage"]
    state: Dict[object, str] = {s: UNINIT for s in _slots_of(storage)}
    for w in unit.get("writes", []):
        slot = w.get("slot")
        if slot not in state:
            raise ValueError(f"write to unknown slot {slot!r}")
        guarded = bool(w.get("guarded", False))
        if not guarded:
            state[slot] = DEFINED
        else:
            # A guarded write defines the slot only on some paths.
            if _ORDER[state[slot]] < _ORDER[MAYBE]:
                state[slot] = MAYBE
    return state


def uninitialized_read(unit: Dict) -> Optional[Tuple[object, str]]:
    """If the unit's read observes an under-defined slot, return ``(slot, state)``.

    Returns ``None`` when the read slot is fully ``DEFINED`` (no UB) — so the
    oracle never fabricates a divergence for a well-initialized read.
    """
    state = analyze_definedness(unit)
    slot = unit.get("read")
    if slot not in state:
        raise ValueError(f"read of unknown slot {slot!r}")
    if _ORDER[state[slot]] < _ORDER[DEFINED]:
        return slot, state[slot]
    return None


# ── source generation ────────────────────────────────────────────────────────

def _c_poison_helper() -> str:
    n = _POISON_N
    return (
        "__attribute__((noinline)) static int _poison(int s){\n"
        f"    int buf[{n}];\n"
        f"    for (int i=0;i<{n};i++) buf[i] = (int)((((unsigned)s*2654435761u) "
        "^ ((unsigned)i*40503u)) | 1u);\n"
        f"    int acc=0; for (int i=0;i<{n};i++) acc ^= buf[i];\n"
        "    return acc;\n"
        "}\n"
    )


def _c_decl_and_ops(storage: Dict, writes: List[Dict], read: object) -> str:
    """Render the body of ``_leak(int s)`` for the given storage/writes/read."""
    n = _POISON_N
    # A sibling stack array forces the read's slot into the poisoned frame.
    pad = f"    int _pad[{n}]; (void)_pad;\n"
    kind = storage["kind"]
    lines = [pad]
    if kind == "scalar":
        lines.append("    int y;\n")
        for w in writes:
            if w.get("guarded"):
                lines.append("        if (s > 1000000000) y = s;\n")
            else:
                lines.append("    y = s;\n")
        lines.append("    return y;\n")
    elif kind == "array":
        length = int(storage["length"])
        lines.append(f"    int a[{length}];\n")
        for w in writes:
            idx = int(w["slot"])
            if w.get("guarded"):
                lines.append(f"        if (s > 1000000000) a[{idx}] = s;\n")
            else:
                lines.append(f"    a[{idx}] = s;\n")
        lines.append(f"    return a[{int(read)}];\n")
    elif kind == "struct":
        fields = list(storage["fields"])
        decl = "; ".join(f"int {f}" for f in fields)
        lines.append(f"    struct P {{ {decl}; }};\n    struct P p;\n")
        for w in writes:
            f = w["slot"]
            if w.get("guarded"):
                lines.append(f"        if (s > 1000000000) p.{f} = s;\n")
            else:
                lines.append(f"    p.{f} = s;\n")
        lines.append(f"    return p.{read};\n")
    else:  # pragma: no cover - guarded earlier
        raise ValueError(kind)
    return "".join(lines)


def _c_source(storage: Dict, writes: List[Dict], read: object) -> str:
    return (
        "#include <stdio.h>\n#include <stdlib.h>\n"
        + _c_poison_helper()
        + "__attribute__((noinline)) static int _leak(int s){\n"
        + _c_decl_and_ops(storage, writes, read)
        + "}\n"
        "int main(int argc, char**argv){\n"
        "    int s = argc > 1 ? atoi(argv[1]) : 1;\n"
        "    volatile int sink = _poison(s); (void)sink;\n"
        "    printf(\"%d\\n\", _leak(s));\n"
        "    return 0;\n"
        "}\n"
    )


# ── target emitters (defined, deterministic translations) ────────────────────

def _rust_source(storage: Dict, writes: List[Dict], read: object) -> str:
    kind = storage["kind"]
    body = ["fn _leak(s: i32) -> i32 {\n"]
    if kind == "scalar":
        body.append("    let mut y: i32 = 0;\n")
        for w in writes:
            body.append("    if s > 1000000000 { y = s; }\n" if w.get("guarded")
                        else "    y = s;\n")
        body.append("    y\n")
    elif kind == "array":
        length = int(storage["length"])
        body.append(f"    let mut a: [i32; {length}] = [0; {length}];\n")
        for w in writes:
            idx = int(w["slot"])
            body.append(f"    if s > 1000000000 {{ a[{idx}] = s; }}\n" if w.get("guarded")
                        else f"    a[{idx}] = s;\n")
        body.append(f"    a[{int(read)}]\n")
    else:  # struct
        fields = list(storage["fields"])
        fdecl = ", ".join(f"{f}: i32" for f in fields)
        finit = ", ".join(f"{f}: 0" for f in fields)
        body.append(f"    struct P {{ {fdecl} }}\n    let mut p = P {{ {finit} }};\n")
        for w in writes:
            f = w["slot"]
            body.append(f"    if s > 1000000000 {{ p.{f} = s; }}\n" if w.get("guarded")
                        else f"    p.{f} = s;\n")
        body.append(f"    p.{read}\n")
    body.append("}\n")
    return (
        "".join(body)
        + "fn main(){\n"
        "    let s: i32 = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(1);\n"
        "    println!(\"{}\", _leak(s));\n"
        "}\n"
    )


def _go_source(storage: Dict, writes: List[Dict], read: object) -> str:
    kind = storage["kind"]
    body = ["func _leak(s int32) int32 {\n"]
    if kind == "scalar":
        body.append("\tvar y int32 = 0\n")
        for w in writes:
            body.append("\tif s > 1000000000 { y = s }\n" if w.get("guarded")
                        else "\ty = s\n")
        body.append("\treturn y\n")
    elif kind == "array":
        length = int(storage["length"])
        body.append(f"\tvar a [{length}]int32\n")
        for w in writes:
            idx = int(w["slot"])
            body.append(f"\tif s > 1000000000 {{ a[{idx}] = s }}\n" if w.get("guarded")
                        else f"\ta[{idx}] = s\n")
        body.append(f"\treturn a[{int(read)}]\n")
    else:  # struct
        fields = list(storage["fields"])
        fdecl = "; ".join(f"{f} int32" for f in fields)
        body.append(f"\ttype P struct {{ {fdecl} }}\n\tvar p P\n")
        for w in writes:
            f = w["slot"]
            body.append(f"\tif s > 1000000000 {{ p.{f} = s }}\n" if w.get("guarded")
                        else f"\tp.{f} = s\n")
        body.append(f"\treturn p.{read}\n")
    body.append("}\n")
    return (
        "package main\n"
        "import (\n\t\"fmt\"\n\t\"os\"\n\t\"strconv\"\n)\n"
        + "".join(body)
        + "func main() {\n"
        "\ts := int32(1)\n"
        "\tif len(os.Args) > 1 { v, _ := strconv.ParseInt(os.Args[1], 10, 32); s = int32(v) }\n"
        "\tfmt.Println(_leak(s))\n"
        "}\n"
    )


def _swift_source(storage: Dict, writes: List[Dict], read: object) -> str:
    kind = storage["kind"]
    body = ["func _leak(_ s: Int32) -> Int32 {\n"]
    if kind == "scalar":
        body.append("    var y: Int32 = 0\n")
        for w in writes:
            body.append("    if s > 1000000000 { y = s }\n" if w.get("guarded")
                        else "    y = s\n")
        body.append("    return y\n")
    elif kind == "array":
        length = int(storage["length"])
        body.append(f"    var a = [Int32](repeating: 0, count: {length})\n")
        for w in writes:
            idx = int(w["slot"])
            body.append(f"    if s > 1000000000 {{ a[{idx}] = s }}\n" if w.get("guarded")
                        else f"    a[{idx}] = s\n")
        body.append(f"    return a[{int(read)}]\n")
    else:  # struct
        fields = list(storage["fields"])
        fdecl = "; ".join(f"var {f}: Int32" for f in fields)
        finit = ", ".join(f"{f}: 0" for f in fields)
        body.append(f"    struct P {{ {fdecl} }}\n    var p = P({finit})\n")
        for w in writes:
            f = w["slot"]
            body.append(f"    if s > 1000000000 {{ p.{f} = s }}\n" if w.get("guarded")
                        else f"    p.{f} = s\n")
        body.append(f"    return p.{read}\n")
    body.append("}\n")
    return (
        "import Foundation\n"
        + "".join(body)
        + "let s = CommandLine.arguments.count > 1 ? (Int32(CommandLine.arguments[1]) ?? 1) : 1\n"
        "print(_leak(s))\n"
    )


_TARGET_EMITTERS = {
    "rust": _rust_source,
    "go": _go_source,
    "swift": _swift_source,
}


# ── the oracle (one instance per target language) ────────────────────────────

class UninitializedReadOracle(DivergenceOracle):
    divergence_class = UNINIT_READ.key
    source_lang = "c"
    target_lang = "rust"
    confirmation_mode = "optimizer_exploited"
    # -O0 reads the poisoned leftover; -O2 elides/zeroes the uninitialized read.
    optimizer_flag_variants = (["-O0"], ["-O2"])

    def applies_to(self, unit: Dict) -> bool:
        if unit.get("probe") not in (None, self.divergence_class):
            return False
        if unit.get("kind") != "uninit_read":
            return False
        storage = unit.get("storage")
        return isinstance(storage, dict) and storage.get("kind") in (
            "scalar", "array", "struct")

    def find_divergence(self, unit: Dict) -> OracleResult:
        if not self.applies_to(unit):
            return OracleResult(OracleVerdict.NOT_APPLICABLE, self.divergence_class,
                                detail="unit is not an uninit_read")
        try:
            flagged = uninitialized_read(unit)
        except (ValueError, KeyError) as e:
            return OracleResult(OracleVerdict.NOT_APPLICABLE, self.divergence_class,
                                detail=f"ill-formed uninit_read unit: {e}")
        if flagged is None:
            return OracleResult(
                OracleVerdict.NO_DIVERGENCE_FOUND, self.divergence_class,
                detail="definedness analysis proves the read slot is initialized")

        slot, sstate = flagged
        storage = unit["storage"]
        writes = list(unit.get("writes", []))
        read = unit["read"]
        ce = self._build(storage, writes, read, slot, sstate)
        return OracleResult(OracleVerdict.DIVERGENT, self.divergence_class,
                            counterexample=ce,
                            detail=f"definedness({slot!r})={sstate}; read is undefined")

    def _build(self, storage: Dict, writes: List[Dict], read: object,
               slot: object, sstate: str) -> Counterexample:
        c_src = _c_source(storage, writes, read)
        emit = _TARGET_EMITTERS[self.target_lang]
        tgt_src = emit(storage, writes, read)
        lang = self.target_lang
        rule = {
            "rust": "Rust's type system forbids reading a possibly-uninitialized "
                    "binding; the idiomatic translation zero-initializes, giving a "
                    "single defined value.",
            "go": "Go zero-initializes all storage, so the read is defined and "
                  "deterministic.",
            "swift": "Swift requires explicit initialization; the translation "
                     "default-initializes to zero, giving a defined value.",
        }[lang]
        note = (
            f"C reads slot {slot!r} of {storage['kind']} storage while its "
            f"definedness is '{sstate}' (uninitialized): UB. Its value is "
            f"indeterminate, and the -O0 and -O2 builds of the *same* C source "
            f"disagree on the printed value. {rule}"
        )
        return Counterexample(
            divergence_class=self.divergence_class,
            source_lang="c", target_lang=lang,
            inputs={"s": 7},  # any input; the harness threads it through argv
            source_snippet=c_src, target_snippet=tgt_src,
            source_definedness=Definedness.UNDEFINED.value,
            divergence_witness=note,
            definedness_witness=(
                f"every input is a valid int; the undefinedness is structural — "
                f"slot {slot!r} is read before being written on all paths."
            ),
        )


class _GoUninitializedReadOracle(UninitializedReadOracle):
    target_lang = "go"


class _SwiftUninitializedReadOracle(UninitializedReadOracle):
    target_lang = "swift"


register(UninitializedReadOracle())
register(_GoUninitializedReadOracle())
register(_SwiftUninitializedReadOracle())
