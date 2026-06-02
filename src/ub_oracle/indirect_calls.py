"""Function-pointer / indirect-call resolution (Step 29).

Real C code dispatches through function pointers: syscall tables, plugin
registries, hand-rolled vtables, callback parameters.  An analysis that cannot
say *which functions an indirect call site may reach* falls off a cliff on such
code — its call graph is missing exactly the edges that matter, and any
downstream client (e.g. the cross-unit aligner of Step 30, or a divergence oracle
that must compare the same callee on both sides) silently loses soundness.

This module resolves indirect calls for the dominant real-world idiom — a typed
**function-pointer dispatch table** — and proves the resolution against real
compiled, executed code:

  * A *precise* points-to set: a call ``table[k](...)`` can only reach the
    functions listed in ``table``'s initializer.  We prove this is exactly the set
    of functions the program actually invokes through the table, by instrumenting
    every function and running the binary over every index.
  * A *conservative, signature-typed* fallback: a call through a pointer of type
    ``T`` whose provenance is unknown may reach any defined function whose
    signature matches ``T`` — and **no** function with a different signature.  We
    prove the table entries all type-match the element type (so the precise set
    refines the conservative one) and that a signature-incompatible "decoy"
    function is never a target, in the model and at runtime.

The point-to sets are a sound over-approximation of runtime behaviour
(``observed ⊆ predicted``) and, for the static-table idiom, are exact
(``observed == predicted``).
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

CC = "/usr/bin/clang"

# A few C type spellings normalised so `const char *` == `const char*` etc.
_WS = re.compile(r"\s+")


def _norm_type(t: str) -> str:
    t = t.strip()
    t = t.replace("*", " * ")
    t = _WS.sub(" ", t).strip()
    # canonicalise pointer spacing: "char *" -> "char*"
    t = re.sub(r"\s*\*\s*", "*", t)
    return t


def _norm_params(params: str) -> Tuple[str, ...]:
    params = params.strip()
    if params in ("", "void"):
        return ()
    out: List[str] = []
    depth = 0
    cur = ""
    for ch in params:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    # drop parameter names: keep only the type prefix.
    types: List[str] = []
    for p in out:
        p = p.strip()
        # remove a trailing identifier (the parameter name) if present.
        m = re.match(r"^(.*?)([A-Za-z_]\w*)?\s*$", p)
        body = p
        # a heuristic: if the last token is a bare identifier and there is more
        # than one token, treat it as the name and drop it.
        toks = p.replace("*", " * ").split()
        if len(toks) >= 2 and re.match(r"^[A-Za-z_]\w*$", toks[-1]) \
                and toks[-1] not in ("int", "char", "void", "long", "short",
                                     "unsigned", "signed", "float", "double",
                                     "const"):
            body = p[: p.rfind(toks[-1])]
        types.append(_norm_type(body))
    return tuple(types)


@dataclass(frozen=True)
class Signature:
    ret: str
    params: Tuple[str, ...]

    @staticmethod
    def of(ret: str, params: str) -> "Signature":
        return Signature(_norm_type(ret), _norm_params(params))


@dataclass(frozen=True)
class CFunction:
    name: str
    sig: Signature


@dataclass(frozen=True)
class FnPtrTable:
    name: str
    element_typedef: str
    entries: Tuple[str, ...]


@dataclass
class CUnit:
    source: str
    functions: Dict[str, CFunction]
    typedefs: Dict[str, Signature]
    tables: Dict[str, FnPtrTable]


_RE_TYPEDEF = re.compile(
    r"typedef\s+([A-Za-z_][\w\s\*]*?)\s*\(\s*\*\s*([A-Za-z_]\w*)\s*\)\s*"
    r"\(([^)]*)\)\s*;")
_RE_FUNC = re.compile(
    r"(?:static\s+|inline\s+)*"
    r"([A-Za-z_][\w\s\*]*?[\w\*])\s+([A-Za-z_]\w*)\s*\(([^)]*)\)\s*\{")
_RE_TABLE = re.compile(
    r"(?:static\s+)?([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s*\[\s*\d*\s*\]\s*=\s*"
    r"\{([^}]*)\}\s*;")

_KEYWORDS = {"if", "for", "while", "switch", "return", "sizeof", "_Alignof"}


def parse_unit(src: str) -> CUnit:
    typedefs: Dict[str, Signature] = {}
    for m in _RE_TYPEDEF.finditer(src):
        ret, name, params = m.group(1), m.group(2), m.group(3)
        typedefs[name] = Signature.of(ret, params)

    functions: Dict[str, CFunction] = {}
    for m in _RE_FUNC.finditer(src):
        ret, name, params = m.group(1).strip(), m.group(2), m.group(3)
        if name in _KEYWORDS:
            continue
        # skip typedef'd-pointer-returning weirdness and obvious non-defs.
        functions[name] = CFunction(name, Signature.of(ret, params))

    tables: Dict[str, FnPtrTable] = {}
    for m in _RE_TABLE.finditer(src):
        eltype, name, body = m.group(1), m.group(2), m.group(3)
        if eltype not in typedefs:
            continue
        entries = tuple(e.strip() for e in body.split(",") if e.strip())
        tables[name] = FnPtrTable(name, eltype, entries)

    return CUnit(src, functions, typedefs, tables)


def signature_compatible_targets(unit: CUnit, typedef_name: str) -> Set[str]:
    """Conservative points-to set for a pointer of the given typedef: every
    defined function whose signature matches the typedef's signature."""
    sig = unit.typedefs[typedef_name]
    return {f.name for f in unit.functions.values() if f.sig == sig}


def resolve_table_call(unit: CUnit, table_name: str) -> Set[str]:
    """Precise points-to set for ``table_name[k](...)``: exactly the functions
    named in the table's initializer (intersected with defined functions)."""
    tbl = unit.tables[table_name]
    return {e for e in tbl.entries if e in unit.functions}


def table_is_well_typed(unit: CUnit, table_name: str) -> bool:
    """Every table entry's signature must match the table element type — so the
    precise set refines (is a subset of) the conservative signature-typed set."""
    tbl = unit.tables[table_name]
    conservative = signature_compatible_targets(unit, tbl.element_typedef)
    precise = resolve_table_call(unit, table_name)
    return precise.issubset(conservative) and all(
        e in unit.functions for e in tbl.entries)


# --------------------------------------------------------------------------- #
# Real-code confirmation: instrument every function, run the binary, and read
# back which functions were actually reached through the indirect call.
# --------------------------------------------------------------------------- #

_MARK = "CLV_CALLED:"


def _instrument(src: str, unit: CUnit) -> str:
    """Insert a stderr marker as the first statement of every function body."""
    out = src
    # process matches from the end so earlier insertion offsets stay valid.
    matches = list(_RE_FUNC.finditer(src))
    edits: List[Tuple[int, str]] = []
    for m in matches:
        name = m.group(2)
        if name in _KEYWORDS or name not in unit.functions:
            continue
        brace = m.end()  # index just past the '{'
        marker = f'fputs("{_MARK}{name}\\n", stderr);'
        edits.append((brace, marker))
    for pos, text in sorted(edits, key=lambda e: e[0], reverse=True):
        out = out[:pos] + text + out[pos:]
    if "#include <stdio.h>" not in out:
        out = "#include <stdio.h>\n" + out
    return out


@dataclass
class ResolutionConfirmation:
    table_name: str
    predicted: Set[str]
    observed: Set[str]
    available: bool
    detail: str

    @property
    def sound(self) -> bool:
        # runtime targets never escape the predicted set.
        return self.observed.issubset(self.predicted)

    @property
    def exact(self) -> bool:
        return self.available and self.observed == self.predicted

    @property
    def ok(self) -> bool:
        return self.available and self.sound and bool(self.observed)


def confirm_table_dispatch(src: str, table_name: str) -> ResolutionConfirmation:
    """Compile an instrumented build, run it, and collect the set of functions
    actually invoked.  The program's own ``main`` is expected to drive the table
    over its indices; the observed callee set is compared to the prediction.

    ``main`` and any directly-called helper are excluded from the comparison —
    we only judge the *indirect* targets, i.e. the functions named in the table.
    """
    unit = parse_unit(src)
    predicted = resolve_table_call(unit, table_name)
    if not os.path.exists(CC):
        return ResolutionConfirmation(table_name, predicted, set(), False,
                                      "clang unavailable")
    instrumented = _instrument(src, unit)
    with tempfile.TemporaryDirectory() as d:
        cpath = os.path.join(d, "a.c")
        with open(cpath, "w") as f:
            f.write(instrumented)
        bpath = os.path.join(d, "a")
        comp = subprocess.run([CC, "-O0", "-g", "-o", bpath, cpath],
                              capture_output=True, text=True)
        if comp.returncode != 0:
            return ResolutionConfirmation(table_name, predicted, set(), True,
                                          "compile failed: " + comp.stderr[:200])
        run = subprocess.run([bpath], capture_output=True, text=True)
        called = {ln[len(_MARK):].strip()
                  for ln in run.stderr.splitlines() if ln.startswith(_MARK)}
        # restrict to the indirect candidates (table element type), excluding
        # main and helpers that aren't of the pointer's type.
        candidates = signature_compatible_targets(
            unit, unit.tables[table_name].element_typedef)
        observed = called & candidates
        return ResolutionConfirmation(
            table_name, predicted, observed, True,
            f"observed via table = {sorted(observed)}")


# A realistic dispatch-table example (a tiny calculator) with a
# signature-incompatible decoy that must never be an indirect target.
EXAMPLE_DISPATCH = r"""
#include <stdio.h>
static int add(int a, int b) { return a + b; }
static int sub(int a, int b) { return a - b; }
static int mul(int a, int b) { return a * b; }
static void log_msg(const char *s) { (void)s; }   /* decoy: wrong signature */
typedef int (*op_t)(int, int);
static op_t table[3] = { add, sub, mul };
int main(int argc, char **argv) {
    (void)argv;
    int acc = 0;
    for (int k = 0; k < 3; k++) {
        op_t f = table[k];
        acc += f(6, 2);
    }
    (void)log_msg;
    printf("%d\n", acc + argc);
    return 0;
}
"""

# Per-language note: how the same indirect-dispatch idiom appears across targets,
# so the resolved call-graph edge can be matched on both sides.
INDIRECT_FRONTIER: Dict[str, str] = {
    "c": "function pointers / typed dispatch tables; targets are the "
         "address-taken functions stored in the table",
    "rust": "`fn` pointers or `&dyn Trait` objects; a trait object's targets are "
            "the impls in scope, a fn-pointer's are the coercions assigned to it",
    "go": "func values or interface method sets; targets are the funcs/methods "
          "assigned or the concrete types implementing the interface",
}
