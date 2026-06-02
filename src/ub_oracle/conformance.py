"""Step 38 — per-language frontend conformance suite.

The fuzzer (Step 36) proves the frontend doesn't *crash* or *drift* on random
input; a conformance suite proves it lowers *specific real-language constructs* to
*specific expected facts*. This is the regression corpus a CI gate runs on every
change: a curated set of C and Rust constructs, each paired with the exact
ingestion result it must produce. If a refactor changes how the frontend reads a
function pointer, an array-decayed parameter, or a by-value ``Vec``, a conformance
case turns red.

Each :class:`ConformanceCase` carries a source snippet and the expected lowering:
for C, the function's return type, ordered ``(name, type)`` parameters and storage
class as clang spells them (``int a[10]`` → ``int *``; ``int (*f)(int)`` →
``int (*)(int)``; a ``typedef``'d return type is preserved); for Rust, the
parameter ownership facts rustc's MIR exposes (a by-value ``Vec``/``Box``/``String``
is *moved*, a reference or ``Copy`` scalar is not).

:func:`run_conformance` executes every applicable case against the real compiler
and returns a pass/fail row per case; :func:`confirm_conformance` asserts the whole
applicable suite passes — the merge gate.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import ir_ingest as _iri

CLANG = _iri.CLANG
RUSTC = _iri.RUSTC


@dataclass(frozen=True)
class ExpectedFunction:
    name: str
    ret_type: str
    params: Tuple[Tuple[Optional[str], str], ...]   # (name, type)
    storage: str = ""
    moved_params: Optional[Tuple[str, ...]] = None  # only checked for Rust


@dataclass(frozen=True)
class ConformanceCase:
    case_id: str
    lang: str                # "c" | "rust"
    source: str
    expected: Tuple[ExpectedFunction, ...]
    note: str = ""


# --------------------------------------------------------------------------- #
# C corpus — clang-AST lowerings of real constructs.
# --------------------------------------------------------------------------- #
C_CASES: List[ConformanceCase] = [
    ConformanceCase(
        "c-typedef-ret", "c",
        "typedef unsigned int u32;\nu32 mk(void){return 0u;}\n",
        (ExpectedFunction("mk", "u32", ()),),
        "a typedef'd return type is preserved verbatim, not desugared",
    ),
    ConformanceCase(
        "c-struct-ptr-param", "c",
        "struct P{int x;};\nint getx(struct P*p){return p->x;}\n",
        (ExpectedFunction("getx", "int", (("p", "struct P *"),)),),
        "struct pointer parameter keeps its tag",
    ),
    ConformanceCase(
        "c-enum-ret", "c",
        "enum C{A,B};\nenum C pick(int n){return n? A:B;}\n",
        (ExpectedFunction("pick", "enum C", (("n", "int"),)),),
        "enum return type keeps its tag",
    ),
    ConformanceCase(
        "c-const-ptr-and-unsigned", "c",
        "int sum(const int*a,unsigned n){int s=0;for(unsigned i=0;i<n;i++)s+=a[i];return s;}\n",
        (ExpectedFunction(
            "sum", "int",
            (("a", "const int *"), ("n", "unsigned int"))),),
        "const-qualified pointer and canonical 'unsigned int' spelling",
    ),
    ConformanceCase(
        "c-array-decay-param", "c",
        "int first(int a[10]){return a[0];}\n",
        (ExpectedFunction("first", "int", (("a", "int *"),)),),
        "array parameter decays to a pointer per C semantics",
    ),
    ConformanceCase(
        "c-function-pointer-param", "c",
        "int apply(int(*f)(int),int x){return f(x);}\n",
        (ExpectedFunction(
            "apply", "int",
            (("f", "int (*)(int)"), ("x", "int"))),),
        "function-pointer parameter type is reconstructed exactly",
    ),
    ConformanceCase(
        "c-static-storage", "c",
        "static long counter(void){return 0L;}\n",
        (ExpectedFunction("counter", "long", (), storage="static"),),
        "internal linkage is read from the AST, not guessed",
    ),
]


# --------------------------------------------------------------------------- #
# Rust corpus — rustc-MIR ownership facts.
# --------------------------------------------------------------------------- #
def _rf(name: str, ret: str, pname: str, ptype: str,
        moved: Tuple[str, ...]) -> ExpectedFunction:
    return ExpectedFunction(name, ret, ((pname, ptype),), moved_params=moved)


RUST_CASES: List[ConformanceCase] = [
    ConformanceCase(
        "rust-vec-byval-moved", "rust",
        "pub fn take(v: Vec<i32>) -> usize { v.len() }\n",
        (_rf("take", "usize", "v", "Vec<i32>", ("v",)),),
        "a by-value Vec is consumed (moved/dropped) per MIR",
    ),
    ConformanceCase(
        "rust-vec-ref-not-moved", "rust",
        "pub fn peek(v: &Vec<i32>) -> usize { v.len() }\n",
        (_rf("peek", "usize", "v", "&Vec<i32>", ()),),
        "a shared reference does not consume its referent",
    ),
    ConformanceCase(
        "rust-box-byval-moved", "rust",
        "pub fn unbox(b: Box<i32>) -> i32 { *b }\n",
        (_rf("unbox", "i32", "b", "Box<i32>", ("b",)),),
        "a by-value Box is consumed",
    ),
    ConformanceCase(
        "rust-string-byval-moved", "rust",
        "pub fn consume(s: String) -> usize { s.len() }\n",
        (_rf("consume", "usize", "s", "String", ("s",)),),
        "a by-value String is consumed",
    ),
    ConformanceCase(
        "rust-copy-scalar-not-moved", "rust",
        "pub fn dbl(x: i64) -> i64 { x.wrapping_mul(2) }\n",
        (_rf("dbl", "i64", "x", "i64", ()),),
        "a Copy scalar is never moved",
    ),
    ConformanceCase(
        "rust-str-ref-not-moved", "rust",
        "pub fn slen(s: &str) -> usize { s.len() }\n",
        (_rf("slen", "usize", "s", "&str", ()),),
        "a &str borrow is not consumed",
    ),
]


ALL_CASES: List[ConformanceCase] = C_CASES + RUST_CASES


def _norm(t: str) -> str:
    return " ".join(t.split())


@dataclass
class CaseResult:
    case_id: str
    lang: str
    applicable: bool
    passed: bool
    mismatches: List[str] = field(default_factory=list)


def _check_case(case: ConformanceCase) -> CaseResult:
    if case.lang == "c":
        if not os.path.exists(CLANG):
            return CaseResult(case.case_id, "c", False, False)
        mod = _iri.ingest_clang(case.source)
    elif case.lang == "rust":
        if not os.path.exists(RUSTC):
            return CaseResult(case.case_id, "rust", False, False)
        mod = _iri.ingest_rustc_mir(case.source)
    else:  # pragma: no cover
        return CaseResult(case.case_id, case.lang, False, False, ["unknown lang"])

    res = CaseResult(case.case_id, case.lang, True, True)
    if mod is None:
        res.passed = False
        res.mismatches.append("ingest returned None")
        return res
    for ef in case.expected:
        rf = mod.functions.get(ef.name)
        if rf is None:
            res.mismatches.append(f"{ef.name}: missing from ingest")
            continue
        if _norm(rf.ret_type) != _norm(ef.ret_type):
            res.mismatches.append(
                f"{ef.name}.ret: expected {ef.ret_type!r} got {rf.ret_type!r}")
        if rf.arity != len(ef.params):
            res.mismatches.append(
                f"{ef.name}.arity: expected {len(ef.params)} got {rf.arity}")
        else:
            for (en, et), rp in zip(ef.params, rf.params):
                if en is not None and rp.name != en:
                    res.mismatches.append(
                        f"{ef.name}.param-name: expected {en!r} got {rp.name!r}")
                if _norm(rp.type) != _norm(et):
                    res.mismatches.append(
                        f"{ef.name}.param-type: expected {et!r} got {rp.type!r}")
        if case.lang == "c" and rf.storage != ef.storage:
            res.mismatches.append(
                f"{ef.name}.storage: expected {ef.storage!r} got {rf.storage!r}")
        if ef.moved_params is not None:
            if tuple(rf.moved_params) != tuple(ef.moved_params):
                res.mismatches.append(
                    f"{ef.name}.moved: expected {ef.moved_params} "
                    f"got {rf.moved_params}")
    res.passed = not res.mismatches
    return res


def run_conformance(cases: Optional[List[ConformanceCase]] = None) -> List[CaseResult]:
    return [_check_case(c) for c in (cases if cases is not None else ALL_CASES)]


@dataclass
class ConformanceConfirmation:
    clang: bool
    rustc: bool
    applicable: int
    passed: int
    results: List[CaseResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        applic = [r for r in self.results if r.applicable]
        return bool(applic) and all(r.passed for r in applic)


def confirm_conformance() -> ConformanceConfirmation:
    results = run_conformance()
    applic = [r for r in results if r.applicable]
    return ConformanceConfirmation(
        clang=os.path.exists(CLANG),
        rustc=os.path.exists(RUSTC),
        applicable=len(applic),
        passed=sum(1 for r in applic if r.passed),
        results=results,
    )


CONFORMANCE_SPI = {
    "run_conformance": run_conformance,
    "confirm_conformance": confirm_conformance,
    "ALL_CASES": ALL_CASES,
}


if __name__ == "__main__":  # pragma: no cover
    c = confirm_conformance()
    print(f"clang={c.clang} rustc={c.rustc} "
          f"applicable={c.applicable} passed={c.passed} ok={c.ok}")
    for r in c.results:
        flag = "skip" if not r.applicable else ("PASS" if r.passed else "FAIL")
        print(f"  [{flag}] {r.case_id}")
        for m in r.mismatches:
            print(f"        - {m}")
