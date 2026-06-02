"""Step 44 — known-bug / CVE corpus.

A curated corpus of *real, UB-rooted bug classes* — each tagged with its CWE (the
unambiguous weakness identifier behind countless CVEs) — translated across
language pairs, where the oracle confirms a genuine **definedness divergence**:
the C program executes undefined behavior (the UBSan/bounds build traps on a
concrete input) while the target translation (Rust or Go) has a fully defined,
deterministic outcome (a clean panic or a defined value). Every entry is proven
end-to-end against the real clang + rustc/go through :class:`reexec.ReexecHarness`
— nothing here is asserted, it is *executed*.

This is the "we catch real bugs" table: each row is a weakness class that has
produced real CVEs in C codebases, and a faithful machine-style translation that
does *not* silently inherit the bug — exactly the divergence a migration auditor
must be told about.

The corpus is data-driven: a new entry is a :class:`CveCase` (CWE id, description,
C source, per-language target sources, concrete inputs). :func:`run_corpus` drives
every applicable case through the harness; :func:`confirm_corpus` asserts every
applicable case confirms a divergence.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .reexec import ReexecHarness, ToolchainStatus, toolchain_available


def _c_main(body: str) -> str:
    return ("#include <stdio.h>\n#include <stdlib.h>\n"
            "int main(int argc, char **argv) {\n" + body + "\n  return 0;\n}\n")


def _rust_main(body: str) -> str:
    return "use std::env;\nfn main() {\n" + body + "\n}\n"


def _go_main(body: str) -> str:
    return ("package main\n\n"
            "import (\n\t\"fmt\"\n\t\"os\"\n\t\"strconv\"\n)\n\n"
            "func main() {\n" + body + "\n}\n")


# --------------------------------------------------------------------------- #
# A single corpus entry and the per-language translations.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CveCase:
    case_id: str
    cwe: str                       # e.g. "CWE-369"
    title: str
    description: str
    c_src: str
    targets: Tuple[Tuple[str, str], ...]   # (lang, target_src)
    inputs: Tuple[str, ...]
    divergence_class: str

    def target_for(self, lang: str) -> Optional[str]:
        for l, s in self.targets:
            if l == lang:
                return s
        return None


# --------------------------------------------------------------------------- #
# The corpus. Every C body reads its integer arguments from argv and prints one
# result; the matching target does the same with the language's *defined*
# semantics (panic or wrapping), so on the trapping input C is undefined while the
# target is defined and deterministic.
# --------------------------------------------------------------------------- #
def _div_c() -> str:
    return _c_main('  int a = atoi(argv[1]);\n  int b = atoi(argv[2]);\n'
                   '  printf("%d\\n", a / b);')


def _div_rust() -> str:
    return _rust_main(
        '  let a: i32 = env::args().nth(1).unwrap().parse().unwrap();\n'
        '  let b: i32 = env::args().nth(2).unwrap().parse().unwrap();\n'
        '  println!("{}", a / b);')


def _div_go() -> str:
    return _go_main(
        '\ta, _ := strconv.Atoi(os.Args[1])\n\tb, _ := strconv.Atoi(os.Args[2])\n'
        '\tfmt.Println(a / b)')


def _oob_c() -> str:
    return _c_main('  int a[4] = {10, 20, 30, 40};\n  int i = atoi(argv[1]);\n'
                   '  printf("%d\\n", a[i]);')


def _oob_rust() -> str:
    return _rust_main(
        '  let a = [10i32, 20, 30, 40];\n'
        '  let i: usize = env::args().nth(1).unwrap().parse().unwrap();\n'
        '  println!("{}", a[i]);')


def _oob_go() -> str:
    return _go_main(
        '\ta := []int{10, 20, 30, 40}\n'
        '\ti, _ := strconv.Atoi(os.Args[1])\n\tfmt.Println(a[i])')


def _ovf_c() -> str:
    return _c_main('  int a = atoi(argv[1]);\n  int b = atoi(argv[2]);\n'
                   '  printf("%d\\n", a + b);')


def _ovf_rust() -> str:
    return _rust_main(
        '  let a: i32 = env::args().nth(1).unwrap().parse().unwrap();\n'
        '  let b: i32 = env::args().nth(2).unwrap().parse().unwrap();\n'
        '  println!("{}", a.wrapping_add(b));')


def _shift_c() -> str:
    return _c_main('  int x = atoi(argv[1]);\n  int s = atoi(argv[2]);\n'
                   '  printf("%d\\n", x << s);')


def _shift_rust() -> str:
    return _rust_main(
        '  let x: i32 = env::args().nth(1).unwrap().parse().unwrap();\n'
        '  let s: u32 = env::args().nth(2).unwrap().parse().unwrap();\n'
        '  println!("{}", x.wrapping_shl(s));')


def _intmin_c() -> str:
    return _c_main('  int a = atoi(argv[1]);\n  int b = atoi(argv[2]);\n'
                   '  printf("%d\\n", a / b);')


def _intmin_rust() -> str:
    return _rust_main(
        '  let a: i32 = env::args().nth(1).unwrap().parse().unwrap();\n'
        '  let b: i32 = env::args().nth(2).unwrap().parse().unwrap();\n'
        '  println!("{}", a.wrapping_div(b));')


CORPUS: List[CveCase] = [
    CveCase(
        "div-by-zero", "CWE-369", "Division by zero",
        "Integer division by zero is undefined behavior in C; the target "
        "languages define it as a deterministic panic. A migration that 'works' "
        "in C by luck (the divisor is never zero on the tested paths) becomes a "
        "hard panic in the target — a behavior change the auditor must see.",
        _div_c(),
        (("rust", _div_rust()), ("go", _div_go())),
        ("7", "0"), "division_by_zero",
    ),
    CveCase(
        "oob-array-read", "CWE-125", "Out-of-bounds array read",
        "Reading past the end of a fixed-size array is undefined behavior in C "
        "(the root of countless information-disclosure CVEs); Rust and Go both "
        "bounds-check and panic deterministically instead of leaking adjacent "
        "memory.",
        _oob_c(),
        (("rust", _oob_rust()), ("go", _oob_go())),
        ("10",), "out_of_bounds_read",
    ),
    CveCase(
        "signed-overflow", "CWE-190", "Signed integer overflow",
        "Signed integer overflow is undefined behavior in C (the weakness behind "
        "many size-computation CVEs); the target's wrapping/panicking semantics "
        "are defined, so the value the program computes on the overflowing input "
        "is no longer whatever the C optimizer chose.",
        _ovf_c(),
        (("rust", _ovf_rust()),),
        ("2147483647", "1"), "signed_overflow",
    ),
    CveCase(
        "oversized-shift", "CWE-758", "Shift by >= bit-width",
        "Shifting an int by an amount >= its width is undefined behavior in C; "
        "the target defines it (wrapping shift / panic). Crypto and serialization "
        "code that shifts by a runtime count is a classic source of this bug.",
        _shift_c(),
        (("rust", _shift_rust()),),
        ("1", "40"), "oversized_shift",
    ),
    CveCase(
        "int-min-div-neg1", "CWE-682", "INT_MIN / -1 overflow",
        "INT_MIN / -1 (and INT_MIN % -1) overflows the signed result and is "
        "undefined behavior in C — it can trap at the hardware level; the target "
        "defines the operation, so the migration's behavior on this boundary "
        "input diverges from any legal C compilation.",
        _intmin_c(),
        (("rust", _intmin_rust()),),
        ("-2147483648", "-1"), "int_min_div_neg1",
    ),
]


# --------------------------------------------------------------------------- #
# Running the corpus through the real re-execution harness.
# --------------------------------------------------------------------------- #
@dataclass
class CorpusEntryResult:
    case_id: str
    cwe: str
    lang: str
    available: bool
    confirmed: bool
    reason: str = ""


@dataclass
class CorpusReport:
    results: List[CorpusEntryResult] = field(default_factory=list)

    @property
    def applicable(self) -> List[CorpusEntryResult]:
        return [r for r in self.results if r.available]

    @property
    def confirmed_count(self) -> int:
        return sum(1 for r in self.applicable if r.confirmed)

    @property
    def ok(self) -> bool:
        applic = self.applicable
        return bool(applic) and all(r.confirmed for r in applic)


def run_corpus(status: Optional[ToolchainStatus] = None,
               langs: Optional[Tuple[str, ...]] = None) -> CorpusReport:
    """Drive every (case, target-language) pair through the harness."""
    st = status or toolchain_available()
    h = ReexecHarness(st)
    rep = CorpusReport()
    for case in CORPUS:
        for lang, tgt in case.targets:
            if langs is not None and lang not in langs:
                continue
            if not st.full_for(lang):
                rep.results.append(CorpusEntryResult(
                    case.case_id, case.cwe, lang, available=False, confirmed=False,
                    reason="toolchain unavailable"))
                continue
            res = h.confirm_trap_vs_defined(
                case.c_src, tgt, list(case.inputs),
                divergence_class=case.divergence_class, target_lang=lang)
            rep.results.append(CorpusEntryResult(
                case.case_id, case.cwe, lang,
                available=res.available, confirmed=res.confirmed,
                reason=res.reason))
    return rep


@dataclass
class CorpusConfirmation:
    available: bool
    ok: bool
    report: CorpusReport


def confirm_corpus(langs: Optional[Tuple[str, ...]] = None) -> CorpusConfirmation:
    rep = run_corpus(langs=langs)
    applic = rep.applicable
    return CorpusConfirmation(available=bool(applic), ok=rep.ok, report=rep)


def coverage_table() -> List[Dict[str, str]]:
    """A compact 'we catch real bugs' table: one row per (CWE, case, langs)."""
    rows = []
    for c in CORPUS:
        rows.append({
            "cwe": c.cwe,
            "case": c.case_id,
            "title": c.title,
            "langs": ", ".join(l for l, _ in c.targets),
        })
    return rows


CVE_CORPUS_SPI = {
    "run_corpus": run_corpus,
    "confirm_corpus": confirm_corpus,
    "coverage_table": coverage_table,
    "CORPUS": CORPUS,
}


if __name__ == "__main__":  # pragma: no cover
    c = confirm_corpus()
    print(f"available={c.available} ok={c.ok} "
          f"confirmed={c.report.confirmed_count}/{len(c.report.applicable)}")
    for r in c.report.results:
        flag = "n/a" if not r.available else ("CATCH" if r.confirmed else "MISS")
        print(f"  [{flag:5s}] {r.cwe:8s} {r.case_id:18s} C->{r.lang}")
        if r.available and not r.confirmed:
            print(f"          {r.reason[:140]}")
