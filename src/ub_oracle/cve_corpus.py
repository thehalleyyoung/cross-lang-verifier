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
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
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


def _ovf_go() -> str:
    return _go_main(
        '\ta64, _ := strconv.ParseInt(os.Args[1], 10, 32)\n'
        '\tb64, _ := strconv.ParseInt(os.Args[2], 10, 32)\n'
        '\ta, b := int32(a64), int32(b64)\n'
        '\tfmt.Println(a + b)')


def _underflow_c() -> str:
    return _c_main('  int a = atoi(argv[1]);\n  int b = atoi(argv[2]);\n'
                   '  printf("%d\\n", a - b);')


def _underflow_rust() -> str:
    return _rust_main(
        '  let a: i32 = env::args().nth(1).unwrap().parse().unwrap();\n'
        '  let b: i32 = env::args().nth(2).unwrap().parse().unwrap();\n'
        '  println!("{}", a.wrapping_sub(b));')


def _underflow_go() -> str:
    return _go_main(
        '\ta64, _ := strconv.ParseInt(os.Args[1], 10, 32)\n'
        '\tb64, _ := strconv.ParseInt(os.Args[2], 10, 32)\n'
        '\ta, b := int32(a64), int32(b64)\n'
        '\tfmt.Println(a - b)')


def _shift_go() -> str:
    return _go_main(
        '\tx64, _ := strconv.ParseInt(os.Args[1], 10, 32)\n'
        '\ts64, _ := strconv.ParseUint(os.Args[2], 10, 32)\n'
        '\tx := int32(x64)\n'
        '\tfmt.Println(x << uint32(s64))')


def _intmin_go() -> str:
    return _go_main(
        '\ta64, _ := strconv.ParseInt(os.Args[1], 10, 32)\n'
        '\tb64, _ := strconv.ParseInt(os.Args[2], 10, 32)\n'
        '\ta, b := int32(a64), int32(b64)\n'
        '\tfmt.Println(a / b)')


def _oob_write_c() -> str:
    return _c_main('  int a[4] = {10, 20, 30, 40};\n  int i = atoi(argv[1]);\n'
                   '  a[i] = 99;\n  printf("%d\\n", a[0]);')


def _oob_write_rust() -> str:
    return _rust_main(
        '  let mut a = [10i32, 20, 30, 40];\n'
        '  let i: usize = env::args().nth(1).unwrap().parse().unwrap();\n'
        '  a[i] = 99;\n  println!("{}", a[0]);')


def _oob_write_go() -> str:
    return _go_main(
        '\ta := []int{10, 20, 30, 40}\n'
        '\ti, _ := strconv.Atoi(os.Args[1])\n\ta[i] = 99\n'
        '\tfmt.Println(a[0])')


def _null_c() -> str:
    return _c_main('  int flag = atoi(argv[1]);\n  int value = 7;\n'
                   '  int *p = flag ? NULL : &value;\n  printf("%d\\n", *p);')


def _null_rust() -> str:
    return _rust_main(
        '  let flag: i32 = env::args().nth(1).unwrap().parse().unwrap();\n'
        '  let value = 7i32;\n'
        '  let p = if flag != 0 { None } else { Some(value) };\n'
        '  println!("{}", p.unwrap());')


def _null_go() -> str:
    return _go_main(
        '\tflag, _ := strconv.Atoi(os.Args[1])\n\tvalue := 7\n'
        '\tvar p *int\n\tif flag == 0 { p = &value }\n'
        '\tfmt.Println(*p)')


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
# Step 157 — historical-CVE weakness-class replays.
#
# These entries deliberately do *not* vendor third-party vulnerable code.  Each
# CVE id below is an NVD-indexed historical entry for the stated CWE family, and
# the generated bundle is a from-scratch minimized replay of that family against
# the same real compiler pipeline.  The scope is therefore "CVE weakness-class
# replay", not "the original vendor bug reproduction".
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HistoricalReplayTemplate:
    template_id: str
    cwe: str
    title: str
    divergence_class: str
    c_src: str
    targets: Tuple[Tuple[str, str], ...]
    witness_inputs: Tuple[str, ...]
    safe_inputs: Tuple[str, ...]

    def target_for(self, lang: str) -> Optional[str]:
        for l, s in self.targets:
            if l == lang:
                return s
        return None

    @property
    def langs(self) -> Tuple[str, ...]:
        return tuple(l for l, _ in self.targets)


@dataclass(frozen=True)
class HistoricalCveCase:
    cve_id: str
    nvd_cwe: str
    replay_template: str
    historical_family: str
    nvd_query: str
    replay_scope: str = (
        "from-scratch minimized weakness-class replay; not original vendor source"
    )


HISTORICAL_TEMPLATES: Tuple[HistoricalReplayTemplate, ...] = (
    HistoricalReplayTemplate(
        "signed-overflow", "CWE-190", "Signed integer overflow",
        "signed_overflow", _ovf_c(),
        (("rust", _ovf_rust()), ("go", _ovf_go())),
        ("2147483647", "1"), ("4", "5"),
    ),
    HistoricalReplayTemplate(
        "signed-underflow", "CWE-191", "Signed integer underflow",
        "signed_underflow", _underflow_c(),
        (("rust", _underflow_rust()), ("go", _underflow_go())),
        ("-2147483648", "1"), ("4", "5"),
    ),
    HistoricalReplayTemplate(
        "division-by-zero", "CWE-369", "Division by zero",
        "division_by_zero", _div_c(),
        (("rust", _div_rust()), ("go", _div_go())),
        ("7", "0"), ("8", "2"),
    ),
    HistoricalReplayTemplate(
        "oob-array-read", "CWE-125", "Out-of-bounds array read",
        "out_of_bounds_read", _oob_c(),
        (("rust", _oob_rust()), ("go", _oob_go())),
        ("10",), ("2",),
    ),
    HistoricalReplayTemplate(
        "oob-array-write", "CWE-787", "Out-of-bounds array write",
        "out_of_bounds_write", _oob_write_c(),
        (("rust", _oob_write_rust()), ("go", _oob_write_go())),
        ("10",), ("2",),
    ),
    HistoricalReplayTemplate(
        "null-deref", "CWE-476", "Null pointer dereference",
        "null_deref", _null_c(),
        (("rust", _null_rust()), ("go", _null_go())),
        ("1",), ("0",),
    ),
    HistoricalReplayTemplate(
        "oversized-shift", "CWE-758", "Reliance on undefined behavior",
        "oversized_shift", _shift_c(),
        (("rust", _shift_rust()), ("go", _shift_go())),
        ("1", "40"), ("1", "3"),
    ),
    HistoricalReplayTemplate(
        "int-min-div-neg1", "CWE-682", "Incorrect calculation boundary",
        "int_min_div_neg1", _intmin_c(),
        (("rust", _intmin_rust()), ("go", _intmin_go())),
        ("-2147483648", "-1"), ("8", "2"),
    ),
)

_TEMPLATE_BY_ID: Dict[str, HistoricalReplayTemplate] = {
    t.template_id: t for t in HISTORICAL_TEMPLATES
}


def _hist(cwe: str, template: str, family: str,
          ids: Tuple[str, ...]) -> Tuple[HistoricalCveCase, ...]:
    return tuple(
        HistoricalCveCase(
            cve_id=cve_id,
            nvd_cwe=cwe,
            replay_template=template,
            historical_family=family,
            nvd_query=f"https://services.nvd.nist.gov/rest/json/cves/2.0?cweId={cwe}",
        )
        for cve_id in ids
    )


HISTORICAL_CVE_CASES: Tuple[HistoricalCveCase, ...] = (
    *_hist("CWE-190", "signed-overflow", "NVD CWE-190 integer overflow", (
        "CVE-2002-0639", "CVE-2002-0391", "CVE-2004-0657",
        "CVE-2004-0788", "CVE-2004-2013", "CVE-2005-0102",
        "CVE-2005-1141", "CVE-2005-1513",
    )),
    *_hist("CWE-191", "signed-underflow", "NVD CWE-191 integer underflow", (
        "CVE-2004-0184", "CVE-2004-0816", "CVE-2004-1002",
        "CVE-2005-0199", "CVE-2005-1891", "CVE-2007-0063",
        "CVE-2009-3301", "CVE-2010-2497",
    )),
    *_hist("CWE-369", "division-by-zero", "NVD CWE-369 divide by zero", (
        "CVE-2004-0804", "CVE-2006-5939", "CVE-2007-2723",
        "CVE-2007-2237", "CVE-2007-3268", "CVE-2009-1887",
        "CVE-2010-4165", "CVE-2011-1012",
    )),
    *_hist("CWE-125", "oob-array-read", "NVD CWE-125 out-of-bounds read", (
        "CVE-2004-0183", "CVE-2004-0221", "CVE-2004-0421",
        "CVE-2004-0112", "CVE-2004-1940", "CVE-2007-3847",
        "CVE-2009-2523", "CVE-2010-4577",
    )),
    *_hist("CWE-787", "oob-array-write", "NVD CWE-787 out-of-bounds write", (
        "CVE-2002-2227", "CVE-2003-0870", "CVE-2003-1396",
        "CVE-2004-0398", "CVE-2004-0488", "CVE-2004-0783",
        "CVE-2004-0574", "CVE-2004-1189",
    )),
    *_hist("CWE-476", "null-deref", "NVD CWE-476 null pointer dereference", (
        "CVE-2001-1559", "CVE-2002-0401", "CVE-2002-1912",
        "CVE-2003-1000", "CVE-2003-1013", "CVE-2004-0365",
        "CVE-2004-0119", "CVE-2004-0389",
    )),
    *_hist("CWE-758", "oversized-shift", "NVD CWE-758 undefined behavior", (
        "CVE-2026-21684", "CVE-2026-21685", "CVE-2026-21686",
        "CVE-2026-21687", "CVE-2026-22858", "CVE-2026-24404",
    )),
    *_hist("CWE-682", "int-min-div-neg1", "NVD CWE-682 incorrect calculation", (
        "CVE-2011-1573", "CVE-2011-3062", "CVE-2016-7433",
        "CVE-2016-9377", "CVE-2017-0545", "CVE-2017-8326",
    )),
)


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


def historical_cve_cases(limit: Optional[int] = None) -> Tuple[HistoricalCveCase, ...]:
    """Return the NVD-indexed historical CVE weakness-class replay manifest."""
    cases = HISTORICAL_CVE_CASES if limit is None else HISTORICAL_CVE_CASES[:limit]
    return tuple(cases)


def historical_coverage_table() -> List[Dict[str, str]]:
    """One row per historical CVE id, explicitly scoped to weakness replays."""
    rows = []
    for c in HISTORICAL_CVE_CASES:
        t = _TEMPLATE_BY_ID[c.replay_template]
        rows.append({
            "cve_id": c.cve_id,
            "cwe": c.nvd_cwe,
            "template": c.replay_template,
            "title": t.title,
            "langs": ", ".join(t.langs),
            "scope": c.replay_scope,
            "nvd_query": c.nvd_query,
        })
    return rows


def _confirm_template(h: ReexecHarness, template: HistoricalReplayTemplate,
                      lang: str, inputs: Tuple[str, ...]):
    target = template.target_for(lang)
    if target is None:
        raise ValueError(f"template {template.template_id!r} has no {lang!r} target")
    return h.confirm_trap_vs_defined(
        template.c_src, target, list(inputs), template.divergence_class, lang)


def run_historical_corpus(status: Optional[ToolchainStatus] = None,
                          langs: Optional[Tuple[str, ...]] = None,
                          limit: Optional[int] = None) -> CorpusReport:
    """Replay historical-CVE weakness classes through the real harness.

    The returned rows are keyed by CVE id, but the source snippets are minimized
    from-scratch witnesses for that CVE's NVD CWE family. This is intentionally
    stronger than a metadata table (the code compiles and traps), while remaining
    honest about not bundling third-party vulnerable source.
    """
    st = status or toolchain_available()
    h = ReexecHarness(st)
    rep = CorpusReport()
    for case in historical_cve_cases(limit):
        template = _TEMPLATE_BY_ID[case.replay_template]
        for lang, _target in template.targets:
            if langs is not None and lang not in langs:
                continue
            if not st.full_for(lang):
                rep.results.append(CorpusEntryResult(
                    case.cve_id, case.nvd_cwe, lang, available=False,
                    confirmed=False, reason="toolchain unavailable"))
                continue
            res = _confirm_template(h, template, lang, template.witness_inputs)
            rep.results.append(CorpusEntryResult(
                case.cve_id, case.nvd_cwe, lang,
                available=res.available, confirmed=res.confirmed,
                reason=res.reason))
    return rep


def confirm_historical_corpus(langs: Optional[Tuple[str, ...]] = None,
                              limit: Optional[int] = None) -> CorpusConfirmation:
    rep = run_historical_corpus(langs=langs, limit=limit)
    applic = rep.applicable
    return CorpusConfirmation(available=bool(applic), ok=rep.ok, report=rep)


def _quote_args(args: Tuple[str, ...]) -> str:
    return " ".join(shlex.quote(a) for a in args)


def _target_source_suffix(lang: str) -> str:
    if lang == "rust":
        return "rs"
    if lang == "go":
        return "go"
    raise ValueError(f"historical CVE bundles support rust/go targets, got {lang!r}")


def _target_compile_script(lang: str) -> str:
    if lang == "rust":
        return (
            'RUSTC="${RUSTC:-rustc}"\n'
            '"$RUSTC" -O "$WORK/target.rs" -o "$WORK/target_bin" '
            '> "$WORK/target_compile.out" 2> "$WORK/target_compile.err"\n'
            'target_compile_rc=$?\n'
        )
    if lang == "go":
        return (
            'GO="${GO:-go}"\n'
            'GOCACHE="$WORK/.gocache" "$GO" build -o "$WORK/target_bin" '
            '"$WORK/target.go" > "$WORK/target_compile.out" '
            '2> "$WORK/target_compile.err"\n'
            'target_compile_rc=$?\n'
        )
    raise ValueError(f"unsupported target language: {lang}")


def _target_allowed_case(lang: str, rc_var: str) -> str:
    if lang == "rust":
        return f'case "${{{rc_var}}}" in 0|101) ;; *) exit 21 ;; esac\n'
    if lang == "go":
        return f'case "${{{rc_var}}}" in 0|2) ;; *) exit 21 ;; esac\n'
    raise ValueError(f"unsupported target language: {lang}")


def historical_reproduction_bundle(case: HistoricalCveCase, lang: str) -> str:
    """Render a self-contained shell bundle for one CVE-family replay.

    The script exits successfully only if the C/UBSan build traps on the
    witnessing input, the target language has a defined outcome on that same
    input, and both sides run cleanly on the safe input.
    """
    template = _TEMPLATE_BY_ID[case.replay_template]
    target_src = template.target_for(lang)
    if target_src is None:
        raise ValueError(f"{case.cve_id} has no {lang} target")
    witness = _quote_args(template.witness_inputs)
    safe = _quote_args(template.safe_inputs)
    suffix = _target_source_suffix(lang)
    compile_target = _target_compile_script(lang)
    allowed_witness = _target_allowed_case(lang, "target_witness_rc")
    allowed_safe = _target_allowed_case(lang, "target_safe_rc")
    return (
        "#!/usr/bin/env bash\n"
        f"# {case.cve_id} ({case.nvd_cwe}) weakness-class replay: {template.title}\n"
        f"# Scope: {case.replay_scope}.\n"
        f"# NVD class query: {case.nvd_query}\n"
        "set -u\n"
        'WORK="$(mktemp -d)"\n'
        'trap \'rm -rf "$WORK"\' EXIT\n'
        'CC="${CC:-clang}"\n'
        "cat > \"$WORK/source.c\" <<'CLV_C_EOF'\n"
        f"{template.c_src.rstrip()}\n"
        "CLV_C_EOF\n"
        f"cat > \"$WORK/target.{suffix}\" <<'CLV_T_EOF'\n"
        f"{target_src.rstrip()}\n"
        "CLV_T_EOF\n"
        '"$CC" -O1 -fsanitize=undefined -fno-sanitize-recover=all '
        '"$WORK/source.c" -o "$WORK/source_san" '
        '> "$WORK/c_compile.out" 2> "$WORK/c_compile.err"\n'
        'c_compile_rc=$?\n'
        'if [ "$c_compile_rc" -ne 0 ]; then cat "$WORK/c_compile.err"; exit 10; fi\n'
        f"{compile_target}"
        'if [ "$target_compile_rc" -ne 0 ]; then '
        'cat "$WORK/target_compile.err"; exit 11; fi\n'
        f'"$WORK/source_san" {witness} '
        '> "$WORK/c_witness.out" 2> "$WORK/c_witness.err"\n'
        'c_witness_rc=$?\n'
        'if [ "$c_witness_rc" -eq 0 ] || '
        '! grep -qi "runtime error" "$WORK/c_witness.err"; then\n'
        '  echo "expected C/UBSan runtime error on witnessing input" >&2\n'
        '  cat "$WORK/c_witness.out" "$WORK/c_witness.err" >&2\n'
        '  exit 20\n'
        'fi\n'
        f'"$WORK/target_bin" {witness} '
        '> "$WORK/target_witness.out" 2> "$WORK/target_witness.err"\n'
        'target_witness_rc=$?\n'
        f"{allowed_witness}"
        f'"$WORK/source_san" {safe} '
        '> "$WORK/c_safe.out" 2> "$WORK/c_safe.err"\n'
        'c_safe_rc=$?\n'
        'if [ "$c_safe_rc" -ne 0 ]; then\n'
        '  echo "expected safe C input to run cleanly" >&2\n'
        '  cat "$WORK/c_safe.out" "$WORK/c_safe.err" >&2\n'
        '  exit 22\n'
        'fi\n'
        f'"$WORK/target_bin" {safe} '
        '> "$WORK/target_safe.out" 2> "$WORK/target_safe.err"\n'
        'target_safe_rc=$?\n'
        f"{allowed_safe}"
        f'echo "{case.cve_id} {lang}: C/UBSan traps on witness; '
        'target is defined; safe input runs cleanly"\n'
    )


@dataclass
class HistoricalBundleCheck:
    generated: int
    linted: int
    executed: int
    ok: bool
    failures: Tuple[str, ...] = ()


def _bundle_name(case: HistoricalCveCase, lang: str) -> str:
    return f"{case.cve_id}-{lang}-{case.replay_template}.sh"


def write_historical_reproduction_bundles(
    out_dir: str,
    langs: Tuple[str, ...] = ("rust", "go"),
    limit: Optional[int] = None,
) -> Tuple[str, ...]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []
    for case in historical_cve_cases(limit):
        template = _TEMPLATE_BY_ID[case.replay_template]
        for lang in langs:
            if lang not in template.langs:
                continue
            path = out / _bundle_name(case, lang)
            path.write_text(historical_reproduction_bundle(case, lang), encoding="utf-8")
            path.chmod(0o755)
            paths.append(str(path))
    return tuple(paths)


def check_historical_reproduction_bundles(
    out_dir: str,
    langs: Tuple[str, ...] = ("rust", "go"),
    limit: Optional[int] = None,
    execute: bool = False,
    status: Optional[ToolchainStatus] = None,
) -> HistoricalBundleCheck:
    paths = write_historical_reproduction_bundles(out_dir, langs, limit)
    failures: List[str] = []
    linted = 0
    executed = 0
    for p in paths:
        chk = subprocess.run(["bash", "-n", p], capture_output=True, text=True)
        if chk.returncode == 0:
            linted += 1
        else:
            failures.append(f"{os.path.basename(p)}: bash -n failed: {chk.stderr}")
    if execute:
        st = status or toolchain_available()
        env = dict(os.environ)
        if st.cc:
            env["CC"] = st.cc
        for p in paths:
            base = os.path.basename(p)
            lang_name = "go" if "-go-" in base else "rust"
            if not st.full_for(lang_name):
                failures.append(f"{base}: {lang_name} toolchain unavailable")
                continue
            run = subprocess.run(
                ["bash", p], capture_output=True, text=True, env=env, timeout=90)
            if run.returncode == 0:
                executed += 1
            else:
                failures.append(
                    f"{base}: execution failed rc={run.returncode}: "
                    f"{(run.stdout + run.stderr)[-500:]}")
    return HistoricalBundleCheck(
        generated=len(paths),
        linted=linted,
        executed=executed,
        ok=(not failures and linted == len(paths)
            and (not execute or executed == len(paths))),
        failures=tuple(failures),
    )


CVE_CORPUS_SPI = {
    "run_corpus": run_corpus,
    "confirm_corpus": confirm_corpus,
    "coverage_table": coverage_table,
    "CORPUS": CORPUS,
    "historical_cve_cases": historical_cve_cases,
    "historical_coverage_table": historical_coverage_table,
    "run_historical_corpus": run_historical_corpus,
    "confirm_historical_corpus": confirm_historical_corpus,
    "historical_reproduction_bundle": historical_reproduction_bundle,
    "write_historical_reproduction_bundles": write_historical_reproduction_bundles,
    "check_historical_reproduction_bundles": check_historical_reproduction_bundles,
    "HISTORICAL_CVE_CASES": HISTORICAL_CVE_CASES,
    "HISTORICAL_TEMPLATES": HISTORICAL_TEMPLATES,
    "CveCase": CveCase,
    "HistoricalCveCase": HistoricalCveCase,
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
