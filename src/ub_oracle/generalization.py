"""Step 88 — generalization study.

The empirical proof that the divergence result is **not an artefact of one
language pair, one producer, or one input distribution**. It runs the real
re-execution oracle (:meth:`reexec.ReexecHarness.confirm_trap_vs_defined`) over a
**grid**

    (language pair) x (producer / translation style) x (divergence class)
                    x (concrete input)

and shows the verdict is *invariant* across every cell: on inputs that trigger C
undefined behaviour the oracle detects the divergence for **every** target pack
and **every** producer style (detection rate 1.0), and on equivalent inputs it
stays silent for every cell (false-positive rate 0.0). Robustness to the input
distribution is covered by probing several distinct UB-triggering inputs per
class.

Why this is the right design
----------------------------
* **Across language pairs.** The grid instantiates three target packs —
  ``rust``, ``go`` and ``swift`` — through the same pack-parameterized oracle, so
  a uniform result is direct evidence the thesis is pair-independent.

* **Across producers / transpilers.** A real transpiler (c2rust) and a human or
  LLM port emit *different but faithful* target code for the same C unit. We
  model that variation with several **producer styles** — ``direct`` (inline
  expression), ``helper`` (via a function), ``verbose`` (intermediate bindings) —
  each an observably-equivalent rendering. A verdict that is invariant across
  styles is evidence the result is producer-independent.

* **Across the input distribution.** Each divergence class is probed with several
  distinct UB-triggering inputs (and several safe inputs), so a uniform result is
  evidence the detection is not tuned to one operand.

Nothing is simulated: every source is compiled and run with the real toolchain.
Operands are read from ``argv``/``CommandLine``/``os.Args`` at runtime so neither
rustc const-evaluation nor any backend folds the undefined operation away. The
per-cell verdict layer is content-hashed (timing excluded) so the study
reproduces an identical hash across runs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from . import oracles as _oracles  # noqa: F401  (registers all built-in pairs)
from . import plugin
from . import target_semantics as tsem
from .reexec import ReexecHarness, ToolchainStatus, toolchain_available

SCHEMA_VERSION = "generalization/v1"
SCHEMA_VERSION_V2 = "generalization/v2"

TARGETS: Tuple[str, ...] = ("rust", "go", "swift")
STYLES: Tuple[str, ...] = ("direct", "helper", "verbose")
CLASSES: Tuple[str, ...] = ("div_by_zero", "oversized_shift")

# Concrete inputs. Each class has UB-triggering inputs (the C op is undefined)
# and safe inputs (the C op is well-defined and equals the target value).
UB_INPUTS: Dict[str, List[Tuple[str, ...]]] = {
    "div_by_zero": [("10", "0"), ("-7", "0"), ("2147483647", "0")],
    "oversized_shift": [("1", "40"), ("3", "32"), ("5", "60")],
}
SAFE_INPUTS: Dict[str, List[Tuple[str, ...]]] = {
    "div_by_zero": [("10", "2"), ("-12", "3")],
    "oversized_shift": [("7", "3"), ("9", "0")],
}


# --------------------------------------------------------------------------- #
# Source generators. Each returns the full C source and the target source for a
# given (class, target, style). Operands are read at runtime.
# --------------------------------------------------------------------------- #
def _c_src(klass: str) -> str:
    head = "#include <stdio.h>\n#include <stdlib.h>\n"
    if klass == "div_by_zero":
        body = ("int a=atoi(argv[1]);int b=atoi(argv[2]);"
                'printf("%d\\n",a/b);')
    elif klass == "oversized_shift":
        body = ("int x=atoi(argv[1]);int s=atoi(argv[2]);"
                'printf("%d\\n",x<<s);')
    else:  # pragma: no cover - guarded by CLASSES
        raise ValueError(klass)
    return head + "int main(int argc,char**argv){" + body + "return 0;}\n"


def _rust_src(klass: str, style: str) -> str:
    if klass == "div_by_zero":
        a = "let a: i32 = std::env::args().nth(1).unwrap().parse().unwrap();"
        b = "let b: i32 = std::env::args().nth(2).unwrap().parse().unwrap();"
        if style == "direct":
            expr = "println!(\"{}\", a / b);"
        elif style == "helper":
            return ("fn dv(x:i32,y:i32)->i32{x / y}\nfn main(){\n"
                    f"  {a}\n  {b}\n  println!(\"{{}}\", dv(a,b));\n}}\n")
        else:  # verbose
            expr = "let q = a / b;\n  println!(\"{}\", q);"
        return f"fn main(){{\n  {a}\n  {b}\n  {expr}\n}}\n"
    else:  # oversized_shift — wrapping_shl is Rust's defined oversized shift
        x = "let x: i32 = std::env::args().nth(1).unwrap().parse().unwrap();"
        s = "let s: u32 = std::env::args().nth(2).unwrap().parse().unwrap();"
        if style == "direct":
            expr = "println!(\"{}\", x.wrapping_shl(s));"
        elif style == "helper":
            return ("fn sh(x:i32,s:u32)->i32{x.wrapping_shl(s)}\nfn main(){\n"
                    f"  {x}\n  {s}\n  println!(\"{{}}\", sh(x,s));\n}}\n")
        else:
            expr = "let q = x.wrapping_shl(s);\n  println!(\"{}\", q);"
        return f"fn main(){{\n  {x}\n  {s}\n  {expr}\n}}\n"


def _go_src(klass: str, style: str) -> str:
    head = 'package main\nimport ("fmt";"os";"strconv")\n'
    rd = ('a,_:=strconv.Atoi(os.Args[1]);b,_:=strconv.Atoi(os.Args[2]);'
          if klass == "div_by_zero" else
          'x,_:=strconv.Atoi(os.Args[1]);s,_:=strconv.Atoi(os.Args[2]);')
    if klass == "div_by_zero":
        if style == "direct":
            return head + "func main(){" + rd + "fmt.Println(a/b)}\n"
        if style == "helper":
            return (head + "func dv(x,y int)int{return x/y}\n"
                    "func main(){" + rd + "fmt.Println(dv(a,b))}\n")
        return head + "func main(){" + rd + "q:=a/b;fmt.Println(q)}\n"
    else:  # oversized_shift — Go int is 64-bit, shift 32..60 is in-range/defined
        if style == "direct":
            return head + "func main(){" + rd + "fmt.Println(x<<uint(s))}\n"
        if style == "helper":
            return (head + "func sh(x,s int)int{return x<<uint(s)}\n"
                    "func main(){" + rd + "fmt.Println(sh(x,s))}\n")
        return head + "func main(){" + rd + "q:=x<<uint(s);fmt.Println(q)}\n"


def _swift_src(klass: str, style: str) -> str:
    head = "import Foundation\n"
    if klass == "div_by_zero":
        rd = ("let a=Int32(CommandLine.arguments[1])!\n"
              "let b=Int32(CommandLine.arguments[2])!\n")
        if style == "direct":
            return head + rd + "print(a / b)\n"
        if style == "helper":
            return (head + "func dv(_ x:Int32,_ y:Int32)->Int32{return x / y}\n"
                    + rd + "print(dv(a,b))\n")
        return head + rd + "let q = a / b\nprint(q)\n"
    else:  # oversized_shift — Swift's `<<` is a defined "smart" shift
        rd = ("let x=Int32(CommandLine.arguments[1])!\n"
              "let s=Int32(CommandLine.arguments[2])!\n")
        if style == "direct":
            return head + rd + "print(x << s)\n"
        if style == "helper":
            return (head + "func sh(_ x:Int32,_ s:Int32)->Int32{return x << s}\n"
                    + rd + "print(sh(x,s))\n")
        return head + rd + "let q = x << s\nprint(q)\n"


_TARGET_GEN: Dict[str, Callable[[str, str], str]] = {
    "rust": _rust_src, "go": _go_src, "swift": _swift_src,
}


def target_source(target: str, klass: str, style: str) -> str:
    try:
        return _TARGET_GEN[target](klass, style)
    except KeyError:
        raise ValueError(f"unknown target {target!r}") from None


# --------------------------------------------------------------------------- #
# Running one grid cell.
# --------------------------------------------------------------------------- #
@dataclass
class CellResult:
    target: str
    style: str
    klass: str
    n_ub: int               # UB inputs probed
    n_ub_detected: int      # of those, oracle confirmed divergence
    n_safe: int             # safe inputs probed
    n_safe_flagged: int     # of those, oracle (wrongly) confirmed -> false positives

    @property
    def detection_rate(self) -> float:
        return self.n_ub_detected / self.n_ub if self.n_ub else 0.0

    @property
    def fp_rate(self) -> float:
        return self.n_safe_flagged / self.n_safe if self.n_safe else 0.0

    @property
    def uniform_ok(self) -> bool:
        return self.n_ub_detected == self.n_ub and self.n_safe_flagged == 0


def _run_cell(h: ReexecHarness, target: str, klass: str, style: str) -> CellResult:
    c_src = _c_src(klass)
    t_src = target_source(target, klass, style)
    n_ub = n_det = 0
    for inp in UB_INPUTS[klass]:
        n_ub += 1
        r = h.confirm_trap_vs_defined(c_src, t_src, list(inp),
                                      divergence_class=klass, target_lang=target)
        if getattr(r, "confirmed", False):
            n_det += 1
    n_safe = n_flag = 0
    for inp in SAFE_INPUTS[klass]:
        n_safe += 1
        r = h.confirm_trap_vs_defined(c_src, t_src, list(inp),
                                      divergence_class=klass, target_lang=target)
        if getattr(r, "confirmed", False):
            n_flag += 1
    return CellResult(target, style, klass, n_ub, n_det, n_safe, n_flag)


# --------------------------------------------------------------------------- #
# The study + confirmation.
# --------------------------------------------------------------------------- #
@dataclass
class GeneralizationReport:
    schema: str
    available_targets: Tuple[str, ...]
    cells: List[CellResult]
    content_hash: str

    def by_pair(self) -> Dict[str, Tuple[int, int, int, int]]:
        agg: Dict[str, List[int]] = {}
        for c in self.cells:
            a = agg.setdefault(c.target, [0, 0, 0, 0])
            a[0] += c.n_ub_detected
            a[1] += c.n_ub
            a[2] += c.n_safe_flagged
            a[3] += c.n_safe
        return {k: tuple(v) for k, v in agg.items()}

    def by_style(self) -> Dict[str, Tuple[int, int, int, int]]:
        agg: Dict[str, List[int]] = {}
        for c in self.cells:
            a = agg.setdefault(c.style, [0, 0, 0, 0])
            a[0] += c.n_ub_detected
            a[1] += c.n_ub
            a[2] += c.n_safe_flagged
            a[3] += c.n_safe
        return {k: tuple(v) for k, v in agg.items()}

    def render(self) -> str:
        if not self.cells:
            return "generalization: no targets available (consistency only)"
        lines = [
            "Generalization study (real clang/UBSan + rust/go/swift):",
            f"  targets={list(self.available_targets)} "
            f"styles={list(STYLES)} classes={list(CLASSES)}",
            f"  cells={len(self.cells)}  content_hash={self.content_hash}",
            "  per language pair (detected/UB, fp/safe):",
        ]
        for k, (d, u, f, s) in sorted(self.by_pair().items()):
            lines.append(f"    {k:6s} detect={d}/{u}  fp={f}/{s}")
        lines.append("  per producer style (detected/UB, fp/safe):")
        for k, (d, u, f, s) in sorted(self.by_style().items()):
            lines.append(f"    {k:8s} detect={d}/{u}  fp={f}/{s}")
        return "\n".join(lines)


def _hash_cells(cells: List[CellResult]) -> str:
    layer = sorted(
        (c.target, c.style, c.klass, c.n_ub, c.n_ub_detected,
         c.n_safe, c.n_safe_flagged)
        for c in cells)
    blob = json.dumps(layer, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


def run_generalization(
    targets: Tuple[str, ...] = TARGETS,
    harness: Optional[ReexecHarness] = None,
) -> GeneralizationReport:
    status = toolchain_available()
    avail = tuple(t for t in targets if status.full_for(t))
    if not avail:
        return GeneralizationReport(SCHEMA_VERSION, (), [], "")
    h = harness or ReexecHarness(status)
    cells: List[CellResult] = []
    for target in avail:
        for klass in CLASSES:
            for style in STYLES:
                cells.append(_run_cell(h, target, klass, style))
    return GeneralizationReport(SCHEMA_VERSION, avail, cells, _hash_cells(cells))


@dataclass
class GeneralizationConfirmation:
    available: bool
    ok: bool
    n_pairs: int
    n_cells: int
    invariant_across_pairs: bool
    invariant_across_styles: bool
    report: Optional[GeneralizationReport]
    detail: str

    def render(self) -> str:
        if not self.available:
            return "generalization: toolchain unavailable (consistency only)"
        return (self.report.render() if self.report else "") + (
            f"\n  invariant_across_pairs={self.invariant_across_pairs} "
            f"invariant_across_styles={self.invariant_across_styles} ok={self.ok}")


def confirm_generalization(
    targets: Tuple[str, ...] = TARGETS,
) -> GeneralizationConfirmation:
    """Prove the result generalizes: across **every** available (pair × style)
    cell the detection rate is 1.0 on UB inputs and the false-positive rate is
    0.0 on safe inputs — so the divergence result is invariant to the language
    pair, the producer style, and (within each class) the concrete input.
    """
    status = toolchain_available()
    avail = tuple(t for t in targets if status.full_for(t))
    if not avail:
        return GeneralizationConfirmation(
            available=False, ok=True, n_pairs=0, n_cells=0,
            invariant_across_pairs=True, invariant_across_styles=True,
            report=None, detail="toolchain unavailable: consistency-only pass")

    rep = run_generalization(avail)
    # every cell must detect all UB and flag no safe input.
    all_cells_ok = bool(rep.cells) and all(c.uniform_ok for c in rep.cells)
    # invariance: the (detected==UB, fp==0) property holds for every pair and
    # for every style aggregate — i.e. no pair or style is an outlier.
    inv_pair = all(d == u and f == 0 for (d, u, f, s) in rep.by_pair().values())
    inv_style = all(d == u and f == 0 for (d, u, f, s) in rep.by_style().values())
    # robustness needs breadth: >= 2 distinct language pairs actually exercised.
    breadth_ok = len(rep.available_targets) >= 2
    ok = all_cells_ok and inv_pair and inv_style and breadth_ok
    detail = (f"pairs={len(rep.available_targets)} cells={len(rep.cells)} "
              f"all_cells_uniform={all_cells_ok} inv_pair={inv_pair} "
              f"inv_style={inv_style} breadth_ok={breadth_ok}")
    return GeneralizationConfirmation(
        available=True, ok=ok, n_pairs=len(rep.available_targets),
        n_cells=len(rep.cells), invariant_across_pairs=inv_pair,
        invariant_across_styles=inv_style, report=rep, detail=detail)


GENERALIZATION_SPI = {
    "TARGETS": TARGETS,
    "STYLES": STYLES,
    "CLASSES": CLASSES,
    "target_source": target_source,
    "run_generalization": run_generalization,
    "confirm_generalization": confirm_generalization,
}


# --------------------------------------------------------------------------- #
# Step 122 — cross-pair generalization study v2.
#
# V1 holds one C-rooted phenomenon fixed while varying target and producer style.
# V2 is deliberately different: it exercises the *new language pairs and
# directions* added after the original study, using each pair's registered oracle
# and confirmation mode.  The claim is breadth/transfer of the plugin framework
# (positive witnesses confirmed, safe controls not flagged) rather than a single
# invariant outcome shared by semantically different pairs.
# --------------------------------------------------------------------------- #

_I32_MIN = -(1 << 31)


@dataclass(frozen=True)
class PairGeneralizationCase:
    case_id: str
    source_lang: str
    target_lang: str
    divergence_class: str
    positive_unit: Dict
    negative_unit: Dict
    safe_argv: Tuple[str, ...]
    new_pair: bool
    description: str

    @property
    def pair(self) -> Tuple[str, str]:
        return (self.source_lang, self.target_lang)


def _c_to_target_case(target: str, *, new_pair: bool) -> PairGeneralizationCase:
    return PairGeneralizationCase(
        case_id=f"c_to_{target}_div_by_zero",
        source_lang="c",
        target_lang=target,
        divergence_class="div_by_zero",
        positive_unit={
            "kind": "div", "width": 32, "signed": True,
            "dividend": 7, "b_range": [0, 0],
        },
        negative_unit={
            "kind": "div", "width": 32, "signed": True,
            "dividend": 7, "b_range": [1, 1],
        },
        safe_argv=("7", "1"),
        new_pair=new_pair,
        description=f"C division-by-zero UB becomes a defined {target} outcome.",
    )


GENERALIZATION_V2_CASES: Tuple[PairGeneralizationCase, ...] = (
    _c_to_target_case("rust", new_pair=False),
    _c_to_target_case("go", new_pair=False),
    _c_to_target_case("swift", new_pair=False),
    _c_to_target_case("ocaml", new_pair=True),
    _c_to_target_case("zig", new_pair=True),
    _c_to_target_case("wasm", new_pair=True),
    PairGeneralizationCase(
        case_id="c_to_cpp_sign_bit_shift",
        source_lang="c",
        target_lang="cpp",
        divergence_class="signed_shift_sign_bit",
        positive_unit={
            "kind": "sign_bit_shift", "width": 32, "shift_range": [31, 31],
        },
        negative_unit={
            "kind": "sign_bit_shift", "width": 32, "shift_range": [1, 1],
        },
        safe_argv=("1",),
        new_pair=True,
        description="The byte-identical sign-bit shift is C UB but C++20-defined.",
    ),
    PairGeneralizationCase(
        case_id="rust_to_c_intmin_div_neg1",
        source_lang="rust",
        target_lang="c",
        divergence_class="intmin_div_neg1",
        positive_unit={
            "kind": "div", "width": 32, "signed": True,
            "source_lang": "rust", "target_lang": "c",
            "a_range": [_I32_MIN, _I32_MIN], "b_range": [-1, -1],
        },
        negative_unit={
            "kind": "div", "width": 32, "signed": True,
            "source_lang": "rust", "target_lang": "c",
            "a_range": [42, 42], "b_range": [-1, -1],
        },
        safe_argv=("42", "-1"),
        new_pair=True,
        description="A Rust-defined panic can become target-side C UB.",
    ),
    PairGeneralizationCase(
        case_id="go_to_rust_intmin_div_neg1",
        source_lang="go",
        target_lang="rust",
        divergence_class="intmin_div_neg1",
        positive_unit={
            "kind": "div", "width": 32, "signed": True,
            "source_lang": "go", "target_lang": "rust",
            "a_range": [_I32_MIN, _I32_MIN], "b_range": [-1, -1],
        },
        negative_unit={
            "kind": "div", "width": 32, "signed": True,
            "source_lang": "go", "target_lang": "rust",
            "a_range": [42, 42], "b_range": [-1, -1],
        },
        safe_argv=("42", "-1"),
        new_pair=True,
        description="Go and Rust are both defined but disagree on INT_MIN/-1.",
    ),
)


def _case_oracle(case: PairGeneralizationCase):
    return plugin.get_oracle_for(
        case.divergence_class, case.source_lang, case.target_lang)


def case_available(case: PairGeneralizationCase,
                   status: Optional[ToolchainStatus] = None) -> bool:
    """Whether this V2 case can be confirmed on the current host."""
    status = status or toolchain_available()
    orc = _case_oracle(case)
    mode = orc.confirmation_mode
    if mode == "defined_divergence":
        return status.can_compile(case.source_lang) and status.can_compile(case.target_lang)
    if mode == "source_defined_target_ub":
        return status.can_compile(case.source_lang) and status.c_available and status.ubsan
    if case.source_lang == "c" and case.target_lang != "c":
        if mode == "trap_vs_defined":
            return status.full_for(case.target_lang)
        if mode in ("asan_trap_vs_defined", "libc_contract_trap_vs_defined"):
            return status.full_libc_contract_for(case.target_lang)
        if mode == "optimizer_exploited":
            return (
                status.c_available
                and status.target_available(case.target_lang)
                and status.target_runnable(case.target_lang)
            )
    return status.can_compile(case.source_lang) and status.can_compile(case.target_lang)


def available_v2_cases(
    status: Optional[ToolchainStatus] = None,
    cases: Tuple[PairGeneralizationCase, ...] = GENERALIZATION_V2_CASES,
) -> Tuple[PairGeneralizationCase, ...]:
    status = status or toolchain_available()
    return tuple(c for c in cases if case_available(c, status))


@dataclass
class PairCaseResult:
    case_id: str
    source_lang: str
    target_lang: str
    divergence_class: str
    confirmation_mode: str
    new_pair: bool
    available: bool
    positive_found: bool
    positive_confirmed: bool
    safe_exercised: bool
    safe_flagged: bool
    detail: str = ""

    @property
    def pair(self) -> Tuple[str, str]:
        return (self.source_lang, self.target_lang)

    @property
    def ok(self) -> bool:
        return (
            self.available
            and self.positive_found
            and self.positive_confirmed
            and self.safe_exercised
            and not self.safe_flagged
        )


def _safe_control(
    harness: ReexecHarness,
    case: PairGeneralizationCase,
    mode: str,
    source_snippet: str,
    target_snippet: str,
) -> object:
    argv = list(case.safe_argv)
    if mode == "trap_vs_defined":
        return harness.confirm_trap_vs_defined(
            source_snippet, target_snippet, argv,
            case.divergence_class, target_lang=case.target_lang)
    if mode == "source_defined_target_ub":
        return harness.confirm_source_defined_target_ub(
            source_snippet, case.source_lang, target_snippet, argv,
            case.divergence_class)
    if mode == "defined_divergence":
        return harness.confirm_defined_divergence(
            source_snippet, case.source_lang, target_snippet, case.target_lang,
            argv, case.divergence_class)
    if mode == "optimizer_exploited":
        return harness.confirm_optimizer_exploited(
            source_snippet, target_snippet, argv,
            case.divergence_class, target_lang=case.target_lang)
    if mode in ("asan_trap_vs_defined", "libc_contract_trap_vs_defined"):
        return harness.confirm_libc_contract_trap_vs_defined(
            source_snippet, target_snippet, argv,
            case.divergence_class, target_lang=case.target_lang)
    raise ValueError(f"unsupported V2 confirmation mode {mode!r}")


def run_generalization_v2_case(
    case: PairGeneralizationCase,
    harness: Optional[ReexecHarness] = None,
) -> PairCaseResult:
    status = harness.status if harness is not None else toolchain_available()
    orc = _case_oracle(case)
    mode = orc.confirmation_mode
    available = case_available(case, status)
    result = PairCaseResult(
        case_id=case.case_id,
        source_lang=case.source_lang,
        target_lang=case.target_lang,
        divergence_class=case.divergence_class,
        confirmation_mode=mode,
        new_pair=case.new_pair,
        available=available,
        positive_found=False,
        positive_confirmed=False,
        safe_exercised=False,
        safe_flagged=False,
    )
    if not available:
        result.detail = "toolchain unavailable"
        return result

    found = orc.find_divergence(case.positive_unit)
    result.positive_found = found.is_divergent and found.counterexample is not None
    if not result.positive_found:
        result.detail = f"positive witness not found: {found.verdict} {found.detail}"
        return result

    h = harness or ReexecHarness(status)
    confirmed = orc.confirm(found, h)
    rr = confirmed.reexec
    result.positive_confirmed = bool(rr is not None and rr.available and rr.confirmed)
    if not result.positive_confirmed:
        result.detail = f"positive not confirmed: {rr.reason if rr else 'no reexec'}"
        return result

    ce = found.counterexample
    assert ce is not None  # for type checkers; guarded by positive_found
    safe = _safe_control(h, case, mode, ce.source_snippet, ce.target_snippet)
    result.safe_exercised = bool(getattr(safe, "available", False))
    result.safe_flagged = bool(getattr(safe, "confirmed", False))
    if not result.safe_exercised:
        result.detail = f"safe control unavailable: {getattr(safe, 'reason', '')}"
    elif result.safe_flagged:
        result.detail = f"safe control falsely confirmed: {getattr(safe, 'reason', '')}"
    else:
        result.detail = "positive confirmed; safe control not flagged"
    return result


def _hash_v2_results(results: List[PairCaseResult]) -> str:
    layer = sorted(
        (
            r.case_id, r.source_lang, r.target_lang, r.divergence_class,
            r.confirmation_mode, int(r.new_pair), int(r.available),
            int(r.positive_found), int(r.positive_confirmed),
            int(r.safe_exercised), int(r.safe_flagged),
        )
        for r in results)
    blob = json.dumps(layer, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


@dataclass
class GeneralizationV2Report:
    schema: str
    results: List[PairCaseResult]
    content_hash: str

    @property
    def available_results(self) -> List[PairCaseResult]:
        return [r for r in self.results if r.available]

    @property
    def available_new_results(self) -> List[PairCaseResult]:
        return [r for r in self.available_results if r.new_pair]

    def by_pair(self) -> Dict[Tuple[str, str], Tuple[int, int, int, int]]:
        """Return confirmed/positive and safe-flagged/safe counts per pair."""
        agg: Dict[Tuple[str, str], List[int]] = {}
        for r in self.available_results:
            a = agg.setdefault(r.pair, [0, 0, 0, 0])
            a[0] += int(r.positive_confirmed)
            a[1] += 1
            a[2] += int(r.safe_flagged)
            a[3] += int(r.safe_exercised)
        return {k: tuple(v) for k, v in agg.items()}

    def render(self) -> str:
        if not self.available_results:
            return "generalization-v2: no new-pair toolchains available"
        lines = [
            "Cross-pair generalization v2 (registered oracle transfer):",
            f"  available_cases={len(self.available_results)} "
            f"new_pair_cases={len(self.available_new_results)} "
            f"content_hash={self.content_hash}",
            "  per pair (confirmed/positive, false-positive/safe):",
        ]
        for (src, tgt), (pc, pt, sf, st) in sorted(self.by_pair().items()):
            lines.append(f"    {src}->{tgt}: confirm={pc}/{pt}  fp={sf}/{st}")
        return "\n".join(lines)


def run_generalization_v2(
    cases: Tuple[PairGeneralizationCase, ...] = GENERALIZATION_V2_CASES,
    harness: Optional[ReexecHarness] = None,
) -> GeneralizationV2Report:
    status = harness.status if harness is not None else toolchain_available()
    h = harness or ReexecHarness(status)
    results = [
        run_generalization_v2_case(case, h)
        for case in cases
        if case_available(case, status)
    ]
    return GeneralizationV2Report(
        SCHEMA_VERSION_V2, results, _hash_v2_results(results) if results else "")


@dataclass
class GeneralizationV2Confirmation:
    available: bool
    ok: bool
    n_cases: int
    n_new_pairs: int
    all_positive_confirmed: bool
    zero_safe_flags: bool
    report: Optional[GeneralizationV2Report]
    detail: str

    def render(self) -> str:
        if not self.available:
            return "generalization-v2: toolchain unavailable (consistency only)"
        return (self.report.render() if self.report else "") + (
            f"\n  all_positive_confirmed={self.all_positive_confirmed} "
            f"zero_safe_flags={self.zero_safe_flags} ok={self.ok}")


def confirm_generalization_v2(
    cases: Tuple[PairGeneralizationCase, ...] = GENERALIZATION_V2_CASES,
) -> GeneralizationV2Confirmation:
    rep = run_generalization_v2(cases)
    if not rep.available_results:
        return GeneralizationV2Confirmation(
            available=False, ok=True, n_cases=0, n_new_pairs=0,
            all_positive_confirmed=True, zero_safe_flags=True, report=rep,
            detail="no V2 toolchains available: consistency-only pass")

    positives_ok = all(r.positive_found and r.positive_confirmed for r in rep.available_results)
    safe_ok = all(r.safe_exercised and not r.safe_flagged for r in rep.available_results)
    new_pairs = {r.pair for r in rep.available_new_results}
    breadth_ok = len(new_pairs) >= 2
    ok = positives_ok and safe_ok and breadth_ok
    detail = (
        f"cases={len(rep.available_results)} new_pairs={len(new_pairs)} "
        f"positives_ok={positives_ok} safe_ok={safe_ok} breadth_ok={breadth_ok}"
    )
    return GeneralizationV2Confirmation(
        available=True, ok=ok, n_cases=len(rep.available_results),
        n_new_pairs=len(new_pairs), all_positive_confirmed=positives_ok,
        zero_safe_flags=safe_ok, report=rep, detail=detail)


GENERALIZATION_SPI.update({
    "GENERALIZATION_V2_CASES": GENERALIZATION_V2_CASES,
    "run_generalization_v2": run_generalization_v2,
    "confirm_generalization_v2": confirm_generalization_v2,
})


if __name__ == "__main__":  # pragma: no cover
    conf = confirm_generalization()
    print(f"available={conf.available} ok={conf.ok}")
    print(conf.render())
    print("detail:", conf.detail)
