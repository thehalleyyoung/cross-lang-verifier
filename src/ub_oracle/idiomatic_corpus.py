"""Step 42 — Tier-2 anchor corpus: human-idiomatic ports.

The toy `a / b` pairs prove the *mechanism*; this module proves the oracle keeps
its guarantees on **idiomatic, value-carrying functions** — the kind of code a
human porting a C utility to Rust/Go actually writes (named helpers, clamping,
checksums, a 64-bit-widened average, the infamous binary-search midpoint).

Each corpus item is a *real-world-shaped* function with provenance to a concrete
algorithm, a declared label, and idiomatic ports per target language:

* **divergent** items: the C relies on undefined behaviour (signed overflow in a
  midpoint, an out-of-range bit-field shift, a division whose divisor can be
  zero) while the idiomatic target port is **defined** (Rust `wrapping_*`/panic,
  Go's 64-bit `int`, …). The oracle must **confirm** the divergence on the
  UB-triggering input and stay **silent** on the safe input.
* **equivalent** items: idiomatic ports that are well-defined on *both* sides
  (a clamp, an additive checksum, a 64-bit-widened average). The oracle must
  **never** flag them — these are the true-negative idiomatic ports that a
  value-level differ would false-positive on.

Every pair is compiled and run with the real toolchain (clang/UBSan + rustc/go);
the per-item verdict layer is content-hashed so the corpus reproduces an
identical hash across runs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .reexec import ReexecHarness, toolchain_available

# --------------------------------------------------------------------------- #
# Source fragments. Operands are always read from argv so nothing is
# const-foldable (rustc's deny-by-default `unconditional_panic` /
# `arithmetic_overflow` lints reject const-evaluable UB).
# --------------------------------------------------------------------------- #
def _c(decl_body: str, reads: str, call: str) -> str:
    return ("#include <stdio.h>\n#include <stdlib.h>\n"
            f"{decl_body}\n"
            "int main(int argc,char**argv){"
            f"{reads}"
            f'printf("%d\\n",{call});return 0;}}\n')


@dataclass(frozen=True)
class IdiomaticItem:
    item_id: str
    provenance: str            # the real-world function this mirrors.
    klass: str
    declared_label: str        # "divergent" | "equivalent".
    c_src: str
    targets: Dict[str, str]    # lang -> idiomatic target source.
    ub_inputs: Tuple[str, ...]    # an input that triggers the C UB (divergent).
    safe_inputs: Tuple[str, ...]  # an input that is well-defined on both sides.


# ---- divergent items -------------------------------------------------------- #
_MIDPOINT_C = _c(
    "static int midpoint(int lo,int hi){return (lo+hi)/2;}",
    "int lo=atoi(argv[1]);int hi=atoi(argv[2]);",
    "midpoint(lo,hi)")
_MIDPOINT_RUST = (
    "fn midpoint(lo:i32,hi:i32)->i32{ lo.wrapping_add(hi) / 2 }\n"
    "fn main(){\n"
    "  let lo: i32 = std::env::args().nth(1).unwrap().parse().unwrap();\n"
    "  let hi: i32 = std::env::args().nth(2).unwrap().parse().unwrap();\n"
    "  println!(\"{}\", midpoint(lo,hi));\n}\n")
_MIDPOINT_GO = (
    "package main\nimport (\"fmt\";\"os\";\"strconv\")\n"
    "func midpoint(lo,hi int)int{return (lo+hi)/2}\n"
    "func main(){lo,_:=strconv.Atoi(os.Args[1]);hi,_:=strconv.Atoi(os.Args[2]);"
    "fmt.Println(midpoint(lo,hi))}\n")

_BITFIELD_C = _c(
    "static int field(int v,int w){return v<<w;}",
    "int v=atoi(argv[1]);int w=atoi(argv[2]);",
    "field(v,w)")
_BITFIELD_RUST = (
    "fn field(v:i32,w:u32)->i32{ v.wrapping_shl(w) }\n"
    "fn main(){\n"
    "  let v: i32 = std::env::args().nth(1).unwrap().parse().unwrap();\n"
    "  let w: u32 = std::env::args().nth(2).unwrap().parse().unwrap();\n"
    "  println!(\"{}\", field(v,w));\n}\n")

_RATE_C = _c(
    "static int rate(int total,int count){return total/count;}",
    "int t=atoi(argv[1]);int c=atoi(argv[2]);",
    "rate(t,c)")
_RATE_RUST = (
    "fn rate(total:i32,count:i32)->i32{ total / count }\n"
    "fn main(){\n"
    "  let t: i32 = std::env::args().nth(1).unwrap().parse().unwrap();\n"
    "  let c: i32 = std::env::args().nth(2).unwrap().parse().unwrap();\n"
    "  println!(\"{}\", rate(t,c));\n}\n")
_RATE_GO = (
    "package main\nimport (\"fmt\";\"os\";\"strconv\")\n"
    "func rate(total,count int)int{return total/count}\n"
    "func main(){t,_:=strconv.Atoi(os.Args[1]);c,_:=strconv.Atoi(os.Args[2]);"
    "fmt.Println(rate(t,c))}\n")

# ---- equivalent items ------------------------------------------------------- #
_AVG_C = _c(
    "static int avg(int a,int b){long s=(long)a+(long)b;return (int)(s/2);}",
    "int a=atoi(argv[1]);int b=atoi(argv[2]);",
    "avg(a,b)")
_AVG_RUST = (
    "fn avg(a:i32,b:i32)->i32{ ((a as i64 + b as i64)/2) as i32 }\n"
    "fn main(){\n"
    "  let a: i32 = std::env::args().nth(1).unwrap().parse().unwrap();\n"
    "  let b: i32 = std::env::args().nth(2).unwrap().parse().unwrap();\n"
    "  println!(\"{}\", avg(a,b));\n}\n")
_AVG_GO = (
    "package main\nimport (\"fmt\";\"os\";\"strconv\")\n"
    "func avg(a,b int)int{return (a+b)/2}\n"
    "func main(){a,_:=strconv.Atoi(os.Args[1]);b,_:=strconv.Atoi(os.Args[2]);"
    "fmt.Println(avg(a,b))}\n")

_CLAMP_C = _c(
    "static int clamp(int v){return v<0?0:(v>255?255:v);}",
    "int v=atoi(argv[1]);",
    "clamp(v)")
_CLAMP_RUST = (
    "fn clamp(v:i32)->i32{ v.max(0).min(255) }\n"
    "fn main(){\n"
    "  let v: i32 = std::env::args().nth(1).unwrap().parse().unwrap();\n"
    "  println!(\"{}\", clamp(v));\n}\n")
_CLAMP_GO = (
    "package main\nimport (\"fmt\";\"os\";\"strconv\")\n"
    "func clamp(v int)int{if v<0{return 0};if v>255{return 255};return v}\n"
    "func main(){v,_:=strconv.Atoi(os.Args[1]);fmt.Println(clamp(v))}\n")

_CKSUM_C = _c(
    "static int cks(int a,int b){return ((unsigned)a+(unsigned)b)&0xff;}",
    "int a=atoi(argv[1]);int b=atoi(argv[2]);",
    "cks(a,b)")
_CKSUM_RUST = (
    "fn cks(a:i32,b:i32)->i32{ ((a as u32).wrapping_add(b as u32) & 0xff) as i32 }\n"
    "fn main(){\n"
    "  let a: i32 = std::env::args().nth(1).unwrap().parse().unwrap();\n"
    "  let b: i32 = std::env::args().nth(2).unwrap().parse().unwrap();\n"
    "  println!(\"{}\", cks(a,b));\n}\n")
_CKSUM_GO = (
    "package main\nimport (\"fmt\";\"os\";\"strconv\")\n"
    "func cks(a,b int)int{return (a+b)&0xff}\n"
    "func main(){a,_:=strconv.Atoi(os.Args[1]);b,_:=strconv.Atoi(os.Args[2]);"
    "fmt.Println(cks(a,b))}\n")


CORPUS: Tuple[IdiomaticItem, ...] = (
    IdiomaticItem(
        "midpoint-overflow",
        "binary-search / merge midpoint `(lo+hi)/2` (the JDK / NIST-famous "
        "signed-overflow bug); idiomatic ports use wrapping (Rust) or 64-bit "
        "`int` (Go), so they are defined where C is UB.",
        "signed_overflow", "divergent",
        _MIDPOINT_C, {"rust": _MIDPOINT_RUST, "go": _MIDPOINT_GO},
        ("2000000000", "2000000000"), ("10", "20")),
    IdiomaticItem(
        "bitfield-shift",
        "bit-field / flag extraction `v << w` (as in packed-struct decoders); a "
        "width >= 32 is out-of-range UB in C, but Rust's `wrapping_shl` is "
        "defined.",
        "oversized_shift", "divergent",
        _BITFIELD_C, {"rust": _BITFIELD_RUST},
        ("1", "40"), ("1", "3")),
    IdiomaticItem(
        "rate-divide",
        "throughput/rate `total/count` (as in coreutils-style accounting); a "
        "zero divisor is UB in C, a defined panic in Rust and a defined panic in "
        "Go.",
        "div_by_zero", "divergent",
        _RATE_C, {"rust": _RATE_RUST, "go": _RATE_GO},
        ("100", "0"), ("100", "4")),
    IdiomaticItem(
        "safe-average",
        "overflow-safe average widening to 64 bits before halving; well-defined "
        "on both sides — the idiomatic fix for the midpoint bug.",
        "none", "equivalent",
        _AVG_C, {"rust": _AVG_RUST, "go": _AVG_GO},
        ("2000000000", "2000000000"), ("10", "20")),
    IdiomaticItem(
        "clamp-byte",
        "saturating clamp to [0,255] (pixel/byte saturation); no UB on either "
        "side, must never be flagged.",
        "none", "equivalent",
        _CLAMP_C, {"rust": _CLAMP_RUST, "go": _CLAMP_GO},
        ("300", ""), ("100", "")),
    IdiomaticItem(
        "additive-checksum",
        "additive checksum mod 256 using unsigned arithmetic (Internet-checksum "
        "shaped); well-defined wrap-around on both sides.",
        "none", "equivalent",
        _CKSUM_C, {"rust": _CKSUM_RUST, "go": _CKSUM_GO},
        ("100", "200"), ("1", "2")),
)


# --------------------------------------------------------------------------- #
# Per-item verdict.
# --------------------------------------------------------------------------- #
@dataclass
class ItemVerdict:
    item_id: str
    lang: str
    declared_label: str
    ub_confirmed: bool        # oracle confirmed on the UB-triggering input.
    safe_confirmed: bool      # oracle confirmed on the safe input (should be 0).
    correct: bool             # verdict matches the declared label.
    detail: str

    def key(self) -> Tuple:
        return (self.item_id, self.lang, self.declared_label,
                self.ub_confirmed, self.safe_confirmed, self.correct)


@dataclass
class CorpusReport:
    available: bool
    langs: Tuple[str, ...]
    verdicts: List[ItemVerdict] = field(default_factory=list)
    content_hash: str = ""

    @property
    def n_items(self) -> int:
        return len(self.verdicts)

    @property
    def n_divergent(self) -> int:
        return sum(1 for v in self.verdicts if v.declared_label == "divergent")

    @property
    def n_equivalent(self) -> int:
        return sum(1 for v in self.verdicts if v.declared_label == "equivalent")

    @property
    def all_correct(self) -> bool:
        return bool(self.verdicts) and all(v.correct for v in self.verdicts)

    def render(self) -> str:
        if not self.available:
            return "idiomatic corpus: toolchain unavailable (consistency only)"
        lines = [f"idiomatic anchor corpus: {self.n_items} (item x lang) verdicts "
                 f"across {list(self.langs)}  hash={self.content_hash[:16]}"]
        for v in self.verdicts:
            mark = "ok" if v.correct else "WRONG"
            lines.append(f"  [{mark:5s}] {v.item_id} ({v.lang}, {v.declared_label}): "
                         f"{v.detail}")
        lines.append(f"  => {'all correct' if self.all_correct else 'FAILURES'}")
        return "\n".join(lines)


def _expected(label: str) -> Tuple[bool, bool]:
    # (ub_confirmed, safe_confirmed) the oracle must produce.
    if label == "divergent":
        return True, False
    return False, False  # equivalent: never flagged, on any input.


def run_corpus(langs: Tuple[str, ...] = ("rust", "go")) -> CorpusReport:
    status = toolchain_available()
    avail = tuple(l for l in langs if status.full_for(l))
    if not avail:
        return CorpusReport(available=False, langs=())
    h = ReexecHarness(status)
    verdicts: List[ItemVerdict] = []
    for item in CORPUS:
        for lang in avail:
            tgt = item.targets.get(lang)
            if tgt is None:
                continue
            ub_args = [a for a in item.ub_inputs if a != ""]
            safe_args = [a for a in item.safe_inputs if a != ""]
            r_ub = h.confirm_trap_vs_defined(item.c_src, tgt, ub_args,
                                             item.klass, lang)
            r_safe = h.confirm_trap_vs_defined(item.c_src, tgt, safe_args,
                                               item.klass, lang)
            if not (r_ub.available and r_safe.available):
                continue
            ubc = bool(r_ub.confirmed)
            sfc = bool(r_safe.confirmed)
            exp_ub, exp_safe = _expected(item.declared_label)
            correct = (ubc == exp_ub) and (sfc == exp_safe)
            detail = (f"ub_input->{'flagged' if ubc else 'silent'}, "
                      f"safe_input->{'flagged' if sfc else 'silent'} "
                      f"(expected {'flag' if exp_ub else 'silent'}/"
                      f"{'flag' if exp_safe else 'silent'})")
            verdicts.append(ItemVerdict(
                item.item_id, lang, item.declared_label, ubc, sfc, correct,
                detail))
    chash = hashlib.sha256(
        json.dumps([v.key() for v in verdicts], sort_keys=True).encode()
    ).hexdigest()
    return CorpusReport(available=True, langs=avail, verdicts=verdicts,
                        content_hash=chash)


@dataclass
class CorpusConfirmation:
    available: bool
    ok: bool
    n_items: int
    n_divergent: int
    n_equivalent: int
    n_langs: int
    hash_stable: bool
    content_hash: str
    report: Optional[CorpusReport]
    detail: str


def confirm_idiomatic_corpus(
        langs: Tuple[str, ...] = ("rust", "go")) -> CorpusConfirmation:
    """Prove the oracle keeps its guarantees on idiomatic, value-carrying ports:
    every **divergent** item is flagged on its UB input and silent on its safe
    input, and every **equivalent** item is never flagged. Run twice to confirm
    the verdict layer is content-hash-stable."""
    status = toolchain_available()
    if not any(status.full_for(l) for l in langs):
        return CorpusConfirmation(
            available=False, ok=True, n_items=0, n_divergent=0, n_equivalent=0,
            n_langs=0, hash_stable=True, content_hash="", report=None,
            detail="toolchain unavailable: consistency-only pass")
    r1 = run_corpus(langs)
    r2 = run_corpus(langs)
    stable = (r1.content_hash == r2.content_hash and bool(r1.content_hash))
    # require both labels represented and at least two languages exercised.
    breadth = r1.n_divergent > 0 and r1.n_equivalent > 0 and len(r1.langs) >= 2
    ok = r1.all_correct and stable and breadth
    detail = (f"items={r1.n_items} divergent={r1.n_divergent} "
              f"equivalent={r1.n_equivalent} langs={list(r1.langs)} "
              f"all_correct={r1.all_correct} hash_stable={stable}")
    return CorpusConfirmation(
        available=True, ok=ok, n_items=r1.n_items, n_divergent=r1.n_divergent,
        n_equivalent=r1.n_equivalent, n_langs=len(r1.langs), hash_stable=stable,
        content_hash=r1.content_hash, report=r1, detail=detail)


IDIOMATIC_CORPUS_SPI = {
    "CORPUS": CORPUS,
    "run_corpus": run_corpus,
    "confirm_idiomatic_corpus": confirm_idiomatic_corpus,
}


if __name__ == "__main__":  # pragma: no cover
    conf = confirm_idiomatic_corpus()
    print(f"ok={conf.ok} {conf.detail}")
    if conf.report is not None:
        print(conf.report.render())
