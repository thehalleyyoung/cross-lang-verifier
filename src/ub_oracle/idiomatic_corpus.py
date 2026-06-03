"""Step 42 — Tier-2 anchor corpus: human-idiomatic ports.

The toy `a / b` pairs prove the *mechanism*; this module proves the oracle keeps
its guarantees on **idiomatic, value-carrying functions** — the kind of code a
human porting a C utility to Rust/Go actually writes (named helpers, clamping,
checksums, checked arithmetic, a 64-bit-widened average, the infamous
binary-search midpoint).  Step 161 extends that anchor with explicit
coreutils/sudo-rs/zlib-rs-class ports that are intentionally not literal
translations, while remaining compiler-confirmed extraction units.

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
from .oracles.uninit_padding import demo_sources as _padding_demo_sources

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

_MEMCPY_C = (
    "#include <stdio.h>\n#include <stdlib.h>\n#include <stdint.h>\n#include <string.h>\n"
    "#ifdef CLV_CHECK_MEMCPY\n"
    "static void *clv_checked_memcpy(void *dst,const void *src,size_t n){"
    "uintptr_t d=(uintptr_t)dst;uintptr_t s=(uintptr_t)src;"
    "if(n>0&&d<s+n&&s<d+n){fprintf(stderr,\"runtime error: memcpy-param-overlap dst=%p src=%p n=%zu\\n\",dst,src,n);abort();}"
    "return memmove(dst,src,n);}\n"
    "#define memcpy(d,s,n) clv_checked_memcpy((d),(s),(n))\n"
    "#endif\n"
    "static void shift(char *buf,int dst,int src,int n){memcpy(buf+dst,buf+src,(size_t)n);}\n"
    "int main(int argc,char**argv){"
    "int dst=atoi(argv[1]);int src=atoi(argv[2]);int n=atoi(argv[3]);"
    "char buf[17]=\"ABCDEFGHIJKLMNOP\";"
    "if(dst<0||src<0||n<0||dst+n>16||src+n>16)return 3;"
    "shift(buf,dst,src,n);printf(\"%s\\n\",buf);return 0;}\n")
_MEMCPY_RUST = (
    "fn main(){\n"
    "  let a: Vec<String> = std::env::args().collect();\n"
    "  let dst: usize = a[1].parse().unwrap();\n"
    "  let src: usize = a[2].parse().unwrap();\n"
    "  let n: usize = a[3].parse().unwrap();\n"
    "  let mut buf = b\"ABCDEFGHIJKLMNOP\".to_vec();\n"
    "  buf.copy_within(src..src+n, dst);\n"
    "  println!(\"{}\", String::from_utf8_lossy(&buf));\n}\n")
_MEMCPY_GO = (
    "package main\nimport (\"fmt\";\"os\";\"strconv\")\n"
    "func main(){dst,_:=strconv.Atoi(os.Args[1]);src,_:=strconv.Atoi(os.Args[2]);"
    "n,_:=strconv.Atoi(os.Args[3]);buf:=[]byte(\"ABCDEFGHIJKLMNOP\");"
    "copy(buf[dst:dst+n],buf[src:src+n]);fmt.Println(string(buf))}\n")

_PADDING_C, _PADDING_RUST, _PADDING_UB, _PADDING_SAFE = _padding_demo_sources("rust")
_, _PADDING_GO, _, _ = _padding_demo_sources("go")

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


# ---- Step 161: idiomatic real-port-family expansion ------------------------- #
_COREUTILS_BLOCKS_C = _c(
    "static int blocks_from_bytes(int bytes){return (bytes+511)/512;}",
    "int b=atoi(argv[1]);",
    "blocks_from_bytes(b)")
_COREUTILS_BLOCKS_RUST = (
    "fn blocks_from_bytes(bytes:i32)->i32{\n"
    "  bytes.checked_add(511).map(|rounded| rounded / 512).unwrap_or(i32::MAX)\n"
    "}\n"
    "fn main(){\n"
    "  let b: i32 = std::env::args().nth(1).unwrap().parse().unwrap();\n"
    "  println!(\"{}\", blocks_from_bytes(b));\n}\n")
_COREUTILS_BLOCKS_GO = (
    "package main\nimport (\"fmt\";\"os\";\"strconv\")\n"
    "func blocksFromBytes(bytes int64)int64{if bytes>2147483136{return 2147483647};"
    "return (bytes+511)/512}\n"
    "func main(){b,_:=strconv.ParseInt(os.Args[1],10,32);"
    "fmt.Println(blocksFromBytes(b))}\n")

_SUDO_TIMEOUT_C = _c(
    "static int timeout_slice(int remaining,int attempts){return remaining/attempts;}",
    "int r=atoi(argv[1]);int a=atoi(argv[2]);",
    "timeout_slice(r,a)")
_SUDO_TIMEOUT_RUST = (
    "fn timeout_slice(remaining:i32,attempts:i32)->i32{\n"
    "  remaining.checked_div(attempts).unwrap_or(remaining)\n"
    "}\n"
    "fn main(){\n"
    "  let r: i32 = std::env::args().nth(1).unwrap().parse().unwrap();\n"
    "  let a: i32 = std::env::args().nth(2).unwrap().parse().unwrap();\n"
    "  println!(\"{}\", timeout_slice(r,a));\n}\n")
_SUDO_TIMEOUT_GO = (
    "package main\nimport (\"fmt\";\"os\";\"strconv\")\n"
    "func timeoutSlice(remaining,attempts int64)int64{if attempts==0{return remaining};"
    "return remaining/attempts}\n"
    "func main(){r,_:=strconv.ParseInt(os.Args[1],10,32);"
    "a,_:=strconv.ParseInt(os.Args[2],10,32);fmt.Println(timeoutSlice(r,a))}\n")

_ZLIB_ADLER_C = _c(
    "static int adler_window_step(int s1,int byte){"
    "unsigned acc=(unsigned)s1+(unsigned)(byte&0xff);return (int)(acc%65521u);}",
    "int s=atoi(argv[1]);int b=atoi(argv[2]);",
    "adler_window_step(s,b)")
_ZLIB_ADLER_RUST = (
    "fn adler_window_step(s1:i32,byte:i32)->i32{\n"
    "  ((s1 as u32).wrapping_add((byte as u32) & 0xff) % 65521) as i32\n"
    "}\n"
    "fn main(){\n"
    "  let s: i32 = std::env::args().nth(1).unwrap().parse().unwrap();\n"
    "  let b: i32 = std::env::args().nth(2).unwrap().parse().unwrap();\n"
    "  println!(\"{}\", adler_window_step(s,b));\n}\n")
_ZLIB_ADLER_GO = (
    "package main\nimport (\"fmt\";\"os\";\"strconv\")\n"
    "func adlerWindowStep(s1, b int64)int64{acc:=uint32(s1)+uint32(b&0xff);"
    "return int64(acc%65521)}\n"
    "func main(){s,_:=strconv.ParseInt(os.Args[1],10,32);"
    "b,_:=strconv.ParseInt(os.Args[2],10,32);fmt.Println(adlerWindowStep(s,b))}\n")

STEP161_FAMILY_IDS = (
    "coreutils-block-rounding",
    "sudo-rs-timeout-slice",
    "zlib-rs-adler-window",
)


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
        "memcpy-overlap",
        "in-place buffer shift written with `memcpy` instead of `memmove`; the "
        "overlapping C call is UB, while Rust `copy_within` and Go `copy` have "
        "defined memmove-like slice semantics.",
        "memcpy_overlap", "divergent",
        _MEMCPY_C, {"rust": _MEMCPY_RUST, "go": _MEMCPY_GO},
        ("1", "0", "4"), ("8", "0", "4")),
    IdiomaticItem(
        "uninit-padding",
        "whole-struct byte serialization after assigning fields; C padding bytes "
        "are indeterminate, while safe Rust/Go serializers start from zeroed bytes "
        "and write only fields.",
        "uninit_padding", "divergent",
        _PADDING_C, {"rust": _PADDING_RUST, "go": _PADDING_GO},
        _PADDING_UB, _PADDING_SAFE),
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
    IdiomaticItem(
        "coreutils-block-rounding",
        "uutils/coreutils-class block-count rounding: C `bytes+511` has a latent "
        "signed-overflow precondition, while the idiomatic Rust/Go ports make the "
        "overflow policy explicit with checked arithmetic and saturation.",
        "signed_overflow", "divergent",
        _COREUTILS_BLOCKS_C,
        {"rust": _COREUTILS_BLOCKS_RUST, "go": _COREUTILS_BLOCKS_GO},
        ("2147483400",), ("1024",)),
    IdiomaticItem(
        "sudo-rs-timeout-slice",
        "sudo-rs-class timeout/backoff calculation: a zero attempt count is C "
        "division UB, while the idiomatic ports route the precondition through "
        "`checked_div` / an explicit zero guard.",
        "div_by_zero", "divergent",
        _SUDO_TIMEOUT_C, {"rust": _SUDO_TIMEOUT_RUST, "go": _SUDO_TIMEOUT_GO},
        ("30", "0"), ("30", "3")),
    IdiomaticItem(
        "zlib-rs-adler-window",
        "zlib-rs-class Adler/window checksum update: the port deliberately uses "
        "unsigned modular arithmetic on both sides, serving as an idiomatic "
        "true-equivalence control for the corpus expansion.",
        "none", "equivalent",
        _ZLIB_ADLER_C, {"rust": _ZLIB_ADLER_RUST, "go": _ZLIB_ADLER_GO},
        ("65520", "255"), ("1", "2")),
)


def step161_items() -> Tuple[IdiomaticItem, ...]:
    """The explicit coreutils/sudo-rs/zlib-rs-class Step-161 expansion."""
    by_id = {it.item_id: it for it in CORPUS}
    return tuple(by_id[item_id] for item_id in STEP161_FAMILY_IDS)


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


def _confirm_item(h: ReexecHarness, item: IdiomaticItem, target_src: str,
                  args: List[str], lang: str):
    if item.klass == "memcpy_overlap":
        return h.confirm_libc_contract_trap_vs_defined(item.c_src, target_src, args,
                                                       item.klass, lang)
    if item.klass == "uninit_padding":
        return h.confirm_uninit_padding_vs_defined(item.c_src, target_src, args,
                                                   item.klass, lang)
    return h.confirm_trap_vs_defined(item.c_src, target_src, args,
                                     item.klass, lang)


def _run_items(items: Tuple[IdiomaticItem, ...],
               langs: Tuple[str, ...] = ("rust", "go")) -> CorpusReport:
    status = toolchain_available()
    avail = tuple(l for l in langs if status.full_for(l)
                  or status.full_libc_contract_for(l)
                  or status.full_uninit_padding_for(l))
    if not avail:
        return CorpusReport(available=False, langs=())
    h = ReexecHarness(status)
    verdicts: List[ItemVerdict] = []
    for item in items:
        for lang in avail:
            tgt = item.targets.get(lang)
            if tgt is None:
                continue
            ub_args = [a for a in item.ub_inputs if a != ""]
            safe_args = [a for a in item.safe_inputs if a != ""]
            r_ub = _confirm_item(h, item, tgt, ub_args, lang)
            r_safe = _confirm_item(h, item, tgt, safe_args, lang)
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


def run_corpus(langs: Tuple[str, ...] = ("rust", "go")) -> CorpusReport:
    return _run_items(CORPUS, langs)


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
    if not any(status.full_for(l) or status.full_libc_contract_for(l)
               or status.full_uninit_padding_for(l) for l in langs):
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


def confirm_step161_expansion(
        langs: Tuple[str, ...] = ("rust", "go")) -> CorpusConfirmation:
    """Confirm the coreutils/sudo-rs/zlib-rs idiomatic-port expansion."""
    status = toolchain_available()
    if not any(status.full_for(l) for l in langs):
        return CorpusConfirmation(
            available=False, ok=True, n_items=0, n_divergent=0, n_equivalent=0,
            n_langs=0, hash_stable=True, content_hash="", report=None,
            detail="toolchain unavailable: Step-161 consistency-only pass")
    items = step161_items()
    r1 = _run_items(items, langs)
    r2 = _run_items(items, langs)
    stable = (r1.content_hash == r2.content_hash and bool(r1.content_hash))
    expected_cells = {
        (it.item_id, lang)
        for it in items
        for lang in r1.langs
        if lang in it.targets
    }
    seen_cells = {(v.item_id, v.lang) for v in r1.verdicts}
    seen_ids = {v.item_id for v in r1.verdicts}
    labels = {v.declared_label for v in r1.verdicts}
    ok = (
        r1.all_correct
        and stable
        and seen_ids == set(STEP161_FAMILY_IDS)
        and seen_cells == expected_cells
        and labels == {"divergent", "equivalent"}
    )
    detail = (f"step161_items={sorted(seen_ids)} cells={len(seen_cells)} "
              f"langs={list(r1.langs)} all_correct={r1.all_correct} "
              f"hash_stable={stable}")
    return CorpusConfirmation(
        available=True, ok=ok, n_items=r1.n_items, n_divergent=r1.n_divergent,
        n_equivalent=r1.n_equivalent, n_langs=len(r1.langs), hash_stable=stable,
        content_hash=r1.content_hash, report=r1, detail=detail)


IDIOMATIC_CORPUS_SPI = {
    "CORPUS": CORPUS,
    "STEP161_FAMILY_IDS": STEP161_FAMILY_IDS,
    "step161_items": step161_items,
    "run_corpus": run_corpus,
    "confirm_idiomatic_corpus": confirm_idiomatic_corpus,
    "confirm_step161_expansion": confirm_step161_expansion,
}


if __name__ == "__main__":  # pragma: no cover
    conf = confirm_idiomatic_corpus()
    print(f"ok={conf.ok} {conf.detail}")
    if conf.report is not None:
        print(conf.report.render())
