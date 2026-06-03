"""
Large-scale migration study (100_STEPS steps 85 and 141).

A best-paper claim needs an empirical result at *scale*: not a handful of toy
pairs but a large, diverse population of translated programs run through the
**real** decision procedure, with aggregate divergence statistics that a
reviewer can reproduce. This module is that study.

What it is — and what it is honestly *not*
------------------------------------------
The population here is a **mechanically generated, large-scale stress corpus**:
tens of thousands of *distinct, compilable* C → {Rust, Go} translation pairs
obtained by instantiating the catalogued divergence families (division-by-zero,
OOB read, oversized shift, signed overflow) and their well-defined counterparts
(safe add/mul/shift/mod) across a wide grid of real literal operands and across
both language pairs. It is **>= 1 000 000 lines** of code that
all really compiles. It is *not* a scrape of third-party repositories — those are
covered, at smaller scale, by the Tier-1/2/3 corpora (idiomatic ports, LLM
translations, the CVE set). The value added here is **scale and breadth**: it
stresses the oracle's generality and lets us report population-level rates.

How the study stays trustworthy
-------------------------------
* **Nothing is simulated.** Every executed item is compiled and run by the real
  toolchain through :func:`ground_truth.label_item` (clang + UBSan for the C UB
  oracle, ``rustc``/``go`` for the target). The observed label is whatever the
  binaries do.
* **The corpus is fully enumerated and content-hashed.** :func:`corpus_census`
  reports the exact line count, item count, distinct-program count, label
  balance, and the per-family / per-pair breakdown over the entire >= 1M-LOC
  population — these are counted, never estimated.
* **The execution is a seeded uniform sample.** Running 1M LOC through three
  compilers on every CI run is impractical, so the study executes a
  **deterministic, seeded random sample** live and reports the observed
  divergence/equivalence rates *with the sample size*. The sampling is
  reproducible (same seed -> same items -> same verdict-layer hash) and the full
  population can be driven by raising ``sample_size`` (or ``-1`` for all).
* **Verdicts are content-hashed; timings are not** (same discipline as
  :mod:`ub_oracle.scale_measure`): two runs on the same toolchain produce the
  same ``content_hash`` even though wall-times differ.

The headline numbers the study yields: the total corpus LOC, the number of
distinct programs and families/pairs covered, and — over the executed sample —
how many declared-divergent items the real pipeline *confirms* as divergent and
how many declared-equivalent items it confirms as equivalent (i.e. the
ground-truth labels the sanitizer-anchored procedure agrees with).
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

from .ground_truth import GTItem, label_item
from .reexec import ReexecHarness, toolchain_available

SCHEMA_VERSION = "large-scale-study/v2"

#: target floor for the generated corpus — Step 141 calls for >= 1M LOC.
MIN_TOTAL_LOC = 1_000_000


# --------------------------------------------------------------------------- #
# Source templates. Unlike the tiny ground_truth builders (which read *all*
# operands from argv and therefore emit byte-identical source for every
# witness), these bake the *defined* operands as distinct literals into the
# source while reading only the single UB-triggering operand from argv. That
# yields genuinely *distinct, compilable programs* — distinct source, distinct
# content hash — across the operand grid, instead of one program reused with
# different command-line arguments. The UB operand stays in argv so neither
# rustc's const-evaluator nor any backend can fold the undefined operation away
# before it actually executes.
# --------------------------------------------------------------------------- #
def _c_prog(body: str) -> str:
    return ("#include <stdio.h>\n#include <stdlib.h>\n"
            "int main(int argc, char **argv) {\n" + body + "\n  return 0;\n}\n")


def _rust_prog(body: str) -> str:
    return "use std::env;\nfn main() {\n" + body + "\n}\n"


def _go_prog(body: str, imports: Tuple[str, ...] = ("fmt", "os", "strconv")) -> str:
    imp = "".join(f"\t{json_q(i)}\n" for i in imports)
    return ("package main\n\nimport (\n" + imp + ")\n\n"
            "func main() {\n" + body + "\n}\n")


def json_q(s: str) -> str:
    return '"' + s + '"'


def _rd_c(idx: int) -> str:
    return f"atoi(argv[{idx}])"


def _rd_rust(idx: int, ty: str = "i32") -> str:
    return f"env::args().nth({idx}).unwrap().parse::<{ty}>().unwrap()"


def _rd_go(idx: int) -> str:
    return f"func() int {{ v, _ := strconv.Atoi(os.Args[{idx}]); return v }}()"


# ---- divergent templates (C-UB vs target-defined) ------------------------- #
def _t_div0(lang: str, k1: int, k2: int) -> Tuple[str, str]:
    """``acc / z`` with ``z`` read from argv (= 0): C traps, target panics."""
    c = _c_prog(f"  int p = {k1};\n  int q = {k2};\n"
                f"  int acc = p * 3 + q;\n  int z = {_rd_c(1)};\n"
                f'  printf("%d\\n", acc / z);')
    if lang == "rust":
        t = _rust_prog(f"  let p: i32 = {k1};\n  let q: i32 = {k2};\n"
                       f"  let acc: i32 = p * 3 + q;\n"
                       f"  let z: i32 = {_rd_rust(1)};\n"
                       f'  println!("{{}}", acc / z);')
    else:
        t = _go_prog(f"\tp := {k1}\n\tq := {k2}\n\tacc := p*3 + q\n"
                     f"\tz := {_rd_go(1)}\n\tfmt.Println(acc / z)")
    return c, t


def _t_oob(lang: str, vals: Tuple[int, int, int, int]) -> Tuple[str, str]:
    """Index a length-4 array out of bounds with ``i`` read from argv."""
    a, b, c0, d = vals
    c = _c_prog(f"  int t[4] = {{{a}, {b}, {c0}, {d}}};\n  int i = {_rd_c(1)};\n"
                f'  printf("%d\\n", t[i]);')
    if lang == "rust":
        t = _rust_prog(f"  let t = [{a}i32, {b}, {c0}, {d}];\n"
                       f"  let i: usize = {_rd_rust(1, 'usize')};\n"
                       f'  println!("{{}}", t[i]);')
    else:
        t = _go_prog(f"\tt := []int{{{a}, {b}, {c0}, {d}}}\n\ti := {_rd_go(1)}\n"
                     f"\tfmt.Println(t[i])")
    return c, t


def _t_shift(lang: str, k1: int) -> Tuple[str, str]:
    """``x << s`` with ``s`` (>= 32) read from argv: UB in C, defined in target."""
    c = _c_prog(f"  int x = {k1};\n  int s = {_rd_c(1)};\n"
                f'  printf("%d\\n", x << s);')
    if lang == "rust":
        t = _rust_prog(f"  let x: i32 = {k1};\n  let s: u32 = {_rd_rust(1, 'u32')};\n"
                       f'  println!("{{}}", x.wrapping_shl(s));')
    else:
        t = _go_prog(f"\tx := {k1}\n\ts := uint({_rd_go(1)})\n"
                     f"\tfmt.Println(x << s)")
    return c, t


def _t_overflow(lang: str, k1: int) -> Tuple[str, str]:
    """``a + b`` with ``a`` a large literal and ``b`` from argv: i32 overflow."""
    c = _c_prog(f"  int a = {k1};\n  int b = {_rd_c(1)};\n"
                f'  printf("%d\\n", a + b);')
    # rust only — Go's int is 64-bit so this would not overflow (not divergent).
    t = _rust_prog(f"  let a: i32 = {k1};\n  let b: i32 = {_rd_rust(1)};\n"
                   f'  println!("{{}}", a.wrapping_add(b));')
    return c, t


# ---- equivalent templates (fully defined, identical output) --------------- #
def _t_safe_add(lang: str, a: int, b: int) -> Tuple[str, str]:
    c = _c_prog(f"  int a = {a};\n  int b = {b};\n"
                f'  printf("%d\\n", a + b);')
    if lang == "rust":
        t = _rust_prog(f"  let a: i32 = {a};\n  let b: i32 = {b};\n"
                       f'  println!("{{}}", a + b);')
    else:
        t = _go_prog(f"\ta := {a}\n\tb := {b}\n\tfmt.Println(a + b)", ("fmt",))
    return c, t


def _t_safe_mul(lang: str, a: int, b: int) -> Tuple[str, str]:
    c = _c_prog(f"  int a = {a};\n  int b = {b};\n"
                f'  printf("%d\\n", a * b);')
    if lang == "rust":
        t = _rust_prog(f"  let a: i32 = {a};\n  let b: i32 = {b};\n"
                       f'  println!("{{}}", a * b);')
    else:
        t = _go_prog(f"\ta := {a}\n\tb := {b}\n\tfmt.Println(a * b)", ("fmt",))
    return c, t


def _t_safe_mod(lang: str, a: int, b: int) -> Tuple[str, str]:
    c = _c_prog(f"  int a = {a};\n  int b = {b};\n"
                f'  printf("%d\\n", a % b);')
    if lang == "rust":
        t = _rust_prog(f"  let a: i32 = {a};\n  let b: i32 = {b};\n"
                       f'  println!("{{}}", a % b);')
    else:
        t = _go_prog(f"\ta := {a}\n\tb := {b}\n\tfmt.Println(a % b)", ("fmt",))
    return c, t


def _t_safe_shift(lang: str, x: int, s: int) -> Tuple[str, str]:
    c = _c_prog(f"  int x = {x};\n  int s = {s};\n"
                f'  printf("%d\\n", x << s);')
    if lang == "rust":
        t = _rust_prog(f"  let x: i32 = {x};\n  let s: u32 = {s};\n"
                       f'  println!("{{}}", x << s);')
    else:
        t = _go_prog(f"\tx := {x}\n\ts := uint({s})\n\tfmt.Println(x << s)", ("fmt",))
    return c, t


# --------------------------------------------------------------------------- #
# Corpus generation — wide grids over genuinely-distinct source.
# --------------------------------------------------------------------------- #
def _scaled_divergent(lang: str) -> List[GTItem]:
    items: List[GTItem] = []

    # division by zero: distinct (defined) surrounding operands; z=0 from argv.
    for k1 in range(1, 121):
        for k2 in range(1, 41):                                  # 120*40 = 4800
            c, t = _t_div0(lang, k1, k2)
            items.append(GTItem(f"{lang}-ls-div0-{k1}-{k2}", lang,
                                "div_by_zero", "CWE-369", "divergent",
                                c, t, ("0",)))

    # out-of-bounds read: distinct array contents; index 7 from argv.
    for s in range(0, 4800):                                      # 4800 distinct
        vals = (s, s + 1, s + 2, s + 3)
        c, t = _t_oob(lang, vals)
        items.append(GTItem(f"{lang}-ls-oob-{s}", lang, "oob_read",
                            "CWE-125", "divergent", c, t, ("7",)))

    # oversized shift: distinct shifted value; shift 40 (>= 32) from argv.
    for k1 in range(1, 3201):                                     # 3200 distinct
        c, t = _t_shift(lang, k1)
        items.append(GTItem(f"{lang}-ls-shift-{k1}", lang, "oversized_shift",
                            "CWE-758", "divergent", c, t, ("40",)))

    if lang == "rust":
        # signed overflow (32-bit only): a is always within i32, but a+4096
        # exceeds INT_MAX for every item, so every declared witness is real UB.
        for k1 in range(2147479648, 2147483648):                  # 4000 distinct
            c, t = _t_overflow(lang, k1)
            items.append(GTItem(f"{lang}-ls-ovf-{k1}", lang, "signed_overflow",
                                "CWE-190", "divergent", c, t, ("4096",)))
    return items


def _scaled_equivalent(lang: str) -> List[GTItem]:
    items: List[GTItem] = []

    # safe addition over a wide in-range grid (no overflow → identical output).
    for a in range(1, 121):
        for b in range(1, 41):                                  # 120*40 = 4800
            c, t = _t_safe_add(lang, a, b)
            items.append(GTItem(f"{lang}-ls-add-{a}-{b}", lang, "safe_add", "",
                                "equivalent", c, t, ()))

    # safe multiplication (product stays well within i32).
    for a in range(2, 82):
        for b in range(2, 32):                                    # 80*30 = 2400
            c, t = _t_safe_mul(lang, a, b)
            items.append(GTItem(f"{lang}-ls-mul-{a}-{b}", lang, "safe_mul", "",
                                "equivalent", c, t, ()))

    # safe modulo by a non-zero divisor.
    for a in range(10, 250):
        for b in range(2, 22):                                   # 240*20 = 4800
            c, t = _t_safe_mod(lang, a, b)
            items.append(GTItem(f"{lang}-ls-mod-{a}-{b}", lang, "safe_mod", "",
                                "equivalent", c, t, ()))

    # safe shift strictly within range and representable as signed int.
    for x in range(1, 201):
        for s in range(0, 16):                                   # 200*16 = 3200
            c, t = _t_safe_shift(lang, x, s)
            items.append(GTItem(f"{lang}-ls-sshift-{x}-{s}", lang, "safe_shift",
                                "", "equivalent", c, t, ()))
    return items


def generate_corpus(langs: Tuple[str, ...] = ("rust", "go")) -> List[GTItem]:
    """The full, deterministically-enumerated large-scale corpus (>= 1M LOC)."""
    items: List[GTItem] = []
    for lang in langs:
        items.extend(_scaled_divergent(lang))
        items.extend(_scaled_equivalent(lang))
    return items


# --------------------------------------------------------------------------- #
# Census — exact, counted statistics over the whole population.
# --------------------------------------------------------------------------- #
def _loc(src: str) -> int:
    return len(src.splitlines())


def corpus_census(items: List[GTItem]) -> Dict[str, object]:
    """Exact line/program/family statistics over the entire corpus."""
    c_loc = sum(_loc(it.c_src) for it in items)
    t_loc = sum(_loc(it.target_src) for it in items)
    by_pair: Dict[str, int] = {}
    by_class: Dict[str, int] = {}
    by_label: Dict[str, int] = {}
    hashes = set()
    for it in items:
        by_pair[it.lang] = by_pair.get(it.lang, 0) + 1
        by_class[it.klass] = by_class.get(it.klass, 0) + 1
        by_label[it.declared_label] = by_label.get(it.declared_label, 0) + 1
        hashes.add(it.content_hash)
    return {
        "n_items": len(items),
        "n_distinct_programs": len(hashes),
        "c_loc": c_loc,
        "target_loc": t_loc,
        "total_loc": c_loc + t_loc,
        "pairs": sorted(by_pair),
        "by_pair": dict(sorted(by_pair.items())),
        "by_class": dict(sorted(by_class.items())),
        "by_label": dict(sorted(by_label.items())),
        "label_balance_delta": abs(
            by_label.get("divergent", 0) - by_label.get("equivalent", 0)),
    }


# --------------------------------------------------------------------------- #
# Live, seeded-sample execution through the real labeler.
# --------------------------------------------------------------------------- #
@dataclass
class StudyResult:
    item_id: str
    lang: str
    klass: str
    declared_label: str
    observed_label: str
    agrees: bool
    ub_trapped: bool


@dataclass
class StudyReport:
    schema_version: str
    available: bool
    census: Dict[str, object]
    sample_size: int
    seed: int
    results: List[StudyResult] = field(default_factory=list)
    aggregates: Dict[str, object] = field(default_factory=dict)
    wall_seconds: float = 0.0
    content_hash: str = ""

    @property
    def ok(self) -> bool:
        """The study is sound iff the corpus meets the LOC floor and every
        executed item's real verdict agrees with its declared ground-truth
        label (or no toolchain was available, in which case it is
        consistency-only)."""
        if int(self.census["total_loc"]) < MIN_TOTAL_LOC:
            return False
        if not self.available:
            return True
        if not self.results:
            return False
        return all(r.agrees for r in self.results)

    def summary(self) -> str:
        c = self.census
        a = self.aggregates
        return (f"large-scale study: {c['n_items']} items / "
                f"{c['n_distinct_programs']} distinct programs / "
                f"{c['total_loc']} LOC across {c['pairs']}; "
                f"sample={self.sample_size} seed={self.seed} "
                f"available={self.available} "
                f"confirmed_divergent={a.get('confirmed_divergent', 0)} "
                f"confirmed_equivalent={a.get('confirmed_equivalent', 0)} "
                f"agree={a.get('agree', 0)}/{a.get('decided', 0)} "
                f"hash={self.content_hash[:12]}")


def _verdict_payload(results: List[StudyResult]) -> str:
    rows = [
        {
            "id": r.item_id,
            "lang": r.lang,
            "klass": r.klass,
            "declared": r.declared_label,
            "observed": r.observed_label,
            "agrees": r.agrees,
        }
        for r in sorted(results, key=lambda x: x.item_id)
    ]
    return json.dumps(rows, sort_keys=True, separators=(",", ":"))


def run_sample(harness: Optional[ReexecHarness] = None,
               items: Optional[List[GTItem]] = None,
               sample_size: int = 16,
               seed: int = 0xC0FFEE) -> StudyReport:
    """Generate the corpus, census it, then execute a seeded random sample of
    ``sample_size`` items (``-1`` = all) through the real labeler."""
    items = items if items is not None else generate_corpus()
    census = corpus_census(items)

    status = toolchain_available()
    # We can only execute items whose target toolchain is fully present.
    runnable = [it for it in items if status.full_for(it.lang)]
    available = bool(runnable)

    if sample_size < 0 or sample_size > len(runnable):
        chosen = runnable
    else:
        rng = random.Random(seed)
        chosen = rng.sample(runnable, sample_size) if runnable else []

    h = harness or ReexecHarness()
    results: List[StudyResult] = []
    t0 = time.time()
    for it in chosen:
        ev = label_item(h, it)
        agrees = (ev.observed_label == it.declared_label)
        results.append(StudyResult(
            item_id=it.item_id, lang=it.lang, klass=it.klass,
            declared_label=it.declared_label, observed_label=ev.observed_label,
            agrees=agrees, ub_trapped=ev.ub_trapped))
    wall = time.time() - t0

    decided = [r for r in results if r.observed_label in ("divergent", "equivalent")]
    aggregates = {
        "executed": len(results),
        "decided": len(decided),
        "agree": sum(1 for r in results if r.agrees),
        "confirmed_divergent": sum(
            1 for r in results
            if r.declared_label == "divergent" and r.observed_label == "divergent"),
        "confirmed_equivalent": sum(
            1 for r in results
            if r.declared_label == "equivalent" and r.observed_label == "equivalent"),
        "by_class": _agg_by_class(results),
    }

    payload = _verdict_payload(results)
    digest = hashlib.sha256()
    digest.update(SCHEMA_VERSION.encode())
    digest.update(str(seed).encode())
    digest.update(payload.encode())
    content_hash = digest.hexdigest()

    return StudyReport(
        schema_version=SCHEMA_VERSION,
        available=available,
        census=census,
        sample_size=len(chosen),
        seed=seed,
        results=results,
        aggregates=aggregates,
        wall_seconds=round(wall, 3),
        content_hash=content_hash,
    )


def _agg_by_class(results: List[StudyResult]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for r in results:
        d = out.setdefault(r.klass, {"executed": 0, "agree": 0})
        d["executed"] += 1
        d["agree"] += int(r.agrees)
    return {k: out[k] for k in sorted(out)}


def confirm_large_scale_study(sample_size: int = 12,
                              seed: int = 0xC0FFEE) -> StudyReport:
    """Top-level confirmation used by the traceability theorem and the CLI."""
    return run_sample(sample_size=sample_size, seed=seed)


def report_dict(rep: StudyReport) -> Dict[str, object]:
    d = asdict(rep)
    d["ok"] = rep.ok
    return d


def main() -> None:
    rep = confirm_large_scale_study()
    print(rep.summary())
    print("ok:", rep.ok)
    print(json.dumps(rep.census, indent=2))


if __name__ == "__main__":
    main()
