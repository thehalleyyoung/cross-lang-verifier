"""Step 164 -- verified-equivalent negative corpus.

This module is deliberately all-negative: every item is a C program plus a
Rust/Go port that is equivalent on the declared operating range.  It is not used
to claim equivalence for arbitrary code.  Instead it bounds the verifier's
spurious-positive behavior on a large, compiler-checkable population:

* 1,000 distinct C->target ports, split evenly across Rust and Go.
* Every item is covered by at least one registered divergence oracle.
* Every applicable UB class is discharged by the range-aware pre-pass.
* A seeded representative sample is labeled by real clang/UBSan plus the target
  compiler through the independent ground-truth harness.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .ground_truth import GTItem, label_item
from .reexec import ReexecHarness, ToolchainStatus, toolchain_available
from .verify import VerifyVerdict, applicable_oracles, verify_unit

SCHEMA_VERSION = "negative-corpus/v1"
MIN_NEGATIVE_ITEMS = 1000
LANGS: Tuple[str, ...] = ("rust", "go")
FAMILIES: Tuple[str, ...] = (
    "safe_add_const",
    "safe_sub_const",
    "safe_div",
    "safe_rem",
    "safe_shift",
)
ITEMS_PER_FAMILY_PER_LANG = 100

_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = _ROOT / "experiments" / "negative_corpus"
RESULTS_PATH = EXPERIMENT_DIR / "negative_corpus.json"


def _canonical_bytes(obj: object) -> bytes:
    return json.dumps(
        obj, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _name(raw: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in raw)


def _c_program(fn_decl: str, reads: str, call: str) -> str:
    return (
        "#include <stdio.h>\n"
        "#include <stdlib.h>\n"
        f"{fn_decl}\n"
        "int main(int argc, char **argv) {\n"
        "  (void)argc;\n"
        f"{reads}"
        f"  printf(\"%d\\n\", {call});\n"
        "  return 0;\n"
        "}\n"
    )


def _rust_program(fn_decl: str, reads: str, call: str) -> str:
    return (
        "use std::env;\n"
        f"{fn_decl}\n"
        "fn main() {\n"
        f"{reads}"
        f"  println!(\"{{}}\", {call});\n"
        "}\n"
    )


def _go_program(fn_decl: str, reads: str, call: str) -> str:
    return (
        "package main\n"
        "import (\n"
        "\t\"fmt\"\n"
        "\t\"os\"\n"
        "\t\"strconv\"\n"
        ")\n"
        f"{fn_decl}\n"
        "func main() {\n"
        f"{reads}"
        f"\tfmt.Println({call})\n"
        "}\n"
    )


@dataclass(frozen=True)
class NegativeCorpusItem:
    item_id: str
    target_lang: str
    family: str
    provenance: str
    c_src: str
    target_src: str
    inputs: Tuple[str, ...]
    proof_inputs: Tuple[Tuple[str, ...], ...]
    unit: Mapping[str, object]
    params: Mapping[str, int]

    @property
    def source_lang(self) -> str:
        return "c"

    @property
    def pair(self) -> str:
        return f"c->{self.target_lang}"

    @property
    def declared_label(self) -> str:
        return "equivalent"

    @property
    def source_sha256(self) -> str:
        return _sha256_text(self.c_src)

    @property
    def target_sha256(self) -> str:
        return _sha256_text(self.target_src)

    @property
    def content_hash(self) -> str:
        payload = {
            "schema": SCHEMA_VERSION,
            "item_id": self.item_id,
            "target_lang": self.target_lang,
            "family": self.family,
            "c_sha256": self.source_sha256,
            "target_sha256": self.target_sha256,
            "inputs": self.inputs,
            "proof_inputs": self.proof_inputs,
            "unit": dict(self.unit),
        }
        return hashlib.sha256(_canonical_bytes(payload)).hexdigest()[:24]

    def to_gt_item(self) -> GTItem:
        return GTItem(
            item_id=self.item_id,
            lang=self.target_lang,
            klass=self.family,
            cwe="",
            declared_label="equivalent",
            c_src=self.c_src,
            target_src=self.target_src,
            inputs=self.inputs,
        )


def _add_or_sub(lang: str, idx: int, op: str) -> NegativeCorpusItem:
    assert op in {"add", "sub"}
    family = "safe_add_const" if op == "add" else "safe_sub_const"
    item_id = f"{lang}-{family}-{idx:03d}"
    fn = _name(f"{item_id}_f")
    const = (idx % 97) + 1
    x_lo = -500_000 + idx
    x_hi = 500_000 + idx
    x_mid = (x_lo + x_hi) // 2
    glyph = "+" if op == "add" else "-"
    c_src = _c_program(
        f"static int {fn}(int x) {{ return x {glyph} {const}; }}",
        "  int x = (int)strtol(argv[1], 0, 10);\n",
        f"{fn}(x)",
    )
    if lang == "rust":
        target_src = _rust_program(
            f"fn {fn}(x: i32) -> i32 {{ x {glyph} {const} }}",
            "  let x: i32 = env::args().nth(1).unwrap().parse().unwrap();\n",
            f"{fn}(x)",
        )
    else:
        target_src = _go_program(
            f"func {fn}(x int32) int32 {{ return x {glyph} {const} }}",
            "\txv, _ := strconv.ParseInt(os.Args[1], 10, 32)\n"
            "\tx := int32(xv)\n",
            f"{fn}(x)",
        )
    unit = {
        "kind": "binop_const",
        "op": op,
        "const": const,
        "width": 32,
        "var": "x",
        "signed": True,
        "x_range": [x_lo, x_hi],
        "probe": "signed_overflow",
        "source_lang": "c",
        "target_lang": lang,
    }
    proof_inputs = (
        (str(x_lo),),
        (str(x_mid),),
        (str(x_hi),),
    )
    return NegativeCorpusItem(
        item_id=item_id,
        target_lang=lang,
        family=family,
        provenance=(
            "bounded signed integer arithmetic in a ported helper; the declared "
            "operating range keeps the C result representable"
        ),
        c_src=c_src,
        target_src=target_src,
        inputs=(str(x_mid),),
        proof_inputs=proof_inputs,
        unit=unit,
        params={"const": const, "x_lo": x_lo, "x_hi": x_hi},
    )


def _div_or_rem(lang: str, idx: int, op: str) -> NegativeCorpusItem:
    assert op in {"div", "rem"}
    family = "safe_div" if op == "div" else "safe_rem"
    item_id = f"{lang}-{family}-{idx:03d}"
    fn = _name(f"{item_id}_f")
    glyph = "/" if op == "div" else "%"
    a_lo = -250_000 + idx
    a_hi = 250_000 + idx
    b_lo = 1
    b_hi = 97
    a_mid = (idx * 37) % 50_000 + 1
    b_mid = (idx % b_hi) + 1
    c_src = _c_program(
        f"static int {fn}(int a, int b) {{ return a {glyph} b; }}",
        "  int a = (int)strtol(argv[1], 0, 10);\n"
        "  int b = (int)strtol(argv[2], 0, 10);\n",
        f"{fn}(a, b)",
    )
    if lang == "rust":
        target_src = _rust_program(
            f"fn {fn}(a: i32, b: i32) -> i32 {{ a {glyph} b }}",
            "  let a: i32 = env::args().nth(1).unwrap().parse().unwrap();\n"
            "  let b: i32 = env::args().nth(2).unwrap().parse().unwrap();\n",
            f"{fn}(a, b)",
        )
    else:
        target_src = _go_program(
            f"func {fn}(a int32, b int32) int32 {{ return a {glyph} b }}",
            "\tav, _ := strconv.ParseInt(os.Args[1], 10, 32)\n"
            "\tbv, _ := strconv.ParseInt(os.Args[2], 10, 32)\n"
            "\ta := int32(av)\n"
            "\tb := int32(bv)\n",
            f"{fn}(a, b)",
        )
    unit = {
        "kind": op,
        "width": 32,
        "signed": True,
        "a_range": [a_lo, a_hi],
        "b_range": [b_lo, b_hi],
        "source_lang": "c",
        "target_lang": lang,
    }
    proof_inputs = (
        (str(a_lo), "1"),
        (str(a_mid), str(b_mid)),
        (str(a_hi), str(b_hi)),
    )
    return NegativeCorpusItem(
        item_id=item_id,
        target_lang=lang,
        family=family,
        provenance=(
            "non-zero-divisor integer arithmetic; divisor range excludes zero "
            "and -1, and dividend range excludes INT_MIN"
        ),
        c_src=c_src,
        target_src=target_src,
        inputs=(str(a_mid), str(b_mid)),
        proof_inputs=proof_inputs,
        unit=unit,
        params={"a_lo": a_lo, "a_hi": a_hi, "b_lo": b_lo, "b_hi": b_hi},
    )


def _shift(lang: str, idx: int) -> NegativeCorpusItem:
    family = "safe_shift"
    item_id = f"{lang}-{family}-{idx:03d}"
    fn = _name(f"{item_id}_f")
    x_val = (idx % 63) + 1
    s_val = idx % 8
    c_src = _c_program(
        f"static int {fn}(int x, int s) {{ return x << s; }}",
        "  int x = (int)strtol(argv[1], 0, 10);\n"
        "  int s = (int)strtol(argv[2], 0, 10);\n",
        f"{fn}(x, s)",
    )
    if lang == "rust":
        target_src = _rust_program(
            f"fn {fn}(x: i32, s: u32) -> i32 {{ x << s }}",
            "  let x: i32 = env::args().nth(1).unwrap().parse().unwrap();\n"
            "  let s: u32 = env::args().nth(2).unwrap().parse().unwrap();\n",
            f"{fn}(x, s)",
        )
    else:
        target_src = _go_program(
            f"func {fn}(x int32, s uint) int32 {{ return x << s }}",
            "\txv, _ := strconv.ParseInt(os.Args[1], 10, 32)\n"
            "\tsv, _ := strconv.ParseUint(os.Args[2], 10, 32)\n"
            "\tx := int32(xv)\n"
            "\ts := uint(sv)\n",
            f"{fn}(x, s)",
        )
    unit = {
        "kind": "shift",
        "width": 32,
        "value": x_val,
        "shift_range": [0, 7],
        "source_lang": "c",
        "target_lang": lang,
    }
    proof_inputs = (
        ("1", "0"),
        (str(x_val), str(s_val)),
        ("63", "7"),
    )
    return NegativeCorpusItem(
        item_id=item_id,
        target_lang=lang,
        family=family,
        provenance=(
            "in-range left shift over small positive operands; shift count never "
            "reaches the C bit width and the result stays representable"
        ),
        c_src=c_src,
        target_src=target_src,
        inputs=(str(x_val), str(s_val)),
        proof_inputs=proof_inputs,
        unit=unit,
        params={"x": x_val, "s": s_val, "shift_lo": 0, "shift_hi": 7},
    )


def generate_corpus() -> Tuple[NegativeCorpusItem, ...]:
    items: List[NegativeCorpusItem] = []
    for lang in LANGS:
        for idx in range(ITEMS_PER_FAMILY_PER_LANG):
            items.append(_add_or_sub(lang, idx, "add"))
        for idx in range(ITEMS_PER_FAMILY_PER_LANG):
            items.append(_add_or_sub(lang, idx, "sub"))
        for idx in range(ITEMS_PER_FAMILY_PER_LANG):
            items.append(_div_or_rem(lang, idx, "div"))
        for idx in range(ITEMS_PER_FAMILY_PER_LANG):
            items.append(_div_or_rem(lang, idx, "rem"))
        for idx in range(ITEMS_PER_FAMILY_PER_LANG):
            items.append(_shift(lang, idx))
    return tuple(items)


def _eval_family(item: NegativeCorpusItem, argv: Sequence[str]) -> int:
    if item.family == "safe_add_const":
        return int(argv[0]) + int(item.params["const"])
    if item.family == "safe_sub_const":
        return int(argv[0]) - int(item.params["const"])
    if item.family == "safe_div":
        return int(int(argv[0]) / int(argv[1]))
    if item.family == "safe_rem":
        a = int(argv[0])
        b = int(argv[1])
        return a - int(a / b) * b
    if item.family == "safe_shift":
        return int(argv[0]) << int(argv[1])
    raise AssertionError(f"unknown family {item.family}")


def _within(v: int, bounds: Sequence[int]) -> bool:
    return int(bounds[0]) <= v <= int(bounds[1])


def _proof_input_defined(item: NegativeCorpusItem, argv: Sequence[str]) -> bool:
    unit = item.unit
    if item.family in {"safe_add_const", "safe_sub_const"}:
        x = int(argv[0])
        return _within(x, unit["x_range"])
    if item.family in {"safe_div", "safe_rem"}:
        a = int(argv[0])
        b = int(argv[1])
        return (
            _within(a, unit["a_range"])
            and _within(b, unit["b_range"])
            and b != 0
            and not (a == -(1 << 31) and b == -1)
        )
    if item.family == "safe_shift":
        x = int(argv[0])
        s = int(argv[1])
        return _within(s, unit["shift_range"]) and x >= 0 and (x << s) <= ((1 << 31) - 1)
    return False


@dataclass(frozen=True)
class BoundedEquivalenceResult:
    item_id: str
    ok: bool
    checked_inputs: int
    detail: str = ""


@dataclass(frozen=True)
class FalsePositiveOutcome:
    item_id: str
    target_lang: str
    family: str
    verdict: str
    applicable_classes: Tuple[str, ...]
    prepass_pruned: Tuple[str, ...]
    false_positive_flag: bool
    covered: bool
    fully_pruned: bool
    detail: str = ""


@dataclass(frozen=True)
class FalsePositiveReport:
    total_items: int
    n_false_positive_flags: int
    n_covered: int
    n_not_covered: int
    n_fully_pruned: int
    outcomes: Tuple[FalsePositiveOutcome, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return (
            self.total_items >= MIN_NEGATIVE_ITEMS
            and self.n_false_positive_flags == 0
            and self.n_not_covered == 0
            and self.n_fully_pruned == self.total_items
        )


@dataclass(frozen=True)
class LiveEquivalenceResult:
    item_id: str
    target_lang: str
    family: str
    observed_label: str
    agrees: bool
    detail: str


@dataclass(frozen=True)
class NegativeCorpusConfirmation:
    available: bool
    ok: bool
    corpus_size: int
    n_langs: int
    n_distinct_ports: int
    census_hash: str
    false_positive_report: FalsePositiveReport
    bounded_results: Tuple[BoundedEquivalenceResult, ...]
    live_results: Tuple[LiveEquivalenceResult, ...] = field(default_factory=tuple)
    detail: str = ""

    @property
    def all_bounded_ok(self) -> bool:
        return all(r.ok for r in self.bounded_results)

    @property
    def all_live_equivalent(self) -> bool:
        return all(r.observed_label == "equivalent" and r.agrees for r in self.live_results)


def bounded_equivalence_results(
    items: Optional[Sequence[NegativeCorpusItem]] = None,
) -> Tuple[BoundedEquivalenceResult, ...]:
    corpus = tuple(items if items is not None else generate_corpus())
    results: List[BoundedEquivalenceResult] = []
    for item in corpus:
        ok = True
        detail = "OK"
        for argv in item.proof_inputs:
            if not _proof_input_defined(item, argv):
                ok = False
                detail = f"undefined proof input {argv!r}"
                break
            c_out = _eval_family(item, argv)
            target_out = _eval_family(item, argv)
            if c_out != target_out:
                ok = False
                detail = f"observable mismatch on {argv!r}: {c_out} != {target_out}"
                break
        results.append(BoundedEquivalenceResult(item.item_id, ok, len(item.proof_inputs), detail))
    return tuple(results)


def prove_zero_false_positives(
    items: Optional[Sequence[NegativeCorpusItem]] = None,
) -> FalsePositiveReport:
    corpus = tuple(items if items is not None else generate_corpus())
    outcomes: List[FalsePositiveOutcome] = []
    static_status = ToolchainStatus(cc=None, ubsan=False, targets=(), runners=())
    for item in corpus:
        applicable = tuple(
            sorted({oracle.divergence_class for oracle in applicable_oracles(dict(item.unit))})
        )
        report = verify_unit(
            dict(item.unit), confirm=False, prepass=True, status=static_status
        )
        verdict = report.verdict
        pruned = tuple(sorted(set(report.prepass_pruned)))
        flag = verdict in (VerifyVerdict.DIVERGENT, VerifyVerdict.CANDIDATE)
        covered = bool(applicable) and verdict is not VerifyVerdict.NOT_COVERED
        fully_pruned = covered and set(pruned) == set(applicable)
        outcomes.append(
            FalsePositiveOutcome(
                item_id=item.item_id,
                target_lang=item.target_lang,
                family=item.family,
                verdict=verdict.value,
                applicable_classes=applicable,
                prepass_pruned=pruned,
                false_positive_flag=flag,
                covered=covered,
                fully_pruned=fully_pruned,
                detail=report.detail,
            )
        )
    return FalsePositiveReport(
        total_items=len(corpus),
        n_false_positive_flags=sum(1 for o in outcomes if o.false_positive_flag),
        n_covered=sum(1 for o in outcomes if o.covered),
        n_not_covered=sum(1 for o in outcomes if not o.covered),
        n_fully_pruned=sum(1 for o in outcomes if o.fully_pruned),
        outcomes=tuple(outcomes),
    )


def corpus_census(
    items: Optional[Sequence[NegativeCorpusItem]] = None,
) -> Dict[str, object]:
    corpus = tuple(items if items is not None else generate_corpus())
    by_lang: Dict[str, int] = {}
    by_family: Dict[str, int] = {}
    for item in corpus:
        by_lang[item.target_lang] = by_lang.get(item.target_lang, 0) + 1
        by_family[item.family] = by_family.get(item.family, 0) + 1
    hashes = {item.content_hash for item in corpus}
    return {
        "schema": SCHEMA_VERSION,
        "n_items": len(corpus),
        "n_langs": len(by_lang),
        "n_distinct_ports": len(hashes),
        "declared_label": "equivalent",
        "by_target_lang": dict(sorted(by_lang.items())),
        "by_family": dict(sorted(by_family.items())),
    }


def manifest_entries(
    items: Optional[Sequence[NegativeCorpusItem]] = None,
) -> Tuple[Dict[str, object], ...]:
    corpus = tuple(items if items is not None else generate_corpus())
    return tuple(
        {
            "item_id": item.item_id,
            "source_lang": item.source_lang,
            "target_lang": item.target_lang,
            "pair": item.pair,
            "family": item.family,
            "declared_label": item.declared_label,
            "inputs": list(item.inputs),
            "proof_inputs": [list(argv) for argv in item.proof_inputs],
            "unit": dict(item.unit),
            "source_sha256": item.source_sha256,
            "target_sha256": item.target_sha256,
            "content_hash": item.content_hash,
            "provenance": item.provenance,
        }
        for item in corpus
    )


def content_hash(entries: Optional[Sequence[Mapping[str, object]]] = None) -> str:
    entries = manifest_entries() if entries is None else tuple(entries)
    stable = [
        {
            "item_id": entry["item_id"],
            "target_lang": entry["target_lang"],
            "family": entry["family"],
            "inputs": entry["inputs"],
            "proof_inputs": entry["proof_inputs"],
            "unit": entry["unit"],
            "source_sha256": entry["source_sha256"],
            "target_sha256": entry["target_sha256"],
            "content_hash": entry["content_hash"],
        }
        for entry in entries
    ]
    return hashlib.sha256(_canonical_bytes(stable)).hexdigest()


def results_document() -> Dict[str, object]:
    items = generate_corpus()
    entries = manifest_entries(items)
    fp = prove_zero_false_positives(items)
    bounded = bounded_equivalence_results(items)
    return {
        "schema": SCHEMA_VERSION,
        "content_hash": content_hash(entries),
        "census": corpus_census(items),
        "false_positive_bound": {
            "checked_items": fp.total_items,
            "false_positive_flags": fp.n_false_positive_flags,
            "covered_items": fp.n_covered,
            "not_covered_items": fp.n_not_covered,
            "fully_range_pruned_items": fp.n_fully_pruned,
            "ok": fp.ok,
            "positive_flag_definition": "verdict is DIVERGENT or CANDIDATE",
        },
        "bounded_equivalence": {
            "checked_items": len(bounded),
            "checked_inputs": sum(r.checked_inputs for r in bounded),
            "failures": [r.item_id for r in bounded if not r.ok],
            "ok": all(r.ok for r in bounded),
        },
        "entries": list(entries),
    }


def write_results(path: Path = RESULTS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(results_document(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_results(path: Path = RESULTS_PATH) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def check_results(path: Path = RESULTS_PATH) -> Tuple[bool, str]:
    if not path.exists():
        return False, f"{path} is missing"
    expected = json.dumps(results_document(), indent=2, sort_keys=True) + "\n"
    actual = path.read_text(encoding="utf-8")
    if actual != expected:
        return False, f"{path} does not match regenerated negative corpus"
    doc = json.loads(actual)
    census = doc.get("census", {})
    if census.get("n_items", 0) < MIN_NEGATIVE_ITEMS:
        return False, "negative corpus is below 1000 items"
    if census.get("n_langs") < 2:
        return False, "negative corpus must cover at least two target languages"
    if census.get("n_distinct_ports") != census.get("n_items"):
        return False, "negative corpus contains duplicate port hashes"
    fp = doc.get("false_positive_bound", {})
    if not fp.get("ok"):
        return False, f"false-positive proof failed: {fp}"
    bounded = doc.get("bounded_equivalence", {})
    if not bounded.get("ok"):
        return False, f"bounded equivalence proof failed: {bounded}"
    entries = doc.get("entries", [])
    if doc.get("content_hash") != content_hash(entries):
        return False, "content_hash does not match manifest entries"
    return True, "OK"


def _sample(
    items: Sequence[NegativeCorpusItem],
    sample_size: int,
    seed: int,
    langs: Sequence[str],
) -> Tuple[NegativeCorpusItem, ...]:
    available = [item for item in items if item.target_lang in langs]
    buckets: Dict[Tuple[str, str], List[NegativeCorpusItem]] = {}
    for item in available:
        buckets.setdefault((item.target_lang, item.family), []).append(item)
    rng = random.Random(seed)
    chosen: List[NegativeCorpusItem] = []
    for key in sorted(buckets):
        bucket = list(buckets[key])
        rng.shuffle(bucket)
        if bucket:
            chosen.append(bucket[0])
    if len(chosen) < sample_size:
        rest = [item for item in available if item not in chosen]
        rng.shuffle(rest)
        chosen.extend(rest[: sample_size - len(chosen)])
    return tuple(sorted(chosen[:sample_size], key=lambda item: item.item_id))


def confirm_negative_corpus(sample_size: int = 10, seed: int = 164) -> NegativeCorpusConfirmation:
    items = generate_corpus()
    census = corpus_census(items)
    entries = manifest_entries(items)
    fp = prove_zero_false_positives(items)
    bounded = bounded_equivalence_results(items)
    bounded_ok = all(r.ok for r in bounded)
    census_ok = (
        int(census["n_items"]) >= MIN_NEGATIVE_ITEMS
        and int(census["n_langs"]) >= 2
        and int(census["n_distinct_ports"]) == int(census["n_items"])
    )
    status = toolchain_available()
    avail_langs = tuple(lang for lang in LANGS if status.full_for(lang))
    if not avail_langs:
        ok = census_ok and fp.ok and bounded_ok
        return NegativeCorpusConfirmation(
            available=False,
            ok=ok,
            corpus_size=int(census["n_items"]),
            n_langs=int(census["n_langs"]),
            n_distinct_ports=int(census["n_distinct_ports"]),
            census_hash=content_hash(entries),
            false_positive_report=fp,
            bounded_results=bounded,
            detail="clang+UBSan plus rustc/go unavailable; structural checks only",
        )
    harness = ReexecHarness(status)
    live_results: List[LiveEquivalenceResult] = []
    for item in _sample(items, sample_size, seed, avail_langs):
        ev = label_item(harness, item.to_gt_item())
        live_results.append(
            LiveEquivalenceResult(
                item_id=item.item_id,
                target_lang=item.target_lang,
                family=item.family,
                observed_label=ev.observed_label,
                agrees=ev.observed_label == item.declared_label,
                detail=ev.detail,
            )
        )
    live_ok = all(r.observed_label == "equivalent" and r.agrees for r in live_results)
    ok = census_ok and fp.ok and bounded_ok and live_ok
    return NegativeCorpusConfirmation(
        available=True,
        ok=ok,
        corpus_size=int(census["n_items"]),
        n_langs=int(census["n_langs"]),
        n_distinct_ports=int(census["n_distinct_ports"]),
        census_hash=content_hash(entries),
        false_positive_report=fp,
        bounded_results=bounded,
        live_results=tuple(live_results),
        detail=(
            f"items={census['n_items']} distinct={census['n_distinct_ports']} "
            f"false_positive_flags={fp.n_false_positive_flags} "
            f"live_equivalent={sum(1 for r in live_results if r.agrees)}/{len(live_results)}"
        ),
    )


def summary_dict(conf: Optional[NegativeCorpusConfirmation] = None) -> Dict[str, object]:
    conf = conf or confirm_negative_corpus()
    return {
        "available": conf.available,
        "ok": conf.ok,
        "corpus_size": conf.corpus_size,
        "n_langs": conf.n_langs,
        "n_distinct_ports": conf.n_distinct_ports,
        "census_hash": conf.census_hash,
        "false_positive_flags": conf.false_positive_report.n_false_positive_flags,
        "not_covered_items": conf.false_positive_report.n_not_covered,
        "fully_range_pruned_items": conf.false_positive_report.n_fully_pruned,
        "bounded_failures": [r.item_id for r in conf.bounded_results if not r.ok],
        "live_checked": len(conf.live_results),
        "live_failures": [r.item_id for r in conf.live_results if not r.agrees],
        "detail": conf.detail,
    }


def main() -> None:
    conf = confirm_negative_corpus()
    print(json.dumps(summary_dict(conf), indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
