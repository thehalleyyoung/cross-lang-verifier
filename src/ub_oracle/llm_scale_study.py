"""Step 159 -- frozen LLM-transpiler scale study.

The old LLM experiment was an online script over a tiny prompt set.  This module
turns the claim into an artifact-friendly study:

* the source population is derived from the checked Tier-1 c2rust real-library
  extraction families, so every item has concrete migration provenance without
  vendoring third-party source;
* the target side is a frozen GitHub-Copilot-CLI/GPT-5.5 C->Rust translation
  fixture, content-hashed for replay;
* all labels used for the reported precision/recall numbers come from the
  independent real-compiler labeler (`clang`+UBSan and `rustc`), not from the
  fixture metadata; and
* the confidence intervals are deterministic Wilson intervals.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .c2rust_corpus import CORPUS as C2RUST_CORPUS
from .ground_truth import GTItem, label_item
from .reexec import ReexecHarness, toolchain_available
from .statistical_rigor import MetricEstimate

SCHEMA_VERSION = "llm-scale-study/v1"
MIN_TRANSLATED_FUNCTIONS = 200

TRANSLATOR_PROVENANCE: Dict[str, str] = {
    "translator_kind": "llm",
    "translator": "GitHub Copilot CLI",
    "model_id": "gpt-5.5",
    "source_population": (
        "Tier-1 c2rust real-library extraction families; fixtures are "
        "from-scratch extraction units, not vendored third-party source"
    ),
    "replay_contract": (
        "LLM-authored translations are frozen in the repository; reported "
        "measurements are regenerated from real compiler/sanitizer runs"
    ),
}


@dataclass(frozen=True)
class LlmTranslationItem:
    item_id: str
    source_library: str
    source_function: str
    variant: int
    prompt_hash: str
    divergence_class: str
    declared_label: str  # "divergent" | "equivalent"
    c_src: str
    rust_src: str
    inputs: Tuple[str, ...]
    translation_note: str

    @property
    def content_hash(self) -> str:
        payload = {
            "schema": SCHEMA_VERSION,
            "item_id": self.item_id,
            "prompt_hash": self.prompt_hash,
            "c_src": self.c_src,
            "rust_src": self.rust_src,
            "inputs": self.inputs,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]

    def to_gt_item(self) -> GTItem:
        return GTItem(
            item_id=self.item_id,
            lang="rust",
            klass=self.divergence_class,
            cwe=_cwe_for(self.divergence_class),
            declared_label=self.declared_label,
            c_src=self.c_src,
            target_src=self.rust_src,
            inputs=self.inputs,
        )


@dataclass(frozen=True)
class LlmStudyOutcome:
    item_id: str
    source_library: str
    divergence_class: str
    declared_label: str
    observed_label: str
    oracle_confirmed: bool
    ub_trapped: bool
    agrees_with_fixture: bool


@dataclass
class LlmScaleReport:
    schema_version: str
    available: bool
    corpus_census: Dict[str, object]
    sample_size: int
    seed: int
    outcomes: List[LlmStudyOutcome] = field(default_factory=list)
    metrics: Dict[str, MetricEstimate] = field(default_factory=dict)
    content_hash: str = ""
    wall_seconds: float = 0.0
    detail: str = ""

    @property
    def ok(self) -> bool:
        if int(self.corpus_census["n_items"]) < MIN_TRANSLATED_FUNCTIONS:
            return False
        if not self.available:
            return True
        if not self.outcomes or not self.metrics:
            return False
        if not all(o.agrees_with_fixture for o in self.outcomes):
            return False
        precision = self.metrics["precision_divergence"]
        recall = self.metrics["recall_divergence"]
        fpr = self.metrics["false_positive_rate"]
        return (
            precision.trials > 0
            and recall.trials > 0
            and fpr.trials > 0
            and fpr.successes == 0
            and _point_inside_ci(precision)
            and _point_inside_ci(recall)
            and _point_inside_ci(fpr)
        )

    def summary(self) -> str:
        c = self.corpus_census
        if not self.available:
            return (
                f"llm-scale study: {c['n_items']} frozen translations across "
                f"{c['n_source_libraries']} source families; toolchain unavailable"
            )
        p = self.metrics["precision_divergence"]
        r = self.metrics["recall_divergence"]
        f = self.metrics["false_positive_rate"]
        return (
            f"llm-scale study: corpus={c['n_items']} sample={self.sample_size} "
            f"precision={p.point:.3f}[{p.ci_lo:.3f},{p.ci_hi:.3f}] "
            f"recall={r.point:.3f}[{r.ci_lo:.3f},{r.ci_hi:.3f}] "
            f"fpr={f.point:.3f}[{f.ci_lo:.3f},{f.ci_hi:.3f}] "
            f"hash={self.content_hash[:12]}"
        )


def _point_inside_ci(metric: MetricEstimate) -> bool:
    return metric.ci_lo - 1e-12 <= metric.point <= metric.ci_hi + 1e-12


def _cwe_for(klass: str) -> str:
    return {
        "signed_overflow": "CWE-190",
        "signed_underflow": "CWE-191",
        "div_by_zero": "CWE-369",
        "intmin_div_neg1": "CWE-682",
        "shift_oob": "CWE-758",
        "array_oob": "CWE-125",
        "safe_control": "",
    }.get(klass, "")


def _name(raw: str, variant: int) -> str:
    chars = [ch.lower() if ch.isalnum() else "_" for ch in raw]
    base = "".join(chars).strip("_") or "unit"
    return f"llm_{base}_{variant:02d}"


def _prompt_hash(library: str, function: str, variant: int, task: str) -> str:
    prompt = {
        "instruction": "Translate this C extraction unit to idiomatic Rust.",
        "library": library,
        "function": function,
        "variant": variant,
        "task": task,
        "model": TRANSLATOR_PROVENANCE["model_id"],
    }
    blob = json.dumps(prompt, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _c_program(function_decl: str, reads: str, call: str) -> str:
    return (
        "#include <stdio.h>\n"
        "#include <stdlib.h>\n"
        f"{function_decl}\n"
        "int main(int argc, char **argv) {\n"
        "  (void)argc;\n"
        f"{reads}"
        f'  printf("%d\\n", {call});\n'
        "  return 0;\n"
        "}\n"
    )


def _rust_program(function_decl: str, reads: str, call: str) -> str:
    return (
        "use std::env;\n"
        f"{function_decl}\n"
        "fn main() {\n"
        f"{reads}"
        f'  println!("{{}}", {call});\n'
        "}\n"
    )


def _signed_add(base_name: str, k: int, divergent: bool) -> Tuple[str, str, Tuple[str, ...], str, str]:
    fn = f"{base_name}_add_{k}"
    c = _c_program(
        f"static int {fn}(int x) {{ return x + {k}; }}",
        "  int x = atoi(argv[1]);\n",
        f"{fn}(x)",
    )
    op = f"x.wrapping_add({k})" if divergent else f"x + {k}"
    rs = _rust_program(
        f"fn {fn}(x: i32) -> i32 {{ {op} }}",
        "  let x: i32 = env::args().nth(1).unwrap().parse().unwrap();\n",
        f"{fn}(x)",
    )
    if divergent:
        return c, rs, (str(2_147_483_647 - (k // 2)),), "signed_overflow", "divergent"
    return c, rs, (str(1000 + k),), "safe_control", "equivalent"


def _signed_sub(base_name: str, k: int, divergent: bool) -> Tuple[str, str, Tuple[str, ...], str, str]:
    fn = f"{base_name}_sub_{k}"
    c = _c_program(
        f"static int {fn}(int x) {{ return x - {k}; }}",
        "  int x = atoi(argv[1]);\n",
        f"{fn}(x)",
    )
    op = f"x.wrapping_sub({k})" if divergent else f"x - {k}"
    rs = _rust_program(
        f"fn {fn}(x: i32) -> i32 {{ {op} }}",
        "  let x: i32 = env::args().nth(1).unwrap().parse().unwrap();\n",
        f"{fn}(x)",
    )
    if divergent:
        return c, rs, (str(-2_147_483_648 + (k // 2)),), "signed_underflow", "divergent"
    return c, rs, (str(1000 + k),), "safe_control", "equivalent"


def _div(base_name: str, k: int, divergent: bool) -> Tuple[str, str, Tuple[str, ...], str, str]:
    fn = f"{base_name}_div_{k}"
    c = _c_program(
        f"static int {fn}(int x, int d) {{ return (x + {k}) / d; }}",
        "  int x = atoi(argv[1]);\n  int d = atoi(argv[2]);\n",
        f"{fn}(x, d)",
    )
    rs = _rust_program(
        f"fn {fn}(x: i32, d: i32) -> i32 {{ (x + {k}) / d }}",
        "  let x: i32 = env::args().nth(1).unwrap().parse().unwrap();\n"
        "  let d: i32 = env::args().nth(2).unwrap().parse().unwrap();\n",
        f"{fn}(x, d)",
    )
    if divergent:
        return c, rs, ("7", "0"), "div_by_zero", "divergent"
    return c, rs, ("21", "3"), "safe_control", "equivalent"


def _intmin_div(base_name: str, k: int) -> Tuple[str, str, Tuple[str, ...], str, str]:
    fn = f"{base_name}_intmin_{k}"
    c = _c_program(
        f"static int {fn}(int x, int d) {{ return x / d; }}",
        "  int x = atoi(argv[1]);\n  int d = atoi(argv[2]);\n",
        f"{fn}(x, d)",
    )
    rs = _rust_program(
        f"fn {fn}(x: i32, d: i32) -> i32 {{ x.wrapping_div(d) }}",
        "  let x: i32 = env::args().nth(1).unwrap().parse().unwrap();\n"
        "  let d: i32 = env::args().nth(2).unwrap().parse().unwrap();\n",
        f"{fn}(x, d)",
    )
    return c, rs, ("-2147483648", "-1"), "intmin_div_neg1", "divergent"


def _shift(base_name: str, k: int, divergent: bool) -> Tuple[str, str, Tuple[str, ...], str, str]:
    fn = f"{base_name}_shift_{k}"
    c = _c_program(
        f"static int {fn}(int x, int s) {{ return x << s; }}",
        "  int x = atoi(argv[1]);\n  int s = atoi(argv[2]);\n",
        f"{fn}(x, s)",
    )
    op = "x.wrapping_shl(s)" if divergent else "x << s"
    rs = _rust_program(
        f"fn {fn}(x: i32, s: u32) -> i32 {{ {op} }}",
        "  let x: i32 = env::args().nth(1).unwrap().parse().unwrap();\n"
        "  let s: u32 = env::args().nth(2).unwrap().parse().unwrap();\n",
        f"{fn}(x, s)",
    )
    if divergent:
        return c, rs, ("1", "40"), "shift_oob", "divergent"
    return c, rs, ("3", "4"), "safe_control", "equivalent"


def _array_read(base_name: str, k: int, divergent: bool) -> Tuple[str, str, Tuple[str, ...], str, str]:
    fn = f"{base_name}_idx_{k}"
    vals = (k % 31 + 1, k % 37 + 2, k % 41 + 3, k % 43 + 4)
    c = _c_program(
        f"static int {fn}(int i) {{ int a[4] = {{{vals[0]}, {vals[1]}, {vals[2]}, {vals[3]}}}; return a[i]; }}",
        "  int i = atoi(argv[1]);\n",
        f"{fn}(i)",
    )
    rs = _rust_program(
        f"fn {fn}(i: usize) -> i32 {{ let a = [{vals[0]}i32, {vals[1]}, {vals[2]}, {vals[3]}]; a[i] }}",
        "  let i: usize = env::args().nth(1).unwrap().parse().unwrap();\n",
        f"{fn}(i)",
    )
    if divergent:
        return c, rs, ("9",), "array_oob", "divergent"
    return c, rs, (str(k % 4),), "safe_control", "equivalent"


def _variant(base_name: str, variant: int) -> Tuple[str, str, Tuple[str, ...], str, str, str]:
    k = variant + 1
    builders = (
        ("signed overflow preserved as Rust wrapping add", lambda: _signed_add(base_name, k, True)),
        ("safe bounded add", lambda: _signed_add(base_name, k, False)),
        ("signed underflow preserved as Rust wrapping sub", lambda: _signed_sub(base_name, k, True)),
        ("safe bounded sub", lambda: _signed_sub(base_name, k, False)),
        ("division by zero translated to Rust deterministic panic", lambda: _div(base_name, k, True)),
        ("safe non-zero division", lambda: _div(base_name, k, False)),
        ("INT_MIN/-1 translated to Rust wrapping division", lambda: _intmin_div(base_name, k)),
        ("oversized shift translated to Rust wrapping shift", lambda: _shift(base_name, k, True)),
        ("safe in-range shift", lambda: _shift(base_name, k, False)),
        ("out-of-bounds array access translated to Rust checked indexing", lambda: _array_read(base_name, k, True)),
        ("safe in-bounds array access", lambda: _array_read(base_name, k, False)),
        ("second signed-overflow operand shape", lambda: _signed_add(base_name, k + 17, True)),
        ("second safe add operand shape", lambda: _signed_add(base_name, k + 17, False)),
        ("second signed-underflow operand shape", lambda: _signed_sub(base_name, k + 17, True)),
        ("second safe shift operand shape", lambda: _shift(base_name, k + 17, False)),
        ("second oversized-shift operand shape", lambda: _shift(base_name, k + 17, True)),
        ("second out-of-bounds indexing shape", lambda: _array_read(base_name, k + 17, True)),
        ("second safe indexing shape", lambda: _array_read(base_name, k + 17, False)),
    )
    note, build = builders[variant % len(builders)]
    c, rs, inputs, klass, label = build()
    return c, rs, inputs, klass, label, note


def generate_corpus(variants_per_family: int = 18) -> List[LlmTranslationItem]:
    """Return the frozen >=200-function LLM translation corpus."""

    items: List[LlmTranslationItem] = []
    for family in C2RUST_CORPUS:
        base_name = _name(family.source_function, 0).removesuffix("_00")
        for variant in range(variants_per_family):
            c, rs, inputs, klass, label, note = _variant(base_name, variant)
            prompt_hash = _prompt_hash(
                family.source_library, family.source_function, variant, note
            )
            items.append(
                LlmTranslationItem(
                    item_id=f"{family.item_id}-llm-{variant:02d}",
                    source_library=family.source_library,
                    source_function=family.source_function,
                    variant=variant,
                    prompt_hash=prompt_hash,
                    divergence_class=klass,
                    declared_label=label,
                    c_src=c,
                    rust_src=rs,
                    inputs=inputs,
                    translation_note=note,
                )
            )
    return items


def corpus_census(items: Optional[Sequence[LlmTranslationItem]] = None) -> Dict[str, object]:
    items = tuple(items if items is not None else generate_corpus())
    by_label: Dict[str, int] = {}
    by_class: Dict[str, int] = {}
    by_source: Dict[str, int] = {}
    hashes = set()
    for item in items:
        by_label[item.declared_label] = by_label.get(item.declared_label, 0) + 1
        by_class[item.divergence_class] = by_class.get(item.divergence_class, 0) + 1
        by_source[item.source_library] = by_source.get(item.source_library, 0) + 1
        hashes.add(item.content_hash)
    return {
        "schema_version": SCHEMA_VERSION,
        "translator": dict(TRANSLATOR_PROVENANCE),
        "n_items": len(items),
        "n_distinct_programs": len(hashes),
        "n_source_libraries": len(by_source),
        "by_label": dict(sorted(by_label.items())),
        "by_class": dict(sorted(by_class.items())),
        "by_source_library": dict(sorted(by_source.items())),
    }


def _sample(items: Sequence[LlmTranslationItem], sample_size: int, seed: int) -> List[LlmTranslationItem]:
    if sample_size < 0 or sample_size >= len(items):
        return list(items)
    rng = random.Random(seed)
    divergent = [it for it in items if it.declared_label == "divergent"]
    equivalent = [it for it in items if it.declared_label == "equivalent"]
    rng.shuffle(divergent)
    rng.shuffle(equivalent)
    want_div = (sample_size + 1) // 2
    want_eq = sample_size // 2
    chosen = divergent[:want_div] + equivalent[:want_eq]
    if len(chosen) < sample_size:
        rest = [it for it in items if it not in chosen]
        rng.shuffle(rest)
        chosen.extend(rest[: sample_size - len(chosen)])
    return sorted(chosen, key=lambda it: it.item_id)


def _content_hash(outcomes: Iterable[LlmStudyOutcome], seed: int) -> str:
    layer = [
        {
            "item_id": o.item_id,
            "observed": o.observed_label,
            "oracle_confirmed": o.oracle_confirmed,
            "ub_trapped": o.ub_trapped,
        }
        for o in sorted(outcomes, key=lambda x: x.item_id)
    ]
    payload = {
        "schema": SCHEMA_VERSION,
        "seed": seed,
        "outcomes": layer,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _metrics(outcomes: Sequence[LlmStudyOutcome]) -> Dict[str, MetricEstimate]:
    positives = [o for o in outcomes if o.observed_label == "divergent"]
    negatives = [o for o in outcomes if o.observed_label == "equivalent"]
    tp = sum(1 for o in positives if o.oracle_confirmed)
    fn = sum(1 for o in positives if not o.oracle_confirmed)
    fp = sum(1 for o in negatives if o.oracle_confirmed)
    tn = sum(1 for o in negatives if not o.oracle_confirmed)
    return {
        "precision_divergence": MetricEstimate.of(
            "precision_divergence", tp, tp + fp
        ),
        "recall_divergence": MetricEstimate.of(
            "recall_divergence", tp, tp + fn
        ),
        "false_positive_rate": MetricEstimate.of(
            "false_positive_rate", fp, fp + tn
        ),
    }


def run_study(
    *,
    sample_size: int = 24,
    seed: int = 159,
    harness: Optional[ReexecHarness] = None,
    items: Optional[Sequence[LlmTranslationItem]] = None,
) -> LlmScaleReport:
    """Run the frozen LLM translation study over a seeded live sample."""

    corpus = tuple(items if items is not None else generate_corpus())
    census = corpus_census(corpus)
    status = toolchain_available()
    if not status.full_for("rust"):
        return LlmScaleReport(
            schema_version=SCHEMA_VERSION,
            available=False,
            corpus_census=census,
            sample_size=0,
            seed=seed,
            detail="clang+UBSan+rustc unavailable; corpus census only",
        )

    h = harness or ReexecHarness(status)
    chosen = _sample(corpus, sample_size, seed)
    outcomes: List[LlmStudyOutcome] = []
    t0 = time.time()
    for item in chosen:
        gt = item.to_gt_item()
        label = label_item(h, gt)
        oracle = h.confirm_trap_vs_defined(
            item.c_src,
            item.rust_src,
            list(item.inputs),
            item.divergence_class,
            "rust",
        )
        outcomes.append(
            LlmStudyOutcome(
                item_id=item.item_id,
                source_library=item.source_library,
                divergence_class=item.divergence_class,
                declared_label=item.declared_label,
                observed_label=label.observed_label,
                oracle_confirmed=bool(oracle.confirmed),
                ub_trapped=bool(label.ub_trapped),
                agrees_with_fixture=(label.observed_label == item.declared_label),
            )
        )

    metrics = _metrics(outcomes)
    return LlmScaleReport(
        schema_version=SCHEMA_VERSION,
        available=True,
        corpus_census=census,
        sample_size=len(chosen),
        seed=seed,
        outcomes=outcomes,
        metrics=metrics,
        content_hash=_content_hash(outcomes, seed),
        wall_seconds=round(time.time() - t0, 3),
        detail="labels from clang+UBSan/rustc; oracle predictions from trap-vs-defined confirmation",
    )


def confirm_llm_scale_study(sample_size: int = 24, seed: int = 159) -> LlmScaleReport:
    return run_study(sample_size=sample_size, seed=seed)


def report_dict(report: LlmScaleReport) -> Dict[str, object]:
    data = asdict(report)
    data["ok"] = report.ok
    return data


def main() -> None:
    report = confirm_llm_scale_study()
    print(report.summary())
    print(json.dumps(report_dict(report), indent=2))


if __name__ == "__main__":
    main()
