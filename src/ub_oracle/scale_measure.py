"""Step 46 — scale measurement infrastructure.

A harness that drives the labeled ground-truth corpus (Step 45) through a real
decision procedure, recording **time, memory, verdict and abstention per item**,
and emitting a **canonical, content-hashed results JSON** suitable for a paper's
results table and for byte-level reproducibility checks.

Design principles that make the artifact trustworthy:

* The *decision procedure* under measurement is the sanitizer-anchored
  :func:`ground_truth.label_item`: it compiles and runs real binaries and returns
  a verdict (``divergent`` / ``equivalent``) or abstains (``inconclusive`` /
  ``uncompilable``). Nothing is simulated.

* **Verdicts are content-hashed; measurements are not.** Wall-time and memory are
  inherently non-deterministic, so the canonical `content_hash` is computed over
  the *verdict layer only* (item id, language, class, declared label, observed
  verdict, decided/abstained flags) with sorted keys and stable separators. Two
  independent runs on the same toolchain therefore produce the **same**
  `content_hash` even though their timings differ — exactly the property a
  reviewer re-running the artifact needs. The timing/memory numbers are emitted
  in a separate, explicitly non-hashed ``measurements`` section.

* Aggregates (decided/abstained fractions, broken down by language pair and by
  divergence class, plus total wall time and peak RSS) are computed from the
  per-item records, never hand-entered.

The JSON schema is versioned so downstream tooling can evolve without silently
reinterpreting old artifacts.
"""

from __future__ import annotations

import hashlib
import json
import os
import resource
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

from . import ground_truth as gt
from .reexec import ReexecHarness, ToolchainStatus, toolchain_available

SCHEMA_VERSION = "scale-measure/v1"

_DECIDED = ("divergent", "equivalent")


@dataclass
class ItemMeasurement:
    item_id: str
    lang: str
    klass: str
    declared_label: str
    observed_label: str
    decided: bool
    abstained: bool
    wall_ms: float
    peak_rss_kb: int

    def verdict_record(self) -> Dict[str, object]:
        """The deterministic, content-hashable projection (no timing/memory)."""
        return {
            "item_id": self.item_id,
            "lang": self.lang,
            "klass": self.klass,
            "declared_label": self.declared_label,
            "observed_label": self.observed_label,
            "decided": self.decided,
            "abstained": self.abstained,
        }


@dataclass
class MeasurementReport:
    available: bool
    schema: str
    langs: Tuple[str, ...]
    total_items: int
    measured: List[ItemMeasurement] = field(default_factory=list)
    total_wall_ms: float = 0.0
    peak_rss_kb: int = 0

    # ── aggregates ──────────────────────────────────────────────────────
    @property
    def decided(self) -> int:
        return sum(1 for m in self.measured if m.decided)

    @property
    def abstained(self) -> int:
        return sum(1 for m in self.measured if m.abstained)

    @property
    def faithful(self) -> bool:
        """Every *decided* item must match its declared (sanitizer-grounded)
        label — the measurement layer must not corrupt the verdict."""
        return all(m.observed_label == m.declared_label
                   for m in self.measured if m.decided)

    def by_pair(self) -> Dict[str, Dict[str, int]]:
        out: Dict[str, Dict[str, int]] = {}
        for m in self.measured:
            d = out.setdefault(m.lang, {"decided": 0, "abstained": 0})
            d["decided" if m.decided else "abstained"] += 1
        return out

    def by_class(self) -> Dict[str, Dict[str, int]]:
        out: Dict[str, Dict[str, int]] = {}
        for m in self.measured:
            d = out.setdefault(m.klass, {"decided": 0, "abstained": 0})
            d["decided" if m.decided else "abstained"] += 1
        return out


def _peak_rss_kb() -> int:
    """Cumulative high-water peak RSS of all child processes, in kibibytes.
    (``ru_maxrss`` is bytes on Darwin, kibibytes on Linux — normalize to KiB.)"""
    raw = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    if raw <= 0:
        return 0
    # Heuristic: Darwin reports bytes (values in the millions for tiny procs),
    # Linux reports KiB. Treat very large values as bytes.
    return raw // 1024 if raw > 1_000_000 else raw


def measure_item(h: ReexecHarness, item: gt.GTItem) -> ItemMeasurement:
    """Run one item through the real decision procedure, timing it and snapshotting
    the child-process peak RSS."""
    t0 = time.perf_counter()
    ev = gt.label_item(h, item)
    wall_ms = (time.perf_counter() - t0) * 1000.0
    decided = ev.observed_label in _DECIDED
    return ItemMeasurement(
        item_id=item.item_id,
        lang=item.lang,
        klass=item.klass,
        declared_label=item.declared_label,
        observed_label=ev.observed_label,
        decided=decided,
        abstained=not decided,
        wall_ms=round(wall_ms, 3),
        peak_rss_kb=_peak_rss_kb(),
    )


def run_scale(langs: Tuple[str, ...] = ("rust", "go"),
              status: Optional[ToolchainStatus] = None,
              sample_per_class: Optional[int] = None) -> MeasurementReport:
    """Drive the corpus (or a per-family sample) through the decision procedure,
    recording per-item time/memory/verdict/abstention."""
    st = status or toolchain_available()
    avail_langs = tuple(l for l in langs if st.full_for(l))
    all_items = gt.enumerate_corpus(langs)
    rep = MeasurementReport(available=bool(avail_langs), schema=SCHEMA_VERSION,
                            langs=avail_langs, total_items=len(all_items))
    if not avail_langs:
        return rep
    items = [it for it in all_items if it.lang in avail_langs]
    if sample_per_class is not None:
        idxs = gt._sample_indices(items, sample_per_class)
        items = [items[i] for i in idxs]
    h = ReexecHarness(st)
    t0 = time.perf_counter()
    for it in items:
        rep.measured.append(measure_item(h, it))
    rep.total_wall_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    rep.peak_rss_kb = _peak_rss_kb()
    return rep


# --------------------------------------------------------------------------- #
# Canonical, content-hashed results JSON.
# --------------------------------------------------------------------------- #
def _canonical_bytes(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("utf-8")


def verdict_layer(report: MeasurementReport) -> List[Dict[str, object]]:
    """The deterministic verdict records, sorted by item id."""
    return sorted((m.verdict_record() for m in report.measured),
                  key=lambda r: r["item_id"])


def content_hash(report: MeasurementReport) -> str:
    """A stable sha256 over the verdict layer (independent of timing/memory)."""
    return hashlib.sha256(_canonical_bytes(verdict_layer(report))).hexdigest()


def results_document(report: MeasurementReport) -> Dict[str, object]:
    """The full results document: a content-hashed verdict layer plus an
    explicitly non-hashed measurements section and computed aggregates."""
    verdicts = verdict_layer(report)
    chash = hashlib.sha256(_canonical_bytes(verdicts)).hexdigest()
    return {
        "schema": report.schema,
        "content_hash": chash,
        "n_items": len(report.measured),
        "n_decided": report.decided,
        "n_abstained": report.abstained,
        "faithful": report.faithful,
        "by_pair": report.by_pair(),
        "by_class": report.by_class(),
        "verdicts": verdicts,
        # NON-HASHED: wall-clock & memory vary run-to-run, by design.
        "measurements": {
            "total_wall_ms": report.total_wall_ms,
            "peak_rss_kb": report.peak_rss_kb,
            "per_item": sorted(
                ({"item_id": m.item_id, "wall_ms": m.wall_ms,
                  "peak_rss_kb": m.peak_rss_kb} for m in report.measured),
                key=lambda r: r["item_id"]),
        },
    }


def emit_results_json(path: str, report: MeasurementReport) -> str:
    """Write the canonical results document and return its content hash."""
    doc = results_document(report)
    with open(path, "w") as f:
        json.dump(doc, f, sort_keys=True, indent=2)
        f.write("\n")
    return str(doc["content_hash"])


# --------------------------------------------------------------------------- #
# Confirmation: the verdict layer is faithful and its content hash is stable.
# --------------------------------------------------------------------------- #
@dataclass
class ScaleConfirmation:
    available: bool
    ok: bool
    n_items: int
    n_decided: int
    n_abstained: int
    content_hash: str
    hash_stable: bool


def confirm_scale(langs: Tuple[str, ...] = ("rust", "go"),
                  sample_per_class: Optional[int] = 1) -> ScaleConfirmation:
    """Run the measurement harness twice over the same sample and confirm:
    (a) every decided item matches its sanitizer-grounded label (faithful),
    (b) at least one item was decided, and
    (c) the content hash of the verdict layer is identical across the two runs
        (the verdict artifact is byte-reproducible even though timings vary)."""
    st = toolchain_available()
    if not any(st.full_for(l) for l in langs):
        return ScaleConfirmation(False, False, 0, 0, 0, "", False)
    r1 = run_scale(langs, status=st, sample_per_class=sample_per_class)
    r2 = run_scale(langs, status=st, sample_per_class=sample_per_class)
    h1, h2 = content_hash(r1), content_hash(r2)
    stable = h1 == h2
    ok = bool(r1.measured) and r1.decided > 0 and r1.faithful and stable
    return ScaleConfirmation(
        available=True, ok=ok, n_items=len(r1.measured),
        n_decided=r1.decided, n_abstained=r1.abstained,
        content_hash=h1, hash_stable=stable)


SCALE_MEASURE_SPI = {
    "run_scale": run_scale,
    "results_document": results_document,
    "emit_results_json": emit_results_json,
    "content_hash": content_hash,
    "confirm_scale": confirm_scale,
}


if __name__ == "__main__":  # pragma: no cover
    rep = run_scale(sample_per_class=1)
    doc = results_document(rep)
    print(f"schema={doc['schema']}")
    print(f"items={doc['n_items']} decided={doc['n_decided']} "
          f"abstained={doc['n_abstained']} faithful={doc['faithful']}")
    print(f"content_hash={doc['content_hash']}")
    print(f"by_pair={doc['by_pair']}")
    print(f"total_wall_ms={doc['measurements']['total_wall_ms']} "
          f"peak_rss_kb={doc['measurements']['peak_rss_kb']}")
    conf = confirm_scale(sample_per_class=1)
    print(f"confirm: ok={conf.ok} hash_stable={conf.hash_stable} "
          f"decided={conf.n_decided}/{conf.n_items}")
