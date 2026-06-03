"""Deterministic parallel ground-truth harness (100_STEPS step 143).

The expensive empirical paths in this repository ultimately reduce to
``ground_truth.label_item``: compile and run a real C/target pair, then emit a
small verdict record.  This module parallelizes that operation without letting
OS scheduling leak into the reproducibility artifact:

* workers may finish in any order, but the merged report is sorted back to the
  input sequence;
* the content hash is computed only over the deterministic verdict layer
  (item id, pair/class, declared/observed labels, decided/abstained), never over
  timings, process ids, stderr snippets, or completion order;
* the label function is explicitly injectable and must be picklable, so tests can
  prove schedule independence under ``spawn``-based process pools without relying
  on parent-process monkeypatches.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .ground_truth import GTItem, LabelEvidence, enumerate_corpus, label_item
from .reexec import ReexecHarness, ToolchainStatus, toolchain_available

SCHEMA_VERSION = "parallel-harness/v1"

LabelFn = Callable[[ReexecHarness, GTItem], LabelEvidence]
_DECIDED = {"divergent", "equivalent"}


def _canonical_bytes(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, default=str).encode("utf-8")


@dataclass(frozen=True)
class ParallelItemResult:
    sequence: int
    item_id: str
    lang: str
    klass: str
    declared_label: str
    observed_label: str
    decided: bool
    abstained: bool
    detail: str

    def verdict_record(self) -> Dict[str, object]:
        """The deterministic, hashable projection; matches scale_measure."""
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
class ParallelRunReport:
    workers: int
    total_items: int
    schema: str = SCHEMA_VERSION
    results: List[ParallelItemResult] = field(default_factory=list)

    @property
    def decided(self) -> int:
        return sum(1 for r in self.results if r.decided)

    @property
    def abstained(self) -> int:
        return sum(1 for r in self.results if r.abstained)

    @property
    def faithful(self) -> bool:
        return all(
            r.observed_label == r.declared_label
            for r in self.results
            if r.decided
        )

    def verdict_layer(self) -> List[Dict[str, object]]:
        return sorted((r.verdict_record() for r in self.results),
                      key=lambda r: (str(r["item_id"]), str(r["lang"])))

    def content_hash(self) -> str:
        return hashlib.sha256(_canonical_bytes(self.verdict_layer())).hexdigest()

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema": self.schema,
            "workers": self.workers,
            "total_items": self.total_items,
            "n_decided": self.decided,
            "n_abstained": self.abstained,
            "faithful": self.faithful,
            "content_hash": self.content_hash(),
            "verdicts": self.verdict_layer(),
            "results": [
                {
                    "sequence": r.sequence,
                    "item_id": r.item_id,
                    "lang": r.lang,
                    "klass": r.klass,
                    "declared_label": r.declared_label,
                    "observed_label": r.observed_label,
                    "decided": r.decided,
                    "abstained": r.abstained,
                    "detail": r.detail,
                }
                for r in self.results
            ],
        }


def _result_from_evidence(seq: int, item: GTItem,
                          ev: LabelEvidence) -> ParallelItemResult:
    decided = ev.observed_label in _DECIDED
    return ParallelItemResult(
        sequence=seq,
        item_id=item.item_id,
        lang=item.lang,
        klass=item.klass,
        declared_label=item.declared_label,
        observed_label=ev.observed_label,
        decided=decided,
        abstained=not decided,
        detail=ev.detail,
    )


def _worker(args: Tuple[int, GTItem, ToolchainStatus, LabelFn]
            ) -> ParallelItemResult:
    seq, item, status, label_fn = args
    return _result_from_evidence(seq, item, label_fn(ReexecHarness(status), item))


def _ordered(results: Iterable[ParallelItemResult]) -> List[ParallelItemResult]:
    return sorted(results, key=lambda r: r.sequence)


def run_parallel(
    items: Sequence[GTItem],
    *,
    workers: int = 1,
    status: Optional[ToolchainStatus] = None,
    label_fn: LabelFn = label_item,
) -> ParallelRunReport:
    """Label ``items`` with a deterministic merge independent of scheduling."""
    if workers < 1:
        raise ValueError("workers must be >= 1")
    st = status or toolchain_available()
    indexed = [(i, item, st, label_fn) for i, item in enumerate(items)]

    if workers == 1 or len(indexed) <= 1:
        results = [_worker(arg) for arg in indexed]
    else:
        finished: List[ParallelItemResult] = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_worker, arg) for arg in indexed]
            for fut in concurrent.futures.as_completed(futures):
                finished.append(fut.result())
        results = _ordered(finished)

    return ParallelRunReport(
        workers=workers,
        total_items=len(items),
        results=_ordered(results),
    )


@dataclass
class ParallelDeterminismProof:
    available: bool
    ok: bool
    n_items: int
    workers: Tuple[int, ...]
    content_hashes: Dict[int, str]
    baseline_hash: str
    mismatches: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "available": self.available,
            "ok": self.ok,
            "n_items": self.n_items,
            "workers": list(self.workers),
            "content_hashes": dict(self.content_hashes),
            "baseline_hash": self.baseline_hash,
            "mismatches": list(self.mismatches),
        }


def _default_sample(status: ToolchainStatus,
                    langs: Tuple[str, ...] = ("rust", "go")) -> List[GTItem]:
    available = tuple(lang for lang in langs if status.full_for(lang))
    if not available:
        return []
    items = [item for item in enumerate_corpus(available)
             if item.lang in available]
    by_label: Dict[str, GTItem] = {}
    for item in items:
        by_label.setdefault(item.declared_label, item)
        if {"divergent", "equivalent"} <= set(by_label):
            break
    return [by_label[k] for k in sorted(by_label)]


def confirm_parallel_determinism(
    items: Optional[Sequence[GTItem]] = None,
    *,
    workers: Tuple[int, ...] = (1, 2),
    status: Optional[ToolchainStatus] = None,
    label_fn: LabelFn = label_item,
) -> ParallelDeterminismProof:
    """Run the same items at multiple worker counts and compare content hashes."""
    st = status or toolchain_available()
    chosen = list(items) if items is not None else _default_sample(st)
    if not chosen:
        return ParallelDeterminismProof(
            available=False,
            ok=False,
            n_items=0,
            workers=tuple(workers),
            content_hashes={},
            baseline_hash="",
            mismatches=["no runnable items"],
        )

    reports = [run_parallel(chosen, workers=w, status=st, label_fn=label_fn)
               for w in workers]
    baseline = reports[0]
    hashes = {report.workers: report.content_hash() for report in reports}
    mismatches: List[str] = []
    for report in reports[1:]:
        if report.verdict_layer() != baseline.verdict_layer():
            mismatches.append(
                f"workers={report.workers} verdict layer differs from "
                f"workers={baseline.workers}")
    if len(set(hashes.values())) != 1:
        mismatches.append("content hashes differ across worker counts")

    return ParallelDeterminismProof(
        available=True,
        ok=not mismatches,
        n_items=len(chosen),
        workers=tuple(workers),
        content_hashes=hashes,
        baseline_hash=baseline.content_hash(),
        mismatches=mismatches,
    )


if __name__ == "__main__":  # pragma: no cover
    proof = confirm_parallel_determinism()
    print(json.dumps(proof.to_dict(), indent=2, sort_keys=True))
