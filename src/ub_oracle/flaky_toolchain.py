"""Flaky-toolchain detector (100_STEPS step 148).

The parallel and sharded harnesses intentionally hash only the verdict layer so
timing and worker scheduling cannot perturb reproducibility claims.  That is the
right artifact boundary, but it is too coarse to catch a compiler/runtime that
emits unstable evidence while still producing the same high-level verdict.

This module reruns a corpus through the real labeler and compares two layers:

* the stable verdict projection used by the rest of the artifact; and
* a fuller evidence projection over ``LabelEvidence`` (UB-trap flag, C output,
  target output, and detail text, where compiler/runtime diagnostics are
  recorded by the existing harness).

Any item whose verdict or evidence projection changes across reruns is
quarantined with per-run hashes.  The report is still deterministic: it stores
hashes and compact records, not wall-clock timing or process metadata.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .ground_truth import GTItem, LabelEvidence, enumerate_corpus, label_item
from .parallel_harness import LabelFn, _canonical_bytes
from .reexec import ReexecHarness, ToolchainStatus, toolchain_available

SCHEMA_VERSION = "flaky-toolchain/v1"
_DECIDED = {"divergent", "equivalent"}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _item_key(item: GTItem) -> Tuple[str, str]:
    return (item.lang, item.item_id)


def _ensure_unique_items(items: Sequence[GTItem]) -> None:
    seen = set()
    for item in items:
        key = _item_key(item)
        if key in seen:
            raise ValueError(f"duplicate corpus item key: {key!r}")
        seen.add(key)


@dataclass(frozen=True)
class EvidenceResult:
    sequence: int
    item_id: str
    lang: str
    klass: str
    declared_label: str
    observed_label: str
    ub_trapped: bool
    c_out: str
    target_out: str
    detail: str

    @property
    def key(self) -> Tuple[str, str]:
        return (self.lang, self.item_id)

    @property
    def decided(self) -> bool:
        return self.observed_label in _DECIDED

    def verdict_record(self) -> Dict[str, object]:
        return {
            "item_id": self.item_id,
            "lang": self.lang,
            "klass": self.klass,
            "declared_label": self.declared_label,
            "observed_label": self.observed_label,
            "decided": self.decided,
            "abstained": not self.decided,
        }

    def evidence_record(self) -> Dict[str, object]:
        return {
            "item_id": self.item_id,
            "lang": self.lang,
            "klass": self.klass,
            "declared_label": self.declared_label,
            "observed_label": self.observed_label,
            "ub_trapped": self.ub_trapped,
            "c_out": self.c_out,
            "target_out": self.target_out,
            "detail": self.detail,
        }

    def evidence_hash(self) -> str:
        return _sha(_canonical_bytes(self.evidence_record()))


def _result_from_evidence(seq: int, item: GTItem, ev: LabelEvidence) -> EvidenceResult:
    return EvidenceResult(
        sequence=seq,
        item_id=item.item_id,
        lang=item.lang,
        klass=item.klass,
        declared_label=item.declared_label,
        observed_label=ev.observed_label,
        ub_trapped=ev.ub_trapped,
        c_out=ev.c_out,
        target_out=ev.target_out,
        detail=ev.detail,
    )


def _worker(args: Tuple[int, GTItem, ToolchainStatus, LabelFn]) -> EvidenceResult:
    seq, item, status, label_fn = args
    return _result_from_evidence(seq, item, label_fn(ReexecHarness(status), item))


def _ordered(results: Iterable[EvidenceResult]) -> List[EvidenceResult]:
    return sorted(results, key=lambda r: r.sequence)


@dataclass
class EvidenceRun:
    run_index: int
    workers: int
    results: List[EvidenceResult] = field(default_factory=list)

    def verdict_layer(self) -> List[Dict[str, object]]:
        return sorted(
            (r.verdict_record() for r in self.results),
            key=lambda r: (str(r["item_id"]), str(r["lang"])),
        )

    def evidence_layer(self) -> List[Dict[str, object]]:
        return sorted(
            (r.evidence_record() for r in self.results),
            key=lambda r: (str(r["item_id"]), str(r["lang"])),
        )

    def verdict_hash(self) -> str:
        return _sha(_canonical_bytes(self.verdict_layer()))

    def evidence_hash(self) -> str:
        return _sha(_canonical_bytes(self.evidence_layer()))

    def to_dict(self, *, include_records: bool = True) -> Dict[str, object]:
        out: Dict[str, object] = {
            "run_index": self.run_index,
            "workers": self.workers,
            "verdict_hash": self.verdict_hash(),
            "evidence_hash": self.evidence_hash(),
        }
        if include_records:
            out["evidence"] = self.evidence_layer()
        return out


def run_evidence_once(
    items: Sequence[GTItem],
    *,
    run_index: int = 0,
    workers: int = 1,
    status: Optional[ToolchainStatus] = None,
    label_fn: LabelFn = label_item,
) -> EvidenceRun:
    """Run one pass over ``items`` and retain deterministic evidence records."""

    if workers < 1:
        raise ValueError("workers must be >= 1")
    item_list = list(items)
    _ensure_unique_items(item_list)
    st = status or toolchain_available()
    indexed = [(i, item, st, label_fn) for i, item in enumerate(item_list)]
    if workers == 1 or len(indexed) <= 1:
        results = [_worker(arg) for arg in indexed]
    else:
        finished: List[EvidenceResult] = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_worker, arg) for arg in indexed]
            for fut in concurrent.futures.as_completed(futures):
                finished.append(fut.result())
        results = _ordered(finished)
    return EvidenceRun(run_index=run_index, workers=workers, results=_ordered(results))


@dataclass(frozen=True)
class QuarantinedItem:
    item_id: str
    lang: str
    klass: str
    declared_label: str
    verdicts: Tuple[str, ...]
    evidence_hashes: Tuple[str, ...]
    verdict_unstable: bool
    evidence_unstable: bool

    def to_dict(self) -> Dict[str, object]:
        return {
            "item_id": self.item_id,
            "lang": self.lang,
            "klass": self.klass,
            "declared_label": self.declared_label,
            "verdicts": list(self.verdicts),
            "evidence_hashes": list(self.evidence_hashes),
            "verdict_unstable": self.verdict_unstable,
            "evidence_unstable": self.evidence_unstable,
        }


@dataclass
class FlakyToolchainReport:
    available: bool
    stable: bool
    verdict_stable: bool
    evidence_stable: bool
    runs: int
    total_items: int
    baseline_verdict_hash: str
    baseline_evidence_hash: str
    run_verdict_hashes: List[str] = field(default_factory=list)
    run_evidence_hashes: List[str] = field(default_factory=list)
    quarantined: List[QuarantinedItem] = field(default_factory=list)
    mismatches: List[str] = field(default_factory=list)
    schema: str = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema": self.schema,
            "available": self.available,
            "stable": self.stable,
            "verdict_stable": self.verdict_stable,
            "evidence_stable": self.evidence_stable,
            "runs": self.runs,
            "total_items": self.total_items,
            "baseline_verdict_hash": self.baseline_verdict_hash,
            "baseline_evidence_hash": self.baseline_evidence_hash,
            "run_verdict_hashes": list(self.run_verdict_hashes),
            "run_evidence_hashes": list(self.run_evidence_hashes),
            "quarantined": [q.to_dict() for q in self.quarantined],
            "mismatches": list(self.mismatches),
        }


def _default_sample(status: ToolchainStatus) -> List[GTItem]:
    available = tuple(lang for lang in ("rust", "go") if status.full_for(lang))
    if not available:
        return []
    by_label: Dict[str, GTItem] = {}
    for item in enumerate_corpus(available):
        by_label.setdefault(item.declared_label, item)
        if {"divergent", "equivalent"} <= set(by_label):
            break
    return [by_label[k] for k in sorted(by_label)]


def _empty_report(runs: int, mismatches: List[str]) -> FlakyToolchainReport:
    return FlakyToolchainReport(
        available=False,
        stable=False,
        verdict_stable=False,
        evidence_stable=False,
        runs=runs,
        total_items=0,
        baseline_verdict_hash="",
        baseline_evidence_hash="",
        mismatches=mismatches,
    )


def _quarantine(runs: Sequence[EvidenceRun]) -> List[QuarantinedItem]:
    by_key: Dict[Tuple[str, str], List[EvidenceResult]] = {}
    for run in runs:
        for result in run.results:
            by_key.setdefault(result.key, []).append(result)

    quarantined: List[QuarantinedItem] = []
    for key in sorted(by_key):
        records = by_key[key]
        verdicts = tuple(r.observed_label for r in records)
        evidence_hashes = tuple(r.evidence_hash() for r in records)
        verdict_unstable = len(set(verdicts)) > 1
        evidence_unstable = len(set(evidence_hashes)) > 1
        if verdict_unstable or evidence_unstable:
            first = records[0]
            quarantined.append(
                QuarantinedItem(
                    item_id=first.item_id,
                    lang=first.lang,
                    klass=first.klass,
                    declared_label=first.declared_label,
                    verdicts=verdicts,
                    evidence_hashes=evidence_hashes,
                    verdict_unstable=verdict_unstable,
                    evidence_unstable=evidence_unstable,
                )
            )
    return quarantined


def detect_flaky_toolchain(
    items: Optional[Sequence[GTItem]] = None,
    *,
    runs: int = 3,
    workers: int = 1,
    status: Optional[ToolchainStatus] = None,
    label_fn: LabelFn = label_item,
) -> FlakyToolchainReport:
    """Rerun ``items`` and quarantine verdict/evidence instability.

    ``runs`` must be at least two: a single pass cannot distinguish a stable
    compiler from a lucky observation of a flaky one.
    """

    if runs < 2:
        raise ValueError("runs must be >= 2")
    st = status or toolchain_available()
    item_list = list(items) if items is not None else _default_sample(st)
    if not item_list:
        return _empty_report(runs, ["no runnable items"])
    _ensure_unique_items(item_list)

    observed_runs = [
        run_evidence_once(
            item_list,
            run_index=i,
            workers=workers,
            status=st,
            label_fn=label_fn,
        )
        for i in range(runs)
    ]
    verdict_hashes = [run.verdict_hash() for run in observed_runs]
    evidence_hashes = [run.evidence_hash() for run in observed_runs]
    quarantined = _quarantine(observed_runs)

    mismatches: List[str] = []
    if len(set(verdict_hashes)) > 1:
        mismatches.append("verdict layer changed across reruns")
    if len(set(evidence_hashes)) > 1:
        mismatches.append("compiler/runtime evidence changed across reruns")
    if quarantined:
        mismatches.append(f"{len(quarantined)} item(s) quarantined as flaky")

    verdict_stable = len(set(verdict_hashes)) == 1
    evidence_stable = len(set(evidence_hashes)) == 1
    return FlakyToolchainReport(
        available=True,
        stable=verdict_stable and evidence_stable and not quarantined,
        verdict_stable=verdict_stable,
        evidence_stable=evidence_stable,
        runs=runs,
        total_items=len(item_list),
        baseline_verdict_hash=verdict_hashes[0],
        baseline_evidence_hash=evidence_hashes[0],
        run_verdict_hashes=verdict_hashes,
        run_evidence_hashes=evidence_hashes,
        quarantined=quarantined,
        mismatches=mismatches,
    )


FLAKY_TOOLCHAIN_SPI = {
    "run_evidence_once": run_evidence_once,
    "detect_flaky_toolchain": detect_flaky_toolchain,
}


if __name__ == "__main__":  # pragma: no cover
    report = detect_flaky_toolchain()
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
