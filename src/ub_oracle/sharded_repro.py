"""Sharded reproducibility proof for migration-scale runs (100_STEPS step 147).

The parallel harness already gives us a deterministic, detail-free verdict layer:
worker order, timings, process ids, and stderr snippets are deliberately excluded
from the content hash.  This module proves that the same discipline composes over
corpus shards:

* a corpus is split into deterministic contiguous shards;
* each shard emits byte-stable canonical verdict bytes; and
* concatenating all shard verdict records and re-canonicalizing them yields the
  exact same hash as an unsharded run over the whole corpus.

The last re-canonicalization uses the same globally sorted verdict projection as
``parallel_harness.ParallelRunReport.content_hash``.  A separate shard-chain hash
records the sequence of per-shard bytes, but it is not confused with the whole
corpus verdict hash.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .ground_truth import GTItem, enumerate_corpus, label_item
from .parallel_harness import (
    LabelFn,
    _canonical_bytes,
    run_parallel,
)
from .reexec import ToolchainStatus, toolchain_available

SCHEMA_VERSION = "sharded-repro/v1"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _verdict_sort_key(record: Dict[str, object]) -> Tuple[str, str]:
    return (str(record["item_id"]), str(record["lang"]))


def _sorted_verdicts(
    verdicts: Iterable[Dict[str, object]],
) -> List[Dict[str, object]]:
    return sorted((dict(v) for v in verdicts), key=_verdict_sort_key)


def _ensure_unique_items(items: Sequence[GTItem]) -> None:
    seen = set()
    for item in items:
        key = (item.item_id, item.lang)
        if key in seen:
            raise ValueError(f"duplicate corpus item key: {key!r}")
        seen.add(key)


@dataclass(frozen=True)
class ShardSpec:
    """A deterministic contiguous slice of the input corpus."""

    index: int
    start: int
    end: int

    @property
    def n_items(self) -> int:
        return self.end - self.start

    def to_dict(self) -> Dict[str, int]:
        return {
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "n_items": self.n_items,
        }


def split_shards(
    items: Sequence[GTItem],
    *,
    shard_size: Optional[int] = None,
    shard_count: Optional[int] = None,
) -> Tuple[ShardSpec, ...]:
    """Split ``items`` into deterministic, non-empty contiguous shards.

    Exactly one of ``shard_size`` or ``shard_count`` must be supplied.  Count-based
    sharding uses a front-loaded remainder so repeated runs and distributed
    workers agree on boundaries byte-for-byte.
    """

    if (shard_size is None) == (shard_count is None):
        raise ValueError("set exactly one of shard_size or shard_count")

    n = len(items)
    if n == 0:
        raise ValueError("cannot shard an empty corpus")

    if shard_size is not None:
        if shard_size < 1:
            raise ValueError("shard_size must be >= 1")
        return tuple(
            ShardSpec(i, start, min(start + shard_size, n))
            for i, start in enumerate(range(0, n, shard_size))
        )

    assert shard_count is not None
    if shard_count < 1:
        raise ValueError("shard_count must be >= 1")
    if shard_count > n:
        raise ValueError("shard_count cannot exceed corpus size")

    base, remainder = divmod(n, shard_count)
    specs: List[ShardSpec] = []
    start = 0
    for index in range(shard_count):
        width = base + (1 if index < remainder else 0)
        end = start + width
        specs.append(ShardSpec(index, start, end))
        start = end
    return tuple(specs)


@dataclass
class ShardRun:
    """The stable verdict artifact for one shard."""

    index: int
    start: int
    end: int
    n_items: int
    byte_hash: str
    decided: int
    abstained: int
    faithful: bool
    verdicts: List[Dict[str, object]] = field(default_factory=list)

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.verdicts)

    def to_dict(self, *, include_verdicts: bool = True) -> Dict[str, object]:
        out: Dict[str, object] = {
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "n_items": self.n_items,
            "byte_hash": self.byte_hash,
            "decided": self.decided,
            "abstained": self.abstained,
            "faithful": self.faithful,
        }
        if include_verdicts:
            out["verdicts"] = list(self.verdicts)
        return out


@dataclass
class ShardedRun:
    """A sharded run plus the hashes needed to compare it with a whole run."""

    total_items: int
    shards: List[ShardRun] = field(default_factory=list)
    schema: str = SCHEMA_VERSION

    @property
    def n_shards(self) -> int:
        return len(self.shards)

    @property
    def faithful(self) -> bool:
        return all(s.faithful for s in self.shards)

    def merged_verdict_layer(self) -> List[Dict[str, object]]:
        return _sorted_verdicts(
            verdict for shard in self.shards for verdict in shard.verdicts
        )

    def concatenated_hash(self) -> str:
        """Hash of all shard verdicts after the same global canonicalization as
        the unsharded run.  This is the value that must match the whole-corpus
        hash."""

        return _sha256(_canonical_bytes(self.merged_verdict_layer()))

    def shard_chain_hash(self) -> str:
        """Hash of the per-shard canonical bytes in shard order.

        This proves the shard files themselves are stable, but it is intentionally
        not compared to the whole-corpus hash because the whole hash is over one
        globally sorted verdict layer.
        """

        h = hashlib.sha256()
        for shard in self.shards:
            data = shard.canonical_bytes()
            h.update(len(data).to_bytes(8, "big"))
            h.update(data)
        return h.hexdigest()

    def to_dict(self, *, include_verdicts: bool = True) -> Dict[str, object]:
        return {
            "schema": self.schema,
            "total_items": self.total_items,
            "n_shards": self.n_shards,
            "faithful": self.faithful,
            "concatenated_hash": self.concatenated_hash(),
            "shard_chain_hash": self.shard_chain_hash(),
            "shards": [
                shard.to_dict(include_verdicts=include_verdicts)
                for shard in self.shards
            ],
        }


def run_sharded(
    items: Sequence[GTItem],
    *,
    shard_size: Optional[int] = None,
    shard_count: Optional[int] = None,
    workers: int = 1,
    status: Optional[ToolchainStatus] = None,
    label_fn: LabelFn = label_item,
) -> ShardedRun:
    """Run every shard and return detail-free, byte-stable verdict artifacts."""

    item_list = list(items)
    _ensure_unique_items(item_list)
    st = status or toolchain_available()
    shards: List[ShardRun] = []
    for spec in split_shards(
        item_list,
        shard_size=shard_size,
        shard_count=shard_count,
    ):
        report = run_parallel(
            item_list[spec.start:spec.end],
            workers=workers,
            status=st,
            label_fn=label_fn,
        )
        verdicts = report.verdict_layer()
        shards.append(
            ShardRun(
                index=spec.index,
                start=spec.start,
                end=spec.end,
                n_items=spec.n_items,
                byte_hash=_sha256(_canonical_bytes(verdicts)),
                decided=report.decided,
                abstained=report.abstained,
                faithful=report.faithful,
                verdicts=verdicts,
            )
        )
    return ShardedRun(total_items=len(item_list), shards=shards)


@dataclass
class ShardedReproProof:
    """Evidence that sharding is byte-stable and equivalent to a whole run."""

    available: bool
    ok: bool
    total_items: int
    n_shards: int
    whole_hash: str
    concatenated_hash: str
    repeat_concatenated_hash: str
    shard_chain_hash: str
    repeat_shard_chain_hash: str
    shard_hashes: Dict[int, str]
    repeat_shard_hashes: Dict[int, str]
    mismatches: List[str] = field(default_factory=list)
    schema: str = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema": self.schema,
            "available": self.available,
            "ok": self.ok,
            "total_items": self.total_items,
            "n_shards": self.n_shards,
            "whole_hash": self.whole_hash,
            "concatenated_hash": self.concatenated_hash,
            "repeat_concatenated_hash": self.repeat_concatenated_hash,
            "shard_chain_hash": self.shard_chain_hash,
            "repeat_shard_chain_hash": self.repeat_shard_chain_hash,
            "shard_hashes": dict(sorted(self.shard_hashes.items())),
            "repeat_shard_hashes": dict(sorted(self.repeat_shard_hashes.items())),
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


def _empty_proof(mismatches: List[str]) -> ShardedReproProof:
    return ShardedReproProof(
        available=False,
        ok=False,
        total_items=0,
        n_shards=0,
        whole_hash="",
        concatenated_hash="",
        repeat_concatenated_hash="",
        shard_chain_hash="",
        repeat_shard_chain_hash="",
        shard_hashes={},
        repeat_shard_hashes={},
        mismatches=mismatches,
    )


def confirm_sharded_reproducibility(
    items: Optional[Sequence[GTItem]] = None,
    *,
    shard_size: Optional[int] = None,
    shard_count: Optional[int] = None,
    workers: int = 1,
    status: Optional[ToolchainStatus] = None,
    label_fn: LabelFn = label_item,
) -> ShardedReproProof:
    """Prove a sharded run reproduces the whole-run verdict hash.

    The proof executes the unsharded run once and the sharded run twice, all with
    the same ``ToolchainStatus`` and ``label_fn``.  It succeeds only if the
    globally canonicalized concatenation of shard verdicts equals the whole-run
    verdict layer and every shard's canonical bytes repeat exactly.
    """

    st = status or toolchain_available()
    item_list = list(items) if items is not None else _default_sample(st)
    if not item_list:
        return _empty_proof(["no runnable items"])
    _ensure_unique_items(item_list)
    if shard_size is None and shard_count is None:
        shard_size = 1

    whole = run_parallel(item_list, workers=workers, status=st, label_fn=label_fn)
    sharded = run_sharded(
        item_list,
        shard_size=shard_size,
        shard_count=shard_count,
        workers=workers,
        status=st,
        label_fn=label_fn,
    )
    repeated = run_sharded(
        item_list,
        shard_size=shard_size,
        shard_count=shard_count,
        workers=workers,
        status=st,
        label_fn=label_fn,
    )

    whole_layer = whole.verdict_layer()
    merged_layer = sharded.merged_verdict_layer()
    whole_hash = whole.content_hash()
    concatenated_hash = sharded.concatenated_hash()
    repeat_concatenated_hash = repeated.concatenated_hash()
    shard_hashes = {s.index: s.byte_hash for s in sharded.shards}
    repeat_shard_hashes = {s.index: s.byte_hash for s in repeated.shards}

    mismatches: List[str] = []
    if merged_layer != whole_layer:
        mismatches.append("merged shard verdict layer differs from whole run")
    if concatenated_hash != whole_hash:
        mismatches.append("concatenated shard hash differs from whole hash")
    if repeat_concatenated_hash != concatenated_hash:
        mismatches.append("concatenated shard hash is not stable across runs")
    if shard_hashes != repeat_shard_hashes:
        mismatches.append("per-shard byte hashes are not stable across runs")
    if sharded.shard_chain_hash() != repeated.shard_chain_hash():
        mismatches.append("per-shard byte chain is not stable across runs")
    if not whole.faithful:
        mismatches.append("whole run produced an unfaithful decided verdict")
    if not sharded.faithful:
        mismatches.append("sharded run produced an unfaithful decided verdict")

    return ShardedReproProof(
        available=True,
        ok=not mismatches,
        total_items=len(item_list),
        n_shards=sharded.n_shards,
        whole_hash=whole_hash,
        concatenated_hash=concatenated_hash,
        repeat_concatenated_hash=repeat_concatenated_hash,
        shard_chain_hash=sharded.shard_chain_hash(),
        repeat_shard_chain_hash=repeated.shard_chain_hash(),
        shard_hashes=shard_hashes,
        repeat_shard_hashes=repeat_shard_hashes,
        mismatches=mismatches,
    )


SHARDED_REPRO_SPI = {
    "split_shards": split_shards,
    "run_sharded": run_sharded,
    "confirm_sharded_reproducibility": confirm_sharded_reproducibility,
}


if __name__ == "__main__":  # pragma: no cover
    proof = confirm_sharded_reproducibility()
    print(json.dumps(proof.to_dict(), indent=2, sort_keys=True))
