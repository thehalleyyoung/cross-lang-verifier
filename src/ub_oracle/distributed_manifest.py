"""Distributed run manifests for deterministic corpus farming (step 149).

Step 147 proves that local shards merge to the same verdict hash as a whole run.
Distributed execution needs one more artifact: a manifest that pins exactly which
items each worker is allowed to run, so separately executed shard results can be
validated and merged without trusting local corpus enumeration.

The manifest records every item key and source/target content hash per shard.
Shard result documents must echo those identities, and the merger rejects missing,
duplicate, drifted, or foreign shards before computing the same canonical verdict
hash used by the parallel and sharded harnesses.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .ground_truth import GTItem, label_item
from .parallel_harness import LabelFn, _canonical_bytes, run_parallel
from .reexec import ToolchainStatus, toolchain_available
from .sharded_repro import ShardSpec, split_shards

SCHEMA_VERSION = "distributed-manifest/v1"
SHARD_RESULT_SCHEMA = "distributed-shard-result/v1"
MERGE_SCHEMA = "distributed-merge/v1"


class ManifestValidationError(ValueError):
    """Raised when a distributed manifest or shard result is inconsistent."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash_obj(obj: object) -> str:
    return _sha(_canonical_bytes(obj))


def _inputs_hash(item: GTItem) -> str:
    return _hash_obj(list(item.inputs))


def item_key(item: GTItem) -> str:
    return f"{item.lang}:{item.item_id}"


def item_identity(item: GTItem) -> Dict[str, object]:
    """The cross-machine identity a worker must echo for a corpus item."""

    return {
        "key": item_key(item),
        "item_id": item.item_id,
        "lang": item.lang,
        "klass": item.klass,
        "declared_label": item.declared_label,
        "content_hash": item.content_hash,
        "inputs_hash": _inputs_hash(item),
    }


def _ensure_unique_identities(identities: Sequence[Dict[str, object]]) -> None:
    seen = set()
    for ident in identities:
        key = str(ident["key"])
        if key in seen:
            raise ManifestValidationError(f"duplicate item key in manifest: {key}")
        seen.add(key)


def _verdict_key(record: Dict[str, object]) -> str:
    return f"{record['lang']}:{record['item_id']}"


def _sorted_verdicts(records: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    return sorted((dict(r) for r in records), key=lambda r: (str(r["item_id"]), str(r["lang"])))


@dataclass(frozen=True)
class ManifestShard:
    index: int
    start: int
    end: int
    identities: Tuple[Dict[str, object], ...]

    @property
    def n_items(self) -> int:
        return self.end - self.start

    @property
    def item_keys(self) -> Tuple[str, ...]:
        return tuple(str(i["key"]) for i in self.identities)

    @property
    def identity_hash(self) -> str:
        return _hash_obj(list(self.identities))

    def spec_dict(self) -> Dict[str, int]:
        return {
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "n_items": self.n_items,
        }

    def to_dict(self) -> Dict[str, object]:
        out: Dict[str, object] = self.spec_dict()
        out.update(
            {
                "item_keys": list(self.item_keys),
                "item_identities": [dict(i) for i in self.identities],
                "identity_hash": self.identity_hash,
            }
        )
        return out


@dataclass(frozen=True)
class DistributedManifest:
    total_items: int
    shards: Tuple[ManifestShard, ...]
    schema: str = SCHEMA_VERSION

    @property
    def n_shards(self) -> int:
        return len(self.shards)

    @property
    def corpus_identities(self) -> Tuple[Dict[str, object], ...]:
        return tuple(ident for shard in self.shards for ident in shard.identities)

    @property
    def corpus_hash(self) -> str:
        return _hash_obj(list(self.corpus_identities))

    def body_dict(self) -> Dict[str, object]:
        return {
            "schema": self.schema,
            "total_items": self.total_items,
            "n_shards": self.n_shards,
            "corpus_hash": self.corpus_hash,
            "shards": [s.to_dict() for s in self.shards],
        }

    @property
    def manifest_hash(self) -> str:
        return _hash_obj(self.body_dict())

    def to_dict(self) -> Dict[str, object]:
        out = self.body_dict()
        out["manifest_hash"] = self.manifest_hash
        return out


def create_manifest(
    items: Sequence[GTItem],
    *,
    shard_size: Optional[int] = None,
    shard_count: Optional[int] = None,
) -> DistributedManifest:
    """Create a deterministic, content-addressed distributed manifest."""

    item_list = list(items)
    identities = [item_identity(item) for item in item_list]
    _ensure_unique_identities(identities)
    shards: List[ManifestShard] = []
    for spec in split_shards(
        item_list,
        shard_size=shard_size,
        shard_count=shard_count,
    ):
        shard_identities = tuple(identities[spec.start:spec.end])
        shards.append(
            ManifestShard(
                index=spec.index,
                start=spec.start,
                end=spec.end,
                identities=shard_identities,
            )
        )
    return DistributedManifest(total_items=len(item_list), shards=tuple(shards))


def _shard_by_index(manifest: DistributedManifest, index: int) -> ManifestShard:
    for shard in manifest.shards:
        if shard.index == index:
            return shard
    raise ManifestValidationError(f"unknown shard index: {index}")


def _validate_local_slice(
    manifest: DistributedManifest,
    items: Sequence[GTItem],
    shard: ManifestShard,
) -> None:
    local = [item_identity(item) for item in items[shard.start:shard.end]]
    if local != [dict(i) for i in shard.identities]:
        raise ManifestValidationError(
            f"local corpus slice for shard {shard.index} does not match manifest"
        )


def _result_hash(doc: Dict[str, object]) -> str:
    body = {k: v for k, v in doc.items() if k != "result_hash"}
    return _hash_obj(body)


def run_manifest_shard(
    manifest: DistributedManifest,
    items: Sequence[GTItem],
    shard_index: int,
    *,
    workers: int = 1,
    status: Optional[ToolchainStatus] = None,
    label_fn: LabelFn = label_item,
) -> Dict[str, object]:
    """Run one manifest shard and emit a self-validating shard result document."""

    shard = _shard_by_index(manifest, shard_index)
    item_list = list(items)
    _validate_local_slice(manifest, item_list, shard)
    report = run_parallel(
        item_list[shard.start:shard.end],
        workers=workers,
        status=status or toolchain_available(),
        label_fn=label_fn,
    )
    verdicts = report.verdict_layer()
    doc: Dict[str, object] = {
        "schema": SHARD_RESULT_SCHEMA,
        "manifest_hash": manifest.manifest_hash,
        "shard": shard.spec_dict(),
        "shard_identity_hash": shard.identity_hash,
        "item_identities": [dict(i) for i in shard.identities],
        "verdict_hash": _hash_obj(verdicts),
        "n_decided": report.decided,
        "n_abstained": report.abstained,
        "faithful": report.faithful,
        "verdicts": verdicts,
    }
    doc["result_hash"] = _result_hash(doc)
    return doc


@dataclass
class DistributedMerge:
    manifest_hash: str
    total_items: int
    n_shards: int
    content_hash: str
    shard_result_hashes: Dict[int, str]
    verdicts: List[Dict[str, object]] = field(default_factory=list)
    schema: str = MERGE_SCHEMA

    def to_dict(self, *, include_verdicts: bool = True) -> Dict[str, object]:
        out: Dict[str, object] = {
            "schema": self.schema,
            "manifest_hash": self.manifest_hash,
            "total_items": self.total_items,
            "n_shards": self.n_shards,
            "content_hash": self.content_hash,
            "shard_result_hashes": dict(sorted(self.shard_result_hashes.items())),
        }
        if include_verdicts:
            out["verdicts"] = list(self.verdicts)
        return out


def _validate_shard_doc(
    manifest: DistributedManifest,
    doc: Dict[str, object],
) -> Tuple[int, List[Dict[str, object]], str]:
    if doc.get("schema") != SHARD_RESULT_SCHEMA:
        raise ManifestValidationError("shard result schema mismatch")
    if doc.get("manifest_hash") != manifest.manifest_hash:
        raise ManifestValidationError("shard result belongs to a different manifest")
    if doc.get("result_hash") != _result_hash(doc):
        raise ManifestValidationError("shard result hash mismatch")

    shard_info = doc.get("shard")
    if not isinstance(shard_info, dict):
        raise ManifestValidationError("shard result missing shard spec")
    index = int(shard_info.get("index", -1))
    shard = _shard_by_index(manifest, index)
    if shard_info != shard.spec_dict():
        raise ManifestValidationError(f"shard {index} boundary mismatch")
    if doc.get("shard_identity_hash") != shard.identity_hash:
        raise ManifestValidationError(f"shard {index} identity hash mismatch")
    if doc.get("item_identities") != [dict(i) for i in shard.identities]:
        raise ManifestValidationError(f"shard {index} item identities mismatch")

    verdicts_obj = doc.get("verdicts")
    if not isinstance(verdicts_obj, list):
        raise ManifestValidationError(f"shard {index} missing verdicts")
    verdicts = [dict(v) for v in verdicts_obj]
    expected_keys = set(shard.item_keys)
    observed_keys = {_verdict_key(v) for v in verdicts}
    if observed_keys != expected_keys:
        raise ManifestValidationError(f"shard {index} verdict keys mismatch")
    if doc.get("verdict_hash") != _hash_obj(_sorted_verdicts(verdicts)):
        raise ManifestValidationError(f"shard {index} verdict hash mismatch")
    return index, verdicts, str(doc["result_hash"])


def merge_shard_results(
    manifest: DistributedManifest,
    shard_results: Sequence[Dict[str, object]],
) -> DistributedMerge:
    """Validate and merge all shard result documents for ``manifest``."""

    seen_shards = set()
    seen_items = set()
    merged: List[Dict[str, object]] = []
    result_hashes: Dict[int, str] = {}
    for doc in shard_results:
        index, verdicts, result_hash = _validate_shard_doc(manifest, dict(doc))
        if index in seen_shards:
            raise ManifestValidationError(f"duplicate shard result: {index}")
        seen_shards.add(index)
        result_hashes[index] = result_hash
        for verdict in verdicts:
            key = _verdict_key(verdict)
            if key in seen_items:
                raise ManifestValidationError(f"duplicate verdict key: {key}")
            seen_items.add(key)
        merged.extend(verdicts)

    expected = {shard.index for shard in manifest.shards}
    if seen_shards != expected:
        missing = sorted(expected - seen_shards)
        extra = sorted(seen_shards - expected)
        raise ManifestValidationError(f"shard coverage mismatch missing={missing} extra={extra}")

    verdicts = _sorted_verdicts(merged)
    return DistributedMerge(
        manifest_hash=manifest.manifest_hash,
        total_items=manifest.total_items,
        n_shards=manifest.n_shards,
        content_hash=_hash_obj(verdicts),
        shard_result_hashes=result_hashes,
        verdicts=verdicts,
    )


@dataclass
class DistributedManifestProof:
    available: bool
    ok: bool
    manifest_hash: str
    whole_hash: str
    merged_hash: str
    n_shards: int
    total_items: int
    mismatches: List[str] = field(default_factory=list)
    schema: str = MERGE_SCHEMA

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema": self.schema,
            "available": self.available,
            "ok": self.ok,
            "manifest_hash": self.manifest_hash,
            "whole_hash": self.whole_hash,
            "merged_hash": self.merged_hash,
            "n_shards": self.n_shards,
            "total_items": self.total_items,
            "mismatches": list(self.mismatches),
        }


def confirm_distributed_manifest(
    items: Sequence[GTItem],
    *,
    shard_size: Optional[int] = None,
    shard_count: Optional[int] = None,
    workers: int = 1,
    status: Optional[ToolchainStatus] = None,
    label_fn: LabelFn = label_item,
) -> DistributedManifestProof:
    """Run all manifest shards, merge them, and compare against a whole run."""

    item_list = list(items)
    if not item_list:
        return DistributedManifestProof(
            available=False,
            ok=False,
            manifest_hash="",
            whole_hash="",
            merged_hash="",
            n_shards=0,
            total_items=0,
            mismatches=["no items"],
        )
    if shard_size is None and shard_count is None:
        shard_size = 1
    st = status or toolchain_available()
    manifest = create_manifest(
        item_list,
        shard_size=shard_size,
        shard_count=shard_count,
    )
    shard_docs = [
        run_manifest_shard(
            manifest,
            item_list,
            shard.index,
            workers=workers,
            status=st,
            label_fn=label_fn,
        )
        for shard in manifest.shards
    ]
    merged = merge_shard_results(manifest, shard_docs)
    whole = run_parallel(item_list, workers=workers, status=st, label_fn=label_fn)
    whole_hash = whole.content_hash()
    mismatches = []
    if merged.content_hash != whole_hash:
        mismatches.append("merged distributed hash differs from whole-run hash")
    if not whole.faithful:
        mismatches.append("whole run produced an unfaithful decided verdict")
    return DistributedManifestProof(
        available=True,
        ok=not mismatches,
        manifest_hash=manifest.manifest_hash,
        whole_hash=whole_hash,
        merged_hash=merged.content_hash,
        n_shards=manifest.n_shards,
        total_items=len(item_list),
        mismatches=mismatches,
    )


DISTRIBUTED_MANIFEST_SPI = {
    "create_manifest": create_manifest,
    "run_manifest_shard": run_manifest_shard,
    "merge_shard_results": merge_shard_results,
    "confirm_distributed_manifest": confirm_distributed_manifest,
}


if __name__ == "__main__":  # pragma: no cover
    from .ground_truth import enumerate_corpus

    sample = enumerate_corpus(("rust",))[:2]
    proof = confirm_distributed_manifest(sample, shard_size=1)
    print(json.dumps(proof.to_dict(), indent=2, sort_keys=True))
