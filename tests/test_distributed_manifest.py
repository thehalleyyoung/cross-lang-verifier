from __future__ import annotations

import pytest

from src.ub_oracle.distributed_manifest import (
    ManifestValidationError,
    confirm_distributed_manifest,
    create_manifest,
    merge_shard_results,
    run_manifest_shard,
)
from src.ub_oracle.ground_truth import GTItem, LabelEvidence
from src.ub_oracle.parallel_harness import run_parallel
from src.ub_oracle.reexec import ToolchainStatus


def _status() -> ToolchainStatus:
    return ToolchainStatus(cc=None, ubsan=False, targets=())


def _item(item_id: str, label: str = "equivalent") -> GTItem:
    return GTItem(
        item_id=item_id,
        lang="rust",
        klass="safe_add" if label == "equivalent" else "div_by_zero",
        cwe="" if label == "equivalent" else "CWE-369",
        declared_label=label,
        c_src=f"/* {item_id} C */",
        target_src=f"// {item_id} Rust",
        inputs=(),
    )


def label(_h, item: GTItem) -> LabelEvidence:
    return LabelEvidence(
        observed_label=item.declared_label,
        ub_trapped=item.declared_label == "divergent",
        c_out="",
        target_out=item.item_id,
        detail="manifest test evidence",
    )


def test_manifest_pins_shard_boundaries_and_item_identities():
    items = [_item("a"), _item("b", "divergent"), _item("c")]
    manifest = create_manifest(items, shard_size=2)
    doc = manifest.to_dict()

    assert doc["schema"] == "distributed-manifest/v1"
    assert doc["total_items"] == 3
    assert doc["n_shards"] == 2
    assert len(doc["manifest_hash"]) == 64
    assert doc["shards"][0]["item_keys"] == ["rust:a", "rust:b"]
    assert doc == create_manifest(items, shard_size=2).to_dict()


def test_distributed_merge_matches_whole_run_hash():
    items = [_item("a"), _item("b", "divergent"), _item("c"), _item("d")]
    manifest = create_manifest(items, shard_size=2)
    shard_docs = [
        run_manifest_shard(
            manifest,
            items,
            shard.index,
            status=_status(),
            label_fn=label,
        )
        for shard in manifest.shards
    ]

    merged = merge_shard_results(manifest, shard_docs)
    whole = run_parallel(items, status=_status(), label_fn=label)

    assert merged.content_hash == whole.content_hash()
    assert merged.total_items == len(items)
    assert set(merged.shard_result_hashes) == {0, 1}


def test_confirm_distributed_manifest_proves_whole_hash_equivalence():
    items = [_item("a"), _item("b", "divergent"), _item("c")]

    proof = confirm_distributed_manifest(
        items,
        shard_count=2,
        status=_status(),
        label_fn=label,
    )

    assert proof.ok, proof.to_dict()
    assert proof.whole_hash == proof.merged_hash


def test_merge_rejects_tampered_item_identity():
    items = [_item("a"), _item("b")]
    manifest = create_manifest(items, shard_size=1)
    shard_docs = [
        run_manifest_shard(
            manifest,
            items,
            shard.index,
            status=_status(),
            label_fn=label,
        )
        for shard in manifest.shards
    ]
    shard_docs[0]["item_identities"][0]["content_hash"] = "tampered"

    with pytest.raises(ManifestValidationError, match="hash mismatch|identities mismatch"):
        merge_shard_results(manifest, shard_docs)


def test_merge_rejects_missing_and_duplicate_shards():
    items = [_item("a"), _item("b")]
    manifest = create_manifest(items, shard_size=1)
    first = run_manifest_shard(
        manifest,
        items,
        0,
        status=_status(),
        label_fn=label,
    )

    with pytest.raises(ManifestValidationError, match="coverage mismatch"):
        merge_shard_results(manifest, [first])
    with pytest.raises(ManifestValidationError, match="duplicate shard"):
        merge_shard_results(manifest, [first, first])
