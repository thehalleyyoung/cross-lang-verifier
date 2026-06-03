"""Focused tests for Step 147's sharded reproducibility proof."""

from __future__ import annotations

import time

import pytest

from src.ub_oracle.ground_truth import GTItem, LabelEvidence, enumerate_corpus
from src.ub_oracle.parallel_harness import run_parallel
from src.ub_oracle.reexec import ToolchainStatus, toolchain_available
from src.ub_oracle.sharded_repro import (
    confirm_sharded_reproducibility,
    run_sharded,
    split_shards,
)


def _status() -> ToolchainStatus:
    return ToolchainStatus(cc=None, ubsan=False, targets=())


def _item(item_id: str, label: str, klass: str = "fake") -> GTItem:
    return GTItem(
        item_id=item_id,
        lang="rust",
        klass=klass,
        cwe="" if label == "equivalent" else "CWE-000",
        declared_label=label,
        c_src=f"/* {item_id} C */",
        target_src=f"// {item_id} Rust",
        inputs=(),
    )


def deterministic_fake_label(_h, item: GTItem) -> LabelEvidence:
    if item.item_id.endswith("slow"):
        time.sleep(0.03)
    return LabelEvidence(
        observed_label=item.declared_label,
        ub_trapped=item.declared_label == "divergent",
        c_out="",
        target_out=item.item_id,
        detail=f"detail excluded from stable bytes: {item.item_id}",
    )


def noisy_detail_fake_label(_h, item: GTItem) -> LabelEvidence:
    return LabelEvidence(
        observed_label=item.declared_label,
        ub_trapped=False,
        c_out="",
        target_out="",
        detail=f"nondeterministic worker detail {item.item_id}: {time.time_ns()}",
    )


def test_split_shards_validates_and_uses_stable_boundaries():
    items = [_item(str(i), "equivalent") for i in range(5)]

    assert [s.to_dict() for s in split_shards(items, shard_count=2)] == [
        {"index": 0, "start": 0, "end": 3, "n_items": 3},
        {"index": 1, "start": 3, "end": 5, "n_items": 2},
    ]
    assert [s.to_dict() for s in split_shards(items, shard_size=2)] == [
        {"index": 0, "start": 0, "end": 2, "n_items": 2},
        {"index": 1, "start": 2, "end": 4, "n_items": 2},
        {"index": 2, "start": 4, "end": 5, "n_items": 1},
    ]

    with pytest.raises(ValueError, match="exactly one"):
        split_shards(items)
    with pytest.raises(ValueError, match="exactly one"):
        split_shards(items, shard_size=1, shard_count=1)
    with pytest.raises(ValueError, match="empty"):
        split_shards([], shard_size=1)
    with pytest.raises(ValueError, match="shard_size"):
        split_shards(items, shard_size=0)
    with pytest.raises(ValueError, match="shard_count"):
        split_shards(items, shard_count=0)
    with pytest.raises(ValueError, match="exceed"):
        split_shards(items, shard_count=6)


def test_sharded_concatenation_matches_whole_verdict_hash():
    items = [
        _item("d-slow", "divergent", "div_by_zero"),
        _item("a-fast", "equivalent", "safe_add"),
        _item("c-slow", "equivalent", "safe_mul"),
        _item("b-fast", "divergent", "oob_read"),
    ]

    proof = confirm_sharded_reproducibility(
        items,
        shard_size=2,
        workers=2,
        status=_status(),
        label_fn=deterministic_fake_label,
    )

    assert proof.ok, proof.to_dict()
    assert proof.whole_hash == proof.concatenated_hash
    assert proof.concatenated_hash == proof.repeat_concatenated_hash
    assert proof.shard_hashes == proof.repeat_shard_hashes
    assert proof.shard_chain_hash == proof.repeat_shard_chain_hash
    assert proof.n_shards == 2


def test_partitioning_and_worker_count_do_not_change_the_merged_hash():
    items = [
        _item("03-slow", "divergent"),
        _item("01-fast", "equivalent"),
        _item("04-slow", "equivalent"),
        _item("02-fast", "divergent"),
    ]
    whole_hash = run_parallel(
        items,
        workers=1,
        status=_status(),
        label_fn=deterministic_fake_label,
    ).content_hash()

    hashes = {
        run_sharded(
            items,
            shard_size=1,
            workers=1,
            status=_status(),
            label_fn=deterministic_fake_label,
        ).concatenated_hash(),
        run_sharded(
            items,
            shard_size=2,
            workers=2,
            status=_status(),
            label_fn=deterministic_fake_label,
        ).concatenated_hash(),
        run_sharded(
            items,
            shard_count=1,
            workers=2,
            status=_status(),
            label_fn=deterministic_fake_label,
        ).concatenated_hash(),
        run_sharded(
            items,
            shard_count=3,
            workers=1,
            status=_status(),
            label_fn=deterministic_fake_label,
        ).concatenated_hash(),
    }

    assert hashes == {whole_hash}


def test_shard_bytes_exclude_nondeterministic_details():
    items = [_item("a", "equivalent"), _item("b", "divergent")]

    proof = confirm_sharded_reproducibility(
        items,
        shard_size=1,
        workers=2,
        status=_status(),
        label_fn=noisy_detail_fake_label,
    )

    assert proof.ok, proof.to_dict()
    assert proof.shard_hashes == proof.repeat_shard_hashes


def test_duplicate_item_keys_are_rejected_loudly():
    items = [_item("dup", "equivalent"), _item("dup", "divergent")]

    with pytest.raises(ValueError, match="duplicate"):
        run_sharded(
            items,
            shard_size=1,
            status=_status(),
            label_fn=deterministic_fake_label,
        )


_TC = toolchain_available()


@pytest.mark.skipif(
    not _TC.full_for("rust"),
    reason=f"needs clang+UBSan+rustc toolchain ({_TC})",
)
def test_sharded_reproducibility_matches_whole_on_real_compilers():
    rust_items = [item for item in enumerate_corpus(("rust",)) if item.lang == "rust"]
    sample = [
        next(item for item in rust_items if item.declared_label == "divergent"),
        next(item for item in rust_items if item.declared_label == "equivalent"),
    ]

    proof = confirm_sharded_reproducibility(
        sample,
        shard_size=1,
        workers=1,
        status=_TC,
    )

    assert proof.ok, proof.to_dict()
    assert proof.whole_hash == proof.concatenated_hash
