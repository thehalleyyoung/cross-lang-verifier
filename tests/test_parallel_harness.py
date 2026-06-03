"""Focused tests for Step 143's deterministic parallel harness."""

from __future__ import annotations

import time

import pytest

from src.ub_oracle.ground_truth import GTItem, LabelEvidence, enumerate_corpus
from src.ub_oracle.parallel_harness import (
    confirm_parallel_determinism,
    run_parallel,
)
from src.ub_oracle.reexec import ToolchainStatus, toolchain_available


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
    # Force completion order to differ from input order in a process pool.
    if item.item_id.endswith("slow"):
        time.sleep(0.05)
    return LabelEvidence(
        observed_label=item.declared_label,
        ub_trapped=item.declared_label == "divergent",
        c_out="",
        target_out=item.item_id,
        detail=f"detail that must not affect the hash: {item.item_id}",
    )


def noisy_detail_fake_label(_h, item: GTItem) -> LabelEvidence:
    return LabelEvidence(
        observed_label=item.declared_label,
        ub_trapped=False,
        c_out="",
        target_out="",
        detail=f"worker-local timing/detail for {item.item_id}: {time.time_ns()}",
    )


def test_parallel_merge_is_in_input_order_but_hash_is_verdict_layer_only():
    items = [
        _item("01-slow", "divergent", "div_by_zero"),
        _item("02-fast", "equivalent", "safe_add"),
        _item("03-slow", "equivalent", "safe_mul"),
        _item("04-fast", "divergent", "oob_read"),
    ]

    sequential = run_parallel(
        items, workers=1, status=_status(), label_fn=deterministic_fake_label)
    parallel = run_parallel(
        items, workers=2, status=_status(), label_fn=deterministic_fake_label)

    assert [r.item_id for r in parallel.results] == [i.item_id for i in items]
    assert parallel.verdict_layer() == sequential.verdict_layer()
    assert parallel.content_hash() == sequential.content_hash()
    assert parallel.decided == 4
    assert parallel.faithful


def test_parallel_content_hash_ignores_nonverdict_details():
    items = [_item("a", "equivalent"), _item("b", "divergent")]

    first = run_parallel(
        items, workers=2, status=_status(), label_fn=noisy_detail_fake_label)
    second = run_parallel(
        items, workers=2, status=_status(), label_fn=noisy_detail_fake_label)

    assert [r.detail for r in first.results] != [r.detail for r in second.results]
    assert first.verdict_layer() == second.verdict_layer()
    assert first.content_hash() == second.content_hash()


def test_parallel_determinism_proof_across_worker_counts():
    items = [
        _item("a-slow", "divergent"),
        _item("b-fast", "equivalent"),
        _item("c-slow", "equivalent"),
    ]

    proof = confirm_parallel_determinism(
        items,
        workers=(1, 2, 3),
        status=_status(),
        label_fn=deterministic_fake_label,
    )

    assert proof.available
    assert proof.ok, proof.mismatches
    assert len(set(proof.content_hashes.values())) == 1
    assert proof.n_items == 3


_TC = toolchain_available()


@pytest.mark.skipif(
    not _TC.full_for("rust"),
    reason=f"needs clang+UBSan+rustc toolchain ({_TC})",
)
def test_parallel_harness_matches_sequential_on_real_compilers():
    rust_items = [item for item in enumerate_corpus(("rust",))
                  if item.lang == "rust"]
    sample = [
        next(item for item in rust_items if item.declared_label == "divergent"),
        next(item for item in rust_items if item.declared_label == "equivalent"),
    ]

    proof = confirm_parallel_determinism(
        sample,
        workers=(1, 2),
        status=_TC,
    )

    assert proof.ok, proof.to_dict()
    assert proof.n_items == 2
