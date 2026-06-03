from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile

import pytest

from src.ub_oracle import c2rust_corpus as corpus
from src.ub_oracle.ir import assert_valid
from src.ub_oracle.reexec import NO_TOOLCHAIN_ENV, toolchain_available


_TC = toolchain_available()


def test_c2rust_corpus_has_a_dozen_distinct_libraries():
    corpus.validate_corpus_shape()
    assert len(corpus.CORPUS) == 12
    assert len({item.source_library for item in corpus.CORPUS}) == 12


def test_c2rust_corpus_units_are_valid_and_verifier_backed():
    doc = corpus.results_document()
    assert doc["schema"] == corpus.SCHEMA_VERSION
    assert doc["n_items"] == 12
    assert doc["n_source_libraries"] == 12
    assert doc["all_generated_by_c2rust_shape"]
    assert doc["all_verdicts_match_expectation"]
    assert doc["by_symbolic_verdict"]["candidate"] >= 1
    assert doc["by_symbolic_verdict"]["no_divergence_found"] >= 1
    for item, case in zip(corpus.CORPUS, doc["cases"]):
        assert_valid(item.unit, label=item.item_id)
        assert case["item_id"] == item.item_id
        assert case["source_sha256"]
        assert case["rust_sha256"]
        assert case["translator"] == "c2rust"
        assert case["observed_symbolic_verdict"] == item.expected_symbolic_verdict


def test_c2rust_corpus_results_json_is_byte_reproducible():
    old_mask = os.environ.pop(NO_TOOLCHAIN_ENV, None)
    try:
        ok, detail = corpus.check_results()
        assert ok, detail
        on_disk = json.loads(corpus.RESULTS_PATH.read_text(encoding="utf-8"))
        assert on_disk["content_hash"] == corpus.results_document()["content_hash"]
    finally:
        if old_mask is not None:
            os.environ[NO_TOOLCHAIN_ENV] = old_mask


@pytest.mark.skipif(
    not (_TC.c_available and _TC.target_available("rust")),
    reason=f"needs C and rustc ({_TC})",
)
def test_c2rust_corpus_sources_and_generated_rust_compile():
    with tempfile.TemporaryDirectory() as tmp:
        for item in corpus.CORPUS:
            c_obj = f"{tmp}/{item.item_id}.o"
            rust_lib = f"{tmp}/lib_{item.item_id.replace('-', '_')}.rlib"
            c_run = subprocess.run(
                [_TC.cc, "-c", str(item.c_path), "-o", c_obj],
                capture_output=True,
                text=True,
                timeout=60,
            )
            assert c_run.returncode == 0, c_run.stderr
            rs_run = subprocess.run(
                [_TC.rustc, "--edition=2021", "--crate-type", "lib",
                 str(item.rust_path), "-o", rust_lib],
                capture_output=True,
                text=True,
                timeout=60,
            )
            assert rs_run.returncode == 0, rs_run.stderr


@pytest.mark.skipif(corpus.c2rust_path() is None, reason="c2rust not installed")
def test_checked_in_rust_is_real_c2rust_output():
    comparison = corpus.compare_generated_to_c2rust()
    assert comparison["ok"], comparison
    assert comparison["c2rust_version"] == corpus.TRANSLATOR_VERSION
