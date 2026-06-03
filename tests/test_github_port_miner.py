from __future__ import annotations

import json
import os
import subprocess
import tempfile

import pytest

from src.ub_oracle import github_port_miner as miner
from src.ub_oracle.ir import assert_valid
from src.ub_oracle.reexec import NO_TOOLCHAIN_ENV, toolchain_available


def test_github_search_parser_ranks_probable_c_to_rust_ports():
    payload = {
        "items": [
            {
                "full_name": "example/noisy-rust-only",
                "html_url": "https://github.com/example/noisy-rust-only",
                "description": "A Rust-only command line utility",
                "stargazers_count": 9000,
                "topics": ["rust"],
            },
            {
                "full_name": "example/zlib-port",
                "html_url": "https://github.com/example/zlib-port",
                "description": "Rust implementation of a C zlib library",
                "stargazers_count": 120,
                "topics": ["rust", "zlib", "c-abi"],
            },
            {
                "full_name": "example/archived-port",
                "description": "Rust rewrite of C tools",
                "stargazers_count": 10_000,
                "topics": ["rust", "c"],
                "archived": True,
            },
        ]
    }

    candidates = miner.parse_github_search_response(payload)

    assert [c.owner_repo for c in candidates] == ["example/zlib-port"]
    assert candidates[0].score >= 70
    assert "zlib" in candidates[0].evidence


def test_seeded_candidates_are_distinct_ranked_real_repos():
    candidates = miner.seeded_candidates()
    repos = [c.owner_repo for c in candidates]
    scores = [c.score for c in candidates]
    assert scores == sorted(scores, reverse=True)
    assert {"uutils/coreutils", "trifectatechfoundation/sudo-rs",
            "trifectatechfoundation/zlib-rs"} <= set(repos)
    assert all(c.url.startswith("https://github.com/") for c in candidates)
    assert all(c.score >= 70 for c in candidates)


def test_github_port_samples_are_valid_and_verifier_backed():
    miner.validate_corpus_shape()
    doc = miner.results_document()
    assert doc["schema"] == miner.SCHEMA_VERSION
    assert doc["n_candidates"] >= 3
    assert doc["n_verified_samples"] == 3
    assert doc["n_source_families"] == 3
    assert doc["all_verdicts_match_expectation"]
    assert doc["by_symbolic_verdict"] == {
        "candidate": 2,
        "no_divergence_found": 1,
    }
    assert doc["by_divergence_class"] == {
        "div_by_zero": 1,
        "signed_overflow": 2,
    }
    for sample, case in zip(miner.SAMPLES, doc["samples"]):
        assert_valid(sample.unit, label=sample.sample_id)
        assert case["sample_id"] == sample.sample_id
        assert case["source_sha256"]
        assert case["rust_sha256"]
        assert case["observed_symbolic_verdict"] == sample.expected_symbolic_verdict


def test_github_port_results_json_is_byte_reproducible():
    old_mask = os.environ.pop(NO_TOOLCHAIN_ENV, None)
    try:
        ok, detail = miner.check_results()
        assert ok, detail
        on_disk = json.loads(miner.RESULTS_PATH.read_text(encoding="utf-8"))
        assert on_disk["content_hash"] == miner.results_document()["content_hash"]
    finally:
        if old_mask is not None:
            os.environ[NO_TOOLCHAIN_ENV] = old_mask


_TC = toolchain_available()


@pytest.mark.skipif(
    not (_TC.c_available and _TC.target_available("rust")),
    reason=f"needs C and rustc ({_TC})",
)
def test_github_port_sample_sources_compile_as_real_code():
    with tempfile.TemporaryDirectory() as tmp:
        for sample in miner.SAMPLES:
            c_obj = f"{tmp}/{sample.sample_id}.o"
            rust_lib = f"{tmp}/lib_{sample.sample_id.replace('-', '_')}.rlib"
            c_run = subprocess.run(
                [_TC.cc, "-c", str(sample.c_path), "-o", c_obj],
                capture_output=True,
                text=True,
                timeout=60,
            )
            assert c_run.returncode == 0, c_run.stderr
            rs_run = subprocess.run(
                [
                    _TC.rustc,
                    "--edition=2021",
                    "--crate-type",
                    "lib",
                    str(sample.rust_path),
                    "-o",
                    rust_lib,
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            assert rs_run.returncode == 0, rs_run.stderr


@pytest.mark.skipif(
    not _TC.full_for("rust"),
    reason=f"needs clang+UBSan+rustc toolchain ({_TC})",
)
def test_github_port_divergent_sample_confirms_against_real_compilers():
    sample = next(s for s in miner.SAMPLES if s.sample_id == "sudo-tty-rate")
    report = miner.confirm_sample(sample)
    assert report.verdict.value == "divergent"
    assert report.divergence is not None
    assert report.divergence.reexec is not None
    assert report.divergence.reexec.confirmed
