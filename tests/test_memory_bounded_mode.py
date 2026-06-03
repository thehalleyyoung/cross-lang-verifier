"""Focused tests for Step 146's memory-bounded verification mode."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from src.ub_oracle import cli
from src.ub_oracle import memory_bound as mb
from src.ub_oracle.memory_bound import prove_memory_bounded_equivalence
from src.ub_oracle.reexec import ReexecHarness, ToolchainStatus, toolchain_available
from src.ub_oracle.verify import VerifyReport, VerifyVerdict


def _status() -> ToolchainStatus:
    return ToolchainStatus(cc=None, ubsan=False, targets=())


def test_rss_supervisor_marks_resource_exhaustion_without_semantic_trap():
    harness = ReexecHarness(
        _status(),
        max_rss_mb=16,
        rss_poll_interval=0.001,
    )

    outcome = harness._run([
        sys.executable,
        "-c",
        "import time; chunks=[bytearray(1024*1024) for _ in range(128)]; time.sleep(2)",
    ])

    assert harness.resource_exhausted
    assert outcome.resource_exhausted
    assert not outcome.ub_trapped
    assert not outcome.asan_trapped
    assert "resource limit exceeded" in outcome.stderr


def test_memory_bound_proof_detects_verdict_layer_drift(monkeypatch):
    def fake_verify(unit, harness=None, **_kwargs):
        if harness is not None:
            return VerifyReport(
                VerifyVerdict.UNKNOWN,
                unit,
                detail="bounded run changed the verdict layer",
            )
        return VerifyReport(
            VerifyVerdict.NO_DIVERGENCE_FOUND,
            unit,
            detail="unbounded verdict layer",
        )

    monkeypatch.setattr(mb, "verify_unit", fake_verify)
    monkeypatch.setattr(
        mb,
        "ReexecHarness",
        lambda *_args, **_kwargs: SimpleNamespace(
            peak_rss_kb=0,
            resource_exhausted=False,
            resource_exhaustions=[],
        ),
    )

    proof = prove_memory_bounded_equivalence(
        [{"name": "unit"}],
        max_rss_mb=256,
        status=_status(),
    )

    assert not proof.ok
    assert "bounded and unbounded verdict-layer signatures differ" in proof.mismatches


def test_memory_bound_proof_rejects_resource_exhaustion(monkeypatch):
    def fake_verify(unit, **_kwargs):
        return VerifyReport(
            VerifyVerdict.NO_DIVERGENCE_FOUND,
            unit,
            detail="same verdict layer",
        )

    monkeypatch.setattr(mb, "verify_unit", fake_verify)
    monkeypatch.setattr(
        mb,
        "ReexecHarness",
        lambda *_args, **_kwargs: SimpleNamespace(
            peak_rss_kb=1025,
            resource_exhausted=True,
            resource_exhaustions=["python: resource limit exceeded"],
        ),
    )

    proof = prove_memory_bounded_equivalence(
        [{"name": "unit"}],
        max_rss_mb=1,
        status=_status(),
    )

    assert not proof.ok
    assert "bounded run exhausted the RSS budget" in proof.mismatches
    assert proof.resource_exhaustions == ["python: resource limit exceeded"]


def test_cli_passes_memory_bound_to_verifier(monkeypatch, tmp_path, capsys):
    manifest = tmp_path / "units.json"
    manifest.write_text(json.dumps({"units": [{"name": "u"}]}), encoding="utf-8")
    seen = []

    def fake_verify(unit, harness=None, **_kwargs):
        seen.append(harness.max_rss_mb if harness is not None else None)
        return VerifyReport(
            VerifyVerdict.NO_DIVERGENCE_FOUND,
            unit,
            detail="checked classes: [fake]",
        )

    monkeypatch.setattr(cli, "toolchain_available", lambda: _status())
    monkeypatch.setattr(cli, "verify_unit", fake_verify)

    rc = cli.run([
        "--units", str(manifest),
        "--no-validate",
        "--format", "json",
        "--max-rss-mb", "64",
    ])

    assert rc == 0
    assert seen == [64]
    payload = json.loads(capsys.readouterr().out)
    assert payload["memory_bound"]["max_rss_mb"] == 64
    assert payload["memory_bound"]["resource_exhausted"] is False


_TC = toolchain_available()


@pytest.mark.skipif(
    not _TC.full,
    reason=f"needs C+UBSan+rustc toolchain ({_TC})",
)
def test_memory_bound_equivalence_against_real_compilers():
    with open("examples/readme_demo_units.json", "r", encoding="utf-8") as fh:
        units = json.load(fh)["units"]

    proof = prove_memory_bounded_equivalence(
        units,
        max_rss_mb=4096,
        status=_TC,
    )

    assert proof.ok, proof.to_dict()
    assert proof.total == 2
    assert proof.bounded_verdicts.get("divergent", 0) == 1
    assert proof.bounded_signature_hash == proof.unbounded_signature_hash
