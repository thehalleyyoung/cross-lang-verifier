from __future__ import annotations

import json
import subprocess

import pytest

from src.ub_oracle import bug_regression_corpus as regressions
from src.ub_oracle import divergence_findings as findings
from src.ub_oracle.reexec import toolchain_available


def test_bug_regression_manifest_tracks_every_finding_and_freezes_sources():
    ok, detail = regressions.check_results()
    assert ok, detail
    on_disk = json.loads(regressions.RESULTS_PATH.read_text(encoding="utf-8"))
    assert on_disk == regressions.results_document()

    finding_ids = {rec.finding_id for rec in findings.finding_records()}
    regression_ids = {entry["finding_id"] for entry in on_disk["regressions"]}
    assert regression_ids == finding_ids
    assert on_disk["n_regressions"] == len(finding_ids) == 1
    assert "not upstream CVEs" in on_disk["evidence_policy"]

    by_id = {rec.finding_id: rec for rec in findings.finding_records()}
    for entry in on_disk["regressions"]:
        rec = by_id[entry["finding_id"]]
        assert entry["c_sha256"] == rec.c_sha256
        assert entry["rust_sha256"] == rec.rust_sha256
        assert entry["witness_input"] == list(rec.witness_input)
        assert entry["safe_input"] == list(rec.safe_input)
        assert entry["bundle_path"] == f"docs/repro/{rec.finding_id}.sh"


def test_bug_regression_docs_and_bundles_are_read_only_and_fresh():
    bundle = regressions._ROOT / f"docs/repro/{findings.FINDING_ID}.sh"
    before = bundle.read_text(encoding="utf-8")

    ok, detail = regressions.check_docs()
    assert ok, detail
    ok, detail = regressions.validate_bundles()
    assert ok, detail
    ok, detail = regressions.validate_frozen_sources()
    assert ok, detail

    after = bundle.read_text(encoding="utf-8")
    assert after == before

    text = regressions.DOC_PATH.read_text(encoding="utf-8")
    assert "Frozen bug-regression corpus" in text
    assert findings.FINDING_ID in text
    assert "not upstream CVEs" in text
    assert "`make bug-regression-check`" in text
    syntax = subprocess.run(["bash", "-n", str(bundle)], capture_output=True, text=True)
    assert syntax.returncode == 0, syntax.stderr


_TC = toolchain_available()


@pytest.mark.skipif(
    not _TC.full_for("rust"),
    reason=f"needs clang+UBSan+rustc toolchain ({_TC})",
)
def test_bug_regression_corpus_replays_every_frozen_finding_live():
    rep = regressions.confirm_regressions()
    assert rep.available and rep.ok, rep.detail
    assert rep.n_confirmed == rep.n_regressions == 1
    conf = rep.confirmations[0]
    assert conf.confirmed
    assert conf.ub_reachable
    assert conf.target_defined
    assert conf.safe_silent
    assert conf.bundle_path == f"docs/repro/{findings.FINDING_ID}.sh"

    out = subprocess.run(
        ["bash", str(regressions._ROOT / conf.bundle_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    combined = out.stdout + out.stderr
    assert out.returncode == 0, combined
    assert "signed integer overflow" in combined
    assert "== Safe input control ==" in combined
