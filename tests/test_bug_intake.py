from __future__ import annotations

import json
import subprocess

import pytest

from src.ub_oracle import bug_intake as intake
from src.ub_oracle import divergence_findings as findings
from src.ub_oracle.reexec import toolchain_available


def test_bug_intake_template_sample_and_results_are_byte_fresh():
    intake.write_artifacts()
    ok, detail = intake.check_results()
    assert ok, detail
    ok, detail = intake.check_docs()
    assert ok, detail

    template = json.loads(intake.TEMPLATE_PATH.read_text(encoding="utf-8"))
    assert template["schema"] == intake.SCHEMA_VERSION
    assert "sandbox" in template["sandbox_warning"]
    for field in intake.REQUIRED_FIELDS:
        assert field in template["required_fields"]

    results = json.loads(intake.RESULTS_PATH.read_text(encoding="utf-8"))
    assert results == intake.results_document()
    assert results["valid_sample_submissions"] == 1
    assert "never serialized" in " ".join(results["intake_contract"])


def test_bug_intake_sample_validates_hashes_and_stays_candidate_only():
    submission = intake.load_submission()
    validation = intake.validate_submission(submission)
    rec = findings.finding_records()[0]

    assert validation.valid, validation.errors
    assert validation.submission_id == f"INTAKE-{findings.FINDING_ID}"
    assert validation.c_sha256 == rec.c_sha256
    assert validation.target_sha256 == rec.rust_sha256
    assert submission["witness_input"] == list(rec.witness_input)
    assert submission["safe_input"] == list(rec.safe_input)
    assert "candidate" in submission["claim_scope"]

    tampered = dict(submission)
    tampered["source"] = dict(submission["source"])
    tampered["source"]["code"] = str(tampered["source"]["code"]) + "\n"
    bad = intake.validate_submission(tampered)
    assert not bad.valid
    assert any("source.sha256" in err for err in bad.errors)


def test_bug_intake_docs_and_bundle_are_checked():
    bundle = intake._ROOT / f"docs/repro/INTAKE-{findings.FINDING_ID}.sh"
    assert bundle.exists()
    syntax = subprocess.run(["bash", "-n", str(bundle)], capture_output=True, text=True)
    assert syntax.returncode == 0, syntax.stderr

    text = intake.DOC_PATH.read_text(encoding="utf-8")
    assert "Bug-bounty-style divergence intake" in text
    assert "Sandbox warning" in text
    assert "`make bug-intake-check`" in text
    assert "not counted as real findings" in text


_TC = toolchain_available()


@pytest.mark.skipif(
    not _TC.full_for("rust"),
    reason=f"needs clang+UBSan+rustc toolchain ({_TC})",
)
def test_bug_intake_sample_replays_against_real_compilers():
    rep = intake.confirm_submission(write_repro_bundle=True)
    assert rep.available and rep.ok, rep.detail
    assert rep.validation_ok
    assert rep.witness_confirmed
    assert rep.ub_reachable
    assert rep.target_defined
    assert rep.safe_silent

    out = subprocess.run(
        ["bash", str(intake._ROOT / rep.bundle_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    combined = out.stdout + out.stderr
    assert out.returncode == 0, combined
    assert "signed integer overflow" in combined
    assert "== Safe input control ==" in combined
