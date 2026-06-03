from __future__ import annotations

import json
from pathlib import Path

from src.ub_oracle.github_action import confirm_github_action


def test_translation_equivalence_guard_runs_on_sample_repo(tmp_path):
    report = confirm_github_action(tmp_path)

    assert report.ok, report.checks
    assert report.metadata_ok
    assert report.script_syntax_ok
    assert report.sample_exit_code == 1
    assert report.sample_sarif_ok
    assert report.sample_outputs_ok
    assert report.operational_error_sarif_ok


def test_sample_action_sarif_contains_candidate_and_source_location(tmp_path):
    report = confirm_github_action(tmp_path)
    assert report.ok, report.checks

    sarif = tmp_path / "sample" / "guard.sarif"
    doc = json.loads(sarif.read_text(encoding="utf-8"))
    results = doc["runs"][0]["results"]
    assert any(result["ruleId"] == "signed_overflow" for result in results)
    assert any(result["level"] == "warning" for result in results)
    uris = [
        result["locations"][0]["physicalLocation"]["artifactLocation"]["uri"]
        for result in results
        if "physicalLocation" in result["locations"][0]
    ]
    assert "c/overflow.c" in uris


def test_sample_repo_workflow_is_a_real_consumer_fixture():
    root = Path(__file__).resolve().parents[1]
    workflow = (
        root
        / "examples"
        / "github_action_sample"
        / ".github"
        / "workflows"
        / "translation-equivalence-guard.yml"
    ).read_text(encoding="utf-8")

    assert "thehalleyyoung/cross-lang-verifier/.github/actions/translation-equivalence-guard@main" in workflow
    assert "paths:" in workflow
    assert "c/**" in workflow
    assert "rust/**" in workflow
    assert "github/codeql-action/upload-sarif@v3" in workflow
