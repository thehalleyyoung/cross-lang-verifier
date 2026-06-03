from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from src.ub_oracle.precommit import confirm_precommit_hook


def test_precommit_hook_runs_on_fixture_repo(tmp_path):
    report = confirm_precommit_hook(tmp_path)

    assert report.ok, report.checks
    assert report.metadata_ok
    assert report.fixture_config_ok
    assert report.git_fixture_ok
    assert report.divergent_exit_code == 1
    assert report.divergent_output_ok
    assert report.safe_exit_code == 0
    assert report.no_manifest_exit_code == 0
    assert report.missing_exit_code == 2
    assert report.aggregate_exit_code == 2


def test_precommit_hook_implicit_manifest_discovery_and_sarif(tmp_path):
    root = Path(__file__).resolve().parents[1]
    fixture = root / "examples" / "pre_commit_sample"
    sarif = tmp_path / "precommit.sarif"
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        f"{root / 'src'}{os.pathsep}{env['PYTHONPATH']}"
        if env.get("PYTHONPATH")
        else str(root / "src")
    )

    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "ub_oracle.precommit",
            "--repo-root",
            str(fixture),
            "--no-confirm",
            "--fail-on",
            "candidate",
            "--sarif",
            str(sarif),
            "c/overflow.c",
        ],
        cwd=str(fixture),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert run.returncode == 1
    assert "CANDIDATE" in run.stdout
    assert "c/overflow.c" in run.stdout
    doc = json.loads(sarif.read_text(encoding="utf-8"))
    assert doc["version"] == "2.1.0"
    assert any(
        result["ruleId"] == "signed_overflow"
        for result in doc["runs"][0]["results"]
    )
