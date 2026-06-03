"""Verification support for the Translation Equivalence Guard GitHub Action."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

_ROOT = Path(__file__).resolve().parents[2]
_ACTION_DIR = _ROOT / ".github" / "actions" / "translation-equivalence-guard"
_ACTION_YML = _ACTION_DIR / "action.yml"
_RUN_SH = _ACTION_DIR / "run.sh"
_EXAMPLE_WORKFLOW = (
    _ROOT / ".github" / "workflows" / "translation-equivalence-guard.example.yml"
)
_SAMPLE_DIR = _ROOT / "examples" / "github_action_sample"
_SAMPLE_WORKFLOW = (
    _SAMPLE_DIR / ".github" / "workflows" / "translation-equivalence-guard.yml"
)


@dataclass(frozen=True)
class GitHubActionReport:
    ok: bool
    metadata_ok: bool
    script_syntax_ok: bool
    sample_exit_code: int
    sample_sarif_ok: bool
    sample_outputs_ok: bool
    operational_error_sarif_ok: bool
    checks: tuple


def _text_has(path: Path, needles: List[str], checks: List[str]) -> bool:
    text = path.read_text(encoding="utf-8")
    missing = [needle for needle in needles if needle not in text]
    checks.append(f"{path.relative_to(_ROOT)} missing={missing}")
    return not missing


def _run_action(tmp_dir: Path, *, manifest: str, fail_on: str = "candidate") -> subprocess.CompletedProcess:
    sarif = tmp_dir / "guard.sarif"
    output = tmp_dir / "github_output.txt"
    env = dict(os.environ)
    env.update({
        "GITHUB_WORKSPACE": str(_ROOT),
        "GITHUB_OUTPUT": str(output),
        "CLV_WORKING_DIRECTORY": "examples/github_action_sample",
        "CLV_MANIFEST": manifest,
        "CLV_SARIF": str(sarif),
        "CLV_FAIL_ON": fail_on,
        "CLV_NO_CONFIRM": "true",
        "CLV_COLOR": "never",
        "CLV_PYTHON": sys.executable,
        "CLV_PYTHON_MODULE": "ub_oracle.cli",
        "PYTHONPATH": (
            f"{_ROOT / 'src'}{os.pathsep}{env['PYTHONPATH']}"
            if env.get("PYTHONPATH")
            else str(_ROOT / "src")
        ),
    })
    return subprocess.run(
        ["bash", str(_RUN_SH)],
        cwd=str(_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _load_sarif(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _sarif_ok(path: Path) -> bool:
    doc = _load_sarif(path)
    if not isinstance(doc, dict):
        return False
    return doc.get("version") == "2.1.0" and isinstance(doc.get("runs"), list)


def _sample_sarif_ok(path: Path) -> bool:
    doc = _load_sarif(path)
    if not isinstance(doc, dict) or doc.get("version") != "2.1.0":
        return False
    try:
        results = doc["runs"][0]["results"]
    except (KeyError, IndexError, TypeError):
        return False
    uris = []
    for result in results:
        locations = result.get("locations") or []
        if not locations or "physicalLocation" not in locations[0]:
            continue
        uris.append(
            locations[0]["physicalLocation"]["artifactLocation"]["uri"]
        )
    return (
        any(result.get("ruleId") == "signed_overflow" for result in results)
        and any(result.get("level") == "warning" for result in results)
        and "c/overflow.c" in uris
    )


def _outputs_ok(path: Path, *, expected_code: int) -> bool:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    out = dict(line.split("=", 1) for line in lines if "=" in line)
    return out.get("exit_code") == str(expected_code) and Path(out.get("sarif", "")).is_absolute()


def confirm_github_action(tmp_dir: Path) -> GitHubActionReport:
    """Run the checked action runner against the sample consumer repository.

    The confirmation is intentionally hermetic: it uses the same shell script the
    composite Action invokes, but points it at the in-tree Python module instead
    of installing from GitHub. The result still proves the published Action's
    behavior-critical path: manifest resolution, verifier invocation, SARIF
    emission, fail gating, and ``GITHUB_OUTPUT`` reporting.
    """
    checks: List[str] = []
    metadata_ok = all([
        _text_has(_ACTION_YML, [
            'using: "composite"',
            '${{ github.action_path }}/run.sh',
            "CLV_MANIFEST: ${{ inputs.manifest }}",
            "CLV_WORKING_DIRECTORY: ${{ inputs.working-directory }}",
            "CLV_EXTRA_ARGS: ${{ inputs.extra-args }}",
        ], checks),
        _text_has(_EXAMPLE_WORKFLOW, [
            "pull_request:",
            "paths:",
            "thehalleyyoung/cross-lang-verifier/.github/actions/translation-equivalence-guard@main",
            "github/codeql-action/upload-sarif@v3",
            "steps.guard.outputs.sarif",
        ], checks),
        _text_has(_SAMPLE_WORKFLOW, [
            "working-directory: examples/github_action_sample",
            "fail-on: candidate",
            "no-confirm: true",
            "github/codeql-action/upload-sarif@v3",
        ], checks),
    ])

    syntax = subprocess.run(
        ["bash", "-n", str(_RUN_SH)],
        capture_output=True,
        text=True,
    )
    script_syntax_ok = syntax.returncode == 0
    checks.append(f"run.sh syntax rc={syntax.returncode}")

    sample_tmp = tmp_dir / "sample"
    sample_tmp.mkdir()
    sample = _run_action(sample_tmp, manifest="units_manifest.json")
    sample_sarif = sample_tmp / "guard.sarif"
    sample_output = sample_tmp / "github_output.txt"
    sample_sarif_ok = _sample_sarif_ok(sample_sarif)
    sample_outputs_ok = _outputs_ok(sample_output, expected_code=1)
    checks.append(
        "sample rc="
        f"{sample.returncode} sarif={sample_sarif_ok} outputs={sample_outputs_ok}"
    )

    error_tmp = tmp_dir / "operational_error"
    error_tmp.mkdir()
    error = _run_action(error_tmp, manifest="missing.json")
    error_sarif = error_tmp / "guard.sarif"
    error_output = error_tmp / "github_output.txt"
    operational_error_sarif_ok = (
        error.returncode == 2
        and _sarif_ok(error_sarif)
        and _outputs_ok(error_output, expected_code=2)
    )
    checks.append(
        "missing-manifest rc="
        f"{error.returncode} sarif={_sarif_ok(error_sarif)}"
    )

    ok = (
        metadata_ok
        and script_syntax_ok
        and sample.returncode == 1
        and sample_sarif_ok
        and sample_outputs_ok
        and operational_error_sarif_ok
    )
    return GitHubActionReport(
        ok=ok,
        metadata_ok=metadata_ok,
        script_syntax_ok=script_syntax_ok,
        sample_exit_code=sample.returncode,
        sample_sarif_ok=sample_sarif_ok,
        sample_outputs_ok=sample_outputs_ok,
        operational_error_sarif_ok=operational_error_sarif_ok,
        checks=tuple(checks),
    )
