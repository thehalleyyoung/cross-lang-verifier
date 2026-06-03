from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.ub_oracle import cli as _cli
from src.ub_oracle import mechanized_soundness as ms
from src.ub_oracle.plugin import OracleResult, OracleVerdict
from src.ub_oracle.reexec import toolchain_available
from src.ub_oracle.replay import Counterexample
from src.ub_oracle.verify import VerifyReport, VerifyVerdict


_LAKE_PRESENT = ms._lake_binary() is not None
_STATUS = toolchain_available()
_ROOT = Path(__file__).resolve().parents[1]


def test_verified_checker_source_binds_to_proven_symbols():
    source = (_ROOT / ms.CHECKER_SOURCE).read_text(encoding="utf-8")

    assert "import ProductSoundness" in source
    assert "productViolated" in source
    assert "oracle_sound" in source
    assert "confirmed" not in source


@pytest.mark.skipif(not _LAKE_PRESENT, reason="Lean/Lake not installed")
def test_lake_builds_verified_checker_and_rejects_false_divergent_claim():
    build = ms.build_verified_checker()
    assert build.ok, ms.render_verified_checker_build(build)

    accepted = ms.run_verified_checker(
        "divergent", True, True, True, build=False)
    assert accepted.ok, accepted.stderr
    assert '"kernel_theorem":"oracle_sound"' in accepted.stdout

    rejected = ms.run_verified_checker(
        "divergent", False, True, True, build=False)
    assert rejected.available
    assert not rejected.accepted
    assert rejected.exit_code == 1
    assert '"accepted":false' in rejected.stdout


def test_cli_verified_check_noops_when_there_is_no_positive_claim(tmp_path, capsys):
    manifest = tmp_path / "safe.json"
    manifest.write_text(json.dumps({"units": [{
        "name": "safe_add_zero",
        "kind": "binop_const",
        "op": "add",
        "const": 0,
        "width": 32,
        "var": "x",
        "signed": True,
        "probe": "signed_overflow",
        "source_lang": "c",
        "target_lang": "rust",
    }]}), encoding="utf-8")

    rc = _cli.run([
        "--units", str(manifest),
        "--no-confirm",
        "--verified-check",
        "--format", "json",
        "--fail-on", "unknown",
    ])
    out = capsys.readouterr().out
    payload = json.loads(out)

    assert rc == 0
    assert payload["verified_check"]["status"] == "no_positive_claims"
    assert payload["verified_check"]["checked"] == 0


def test_verified_check_skips_positive_claims_outside_source_ub_scope():
    report = VerifyReport(
        VerifyVerdict.DIVERGENT,
        {"source_lang": "go", "target_lang": "rust"},
        divergence=OracleResult(
            OracleVerdict.DIVERGENT,
            "defined_divergence",
            counterexample=Counterexample(
                divergence_class="defined_divergence",
                source_lang="go",
                target_lang="rust",
                inputs={},
                source_snippet="package main\nfunc main(){}\n",
                target_snippet="fn main() {}\n",
                source_definedness="defined",
            ),
        ),
    )

    summary, err = _cli._run_verified_checks([report])

    assert err is None
    assert summary["status"] == "no_positive_claims"
    assert summary["checked"] == 0
    assert summary["skipped_non_ub_rooted"] == 1


@pytest.mark.skipif(
    not (_LAKE_PRESENT and _STATUS.full_for("rust")),
    reason="needs Lean/Lake plus C+UBSan+rustc",
)
def test_cli_verified_check_accepts_confirmed_readme_demo(capsys):
    rc = _cli.run([
        "--units", str(_ROOT / "examples" / "readme_demo_units.json"),
        "--verified-check",
        "--color", "never",
        "--fail-on", "unknown",
    ])
    out = capsys.readouterr().out

    assert rc == 0
    assert "DIVERGENT" in out
    assert (
        "verified-check: accepted (1/1 source-UB claim(s); "
        "skipped_non_ub=0; theorem=oracle_sound)"
    ) in out
