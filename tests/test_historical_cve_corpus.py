from __future__ import annotations

import re
import subprocess

import pytest

from src.ub_oracle import cve_corpus as cve
from src.ub_oracle.reexec import ReexecHarness, toolchain_available


_TC = toolchain_available()


def test_historical_cve_manifest_is_large_unique_and_honestly_scoped():
    cases = cve.historical_cve_cases()
    assert len(cases) >= 50
    assert len({case.cve_id for case in cases}) == len(cases)
    assert all(re.fullmatch(r"CVE-\d{4}-\d{4,}", case.cve_id) for case in cases)

    templates = {template.template_id: template for template in cve.HISTORICAL_TEMPLATES}
    assert len(templates) >= 8
    for case in cases:
        template = templates[case.replay_template]
        assert case.nvd_cwe == template.cwe
        assert "not original vendor source" in case.replay_scope
        assert case.nvd_query.endswith(f"cweId={case.nvd_cwe}")
        assert {"rust", "go"}.issubset(set(template.langs))

    rows = cve.historical_coverage_table()
    assert len(rows) == len(cases)
    assert all(
        set(row) >= {"cve_id", "cwe", "template", "title", "langs", "scope", "nvd_query"}
        for row in rows
    )


def test_historical_reproduction_bundles_lint_for_every_cve_and_pair(tmp_path):
    check = cve.check_historical_reproduction_bundles(
        str(tmp_path), langs=("rust", "go"), execute=False)
    assert check.generated == len(cve.HISTORICAL_CVE_CASES) * 2
    assert check.linted == check.generated
    assert check.ok, check.failures[:3]

    sample = next(tmp_path.glob("*.sh"))
    text = sample.read_text()
    assert "weakness-class replay" in text
    assert "not original vendor source" in text


@pytest.mark.skipif(not _TC.full_for("rust"), reason="C/UBSan/rust unavailable")
def test_each_historical_replay_template_confirms_against_real_rust():
    harness = ReexecHarness(_TC)
    for template in cve.HISTORICAL_TEMPLATES:
        target = template.target_for("rust")
        assert target is not None
        res = harness.confirm_trap_vs_defined(
            template.c_src,
            target,
            list(template.witness_inputs),
            template.divergence_class,
            "rust",
        )
        assert res.available, res.reason
        assert res.confirmed, f"{template.template_id}: {res.reason}"

        safe = harness.confirm_trap_vs_defined(
            template.c_src,
            target,
            list(template.safe_inputs),
            template.divergence_class,
            "rust",
        )
        assert safe.available, safe.reason
        assert safe.rust_defined, safe.reason
        assert not safe.ub_reachable
        assert not safe.confirmed


@pytest.mark.skipif(not _TC.full_for("go"), reason="C/UBSan/go unavailable")
def test_each_historical_replay_template_confirms_against_real_go():
    harness = ReexecHarness(_TC)
    for template in cve.HISTORICAL_TEMPLATES:
        target = template.target_for("go")
        assert target is not None
        res = harness.confirm_trap_vs_defined(
            template.c_src,
            target,
            list(template.witness_inputs),
            template.divergence_class,
            "go",
        )
        assert res.available, res.reason
        assert res.confirmed, f"{template.template_id}: {res.reason}"


@pytest.mark.skipif(not _TC.full_for("rust"), reason="C/UBSan/rust unavailable")
def test_representative_historical_bundle_executes_end_to_end(tmp_path):
    representatives = []
    seen = set()
    for case in cve.HISTORICAL_CVE_CASES:
        if case.replay_template in seen:
            continue
        seen.add(case.replay_template)
        representatives.append(case)

    for case in representatives:
        script = tmp_path / f"{case.cve_id}-rust.sh"
        script.write_text(cve.historical_reproduction_bundle(case, "rust"))
        script.chmod(0o755)
        lint = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
        assert lint.returncode == 0, lint.stderr
        run = subprocess.run(["bash", str(script)], capture_output=True, text=True, timeout=90)
        assert run.returncode == 0, (case.cve_id, run.stdout, run.stderr)
