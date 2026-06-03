from __future__ import annotations

import pytest

from experiments.cex_quality import run as cex_run
from experiments.cross_pair_matrix import run as matrix_run
from experiments.perf_curves import run as perf_run
from experiments.redteam import run as redteam_run
from experiments.ub_divergence import run as ub_run
from src.ub_oracle import regression_matrix as matrix
from src.ub_oracle.cache import (
    ToolchainMismatch,
    _tool_version,
    toolchain_provenance,
    validate_toolchain_provenance,
)
from src.ub_oracle.reexec import ReexecHarness, ToolchainStatus


def _fake_tool(tmp_path, name: str, version: str, *, go_style: bool = False) -> str:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if go_style:
        body = (
            "#!/bin/sh\n"
            "if [ \"$1\" = version ]; then\n"
            f"  echo '{version}'\n"
            "else\n"
            "  echo 'bad version flag' >&2\n"
            "  exit 2\n"
            "fi\n"
        )
    else:
        body = f"#!/bin/sh\necho '{version}'\n"
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o111)
    return str(path)


def _status(tmp_path, *, rust_version: str = "rustc 1.99.0-test") -> ToolchainStatus:
    return ToolchainStatus(
        cc=_fake_tool(tmp_path, "clang", "clang version 99.0.0-test"),
        ubsan=True,
        targets=(
            ("rust", _fake_tool(tmp_path, "rustc", rust_version)),
            ("go", _fake_tool(tmp_path, "go", "go version go1.99.0 test/os", go_style=True)),
        ),
        runners=(),
    )


def test_tool_version_uses_go_version_subcommand(tmp_path):
    go = _fake_tool(tmp_path, "go", "go version go1.99.0 test/os", go_style=True)
    cp = _fake_tool(tmp_path, "cp", "this should not be probed")

    assert _tool_version(go, tool_name="go") == "go version go1.99.0 test/os"
    assert _tool_version(go, tool_name="rust") == "unknown"
    assert _tool_version(cp, tool_name="wasm") == "lossless-copy-stager"


def test_toolchain_provenance_records_exact_versions(tmp_path):
    provenance = toolchain_provenance(_status(tmp_path))

    assert provenance["schema"] == "toolchain-provenance/v1"
    assert provenance["cc"]["version"] == "clang version 99.0.0-test"
    assert provenance["targets"]["rust"]["version"] == "rustc 1.99.0-test"
    assert provenance["targets"]["go"]["version"] == "go version go1.99.0 test/os"
    assert provenance["fingerprint"]["go"] == "go version go1.99.0 test/os"


def test_toolchain_validation_ignores_paths_but_rejects_version_changes(tmp_path):
    recorded = {
        "toolchain_provenance": toolchain_provenance(_status(tmp_path / "old"))
    }

    same_versions_different_paths = _status(tmp_path / "new")
    ok = validate_toolchain_provenance(recorded, same_versions_different_paths)
    assert ok.ok

    changed_rust = _status(tmp_path / "changed", rust_version="rustc 2.0.0-test")
    with pytest.raises(ToolchainMismatch, match="version mismatch"):
        validate_toolchain_provenance(recorded, changed_rust)


def test_strict_validation_rejects_missing_or_unknown_provenance(tmp_path):
    with pytest.raises(ToolchainMismatch, match="toolchain_provenance"):
        validate_toolchain_provenance({}, _status(tmp_path))

    with pytest.raises(ToolchainMismatch, match="unknown"):
        validate_toolchain_provenance(
            {"toolchain_fingerprint": {"cc": "unknown"}},
            _status(tmp_path),
        )


def test_matrix_confirmations_include_replayable_toolchain_provenance(tmp_path, monkeypatch):
    status = _status(tmp_path)
    monkeypatch.setattr(matrix, "_sorted_oracles", lambda: [])
    artifact = matrix.confirm_matrix(ReexecHarness(status))

    assert artifact["schema"] == "cross-pair-confirmations/v2"
    assert artifact["toolchain_fingerprint"]["cc"] == "clang version 99.0.0-test"
    assert artifact["toolchain_provenance"]["targets"]["go"]["version"].startswith("go version ")
    assert validate_toolchain_provenance(artifact, status).ok


def test_ub_confirm_check_returns_nonzero_on_toolchain_mismatch(tmp_path, monkeypatch):
    path = tmp_path / "confirmations.json"
    path.write_text(
        '{"toolchain_fingerprint": {"cc": "clang version impossible"}}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(ub_run, "CONFIRMATIONS_PATH", str(path))

    assert ub_run.main(["--confirm", "--check"]) == 1


@pytest.mark.parametrize(
    ("module", "path_attr", "argv", "setup_attr"),
    [
        (matrix_run, "CONFIRMATIONS_PATH", ["--confirm", "--check"], "build_matrix"),
        (cex_run, "MINIMIZED_PATH", ["--minimize", "--check"], "build_baseline"),
        (redteam_run, "ATTACK_PATH", ["--attack", "--check"], "_deterministic_grid"),
        (perf_run, "TIMINGS_PATH", ["--measure", "--check"], "deterministic_grid"),
    ],
)
def test_all_environment_artifact_checks_fail_on_mismatch(
    tmp_path, monkeypatch, module, path_attr, argv, setup_attr
):
    path = tmp_path / "artifact.json"
    path.write_text(
        '{"toolchain_fingerprint": {"cc": "clang version impossible"}}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(module, path_attr, str(path))
    if module is perf_run:
        monkeypatch.setattr(module.perf, setup_attr, lambda: {})
    elif module is matrix_run:
        monkeypatch.setattr(module.M, setup_attr, lambda: {})
    else:
        monkeypatch.setattr(module, setup_attr, lambda: {})

    assert module.main(argv) == 1
