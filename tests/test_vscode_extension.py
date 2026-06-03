from __future__ import annotations

import pytest

from src.ub_oracle import vscode_ext as _vsx


def test_vscode_extension_manifest_wraps_lsp():
    manifest = _vsx._load_manifest()
    assert _vsx._manifest_contributes_command()
    assert _vsx._manifest_lsp_contract()
    assert manifest["contributes"]["configuration"]["properties"][
        "crossLangVerifier.lspModule"
    ]["default"] == "ub_oracle.lsp"
    assert "vscode-languageclient" in manifest["dependencies"]
    assert "workspaceContains:units_manifest.json" in manifest["activationEvents"]


@pytest.mark.skipif(_vsx._node_tools() is None, reason="npm not installed")
def test_vscode_extension_compiles_packages_and_references_lsp():
    report = _vsx.confirm_vscode_extension()
    assert report.available and report.ok, report.detail
    assert report.manifest_ok
    assert report.compiled and report.entry_built
    assert report.lsp_backed
    assert report.packaged and report.package_built
    assert report.package_contains_runtime
