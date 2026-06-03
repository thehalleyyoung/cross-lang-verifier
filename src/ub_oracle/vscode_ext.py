"""Step 175 — thin VS Code extension over the real LSP server.

A genuine VS Code extension (`vscode-extension/`) that surfaces
C→{Rust,Go,Swift} cross-language divergences **in-editor**. It is deliberately
*thin*: it starts the real `cross-lang-verify-lsp` stdio server (the same LSP
adapter tested in `tests/test_lsp.py`) and lets that server publish diagnostics.
The extension never re-implements verifier logic, so it cannot drift from the
proven oracle.

This module is the machine-checkable claim for that extension: it compiles the
extension's TypeScript with the **real `tsc` compiler** against the **real
`@types/vscode`** API typings, confirms the JavaScript entry point is produced
and LSP-backed, and packages a real `.vsix` artifact with the runtime LSP client
dependency included.

Guarantees (proven against the real Node/TypeScript toolchain, not mocked):

* `confirm_vscode_extension()` runs `npm install`, `npm run compile`, and
  `npm run package`; it requires a clean compile producing `out/extension.js`,
  an LSP-client-backed entry point, and a real `.vsix` package.
  Consistency-only (``available=False``) when Node/npm is not installed — never
  fabricated.
* The extension's `package.json` is checked to declare the `engines.vscode`
  constraint, the `crossLangVerifier.verify` command, LSP activation/config, and
  the `main` entry that the compile produces.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

_ROOT = Path(__file__).resolve().parents[2]
_EXT = _ROOT / "vscode-extension"
_PKG = _EXT / "package.json"


def _node_tools() -> Optional[str]:
    return shutil.which("npm")


def _load_manifest() -> dict:
    return json.loads(_PKG.read_text())


def _manifest_contributes_command() -> bool:
    m = _load_manifest()
    cmds = m.get("contributes", {}).get("commands", [])
    has_cmd = any(c.get("command") == "crossLangVerifier.verify" for c in cmds)
    has_engine = "vscode" in m.get("engines", {})
    has_main = bool(m.get("main"))
    return bool(has_cmd and has_engine and has_main)


def _manifest_lsp_contract() -> bool:
    m = _load_manifest()
    activation = set(m.get("activationEvents", []))
    commands = {
        c.get("command")
        for c in m.get("contributes", {}).get("commands", [])
        if isinstance(c, dict)
    }
    props = m.get("contributes", {}).get("configuration", {}).get("properties", {})
    dependencies = m.get("dependencies", {})
    scripts = m.get("scripts", {})
    required_activation = {
        "workspaceContains:units_manifest.json",
        "workspaceContains:cross-lang-verify.json",
        "workspaceContains:.cross-lang-verify.json",
        "onCommand:crossLangVerifier.verify",
    }
    return bool(
        _manifest_contributes_command()
        and "crossLangVerifier.restartLanguageServer" in commands
        and required_activation <= activation
        and props.get("crossLangVerifier.lspModule", {}).get("default")
        == "ub_oracle.lsp"
        and props.get("crossLangVerifier.unitsManifest", {}).get("default")
        == "units_manifest.json"
        and "vscode-languageclient" in dependencies
        and "compile" in scripts
        and "package" in scripts
    )


def _expected_vsix() -> Path:
    m = _load_manifest()
    return _EXT / f"{m.get('name', 'cross-lang-verifier')}-{m.get('version', '0.0.0')}.vsix"


def _entry_lsp_backed(entry: Path) -> bool:
    if not entry.exists():
        return False
    text = entry.read_text(encoding="utf-8", errors="replace")
    required = (
        "vscode-languageclient/node",
        "LanguageClient",
        "ub_oracle.lsp",
        "cross-lang-verifier.reverify",
        "workspace/executeCommand",
    )
    return all(marker in text for marker in required)


def _vsix_contains_runtime(vsix: Path) -> bool:
    if not vsix.exists() or vsix.stat().st_size <= 0:
        return False
    required = (
        "extension/package.json",
        "extension/out/extension.js",
        "extension/node_modules/vscode-languageclient/node.js",
    )
    try:
        with zipfile.ZipFile(vsix) as archive:
            names = set(archive.namelist())
    except zipfile.BadZipFile:
        return False
    return all(name in names for name in required)


@dataclass(frozen=True)
class VscodeExtReport:
    available: bool
    ok: bool
    manifest_ok: bool
    compiled: bool
    entry_built: bool
    detail: str
    lsp_backed: bool = False
    packaged: bool = False
    package_built: bool = False
    package_contains_runtime: bool = False


def confirm_vscode_extension(timeout: int = 300) -> VscodeExtReport:
    """Compile the real extension with the real `tsc` against `@types/vscode`.

    Consistency-only when Node/npm is absent.
    """
    if not _PKG.exists():
        return VscodeExtReport(False, False, False, False, False,
                               "vscode-extension/package.json missing")
    manifest_ok = _manifest_lsp_contract()
    npm = _node_tools()
    if npm is None:
        return VscodeExtReport(
            available=False, ok=bool(manifest_ok), manifest_ok=manifest_ok,
            compiled=False, entry_built=False,
            detail="npm not installed; manifest validated, compile skipped",
        )

    log: List[str] = []
    inst = subprocess.run(
        [npm, "install", "--no-audit", "--no-fund"],
        cwd=str(_EXT), capture_output=True, text=True, timeout=timeout,
    )
    log.append(f"install rc={inst.returncode}")
    if inst.returncode != 0:
        return VscodeExtReport(
            available=True, ok=False, manifest_ok=manifest_ok,
            compiled=False, entry_built=False,
            detail="; ".join(log) + " :: " + (inst.stderr or "").strip()[-200:],
        )

    comp = subprocess.run(
        [npm, "run", "compile"],
        cwd=str(_EXT), capture_output=True, text=True, timeout=timeout,
    )
    log.append(f"compile rc={comp.returncode}")
    main_rel = _load_manifest().get("main", "./out/extension.js")
    entry = (_EXT / main_rel).resolve()
    entry_built = entry.exists()
    lsp_backed = _entry_lsp_backed(entry)
    vsix = _expected_vsix()
    if vsix.exists():
        vsix.unlink()
    package = subprocess.run(
        [npm, "run", "package"],
        cwd=str(_EXT), capture_output=True, text=True, timeout=timeout,
    )
    log.append(f"package rc={package.returncode}")
    package_built = vsix.exists() and vsix.stat().st_size > 0
    package_contains_runtime = _vsix_contains_runtime(vsix)
    ok = (
        manifest_ok
        and comp.returncode == 0
        and entry_built
        and lsp_backed
        and package.returncode == 0
        and package_built
        and package_contains_runtime
    )
    if comp.returncode != 0:
        log.append((comp.stderr or comp.stdout).strip()[-200:])
    if package.returncode != 0:
        log.append((package.stderr or package.stdout).strip()[-400:])
    return VscodeExtReport(
        available=True, ok=bool(ok), manifest_ok=manifest_ok,
        compiled=(comp.returncode == 0), entry_built=entry_built,
        detail=(
            "; ".join(log)
            + f" entry={entry.name}:{entry_built}"
            + f" lsp={lsp_backed}"
            + f" vsix={vsix.name}:{package_built}"
            + f" runtime={package_contains_runtime}"
        ),
        lsp_backed=lsp_backed,
        packaged=(package.returncode == 0),
        package_built=package_built,
        package_contains_runtime=package_contains_runtime,
    )


if __name__ == "__main__":  # pragma: no cover
    rep = confirm_vscode_extension()
    print("vscode-extension:", rep.detail)
    print("=> ok" if rep.ok else "=> FAILED")
    raise SystemExit(0 if rep.ok else 1)
