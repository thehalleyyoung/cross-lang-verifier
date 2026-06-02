"""Step 58 — thin VS Code extension.

A genuine VS Code extension (`vscode-extension/`) that surfaces
C→{Rust,Go,Swift} cross-language divergences **in-editor**. It is deliberately
*thin*: it shells out to the real `cross-lang-verify` CLI (the same oracle the
test-suite proves), parses its JSON, and turns each divergent translation unit
into a `vscode.Diagnostic` — it never re-implements any analysis, so it cannot
drift from the proven oracle.

This module is the machine-checkable claim for that extension: it compiles the
extension's TypeScript with the **real `tsc` compiler** against the **real
`@types/vscode`** API typings and confirms the JavaScript entry point is
produced, and it validates that `package.json` actually contributes the command.

Guarantees (proven against the real Node/TypeScript toolchain, not mocked):

* `confirm_vscode_extension()` runs `npm install` (once) and `npx tsc -p .` and
  requires a clean compile producing `out/extension.js`. Consistency-only
  (``available=False``) when Node/npm is not installed — never fabricated.
* The extension's `package.json` is checked to declare the `engines.vscode`
  constraint, the `crossLangVerifier.verify` command, and the `main` entry that
  the compile produces.
"""

from __future__ import annotations

import json
import shutil
import subprocess
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


@dataclass(frozen=True)
class VscodeExtReport:
    available: bool
    ok: bool
    manifest_ok: bool
    compiled: bool
    entry_built: bool
    detail: str


def confirm_vscode_extension(timeout: int = 300) -> VscodeExtReport:
    """Compile the real extension with the real `tsc` against `@types/vscode`.

    Consistency-only when Node/npm is absent.
    """
    if not _PKG.exists():
        return VscodeExtReport(False, False, False, False, False,
                               "vscode-extension/package.json missing")
    manifest_ok = _manifest_contributes_command()
    npm = _node_tools()
    if npm is None:
        return VscodeExtReport(
            available=False, ok=bool(manifest_ok), manifest_ok=manifest_ok,
            compiled=False, entry_built=False,
            detail="npm not installed; manifest validated, compile skipped",
        )

    log: List[str] = []
    if not (_EXT / "node_modules").exists():
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
    ok = manifest_ok and comp.returncode == 0 and entry_built
    if comp.returncode != 0:
        log.append((comp.stderr or comp.stdout).strip()[-200:])
    return VscodeExtReport(
        available=True, ok=bool(ok), manifest_ok=manifest_ok,
        compiled=(comp.returncode == 0), entry_built=entry_built,
        detail="; ".join(log) + f" entry={entry.name}:{entry_built}",
    )


if __name__ == "__main__":  # pragma: no cover
    rep = confirm_vscode_extension()
    print("vscode-extension:", rep.detail)
    print("=> ok" if rep.ok else "=> FAILED")
    raise SystemExit(0 if rep.ok else 1)
