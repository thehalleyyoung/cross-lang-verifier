"""Ecosystem flywheel: a frozen public API with a SemVer guard, plus a real
`cargo` subcommand integration.

Step 100 of the roadmap asks for the pieces that make a tool *dependable and
integrated*: a stable plugin/public API with SemVer guarantees, and ecosystem
integrations (the repo already ships a VS Code extension, a composite GitHub
Action and CI templates).  This module adds the two missing, *provable* pieces:

  1. **A frozen v1 public-API surface + SemVer guard.**  ``PUBLIC_API_V1`` is the
     committed set of symbols downstream code may depend on.  ``confirm_api_surface``
     imports the package and asserts every promised symbol is still exported in
     ``__all__`` and is a live, importable object — so any *removal or rename*
     (a breaking change under SemVer) is caught mechanically before release.  The
     surface is also snapshotted to ``integrations/api_surface_v1.json`` so a diff
     review shows exactly what the stability promise covers.

  2. **Real `cargo` subcommands.**  ``cargo-cross-lang-verify`` and the shorter
     ``cargo-cross-verify`` are thin shims following cargo's ``cargo-<name>``
     discovery convention; dropping them on ``PATH`` makes both
     ``cargo cross-lang-verify --units …`` and ``cargo cross-verify --units …``
     work.  They forward to the proven CLI, so they cannot drift from the oracle.
     ``confirm_ecosystem`` runs both shims *end to end* against a real manifest and
     requires each to emit the same oracle JSON the library produces.

Everything here is pure stdlib and verified against real execution.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
_INTEGRATIONS = _ROOT / "integrations"
_CARGO_SHIM = _INTEGRATIONS / "cargo-cross-lang-verify"
_CARGO_VERIFY_SHIM = _INTEGRATIONS / "cargo-cross-verify"
_API_SNAPSHOT = _INTEGRATIONS / "api_surface_v1.json"
_EXAMPLE_MANIFEST = _ROOT / "examples" / "units_manifest.json"
_CARGO_SUBCOMMANDS: Tuple[Tuple[str, Path], ...] = (
    ("cross-lang-verify", _CARGO_SHIM),
    ("cross-verify", _CARGO_VERIFY_SHIM),
)

# --------------------------------------------------------------------------
# The frozen v1 public API.  Removing or renaming any of these is a *breaking*
# change under SemVer and must bump the major version.  Keep this list in sync
# with the documented surface; the guard below enforces it can never silently
# shrink.
# --------------------------------------------------------------------------
PUBLIC_API_V1: Tuple[str, ...] = (
    # core verdict pipeline
    "verify_unit",
    "VerifyVerdict",
    "VerifyReport",
    "applicable_oracles",
    # divergence catalogue
    "DivergenceClass",
    "Severity",
    "Definedness",
    "DivergenceEntry",
    "CATALOGUE",
    "c_ub_classes",
    "entry_for",
    # oracle plugin SPI
    "DivergenceOracle",
    "OracleVerdict",
    "OracleResult",
    "register",
    "get_oracle",
    "get_oracle_for",
    "oracles_for",
    "language_pairs",
    "list_oracles",
    # ground-truth re-execution
    "ReexecResult",
    "ReexecHarness",
    "toolchain_available",
    "ToolchainStatus",
    # IR contract
    "validate_unit",
    "is_valid",
    "assert_valid",
    "KNOWN_KINDS",
    "IRError",
    "IRValidationError",
    # reporting
    "aggregate_reports",
    "to_sarif",
    "pair_of",
    # traceability
    "Claim",
    "CLAIMS",
    "verify_traceability",
    "claim_ids",
)

SEMVER = "1.0.0"

_CARGO_SHIM_BODY = """\
#!/usr/bin/env bash
# {shim_name} -- a `cargo` subcommand wrapper for the cross-language divergence
# oracle.  Cargo discovers any executable named `cargo-<name>` on PATH and exposes
# it as `cargo <name>`, so installing this shim (chmod +x and place on PATH)
# enables:
#
#     cargo {subcommand} --units units_manifest.json --format json
#
# It is a thin forwarder to the proven CLI; it adds no semantics of its own and
# therefore cannot drift from the oracle.
set -euo pipefail

# Cargo invokes `cargo-<name> <name> <args...>`; drop the leading subcommand
# token so the CLI sees only its real flags.
if [[ "${1:-}" == "{subcommand}" ]]; then
  shift
fi

# Prefer an installed console script; fall back to running from a source tree.
if command -v cross-lang-verify >/dev/null 2>&1; then
  exec cross-lang-verify "$@"
fi

PYTHON="${CROSS_LANG_VERIFY_PYTHON:-python3}"
SRC="${CROSS_LANG_VERIFY_SRC:-}"
if [[ -n "${SRC}" ]]; then
  PYTHONPATH="${SRC}${PYTHONPATH:+:${PYTHONPATH}}" exec "${PYTHON}" -m ub_oracle.cli "$@"
fi
exec "${PYTHON}" -m ub_oracle.cli "$@"
"""


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------
def _import_pkg():
    try:
        import ub_oracle as _pkg
    except ModuleNotFoundError:  # source-tree import via `src.` package
        import src.ub_oracle as _pkg  # type: ignore
    return _pkg


def _current_surface() -> Dict[str, object]:
    ub_oracle = _import_pkg()

    exported = set(getattr(ub_oracle, "__all__", ()))
    present: Dict[str, bool] = {}
    for name in PUBLIC_API_V1:
        obj = getattr(ub_oracle, name, None)
        present[name] = (name in exported) and (obj is not None)
    return {
        "schema": "cross-lang-verifier/public-api/1",
        "semver": SEMVER,
        "symbols": list(PUBLIC_API_V1),
        "exported": sorted(exported),
    }


def _surface_text() -> str:
    return json.dumps(_current_surface(), indent=2, sort_keys=True) + "\n"


def _cargo_shim_body(subcommand: str) -> str:
    return (
        _CARGO_SHIM_BODY
        .replace("{shim_name}", f"cargo-{subcommand}")
        .replace("{subcommand}", subcommand)
    )


def generate_artifacts(integrations_dir: Path = _INTEGRATIONS) -> Tuple[Path, Path]:
    integrations_dir.mkdir(parents=True, exist_ok=True)
    cargo_shim = integrations_dir / "cargo-cross-lang-verify"
    cargo_verify_shim = integrations_dir / "cargo-cross-verify"
    api_snapshot = integrations_dir / "api_surface_v1.json"
    for subcommand, path in (
        ("cross-lang-verify", cargo_shim),
        ("cross-verify", cargo_verify_shim),
    ):
        path.write_text(_cargo_shim_body(subcommand), encoding="utf-8")
        os.chmod(path, 0o755)
    api_snapshot.write_text(_surface_text(), encoding="utf-8")
    return cargo_shim, api_snapshot


def api_snapshot_fresh(path: Path = _API_SNAPSHOT) -> Tuple[bool, str]:
    """Read-only guard that the committed v1 API snapshot is up to date."""
    expected = _surface_text()
    if not path.exists():
        return False, f"{path} is missing"
    actual = path.read_text(encoding="utf-8")
    if actual != expected:
        return False, (
            f"{path} is stale; regenerate with "
            "python -m src.ub_oracle.ecosystem"
        )
    return True, "OK"


# --------------------------------------------------------------------------
# Confirmation
# --------------------------------------------------------------------------
@dataclass
class EcosystemReport:
    ok: bool
    api_ok: bool
    api_snapshot_ok: bool
    missing_symbols: Tuple[str, ...]
    shim_syntax_ok: bool
    shim_ran: bool
    shim_matches_library: bool
    shim_results: Dict[str, Dict[str, bool]] = field(default_factory=dict)
    semver: str = SEMVER
    checks: Tuple[str, ...] = field(default_factory=tuple)
    detail: str = ""


def _run_cargo_shim(
    subcommand: str,
    path: Path,
    cli_args: List[str],
    env: Dict[str, str],
    expected_json: Dict[str, object],
) -> Dict[str, bool]:
    syntax = subprocess.run(
        ["bash", "-n", str(path)],
        capture_output=True, text=True,
    )
    syntax_ok = syntax.returncode == 0

    ran = False
    matches = False
    if syntax_ok:
        run = subprocess.run(
            ["bash", str(path), subcommand, *cli_args],
            capture_output=True, text=True, env=env, timeout=120,
        )
        ran = run.returncode == 0
        if ran:
            matches = json.loads(run.stdout) == expected_json

    cargo_available = shutil.which("cargo", path=env.get("PATH")) is not None
    cargo_ran = False
    cargo_matches = False
    if cargo_available and syntax_ok:
        cargo_env = dict(env)
        cargo_env["PATH"] = f"{path.parent}{os.pathsep}{cargo_env.get('PATH', '')}"
        cargo = subprocess.run(
            ["cargo", subcommand, *cli_args],
            cwd=str(_ROOT), capture_output=True, text=True,
            env=cargo_env, timeout=120,
        )
        cargo_ran = cargo.returncode == 0
        if cargo_ran:
            cargo_matches = json.loads(cargo.stdout) == expected_json

    return {
        "syntax_ok": syntax_ok,
        "ran": ran,
        "matches_library": matches,
        "cargo_available": cargo_available,
        "cargo_ran": cargo_ran,
        "cargo_matches_library": cargo_matches,
    }


def _confirm_api() -> Tuple[bool, List[str]]:
    ub_oracle = _import_pkg()

    exported = set(getattr(ub_oracle, "__all__", ()))
    missing: List[str] = []
    for name in PUBLIC_API_V1:
        if name not in exported or getattr(ub_oracle, name, None) is None:
            missing.append(name)
    return (len(missing) == 0), missing


def confirm_ecosystem() -> EcosystemReport:
    """Prove the stability promise and the cargo integration both hold:

    * every v1 public-API symbol is still exported and live (SemVer guard), and
    * the `cargo cross-lang-verify` and `cargo cross-verify` shims are valid
      shell, run end-to-end against a real manifest (through cargo discovery when
      cargo is installed), and emit JSON identical to the in-process library.
    """
    checks: List[str] = []

    api_ok, missing = _confirm_api()
    checks.append(f"api: {len(PUBLIC_API_V1)} v1 symbols, missing={missing}")
    api_snapshot_ok, snapshot_detail = api_snapshot_fresh()
    checks.append(f"api snapshot: {snapshot_detail}")

    # 1. Compute the library's JSON once, then require every cargo shim to match.
    env = dict(os.environ)
    env["CROSS_LANG_VERIFY_PYTHON"] = sys.executable
    env["CROSS_LANG_VERIFY_SRC"] = str(_SRC)
    env["PYTHONPATH"] = (
        f"{_SRC}{os.pathsep}{env['PYTHONPATH']}"
        if env.get("PYTHONPATH")
        else str(_SRC)
    )
    cli_args = ["--units", str(_EXAMPLE_MANIFEST),
                "--format", "json", "--no-confirm"]
    shim_results: Dict[str, Dict[str, bool]] = {}
    shim_syntax_ok = False
    shim_ran = False
    shim_matches = False
    try:
        lib = subprocess.run(
            [sys.executable, "-m", "ub_oracle.cli", *cli_args],
            capture_output=True, text=True, env=env, timeout=120,
        )
        lib_json = json.loads(lib.stdout)
        checks.append(f"library cli rc={lib.returncode}")
        for subcommand, path in _CARGO_SUBCOMMANDS:
            result = _run_cargo_shim(
                subcommand, path, cli_args, env, lib_json
            )
            shim_results[subcommand] = result
            checks.append(
                f"cargo {subcommand}: syntax={result['syntax_ok']} "
                f"run={result['ran']} matches={result['matches_library']}"
            )
        shim_syntax_ok = all(r["syntax_ok"] for r in shim_results.values())
        shim_ran = all(
            r["ran"] and (not r["cargo_available"] or r["cargo_ran"])
            for r in shim_results.values()
        )
        shim_matches = all(
            r["matches_library"]
            and (not r["cargo_available"] or r["cargo_matches_library"])
            for r in shim_results.values()
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        checks.append(f"shim error: {exc!r}")

    ok = bool(api_ok and api_snapshot_ok and shim_syntax_ok and shim_ran
              and shim_matches)
    return EcosystemReport(
        ok=ok,
        api_ok=api_ok,
        api_snapshot_ok=api_snapshot_ok,
        missing_symbols=tuple(missing),
        shim_syntax_ok=shim_syntax_ok,
        shim_ran=shim_ran,
        shim_matches_library=shim_matches,
        shim_results=shim_results,
        checks=tuple(checks),
        detail="v1 API intact and cargo subcommands match the library"
        if ok else "api/shim check failed",
    )


if __name__ == "__main__":  # pragma: no cover
    rep = confirm_ecosystem()
    print(f"ecosystem ok={rep.ok} api_ok={rep.api_ok} "
          f"shim_matches={rep.shim_matches_library} semver={rep.semver}")
    for c in rep.checks:
        print("  -", c)
