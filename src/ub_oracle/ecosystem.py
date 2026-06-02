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

  2. **A real `cargo` subcommand.**  ``cargo-cross-lang-verify`` is a thin shim
     following cargo's ``cargo-<name>`` discovery convention; dropping it on
     ``PATH`` makes ``cargo cross-lang-verify --units …`` work.  It forwards to the
     proven CLI, so it cannot drift from the oracle.  ``confirm_ecosystem`` runs
     the shim *end to end* against a real manifest and requires it to emit the
     same oracle JSON the library produces.

Everything here is pure stdlib and verified against real execution.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
_INTEGRATIONS = _ROOT / "integrations"
_CARGO_SHIM = _INTEGRATIONS / "cargo-cross-lang-verify"
_API_SNAPSHOT = _INTEGRATIONS / "api_surface_v1.json"
_EXAMPLE_MANIFEST = _ROOT / "examples" / "units_manifest.json"

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
# cargo-cross-lang-verify -- a `cargo` subcommand wrapper for the
# cross-language divergence oracle.  Cargo discovers any executable named
# `cargo-<name>` on PATH and exposes it as `cargo <name>`, so installing this
# shim (chmod +x and place on PATH) enables:
#
#     cargo cross-lang-verify --units units_manifest.json --format json
#
# It is a thin forwarder to the proven CLI; it adds no semantics of its own and
# therefore cannot drift from the oracle.
set -euo pipefail

# Cargo invokes `cargo-<name> <name> <args...>`; drop the leading subcommand
# token so the CLI sees only its real flags.
if [[ "${1:-}" == "cross-lang-verify" ]]; then
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


def generate_artifacts() -> Tuple[Path, Path]:
    _INTEGRATIONS.mkdir(parents=True, exist_ok=True)
    _CARGO_SHIM.write_text(_CARGO_SHIM_BODY)
    os.chmod(_CARGO_SHIM, 0o755)
    _API_SNAPSHOT.write_text(
        json.dumps(_current_surface(), indent=2, sort_keys=True) + "\n"
    )
    return _CARGO_SHIM, _API_SNAPSHOT


# --------------------------------------------------------------------------
# Confirmation
# --------------------------------------------------------------------------
@dataclass
class EcosystemReport:
    ok: bool
    api_ok: bool
    missing_symbols: Tuple[str, ...]
    shim_syntax_ok: bool
    shim_ran: bool
    shim_matches_library: bool
    semver: str = SEMVER
    checks: Tuple[str, ...] = field(default_factory=tuple)
    detail: str = ""


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
    * the `cargo cross-lang-verify` shim is valid shell, runs end-to-end against
      a real manifest, and emits JSON identical to the in-process library.
    """
    generate_artifacts()
    checks: List[str] = []

    api_ok, missing = _confirm_api()
    checks.append(f"api: {len(PUBLIC_API_V1)} v1 symbols, missing={missing}")

    # 1. shim is valid bash (static syntax check).
    syntax = subprocess.run(
        ["bash", "-n", str(_CARGO_SHIM)],
        capture_output=True, text=True,
    )
    shim_syntax_ok = syntax.returncode == 0
    checks.append(f"shim bash -n rc={syntax.returncode}")

    # 2. run the shim end-to-end and compare to the library CLI output.
    env = dict(os.environ)
    env["CROSS_LANG_VERIFY_PYTHON"] = sys.executable
    env["CROSS_LANG_VERIFY_SRC"] = str(_SRC)
    cli_args = ["--units", str(_EXAMPLE_MANIFEST),
                "--format", "json", "--no-confirm"]
    shim_args = ["cross-lang-verify", *cli_args]
    shim_ran = False
    shim_matches = False
    try:
        run = subprocess.run(
            ["bash", str(_CARGO_SHIM), *shim_args],
            capture_output=True, text=True, env=env, timeout=120,
        )
        shim_ran = run.returncode == 0
        checks.append(f"shim run rc={run.returncode}")
        if shim_ran:
            shim_json = json.loads(run.stdout)
            lib = subprocess.run(
                [sys.executable, "-m", "ub_oracle.cli", *cli_args],
                capture_output=True, text=True,
                env={**env, "PYTHONPATH": str(_SRC)}, timeout=120,
            )
            lib_json = json.loads(lib.stdout)
            shim_matches = (shim_json == lib_json)
            checks.append(f"shim matches library: {shim_matches}")
    except Exception as exc:  # pragma: no cover - environment dependent
        checks.append(f"shim error: {exc!r}")

    ok = bool(api_ok and shim_syntax_ok and shim_ran and shim_matches)
    return EcosystemReport(
        ok=ok,
        api_ok=api_ok,
        missing_symbols=tuple(missing),
        shim_syntax_ok=shim_syntax_ok,
        shim_ran=shim_ran,
        shim_matches_library=shim_matches,
        checks=tuple(checks),
        detail="v1 API intact and cargo subcommand matches the library"
        if ok else "api/shim check failed",
    )


if __name__ == "__main__":  # pragma: no cover
    rep = confirm_ecosystem()
    print(f"ecosystem ok={rep.ok} api_ok={rep.api_ok} "
          f"shim_matches={rep.shim_matches_library} semver={rep.semver}")
    for c in rep.checks:
        print("  -", c)
