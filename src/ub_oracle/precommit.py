"""Pre-commit integration for cross-lang-verifier.

The hook is intentionally a thin adapter over ``ub_oracle.cli``: pre-commit
supplies staged filenames, this module resolves the affected verifier
manifest(s), then the same CLI path used in CI/GitHub Actions performs the
actual semantic check.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from . import cli

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
_HOOKS_FILE = _ROOT / ".pre-commit-hooks.yaml"
_PYPROJECT = _ROOT / "pyproject.toml"
_SAMPLE_DIR = _ROOT / "examples" / "pre_commit_sample"
_SAMPLE_CONFIG = _SAMPLE_DIR / ".pre-commit-config.yaml"

_DEFAULT_MANIFEST_NAMES = (
    "units_manifest.json",
    "cross-lang-verify.json",
    ".cross-lang-verify.json",
)
_SOURCE_SUFFIXES = {
    ".c",
    ".h",
    ".cc",
    ".cpp",
    ".cxx",
    ".hpp",
    ".rs",
    ".go",
    ".zig",
    ".ml",
    ".wasm",
}
_FAIL_CHOICES = ("candidate", "divergent", "not-covered", "unknown")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cross-lang-verify-pre-commit",
        description=(
            "Resolve staged cross-language translation files to verifier "
            "manifest(s) and run cross-lang-verify before commit."
        ),
    )
    parser.add_argument(
        "--manifest",
        action="append",
        default=[],
        metavar="MANIFEST.json",
        help=(
            "manifest to verify; repeatable. When supplied, filename discovery "
            "is used only for the status banner."
        ),
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        metavar="DIR",
        help="repository root used for relative paths and manifest discovery",
    )
    parser.add_argument(
        "--confirm",
        dest="confirm",
        action="store_true",
        default=False,
        help=(
            "run real compiler re-execution. The default is deterministic "
            "--no-confirm mode, suitable for fast pre-commit hooks."
        ),
    )
    parser.add_argument(
        "--no-confirm",
        dest="confirm",
        action="store_false",
        help="keep symbolic witnesses as CANDIDATE findings (default)",
    )
    parser.add_argument(
        "--fail-on",
        action="append",
        choices=_FAIL_CHOICES,
        default=None,
        help=(
            "verdict that blocks the commit; repeatable. Default: candidate, "
            "because pre-commit normally runs in --no-confirm mode."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="verifier output format (default: text)",
    )
    parser.add_argument(
        "--color",
        choices=("auto", "always", "never"),
        default="never",
        help="verifier color mode (default: never)",
    )
    parser.add_argument(
        "--sarif",
        metavar="OUT.sarif",
        help=(
            "also write SARIF. With multiple manifests, the hook writes "
            "one sibling file per manifest to avoid overwriting results."
        ),
    )
    parser.add_argument("filenames", nargs="*", help="filenames supplied by pre-commit")
    return parser


def _resolve(root: Path, token: str) -> Path:
    path = Path(token)
    return path if path.is_absolute() else root / path


def _rel(root: Path, path: Path) -> str:
    try:
        return str(path.resolve(strict=False).relative_to(root.resolve(strict=False)))
    except ValueError:
        return str(path)


def _is_probable_manifest_name(path: Path) -> bool:
    return path.name in _DEFAULT_MANIFEST_NAMES or path.name.endswith("_units.json")


def _looks_like_units_manifest(path: Path) -> bool:
    if not path.is_file() or path.suffix.lower() != ".json":
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if isinstance(data, dict):
        units = data.get("units")
        return isinstance(units, list) and all(isinstance(unit, dict) for unit in units)
    return isinstance(data, list) and all(isinstance(unit, dict) for unit in data)


def _within(root: Path, path: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return False
    return True


def _nearest_manifest(root: Path, changed: Path) -> Optional[Path]:
    directory = changed if changed.is_dir() else changed.parent
    current = directory.resolve(strict=False)
    root_resolved = root.resolve(strict=False)
    while _within(root, current):
        for name in _DEFAULT_MANIFEST_NAMES:
            candidate = current / name
            if candidate.is_file():
                return candidate
        if current == root_resolved:
            break
        current = current.parent
    return None


def _add_manifest(
    manifests: Dict[Path, List[str]],
    root: Path,
    manifest: Path,
    reason: str,
) -> None:
    key = manifest.resolve(strict=False)
    manifests.setdefault(key, [])
    rel_reason = reason if reason.startswith("--") else _rel(root, Path(reason))
    if rel_reason not in manifests[key]:
        manifests[key].append(rel_reason)


def _resolve_manifests(
    root: Path,
    explicit_manifests: Iterable[str],
    filenames: Iterable[str],
) -> Dict[Path, List[str]]:
    manifests: Dict[Path, List[str]] = {}
    explicit = list(explicit_manifests)
    if explicit:
        for token in explicit:
            _add_manifest(manifests, root, _resolve(root, token), "--manifest")
        for token in filenames:
            if _resolve(root, token).exists():
                for manifest in list(manifests):
                    _add_manifest(manifests, root, manifest, token)
        return manifests

    for token in filenames:
        changed = _resolve(root, token)
        if not changed.exists():
            continue
        if changed.suffix.lower() == ".json":
            if _looks_like_units_manifest(changed) or _is_probable_manifest_name(changed):
                _add_manifest(manifests, root, changed, token)
                continue
        if changed.suffix.lower() in _SOURCE_SUFFIXES:
            nearest = _nearest_manifest(root, changed)
            if nearest is not None:
                _add_manifest(manifests, root, nearest, token)
    return manifests


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "manifest"


def _sarif_for(base: Optional[str], root: Path, manifest: Path, index: int, total: int) -> Optional[str]:
    if base is None:
        return None
    path = _resolve(root, base)
    if total == 1:
        return str(path)
    suffix = path.suffix or ".sarif"
    manifest_slug = _slug(_rel(root, manifest))
    return str(path.with_name(f"{path.stem}.{index + 1}.{manifest_slug}{suffix}"))


def _banner(root: Path, manifest: Path, reasons: List[str]) -> None:
    sys.stdout.write(
        f"cross-lang-verify pre-commit: checking {_rel(root, manifest)}"
    )
    visible = [reason for reason in reasons if not reason.startswith("--")]
    if visible:
        shown = ", ".join(visible[:5])
        if len(visible) > 5:
            shown += f", ... (+{len(visible) - 5} more)"
        sys.stdout.write(f" (triggered by {shown})")
    sys.stdout.write("\n")


def _run_cli_for_manifest(
    root: Path,
    manifest: Path,
    *,
    confirm: bool,
    fail_on: Sequence[str],
    fmt: str,
    color: str,
    sarif: Optional[str],
) -> int:
    argv: List[str] = [
        "--units",
        str(manifest),
        "--format",
        fmt,
        "--color",
        color,
    ]
    if not confirm:
        argv.append("--no-confirm")
    for verdict in fail_on:
        argv.extend(["--fail-on", verdict])
    if sarif is not None:
        argv.extend(["--sarif", sarif])

    old_cwd = Path.cwd()
    try:
        os.chdir(root)
        return int(cli.run(argv))
    finally:
        os.chdir(old_cwd)


def run(argv: Optional[Sequence[str]] = None) -> int:
    args = create_parser().parse_args(argv)
    root = Path(args.repo_root).resolve(strict=False)
    fail_on = args.fail_on or ["candidate"]
    manifests = _resolve_manifests(root, args.manifest, args.filenames)
    if not manifests:
        return 0

    exit_code = 0
    items = list(manifests.items())
    for index, (manifest, reasons) in enumerate(items):
        _banner(root, manifest, reasons)
        sarif = _sarif_for(args.sarif, root, manifest, index, len(items))
        rc = _run_cli_for_manifest(
            root,
            manifest,
            confirm=args.confirm,
            fail_on=fail_on,
            fmt=args.format,
            color=args.color,
            sarif=sarif,
        )
        exit_code = max(exit_code, rc)
    return exit_code


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(argv)


@dataclass(frozen=True)
class PreCommitReport:
    ok: bool
    metadata_ok: bool
    fixture_config_ok: bool
    git_fixture_ok: bool
    divergent_exit_code: int
    divergent_output_ok: bool
    safe_exit_code: int
    no_manifest_exit_code: int
    missing_exit_code: int
    aggregate_exit_code: int
    checks: Tuple[str, ...]


def _text_has(path: Path, needles: Sequence[str], checks: List[str]) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        checks.append(f"{path} read error={exc}")
        return False
    missing = [needle for needle in needles if needle not in text]
    checks.append(f"{path.relative_to(_ROOT)} missing={missing}")
    return not missing


def _run_hook(fixture: Path, args: Sequence[str]) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        f"{_SRC}{os.pathsep}{env['PYTHONPATH']}"
        if env.get("PYTHONPATH")
        else str(_SRC)
    )
    return subprocess.run(
        [sys.executable, "-m", "ub_oracle.precommit", *args],
        cwd=str(fixture),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def confirm_precommit_hook(tmp_dir: Path) -> PreCommitReport:
    """Verify the hook metadata and exercise it on a real fixture repository."""
    checks: List[str] = []
    metadata_ok = _text_has(
        _HOOKS_FILE,
        [
            "id: cross-lang-verify",
            "entry: cross-lang-verify-pre-commit",
            "language: python",
            "pass_filenames: true",
        ],
        checks,
    ) and _text_has(
        _PYPROJECT,
        ['cross-lang-verify-pre-commit = "ub_oracle.precommit:main"'],
        checks,
    )
    fixture_config_ok = _text_has(
        _SAMPLE_CONFIG,
        [
            "repo: local",
            "id: cross-lang-verify",
            "entry: cross-lang-verify-pre-commit",
            "pass_filenames: true",
        ],
        checks,
    )

    fixture = tmp_dir / "fixture"
    shutil.copytree(_SAMPLE_DIR, fixture)
    git_fixture_ok = False
    if shutil.which("git") is not None:
        init = subprocess.run(["git", "init", "--quiet"], cwd=fixture)
        add = subprocess.run(["git", "add", "."], cwd=fixture)
        git_fixture_ok = init.returncode == 0 and add.returncode == 0
    checks.append(f"fixture git init/add ok={git_fixture_ok}")

    common = ["--repo-root", ".", "--no-confirm", "--fail-on", "candidate"]
    divergent = _run_hook(fixture, [*common, "c/overflow.c", "rust/src/lib.rs"])
    divergent_output_ok = (
        divergent.returncode == 1
        and "CANDIDATE" in divergent.stdout
        and "c/overflow.c" in divergent.stdout
    )
    checks.append(
        "divergent rc="
        f"{divergent.returncode} output_ok={divergent_output_ok}"
    )

    safe = _run_hook(
        fixture,
        [
            *common,
            "--manifest",
            "safe_units_manifest.json",
            "c/safe.c",
            "rust/src/safe.rs",
        ],
    )
    checks.append(f"safe rc={safe.returncode}")

    no_manifest = _run_hook(fixture, [*common, "docs/notes.txt"])
    checks.append(f"no-manifest rc={no_manifest.returncode}")

    ordinary_json = _run_hook(fixture, [*common, "package.json"])
    checks.append(f"ordinary-json rc={ordinary_json.returncode}")

    missing = _run_hook(fixture, [*common, "--manifest", "missing.json"])
    checks.append(f"missing-manifest rc={missing.returncode}")

    aggregate = _run_hook(
        fixture,
        [
            *common,
            "--manifest",
            "units_manifest.json",
            "--manifest",
            "missing.json",
        ],
    )
    checks.append(f"aggregate rc={aggregate.returncode}")

    ok = (
        metadata_ok
        and fixture_config_ok
        and git_fixture_ok
        and divergent_output_ok
        and safe.returncode == 0
        and no_manifest.returncode == 0
        and ordinary_json.returncode == 0
        and missing.returncode == 2
        and aggregate.returncode == 2
    )
    return PreCommitReport(
        ok=ok,
        metadata_ok=metadata_ok,
        fixture_config_ok=fixture_config_ok,
        git_fixture_ok=git_fixture_ok,
        divergent_exit_code=divergent.returncode,
        divergent_output_ok=divergent_output_ok,
        safe_exit_code=safe.returncode,
        no_manifest_exit_code=no_manifest.returncode,
        missing_exit_code=missing.returncode,
        aggregate_exit_code=aggregate.returncode,
        checks=tuple(checks),
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
