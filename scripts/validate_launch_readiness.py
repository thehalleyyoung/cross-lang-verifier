#!/usr/bin/env python3
"""Validate the public launch packet against live repository evidence."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = (
    "README.md",
    "docs/launch.md",
    "docs/assets/readme_demo.gif",
    "docs/assets/demo_video.mp4",
    "docs/assets/demo_video_poster.png",
    "docs/SOUNDNESS_COMPENDIUM.md",
    "docs/SDK.md",
    "docs/TRANSPILER_RECIPES.md",
    "CONTRIBUTING.md",
    ".github/ISSUE_TEMPLATE/divergence_report.yml",
    ".github/ISSUE_TEMPLATE/plugin_request.yml",
    ".github/PULL_REQUEST_TEMPLATE.md",
    ".github/actions/translation-equivalence-guard/action.yml",
    ".github/workflows/translation-equivalence-guard.example.yml",
)

README_MARKERS = (
    "docs/assets/readme_demo.gif",
    "docs/assets/demo_video.mp4",
    "examples/readme_demo_units.json",
    "python3 -m src.ub_oracle.cli",
    "confirmed",
)

LAUNCH_MARKERS = (
    "Show HN",
    "r/rust",
    "r/programming",
    "c2rust",
    "TRACTOR",
    "Blog series",
    "Conference talk",
    "Good-first",
    "External posting",
)


class LaunchValidationError(RuntimeError):
    """Raised when a launch-readiness invariant is not satisfied."""


def _read(root: Path, relative: str) -> str:
    return (root / relative).read_text(encoding="utf-8")


def _require_files(root: Path, files: Iterable[str]) -> List[str]:
    present: List[str] = []
    for relative in files:
        path = root / relative
        if not path.exists():
            raise LaunchValidationError(f"missing required launch file: {relative}")
        present.append(relative)
    return present


def _require_markers(text: str, markers: Iterable[str], *, label: str) -> List[str]:
    missing = [marker for marker in markers if marker not in text]
    if missing:
        raise LaunchValidationError(
            f"{label} is missing required marker(s): {', '.join(missing)}"
        )
    return list(markers)


def _demo_command(root: Path, sarif_path: Path) -> List[str]:
    return [
        sys.executable,
        "-m",
        "src.ub_oracle.cli",
        "--units",
        str(root / "examples/readme_demo_units.json"),
        "--format",
        "text",
        "--color",
        "never",
        "--sarif",
        str(sarif_path),
        "--fail-on",
        "unknown",
    ]


def _run_demo(root: Path) -> Dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="clv-launch-") as tmp:
        sarif_path = Path(tmp) / "readme_demo.sarif"
        proc = subprocess.run(
            _demo_command(root, sarif_path),
            cwd=root,
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )
        if proc.returncode != 0:
            raise LaunchValidationError(
                "README demo command failed with exit "
                f"{proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )
        stdout = proc.stdout
        required_output = (
            "DIVERGENT",
            "c_rust_signed_overflow",
            "UB reachable",
            "Rust defined",
            "NO-DIVERGENCE",
            "c_rust_safe_add_zero",
            "abstract-interpretation",
        )
        _require_markers(stdout, required_output, label="README demo output")
        if not sarif_path.exists():
            raise LaunchValidationError("README demo did not write SARIF output")
        sarif = json.loads(sarif_path.read_text(encoding="utf-8"))
        if sarif.get("version") != "2.1.0":
            raise LaunchValidationError("SARIF output is not version 2.1.0")
        results = sarif.get("runs", [{}])[0].get("results", [])
        if not any(result.get("level") == "error" for result in results):
            raise LaunchValidationError("SARIF output has no error-level divergence")
        return {
            "stdout": stdout,
            "sarif_results": len(results),
            "sarif_rule_ids": sorted(
                {str(result.get("ruleId")) for result in results if result.get("ruleId")}
            ),
        }


def _tool_versions(root: Path) -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for name, command in {
        "python": [sys.executable, "--version"],
        "clang": ["clang", "--version"],
        "rustc": ["rustc", "--version"],
    }.items():
        proc = subprocess.run(
            command,
            cwd=root,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        if proc.returncode != 0:
            raise LaunchValidationError(
                f"required launch toolchain command failed: {' '.join(command)}"
            )
        versions[name] = (proc.stdout or proc.stderr).splitlines()[0].strip()
    return versions


def validate(root: Path = REPO_ROOT) -> Dict[str, object]:
    root = root.resolve()
    files = _require_files(root, REQUIRED_FILES)
    readme_markers = _require_markers(
        _read(root, "README.md"), README_MARKERS, label="README.md"
    )
    launch_markers = _require_markers(
        _read(root, "docs/launch.md"), LAUNCH_MARKERS, label="docs/launch.md"
    )
    contributing_markers = _require_markers(
        _read(root, "CONTRIBUTING.md"),
        ("Positive witness", "Safe negative control", "targeted command"),
        label="CONTRIBUTING.md",
    )
    demo = _run_demo(root)
    return {
        "ok": True,
        "files": files,
        "readme_markers": readme_markers,
        "launch_markers": launch_markers,
        "contributing_markers": contributing_markers,
        "tool_versions": _tool_versions(root),
        "demo": demo,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable evidence")
    args = parser.parse_args(argv)
    evidence = validate(REPO_ROOT)
    if args.json:
        print(json.dumps(evidence, indent=2, sort_keys=True))
    else:
        print("launch-readiness: PASS")
        for name, version in evidence["tool_versions"].items():
            print(f"  {name}: {version}")
        demo = evidence["demo"]
        print(
            "  readme-demo: confirmed DIVERGENT + NO-DIVERGENCE "
            f"({demo['sarif_results']} SARIF finding(s))"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
