"""Step 60 — a single-binary distribution path for non-Python users.

`docker run` (the Step-54 image) already gives a hermetic, toolchain-complete
path.  This module adds the *other* half of Step 60: a **single-file
executable** so someone who does not want to manage a Python environment can
grab one artifact and run the analyzer.

We build a **zipapp** (`python -m zipapp`): a single `.pyz` file that bundles the
entire `ub_oracle` package with a `__main__` shim pointing at the CLI.  A `.pyz`
is directly executable (`./cross-lang-verify.pyz --units …`) given any Python 3
interpreter — no `pip install`, no virtualenv, no source tree.  This is the
stdlib-blessed single-file path; unlike a PyInstaller binary it needs no native
toolchain to *produce*, so it is reproducible anywhere.

The build is proven end to end against the real CLI, not mocked:
`confirm_single_binary()` builds the `.pyz`, then **runs it as a subprocess**
against a real units manifest and checks that its JSON output matches what the
in-process CLI produces — i.e. the shipped single file is behaviourally
identical to the library.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import zipapp
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
_PKG = _SRC / "ub_oracle"
_EXAMPLE_MANIFEST = _ROOT / "examples" / "units_manifest.json"

# The __main__ shim placed at the archive root so `python app.pyz` runs the CLI.
_MAIN_SHIM = (
    "import sys\n"
    "from ub_oracle.cli import main\n"
    "if __name__ == '__main__':\n"
    "    sys.exit(main())\n"
)


def build_pyz(out_path: Path, interpreter: str = "/usr/bin/env python3") -> Path:
    """Build a single-file executable `.pyz` bundling `ub_oracle`.

    Copies the package into a staging dir, drops a root `__main__.py` shim, and
    zipapps it with an interpreter shebang so the result is directly runnable.
    """
    import shutil

    with tempfile.TemporaryDirectory() as td:
        stage = Path(td) / "app"
        stage.mkdir()
        shutil.copytree(_PKG, stage / "ub_oracle",
                        ignore=shutil.ignore_patterns(
                            "__pycache__", "*.pyc", ".coverage"))
        (stage / "__main__.py").write_text(_MAIN_SHIM, encoding="utf-8")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        zipapp.create_archive(stage, target=str(out_path),
                              interpreter=interpreter)
    return out_path


def _run_pyz(pyz: Path, args: List[str], timeout: int = 180) -> subprocess.CompletedProcess:
    # invoke via the current interpreter for hermetic, portable execution
    # (equivalent to `./app.pyz` when the shebang interpreter is on PATH).
    return subprocess.run(
        [sys.executable, str(pyz)] + args,
        cwd=str(_ROOT), capture_output=True, text=True, timeout=timeout)


def _inprocess_cli_json(manifest: Path) -> dict:
    import io
    from contextlib import redirect_stdout
    from .cli import main as cli_main

    buf = io.StringIO()
    with redirect_stdout(buf):
        cli_main(["--units", str(manifest), "--format", "json", "--no-confirm"])
    return json.loads(buf.getvalue())


@dataclass(frozen=True)
class SingleBinaryReport:
    available: bool
    ok: bool
    built: bool
    runs: bool
    matches_library: bool
    size_bytes: int
    detail: str


def confirm_single_binary() -> SingleBinaryReport:
    """Build the `.pyz`, run it as a subprocess on a real manifest, and prove
    its JSON output matches the in-process CLI byte for byte (after canonical
    JSON re-encoding).  Always available (pure stdlib)."""
    if not _EXAMPLE_MANIFEST.exists():
        return SingleBinaryReport(False, False, False, False, False, 0,
                                  "example manifest missing")
    with tempfile.TemporaryDirectory() as td:
        pyz = Path(td) / "cross-lang-verify.pyz"
        try:
            build_pyz(pyz)
        except Exception as e:  # pragma: no cover - build failures are real
            return SingleBinaryReport(True, False, False, False, False, 0,
                                      f"build failed: {e}")
        built = pyz.exists()
        size = pyz.stat().st_size if built else 0

        proc = _run_pyz(pyz, ["--units", str(_EXAMPLE_MANIFEST),
                              "--format", "json", "--no-confirm"])
        runs = proc.returncode == 0 and bool(proc.stdout.strip())
        matches = False
        if runs:
            try:
                from_binary = json.loads(proc.stdout)
                from_lib = _inprocess_cli_json(_EXAMPLE_MANIFEST)
                canon = lambda d: json.dumps(d, sort_keys=True)
                matches = canon(from_binary) == canon(from_lib)
            except Exception as e:  # pragma: no cover
                return SingleBinaryReport(True, False, built, runs, False, size,
                                          f"output compare failed: {e}")
        ok = built and runs and matches
        detail = (f"pyz={size}B run_rc={proc.returncode} "
                  f"matches_library={matches}")
        if not runs:
            detail += f" :: {(proc.stderr or '').strip()[-200:]}"
        return SingleBinaryReport(
            available=True, ok=bool(ok), built=built, runs=runs,
            matches_library=matches, size_bytes=size, detail=detail)


if __name__ == "__main__":  # pragma: no cover
    rep = confirm_single_binary()
    print("single-binary:", rep.detail)
    print("=> ok" if rep.ok else "=> FAILED")
    raise SystemExit(0 if rep.ok else 1)
