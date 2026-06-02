"""Step 69 — documentation site (mkdocs).

A real, buildable `mkdocs` site that gathers the project's existing reference
docs into a navigable site **plus** an auto-generated *gallery of caught
divergences* derived from the live corpora (`idiomatic_corpus`,
`multipair_corpus`) so the gallery can never drift from the actual catalogue.

Guarantees (proven against the real `mkdocs` toolchain, not mocked):

* `generate_gallery()` renders `docs/gallery.md` deterministically from the
  in-repo corpora — every catalogued divergence (provenance, class, label,
  language pairs) appears, machine-generated.
* `confirm_docs_site()` writes the gallery, then invokes the **real `mkdocs`
  binary** with ``build --strict`` (which fails on any broken nav entry, bad
  link, or missing page) and confirms a site is produced. Consistency-only
  (``available=False``) when `mkdocs` is not installed — never fabricated.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from . import idiomatic_corpus as _idio
from . import multipair_corpus as _multi

_ROOT = Path(__file__).resolve().parents[2]
_DOCS = _ROOT / "docs"
_MKDOCS_YML = _ROOT / "mkdocs.yml"
_GALLERY = _DOCS / "gallery.md"


# ── gallery generation (from the live corpora) ───────────────────────────────


def _gallery_markdown() -> str:
    lines: List[str] = [
        "# Gallery of caught divergences",
        "",
        "*Auto-generated from the in-repo corpora "
        "(`ub_oracle.idiomatic_corpus`, `ub_oracle.multipair_corpus`) — do not "
        "edit by hand; run `python -m ub_oracle.docs_site`.*",
        "",
        "Each row is a real, compilable C function and its translation(s); the "
        "**verdict** is what the oracle proves live against clang/UBSan + the "
        "target compiler. *Divergent* rows are UB-rooted bugs the oracle flags; "
        "*equivalent* rows are safe translations it stays silent on.",
        "",
        "## Tier-2 — idiomatic anchors (one target each)",
        "",
        "| id | mirrors | class | target(s) | verdict |",
        "|----|---------|-------|-----------|---------|",
    ]
    for it in _idio.CORPUS:
        pairs = ", ".join(sorted(it.targets))
        lines.append(
            f"| `{it.item_id}` | {it.provenance} | `{it.klass}` | {pairs} | "
            f"**{it.declared_label}** |"
        )
    lines += [
        "",
        "## Tier-3 — multi-pair (every target at once)",
        "",
        "| id | mirrors | class | pairs | verdict |",
        "|----|---------|-------|-------|---------|",
    ]
    for fn in _multi.CORPUS:
        pairs = ", ".join(sorted(fn.targets))
        lines.append(
            f"| `{fn.func_id}` | {fn.provenance} | `{fn.klass}` | {pairs} | "
            f"**{fn.declared_label}** |"
        )
    n_div = sum(1 for it in _idio.CORPUS if it.declared_label == "divergent")
    n_div += sum(1 for fn in _multi.CORPUS if fn.declared_label == "divergent")
    lines += [
        "",
        f"*{n_div} catalogued UB-rooted divergences across "
        f"{len(_idio.CORPUS) + len(_multi.CORPUS)} functions. Every verdict is "
        "reproduced live by the test-suite and the traceability check.*",
        "",
    ]
    return "\n".join(lines)


def generate_gallery() -> Path:
    """Write `docs/gallery.md` from the live corpora; returns its path."""
    _GALLERY.write_text(_gallery_markdown(), encoding="utf-8")
    return _GALLERY


# ── building the real site ───────────────────────────────────────────────────


def _mkdocs_invocation() -> Optional[Sequence[str]]:
    """Locate a runnable mkdocs: a `mkdocs` binary on PATH, or the importable
    `mkdocs` module under the current interpreter (covers venv installs that are
    not on PATH). Returns the argv prefix, or None if mkdocs is unavailable."""
    onpath = shutil.which("mkdocs")
    if onpath:
        return [onpath]
    try:
        import importlib.util
        if importlib.util.find_spec("mkdocs") is not None:
            return [sys.executable, "-m", "mkdocs"]
    except Exception:
        pass
    return None


@dataclass(frozen=True)
class DocsSiteReport:
    available: bool
    ok: bool
    strict_build: bool
    pages_built: int
    detail: str


def confirm_docs_site(timeout: int = 240) -> DocsSiteReport:
    """Generate the gallery and run the **real** `mkdocs build --strict`.

    ``--strict`` turns any warning (dangling nav entry, broken internal link,
    missing page) into a failure, so a clean build is a real check that the
    site's structure is sound. Consistency-only when `mkdocs` is absent.
    """
    if not _MKDOCS_YML.exists():
        return DocsSiteReport(False, False, False, 0, "mkdocs.yml missing")
    generate_gallery()
    mk = _mkdocs_invocation()
    if mk is None:
        return DocsSiteReport(
            available=False, ok=True, strict_build=False, pages_built=0,
            detail="mkdocs not installed; gallery generated, build skipped",
        )
    with tempfile.TemporaryDirectory() as td:
        site = Path(td) / "site"
        proc = subprocess.run(
            list(mk) + ["build", "--strict", "--site-dir", str(site)],
            cwd=str(_ROOT), capture_output=True, text=True, timeout=timeout,
        )
        built = list(site.rglob("*.html")) if site.exists() else []
        ok = proc.returncode == 0 and len(built) > 0
        detail = (f"rc={proc.returncode} html_pages={len(built)} "
                  f"{(proc.stderr or proc.stdout).strip().splitlines()[-1:] }")
        return DocsSiteReport(
            available=True, ok=ok, strict_build=(proc.returncode == 0),
            pages_built=len(built), detail=detail,
        )


if __name__ == "__main__":  # pragma: no cover
    rep = confirm_docs_site()
    print("docs-site:", rep.detail)
    print("=> ok" if rep.ok else "=> FAILED")
    raise SystemExit(0 if rep.ok else 1)
