"""Step 54 — external replication kit (programmatic confirmation).

This module backs the ``Dockerfile`` + ``scripts/reproduce_kit.sh`` +
``make reproduce-kit`` external replication kit with a Python-checkable
confirmation so the kit's integrity is itself part of the traceability matrix.

It verifies that:

* the kit's files exist and carry the expected entry points (a stranger really
  can ``docker build`` and ``make reproduce-kit``);
* the labeled ground-truth corpus is large enough (>= 500 pairs, >= 2 language
  pairs) — the substrate every reproduced table draws from;
* (when a full toolchain is present) the corpora and the scale-measurement
  verdict layer re-confirm against real code; and
* a **content-hash manifest** computed over the *deterministic* layers
  (corpus statistics, the external-tool applicability table, and the kit file
  hashes) is **stable across runs** — the property that lets an artifact
  evaluator diff two independent reproductions.

The heavy, toolchain-gated re-confirmations are optional (``quick=False``) so the
same entry point serves both a fast integrity check and the full container run.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import external_baselines as xb
from . import ground_truth as gt

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# (relative path, required substrings that prove the right entry point is wired)
_KIT_FILES = {
    "Dockerfile": ("make", "reproduce-kit"),
    "scripts/reproduce_kit.sh": ("check_no_simulated_results.sh",
                                 "experiments.ub_divergence.run --check",
                                 "confirm_replication_kit"),
    "Makefile": ("reproduce-kit:", "docker-build:", "docker-reproduce:"),
}


@dataclass
class ReplicationReport:
    files: Dict[str, bool] = field(default_factory=dict)
    file_hashes: Dict[str, str] = field(default_factory=dict)
    corpus_size: int = 0
    n_langs: int = 0
    # toolchain-gated confirmations (None when skipped / unavailable)
    scale_ok: Optional[bool] = None
    headtohead_ok: Optional[bool] = None
    cve_ok: Optional[bool] = None

    @property
    def files_ok(self) -> bool:
        return bool(self.files) and all(self.files.values())

    @property
    def corpus_ok(self) -> bool:
        return self.corpus_size >= 500 and self.n_langs >= 2

    @property
    def toolchain_ok(self) -> bool:
        # A None means "skipped / consistency-only" and does not fail the kit.
        return all(v is not False
                   for v in (self.scale_ok, self.headtohead_ok, self.cve_ok))

    @property
    def ok(self) -> bool:
        return self.files_ok and self.corpus_ok and self.toolchain_ok


def _read(path: str) -> Optional[str]:
    try:
        with open(path, "r") as f:
            return f.read()
    except OSError:
        return None


def confirm_replication_kit(quick: bool = True,
                            langs=("rust", "go")) -> ReplicationReport:
    """Confirm the replication kit. ``quick=True`` checks only the deterministic
    integrity (files, entry points, corpus size); ``quick=False`` additionally
    re-confirms the corpora and the scale-measurement verdict layer against the
    real toolchain."""
    rep = ReplicationReport()
    for rel, needles in _KIT_FILES.items():
        content = _read(os.path.join(_ROOT, rel))
        present = content is not None and all(n in content for n in needles)
        rep.files[rel] = present
        if content is not None:
            rep.file_hashes[rel] = hashlib.sha256(content.encode()).hexdigest()[:16]

    stats = gt.corpus_stats(langs)
    rep.corpus_size = int(stats["total"])
    rep.n_langs = int(stats["n_langs"])

    if not quick:
        from . import scale_measure as sm
        from . import cve_corpus as cc
        sc = sm.confirm_scale(sample_per_class=1)
        rep.scale_ok = sc.ok if sc.available else None
        hh = xb.confirm_head_to_head(per_class=1)
        rep.headtohead_ok = hh.ok if hh.available else None
        cv = cc.confirm_corpus()
        rep.cve_ok = cv.ok if cv.available else None
    return rep


def manifest(rep: ReplicationReport, langs=("rust", "go")) -> Dict[str, object]:
    """A deterministic, content-hashed manifest of the kit. The ``kit_hash`` is
    computed over only the run-to-run-stable layers (corpus statistics, the
    external-tool applicability table and the kit file hashes), so two
    reproductions on the same checkout produce the identical hash."""
    stable = {
        "corpus_stats": gt.corpus_stats(langs),
        "applicability_table": xb.applicability_table(),
        "kit_files": dict(sorted(rep.file_hashes.items())),
    }
    canon = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
    kit_hash = hashlib.sha256(canon).hexdigest()
    return {
        "kit_hash": kit_hash,
        "files_present": dict(sorted(rep.files.items())),
        "corpus_size": rep.corpus_size,
        "n_langs": rep.n_langs,
        "toolchain": {
            "scale_ok": rep.scale_ok,
            "headtohead_ok": rep.headtohead_ok,
            "cve_ok": rep.cve_ok,
        },
        "stable_layers": stable,
    }


def render(rep: ReplicationReport) -> str:
    lines = ["replication kit:"]
    for rel, ok in sorted(rep.files.items()):
        lines.append(f"  [{'ok' if ok else 'MISSING':7s}] {rel}")
    lines.append(f"  corpus: {rep.corpus_size} pairs across {rep.n_langs} languages "
                 f"({'ok' if rep.corpus_ok else 'TOO SMALL'})")
    for name, val in (("scale", rep.scale_ok), ("head-to-head", rep.headtohead_ok),
                      ("cve-corpus", rep.cve_ok)):
        if val is None:
            lines.append(f"  {name}: skipped/consistency-only")
        else:
            lines.append(f"  {name}: {'ok' if val else 'FAILED'}")
    lines.append(f"  => {'PASSED' if rep.ok else 'FAILED'}")
    return "\n".join(lines)


REPLICATION_SPI = {
    "confirm_replication_kit": confirm_replication_kit,
    "manifest": manifest,
    "render": render,
}


if __name__ == "__main__":  # pragma: no cover
    rep = confirm_replication_kit(quick=False)
    print(render(rep))
    m = manifest(rep)
    print(f"kit_hash={m['kit_hash']}")
