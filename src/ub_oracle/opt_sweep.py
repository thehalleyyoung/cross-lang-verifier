"""
Optimization-level sweep (100_STEPS step 152).

The "no sanitizer can trap it" divergence classes (strict aliasing, ``restrict``
violation, ``-ffast-math`` reassociation) are *latent* at ``-O0``: the program
runs the way a naive reader expects.  Their hazard only becomes **observable**
once the optimizer starts to *rely* on the undefined/under-specified behaviour.
A natural, reviewer-legible question is therefore: **at which optimization level
does each C UB first change the program's observable output?**

This module answers it empirically.  For each optimizer-exploited oracle it takes
that oracle's own Z3-found witness, compiles the witnessing C source at
``-O0, -O1, -O2, -O3`` (plus any class-specific flag pair the oracle declares),
runs each build on the witness input, and records the first level whose stdout
diverges from the ``-O0`` baseline — the *onset level* of the UB.

Two layers, mirroring the rest of the project:

* :func:`sweep_grid` is deterministic and toolchain-free: it just enumerates the
  ``(class, levels)`` cells that *would* be swept, so the grid shape is a
  byte-stable artifact.
* :func:`sweep` shells out to the real C compiler and fills each cell with the
  observed stdout + onset level — environment-dependent ground truth, exactly
  like the cross-pair confirmation layer.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import plugin
from .reexec import ReexecHarness, toolchain_available

#: the optimization levels swept, in increasing aggressiveness.
OPT_LEVELS: Tuple[str, ...] = ("-O0", "-O1", "-O2", "-O3")

#: canonical units for the optimizer-exploited classes (no sanitizer traps
#: them; their evidence is an opt-level output flip).
_SWEEP_UNITS: Dict[str, Dict] = {
    "strict_aliasing": {"kind": "type_pun"},
    "restrict_violation": {"kind": "restrict_pair"},
    "fast_math_reassoc": {"kind": "fp_reassoc"},
}


def sweep_classes() -> List[str]:
    """The divergence classes whose hazard is opt-level latent (deterministic)."""
    return sorted(
        c for c in _SWEEP_UNITS
        if plugin.oracles_for(source_lang="c", divergence_class=c))


def sweep_grid() -> Dict[str, object]:
    """The toolchain-free shape of the sweep: classes x levels (byte-stable)."""
    classes = sweep_classes()
    return {
        "artifact": "optimization_level_sweep_grid",
        "levels": list(OPT_LEVELS),
        "classes": classes,
        "n_cells": len(classes) * len(OPT_LEVELS),
    }


@dataclass
class SweepRow:
    divergence_class: str
    flag_base: List[str]
    outputs: Dict[str, str] = field(default_factory=dict)
    onset_level: Optional[str] = None
    available: bool = True
    reason: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "divergence_class": self.divergence_class,
            "flag_base": self.flag_base,
            "outputs": self.outputs,
            "onset_level": self.onset_level,
            "available": self.available,
            "reason": self.reason,
        }


def _oracle_for(divergence_class: str):
    # any pair works — the C *source* (the thing we sweep) is shared across pairs.
    matches = plugin.oracles_for(source_lang="c",
                                 divergence_class=divergence_class)
    return matches[0] if matches else None


def _base_flags(oracle) -> List[str]:
    """The non-level flags an oracle needs (e.g. fast-math's licence flag).

    The optimizer-exploited oracles encode their resolving flag pair in
    ``optimizer_flag_variants`` as (defined_build, exploited_build).  Everything
    in the exploited build that is *not* an ``-O`` level is a base flag the
    sweep must keep at every level (so the UB stays *licensed* as we raise -O).
    """
    variants = getattr(oracle, "optimizer_flag_variants", None)
    if not variants:
        return []
    exploited = list(variants[1])
    return [f for f in exploited if not f.startswith("-O")]


def sweep(harness: Optional[ReexecHarness] = None) -> Dict[str, object]:
    """Run the real-compiler sweep over every opt-level-latent class."""
    harness = harness or ReexecHarness(toolchain_available())
    cc = harness.status.cc
    rows: List[SweepRow] = []
    for cls in sweep_classes():
        oracle = _oracle_for(cls)
        res = oracle.find_divergence(dict(_SWEEP_UNITS[cls]))
        ce = res.counterexample
        argv = [str(v) for v in ce.inputs.values()]
        base = _base_flags(oracle)
        row = SweepRow(divergence_class=cls, flag_base=base)
        if not (harness.status.c_available and ce.source_snippet):
            row.available = False
            row.reason = "no C compiler"
            rows.append(row)
            continue
        with tempfile.TemporaryDirectory() as d:
            cpath = os.path.join(d, "s.c")
            with open(cpath, "w") as f:
                f.write(ce.source_snippet)
            for lvl in OPT_LEVELS:
                opath = os.path.join(d, f"s{lvl}.out")
                cr = subprocess.run([cc, lvl, *base, "-o", opath, cpath],
                                    capture_output=True, text=True, timeout=60)
                if cr.returncode != 0:
                    row.outputs[lvl] = "<compile-failed>"
                    continue
                rr = subprocess.run([opath, *argv], capture_output=True,
                                    text=True, timeout=60)
                row.outputs[lvl] = rr.stdout.strip()
        baseline = row.outputs.get("-O0")
        for lvl in OPT_LEVELS[1:]:
            if row.outputs.get(lvl) != baseline:
                row.onset_level = lvl
                break
        rows.append(row)
    confirmed = [r for r in rows if r.available and r.onset_level is not None]
    return {
        "artifact": "optimization_level_sweep",
        "levels": list(OPT_LEVELS),
        "rows": [r.to_dict() for r in rows],
        "n_with_onset": len(confirmed),
    }


def render_table(report: Dict[str, object]) -> str:
    levels = report["levels"]
    head = ["class".ljust(22)] + [l.rjust(6) for l in levels] + ["  onset"]
    lines = [" ".join(head), "-" * (len(" ".join(head)))]
    for r in report["rows"]:
        cells = [r["divergence_class"].ljust(22)]
        for l in levels:
            cells.append((r["outputs"].get(l, "-") or "-").rjust(6))
        cells.append("  " + (r["onset_level"] or "(none)"))
        lines.append(" ".join(cells))
    return "\n".join(lines)
