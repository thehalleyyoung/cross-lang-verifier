"""
Cross-pair regression matrix (100_STEPS step 40).

The matrix is the *living evidence of generality*: it drives **every** registered
divergence oracle across **every** supported ``(source, target)`` language pair
and records, for each cell, the symbolic verdict (and — when a real toolchain is
present — the ground-truth confirmation against actual compilers).

Two layers, mirroring the rest of the project:

* :func:`build_matrix` is **deterministic and toolchain-free** — it only runs the
  Z3 witness search, so the resulting artifact is byte-reproducible and can be
  asserted in CI exactly like ``experiments/ub_divergence/results.json``.
* :func:`confirm_matrix` additionally compiles & runs each witness against the
  real C (under UBSan) and the real target compiler, via the ground-truth
  harness, for whatever pairs the host toolchain supports.

Crucially, this file contains **no per-language branches**: it discovers pairs
and oracles from the plugin registry and definedness from the target-semantics
packs, so a newly registered pair shows up in the matrix automatically.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Tuple

from . import plugin
from .target_semantics import get_pack
from .plugin import OracleVerdict

# One canonical, fixed unit per divergence class.  Because a generated pair
# oracle shares its anchor's ``divergence_class`` key, a single unit per class
# drives that class across *every* pair that implements it.
CANONICAL_UNITS: Dict[str, Dict[str, Any]] = {
    "signed_overflow": {"kind": "binop_const", "op": "add", "const": 2147483647,
                        "width": 32, "var": "x", "signed": True},
    "shift_oob": {"kind": "shift", "width": 32, "value": 1},
    "div_by_zero": {"kind": "div", "width": 32, "a": "a", "b": "b"},
    "intmin_div_neg1": {"kind": "div", "width": 32, "signed": True},
    "array_oob": {"kind": "array_index", "length": 4},
    "strict_aliasing": {"kind": "type_pun"},
    "fp_contraction": {"kind": "fp_fma", "probe": "fp_contraction"},
    "uninit_read": {"kind": "uninit_read",
                    "storage": {"kind": "struct", "fields": ["a", "b"]},
                    "writes": [{"slot": "a"}], "read": "b"},
    "uninit_padding": {"kind": "uninit_padding"},
    "vla_bound": {"kind": "vla", "width": 32, "var": "n"},
    "float_cast_overflow": {"kind": "float_cast", "width": 32, "var": "x"},
    "fast_math_reassoc": {"kind": "fp_reassoc"},
    "restrict_violation": {"kind": "restrict_pair"},
    "pointer_provenance": {"kind": "pointer_offset", "width": 32, "var": "n"},
    "signed_shift_sign_bit": {"kind": "sign_bit_shift", "width": 32, "var": "n"},
    "bitfield_layout": {"kind": "bitfield_struct"},
    "enum_out_of_range": {"kind": "enum_cast"},
    "memcpy_overlap": {"kind": "memcpy_overlap", "buffer_len": 16},
    "eval_order": {"kind": "unsequenced", "pattern": "postinc_read_add"},
    "longjmp_vla": {"kind": "longjmp_vla", "var": "n"},
    "atomic_ordering": {"kind": "atomic_litmus", "pattern": "store_buffering",
                        "source_order": "relaxed", "target_order": "seq_cst"},
}


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _defined_returncodes_for(lang: str) -> Tuple[int, ...]:
    """Defined outcome return codes for matrix metadata.

    ``c`` is special: it is not a normal safe target pack in the Rust->C reverse
    oracle, but a clean value return (0) is still the only defined C process
    outcome the harness recognizes outside sanitizer-confirmed UB.
    """
    if lang == "c":
        return (0,)
    return get_pack(lang).defined_returncodes


def _oracle_available(status, oracle) -> bool:
    """Whether ``oracle`` should be attempted by the real-compiler matrix."""
    src = oracle.source_lang
    tgt = oracle.target_lang
    if oracle.confirmation_mode == "source_defined_target_ub":
        return bool(
            status.can_compile(src)
            and getattr(status, "c_available", False)
            and getattr(status, "ubsan", False)
        )
    if src == "c":
        if oracle.confirmation_mode in ("asan_trap_vs_defined",
                                        "libc_contract_trap_vs_defined"):
            checker = getattr(status, "full_libc_contract_for", None)
            return bool(checker(tgt)) if checker is not None else False
        if oracle.confirmation_mode == "static_ub_vs_defined":
            target_available = getattr(status, "target_available", lambda _t: False)
            target_runnable = getattr(status, "target_runnable", lambda _t: True)
            return bool(getattr(status, "c_available", False)
                        and target_available(tgt) and target_runnable(tgt))
        if oracle.confirmation_mode == "model_level_divergence":
            target_available = getattr(status, "target_available", lambda _t: False)
            target_runnable = getattr(status, "target_runnable", lambda _t: True)
            return bool(getattr(status, "c_available", False)
                        and target_available(tgt) and target_runnable(tgt))
        if oracle.confirmation_mode == "uninit_padding":
            checker = getattr(status, "full_uninit_padding_for", None)
            return bool(checker(tgt)) if checker is not None else False
        full_for = getattr(status, "full_for", lambda _t: False)
        return bool(full_for(tgt))
    return bool(status.can_compile(src) and status.can_compile(tgt))


def canonical_unit_for(oracle) -> Dict[str, Any]:
    """The canonical unit for ``oracle``'s class, tagged with its language pair."""
    cls = oracle.divergence_class
    if cls not in CANONICAL_UNITS:
        raise KeyError(
            f"no canonical unit registered for divergence class {cls!r}; "
            f"add one to regression_matrix.CANONICAL_UNITS")
    unit = dict(CANONICAL_UNITS[cls])
    unit["source_lang"] = oracle.source_lang
    unit["target_lang"] = oracle.target_lang
    return unit


def _sorted_oracles() -> List:
    """All registered oracles in a deterministic order."""
    return sorted(
        plugin.ALL_ORACLES,
        key=lambda o: (o.source_lang, o.target_lang, o.divergence_class),
    )


def build_matrix() -> Dict[str, Any]:
    """Run the symbolic oracle suite across every supported pair (no toolchain).

    This is deterministic *per fresh process*: every oracle either constructs its
    witness directly or takes the first model from a fixed SMT query, so a clean
    regeneration is byte-identical to the committed artifact (asserted by
    ``--check``, which always runs in a fresh interpreter).
    """
    cells: List[Dict[str, Any]] = []
    for oracle in _sorted_oracles():
        unit = canonical_unit_for(oracle)
        res = oracle.find_divergence(unit)
        ce = res.counterexample
        defined_codes = _defined_returncodes_for(oracle.target_lang)
        cells.append({
            "source_lang": oracle.source_lang,
            "target_lang": oracle.target_lang,
            "divergence_class": oracle.divergence_class,
            "verdict": str(res.verdict),
            "confirmation_mode": oracle.confirmation_mode,
            "source_definedness": ce.source_definedness,
            "target_defined_returncodes": list(defined_codes),
            "source_sha16": _sha(ce.source_snippet),
            "target_sha16": _sha(ce.target_snippet),
            "witness": ce.inputs,
            "divergence_witness": ce.divergence_witness,
        })

    pairs = sorted({(c["source_lang"], c["target_lang"]) for c in cells})
    classes = sorted({c["divergence_class"] for c in cells})
    coverage = []
    for src, tgt in pairs:
        covered = sorted(c["divergence_class"] for c in cells
                         if c["source_lang"] == src and c["target_lang"] == tgt)
        divergent = sum(
            1 for c in cells
            if c["source_lang"] == src and c["target_lang"] == tgt
            and c["verdict"] == str(OracleVerdict.DIVERGENT))
        coverage.append({
            "source_lang": src,
            "target_lang": tgt,
            "classes_covered": covered,
            "n_classes": len(covered),
            "n_divergent": divergent,
        })

    return {
        "artifact": "cross_pair_regression_matrix",
        "language_pairs": ["%s->%s" % p for p in pairs],
        "divergence_classes": classes,
        "n_cells": len(cells),
        "coverage": coverage,
        "cells": cells,
    }


def confirm_matrix(harness) -> Dict[str, Any]:
    """Confirm each cell against real compilers, for pairs the host supports.

    Cells whose target toolchain is unavailable are recorded as ``skipped`` (with
    a reason) rather than silently dropped, so the artifact is honest about what
    the host could actually prove.
    """
    status = harness.status
    cells: List[Dict[str, Any]] = []
    for oracle in _sorted_oracles():
        src = oracle.source_lang
        tgt = oracle.target_lang
        entry: Dict[str, Any] = {
            "source_lang": src,
            "target_lang": tgt,
            "divergence_class": oracle.divergence_class,
        }
        ok = _oracle_available(status, oracle)
        if not ok:
            entry.update(skipped=True,
                         reason=f"toolchain not available for {src}->{tgt}")
            cells.append(entry)
            continue
        res = oracle.confirm(oracle.find_divergence(canonical_unit_for(oracle)),
                             harness)
        rr = res.reexec
        entry.update(
            skipped=False,
            available=rr.available,
            ub_reachable=rr.ub_reachable,
            target_defined=rr.rust_defined,
            confirmed=rr.confirmed,
            target_returncode=(rr.rust_run.returncode
                               if rr.rust_run is not None else None),
            reason=rr.reason,
        )
        cells.append(entry)

    confirmed = [c for c in cells if c.get("confirmed")]
    attempted = [c for c in cells if not c.get("skipped")]
    return {
        "artifact": "cross_pair_regression_matrix_confirmations",
        "n_cells": len(cells),
        "n_attempted": len(attempted),
        "n_confirmed": len(confirmed),
        "all_attempted_confirmed": bool(attempted) and len(confirmed) == len(attempted),
        "cells": cells,
    }


def render_table(matrix: Optional[Dict[str, Any]] = None) -> str:
    """A compact text grid: rows = divergence classes, columns = language pairs."""
    matrix = matrix or build_matrix()
    pairs = matrix["language_pairs"]
    classes = matrix["divergence_classes"]
    have: Dict[Tuple[str, str], str] = {}
    for c in matrix["cells"]:
        key = (c["divergence_class"], "%s->%s" % (c["source_lang"], c["target_lang"]))
        have[key] = "D" if c["verdict"] == str(OracleVerdict.DIVERGENT) else "."

    cw = max([len(p) for p in pairs] + [3])
    rw = max([len(c) for c in classes] + [len("divergence_class")])
    header = "divergence_class".ljust(rw) + " | " + " | ".join(p.ljust(cw) for p in pairs)
    sep = "-" * len(header)
    lines = [header, sep]
    for cls in classes:
        row = cls.ljust(rw) + " | " + " | ".join(
            have.get((cls, p), "-").center(cw) for p in pairs)
        lines.append(row)
    lines.append(sep)
    lines.append(f"legend: D=divergent (symbolic witness found), .=non-divergent, "
                 f"-=class not implemented for that pair")
    return "\n".join(lines)
