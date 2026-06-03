"""
Incremental verification cache (100_STEPS steps 66 and 142).

Confirming a divergence is *expensive*: every unit compiles C three ways (plus
UBSan), compiles the target, and runs all of them.  On a large migration in CI,
re-doing that for unchanged units on every push is wasteful.  This module adds a
content-addressed cache so that **only changed units are re-verified**.

The design is built around the project's soundness contract — a cache must never
let a stale verdict outlive a change that could invalidate it:

* The cache **key** is a SHA-256 over three things:
    1. a ``SEMANTICS_VERSION`` constant, bumped whenever oracle logic changes,
    2. a **toolchain fingerprint** — the real ``clang``/``rustc``/``go``/``swiftc``
       ``--version`` strings — so a compiler upgrade (which can change what UB
       actually does) invalidates every affected entry automatically, and
   3. the **canonicalised unit content hash** (JSON with sorted keys).
  Change any of these and the key changes, forcing a fresh real-compiler run.

* The cache **value** records the deterministic verdict and, for confirmed
  divergences, the proof-carrying counterexample emitted by the real run.  A
  cache hit reconstructs a :class:`~ub_oracle.verify.VerifyReport` carrying
  exactly the verdict and certificate-bearing witness produced earlier under an
  identical toolchain.

* ``DIVERGENT`` / ``NO_DIVERGENCE_FOUND`` / ``NOT_COVERED`` are deterministic and
  safe to cache.  ``UNKNOWN`` (a solver abstained / timed out) and un-confirmable
  ``CANDIDATE`` results are **never** cached, since they are environment/timeout
  dependent and re-running them may legitimately change the answer.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .reexec import ReexecHarness, ToolchainStatus, toolchain_available
from .replay import Counterexample
from .plugin import OracleResult, OracleVerdict
from .verify import VerifyReport, VerifyVerdict, verify_unit
from .report import pair_of, _class_of

#: bump this whenever oracle/confirmation logic changes in a way that could
#: change a verdict for an unchanged unit + unchanged toolchain.
SEMANTICS_VERSION = "2"

CACHE_FORMAT_VERSION = 3

#: only these verdicts are deterministic enough to cache safely.
_CACHEABLE = {
    VerifyVerdict.DIVERGENT,
    VerifyVerdict.NO_DIVERGENCE_FOUND,
    VerifyVerdict.NOT_COVERED,
}


def _tool_version(path: Optional[str]) -> str:
    """The first line of ``<tool> --version``, or ``"absent"``.

    Failures degrade to a stable sentinel so fingerprinting never raises; the
    point is only that *different* toolchains hash differently.
    """
    if not path:
        return "absent"
    try:
        r = subprocess.run([path, "--version"], capture_output=True, text=True,
                           timeout=30)
        out = (r.stdout or r.stderr or "").strip().splitlines()
        return out[0] if out else "unknown"
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def toolchain_fingerprint(status: Optional[ToolchainStatus] = None) -> Dict[str, str]:
    """A stable dict of the real compiler version strings in this environment."""
    status = status or toolchain_available()
    fp: Dict[str, str] = {
        "cc": _tool_version(status.cc),
        "ubsan": "yes" if status.ubsan else "no",
    }
    for name, path in status.targets:
        fp[name] = _tool_version(path)
    return fp


def canonical_unit(unit: Dict) -> str:
    """Deterministic JSON for a unit (sorted keys), used in the cache key."""
    return json.dumps(unit, sort_keys=True, default=str)


def unit_content_hash(unit: Dict) -> str:
    """Stable SHA-256 over the canonical unit content.

    Step 142's cache contract is intentionally stated in terms of the source/unit
    hash rather than process-local object identity.  Two dicts with the same
    semantic content therefore address the same cache entry even if their key
    insertion order differs.
    """
    return hashlib.sha256(canonical_unit(unit).encode("utf-8")).hexdigest()


def cache_key(unit: Dict, fingerprint: Dict[str, str]) -> str:
    """Content-addressed key binding unit + toolchain + semantics version."""
    payload = "\u241f".join([
        SEMANTICS_VERSION,
        json.dumps(fingerprint, sort_keys=True),
        unit_content_hash(unit),
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class CacheEntry:
    verdict: str
    detail: str
    divergence_class: Optional[str]
    pair: str
    counterexample: Optional[Dict] = None

    def to_dict(self) -> Dict:
        out = {
            "verdict": self.verdict,
            "detail": self.detail,
            "divergence_class": self.divergence_class,
            "pair": self.pair,
        }
        if self.counterexample is not None:
            out["counterexample"] = self.counterexample
        return out

    @staticmethod
    def from_dict(d: Dict) -> "CacheEntry":
        return CacheEntry(
            verdict=d["verdict"],
            detail=d.get("detail", ""),
            divergence_class=d.get("divergence_class"),
            pair=d.get("pair", "unknown->unknown"),
            counterexample=d.get("counterexample"),
        )


class VerificationCache:
    """A JSON-file-backed, content-addressed verdict cache."""

    def __init__(self, fingerprint: Optional[Dict[str, str]] = None):
        self.fingerprint = fingerprint if fingerprint is not None \
            else toolchain_fingerprint()
        self._entries: Dict[str, CacheEntry] = {}

    # ── persistence ──────────────────────────────────────────────────────────
    @classmethod
    def load(cls, path: str,
             fingerprint: Optional[Dict[str, str]] = None) -> "VerificationCache":
        c = cls(fingerprint=fingerprint)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except (OSError, ValueError):
                return c  # a corrupt cache is simply ignored (cold start)
            if isinstance(data, dict) and \
                    data.get("format") == CACHE_FORMAT_VERSION:
                for k, v in (data.get("entries") or {}).items():
                    try:
                        c._entries[k] = CacheEntry.from_dict(v)
                    except (KeyError, TypeError):
                        continue
        return c

    def save(self, path: str) -> None:
        payload = {
            "format": CACHE_FORMAT_VERSION,
            "semantics_version": SEMANTICS_VERSION,
            "fingerprint": self.fingerprint,
            "entries": {k: e.to_dict() for k, e in sorted(self._entries.items())},
        }
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")

    # ── core ─────────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self._entries)

    def key_for(self, unit: Dict) -> str:
        return cache_key(unit, self.fingerprint)

    def get(self, unit: Dict) -> Optional[CacheEntry]:
        return self._entries.get(self.key_for(unit))

    def put(self, unit: Dict, report: VerifyReport) -> bool:
        """Cache a report's verdict iff it is deterministic. Returns stored?."""
        if report.verdict not in _CACHEABLE:
            return False
        cls = (report.divergence.divergence_class
               if report.divergence is not None else None)
        ce = None
        if report.divergence is not None and report.divergence.counterexample is not None:
            ce = report.divergence.counterexample.to_dict()
        self._entries[self.key_for(unit)] = CacheEntry(
            verdict=report.verdict.value,
            detail=report.detail,
            divergence_class=cls,
            pair=pair_of(report),
            counterexample=ce,
        )
        return True

    def prune_to(self, units: Sequence[Dict]) -> int:
        """Drop entries not referenced by ``units`` (keeps the cache bounded).

        Returns the number of entries removed.
        """
        live = {self.key_for(u) for u in units}
        stale = [k for k in self._entries if k not in live]
        for k in stale:
            del self._entries[k]
        return len(stale)


def _report_from_entry(unit: Dict, entry: CacheEntry) -> VerifyReport:
    """Reconstruct a VerifyReport from a cache hit (verdict-faithful)."""
    verdict = VerifyVerdict(entry.verdict)
    divergence = None
    if verdict is VerifyVerdict.DIVERGENT and entry.divergence_class:
        ce = None
        if entry.counterexample is not None:
            ce = Counterexample.from_dict(entry.counterexample)
        divergence = OracleResult(
            OracleVerdict.DIVERGENT,
            entry.divergence_class,
            counterexample=ce,
        )
    return VerifyReport(
        verdict, unit,
        divergence=divergence,
        detail=entry.detail + "  [cached]",
    )


@dataclass
class IncrementalResult:
    reports: List[VerifyReport] = field(default_factory=list)
    hits: int = 0
    misses: int = 0
    stored: int = 0

    @property
    def total(self) -> int:
        return len(self.reports)

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total else 0.0

    def to_dict(self) -> Dict:
        return {
            "total": self.total,
            "hits": self.hits,
            "misses": self.misses,
            "stored": self.stored,
            "hit_rate": self.hit_rate,
        }


def _stable_json_hash(obj: object) -> str:
    data = json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, default=str)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _normalised_detail(detail: str) -> str:
    suffix = "  [cached]"
    while detail.endswith(suffix):
        detail = detail[:-len(suffix)]
    return detail


def report_signature(unit: Dict, report: VerifyReport) -> Dict[str, object]:
    """Verdict-layer signature shared by cold and cache-hit reports.

    The cache deliberately rehydrates only the deterministic verdict layer:
    verdict, pair, class, and the proof-carrying counterexample.  Runtime fields
    such as ``toolchain_available`` or pre-pass bookkeeping are not part of the
    persisted cache value and therefore are not part of this equality proof.
    """
    cls = None
    counterexample_hash = None
    if report.divergence is not None:
        cls = report.divergence.divergence_class
        if report.divergence.counterexample is not None:
            counterexample_hash = _stable_json_hash(
                report.divergence.counterexample.to_dict())
    return {
        "unit_content_hash": unit_content_hash(unit),
        "unit_name": str(unit.get("name") or unit.get("id") or ""),
        "verdict": report.verdict.value,
        "pair": pair_of(report),
        "divergence_class": cls,
        "detail": _normalised_detail(report.detail),
        "counterexample_hash": counterexample_hash,
    }


def report_signature_hash(units: Sequence[Dict],
                          reports: Sequence[VerifyReport]) -> str:
    """Stable hash of a sequence of verdict-layer report signatures."""
    records = [
        report_signature(unit, report)
        for unit, report in zip(units, reports)
    ]
    return _stable_json_hash({
        "n_units": len(units),
        "n_reports": len(reports),
        "records": records,
    })


@dataclass
class CacheEquivalenceProof:
    """Evidence that cache hits replay the same verdict layer as cold runs."""

    ok: bool
    total: int
    cacheable: int
    cold_hits: int
    cold_misses: int
    warm_hits: int
    warm_misses: int
    cold_signature_hash: str
    warm_signature_hash: str
    fingerprint_hash: str
    mismatches: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "ok": self.ok,
            "total": self.total,
            "cacheable": self.cacheable,
            "cold_hits": self.cold_hits,
            "cold_misses": self.cold_misses,
            "warm_hits": self.warm_hits,
            "warm_misses": self.warm_misses,
            "cold_signature_hash": self.cold_signature_hash,
            "warm_signature_hash": self.warm_signature_hash,
            "fingerprint_hash": self.fingerprint_hash,
            "mismatches": list(self.mismatches),
        }


def prove_cache_equivalence(
    units: Sequence[Dict],
    *,
    fingerprint: Optional[Dict[str, str]] = None,
    harness: Optional[ReexecHarness] = None,
    confirm: bool = True,
    status: Optional[ToolchainStatus] = None,
) -> CacheEquivalenceProof:
    """Cold-run, warm-run, and hash-check the incremental cache contract.

    A valid proof establishes that:

    * a fresh cache has no hits on the cold pass;
    * every cacheable cold verdict is served as a warm-pass hit under the same
      unit-content hash and toolchain fingerprint;
    * the warm-pass verdict-layer signature is byte-identical to the cold pass.

    Non-cacheable verdicts (``CANDIDATE``/``UNKNOWN``) may legitimately miss on
    the warm pass; they are counted separately rather than treated as failures.
    """
    status = status or toolchain_available()
    fp = fingerprint if fingerprint is not None else toolchain_fingerprint(status)
    cache = VerificationCache(fingerprint=fp)
    cold = verify_incremental(
        units, cache, harness=harness, confirm=confirm, status=status)
    warm = verify_incremental(
        units, cache, harness=harness, confirm=confirm, status=status)

    cold_hash = report_signature_hash(units, cold.reports)
    warm_hash = report_signature_hash(units, warm.reports)
    cacheable = sum(1 for r in cold.reports if r.verdict in _CACHEABLE)

    mismatches: List[str] = []
    if cold.hits != 0:
        mismatches.append(f"fresh cold run had {cold.hits} cache hit(s)")
    if cold.misses != len(units):
        mismatches.append(
            f"fresh cold run had {cold.misses} miss(es) for {len(units)} unit(s)")
    if warm.hits != cacheable:
        mismatches.append(
            f"warm run had {warm.hits} hit(s), expected {cacheable}")
    expected_warm_misses = len(units) - cacheable
    if warm.misses != expected_warm_misses:
        mismatches.append(
            f"warm run had {warm.misses} miss(es), expected {expected_warm_misses}")
    if cold_hash != warm_hash:
        mismatches.append("cold and warm verdict-layer signatures differ")

    return CacheEquivalenceProof(
        ok=not mismatches,
        total=len(units),
        cacheable=cacheable,
        cold_hits=cold.hits,
        cold_misses=cold.misses,
        warm_hits=warm.hits,
        warm_misses=warm.misses,
        cold_signature_hash=cold_hash,
        warm_signature_hash=warm_hash,
        fingerprint_hash=_stable_json_hash(fp),
        mismatches=mismatches,
    )


def verify_incremental(units: Sequence[Dict],
                       cache: VerificationCache,
                       *,
                       harness: Optional[ReexecHarness] = None,
                       confirm: bool = True,
                       status: Optional[ToolchainStatus] = None
                       ) -> IncrementalResult:
    """Verify ``units``, reusing cached verdicts for unchanged units.

    A unit is a *hit* when an entry exists under its content+toolchain key; only
    *misses* are actually (re-)verified against the real compilers.  Freshly
    produced deterministic verdicts are written back into ``cache`` (the caller
    decides when to ``save``).
    """
    status = status or toolchain_available()
    result = IncrementalResult()
    for unit in units:
        entry = cache.get(unit)
        if entry is not None:
            result.reports.append(_report_from_entry(unit, entry))
            result.hits += 1
            continue
        rep = verify_unit(unit, harness=harness, confirm=confirm, status=status)
        result.reports.append(rep)
        result.misses += 1
        if cache.put(unit, rep):
            result.stored += 1
    return result
