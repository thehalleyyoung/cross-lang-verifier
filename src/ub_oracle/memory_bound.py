"""Memory-bounded verification mode (100_STEPS step 146).

The real-compiler confirmation path is allowed to be expensive, but it should
not be allowed to consume unbounded memory in CI or artifact runs.  This module
adds a verdict-layer proof for the bounded mode:

* the bounded run uses :class:`ub_oracle.reexec.ReexecHarness`'s process-tree RSS
  supervisor, which kills compiler/program subprocesses that exceed the cap;
* resource exhaustion is treated as a failed confirmation/abstention signal, not
  as sanitizer evidence or target-language definedness; and
* the proof compares the deterministic verdict-layer signatures of unbounded and
  bounded runs, reusing the same hash contract as the incremental-cache proof.

The cap is deliberately over subprocess-tree resident set size rather than
address-space rlimits: sanitizer runtimes reserve large virtual address spaces,
so virtual-memory caps would reject valid UBSan/ASan/MSan executions that use
modest actual RSS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .cache import report_signature_hash
from .reexec import ReexecHarness, ToolchainStatus, toolchain_available
from .verify import VerifyReport, verify_unit


def _verdict_counts(reports: Sequence[VerifyReport]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for report in reports:
        out[report.verdict.value] = out.get(report.verdict.value, 0) + 1
    return out


@dataclass
class MemoryBoundProof:
    """Evidence that an RSS-capped run preserves the unbounded verdict layer."""

    ok: bool
    total: int
    max_rss_mb: int
    observed_peak_rss_kb: int
    unbounded_signature_hash: str
    bounded_signature_hash: str
    unbounded_verdicts: Dict[str, int]
    bounded_verdicts: Dict[str, int]
    resource_exhausted: bool
    resource_exhaustions: List[str] = field(default_factory=list)
    mismatches: List[str] = field(default_factory=list)

    @property
    def max_rss_kb(self) -> int:
        return self.max_rss_mb * 1024

    def to_dict(self) -> Dict[str, object]:
        return {
            "ok": self.ok,
            "total": self.total,
            "max_rss_mb": self.max_rss_mb,
            "max_rss_kb": self.max_rss_kb,
            "observed_peak_rss_kb": self.observed_peak_rss_kb,
            "unbounded_signature_hash": self.unbounded_signature_hash,
            "bounded_signature_hash": self.bounded_signature_hash,
            "unbounded_verdicts": dict(self.unbounded_verdicts),
            "bounded_verdicts": dict(self.bounded_verdicts),
            "resource_exhausted": self.resource_exhausted,
            "resource_exhaustions": list(self.resource_exhaustions),
            "mismatches": list(self.mismatches),
        }


def prove_memory_bounded_equivalence(
    units: Sequence[Dict],
    *,
    max_rss_mb: int,
    status: Optional[ToolchainStatus] = None,
    confirm: bool = True,
) -> MemoryBoundProof:
    """Run ``units`` unbounded and RSS-bounded, then compare verdict signatures.

    The proof is intentionally verdict-layer only.  Memory measurements and
    compiler stderr are operational evidence, not semantic results; if the cap is
    too low and kills any subprocess, the proof fails rather than accepting a
    resource-limited abstention as evidence of semantic equivalence.
    """
    st = status or toolchain_available()
    unbounded_reports = [
        verify_unit(unit, confirm=confirm, status=st)
        for unit in units
    ]

    bounded_harness = ReexecHarness(st, max_rss_mb=max_rss_mb)
    bounded_reports = [
        verify_unit(unit, harness=bounded_harness, confirm=confirm, status=st)
        for unit in units
    ]

    unbounded_hash = report_signature_hash(units, unbounded_reports)
    bounded_hash = report_signature_hash(units, bounded_reports)
    mismatches: List[str] = []
    if unbounded_hash != bounded_hash:
        mismatches.append("bounded and unbounded verdict-layer signatures differ")
    if bounded_harness.resource_exhausted:
        mismatches.append("bounded run exhausted the RSS budget")
    if bounded_harness.peak_rss_kb > max_rss_mb * 1024:
        mismatches.append(
            f"observed peak RSS {bounded_harness.peak_rss_kb} KiB exceeds "
            f"cap {max_rss_mb * 1024} KiB")

    return MemoryBoundProof(
        ok=not mismatches,
        total=len(units),
        max_rss_mb=max_rss_mb,
        observed_peak_rss_kb=bounded_harness.peak_rss_kb,
        unbounded_signature_hash=unbounded_hash,
        bounded_signature_hash=bounded_hash,
        unbounded_verdicts=_verdict_counts(unbounded_reports),
        bounded_verdicts=_verdict_counts(bounded_reports),
        resource_exhausted=bounded_harness.resource_exhausted,
        resource_exhaustions=list(bounded_harness.resource_exhaustions),
        mismatches=mismatches,
    )
