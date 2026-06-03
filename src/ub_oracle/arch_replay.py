"""Cross-architecture witness replay (100_STEPS step 153).

The project's confirmation discipline is intentionally grounded in real
toolchains.  This module extends that discipline across CPU architectures without
pretending that an unavailable ISA was tested:

* the host architecture is normalized to a small, auditable vocabulary;
* native replay executes the same ground-truth witnesses through the real
  ``ReexecHarness`` and records a deterministic verdict-layer hash;
* non-native architectures are recorded as unavailable unless a future runner is
  wired in explicitly; and
* the arch-difference detector is unit-tested with synthetic fixtures that are
  clearly marked as synthetic and never used as empirical evidence.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .cache import toolchain_fingerprint
from .ground_truth import GTItem, enumerate_corpus, label_item
from .parallel_harness import LabelFn, _canonical_bytes, run_parallel
from .reexec import ToolchainStatus, toolchain_available

SCHEMA_VERSION = "cross-architecture-replay/v1"
CANONICAL_ARCHES = ("arm64", "x86_64")
DEFAULT_TARGET_LANGS = ("rust", "go")


def normalize_arch(machine: Optional[str] = None) -> str:
    """Normalize common platform names while leaving unknown ISAs explicit."""

    raw = (machine if machine is not None else platform.machine()).strip().lower()
    if raw in {"x86_64", "amd64", "x64"}:
        return "x86_64"
    if raw in {"arm64", "aarch64"}:
        return "arm64"
    return raw or "unknown"


def _sorted_verdicts(
    verdicts: Iterable[Mapping[str, object]],
) -> Tuple[Dict[str, object], ...]:
    return tuple(
        sorted(
            (dict(v) for v in verdicts),
            key=lambda r: (str(r["item_id"]), str(r["lang"]), str(r["klass"])),
        )
    )


def arch_verdict_hash(verdicts: Iterable[Mapping[str, object]]) -> str:
    """Stable hash over an architecture's semantic verdict layer only."""

    return hashlib.sha256(_canonical_bytes(_sorted_verdicts(verdicts))).hexdigest()


@dataclass(frozen=True)
class ArchReplayResult:
    arch: str
    available: bool
    executed: int
    content_hash: str = ""
    verdicts: Tuple[Dict[str, object], ...] = ()
    faithful: bool = True
    toolchain_fingerprint: Dict[str, str] = field(default_factory=dict)
    detail: str = ""
    synthetic: bool = False

    def to_dict(self) -> Dict[str, object]:
        return {
            "arch": self.arch,
            "available": self.available,
            "executed": self.executed,
            "content_hash": self.content_hash,
            "verdicts": list(self.verdicts),
            "faithful": self.faithful,
            "toolchain_fingerprint": dict(sorted(self.toolchain_fingerprint.items())),
            "detail": self.detail,
            "synthetic": self.synthetic,
        }


@dataclass(frozen=True)
class CrossArchReport:
    requested_arches: Tuple[str, ...]
    host_arch: str
    results: Tuple[ArchReplayResult, ...]
    schema: str = SCHEMA_VERSION
    synthetic: bool = False

    @property
    def available_arches(self) -> Tuple[str, ...]:
        return tuple(r.arch for r in self.results if r.available)

    @property
    def unavailable_arches(self) -> Tuple[str, ...]:
        return tuple(r.arch for r in self.results if not r.available)

    @property
    def available(self) -> bool:
        return bool(self.available_arches)

    @property
    def arch_dependency_witnesses(self) -> Tuple[Dict[str, object], ...]:
        by_item: Dict[Tuple[str, str, str], Dict[str, str]] = {}
        for result in self.results:
            if not result.available:
                continue
            for verdict in result.verdicts:
                key = (
                    str(verdict["item_id"]),
                    str(verdict["lang"]),
                    str(verdict["klass"]),
                )
                by_item.setdefault(key, {})[result.arch] = str(verdict["observed_label"])

        witnesses: List[Dict[str, object]] = []
        for (item_id, lang, klass), labels_by_arch in sorted(by_item.items()):
            if len(set(labels_by_arch.values())) > 1:
                witnesses.append({
                    "item_id": item_id,
                    "lang": lang,
                    "klass": klass,
                    "observed_by_arch": dict(sorted(labels_by_arch.items())),
                })
        return tuple(witnesses)

    @property
    def arch_dependency_detected(self) -> bool:
        return bool(self.arch_dependency_witnesses)

    @property
    def ok(self) -> bool:
        return all((not r.available and r.executed == 0) or (r.executed > 0 and r.faithful)
                   for r in self.results)

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema": self.schema,
            "requested_arches": list(self.requested_arches),
            "host_arch": self.host_arch,
            "available_arches": list(self.available_arches),
            "unavailable_arches": list(self.unavailable_arches),
            "arch_dependency_detected": self.arch_dependency_detected,
            "arch_dependency_witnesses": list(self.arch_dependency_witnesses),
            "ok": self.ok,
            "synthetic": self.synthetic,
            "results": [r.to_dict() for r in self.results],
        }

    def summary(self) -> str:
        prefix = "synthetic cross-architecture detector" if self.synthetic else "cross-architecture replay"
        return (
            f"{prefix}: host={self.host_arch} available={list(self.available_arches)} "
            f"unavailable={list(self.unavailable_arches)} "
            f"arch_dependency={self.arch_dependency_detected}"
        )


def _unique_arches(arches: Sequence[str]) -> Tuple[str, ...]:
    seen = set()
    out: List[str] = []
    for arch in (normalize_arch(a) for a in arches):
        if arch not in seen:
            seen.add(arch)
            out.append(arch)
    return tuple(out)


def _default_sample(status: ToolchainStatus) -> Tuple[GTItem, ...]:
    """Pick one divergent and one equivalent item per runnable language."""

    items = [it for it in enumerate_corpus(DEFAULT_TARGET_LANGS) if status.full_for(it.lang)]
    chosen: Dict[Tuple[str, str], GTItem] = {}
    for item in items:
        chosen.setdefault((item.lang, item.declared_label), item)
    return tuple(chosen[k] for k in sorted(chosen))


def _native_arch_result(
    arch: str,
    items: Sequence[GTItem],
    status: ToolchainStatus,
    *,
    label_fn: LabelFn = label_item,
) -> ArchReplayResult:
    runnable = [item for item in items if status.full_for(item.lang)]
    if not runnable:
        return ArchReplayResult(
            arch=arch,
            available=False,
            executed=0,
            toolchain_fingerprint=toolchain_fingerprint(status),
            detail="no requested witnesses have a full native toolchain",
        )

    report = run_parallel(runnable, workers=1, status=status, label_fn=label_fn)
    verdicts = _sorted_verdicts(report.verdict_layer())
    return ArchReplayResult(
        arch=arch,
        available=True,
        executed=len(runnable),
        content_hash=arch_verdict_hash(verdicts),
        verdicts=verdicts,
        faithful=report.faithful,
        toolchain_fingerprint=toolchain_fingerprint(status),
        detail="native host replay through ReexecHarness",
    )


def confirm_cross_architecture_replay(
    items: Optional[Sequence[GTItem]] = None,
    *,
    requested_arches: Sequence[str] = CANONICAL_ARCHES,
    status: Optional[ToolchainStatus] = None,
    label_fn: LabelFn = label_item,
) -> CrossArchReport:
    """Replay witnesses on every genuinely available architecture.

    Today the repository has a native runner only.  The non-host ISA is therefore
    recorded as unavailable rather than inferred, emulated implicitly, or filled
    with synthetic evidence.  Future QEMU/remote-runner wiring can add real
    non-host ``ArchReplayResult`` objects without changing the report schema.
    """

    st = status or toolchain_available()
    host = normalize_arch()
    arches = _unique_arches(requested_arches)
    chosen = tuple(items) if items is not None else _default_sample(st)
    results: List[ArchReplayResult] = []

    for arch in arches:
        if arch == host:
            results.append(_native_arch_result(arch, chosen, st, label_fn=label_fn))
        else:
            results.append(ArchReplayResult(
                arch=arch,
                available=False,
                executed=0,
                toolchain_fingerprint=toolchain_fingerprint(st),
                detail="no native or emulated runner registered for this architecture",
            ))

    return CrossArchReport(
        requested_arches=arches,
        host_arch=host,
        results=tuple(results),
        synthetic=False,
    )


def synthetic_arch_report(
    verdicts_by_arch: Mapping[str, Sequence[Mapping[str, object]]],
) -> CrossArchReport:
    """Build a clearly-marked synthetic detector fixture for tests/theorems."""

    results: List[ArchReplayResult] = []
    for arch, verdicts in sorted(verdicts_by_arch.items()):
        sorted_verdicts = _sorted_verdicts(verdicts)
        results.append(ArchReplayResult(
            arch=normalize_arch(arch),
            available=True,
            executed=len(sorted_verdicts),
            content_hash=arch_verdict_hash(sorted_verdicts),
            verdicts=sorted_verdicts,
            faithful=True,
            detail="synthetic detector fixture; not empirical evidence",
            synthetic=True,
        ))
    return CrossArchReport(
        requested_arches=tuple(r.arch for r in results),
        host_arch="synthetic",
        results=tuple(results),
        synthetic=True,
    )


def report_json(report: CrossArchReport) -> str:
    return json.dumps(report.to_dict(), sort_keys=True, indent=2)


if __name__ == "__main__":  # pragma: no cover
    print(report_json(confirm_cross_architecture_replay()))
