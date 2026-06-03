"""Step 123 — executable per-pair soundness statements for newly added pairs.

The repository already states a global, one-sided contract: a ``DIVERGENT``
verdict must be backed by a witness whose confirmation mode matches the semantic
polarity of the pair.  This module makes that contract explicit for the language
pairs added in steps 116--121:

* C -> Zig
* C -> C++20
* Rust -> C
* C -> WebAssembly
* Go -> Rust
* C -> OCaml

Each statement is machine-checked against the live plugin registry, target
semantics packs, the oracle's symbolic positive and negative controls, and (when
the host has the relevant compiler/runtime) real re-execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import oracles as _oracles  # noqa: F401  (registers all built-in pairs)
from . import generalization as gen
from . import plugin
from .plugin import OracleVerdict
from .reexec import ReexecHarness, ToolchainStatus, toolchain_available
from .target_semantics import PACKS


NEW_PAIR_CASE_IDS: Tuple[str, ...] = (
    "c_to_zig_div_by_zero",
    "c_to_cpp_sign_bit_shift",
    "rust_to_c_intmin_div_neg1",
    "c_to_wasm_div_by_zero",
    "go_to_rust_intmin_div_neg1",
    "c_to_ocaml_div_by_zero",
)

EXPECTED_MODES: Dict[str, str] = {
    "c_to_zig_div_by_zero": "trap_vs_defined",
    "c_to_cpp_sign_bit_shift": "trap_vs_defined",
    "rust_to_c_intmin_div_neg1": "source_defined_target_ub",
    "c_to_wasm_div_by_zero": "trap_vs_defined",
    "go_to_rust_intmin_div_neg1": "defined_divergence",
    "c_to_ocaml_div_by_zero": "trap_vs_defined",
}


@dataclass(frozen=True)
class PairSoundnessStatement:
    case: gen.PairGeneralizationCase

    @property
    def case_id(self) -> str:
        return self.case.case_id

    @property
    def pair(self) -> Tuple[str, str]:
        return self.case.pair

    @property
    def expected_mode(self) -> str:
        return EXPECTED_MODES[self.case_id]

    @property
    def statement(self) -> str:
        src, tgt = self.pair
        if self.expected_mode == "defined_divergence":
            return (
                f"{src}->{tgt}: both programs are language-defined and "
                "deterministic; a divergent verdict requires an observable "
                "defined-behavior difference on the same input."
            )
        if self.expected_mode == "source_defined_target_ub":
            return (
                f"{src}->{tgt}: the source program is language-defined and "
                "deterministic; a divergent verdict requires the translated C "
                "target to trap under UBSan on the same input."
            )
        return (
            f"{src}->{tgt}: the C source reaches the named undefined behavior "
            "under UBSan, while the target has a deterministic language-defined "
            "outcome on the same input."
        )


def statements() -> Tuple[PairSoundnessStatement, ...]:
    by_id = {case.case_id: case for case in gen.GENERALIZATION_V2_CASES}
    return tuple(PairSoundnessStatement(by_id[case_id])
                 for case_id in NEW_PAIR_CASE_IDS)


@dataclass
class PairSoundnessResult:
    case_id: str
    source_lang: str
    target_lang: str
    divergence_class: str
    expected_mode: str
    registry_ok: bool
    mode_ok: bool
    pack_ok: bool
    positive_ok: bool
    negative_ok: bool
    live_available: bool
    live_confirmed: bool
    failures: List[str] = field(default_factory=list)
    detail: str = ""

    @property
    def pair(self) -> Tuple[str, str]:
        return (self.source_lang, self.target_lang)

    @property
    def static_ok(self) -> bool:
        return (
            self.registry_ok and self.mode_ok and self.pack_ok
            and self.positive_ok and self.negative_ok and not self.failures
        )

    @property
    def ok(self) -> bool:
        return self.static_ok and (self.live_confirmed if self.live_available else True)


def _pack_covers(case: gen.PairGeneralizationCase) -> bool:
    if case.target_lang == "c":
        return True
    pack = PACKS.get(case.target_lang)
    if pack is None:
        return False
    return case.divergence_class in pack.class_resolution


def check_statement(
    stmt: PairSoundnessStatement,
    harness: Optional[ReexecHarness] = None,
) -> PairSoundnessResult:
    case = stmt.case
    status = harness.status if harness is not None else toolchain_available()
    failures: List[str] = []

    try:
        orc = plugin.get_oracle_for(
            case.divergence_class, case.source_lang, case.target_lang)
        registry_ok = True
    except KeyError as exc:
        orc = None
        registry_ok = False
        failures.append(str(exc))

    mode = getattr(orc, "confirmation_mode", "")
    mode_ok = mode == stmt.expected_mode
    if not mode_ok:
        failures.append(f"mode {mode!r} != expected {stmt.expected_mode!r}")

    pack_ok = _pack_covers(case)
    if not pack_ok:
        failures.append(
            f"target pack {case.target_lang!r} does not cover {case.divergence_class!r}")

    positive_ok = False
    negative_ok = False
    live_available = False
    live_confirmed = False
    detail = ""

    if orc is not None:
        pos = orc.find_divergence(case.positive_unit)
        positive_ok = pos.verdict is OracleVerdict.DIVERGENT and pos.counterexample is not None
        if not positive_ok:
            failures.append(f"positive control did not diverge: {pos.verdict} {pos.detail}")

        neg = orc.find_divergence(case.negative_unit)
        negative_ok = neg.verdict is not OracleVerdict.DIVERGENT
        if not negative_ok:
            failures.append("negative control produced a divergent symbolic witness")

        live_available = gen.case_available(case, status)
        if live_available and positive_ok:
            h = harness or ReexecHarness(status)
            live = orc.confirm(pos, h)
            rr = live.reexec
            live_confirmed = bool(rr is not None and rr.available and rr.confirmed)
            if not live_confirmed:
                failures.append(f"live confirmation failed: {rr.reason if rr else 'no reexec'}")
            else:
                detail = rr.reason if rr else ""
        elif not live_available:
            detail = "live confirmation unavailable on this host"

    return PairSoundnessResult(
        case_id=case.case_id,
        source_lang=case.source_lang,
        target_lang=case.target_lang,
        divergence_class=case.divergence_class,
        expected_mode=stmt.expected_mode,
        registry_ok=registry_ok,
        mode_ok=mode_ok,
        pack_ok=pack_ok,
        positive_ok=positive_ok,
        negative_ok=negative_ok,
        live_available=live_available,
        live_confirmed=live_confirmed,
        failures=failures,
        detail=detail,
    )


def check_pair_soundness(
    harness: Optional[ReexecHarness] = None,
) -> List[PairSoundnessResult]:
    return [check_statement(stmt, harness) for stmt in statements()]


@dataclass
class PairSoundnessConfirmation:
    ok: bool
    results: List[PairSoundnessResult]

    @property
    def live_confirmed_pairs(self) -> List[Tuple[str, str]]:
        return [r.pair for r in self.results if r.live_confirmed]

    def render(self) -> str:
        lines = ["Per-pair soundness statements (steps 116-121):"]
        for r in self.results:
            live = "confirmed" if r.live_confirmed else (
                "unavailable" if not r.live_available else "failed")
            lines.append(
                f"  {r.source_lang}->{r.target_lang} {r.divergence_class}: "
                f"static_ok={r.static_ok} live={live}")
        lines.append(f"  ok={self.ok}")
        return "\n".join(lines)


def confirm_pair_soundness(
    harness: Optional[ReexecHarness] = None,
) -> PairSoundnessConfirmation:
    results = check_pair_soundness(harness)
    return PairSoundnessConfirmation(all(r.ok for r in results), results)


if __name__ == "__main__":  # pragma: no cover
    print(confirm_pair_soundness().render())
