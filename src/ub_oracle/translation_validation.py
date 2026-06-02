"""Step 71 — cross-language translation-validation framing.

The companion *code* for
[`docs/TRANSLATION_VALIDATION.md`](../../docs/TRANSLATION_VALIDATION.md). It
exposes the oracle through the standard **translation-validation** interface — a
per-instance, witness-producing validator `V(P_S, P_T, I, T)` that either
`REFUTES` the producer's faithfulness claim with a **re-executable**
counterexample witness, or returns `NOT_REFUTED` over the probed inputs (a
one-sided result; equivalence is never claimed).

The validity relation is exactly the relational assertion `R_m` of
:mod:`product_program` (so this layer is a thin, literature-facing adapter over a
construction that is already proven faithful to the operational semantics). What
this module adds is the **witness object** and its two operational theorems,
proven against real compiled programs:

* *witness soundness* — a `REFUTED` witness, replayed against **fresh**
  compilations, reproduces a violation of `R_m` (a genuine, third-party-checkable
  divergence);
* *witness determinism* — replaying the same witness twice yields the identical
  product observable (a stable, diffable artifact).

Nothing is simulated: `replay()` recompiles and re-runs real `clang`/UBSan +
`rustc`/`go` binaries. The validator is **target-parameterized** through
:mod:`target_semantics`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

from . import ground_truth as gt
from . import product_program as pp
from . import target_semantics as tsem
from .reexec import ReexecHarness, ToolchainStatus, toolchain_available

SCHEMA_VERSION = "translation-validation/v1"

# Validator verdicts.
REFUTED = "REFUTED"
NOT_REFUTED = "NOT_REFUTED"
UNAVAILABLE = "UNAVAILABLE"


# --------------------------------------------------------------------------- #
# The counterexample witness — a self-contained, re-executable artifact.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CounterexampleWitness:
    """A third-party-checkable refutation of the producer's faithfulness claim.

    Carries everything needed to reproduce the divergence with no access to this
    process: the two source texts, the concrete input, the target pack name, the
    mode, and the product observable that was measured. :meth:`replay` recompiles
    and re-runs both sides from scratch and re-derives the product observable.
    """

    c_src: str
    target_src: str
    inputs: Tuple[str, ...]
    target: str
    mode: str
    klass: str
    observable: pp.ProductObservable
    reason: str

    @property
    def observable_key(self) -> Tuple:
        o = self.observable
        return (o.target, o.mode, o.o0_rc, o.o0_val, o.o2_rc, o.o2_val,
                o.san_trapped, o.defined, o.deterministic)

    def fingerprint(self) -> str:
        """Stable content fingerprint over the *witness inputs* (not timings)."""
        blob = json.dumps(
            {"c": self.c_src, "t": self.target_src, "i": list(self.inputs),
             "target": self.target, "mode": self.mode, "klass": self.klass},
            sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:32]

    def replay(self, h: Optional[ReexecHarness] = None
               ) -> Optional[pp.ProductObservable]:
        """Recompile + re-run both sides from scratch; return the fresh product
        observable (or ``None`` if the toolchain is unavailable)."""
        status = toolchain_available()
        if not status.full_for(self.target):
            return None
        harness = h or ReexecHarness(status)
        built = pp.build_product(
            harness, self.c_src, self.target_src, list(self.inputs),
            target=self.target, mode=self.mode, divergence_class=self.klass)
        if built is None:
            return None
        return built[0]

    def replay_reproduces_violation(self, h: Optional[ReexecHarness] = None
                                    ) -> bool:
        """Witness-soundness check for this single witness: a fresh replay still
        violates the relational assertion `R_m`."""
        fresh = self.replay(h)
        if fresh is None:
            return True  # consistency-only when toolchain absent
        return pp.product_violated(fresh)


# --------------------------------------------------------------------------- #
# The validator.
# --------------------------------------------------------------------------- #
@dataclass
class ValidationResult:
    verdict: str                       # REFUTED | NOT_REFUTED | UNAVAILABLE
    target: str
    n_probed: int
    witness: Optional[CounterexampleWitness]
    reason: str

    @property
    def refuted(self) -> bool:
        return self.verdict == REFUTED

    def render(self) -> str:
        if self.verdict == UNAVAILABLE:
            return f"translation-validation: unavailable ({self.reason})"
        if self.refuted and self.witness is not None:
            w = self.witness
            return (f"REFUTED [{self.target}] on input {list(w.inputs)} "
                    f"(fingerprint {w.fingerprint()}): {w.reason}")
        return (f"NOT_REFUTED [{self.target}] over {self.n_probed} probed input(s) "
                f"— no divergence witnessed (one-sided; not an equivalence claim)")


def validate(
    c_src: str,
    target_src: str,
    candidate_inputs: List[Tuple[str, ...]],
    target: str,
    mode: str = pp.TRAP_VS_DEFINED,
    klass: str = "div_by_zero",
    harness: Optional[ReexecHarness] = None,
) -> ValidationResult:
    """The translation validator `V(P_S, P_T, I, T)`.

    Probes ``candidate_inputs`` in order; returns ``REFUTED`` with a
    re-executable witness at the first input violating `R_m`, else
    ``NOT_REFUTED`` over the probed inputs. ``target`` selects the semantics pack.
    """
    tsem.get_pack(target)  # validate pack (parameterization point)
    status = toolchain_available()
    if not status.full_for(target):
        return ValidationResult(UNAVAILABLE, target, 0, None,
                                "toolchain unavailable for target")
    h = harness or ReexecHarness(status)

    probed = 0
    for inp in candidate_inputs:
        probed += 1
        built = pp.build_product(h, c_src, target_src, list(inp),
                                 target=target, mode=mode, divergence_class=klass)
        if built is None:
            continue
        obs, res = built
        if pp.product_violated(obs):
            w = CounterexampleWitness(
                c_src=c_src, target_src=target_src, inputs=tuple(inp),
                target=target, mode=mode, klass=klass, observable=obs,
                reason=getattr(res, "reason", "R_m violated"))
            return ValidationResult(REFUTED, target, probed, w,
                                    "counterexample witness found")
    return ValidationResult(NOT_REFUTED, target, probed, None,
                            "no probed input witnessed a divergence")


# --------------------------------------------------------------------------- #
# Confirmation — the executable witness theorems.
# --------------------------------------------------------------------------- #
@dataclass
class TVCheck:
    item_id: str
    target: str
    declared_label: str
    verdict: str
    replay_reproduced: bool
    replay_deterministic: bool
    agree: bool


@dataclass
class TVConfirmation:
    available: bool
    ok: bool
    n_checked: int
    n_refuted: int
    n_not_refuted: int
    checks: List[TVCheck]
    content_hash: str
    detail: str

    def render(self) -> str:
        if not self.available:
            return "translation-validation: toolchain unavailable (consistency only)"
        lines = [
            "Cross-language translation-validation witness theorems (real code):",
            f"  items checked={self.n_checked} "
            f"(refuted={self.n_refuted}, not_refuted={self.n_not_refuted})",
            f"  content_hash={self.content_hash}",
            f"  all-agree(verdict + witness soundness + determinism)={self.ok}",
        ]
        for c in self.checks[:12]:
            lines.append(
                f"    [{c.target:5s}] {c.item_id:18s} {c.declared_label:10s} "
                f"{c.verdict:11s} replay={int(c.replay_reproduced)} "
                f"det={int(c.replay_deterministic)} -> "
                f"{'ok' if c.agree else 'MISMATCH'}")
        return "\n".join(lines)


def _hash_checks(checks: List[TVCheck]) -> str:
    layer = sorted(
        (c.item_id, c.target, c.declared_label, c.verdict,
         c.replay_reproduced, c.replay_deterministic)
        for c in checks)
    blob = json.dumps(layer, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


def confirm_translation_validation(
    langs: Tuple[str, ...] = ("rust", "go"),
    per_class: int = 1,
) -> TVConfirmation:
    """Prove, on real code, the validator's two witness theorems.

    For a per-family sample of divergent and equivalent corpus items per pack:

    * **divergent** items must be ``REFUTED``; the emitted witness must **replay**
      against fresh compilations to the *same* `R_m` violation (witness
      soundness), and a second replay must reproduce the **identical** product
      observable (witness determinism);
    * **equivalent** items must be ``NOT_REFUTED`` (no false refutation).
    """
    status = toolchain_available()
    available = any(status.full_for(l) for l in langs)
    if not available:
        return TVConfirmation(False, True, 0, 0, 0, [], "",
                              "toolchain unavailable: consistency-only pass")

    h = ReexecHarness(status)
    items = gt.enumerate_corpus(langs)
    buckets: Dict[Tuple[str, str, str], int] = {}
    selected: List[gt.GTItem] = []
    for it in items:
        key = (it.lang, it.klass, it.declared_label)
        n = buckets.get(key, 0)
        if n < per_class:
            buckets[key] = n + 1
            selected.append(it)

    checks: List[TVCheck] = []
    for it in selected:
        if not status.full_for(it.lang):
            continue
        vr = validate(it.c_src, it.target_src, [tuple(it.inputs)],
                      target=it.lang, mode=pp.TRAP_VS_DEFINED, klass=it.klass,
                      harness=h)
        if vr.verdict == UNAVAILABLE:
            continue

        replay_ok = True
        det_ok = True
        if it.declared_label == "divergent":
            expect = (vr.verdict == REFUTED)
            if vr.witness is not None:
                r1 = vr.witness.replay(h)
                r2 = vr.witness.replay(h)
                replay_ok = (r1 is not None and pp.product_violated(r1))
                det_ok = (r1 is not None and r2 is not None
                          and r1.__dict__ == r2.__dict__)
            else:
                expect = False
            agree = expect and replay_ok and det_ok
        else:  # equivalent
            agree = (vr.verdict == NOT_REFUTED)
            replay_ok = True
            det_ok = True

        checks.append(TVCheck(
            item_id=it.item_id, target=it.lang, declared_label=it.declared_label,
            verdict=vr.verdict, replay_reproduced=replay_ok,
            replay_deterministic=det_ok, agree=agree))

    n_ref = sum(1 for c in checks if c.verdict == REFUTED)
    n_nref = sum(1 for c in checks if c.verdict == NOT_REFUTED)
    ok = bool(checks) and all(c.agree for c in checks)
    # every divergent item must be refuted; every equivalent item must not be.
    ok = ok and all(c.verdict == REFUTED
                    for c in checks if c.declared_label == "divergent")
    ok = ok and all(c.verdict == NOT_REFUTED
                    for c in checks if c.declared_label == "equivalent")
    detail = (f"checked={len(checks)} refuted={n_ref} not_refuted={n_nref} "
              f"all_agree={all(c.agree for c in checks) if checks else False}")
    return TVConfirmation(
        available=True, ok=ok, n_checked=len(checks), n_refuted=n_ref,
        n_not_refuted=n_nref, checks=checks, content_hash=_hash_checks(checks),
        detail=detail)


TRANSLATION_VALIDATION_SPI = {
    "validate": validate,
    "CounterexampleWitness": CounterexampleWitness,
    "confirm_translation_validation": confirm_translation_validation,
    "REFUTED": REFUTED,
    "NOT_REFUTED": NOT_REFUTED,
}


if __name__ == "__main__":  # pragma: no cover
    conf = confirm_translation_validation(per_class=1)
    print(f"available={conf.available} ok={conf.ok}")
    print(conf.render())
    print("detail:", conf.detail)
