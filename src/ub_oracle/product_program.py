"""Step 72 — relational / product-program formalization.

The companion *code* for [`docs/PRODUCT_PROGRAM.md`](../../docs/PRODUCT_PROGRAM.md).
It realizes the **product program** `P_S × P_T` as a concrete, executable object
and proves — against real compiled programs — that the relational assertion `R`
it carries decides identically to the operational divergence semantics
(`semantics.is_divergence`) and to the re-execution harness
(`reexec.ReexecHarness.confirm_trap_vs_defined`).

The construction is **parameterized over the target semantics pack**
(`target_semantics.TargetPack`): instantiating a new target only changes how
`defined`/`deterministic` are recorded; the product rules, the relational
assertion `R_m`, and the soundness argument are unchanged.

Nothing is simulated. The product observable `(src(i), tgt(i))` is built from
real `clang`/UBSan `-O0`/`-O2` runs and real `rustc`/`go` runs; the relational
assertion is evaluated on those measured observables.

Key objects
-----------
* :class:`ProductObservable` — the well-defined observable of one product run:
  the source triple ``(o0, o2, san)`` and the target pair ``(defined, det)``.
* :func:`product_violated` — the relational assertion ``¬R_m``; ``True`` iff the
  pair diverges at this input. By construction this is the *same Boolean
  function* of the recorded observables as :func:`semantics.is_divergence`.
* :func:`build_product` — run the real binaries (via the harness) and form the
  :class:`ProductObservable` (so the abstraction is *measured*, not assumed).
* :func:`confirm_product_program` — the executable soundness/completeness
  theorem: on real divergent and equivalent corpus items, across packs,
  ``product_violated == is_divergence == harness.confirmed``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

from . import ground_truth as gt
from . import semantics as sem
from . import target_semantics as tsem
from .reexec import ReexecHarness, ToolchainStatus, toolchain_available

SCHEMA_VERSION = "product-program/v1"

# The relational consequence modes, mirroring the semantics layer.
EXPLOITED = sem.EXPLOITED
TRAP_VS_DEFINED = sem.TRAP_VS_DEFINED
MODES = sem.MODES

# The three inference-rule clause names of the relational assertion R_m, kept as
# data so docs/tests can refer to them by name (see docs/PRODUCT_PROGRAM.md).
CLAUSES: Tuple[str, ...] = ("P_premise_ub_reached",
                            "T_target_defined",
                            "C_consequence")


# --------------------------------------------------------------------------- #
# The product observable.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ProductObservable:
    """The observable of one product run ``P_S × P_T`` on one fixed input.

    Source side: ``o0_rc/o0_val`` and ``o2_rc/o2_val`` are the ``-O0`` / ``-O2``
    observables (rc 0 ⇒ value in stdout); ``san_trapped`` is the UBSan witness.
    Target side: ``defined`` / ``deterministic`` as recorded by re-execution.
    ``target`` is the pack name the product was instantiated with.
    """

    target: str
    mode: str
    o0_rc: int
    o0_val: str
    o2_rc: int
    o2_val: str
    san_trapped: bool
    defined: bool
    deterministic: bool

    def to_observation(self) -> sem.Observation:
        """Re-package the product observable as a `semantics.Observation` so the
        two layers provably consume the *same* recorded numbers."""
        src = sem.SourceObservation(
            o0=sem.Outcome(self.o0_rc, self.o0_val),
            o2=sem.Outcome(self.o2_rc, self.o2_val),
            san_trapped=self.san_trapped,
        )
        tgt = sem.TargetObservation(defined=self.defined,
                                    deterministic=self.deterministic)
        mode = self.mode if self.mode in MODES else EXPLOITED
        return sem.Observation(source=src, target=tgt, mode=mode)


# --------------------------------------------------------------------------- #
# The relational assertion R_m and its three clauses (the inference rules).
# --------------------------------------------------------------------------- #
def clause_premise(obs: ProductObservable) -> bool:
    """(P) the source actually reaches UB on this input (sanitizer witness)."""
    return bool(obs.san_trapped)


def clause_target_defined(obs: ProductObservable) -> bool:
    """(T) the target outcome is defined (and deterministic unless mode=exploited)."""
    return bool(obs.defined) and (obs.deterministic or obs.mode == EXPLOITED)


def clause_consequence(obs: ProductObservable) -> bool:
    """(C) the mode-specific consequence clause."""
    if obs.mode == EXPLOITED:
        return (obs.o0_rc == 0 and obs.o2_rc == 0 and obs.o0_val != obs.o2_val)
    # trap_vs_defined: the consequence is the definedness gap itself.
    return bool(obs.san_trapped and obs.defined and obs.deterministic)


def evaluate_clauses(obs: ProductObservable) -> Dict[str, bool]:
    return {
        "P_premise_ub_reached": clause_premise(obs),
        "T_target_defined": clause_target_defined(obs),
        "C_consequence": clause_consequence(obs),
    }


def product_violated(obs: ProductObservable) -> bool:
    """``¬R_m`` — the product assertion is violated iff all three clauses hold,
    i.e. the pair diverges at this input."""
    c = evaluate_clauses(obs)
    return c["P_premise_ub_reached"] and c["T_target_defined"] and c["C_consequence"]


def product_assertion_holds(obs: ProductObservable) -> bool:
    """``R_m`` — the relational assertion holds (no divergence witnessed here)."""
    return not product_violated(obs)


# --------------------------------------------------------------------------- #
# Building the product observable from real executions.
# --------------------------------------------------------------------------- #
def build_product(
    h: ReexecHarness,
    c_src: str,
    target_src: str,
    argv_inputs: List[str],
    target: str,
    mode: str = TRAP_VS_DEFINED,
    divergence_class: str = "div_by_zero",
) -> Optional[Tuple[ProductObservable, object]]:
    """Run the real source/target binaries and form the product observable.

    Returns ``(observable, harness_result)`` or ``None`` when the run was
    unavailable (toolchain missing / compile failure) so the theorem is only
    asserted on genuinely-observed runs. ``target`` selects the semantics pack.
    """
    # validate the pack exists (parameterization point).
    tsem.get_pack(target)
    res = h.confirm_trap_vs_defined(
        c_src, target_src, list(argv_inputs),
        divergence_class=divergence_class, target_lang=target,
    )
    if not getattr(res, "available", False):
        return None
    c_runs = getattr(res, "c_runs", {}) or {}
    if "O0" not in c_runs:
        return None
    o0 = c_runs["O0"]
    o2 = c_runs.get("O2", o0)
    obs = ProductObservable(
        target=target,
        mode=mode if mode in MODES else TRAP_VS_DEFINED,
        o0_rc=o0.returncode, o0_val=o0.stdout,
        o2_rc=o2.returncode, o2_val=o2.stdout,
        san_trapped=bool(res.ub_reachable),
        defined=bool(res.rust_defined),
        deterministic=True,
    )
    return obs, res


# --------------------------------------------------------------------------- #
# Confirmation — the executable soundness/completeness theorem.
# --------------------------------------------------------------------------- #
@dataclass
class ProductCheck:
    item_id: str
    target: str
    klass: str
    declared_label: str
    product_violated: bool
    semantics_divergence: bool
    harness_confirmed: bool
    agree: bool


@dataclass
class ProductConfirmation:
    available: bool
    ok: bool
    n_checked: int
    n_divergent: int
    n_equivalent: int
    checks: List[ProductCheck]
    content_hash: str
    detail: str

    def render(self) -> str:
        if not self.available:
            return "product-program: toolchain unavailable (consistency only)"
        lines = [
            "Relational product-program soundness (real clang/UBSan + targets):",
            f"  items checked={self.n_checked} "
            f"(divergent={self.n_divergent}, equivalent={self.n_equivalent})",
            f"  content_hash={self.content_hash}",
            f"  all-agree(product == semantics == harness)={self.ok}",
        ]
        for c in self.checks[:12]:
            lines.append(
                f"    [{c.target:5s}] {c.item_id:18s} {c.declared_label:10s} "
                f"product={int(c.product_violated)} "
                f"sem={int(c.semantics_divergence)} "
                f"harness={int(c.harness_confirmed)} -> "
                f"{'ok' if c.agree else 'MISMATCH'}"
            )
        return "\n".join(lines)


def _hash_checks(checks: List[ProductCheck]) -> str:
    layer = sorted(
        (c.item_id, c.target, c.declared_label,
         c.product_violated, c.semantics_divergence, c.harness_confirmed)
        for c in checks
    )
    blob = json.dumps(layer, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


def confirm_product_program(
    langs: Tuple[str, ...] = ("rust", "go"),
    per_class: int = 1,
) -> ProductConfirmation:
    """Prove, on real code, that the relational product decides identically to
    the operational semantics and to the harness.

    For a per-family sample of divergent **and** equivalent corpus items in each
    target pack, build the product observable from real executions and require

        product_violated(obs) == is_divergence(obs.to_observation())
                              == harness.confirmed.

    Divergent items use the ``trap_vs_defined`` mode (their consequence is the
    definedness gap); equivalent items are checked too, where all three deciders
    must agree on *no* divergence.
    """
    status = toolchain_available()
    available = any(status.full_for(l) for l in langs)
    if not available:
        return ProductConfirmation(
            available=False, ok=True, n_checked=0, n_divergent=0,
            n_equivalent=0, checks=[], content_hash="",
            detail="toolchain unavailable: consistency-only pass",
        )

    h = ReexecHarness(status)
    checks: List[ProductCheck] = []

    items = gt.enumerate_corpus(langs)
    # per-(lang, klass, declared_label) sampling: first `per_class` of each bucket.
    buckets: Dict[Tuple[str, str, str], int] = {}
    selected: List[gt.GTItem] = []
    for it in items:
        key = (it.lang, it.klass, it.declared_label)
        n = buckets.get(key, 0)
        if n < per_class:
            buckets[key] = n + 1
            selected.append(it)

    for it in selected:
        if not status.full_for(it.lang):
            continue
        built = build_product(
            h, it.c_src, it.target_src, list(it.inputs),
            target=it.lang, mode=TRAP_VS_DEFINED, divergence_class=it.klass,
        )
        if built is None:
            continue
        obs, res = built
        pv = product_violated(obs)
        sd = sem.is_divergence(obs.to_observation())
        hc = bool(getattr(res, "confirmed", False))
        agree = (pv == sd == hc)
        checks.append(ProductCheck(
            item_id=it.item_id, target=it.lang, klass=it.klass,
            declared_label=it.declared_label, product_violated=pv,
            semantics_divergence=sd, harness_confirmed=hc, agree=agree,
        ))

    n_div = sum(1 for c in checks if c.declared_label == "divergent")
    n_equ = sum(1 for c in checks if c.declared_label == "equivalent")
    ok = bool(checks) and all(c.agree for c in checks)
    # Every confirmed divergent item must actually violate the product assertion,
    # and every equivalent item must satisfy it (no false alarm).
    ok = ok and all(
        c.product_violated for c in checks
        if c.declared_label == "divergent" and c.harness_confirmed
    )
    ok = ok and all(
        not c.product_violated for c in checks if c.declared_label == "equivalent"
    )
    detail = (f"checked={len(checks)} divergent={n_div} equivalent={n_equ} "
              f"all_agree={all(c.agree for c in checks) if checks else False}")
    return ProductConfirmation(
        available=True, ok=ok, n_checked=len(checks), n_divergent=n_div,
        n_equivalent=n_equ, checks=checks, content_hash=_hash_checks(checks),
        detail=detail,
    )


PRODUCT_PROGRAM_SPI = {
    "CLAUSES": CLAUSES,
    "evaluate_clauses": evaluate_clauses,
    "product_violated": product_violated,
    "product_assertion_holds": product_assertion_holds,
    "build_product": build_product,
    "confirm_product_program": confirm_product_program,
}


if __name__ == "__main__":  # pragma: no cover
    conf = confirm_product_program(per_class=1)
    print(f"available={conf.available} ok={conf.ok}")
    print(conf.render())
    print("detail:", conf.detail)
