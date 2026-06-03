"""Soundness-regression gate for built-in oracle plugins.

Step 137's rule is intentionally mechanical: adding a built-in oracle instance
without adding a corresponding soundness statement must fail CI.  The statement
is not a substitute for compiler re-execution or the Lean checker; it is the
metadata join point that binds a registered plugin instance to its declared
confirmation mode, its source-definedness premise, a theorem/claim reference, and
a concrete unit that exercises the real oracle implementation.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .catalogue import CATALOGUE, Definedness
from .plugin import ALL_ORACLES, DivergenceOracle, OracleVerdict

OracleKey = Tuple[str, str, str]


@dataclass(frozen=True)
class SoundnessStatement:
    """A checked statement for one registered oracle instance."""

    id: str
    source_lang: str
    target_lang: str
    divergence_class: str
    expected_confirmation_mode: str
    source_definedness: str
    evidence_kind: str
    theorem_refs: Tuple[str, ...]
    witness_unit: Mapping[str, object]
    statement: str

    @property
    def key(self) -> OracleKey:
        return (self.source_lang, self.target_lang, self.divergence_class)


@dataclass(frozen=True)
class WitnessProbe:
    key: OracleKey
    oracle_type: str
    inputs: Mapping[str, object]
    source_bytes: int
    target_bytes: int


@dataclass(frozen=True)
class SoundnessProblem:
    key: OracleKey
    kind: str
    detail: str


@dataclass(frozen=True)
class SoundnessAudit:
    registered: Tuple[OracleKey, ...]
    statements: Tuple[OracleKey, ...]
    probes: Tuple[WitnessProbe, ...] = ()
    problems: Tuple[SoundnessProblem, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.problems

    def detail(self) -> str:
        if self.ok:
            return (
                f"{len(self.registered)} oracle(s), "
                f"{len(self.statements)} statement(s), "
                f"{len(self.probes)} witness probe(s): ok"
            )
        lines = [
            f"{p.kind} {p.key[0]}->{p.key[1]}:{p.key[2]}: {p.detail}"
            for p in self.problems
        ]
        return "\n".join(lines)


_BASE_UNITS: Dict[str, Dict[str, object]] = {
    "array_oob": {"kind": "array_index", "length": 4},
    "atomic_ordering": {"kind": "atomic_litmus", "pattern": "store_buffering"},
    "bitfield_layout": {"kind": "bitfield_struct"},
    "div_by_zero": {"kind": "div", "width": 32, "signed": True},
    "enum_out_of_range": {"kind": "enum_cast"},
    "eval_order": {"kind": "unsequenced", "pattern": "postinc_read_add"},
    "fast_math_reassoc": {"kind": "fp_reassoc"},
    "float_cast_overflow": {"kind": "float_cast", "width": 32},
    "fp_contraction": {"kind": "fp_fma"},
    "intmin_div_neg1": {"kind": "div", "width": 32, "signed": True},
    "longjmp_vla": {"kind": "longjmp_vla", "preferred_bound": 4},
    "memcpy_overlap": {"kind": "memcpy_overlap", "buffer_len": 16},
    "pointer_provenance": {"kind": "pointer_offset", "width": 32},
    "restrict_violation": {"kind": "restrict_pair"},
    "shift_oob": {"kind": "shift", "width": 32, "value": 1},
    "signed_overflow": {
        "kind": "binop_const",
        "op": "add",
        "const": 1,
        "width": 32,
        "signed": True,
    },
    "signed_shift_sign_bit": {"kind": "sign_bit_shift", "width": 32},
    "strict_aliasing": {"kind": "type_pun"},
    "uninit_padding": {"kind": "uninit_padding"},
    "uninit_read": {
        "kind": "uninit_read",
        "storage": {"kind": "scalar"},
        "writes": [],
        "read": None,
    },
    "vla_bound": {"kind": "vla", "width": 32},
}

_MODE_BY_CLASS: Dict[str, str] = {
    "array_oob": "trap_vs_defined",
    "atomic_ordering": "model_level_divergence",
    "bitfield_layout": "defined_divergence",
    "div_by_zero": "trap_vs_defined",
    "enum_out_of_range": "defined_divergence",
    "eval_order": "static_ub_vs_defined",
    "fast_math_reassoc": "optimizer_exploited",
    "float_cast_overflow": "trap_vs_defined",
    "fp_contraction": "optimizer_exploited",
    "intmin_div_neg1": "trap_vs_defined",
    "longjmp_vla": "libc_contract_trap_vs_defined",
    "memcpy_overlap": "libc_contract_trap_vs_defined",
    "pointer_provenance": "trap_vs_defined",
    "restrict_violation": "optimizer_exploited",
    "shift_oob": "trap_vs_defined",
    "signed_overflow": "exploited",
    "signed_shift_sign_bit": "trap_vs_defined",
    "strict_aliasing": "optimizer_exploited",
    "uninit_padding": "uninit_padding",
    "uninit_read": "optimizer_exploited",
    "vla_bound": "trap_vs_defined",
}

_CLASS_CLAIM_REFS: Dict[str, Tuple[str, ...]] = {
    "array_oob": ("claim:C31-cve-corpus",),
    "atomic_ordering": ("claim:C37-product-program",),
    "bitfield_layout": ("claim:C15-abi-layout",),
    "div_by_zero": ("claim:C31-cve-corpus",),
    "enum_out_of_range": ("claim:C10-ub-catalogue",),
    "eval_order": ("claim:C30-eval-order",),
    "fast_math_reassoc": ("claim:C60-fast-math-reassociation-divergence",),
    "float_cast_overflow": ("claim:C59-float-cast-overflow-divergence",),
    "fp_contraction": ("claim:C27-solver-portfolio",),
    "intmin_div_neg1": ("claim:C31-cve-corpus",),
    "longjmp_vla": ("claim:C10-ub-catalogue",),
    "memcpy_overlap": ("claim:C24-libc-model",),
    "pointer_provenance": (
        "claim:C62-pointer-provenance-divergence",
        "lean:pointer_provenance_oracle_sound",
    ),
    "restrict_violation": ("claim:C61-restrict-violation-divergence",),
    "shift_oob": ("claim:C31-cve-corpus",),
    "signed_overflow": ("claim:C37-product-program", "lean:oracle_sound"),
    "signed_shift_sign_bit": ("claim:C10-ub-catalogue",),
    "strict_aliasing": (
        "claim:C41-mechanized-soundness",
        "lean:strict_aliasing_oracle_sound",
    ),
    "uninit_padding": ("claim:C10-ub-catalogue",),
    "uninit_read": ("claim:C12-uninit-definedness",),
    "vla_bound": ("claim:C58-vla-bound-divergence",),
}

_SOURCE_DEFINEDNESS_OVERRIDES: Dict[OracleKey, str] = {
    ("c", "rust", "eval_order"): Definedness.UNDEFINED.value,
    ("go", "rust", "intmin_div_neg1"): Definedness.DEFINED.value,
    ("rust", "c", "intmin_div_neg1"): Definedness.DEFINED.value,
}

_MODE_OVERRIDES: Dict[OracleKey, str] = {
    ("go", "rust", "intmin_div_neg1"): "defined_divergence",
    ("rust", "c", "intmin_div_neg1"): "source_defined_target_ub",
}


def _oracle_key(oracle: DivergenceOracle) -> OracleKey:
    return (oracle.source_lang, oracle.target_lang, oracle.divergence_class)


def _statement_id(source: str, target: str, cls: str) -> str:
    return f"S-{source}-{target}-{cls}".replace("_", "-")


def _unit_for(source: str, target: str, cls: str) -> Dict[str, object]:
    base = dict(_BASE_UNITS[cls])
    base.update({"source_lang": source, "target_lang": target, "probe": cls})
    return base


def _catalogue_definedness(source: str, target: str, cls: str) -> str:
    override = _SOURCE_DEFINEDNESS_OVERRIDES.get((source, target, cls))
    if override is not None:
        return override
    return CATALOGUE[cls].source_definedness.value


def _statement_for(source: str, target: str, cls: str) -> SoundnessStatement:
    mode = _MODE_OVERRIDES.get((source, target, cls), _MODE_BY_CLASS[cls])
    refs = ("claim:C37-product-program",) + _CLASS_CLAIM_REFS.get(cls, ())
    definedness = _catalogue_definedness(source, target, cls)
    pair = f"{source}->{target}"
    stmt = (
        f"The {pair} {cls} oracle is admitted only in confirmation mode "
        f"{mode!r}; its witness unit exercises the registered plugin, emits real "
        f"source/target code for the pair, and carries source_definedness="
        f"{definedness!r}. Positive claims are delegated to the referenced "
        f"traceability/mechanized evidence, not to unchecked registry convention."
    )
    return SoundnessStatement(
        id=_statement_id(source, target, cls),
        source_lang=source,
        target_lang=target,
        divergence_class=cls,
        expected_confirmation_mode=mode,
        source_definedness=definedness,
        evidence_kind=mode,
        theorem_refs=refs,
        witness_unit=_unit_for(source, target, cls),
        statement=stmt,
    )


def _built_in_statement_keys() -> Tuple[OracleKey, ...]:
    keys: List[OracleKey] = []
    keys.extend(("c", "rust", cls) for cls in (
        "signed_overflow",
        "shift_oob",
        "div_by_zero",
        "intmin_div_neg1",
        "array_oob",
        "strict_aliasing",
        "uninit_read",
        "fp_contraction",
        "vla_bound",
        "float_cast_overflow",
        "fast_math_reassoc",
        "restrict_violation",
        "pointer_provenance",
        "bitfield_layout",
        "enum_out_of_range",
        "memcpy_overlap",
        "eval_order",
        "longjmp_vla",
        "atomic_ordering",
        "uninit_padding",
    ))
    keys.extend(("c", "go", cls) for cls in (
        "signed_overflow",
        "shift_oob",
        "div_by_zero",
        "intmin_div_neg1",
        "array_oob",
        "strict_aliasing",
        "uninit_read",
        "vla_bound",
        "float_cast_overflow",
        "fast_math_reassoc",
        "restrict_violation",
        "pointer_provenance",
        "bitfield_layout",
        "enum_out_of_range",
        "memcpy_overlap",
        "longjmp_vla",
        "atomic_ordering",
        "uninit_padding",
    ))
    keys.extend(("c", "swift", cls) for cls in (
        "signed_overflow",
        "shift_oob",
        "div_by_zero",
        "intmin_div_neg1",
        "array_oob",
        "uninit_read",
    ))
    keys.extend(("c", "ocaml", cls) for cls in (
        "signed_overflow",
        "div_by_zero",
        "intmin_div_neg1",
        "array_oob",
    ))
    keys.extend(("c", "zig", cls) for cls in (
        "signed_overflow",
        "shift_oob",
        "div_by_zero",
        "intmin_div_neg1",
        "array_oob",
    ))
    keys.extend(("c", "wasm", cls) for cls in (
        "signed_overflow",
        "shift_oob",
        "div_by_zero",
        "intmin_div_neg1",
    ))
    keys.append(("c", "cpp", "signed_shift_sign_bit"))
    keys.append(("go", "rust", "intmin_div_neg1"))
    keys.append(("rust", "c", "intmin_div_neg1"))
    return tuple(keys)


SOUNDNESS_STATEMENTS: Tuple[SoundnessStatement, ...] = tuple(
    _statement_for(*key) for key in _built_in_statement_keys()
)


def _registered_oracles(oracles: Optional[Iterable[DivergenceOracle]] = None) -> Tuple[DivergenceOracle, ...]:
    if oracles is not None:
        return tuple(oracles)
    from . import oracles as _builtin_oracles  # noqa: F401  (registers plugins)
    return tuple(ALL_ORACLES)


def _validate_refs(statement: SoundnessStatement) -> List[SoundnessProblem]:
    problems: List[SoundnessProblem] = []
    claim_ids: Optional[set] = None
    lean_theorems: Optional[set] = None
    for ref in statement.theorem_refs:
        if ref.startswith("claim:"):
            if claim_ids is None:
                from . import traceability
                claim_ids = set(traceability.claim_ids())
            cid = ref.split(":", 1)[1]
            if cid not in claim_ids:
                problems.append(SoundnessProblem(
                    statement.key, "theorem-ref", f"unknown traceability claim {cid!r}"
                ))
        elif ref.startswith("lean:"):
            if lean_theorems is None:
                from . import mechanized_soundness as ms
                lean_theorems = set(ms.REQUIRED_THEOREMS)
            thm = ref.split(":", 1)[1]
            if thm not in lean_theorems:
                problems.append(SoundnessProblem(
                    statement.key, "theorem-ref", f"unknown Lean theorem {thm!r}"
                ))
        else:
            problems.append(SoundnessProblem(
                statement.key, "theorem-ref", f"unsupported theorem ref {ref!r}"
            ))
    return problems


def _probe_witness(
    oracle: DivergenceOracle,
    statement: SoundnessStatement,
) -> Tuple[Optional[WitnessProbe], List[SoundnessProblem]]:
    key = statement.key
    problems: List[SoundnessProblem] = []
    try:
        result = oracle.find_divergence(dict(statement.witness_unit))
    except Exception as exc:  # pragma: no cover - exercised by negative tests if needed
        return None, [SoundnessProblem(key, "witness", f"oracle raised {exc!r}")]

    if result.verdict is not OracleVerdict.DIVERGENT:
        problems.append(SoundnessProblem(
            key, "witness", f"expected DIVERGENT, got {result.verdict.value}"
        ))
        return None, problems
    ce = result.counterexample
    if ce is None:
        problems.append(SoundnessProblem(key, "witness", "divergent result has no counterexample"))
        return None, problems
    if ce.source_lang != statement.source_lang or ce.target_lang != statement.target_lang:
        problems.append(SoundnessProblem(
            key,
            "witness",
            f"counterexample pair {ce.source_lang}->{ce.target_lang} does not match statement",
        ))
    if ce.divergence_class != statement.divergence_class:
        problems.append(SoundnessProblem(
            key, "witness", f"counterexample class {ce.divergence_class!r} does not match"
        ))
    if ce.source_definedness != statement.source_definedness:
        problems.append(SoundnessProblem(
            key,
            "witness",
            f"source_definedness {ce.source_definedness!r} != {statement.source_definedness!r}",
        ))
    if not ce.source_snippet or not ce.target_snippet:
        problems.append(SoundnessProblem(key, "witness", "counterexample snippets must be non-empty"))
    if not ce.definedness_witness or not ce.divergence_witness:
        problems.append(SoundnessProblem(key, "witness", "counterexample witness prose is missing"))
    if problems:
        return None, problems
    return WitnessProbe(
        key=key,
        oracle_type=f"{type(oracle).__module__}.{type(oracle).__name__}",
        inputs=dict(ce.inputs),
        source_bytes=len(ce.source_snippet.encode("utf-8")),
        target_bytes=len(ce.target_snippet.encode("utf-8")),
    ), []


def audit_soundness_statements(
    *,
    oracles: Optional[Iterable[DivergenceOracle]] = None,
    statements: Sequence[SoundnessStatement] = SOUNDNESS_STATEMENTS,
    probe_witnesses: bool = True,
) -> SoundnessAudit:
    """Audit registered oracle instances against static soundness statements."""

    registered_oracles = _registered_oracles(oracles)
    registered_by_key: Dict[OracleKey, DivergenceOracle] = {}
    problems: List[SoundnessProblem] = []
    for oracle in registered_oracles:
        key = _oracle_key(oracle)
        if key in registered_by_key:
            problems.append(SoundnessProblem(key, "registry", "duplicate registered oracle key"))
        registered_by_key[key] = oracle

    statement_by_key: Dict[OracleKey, SoundnessStatement] = {}
    for st in statements:
        if st.key in statement_by_key:
            problems.append(SoundnessProblem(st.key, "statement", "duplicate soundness statement"))
        statement_by_key[st.key] = st
        if not st.id or not st.statement or len(st.statement) < 80:
            problems.append(SoundnessProblem(st.key, "statement", "statement is too terse"))
        if st.divergence_class not in CATALOGUE:
            problems.append(SoundnessProblem(st.key, "statement", "divergence class missing from catalogue"))
        if st.divergence_class not in _BASE_UNITS:
            problems.append(SoundnessProblem(st.key, "statement", "no witness unit template"))
        problems.extend(_validate_refs(st))

    for key in sorted(set(registered_by_key) - set(statement_by_key)):
        problems.append(SoundnessProblem(key, "coverage", "registered oracle has no soundness statement"))
    for key in sorted(set(statement_by_key) - set(registered_by_key)):
        problems.append(SoundnessProblem(key, "coverage", "soundness statement has no registered oracle"))

    probes: List[WitnessProbe] = []
    for key in sorted(set(registered_by_key) & set(statement_by_key)):
        oracle = registered_by_key[key]
        st = statement_by_key[key]
        if oracle.confirmation_mode != st.expected_confirmation_mode:
            problems.append(SoundnessProblem(
                key,
                "mode",
                f"registry mode {oracle.confirmation_mode!r} != statement mode {st.expected_confirmation_mode!r}",
            ))
        if st.evidence_kind != st.expected_confirmation_mode:
            problems.append(SoundnessProblem(key, "statement", "evidence kind must match confirmation mode"))
        if probe_witnesses:
            probe, pproblems = _probe_witness(oracle, st)
            problems.extend(pproblems)
            if probe is not None:
                probes.append(probe)

    return SoundnessAudit(
        registered=tuple(sorted(registered_by_key)),
        statements=tuple(sorted(statement_by_key)),
        probes=tuple(probes),
        problems=tuple(problems),
    )


def confirm_soundness_registry(*, probe_witnesses: bool = True) -> SoundnessAudit:
    """Run the default built-in-oracle soundness-regression gate."""

    return audit_soundness_statements(probe_witnesses=probe_witnesses)


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the gate finds a problem")
    parser.add_argument(
        "--no-witness-probes",
        action="store_true",
        help="check registry/metadata coverage only; do not call find_divergence",
    )
    args = parser.parse_args(argv)
    audit = confirm_soundness_registry(probe_witnesses=not args.no_witness_probes)
    print(audit.detail())
    if args.check and not audit.ok:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_main())
