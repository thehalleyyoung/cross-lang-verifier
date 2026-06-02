"""
Tests for the cross-language divergence-oracle framework (src/ub_oracle).

These tests prove the anchor (C->Rust signed-overflow UB divergence) end-to-end.
The re-execution tests compile and run *real* C (under UndefinedBehaviorSanitizer)
and *real* Rust; they skip cleanly if the toolchain is unavailable.
"""

from __future__ import annotations

import json

import pytest

from src.ub_oracle import (
    CATALOGUE,
    Definedness,
    Severity,
    c_ub_classes,
    entry_for,
    list_oracles,
    get_oracle,
    Counterexample,
    REPLAY_SCHEMA_VERSION,
    ReexecHarness,
    ToolchainStatus,
    toolchain_available,
)
from src.ub_oracle import oracles  # noqa: F401  (registers plugins)
from src.ub_oracle.plugin import OracleVerdict
from src.ub_oracle.diff_testing import measure_fuzzing_gap, overflow_input_count


# ── catalogue (steps 12-13) ──────────────────────────────────────────────────

def test_catalogue_has_core_ub_classes():
    for key in ("signed_overflow", "shift_oob", "div_by_zero",
                "array_oob", "null_deref"):
        assert key in CATALOGUE, f"missing catalogue entry {key}"


def test_catalogue_entries_are_well_formed():
    for key, e in CATALOGUE.items():
        assert e.cls.key == key
        assert isinstance(e.severity, Severity)
        assert isinstance(e.source_definedness, Definedness)
        assert e.source_rule and e.target_rule
        assert e.c_standard_ref, f"{key} lacks a C-standard citation"


def test_ub_subtable_is_nonempty_and_consistent():
    ub = c_ub_classes()
    assert len(ub) >= 6
    for e in ub:
        assert e.is_ub_rooted()
        assert e.source_definedness is Definedness.UNDEFINED


def test_signed_overflow_maps_to_defined_rust_outcome():
    e = entry_for("signed_overflow")
    assert e.is_ub_rooted()
    assert e.rust_outcome.value in ("wrap", "panic")


# ── plugin registry (step 11) ────────────────────────────────────────────────

def test_signed_overflow_oracle_registered():
    assert "signed_overflow" in list_oracles()


def test_oracle_applies_only_to_relevant_units():
    orc = get_oracle("signed_overflow")
    assert orc.applies_to({"kind": "binop_const", "op": "add",
                           "const": 1, "width": 32, "signed": True})
    assert not orc.applies_to({"kind": "binop_const", "op": "xor",
                               "const": 1, "width": 32, "signed": True})
    assert not orc.applies_to({"kind": "something_else"})


# ── replay format (step 9) ───────────────────────────────────────────────────

def test_counterexample_roundtrips_json():
    ce = Counterexample(
        divergence_class="signed_overflow", source_lang="c", target_lang="rust",
        inputs={"x": 2147483647}, source_snippet="int f...", target_snippet="fn f...",
        divergence_witness="w", definedness_witness="d",
    )
    s = ce.to_json()
    back = Counterexample.from_json(s)
    assert back.to_dict() == ce.to_dict()
    assert json.loads(s)["schema_version"] == REPLAY_SCHEMA_VERSION


# ── the oracle finds a witness via Z3 (steps 13-14) ──────────────────────────

@pytest.mark.parametrize("op,c,width,expected", [
    ("add", 1, 32, 2147483647),
    ("sub", 1, 32, -2147483648),
    ("add", 1, 64, 9223372036854775807),
])
def test_oracle_finds_overflow_witness(op, c, width, expected):
    orc = get_oracle("signed_overflow")
    unit = {"kind": "binop_const", "op": op, "const": c,
            "width": width, "var": "x", "signed": True}
    res = orc.find_divergence(unit)
    assert res.verdict is OracleVerdict.DIVERGENT
    assert res.counterexample is not None
    # The witness must actually overflow.
    x = res.counterexample.inputs["x"]
    real = x + c if op == "add" else x - c
    lo, hi = -(1 << (width - 1)), (1 << (width - 1)) - 1
    assert real < lo or real > hi
    assert x == expected


def test_oracle_generates_compilable_looking_sources():
    orc = get_oracle("signed_overflow")
    res = orc.find_divergence({"kind": "binop_const", "op": "add",
                               "const": 1, "width": 32, "var": "x", "signed": True})
    ce = res.counterexample
    assert "int f(int x)" in ce.source_snippet
    assert "wrapping_add" in ce.target_snippet
    assert ce.source_definedness == Definedness.UNDEFINED.value


# ── differential-testing gap (steps 22, 24) ──────────────────────────────────

def test_overflow_input_count_is_exact():
    # exactly `c` inputs overflow for x+c (c>0) at any width
    assert overflow_input_count("add", 1, 32) == 1
    assert overflow_input_count("add", 7, 32) == 7
    assert overflow_input_count("sub", 1, 32) == 1


def test_fuzzer_misses_what_oracle_finds():
    gap = measure_fuzzing_gap("add", 1, 32, trials=200_000, seed=0)
    assert gap.overflow_inputs == 1
    assert gap.empirical_hits == 0          # needle in 4.3e9 haystack
    assert gap.oracle_hit_probability == 1.0
    assert gap.hit_probability < 1e-9


# ── ground-truth re-execution against real compilers (step 10) ───────────────

_TC = toolchain_available()
_requires_toolchain = pytest.mark.skipif(
    not _TC.full, reason=f"needs C+UBSan+rustc toolchain ({_TC})")


@_requires_toolchain
@pytest.mark.parametrize("op,c,width", [
    ("add", 1, 32),
    ("sub", 1, 32),
    ("add", 7, 32),
    ("add", 1, 64),
])
def test_ub_divergence_confirmed_against_real_compilers(op, c, width):
    orc = get_oracle("signed_overflow")
    unit = {"kind": "binop_const", "op": op, "const": c,
            "width": width, "var": "x", "signed": True}
    res = orc.confirm(orc.find_divergence(unit), ReexecHarness(_TC))
    rr = res.reexec
    assert rr.available
    assert rr.ub_reachable, "UBSan should trap on the witness"
    assert rr.ub_consequential, "O0 and O2 must disagree (UB exploited)"
    assert rr.rust_defined, "Rust must produce a defined value"
    assert rr.confirmed
    # And the counterexample is now marked confirmed with observations.
    ce = res.counterexample
    assert ce.confirmed
    assert ce.source_observed["O0"] != ce.source_observed["O2"]


@_requires_toolchain
def test_equivalent_translation_is_not_flagged_by_reexec():
    """A *correct* (wrapping-everywhere) translation must NOT show O0/O2 divergence."""
    harness = ReexecHarness(_TC)
    # Both sides wrap: no UB in C, so no consequential divergence.
    c_src = (
        "#include <stdio.h>\n#include <stdlib.h>\n"
        "int f(unsigned x){ return (x + 1u) > x; }\n"
        "int main(int c,char**v){ if(c<2) return 2; "
        "unsigned x=(unsigned)strtoull(v[1],0,10); printf(\"%d\\n\",f(x)); return 0; }\n"
    )
    rust_src = (
        "fn f(x:u32)->i32{ (x.wrapping_add(1) > x) as i32 }\n"
        "fn main(){ let x:u32 = std::env::args().nth(1).unwrap().parse().unwrap(); "
        "println!(\"{}\", f(x)); }\n"
    )
    rr = harness.confirm_ub_divergence(c_src, rust_src, ["4294967295"])
    # Unsigned wraparound is *defined* in C, so the sanitizer must not trap and
    # O0/O2 must agree -> not confirmed as a divergence.
    assert rr.available
    assert not rr.ub_reachable
    assert not rr.confirmed


# ── integer-model oracles: shift / div-by-zero / INT_MIN÷-1 (step 18) ────────
from src.ub_oracle import verify_unit, VerifyVerdict, applicable_oracles
from src.ub_oracle.metrics import (
    evaluate_symbolic,
    evaluate_confirmed,
    POSITIVE_CASES,
    NEGATIVE_CASES,
    ALL_CASES,
)


def test_integer_ub_oracles_registered():
    names = set(list_oracles())
    assert {"shift_oob", "div_by_zero", "intmin_div_neg1"} <= names


@pytest.mark.parametrize("key,unit", [
    ("shift_oob", {"kind": "shift", "width": 32}),
    ("shift_oob", {"kind": "shift", "width": 64}),
    ("div_by_zero", {"kind": "div", "width": 32}),
    ("div_by_zero", {"kind": "rem", "width": 32}),
    ("intmin_div_neg1", {"kind": "div", "width": 32, "signed": True}),
])
def test_integer_oracle_finds_witness_symbolically(key, unit):
    orc = get_oracle(key)
    res = orc.find_divergence(unit)
    assert res.verdict is OracleVerdict.DIVERGENT
    ce = res.counterexample
    assert ce is not None and ce.source_snippet and ce.target_snippet
    # divergence is a definedness divergence: source is UB.
    assert ce.source_definedness == Definedness.UNDEFINED.value


def test_shift_witness_is_smallest_out_of_range_amount():
    res = get_oracle("shift_oob").find_divergence({"kind": "shift", "width": 32})
    assert res.counterexample.inputs["s"] == 32  # exactly the bit width


def test_intmin_div_witness_is_the_unique_overflow_pair():
    res = get_oracle("intmin_div_neg1").find_divergence(
        {"kind": "div", "width": 32, "signed": True})
    inp = res.counterexample.inputs
    assert inp["a"] == -(2 ** 31) and inp["b"] == -1


def test_probe_routes_div_unit_to_a_single_oracle():
    # A bare div unit is understood by both div oracles ...
    assert len(applicable_oracles({"kind": "div", "width": 32})) == 2
    # ... but a probed unit is routed to exactly one.
    only = applicable_oracles({"kind": "div", "width": 32, "probe": "div_by_zero"})
    assert [o.divergence_class for o in only] == ["div_by_zero"]


def test_integer_oracles_declare_trap_vs_defined_mode():
    for key in ("shift_oob", "div_by_zero", "intmin_div_neg1"):
        assert get_oracle(key).confirmation_mode == "trap_vs_defined"


# ── sound-for-divergence verify entry point (steps 5 & 33) ───────────────────
def test_verify_unit_never_claims_equivalence():
    for verdict in VerifyVerdict:
        assert verdict.claims_equivalence is False


def test_verify_unit_not_covered_is_loud():
    r = verify_unit({"kind": "string_concat", "width": 32})
    assert r.verdict is VerifyVerdict.NOT_COVERED
    assert not r.is_sound_claim
    assert "NOT COVERED" in r.banner()


def test_verify_unit_no_divergence_found_is_not_equivalence():
    # add of 0 never overflows: the oracle applies but finds nothing.
    r = verify_unit({"kind": "binop_const", "op": "add", "const": 0,
                     "width": 32, "var": "x", "signed": True})
    assert r.verdict is VerifyVerdict.NO_DIVERGENCE_FOUND
    assert not r.verdict.claims_equivalence
    assert "NOT a proof of equivalence" in r.banner()


def test_verify_unit_candidate_when_confirmation_disabled():
    # With confirm disabled, a symbolic witness must NOT be asserted as a
    # divergence (soundness): it is only a CANDIDATE.
    r = verify_unit({"kind": "shift", "width": 32}, confirm=False)
    assert r.verdict is VerifyVerdict.CANDIDATE
    assert not r.verdict.claims_divergence


# ── precision / recall harness (step 23) ─────────────────────────────────────
def test_symbolic_precision_recall_is_perfect():
    m = evaluate_symbolic()
    assert m["num_positive"] == len(POSITIVE_CASES)
    assert m["num_negative"] == len(NEGATIVE_CASES)
    assert m["overall"]["precision"] == 1.0
    assert m["overall"]["recall"] == 1.0
    assert m["overall"]["fp"] == 0 and m["overall"]["fn"] == 0
    for key, sc in m["per_class"].items():
        assert sc["precision"] == 1.0, key
        assert sc["recall"] == 1.0, key


def test_symbolic_metrics_cover_every_registered_oracle():
    m = evaluate_symbolic()
    assert set(m["per_class"]) == set(list_oracles())


# ── toolchain-backed confirmations for the new classes ───────────────────────
@_requires_toolchain
@pytest.mark.parametrize("key,unit", [
    ("shift_oob", {"kind": "shift", "width": 32}),
    ("div_by_zero", {"kind": "div", "width": 32}),
    ("intmin_div_neg1", {"kind": "div", "width": 32, "signed": True}),
])
def test_trap_vs_defined_confirmed_against_real_compilers(key, unit):
    orc = get_oracle(key)
    res = orc.confirm(orc.find_divergence(unit), ReexecHarness(_TC))
    rr = res.reexec
    assert rr.available
    assert rr.mode == "trap_vs_defined"
    assert rr.ub_reachable, "UBSan must trap on the witness (C is UB)"
    assert rr.rust_defined, "Rust must be defined & deterministic"
    assert rr.confirmed
    assert res.counterexample.confirmed


@_requires_toolchain
def test_division_negative_control_is_not_confirmed():
    # b = 1 (a defined divisor): C is NOT UB and Rust is defined -> no trap,
    # so the trap_vs_defined confirmation must fail.
    harness = ReexecHarness(_TC)
    c_src = (
        "#include <stdio.h>\n#include <stdlib.h>\n"
        "int f(int a,int b){ return a/b; }\n"
        "int main(int c,char**v){ if(c<3) return 2; "
        "int a=atoi(v[1]),b=atoi(v[2]); printf(\"%d\\n\",f(a,b)); return 0; }\n"
    )
    rust_src = (
        "fn f(a:i32,b:i32)->i32{ a/b }\n"
        "fn main(){ let v:Vec<String>=std::env::args().collect(); "
        "let a:i32=v[1].parse().unwrap(); let b:i32=v[2].parse().unwrap(); "
        "println!(\"{}\",f(a,b)); }\n"
    )
    rr = harness.confirm_trap_vs_defined(c_src, rust_src, ["7", "1"], "div_by_zero")
    assert rr.available
    assert not rr.ub_reachable
    assert not rr.confirmed


@_requires_toolchain
def test_confirmed_precision_recall_is_perfect():
    m = evaluate_confirmed(ReexecHarness(_TC))
    assert m["overall"]["precision"] == 1.0
    assert m["overall"]["recall"] == 1.0
    for key, sc in m["per_class"].items():
        assert sc["precision"] == 1.0, key
        assert sc["recall"] == 1.0, key


@_requires_toolchain
def test_verify_unit_confirms_divergence_end_to_end():
    r = verify_unit({"kind": "div", "width": 32, "probe": "div_by_zero"})
    assert r.verdict is VerifyVerdict.DIVERGENT
    assert r.is_sound_claim
    assert r.divergence is not None
    assert r.divergence.reexec.confirmed


# ── memory-shape oracles: array-OOB & strict-aliasing (steps 15-16) ──────────
def test_memory_shape_oracles_registered():
    names = set(list_oracles())
    assert {"array_oob", "strict_aliasing"} <= names


def test_oob_oracle_finds_smallest_out_of_bounds_index():
    res = get_oracle("array_oob").find_divergence(
        {"kind": "array_index", "length": 4})
    assert res.verdict is OracleVerdict.DIVERGENT
    assert res.counterexample.inputs["i"] == 4  # first OOB index
    assert res.counterexample.source_definedness == Definedness.UNDEFINED.value


def test_oob_oracle_confirmation_mode():
    assert get_oracle("array_oob").confirmation_mode == "trap_vs_defined"


def test_aliasing_oracle_finds_distinguishing_values():
    res = get_oracle("strict_aliasing").find_divergence({"kind": "type_pun"})
    assert res.verdict is OracleVerdict.DIVERGENT
    # A must differ from the low 32 bits of B (that is what the optimizer can
    # resolve two different ways).
    assert "A=" in res.detail and "B=" in res.detail


def test_aliasing_oracle_confirmation_mode():
    assert get_oracle("strict_aliasing").confirmation_mode == "optimizer_exploited"


def test_aliasing_unit_not_matched_by_arithmetic_oracles():
    only = applicable_oracles({"kind": "type_pun"})
    assert [o.divergence_class for o in only] == ["strict_aliasing"]


@_requires_toolchain
def test_oob_confirmed_against_real_compilers():
    orc = get_oracle("array_oob")
    res = orc.confirm(orc.find_divergence({"kind": "array_index", "length": 4}),
                      ReexecHarness(_TC))
    rr = res.reexec
    assert rr.available and rr.mode == "trap_vs_defined"
    assert rr.ub_reachable and rr.rust_defined and rr.confirmed
    assert res.counterexample.confirmed


@_requires_toolchain
def test_strict_aliasing_confirmed_via_optimizer_exploited():
    orc = get_oracle("strict_aliasing")
    res = orc.confirm(orc.find_divergence({"kind": "type_pun"}), ReexecHarness(_TC))
    rr = res.reexec
    assert rr.available and rr.mode == "optimizer_exploited"
    # the very same C source diverges across the two conforming builds ...
    assert rr.c_runs["A"].stdout != rr.c_runs["B"].stdout
    # ... while Rust is defined & deterministic.
    assert rr.rust_defined and rr.confirmed


@_requires_toolchain
def test_optimizer_exploited_negative_control():
    """A well-defined C program must NOT diverge across -O0 / -O2."""
    harness = ReexecHarness(_TC)
    c_src = (
        "#include <stdio.h>\n#include <stdlib.h>\n"
        "int main(int c,char**v){ if(c<2) return 2; int x=atoi(v[1]); "
        "printf(\"%d\\n\", x*2); return 0; }\n"
    )
    rust_src = (
        "fn main(){ let x:i32=std::env::args().nth(1).unwrap().parse().unwrap(); "
        "println!(\"{}\", x.wrapping_mul(2)); }\n"
    )
    rr = harness.confirm_optimizer_exploited(c_src, rust_src, ["3"], "strict_aliasing")
    assert rr.available
    assert not rr.ub_consequential   # O0 and O2 agree
    assert not rr.confirmed

# ── floating-point contraction oracle (step 19) ──────────────────────────────

def test_fp_contraction_in_catalogue():
    e = entry_for("fp_contraction")
    assert e.source_definedness is Definedness.UNSPECIFIED
    assert e.rust_outcome.value == "defined_value"
    assert e.c_standard_ref


def test_fp_contraction_symbolic_find():
    orc = get_oracle("fp_contraction")
    assert orc.applies_to({"kind": "fp_fma"})
    res = orc.find_divergence({"kind": "fp_fma", "probe": "fp_contraction"})
    assert res.verdict is OracleVerdict.DIVERGENT
    ce = res.counterexample
    assert set(ce.inputs) == {"a", "b", "c"}
    for v in ce.inputs.values():
        assert isinstance(v, float)
    # fma(a,b,c) must actually differ from round(round(a*b)+c)
    import math
    a, b, c = ce.inputs["a"], ce.inputs["b"], ce.inputs["c"]
    assert math.fma(a, b, c) != (a * b) + c


def test_fp_contraction_does_not_apply_to_other_kinds():
    orc = get_oracle("fp_contraction")
    assert not orc.applies_to({"kind": "array_index", "length": 4})
    assert not orc.applies_to({"kind": "fp_fma", "probe": "signed_overflow"})


@_requires_toolchain
def test_fp_contraction_confirmed_off_vs_fast():
    orc = get_oracle("fp_contraction")
    res = orc.confirm(
        orc.find_divergence({"kind": "fp_fma", "probe": "fp_contraction"}),
        ReexecHarness(_TC))
    rr = res.reexec
    assert rr.available and rr.mode == "optimizer_exploited"
    # -ffp-contract=off and -ffp-contract=fast disagree on the same source ...
    assert rr.c_runs["A"].stdout != rr.c_runs["B"].stdout
    # ... while Rust (always two roundings) is defined & deterministic.
    assert rr.rust_defined and rr.confirmed

# ── honest aggregate reporting + SARIF + CLI (steps 47, 55, 57) ───────────────

from src.ub_oracle.report import aggregate_reports, to_sarif, pair_of  # noqa: E402
from src.ub_oracle.verify import verify_unit, VerifyVerdict  # noqa: E402
from src.ub_oracle import cli as _cli  # noqa: E402


def _reports_no_confirm():
    """A representative spread of verdicts without needing the toolchain."""
    units = [
        {"name": "ovf", "kind": "binop_const", "op": "add", "const": 1,
         "width": 32, "var": "x", "signed": True, "probe": "signed_overflow",
         "source_lang": "c", "target_lang": "rust"},
        {"name": "noovf", "kind": "binop_const", "op": "add", "const": 0,
         "width": 32, "var": "x", "signed": True, "probe": "signed_overflow",
         "source_lang": "c", "target_lang": "rust"},
        {"name": "opaque", "kind": "string_concat", "width": 32,
         "source_lang": "c", "target_lang": "rust"},
        {"name": "go_unit", "kind": "binop_const", "op": "add", "const": 1,
         "width": 32, "var": "x", "signed": True,
         "source_lang": "go", "target_lang": "rust"},
    ]
    return units, [verify_unit(u, confirm=False) for u in units]


def test_declared_unsupported_pair_is_not_covered():
    # a go->rust unit must match NO oracle (honest pair gating), not be treated
    # as the c->rust anchor.
    rep = verify_unit({"kind": "binop_const", "op": "add", "const": 1,
                       "width": 32, "var": "x", "signed": True,
                       "source_lang": "go", "target_lang": "rust"},
                      confirm=False)
    assert rep.verdict is VerifyVerdict.NOT_COVERED
    assert pair_of(rep) == "go->rust"


def test_aggregate_reports_buckets_and_fractions():
    _, reports = _reports_no_confirm()
    agg = aggregate_reports(reports)
    ov = agg["overall"]
    assert ov["total"] == 4
    # ovf -> CANDIDATE (symbolic, not confirmed) => abstained
    # noovf -> NO_DIVERGENCE_FOUND => decided
    # opaque (c->rust, unknown kind) -> NOT_COVERED => abstained
    # go_unit -> NOT_COVERED => abstained
    assert ov["candidate"] == 1
    assert ov["no_divergence_found"] == 1
    assert ov["not_covered"] == 2
    assert ov["decided"] == 1 and ov["abstained"] == 3 and ov["unknown"] == 0
    assert abs(ov["decided_fraction"] - 0.25) < 1e-9
    # per-pair breakdown surfaces the uncovered go pair.
    assert agg["by_pair"]["go->rust"]["not_covered"] == 1
    assert "equivalence" in agg["disclaimer"].lower()


def test_to_sarif_shape_for_candidate_and_clean():
    _, reports = _reports_no_confirm()
    doc = to_sarif(reports)
    assert doc["version"] == "2.1.0" and "$schema" in doc
    run = doc["runs"][0]
    assert run["tool"]["driver"]["name"] == "cross-lang-verifier"
    ruleids = {r["id"] for r in run["tool"]["driver"]["rules"]}
    # only the CANDIDATE (warning) becomes a finding here; clean/not-covered do not.
    assert len(run["results"]) == 1
    res = run["results"][0]
    assert res["level"] == "warning"
    assert res["ruleId"] in ruleids
    assert res["message"]["text"]
    assert res["locations"][0]["logicalLocations"][0]["name"] == "ovf"
    assert res["partialFingerprints"]


def test_to_sarif_physical_location_only_when_declared():
    rep = verify_unit({"name": "with_loc", "kind": "binop_const", "op": "add",
                       "const": 1, "width": 32, "var": "x", "signed": True,
                       "probe": "signed_overflow", "source_file": "src/a.c",
                       "line": 42}, confirm=False)
    res = to_sarif([rep])["runs"][0]["results"][0]
    phys = res["locations"][0]["physicalLocation"]
    assert phys["artifactLocation"]["uri"] == "src/a.c"
    assert phys["region"]["startLine"] == 42


def test_cli_text_and_exit_code(tmp_path, capsys):
    manifest = tmp_path / "units.json"
    manifest.write_text(json.dumps({"units": [
        {"name": "ovf", "kind": "binop_const", "op": "add", "const": 1,
         "width": 32, "var": "x", "signed": True, "probe": "signed_overflow"},
        {"name": "go_unit", "kind": "binop_const", "op": "add", "const": 1,
         "width": 32, "var": "x", "signed": True,
         "source_lang": "go", "target_lang": "rust"},
    ]}))
    # without confirmation the overflow unit is CANDIDATE, not DIVERGENT, so the
    # default (fail-on divergent) yields exit 0.
    rc = _cli.run(["--units", str(manifest), "--no-confirm", "--color", "never"])
    out = capsys.readouterr().out
    assert "CANDIDATE" in out and "NOT-COVERED" in out
    assert "Summary" in out and "by pair" in out
    assert rc == 0
    # but --fail-on candidate must flip the exit code.
    rc2 = _cli.run(["--units", str(manifest), "--no-confirm", "--color", "never",
                    "--fail-on", "candidate"])
    capsys.readouterr()
    assert rc2 == 1


def test_cli_json_format_and_sarif_file(tmp_path, capsys):
    manifest = tmp_path / "units.json"
    manifest.write_text(json.dumps([
        {"name": "noovf", "kind": "binop_const", "op": "add", "const": 0,
         "width": 32, "var": "x", "signed": True, "probe": "signed_overflow"},
    ]))
    sarif = tmp_path / "out.sarif"
    rc = _cli.run(["--units", str(manifest), "--no-confirm", "--format", "json",
                   "--sarif", str(sarif)])
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["summary"]["overall"]["total"] == 1
    assert parsed["units"][0]["verdict"] == "no_divergence_found"
    assert json.loads(sarif.read_text())["version"] == "2.1.0"
    assert rc == 0


def test_cli_bad_manifest_returns_2(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    rc = _cli.run(["--units", str(bad)])
    assert rc == 2

# ── ablation study (step 53) ─────────────────────────────────────────────────

from src.ub_oracle.ablation import ablate_each_class  # noqa: E402
from src.ub_oracle.metrics import LabeledCase  # noqa: E402


def _small_positive_set():
    # a fast subset (no FP Z3 search) covering three independent classes.
    return [
        LabeledCase("ovf_add1_w32",
                    {"kind": "binop_const", "op": "add", "const": 1, "width": 32,
                     "var": "x", "signed": True, "probe": "signed_overflow"},
                    "signed_overflow"),
        LabeledCase("shift_w32",
                    {"kind": "shift", "width": 32, "probe": "shift_oob"},
                    "shift_oob"),
        LabeledCase("oob_len4",
                    {"kind": "array_index", "length": 4, "probe": "array_oob"},
                    "array_oob"),
    ]


def test_ablation_each_class_misses_only_its_own_positives():
    pos = _small_positive_set()
    rep = ablate_each_class(pos)
    assert rep["recall_full"] == 1.0
    # disabling each implemented class drops recall and misses exactly that
    # class's own positive(s), with NO cross-class leakage.
    for cls in ("signed_overflow", "shift_oob", "array_oob"):
        row = rep["per_class"][cls]
        assert row["recall_drop"] > 0, f"{cls} should be load-bearing"
        assert row["cross_class_leak"] == [], f"{cls} leaked into another class"
        owned = {c.name for c in pos if c.truth_class == cls}
        assert set(row["newly_missed"]) == owned


def test_ablation_unrelated_class_is_inert():
    # disabling a class with no positive in the set must not change recall.
    pos = _small_positive_set()
    rep = ablate_each_class(pos)
    row = rep["per_class"]["div_by_zero"]
    assert row["recall_drop"] == 0.0
    assert row["newly_missed"] == []


# ── head-to-head vs differential testing (step 48) ───────────────────────────

from src.ub_oracle.headtohead import (  # noqa: E402
    FuzzUnit, head_to_head, differential_fuzz, default_units,
)


@_requires_toolchain
def test_head_to_head_sparse_ub_gap_and_dense_ub_parity():
    units = [
        FuzzUnit("ovf_add1_w32",
                 {"kind": "binop_const", "op": "add", "const": 1, "width": 32,
                  "var": "x", "signed": True, "probe": "signed_overflow"},
                 "signed_overflow", {"x": ("int", -(2 ** 31), 2 ** 31 - 1)}),
        FuzzUnit("shift_w32",
                 {"kind": "shift", "width": 32, "probe": "shift_oob"},
                 "shift_oob",
                 {"x": ("int", -(2 ** 31), 2 ** 31 - 1),
                  "s": ("int", 0, 2 ** 16 - 1)}),
    ]
    rep = head_to_head(units, trials=250, seed=0, harness=ReexecHarness(_TC))
    by = {r["name"]: r for r in rep["rows"]}
    # the oracle confirms both divergences via real re-execution.
    assert by["ovf_add1_w32"]["oracle_confirmed"]
    assert by["shift_w32"]["oracle_confirmed"]
    # equal-budget differential testing: misses the sparse signed-overflow UB ...
    assert by["ovf_add1_w32"]["fuzz_hits"] == 0
    assert by["ovf_add1_w32"]["false_negative_gap"] is True
    # ... but finds the dense out-of-range-shift UB immediately (not rigged).
    assert by["shift_w32"]["fuzz_found"] is True
    assert by["shift_w32"]["false_negative_gap"] is False
    assert "ovf_add1_w32" in rep["false_negative_gap_units"]
    assert "shift_w32" not in rep["false_negative_gap_units"]


@_requires_toolchain
def test_differential_fuzz_returns_counts():
    fu = FuzzUnit("shift_w32",
                  {"kind": "shift", "width": 32, "probe": "shift_oob"},
                  "shift_oob",
                  {"x": ("int", -8, 8), "s": ("int", 0, 2 ** 16 - 1)})
    hits, first = differential_fuzz(ReexecHarness(_TC), fu, trials=50, seed=1)
    assert hits == 1 and first is not None and first >= 1


def test_default_units_are_well_formed():
    units = default_units()
    assert {u.divergence_class for u in units} == {"signed_overflow", "shift_oob"}
    for u in units:
        assert u.domains, "each unit declares a sampling domain"
        for name, dom in u.domains.items():
            assert isinstance(name, str)
            assert dom[0] in ("int", "float") and dom[1] <= dom[2]

# ── GitHub Action: Translation Equivalence Guard (step 56) ───────────────────

import os as _os  # noqa: E402
import sys as _sys  # noqa: E402

_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_ACTION_YML = _os.path.join(
    _REPO_ROOT, ".github", "actions", "translation-equivalence-guard", "action.yml")
_WORKFLOW_YML = _os.path.join(
    _REPO_ROOT, ".github", "workflows", "translation-equivalence-guard.example.yml")


def test_guard_action_is_well_formed():
    yaml = pytest.importorskip("yaml")
    with open(_ACTION_YML) as fh:
        action = yaml.safe_load(fh)
    assert action["name"] == "Translation Equivalence Guard"
    assert action["runs"]["using"] == "composite"
    # the manifest input is required; sarif/fail-on have sane defaults.
    assert action["inputs"]["manifest"]["required"] is True
    assert action["inputs"]["fail-on"]["default"] == "divergent"
    # the action must surface the SARIF path and exit code as outputs.
    assert "sarif" in action["outputs"] and "exit-code" in action["outputs"]
    # at least one step actually invokes the CLI we ship.
    cmds = " ".join(str(s.get("run", "")) for s in action["runs"]["steps"])
    assert "cross-lang-verify --units" in cmds
    assert "--sarif" in cmds and "--fail-on" in cmds


def test_guard_example_workflow_uploads_sarif():
    yaml = pytest.importorskip("yaml")
    with open(_WORKFLOW_YML) as fh:
        wf = yaml.safe_load(fh)
    # PyYAML parses the bare `on:` key as boolean True; tolerate both.
    triggers = wf.get("on", wf.get(True))
    assert "pull_request" in triggers
    # needs security-events: write to upload to code scanning.
    assert wf["permissions"]["security-events"] == "write"
    steps = wf["jobs"]["guard"]["steps"]
    uses = [s.get("uses", "") for s in steps]
    assert any(u.endswith("translation-equivalence-guard") for u in uses)
    assert any("codeql-action/upload-sarif" in u for u in uses)


def test_guard_manifest_example_verifies_and_emits_sarif(tmp_path):
    # the manifest the example workflow ships must really drive the CLI to a
    # SARIF file (proving the action's core command works end-to-end).
    manifest = _os.path.join(_REPO_ROOT, "examples", "units_manifest.json")
    sarif = tmp_path / "guard.sarif"
    rc = _cli.run(["--units", manifest, "--no-confirm", "--color", "never",
                   "--sarif", str(sarif), "--fail-on", "candidate"])
    doc = json.loads(sarif.read_text())
    assert doc["version"] == "2.1.0"
    # without confirmation the overflow/shift/etc. units are CANDIDATE warnings.
    assert any(r["level"] == "warning" for r in doc["runs"][0]["results"])
    assert rc == 1  # --fail-on candidate trips on the symbolic witnesses


# ── C -> Go: the second language pair (step 37) ──────────────────────────────

from src.ub_oracle.verify import verify_unit, applicable_oracles, VerifyVerdict
from src.ub_oracle import plugin as _plugin
from src.ub_oracle.reexec import RunOutcome

_requires_go = pytest.mark.skipif(
    not _TC.full_for("go"),
    reason=f"needs C+UBSan+go toolchain ({_TC})")

# (probe-free) C->Go units, one per Go oracle.
_GO_UNITS = {
    "signed_overflow": {"kind": "binop_const", "op": "add", "const": 2147483647,
                        "width": 32, "var": "x", "signed": True},
    "shift_oob": {"kind": "shift", "width": 32, "value": 1},
    "div_by_zero": {"kind": "div", "width": 32, "a": "a", "b": "b"},
    "array_oob": {"kind": "array_index", "length": 4},
}


def _go_unit(class_key):
    u = dict(_GO_UNITS[class_key])
    u["source_lang"], u["target_lang"] = "c", "go"
    return u


def test_go_pair_is_registered_as_a_second_language_pair():
    pairs = _plugin.language_pairs()
    assert ("c", "rust") in pairs
    assert ("c", "go") in pairs
    go_oracles = _plugin.oracles_for(source_lang="c", target_lang="go")
    classes = {o.divergence_class for o in go_oracles}
    # the four argv-driven classes plus INT_MIN/-1 are covered for Go.
    assert {"signed_overflow", "shift_oob", "div_by_zero",
            "array_oob", "intmin_div_neg1"} <= classes


def test_anchor_registry_is_unchanged_by_the_go_pair():
    # the legacy REGISTRY must still expose exactly the C->Rust anchor oracles
    # (one per class) so metrics/ablation/head-to-head are untouched.
    for key, orc in _plugin.REGISTRY.items():
        assert (orc.source_lang, orc.target_lang) == ("c", "rust")
        assert orc.divergence_class == key


@pytest.mark.parametrize("class_key", list(_GO_UNITS))
def test_go_oracle_emits_go_target_source_symbolically(class_key):
    orc = _plugin.get_oracle_for(class_key, "c", "go")
    res = orc.find_divergence(_go_unit(class_key))
    assert res.verdict is OracleVerdict.DIVERGENT
    ce = res.counterexample
    assert ce.target_lang == "go"
    # real Go source markers (no toolchain needed to check this).
    assert ce.target_snippet.startswith("package main")
    assert "func main()" in ce.target_snippet
    # the C source is identical to the C->Rust anchor's (generality reuse).
    anchor = _plugin.get_oracle_for(class_key, "c", "rust")
    anchor_ce = anchor.find_divergence(dict(_GO_UNITS[class_key])).counterexample
    assert ce.source_snippet == anchor_ce.source_snippet


def test_go_unit_is_routed_only_to_go_oracles():
    unit = _go_unit("div_by_zero")
    applicable = applicable_oracles(unit)
    assert applicable, "a C->Go unit must match the Go oracles"
    assert all(o.target_lang == "go" for o in applicable)


def test_c_rust_unit_is_not_routed_to_go_oracles():
    # a unit that omits languages defaults to the C->Rust anchor — Go oracles
    # must never fire on it.
    unit = {"kind": "div", "width": 32, "a": "a", "b": "b"}
    applicable = applicable_oracles(unit)
    assert applicable
    assert all(o.target_lang == "rust" for o in applicable)


def test_run_outcome_go_definedness_predicate():
    # Go's defined return codes: 0 (value) or 2 (runtime panic). 101 (Rust's
    # panic code) is NOT a defined Go outcome.
    assert RunOutcome(0, "x", "").target_outcome_defined("go")
    assert RunOutcome(2, "", "panic").target_outcome_defined("go")
    assert not RunOutcome(101, "", "").target_outcome_defined("go")
    assert not RunOutcome(139, "", "").target_outcome_defined("go")
    # Rust's predicate is unchanged.
    assert RunOutcome(101, "", "").target_outcome_defined("rust")
    assert not RunOutcome(2, "", "").target_outcome_defined("rust")


@_requires_go
@pytest.mark.parametrize("class_key", list(_GO_UNITS))
def test_go_divergence_confirmed_against_real_clang_and_go(class_key):
    orc = _plugin.get_oracle_for(class_key, "c", "go")
    res = orc.confirm(orc.find_divergence(_go_unit(class_key)), ReexecHarness(_TC))
    rr = res.reexec
    assert rr.available, rr.reason
    assert rr.ub_reachable, "UBSan must trap on the witness (C is UB)"
    assert rr.rust_defined, "Go must produce a defined, deterministic outcome"
    assert rr.confirmed, rr.reason
    assert res.counterexample.confirmed
    assert res.counterexample.target_observed is not None


@_requires_go
def test_go_intmin_div_neg1_confirmed_end_to_end():
    from src.ub_oracle.catalogue import INT_MIN_DIV_NEG1
    unit = {"kind": "div", "width": 32, "signed": True, "a": "a", "b": "b",
            "source_lang": "c", "target_lang": "go",
            "probe": INT_MIN_DIV_NEG1.key}
    report = verify_unit(unit)
    assert report.verdict is VerifyVerdict.DIVERGENT
    assert report.divergence.divergence_class == INT_MIN_DIV_NEG1.key


@_requires_go
def test_equivalent_go_translation_is_not_confirmed():
    # negative control: a Go program whose answer matches a *defined* C program
    # (no UB) must NOT be confirmed as a divergence — the sanitizer never traps.
    harness = ReexecHarness(_TC)
    c_src = ("#include <stdio.h>\n#include <stdlib.h>\n"
             "int f(int a,int b){ return a + b; }\n"
             "int main(int c,char**v){ if(c<3)return 2;"
             " printf(\"%d\\n\", f(atoi(v[1]),atoi(v[2]))); return 0; }\n")
    go_src = ("package main\nimport (\n\t\"fmt\"\n\t\"os\"\n\t\"strconv\"\n)\n"
              "func f(a int32, b int32) int32 { return a + b }\n"
              "func main(){ av,_:=strconv.Atoi(os.Args[1]); bv,_:=strconv.Atoi(os.Args[2]);"
              " fmt.Println(f(int32(av), int32(bv))) }\n")
    rr = harness.confirm_trap_vs_defined(c_src, go_src, ["2", "3"],
                                         "division_by_zero", target_lang="go")
    assert rr.available
    assert not rr.ub_reachable
    assert not rr.confirmed


# ── Step 39: pluggable target-semantics packs (C->Swift third pair) ──────────

from src.ub_oracle.target_semantics import PACKS, TargetPack, get_pack
from src.ub_oracle.oracles import target_pairs as _tp
from src.ub_oracle.oracles.signed_overflow import SignedOverflowOracle
from src.ub_oracle.oracles.integer_ub import DivisionByZeroOracle
from src.ub_oracle.oracles.memory_shape import ArrayOutOfBoundsOracle

_requires_swift = pytest.mark.skipif(
    not _TC.full_for("swift"),
    reason=f"needs C+UBSan+swiftc toolchain ({_TC})")

_SWIFT_UNITS = {
    "signed_overflow": {"kind": "binop_const", "op": "add", "const": 2147483647,
                        "width": 32, "var": "x", "signed": True},
    "shift_oob": {"kind": "shift", "width": 32, "value": 1},
    "div_by_zero": {"kind": "div", "width": 32, "a": "a", "b": "b"},
    "array_oob": {"kind": "array_index", "length": 4},
}


def _swift_unit(class_key):
    u = dict(_SWIFT_UNITS[class_key])
    u["source_lang"], u["target_lang"] = "c", "swift"
    return u


def test_target_packs_encode_defined_returncodes_as_data():
    assert PACKS["rust"].defined_returncodes == (0, 101)
    assert PACKS["go"].defined_returncodes == (0, 2)
    # Swift fatal traps are SIGTRAP, which Python's subprocess reports as -5.
    assert PACKS["swift"].defined_returncodes == (0, -5)
    # every pack documents how it resolves each core UB class (data-driven).
    for name in ("rust", "go", "swift"):
        res = PACKS[name].class_resolution
        assert {"signed_overflow", "div_by_zero", "array_oob"} <= set(res)


def test_get_pack_raises_loudly_for_unknown_target():
    with pytest.raises(ValueError):
        get_pack("haskell")


def test_swift_pair_is_a_third_registered_language_pair():
    pairs = _plugin.language_pairs()
    assert ("c", "swift") in pairs
    sw = _plugin.oracles_for(source_lang="c", target_lang="swift")
    classes = {o.divergence_class for o in sw}
    assert {"signed_overflow", "shift_oob", "div_by_zero",
            "array_oob", "intmin_div_neg1"} <= classes


def test_generated_target_oracles_reuse_anchor_search_not_new_code():
    # adding a target is *configuration*: the generated oracle is a subclass of
    # the very anchor oracle whose Z3 witness search it reuses unchanged.
    assert isinstance(_plugin.get_oracle_for("signed_overflow", "c", "swift"),
                      SignedOverflowOracle)
    assert isinstance(_plugin.get_oracle_for("div_by_zero", "c", "swift"),
                      DivisionByZeroOracle)
    assert isinstance(_plugin.get_oracle_for("array_oob", "c", "go"),
                      ArrayOutOfBoundsOracle)


@pytest.mark.parametrize("class_key", list(_SWIFT_UNITS))
def test_swift_oracle_emits_swift_source_symbolically(class_key):
    orc = _plugin.get_oracle_for(class_key, "c", "swift")
    res = orc.find_divergence(_swift_unit(class_key))
    assert res.verdict is OracleVerdict.DIVERGENT
    ce = res.counterexample
    assert ce.target_lang == "swift"
    assert ce.target_snippet.startswith("import Foundation")
    assert "func f(" in ce.target_snippet
    # the C source is byte-identical to the anchor's (single witness search).
    anchor = _plugin.get_oracle_for(class_key, "c", "rust")
    anchor_ce = anchor.find_divergence(dict(_SWIFT_UNITS[class_key])).counterexample
    assert ce.source_snippet == anchor_ce.source_snippet


def test_run_outcome_definedness_is_pack_driven():
    assert RunOutcome(0, "v", "").target_outcome_defined("swift")
    assert RunOutcome(-5, "", "trap").target_outcome_defined("swift")  # SIGTRAP
    assert not RunOutcome(2, "", "").target_outcome_defined("swift")
    assert not RunOutcome(101, "", "").target_outcome_defined("swift")
    # cross-checks: go/rust predicates are independent data.
    assert RunOutcome(2, "", "").target_outcome_defined("go")
    assert not RunOutcome(-5, "", "").target_outcome_defined("go")
    with pytest.raises(ValueError):
        RunOutcome(0, "", "").target_outcome_defined("haskell")


def test_toolchain_status_is_pack_driven_and_pair_aware():
    # a status with only the swift compiler present is full *for swift* only.
    st = ToolchainStatus(cc="/usr/bin/clang", ubsan=True,
                         targets=(("rust", None), ("go", None),
                                  ("swift", "/usr/bin/swiftc")))
    assert st.full_for("swift")
    assert not st.full_for("rust")
    assert not st.full_for("go")
    assert st.target_path("swift") == "/usr/bin/swiftc"


@_requires_swift
@pytest.mark.parametrize("class_key", ["signed_overflow", "div_by_zero", "array_oob"])
def test_swift_divergence_confirmed_against_real_clang_and_swiftc(class_key):
    orc = _plugin.get_oracle_for(class_key, "c", "swift")
    res = orc.confirm(orc.find_divergence(_swift_unit(class_key)), ReexecHarness(_TC))
    rr = res.reexec
    assert rr.available, rr.reason
    assert rr.ub_reachable, "UBSan must trap on the witness (C is UB)"
    assert rr.rust_defined, "Swift must produce a defined, deterministic outcome"
    assert rr.confirmed, rr.reason
    assert res.counterexample.confirmed


@_requires_swift
def test_swift_value_vs_trap_resolutions_are_both_confirmed():
    # signed overflow resolves to a *value* (rc 0); div-by-zero resolves to a
    # *trap* (rc -5). Both are language-defined per the Swift pack, and both
    # confirm — proving the pack's two definedness flavours work end-to-end.
    harness = ReexecHarness(_TC)
    sov = _plugin.get_oracle_for("signed_overflow", "c", "swift")
    rr1 = sov.confirm(sov.find_divergence(_swift_unit("signed_overflow")), harness).reexec
    assert rr1.confirmed and rr1.rust_run.returncode == 0
    dv = _plugin.get_oracle_for("div_by_zero", "c", "swift")
    rr2 = dv.confirm(dv.find_divergence(_swift_unit("div_by_zero")), harness).reexec
    assert rr2.confirmed and rr2.rust_run.returncode == -5


# ── Step 40: cross-pair regression matrix (living evidence of generality) ────

from src.ub_oracle import regression_matrix as _matrix


def test_matrix_covers_every_registered_pair_and_class():
    m = _matrix.build_matrix()
    # one cell per registered oracle.
    assert m["n_cells"] == len(_plugin.ALL_ORACLES)
    pairs = {(c["source_lang"], c["target_lang"]) for c in m["cells"]}
    assert pairs == set(_plugin.language_pairs())
    assert {"c->rust", "c->go", "c->swift"} <= set(m["language_pairs"])


def test_matrix_every_cell_is_divergent_symbolically():
    m = _matrix.build_matrix()
    for c in m["cells"]:
        assert c["verdict"] == str(OracleVerdict.DIVERGENT), c
    # coverage totals: every covered class in every pair is divergent.
    for cov in m["coverage"]:
        assert cov["n_divergent"] == cov["n_classes"] == len(cov["classes_covered"])


def test_matrix_pair_coverage_is_honest_about_class_breadth():
    m = _matrix.build_matrix()
    cov = {(c["source_lang"], c["target_lang"]): c for c in m["coverage"]}
    # the anchor implements the full catalogue of executable oracles; the
    # generated pairs implement the five argv-driven integer/memory classes
    # plus the optimizer-exploited uninitialized-read class (which is confirmed
    # by two conforming C builds disagreeing while the target stays defined, so
    # it transfers to every target pair, not just the anchor).
    assert {"strict_aliasing", "fp_contraction"} <= set(cov[("c", "rust")]["classes_covered"])
    for tgt in ("go", "swift"):
        covered = set(cov[("c", tgt)]["classes_covered"])
        assert {"signed_overflow", "shift_oob", "div_by_zero",
                "intmin_div_neg1", "array_oob", "uninit_read"} == covered


def test_matrix_is_byte_reproducible_in_a_fresh_process():
    # The matrix is byte-reproducible per *fresh* process (that is exactly the
    # contract `make matrix-check` enforces in CI). We therefore verify it the
    # same way: spawn a clean interpreter and assert the artifact regenerates
    # identically. (Doing it in-process would be confounded by the SMT solver's
    # per-process RNG, which other tests in this module have already advanced.)
    import subprocess
    from experiments.cross_pair_matrix.run import RESULTS_PATH
    assert _os.path.exists(RESULTS_PATH), "run `make matrix` to materialise the artifact"
    root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    proc = subprocess.run(
        [_sys.executable, "-m", "experiments.cross_pair_matrix.run", "--check"],
        cwd=root, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_matrix_records_pack_defined_returncodes_per_cell():
    m = _matrix.build_matrix()
    by_tgt = {}
    for c in m["cells"]:
        by_tgt.setdefault(c["target_lang"], set()).add(tuple(c["target_defined_returncodes"]))
    assert by_tgt["rust"] == {(0, 101)}
    assert by_tgt["go"] == {(0, 2)}
    assert by_tgt["swift"] == {(0, -5)}


def test_matrix_render_table_is_a_grid_over_pairs_and_classes():
    table = _matrix.render_table()
    assert "c->rust" in table and "c->go" in table and "c->swift" in table
    assert "signed_overflow" in table and "fp_contraction" in table
    # a class not implemented for a pair is shown as '-', not a false 'D'.
    assert "-" in table and "D" in table


def test_matrix_confirm_marks_unavailable_pairs_skipped_not_dropped():
    # a host with no compilers must still account for *every* cell, as skipped.
    class _Dead:
        def full_for(self, _t):
            return False
    class _Harness:
        status = _Dead()
    conf = _matrix.confirm_matrix(_Harness())
    assert conf["n_cells"] == len(_plugin.ALL_ORACLES)
    assert conf["n_attempted"] == 0
    assert all(c["skipped"] for c in conf["cells"])


@_requires_swift
def test_matrix_confirm_runs_every_cell_against_real_compilers():
    # the full matrix, end-to-end, against real clang+UBSan + rustc + go + swiftc.
    conf = _matrix.confirm_matrix(ReexecHarness(_TC))
    assert conf["n_attempted"] == len(_plugin.ALL_ORACLES)
    assert conf["all_attempted_confirmed"], \
        [c for c in conf["cells"] if not c.get("confirmed")]


# ── Step 59: real, installable packaging (pip install cross-lang-verifier) ───

def _load_pyproject():
    try:
        import tomllib  # py3.11+
    except ModuleNotFoundError:  # pragma: no cover
        import tomli as tomllib  # type: ignore
    root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    with open(_os.path.join(root, "pyproject.toml"), "rb") as f:
        return tomllib.load(f)


def test_pyproject_declares_distribution_and_fixed_entry_points():
    pp = _load_pyproject()
    proj = pp["project"]
    assert proj["name"] == "cross-lang-verifier"
    # version is pinned (a concrete, parseable version string).
    assert proj["version"].count(".") >= 2
    # both console scripts point at the real, importable CLI entry.
    scripts = proj["scripts"]
    assert scripts["cross-lang-verify"] == "ub_oracle.cli:main"
    assert scripts["cross-lang-verifier"] == "ub_oracle.cli:main"
    # z3 is the single runtime dependency, version-bounded (no stray openai).
    deps = " ".join(proj["dependencies"])
    assert "z3-solver" in deps and ">=" in deps
    assert "openai" not in deps


def test_pyproject_packages_only_the_self_contained_ub_oracle_tree():
    pp = _load_pyproject()
    st = pp["tool"]["setuptools"]
    assert st["package-dir"] == {"": "src"}
    find = st["packages"]["find"]
    assert find["where"] == ["src"]
    # scoped discovery: never package src/tests/experiments/benchmarks.
    assert find["include"] == ["ub_oracle*"]


def test_no_stale_setup_py_with_mismatched_name():
    # the historical setup.py declared name="semrec" with an src.* entry point;
    # Step 59 removes that name/script mismatch in favour of pyproject.
    root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    assert not _os.path.exists(_os.path.join(root, "setup.py"))


def test_cli_main_entry_point_is_importable_and_callable():
    # the console script resolves to `ub_oracle.cli:main`; in-repo that module is
    # `src.ub_oracle.cli` (the installed-wheel top-level import is proven
    # separately by scripts/verify_packaging.sh).
    from src.ub_oracle.cli import main as _entry
    assert callable(_entry)
    # a trivial empty manifest verifies the entry runs without a toolchain.
    import tempfile, json as _json
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        _json.dump({"units": []}, fh)
        path = fh.name
    try:
        rc = _entry(["--units", path, "--no-confirm"])
    finally:
        _os.unlink(path)
    assert rc == 0


# --------------------------------------------------------------------------- #
# Step 52: counterexample quality / minimizer.
# --------------------------------------------------------------------------- #
from src.ub_oracle.minimizer import (  # noqa: E402
    simplicity_cost,
    _reduction_ladder,
    MinimizationResult,
    minimize_counterexample,
)
from src.ub_oracle.regression_matrix import canonical_unit_for  # noqa: E402


def test_simplicity_cost_orders_small_nonnegative_first():
    # 0 is the simplest; smaller magnitude beats larger; ties broken so the
    # non-negative representative is simpler than its negative counterpart.
    assert simplicity_cost(0) < simplicity_cost(1)
    assert simplicity_cost(1) < simplicity_cost(2)
    assert simplicity_cost(1) < simplicity_cost(-1)
    assert simplicity_cost(2) < simplicity_cost(-3)
    assert simplicity_cost(5) < simplicity_cost(-5)


def test_reduction_ladder_is_strictly_simpler_and_sorted():
    v = 1073741824
    ladder = _reduction_ladder(v)
    # every candidate is strictly simpler than v ...
    assert all(simplicity_cost(c) < simplicity_cost(v) for c in ladder)
    # ... presented simplest-first ...
    costs = [simplicity_cost(c) for c in ladder]
    assert costs == sorted(costs)
    # ... and the canonical anchors 0 and 1 are reachable.
    assert 0 in ladder and 1 in ladder
    # a value already at the floor has an empty ladder.
    assert _reduction_ladder(0) == []


def test_ub_category_distinguishes_ubsan_diagnostics():
    from src.ub_oracle.reexec import RunOutcome
    def cat(msg):
        return RunOutcome(1, "", f"x.c:3:5: runtime error: {msg}").ub_category
    overflow = cat("signed integer overflow: 2147483647 + 1 cannot be "
                   "represented in type 'int'")
    shift_big = cat("shift exponent 32 is too large for 32-bit type 'int'")
    shift_neg = cat("shift exponent -1 is negative")
    divzero = cat("division by zero")
    intmin = cat("division of -2147483648 by -1 cannot be represented in "
                 "type 'int'")
    oob = cat("index 4 out of bounds for type 'int[4]'")
    # numbers and quoted types are stripped, so witnesses of the same KIND
    # share a category regardless of the specific operands ...
    assert cat("index 99 out of bounds for type 'int[8]'") == oob
    # ... but genuinely different undefined behaviors stay distinct.
    cats = {overflow, shift_big, shift_neg, divzero, intmin, oob}
    assert len(cats) == 6
    # crucially the two pairs the minimizer must never conflate:
    assert shift_big != shift_neg
    assert divzero != intmin
    # a clean (non-UBSan) run has no category.
    assert RunOutcome(0, "ok", "").ub_category == ""


def test_minimization_result_to_dict_roundtrips_and_reports_reduction():
    r = MinimizationResult(
        divergence_class="signed_overflow", source_lang="c", target_lang="rust",
        original_inputs={"x": 1073741824}, minimized_inputs={"x": 1},
        fields_reduced=["x"], probes=5, confirmed=True,
        certified_locally_minimal=True, already_minimal=False)
    assert r.reduced is True
    d = r.to_dict()
    assert d["divergence_class"] == "signed_overflow"
    assert d["original_inputs"] == {"x": 1073741824}
    assert d["minimized_inputs"] == {"x": 1}
    assert d["fields_reduced"] == ["x"]
    assert d["confirmed"] is True and d["certified_locally_minimal"] is True


def _anchor_oracle(class_key):
    return next(o for o in _plugin.ALL_ORACLES
               if o.divergence_class == class_key
               and o.source_lang == "c" and o.target_lang == "rust")


@_requires_toolchain
def test_signed_overflow_witness_minimizes_to_one_against_real_compilers():
    orc = _anchor_oracle("signed_overflow")
    res = orc.find_divergence(canonical_unit_for(orc))
    assert res.counterexample is not None
    m = minimize_counterexample(orc, res, ReexecHarness(), max_probes=400)
    assert m.confirmed
    assert m.minimized_inputs == {"x": 1}
    assert m.fields_reduced == ["x"]
    assert m.certified_locally_minimal


@_requires_toolchain
def test_intmin_div_neg1_does_not_drift_into_division_by_zero():
    # The faithfulness guarantee: magnitude minimization must NOT collapse the
    # INT_MIN/-1 witness to {0,0} (a *different* UB: division by zero).  UBSan
    # category preservation keeps it pinned at the canonical pair.
    orc = _anchor_oracle("intmin_div_neg1")
    res = orc.find_divergence(canonical_unit_for(orc))
    assert res.counterexample is not None
    m = minimize_counterexample(orc, res, ReexecHarness(), max_probes=400)
    assert m.confirmed
    assert m.minimized_inputs == {"a": -2147483648, "b": -1}
    assert m.minimized_inputs != {"a": 0, "b": 0}
    assert m.certified_locally_minimal


@_requires_toolchain
def test_shift_oob_keeps_too_large_exponent_category():
    # s=32 ("too large") must NOT be minimized into s=-1 ("negative"), which is
    # a different UBSan category, even though -1 has smaller magnitude.
    orc = _anchor_oracle("shift_oob")
    res = orc.find_divergence(canonical_unit_for(orc))
    assert res.counterexample is not None
    m = minimize_counterexample(orc, res, ReexecHarness(), max_probes=400)
    assert m.confirmed
    assert m.minimized_inputs.get("s") == 32
    assert m.certified_locally_minimal


# --------------------------------------------------------------------------- #
# Step 65: divergence triage UX.
# --------------------------------------------------------------------------- #
from src.ub_oracle import triage as _triage  # noqa: E402
from src.ub_oracle.triage import (  # noqa: E402
    Tier, TriageItem, severity_of_class, triage_reports, render_triage,
)
from src.ub_oracle.verify import VerifyReport, VerifyVerdict  # noqa: E402
from src.ub_oracle.plugin import OracleResult, OracleVerdict  # noqa: E402


def _divergent_report(name, cls):
    r = VerifyReport(VerifyVerdict.DIVERGENT,
                     {"name": name, "source_lang": "c", "target_lang": "rust"})
    r.divergence = OracleResult(OracleVerdict.DIVERGENT, cls)
    return r


def _report(verdict, name, probe=None, pair=("c", "rust")):
    u = {"name": name, "source_lang": pair[0], "target_lang": pair[1]}
    if probe:
        u["probe"] = probe
    return VerifyReport(verdict, u)


def test_severity_of_class_reads_the_catalogue():
    assert severity_of_class("signed_overflow").value == "critical"
    assert severity_of_class("fp_contraction").value == "moderate"
    assert severity_of_class("nonexistent_class") is None


def test_triage_ranks_confirmed_above_candidate_above_abstention():
    reports = [
        _report(VerifyVerdict.NOT_COVERED, "z", pair=("go", "rust")),
        _report(VerifyVerdict.CANDIDATE, "cand", probe="array_oob"),
        _divergent_report("crit", "signed_overflow"),     # CRITICAL
        _divergent_report("mod", "fp_contraction"),        # MODERATE
        _report(VerifyVerdict.NO_DIVERGENCE_FOUND, "clean", probe="signed_overflow"),
        _report(VerifyVerdict.UNKNOWN, "huh", probe="div_by_zero"),
    ]
    s = triage_reports(reports)
    order = [it.unit_label for it in s.items]
    # confirmed-critical first, then confirmed-moderate, then candidate, then
    # unknown, then not-covered, then no-divergence (informational) last.
    assert order == ["crit", "mod", "cand", "huh", "z", "clean"]
    assert s.top_tier is Tier.CONFIRMED_CRITICAL
    # everything but the clean no-divergence row needs a human's attention.
    assert s.actionable == 5


def test_triage_is_deterministic_within_a_tier():
    # two critical confirmations ordered by class then unit name, stably.
    reports = [
        _divergent_report("u_b", "signed_overflow"),
        _divergent_report("u_a", "signed_overflow"),
        _divergent_report("arr", "array_oob"),
    ]
    s1 = triage_reports(reports)
    s2 = triage_reports(list(reversed(reports)))
    assert [it.unit_label for it in s1.items] == [it.unit_label for it in s2.items]
    # array_oob sorts before signed_overflow; within signed_overflow, u_a < u_b.
    assert [it.unit_label for it in s1.items] == ["arr", "u_a", "u_b"]


def test_triage_summary_to_dict_counts_each_tier():
    reports = [
        _divergent_report("c1", "signed_overflow"),
        _divergent_report("c2", "array_oob"),
        _report(VerifyVerdict.CANDIDATE, "cand", probe="shift_oob"),
    ]
    d = triage_reports(reports).to_dict()
    assert d["total"] == 3
    assert d["actionable"] == 3
    assert d["top_tier"] == int(Tier.CONFIRMED_CRITICAL)
    assert d["by_tier"][int(Tier.CONFIRMED_CRITICAL)] == 2
    assert d["by_tier"][int(Tier.CANDIDATE)] == 1


def test_render_triage_groups_and_flags_critical():
    reports = [_divergent_report("boom", "signed_overflow")]
    txt = render_triage(triage_reports(reports))
    assert "Triage" in txt
    assert "confirmed-divergence (critical)" in txt
    assert "boom" in txt and "signed_overflow" in txt
    # the urgent marker appears on the critical group.
    assert "\u203c" in txt


def test_tier_actionability_excludes_only_clean():
    assert Tier.CONFIRMED_CRITICAL.actionable
    assert Tier.CANDIDATE.actionable
    assert Tier.NOT_COVERED.actionable
    assert not Tier.NO_DIVERGENCE.actionable


# --------------------------------------------------------------------------- #
# Step 67: config + suppression / baseline files.
# --------------------------------------------------------------------------- #
import datetime as _datetime  # noqa: E402
from src.ub_oracle import suppress as _suppress  # noqa: E402
from src.ub_oracle.suppress import (  # noqa: E402
    Suppression, load_suppressions, apply_suppressions, build_baseline,
    fingerprint_of,
)


def test_suppression_matches_by_class_pair_and_unit_glob():
    r = _divergent_report("checksum_v2", "signed_overflow")
    assert Suppression("ok", divergence_class="signed_overflow").matches(r)
    assert not Suppression("ok", divergence_class="array_oob").matches(r)
    assert Suppression("ok", pair="c->rust").matches(r)
    assert not Suppression("ok", pair="c->go").matches(r)
    assert Suppression("ok", unit="checksum_*").matches(r)
    assert not Suppression("ok", unit="other_*").matches(r)
    # all-of: every specified field must match.
    assert Suppression("ok", divergence_class="signed_overflow",
                       unit="checksum_*").matches(r)
    assert not Suppression("ok", divergence_class="signed_overflow",
                           unit="nope_*").matches(r)


def test_suppression_matches_by_fingerprint_exactly():
    r = _divergent_report("u", "signed_overflow")
    fp = fingerprint_of(r)
    assert Suppression("ok", fingerprint=fp).matches(r)
    assert not Suppression("ok", fingerprint="deadbeefdeadbeef").matches(r)


def test_suppression_only_targets_findings_not_abstentions():
    clean = _report(VerifyVerdict.NO_DIVERGENCE_FOUND, "ok", probe="signed_overflow")
    notcov = _report(VerifyVerdict.NOT_COVERED, "x", pair=("go", "rust"))
    # an empty rule matches every *finding* but never a clean/abstained report.
    empty = Suppression("blanket")
    assert empty.is_empty_match()
    assert not empty.matches(clean)
    assert not empty.matches(notcov)
    assert empty.matches(_divergent_report("d", "signed_overflow"))


def test_suppression_expiry_is_enforced():
    r = _divergent_report("u", "signed_overflow")
    past = Suppression("ok", divergence_class="signed_overflow", expires="2000-01-01")
    future = Suppression("ok", divergence_class="signed_overflow", expires="2999-01-01")
    assert past.expired()
    assert not future.expired()
    assert not past.matches(r)              # expired never suppresses
    assert future.matches(r)
    # a malformed expiry fails safe (treated as expired, never silently hides).
    assert Suppression("ok", expires="not-a-date").expired()


def test_load_suppressions_requires_a_reason(tmp_path):
    good = tmp_path / "good.json"
    good.write_text(_json_dumps({"suppressions": [
        {"divergence_class": "signed_overflow", "reason": "intended wrap"}]}))
    rules = load_suppressions(str(good))
    assert len(rules) == 1 and rules[0].reason == "intended wrap"

    bad = tmp_path / "bad.json"
    bad.write_text(_json_dumps({"suppressions": [{"divergence_class": "x"}]}))
    with pytest.raises(ValueError):
        load_suppressions(str(bad))


def test_apply_suppressions_reports_unused_expired_and_empty():
    reports = [
        _divergent_report("a", "signed_overflow"),
        _divergent_report("b", "array_oob"),
    ]
    rules = [
        Suppression("match-a", divergence_class="signed_overflow"),
        Suppression("never", divergence_class="div_by_zero"),         # unused
        Suppression("stale", divergence_class="array_oob",
                    expires="2000-01-01"),                            # expired
    ]
    res = apply_suppressions(reports, rules)
    assert res.suppressed_count == 1                  # only 'a' suppressed
    assert [o.suppressed for o in res.outcomes] == [True, False]
    assert any(u.reason == "never" for u in res.unused_rules)
    assert any(e.reason == "stale" for e in res.expired_rules)
    # the expired rule did NOT suppress b.
    assert res.outcomes[1].effective_finding


def test_build_baseline_is_deterministic_and_pins_fingerprints():
    reports = [
        _divergent_report("b", "array_oob"),
        _divergent_report("a", "signed_overflow"),
        _report(VerifyVerdict.NO_DIVERGENCE_FOUND, "clean", probe="signed_overflow"),
    ]
    base = build_baseline(reports)
    assert base["version"] == 1
    # only the two findings are baselined (clean report excluded), sorted by class.
    classes = [s["divergence_class"] for s in base["suppressions"]]
    assert classes == ["array_oob", "signed_overflow"]
    assert all("fingerprint" in s and s["reason"] for s in base["suppressions"])
    # regenerating yields byte-identical content.
    assert build_baseline(reports) == base
    # the emitted baseline, fed back in, suppresses exactly those findings.
    rules = [Suppression.from_dict(s) for s in base["suppressions"]]
    res = apply_suppressions(reports, rules)
    assert res.suppressed_count == 2


def _json_dumps(obj):
    import json as _j
    return _j.dumps(obj)


@_requires_toolchain
def test_cli_baseline_flips_the_fail_gate_against_real_compilers(tmp_path):
    # A real, confirmed signed-overflow divergence makes the CLI exit 1 ...
    from src.ub_oracle.cli import run
    manifest = tmp_path / "units.json"
    manifest.write_text(_json_dumps({"units": [
        {"name": "add1_w32", "kind": "binop_const", "op": "add", "const": 1,
         "width": 32, "var": "x", "signed": True, "probe": "signed_overflow",
         "source_lang": "c", "target_lang": "rust"}]}))
    rc = run(["--units", str(manifest), "--color", "never"])
    assert rc == 1

    # ... writing a baseline captures that one finding ...
    baseline = tmp_path / "baseline.json"
    rc = run(["--units", str(manifest), "--write-baseline", str(baseline)])
    assert rc == 0
    payload = json.loads(baseline.read_text())
    assert len(payload["suppressions"]) == 1
    assert payload["suppressions"][0]["divergence_class"] == "signed_overflow"

    # ... and re-running under that baseline flips the gate green (exit 0),
    # because the known-accepted divergence is suppressed.
    rc = run(["--units", str(manifest), "--suppress", str(baseline),
              "--color", "never"])
    assert rc == 0


# --------------------------------------------------------------------------- #
# Step 66: incremental verification cache.
# --------------------------------------------------------------------------- #
from src.ub_oracle import cache as _cache  # noqa: E402
from src.ub_oracle.cache import (  # noqa: E402
    VerificationCache, CacheEntry, cache_key, canonical_unit,
    toolchain_fingerprint, verify_incremental, SEMANTICS_VERSION,
)


def _fp(**kw):
    base = {"cc": "clang X", "ubsan": "yes", "rust": "rustc Y"}
    base.update(kw)
    return base


def test_canonical_unit_is_order_independent():
    a = {"name": "u", "op": "add", "const": 1}
    b = {"const": 1, "op": "add", "name": "u"}
    assert canonical_unit(a) == canonical_unit(b)


def test_cache_key_changes_with_unit_toolchain_and_semantics():
    u1 = {"name": "u", "op": "add", "const": 1}
    u2 = {"name": "u", "op": "add", "const": 2}
    fp1 = _fp()
    fp2 = _fp(cc="clang DIFFERENT")
    k = cache_key(u1, fp1)
    # changing the unit changes the key ...
    assert cache_key(u2, fp1) != k
    # ... changing the toolchain fingerprint changes the key (soundness!) ...
    assert cache_key(u1, fp2) != k
    # ... and the same inputs are stable.
    assert cache_key(u1, fp1) == k


def test_cache_only_stores_deterministic_verdicts():
    c = VerificationCache(fingerprint=_fp())
    div = _divergent_report("d", "signed_overflow")
    cand = _report(VerifyVerdict.CANDIDATE, "c", probe="array_oob")
    unk = _report(VerifyVerdict.UNKNOWN, "u", probe="div_by_zero")
    clean = _report(VerifyVerdict.NO_DIVERGENCE_FOUND, "ok", probe="signed_overflow")
    assert c.put({"name": "d"}, div) is True
    assert c.put({"name": "ok"}, clean) is True
    # candidate/unknown are environment/timeout dependent: never cached.
    assert c.put({"name": "c"}, cand) is False
    assert c.put({"name": "u"}, unk) is False
    assert len(c) == 2


def test_cache_save_load_roundtrip(tmp_path):
    c = VerificationCache(fingerprint=_fp())
    c.put({"name": "d"}, _divergent_report("d", "signed_overflow"))
    path = tmp_path / "cache.json"
    c.save(str(path))
    c2 = VerificationCache.load(str(path), fingerprint=_fp())
    assert len(c2) == 1
    entry = c2.get({"name": "d"})
    assert entry is not None
    assert entry.verdict == "divergent"
    assert entry.divergence_class == "signed_overflow"


def test_cache_miss_when_fingerprint_differs(tmp_path):
    # A cache written under one toolchain must MISS under a different toolchain,
    # so a compiler upgrade never serves a stale confirmation.
    c = VerificationCache(fingerprint=_fp())
    c.put({"name": "d"}, _divergent_report("d", "signed_overflow"))
    path = tmp_path / "cache.json"
    c.save(str(path))
    other = VerificationCache.load(str(path), fingerprint=_fp(rust="rustc NEWER"))
    assert other.get({"name": "d"}) is None


def test_cache_prune_drops_unreferenced_entries():
    c = VerificationCache(fingerprint=_fp())
    c.put({"name": "a"}, _divergent_report("a", "signed_overflow"))
    c.put({"name": "b"}, _divergent_report("b", "array_oob"))
    removed = c.prune_to([{"name": "a"}])
    assert removed == 1
    assert c.get({"name": "a"}) is not None
    assert c.get({"name": "b"}) is None


@_requires_toolchain
def test_incremental_warm_run_is_full_reuse_and_verdict_faithful():
    # Cold run really compiles+runs; warm run serves every verdict from cache,
    # and the cached verdicts match the freshly-computed ones exactly.
    units = [
        {"name": "add1_w32", "kind": "binop_const", "op": "add", "const": 1,
         "width": 32, "var": "x", "signed": True, "probe": "signed_overflow",
         "source_lang": "c", "target_lang": "rust"},
        {"name": "noovf", "kind": "binop_const", "op": "add", "const": 0,
         "width": 32, "var": "x", "signed": True, "probe": "signed_overflow",
         "source_lang": "c", "target_lang": "rust"},
    ]
    fp = toolchain_fingerprint()
    cold_cache = VerificationCache(fingerprint=fp)
    cold = verify_incremental(units, cold_cache)
    assert cold.hits == 0 and cold.misses == 2
    cold_verdicts = [r.verdict for r in cold.reports]
    assert VerifyVerdict.DIVERGENT in cold_verdicts
    assert VerifyVerdict.NO_DIVERGENCE_FOUND in cold_verdicts

    warm = verify_incremental(units, cold_cache)
    assert warm.hits == 2 and warm.misses == 0
    assert warm.hit_rate == 1.0
    # cached verdicts are faithful to the cold (real-compiler) run.
    assert [r.verdict for r in warm.reports] == cold_verdicts
    # a cached DIVERGENT still carries its class (so the fail gate / triage work).
    div = next(r for r in warm.reports if r.verdict is VerifyVerdict.DIVERGENT)
    assert div.divergence is not None
    assert div.divergence.divergence_class == "signed_overflow"


@_requires_toolchain
def test_incremental_changed_unit_is_reverified():
    units = [{"name": "add1_w32", "kind": "binop_const", "op": "add", "const": 1,
              "width": 32, "var": "x", "signed": True, "probe": "signed_overflow",
              "source_lang": "c", "target_lang": "rust"}]
    c = VerificationCache(fingerprint=toolchain_fingerprint())
    verify_incremental(units, c)
    # mutate the unit (const 1 -> 5): a different unit, hence a cache MISS.
    changed = [dict(units[0], const=5, name="add5_w32")]
    second = verify_incremental(changed, c)
    assert second.misses == 1 and second.hits == 0


# --------------------------------------------------------------------------- #
# Step 68: local, telemetry-free quality dashboard.
# --------------------------------------------------------------------------- #
from src.ub_oracle import dashboard as _dashboard  # noqa: E402
from src.ub_oracle.dashboard import (  # noqa: E402
    dashboard_data, class_risks, render_dashboard,
)


def _mixed_reports():
    return [
        _divergent_report("a", "signed_overflow"),   # CRITICAL confirmed
        _divergent_report("b", "fp_contraction"),    # MODERATE confirmed
        _report(VerifyVerdict.CANDIDATE, "c", probe="array_oob"),
        _report(VerifyVerdict.NO_DIVERGENCE_FOUND, "d", probe="signed_overflow"),
        _report(VerifyVerdict.NOT_COVERED, "e", pair=("go", "rust")),
    ]


def test_dashboard_data_counts_each_verdict():
    d = dashboard_data(_mixed_reports())
    assert d.total == 5
    assert d.confirmed == 2
    assert d.candidate == 1
    assert d.clean == 1
    assert d.not_covered == 1
    assert d.posture == "AT RISK"   # any confirmed divergence => AT RISK


def test_dashboard_posture_degrades_gracefully():
    only_clean = [_report(VerifyVerdict.NO_DIVERGENCE_FOUND, "x",
                          probe="signed_overflow")]
    assert dashboard_data(only_clean).posture == "NO DIVERGENCE FOUND"
    cand = [_report(VerifyVerdict.CANDIDATE, "x", probe="array_oob")]
    assert dashboard_data(cand).posture == "NEEDS REVIEW"


def test_class_risks_rank_critical_confirmed_first():
    rows = class_risks(_mixed_reports())
    # signed_overflow (critical, confirmed) outranks fp_contraction (moderate).
    assert rows[0].divergence_class == "signed_overflow"
    assert rows[0].risk_score > rows[1].risk_score
    so = next(r for r in rows if r.divergence_class == "signed_overflow")
    assert so.confirmed == 1 and so.severity == "critical"


def test_render_dashboard_is_offline_and_self_contained():
    html_doc = render_dashboard(_mixed_reports(), generated_at="2026-01-01 00:00")
    assert html_doc.startswith("<!DOCTYPE html>")
    # absolutely no network egress: no external scripts, fonts, or URLs.
    for bad in ("http://", "https://", "<script", "googleapis", "cdn."):
        assert bad not in html_doc
    # the honesty disclaimer is rendered in the page itself.
    assert "not a proof of equivalence" in html_doc.lower()
    # the riskiest classes and the posture are visible.
    assert "AT RISK" in html_doc
    assert "signed_overflow" in html_doc and "fp_contraction" in html_doc
    # deterministic given a fixed timestamp.
    assert html_doc == render_dashboard(_mixed_reports(),
                                        generated_at="2026-01-01 00:00")


def test_dashboard_escapes_unit_names():
    r = _divergent_report("<img src=x onerror=alert(1)>", "signed_overflow")
    html_doc = render_dashboard([r], generated_at="2026-01-01 00:00")
    # the malicious unit name is HTML-escaped, never injected raw.
    assert "<img src=x" not in html_doc
    assert "&lt;img" in html_doc


# ── performance / scalability curves (step 50) ───────────────────────────────

from src.ub_oracle import perf as _perf
from src.ub_oracle.plugin import ALL_ORACLES as _ALL_ORACLES, REGISTRY as _REGISTRY


def test_perf_deterministic_grid_is_reproducible():
    # The grid (classes x pairs x widths x SMT sizes + verdicts) carries NO
    # timings and must regenerate identically within a process.
    g1 = _perf.deterministic_grid()
    g2 = _perf.deterministic_grid()
    assert g1 == g2
    assert json.dumps(g1, sort_keys=True) == json.dumps(g2, sort_keys=True)


def test_perf_grid_covers_all_pairs_and_widths():
    g = _perf.deterministic_grid()
    pairs = {(r["source_lang"], r["target_lang"]) for r in g["class_pair_profile"]}
    assert ("c", "rust") in pairs and ("c", "go") in pairs and ("c", "swift") in pairs
    # every profiled search actually found a divergence (sanity of the grid).
    assert all(r["verdict"] == "divergent" for r in g["class_pair_profile"])
    # width sweep spans both supported integer widths for each scalable class.
    widths = {(r["divergence_class"], r["width"]) for r in g["width_scaling"]}
    for cls in ("signed_overflow", "shift_oob", "div_by_zero", "intmin_div_neg1"):
        assert (cls, 32) in widths and (cls, 64) in widths
    # SMT scaling stays satisfiable all the way up to 512-bit widths.
    assert [r["width"] for r in g["smt_scaling"]] == [8, 16, 32, 64, 128, 256, 512]
    assert all(r["result"] == "sat" for r in g["smt_scaling"])


def test_perf_smt_scaling_curve_is_not_pathological():
    # The bitvector overflow search must scale gracefully with bit width: the
    # geometric-mean per-doubling growth stays well under the pathology bar.
    curve = _perf.smt_scaling_curve(repeats=2)
    assert len(curve.seconds) == len(curve.sizes)
    assert all(s >= 0.0 for s in curve.seconds)
    assert all(v == "sat" for v in curve.verdicts)
    assert not curve.pathological
    assert curve.growth_ratio < curve.threshold
    d = curve.to_dict()
    assert d["sizes"][0] == 8 and d["sizes"][-1] == 512


def test_perf_class_pair_profile_times_real_searches():
    rows = _perf.class_pair_profile(repeats=1)
    assert rows, "expected at least one timed oracle"
    by_label = {r.label for r in rows}
    assert "c->rust:signed_overflow" in by_label
    for r in rows:
        assert r.seconds >= 0.0
        assert r.verdict == "divergent"


# ── plugin SDK: a third-party oracle registers without forking (step 70) ─────

@pytest.fixture
def _isolated_registry():
    """Ensure the external SDK oracle is registered for the test, then remove it.

    The module registers on first import; Python caches the module, so re-import
    is a no-op. This fixture therefore registers idempotently and tears down by
    class, leaving the global registry free of this non-core class so the rest of
    the suite (matrix/metrics over the core oracles) is unaffected.
    """
    from src.ub_oracle.plugin import register, oracles_for
    from examples.plugins import float_cast_overflow_oracle as ext

    cls = ext.FLOAT_CAST_OVERFLOW
    if not oracles_for("c", "rust", cls):
        register(ext.FloatCastOverflowOracle())
    try:
        yield
    finally:
        _ALL_ORACLES[:] = [o for o in _ALL_ORACLES if o.divergence_class != cls]
        _REGISTRY.pop(cls, None)


def test_external_plugin_registers_and_is_discoverable(_isolated_registry):
    from src.ub_oracle.plugin import oracles_for, get_oracle_for
    from examples.plugins import float_cast_overflow_oracle as ext

    # importing the external module is the ONLY integration step.
    matches = oracles_for("c", "rust", ext.FLOAT_CAST_OVERFLOW)
    assert len(matches) == 1
    assert isinstance(matches[0], ext.FloatCastOverflowOracle)
    # the engine can resolve it by class through the public helper.
    assert get_oracle_for(ext.FLOAT_CAST_OVERFLOW).divergence_class == \
        ext.FLOAT_CAST_OVERFLOW


def test_external_plugin_finds_divergence_symbolically(_isolated_registry):
    from examples.plugins import float_cast_overflow_oracle as ext

    oracle = ext.FloatCastOverflowOracle()
    assert oracle.applies_to(ext.EXAMPLE_UNIT)
    res = oracle.find_divergence(ext.EXAMPLE_UNIT)
    assert res.verdict is OracleVerdict.DIVERGENT
    ce = res.counterexample
    assert ce is not None and ce.divergence_class == ext.FLOAT_CAST_OVERFLOW
    # the witness is a genuine out-of-int-range finite double, Z3-found.
    x = ce.inputs["x"]
    assert x > (1 << 31) - 1
    assert "(int)x" in ce.source_snippet and "x as i32" in ce.target_snippet


def test_external_plugin_integrates_with_verify_unit(_isolated_registry):
    from examples.plugins import float_cast_overflow_oracle as ext
    from src.ub_oracle.verify import verify_unit, VerifyVerdict

    report = verify_unit(ext.EXAMPLE_UNIT, confirm=False)
    # without confirmation the witness is a CANDIDATE, never silently DIVERGENT.
    assert report.verdict in (VerifyVerdict.CANDIDATE, VerifyVerdict.DIVERGENT)
    assert report.oracle_results
    assert any(r.divergence_class == ext.FLOAT_CAST_OVERFLOW
               for r in report.oracle_results)


@_requires_toolchain
def test_external_plugin_confirms_against_real_compilers(_isolated_registry):
    from examples.plugins import float_cast_overflow_oracle as ext
    from src.ub_oracle.verify import verify_unit, VerifyVerdict

    report = verify_unit(ext.EXAMPLE_UNIT, harness=ReexecHarness(_TC))
    # C (int)x overflow traps under UBSan; Rust `x as i32` saturates (defined).
    assert report.verdict is VerifyVerdict.DIVERGENT
    assert report.divergence is not None
    rr = report.divergence.reexec
    assert rr is not None and rr.confirmed
    assert rr.ub_reachable and rr.rust_defined
    # the defined Rust outcome is the saturated i32::MAX.
    assert rr.rust_run.stdout.strip() == "2147483647"


# ── internal red-team: no false "looks equivalent" (step 84) ─────────────────

from src.ub_oracle import redteam as _redteam
from src.ub_oracle.reexec import ToolchainStatus as _ToolchainStatus

_NO_TC = _ToolchainStatus(cc=None, ubsan=False, targets=())


def test_redteam_covers_every_oracle_and_pair():
    cases = _redteam.build_cases()
    assert len(cases) >= 60
    pairs = {(c.source_lang, c.target_lang) for c in cases}
    assert ("c", "rust") in pairs and ("c", "go") in pairs and ("c", "swift") in pairs
    # every supported integer class gets its multi-width / multi-const mutations.
    classes = {c.divergence_class for c in cases}
    for cls in ("signed_overflow", "shift_oob", "div_by_zero", "intmin_div_neg1",
                "array_oob"):
        assert cls in classes
    # the signed-overflow adversary varies the constant (not just INT_MAX).
    so_consts = {c.unit.get("const") for c in cases
                 if c.divergence_class == "signed_overflow"}
    assert len(so_consts) >= 2


def test_redteam_grid_is_byte_reproducible():
    g1 = _redteam.run_redteam(status=_NO_TC, confirm=False).to_dict()
    g2 = _redteam.run_redteam(status=_NO_TC, confirm=False).to_dict()
    assert json.dumps(g1, sort_keys=True) == json.dumps(g2, sort_keys=True)


def test_redteam_symbolic_path_makes_no_equivalence_claim():
    # With no toolchain every genuinely-divergent case is a sound CANDIDATE —
    # never NO_DIVERGENCE_FOUND (the one verdict that would look "equivalent").
    report = _redteam.run_redteam(status=_NO_TC, confirm=False)
    assert report.n_cases >= 60
    assert report.sound and not report.breaches
    assert all(c.verdict == "candidate" for c in report.cases)


def test_redteam_never_reports_no_divergence_for_divergent_units():
    # Direct property check against the verifier's forbidden verdict.
    report = _redteam.run_redteam(status=_NO_TC, confirm=False)
    assert all(c.verdict != "no_divergence_found" for c in report.cases)


@_requires_toolchain
def test_redteam_confirms_all_adversarial_cases_against_real_compilers():
    # The full adversary: every semantics-preserving divergent mutation, across
    # every supported pair, must be CONFIRMED divergent by real re-execution and
    # leave ZERO soundness breaches. This is the executable form of the
    # sound-for-divergence guarantee.
    report = _redteam.run_redteam(harness=ReexecHarness(_TC), status=_TC)
    assert report.toolchain_full
    assert report.n_cases >= 60
    assert not report.breaches, [b.label for b in report.breaches]
    assert report.n_confirmed_divergent == report.n_cases
    assert report.sound


# --- Step 8: coverage ratchet gate (pure-function policy tests) -------------
import importlib.util as _ilu
_cov_gate_path = _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
    "scripts", "coverage_gate.py",
)
_cg_spec = _ilu.spec_from_file_location("coverage_gate", _cov_gate_path)
_cov_gate = _ilu.module_from_spec(_cg_spec)
_cg_spec.loader.exec_module(_cov_gate)


def test_coverage_gate_total_is_mean_of_modules():
    assert _cov_gate._total({"a": 90.0, "b": 80.0}) == 85.0
    assert _cov_gate._total({}) == 0.0


def test_coverage_gate_passes_when_at_or_above_floor():
    floor = {"mean_floor": 85.0, "modules": {"a": 80.0, "b": 90.0}}
    current = {"a": 80.0, "b": 90.0}
    ok, failures, head = _cov_gate.evaluate(current, floor)
    assert ok
    assert failures == []
    assert head == 0.0


def test_coverage_gate_fails_when_a_module_drops():
    floor = {"mean_floor": 85.0, "modules": {"a": 80.0, "b": 90.0}}
    current = {"a": 70.0, "b": 95.0}  # mean still 82.5 < 85 too
    ok, failures, _ = _cov_gate.evaluate(current, floor)
    assert not ok
    assert any("a:" in f for f in failures)
    assert any("MEAN" in f for f in failures)


def test_coverage_gate_fails_on_mean_even_if_modules_ok():
    floor = {"mean_floor": 90.0, "modules": {"a": 80.0}}
    current = {"a": 85.0}  # module above its 80 floor, but mean 85 < 90
    ok, failures, head = _cov_gate.evaluate(current, floor)
    assert not ok
    assert any("MEAN" in f for f in failures)
    assert head == -5.0


def test_coverage_gate_headroom_is_mean_minus_floor():
    floor = {"mean_floor": 80.0, "modules": {}}
    ok, _, head = _cov_gate.evaluate({"a": 90.0, "b": 92.0}, floor)
    assert ok
    assert head == 11.0


def test_coverage_gate_tolerates_subepsilon_jitter():
    floor = {"mean_floor": 85.0, "modules": {"a": 85.0}}
    ok, failures, _ = _cov_gate.evaluate({"a": 85.0 - 1e-10}, floor)
    assert ok and not failures


def test_committed_floor_is_well_formed_and_covers_core_modules():
    with open(_cov_gate.FLOOR_PATH) as f:
        floor = json.load(f)
    assert "mean_floor" in floor and floor["mean_floor"] >= 85.0
    for rel in _cov_gate.CORE_MODULES:
        assert rel in floor["modules"], rel
        assert 0.0 <= floor["modules"][rel] <= 100.0


# --- Step 78: abstract-interpretation pre-pass ------------------------------
from src.ub_oracle import abstract_interp as _ai  # noqa: E402
from src.ub_oracle.abstract_interp import Interval as _Interval  # noqa: E402
from src.ub_oracle.plugin import ALL_ORACLES as _ALL_ORACLES_AI  # noqa: E402


def _oracle_for_class(cls):
    return [o for o in _ALL_ORACLES_AI if o.divergence_class == cls][0]


def test_interval_transfer_functions_are_exact():
    iv = _Interval(0, 10)
    assert iv.add_const(5) == _Interval(5, 15)
    assert iv.sub_const(3) == _Interval(-3, 7)
    assert iv.contains(10) and not iv.contains(11)
    assert _Interval(0, 10).intersects(_Interval(10, 20))
    assert not _Interval(0, 10).intersects(_Interval(11, 20))
    assert _Interval(2, 5).subset_of(_Interval(0, 10))
    assert not _Interval(2, 15).subset_of(_Interval(0, 10))
    bottom = _Interval(1, 0)
    assert bottom.is_bottom and bottom.subset_of(_Interval(0, 10))
    assert not bottom.intersects(_Interval(0, 10))


def test_repr_interval_matches_signed_width():
    assert _ai.repr_interval(32) == _Interval(-2147483648, 2147483647)
    assert _ai.repr_interval(8) == _Interval(-128, 127)


def test_parse_range_clamps_to_type_and_rejects_garbage():
    # a declared range is intersected with the representable interval.
    iv = _ai.parse_range({"x_range": [-10 ** 12, 10 ** 12]}, "x_range", 32)
    assert iv == _ai.repr_interval(32)
    assert _ai.parse_range({}, "x_range", 32) is None
    with pytest.raises(ValueError):
        _ai.parse_range({"x_range": [5, 1]}, "x_range", 32)
    with pytest.raises(ValueError):
        _ai.parse_range({"x_range": [1, 2, 3]}, "x_range", 32)


def test_prepass_prunes_structurally_safe_add_zero():
    u = {"kind": "binop_const", "op": "add", "const": 0, "width": 32,
         "signed": True}
    pruned = _ai.prunable_classes(u)
    assert "signed_overflow" in pruned


def test_prepass_prunes_when_declared_range_forbids_overflow():
    u = {"kind": "binop_const", "op": "add", "const": 5, "width": 32,
         "signed": True, "x_range": [0, 10]}
    res = _ai.analyze_unit(u)["signed_overflow"]
    assert res.verdict is _ai.PrePassVerdict.NO_UB_REACHABLE


def test_prepass_defers_when_range_still_allows_overflow():
    u = {"kind": "binop_const", "op": "add", "const": 5, "width": 32,
         "signed": True, "x_range": [2147483640, 2147483647]}
    assert "signed_overflow" not in _ai.prunable_classes(u)


def test_prepass_prunes_div_by_zero_when_divisor_excludes_zero():
    u = {"kind": "div", "width": 32, "signed": True, "b_range": [1, 9]}
    pruned = _ai.prunable_classes(u)
    # both division-by-zero and INT_MIN/-1 become unreachable (b != 0, b != -1).
    assert "div_by_zero" in pruned and "intmin_div_neg1" in pruned


def test_prepass_defers_div_by_zero_when_range_includes_zero():
    u = {"kind": "div", "width": 32, "signed": True, "b_range": [-3, 3]}
    assert "div_by_zero" not in _ai.prunable_classes(u)


def test_prepass_prunes_intmin_when_dividend_excludes_int_min():
    u = {"kind": "div", "width": 32, "signed": True,
         "a_range": [-100, 100], "b_range": [-5, -1]}
    pruned = _ai.prunable_classes(u)
    # b can be -1 and 0 excluded, but a can never be INT_MIN -> intmin pruned,
    # div_by_zero pruned too (0 not in [-5,-1]).
    assert "intmin_div_neg1" in pruned and "div_by_zero" in pruned


def test_prepass_prunes_shift_within_width():
    u = {"kind": "shift", "width": 32, "shift_range": [0, 31]}
    assert "shift_oob" in _ai.prunable_classes(u)
    u2 = {"kind": "shift", "width": 32, "shift_range": [0, 64]}
    assert "shift_oob" not in _ai.prunable_classes(u2)
    u3 = {"kind": "shift", "width": 32, "shift_range": [-1, 10]}  # negative is UB
    assert "shift_oob" not in _ai.prunable_classes(u3)


def test_prepass_is_consistent_with_the_smt_search():
    # SOUNDNESS-CRITICAL: whenever the pre-pass prunes a class, the (now
    # range-aware) oracle's own SMT search must agree that there is no
    # divergence — so pruning is a pure accelerator, never a behavior change.
    safe_units = [
        {"kind": "binop_const", "op": "add", "const": 5, "width": 32,
         "signed": True, "x_range": [0, 10]},
        {"kind": "binop_const", "op": "sub", "const": 3, "width": 32,
         "signed": True, "x_range": [0, 10]},
        {"kind": "div", "width": 32, "signed": True, "b_range": [1, 9]},
        {"kind": "shift", "width": 32, "shift_range": [0, 31]},
    ]
    for u in safe_units:
        pruned = _ai.prunable_classes(u)
        assert pruned, u
        for cls in pruned:
            orc = _oracle_for_class(cls)
            res = orc.find_divergence(u)
            assert res.verdict is OracleVerdict.NO_DIVERGENCE_FOUND, (cls, u)


def test_verify_unit_prepass_skips_the_solver_entirely(monkeypatch):
    # When a class is pruned, the oracle's find_divergence must never be called.
    u = {"kind": "binop_const", "op": "add", "const": 5, "width": 32,
         "signed": True, "x_range": [0, 10]}
    orc = _oracle_for_class("signed_overflow")

    def _boom(_unit):  # pragma: no cover - must not run
        raise AssertionError("solver was invoked despite a sound prune")

    monkeypatch.setattr(orc, "find_divergence", _boom)
    r = verify_unit(u, confirm=False, status=_NO_TC)
    assert r.verdict is VerifyVerdict.NO_DIVERGENCE_FOUND
    assert "signed_overflow" in r.prepass_pruned


def test_verify_unit_prepass_does_not_change_the_verdict():
    # prepass=True and prepass=False must reach the SAME verdict (the pre-pass
    # only ever discharges no-divergence, never invents or hides one).
    units = [
        {"kind": "binop_const", "op": "add", "const": 5, "width": 32,
         "signed": True, "x_range": [0, 10]},                 # pruned -> NDF
        {"kind": "binop_const", "op": "add", "const": 1073741824,
         "width": 32, "signed": True},                        # not pruned -> CAND
        {"kind": "div", "width": 32, "signed": True, "b_range": [1, 9]},  # NDF
        {"kind": "div", "width": 32, "signed": True},          # CAND
    ]
    for u in units:
        a = verify_unit(u, confirm=False, status=_NO_TC, prepass=True)
        b = verify_unit(u, confirm=False, status=_NO_TC, prepass=False)
        assert a.verdict is b.verdict, u


def test_verify_unit_malformed_range_falls_back_to_full_search():
    # a malformed range must not crash verify_unit; it falls back to SMT.
    u = {"kind": "binop_const", "op": "add", "const": 1073741824,
         "width": 32, "signed": True, "x_range": [10, 0]}
    r = verify_unit(u, confirm=False, status=_NO_TC)
    assert r.verdict in (VerifyVerdict.CANDIDATE, VerifyVerdict.NO_DIVERGENCE_FOUND)
    assert r.prepass_pruned == []


@_requires_toolchain
def test_prepass_never_prunes_a_real_confirmable_divergence():
    # SOUNDNESS against real compilers: a unit whose declared range STILL admits
    # overflow must NOT be pruned, and verify_unit must confirm DIVERGENT by
    # real re-execution — while the same shape with a safe range is discharged
    # without SMT and is NOT divergent.
    risky = {"kind": "binop_const", "op": "add", "const": 5, "width": 32,
             "signed": True, "x_range": [2147483640, 2147483647]}
    assert "signed_overflow" not in _ai.prunable_classes(risky)
    r = verify_unit(risky, ReexecHarness(_TC), status=_TC)
    assert r.verdict is VerifyVerdict.DIVERGENT
    assert -2147483648 <= r.divergence.counterexample.inputs["x"] <= 2147483647

    safe = {"kind": "binop_const", "op": "add", "const": 5, "width": 32,
            "signed": True, "x_range": [0, 10]}
    r2 = verify_unit(safe, ReexecHarness(_TC), status=_TC)
    assert r2.verdict is VerifyVerdict.NO_DIVERGENCE_FOUND
    assert "signed_overflow" in r2.prepass_pruned


# --- Step 6: shared semantic-IR contract + validator ------------------------
from src.ub_oracle import ir as _ir  # noqa: E402
from src.ub_oracle.ir import (  # noqa: E402
    validate_unit as _validate_unit,
    is_valid as _is_valid,
    assert_valid as _assert_valid,
    IRValidationError as _IRValidationError,
)


def test_ir_accepts_every_canonical_unit_shape():
    good = [
        {"kind": "binop_const", "op": "add", "const": 1, "width": 32,
         "signed": True},
        {"kind": "binop_const", "op": "sub", "const": 7, "width": 64,
         "x_range": [0, 10]},
        {"kind": "shift", "width": 32, "shift_range": [0, 31]},
        {"kind": "div", "width": 32, "b_range": [1, 9]},
        {"kind": "rem", "width": 64, "a_range": [-5, 5], "b_range": [1, 2]},
        {"kind": "array_index", "length": 8},
        {"kind": "type_pun"},
        {"kind": "fp_fma"},
        {"kind": "binop_const", "op": "add", "const": 1, "target_lang": "go"},
        {"kind": "binop_const", "op": "add", "const": 1, "target_lang": "swift"},
        {"kind": "div", "width": 32, "probe": "div_by_zero"},
    ]
    for u in good:
        assert _validate_unit(u) == [], (u, _validate_unit(u))
        assert _is_valid(u)


def test_ir_rejects_bad_envelope():
    assert _validate_unit("not a dict")[0].field == "<unit>"
    assert any(e.field == "kind" for e in _validate_unit({}))
    assert any(e.field == "kind" for e in _validate_unit({"kind": 5}))


def test_ir_rejects_unsupported_width():
    errs = _validate_unit({"kind": "binop_const", "op": "add", "const": 1,
                           "width": 7})
    assert any(e.field == "width" for e in errs)


def test_ir_rejects_missing_binop_operands():
    errs = _validate_unit({"kind": "binop_const", "width": 32})
    fields = {e.field for e in errs}
    assert "op" in fields and "const" in fields


def test_ir_rejects_non_integer_const():
    errs = _validate_unit({"kind": "binop_const", "op": "add", "const": "x",
                           "width": 32})
    assert any(e.field == "const" for e in errs)
    # bool is not a valid integer operand even though bool is an int subclass.
    errs2 = _validate_unit({"kind": "binop_const", "op": "add", "const": True,
                            "width": 32})
    assert any(e.field == "const" for e in errs2)


def test_ir_rejects_bad_array_length():
    assert any(e.field == "length"
               for e in _validate_unit({"kind": "array_index"}))
    assert any(e.field == "length"
               for e in _validate_unit({"kind": "array_index", "length": 0}))
    assert any(e.field == "length"
               for e in _validate_unit({"kind": "array_index", "length": -3}))


def test_ir_rejects_malformed_range():
    for bad in ([1, 2, 3], [5, 1], "nope", [1.5, 2.0]):
        errs = _validate_unit({"kind": "binop_const", "op": "add", "const": 1,
                               "width": 32, "x_range": bad})
        assert any(e.field == "x_range" for e in errs), bad


def test_ir_rejects_unknown_language_and_probe():
    assert any(e.field == "target_lang"
               for e in _validate_unit({"kind": "type_pun",
                                        "target_lang": "haskell"}))
    assert any(e.field == "source_lang"
               for e in _validate_unit({"kind": "type_pun", "source_lang": 3}))
    assert any(e.field == "probe"
               for e in _validate_unit({"kind": "div", "width": 32,
                                        "probe": "no_such_class"}))


def test_ir_require_known_kind_flag():
    u = {"kind": "string_concat", "width": 32}
    assert _validate_unit(u) == []                       # well-formed envelope
    assert any(e.field == "kind"
               for e in _validate_unit(u, require_known_kind=True))


def test_ir_assert_valid_raises_with_all_errors():
    with pytest.raises(_IRValidationError) as ei:
        _assert_valid({"kind": "binop_const", "width": 99}, label="u0")
    msg = str(ei.value)
    assert "u0" in msg and "width" in msg and "op" in msg and "const" in msg
    assert ei.value.errors


def test_ir_bundled_manifest_validates():
    import json as _json
    import os as _os2
    root = _os2.path.dirname(_os2.path.dirname(_os2.path.abspath(__file__)))
    with open(_os2.path.join(root, "examples", "units_manifest.json")) as fh:
        data = _json.load(fh)
    units = data["units"] if isinstance(data, dict) else data
    for u in units:
        assert _validate_unit(u) == [], (u, _validate_unit(u))


def test_cli_rejects_ill_formed_manifest(tmp_path):
    from src.ub_oracle import cli as _cli2
    bad = tmp_path / "bad.json"
    bad.write_text('{"units":[{"kind":"binop_const","width":7}]}')
    rc = _cli2.run(["--units", str(bad), "--no-confirm"])
    assert rc == 2


def test_cli_no_validate_bypasses_rejection(tmp_path):
    from src.ub_oracle import cli as _cli3
    # an ill-formed-but-loadable unit: --no-validate must skip the IR gate so
    # the engine proceeds (and simply reports it NOT_COVERED, exit 0).
    bad = tmp_path / "bad.json"
    bad.write_text('{"units":[{"kind":"binop_const","op":"add","const":1,'
                   '"width":7}]}')
    rc = _cli3.run(["--units", str(bad), "--no-confirm", "--no-validate"])
    assert rc == 0


# --- Step 81: per-class completeness characterization -----------------------
from src.ub_oracle import completeness as _comp  # noqa: E402


def test_completeness_fragments_cover_the_integer_classes():
    assert set(_comp.FRAGMENTS) == {
        "signed_overflow", "shift_oob", "div_by_zero", "intmin_div_neg1"}
    for frag in _comp.FRAGMENTS.values():
        assert frag.description and callable(frag.enumerate_points)


@pytest.mark.parametrize("cls", sorted(_comp.FRAGMENTS))
def test_oracle_is_complete_on_its_characterized_fragment(cls):
    # The decisive completeness evidence: over the whole fragment grid the
    # symbolic search agrees EXACTLY with brute-forced ground truth (it reports
    # DIVERGENT iff a triggering input exists in the declared range), and every
    # reported witness is in range and genuinely triggers the UB.
    res = _comp.check_class_completeness(cls)
    assert res.n_points > 0
    # the grid must exercise BOTH sides of the boundary (some diverge, some not).
    assert 0 < res.n_with_divergence < res.n_points
    assert res.mismatches == [], res.mismatches[:5]
    assert res.bad_witnesses == [], res.bad_witnesses[:5]
    assert res.complete


def test_ground_truth_predicates_match_the_c_rules():
    # the brute-force predicates are the spec; pin their behavior at boundaries.
    assert _comp._ub_signed_overflow("add", 1, 32, _comp._smax(32))
    assert not _comp._ub_signed_overflow("add", 1, 32, _comp._smax(32) - 1)
    assert _comp._ub_signed_overflow("sub", 1, 32, _comp._smin(32))
    assert _comp._ub_shift(32, 32) and _comp._ub_shift(32, 33)
    assert not _comp._ub_shift(32, 31)
    assert _comp._ub_div_by_zero(0) and not _comp._ub_div_by_zero(1)
    assert _comp._ub_intmin_div_neg1(32, _comp._smin(32), -1)
    assert not _comp._ub_intmin_div_neg1(32, _comp._smin(32) + 1, -1)
    assert not _comp._ub_intmin_div_neg1(32, _comp._smin(32), 1)


def test_completeness_holds_across_every_registered_pair():
    # completeness established on the shared symbolic search transfers to every
    # language pair (C->Rust/Go/Swift), since the pairs differ only in the
    # emitted target program, not the witness search.
    by_pair = _comp.check_pair_completeness()
    assert ("c", "rust") in by_pair and ("c", "go") in by_pair \
        and ("c", "swift") in by_pair
    for pair, results in by_pair.items():
        assert results, pair
        for r in results:
            assert r.complete, (pair, r.divergence_class,
                                r.mismatches[:3], r.bad_witnesses[:3])


def test_out_of_fragment_classes_are_documented_not_claimed():
    # honesty: memory-shape / FP classes are explicitly NOT claimed complete.
    for k in _comp.OUT_OF_FRAGMENT:
        assert k not in _comp.FRAGMENTS


@_requires_toolchain
def test_completeness_witnesses_confirm_against_real_compilers():
    # spot-check that the completeness fragment's *positive* points are not just
    # symbolically-sound but really divergent: take one diverging unit per class
    # and confirm it end-to-end with real clang+UBSan + the target compiler.
    from src.ub_oracle.verify import verify_unit as _vu, VerifyVerdict as _VV
    confirmed = 0
    for cls, frag in _comp.FRAGMENTS.items():
        pos = [p for p in frag.enumerate_points() if p.has_divergence]
        assert pos, cls
        rep = _vu(pos[0].unit, ReexecHarness(_TC), status=_TC)
        assert rep.verdict is _VV.DIVERGENT, (cls, rep.detail)
        confirmed += 1
    assert confirmed == len(_comp.FRAGMENTS)


# ── formal divergence semantics (Step 80) ────────────────────────────────────

from src.ub_oracle import semantics as _sem


def _src(o0_rc, o0_v, o2_rc, o2_v, san):
    return _sem.SourceObservation(
        o0=_sem.Outcome(o0_rc, o0_v), o2=_sem.Outcome(o2_rc, o2_v), san_trapped=san)


def test_semantics_exploited_requires_all_three_clauses():
    # Canonical positive: UB reached, optimizer flips the value, target defined.
    pos = _sem.Observation(
        source=_src(0, "0", 0, "1", san=True),
        target=_sem.TargetObservation(defined=True), mode=_sem.EXPLOITED)
    j = _sem.judge(pos)
    assert j.diverges and j.premise_ub_reached and j.consequence_met
    assert _sem.is_divergence(pos)

    # (P) fails: sanitizer did not trap -> not rooted in UB -> NOT a divergence,
    # even though O0 and O2 disagree.
    no_ub = _sem.Observation(
        source=_src(0, "0", 0, "1", san=False),
        target=_sem.TargetObservation(defined=True), mode=_sem.EXPLOITED)
    assert not _sem.is_divergence(no_ub)
    assert "clause (P)" in _sem.judge(no_ub).reason

    # (C) fails: UB reached but O0/O2 agree (benign) -> no observed consequence.
    benign = _sem.Observation(
        source=_src(0, "5", 0, "5", san=True),
        target=_sem.TargetObservation(defined=True), mode=_sem.EXPLOITED)
    assert not _sem.is_divergence(benign)
    assert "clause (C)" in _sem.judge(benign).reason

    # (T) fails: target itself undefined -> we do not claim a divergence.
    tgt_ub = _sem.Observation(
        source=_src(0, "0", 0, "1", san=True),
        target=_sem.TargetObservation(defined=False), mode=_sem.EXPLOITED)
    assert not _sem.is_divergence(tgt_ub)
    assert "clause (T)" in _sem.judge(tgt_ub).reason


def test_semantics_trap_vs_defined_reduces_to_premise_and_target():
    # In trap_vs_defined mode the consequence IS the definedness gap: O0/O2 need
    # not differ (the program traps rather than returning a different value).
    pos = _sem.Observation(
        source=_src(-6, "", -6, "", san=True),  # crashes, no stdout
        target=_sem.TargetObservation(defined=True, deterministic=True),
        mode=_sem.TRAP_VS_DEFINED)
    assert _sem.is_divergence(pos)

    # Non-deterministic target outcome must NOT be accepted as defined.
    flaky = _sem.Observation(
        source=_src(-6, "", -6, "", san=True),
        target=_sem.TargetObservation(defined=True, deterministic=False),
        mode=_sem.TRAP_VS_DEFINED)
    assert not _sem.is_divergence(flaky)

    # No UB reached -> not a divergence.
    no_ub = _sem.Observation(
        source=_src(0, "1", 0, "1", san=False),
        target=_sem.TargetObservation(defined=True, deterministic=True),
        mode=_sem.TRAP_VS_DEFINED)
    assert not _sem.is_divergence(no_ub)


def test_semantics_rejects_unknown_mode():
    with pytest.raises(ValueError):
        _sem.Observation(
            source=_src(0, "0", 0, "1", san=True),
            target=_sem.TargetObservation(defined=True), mode="bogus")


def test_semantics_observation_from_reexec_is_none_when_unavailable():
    from src.ub_oracle.reexec import ReexecResult
    r = ReexecResult(available=False, divergence_class="signed_overflow", inputs={})
    assert _sem.observation_from_reexec(r) is None
    # The coincidence theorem holds vacuously on unavailable runs.
    assert _sem.coincides_with_harness(r)


@_requires_toolchain
def test_semantics_predicate_coincides_with_harness_on_real_programs():
    # The formal predicate must equal the harness's confirmed flag on BOTH a
    # genuinely-diverging unit and an equivalent (non-diverging) one, compiled
    # and run for real.
    harness = ReexecHarness(_TC)

    # (1) Exploited signed-overflow divergence (confirmed=True expected).
    orc = get_oracle("signed_overflow")
    unit = {"kind": "binop_const", "op": "add", "const": 1,
            "width": 32, "var": "x", "signed": True}
    res = orc.confirm(orc.find_divergence(unit), harness)
    rr = res.reexec
    assert rr.available and rr.confirmed
    obs = _sem.observation_from_reexec(rr)
    assert obs is not None
    assert _sem.is_divergence(obs) == rr.confirmed == True
    assert _sem.coincides_with_harness(rr)

    # (2) Equivalent (wrapping) translation: no UB, harness not confirmed, and
    # the formal predicate must agree (False == False).
    c_src = (
        "#include <stdio.h>\n#include <stdlib.h>\n"
        "int f(unsigned x){ return (x + 1u) > x; }\n"
        "int main(int c,char**v){ if(c<2) return 2; "
        "unsigned x=(unsigned)strtoull(v[1],0,10); printf(\"%d\\n\",f(x)); return 0; }\n"
    )
    rust_src = (
        "fn f(x:u32)->i32{ (x.wrapping_add(1) > x) as i32 }\n"
        "fn main(){ let x:u32 = std::env::args().nth(1).unwrap().parse().unwrap(); "
        "println!(\"{}\", f(x)); }\n"
    )
    rr2 = harness.confirm_ub_divergence(c_src, rust_src, ["4294967295"])
    assert rr2.available and not rr2.confirmed
    obs2 = _sem.observation_from_reexec(rr2)
    assert obs2 is not None
    assert _sem.is_divergence(obs2) == rr2.confirmed == False
    assert _sem.coincides_with_harness(rr2)


# ── theory<->implementation traceability (Step 83) ───────────────────────────

from src.ub_oracle import traceability as _trace


def test_traceability_every_claim_maps_to_code():
    # Every claim's module imports, every referenced symbol exists, and every
    # attached executable theorem core evaluates to True.
    problems = _trace.verify_traceability(run_theorems=True)
    assert problems == [], [ (p.claim_id, p.kind, p.detail) for p in problems ]


def test_traceability_claim_ids_are_unique_and_nonempty():
    ids = _trace.claim_ids()
    assert ids and len(ids) == len(set(ids))
    assert all(i.strip() for i in ids)


def test_traceability_doc_lists_every_claim_id():
    # The generated docs/TRACEABILITY.md must cite exactly the claim ids defined
    # in code — neither orphaned doc rows nor undocumented claims.
    doc = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)),
                        "docs", "TRACEABILITY.md")
    with open(doc) as f:
        text = f.read()
    for cid in _trace.claim_ids():
        assert f"`{cid}`" in text, f"claim {cid} missing from TRACEABILITY.md"


def test_traceability_structural_pass_without_theorems_is_fast_and_clean():
    # The symbol/import-only structural pass must also be clean (and not depend
    # on the theorem cores).
    assert _trace.verify_traceability(run_theorems=False) == []


def test_traceability_detects_a_broken_claim():
    # Negative control: a claim pointing at a missing symbol is reported.
    bad = _trace.Claim("X-bogus", "nonexistent", "ub_oracle.semantics",
                       ("this_symbol_does_not_exist",))
    saved = list(_trace.CLAIMS)
    _trace.CLAIMS.append(bad)
    try:
        problems = _trace.verify_traceability(run_theorems=False)
        assert any(p.claim_id == "X-bogus" and p.kind == "symbol"
                   for p in problems)
    finally:
        _trace.CLAIMS[:] = saved


# ── positioning vs adjacent verifiers (Step 82) ──────────────────────────────

def test_positioning_doc_covers_each_adjacent_tool_family():
    doc = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)),
                        "docs", "POSITIONING.md")
    with open(doc) as f:
        text = f.read()
    # Each adjacent family the step requires must be addressed by name.
    for needle in ("CBMC", "Kani", "Prusti", "equivalence", "differential",
                   "Rust", "Go", "Swift"):
        assert needle in text, f"POSITIONING.md does not address {needle!r}"
    # And it must state the gap as the design goal (the summary row).
    assert "design goal" in text


# ── uninitialized-read / definedness oracle (Step 17) ────────────────────────

from src.ub_oracle.oracles import uninit_read as _uninit
from src.ub_oracle.plugin import get_oracle_for as _get_oracle_for


def test_definedness_lattice_tracks_writes():
    # never written -> uninit; unconditional write -> defined; guarded -> maybe.
    scalar = {"kind": "uninit_read", "storage": {"kind": "scalar"},
              "writes": [], "read": None}
    assert _uninit.analyze_definedness(scalar) == {None: _uninit.UNINIT}

    arr = {"kind": "uninit_read", "storage": {"kind": "array", "length": 4},
           "writes": [{"slot": 0}], "read": 3}
    st = _uninit.analyze_definedness(arr)
    assert st[0] == _uninit.DEFINED and st[3] == _uninit.UNINIT

    cond = {"kind": "uninit_read", "storage": {"kind": "scalar"},
            "writes": [{"slot": None, "guarded": True}], "read": None}
    assert _uninit.analyze_definedness(cond) == {None: _uninit.MAYBE}


def test_uninitialized_read_detects_undefined_and_clears_defined():
    # A read of a defined slot is NOT flagged (no fabricated divergence).
    defined = {"kind": "uninit_read", "storage": {"kind": "scalar"},
               "writes": [{"slot": None}], "read": None}
    assert _uninit.uninitialized_read(defined) is None

    # A maybe-defined (guarded-only) read IS flagged.
    cond = {"kind": "uninit_read", "storage": {"kind": "scalar"},
            "writes": [{"slot": None, "guarded": True}], "read": None}
    assert _uninit.uninitialized_read(cond) == (None, _uninit.MAYBE)


def test_uninit_oracle_returns_no_divergence_for_initialized_read():
    orc = _get_oracle_for("uninit_read", "c", "rust")
    defined = {"kind": "uninit_read", "storage": {"kind": "array", "length": 3},
               "writes": [{"slot": 0}, {"slot": 1}, {"slot": 2}], "read": 1}
    res = orc.find_divergence(defined)
    assert res.verdict is OracleVerdict.NO_DIVERGENCE_FOUND


def test_uninit_oracle_registered_for_all_three_pairs():
    for tl in ("rust", "go", "swift"):
        orc = _get_oracle_for("uninit_read", "c", tl)
        assert orc.target_lang == tl
        assert orc.confirmation_mode == "optimizer_exploited"


def test_uninit_ir_validation_rejects_illformed_units():
    from src.ub_oracle.ir import validate_unit
    bad = {"kind": "uninit_read", "storage": {"kind": "array", "length": 0},
           "writes": [{"slot": 5}], "read": 9}
    errs = validate_unit(bad)
    fields = {e.field for e in errs}
    assert "storage.length" in fields
    good = {"kind": "uninit_read", "storage": {"kind": "struct", "fields": ["a", "b"]},
            "writes": [{"slot": "a"}], "read": "b"}
    assert validate_unit(good) == []


@_requires_toolchain
@pytest.mark.parametrize("target", ["rust", "go", "swift"])
@pytest.mark.parametrize("unit_name", ["scalar", "array", "struct", "cond"])
def test_uninit_divergence_confirmed_against_real_compilers(target, unit_name):
    units = {
        "scalar": {"kind": "uninit_read", "storage": {"kind": "scalar"},
                   "writes": [], "read": None},
        "array": {"kind": "uninit_read", "storage": {"kind": "array", "length": 4},
                  "writes": [{"slot": 0}], "read": 3},
        "struct": {"kind": "uninit_read",
                   "storage": {"kind": "struct", "fields": ["a", "b"]},
                   "writes": [{"slot": "a"}], "read": "b"},
        "cond": {"kind": "uninit_read", "storage": {"kind": "scalar"},
                 "writes": [{"slot": None, "guarded": True}], "read": None},
    }
    if not _TC.full_for(target):
        pytest.skip(f"toolchain missing for {target}")
    orc = _get_oracle_for("uninit_read", "c", target)
    res = orc.find_divergence(units[unit_name])
    assert res.verdict is OracleVerdict.DIVERGENT
    out = orc.confirm(res, ReexecHarness(_TC))
    rr = out.reexec
    assert rr.available, rr.reason
    # same C source under two conforming builds disagrees (UB under-determined)...
    assert rr.ub_consequential, rr.reason
    # ...while the target is defined and deterministic.
    assert rr.rust_defined, rr.reason
    assert rr.confirmed, rr.reason
    assert out.counterexample.confirmed
    # the two C builds really printed different values:
    obs = out.counterexample.source_observed
    assert obs["A"] != obs["B"], obs


# ── Step 73: counterexample-guided abstraction refinement ────────────────────
from src.ub_oracle import cegar as _cegar  # noqa: E402


def _cegar_queries():
    """A spread of guarded fragments: some equivalent, some divergent, with and
    without forced refinement."""
    return {
        # only INT_MAX (odd) overflows x+1, but the guard demands even ⟹ no UB.
        "even_guard_prunes_add1":
            _cegar.GuardedQuery("add", 1, 32, "x", (_cegar.even(),)),
        # x-1 underflows only at INT_MIN (even); odd guard ⟹ no UB (width 64).
        "odd_guard_prunes_sub1_w64":
            _cegar.GuardedQuery("sub", 1, 64, "x", (_cegar.odd(),)),
        # add 2 overflows {INT_MAX-1 (even), INT_MAX (odd)}; even guard keeps one.
        "even_guard_keeps_witness":
            _cegar.GuardedQuery("add", 2, 32, "x",
                                (_cegar.even(), _cegar.at_least(0))),
        # no guards at all ⟹ immediate witness, zero refinement.
        "unguarded_add1":
            _cegar.GuardedQuery("add", 1, 32),
        # several guards, all satisfiable together with an overflowing input.
        "multi_guard_divergent":
            _cegar.GuardedQuery("add", 5, 32, "x",
                                (_cegar.at_least(0), _cegar.multiple_of(1))),
        # contradictory guards (<=10 yet must overflow at the top) ⟹ equivalent.
        "contradictory_bound":
            _cegar.GuardedQuery("add", 1, 32, "x", (_cegar.at_most(10),)),
    }


def test_cegar_agrees_with_brute_force_ground_truth():
    """CEGAR's verdict must equal exact enumeration of the UB region on every
    guarded fragment, and a divergent verdict's witness must really overflow and
    satisfy every guard."""
    for name, q in _cegar_queries().items():
        res = _cegar.run_cegar(q)
        bf = _cegar.brute_force_witness(q)
        if res.verdict is _cegar.CegarVerdict.DIVERGENT:
            assert bf is not None, f"{name}: CEGAR says divergent, brute force finds none"
            assert res.witness is not None
            assert _cegar._overflows_concretely(res.witness, q.op, q.const, q.width), name
            assert all(p.holds_at(res.witness) for p in q.assumes), \
                f"{name}: witness violates a guard"
        else:
            assert bf is None, f"{name}: CEGAR says equivalent, brute force found {bf}"


def test_cegar_refinement_is_real_and_measured():
    """The loop must actually *refine* on path-sensitive fragments the interval
    domain cannot handle, and report honest statistics."""
    q = _cegar.GuardedQuery("add", 1, 32, "x", (_cegar.even(),))
    res = _cegar.run_cegar(q)
    assert res.verdict is _cegar.CegarVerdict.EQUIVALENT
    # it had to *learn* the even-guard (the coarse abstraction was too weak)…
    assert res.refinements >= 1
    assert res.learned_predicates == ["x % 2 == 0"]
    # …and the solver-call count is refinements + the final discharging check.
    assert res.solver_calls == res.refinements + 1
    # the trace records the spurious model that drove the refinement.
    assert any("refine" in line for line in res.trace)
    assert any("UNSAT" in line for line in res.trace)


def test_cegar_unguarded_query_needs_no_refinement():
    res = _cegar.run_cegar(_cegar.GuardedQuery("add", 1, 32))
    assert res.verdict is _cegar.CegarVerdict.DIVERGENT
    assert res.refinements == 0
    assert res.solver_calls == 1
    assert res.witness == 2147483647  # INT_MAX is the sole overflowing input


def test_cegar_loop_is_bounded_by_guard_count():
    """At most one refinement per guard, plus the final check — never more."""
    for q in _cegar_queries().values():
        res = _cegar.run_cegar(q)
        assert res.refinements <= len(q.assumes)
        assert res.solver_calls <= len(q.assumes) + 1


def test_cegar_is_deterministic():
    q = _cegar.GuardedQuery("add", 2, 32, "x", (_cegar.even(), _cegar.at_least(0)))
    a = _cegar.run_cegar(q)
    b = _cegar.run_cegar(q)
    assert (a.verdict, a.refinements, a.solver_calls, a.witness) == \
           (b.verdict, b.refinements, b.solver_calls, b.witness)


@_requires_toolchain
def test_cegar_divergent_witness_confirmed_against_real_compilers():
    """Close the loop: a CEGAR-discovered witness for a *guarded* fragment really
    makes the compiled C ``-O0``/``-O2`` builds disagree while Rust is defined."""
    q = _cegar.GuardedQuery("add", 2, 32, "x",
                            (_cegar.even(), _cegar.at_least(0)))
    res = _cegar.run_cegar(q)
    assert res.verdict is _cegar.CegarVerdict.DIVERGENT
    unit = _cegar.to_signed_overflow_unit(q, res.witness)
    orc = get_oracle("signed_overflow")
    out = orc.confirm(orc.find_divergence(unit), ReexecHarness(_TC))
    rr = out.reexec
    assert rr.available, rr.reason
    assert rr.ub_reachable, "UBSan should trap on the CEGAR witness"
    assert rr.ub_consequential, "O0 and O2 must disagree on the CEGAR witness"
    assert rr.rust_defined, "Rust must stay defined"
    assert rr.confirmed
    # and the oracle really used the CEGAR witness (range was pinned to it).
    assert out.counterexample.inputs["x"] == res.witness


# ── Step 74: k-induction for loops beyond bounded unrolling ──────────────────
from src.ub_oracle import kinduction as _ki  # noqa: E402


def test_kinduction_proves_safe_loop_with_auxiliary_invariant():
    """The modular counter never overflows; the bare no-overflow property is not
    inductive, but strengthening with the (itself-inductive) range invariant
    closes the induction at k=1 — proving safety for *unbounded* iterations."""
    ts, aux = _ki.saturating_counter(1000)
    res = _ki.prove(ts, max_k=8, aux=aux)
    assert res.verdict is _ki.KIndVerdict.SAFE
    assert res.k == 1
    assert res.aux_invariants_used == ["0<=i<M"]
    # without the strengthening lemma the property is genuinely not k-inductive
    # within the budget (a spurious unreachable CTI keeps breaking the step).
    bare = _ki.prove(ts, max_k=6)
    assert bare.verdict is _ki.KIndVerdict.UNKNOWN


def test_kinduction_rejects_non_inductive_auxiliary_lemma():
    """An auxiliary lemma that is not itself inductive must be discarded, so it
    cannot be used to (unsoundly) close the induction."""
    ts, _ = _ki.saturating_counter(1000)
    # 'i <= 10' is true initially but NOT preserved by trans (i reaches 999),
    # so the engine must refuse to use it and therefore not prove SAFE via it.
    bogus = [("i<=10", lambda s: s["i"] <= 10)]
    res = _ki.prove(ts, max_k=4, aux=bogus)
    assert res.aux_invariants_used == []           # lemma was rejected
    assert res.verdict is _ki.KIndVerdict.UNKNOWN  # and it did not help


def test_kinduction_finds_genuine_overflow_witness():
    """The accumulator reaches INT_MAX and the next update overflows; the base
    case (which starts from init) returns a *reachable* counterexample at the
    exact iteration the loop first goes undefined."""
    ts = _ki.accumulator_overflow(_ki.INT32_MAX - 2, 1)
    res = _ki.prove(ts, max_k=8)
    assert res.verdict is _ki.KIndVerdict.DIVERGENT
    assert res.witness_depth == 2
    # the witness trace ends on the last in-range state (INT_MAX); its next
    # increment is the UB.
    assert res.witness_trace[-1]["acc"] == _ki.INT32_MAX


def test_kinduction_agrees_with_brute_force_simulation():
    """k-induction's verdict and witness depth must match an independent concrete
    simulation of the same transition system."""
    # SAFE case: simulate far beyond any unrolling and observe no violation.
    ts_safe, aux = _ki.saturating_counter(1000)
    safe, viol, _ = _ki.simulate(ts_safe, 5000)
    assert safe and viol is None
    assert _ki.prove(ts_safe, max_k=8, aux=aux).proved_safe
    # DIVERGENT case: the simulator's first-violation depth equals the witness.
    ts_div = _ki.accumulator_overflow(_ki.INT32_MAX - 2, 1)
    ok, vdepth, _ = _ki.simulate(ts_div, 10)
    assert not ok
    res = _ki.prove(ts_div, max_k=8)
    assert res.witness_depth == vdepth


def test_kinduction_is_deterministic():
    ts, aux = _ki.saturating_counter(777)
    a = _ki.prove(ts, max_k=6, aux=aux)
    b = _ki.prove(ts, max_k=6, aux=aux)
    assert (a.verdict, a.k, a.aux_invariants_used) == (b.verdict, b.k, b.aux_invariants_used)


@_requires_toolchain
def test_kinduction_overflow_witness_traps_at_predicted_trip_count():
    """Close the loop on real compilers: at exactly the k-induction-predicted
    trip count the C loop executes UB (UBSan traps) while Rust stays defined —
    and one trip earlier it does *not*, so the predicted boundary is exact."""
    start, step = _ki.INT32_MAX - 2, 1
    trips = _ki.trips_to_overflow(start, step)
    # the engine's witness depth + 1 (the overflowing update) equals the trip count
    res = _ki.prove(_ki.accumulator_overflow(start, step), max_k=8)
    assert res.witness_depth + 1 == trips
    c = _ki.accumulator_c_source(start, step)
    rs = _ki.accumulator_rust_source(start, step)
    h = ReexecHarness(_TC)
    out = h.confirm_trap_vs_defined(c, rs, [str(trips)], "signed_overflow", "rust")
    assert out.confirmed, out.reason
    assert out.ub_reachable, "UBSan must trap at the predicted trip count"
    assert out.rust_defined
    # one fewer iteration stays in range — no UB, proving the boundary is exact.
    safe = h.confirm_trap_vs_defined(c, rs, [str(trips - 1)], "signed_overflow", "rust")
    assert not safe.ub_reachable, "the loop must not be UB one trip earlier"


@_requires_toolchain
@pytest.mark.parametrize("target", ["rust", "go", "swift"])
def test_kinduction_safe_loop_no_divergence_across_pairs(target):
    """The SAFE verdict transfers across pairs: at a large (effectively unbounded)
    trip count the C modular counter and the target produce the *same* defined,
    deterministic value with no UB — i.e. no divergence, exactly as k-induction
    proved for unbounded iterations."""
    if not _TC.full_for(target):
        pytest.skip(f"toolchain missing for {target}")
    M = 1000
    c = _ki.saturating_c_source(M)
    tgt_src = _ki._SATURATING_EMITTERS[target](M)
    h = ReexecHarness(_TC)
    res = h.confirm_ub_divergence(c, tgt_src, ["5000000"], "signed_overflow", target)
    assert not res.ub_reachable, "the modular counter must never be UB"
    assert not res.ub_consequential, "O0 and O2 must agree (defined)"
    assert res.rust_defined
    # C and the target agree on the defined result (5_000_000 mod 1000 == 0).
    assert res.c_runs["O0"].stdout == res.rust_run.stdout == "0"


# ---------------------------------------------------------------------------
# Step 79 — ABI / interop-divergence checks at FFI boundaries.
# ---------------------------------------------------------------------------

from src.ub_oracle import abi_layout as _abi  # noqa: E402


def test_abi_c_layout_matches_hand_computed_padding():
    lay = _abi.c_layout(_abi.hazard_struct())
    assert lay.size == 12
    assert lay.offsets == {"a": 0, "b": 4, "c": 8}
    assert lay.align == 4


def test_abi_optimized_layout_reorders_by_descending_alignment():
    opt = _abi.optimized_layout(_abi.hazard_struct())
    # int (align 4) floats to the front; the two chars pack into the tail.
    assert opt.size == 8
    assert opt.offsets == {"b": 0, "a": 4, "c": 5}


def test_abi_hazard_detected_for_suboptimal_declaration_order():
    res = _abi.abi_divergence(_abi.hazard_struct())
    assert res.is_hazard
    assert res.verdict is _abi.AbiVerdict.INTEROP_HAZARD
    assert set(res.moved_fields) == {"a", "b", "c"}
    assert "misread" in res.reason


def test_abi_safe_when_declaration_order_already_optimal():
    res = _abi.abi_divergence(_abi.safe_struct())
    assert not res.is_hazard
    assert res.verdict is _abi.AbiVerdict.INTEROP_SAFE
    assert res.moved_fields == []


def test_abi_uniform_alignment_never_diverges():
    res = _abi.abi_divergence(_abi.uniform_struct())
    assert not res.is_hazard
    assert res.c.size == 12 and res.optimized.size == 12


def test_abi_divergence_is_deterministic():
    a = _abi.abi_divergence(_abi.hazard_struct())
    b = _abi.abi_divergence(_abi.hazard_struct())
    assert (a.verdict, a.c.size, a.optimized.size, a.moved_fields) == \
           (b.verdict, b.c.size, b.optimized.size, b.moved_fields)


def test_abi_layout_differs_from_is_offset_sensitive():
    c = _abi.c_layout(_abi.hazard_struct())
    opt = _abi.optimized_layout(_abi.hazard_struct())
    assert c.differs_from(opt)
    assert not c.differs_from(c)


@_requires_toolchain
def test_abi_model_matches_real_clang_offsetof():
    """The C-ABI model must reproduce real ``clang`` ``sizeof``/``offsetof``
    field-by-field for every struct we reason about."""
    for fields in (_abi.hazard_struct(), _abi.safe_struct(), _abi.uniform_struct()):
        conf = _abi.confirm_abi(fields, cc=_TC.cc)
        assert conf.available, conf.reason
        model = _abi.c_layout(fields)
        assert conf.c_size == model.size, fields
        assert conf.c_offsets == model.offsets, fields


@_requires_toolchain
def test_abi_hazard_confirmed_by_real_rustc_default_repr():
    """Close the loop on real ``rustc``: ``#[repr(C)]`` reproduces the C layout
    exactly (interop-safe), while the default repr genuinely diverges — and the
    observed divergent layout matches our optimized model byte-for-byte."""
    rustc = _TC.target_path("rust")
    if rustc is None:
        pytest.skip("rustc unavailable")
    fields = _abi.hazard_struct()
    conf = _abi.confirm_abi(fields, cc=_TC.cc, rustc=rustc)
    assert conf.available, conf.reason
    # repr(C) is an exact mirror of the C layout (safe FFI representation).
    assert conf.rust_reprc_matches_c is True
    # the default repr really diverges, exactly as the oracle predicted.
    assert _abi.abi_divergence(fields).is_hazard
    assert conf.rust_natural_diverges is True
    # and the real divergent layout equals our optimized model.
    opt = _abi.optimized_layout(fields)
    assert conf.rust_natural_size == opt.size
    assert conf.rust_natural_offsets == opt.offsets


@_requires_toolchain
def test_abi_safe_struct_does_not_diverge_under_real_rustc():
    """Soundness: when the oracle abstains, real ``rustc`` default repr does NOT
    diverge — we never flag an interop hazard the compiler doesn't exhibit."""
    rustc = _TC.target_path("rust")
    if rustc is None:
        pytest.skip("rustc unavailable")
    for fields in (_abi.safe_struct(), _abi.uniform_struct()):
        assert not _abi.abi_divergence(fields).is_hazard
        conf = _abi.confirm_abi(fields, cc=_TC.cc, rustc=rustc)
        assert conf.rust_reprc_matches_c is True
        assert conf.rust_natural_diverges is False, fields


@_requires_toolchain
def test_abi_go_struct_is_layout_stable_like_c():
    """Go lays structs out in declaration order, so a Go struct reproduces the C
    layout exactly — the hazard is specific to reordering representations."""
    go = _TC.target_path("go")
    if go is None:
        pytest.skip("go unavailable")
    for fields in (_abi.hazard_struct(), _abi.safe_struct(), _abi.uniform_struct()):
        conf = _abi.confirm_abi(fields, cc=_TC.cc, go=go)
        assert conf.available, conf.reason
        assert conf.go_matches_c is True, fields


# ---------------------------------------------------------------------------
# Step 28 — ABI fidelity: union / enum / nested-struct layouts.
# ---------------------------------------------------------------------------


def test_abi_nested_struct_layout_matches_hand_computed():
    lay = _abi.c_layout(_abi.nested_struct())
    # inner {int p; char q} has size 8, align 4 -> outer: x@0, in@4, y@12, size 16
    assert lay.size == 16
    assert lay.offsets == {"x": 0, "in": 4, "y": 12}
    assert lay.align == 4


def test_abi_union_layout_follows_widest_member():
    lay = _abi.union_layout(_abi.mixed_union())
    assert lay.size == 8 and lay.align == 8
    assert lay.offsets == {"a": 0, "b": 0, "c": 0}


def test_abi_enum_default_repr_width_table():
    assert _abi.rust_default_enum_size(3) == 1
    assert _abi.rust_default_enum_size(256) == 1
    assert _abi.rust_default_enum_size(257) == 2
    assert _abi.rust_default_enum_size(65536) == 2
    assert _abi.rust_default_enum_size(65537) == 4


def test_abi_enum_divergence_flagged_for_small_enums():
    res = _abi.enum_abi_divergence(3)
    assert res.is_hazard
    assert res.c_size == 4 and res.rust_default_size == 1
    assert "repr(C)" in res.reason


def test_abi_enum_divergence_safe_when_widths_coincide():
    # a >2**16-variant enum needs 4 bytes in Rust too -> matches C int.
    res = _abi.enum_abi_divergence(70000)
    assert not res.is_hazard
    assert res.c_size == res.rust_default_size == 4


@_requires_toolchain
def test_abi_nested_struct_matches_real_clang():
    fields = _abi.nested_struct()
    conf = _abi.confirm_abi(fields, cc=_TC.cc)
    assert conf.available, conf.reason
    model = _abi.c_layout(fields)
    assert conf.c_size == model.size
    assert conf.c_offsets == model.offsets


@_requires_toolchain
def test_abi_union_matches_real_clang():
    conf = _abi.confirm_union(_abi.mixed_union(), cc=_TC.cc)
    assert conf.available, conf.reason
    model = _abi.union_layout(_abi.mixed_union())
    assert conf.c_size == model.size
    assert conf.c_align == model.align


@_requires_toolchain
def test_abi_enum_divergence_confirmed_by_real_rustc():
    """The enum width divergence is real: C enum is 4 bytes, the default-repr
    Rust fieldless enum is narrower, and #[repr(C)] restores the C width."""
    rustc = _TC.target_path("rust")
    if rustc is None:
        pytest.skip("rustc unavailable")
    conf = _abi.confirm_enum(3, cc=_TC.cc, rustc=rustc)
    assert conf.available, conf.reason
    assert conf.c_size == 4
    assert conf.rust_default_size == 1
    assert conf.rust_reprc_size == 4
    assert conf.rust_default_diverges is True
    # the model predicted exactly this.
    model = _abi.enum_abi_divergence(3)
    assert model.is_hazard
    assert model.c_size == conf.c_size
    assert model.rust_default_size == conf.rust_default_size


@_requires_toolchain
def test_abi_enum_model_tracks_rustc_across_variant_counts():
    rustc = _TC.target_path("rust")
    if rustc is None:
        pytest.skip("rustc unavailable")
    for n in (3, 300):
        conf = _abi.confirm_enum(n, cc=_TC.cc, rustc=rustc)
        assert conf.rust_default_size == _abi.rust_default_enum_size(n), n
        assert conf.rust_reprc_size == _abi.C_ENUM_SIZE


# ---------------------------------------------------------------------------
# Step 21 — byte-addressed provenance memory model (spatial/temporal safety).
# ---------------------------------------------------------------------------

from src.ub_oracle import memory_model as _mm  # noqa: E402


def test_memory_model_flags_spatial_oob():
    f = _mm.first_fault(_mm.oob_trace())
    assert f is not None and f.kind is _mm.FaultKind.OOB_SPATIAL


def test_memory_model_flags_use_after_free():
    f = _mm.first_fault(_mm.uaf_trace())
    assert f is not None and f.kind is _mm.FaultKind.USE_AFTER_FREE


def test_memory_model_flags_double_free():
    f = _mm.first_fault(_mm.double_free_trace())
    assert f is not None and f.kind is _mm.FaultKind.DOUBLE_FREE


def test_memory_model_accepts_safe_traces():
    assert _mm.first_fault(_mm.safe_trace()) is None
    assert _mm.first_fault(_mm.safe_boundary_trace()) is None


def test_memory_model_boundary_is_exact():
    # last legal byte of an 8-byte object is in bounds; one past is not.
    assert _mm.first_fault([_mm.Alloc("p", 8), _mm.Load("p", 7, 1)]) is None
    assert _mm.first_fault([_mm.Alloc("p", 8), _mm.Load("p", 8, 1)]).kind \
        is _mm.FaultKind.OOB_SPATIAL
    # a 4-byte load straddling the end is OOB even though byte 7 alone is legal.
    assert _mm.first_fault([_mm.Alloc("p", 8), _mm.Load("p", 5, 4)]).kind \
        is _mm.FaultKind.OOB_SPATIAL


def test_memory_model_provenance_is_per_allocation():
    # an in-bounds access to q must not be excused by p's size and vice versa;
    # freeing p does not free q.
    trace = [_mm.Alloc("p", 4), _mm.Alloc("q", 16), _mm.Free("p"),
             _mm.Load("q", 12, 4)]
    assert _mm.first_fault(trace) is None
    # but reading p after it is freed is a UAF, regardless of q being alive.
    trace2 = trace + [_mm.Load("p", 0, 1)]
    assert _mm.first_fault(trace2).kind is _mm.FaultKind.USE_AFTER_FREE


def test_memory_model_is_deterministic():
    a = _mm.simulate(_mm.oob_trace())
    b = _mm.simulate(_mm.oob_trace())
    assert (a.fault.kind, a.steps) == (b.fault.kind, b.steps)


@_requires_toolchain
@pytest.mark.parametrize("name,trace,expect_kind", [
    ("oob", _mm.oob_trace(), "oob_spatial"),
    ("uaf", _mm.uaf_trace(), "use_after_free"),
    ("double_free", _mm.double_free_trace(), "double_free"),
])
def test_memory_model_faults_confirmed_by_real_asan(name, trace, expect_kind):
    """Each predicted fault is confirmed by AddressSanitizer on real compiled
    code — and ASan's reported fault *kind* matches the model's."""
    conf = _mm.confirm_memory(trace, cc=_TC.cc)
    assert conf.available, conf.reason
    assert conf.predicted_fault is not None
    assert conf.asan_trapped is True
    assert conf.consistent
    assert _mm._asan_kind(conf.asan_report) == expect_kind


@_requires_toolchain
@pytest.mark.parametrize("trace", [_mm.safe_trace(), _mm.safe_boundary_trace()])
def test_memory_model_safe_traces_run_clean_under_asan(trace):
    """Soundness: when the model abstains, ASan does not trap — no fabricated
    memory bug."""
    conf = _mm.confirm_memory(trace, cc=_TC.cc)
    assert conf.available, conf.reason
    assert conf.predicted_fault is None
    assert conf.asan_trapped is False
    assert conf.consistent


# ---------------------------------------------------------------------------
# Step 77 — pointer-provenance (PNVI) memory model.
# ---------------------------------------------------------------------------

from src.ub_oracle import provenance as _pv  # noqa: E402


def test_provenance_one_past_formable_but_not_dereferenceable():
    # forming the one-past-the-end pointer is legal...
    assert _pv.first_fault(_pv.one_past_form_only()) is None
    # ...but dereferencing it is out of bounds.
    f = _pv.first_fault(_pv.one_past_form_then_deref())
    assert f is not None and f.kind is _pv.ProvFault.DEREF_OOB
    assert "one-past-the-end" in f.detail


def test_provenance_preserved_across_arithmetic_roundtrip():
    assert _pv.first_fault(_pv.arithmetic_roundtrip()) is None


def test_provenance_formation_oob_is_a_fault_before_any_deref():
    f = _pv.first_fault(_pv.formation_out_of_bounds())
    assert f is not None and f.kind is _pv.ProvFault.FORMATION_OOB


def test_provenance_integer_roundtrip_requires_exposure():
    # exposed pointer -> provenance recovered -> deref safe.
    assert _pv.first_fault(_pv.exposed_roundtrip_recovers_provenance()) is None
    # opaque integer -> no provenance -> deref undefined.
    f = _pv.first_fault(_pv.opaque_int_has_no_provenance())
    assert f is not None and f.kind is _pv.ProvFault.NO_PROVENANCE


def test_provenance_unexposed_roundtrip_loses_provenance():
    # same as exposed roundtrip but WITHOUT the Expose step: provenance is not
    # recovered, so the deref of the rebuilt pointer is undefined.
    trace = [_pv.Alloc("a", 16), _pv.Form("p", "a", 0),
             _pv.FromExposedAddr("q", "p"), _pv.Deref("q", 4)]
    f = _pv.first_fault(trace)
    assert f is not None and f.kind is _pv.ProvFault.NO_PROVENANCE


def test_provenance_free_revokes_provenance():
    f = _pv.first_fault(_pv.use_after_free_via_provenance())
    assert f is not None and f.kind is _pv.ProvFault.USE_AFTER_FREE


def test_provenance_arithmetic_out_of_range_is_formation_fault():
    # offsetting past one-past-the-end is an out-of-bounds pointer formation.
    trace = [_pv.Alloc("a", 16), _pv.Form("p", "a", 0), _pv.Add("p", "p", 17)]
    f = _pv.first_fault(trace)
    assert f is not None and f.kind is _pv.ProvFault.FORMATION_OOB
    # exactly one-past (offset == size) is allowed to form.
    ok = [_pv.Alloc("a", 16), _pv.Form("p", "a", 0), _pv.Add("p", "p", 16)]
    assert _pv.first_fault(ok) is None


def test_provenance_interface_is_documented():
    keys = _pv.PROVENANCE_INTERFACE
    for required in ("pointer_carries_provenance",
                     "arithmetic_preserves_provenance",
                     "one_past_the_end_is_formable_not_dereferenceable",
                     "integer_roundtrip_requires_exposure",
                     "cross_provenance_access_is_undefined",
                     "free_revokes_provenance"):
        assert required in keys and keys[required]


def test_provenance_is_deterministic():
    a = _pv.simulate(_pv.one_past_form_then_deref())
    b = _pv.simulate(_pv.one_past_form_then_deref())
    assert (a.fault.kind, a.steps) == (b.fault.kind, b.steps)


@_requires_toolchain
@pytest.mark.parametrize("scenario", sorted(_pv.CONFIRMABLE))
def test_provenance_scenarios_confirmed_by_real_asan(scenario):
    """The PNVI distinctions are confirmed on real compiled code: forming and
    comparing a one-past-the-end pointer runs clean, dereferencing it traps under
    ASan, and an in-bounds arithmetic round-trip is safe — each exactly as the
    model predicts."""
    conf = _pv.confirm_provenance(scenario, cc=_TC.cc)
    assert conf.available, conf.reason
    assert conf.consistent, (scenario, conf.predicted_fault, conf.asan_trapped)
    _, predicts_fault = _pv.CONFIRMABLE[scenario]
    assert conf.asan_trapped is predicts_fault


# ---------------------------------------------------------------------------
# Step 76 — ownership / borrow facts from the real Rust borrow checker.
# ---------------------------------------------------------------------------

from src.ub_oracle import ownership as _ow  # noqa: E402


def test_ownership_patterns_are_well_formed():
    for name, pat in _ow.PATTERNS.items():
        assert pat.name == name
        assert pat.rust_src and pat.c_gloss and pat.consequence
        # rejected patterns must name an error code; accepted ones must not.
        assert bool(pat.error_code) == (not pat.accepts)


def test_ownership_interface_is_documented():
    keys = _ow.OWNERSHIP_INTERFACE
    for required in ("ownership_fact_is_a_checker_verdict",
                     "rejection_forces_a_translation_choice",
                     "acceptance_licenses_alias_assumptions",
                     "retargetable"):
        assert required in keys and keys[required]


def test_ownership_unknown_pattern_raises():
    with pytest.raises(KeyError):
        _ow.pattern("does_not_exist")


@_requires_toolchain
@pytest.mark.parametrize("name", sorted(_ow.PATTERNS))
def test_ownership_facts_confirmed_by_real_rustc(name):
    """Every predicted borrow-check verdict is confirmed by the real rustc
    borrow checker, including the exact error code on rejection."""
    rustc = _TC.target_path("rust")
    if rustc is None:
        pytest.skip("rustc unavailable")
    pat = _ow.pattern(name)
    conf = _ow.confirm_ownership(name, rustc=rustc)
    assert conf.available, conf.reason
    assert conf.accepted is pat.accepts, (name, conf.stderr)
    assert conf.matches(pat), (name, conf.error_code, conf.stderr)


@_requires_toolchain
def test_ownership_mutable_aliasing_is_rejected_but_unsafe_compiles():
    """The headline ownership fact: C mutable aliasing has no safe Rust analogue
    (borrow-check rejects two &mut), yet the unsafe raw-pointer re-expression
    compiles — exactly the translator's dilemma."""
    rustc = _TC.target_path("rust")
    if rustc is None:
        pytest.skip("rustc unavailable")
    rejected = _ow.confirm_ownership("two_mut_borrows", rustc=rustc)
    assert rejected.accepted is False
    assert rejected.error_code == "E0499"
    unsafe = _ow.confirm_ownership("raw_ptr_aliasing", rustc=rustc)
    assert unsafe.accepted is True


# ---------------------------------------------------------------------------
# Step 30 — robust cross-unit function alignment (signature + call-graph).
# ---------------------------------------------------------------------------

from src.ub_oracle import unit_alignment as _ua  # noqa: E402


def test_alignment_type_compatibility_uses_c_to_target_map():
    assert _ua.types_compatible("int", "i32")
    assert _ua.types_compatible("char*", "*const u8")
    assert _ua.types_compatible("double", "f64")
    assert _ua.types_compatible("void", "()")
    assert not _ua.types_compatible("int", "i64")
    assert not _ua.types_compatible("int", "*const u8")


def test_alignment_signature_score_rewards_exact_match():
    a = _ua.FunctionSig("f", ("int", "int"), "int")
    b = _ua.FunctionSig("g", ("i32", "i32"), "i32")
    assert _ua.signature_score(a, b) == 1.0
    # arity mismatch is a hard zero.
    c = _ua.FunctionSig("h", ("i32",), "i32")
    assert _ua.signature_score(a, c) == 0.0


def test_alignment_recovers_true_pairs_on_renamed_module():
    c, t = _ua.example_c_unit(), _ua.example_target_unit()
    truth = _ua.example_ground_truth()
    res = _ua.align(c, t)
    assert res.mapping == truth
    assert _ua.alignment_accuracy(res.mapping, truth) == 1.0
    assert not res.unmatched_c and not res.unmatched_target


def test_alignment_beats_name_only_baseline_on_adversarial_names():
    c, t = _ua.example_c_unit(), _ua.example_target_unit()
    truth = _ua.example_ground_truth()
    structural = _ua.alignment_accuracy(_ua.align(c, t).mapping, truth)
    baseline = _ua.alignment_accuracy(_ua.name_only_align(c, t), truth)
    # the structural matcher is perfect; name-only is misled by colliding names.
    assert structural == 1.0
    assert baseline < structural
    # specifically, name-only mis-pairs the 2-arg `add` with 1-arg `add_one`.
    assert _ua.name_only_align(c, t)["add"] == "add_one"


def test_alignment_honours_user_pins():
    c, t = _ua.example_c_unit(), _ua.example_target_unit()
    # pin a deliberately wrong pair; the solver must respect it.
    res = _ua.align(c, t, pins={"add": "add_one"})
    assert res.mapping["add"] == "add_one"
    assert res.scores["add"] == 1.0
    # the rest are still aligned around the pin without reusing the pinned target.
    assert "add_one" not in [v for k, v in res.mapping.items() if k != "add"]


def test_alignment_reports_unmatched_when_confidence_too_low():
    c = _ua.Unit((_ua.FunctionSig("lonely", ("int", "int", "int"), "int"),))
    t = _ua.Unit((_ua.FunctionSig("totally_different", ("f64",), "f64"),))
    res = _ua.align(c, t)
    # arity/type-incompatible and name-distant -> below the confidence floor.
    assert "lonely" in res.unmatched_c
    assert "totally_different" in res.unmatched_target
    assert "lonely" not in res.mapping


def test_alignment_is_deterministic():
    c, t = _ua.example_c_unit(), _ua.example_target_unit()
    assert _ua.align(c, t).mapping == _ua.align(c, t).mapping


# ---------------------------------------------------------------------------
# Step 35 — foreign-effect / soundness-frontier detector (abstain loudly).
# ---------------------------------------------------------------------------

from src.ub_oracle import foreign_effects as _fe  # noqa: E402

_clang_present = pytest.mark.skipif(
    not _os.path.exists(_fe.CC), reason="clang not available")


def test_frontier_clear_on_pure_fragment():
    pure = "int add(int a,int b){return a+b;}\nint dbl(int*p){return *p+*p;}"
    v = _fe.decide(pure)
    assert v.clear and v.status == "CLEAR"
    assert v.reasons == []


@pytest.mark.parametrize("src,kind", [
    ("int f(volatile int*p){return *p;}", _fe.ForeignKind.VOLATILE),
    ('int f(int x){int y;__asm__("mov %1,%0":"=r"(y):"r"(x));return y;}',
     _fe.ForeignKind.INLINE_ASM),
    ("extern int g(int);\nint f(int x){return g(x);}",
     _fe.ForeignKind.FOREIGN_CALL),
    ("#include <stdatomic.h>\nint f(_Atomic int*p){return atomic_load(p);}",
     _fe.ForeignKind.ATOMIC),
    ("#include <setjmp.h>\njmp_buf b;\nint f(){return setjmp(b);}",
     _fe.ForeignKind.NONLOCAL_JUMP),
    ("#include <signal.h>\nvoid f(){signal(2,0);}", _fe.ForeignKind.SIGNAL),
])
def test_frontier_abstains_on_each_foreign_construct(src, kind):
    v = _fe.decide(src)
    assert not v.clear and v.status == "ABSTAIN"
    assert kind in v.kinds
    # abstention must be loud: a human-readable reason naming the construct.
    assert any(kind.value in r for r in v.reasons)
    assert "ABSTAIN" in v.loud_message()


def test_frontier_ignores_comments_and_string_literals():
    cmt = "int f(int x){/* volatile asm setjmp */ return x;}\n// volatile longjmp"
    strlit = 'int f(){const char*s="volatile asm atomic_load"; return 0;}'
    assert _fe.decide(cmt).clear
    assert _fe.decide(strlit).clear


def test_frontier_does_not_flag_pure_libc_or_keywords():
    src = ("int f(int*p,int n){int s=0;for(int i=0;i<n;i++){if(p[i])"
           "s+=p[i];}return s+abs(s);}")
    assert _fe.decide(src).clear


@_clang_present
def test_frontier_volatile_opacity_confirmed_by_clang_ir():
    c = _fe.confirm_volatile_opaque()
    assert c is not None and c.ok
    # the pure-model fold (one load) is provably unsound: clang keeps >= 4.
    assert "kept=4" in c.detail


@_clang_present
def test_frontier_inline_asm_opacity_confirmed_by_clang_ir():
    c = _fe.confirm_inline_asm_opaque()
    assert c is not None and c.ok


@_clang_present
def test_frontier_foreign_call_opacity_confirmed_by_clang_ir():
    c = _fe.confirm_foreign_call_opaque()
    assert c is not None and c.ok


@_clang_present
def test_frontier_atomic_opacity_confirmed_by_clang_ir():
    c = _fe.confirm_atomic_opaque()
    assert c is not None and c.ok


@_clang_present
def test_frontier_all_confirmations_pass():
    cs = _fe.confirm_all()
    assert len(cs) == 4 and all(c.ok for c in cs)


# ---------------------------------------------------------------------------
# Step 34 — concurrency / data-race awareness (TSan + Go race detector).
# ---------------------------------------------------------------------------

from src.ub_oracle import concurrency as _co  # noqa: E402

_tsan_present = pytest.mark.skipif(
    not _os.path.exists(_co.CC), reason="clang/tsan not available")
_go_present = pytest.mark.skipif(
    not _os.path.exists(_co.GO), reason="go not available")


def test_concurrency_pattern_catalogue_is_consistent():
    # exactly one racy pattern; every pattern carries a Rust migration story.
    racy = [p for p in _co.PATTERNS.values() if p.races]
    assert [p.name for p in racy] == ["unsynchronized_counter"]
    for p in _co.PATTERNS.values():
        assert p.c_source and p.go_source and p.rust_story


@_tsan_present
def test_concurrency_unsynchronized_counter_is_a_real_tsan_race():
    r = _co.confirm_race("unsynchronized_counter", check_go=False)
    assert r.c.available and r.c.race_detected is True
    assert r.ok


@_tsan_present
@pytest.mark.parametrize("name", ["mutex_counter", "atomic_counter",
                                  "readonly_shared"])
def test_concurrency_synchronized_patterns_are_race_free_under_tsan(name):
    r = _co.confirm_race(name, check_go=False)
    assert r.c.available and r.c.race_detected is False
    assert r.ok


@_tsan_present
@_go_present
def test_concurrency_c_and_go_detectors_agree_on_the_race():
    # the SAME unsynchronized-counter idiom is a race on both the C source
    # (ThreadSanitizer) and the Go target (go run -race) — the cross-language story.
    r = _co.confirm_race("unsynchronized_counter", check_go=True)
    assert r.c.race_detected is True
    assert r.go.available and r.go.race_detected is True
    assert r.ok


@_tsan_present
@_go_present
def test_concurrency_mutex_is_clean_on_both_c_and_go():
    r = _co.confirm_race("mutex_counter", check_go=True)
    assert r.c.race_detected is False
    assert r.go.race_detected is False
    assert r.ok


# ---------------------------------------------------------------------------
# Step 29 — function-pointer / indirect-call resolution (dispatch tables).
# ---------------------------------------------------------------------------

from src.ub_oracle import indirect_calls as _ic  # noqa: E402

_clang_for_ic = pytest.mark.skipif(
    not _os.path.exists(_ic.CC), reason="clang not available")


def test_indirect_parser_extracts_functions_typedef_and_table():
    u = _ic.parse_unit(_ic.EXAMPLE_DISPATCH)
    assert set(u.functions) == {"add", "sub", "mul", "log_msg", "main"}
    assert u.functions["add"].sig == _ic.Signature("int", ("int", "int"))
    assert u.functions["log_msg"].sig == _ic.Signature("void", ("const char*",))
    assert u.typedefs["op_t"] == _ic.Signature("int", ("int", "int"))
    assert u.tables["table"].entries == ("add", "sub", "mul")


def test_indirect_precise_points_to_is_the_table_entries():
    u = _ic.parse_unit(_ic.EXAMPLE_DISPATCH)
    assert _ic.resolve_table_call(u, "table") == {"add", "sub", "mul"}


def test_indirect_signature_typing_excludes_incompatible_decoy():
    u = _ic.parse_unit(_ic.EXAMPLE_DISPATCH)
    # the conservative signature-typed set still excludes log_msg (wrong sig),
    # and never main (different sig) — so signature typing is the precision lever.
    compat = _ic.signature_compatible_targets(u, "op_t")
    assert compat == {"add", "sub", "mul"}
    assert "log_msg" not in compat and "main" not in compat


def test_indirect_table_is_well_typed_precise_refines_conservative():
    u = _ic.parse_unit(_ic.EXAMPLE_DISPATCH)
    assert _ic.table_is_well_typed(u, "table")
    precise = _ic.resolve_table_call(u, "table")
    conservative = _ic.signature_compatible_targets(u, "op_t")
    assert precise.issubset(conservative)


@_clang_for_ic
def test_indirect_resolution_is_exact_against_real_execution():
    c = _ic.confirm_table_dispatch(_ic.EXAMPLE_DISPATCH, "table")
    assert c.available and c.ok
    # the program drives every index, so the observed indirect targets are
    # exactly the predicted points-to set (precision), and never escape it.
    assert c.sound
    assert c.exact
    assert c.observed == {"add", "sub", "mul"}


@_clang_for_ic
def test_indirect_decoy_function_is_never_an_observed_target():
    c = _ic.confirm_table_dispatch(_ic.EXAMPLE_DISPATCH, "table")
    assert "log_msg" not in c.observed
    assert "main" not in c.observed
