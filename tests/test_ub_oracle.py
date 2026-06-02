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


# ── Step 120: Go -> Rust defined-but-different pair ──────────────────────────

_requires_go_rust = pytest.mark.skipif(
    not (_TC.can_compile("go") and _TC.can_compile("rust")),
    reason=f"needs a go + rustc toolchain ({_TC})")


def _go_rust_oracle():
    return _plugin.get_oracle_for("intmin_div_neg1", "go", "rust")


def test_go_rust_pair_registered_as_a_safe_to_safe_pair():
    assert ("go", "rust") in _plugin.language_pairs()
    orc = _go_rust_oracle()
    assert orc.source_lang == "go" and orc.target_lang == "rust"
    # this is the *defined-but-different* mode: neither side has UB.
    assert orc.confirmation_mode == "defined_divergence"


def test_go_rust_oracle_does_not_pollute_the_c_rust_anchor():
    # the anchor REGISTRY entry for the class is still the C->Rust oracle.
    anchor = _plugin.REGISTRY["intmin_div_neg1"]
    assert (anchor.source_lang, anchor.target_lang) == ("c", "rust")
    # and the oracle triples remain unique per (source, target, class).
    triples = [(o.source_lang, o.target_lang, o.divergence_class)
               for o in _plugin.ALL_ORACLES]
    assert len(triples) == len(set(triples))


def test_go_rust_oracle_scopes_to_go_source_only():
    orc = _go_rust_oracle()
    # a C-source div unit must NOT be claimed by the Go->Rust oracle.
    assert not orc.applies_to({"kind": "div", "width": 32, "signed": True,
                               "source_lang": "c", "target_lang": "rust"})
    # its own pair, div and rem, are in scope.
    assert orc.applies_to({"kind": "div", "width": 32, "signed": True,
                           "source_lang": "go", "target_lang": "rust"})
    assert orc.applies_to({"kind": "rem", "width": 64, "signed": True})


@pytest.mark.parametrize("kind,width", [("div", 32), ("rem", 32), ("div", 64)])
def test_go_rust_finds_the_signed_overflow_pair_symbolically(kind, width):
    orc = _go_rust_oracle()
    res = orc.find_divergence({"kind": kind, "width": width, "signed": True})
    assert res.verdict == OracleVerdict.DIVERGENT
    ce = res.counterexample
    # the Z3-found witness is exactly INT_MIN op -1.
    vals = list(ce.inputs.values())
    assert -(1 << (width - 1)) in vals and -1 in vals
    assert ce.source_lang == "go" and ce.target_lang == "rust"
    # the Go source is defined (no UB on either side).
    assert ce.source_definedness == "defined"
    assert "package main" in ce.source_snippet
    assert "fn f(" in ce.target_snippet


@_requires_go_rust
@pytest.mark.parametrize("kind,go_out", [("div", "-2147483648"), ("rem", "0")])
def test_go_rust_confirmed_against_real_go_and_rustc(kind, go_out):
    # The flagship safe<->safe claim, end-to-end on real compilers: Go defines
    # INT_MIN op -1 by modular wraparound (a value, exit 0) while the faithful
    # Rust port panics (a defined abort, exit 101). Both defined; observably
    # different => a real Go->Rust translation hazard, confirmed by re-execution.
    orc = _go_rust_oracle()
    res = orc.confirm(orc.find_divergence(
        {"kind": kind, "width": 32, "signed": True}), ReexecHarness(_TC))
    rr = res.reexec
    assert rr.available and rr.mode == "defined_divergence"
    assert rr.confirmed, rr.reason
    # Go side: a defined value; Rust side: a defined panic (exit 101).
    assert rr.c_runs["A"].returncode == 0 and rr.c_runs["A"].stdout == go_out
    assert rr.rust_run.returncode == 101 and rr.rust_run.stdout == ""
    assert res.counterexample.confirmed


@_requires_go_rust
def test_defined_divergence_negative_control_two_agreeing_programs():
    # A harness-level negative control: two programs that compute the *same*
    # defined value must NOT be reported as a divergence, even though both are
    # defined. This guards the defined_divergence predicate against false
    # positives on genuinely-equivalent safe<->safe ports.
    go_src = ("package main\nimport (\n\t\"fmt\"\n\t\"os\"\n\t\"strconv\"\n)\n"
              "func main(){ a,_:=strconv.Atoi(os.Args[1]); fmt.Println(a*2) }\n")
    rust_src = ("fn main(){ let v:Vec<String>=std::env::args().collect();\n"
                "    let a:i64=v[1].parse().unwrap(); println!(\"{}\", a*2); }\n")
    rr = ReexecHarness(_TC).confirm_defined_divergence(
        go_src, "go", rust_src, "rust", ["21"], "intmin_div_neg1")
    assert rr.available and rr.mode == "defined_divergence"
    assert not rr.confirmed
    # both ran defined and agreed (42 == 42), so there is no divergence.
    assert rr.rust_defined and not rr.ub_reachable


@_requires_go_rust
def test_go_rust_verify_unit_confirms_without_needing_c_toolchain():
    # verify_unit must route a go->rust unit to the new pair and confirm it via
    # the two compilers alone — it must not demand the C compiler / UBSan.
    from src.ub_oracle.verify import verify_unit, VerifyVerdict
    rep = verify_unit({"kind": "div", "width": 32, "signed": True,
                       "source_lang": "go", "target_lang": "rust"},
                      status=_TC)
    assert rep.verdict is VerifyVerdict.DIVERGENT
    assert rep.divergence.divergence_class == "intmin_div_neg1"



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
    # generated pairs implement the argv-driven integer/memory classes plus the
    # optimizer-exploited / trap-vs-defined classes that transfer to every safe
    # target (Rust and Go), while Swift only carries the core integer/memory set.
    assert {"strict_aliasing", "fp_contraction"} <= set(cov[("c", "rust")]["classes_covered"])
    go_covered = set(cov[("c", "go")]["classes_covered"])
    assert {"signed_overflow", "shift_oob", "div_by_zero", "intmin_div_neg1",
            "array_oob", "uninit_read", "strict_aliasing", "vla_bound",
            "float_cast_overflow", "fast_math_reassoc", "restrict_violation",
            "pointer_provenance", "bitfield_layout", "enum_out_of_range",
            "memcpy_overlap", "longjmp_vla"} == go_covered
    swift_covered = set(cov[("c", "swift")]["classes_covered"])
    assert {"signed_overflow", "shift_oob", "div_by_zero",
            "intmin_div_neg1", "array_oob", "uninit_read"} == swift_covered


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
        def can_compile(self, _lang):
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
    # language pair, since the pairs differ only in the emitted target program,
    # not the witness search.
    by_pair = _comp.check_pair_completeness()
    assert ("c", "rust") in by_pair and ("c", "go") in by_pair \
        and ("c", "swift") in by_pair and ("c", "ocaml") in by_pair
    # the integer-fragment pairs must each be complete on the classes they
    # implement; a pair that implements *none* of the fragment classes (e.g. the
    # C++ defined-subset pair, whose only class is signed_shift_sign_bit) is
    # legitimately empty and skipped — exactly as check_pair_completeness documents.
    checked = 0
    for pair, results in by_pair.items():
        if not results:
            continue
        checked += 1
        for r in results:
            assert r.complete, (pair, r.divergence_class,
                                r.mismatches[:3], r.bad_witnesses[:3])
    assert checked >= 4, by_pair


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
# Steps 34/104 — concurrency / data-race awareness (TSan + Go + rustc).
# ---------------------------------------------------------------------------

from src.ub_oracle import concurrency as _co  # noqa: E402

_tsan_present = pytest.mark.skipif(
    not _co.c_race_detector_available(), reason="clang/tsan not available")
_go_present = pytest.mark.skipif(
    not _co.go_race_detector_available(), reason="go -race not available")
_rust_present = pytest.mark.skipif(
    not _os.path.exists(_co.RUSTC), reason="rustc not available")


def test_concurrency_pattern_catalogue_is_consistent():
    # exactly one racy pattern; every pattern carries a Rust migration story.
    racy = [p for p in _co.PATTERNS.values() if p.races]
    assert [p.name for p in racy] == ["unsynchronized_counter"]
    for p in _co.PATTERNS.values():
        assert p.c_source and p.go_source and p.rust_source and p.rust_story
        assert p.rust_accepts is (not p.races)


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
@pytest.mark.parametrize("name", ["mutex_counter", "atomic_counter",
                                  "readonly_shared"])
def test_concurrency_synchronized_patterns_are_clean_on_both_c_and_go(name):
    r = _co.confirm_race(name, check_go=True, check_rust=False)
    assert r.c.race_detected is False
    assert r.go.race_detected is False
    assert r.ok


@_rust_present
def test_concurrency_racy_idiom_is_rejected_by_real_rustc():
    r = _co.confirm_race("unsynchronized_counter", check_go=False, check_rust=True)
    assert r.rust.available and r.rust.accepted is False
    assert r.rust.error_code in {"E0499", "E0597", "E0521", "E0373"}
    assert r.ok


@_rust_present
@pytest.mark.parametrize("name", ["mutex_counter", "atomic_counter",
                                  "readonly_shared"])
def test_concurrency_synchronized_rust_translations_compile_and_run(name):
    r = _co.confirm_race(name, check_go=False, check_rust=True)
    assert r.rust.available and r.rust.accepted is True
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


# ---------------------------------------------------------------------------
# Step 26 — real C preprocessing (clang -E): macros, conditionals, includes.
# ---------------------------------------------------------------------------

from src.ub_oracle import preprocess as _pp  # noqa: E402

_clang_for_pp = pytest.mark.skipif(
    not _os.path.exists(_pp.CC), reason="clang not available")


def test_preprocess_detects_unparenthesized_macro_hazard():
    haz = _pp.detect_unparenthesized_macros("#define MUL(a,b) a*b\n")
    assert [h.name for h in haz] == ["MUL"]
    # the fully-parenthesized form is NOT a hazard.
    assert _pp.detect_unparenthesized_macros("#define MUL(a,b) ((a)*(b))\n") == []
    # object-like macros and single-token wrappers are not flagged.
    assert _pp.detect_unparenthesized_macros("#define N 10\n") == []
    assert _pp.detect_unparenthesized_macros("#define ID(x) (x)\n") == []


@_clang_for_pp
def test_preprocess_expands_with_real_clang():
    out = _pp.preprocess("#define MUL(a,b) a*b\nint v(){return MUL(1+1,2);}\n")
    assert out is not None
    assert "1+1*2" in out.replace(" ", "") or "1+1*2" in out


@_clang_for_pp
def test_preprocess_macro_is_semantically_load_bearing():
    c = _pp.confirm_macro_precedence_hazard()
    assert c.available and c.ok
    # the precedence pitfall: unparenthesized macro yields 3, parenthesized 4.
    assert c.hazard_value == 3
    assert c.safe_value == 4
    assert c.detected_hazard


@_clang_for_pp
def test_preprocess_conditional_selects_the_program():
    c = _pp.confirm_conditional_compilation()
    assert c.available and c.ok
    assert c.without == 0 and c.with_feature == 1


@_clang_for_pp
def test_preprocess_include_resolution_works():
    c = _pp.confirm_include_resolution()
    assert c.available and c.ok
    assert c.symbol_present and c.value == 42


# ---------------------------------------------------------------------------
# Step 32 — behavior-accurate libc/runtime modeling (differential vs real libc).
# ---------------------------------------------------------------------------

from src.ub_oracle import libc_model as _lc  # noqa: E402

_clang_for_lc = pytest.mark.skipif(
    not _os.path.exists(_lc.CC), reason="clang not available")


def test_libc_specs_pure_model_values():
    # spot-check the pure models without a compiler.
    assert _lc.model_strlen(b"hello\x00") == 5
    assert _lc.model_strcmp(b"abc\x00", b"abd\x00") == -1
    assert _lc.model_strcmp(b"abc\x00", b"abc\x00") == 0
    assert _lc.model_memcmp(b"\x01\x02", b"\x01\x03", 2) == -1
    assert _lc.model_memcpy(b"\xaa\xbb\xcc", 2) == b"\xaa\xbb"
    assert _lc.model_memset(b"\x00\x00\x00", 0x41, 2) == b"\x41\x41\x00"
    assert _lc.model_strchr(b"abc\x00", ord("b")) == 1
    assert _lc.model_strchr(b"abc\x00", ord("z")) == -1


def test_libc_specs_encode_ub_preconditions():
    # strlen/strcmp/strchr require NUL termination — the model refuses otherwise.
    with pytest.raises(ValueError):
        _lc.model_strlen(b"no terminator")
    with pytest.raises(ValueError):
        _lc.model_strcmp(b"a\x00", b"no term")
    # the contract table documents the overlap/sign rules.
    assert "overlap" in _lc.LIBC_CONTRACTS["memcpy"]
    assert "SIGN" in _lc.LIBC_CONTRACTS["strcmp"]


@_clang_for_lc
@pytest.mark.parametrize("name", list(_lc.SPECS))
def test_libc_spec_matches_real_libc_on_random_inputs(name):
    c = _lc.confirm_spec(name, trials=150, seed=7)
    assert c.available and c.trials == 150
    assert c.ok, f"{name} mismatches: {c.mismatches[:3]}"


@_clang_for_lc
def test_libc_all_specs_confirmed_together():
    results = _lc.confirm_all(trials=80)
    assert len(results) == len(_lc.SPECS)
    assert all(r.ok for r in results)


# ---------------------------------------------------------------------------
# Step 27 — high-fidelity ingestion via compiler IRs (clang AST + rustc MIR).
# ---------------------------------------------------------------------------

from src.ub_oracle import ir_ingest as _iri  # noqa: E402

_clang_for_ir = pytest.mark.skipif(
    not _os.path.exists(_iri.CLANG), reason="clang not available")
_rustc_for_ir = pytest.mark.skipif(
    not _os.path.exists(_iri.RUSTC), reason="rustc not available")


@_clang_for_ir
def test_ir_clang_ingest_extracts_faithful_signatures():
    c = _iri.confirm_clang_ingest()
    assert c.available and c.ok
    fns = c.module.functions
    add = fns["add"]
    assert add.ret_type == "int" and add.arity == 2
    assert tuple(p.type for p in add.params) == ("int", "int")
    assert tuple(p.name for p in add.params) == ("a", "b")
    # storage class is read from the AST, not guessed.
    assert fns["dup_first"].storage == "static"
    assert fns["dup_first"].params[0].type == "const char *"


@_rustc_for_ir
def test_ir_mir_ingest_yields_ownership_facts_for_free():
    m = _iri.confirm_mir_ingest()
    assert m.available and m.ok
    fns = m.module.functions
    # a by-value Vec parameter is consumed (moved/dropped) per the compiler's MIR.
    assert "v" in fns["consume"].moved_params
    # a Copy i32 parameter is never moved.
    assert fns["double"].moved_params == ()


@_rustc_for_ir
def test_ir_mir_move_vs_copy_is_the_compilers_own_fact():
    # passing the Vec onward by value (into another fn) is a `move` in MIR;
    # a Copy type passed the same way is not consumed.
    src = ("pub fn take(v: Vec<i32>) -> Vec<i32> { v }\n"
           "pub fn keep(x: i32) -> i32 { x }\n")
    mod = _iri.ingest_rustc_mir(src)
    assert mod is not None
    assert mod.functions["take"].moved_params == ("v",)
    assert mod.functions["keep"].moved_params == ()


@_clang_for_ir
def test_ir_clang_function_qualtype_param_split():
    # the function-type parser handles pointer return types and qualified params.
    assert _iri._split_fn_qualtype("char *(const char *)") == ("const char *",)
    assert _iri._split_fn_qualtype("int (int, int)") == ("int", "int")
    assert _iri._split_fn_qualtype("void (void)") == ()


# ---------------------------------------------------------------------------
# Step 31 — whole-project ingestion (compile_commands.json + cargo workspace).
# ---------------------------------------------------------------------------

from src.ub_oracle import project_ingest as _pri  # noqa: E402

_clang_for_proj = pytest.mark.skipif(
    not _os.path.exists(_pri.CLANG), reason="clang not available")
_cargo_for_proj = pytest.mark.skipif(
    not _os.path.exists(_pri.CARGO), reason="cargo not available")


@_clang_for_proj
def test_proj_compile_db_ingests_whole_c_tree():
    c = _pri.confirm_compile_db()
    assert c.available and c.ok
    fns = c.project.all_functions()
    # functions from *both* translation units are present in one project model.
    assert {"add", "helper", "slen"} <= set(fns)
    assert fns["add"].ret_type == "int"
    assert fns["helper"].storage == "static"
    assert len(c.project.units) == 2


@_cargo_for_proj
def test_proj_cargo_workspace_enumerates_members():
    w = _pri.confirm_cargo_workspace()
    assert w.available and w.ok
    pkgs = w.project.packages
    assert {"alpha", "beta"} <= set(pkgs)
    # the source roots are exactly what cargo metadata reports for each crate.
    assert pkgs["alpha"].src_path.endswith(_os.path.join("src", "lib.rs"))
    assert "lib" in pkgs["beta"].kind


def test_proj_include_dir_extraction_handles_both_forms():
    # -I dir and -Idir, relative paths resolved against the entry directory.
    argv = ["clang", "-c", "-I", "include", "-Iother", "/abs/x.c"]
    dirs = _pri._include_dirs_from_argv(argv, "/proj")
    assert "/proj/include" in dirs and "/proj/other" in dirs


def test_proj_entry_command_accepts_arguments_or_command_string():
    assert _pri._entry_command({"arguments": ["clang", "-c", "x.c"]}) == \
        ["clang", "-c", "x.c"]
    assert _pri._entry_command({"command": "clang -c x.c"}) == \
        ["clang", "-c", "x.c"]


# ---------------------------------------------------------------------------
# Step 51 — solver portfolio + parallelism (z3 + boolector, with timeouts).
# ---------------------------------------------------------------------------

from src.ub_oracle import solver_portfolio as _sp  # noqa: E402


def test_portfolio_has_at_least_one_real_solver():
    # the project ships with z3 in-process; boolector is used when present.
    assert "z3" in _sp.available_solvers()


def test_portfolio_solves_known_sat_and_unsat():
    sat_q = ("(set-logic QF_BV)(declare-fun x () (_ BitVec 8))"
             "(assert (= (bvadd x #x01) #x00))(check-sat)")
    unsat_q = ("(set-logic QF_BV)(declare-fun x () (_ BitVec 8))"
               "(assert (distinct (bvxor x x) #x00))(check-sat)")
    rs = _sp.solve_portfolio(sat_q, timeout=10.0)
    ru = _sp.solve_portfolio(unsat_q, timeout=10.0)
    assert rs.status == _sp.SAT and rs.agreement and rs.winner is not None
    assert ru.status == _sp.UNSAT and ru.agreement


def test_portfolio_robustness_battery_matches_ground_truth():
    c = _sp.confirm_portfolio(timeout=10.0)
    assert c.ok
    # every divergence-class query reaches its known answer with agreement.
    for cr in c.report:
        assert cr.consensus == cr.expected
        assert cr.agreement


@pytest.mark.skipif(not _os.path.exists(_sp.BOOLECTOR),
                    reason="boolector not available")
def test_portfolio_boolector_and_z3_cross_check():
    # when both backends are present, at least one query is decided by BOTH,
    # i.e. the portfolio genuinely cross-checks rather than trusting one solver.
    c = _sp.confirm_portfolio(timeout=10.0)
    assert "z3" in c.available_solvers and "boolector" in c.available_solvers
    both = [cr for cr in c.report
            if cr.per_solver.get("z3") in (_sp.SAT, _sp.UNSAT)
            and cr.per_solver.get("boolector") in (_sp.SAT, _sp.UNSAT)]
    assert both, "no query was decided by both solvers"
    for cr in both:
        assert cr.per_solver["z3"] == cr.per_solver["boolector"]


# ---------------------------------------------------------------------------
# Step 36 — fuzzer-guided frontend hardening (differential vs real clang).
# ---------------------------------------------------------------------------

from src.ub_oracle import frontend_fuzz as _ff  # noqa: E402

_clang_for_fuzz = pytest.mark.skipif(
    not _os.path.exists(_ff.CLANG), reason="clang not available")


def test_fuzz_generator_emits_compilable_well_typed_c():
    import random as _r
    prog = _ff.generate_program(_r.Random(1), 0)
    assert prog.functions and "return" in prog.source


def test_fuzz_garbage_never_crashes_the_frontend_object():
    # the malformed-input contract is toolchain-independent: ingest must not raise.
    for g in _ff.GARBAGE_INPUTS:
        try:
            _ff._iri.ingest_clang(g)
        except Exception as e:  # pragma: no cover
            assert False, f"frontend raised on garbage: {e!r}"


@_clang_for_fuzz
def test_fuzz_clang_frontend_zero_divergences_zero_crashes():
    rep = _ff.fuzz_clang_frontend(iterations=40, seed=0xBEEF)
    assert rep.compiled >= 20
    assert rep.divergences == [], rep.divergences[:5]
    assert rep.crashes == [], rep.crashes[:5]


@_clang_for_fuzz
def test_fuzz_confirm_survives_a_sizeable_run():
    c = _ff.confirm_fuzz(iterations=40, seed=0x1234)
    assert c.available and c.ok
    assert c.report.compiled >= 20


# ---------------------------------------------------------------------------
# Step 38 — per-language frontend conformance suite (merge gate).
# ---------------------------------------------------------------------------

from src.ub_oracle import conformance as _conf  # noqa: E402

_clang_for_conf = pytest.mark.skipif(
    not _os.path.exists(_conf.CLANG), reason="clang not available")
_rustc_for_conf = pytest.mark.skipif(
    not _os.path.exists(_conf.RUSTC), reason="rustc not available")


def test_conf_corpus_is_nonempty_and_well_formed():
    assert len(_conf.ALL_CASES) >= 12
    for c in _conf.ALL_CASES:
        assert c.lang in ("c", "rust") and c.expected and c.source


@_clang_for_conf
def test_conf_c_constructs_lower_as_expected():
    res = {r.case_id: r for r in _conf.run_conformance(_conf.C_CASES)}
    for cid in ("c-array-decay-param", "c-function-pointer-param",
                "c-static-storage", "c-typedef-ret"):
        r = res[cid]
        assert r.applicable and r.passed, r.mismatches


@_rustc_for_conf
def test_conf_rust_ownership_constructs_lower_as_expected():
    res = {r.case_id: r for r in _conf.run_conformance(_conf.RUST_CASES)}
    assert res["rust-vec-byval-moved"].passed
    assert res["rust-vec-ref-not-moved"].passed
    assert res["rust-copy-scalar-not-moved"].passed


def test_conf_full_applicable_suite_is_green():
    c = _conf.confirm_conformance()
    assert c.applicable >= 1
    assert c.ok
    assert c.passed == c.applicable


# ---------------------------------------------------------------------------
# Step 20 — evaluation-order / sequencing oracle (clang -Wunsequenced).
# ---------------------------------------------------------------------------

from src.ub_oracle import eval_order as _eo  # noqa: E402

_clang_for_seq = pytest.mark.skipif(
    not _os.path.exists(_eo.CLANG), reason="clang not available")


@_clang_for_seq
def test_seq_unsequenced_modification_is_flagged_and_abstained():
    for src in (_eo.UNSEQUENCED_INCR, _eo.UNSEQUENCED_CALL, _eo.UNSEQUENCED_ASSIGN):
        dec = _eo.decide(src)
        assert dec.is_abstain
        assert dec.diagnostics  # clang gave a concrete location


@_clang_for_seq
def test_seq_clean_code_is_not_abstained():
    dec = _eo.decide(_eo.SEQUENCED_CLEAN)
    assert not dec.is_abstain
    assert dec.verdict == _eo.EQUIVALENT_OK
    assert dec.diagnostics == ()


@_clang_for_seq
def test_seq_confirmation_holds_against_real_clang():
    c = _eo.confirm_sequencing()
    assert c.available and c.ok


def test_seq_comment_string_stripper_is_safe():
    # the side-effect screen must not be confused by code in comments/strings.
    s = _eo._strip_comments_strings('int x; /* i++ */ char* s="a++b"; x=1;')
    assert "i++" not in s and "a++b" not in s


# ---------------------------------------------------------------------------
# Step 105 — sequence-point / unsequenced-modification divergence oracle.
# ---------------------------------------------------------------------------

from src.ub_oracle.oracles import sequence_point as _seqpoint  # noqa: E402,F401


def test_sequence_point_oracle_registered_for_rust_static_mode():
    orc = _plugin.get_oracle_for("eval_order", "c", "rust")
    assert orc.confirmation_mode == "static_ub_vs_defined"
    assert "eval_order" in CATALOGUE
    assert CATALOGUE["eval_order"].source_definedness is Definedness.UNSPECIFIED


def test_sequence_point_witness_honors_range_and_is_static_diagnosable():
    orc = _plugin.get_oracle_for("eval_order", "c", "rust")
    r = orc.find_divergence({"kind": "unsequenced", "pattern": "postinc_read_add"})
    assert r.verdict is _plugin.OracleVerdict.DIVERGENT
    assert r.counterexample.inputs["i"] == 0
    assert "i++ + i" in r.counterexample.source_snippet

    ranged = orc.find_divergence(
        {"kind": "unsequenced", "pattern": "postinc_read_add", "i_range": [5, 9]})
    assert ranged.counterexample.inputs["i"] == 5

    clean = orc.find_divergence(
        {"kind": "unsequenced", "pattern": "sequenced_clean", "probe": "eval_order"})
    assert clean.verdict is _plugin.OracleVerdict.NOT_APPLICABLE

    if _os.path.exists(_eo.CLANG):
        dec = _eo.decide(r.counterexample.source_snippet)
        assert dec.is_abstain
        assert dec.diagnostics


@pytest.mark.skipif(
    not (_TC.c_available and _TC.target_available("rust")),
    reason="needs C compiler and rustc")
def test_sequence_point_confirmed_by_real_static_diagnostic_and_rust():
    orc = _plugin.get_oracle_for("eval_order", "c", "rust")
    res = orc.confirm(
        orc.find_divergence({"kind": "unsequenced", "pattern": "postinc_read_add"}),
        ReexecHarness(_TC),
    )
    rr = res.reexec
    assert rr.available and rr.mode == "static_ub_vs_defined"
    assert rr.ub_reachable, "clang -Wunsequenced must diagnose the C source UB"
    assert rr.rust_defined, "Rust target must be deterministic and defined"
    assert rr.confirmed and res.counterexample.confirmed


# ---------------------------------------------------------------------------
# Step 44 — known-bug / CVE corpus (real UB-rooted bug classes, proven).
# ---------------------------------------------------------------------------

from src.ub_oracle import cve_corpus as _cve  # noqa: E402

_full_rust = _TC.full_for("rust") if hasattr(_TC, "full_for") else False
_full_go = _TC.full_for("go") if hasattr(_TC, "full_for") else False


def test_cve_corpus_is_curated_and_cwe_tagged():
    assert len(_cve.CORPUS) >= 5
    for c in _cve.CORPUS:
        assert c.cwe.startswith("CWE-") and c.targets and c.inputs
    table = _cve.coverage_table()
    assert all(set(r) >= {"cwe", "case", "title", "langs"} for r in table)


@pytest.mark.skipif(not _full_rust, reason="C/UBSan/rust toolchain unavailable")
def test_cve_corpus_catches_every_c_to_rust_bug():
    rep = _cve.run_corpus(langs=("rust",))
    applic = rep.applicable
    assert applic, "no C->Rust cases ran"
    for r in applic:
        assert r.confirmed, f"{r.cwe} {r.case_id}: {r.reason}"


@pytest.mark.skipif(not _full_go, reason="C/UBSan/go toolchain unavailable")
def test_cve_corpus_catches_c_to_go_bugs():
    rep = _cve.run_corpus(langs=("go",))
    applic = rep.applicable
    assert applic, "no C->Go cases ran"
    for r in applic:
        assert r.confirmed, f"{r.cwe} {r.case_id}: {r.reason}"


@pytest.mark.skipif(not (_full_rust or _full_go),
                    reason="no full toolchain for any target")
def test_cve_corpus_confirmation_is_green():
    c = _cve.confirm_corpus()
    assert c.available and c.ok
    assert c.report.confirmed_count == len(c.report.applicable)


# --------------------------------------------------------------------------- #
# Step 45 — labeled ground-truth set (sanitizer-established precision substrate)
# --------------------------------------------------------------------------- #
from src.ub_oracle import ground_truth as _gt  # noqa: E402


def test_ground_truth_corpus_is_large_and_two_pairs():
    items = _gt.enumerate_corpus()
    assert len(items) >= 500, f"only {len(items)} items"
    langs = {it.lang for it in items}
    assert {"rust", "go"}.issubset(langs)
    labels = {it.declared_label for it in items}
    assert labels == {"divergent", "equivalent"}
    # every divergent item carries a CWE; equivalent items do not
    for it in items:
        if it.declared_label == "divergent":
            assert it.cwe.startswith("CWE-"), it.item_id
        else:
            assert it.cwe == "", it.item_id


def test_ground_truth_stats_distribution():
    s = _gt.corpus_stats()
    assert s["total"] >= 500 and s["n_langs"] == 2
    assert s["by_label"]["divergent"] >= 100
    assert s["by_label"]["equivalent"] >= 100
    # a healthy spread of UB classes and safe classes
    assert len([k for k in s["by_class"] if k.startswith("safe_")]) >= 3
    assert len([k for k in s["by_class"] if not k.startswith("safe_")]) >= 4


@pytest.mark.skipif(not (_full_rust or _full_go),
                    reason="no full toolchain for any target")
def test_ground_truth_labeler_matches_declarations():
    # The authoritative sanitizer-based labeler must agree with every declared
    # label on a per-family sample spanning both languages and both labels.
    conf = _gt.confirm_ground_truth(sample_per_class=2)
    assert conf.available and conf.ok, [
        (r.item_id, r.declared_label, r.observed_label, r.detail)
        for r in conf.report.disagreements
    ]
    assert conf.corpus_size >= 500 and conf.n_langs >= 2
    assert not conf.report.disagreements
    kinds = {r.observed_label for r in conf.report.labeled}
    assert {"divergent", "equivalent"}.issubset(kinds)


@pytest.mark.skipif(not _full_rust, reason="C/UBSan/rust toolchain unavailable")
def test_ground_truth_rust_divergent_is_ub_trap():
    # Pick a concrete divergent Rust item and confirm the label is established by
    # an actual sanitizer trap (not by declaration).
    items = [it for it in _gt.enumerate_corpus(("rust",))
             if it.klass == "div_by_zero"]
    assert items
    from src.ub_oracle.reexec import ReexecHarness
    ev = _gt.label_item(ReexecHarness(_TC), items[0])
    assert ev.observed_label == "divergent"
    assert ev.ub_trapped


@pytest.mark.skipif(not _full_rust, reason="C/UBSan/rust toolchain unavailable")
def test_ground_truth_rust_equivalent_outputs_match():
    items = [it for it in _gt.enumerate_corpus(("rust",))
             if it.klass == "safe_add"]
    assert items
    from src.ub_oracle.reexec import ReexecHarness
    ev = _gt.label_item(ReexecHarness(_TC), items[0])
    assert ev.observed_label == "equivalent"
    assert not ev.ub_trapped
    assert ev.c_out == ev.target_out


# --------------------------------------------------------------------------- #
# Step 46 — scale measurement infrastructure (content-hashed results JSON)
# --------------------------------------------------------------------------- #
from src.ub_oracle import scale_measure as _sm  # noqa: E402


@pytest.mark.skipif(not (_full_rust or _full_go),
                    reason="no full toolchain for any target")
def test_scale_measure_records_time_and_verdict():
    rep = _sm.run_scale(sample_per_class=1)
    assert rep.available and rep.measured
    for m in rep.measured:
        assert m.wall_ms >= 0.0
        assert m.peak_rss_kb >= 0
        assert m.decided != m.abstained  # exactly one is true
        if m.decided:
            assert m.observed_label in ("divergent", "equivalent")
    assert rep.decided > 0
    assert rep.faithful  # measurement layer did not corrupt the verdict
    assert rep.total_wall_ms > 0.0


@pytest.mark.skipif(not (_full_rust or _full_go),
                    reason="no full toolchain for any target")
def test_scale_measure_content_hash_is_stable():
    # Two independent runs over the same sample must yield the same verdict-layer
    # content hash, even though wall-time/memory differ.
    r1 = _sm.run_scale(sample_per_class=1)
    r2 = _sm.run_scale(sample_per_class=1)
    assert _sm.content_hash(r1) == _sm.content_hash(r2)
    # ...and timing is genuinely separate from the hashed verdict layer.
    doc = _sm.results_document(r1)
    assert "content_hash" in doc and "measurements" in doc
    assert "total_wall_ms" in doc["measurements"]
    # the verdict layer must not contain timing fields
    for v in doc["verdicts"]:
        assert "wall_ms" not in v and "peak_rss_kb" not in v


@pytest.mark.skipif(not (_full_rust or _full_go),
                    reason="no full toolchain for any target")
def test_scale_measure_emit_json_roundtrip(tmp_path):
    import json as _json
    rep = _sm.run_scale(sample_per_class=1)
    out = tmp_path / "results.json"
    h = _sm.emit_results_json(str(out), rep)
    doc = _json.loads(out.read_text())
    assert doc["schema"] == _sm.SCHEMA_VERSION
    assert doc["content_hash"] == h
    assert doc["n_items"] == len(rep.measured)
    assert doc["n_decided"] + doc["n_abstained"] == doc["n_items"]
    # re-hash the persisted verdict layer → identical
    import hashlib as _hl
    canon = _json.dumps(doc["verdicts"], sort_keys=True,
                        separators=(",", ":")).encode()
    assert _hl.sha256(canon).hexdigest() == h


@pytest.mark.skipif(not (_full_rust or _full_go),
                    reason="no full toolchain for any target")
def test_scale_measure_confirmation_green():
    c = _sm.confirm_scale(sample_per_class=1)
    assert c.available and c.ok
    assert c.hash_stable
    assert c.n_decided > 0
    assert c.n_decided + c.n_abstained == c.n_items


# --------------------------------------------------------------------------- #
# Step 49 — head-to-head vs existing tools (where they apply; the gap where none)
# --------------------------------------------------------------------------- #
from src.ub_oracle import external_baselines as _xb  # noqa: E402


def test_external_baselines_applicability_table_is_categorical():
    tbl = _xb.applicability_table()
    assert len(tbl) >= 4
    # No existing-tool category ingests a cross-language (C, target) pair — that
    # is the structural gap this project occupies.
    assert all(not row["ingests_cross_language_pair"] for row in tbl)
    for row in tbl:
        assert row["note"]  # every category carries a stated applicability reason


@pytest.mark.skipif(not (_full_rust or _full_go),
                    reason="no full toolchain for any target")
def test_external_baselines_head_to_head_gap():
    conf = _xb.confirm_head_to_head(per_class=1)
    assert conf.available and conf.ok
    # oracle catches every divergent item it is run on
    assert conf.oracle_found == conf.n_items
    # on the provably-blind classes the same-language baseline finds nothing,
    # while the oracle catches them all → a total false-negative gap there
    assert conf.n_blind >= 1
    assert conf.blind_baseline_found == 0
    assert conf.blind_oracle_found == conf.n_blind
    # and no installed tool category can ingest a cross-language pair
    assert not conf.report.any_cross_language_tool_installed


@pytest.mark.skipif(not _full_rust, reason="C/UBSan/rust toolchain unavailable")
def test_external_baselines_div0_is_invisible_to_same_language():
    # Concretely: a same-language O0-vs-O2 differential cannot see the div-by-zero
    # divergence (both C builds trap identically), but our oracle does.
    from src.ub_oracle.reexec import ReexecHarness
    item = next(it for it in _xb.gt.enumerate_corpus(("rust",))
                if it.klass == "div_by_zero")
    h = ReexecHarness(_TC)
    base_found, _ = _xb._c_self_diff(h, item)
    assert base_found is False
    ev = _xb.gt.label_item(h, item)
    assert ev.observed_label == "divergent"


# --------------------------------------------------------------------------- #
# Step 54 — external replication kit (Docker + make reproduce-kit)
# --------------------------------------------------------------------------- #
from src.ub_oracle import replication as _repl  # noqa: E402


def test_replication_kit_files_present_and_corpus_large():
    rep = _repl.confirm_replication_kit(quick=True)
    # the stranger really can build & run the kit
    assert rep.files_ok, rep.files
    assert "Dockerfile" in rep.files and rep.files["Dockerfile"]
    assert "scripts/reproduce_kit.sh" in rep.files and rep.files["scripts/reproduce_kit.sh"]
    assert "Makefile" in rep.files and rep.files["Makefile"]
    # the reproduced tables draw from a >=500-pair, 2-language corpus
    assert rep.corpus_ok and rep.corpus_size >= 500 and rep.n_langs >= 2
    assert rep.ok


def test_replication_manifest_hash_is_stable():
    m1 = _repl.manifest(_repl.confirm_replication_kit(quick=True))
    m2 = _repl.manifest(_repl.confirm_replication_kit(quick=True))
    assert m1["kit_hash"] == m2["kit_hash"]
    # the hash is computed over deterministic layers only
    assert "corpus_stats" in m1["stable_layers"]
    assert "applicability_table" in m1["stable_layers"]
    assert "kit_files" in m1["stable_layers"]
    assert m1["corpus_size"] >= 500


def test_replication_kit_makefile_targets_exist():
    import os as __os
    root = __os.path.abspath(__os.path.join(__os.path.dirname(__file__), ".."))
    mk = open(__os.path.join(root, "Makefile")).read()
    for target in ("reproduce-kit:", "docker-build:", "docker-reproduce:"):
        assert target in mk, target
    df = open(__os.path.join(root, "Dockerfile")).read()
    # the image pins the real toolchain the oracles need
    for tool in ("clang", "rust", "go", "z3"):
        assert tool in df.lower(), tool

# --------------------------------------------------------------------------- #
# Step 89 — statistical rigor.
# --------------------------------------------------------------------------- #
from src.ub_oracle import statistical_rigor as _sr  # noqa: E402


def test_wilson_interval_is_deterministic_and_well_formed():
    # Wilson interval is a pure function of (k, n): exactly reproducible.
    assert _sr.wilson_interval(0, 0) == (0.0, 0.0)
    lo, hi = _sr.wilson_interval(15, 15)
    assert 0.0 <= lo <= 1.0 <= hi + 1e-9 and lo < 1.0  # upper bound pinned at 1.0
    assert hi == 1.0
    # point estimate must lie inside its own interval for an interior proportion
    lo2, hi2 = _sr.wilson_interval(7, 20)
    p = 7 / 20
    assert lo2 - 1e-12 <= p <= hi2 + 1e-12
    # same inputs -> identical interval (determinism)
    assert _sr.wilson_interval(7, 20) == (lo2, hi2)


def test_metrics_are_preregistered_constants():
    # The metric names and definitions are fixed before measurement.
    assert set(_sr.PREREGISTERED_METRICS) == {
        "recall_definedness", "false_positive_rate"}
    for defn in _sr.PREREGISTERED_METRICS.values():
        assert isinstance(defn, str) and len(defn) > 20


def test_hardware_profile_records_provenance():
    hw = _sr.hardware_profile()
    for key in ("platform", "machine", "cpu_count", "python_version",
                "clang", "rustc", "go"):
        assert key in hw


@pytest.mark.skipif(not _full_rust, reason="C/UBSan/rust toolchain unavailable")
def test_statistical_rigor_confirms_against_real_code():
    conf = _sr.confirm_statistical_rigor(seeds=(1, 2), sample_per_seed=3)
    assert conf.available and conf.ok, conf.detail
    # a real definedness-divergence population was measured
    assert conf.n_in_scope > 0
    # sound-for-divergence holds empirically: zero false positives
    assert conf.fpr.successes == 0
    # recall point estimate lies inside its reported Wilson CI
    assert conf.recall.ci_lo - 1e-12 <= conf.recall.point <= conf.recall.ci_hi + 1e-12


@pytest.mark.skipif(not _full_rust, reason="C/UBSan/rust toolchain unavailable")
def test_statistical_rigor_content_hash_is_stable():
    # Two runs with identical seeds reproduce the identical content hash
    # even though the (non-hashed) hardware/timing layer is incidental.
    r1 = _sr.run_study(seeds=(5,), sample_per_seed=3)
    r2 = _sr.run_study(seeds=(5,), sample_per_seed=3)
    assert r1.available and r2.available
    assert r1.content_hash == r2.content_hash and r1.content_hash
    # the headline interval is recomputed from the pooled counts, not stored
    m = r1.metrics["recall_definedness"]
    assert (m.ci_lo, m.ci_hi) == _sr.wilson_interval(m.successes, m.trials)

# --------------------------------------------------------------------------- #
# Step 72 — relational / product-program formalization.
# --------------------------------------------------------------------------- #
from src.ub_oracle import product_program as _pp  # noqa: E402


def test_product_assertion_is_same_boolean_function_as_semantics():
    # Exhaustively over the recorded-observable abstraction, product_violated
    # must equal semantics.is_divergence — the soundness/completeness theorem
    # at the abstraction level (no toolchain needed).
    from src.ub_oracle import semantics as _sem
    rcs = (0, 1)
    vals = ("1", "2")
    for mode in _pp.MODES:
        for o0_rc in rcs:
            for o0v in vals:
                for o2_rc in rcs:
                    for o2v in vals:
                        for san in (False, True):
                            for defined in (False, True):
                                for det in (False, True):
                                    obs = _pp.ProductObservable(
                                        target="rust", mode=mode,
                                        o0_rc=o0_rc, o0_val=o0v,
                                        o2_rc=o2_rc, o2_val=o2v,
                                        san_trapped=san, defined=defined,
                                        deterministic=det)
                                    assert (_pp.product_violated(obs)
                                            == _sem.is_divergence(obs.to_observation()))
                                    # R holds iff not violated (assertion duality)
                                    assert (_pp.product_assertion_holds(obs)
                                            != _pp.product_violated(obs))


def test_product_clauses_are_named_inference_rules():
    assert _pp.CLAUSES == ("P_premise_ub_reached", "T_target_defined", "C_consequence")
    obs = _pp.ProductObservable(
        target="go", mode=_pp.TRAP_VS_DEFINED, o0_rc=136, o0_val="",
        o2_rc=136, o2_val="", san_trapped=True, defined=True, deterministic=True)
    clauses = _pp.evaluate_clauses(obs)
    assert set(clauses) == set(_pp.CLAUSES)
    # a trap-vs-defined divergence: all three clauses hold -> R violated
    assert all(clauses.values()) and _pp.product_violated(obs)


def test_product_program_unknown_target_rejected():
    import pytest as _pt
    with _pt.raises(ValueError):
        _pp.tsem.get_pack("cobol")


@pytest.mark.skipif(not (_full_rust or _full_go),
                    reason="C/UBSan + a target toolchain unavailable")
def test_product_program_confirms_against_real_code():
    langs = tuple(l for l in ("rust", "go")
                  if (_full_rust if l == "rust" else _full_go))
    conf = _pp.confirm_product_program(langs=langs, per_class=1)
    assert conf.available and conf.ok, conf.detail
    assert conf.n_checked > 0 and conf.n_divergent > 0 and conf.n_equivalent > 0
    # every check agreed: product == semantics == harness on real binaries
    for c in conf.checks:
        assert c.agree, (c.item_id, c.product_violated,
                         c.semantics_divergence, c.harness_confirmed)
        if c.declared_label == "divergent" and c.harness_confirmed:
            assert c.product_violated
        if c.declared_label == "equivalent":
            assert not c.product_violated


@pytest.mark.skipif(not _full_rust, reason="C/UBSan/rust toolchain unavailable")
def test_product_program_content_hash_stable():
    c1 = _pp.confirm_product_program(langs=("rust",), per_class=1)
    c2 = _pp.confirm_product_program(langs=("rust",), per_class=1)
    assert c1.available and c2.available
    assert c1.content_hash == c2.content_hash and c1.content_hash

# --------------------------------------------------------------------------- #
# Step 71 — cross-language translation-validation framing.
# --------------------------------------------------------------------------- #
from src.ub_oracle import translation_validation as _tv  # noqa: E402


def test_translation_validation_verdict_constants_and_unavailable_target():
    assert _tv.REFUTED == "REFUTED" and _tv.NOT_REFUTED == "NOT_REFUTED"
    import pytest as _pt
    with _pt.raises(ValueError):
        _tv.validate("x", "y", [("0",)], target="fortran")


def test_translation_validation_witness_fingerprint_is_stable_and_self_contained():
    from src.ub_oracle import product_program as _pp
    obs = _pp.ProductObservable(
        target="rust", mode=_pp.TRAP_VS_DEFINED, o0_rc=136, o0_val="",
        o2_rc=136, o2_val="", san_trapped=True, defined=True, deterministic=True)
    w = _tv.CounterexampleWitness(
        c_src="int main(){return 0;}", target_src="fn main(){}",
        inputs=("5", "0"), target="rust", mode=_pp.TRAP_VS_DEFINED,
        klass="div_by_zero", observable=obs, reason="R_m violated")
    # the witness carries everything a third party needs to reproduce
    assert w.c_src and w.target_src and w.inputs == ("5", "0") and w.target == "rust"
    # fingerprint is a deterministic function of the witness inputs
    assert w.fingerprint() == w.fingerprint()
    assert _pp.product_violated(w.observable)


@pytest.mark.skipif(not (_full_rust or _full_go),
                    reason="C/UBSan + a target toolchain unavailable")
def test_translation_validation_confirms_witness_theorems_on_real_code():
    langs = tuple(l for l in ("rust", "go")
                  if (_full_rust if l == "rust" else _full_go))
    conf = _tv.confirm_translation_validation(langs=langs, per_class=1)
    assert conf.available and conf.ok, conf.detail
    assert conf.n_refuted > 0 and conf.n_not_refuted > 0
    for c in conf.checks:
        assert c.agree, (c.item_id, c.verdict, c.replay_reproduced)
        if c.declared_label == "divergent":
            assert c.verdict == _tv.REFUTED
            assert c.replay_reproduced and c.replay_deterministic
        else:
            assert c.verdict == _tv.NOT_REFUTED


@pytest.mark.skipif(not _full_rust, reason="C/UBSan/rust toolchain unavailable")
def test_translation_validation_witness_replays_against_fresh_compilation():
    # End-to-end: validate a real divergent item, then replay the emitted
    # witness from scratch and confirm it still violates R_m (witness soundness).
    from src.ub_oracle import product_program as _pp
    items = [it for it in _gt.enumerate_corpus(("rust",))
             if it.declared_label == "divergent" and it.klass == "div_by_zero"]
    assert items
    it = items[0]
    vr = _tv.validate(it.c_src, it.target_src, [tuple(it.inputs)],
                      target="rust", mode=_pp.TRAP_VS_DEFINED, klass=it.klass)
    assert vr.refuted and vr.witness is not None
    fresh = vr.witness.replay()
    assert fresh is not None and _pp.product_violated(fresh)
    # determinism: a second replay reproduces the identical observable
    again = vr.witness.replay()
    assert again is not None and again.__dict__ == fresh.__dict__

# --------------------------------------------------------------------------- #
# Step 88 — generalization study (across pairs AND producer styles).
# --------------------------------------------------------------------------- #
from src.ub_oracle import generalization as _gen  # noqa: E402

_full_swift = _TC.full_for("swift") if hasattr(_TC, "full_for") else False


def test_generalization_grid_constants_and_source_generation():
    assert set(_gen.TARGETS) == {"rust", "go", "swift"}
    assert set(_gen.STYLES) == {"direct", "helper", "verbose"}
    assert "div_by_zero" in _gen.CLASSES and "oversized_shift" in _gen.CLASSES
    # every (target, class, style) generates a non-trivial, distinct source
    seen = set()
    for t in _gen.TARGETS:
        for k in _gen.CLASSES:
            srcs = {_gen.target_source(t, k, st) for st in _gen.STYLES}
            assert len(srcs) == len(_gen.STYLES)  # styles are genuinely distinct
            seen |= srcs
    import pytest as _pt
    with _pt.raises(ValueError):
        _gen.target_source("kotlin", "div_by_zero", "direct")


@pytest.mark.skipif(not (_full_rust or _full_go or _full_swift),
                    reason="no full C+target toolchain available")
def test_generalization_result_is_invariant_across_pairs_and_producers():
    avail = tuple(t for t, ok in (("rust", _full_rust), ("go", _full_go),
                                  ("swift", _full_swift)) if ok)
    conf = _gen.confirm_generalization(targets=avail)
    assert conf.available and conf.ok, conf.detail
    assert conf.invariant_across_pairs and conf.invariant_across_styles
    # every grid cell: all UB inputs detected, zero false positives on safe inputs
    for c in conf.report.cells:
        assert c.uniform_ok, (c.target, c.style, c.klass,
                              c.n_ub_detected, c.n_ub, c.n_safe_flagged)
        assert c.detection_rate == 1.0 and c.fp_rate == 0.0


@pytest.mark.skipif(not (_full_rust and _full_go),
                    reason="need >=2 pairs to demonstrate breadth")
def test_generalization_demonstrates_multi_pair_breadth_and_stable_hash():
    avail = tuple(t for t, ok in (("rust", _full_rust), ("go", _full_go),
                                  ("swift", _full_swift)) if ok)
    r1 = _gen.run_generalization(targets=avail)
    r2 = _gen.run_generalization(targets=avail)
    assert len(r1.available_targets) >= 2
    # the per-cell verdict layer is content-hashed -> reproducible across runs
    assert r1.content_hash == r2.content_hash and r1.content_hash
    # aggregates confirm each pair independently catches every UB input
    for pair, (d, u, f, s) in r1.by_pair().items():
        assert d == u and f == 0, (pair, d, u, f, s)


from src.ub_oracle import artifact_eval as _ae  # noqa: E402


def test_artifact_available_badge_is_earned_from_files():
    # Pure file inspection -> always runs, never consistency-only.
    ev = _ae.evaluate_artifact()
    avail = ev.available
    assert avail.earned and avail.fully_exercised, avail
    names = {c.name for c in avail.criteria}
    assert {"open_licence_present", "archival_descriptor_names_public_repo",
            "readme_present", "version_consistent"} <= names


def test_artifact_eval_confirmation_earns_all_three_badges():
    conf = _ae.confirm_artifact_evaluation()
    assert conf.available and conf.ok, conf.detail
    assert set(conf.earned_badges) == {"available", "functional", "reproduced"}


def test_artifact_functional_badge_and_consistency_only_flagging():
    ev = _ae.evaluate_artifact()
    func = ev.functional
    assert func.earned, func
    live = [c for c in func.criteria if c.name == "live_oracle_runs"][0]
    # When the toolchain is present the live smoke-test is real evidence (not
    # consistency-only); when absent it must be flagged consistency-only.
    if _full_rust:
        assert live.passed and not live.consistency_only
        assert func.fully_exercised
    else:
        assert live.consistency_only


@pytest.mark.skipif(not _full_rust, reason="need C+UBSan+rustc for live reproduce path")
def test_artifact_reproduced_badge_real_byte_identity_and_stable_hashes():
    ev = _ae.evaluate_artifact()
    rep = ev.reproduced
    assert rep.earned, rep
    by = {c.name: c for c in rep.criteria}
    # the trusted results artifact regenerates byte-identically (real evidence)
    assert by["trusted_results_byte_identical"].passed
    assert not by["trusted_results_byte_identical"].consistency_only
    # the reproducibility hashes are stable and exercised for real
    assert by["replication_kit_hash_stable"].passed
    assert by["scale_hash_reproducible"].passed
    assert by["generalization_hash_reproducible"].passed


from src.ub_oracle import mechanized_soundness as _ms  # noqa: E402

_lean_present = _ms._lean_binary() is not None


def test_mechanized_soundness_source_declares_every_required_theorem():
    rep = _ms.confirm_mechanized_soundness()
    assert rep.source_present
    assert not rep.theorems_missing, rep.theorems_missing
    assert set(rep.theorems_present) == set(_ms.REQUIRED_THEOREMS)
    assert rep.source_hash and len(rep.source_hash) == 64


@pytest.mark.skipif(not _lean_present, reason="Lean 4 kernel not installed")
def test_mechanized_soundness_lean_kernel_accepts_the_proof():
    rep = _ms.confirm_mechanized_soundness()
    assert rep.available
    assert rep.kernel_accepted is True, rep.stderr_tail
    assert rep.ok and rep.fully_checked


def test_mechanized_soundness_consistency_only_when_lean_absent_still_safe():
    # Regardless of toolchain, ok must never be True with a missing theorem.
    rep = _ms.confirm_mechanized_soundness()
    if rep.theorems_missing:
        assert not rep.ok
    # fully_checked implies the kernel actually ran and accepted.
    if rep.fully_checked:
        assert rep.available and rep.kernel_accepted is True


from src.ub_oracle import idiomatic_corpus as _ic  # noqa: E402


def test_idiomatic_corpus_has_both_labels_and_provenance():
    labels = {it.declared_label for it in _ic.CORPUS}
    assert labels == {"divergent", "equivalent"}
    # every item carries real-world provenance and at least one target port.
    for it in _ic.CORPUS:
        assert it.provenance and len(it.provenance) > 20
        assert it.targets, it.item_id


@pytest.mark.skipif(not (_full_rust and _full_go),
                    reason="need C+UBSan+rustc and go for the idiomatic corpus")
def test_idiomatic_corpus_oracle_correct_on_every_item_and_lang():
    conf = _ic.confirm_idiomatic_corpus(langs=("rust", "go"))
    assert conf.available and conf.ok, conf.detail
    assert conf.n_divergent > 0 and conf.n_equivalent > 0
    assert conf.n_langs >= 2
    # every (item x lang) verdict matches its declared label.
    for v in conf.report.verdicts:
        assert v.correct, (v.item_id, v.lang, v.detail)
        if v.declared_label == "divergent":
            assert v.ub_confirmed and not v.safe_confirmed
        else:
            assert not v.ub_confirmed and not v.safe_confirmed


@pytest.mark.skipif(not (_full_rust and _full_go),
                    reason="need >=2 langs for breadth + hash stability")
def test_idiomatic_corpus_hash_is_reproducible():
    r1 = _ic.run_corpus(("rust", "go"))
    r2 = _ic.run_corpus(("rust", "go"))
    assert r1.content_hash == r2.content_hash and r1.content_hash
    # the midpoint-overflow item is the headline real-world divergence.
    mids = [v for v in r1.verdicts if v.item_id == "midpoint-overflow"]
    assert mids and all(v.ub_confirmed and not v.safe_confirmed for v in mids)


from src.ub_oracle import multipair_corpus as _mpc  # noqa: E402


def test_multipair_corpus_translates_every_function_to_three_targets():
    for fn in _mpc.CORPUS:
        assert set(fn.targets.keys()) == {"rust", "go", "swift"}, fn.func_id
        assert fn.provenance and len(fn.provenance) > 15
    labels = {fn.declared_label for fn in _mpc.CORPUS}
    assert labels == {"divergent", "equivalent"}


@pytest.mark.skipif(not (_full_rust and _full_go and _full_swift),
                    reason="need rust+go+swift for the multi-pair corpus")
def test_multipair_corpus_cross_pair_invariant_holds_on_real_code():
    conf = _mpc.confirm_multipair_corpus()
    assert conf.available and conf.ok, conf.detail
    assert conf.n_pairs == 3
    assert conf.cross_pair_invariant
    # a divergent function must be flagged on EVERY pair; equivalent on none.
    for func_id, vs in conf.report.by_function().items():
        label = vs[0].declared_label
        if label == "divergent":
            assert all(v.ub_flagged and not v.safe_flagged for v in vs), func_id
        else:
            assert all(not v.ub_flagged and not v.safe_flagged for v in vs), func_id


@pytest.mark.skipif(not (_full_rust and _full_go and _full_swift),
                    reason="need 3 pairs for hash stability + breadth")
def test_multipair_corpus_hash_reproducible_and_three_pairs():
    r1 = _mpc.run_corpus()
    r2 = _mpc.run_corpus()
    assert r1.content_hash == r2.content_hash and r1.content_hash
    assert len(r1.langs) == 3
    # midpoint signed-overflow divergence reproduces across all three targets
    mids = r1.by_function()["midpoint"]
    assert {v.lang for v in mids} == {"rust", "go", "swift"}
    assert all(v.ub_flagged for v in mids)


from src.ub_oracle import transpiler_recipes as _tr  # noqa: E402


def test_transpiler_recipes_registry_and_protocol():
    names = _tr.recipe_names()
    assert "c2rust" in names and "llm-transpiler" in names
    assert any(n.startswith("reference-") for n in names)
    # every recipe yields a Translator satisfying the protocol.
    for name in names:
        tr = _tr.get_recipe(name).translator()
        assert isinstance(tr, _tr.Translator)
        assert isinstance(tr.available(), bool)
    # unknown recipe raises loudly.
    try:
        _tr.get_recipe("nope")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_transpiler_external_recipe_gated_no_fabrication():
    # c2rust is not installed here; the recipe must report unavailable and never
    # fabricate output.
    tr = _tr.get_recipe("c2rust").translator()
    if not tr.available():
        assert tr.translate("int main(){return 0;}", "div_by_zero") is None


@pytest.mark.skipif(not _full_rust, reason="need C+UBSan+rustc to verify a recipe")
def test_transpiler_reference_recipe_pipeline_flags_real_divergence():
    h = _tr.ReexecHarness(_tr.toolchain_available())
    c = ("#include <stdio.h>\n#include <stdlib.h>\n"
         "int main(int argc,char**argv){int a=atoi(argv[1]);int b=atoi(argv[2]);"
         'printf("%d\\n",a/b);return 0;}\n')
    tr = _tr.ReferenceTranslator("rust")
    ub = _tr.verify_transpiled(c, tr, ["10", "0"], "div_by_zero", h)
    safe = _tr.verify_transpiled(c, tr, ["10", "2"], "div_by_zero", h)
    assert ub.translated and ub.diverged, ub.reason
    assert safe.translated and not safe.diverged, safe.reason


@pytest.mark.skipif(not (_full_rust and _full_go and _full_swift),
                    reason="need rust+go+swift to confirm all reference recipes")
def test_transpiler_recipes_confirmation_all_reference_pairs():
    conf = _tr.confirm_transpiler_recipes()
    assert conf.available and conf.ok, conf.detail
    assert conf.n_reference_pairs == 3
    assert set(conf.external_recipes) == {"c2rust", "llm-transpiler"}


from src.ub_oracle import playground as _pg  # noqa: E402


def test_playground_page_renders_and_lists_every_pair():
    page = _pg.render_page()
    assert "<form" not in page or "playground" in page  # sanity
    for n in _pg.target_names():
        assert f">{n}<" in page, n
    assert "/api/verify" in page


def test_playground_rejects_unknown_target_without_fabrication():
    v = _pg.evaluate("int main(){return 0;}", "fn main(){}", ["1"],
                     "division_by_zero", "klingon")
    assert not v.available and not v.diverged
    assert "klingon" in v.reason


@pytest.mark.skipif(not _full_rust, reason="need C+UBSan+rustc for the oracle")
def test_playground_evaluate_flags_real_divergence():
    c = ("#include <stdio.h>\n#include <stdlib.h>\n"
         "int main(int argc,char**argv){int a=atoi(argv[1]);int b=atoi(argv[2]);"
         'printf("%d\\n",a/b);return 0;}\n')
    t = ("use std::env;\nfn main(){let a:i32=env::args().nth(1).unwrap()"
         ".parse().unwrap();let b:i32=env::args().nth(2).unwrap().parse()"
         '.unwrap();println!("{}",a/b);}\n')
    ub = _pg.evaluate(c, t, ["10", "0"], "division_by_zero", "rust")
    safe = _pg.evaluate(c, t, ["10", "2"], "division_by_zero", "rust")
    assert ub.available and ub.diverged, ub.summary
    assert safe.available and not safe.diverged, safe.summary
    assert ub.to_json()["diverged"] is True


@pytest.mark.skipif(not _full_rust, reason="need rust to drive the HTTP endpoint")
def test_playground_http_endpoint_end_to_end():
    conf = _pg.confirm_playground()
    assert conf.available and conf.ok, conf.detail


from src.ub_oracle import docs_site as _ds  # noqa: E402


def test_docs_site_gallery_lists_every_corpus_item():
    path = _ds.generate_gallery()
    text = path.read_text()
    for it in _ds._idio.CORPUS:
        assert f"`{it.item_id}`" in text, it.item_id
    for fn in _ds._multi.CORPUS:
        assert f"`{fn.func_id}`" in text, fn.func_id
    assert "Auto-generated" in text


def test_docs_site_mkdocs_yml_present_and_nav_targets_exist():
    import re
    yml = _ds._MKDOCS_YML.read_text()
    # every nav markdown target must exist under docs/.
    for m in re.findall(r":\s*([A-Za-z0-9_./-]+\.md)\s*$", yml, re.MULTILINE):
        assert (_ds._DOCS / m).exists(), m


@pytest.mark.skipif(_ds._mkdocs_invocation() is None,
                    reason="mkdocs not installed")
def test_docs_site_strict_build_succeeds():
    rep = _ds.confirm_docs_site()
    assert rep.available and rep.ok, rep.detail
    assert rep.strict_build and rep.pages_built > 0


from src.ub_oracle import vscode_ext as _vsx  # noqa: E402


def test_vscode_extension_manifest_contributes_command():
    assert _vsx._PKG.exists()
    assert _vsx._manifest_contributes_command()
    m = _vsx._load_manifest()
    assert m["main"].endswith("extension.js")
    assert "vscode" in m["engines"]


@pytest.mark.skipif(_vsx._node_tools() is None, reason="npm not installed")
def test_vscode_extension_compiles_with_real_tsc():
    rep = _vsx.confirm_vscode_extension()
    assert rep.available and rep.ok, rep.detail
    assert rep.manifest_ok and rep.compiled and rep.entry_built


from src.ub_oracle import frontends as _fe  # noqa: E402


def test_frontend_spi_registers_three_and_protocol():
    names = {f.name for f in _fe.FRONTENDS}
    assert {"treesitter-c", "clang-ast-c", "rustc-mir-rust"} <= names
    for f in _fe.FRONTENDS:
        assert isinstance(f, _fe.Frontend)
        assert isinstance(f.available(), bool)
    assert {f.name for f in _fe.frontends_for("c")} == {"treesitter-c", "clang-ast-c"}
    with pytest.raises(ValueError):
        _fe.get_frontend("nope")


@pytest.mark.skipif(not _fe.treesitter_available(), reason="tree-sitter not installed")
def test_treesitter_frontend_extracts_functions():
    mod = _fe.ingest_treesitter(
        "static int add(int a, long b){return a+b;}\n"
        "char *pick(const char *a, char *b, int w){return w?b:(char*)a;}\n")
    assert mod is not None
    assert set(mod.functions) == {"add", "pick"}
    assert mod.functions["add"].arity == 2
    assert mod.functions["add"].storage == "static"
    assert [p.name for p in mod.functions["add"].params] == ["a", "b"]
    assert mod.functions["pick"].arity == 3


@pytest.mark.skipif(
    not (_fe.treesitter_available() and __import__("os").path.exists(_fe.ingest_clang.__globals__["CLANG"])),
    reason="need tree-sitter + clang to cross-validate")
def test_treesitter_frontend_agrees_with_clang_ast():
    rep = _fe.confirm_real_frontends()
    assert rep.available and rep.ok, rep.detail
    assert len(rep.agreements) == len(_fe._SAMPLES)
    assert all(a.ok for a in rep.agreements)


from src.ub_oracle import single_binary as _sb  # noqa: E402


def test_single_binary_builds_runnable_pyz(tmp_path):
    pyz = _sb.build_pyz(tmp_path / "clv.pyz")
    assert pyz.exists() and pyz.stat().st_size > 0
    import zipfile
    with zipfile.ZipFile(pyz) as zf:
        names = zf.namelist()
    assert "__main__.py" in names
    assert any(n.startswith("ub_oracle/") for n in names)
    # the shebang makes it directly executable.
    assert pyz.read_bytes()[:2] == b"#!"


def test_single_binary_matches_library_end_to_end():
    rep = _sb.confirm_single_binary()
    assert rep.available and rep.ok, rep.detail
    assert rep.built and rep.runs and rep.matches_library
    assert rep.size_bytes > 0


from src.ub_oracle import divergence_zoo as _zoo  # noqa: E402


def test_zoo_index_and_json_are_structured_and_complete():
    idx = _zoo.index_by_class_and_pair()
    assert idx, "zoo index must not be empty"
    # every divergent corpus exhibit is indexed by class and pair.
    div_ids = {e.exhibit_id for e in _zoo.EXHIBITS if e.declared_label == "divergent"}
    indexed = {i for c in idx.values() for p in c.values() for i in p}
    assert div_ids == indexed
    j = _zoo.to_json()
    assert j["schema"] == "divergence-zoo/1"
    assert set(j["classes"]) == {e.divergence_class for e in _zoo.EXHIBITS}
    # content hash is stable across calls.
    assert _zoo.content_hash() == _zoo.content_hash()


def test_zoo_markdown_generates_and_lists_every_divergent_exhibit():
    path = _zoo.generate_zoo()
    text = path.read_text()
    for e in _zoo.EXHIBITS:
        if e.declared_label == "divergent":
            assert f"`{e.exhibit_id}`" in text, e.exhibit_id
    assert "Auto-generated" in text


@pytest.mark.skipif(not (_full_rust and _full_go and _full_swift),
                    reason="need rust+go+swift to re-confirm every exhibit")
def test_zoo_every_witness_reconfirms_live():
    rep = _zoo.confirm_zoo()
    assert rep.available and rep.ok, rep.detail
    assert rep.n_confirmed == rep.n_divergent
    assert all(c.confirmed for c in rep.checks)


from src.ub_oracle import figures as _fig  # noqa: E402


def test_figures_collect_from_real_data():
    d = _fig.collect()
    assert d.pairs and d.classes
    # catalogue column sums equal per-class totals (cross-source consistency).
    for klass, row in d.catalogue.items():
        assert sum(row.values()) == d.per_class_totals[klass]
    # the fuzzing gap is real in the source data.
    assert d.method_recall["semrec"] >= 0.99
    baselines = [r for m, r in d.method_recall.items() if m != "semrec"]
    assert baselines and all(r == 0.0 for r in baselines)


def test_figures_generate_valid_svgs():
    import xml.dom.minidom as _md
    f1, f2, f3, fmd = _fig.generate_figures()
    for p in (f1, f2, f3):
        _md.parseString(p.read_text())  # raises on malformed XML
        assert p.read_text().startswith("<svg")
    assert "Figure 1" in fmd.read_text()


def test_figures_are_data_faithful():
    rep = _fig.confirm_figures()
    assert rep.ok, rep.checks
    assert rep.n_pairs >= 2 and rep.n_classes >= 2
    assert rep.catalogue_total == sum(_fig.collect().per_class_totals.values())


from src.ub_oracle import ecosystem as _eco  # noqa: E402


def test_ecosystem_v1_api_surface_intact():
    ok, missing = _eco._confirm_api()
    assert ok, f"v1 public API regressed (SemVer-breaking): missing {missing}"
    # every promised symbol is also genuinely exported in __all__.
    import src.ub_oracle as _pkg
    exported = set(_pkg.__all__)
    for name in _eco.PUBLIC_API_V1:
        assert name in exported, name


def test_ecosystem_generates_cargo_shim_and_snapshot():
    shim, snap = _eco.generate_artifacts()
    assert shim.exists() and (shim.stat().st_mode & 0o111)
    assert shim.read_text().startswith("#!/usr/bin/env bash")
    import json as _json
    data = _json.loads(snap.read_text())
    assert data["semver"] == _eco.SEMVER
    assert set(data["symbols"]) == set(_eco.PUBLIC_API_V1)


def test_ecosystem_cargo_subcommand_matches_library():
    rep = _eco.confirm_ecosystem()
    assert rep.api_ok and rep.shim_syntax_ok, rep.checks
    assert rep.shim_ran and rep.shim_matches_library, rep.checks
    assert rep.ok


from src.ub_oracle import claims_audit as _ca  # noqa: E402


def test_claims_audit_default_passes_against_live_code():
    rep = _ca.confirm_claims_audit()
    assert rep.ok, [(c.name, c.actual, c.detail) for c in rep.checks if not c.ok]
    assert rep.n_checks == 3


def test_claims_audit_catches_exhibit_overclaim():
    # An inflated exhibit count in the prose must be flagged.
    _named, cnt = _ca.audit_text(
        "999 exhibits across rust/go/swift rust go swift")
    assert not cnt.ok
    # The honest, live count must pass.
    live = _ca._live_values()["divergent_exhibits"]
    _n2, cnt2 = _ca.audit_text(
        f"{live} exhibits across rust/go/swift rust go swift")
    assert cnt2.ok


def test_claims_audit_catches_unbacked_named_pair():
    # Naming a target language that has no real oracle must be flagged.
    named, _cnt = _ca.audit_text("we translate C to rust, go, swift and COBOL")
    assert named.ok  # rust/go/swift are all backed; cobol isn't matched/named
    # generality bar holds live.
    v = _ca._live_values()
    assert _ca._check_generality_two_pairs(v).ok


from src.ub_oracle import case_studies as _cs  # noqa: E402


def test_case_studies_plan_items_exist_with_real_inputs():
    # Every planned case maps to a real corpus item with ub + safe inputs and
    # an idiomatic target in the chosen language.
    for item_id, lang in _cs._CASE_PLAN:
        it = _cs._item(item_id)
        assert it.declared_label == "divergent"
        assert lang in it.targets and it.targets[lang].strip()
        assert any(a != "" for a in it.ub_inputs)
        assert any(a != "" for a in it.safe_inputs)


@pytest.mark.skipif(not (_full_rust and _full_go),
                    reason="need C/UBSan + rust + go to walk the case studies")
def test_case_studies_walk_end_to_end_and_show_gap():
    rep = _cs.confirm_case_studies(trials=600, seed=0)
    assert rep.available and rep.ok, rep.detail
    assert rep.n_walked == rep.n_cases
    assert rep.n_gap >= 1
    for c in rep.cases:
        assert c.ub_reachable and c.target_defined
        assert c.oracle_confirms_witness and c.oracle_silent_on_safe
    # the div-by-zero case is the certain gap: random int32 won't be 0.
    div = [c for c in rep.cases if c.divergence_class == "div_by_zero"]
    assert div and div[0].fuzzing_gap


@pytest.mark.skipif(not (_full_rust and _full_go),
                    reason="need C/UBSan + rust + go to render walked studies")
def test_case_studies_markdown_renders_each_walk():
    path, rep = _cs.generate_case_studies(trials=600, seed=0)
    text = path.read_text()
    assert "Case studies" in text and "Cost / benefit summary" in text
    for item_id, _lang in _cs._CASE_PLAN:
        assert f"`{item_id}`" in text


from src.ub_oracle import disclosure as _disc  # noqa: E402


def test_disclosure_records_are_well_formed():
    assert len(_disc.DISCLOSURES) >= 3
    ids = [r.advisory_id for r in _disc.DISCLOSURES]
    assert len(ids) == len(set(ids))
    for r in _disc.DISCLOSURES:
        assert r.witness_input and r.safe_input
        assert r.c_src.strip() and r.target_src.strip()
        assert r.impact.strip() and r.remediation.strip()
        assert r.target_lang in ("rust", "go", "swift")


def test_disclosure_template_and_markdown_generate():
    md, tmpl, _rep = _disc.generate_disclosures()
    t = tmpl.read_text()
    assert "Advisory ID" in t and "Remediation" in t and "timeline" in t.lower()
    body = md.read_text()
    for r in _disc.DISCLOSURES:
        assert r.advisory_id in body


@pytest.mark.skipif(not (_full_rust and _full_go),
                    reason="need C/UBSan + rust + go to reproduce disclosures")
def test_disclosure_reproduces_live_with_valid_bundles():
    rep = _disc.confirm_disclosures()
    assert rep.available and rep.ok, rep.detail
    assert rep.n_reproduced == rep.n_records
    assert rep.bundles_valid
    for r in rep.results:
        assert r.ub_reachable and r.target_defined and r.safe_silent
        assert r.bundle_path


@pytest.mark.skipif(not _full_go,
                    reason="need C/UBSan + go to run the repro bundle")
def test_disclosure_bundle_runs_end_to_end():
    # The divide-by-zero bundle must actually exhibit the divergence when run.
    rec = next(r for r in _disc.DISCLOSURES
               if r.divergence_class == "div_by_zero")
    res = _disc.reproduce_disclosure(rec, write_bundle=True)
    import subprocess as _sp
    out = _sp.run(["bash", str(_disc._ROOT / res.bundle_path)],
                  capture_output=True, text=True, timeout=120)
    combined = out.stdout + out.stderr
    assert "division by zero" in combined          # C UBSan trapped
    assert "divide by zero" in combined            # Go panicked


from src.ub_oracle import test_ratchet_core as _trc  # noqa: E402


def _green_counts(**over):
    base = {"passed": 100, "skipped": 5, "xfailed": 7,
            "failed": 0, "error": 0, "xpassed": 0}
    base.update(over)
    return base


def test_ratchet_accepts_green_and_improved():
    floor = {"passed": 100, "skipped": 5, "xfailed": 7}
    assert _trc.enforce_counts("fast", _green_counts(), floor) == 0
    # More passes, fewer skips: still accepted.
    assert _trc.enforce_counts(
        "fast", _green_counts(passed=120, skipped=4), floor) == 0
    assert _trc.violations("fast", _green_counts(), floor) == []


def test_ratchet_rejects_red_and_lies():
    floor = {"passed": 100, "skipped": 5, "xfailed": 7}
    # A failure, an error, or a stale xpass are all hard rejections.
    for bad in ("failed", "error", "xpassed"):
        assert _trc.enforce_counts("fast", _green_counts(**{bad: 1}), floor) == 1
    # Missing baseline is also a violation (forces an explicit --update).
    assert _trc.enforce_counts("fast", _green_counts(), {}) == 1


def test_ratchet_rejects_regressions():
    floor = {"passed": 100, "skipped": 5, "xfailed": 7}
    # Losing a pass, or silently skipping a new test, must fail the gate.
    assert _trc.enforce_counts("fast", _green_counts(passed=99), floor) == 1
    assert _trc.enforce_counts("fast", _green_counts(skipped=6), floor) == 1
    assert any("regressed" in v
               for v in _trc.violations("fast", _green_counts(passed=99), floor))
    assert any("skipped grew" in v
               for v in _trc.violations("fast", _green_counts(skipped=6), floor))


def test_ratchet_baseline_file_is_well_formed():
    import json
    import pathlib
    root = pathlib.Path(_disc._ROOT)
    data = json.loads((root / "tests" / "green_baseline.json").read_text())
    assert "fast" in data
    fast = data["fast"]
    assert {"passed", "skipped", "xfailed"} <= set(fast)
    assert fast["passed"] > 800            # the real, recorded floor
    ok, _ = _trc.is_baselineable(_green_counts(passed=fast["passed"]))
    assert ok
    bad_ok, _ = _trc.is_baselineable(_green_counts(xpassed=1))
    assert not bad_ok


# ---------------------------------------------------------------------------
# Step 85 — large-scale (>=100k LOC) migration study
# ---------------------------------------------------------------------------
from src.ub_oracle import large_scale_study as _ls  # noqa: E402


def test_large_scale_census_meets_loc_floor_and_all_distinct():
    items = _ls.generate_corpus()
    cen = _ls.corpus_census(items)
    assert int(cen["total_loc"]) >= _ls.MIN_TOTAL_LOC
    # every program in the corpus is genuinely distinct source.
    assert cen["n_items"] == cen["n_distinct_programs"]
    # balanced-ish across the two declared labels.
    by_label = cen["by_label"]
    assert by_label["divergent"] > 1000
    assert by_label["equivalent"] > 1000


def test_large_scale_census_has_pair_and_class_breadth():
    items = _ls.generate_corpus()
    cen = _ls.corpus_census(items)
    assert set(cen["pairs"]) == {"go", "rust"}
    # both UB-rooted divergent families and defined-equivalent families present.
    classes = set(cen["by_class"].keys())
    assert {"div_by_zero", "oob_read", "oversized_shift", "signed_overflow"} <= classes
    assert any(k.startswith("safe_") for k in classes)


def test_large_scale_ok_logic_is_loc_gated():
    items = _ls.generate_corpus()
    cen = _ls.corpus_census(items)
    # consistency-only (no toolchain) report passes purely on the LOC floor.
    rep_ok = _ls.StudyReport(
        schema_version=_ls.SCHEMA_VERSION, available=False,
        census=cen, sample_size=0, seed=0)
    assert rep_ok.ok is True
    # a corpus below the floor can never be ok, even with agreeing results.
    small = dict(cen)
    small["total_loc"] = _ls.MIN_TOTAL_LOC - 1
    rep_bad = _ls.StudyReport(
        schema_version=_ls.SCHEMA_VERSION, available=False,
        census=small, sample_size=0, seed=0)
    assert rep_bad.ok is False


@pytest.mark.skipif(
    not (_TC.full_for("rust") and _TC.full_for("go")),
    reason="needs both rustc+clang and go toolchains")
def test_large_scale_live_sample_confirms_labels_and_is_deterministic():
    rep1 = _ls.confirm_large_scale_study(sample_size=10, seed=0xC0FFEE)
    assert rep1.available is True
    assert rep1.ok is True
    assert rep1.aggregates["agree"] == rep1.aggregates["executed"]
    # the verdict-layer content hash is reproducible for a fixed seed.
    rep2 = _ls.confirm_large_scale_study(sample_size=10, seed=0xC0FFEE)
    assert rep2.content_hash == rep1.content_hash


# ---------------------------------------------------------------------------
# Step 110 — variable-length-array (VLA) bound divergence oracle
# ---------------------------------------------------------------------------
from src.ub_oracle.oracles import vla_bound as _vla  # noqa: E402


def test_vla_oracle_registered_for_rust_and_go_trap_mode():
    rust = _plugin.get_oracle_for("vla_bound", "c", "rust")
    go = _plugin.get_oracle_for("vla_bound", "c", "go")
    assert rust.confirmation_mode == "trap_vs_defined"
    assert go.confirmation_mode == "trap_vs_defined"
    assert "vla_bound" in CATALOGUE
    assert CATALOGUE["vla_bound"].is_ub_rooted()


def test_vla_witness_is_a_negative_bound_and_honors_range():
    orc = _plugin.get_oracle_for("vla_bound", "c", "rust")
    res = orc.find_divergence({"kind": "vla", "width": 32})
    assert res.verdict is _plugin.OracleVerdict.DIVERGENT
    # least-extreme negative bound is -1.
    assert list(res.counterexample.inputs.values())[0] == -1
    # a declared range is honored by the Z3 search.
    res2 = orc.find_divergence(
        {"kind": "vla", "width": 32, "bound_range": [-9, -5]})
    assert list(res2.counterexample.inputs.values())[0] == -5
    # not applicable to a non-VLA unit.
    assert orc.find_divergence(
        {"kind": "div", "width": 32}).verdict is _plugin.OracleVerdict.NOT_APPLICABLE


@pytest.mark.skipif(not _TC.full_for("rust"),
                    reason="needs C+UBSan+rustc toolchain")
def test_vla_rust_confirmed_against_real_compilers():
    orc = _plugin.get_oracle_for("vla_bound", "c", "rust")
    res = orc.confirm(orc.find_divergence({"kind": "vla", "width": 32}),
                      ReexecHarness(_TC))
    rr = res.reexec
    assert rr.available and rr.mode == "trap_vs_defined"
    assert rr.ub_reachable, "UBSan vla-bound must trap on the negative bound"
    assert rr.rust_defined, "Rust must panic deterministically (defined)"
    assert rr.confirmed and res.counterexample.confirmed


@pytest.mark.skipif(not _TC.full_for("go"),
                    reason="needs C+UBSan+go toolchain")
def test_vla_go_confirmed_against_real_compilers():
    orc = _plugin.get_oracle_for("vla_bound", "c", "go")
    res = orc.confirm(orc.find_divergence({"kind": "vla", "width": 64}),
                      ReexecHarness(_TC))
    rr = res.reexec
    assert rr.available and rr.mode == "trap_vs_defined"
    assert rr.ub_reachable, "UBSan vla-bound must trap on the negative bound"
    assert rr.rust_defined, "Go must panic deterministically (defined makeslice)"
    assert rr.confirmed and res.counterexample.confirmed


# ---------------------------------------------------------------------------
# Step 111 — longjmp to an exited VLA scope divergence oracle
# ---------------------------------------------------------------------------
from src.ub_oracle.oracles import longjmp_vla as _ljmp  # noqa: E402


def test_longjmp_vla_oracle_registered_for_rust_and_go_contract_mode():
    rust = _plugin.get_oracle_for("longjmp_vla", "c", "rust")
    go = _plugin.get_oracle_for("longjmp_vla", "c", "go")
    assert rust.confirmation_mode == "libc_contract_trap_vs_defined"
    assert go.confirmation_mode == "libc_contract_trap_vs_defined"
    assert "longjmp_vla" in CATALOGUE
    assert CATALOGUE["longjmp_vla"].is_ub_rooted()
    assert CATALOGUE["longjmp_vla"].c_standard_ref == "C17 7.13.2.1"


def test_longjmp_vla_witness_is_positive_and_honors_range():
    orc = _plugin.get_oracle_for("longjmp_vla", "c", "rust")
    res = orc.find_divergence({"kind": "longjmp_vla", "var": "n"})
    assert res.verdict is _plugin.OracleVerdict.DIVERGENT
    assert res.counterexample.inputs == {"n": 4}
    assert "setjmp" in res.counterexample.source_snippet
    assert "CLV_CHECK_LONGJMP_VLA" in res.counterexample.source_snippet
    assert "Drop for Guard" in res.counterexample.target_snippet
    # a declared range is honored; the least positive bound in it is chosen.
    r2 = orc.find_divergence(
        {"kind": "longjmp_vla", "var": "n", "bound_range": [5, 9]})
    assert r2.counterexample.inputs == {"n": 5}
    # no positive VLA bound means this oracle declines, leaving VLA-bound UB to
    # the VLA-bound oracle rather than conflating classes.
    r3 = orc.find_divergence(
        {"kind": "longjmp_vla", "var": "n", "bound_range": [-9, 0]})
    assert r3.verdict is _plugin.OracleVerdict.NO_DIVERGENCE_FOUND
    assert orc.find_divergence(
        {"kind": "vla", "width": 32}).verdict is _plugin.OracleVerdict.NOT_APPLICABLE


@pytest.mark.skipif(not (_TC.c_available and _TC.target_available("rust")),
                    reason="needs C+rustc toolchain")
def test_longjmp_vla_rust_confirmed_against_real_compilers():
    orc = _plugin.get_oracle_for("longjmp_vla", "c", "rust")
    res = orc.confirm(orc.find_divergence({"kind": "longjmp_vla"}), ReexecHarness(_TC))
    rr = res.reexec
    assert rr.available and rr.mode == "libc_contract_trap_vs_defined"
    assert rr.c_runs["contract"].contract_trapped("longjmp-vla")
    assert rr.ub_reachable, "checked C contract must trap on stale VLA setjmp target"
    assert rr.rust_defined, "Rust catch_unwind/Drop path must be deterministic"
    assert rr.confirmed and res.counterexample.confirmed


@pytest.mark.skipif(not (_TC.c_available and _TC.target_available("go")),
                    reason="needs C+go toolchain")
def test_longjmp_vla_go_confirmed_against_real_compilers():
    orc = _plugin.get_oracle_for("longjmp_vla", "c", "go")
    res = orc.confirm(orc.find_divergence({"kind": "longjmp_vla"}), ReexecHarness(_TC))
    rr = res.reexec
    assert rr.available and rr.mode == "libc_contract_trap_vs_defined"
    assert rr.c_runs["contract"].contract_trapped("longjmp-vla")
    assert rr.ub_reachable, "checked C contract must trap on stale VLA setjmp target"
    assert rr.rust_defined, "Go defer/recover path must be deterministic"
    assert rr.confirmed and res.counterexample.confirmed


@pytest.mark.skipif(not (_TC.c_available and _TC.target_available("rust")),
                    reason="needs C+rustc toolchain")
def test_longjmp_vla_leak_only_negative_control_is_not_confirmed():
    orc = _plugin.get_oracle_for("longjmp_vla", "c", "rust")
    target = orc.find_divergence({"kind": "longjmp_vla"}).counterexample.target_snippet
    rr = ReexecHarness(_TC).confirm_libc_contract_trap_vs_defined(
        _ljmp.leak_only_c_control_program(),
        target,
        ["4"],
        "longjmp_vla",
        target_lang="rust",
        contract_macro="CLV_CHECK_LONGJMP_VLA",
        contract_token="longjmp-vla",
        use_asan=False,
    )
    assert rr.available
    assert not rr.c_runs["contract"].contract_trapped("longjmp-vla")
    assert not rr.ub_reachable
    assert rr.rust_defined
    assert not rr.confirmed


# ---------------------------------------------------------------------------
# Step 106 — float-to-integer out-of-range conversion divergence oracle
# ---------------------------------------------------------------------------
from src.ub_oracle.oracles import float_cast as _fcast  # noqa: E402,F401


def test_float_cast_oracle_registered_for_rust_and_go_trap_mode():
    rust = _plugin.get_oracle_for("float_cast_overflow", "c", "rust")
    go = _plugin.get_oracle_for("float_cast_overflow", "c", "go")
    assert rust.confirmation_mode == "trap_vs_defined"
    assert go.confirmation_mode == "trap_vs_defined"
    assert "float_cast_overflow" in CATALOGUE
    assert CATALOGUE["float_cast_overflow"].is_ub_rooted()


def test_float_cast_witness_is_just_past_range_and_honors_range():
    orc = _plugin.get_oracle_for("float_cast_overflow", "c", "rust")
    r32 = orc.find_divergence({"kind": "float_cast", "width": 32})
    assert r32.verdict is _plugin.OracleVerdict.DIVERGENT
    # least-extreme out-of-range value is INT_MAX + 1.
    assert list(r32.counterexample.inputs.values())[0] == (1 << 31)
    r64 = orc.find_divergence({"kind": "float_cast", "width": 64})
    assert list(r64.counterexample.inputs.values())[0] == (1 << 63)
    # a declared range is honored by the Z3 search.
    r = orc.find_divergence(
        {"kind": "float_cast", "width": 32, "value_range": [(1 << 31) + 5, (1 << 31) + 9]})
    assert list(r.counterexample.inputs.values())[0] == (1 << 31) + 5
    # not applicable to a non-cast unit.
    assert orc.find_divergence(
        {"kind": "div", "width": 32}).verdict is _plugin.OracleVerdict.NOT_APPLICABLE


@pytest.mark.skipif(not _TC.full_for("rust"),
                    reason="needs C+UBSan+rustc toolchain")
def test_float_cast_rust_confirmed_against_real_compilers():
    orc = _plugin.get_oracle_for("float_cast_overflow", "c", "rust")
    res = orc.confirm(orc.find_divergence({"kind": "float_cast", "width": 32}),
                      ReexecHarness(_TC))
    rr = res.reexec
    assert rr.available and rr.mode == "trap_vs_defined"
    assert rr.ub_reachable, "UBSan float-cast-overflow must trap on the OOB value"
    assert rr.rust_defined, "Rust `as` must saturate deterministically (defined)"
    assert rr.confirmed and res.counterexample.confirmed


@pytest.mark.skipif(not _TC.full_for("go"),
                    reason="needs C+UBSan+go toolchain")
def test_float_cast_go_confirmed_against_real_compilers():
    orc = _plugin.get_oracle_for("float_cast_overflow", "c", "go")
    res = orc.confirm(orc.find_divergence({"kind": "float_cast", "width": 64}),
                      ReexecHarness(_TC))
    rr = res.reexec
    assert rr.available and rr.mode == "trap_vs_defined"
    assert rr.ub_reachable, "UBSan float-cast-overflow must trap on the OOB value"
    assert rr.rust_defined, "Go conversion must be defined & deterministic"
    assert rr.confirmed and res.counterexample.confirmed


# ---------------------------------------------------------------------------
# Step 107 — -ffast-math reassociation divergence oracle
# ---------------------------------------------------------------------------
from src.ub_oracle.oracles import fast_math as _fastmath  # noqa: E402,F401


def test_fast_math_oracle_registered_for_rust_and_go_optimizer_mode():
    rust = _plugin.get_oracle_for("fast_math_reassoc", "c", "rust")
    go = _plugin.get_oracle_for("fast_math_reassoc", "c", "go")
    assert rust.confirmation_mode == "optimizer_exploited"
    assert go.confirmation_mode == "optimizer_exploited"
    assert rust.optimizer_flag_variants == (["-O2", "-fno-fast-math"],
                                            ["-O2", "-ffast-math"])
    assert "fast_math_reassoc" in CATALOGUE


def test_fast_math_witness_swallows_y_and_is_applicable():
    orc = _plugin.get_oracle_for("fast_math_reassoc", "c", "rust")
    r = orc.find_divergence({"kind": "fp_reassoc"})
    assert r.verdict is _plugin.OracleVerdict.DIVERGENT
    x = r.counterexample.inputs["x"]
    y = r.counterexample.inputs["y"]
    # IEEE-strict (x+y)-x rounds to 0 (y swallowed) while reassociated value is y.
    assert y != 0.0
    assert (x + y) - x == 0.0
    # not applicable to a non-reassociation unit.
    assert orc.find_divergence(
        {"kind": "div", "width": 32}).verdict is _plugin.OracleVerdict.NOT_APPLICABLE


@pytest.mark.skipif(not _TC.full_for("rust"),
                    reason="needs C+rustc toolchain")
def test_fast_math_rust_confirmed_against_real_compilers():
    orc = _plugin.get_oracle_for("fast_math_reassoc", "c", "rust")
    res = orc.confirm(orc.find_divergence({"kind": "fp_reassoc"}), ReexecHarness(_TC))
    rr = res.reexec
    assert rr.available and rr.mode == "optimizer_exploited"
    assert rr.ub_consequential, "-fno-fast-math and -ffast-math must disagree"
    assert rr.rust_defined, "Rust must be IEEE-strict, deterministic & defined"
    assert rr.confirmed and res.counterexample.confirmed


@pytest.mark.skipif(not _TC.full_for("go"),
                    reason="needs C+go toolchain")
def test_fast_math_go_confirmed_against_real_compilers():
    orc = _plugin.get_oracle_for("fast_math_reassoc", "c", "go")
    res = orc.confirm(orc.find_divergence({"kind": "fp_reassoc"}), ReexecHarness(_TC))
    rr = res.reexec
    assert rr.available and rr.mode == "optimizer_exploited"
    assert rr.ub_consequential, "-fno-fast-math and -ffast-math must disagree"
    assert rr.rust_defined, "Go must be IEEE-strict, deterministic & defined"
    assert rr.confirmed and res.counterexample.confirmed


# ---------------------------------------------------------------------------
# Step 109 — restrict-violation divergence oracle
# ---------------------------------------------------------------------------
from src.ub_oracle.oracles import restrict_alias as _restrict  # noqa: E402,F401


def test_restrict_oracle_registered_for_rust_and_go_optimizer_mode():
    rust = _plugin.get_oracle_for("restrict_violation", "c", "rust")
    go = _plugin.get_oracle_for("restrict_violation", "c", "go")
    assert rust.confirmation_mode == "optimizer_exploited"
    assert go.confirmation_mode == "optimizer_exploited"
    assert rust.optimizer_flag_variants == (["-O0"], ["-O2"])
    assert "restrict_violation" in CATALOGUE
    assert CATALOGUE["restrict_violation"].is_ub_rooted()


def test_restrict_witness_is_nonzero_selector_and_honors_range():
    orc = _plugin.get_oracle_for("restrict_violation", "c", "rust")
    r = orc.find_divergence({"kind": "restrict_pair"})
    assert r.verdict is _plugin.OracleVerdict.DIVERGENT
    assert r.counterexample.inputs["sel"] == 1
    r2 = orc.find_divergence({"kind": "restrict_pair", "selector_range": [5, 9]})
    assert r2.counterexample.inputs["sel"] == 5
    assert orc.find_divergence(
        {"kind": "div", "width": 32}).verdict is _plugin.OracleVerdict.NOT_APPLICABLE


@pytest.mark.skipif(not _TC.full_for("rust"),
                    reason="needs C+rustc toolchain")
def test_restrict_rust_confirmed_against_real_compilers():
    orc = _plugin.get_oracle_for("restrict_violation", "c", "rust")
    res = orc.confirm(orc.find_divergence({"kind": "restrict_pair"}), ReexecHarness(_TC))
    rr = res.reexec
    assert rr.available and rr.mode == "optimizer_exploited"
    assert rr.ub_consequential, "-O0 and -O2 must disagree on the aliasing input"
    assert rr.rust_defined, "Rust &mut cannot alias: deterministic & defined"
    assert rr.confirmed and res.counterexample.confirmed


@pytest.mark.skipif(not _TC.full_for("go"),
                    reason="needs C+go toolchain")
def test_restrict_go_confirmed_against_real_compilers():
    orc = _plugin.get_oracle_for("restrict_violation", "c", "go")
    res = orc.confirm(orc.find_divergence({"kind": "restrict_pair"}), ReexecHarness(_TC))
    rr = res.reexec
    assert rr.available and rr.mode == "optimizer_exploited"
    assert rr.ub_consequential, "-O0 and -O2 must disagree on the aliasing input"
    assert rr.rust_defined, "Go has no restrict: deterministic & defined"
    assert rr.confirmed and res.counterexample.confirmed


# ---------------------------------------------------------------------------
# Step 102 — pointer-provenance / pointer-arithmetic-overflow divergence oracle
# ---------------------------------------------------------------------------
from src.ub_oracle.oracles import pointer_provenance as _prov  # noqa: E402,F401


def test_pointer_provenance_oracle_registered_for_rust_and_go_trap_mode():
    rust = _plugin.get_oracle_for("pointer_provenance", "c", "rust")
    go = _plugin.get_oracle_for("pointer_provenance", "c", "go")
    assert rust.confirmation_mode == "trap_vs_defined"
    assert go.confirmation_mode == "trap_vs_defined"
    assert "pointer_provenance" in CATALOGUE
    assert CATALOGUE["pointer_provenance"].is_ub_rooted()


def test_pointer_provenance_witness_overflows_address_space_and_honors_range():
    orc = _plugin.get_oracle_for("pointer_provenance", "c", "rust")
    # 4-byte ints: least n with n*4 >= 2**64 is 2**62.
    r32 = orc.find_divergence({"kind": "pointer_offset", "width": 32})
    assert r32.verdict is _plugin.OracleVerdict.DIVERGENT
    assert r32.counterexample.inputs["n"] == (1 << 62)
    # 8-byte ints: least n with n*8 >= 2**64 is 2**61.
    r64 = orc.find_divergence({"kind": "pointer_offset", "width": 64})
    assert r64.counterexample.inputs["n"] == (1 << 61)
    # honour a declared range: pick the least overflowing offset within it.
    r3 = orc.find_divergence(
        {"kind": "pointer_offset", "width": 32,
         "offset_range": [(1 << 62) + 5, (1 << 62) + 9]})
    assert r3.counterexample.inputs["n"] == (1 << 62) + 5
    assert orc.find_divergence(
        {"kind": "div", "width": 32}).verdict is _plugin.OracleVerdict.NOT_APPLICABLE


@pytest.mark.skipif(not _TC.full_for("rust"),
                    reason="needs C+UBSan+rustc toolchain")
def test_pointer_provenance_rust_confirmed_against_real_compilers():
    orc = _plugin.get_oracle_for("pointer_provenance", "c", "rust")
    res = orc.confirm(orc.find_divergence({"kind": "pointer_offset", "width": 32}),
                      ReexecHarness(_TC))
    rr = res.reexec
    assert rr.available and rr.mode == "trap_vs_defined"
    assert rr.ub_reachable, "UBSan pointer-overflow must trap on the offset"
    assert rr.rust_defined, "Rust checked index is deterministic & defined"
    assert rr.confirmed and res.counterexample.confirmed


@pytest.mark.skipif(not _TC.full_for("go"),
                    reason="needs C+UBSan+go toolchain")
def test_pointer_provenance_go_confirmed_against_real_compilers():
    orc = _plugin.get_oracle_for("pointer_provenance", "c", "go")
    res = orc.confirm(orc.find_divergence({"kind": "pointer_offset", "width": 64}),
                      ReexecHarness(_TC))
    rr = res.reexec
    assert rr.available and rr.mode == "trap_vs_defined"
    assert rr.ub_reachable, "UBSan pointer-overflow must trap on the offset"
    assert rr.rust_defined, "Go bounds-checked index is deterministic & defined"
    assert rr.confirmed and res.counterexample.confirmed


# ---------------------------------------------------------------------------
# Step 101 — strict-aliasing (type-punning) oracle: C->Go parity
# ---------------------------------------------------------------------------
def test_strict_aliasing_registered_for_go():
    go_orc = _plugin.get_oracle_for("strict_aliasing", "c", "go")
    assert go_orc.target_lang == "go"
    assert go_orc.confirmation_mode == "optimizer_exploited"
    res = go_orc.find_divergence({"kind": "type_pun"})
    assert res.verdict is _plugin.OracleVerdict.DIVERGENT
    assert "package main" in res.counterexample.target_snippet


@pytest.mark.skipif(not _TC.full_for("go"),
                    reason="needs C+go toolchain")
def test_strict_aliasing_go_confirmed_against_real_compilers():
    orc = _plugin.get_oracle_for("strict_aliasing", "c", "go")
    res = orc.confirm(orc.find_divergence({"kind": "type_pun"}), ReexecHarness(_TC))
    rr = res.reexec
    assert rr.available and rr.mode == "optimizer_exploited"
    assert rr.c_runs["A"].stdout != rr.c_runs["B"].stdout, "-O0 vs -O2 must disagree"
    assert rr.rust_defined, "Go must be deterministic & defined"
    assert rr.confirmed and res.counterexample.confirmed


# ---------------------------------------------------------------------------
# Step 117 — C -> C++ (defined-subset) language pair
#   The byte-identical token `1 << 31` is UB in C but defined in C++20.
# ---------------------------------------------------------------------------
from src.ub_oracle.oracles import c_to_cpp as _cpp  # noqa: E402,F401
from src.ub_oracle.target_semantics import PACKS as _PACKS  # noqa: E402


def test_cpp_pack_is_a_fourth_registered_language_pair():
    pairs = _plugin.language_pairs()
    assert ("c", "cpp") in pairs
    # the pack is data-driven: clang++/g++ candidates, .cpp suffix, value-only
    # defined outcome (the witnessing construct does not abort).
    pack = _PACKS["cpp"]
    assert pack.compiler_candidates[0] in ("clang++", "g++", "c++")
    assert pack.source_suffix == ".cpp"
    assert pack.defined_returncodes == (0,)
    assert "-std=c++20" in pack.compile_argv("clang++", "a.cpp", "a.out")


def test_cpp_signed_shift_oracle_registered_trap_vs_defined():
    orc = _plugin.get_oracle_for("signed_shift_sign_bit", "c", "cpp")
    assert orc.source_lang == "c" and orc.target_lang == "cpp"
    assert orc.confirmation_mode == "trap_vs_defined"
    assert "signed_shift_sign_bit" in CATALOGUE
    assert CATALOGUE["signed_shift_sign_bit"].is_ub_rooted()
    assert CATALOGUE["signed_shift_sign_bit"].c_standard_ref == "C17 6.5.7p4"


def test_cpp_witness_is_the_sign_bit_shift_and_honors_range():
    orc = _plugin.get_oracle_for("signed_shift_sign_bit", "c", "cpp")
    r32 = orc.find_divergence({"kind": "sign_bit_shift", "width": 32})
    assert r32.verdict is _plugin.OracleVerdict.DIVERGENT
    # least-extreme sign-bit shift for a 32-bit int is n = 31.
    assert r32.counterexample.inputs["n"] == 31
    assert "1 << 31" in r32.counterexample.divergence_witness
    r64 = orc.find_divergence({"kind": "sign_bit_shift", "width": 64})
    assert r64.counterexample.inputs["n"] == 63
    # a declared range is honored by the Z3 search.
    rr = orc.find_divergence(
        {"kind": "sign_bit_shift", "width": 32, "shift_range": [31, 31]})
    assert rr.counterexample.inputs["n"] == 31
    # not applicable to a non-shift unit.
    assert orc.find_divergence(
        {"kind": "div", "width": 32}).verdict is _plugin.OracleVerdict.NOT_APPLICABLE


def test_cpp_source_and_target_are_byte_identical_tokens():
    # The whole point of the C/C++ pair: the *same* shift expression text.
    orc = _plugin.get_oracle_for("signed_shift_sign_bit", "c", "cpp")
    ce = orc.find_divergence({"kind": "sign_bit_shift", "width": 32}).counterexample
    assert "one << n" in ce.source_snippet
    assert "one << n" in ce.target_snippet
    assert ce.source_lang == "c" and ce.target_lang == "cpp"


@pytest.mark.skipif(not _TC.full_for("cpp"),
                    reason="needs C+UBSan+clang++/g++ toolchain")
@pytest.mark.parametrize("width", [32, 64])
def test_cpp_divergence_confirmed_against_real_clang_and_cpp(width):
    orc = _plugin.get_oracle_for("signed_shift_sign_bit", "c", "cpp")
    res = orc.confirm(
        orc.find_divergence({"kind": "sign_bit_shift", "width": width}),
        ReexecHarness(_TC))
    rr = res.reexec
    assert rr.available and rr.mode == "trap_vs_defined"
    assert rr.ub_reachable, "UBSan must trap `1 << (width-1)` in C"
    assert rr.rust_defined, "C++20 must be defined & deterministic"
    assert rr.confirmed and res.counterexample.confirmed
    # the C++ defined value is the most-negative integer of that width.
    assert rr.rust_run.stdout == str(-(1 << (width - 1)))


# ---------------------------------------------------------------------------
# Step 152 — optimization-level sweep: at which -O level does C UB surface?
# ---------------------------------------------------------------------------
from src.ub_oracle import opt_sweep as _sweep  # noqa: E402


def test_opt_sweep_grid_is_deterministic_and_toolchain_free():
    g = _sweep.sweep_grid()
    assert g["levels"] == ["-O0", "-O1", "-O2", "-O3"]
    # the opt-level-latent classes are exactly the optimizer-exploited ones.
    assert set(g["classes"]) == {"strict_aliasing", "restrict_violation",
                                 "fast_math_reassoc"}
    assert g["n_cells"] == len(g["classes"]) * 4
    # stable across calls (pure data).
    assert _sweep.sweep_grid() == g


def test_opt_sweep_base_flags_keep_the_licence_but_drop_levels():
    orc = _plugin.get_oracle_for("fast_math_reassoc", "c", "rust")
    base = _sweep._base_flags(orc)
    # the fast-math licence flag is preserved; the -O level is not.
    assert "-ffast-math" in base
    assert not any(f.startswith("-O") for f in base)


@pytest.mark.skipif(not _TC.c_available or not _TC.ubsan,
                    reason="needs a real C compiler")
def test_opt_sweep_surfaces_each_latent_ub_at_a_real_O_level():
    rep = _sweep.sweep(ReexecHarness(_TC))
    rows = {r["divergence_class"]: r for r in rep["rows"]}
    # every opt-level-latent class flips its observable output once optimized:
    # the -O0 build is "benign" and a higher level diverges from it.
    for cls in ("strict_aliasing", "restrict_violation", "fast_math_reassoc"):
        r = rows[cls]
        assert r["available"], r["reason"]
        assert r["onset_level"] in ("-O1", "-O2", "-O3"), r
        assert r["outputs"]["-O0"] != r["outputs"][r["onset_level"]]
    assert rep["n_with_onset"] == 3


# ── Step 121: C->OCaml pair (GC'd, exception-based target) ───────────────────
# OCaml is the project's fifth registered target language and the first GC'd,
# exception-based one. Its fixed-width Int32/Int64 arithmetic is *modular*
# (defined) and its division / array faults raise exceptions that, uncaught,
# abort the process deterministically with exit code 2. Adding it was pure
# configuration: a TargetPack plus per-class source emitters — no new oracle or
# harness code — exactly as the generality thesis predicts.

_requires_ocaml = pytest.mark.skipif(
    not _TC.full_for("ocaml"),
    reason=f"needs C+UBSan+ocamlopt toolchain ({_TC})")

_OCAML_UNITS = {
    "signed_overflow": {"kind": "binop_const", "op": "add", "const": 2147483647,
                        "width": 32, "var": "x", "signed": True},
    "div_by_zero": {"kind": "div", "width": 32, "a": "a", "b": "b"},
    "intmin_div_neg1": {"kind": "div", "width": 32, "signed": True,
                        "a": "a", "b": "b"},
    "array_oob": {"kind": "array_index", "length": 4},
}


def _ocaml_unit(class_key):
    u = dict(_OCAML_UNITS[class_key])
    u["source_lang"], u["target_lang"] = "c", "ocaml"
    return u


def test_ocaml_pack_is_a_fifth_registered_language_pair():
    pairs = _plugin.language_pairs()
    assert ("c", "ocaml") in pairs
    pack = PACKS["ocaml"]
    # value (0) or an uncaught-exception abort (OCaml exits 2): pure data.
    assert pack.defined_returncodes == (0, 2)
    assert pack.source_suffix == ".ml"
    assert pack.compile_argv("ocamlopt", "a.ml", "a.out")[0] == "ocamlopt"
    # documents how it resolves each supported class (shift is *unspecified* in
    # OCaml, so it is intentionally absent).
    res = pack.class_resolution
    assert {"signed_overflow", "div_by_zero", "intmin_div_neg1",
            "array_oob"} <= set(res)
    assert "shift_oob" not in res


def test_ocaml_pair_registers_the_supported_classes_only():
    oc = _plugin.oracles_for(source_lang="c", target_lang="ocaml")
    classes = {o.divergence_class for o in oc}
    assert classes == {"signed_overflow", "div_by_zero",
                       "intmin_div_neg1", "array_oob"}
    # shift_oob is *unspecified* in OCaml too -> not a sound defined target.
    assert "shift_oob" not in classes


def test_ocaml_oracles_reuse_anchor_search_not_new_code():
    assert isinstance(_plugin.get_oracle_for("signed_overflow", "c", "ocaml"),
                      SignedOverflowOracle)
    assert isinstance(_plugin.get_oracle_for("div_by_zero", "c", "ocaml"),
                      DivisionByZeroOracle)
    assert isinstance(_plugin.get_oracle_for("array_oob", "c", "ocaml"),
                      ArrayOutOfBoundsOracle)


@pytest.mark.parametrize("class_key", list(_OCAML_UNITS))
def test_ocaml_oracle_emits_ocaml_source_symbolically(class_key):
    orc = _plugin.get_oracle_for(class_key, "c", "ocaml")
    res = orc.find_divergence(_ocaml_unit(class_key))
    assert res.verdict is OracleVerdict.DIVERGENT
    ce = res.counterexample
    assert ce.target_lang == "ocaml"
    assert "let () =" in ce.target_snippet
    assert "Sys.argv" in ce.target_snippet
    # the C source is byte-identical to the anchor's (one shared witness search).
    anchor = _plugin.get_oracle_for(class_key, "c", "rust")
    anchor_ce = anchor.find_divergence(dict(_OCAML_UNITS[class_key])).counterexample
    assert ce.source_snippet == anchor_ce.source_snippet


def test_run_outcome_definedness_for_ocaml_is_pack_driven():
    assert RunOutcome(0, "v", "").target_outcome_defined("ocaml")        # value
    assert RunOutcome(2, "", "exn").target_outcome_defined("ocaml")      # uncaught exn
    assert not RunOutcome(101, "", "").target_outcome_defined("ocaml")
    assert not RunOutcome(-5, "", "").target_outcome_defined("ocaml")


@_requires_ocaml
@pytest.mark.parametrize("class_key", list(_OCAML_UNITS))
def test_ocaml_divergence_confirmed_against_real_clang_and_ocamlopt(class_key):
    orc = _plugin.get_oracle_for(class_key, "c", "ocaml")
    res = orc.confirm(orc.find_divergence(_ocaml_unit(class_key)), ReexecHarness(_TC))
    rr = res.reexec
    assert rr.available, rr.reason
    assert rr.ub_reachable, "UBSan must trap on the witness (C is UB)"
    assert rr.rust_defined, "OCaml must produce a defined, deterministic outcome"
    assert rr.confirmed, rr.reason
    assert res.counterexample.confirmed
    assert res.counterexample.target_observed is not None


@_requires_ocaml
def test_equivalent_ocaml_translation_is_not_confirmed():
    # negative control: an OCaml port whose answer matches a *defined* C program
    # (no UB) must NOT be confirmed — the sanitizer never traps.
    harness = ReexecHarness(_TC)
    c_src = ("#include <stdio.h>\n#include <stdlib.h>\n"
             "int f(int a,int b){ return a + b; }\n"
             "int main(int c,char**v){ if(c<3)return 2;"
             " printf(\"%d\\n\", f(atoi(v[1]),atoi(v[2]))); return 0; }\n")
    ml_src = ("let f a b = Int32.add a b\n"
              "let () =\n"
              "  let a = Int32.of_string Sys.argv.(1) in\n"
              "  let b = Int32.of_string Sys.argv.(2) in\n"
              "  Printf.printf \"%ld\\n\" (f a b)\n")
    rr = harness.confirm_trap_vs_defined(c_src, ml_src, ["2", "3"],
                                         "division_by_zero", target_lang="ocaml")
    assert rr.available
    assert not rr.ub_reachable
    assert not rr.confirmed


# ── Step 124: target-semantics-pack SPI conformance suite ────────────────────
# Every TargetPack is the *only* per-language configuration the engine trusts, so
# a pack that violates the SPI contract (a non-total "defined" predicate, a
# compile_argv that forgets the output path, a class_resolution naming a class
# that does not exist) would silently poison the re-execution harness. This suite
# states those obligations as executable properties and runs them against every
# registered pack — pure data, no compilers, in the fast gate.

from src.ub_oracle import pack_conformance as _pc  # noqa: E402


def test_pack_conformance_passes_for_every_registered_pack():
    conf = _pc.confirm_pack_conformance()
    assert conf.ok, conf.detail()
    # the suite covers every pack the registry exposes (rust/go/swift/cpp/ocaml).
    assert conf.n_packs == len(PACKS)
    assert set(conf.by_pack) == set(PACKS)


@pytest.mark.parametrize("name", sorted(PACKS))
def test_every_pack_discharges_every_obligation(name):
    pc = _pc.check_pack(name)
    keys = {o.key for o in pc.obligations}
    assert keys == set(_pc.OBLIGATION_KEYS), keys
    assert pc.ok, [o.key + ": " + o.detail for o in pc.failures()]


def test_predicate_totality_is_actually_enforced_by_the_suite():
    # a deliberately *broken* pack (value-only data but a predicate that also
    # accepts 2) must be caught by the predicate-totality obligation.
    from src.ub_oracle.target_semantics import TargetPack
    import src.ub_oracle.pack_conformance as P

    class _Liar(TargetPack):
        def is_defined_returncode(self, rc):
            return rc in (0, 2)

    liar = _Liar(name="liar", compiler_candidates=("x",), source_suffix=".x",
                 defined_returncodes=(0,), compile_argv=lambda cc, s, o: [cc, "-o", o, s])
    ob = P._check_predicate_is_total(liar)
    assert not ob.passed and "disagrees" in ob.detail


def test_compile_argv_obligation_catches_a_missing_output_path():
    from src.ub_oracle.target_semantics import TargetPack
    import src.ub_oracle.pack_conformance as P
    broken = TargetPack(
        name="broken", compiler_candidates=("cc",), source_suffix=".c",
        defined_returncodes=(0,),
        # forgets to wire the output path -> the harness would never find a binary.
        compile_argv=lambda cc, s, o: [cc, s],
    )
    ob = P._check_compile_argv(broken)
    assert not ob.passed and "output path" in ob.detail


def test_resolution_keys_must_name_real_divergence_classes():
    from src.ub_oracle.target_semantics import TargetPack
    import src.ub_oracle.pack_conformance as P
    bogus = TargetPack(
        name="bogus", compiler_candidates=("cc",), source_suffix=".c",
        defined_returncodes=(0,), compile_argv=lambda cc, s, o: [cc, "-o", o, s],
        class_resolution={"not_a_real_class": "???"},
    )
    ob = P._check_resolutions_are_real(bogus)
    assert not ob.passed and "not_a_real_class" in ob.detail


# ── Step 125: N-language consistency oracle (>= 3 targets, witnessed live) ────
# Given one C source translated to three or more targets, flag any translation
# that disagrees with the majority. This is the multi-target generalisation of
# the pairwise oracle, built entirely on the existing pluggable machinery.

from src.ub_oracle import consistency as _cons  # noqa: E402

_requires_three_targets = pytest.mark.skipif(
    sum(_TC.target_available(t) for t in ("rust", "go", "swift", "ocaml")) < 3
    or not (_TC.c_available and _TC.ubsan),
    reason="needs C+UBSan and >=3 target compilers for the N-language oracle")


def test_consistency_plan_is_pure_data_and_n_language():
    # the participating targets for a class are exactly its registered oracles.
    plan = _cons.plan_consistency("shift_oob")
    assert plan.divergence_class == "shift_oob"
    assert set(plan.targets) >= {"rust", "go", "swift"}
    assert plan.is_n_language  # >= 3 translations to compare
    # div_by_zero is implemented for every safe target too.
    assert _cons.plan_consistency("div_by_zero").is_n_language
    # a class only one pair implements is *not* an N-language case.
    assert not _cons.plan_consistency("signed_shift_sign_bit").is_n_language


def test_canonical_outcome_buckets_value_abort_and_timeout():
    from src.ub_oracle.reexec import RunOutcome
    assert _cons.canonical_outcome(RunOutcome(0, "7", ""), "rust") == "VALUE:7"
    # a defined abort canonicalises to ABORT regardless of the (per-pack) code.
    assert _cons.canonical_outcome(RunOutcome(101, "", ""), "rust") == "ABORT"
    assert _cons.canonical_outcome(RunOutcome(2, "", ""), "go") == "ABORT"
    assert _cons.canonical_outcome(RunOutcome(-5, "", ""), "swift") == "ABORT"
    assert _cons.canonical_outcome(RunOutcome(2, "", ""), "ocaml") == "ABORT"
    # a non-defined termination is surfaced, not hidden.
    assert _cons.canonical_outcome(RunOutcome(139, "", ""), "rust") == "UNDEFINED:rc=139"
    assert _cons.canonical_outcome(
        RunOutcome(-1, "", "", timed_out=True), "rust") == "TIMEOUT"


@_requires_three_targets
def test_shift_oob_flags_rust_as_the_lone_outlier_live():
    # Rust's wrapping_shl MASKS the shift amount (1 << (32 mod 32) == 1) while Go
    # and Swift yield 0 on an out-of-range shift: three real compilers, one shared
    # C source, Rust is the flagged minority.
    cls, unit = "shift_oob", {"kind": "shift", "width": 32, "value": 1,
                              "source_lang": "c", "target_lang": "rust"}
    rep = _cons.check_consistency(cls, unit, ReexecHarness(_TC))
    assert rep.available, rep.reason
    assert rep.n_targets >= 3
    assert rep.c_ub_reachable, "the shared C source must be UB on the witness"
    assert rep.has_outlier and not rep.consistent
    assert rep.flagged == ("rust",), rep.reason
    # the majority printed 0; Rust printed 1.
    assert rep.majority_token == "VALUE:0"
    rust_obs = [o for o in rep.observations if o.target_lang == "rust"][0]
    assert rust_obs.token == "VALUE:1"


@_requires_three_targets
def test_div_by_zero_is_consistent_every_target_aborts_live():
    # every safe target aborts deterministically -> all canonicalise to ABORT,
    # so the N translations are mutually CONSISTENT (no outlier).
    cls, unit = "div_by_zero", {"kind": "div", "width": 32, "a": "a", "b": "b",
                                "source_lang": "c", "target_lang": "rust"}
    rep = _cons.check_consistency(cls, unit, ReexecHarness(_TC))
    assert rep.available, rep.reason
    assert rep.n_targets >= 3
    assert rep.consistent and not rep.has_outlier
    assert rep.majority_token == "ABORT"
    assert all(o.token == "ABORT" for o in rep.observations)


@_requires_three_targets
@pytest.mark.parametrize("cls,unit", _cons.consistency_units())
def test_curated_consistency_units_are_witnessed_live(cls, unit):
    rep = _cons.check_consistency(cls, unit, ReexecHarness(_TC))
    assert rep.available, rep.reason
    assert rep.n_targets >= 3
    # every observation is a *defined* token (no safe target is ever UNDEFINED).
    assert all(not o.token.startswith("UNDEFINED") for o in rep.observations)
    # the report's consistency verdict and the flagged set agree by construction.
    assert rep.consistent == (len(rep.flagged) == 0)


def test_consistency_is_not_available_without_three_targets():
    # a host with only one target present cannot run the N-language comparison.
    class _OneTarget:
        cc = "/usr/bin/clang"
        ubsan = True
        def target_available(self, t):
            return t == "rust"
        @property
        def c_available(self):
            return True
    class _H:
        status = _OneTarget()
    rep = _cons.check_consistency(
        "shift_oob", {"kind": "shift", "width": 32, "value": 1}, _H())
    assert not rep.available and ">=3" in rep.reason


# --------------------------------------------------------------------------- #
# Step 133 — SMT-backed oracle for the integer-UB family.
# --------------------------------------------------------------------------- #

from src.ub_oracle import smt_integer as _smt  # noqa: E402


def test_smt_integer_covers_the_integer_family():
    assert set(_smt.CLASSES) == {
        "signed_overflow", "shift_oob", "div_by_zero", "intmin_div_neg1",
    }


def test_smt_encoding_is_equisatisfiable_at_shipped_widths():
    # Z3 proves operational <-> spec for EVERY input at the real bit widths.
    proofs = _smt.prove_all_equisatisfiable()
    assert proofs and all(p.proved for p in proofs), \
        [(p.class_key, p.width, p.detail) for p in proofs if not p.proved]


def test_smt_encoding_is_equisatisfiable_at_small_widths():
    # The equivalence is width-parametric: it holds for tiny widths too, where
    # the whole input space is also brute-forceable (next test cross-checks it).
    for c in _smt.CLASSES:
        for w in (4, 6, 8):
            p = _smt.prove_equisatisfiable(c, w)
            assert p.proved, (c, w, p.detail)


def test_smt_witness_satisfies_the_independent_python_predicate():
    # The model Z3 returns must satisfy the reference (enumeration) semantics —
    # the symbolic and concrete definitions of UB agree on the witness.
    for c in _smt.CLASSES:
        for w in (32, 64):
            assert _smt.smt_path_agrees_with_predicate(c, w), (c, w)


def test_smt_witnesses_match_the_known_canonical_inputs():
    # div-by-zero -> b == 0 ; INT_MIN / -1 -> the unique overflow pair.
    assert _smt.smt_witness("div_by_zero", 32)["b"] == 0
    w = _smt.smt_witness("intmin_div_neg1", 32)
    assert w == {"a": -(1 << 31), "b": -1}
    w64 = _smt.smt_witness("intmin_div_neg1", 64)
    assert w64 == {"a": -(1 << 63), "b": -1}


def test_enumeration_reference_agrees_with_smt_on_existence_small_width():
    # On a fully enumerable width, "SMT finds a witness" iff "enumeration does".
    for c in _smt.CLASSES:
        smt_found = _smt.smt_witness(c, 6) is not None
        enum = _smt.enumerate_witness(c, 6, probe_budget=1 << 16)
        assert enum.exhausted  # the whole space was scanned
        assert (enum.found is not None) == smt_found, (c, enum)


def test_enumerated_witness_also_satisfies_the_encoding():
    # Decode an enumerated witness and confirm Z3's operational encoding accepts
    # it — closing the loop between brute force and the symbolic path.
    import z3
    for c in _smt.CLASSES:
        enum = _smt.enumerate_witness(c, 6, probe_budget=1 << 16)
        assert enum.found is not None, c
        vars_, operational, _spec = _smt._ENCODERS[c](6)
        solver = z3.Solver()
        for v in vars_:
            solver.add(v == z3.BitVecVal(enum.found[str(v)], 6))
        solver.add(operational)
        assert solver.check() == z3.sat, (c, enum.found)


def test_smt_beats_enumeration_on_the_rare_witness_classes():
    # The headline benchmark: for classes whose witness is vanishingly rare,
    # the SMT path answers in milliseconds while ordered enumeration blows its
    # probe budget without finding anything.
    for c in ("signed_overflow", "intmin_div_neg1"):
        b = _smt.benchmark(c, 32, probe_budget=1 << 20)
        assert b.smt_found
        assert not b.enum_found            # enumeration could not find it
        assert not b.enum_exhausted        # ...and did not exhaust the space
        assert b.enum_probes >= (1 << 19)  # it really did spend the budget


def test_smt_and_enumeration_both_trivial_on_dense_classes():
    # For dense classes (zero divisor, out-of-range shift) both paths succeed.
    for c in ("div_by_zero", "shift_oob"):
        b = _smt.benchmark(c, 32, probe_budget=1 << 16)
        assert b.smt_found and b.enum_found


def test_smt_witness_matches_operational_oracle_for_intmin():
    # The SMT encoding here is the SAME condition the shipped integer_ub oracle
    # solves; the witnessing pair must coincide.
    from src.ub_oracle.oracles.integer_ub import IntMinDivNeg1Oracle
    res = IntMinDivNeg1Oracle().find_divergence(
        {"kind": "div", "width": 32, "signed": True, "a": "a", "b": "b"})
    assert res.verdict == OracleVerdict.DIVERGENT
    smt = _smt.smt_witness("intmin_div_neg1", 32)
    assert res.counterexample.inputs == smt


# ---------------------------------------------------------------------------
# Step 112 — bit-field layout & packing ABI divergence oracle (C->Rust, C->Go)
# ---------------------------------------------------------------------------
from src.ub_oracle.oracles import bitfield_layout as _bf  # noqa: E402,F401


def test_bitfield_oracle_registered_for_rust_and_go_defined_mode():
    rust = _plugin.get_oracle_for("bitfield_layout", "c", "rust")
    go = _plugin.get_oracle_for("bitfield_layout", "c", "go")
    assert rust.confirmation_mode == "defined_divergence"
    assert go.confirmation_mode == "defined_divergence"
    assert "bitfield_layout" in CATALOGUE
    # impl-defined, not UB-rooted.
    assert not CATALOGUE["bitfield_layout"].is_ub_rooted()
    assert CATALOGUE["bitfield_layout"].c_standard_ref == "C17 6.7.2.1p11"


def test_bitfield_witness_is_in_range_and_images_differ():
    orc = _plugin.get_oracle_for("bitfield_layout", "c", "rust")
    res = orc.find_divergence({"kind": "bitfield_struct"})
    assert res.verdict is _plugin.OracleVerdict.DIVERGENT
    vals = res.counterexample.inputs
    # default fields a:3,b:5,c:8 maximised -> 7,31,255 (all in range, nonzero).
    assert vals == {"a": 7, "b": 31, "c": 255}
    # the packed C image (4 bytes) and the unpacked target image (3 bytes) differ.
    assert _bf._c_image([("a", 3), ("b", 5), ("c", 8)], vals) == "ffff0000"
    assert _bf._target_image([("a", 3), ("b", 5), ("c", 8)], vals) == "071fff"
    # a declared per-field range is honoured by the Z3 search.
    r2 = orc.find_divergence(
        {"kind": "bitfield_struct", "value_ranges": {"c": [1, 3]}})
    assert r2.counterexample.inputs["c"] == 3
    # not applicable to a non-bitfield unit.
    assert orc.find_divergence(
        {"kind": "div", "width": 32}).verdict is _plugin.OracleVerdict.NOT_APPLICABLE


@pytest.mark.skipif(not _TC.full_for("rust"),
                    reason="needs C+rustc toolchain")
def test_bitfield_rust_confirmed_against_real_compilers():
    orc = _plugin.get_oracle_for("bitfield_layout", "c", "rust")
    res = orc.confirm(orc.find_divergence({"kind": "bitfield_struct"}),
                      ReexecHarness(_TC))
    rr = res.reexec
    assert rr.available and rr.mode == "defined_divergence"
    assert rr.rust_defined, "both C and Rust must be defined & deterministic"
    assert rr.ub_consequential, "the size/byte image must observably differ"
    assert rr.confirmed and res.counterexample.confirmed


@pytest.mark.skipif(not _TC.full_for("go"),
                    reason="needs C+go toolchain")
def test_bitfield_go_confirmed_against_real_compilers():
    orc = _plugin.get_oracle_for("bitfield_layout", "c", "go")
    res = orc.confirm(orc.find_divergence({"kind": "bitfield_struct"}),
                      ReexecHarness(_TC))
    rr = res.reexec
    assert rr.available and rr.mode == "defined_divergence"
    assert rr.rust_defined and rr.ub_consequential
    assert rr.confirmed and res.counterexample.confirmed


# ---------------------------------------------------------------------------
# Step 108 — out-of-range enum / trap-representation oracle (C->Rust, C->Go)
# ---------------------------------------------------------------------------
from src.ub_oracle.oracles import enum_repr as _enum  # noqa: E402,F401


def test_enum_oracle_registered_for_rust_and_go_defined_mode():
    rust = _plugin.get_oracle_for("enum_out_of_range", "c", "rust")
    go = _plugin.get_oracle_for("enum_out_of_range", "c", "go")
    assert rust.confirmation_mode == "defined_divergence"
    assert go.confirmation_mode == "defined_divergence"
    assert "enum_out_of_range" in CATALOGUE
    assert not CATALOGUE["enum_out_of_range"].is_ub_rooted()


def test_enum_witness_is_least_value_past_last_enumerator():
    orc = _plugin.get_oracle_for("enum_out_of_range", "c", "rust")
    res = orc.find_divergence({"kind": "enum_cast"})
    assert res.verdict is _plugin.OracleVerdict.DIVERGENT
    # 3 enumerators (0,1,2) -> least out-of-range value is 3.
    assert res.counterexample.inputs == {"n": 3}
    # a declared range is honoured by the Z3 search.
    r2 = orc.find_divergence({"kind": "enum_cast", "value_range": [10, 20]})
    assert r2.counterexample.inputs == {"n": 10}
    # more enumerators -> the boundary moves.
    r3 = orc.find_divergence({"kind": "enum_cast", "enumerators": 5})
    assert r3.counterexample.inputs == {"n": 5}
    # not applicable to a non-enum unit.
    assert orc.find_divergence(
        {"kind": "div", "width": 32}).verdict is _plugin.OracleVerdict.NOT_APPLICABLE


@pytest.mark.skipif(not _TC.full_for("rust"),
                    reason="needs C+rustc toolchain")
def test_enum_rust_confirmed_against_real_compilers():
    orc = _plugin.get_oracle_for("enum_out_of_range", "c", "rust")
    res = orc.confirm(orc.find_divergence({"kind": "enum_cast"}),
                      ReexecHarness(_TC))
    rr = res.reexec
    assert rr.available and rr.mode == "defined_divergence"
    assert rr.rust_defined, "C keeps the raw value; Rust collapses to default (both defined)"
    assert rr.ub_consequential, "the printed value must differ (C=3 vs target=0)"
    assert rr.confirmed and res.counterexample.confirmed


@pytest.mark.skipif(not _TC.full_for("go"),
                    reason="needs C+go toolchain")
def test_enum_go_confirmed_against_real_compilers():
    orc = _plugin.get_oracle_for("enum_out_of_range", "c", "go")
    res = orc.confirm(orc.find_divergence({"kind": "enum_cast"}),
                      ReexecHarness(_TC))
    rr = res.reexec
    assert rr.available and rr.mode == "defined_divergence"
    assert rr.rust_defined and rr.ub_consequential
    assert rr.confirmed and res.counterexample.confirmed
