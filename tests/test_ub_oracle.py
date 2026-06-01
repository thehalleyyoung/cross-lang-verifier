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
