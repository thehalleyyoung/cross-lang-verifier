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
