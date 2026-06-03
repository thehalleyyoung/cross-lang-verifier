from __future__ import annotations

import pytest

from src.ub_oracle import oracles  # noqa: F401  (registers plugins)
from src.ub_oracle import plugin as _plugin
from src.ub_oracle.plugin import OracleVerdict
from src.ub_oracle.reexec import ReexecHarness, toolchain_available
from src.ub_oracle.verify import VerifyVerdict, verify_unit


_TC = toolchain_available()

_requires_rust_c_ubsan = pytest.mark.skipif(
    not (_TC.can_compile("rust") and _TC.c_available and _TC.ubsan),
    reason=f"needs rustc + C compiler + UBSan ({_TC})",
)


def _oracle():
    return _plugin.get_oracle_for("intmin_div_neg1", "rust", "c")


def test_rust_to_c_reverse_pair_registered():
    assert ("rust", "c") in _plugin.language_pairs()
    orc = _oracle()
    assert orc.source_lang == "rust"
    assert orc.target_lang == "c"
    assert orc.confirmation_mode == "source_defined_target_ub"


def test_rust_to_c_oracle_scopes_to_reverse_pair_only():
    orc = _oracle()
    assert orc.applies_to({
        "kind": "div", "width": 32, "signed": True,
        "source_lang": "rust", "target_lang": "c",
    })
    assert orc.applies_to({
        "kind": "rem", "width": 64, "signed": True,
        "source_lang": "rust", "target_lang": "c",
    })
    assert not orc.applies_to({
        "kind": "div", "width": 32, "signed": True,
        "source_lang": "c", "target_lang": "rust",
    })
    assert not orc.applies_to({
        "kind": "div", "width": 32, "signed": False,
        "source_lang": "rust", "target_lang": "c",
    })


@pytest.mark.parametrize("kind,width", [("div", 32), ("rem", 32), ("div", 64)])
def test_rust_to_c_finds_intmin_neg1_symbolically(kind, width):
    res = _oracle().find_divergence({
        "kind": kind, "width": width, "signed": True,
        "source_lang": "rust", "target_lang": "c",
    })
    assert res.verdict is OracleVerdict.DIVERGENT
    ce = res.counterexample
    assert ce.source_lang == "rust" and ce.target_lang == "c"
    assert ce.source_definedness == "defined"
    vals = list(ce.inputs.values())
    assert -(1 << (width - 1)) in vals
    assert -1 in vals
    assert "fn f(" in ce.source_snippet
    assert "#include <stdint.h>" in ce.target_snippet
    assert "UBSan" in ce.divergence_witness


def test_rust_to_c_range_without_overflow_reports_no_divergence():
    res = _oracle().find_divergence({
        "kind": "div", "width": 32, "signed": True,
        "a_range": [0, 100], "b_range": [1, 10],
        "source_lang": "rust", "target_lang": "c",
    })
    assert res.verdict is OracleVerdict.NO_DIVERGENCE_FOUND


@_requires_rust_c_ubsan
@pytest.mark.parametrize("kind", ["div", "rem"])
def test_rust_to_c_confirmed_against_real_rustc_and_ubsan(kind):
    res = _oracle().confirm(
        _oracle().find_divergence({
            "kind": kind, "width": 32, "signed": True,
            "source_lang": "rust", "target_lang": "c",
        }),
        ReexecHarness(_TC),
    )
    rr = res.reexec
    assert rr.available and rr.mode == "source_defined_target_ub"
    assert rr.rust_defined, rr.reason
    assert rr.ub_reachable, rr.reason
    assert rr.confirmed, rr.reason
    assert rr.c_runs["source"].returncode == 101
    assert rr.c_runs["target_san"].ub_trapped
    assert res.counterexample.confirmed


@_requires_rust_c_ubsan
def test_source_defined_target_ub_negative_control_safe_division():
    rust_src = (
        "fn main(){ let v:Vec<String>=std::env::args().collect();"
        " let a:i32=v[1].parse().unwrap(); let b:i32=v[2].parse().unwrap();"
        " println!(\"{}\", a / b); }\n"
    )
    c_src = (
        "#include <stdio.h>\n#include <stdlib.h>\n"
        "int main(int argc,char**argv){ if(argc<3)return 2;"
        " int a=atoi(argv[1]); int b=atoi(argv[2]);"
        " printf(\"%d\\n\", a / b); return 0; }\n"
    )
    rr = ReexecHarness(_TC).confirm_source_defined_target_ub(
        rust_src, "rust", c_src, ["6", "3"], "intmin_div_neg1")
    assert rr.available and rr.mode == "source_defined_target_ub"
    assert rr.rust_defined
    assert not rr.ub_reachable
    assert not rr.confirmed


@_requires_rust_c_ubsan
def test_verify_unit_routes_rust_to_c_reverse_pair():
    rep = verify_unit({
        "kind": "div", "width": 32, "signed": True,
        "source_lang": "rust", "target_lang": "c",
    }, status=_TC)
    assert rep.verdict is VerifyVerdict.DIVERGENT
    assert rep.divergence.divergence_class == "intmin_div_neg1"
