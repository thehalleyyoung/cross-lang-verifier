from __future__ import annotations

import pytest

from src.ub_oracle import generalization as gen
from src.ub_oracle import pair_soundness as ps
from src.ub_oracle import plugin
from src.ub_oracle.plugin import OracleVerdict
from src.ub_oracle.reexec import toolchain_available


_TC = toolchain_available()


def _available_new_cases():
    return [c for c in gen.available_v2_cases(_TC) if c.new_pair]


def test_generalization_v2_cases_cover_new_pairs_and_controls():
    case_ids = {c.case_id for c in gen.GENERALIZATION_V2_CASES}
    assert set(ps.NEW_PAIR_CASE_IDS).issubset(case_ids)

    pairs = {c.pair for c in gen.GENERALIZATION_V2_CASES}
    assert {
        ("c", "zig"),
        ("c", "cpp"),
        ("rust", "c"),
        ("c", "wasm"),
        ("go", "rust"),
        ("c", "ocaml"),
    }.issubset(pairs)

    for case in gen.GENERALIZATION_V2_CASES:
        orc = plugin.get_oracle_for(
            case.divergence_class, case.source_lang, case.target_lang)
        positive = orc.find_divergence(case.positive_unit)
        negative = orc.find_divergence(case.negative_unit)
        assert positive.verdict is OracleVerdict.DIVERGENT, case.case_id
        assert positive.counterexample is not None, case.case_id
        assert negative.verdict is not OracleVerdict.DIVERGENT, case.case_id


@pytest.mark.skipif(
    len(_available_new_cases()) < 2,
    reason="needs at least two available new-pair toolchains",
)
def test_generalization_v2_confirms_available_new_pairs_with_stable_hash():
    report1 = gen.run_generalization_v2()
    report2 = gen.run_generalization_v2()
    assert report1.content_hash and report1.content_hash == report2.content_hash

    available_new_pairs = {r.pair for r in report1.available_new_results}
    assert len(available_new_pairs) >= 2
    for result in report1.available_results:
        assert result.ok, (result.case_id, result.detail)

    conf = gen.confirm_generalization_v2()
    assert conf.available and conf.ok, conf.detail
    assert conf.n_new_pairs >= 2
    assert conf.all_positive_confirmed and conf.zero_safe_flags


def test_pair_soundness_statements_are_machine_checked():
    confirmation = ps.confirm_pair_soundness()
    assert confirmation.ok, confirmation.render()

    by_id = {r.case_id: r for r in confirmation.results}
    assert set(by_id) == set(ps.NEW_PAIR_CASE_IDS)
    for stmt in ps.statements():
        result = by_id[stmt.case_id]
        assert result.static_ok, (stmt.statement, result.failures)
        assert result.expected_mode == ps.EXPECTED_MODES[stmt.case_id]
        if result.live_available:
            assert result.live_confirmed, (result.case_id, result.failures)
