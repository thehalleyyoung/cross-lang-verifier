from __future__ import annotations

from dataclasses import replace
from typing import Dict

from src.ub_oracle import plugin
from src.ub_oracle import soundness_gate as sg
from src.ub_oracle.plugin import DivergenceOracle, OracleResult, OracleVerdict


def test_soundness_gate_covers_every_registered_oracle_with_live_witnesses():
    audit = sg.confirm_soundness_registry(probe_witnesses=True)

    assert audit.ok, audit.detail()
    assert len(audit.registered) == 60
    assert len(audit.statements) == len(audit.registered)
    assert len(audit.probes) == len(audit.registered)
    assert {p.key for p in audit.probes} == set(audit.registered)
    assert all(p.source_bytes > 20 and p.target_bytes > 20 for p in audit.probes)


def test_soundness_gate_negative_control_rejects_unstatemented_oracle():
    class UndocumentedOracle(DivergenceOracle):
        divergence_class = "signed_overflow"
        source_lang = "c"
        target_lang = "neverland"

        def applies_to(self, unit: Dict) -> bool:
            return True

        def find_divergence(self, unit: Dict) -> OracleResult:
            return OracleResult(OracleVerdict.NO_DIVERGENCE_FOUND, self.divergence_class)

    registered = tuple(plugin.ALL_ORACLES) + (UndocumentedOracle(),)

    audit = sg.audit_soundness_statements(
        oracles=registered,
        probe_witnesses=False,
    )

    assert not audit.ok
    assert any(
        problem.kind == "coverage"
        and problem.key == ("c", "neverland", "signed_overflow")
        and "no soundness statement" in problem.detail
        for problem in audit.problems
    )


def test_soundness_gate_negative_control_rejects_mode_drift():
    first = sg.SOUNDNESS_STATEMENTS[0]
    bad = replace(
        first,
        expected_confirmation_mode="defined_divergence",
        evidence_kind="defined_divergence",
    )
    statements = (bad,) + sg.SOUNDNESS_STATEMENTS[1:]

    audit = sg.audit_soundness_statements(
        statements=statements,
        probe_witnesses=False,
    )

    assert not audit.ok
    assert any(
        problem.kind == "mode"
        and problem.key == first.key
        and "registry mode" in problem.detail
        for problem in audit.problems
    )


def test_soundness_gate_metadata_only_pass_is_fast_and_complete():
    audit = sg.confirm_soundness_registry(probe_witnesses=False)

    assert audit.ok, audit.detail()
    assert audit.probes == ()
    assert len(audit.statements) == len(audit.registered) == 60
    assert all(st.statement for st in sg.SOUNDNESS_STATEMENTS)
