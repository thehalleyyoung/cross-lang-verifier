from __future__ import annotations

from src.ub_oracle import mechanized_soundness as ms
from src.ub_oracle import soundness_compendium as sc
from src.ub_oracle import soundness_gate as sg
from src.ub_oracle import traceability


def test_compendium_renders_every_oracle_statement_and_required_theorem():
    text = sc.render_compendium()
    rows = sc.compendium_rows()

    assert len(rows) == len(sg.SOUNDNESS_STATEMENTS)
    for row in rows:
        assert f"`{row.statement_id}`" in text
        assert f"`{row.oracle}`" in text
        assert row.witness_unit_json in text
    for theorem in ms.REQUIRED_THEOREMS:
        assert f"`{theorem}`" in text
    for claim_id in traceability.claim_ids():
        assert f"`{claim_id}`" in text


def test_soundness_compendium_doc_is_fresh_and_refs_are_valid():
    check = sc.confirm_compendium(probe_witnesses=False)

    assert check.ok, check.detail()
    assert check.doc_fresh
    assert check.audit_ok
    assert check.missing_claim_refs == ()
    assert check.missing_lean_refs == ()


def test_soundness_compendium_witness_units_exercise_real_oracle_paths():
    check = sc.confirm_compendium(probe_witnesses=True)

    assert check.ok, check.detail()
    assert check.row_count == len(sg.SOUNDNESS_STATEMENTS)
