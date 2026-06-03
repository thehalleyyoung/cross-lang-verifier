from __future__ import annotations

from pathlib import Path

import pytest

from src.ub_oracle import mechanized_soundness as ms


_ROOT = Path(__file__).resolve().parents[1]


def test_coq_crosscheck_source_declares_independent_core_theorems():
    report = ms.confirm_coq_crosscheck()

    assert set(ms.REQUIRED_COQ_THEOREMS) <= set(report.theorems_present)
    assert not report.theorems_missing, report.theorems_missing
    assert report.ok, ms.render_coq_crosscheck(report)

    source = (_ROOT / ms.COQ_SOURCE).read_text(encoding="utf-8")
    assert "ProductSoundness" not in source
    assert "oracle_sound_coq" in source
    assert "product_program_preserves_divergence_witness_coq" in source


@pytest.mark.skipif(ms._coqc_binary() is None, reason="Coq/coqc not installed")
def test_coq_kernel_accepts_core_soundness_crosscheck():
    report = ms.confirm_coq_crosscheck()

    assert report.available
    assert report.kernel_accepted is True, report.stderr_tail
    assert report.fully_checked

