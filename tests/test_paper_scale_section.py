from __future__ import annotations

from pathlib import Path

from src.ub_oracle import large_scale_study as ls
from src.ub_oracle import paper_scale_section as ps


ROOT = Path(__file__).resolve().parents[1]


def test_scale_section_structural_numbers_are_generated_from_census():
    census = ls.corpus_census(ls.generate_corpus())
    text = ps.render_scale_structural_core(census)

    assert "60{,}000" in text
    assert "1{,}044{,}000" in text
    assert "499{,}200 / 544{,}800" in text
    assert "32{,}000 / 28{,}000" in text
    assert "29{,}600 / 30{,}400" in text
    for klass in census["by_class"]:
        assert klass.replace("_", "\\_") in text


def test_checked_in_scale_section_structural_core_is_fresh():
    committed = (ROOT / "docs" / "generated_scale_section.tex").read_text(
        encoding="utf-8"
    )

    assert ps.extract_structural_core(committed) == ps.render_scale_structural_core()


def test_scale_paper_section_check_is_toolchain_free_by_default():
    check = ps.confirm_scale_paper_section(run_live=False)

    assert check.ok, check.to_dict()
    assert check.structural_fresh
    assert check.live_ok
    assert check.live_available is False
