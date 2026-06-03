from __future__ import annotations

import re
from pathlib import Path

from src.ub_oracle import completeness as comp
from src.ub_oracle import mechanized_soundness as ms

_ROOT = Path(__file__).resolve().parents[1]


def _boundary_source() -> str:
    return (_ROOT / ms.COMPLETENESS_BOUNDARY_SOURCE).read_text(encoding="utf-8")


def _lean_list(source: str, name: str) -> tuple[str, ...]:
    match = re.search(rf"def {name} : List String :=\s*\[(.*?)\]", source, re.S)
    assert match, f"missing Lean list {name}"
    return tuple(re.findall(r'"([^"]+)"', match.group(1)))


def test_lean_boundary_partition_matches_runtime_completeness_registry():
    source = _boundary_source()
    class_key_block = source.split("def classKey", 1)[1].split(
        "/-- Classes complete", 1)[0]
    lean_class_keys = set(re.findall(r'=> "([^"]+)"', class_key_block))

    guaranteed = set(ms.COMPLETENESS_GUARANTEED_KEYS)
    may_abstain = set(ms.COMPLETENESS_MAY_ABSTAIN_KEYS)

    assert guaranteed == set(comp.FRAGMENTS)
    assert may_abstain == set(comp.OUT_OF_FRAGMENT)
    assert guaranteed.isdisjoint(may_abstain)
    assert lean_class_keys == guaranteed | may_abstain
    assert set(_lean_list(source, "guaranteedClassKeys")) == guaranteed
    assert set(_lean_list(source, "mayAbstainClassKeys")) == may_abstain


def test_mechanized_completeness_boundary_declares_and_checks_required_theorems():
    report = ms.confirm_mechanized_completeness_boundary()

    assert report.source_present
    assert report.lakefile_present
    assert report.source_hash and len(report.source_hash) == 64
    assert set(report.theorems_present) == set(
        ms.REQUIRED_COMPLETENESS_BOUNDARY_THEOREMS)
    assert not report.theorems_missing, report.theorems_missing
    assert report.ok, ms.render_completeness_boundary(report)
    if report.available:
        assert report.kernel_accepted is True, report.stderr_tail
        assert report.fully_checked


def test_boundary_theorem_is_scoped_to_the_registered_fragment_not_overclaimed():
    source = _boundary_source()

    assert "does not replace the Python/Z3/brute-force" in source
    assert "complete_fragment_decides_recorded_observation" in source
    assert "oracle_decides" in source
