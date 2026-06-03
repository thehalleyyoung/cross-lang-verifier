"""Focused tests for Step 126 strict-aliasing mechanized soundness."""

from __future__ import annotations

import re

import pytest

from src.ub_oracle import Definedness, ReexecHarness, toolchain_available
from src.ub_oracle import mechanized_soundness as ms
from src.ub_oracle import oracles as _oracles  # noqa: F401  (register plugins)
from src.ub_oracle.plugin import OracleVerdict, get_oracle_for

_STATUS = toolchain_available()


def test_lean_contract_requires_strict_aliasing_soundness_theorems():
    report = ms.confirm_mechanized_soundness()

    required = {
        "strict_aliasing_oracle_sound",
        "strict_aliasing_report_implies_type_pun",
        "strict_aliasing_report_implies_optimizer_exploited",
    }
    assert required <= set(ms.REQUIRED_THEOREMS)
    assert required <= set(report.theorems_present)
    assert not report.theorems_missing, report.theorems_missing
    assert report.ok, report.stderr_tail


@pytest.mark.skipif(ms._lean_binary() is None, reason="Lean 4 kernel not installed")
def test_lean_kernel_accepts_strict_aliasing_extension():
    report = ms.confirm_mechanized_soundness()

    assert report.available
    assert report.kernel_accepted is True, report.stderr_tail
    assert report.fully_checked


def test_strict_aliasing_oracle_witness_matches_mechanized_contract():
    oracle = get_oracle_for("strict_aliasing", "c", "rust")
    result = oracle.find_divergence({"kind": "type_pun"})

    assert oracle.confirmation_mode == "optimizer_exploited"
    assert result.verdict is OracleVerdict.DIVERGENT
    assert result.counterexample is not None
    assert result.counterexample.source_definedness == Definedness.UNDEFINED.value

    witness = result.counterexample.divergence_witness
    match = re.search(r"A=(\d+), B=(\d+)", witness)
    assert match, witness
    a_val = int(match.group(1))
    b_val = int(match.group(2))
    assert a_val != (b_val & 0xFFFFFFFF)
    assert "int *" in result.counterexample.source_snippet
    assert "long *" in result.counterexample.source_snippet


@pytest.mark.skipif(
    not (
        _STATUS.c_available
        and _STATUS.target_available("rust")
        and _STATUS.target_runnable("rust")
    ),
    reason="needs C compiler + rustc toolchain",
)
def test_strict_aliasing_real_confirmation_supplies_optimizer_signal():
    oracle = get_oracle_for("strict_aliasing", "c", "rust")
    result = oracle.confirm(
        oracle.find_divergence({"kind": "type_pun"}),
        ReexecHarness(_STATUS),
    )
    reexec = result.reexec

    assert reexec is not None and reexec.available
    assert reexec.mode == "optimizer_exploited"
    assert reexec.c_runs["A"].returncode == 0
    assert reexec.c_runs["B"].returncode == 0
    assert reexec.c_runs["A"].stdout != reexec.c_runs["B"].stdout
    assert reexec.ub_reachable and reexec.ub_consequential
    assert reexec.rust_defined and reexec.confirmed
    assert result.counterexample is not None and result.counterexample.confirmed
