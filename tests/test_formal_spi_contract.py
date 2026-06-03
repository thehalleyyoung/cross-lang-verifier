"""Focused tests for 100_STEPS step 136: formal SPI typeclass contract."""

from __future__ import annotations

import pytest

from src.ub_oracle import pack_conformance as pc
from src.ub_oracle import plugin
from src.ub_oracle import spi_contract as spi
from src.ub_oracle.target_semantics import PACKS


def test_formal_spi_contract_source_matches_python_registry():
    report = spi.confirm_formal_spi_contract()

    assert report.ok, report.detail()
    assert not report.declarations_missing
    assert set(report.pack_instances_present) == {
        f"{name}TargetPackSPI" for name in PACKS
    }
    assert set(report.obligation_keys_present) == set(pc.OBLIGATION_KEYS)
    assert set(report.plugin_methods_present) == set(spi.PLUGIN_METHOD_KEYS)
    assert report.plugin_methods_match_python
    assert report.pack_conformance_ok


def test_formal_spi_contract_reuses_executable_pack_obligations():
    assert spi.TARGET_PACK_OBLIGATION_KEYS == pc.OBLIGATION_KEYS

    pack_conf = pc.confirm_pack_conformance()
    report = spi.confirm_formal_spi_contract()

    assert pack_conf.ok, pack_conf.detail()
    assert report.pack_conformance_ok
    assert report.obligation_keys_missing == ()


def test_oracle_plugin_method_contract_tracks_real_abc():
    assert set(plugin.DivergenceOracle.__abstractmethods__) == {
        "applies_to",
        "find_divergence",
    }
    assert callable(plugin.DivergenceOracle.confirm)
    assert spi.python_oracle_spi_methods_ok()


@pytest.mark.skipif(spi._lake_binary() is None, reason="Lean/Lake not installed")
def test_lake_kernel_accepts_formal_spi_contract():
    report = spi.confirm_formal_spi_contract()

    assert report.available
    assert report.kernel_accepted is True, report.stderr_tail or report.stdout_tail
    assert report.fully_checked
