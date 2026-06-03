from __future__ import annotations

import pytest

from src.ub_oracle import idiomatic_corpus as ic
from src.ub_oracle.reexec import ReexecHarness, toolchain_available


_TC = toolchain_available()
_FULL_RUST_GO = _TC.full_for("rust") and _TC.full_for("go")


def test_step161_expansion_has_three_named_idiomatic_families():
    items = ic.step161_items()
    assert tuple(it.item_id for it in items) == ic.STEP161_FAMILY_IDS
    assert {it.declared_label for it in items} == {"divergent", "equivalent"}
    by_id = {it.item_id: it for it in items}

    assert "uutils/coreutils-class" in by_id["coreutils-block-rounding"].provenance
    assert "sudo-rs-class" in by_id["sudo-rs-timeout-slice"].provenance
    assert "zlib-rs-class" in by_id["zlib-rs-adler-window"].provenance

    assert "checked_add" in by_id["coreutils-block-rounding"].targets["rust"]
    assert "checked_div" in by_id["sudo-rs-timeout-slice"].targets["rust"]
    assert "wrapping_add" in by_id["zlib-rs-adler-window"].targets["rust"]
    for item in items:
        assert set(item.targets) == {"rust", "go"}
        assert item.c_src.strip()
        assert all(src.strip() for src in item.targets.values())


@pytest.mark.skipif(
    not _FULL_RUST_GO,
    reason=f"needs clang+UBSan+rustc+go toolchains ({_TC})",
)
def test_step161_expansion_confirms_against_real_compilers():
    conf = ic.confirm_step161_expansion(langs=("rust", "go"))
    assert conf.available and conf.ok, conf.detail
    assert conf.hash_stable and conf.content_hash
    assert conf.n_items == 6
    assert conf.n_divergent == 4
    assert conf.n_equivalent == 2

    verdicts = {(v.item_id, v.lang): v for v in conf.report.verdicts}
    for item_id in ("coreutils-block-rounding", "sudo-rs-timeout-slice"):
        for lang in ("rust", "go"):
            v = verdicts[(item_id, lang)]
            assert v.correct, v.detail
            assert v.ub_confirmed and not v.safe_confirmed
    for lang in ("rust", "go"):
        v = verdicts[("zlib-rs-adler-window", lang)]
        assert v.correct, v.detail
        assert not v.ub_confirmed and not v.safe_confirmed


@pytest.mark.skipif(
    not _FULL_RUST_GO,
    reason=f"needs clang+rustc+go toolchains ({_TC})",
)
def test_zlib_step161_control_is_defined_and_observably_equal():
    item = next(it for it in ic.step161_items() if it.item_id == "zlib-rs-adler-window")
    harness = ReexecHarness(_TC)
    for lang, target_src in item.targets.items():
        for inputs in (item.ub_inputs, item.safe_inputs):
            args = [a for a in inputs if a != ""]
            res = harness.confirm_defined_divergence(
                item.c_src,
                "c",
                target_src,
                lang,
                args,
                "zlib_adler_equivalence",
            )
            assert res.available, res.reason
            assert res.rust_defined, res.reason
            assert not res.ub_reachable, res.reason
            assert not res.confirmed, res.reason
