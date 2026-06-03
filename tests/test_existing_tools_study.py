from __future__ import annotations

import pytest

from src.ub_oracle import c2rust_corpus
from src.ub_oracle import existing_tools_study as study
from src.ub_oracle.reexec import toolchain_available
from src.ub_oracle.verify import VerifyVerdict

_TC = toolchain_available()


def test_step162_subjects_are_exactly_the_c2rust_corpus():
    subjects = study.build_subjects()

    assert len(subjects) == len(c2rust_corpus.CORPUS) == 12
    assert sum(s.expected_divergent for s in subjects) == 8
    assert sum(not s.expected_divergent for s in subjects) == 4

    by_id = {s.item_id: s for s in subjects}
    assert by_id["nginx-rate"].arg_names == ("bytes", "seconds")
    assert by_id["nginx-rate"].witness_inputs == ("7", "0")
    assert by_id["musl-next-char"].witness_inputs == ("2147483647",)
    assert by_id["bzip2-block-div"].witness_inputs == ("-2147483648", "-1")
    assert by_id["xz-range-shift"].expected_symbolic_verdict == (
        VerifyVerdict.NO_DIVERGENCE_FOUND.value)
    assert by_id["xz-range-shift"].witness_inputs is None

    for subject in subjects:
        assert subject.c_sha256 and subject.rust_sha256
        assert set(subject.domains) == set(subject.arg_names)
        if subject.expected_divergent:
            assert subject.witness_inputs is not None
            assert len(subject.witness_inputs) == len(subject.arg_names)
        else:
            assert subject.witness_inputs is None

    h1 = study._report_hash(subjects)
    h2 = study._report_hash(study.build_subjects())
    assert h1 == h2


@pytest.mark.skipif(not _TC.full_for("rust"),
                    reason=f"requires C/UBSan/rustc toolchain ({_TC})")
def test_step162_head_to_head_on_real_c2rust_generated_code():
    conf = study.confirm_step162_existing_tools(trials=128, seed=study.DEFAULT_SEED)

    assert conf.available, conf.detail
    assert conf.ok, conf.detail
    assert conf.n_items == 12
    assert conf.expected_divergent == 8
    assert conf.safe_controls == 4
    assert conf.semrec_found == conf.expected_divergent
    assert conf.c2rust_tests_found == 0
    assert conf.fuzzer_found < conf.semrec_found
    assert conf.miri_found == 0
    assert conf.miri_status in {"ran", "unavailable"}
    assert {"musl-next-char", "zlib-prev-window", "sqlite-varint-advance",
            "bzip2-block-div"}.issubset(set(conf.only_semrec_units))
