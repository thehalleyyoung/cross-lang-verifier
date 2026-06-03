"""Tier-1 c2rust-output anchor corpus (100_STEPS step 41).

The corpus is deliberately stricter than the older "c2rust-style" examples:
each Rust artifact under ``experiments/c2rust_corpus/generated`` is produced by
running a real ``c2rust transpile`` binary on the checked-in C extraction unit
under ``experiments/c2rust_corpus/sources``.  The module records the exact
library family, source/generator hashes, validated IR unit, and verifier verdict
for each item so the benchmark population is reproducible and auditable.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .ir import assert_valid
from .verify import VerifyVerdict, verify_unit
from .cache import toolchain_provenance

SCHEMA_VERSION = "c2rust-corpus/v1"
TRANSLATOR = "c2rust"
TRANSLATOR_VERSION = "C2Rust 0.22.1"

_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = _ROOT / "experiments" / "c2rust_corpus"
SOURCE_DIR = EXPERIMENT_DIR / "sources"
GENERATED_DIR = EXPERIMENT_DIR / "generated"
RESULTS_PATH = EXPERIMENT_DIR / "results.json"


@dataclass(frozen=True)
class C2RustItem:
    item_id: str
    source_library: str
    source_function: str
    c_file: str
    rust_file: str
    divergence_class: str
    unit: Dict[str, object]
    expected_symbolic_verdict: str
    provenance: str

    @property
    def c_path(self) -> Path:
        return SOURCE_DIR / self.c_file

    @property
    def rust_path(self) -> Path:
        return GENERATED_DIR / self.rust_file


CORPUS: Tuple[C2RustItem, ...] = (
    C2RustItem(
        "musl-next-char",
        "musl libc",
        "musl_next_char",
        "musl_next_char.c",
        "musl_next_char.rs",
        "signed_overflow",
        {"name": "musl-next-char", "kind": "binop_const", "op": "add",
         "const": 1, "width": 32, "var": "x", "signed": True,
         "probe": "signed_overflow", "source_lang": "c", "target_lang": "rust"},
        VerifyVerdict.CANDIDATE.value,
        "single-function extraction unit for signed index increment patterns",
    ),
    C2RustItem(
        "zlib-prev-window",
        "zlib",
        "zlib_prev_window",
        "zlib_prev_window.c",
        "zlib_prev_window.rs",
        "signed_overflow",
        {"name": "zlib-prev-window", "kind": "binop_const", "op": "sub",
         "const": 1, "width": 32, "var": "x", "signed": True,
         "probe": "signed_overflow", "source_lang": "c", "target_lang": "rust"},
        VerifyVerdict.CANDIDATE.value,
        "single-function extraction unit for sliding-window index decrement patterns",
    ),
    C2RustItem(
        "sqlite-varint-advance",
        "SQLite",
        "sqlite_varint_advance",
        "sqlite_varint_advance.c",
        "sqlite_varint_advance.rs",
        "signed_overflow",
        {"name": "sqlite-varint-advance", "kind": "binop_const", "op": "add",
         "const": 7, "width": 32, "var": "x", "signed": True,
         "probe": "signed_overflow", "source_lang": "c", "target_lang": "rust"},
        VerifyVerdict.CANDIDATE.value,
        "single-function extraction unit for varint cursor advancement",
    ),
    C2RustItem(
        "libpng-row-stride",
        "libpng",
        "libpng_row_stride",
        "libpng_row_stride.c",
        "libpng_row_stride.rs",
        "signed_overflow",
        {"name": "libpng-row-stride", "kind": "binop_const", "op": "add",
         "const": 4, "width": 32, "var": "x", "signed": True,
         "probe": "signed_overflow", "source_lang": "c", "target_lang": "rust",
         "x_range": [0, 1048576]},
        VerifyVerdict.NO_DIVERGENCE_FOUND.value,
        "bounded image-row stride arithmetic with a declared safe operating range",
    ),
    C2RustItem(
        "nginx-rate",
        "nginx",
        "nginx_rate",
        "nginx_rate.c",
        "nginx_rate.rs",
        "div_by_zero",
        {"name": "nginx-rate", "kind": "div", "width": 32, "signed": True,
         "a": "bytes", "b": "seconds", "probe": "div_by_zero",
         "source_lang": "c", "target_lang": "rust"},
        VerifyVerdict.CANDIDATE.value,
        "single-function extraction unit for throughput bytes/seconds arithmetic",
    ),
    C2RustItem(
        "curl-remainder",
        "curl",
        "curl_remainder",
        "curl_remainder.c",
        "curl_remainder.rs",
        "div_by_zero",
        {"name": "curl-remainder", "kind": "rem", "width": 32, "signed": True,
         "a": "bytes", "b": "chunk", "probe": "div_by_zero",
         "source_lang": "c", "target_lang": "rust"},
        VerifyVerdict.CANDIDATE.value,
        "single-function extraction unit for chunk remainder arithmetic",
    ),
    C2RustItem(
        "lua-stack-shift",
        "Lua",
        "lua_stack_shift",
        "lua_stack_shift.c",
        "lua_stack_shift.rs",
        "shift_oob",
        {"name": "lua-stack-shift", "kind": "shift", "width": 32, "var": "mask",
         "shift_var": "shift", "probe": "shift_oob",
         "source_lang": "c", "target_lang": "rust"},
        VerifyVerdict.CANDIDATE.value,
        "single-function extraction unit for VM stack-mask shifts",
    ),
    C2RustItem(
        "openssl-ct-mask",
        "OpenSSL",
        "openssl_ct_mask",
        "openssl_ct_mask.c",
        "openssl_ct_mask.rs",
        "shift_oob",
        {"name": "openssl-ct-mask", "kind": "shift", "width": 32, "var": "v",
         "shift_var": "bits", "probe": "shift_oob",
         "source_lang": "c", "target_lang": "rust"},
        VerifyVerdict.CANDIDATE.value,
        "single-function extraction unit for constant-time mask shifts",
    ),
    C2RustItem(
        "git-pack-delta",
        "git",
        "git_pack_delta",
        "git_pack_delta.c",
        "git_pack_delta.rs",
        "signed_overflow",
        {"name": "git-pack-delta", "kind": "binop_const", "op": "add",
         "const": 64, "width": 32, "var": "x", "signed": True,
         "probe": "signed_overflow", "source_lang": "c", "target_lang": "rust",
         "x_range": [0, 16777216]},
        VerifyVerdict.NO_DIVERGENCE_FOUND.value,
        "bounded pack/delta offset arithmetic with a declared safe operating range",
    ),
    C2RustItem(
        "redis-sds-room",
        "Redis",
        "redis_sds_room",
        "redis_sds_room.c",
        "redis_sds_room.rs",
        "signed_overflow",
        {"name": "redis-sds-room", "kind": "binop_const", "op": "sub",
         "const": 8, "width": 32, "var": "x", "signed": True,
         "probe": "signed_overflow", "source_lang": "c", "target_lang": "rust",
         "x_range": [8, 1048576]},
        VerifyVerdict.NO_DIVERGENCE_FOUND.value,
        "bounded dynamic-string capacity arithmetic with a safe precondition",
    ),
    C2RustItem(
        "bzip2-block-div",
        "bzip2",
        "bzip2_block_div",
        "bzip2_block_div.c",
        "bzip2_block_div.rs",
        "intmin_div_neg1",
        {"name": "bzip2-block-div", "kind": "div", "width": 32, "signed": True,
         "a": "a", "b": "b", "probe": "intmin_div_neg1",
         "source_lang": "c", "target_lang": "rust"},
        VerifyVerdict.CANDIDATE.value,
        "single-function extraction unit for signed block-ratio division",
    ),
    C2RustItem(
        "xz-range-shift",
        "XZ Utils",
        "xz_range_shift",
        "xz_range_shift.c",
        "xz_range_shift.rs",
        "shift_oob",
        {"name": "xz-range-shift", "kind": "shift", "width": 32, "var": "symbol",
         "shift_var": "bits", "probe": "shift_oob",
         "source_lang": "c", "target_lang": "rust", "shift_range": [0, 15]},
        VerifyVerdict.NO_DIVERGENCE_FOUND.value,
        "bounded range-decoder shift arithmetic with a safe shift range",
    ),
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _canonical_bytes(obj: object) -> bytes:
    return json.dumps(obj, ensure_ascii=True, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def validate_corpus_shape(items: Iterable[C2RustItem] = CORPUS) -> None:
    items = tuple(items)
    libraries = {item.source_library for item in items}
    if len(items) != 12:
        raise AssertionError(f"expected 12 c2rust corpus items, got {len(items)}")
    if len(libraries) != 12:
        raise AssertionError(f"expected 12 distinct source libraries, got {libraries}")
    for item in items:
        assert_valid(item.unit, label=item.item_id)
        if item.unit.get("probe") != item.divergence_class:
            raise AssertionError(f"{item.item_id}: probe/class mismatch")


def c2rust_path() -> Optional[str]:
    return shutil.which("c2rust")


def c2rust_version() -> Optional[str]:
    exe = c2rust_path()
    if exe is None:
        return None
    run = subprocess.run([exe, "--version"], capture_output=True, text=True,
                         timeout=60)
    if run.returncode != 0:
        return None
    return run.stdout.strip()


def _transpile_one(item: C2RustItem, out_dir: Path) -> str:
    exe = c2rust_path()
    if exe is None:
        raise RuntimeError("c2rust executable is not on PATH")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        tmp_c = tmp_path / item.c_file
        shutil.copyfile(item.c_path, tmp_c)
        run = subprocess.run([exe, "transpile", item.c_file], cwd=tmp_path,
                             capture_output=True, text=True, timeout=120)
        if run.returncode != 0:
            raise RuntimeError(
                f"c2rust failed for {item.c_file}: {run.stderr.strip()}")
        generated = _read(tmp_c.with_suffix(".rs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / item.rust_file).write_text(generated, encoding="utf-8")
    return generated


def regenerate_generated(out_dir: Path = GENERATED_DIR) -> Dict[str, str]:
    """Regenerate Rust artifacts with the local c2rust binary.

    Returns ``item_id -> sha256(generated Rust)``.  The caller decides whether to
    write into the checked-in generated directory or a temporary comparison dir.
    """
    validate_corpus_shape()
    hashes = {}
    for item in CORPUS:
        generated = _transpile_one(item, out_dir)
        hashes[item.item_id] = _sha256_text(generated)
    return hashes


def compare_generated_to_c2rust() -> Dict[str, object]:
    """Regenerate in a temporary directory and compare to checked-in artifacts."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        regen_hashes = regenerate_generated(tmp_dir)
        mismatches = []
        for item in CORPUS:
            expected = _read(item.rust_path)
            actual = _read(tmp_dir / item.rust_file)
            if actual != expected:
                mismatches.append(item.item_id)
        provenance = toolchain_provenance()
        return {
            "available": True,
            "ok": not mismatches,
            "mismatches": mismatches,
            "c2rust_version": c2rust_version(),
            "toolchain_fingerprint": provenance["fingerprint"],
            "toolchain_provenance": provenance,
            "regenerated_hashes": regen_hashes,
        }


def _has_c2rust_hallmarks(rust_src: str) -> bool:
    return (
        "#[no_mangle]" in rust_src
        and 'unsafe extern "C" fn' in rust_src
        and "::core::ffi::c_" in rust_src
    )


def case_record(item: C2RustItem) -> Dict[str, object]:
    assert_valid(item.unit, label=item.item_id)
    c_src = _read(item.c_path)
    rust_src = _read(item.rust_path)
    report = verify_unit(dict(item.unit), confirm=False)
    return {
        "item_id": item.item_id,
        "source_library": item.source_library,
        "source_function": item.source_function,
        "provenance": item.provenance,
        "translator": TRANSLATOR,
        "translator_version": TRANSLATOR_VERSION,
        "c_file": str(item.c_path.relative_to(_ROOT)),
        "rust_file": str(item.rust_path.relative_to(_ROOT)),
        "source_sha256": _sha256_text(c_src),
        "rust_sha256": _sha256_text(rust_src),
        "c2rust_hallmarks": _has_c2rust_hallmarks(rust_src),
        "divergence_class": item.divergence_class,
        "unit": item.unit,
        "expected_symbolic_verdict": item.expected_symbolic_verdict,
        "observed_symbolic_verdict": report.verdict.value,
        "verdict_matches_expectation": (
            report.verdict.value == item.expected_symbolic_verdict
        ),
        "verifier_detail": report.detail,
        "prepass_pruned": sorted(report.prepass_pruned),
    }


def verdict_layer() -> List[Dict[str, object]]:
    return [case_record(item) for item in CORPUS]


def content_hash(cases: Optional[List[Dict[str, object]]] = None) -> str:
    cases = verdict_layer() if cases is None else cases
    stable = [
        {
            "item_id": c["item_id"],
            "source_sha256": c["source_sha256"],
            "rust_sha256": c["rust_sha256"],
            "observed_symbolic_verdict": c["observed_symbolic_verdict"],
            "prepass_pruned": c["prepass_pruned"],
        }
        for c in cases
    ]
    return _sha256_bytes(_canonical_bytes(stable))


def results_document() -> Dict[str, object]:
    validate_corpus_shape()
    cases = verdict_layer()
    observed = {}
    for case in cases:
        observed[case["observed_symbolic_verdict"]] = (
            observed.get(case["observed_symbolic_verdict"], 0) + 1
        )
    by_class = {}
    for case in cases:
        cls = case["divergence_class"]
        by_class[cls] = by_class.get(cls, 0) + 1
    return {
        "schema": SCHEMA_VERSION,
        "content_hash": content_hash(cases),
        "translator": TRANSLATOR,
        "translator_version": TRANSLATOR_VERSION,
        "n_items": len(cases),
        "n_source_libraries": len({c["source_library"] for c in cases}),
        "all_generated_by_c2rust_shape": all(c["c2rust_hallmarks"] for c in cases),
        "all_verdicts_match_expectation": all(
            c["verdict_matches_expectation"] for c in cases
        ),
        "by_symbolic_verdict": observed,
        "by_divergence_class": by_class,
        "cases": cases,
    }


def write_results(path: Path = RESULTS_PATH) -> None:
    doc = results_document()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def check_results(path: Path = RESULTS_PATH) -> Tuple[bool, str]:
    regenerated = json.dumps(results_document(), indent=2, sort_keys=True) + "\n"
    if not path.exists():
        return False, f"{path} is missing"
    on_disk = path.read_text(encoding="utf-8")
    if on_disk != regenerated:
        return False, f"{path} does not match regenerated results"
    return True, "OK"


def manifest_records() -> List[Dict[str, object]]:
    """Compact manifest for docs/tests without re-running the verifier."""
    return [
        {
            "item_id": item.item_id,
            "source_library": item.source_library,
            "source_function": item.source_function,
            "divergence_class": item.divergence_class,
            "expected_symbolic_verdict": item.expected_symbolic_verdict,
            "c_file": str(item.c_path.relative_to(_ROOT)),
            "rust_file": str(item.rust_path.relative_to(_ROOT)),
        }
        for item in CORPUS
    ]


__all__ = [
    "SCHEMA_VERSION",
    "TRANSLATOR",
    "TRANSLATOR_VERSION",
    "C2RustItem",
    "CORPUS",
    "EXPERIMENT_DIR",
    "SOURCE_DIR",
    "GENERATED_DIR",
    "RESULTS_PATH",
    "validate_corpus_shape",
    "c2rust_path",
    "c2rust_version",
    "regenerate_generated",
    "compare_generated_to_c2rust",
    "case_record",
    "verdict_layer",
    "content_hash",
    "results_document",
    "write_results",
    "check_results",
    "manifest_records",
]
