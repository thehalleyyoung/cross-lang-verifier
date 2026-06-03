"""Step 162 -- same-corpus comparison against existing baselines.

This module is deliberately narrower and more concrete than the older
``external_baselines`` applicability table.  It takes the checked-in Tier-1
c2rust extraction corpus, wraps the *actual* generated Rust functions into
runnable programs, and compares four questions on the same items:

* SemRec: does the oracle confirm the C-UB-vs-Rust-defined split on the witness?
* c2rust-style regression tests: do compile + safe-input tests pass?
* Miri: does a Rust-only interpreter see a target-side issue?  Miri is recorded
  as structurally single-language; it is not counted as a cross-language oracle.
* equal-budget fuzzing: does random differential testing hit the witness?

The point is not to "beat" tools on questions they do not claim to answer.  The
point is to make the gap executable: c2rust-style tests and Miri can pass, the
fuzzer catches dense UB but misses sparse boundary witnesses, and SemRec still
confirms the divergence against real checked-in c2rust output.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from . import c2rust_corpus as c2rust
from . import oracles as _oracles  # noqa: F401 - populate oracle registry
from .headtohead import Domain, _sample, _trial_diverges
from .plugin import REGISTRY, OracleVerdict
from .reexec import ReexecHarness, RunOutcome, ToolchainStatus, toolchain_available
from .verify import VerifyVerdict

SCHEMA_VERSION = "existing-tools-study/v1"
DEFAULT_TRIALS = 256
DEFAULT_SEED = 162

I32: Domain = ("int", -(2 ** 31), 2 ** 31 - 1)
SMALL_NONNEG_SHIFTED: Domain = ("int", 0, 1024)
U16_SHIFT: Domain = ("int", 0, 2 ** 16 - 1)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_bytes(obj: object) -> bytes:
    return json.dumps(obj, ensure_ascii=True, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _unit_arg_names(unit: Dict[str, object]) -> Tuple[str, ...]:
    kind = unit.get("kind")
    if kind == "binop_const":
        return (str(unit.get("var", "x")),)
    if kind == "shift":
        return (str(unit.get("var", "x")), str(unit.get("shift_var", "s")))
    if kind in ("div", "rem"):
        return (str(unit.get("a", "a")), str(unit.get("b", "b")))
    raise ValueError(f"unsupported c2rust corpus unit kind: {kind!r}")


def _safe_inputs(unit: Dict[str, object], arg_names: Tuple[str, ...]) -> Tuple[str, ...]:
    kind = unit.get("kind")
    if kind == "binop_const":
        xr = unit.get("x_range")
        x = int(xr[0]) if xr is not None else 0
        return (str(x),)
    if kind == "shift":
        sr = unit.get("shift_range")
        shift = int(sr[0]) if sr is not None else 1
        return ("1", str(shift))
    if kind in ("div", "rem"):
        return ("7", "1")
    raise ValueError(f"unsupported c2rust corpus unit kind: {kind!r}")


def _domains(unit: Dict[str, object], arg_names: Tuple[str, ...]) -> Dict[str, Domain]:
    kind = unit.get("kind")
    if kind == "binop_const":
        xr = unit.get("x_range")
        if xr is not None:
            return {arg_names[0]: ("int", int(xr[0]), int(xr[1]))}
        return {arg_names[0]: I32}
    if kind == "shift":
        sr = unit.get("shift_range")
        shift_domain: Domain = (
            "int", int(sr[0]), int(sr[1])) if sr is not None else U16_SHIFT
        # The c2rust shift extraction units model the shift-amount boundary.  Keep
        # the shifted value non-negative so the fuzzer does not switch questions
        # to C's separate "left shift of a negative value" UB.
        return {arg_names[0]: SMALL_NONNEG_SHIFTED, arg_names[1]: shift_domain}
    if kind in ("div", "rem"):
        ar = unit.get("a_range")
        br = unit.get("b_range")
        a_domain: Domain = ("int", int(ar[0]), int(ar[1])) if ar is not None else I32
        b_domain: Domain = ("int", int(br[0]), int(br[1])) if br is not None else I32
        return {arg_names[0]: a_domain, arg_names[1]: b_domain}
    raise ValueError(f"unsupported c2rust corpus unit kind: {kind!r}")


def _witness_inputs(
    item: c2rust.C2RustItem,
    arg_names: Tuple[str, ...],
) -> Optional[Tuple[str, ...]]:
    oracle = REGISTRY.get(item.divergence_class)
    if oracle is None:
        return None
    res = oracle.find_divergence(dict(item.unit))
    canonical = _canonical_witness_inputs(item.unit, arg_names)
    if canonical is None:
        return None
    if res.verdict is not OracleVerdict.DIVERGENT or res.counterexample is None:
        raise AssertionError(f"{item.item_id}: expected oracle witness for {item.unit}")
    got = set(res.counterexample.inputs)
    expected = set(arg_names)
    if got != expected:
        raise AssertionError(
            f"{item.item_id}: oracle witness inputs {sorted(got)} do not match "
            f"c2rust function parameters {sorted(expected)}")
    return canonical


def _inside(unit: Dict[str, object], key: str, value: int) -> bool:
    rng = unit.get(key)
    if rng is None:
        return True
    return int(rng[0]) <= value <= int(rng[1])


def _signed_bounds(width: int) -> Tuple[int, int]:
    return -(1 << (width - 1)), (1 << (width - 1)) - 1


def _canonical_witness_inputs(
    unit: Dict[str, object],
    arg_names: Tuple[str, ...],
) -> Optional[Tuple[str, ...]]:
    kind = unit.get("kind")
    width = int(unit.get("width", 32))
    lo, hi = _signed_bounds(width)
    if kind == "binop_const":
        op = unit.get("op")
        const = int(unit.get("const", 0))
        if op == "add" and const > 0:
            x = hi - const + 1
        elif op == "add" and const < 0:
            x = lo - const - 1
        elif op == "sub" and const > 0:
            x = lo + const - 1
        elif op == "sub" and const < 0:
            x = hi + const + 1
        else:
            return None
        if not _inside(unit, "x_range", x):
            return None
        return (str(x),)
    if kind == "shift":
        shift = width
        if not _inside(unit, "shift_range", shift):
            return None
        return ("1", str(shift))
    if kind in ("div", "rem"):
        if unit.get("probe") == "intmin_div_neg1":
            a, b = lo, -1
            if not (_inside(unit, "a_range", a) and _inside(unit, "b_range", b)):
                return None
            return (str(a), str(b))
        b = 0
        if not _inside(unit, "b_range", b):
            return None
        return (str(int(unit.get("dividend", 7))), "0")
    raise ValueError(f"unsupported c2rust corpus unit kind: {kind!r}")


@dataclass(frozen=True)
class CorpusSubject:
    item_id: str
    source_library: str
    source_function: str
    divergence_class: str
    expected_symbolic_verdict: str
    unit: Dict[str, object]
    arg_names: Tuple[str, ...]
    safe_inputs: Tuple[str, ...]
    witness_inputs: Optional[Tuple[str, ...]]
    domains: Dict[str, Domain]
    c_source: str
    rust_source: str
    c_sha256: str
    rust_sha256: str

    @property
    def expected_divergent(self) -> bool:
        return self.expected_symbolic_verdict == VerifyVerdict.CANDIDATE.value

    @property
    def c_wrapper(self) -> str:
        reads = "".join(
            f"    int {name} = atoi(argv[{idx + 1}]);\n"
            for idx, name in enumerate(self.arg_names)
        )
        args = ", ".join(self.arg_names)
        return (
            "#include <stdio.h>\n"
            "#include <stdlib.h>\n"
            f"{self.c_source.rstrip()}\n\n"
            "int main(int argc, char **argv) {\n"
            f"    if (argc < {len(self.arg_names) + 1}) return 2;\n"
            f"{reads}"
            f"    printf(\"%d\\n\", {self.source_function}({args}));\n"
            "    return 0;\n"
            "}\n"
        )

    @property
    def rust_wrapper(self) -> str:
        reads = "".join(
            f"    let {name}: ::core::ffi::c_int = args[{idx + 1}].parse().unwrap();\n"
            for idx, name in enumerate(self.arg_names)
        )
        args = ", ".join(self.arg_names)
        rust_source = self.rust_source.rstrip()
        if "::core::" in rust_source and "extern crate core;" not in rust_source:
            first_item = rust_source.find("#[no_mangle]")
            if first_item == -1:
                raise ValueError(f"{self.item_id}: c2rust generated item marker missing")
            rust_source = (
                rust_source[:first_item]
                + "extern crate core;\n"
                + rust_source[first_item:]
            )
        # Keep the c2rust crate-level inner attributes at the beginning of the
        # crate by concatenating generated source first, then appending main.
        return (
            f"{rust_source}\n\n"
            "fn main() {\n"
            "    let args: Vec<String> = std::env::args().collect();\n"
            f"    if args.len() < {len(self.arg_names) + 1} {{ std::process::exit(2); }}\n"
            f"{reads}"
            f"    let out = unsafe {{ {self.source_function}({args}) }};\n"
            "    println!(\"{}\", out);\n"
            "}\n"
        )


def build_subjects(items: Iterable[c2rust.C2RustItem] = c2rust.CORPUS) -> List[CorpusSubject]:
    subjects: List[CorpusSubject] = []
    for item in items:
        c_src = item.c_path.read_text(encoding="utf-8")
        rust_src = item.rust_path.read_text(encoding="utf-8")
        arg_names = _unit_arg_names(item.unit)
        subjects.append(CorpusSubject(
            item_id=item.item_id,
            source_library=item.source_library,
            source_function=item.source_function,
            divergence_class=item.divergence_class,
            expected_symbolic_verdict=item.expected_symbolic_verdict,
            unit=dict(item.unit),
            arg_names=arg_names,
            safe_inputs=_safe_inputs(item.unit, arg_names),
            witness_inputs=_witness_inputs(item, arg_names),
            domains=_domains(item.unit, arg_names),
            c_source=c_src,
            rust_source=rust_src,
            c_sha256=_sha256_text(c_src),
            rust_sha256=_sha256_text(rust_src),
        ))
    return subjects


@dataclass(frozen=True)
class BaselineRun:
    available: bool
    ran: bool
    found: bool
    status: str
    detail: str = ""
    first_hit_trial: Optional[int] = None


def _c2rust_regression_baseline(
    subject: CorpusSubject,
    harness: ReexecHarness,
) -> BaselineRun:
    if not (harness.status.c_available and harness.status.target_available("rust")):
        return BaselineRun(False, False, False, "unavailable",
                           "C compiler or rustc unavailable")
    with tempfile.TemporaryDirectory() as d:
        c_bin = harness._compile_c(subject.c_wrapper, ["-O2"], d, "c_safe")
        rs_bin = harness._compile_target(subject.rust_wrapper, "rust", d, "rs_safe")
        if not (c_bin and rs_bin):
            return BaselineRun(True, True, True, "failed",
                               f"compile failed (c={bool(c_bin)} rust={bool(rs_bin)})")
        c = harness._run([c_bin, *subject.safe_inputs])
        r = harness._run([rs_bin, *subject.safe_inputs])
    passed = (
        c.returncode == 0
        and r.target_outcome_defined("rust")
        and r.returncode == 0
        and c.stdout == r.stdout
    )
    if passed:
        return BaselineRun(True, True, False, "passed",
                           f"safe input {subject.safe_inputs} agrees: {c.stdout!r}")
    return BaselineRun(
        True, True, True, "failed",
        f"safe input disagreement C(rc={c.returncode}, out={c.stdout!r}) "
        f"Rust(rc={r.returncode}, out={r.stdout!r}, err={r.stderr[:120]!r})")


def _semrec_on_actual_c2rust(
    subject: CorpusSubject,
    harness: ReexecHarness,
) -> BaselineRun:
    if not subject.expected_divergent:
        return BaselineRun(True, False, False, "control",
                           "abstract-interpretation/range control; no witness expected")
    if subject.witness_inputs is None:
        return BaselineRun(True, False, False, "no-witness",
                           "oracle produced no witness for candidate item")
    if not harness.status.full_for("rust"):
        return BaselineRun(False, False, False, "unavailable",
                           "C/UBSan/rustc toolchain unavailable")
    with tempfile.TemporaryDirectory() as d:
        san = harness._compile_c(
            subject.c_wrapper,
            ["-O1", "-fsanitize=undefined", "-fno-sanitize-recover=all"],
            d,
            "c_san",
        )
        rs = harness._compile_target(subject.rust_wrapper, "rust", d, "rs_step162")
        if not (san and rs):
            return BaselineRun(True, True, False, "compile-failed",
                               f"compile failed (san={bool(san)} rust={bool(rs)})")
        c = harness._run([san, *subject.witness_inputs])
        r1 = harness._run([rs, *subject.witness_inputs])
        r2 = harness._run([rs, *subject.witness_inputs])

    rust_deterministic = (
        r1.returncode == r2.returncode
        and r1.stdout == r2.stdout
        and not r1.timed_out
        and not r2.timed_out
    )
    rust_defined = rust_deterministic and (
        r1.target_outcome_defined("rust") or _c2rust_extern_abort_is_defined(r1)
    )
    confirmed = c.ub_trapped and rust_defined
    status = "confirmed" if confirmed else "missed"
    return BaselineRun(
        True,
        True,
        confirmed,
        status,
        f"c_ubsan_trap={c.ub_trapped}; rust_defined_or_deterministic_abort="
        f"{rust_defined}; rust_rc={r1.returncode}; rust_stderr={r1.stderr[:160]!r}",
    )


def _c2rust_extern_abort_is_defined(run: RunOutcome) -> bool:
    """c2rust emits ``extern "C"`` Rust functions.

    Division by zero and INT_MIN/-1 are Rust panics, but panicking through an
    ``extern "C"`` ABI aborts instead of unwinding with exit 101.  That abort is a
    deterministic Rust runtime outcome for this generated ABI, so Step 162 treats
    it as target-defined for the purpose of the C-UB-vs-target-deterministic
    comparison.
    """
    if run.timed_out or run.resource_exhausted:
        return False
    err = run.stderr.lower()
    return (
        run.returncode in (134, -6)
        and "panicked at" in err
        and (
            "panic in a function that cannot unwind" in err
            or "non-unwinding panic" in err
        )
    )


def _fuzzer_baseline(
    subject: CorpusSubject,
    harness: ReexecHarness,
    *,
    trials: int,
    seed: int,
) -> BaselineRun:
    if not harness.status.full_for("rust"):
        return BaselineRun(False, False, False, "unavailable",
                           "C/UBSan/rustc toolchain unavailable")
    with tempfile.TemporaryDirectory() as d:
        c_bin = harness._compile_c(
            subject.c_wrapper,
            ["-O0", "-fsanitize=undefined", "-fno-sanitize-recover=all"],
            d,
            "c_fuzz",
        )
        rs_bin = harness._compile_target(subject.rust_wrapper, "rust", d, "rs_fuzz")
        if not (c_bin and rs_bin):
            return BaselineRun(True, True, False, "compile-failed",
                               f"compile failed (c={bool(c_bin)} rust={bool(rs_bin)})")
        import random

        rng = random.Random(seed)
        first: Optional[int] = None
        hits = 0
        for trial in range(1, trials + 1):
            argv = [_sample(rng, subject.domains[name]) for name in subject.arg_names]
            if _trial_diverges(harness, c_bin, rs_bin, argv):
                hits = 1
                first = trial
                break
    return BaselineRun(
        True,
        True,
        hits > 0,
        "found" if hits else "missed",
        f"{hits}/{trials} hits over domains {subject.domains}",
        first_hit_trial=first,
    )


def _cargo_miri_available() -> Tuple[bool, str]:
    cargo = shutil.which("cargo")
    if cargo is None:
        return False, "cargo unavailable"
    try:
        run = subprocess.run(
            [cargo, "miri", "--version"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - host-dependent
        return False, f"cargo miri probe failed: {exc}"
    if run.returncode != 0:
        detail = (run.stderr or run.stdout).strip()
        return False, detail or "cargo miri unavailable"
    return True, run.stdout.strip()


def _miri_baseline(subject: CorpusSubject, *, run_miri: bool) -> BaselineRun:
    if not run_miri:
        return BaselineRun(False, False, False, "not-run",
                           "Miri baseline disabled by caller")
    ok, detail = _cargo_miri_available()
    if not ok:
        return BaselineRun(False, False, False, "unavailable", detail)
    argv = subject.witness_inputs if subject.witness_inputs is not None else subject.safe_inputs
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "Cargo.toml").write_text(
            "[package]\nname = \"clv_step162_miri\"\nversion = \"0.0.0\"\n"
            "edition = \"2021\"\n\n[profile.release]\noverflow-checks = false\n",
            encoding="utf-8",
        )
        src = root / "src"
        src.mkdir()
        (src / "main.rs").write_text(subject.rust_wrapper, encoding="utf-8")
        run = subprocess.run(
            ["cargo", "miri", "run", "--release", "--quiet", "--", *argv],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=180,
        )
    outcome = RunOutcome(run.returncode, run.stdout.strip(), run.stderr.strip())
    rust_defined = outcome.target_outcome_defined("rust")
    stderr_l = outcome.stderr.lower()
    target_ub = (not rust_defined) and (
        "undefined behavior" in stderr_l or "undefined behaviour" in stderr_l)
    return BaselineRun(
        True,
        True,
        False,
        "target-ub" if target_ub else "no-cross-language-finding",
        "Miri is Rust-only; "
        f"rc={outcome.returncode} rust_defined={rust_defined} "
        f"target_ub={target_ub} stderr={outcome.stderr[:160]!r}",
    )


@dataclass
class ExistingToolsRow:
    item_id: str
    source_library: str
    divergence_class: str
    expected_divergent: bool
    arg_names: Tuple[str, ...]
    safe_inputs: Tuple[str, ...]
    witness_inputs: Optional[Tuple[str, ...]]
    semrec: BaselineRun
    c2rust_tests: BaselineRun
    fuzzer: BaselineRun
    miri: BaselineRun

    @property
    def only_semrec(self) -> bool:
        return (
            self.expected_divergent
            and self.semrec.found
            and not self.c2rust_tests.found
            and not self.fuzzer.found
            and not self.miri.found
        )


@dataclass
class ExistingToolsReport:
    schema: str
    trials: int
    seed: int
    rows: List[ExistingToolsRow] = field(default_factory=list)
    content_hash: str = ""

    @property
    def n_items(self) -> int:
        return len(self.rows)

    @property
    def expected_divergent(self) -> int:
        return sum(1 for row in self.rows if row.expected_divergent)

    @property
    def safe_controls(self) -> int:
        return sum(1 for row in self.rows if not row.expected_divergent)

    @property
    def semrec_found(self) -> int:
        return sum(1 for row in self.rows if row.semrec.found)

    @property
    def c2rust_tests_found(self) -> int:
        return sum(1 for row in self.rows if row.c2rust_tests.found)

    @property
    def fuzzer_found(self) -> int:
        return sum(1 for row in self.rows if row.fuzzer.found)

    @property
    def miri_found(self) -> int:
        return sum(1 for row in self.rows if row.miri.found)

    @property
    def only_semrec_units(self) -> List[str]:
        return sorted(row.item_id for row in self.rows if row.only_semrec)

    @property
    def c2rust_tests_passed(self) -> bool:
        return bool(self.rows) and all(
            row.c2rust_tests.ran and row.c2rust_tests.status == "passed"
            for row in self.rows
        )

    @property
    def miri_status(self) -> str:
        if any(row.miri.ran for row in self.rows):
            return "ran"
        if all(row.miri.status == "unavailable" for row in self.rows):
            return "unavailable"
        return "not-run"

    def summary(self) -> Dict[str, object]:
        return {
            "schema": self.schema,
            "trials": self.trials,
            "seed": self.seed,
            "n_items": self.n_items,
            "expected_divergent": self.expected_divergent,
            "safe_controls": self.safe_controls,
            "semrec_found": self.semrec_found,
            "c2rust_tests_found": self.c2rust_tests_found,
            "fuzzer_found": self.fuzzer_found,
            "miri_found": self.miri_found,
            "miri_status": self.miri_status,
            "only_semrec_units": self.only_semrec_units,
            "content_hash": self.content_hash,
        }


def _report_hash(subjects: List[CorpusSubject]) -> str:
    stable = [
        {
            "item_id": s.item_id,
            "class": s.divergence_class,
            "expected": s.expected_symbolic_verdict,
            "args": s.arg_names,
            "safe": s.safe_inputs,
            "witness": s.witness_inputs,
            "domains": s.domains,
            "c_sha256": s.c_sha256,
            "rust_sha256": s.rust_sha256,
        }
        for s in subjects
    ]
    return hashlib.sha256(_canonical_bytes(stable)).hexdigest()


def run_existing_tools_study(
    *,
    trials: int = DEFAULT_TRIALS,
    seed: int = DEFAULT_SEED,
    status: Optional[ToolchainStatus] = None,
    run_miri: bool = True,
) -> ExistingToolsReport:
    subjects = build_subjects()
    harness = ReexecHarness(status or toolchain_available())
    rows: List[ExistingToolsRow] = []
    for subject in subjects:
        rows.append(ExistingToolsRow(
            item_id=subject.item_id,
            source_library=subject.source_library,
            divergence_class=subject.divergence_class,
            expected_divergent=subject.expected_divergent,
            arg_names=subject.arg_names,
            safe_inputs=subject.safe_inputs,
            witness_inputs=subject.witness_inputs,
            semrec=_semrec_on_actual_c2rust(subject, harness),
            c2rust_tests=_c2rust_regression_baseline(subject, harness),
            fuzzer=_fuzzer_baseline(subject, harness, trials=trials, seed=seed),
            miri=_miri_baseline(subject, run_miri=run_miri),
        ))
    return ExistingToolsReport(
        schema=SCHEMA_VERSION,
        trials=trials,
        seed=seed,
        rows=rows,
        content_hash=_report_hash(subjects),
    )


@dataclass
class Step162Confirmation:
    available: bool
    ok: bool
    n_items: int
    expected_divergent: int
    safe_controls: int
    semrec_found: int
    c2rust_tests_found: int
    fuzzer_found: int
    miri_found: int
    miri_status: str
    only_semrec_units: List[str]
    content_hash: str
    report: ExistingToolsReport
    detail: str


def confirm_step162_existing_tools(
    *,
    trials: int = DEFAULT_TRIALS,
    seed: int = DEFAULT_SEED,
    run_miri: bool = True,
) -> Step162Confirmation:
    report = run_existing_tools_study(trials=trials, seed=seed, run_miri=run_miri)
    available = (
        report.n_items > 0
        and all(row.semrec.available for row in report.rows if row.expected_divergent)
        and all(row.c2rust_tests.available for row in report.rows)
        and all(row.fuzzer.available for row in report.rows)
    )
    sparse_misses = {
        "musl-next-char",
        "zlib-prev-window",
        "sqlite-varint-advance",
        "bzip2-block-div",
    }.issubset(set(report.only_semrec_units))
    controls_clean = all(
        not row.semrec.found and not row.c2rust_tests.found and not row.fuzzer.found
        for row in report.rows
        if not row.expected_divergent
    )
    ok = (
        available
        and report.n_items == len(c2rust.CORPUS)
        and report.expected_divergent == 8
        and report.safe_controls == 4
        and report.semrec_found == report.expected_divergent
        and report.c2rust_tests_found == 0
        and report.fuzzer_found < report.semrec_found
        and sparse_misses
        and controls_clean
        and report.miri_found == 0
        and report.c2rust_tests_passed
    )
    detail = (
        f"items={report.n_items} expected_divergent={report.expected_divergent} "
        f"semrec={report.semrec_found} c2rust_tests={report.c2rust_tests_found} "
        f"fuzzer={report.fuzzer_found} miri={report.miri_found} "
        f"miri_status={report.miri_status} only_semrec={report.only_semrec_units} "
        f"hash={report.content_hash[:12]}"
    )
    return Step162Confirmation(
        available=available,
        ok=ok,
        n_items=report.n_items,
        expected_divergent=report.expected_divergent,
        safe_controls=report.safe_controls,
        semrec_found=report.semrec_found,
        c2rust_tests_found=report.c2rust_tests_found,
        fuzzer_found=report.fuzzer_found,
        miri_found=report.miri_found,
        miri_status=report.miri_status,
        only_semrec_units=report.only_semrec_units,
        content_hash=report.content_hash,
        report=report,
        detail=detail,
    )


EXISTING_TOOLS_STUDY_SPI = {
    "SCHEMA_VERSION": SCHEMA_VERSION,
    "build_subjects": build_subjects,
    "run_existing_tools_study": run_existing_tools_study,
    "confirm_step162_existing_tools": confirm_step162_existing_tools,
}
