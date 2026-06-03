"""Bug-bounty-style intake for suspected cross-language divergences.

Step 169 adds a submission format and harness for user-reported divergences.
The committed artifacts are deliberately deterministic: they validate schema,
source hashes, witness inputs, and reproduction-bundle syntax without recording
host-dependent compiler results.  Live replay is exposed as a runtime check and
is gated on the local toolchain, matching the rest of the evidence pipeline.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from . import divergence_findings as findings
from .reexec import ReexecHarness, ReexecResult, toolchain_available

SCHEMA_VERSION = "bug-intake/v1"

_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = _ROOT / "experiments" / "bug_intake"
TEMPLATE_PATH = EXPERIMENT_DIR / "intake_template.json"
SAMPLE_SUBMISSION_PATH = EXPERIMENT_DIR / "sample_submission.json"
RESULTS_PATH = EXPERIMENT_DIR / "intake_results.json"
DOC_PATH = _ROOT / "docs" / "bug_intake.md"
_REPRO_DIR = _ROOT / "docs" / "repro"

SUPPORTED_AUTO_REPRO = (
    "C -> Rust single-function submissions whose arguments are signed 32-bit "
    "integers and whose source side can be checked with clang/UBSan."
)

INTAKE_CONTRACT = (
    "submission is a candidate, not a promoted finding",
    "source and target code are included inline and hash-checked",
    "witness and safe-control argv values are supplied explicitly",
    "auto-reproduction runs only in a sandboxed environment because submissions "
    "are untrusted code",
    "live compiler replay is never serialized into committed artifacts",
)

REQUIRED_FIELDS = (
    "schema",
    "submission_id",
    "title",
    "source.language",
    "source.code",
    "source.sha256",
    "target.language",
    "target.code",
    "target.sha256",
    "function.name",
    "function.arguments",
    "function.return_type",
    "divergence.class",
    "divergence.claim",
    "witness_input",
    "safe_input",
    "claim_scope",
)


@dataclass(frozen=True)
class IntakeValidation:
    submission_id: str
    valid: bool
    errors: Tuple[str, ...] = field(default_factory=tuple)
    c_sha256: str = ""
    target_sha256: str = ""
    bundle_path: str = ""


@dataclass(frozen=True)
class IntakeReplayReport:
    submission_id: str
    available: bool
    ok: bool
    validation_ok: bool
    witness_confirmed: bool
    ub_reachable: bool
    target_defined: bool
    safe_silent: bool
    bundle_path: str = ""
    detail: str = ""


def _canonical_bytes(obj: object) -> bytes:
    return json.dumps(
        obj, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _sample_record() -> findings.FindingRecord:
    return findings.finding_records()[0]


def _sample_submission_dict() -> Dict[str, object]:
    rec = _sample_record()
    c_code = _read(_ROOT / rec.c_file)
    rust_code = _read(_ROOT / rec.rust_file)
    return {
        "schema": SCHEMA_VERSION,
        "submission_id": f"INTAKE-{rec.finding_id}",
        "title": "Signed-overflow divergence in a checked extraction unit",
        "submitter": {
            "contact": "redacted@example.invalid",
            "permission_to_contact": False,
        },
        "source": {
            "language": "c",
            "path": rec.c_file,
            "code": c_code,
            "sha256": _sha256_text(c_code),
        },
        "target": {
            "language": rec.target_lang,
            "path": rec.rust_file,
            "code": rust_code,
            "sha256": _sha256_text(rust_code),
        },
        "function": {
            "name": "coreutils_size_accumulate",
            "arguments": [{"name": "bytes", "type": "i32"}],
            "return_type": "i32",
        },
        "divergence": {
            "class": rec.divergence_class,
            "claim": (
                "The source-side C addition is undefined on the witness under "
                "UBSan, while the Rust target has a deterministic wrapping value."
            ),
        },
        "witness_input": list(rec.witness_input),
        "safe_input": list(rec.safe_input),
        "candidate_repo": rec.candidate_repo,
        "candidate_url": rec.candidate_url,
        "source_family": rec.source_family,
        "claim_scope": (
            "trusted checked-in sample for the intake harness; ordinary intake "
            "submissions remain candidates until promoted through the responsible "
            "findings lane"
        ),
        "notes": [
            "This sample mirrors CLV-DIV-0001 so the intake path is proven on real code.",
            "Do not run arbitrary third-party submissions outside a sandbox.",
        ],
    }


def template_document() -> Dict[str, object]:
    return {
        "schema": SCHEMA_VERSION,
        "description": "Template for suspected cross-language divergence reports.",
        "supported_auto_reproduction": SUPPORTED_AUTO_REPRO,
        "required_fields": list(REQUIRED_FIELDS),
        "intake_contract": list(INTAKE_CONTRACT),
        "promotion_policy": (
            "A valid intake submission is only a candidate. It becomes a finding "
            "after source audit, source-hash pinning, maintainer-ready disclosure "
            "review, and live compiler replay in the responsible findings lane."
        ),
        "sandbox_warning": (
            "The auto-reproduction harness compiles and runs submitted code. Treat "
            "submissions as untrusted and run them only in a sandboxed CI/container."
        ),
        "submission_template": {
            "schema": SCHEMA_VERSION,
            "submission_id": "INTAKE-YYYY-NNNN",
            "title": "Short suspected-divergence summary",
            "submitter": {"contact": "", "permission_to_contact": False},
            "source": {
                "language": "c",
                "path": "",
                "code": "",
                "sha256": "",
            },
            "target": {
                "language": "rust",
                "path": "",
                "code": "",
                "sha256": "",
            },
            "function": {
                "name": "",
                "arguments": [{"name": "x", "type": "i32"}],
                "return_type": "i32",
            },
            "divergence": {"class": "", "claim": ""},
            "witness_input": [],
            "safe_input": [],
            "candidate_repo": "",
            "candidate_url": "",
            "source_family": "",
            "claim_scope": "candidate only; not yet a confirmed upstream defect",
            "notes": [],
        },
    }


def sample_submission() -> Dict[str, object]:
    return _sample_submission_dict()


def _lookup(mapping: Mapping[str, object], dotted: str) -> object:
    cur: object = mapping
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _source_hash(side: object) -> str:
    if not isinstance(side, Mapping):
        return ""
    code = side.get("code")
    return _sha256_text(code) if isinstance(code, str) else ""


def _arg_names(submission: Mapping[str, object]) -> Tuple[str, ...]:
    function = submission.get("function")
    if not isinstance(function, Mapping):
        return ()
    raw_args = function.get("arguments")
    if not isinstance(raw_args, list):
        return ()
    names: List[str] = []
    for item in raw_args:
        if isinstance(item, Mapping) and isinstance(item.get("name"), str):
            names.append(str(item["name"]))
    return tuple(names)


def validate_submission(submission: Mapping[str, object]) -> IntakeValidation:
    submission_id = str(submission.get("submission_id", "<missing>"))
    errors: List[str] = []
    for field in REQUIRED_FIELDS:
        value = _lookup(submission, field)
        if value in (None, "", []):
            errors.append(f"missing required field: {field}")
    if submission.get("schema") != SCHEMA_VERSION:
        errors.append(f"unexpected schema {submission.get('schema')!r}")

    source = submission.get("source")
    target = submission.get("target")
    if not isinstance(source, Mapping):
        errors.append("source must be an object")
        source = {}
    if not isinstance(target, Mapping):
        errors.append("target must be an object")
        target = {}
    if source.get("language") != "c":
        errors.append("current auto-reproduction harness supports C sources only")
    if target.get("language") != "rust":
        errors.append("current auto-reproduction harness supports Rust targets only")

    c_sha = _source_hash(source)
    target_sha = _source_hash(target)
    if c_sha and source.get("sha256") != c_sha:
        errors.append("source.sha256 does not match source.code")
    if target_sha and target.get("sha256") != target_sha:
        errors.append("target.sha256 does not match target.code")

    function = submission.get("function")
    raw_args = function.get("arguments") if isinstance(function, Mapping) else None
    if not isinstance(raw_args, list) or not raw_args:
        errors.append("function.arguments must be a non-empty list")
    else:
        for idx, arg in enumerate(raw_args):
            if not isinstance(arg, Mapping):
                errors.append(f"function.arguments[{idx}] must be an object")
                continue
            if arg.get("type") != "i32":
                errors.append(
                    f"function.arguments[{idx}].type={arg.get('type')!r}; only i32 is currently auto-reproduced"
                )

    argc = len(_arg_names(submission))
    for field in ("witness_input", "safe_input"):
        raw = submission.get(field)
        if not isinstance(raw, list):
            errors.append(f"{field} must be a list")
        elif argc and len(raw) != argc:
            errors.append(f"{field} length {len(raw)} does not match argument count {argc}")

    bundle_path = f"docs/repro/{submission_id}.sh"
    return IntakeValidation(
        submission_id=submission_id,
        valid=not errors,
        errors=tuple(errors),
        c_sha256=c_sha,
        target_sha256=target_sha,
        bundle_path=bundle_path,
    )


def _c_program(submission: Mapping[str, object]) -> str:
    source = submission["source"]
    function = submission["function"]
    assert isinstance(source, Mapping)
    assert isinstance(function, Mapping)
    names = _arg_names(submission)
    reads = "\n".join(
        f"    int32_t {name} = (int32_t)strtol(argv[{idx}], 0, 10);"
        for idx, name in enumerate(names, start=1)
    )
    return (
        str(source["code"]).rstrip()
        + "\n\n#include <stdio.h>\n#include <stdlib.h>\n"
        + "int main(int argc, char **argv) {\n"
        + f"    if (argc < {len(names) + 1}) return 2;\n"
        + reads
        + "\n"
        + f"    printf(\"%d\\n\", {function['name']}({', '.join(names)}));\n"
        + "    return 0;\n"
        + "}\n"
    )


def _rust_program(submission: Mapping[str, object]) -> str:
    target = submission["target"]
    function = submission["function"]
    assert isinstance(target, Mapping)
    assert isinstance(function, Mapping)
    names = _arg_names(submission)
    reads = "\n".join(
        f"    let {name}: i32 = argv[{idx}].parse().unwrap();"
        for idx, name in enumerate(names, start=1)
    )
    return (
        str(target["code"]).rstrip()
        + "\n\nfn main() {\n"
        + "    let argv: Vec<String> = std::env::args().collect();\n"
        + f"    if argv.len() < {len(names) + 1} {{ std::process::exit(2); }}\n"
        + reads
        + "\n"
        + f"    println!(\"{{}}\", {function['name']}({', '.join(names)}));\n"
        + "}\n"
    )


def _bundle_script(submission: Mapping[str, object]) -> str:
    validation = validate_submission(submission)
    witness = " ".join(str(v) for v in submission["witness_input"])
    safe = " ".join(str(v) for v in submission["safe_input"])
    return (
        "#!/usr/bin/env bash\n"
        f"# Intake reproduction for {validation.submission_id}\n"
        "# Evidence tier: candidate intake submission, not an upstream defect claim.\n"
        "# WARNING: submitted code is untrusted; run only in a sandbox/container.\n"
        "set -euo pipefail\n"
        'WORK="$(mktemp -d)"\n'
        'trap \'rm -rf "$WORK"\' EXIT\n'
        "cat > \"$WORK/case.c\" <<'CEOF'\n"
        f"{_c_program(submission).rstrip()}\n"
        "CEOF\n"
        "cat > \"$WORK/case.rs\" <<'REOF'\n"
        f"{_rust_program(submission).rstrip()}\n"
        "REOF\n"
        'echo "== C intake submission under UBSan on witness =="\n'
        'clang -O1 -fsanitize=undefined -fno-sanitize-recover=all '
        '"$WORK/case.c" -o "$WORK/c_bin"\n'
        f'"$WORK/c_bin" {witness} || echo "  (C trapped / nonzero: UB is reachable)"\n'
        'echo "== Rust intake target on the same witness =="\n'
        'rustc -O "$WORK/case.rs" -o "$WORK/rs_bin"\n'
        f'"$WORK/rs_bin" {witness} || echo "  (Rust nonzero exit is checked by the target semantics pack)"\n'
        'echo "== Safe input control =="\n'
        f'"$WORK/c_bin" {safe}\n'
        f'"$WORK/rs_bin" {safe}\n'
    )


def load_submission(path: Path = SAMPLE_SUBMISSION_PATH) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def content_hash(submissions: Optional[Sequence[Mapping[str, object]]] = None) -> str:
    submissions = (sample_submission(),) if submissions is None else submissions
    stable = {
        "schema": SCHEMA_VERSION,
        "template_required_fields": list(REQUIRED_FIELDS),
        "submissions": [
            {
                "submission_id": str(sub["submission_id"]),
                "source_sha256": str(_lookup(sub, "source.sha256")),
                "target_sha256": str(_lookup(sub, "target.sha256")),
                "function": str(_lookup(sub, "function.name")),
                "arguments": list(_arg_names(sub)),
                "divergence_class": str(_lookup(sub, "divergence.class")),
                "witness_input": list(sub.get("witness_input", [])),
                "safe_input": list(sub.get("safe_input", [])),
            }
            for sub in submissions
        ],
    }
    return hashlib.sha256(_canonical_bytes(stable)).hexdigest()


def _validation_json(validation: IntakeValidation) -> Dict[str, object]:
    return {
        "submission_id": validation.submission_id,
        "valid": validation.valid,
        "errors": list(validation.errors),
        "c_sha256": validation.c_sha256,
        "target_sha256": validation.target_sha256,
        "bundle_path": validation.bundle_path,
    }


def results_document() -> Dict[str, object]:
    submission = sample_submission()
    validation = validate_submission(submission)
    return {
        "schema": SCHEMA_VERSION,
        "content_hash": content_hash((submission,)),
        "supported_auto_reproduction": SUPPORTED_AUTO_REPRO,
        "intake_contract": list(INTAKE_CONTRACT),
        "evidence_policy": (
            "Intake submissions are untrusted candidates. They are not counted as "
            "findings or upstream defects until separately promoted through the "
            "responsible findings and frozen bug-regression corpus."
        ),
        "sandbox_warning": template_document()["sandbox_warning"],
        "template_path": str(TEMPLATE_PATH.relative_to(_ROOT)),
        "sample_submission_path": str(SAMPLE_SUBMISSION_PATH.relative_to(_ROOT)),
        "n_sample_submissions": 1,
        "valid_sample_submissions": 1 if validation.valid else 0,
        "sample_submissions": [_validation_json(validation)],
    }


def write_template(path: Path = TEMPLATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(template_document(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_sample_submission(path: Path = SAMPLE_SUBMISSION_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sample_submission(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_results(path: Path = RESULTS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(results_document(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_bundle(
    submission: Optional[Mapping[str, object]] = None,
    *,
    repro_dir: Path = _REPRO_DIR,
) -> Path:
    submission = sample_submission() if submission is None else submission
    validation = validate_submission(submission)
    if not validation.valid:
        raise ValueError("; ".join(validation.errors))
    repro_dir.mkdir(parents=True, exist_ok=True)
    path = _ROOT / validation.bundle_path
    path.write_text(_bundle_script(submission), encoding="utf-8")
    path.chmod(0o755)
    return path


def _entries(doc: Mapping[str, object]) -> Sequence[Mapping[str, object]]:
    raw = doc.get("sample_submissions")
    if not isinstance(raw, list):
        raise AssertionError("sample_submissions must be a list")
    return raw


def load_results(path: Path = RESULTS_PATH) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def check_results(path: Path = RESULTS_PATH) -> Tuple[bool, str]:
    if not path.exists():
        return False, f"{path} is missing"
    expected = json.dumps(results_document(), indent=2, sort_keys=True) + "\n"
    if path.read_text(encoding="utf-8") != expected:
        return False, f"{path} does not match regenerated intake results"
    doc = json.loads(expected)
    if doc.get("schema") != SCHEMA_VERSION:
        return False, f"unexpected schema {doc.get('schema')!r}"
    if doc.get("content_hash") != content_hash((sample_submission(),)):
        return False, "content_hash does not match sample submission"
    entries = _entries(doc)
    if len(entries) != 1 or not entries[0].get("valid"):
        return False, "sample intake submission is not valid"
    sample_on_disk = SAMPLE_SUBMISSION_PATH.read_text(encoding="utf-8")
    regenerated_sample = json.dumps(sample_submission(), indent=2, sort_keys=True) + "\n"
    if sample_on_disk != regenerated_sample:
        return False, f"{SAMPLE_SUBMISSION_PATH} does not match regenerated sample"
    template_on_disk = TEMPLATE_PATH.read_text(encoding="utf-8")
    regenerated_template = json.dumps(template_document(), indent=2, sort_keys=True) + "\n"
    if template_on_disk != regenerated_template:
        return False, f"{TEMPLATE_PATH} does not match regenerated template"
    bundle = _ROOT / str(entries[0]["bundle_path"])
    if not bundle.exists():
        return False, f"{bundle} is missing"
    syntax = subprocess.run(
        ["bash", "-n", str(bundle)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if syntax.returncode != 0:
        return False, f"{bundle} syntax failed: {syntax.stderr}"
    return True, "OK"


def confirm_submission(
    submission: Optional[Mapping[str, object]] = None,
    *,
    harness: Optional[ReexecHarness] = None,
    write_repro_bundle: bool = False,
) -> IntakeReplayReport:
    submission = sample_submission() if submission is None else submission
    validation = validate_submission(submission)
    if not validation.valid:
        return IntakeReplayReport(
            submission_id=validation.submission_id,
            available=False,
            ok=False,
            validation_ok=False,
            witness_confirmed=False,
            ub_reachable=False,
            target_defined=False,
            safe_silent=False,
            detail="; ".join(validation.errors),
        )
    if write_repro_bundle:
        write_bundle(submission)
    status = toolchain_available()
    target_lang = str(_lookup(submission, "target.language"))
    if not status.full_for(target_lang):
        return IntakeReplayReport(
            submission_id=validation.submission_id,
            available=False,
            ok=True,
            validation_ok=True,
            witness_confirmed=False,
            ub_reachable=False,
            target_defined=False,
            safe_silent=False,
            bundle_path=validation.bundle_path,
            detail=f"{target_lang} toolchain absent; structural intake checks passed",
        )
    h = harness or ReexecHarness(status)
    witness: ReexecResult = h.confirm_trap_vs_defined(
        _c_program(submission),
        _rust_program(submission),
        [str(v) for v in submission["witness_input"]],
        str(_lookup(submission, "divergence.class")),
        target_lang,
    )
    safe: ReexecResult = h.confirm_trap_vs_defined(
        _c_program(submission),
        _rust_program(submission),
        [str(v) for v in submission["safe_input"]],
        str(_lookup(submission, "divergence.class")),
        target_lang,
    )
    safe_silent = bool(
        safe.available and not safe.ub_reachable and safe.rust_defined and not safe.confirmed
    )
    ok = bool(witness.confirmed and witness.ub_reachable and witness.rust_defined and safe_silent)
    return IntakeReplayReport(
        submission_id=validation.submission_id,
        available=True,
        ok=ok,
        validation_ok=True,
        witness_confirmed=bool(witness.confirmed),
        ub_reachable=bool(witness.ub_reachable),
        target_defined=bool(witness.rust_defined),
        safe_silent=safe_silent,
        bundle_path=validation.bundle_path,
        detail=(
            "sample intake submission replays live and safe control is silent"
            if ok
            else f"witness={witness.reason}; safe={safe.reason}"
        ),
    )


def markdown_document(doc: Optional[Mapping[str, object]] = None) -> str:
    doc = results_document() if doc is None else doc
    entry = _entries(doc)[0]
    lines = [
        "# Bug-bounty-style divergence intake",
        "",
        "This page is generated by `ub_oracle.bug_intake`. It defines a "
        "structured intake format for suspected cross-language divergences and "
        "a focused auto-reproduction harness for C -> Rust single-function "
        "reports.",
        "",
        f"- **Schema:** `{doc['schema']}`",
        f"- **Supported auto-reproduction:** {doc['supported_auto_reproduction']}",
        f"- **Template:** `{doc['template_path']}`",
        f"- **Checked sample submission:** `{doc['sample_submission_path']}`",
        f"- **Sample bundle:** `{entry['bundle_path']}`",
        f"- **Content hash:** `{doc['content_hash']}`",
        "",
        "**Evidence policy.** Intake reports are candidates only. They are not "
        "counted as real findings, CVEs, or upstream repository defects until "
        "they pass source audit, maintainer-ready disclosure review, live replay, "
        "and promotion into the responsible findings lane.",
        "",
        "**Sandbox warning.** The harness compiles and runs submitted C/Rust code. "
        "Treat third-party submissions as untrusted and run them only in an "
        "isolated container or CI sandbox.",
        "",
        "## Static validation contract",
        "",
    ]
    for item in INTAKE_CONTRACT:
        lines.append(f"- {item}")
    lines += [
        "",
        "## Proven sample",
        "",
        f"`{entry['submission_id']}` validates the intake path against the frozen "
        f"`{findings.FINDING_ID}` extraction unit: the submitted source hashes "
        "match the checked-in C/Rust snippets, the witness and safe control are "
        "declared explicitly, and the generated shell bundle is syntax-checked. "
        "When `clang`/UBSan and `rustc` are present, the focused test also replays "
        "the witness against real binaries.",
        "",
        "Run `make bug-intake-check` to validate the template, sample, bundle, "
        "and optional live replay.",
        "",
    ]
    return "\n".join(lines)


def write_docs(path: Path = DOC_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown_document(load_results()), encoding="utf-8")


def check_docs(path: Path = DOC_PATH) -> Tuple[bool, str]:
    if not path.exists():
        return False, f"{path} is missing"
    expected = markdown_document(load_results())
    if path.read_text(encoding="utf-8") != expected:
        return False, f"{path} does not match regenerated intake docs"
    return True, "OK"


def write_artifacts() -> Tuple[Path, Path, Path, Path, Path]:
    write_template()
    write_sample_submission()
    write_results()
    bundle = write_bundle()
    write_docs()
    return TEMPLATE_PATH, SAMPLE_SUBMISSION_PATH, RESULTS_PATH, DOC_PATH, bundle


if __name__ == "__main__":  # pragma: no cover
    template, sample, results, docs, bundle = write_artifacts()
    ok, detail = check_results(results)
    doc_ok, doc_detail = check_docs(docs)
    rep = confirm_submission(write_repro_bundle=False)
    print(
        "bug-intake "
        f"manifest={ok} docs={doc_ok} live_available={rep.available} "
        f"live_ok={rep.ok} validation={rep.validation_ok}"
    )
    if not ok:
        print(detail)
    if not doc_ok:
        print(doc_detail)
    print(f"wrote {template}, {sample}, {results}, {docs}, and {bundle}")
