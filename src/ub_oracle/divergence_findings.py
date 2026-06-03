"""Evidence-tiered divergence findings for actively maintained ports.

Step 160 asks for a real bug-finding workflow on actively maintained C->Rust
ports.  This module is deliberately conservative about what the checked-in
evidence proves: the oracle confirms a divergence on a minimized extraction unit
from a live port family, emits a maintainer-ready disclosure draft and runnable
bundle, and labels the upstream-repository claim as pending until the same
pattern is confirmed in the upstream source.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from . import github_port_miner as miner
from .reexec import ReexecHarness, ReexecResult, toolchain_available
from .verify import verify_unit

SCHEMA_VERSION = "divergence-findings/v1"
FINDING_ID = "CLV-DIV-0001"

_ROOT = Path(__file__).resolve().parents[2]
_DOCS = _ROOT / "docs"
_REPRO = _DOCS / "repro"
_DOC_PATH = _DOCS / "divergence_findings.md"
RESULTS_PATH = miner.EXPERIMENT_DIR / "divergence_findings.json"


_MAINTENANCE_SNAPSHOTS: Mapping[str, Mapping[str, object]] = {
    "uutils/coreutils": {
        "source": "gh repo view snapshot",
        "source_command": (
            "gh repo view uutils/coreutils "
            "--json nameWithOwner,pushedAt,updatedAt,isArchived,stargazerCount,url,description"
        ),
        "checked_at_utc": "2026-06-03T10:11:01Z",
        "pushed_at": "2026-06-03T08:05:36Z",
        "updated_at": "2026-06-03T09:44:48Z",
        "archived": False,
        "stars": 23377,
        "active_evidence": (
            "GitHub reported a same-day push and update when this finding was "
            "prepared; the repository was not archived."
        ),
    },
    "trifectatechfoundation/sudo-rs": {
        "source": "gh repo view snapshot",
        "source_command": (
            "gh repo view trifectatechfoundation/sudo-rs "
            "--json nameWithOwner,pushedAt,updatedAt,isArchived,stargazerCount,url,description"
        ),
        "checked_at_utc": "2026-06-03T10:11:01Z",
        "pushed_at": "2026-06-03T08:19:26Z",
        "updated_at": "2026-06-03T08:19:29Z",
        "archived": False,
        "stars": 4390,
        "active_evidence": (
            "GitHub reported a same-day push and update when this finding was "
            "prepared; the repository was not archived."
        ),
    },
}


@dataclass(frozen=True)
class FindingRecord:
    finding_id: str
    sample_id: str
    candidate_repo: str
    candidate_url: str
    source_family: str
    divergence_class: str
    target_lang: str
    evidence_tier: str
    upstream_status: str
    disclosure_status: str
    witness_input: Tuple[str, ...]
    safe_input: Tuple[str, ...]
    c_file: str
    rust_file: str
    c_sha256: str
    rust_sha256: str
    maintenance_snapshot: Mapping[str, object]
    remediation: str
    note: str


@dataclass(frozen=True)
class FindingConfirmation:
    finding_id: str
    available: bool
    confirmed: bool
    ub_reachable: bool
    target_defined: bool
    safe_silent: bool
    bundle_path: str = ""
    detail: str = ""


@dataclass(frozen=True)
class FindingsReport:
    available: bool
    ok: bool
    n_records: int
    n_confirmed: int
    bundles_valid: bool
    confirmations: Tuple[FindingConfirmation, ...] = field(default_factory=tuple)
    detail: str = ""


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_bytes(obj: object) -> bytes:
    return json.dumps(
        obj, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sample() -> miner.GitHubPortSample:
    return next(s for s in miner.SAMPLES if s.sample_id == "coreutils-size-accumulate")


def _candidate(sample: miner.GitHubPortSample) -> miner.GitHubPortCandidate:
    return next(c for c in miner.SEED_CANDIDATES if c.owner_repo == sample.candidate_repo)


def _function_name(sample: miner.GitHubPortSample) -> str:
    return sample.sample_id.replace("-", "_")


def _arg_names(sample: miner.GitHubPortSample) -> Tuple[str, ...]:
    unit = sample.unit
    if unit.get("kind") in ("div", "rem"):
        return str(unit.get("a", "a")), str(unit.get("b", "b"))
    if unit.get("kind") == "binop_const":
        return (str(unit.get("var", "x")),)
    raise ValueError(f"{sample.sample_id}: unsupported finding unit kind {unit.get('kind')!r}")


def _counterexample_inputs(sample: miner.GitHubPortSample) -> Mapping[str, object]:
    report = verify_unit(dict(sample.unit), confirm=False)
    for result in report.oracle_results:
        if result.counterexample is not None:
            return result.counterexample.inputs
    raise RuntimeError(f"{sample.sample_id}: no symbolic counterexample was produced")


def witness_input(sample: miner.GitHubPortSample) -> Tuple[str, ...]:
    inputs = _counterexample_inputs(sample)
    return tuple(str(inputs[name]) for name in _arg_names(sample))


def safe_input(sample: miner.GitHubPortSample) -> Tuple[str, ...]:
    unit = sample.unit
    if unit.get("kind") in ("div", "rem"):
        return str(unit.get("dividend", 7)), "1"
    if unit.get("kind") == "binop_const":
        return ("0",)
    raise ValueError(f"{sample.sample_id}: unsupported finding unit kind {unit.get('kind')!r}")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _c_wrapper(sample: miner.GitHubPortSample) -> str:
    args = _arg_names(sample)
    fn = _function_name(sample)
    reads = "\n".join(
        f"    int32_t {name} = (int32_t)strtol(argv[{idx}], 0, 10);"
        for idx, name in enumerate(args, start=1)
    )
    joined = ", ".join(args)
    return (
        _read(sample.c_path).rstrip()
        + "\n\n#include <stdio.h>\n#include <stdlib.h>\n"
        + "int main(int argc, char **argv) {\n"
        + f"    if (argc < {len(args) + 1}) return 2;\n"
        + reads
        + "\n"
        + f"    printf(\"%d\\n\", {fn}({joined}));\n"
        + "    return 0;\n"
        + "}\n"
    )


def _rust_wrapper(sample: miner.GitHubPortSample) -> str:
    args = _arg_names(sample)
    fn = _function_name(sample)
    reads = "\n".join(
        f"    let {name}: i32 = argv[{idx}].parse().unwrap();"
        for idx, name in enumerate(args, start=1)
    )
    joined = ", ".join(args)
    return (
        _read(sample.rust_path).rstrip()
        + "\n\nfn main() {\n"
        + "    let argv: Vec<String> = std::env::args().collect();\n"
        + f"    if argv.len() < {len(args) + 1} {{ std::process::exit(2); }}\n"
        + reads
        + "\n"
        + f"    println!(\"{{}}\", {fn}({joined}));\n"
        + "}\n"
    )


def finding_records() -> Tuple[FindingRecord, ...]:
    sample = _sample()
    cand = _candidate(sample)
    c_src = _read(sample.c_path)
    rust_src = _read(sample.rust_path)
    return (
        FindingRecord(
            finding_id=FINDING_ID,
            sample_id=sample.sample_id,
            candidate_repo=sample.candidate_repo,
            candidate_url=cand.url,
            source_family=sample.source_family,
            divergence_class=sample.divergence_class,
            target_lang="rust",
            evidence_tier="confirmed extraction-unit finding",
            upstream_status=(
                "upstream-instance source audit pending; this record must not be "
                f"cited as an upstream {sample.candidate_repo} CVE or confirmed repository defect"
            ),
            disclosure_status=(
                "maintainer-ready coordinated-disclosure draft generated; send "
                "only after auditing the upstream source for the same pattern"
            ),
            witness_input=witness_input(sample),
            safe_input=safe_input(sample),
            c_file=str(sample.c_path.relative_to(_ROOT)),
            rust_file=str(sample.rust_path.relative_to(_ROOT)),
            c_sha256=_sha256_text(c_src),
            rust_sha256=_sha256_text(rust_src),
            maintenance_snapshot=_MAINTENANCE_SNAPSHOTS[sample.candidate_repo],
            remediation=(
                "Accumulate byte counts in an unsigned or widened type, or guard the "
                "maximum representable value before translating to wrapping target arithmetic."
            ),
            note=sample.note,
        ),
    )


def _record_json(rec: FindingRecord) -> Dict[str, object]:
    return {
        "finding_id": rec.finding_id,
        "sample_id": rec.sample_id,
        "candidate_repo": rec.candidate_repo,
        "candidate_url": rec.candidate_url,
        "source_family": rec.source_family,
        "divergence_class": rec.divergence_class,
        "target_lang": rec.target_lang,
        "evidence_tier": rec.evidence_tier,
        "upstream_status": rec.upstream_status,
        "disclosure_status": rec.disclosure_status,
        "witness_input": list(rec.witness_input),
        "safe_input": list(rec.safe_input),
        "c_file": rec.c_file,
        "rust_file": rec.rust_file,
        "c_sha256": rec.c_sha256,
        "rust_sha256": rec.rust_sha256,
        "maintenance_snapshot": dict(rec.maintenance_snapshot),
        "remediation": rec.remediation,
        "note": rec.note,
    }


def content_hash(records: Optional[Sequence[FindingRecord]] = None) -> str:
    records = finding_records() if records is None else records
    stable = [_record_json(rec) for rec in records]
    return hashlib.sha256(_canonical_bytes(stable)).hexdigest()


def results_document() -> Dict[str, object]:
    records = finding_records()
    return {
        "schema": SCHEMA_VERSION,
        "content_hash": content_hash(records),
        "n_findings": len(records),
        "n_confirmed_extraction_unit_findings": len(records),
        "evidence_policy": (
            "Confirmed extraction-unit divergences are maintainer-ready findings, "
            "not upstream CVEs or repository-defect claims until source audit confirms "
            "the same pattern in the upstream project."
        ),
        "findings": [_record_json(rec) for rec in records],
    }


def write_results(path: Path = RESULTS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results_document(), indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def check_results(path: Path = RESULTS_PATH) -> Tuple[bool, str]:
    regenerated = json.dumps(results_document(), indent=2, sort_keys=True) + "\n"
    if not path.exists():
        return False, f"{path} is missing"
    if path.read_text(encoding="utf-8") != regenerated:
        return False, f"{path} does not match regenerated divergence findings"
    return True, "OK"


def _bundle_script(rec: FindingRecord, sample: miner.GitHubPortSample) -> str:
    return (
        "#!/usr/bin/env bash\n"
        f"# Reproduction for {rec.finding_id}: {rec.sample_id}\n"
        "# Evidence tier: confirmed extraction-unit finding, not an upstream CVE.\n"
        "set -euo pipefail\n"
        'WORK="$(mktemp -d)"\n'
        'trap \'rm -rf "$WORK"\' EXIT\n'
        "cat > \"$WORK/case.c\" <<'CEOF'\n"
        f"{_c_wrapper(sample).rstrip()}\n"
        "CEOF\n"
        "cat > \"$WORK/case.rs\" <<'REOF'\n"
        f"{_rust_wrapper(sample).rstrip()}\n"
        "REOF\n"
        'echo "== C extraction unit under UBSan on witness =="\n'
        'clang -O1 -fsanitize=undefined -fno-sanitize-recover=all '
        '"$WORK/case.c" -o "$WORK/c_bin"\n'
        f'"$WORK/c_bin" {" ".join(rec.witness_input)} || '
        'echo "  (C trapped / nonzero: UB is reachable)"\n'
        'echo "== Rust extraction unit on the same witness =="\n'
        'rustc -O "$WORK/case.rs" -o "$WORK/rs_bin"\n'
        f'"$WORK/rs_bin" {" ".join(rec.witness_input)} || '
        'echo "  (Rust nonzero exit is checked by the target semantics pack)"\n'
        'echo "== Safe input control =="\n'
        f'"$WORK/c_bin" {" ".join(rec.safe_input)}\n'
        f'"$WORK/rs_bin" {" ".join(rec.safe_input)}\n'
    )


def reproduce_finding(
    rec: FindingRecord,
    *,
    harness: Optional[ReexecHarness] = None,
    write_bundle: bool = True,
) -> FindingConfirmation:
    sample = _sample()
    status = toolchain_available()
    if not status.full_for(rec.target_lang):
        return FindingConfirmation(
            rec.finding_id,
            available=False,
            confirmed=False,
            ub_reachable=False,
            target_defined=False,
            safe_silent=False,
            detail=f"{rec.target_lang} toolchain absent",
        )
    h = harness or ReexecHarness(status)
    ub: ReexecResult = h.confirm_trap_vs_defined(
        _c_wrapper(sample),
        _rust_wrapper(sample),
        list(rec.witness_input),
        rec.divergence_class,
        rec.target_lang,
    )
    safe: ReexecResult = h.confirm_trap_vs_defined(
        _c_wrapper(sample),
        _rust_wrapper(sample),
        list(rec.safe_input),
        rec.divergence_class,
        rec.target_lang,
    )
    bundle_path = ""
    if write_bundle:
        _REPRO.mkdir(parents=True, exist_ok=True)
        out = _REPRO / f"{rec.finding_id}.sh"
        out.write_text(_bundle_script(rec, sample), encoding="utf-8")
        out.chmod(0o755)
        bundle_path = str(out.relative_to(_ROOT))
    safe_silent = bool(safe.available and not safe.ub_reachable and safe.rust_defined and not safe.confirmed)
    confirmed = bool(ub.confirmed and ub.ub_reachable and ub.rust_defined and safe_silent)
    return FindingConfirmation(
        rec.finding_id,
        available=True,
        confirmed=confirmed,
        ub_reachable=bool(ub.ub_reachable),
        target_defined=bool(ub.rust_defined),
        safe_silent=safe_silent,
        bundle_path=bundle_path,
        detail=ub.reason if confirmed else f"witness={ub.reason}; safe={safe.reason}",
    )


def confirm_findings() -> FindingsReport:
    records = finding_records()
    status = toolchain_available()
    if not all(status.full_for(rec.target_lang) for rec in records):
        return FindingsReport(
            available=False,
            ok=True,
            n_records=len(records),
            n_confirmed=0,
            bundles_valid=True,
            detail="target toolchain(s) absent; findings defined but not reproduced",
        )
    h = ReexecHarness(status)
    confirmations: List[FindingConfirmation] = []
    bundles_valid = True
    for rec in records:
        conf = reproduce_finding(rec, harness=h, write_bundle=True)
        confirmations.append(conf)
        if conf.bundle_path:
            chk = subprocess.run(
                ["bash", "-n", str(_ROOT / conf.bundle_path)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            bundles_valid &= chk.returncode == 0
    n_confirmed = sum(1 for conf in confirmations if conf.confirmed)
    ok = n_confirmed == len(records) and bundles_valid
    return FindingsReport(
        available=True,
        ok=ok,
        n_records=len(records),
        n_confirmed=n_confirmed,
        bundles_valid=bundles_valid,
        confirmations=tuple(confirmations),
        detail="every extraction-unit finding reproduces live and bundles lint"
        if ok else "a finding failed live reproduction or bundle lint",
    )


def _markdown(records: Sequence[FindingRecord], rep: FindingsReport) -> str:
    by_id = {conf.finding_id: conf for conf in rep.confirmations}
    lines = [
        "# Divergence findings — responsible evidence tier",
        "",
        "This page is generated by `ub_oracle.divergence_findings`. It records "
        "newly confirmed divergence findings from actively maintained C->Rust "
        "port families while separating three evidence tiers:",
        "",
        "1. **Confirmed extraction-unit finding:** the checked-in minimized C and "
        "Rust functions compile and reproduce the divergence against real "
        "`clang`/UBSan and `rustc`.",
        "2. **Maintainer-ready disclosure draft:** the bundle is suitable for a "
        "coordinated report once the same pattern is audited in upstream source.",
        "3. **Upstream defect claim:** intentionally **not** made here until that "
        "source audit is complete.",
        "",
    ]
    for rec in records:
        conf = by_id.get(rec.finding_id)
        status = "reproduced live" if conf and conf.confirmed else "not reproduced in this environment"
        bundle = conf.bundle_path if conf and conf.bundle_path else f"docs/repro/{rec.finding_id}.sh"
        snap = rec.maintenance_snapshot
        lines += [
            f"## {rec.finding_id} — `{rec.sample_id}` ({rec.divergence_class})",
            "",
            f"- **Repository family:** [{rec.candidate_repo}]({rec.candidate_url})",
            f"- **Active-maintenance evidence:** {snap.get('active_evidence')} "
            f"(snapshot source: `{snap.get('source_command')}`; pushed at "
            f"`{snap.get('pushed_at')}`; archived = `{snap.get('archived')}`)",
            f"- **Evidence tier:** {rec.evidence_tier}",
            f"- **Upstream status:** {rec.upstream_status}",
            f"- **Disclosure status:** {rec.disclosure_status}",
            f"- **Live reproduction status:** {status}",
            f"- **Witness / safe input:** `{' '.join(rec.witness_input)}` / "
            f"`{' '.join(rec.safe_input)}`",
            f"- **Reproduction bundle:** `{bundle}`",
            f"- **Checked sources:** `{rec.c_file}` "
            f"(`{rec.c_sha256[:12]}`), `{rec.rust_file}` (`{rec.rust_sha256[:12]}`)",
            f"- **Remediation:** {rec.remediation}",
            "",
            "**Why it matters.** The C extraction unit executes undefined behavior "
            "when the byte counter crosses the signed 32-bit limit, while the Rust "
            "extraction unit uses explicit wrapping arithmetic and returns a "
            "deterministic value on the same input. A port that preserves this "
            "shape has made a latent C signed-overflow precondition into defined "
            "target behavior; the fix is to make the counter range and overflow "
            "policy explicit before translation.",
            "",
        ]
    lines.append(f"Content hash: `{content_hash(records)[:16]}`.")
    lines.append("")
    return "\n".join(lines)


def generate_findings() -> Tuple[Path, Path, FindingsReport]:
    records = finding_records()
    rep = confirm_findings()
    write_results()
    _DOCS.mkdir(parents=True, exist_ok=True)
    _DOC_PATH.write_text(_markdown(records, rep), encoding="utf-8")
    bundle_path = _REPRO / f"{records[0].finding_id}.sh"
    if not bundle_path.exists():
        _REPRO.mkdir(parents=True, exist_ok=True)
        bundle_path.write_text(_bundle_script(records[0], _sample()), encoding="utf-8")
        bundle_path.chmod(0o755)
    return _DOC_PATH, bundle_path, rep


if __name__ == "__main__":  # pragma: no cover
    doc, bundle, rep = generate_findings()
    print(
        f"divergence-findings available={rep.available} ok={rep.ok} "
        f"confirmed={rep.n_confirmed}/{rep.n_records} bundles_valid={rep.bundles_valid}"
    )
    print(f"wrote {RESULTS_PATH}, {doc}, and {bundle}")
