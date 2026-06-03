"""External pre-review and open-artifact release gate.

Step 94 is intentionally operational: before submission, PL and security
reviewers should be able to red-team the draft while the artifact is already
open, reproducible, and mapped to concrete evidence.  This module turns the
repository-side part of that workflow into a deterministic packet:

* reviewer lanes for PL soundness, security/UB, systems artifact, and empirical
  methods;
* every lane tied to real in-repo evidence paths and runnable commands;
* release checklist items for the open-source artifact state; and
* a stable packet hash so the packet can be diffed during pre-review.

No live compiler availability is folded into the packet hash.  The gate uses
only file metadata and fast registry/doc checks, so it is suitable for a release
check and cannot pass because of environment-specific compiler luck.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from . import artifact_eval, claims_audit, soundness_gate, traceability

SCHEMA_VERSION = "pre-review/v1"

_ROOT = Path(__file__).resolve().parents[2]
_DOC_PATH = _ROOT / "docs" / "PRE_REVIEW.md"
_MAKEFILE = _ROOT / "Makefile"

_REQUIRED_CHECKLIST = {
    "open-license",
    "public-citation",
    "artifact-appendix",
    "reproducibility-kit",
    "paper-sources",
    "claims-audit",
    "soundness-registry",
    "review-lanes",
}


@dataclass(frozen=True)
class EvidenceItem:
    id: str
    title: str
    path: str
    kind: str
    command: str = ""
    summary: str = ""

    def as_dict(self) -> Dict[str, str]:
        return {
            "id": self.id,
            "title": self.title,
            "path": self.path,
            "kind": self.kind,
            "command": self.command,
            "summary": self.summary,
        }


@dataclass(frozen=True)
class ReviewerLane:
    id: str
    title: str
    expertise: str
    questions: Tuple[str, ...]
    evidence_ids: Tuple[str, ...]

    def as_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "title": self.title,
            "expertise": self.expertise,
            "questions": list(self.questions),
            "evidence_ids": list(self.evidence_ids),
        }


@dataclass(frozen=True)
class ReleaseChecklistItem:
    id: str
    title: str
    passed: bool
    evidence_ids: Tuple[str, ...]
    detail: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "title": self.title,
            "passed": self.passed,
            "evidence_ids": list(self.evidence_ids),
            "detail": self.detail,
        }


@dataclass(frozen=True)
class PreReviewPacket:
    schema: str
    claims_ok: bool
    soundness_ok: bool
    soundness_registered: int
    artifact_available_ok: bool
    traceability_claims: int
    evidence: Tuple[EvidenceItem, ...]
    reviewer_lanes: Tuple[ReviewerLane, ...]
    release_checklist: Tuple[ReleaseChecklistItem, ...]
    notes: Tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> Dict[str, object]:
        return {
            "schema": self.schema,
            "claims_ok": self.claims_ok,
            "soundness_ok": self.soundness_ok,
            "soundness_registered": self.soundness_registered,
            "artifact_available_ok": self.artifact_available_ok,
            "traceability_claims": self.traceability_claims,
            "evidence": [e.as_dict() for e in sorted(self.evidence, key=lambda x: x.id)],
            "reviewer_lanes": [
                lane.as_dict() for lane in sorted(self.reviewer_lanes, key=lambda x: x.id)
            ],
            "release_checklist": [
                item.as_dict()
                for item in sorted(self.release_checklist, key=lambda x: x.id)
            ],
            "notes": list(self.notes),
        }

    @property
    def content_hash(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PreReviewCheck:
    ok: bool
    packet_hash: str
    packet: PreReviewPacket
    problems: Tuple[str, ...] = ()

    @property
    def detail(self) -> str:
        if self.ok:
            return (
                f"pre-review packet ok: {len(self.packet.evidence)} evidence item(s), "
                f"{len(self.packet.reviewer_lanes)} reviewer lane(s), "
                f"{len(self.packet.release_checklist)} release check(s), "
                f"hash={self.packet_hash[:12]}"
            )
        return "\n".join(self.problems)


def _rel(path: Path) -> str:
    return path.relative_to(_ROOT).as_posix()


def _exists(rel: str) -> bool:
    return (_ROOT / rel).exists()


def _read(rel: str) -> str:
    try:
        return (_ROOT / rel).read_text(encoding="utf-8")
    except OSError:
        return ""


def _make_targets() -> set[str]:
    targets: set[str] = set()
    text = _read("Makefile")
    for line in text.splitlines():
        if line.startswith("\t") or not line or line.startswith("."):
            continue
        m = re.match(r"^([A-Za-z0-9_.-]+)\s*:", line)
        if m:
            targets.add(m.group(1))
    return targets


def _command_problem(command: str, targets: set[str]) -> Optional[str]:
    if not command:
        return None
    parts = shlex.split(command)
    if not parts:
        return "empty command"

    if parts[0] == "make":
        requested = [
            p for p in parts[1:]
            if not p.startswith("-") and "=" not in p
        ]
        missing = [p for p in requested if p not in targets]
        if missing:
            return f"make target(s) do not exist: {missing}"
        return None

    if parts[0] in {"python", "python3"} and len(parts) >= 3 and parts[1] == "-m":
        mod = parts[2]
        bits = mod.split(".")
        if bits[0] == "src":
            rel = Path(*bits).with_suffix(".py")
        else:
            rel = Path("src", *bits).with_suffix(".py")
        if not (_ROOT / rel).exists():
            return f"python module does not exist: {mod}"
        return None

    if parts[0] == "bash" and len(parts) >= 2:
        script = _ROOT / parts[1]
        if not script.exists():
            return f"bash script does not exist: {parts[1]}"
        return None

    if parts[0] == "pdflatex":
        tex_args = [p for p in parts[1:] if p.endswith(".tex")]
        if not tex_args:
            return "pdflatex command names no .tex source"
        missing = [p for p in tex_args if not (_ROOT / p).exists()]
        if missing:
            return f"pdflatex source(s) missing: {missing}"
        return None

    return f"unsupported evidence command prefix: {parts[0]}"


def _evidence_items() -> Tuple[EvidenceItem, ...]:
    return (
        EvidenceItem(
            "paper-source",
            "Submission draft source",
            "tool_paper.tex",
            "paper",
            "pdflatex -interaction=nonstopmode -halt-on-error tool_paper.tex",
            "LaTeX source for the tool paper, built into the review PDF.",
        ),
        EvidenceItem(
            "paper-pdf",
            "Submission draft PDF",
            "tool_paper.pdf",
            "paper",
            "",
            "Compiled PDF for external pre-review.",
        ),
        EvidenceItem(
            "artifact-appendix",
            "ACM artifact-evaluation appendix",
            "docs/ARTIFACT.md",
            "artifact",
            "python -m src.ub_oracle.artifact_eval",
            "Badge criteria and release evidence for available/functional/reproduced.",
        ),
        EvidenceItem(
            "traceability",
            "Claim-to-code traceability matrix",
            "docs/TRACEABILITY.md",
            "soundness",
            "python -m src.ub_oracle.traceability",
            "Every paper/README claim maps to importable code symbols and checks.",
        ),
        EvidenceItem(
            "soundness-compendium",
            "Oracle soundness compendium",
            "docs/SOUNDNESS_COMPENDIUM.md",
            "soundness",
            "make soundness-check",
            "Registered oracles, theorem references, and witness units.",
        ),
        EvidenceItem(
            "mechanized-soundness",
            "Lean mechanized soundness notes",
            "docs/MECHANIZED_SOUNDNESS.md",
            "soundness",
            "make verified-check",
            "Mechanized product-program and checker evidence.",
        ),
        EvidenceItem(
            "reproduction-kit",
            "External replication kit",
            "scripts/reproduce_kit.sh",
            "reproducibility",
            "bash scripts/reproduce_kit.sh",
            "One-command reproduction path used by artifact evaluators.",
        ),
        EvidenceItem(
            "dockerfile",
            "Hermetic artifact container",
            "Dockerfile",
            "reproducibility",
            "make docker-build",
            "Containerized environment for the open artifact.",
        ),
        EvidenceItem(
            "figures",
            "Generated paper figures",
            "docs/figures.md",
            "evaluation",
            "python -m src.ub_oracle.figures",
            "Paper figures generated from in-repo data.",
        ),
        EvidenceItem(
            "disclosures",
            "Responsible-disclosure exemplars",
            "docs/disclosures.md",
            "security",
            "python -m src.ub_oracle.disclosure",
            "Maintainer-facing disclosure records and reproductions.",
        ),
        EvidenceItem(
            "divergence-findings",
            "Evidence-tiered divergence findings",
            "docs/divergence_findings.md",
            "security",
            "python -m src.ub_oracle.divergence_findings",
            "Compiler-confirmed extraction-unit findings with upstream-claim boundaries.",
        ),
        EvidenceItem(
            "zoo",
            "Divergence zoo",
            "docs/zoo.md",
            "security",
            "python -m src.ub_oracle.divergence_zoo",
            "Indexed catalogue of reproducible cross-language divergence exhibits.",
        ),
        EvidenceItem(
            "positioning",
            "Adjacent-work positioning",
            "docs/POSITIONING.md",
            "related-work",
            "",
            "Scope and comparison notes for external reviewers.",
        ),
        EvidenceItem(
            "plugin-sdk",
            "Plugin SDK",
            "docs/SDK.md",
            "ecosystem",
            "",
            "Community extension surface for new languages and divergence classes.",
        ),
        EvidenceItem(
            "pre-review-check",
            "Pre-review packet gate",
            "src/ub_oracle/pre_review.py",
            "release",
            "make pre-review-check",
            "Deterministic gate that validates this packet.",
        ),
    )


def _reviewer_lanes() -> Tuple[ReviewerLane, ...]:
    return (
        ReviewerLane(
            "pl-soundness",
            "PL / formal-methods soundness review",
            "Language semantics, product programs, mechanized proof boundaries.",
            (
                "Does the divergence definition match the source/target semantics claimed?",
                "Are source-definedness, target-definedness, and consequence separated?",
                "Are UNKNOWN/ABSTAIN frontiers loud enough to preserve one-sided soundness?",
                "Do Lean/Coq checker claims line up with the executable oracle?",
            ),
            (
                "paper-source",
                "traceability",
                "soundness-compendium",
                "mechanized-soundness",
            ),
        ),
        ReviewerLane(
            "security-ub",
            "Security / undefined-behavior red team",
            "C UB exploitation, sanitizer evidence, disclosure quality.",
            (
                "Are the flagship UB classes security-relevant and not overclaimed?",
                "Can a maintainer reproduce a witness from the disclosure materials?",
                "Do sanitizer and checked-contract paths cover the claimed bug class?",
                "Are non-CVE weakness replays labelled honestly?",
            ),
            (
                "disclosures",
                "zoo",
                "soundness-compendium",
                "artifact-appendix",
            ),
        ),
        ReviewerLane(
            "systems-artifact",
            "Systems / artifact-evaluation dry run",
            "Open-source packaging, reproducibility, containerized execution.",
            (
                "Can a fresh evaluator identify the exact commands to run?",
                "Does the artifact state satisfy open licence, citation, and packaging checks?",
                "Are reproduction commands real repository entry points?",
                "Is the paper PDF built from checked-in source?",
            ),
            (
                "artifact-appendix",
                "reproduction-kit",
                "dockerfile",
                "paper-source",
                "paper-pdf",
            ),
        ),
        ReviewerLane(
            "empirical-methods",
            "Empirical-methods / benchmarks review",
            "Baselines, generated figures, statistics, and threats to validity.",
            (
                "Are paper figures generated from real checked-in data?",
                "Are baseline comparisons scoped to tools that can ingest the pair?",
                "Do claims-audit checks keep the public prose within proven scope?",
                "Are negative results and abstentions reported alongside wins?",
            ),
            (
                "figures",
                "positioning",
                "traceability",
                "paper-source",
            ),
        ),
    )


def _release_checklist(
    *,
    claims_ok: bool,
    soundness_ok: bool,
    artifact_available_ok: bool,
    evidence_ids: Iterable[str],
    lane_count: int,
) -> Tuple[ReleaseChecklistItem, ...]:
    evidence_set = set(evidence_ids)
    licence = _read("LICENSE")
    citation = _read("CITATION.cff")
    return (
        ReleaseChecklistItem(
            "open-license",
            "Open-source licence is present",
            "MIT License" in licence or "Permission is hereby granted" in licence,
            (),
            "LICENSE contains an OSI-style grant.",
        ),
        ReleaseChecklistItem(
            "public-citation",
            "Citation descriptor names the public repository",
            "github.com/thehalleyyoung/cross-lang-verifier" in citation,
            (),
            "CITATION.cff points at the public code repository.",
        ),
        ReleaseChecklistItem(
            "artifact-appendix",
            "Artifact-available badge criteria pass",
            artifact_available_ok,
            ("artifact-appendix",),
            "Availability checks are pure file/metadata checks.",
        ),
        ReleaseChecklistItem(
            "reproducibility-kit",
            "Reproduction kit and container entry points exist",
            {"reproduction-kit", "dockerfile"}.issubset(evidence_set),
            ("reproduction-kit", "dockerfile"),
            "A reviewer can choose shell or container entry points.",
        ),
        ReleaseChecklistItem(
            "paper-sources",
            "Paper source and compiled PDF are in the packet",
            {"paper-source", "paper-pdf"}.issubset(evidence_set),
            ("paper-source", "paper-pdf"),
            "The pre-review draft is rebuilt from tracked source.",
        ),
        ReleaseChecklistItem(
            "claims-audit",
            "Public claims are audited against live code metadata",
            claims_ok,
            ("traceability",),
            "The claims audit rejects unsupported language-pair/count claims.",
        ),
        ReleaseChecklistItem(
            "soundness-registry",
            "Registered oracle metadata has soundness statements",
            soundness_ok,
            ("soundness-compendium", "mechanized-soundness"),
            "Every registered oracle is covered by the soundness registry.",
        ),
        ReleaseChecklistItem(
            "review-lanes",
            "Independent reviewer lanes are populated",
            lane_count >= 4,
            tuple(sorted(evidence_set)),
            "PL, security, systems-artifact, and empirical review lanes are present.",
        ),
    )


def build_pre_review_packet() -> PreReviewPacket:
    claims = claims_audit.confirm_claims_audit()
    soundness = soundness_gate.confirm_soundness_registry(probe_witnesses=False)
    available = artifact_eval._check_available()  # availability-only; avoids live compiler work.

    evidence = _evidence_items()
    lanes = _reviewer_lanes()
    checklist = _release_checklist(
        claims_ok=claims.ok,
        soundness_ok=soundness.ok,
        artifact_available_ok=available.earned,
        evidence_ids=(e.id for e in evidence),
        lane_count=len(lanes),
    )
    return PreReviewPacket(
        schema=SCHEMA_VERSION,
        claims_ok=claims.ok,
        soundness_ok=soundness.ok,
        soundness_registered=len(soundness.registered),
        artifact_available_ok=available.earned,
        traceability_claims=len(traceability.CLAIMS),
        evidence=evidence,
        reviewer_lanes=lanes,
        release_checklist=checklist,
        notes=(
            "External reviewers receive the same evidence paths and commands the tests validate.",
            "The packet is toolchain-independent; live compiler confirmations remain in the cited gates.",
        ),
    )


def validate_packet(packet: PreReviewPacket) -> PreReviewCheck:
    problems: List[str] = []
    targets = _make_targets()

    if packet.schema != SCHEMA_VERSION:
        problems.append(f"schema mismatch: {packet.schema!r}")
    if not packet.claims_ok:
        problems.append("claims audit did not pass")
    if not packet.soundness_ok:
        problems.append("soundness registry did not pass")
    if packet.soundness_registered < 1:
        problems.append("soundness registry is empty")
    if not packet.artifact_available_ok:
        problems.append("artifact availability checks did not pass")
    if packet.traceability_claims < 1:
        problems.append("traceability claim registry is empty")

    ids: Dict[str, EvidenceItem] = {}
    for item in packet.evidence:
        if item.id in ids:
            problems.append(f"duplicate evidence id: {item.id}")
        ids[item.id] = item
        if not _exists(item.path):
            problems.append(f"missing evidence path for {item.id}: {item.path}")
        cmd_problem = _command_problem(item.command, targets)
        if cmd_problem is not None:
            problems.append(f"invalid evidence command for {item.id}: {cmd_problem}")

    if len(packet.reviewer_lanes) < 4:
        problems.append("fewer than four independent reviewer lanes")
    for lane in packet.reviewer_lanes:
        if len(lane.questions) < 3:
            problems.append(f"reviewer lane has too few questions: {lane.id}")
        if len(lane.evidence_ids) < 3:
            problems.append(f"reviewer lane has too little evidence: {lane.id}")
        missing = [eid for eid in lane.evidence_ids if eid not in ids]
        if missing:
            problems.append(f"reviewer lane {lane.id} references unknown evidence: {missing}")

    checklist_ids = {item.id for item in packet.release_checklist}
    missing_checklist = sorted(_REQUIRED_CHECKLIST - checklist_ids)
    if missing_checklist:
        problems.append(f"missing release checklist item(s): {missing_checklist}")
    for item in packet.release_checklist:
        if not item.passed:
            problems.append(f"release checklist failed: {item.id}: {item.detail}")
        unknown = [eid for eid in item.evidence_ids if eid not in ids]
        if unknown:
            problems.append(f"release checklist {item.id} references unknown evidence: {unknown}")

    return PreReviewCheck(
        ok=not problems,
        packet_hash=packet.content_hash,
        packet=packet,
        problems=tuple(problems),
    )


def confirm_pre_review_packet() -> PreReviewCheck:
    return validate_packet(build_pre_review_packet())


def _md_table(rows: Iterable[Sequence[str]]) -> str:
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def render_pre_review_markdown(packet: PreReviewPacket) -> str:
    check = validate_packet(packet)
    status = "PASS" if check.ok else "FAIL"
    evidence_rows = [
        ("ID", "Evidence", "Path", "Command"),
        ("---", "---", "---", "---"),
    ]
    for item in sorted(packet.evidence, key=lambda x: x.id):
        cmd = f"`{item.command}`" if item.command else "n/a"
        evidence_rows.append((
            f"`{item.id}`",
            item.title,
            f"`{item.path}`",
            cmd,
        ))

    checklist_rows = [
        ("Gate", "Status", "Evidence", "Detail"),
        ("---", "---", "---", "---"),
    ]
    for item in sorted(packet.release_checklist, key=lambda x: x.id):
        evidence = ", ".join(f"`{eid}`" for eid in item.evidence_ids) or "file metadata"
        checklist_rows.append((
            f"`{item.id}`",
            "PASS" if item.passed else "FAIL",
            evidence,
            item.detail,
        ))

    lines = [
        "# External pre-review packet",
        "",
        "This packet is the repository-side evidence bundle for domain-expert "
        "pre-review and simultaneous open-artifact release. It is generated by "
        "`src/ub_oracle/pre_review.py` and validated by `make pre-review-check`.",
        "",
        f"- **Schema:** `{packet.schema}`",
        f"- **Gate status:** **{status}**",
        f"- **Packet hash:** `{check.packet_hash}`",
        f"- **Registered oracle statements:** {packet.soundness_registered}",
        f"- **Traceability claims:** {packet.traceability_claims}",
        "",
        "## Reviewer lanes",
        "",
    ]
    for lane in sorted(packet.reviewer_lanes, key=lambda x: x.id):
        lines.extend([
            f"### {lane.title}",
            "",
            f"**Expertise:** {lane.expertise}",
            "",
            "**Red-team questions:**",
        ])
        lines.extend(f"1. {q}" for q in lane.questions)
        lines.extend([
            "",
            "**Evidence:** " + ", ".join(f"`{eid}`" for eid in lane.evidence_ids),
            "",
        ])

    lines.extend([
        "## Release checklist",
        "",
        _md_table(checklist_rows),
        "",
        "## Evidence manifest",
        "",
        _md_table(evidence_rows),
        "",
        "## Notes",
        "",
    ])
    lines.extend(f"- {note}" for note in packet.notes)
    if check.problems:
        lines.extend(["", "## Validation problems", ""])
        lines.extend(f"- {problem}" for problem in check.problems)
    lines.append("")
    return "\n".join(lines)


def write_pre_review_doc(path: Path | str = _DOC_PATH) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_pre_review_markdown(build_pre_review_packet()), encoding="utf-8")
    return out


PRE_REVIEW_SPI = {
    "build_pre_review_packet": build_pre_review_packet,
    "validate_packet": validate_packet,
    "confirm_pre_review_packet": confirm_pre_review_packet,
    "render_pre_review_markdown": render_pre_review_markdown,
    "write_pre_review_doc": write_pre_review_doc,
}


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the packet is invalid")
    parser.add_argument("--write-doc", action="store_true", help="write docs/PRE_REVIEW.md")
    args = parser.parse_args(argv)

    if args.write_doc:
        out = write_pre_review_doc()
        print(_rel(out))

    check = confirm_pre_review_packet()
    print(check.detail)
    if args.check and not check.ok:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_main())
