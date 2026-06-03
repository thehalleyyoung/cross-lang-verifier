"""Step 166 -- generated datasheet for every corpus-like artifact.

The datasheet is intentionally derived from the live corpus modules rather than
hand-maintained prose.  It documents provenance, population balance, validation
commands, real-code evidence, and limitations for the corpora and corpus-backed
benchmarks that support the paper claims.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from . import adversarial_corpus
from . import bug_regression_corpus
from . import c2rust_corpus
from . import cve_corpus
from . import divergence_findings
from . import divergence_zoo
from . import existing_tools_study
from . import github_port_miner
from . import idiomatic_corpus
from . import large_scale_study
from . import llm_scale_study
from . import multipair_corpus
from . import negative_corpus

SCHEMA_VERSION = "corpus-datasheet/v1"

_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = _ROOT / "experiments" / "corpus_datasheet"
RESULTS_PATH = EXPERIMENT_DIR / "corpus_datasheet.json"
DOC_PATH = _ROOT / "docs" / "corpus_datasheet.md"

DATASET_SCOPE = (
    "Included records are persistent corpora, generated corpus manifests, and "
    "corpus-backed benchmark populations.  Pure execution studies that do not "
    "own a subject population are referenced as validation commands instead of "
    "being double-counted as separate datasets."
)


def _canonical_bytes(obj: object) -> bytes:
    return json.dumps(
        obj, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_obj(obj: object) -> str:
    return hashlib.sha256(_canonical_bytes(obj)).hexdigest()


def _rel(path: Path) -> str:
    return str(path.relative_to(_ROOT))


def _counts(values: Iterable[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items()))


def _pairs_from_targets(source: str, targets: Iterable[str]) -> List[str]:
    return sorted(f"{source}->{target}" for target in targets)


def _stable_hash_short(payload: object) -> str:
    return _sha256_obj(payload)[:24]


@dataclass(frozen=True)
class DatasheetRecord:
    corpus_id: str
    title: str
    kind: str
    source_module: str
    artifact_paths: Tuple[str, ...]
    population_size: int
    unit: str
    language_pairs: Tuple[str, ...]
    divergence_classes: Tuple[str, ...]
    label_balance: Mapping[str, int]
    provenance: str
    construction: str
    validation_commands: Tuple[str, ...]
    real_code_evidence: str
    limitations: Tuple[str, ...]
    content_hash: str
    notes: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, object]:
        return {
            "corpus_id": self.corpus_id,
            "title": self.title,
            "kind": self.kind,
            "source_module": self.source_module,
            "artifact_paths": list(self.artifact_paths),
            "population_size": self.population_size,
            "unit": self.unit,
            "language_pairs": list(self.language_pairs),
            "divergence_classes": list(self.divergence_classes),
            "label_balance": dict(sorted(self.label_balance.items())),
            "provenance": self.provenance,
            "construction": self.construction,
            "validation_commands": list(self.validation_commands),
            "real_code_evidence": self.real_code_evidence,
            "limitations": list(self.limitations),
            "content_hash": self.content_hash,
            "notes": list(self.notes),
        }


def _c2rust_record() -> DatasheetRecord:
    doc = c2rust_corpus.results_document()
    return DatasheetRecord(
        corpus_id="c2rust-output",
        title="Tier-1 c2rust-output extraction corpus",
        kind="machine-translated corpus",
        source_module="ub_oracle.c2rust_corpus",
        artifact_paths=(
            _rel(c2rust_corpus.RESULTS_PATH),
            _rel(c2rust_corpus.SOURCE_DIR),
            _rel(c2rust_corpus.GENERATED_DIR),
        ),
        population_size=int(doc["n_items"]),
        unit="c2rust extraction unit",
        language_pairs=("c->rust",),
        divergence_classes=tuple(sorted(str(k) for k in doc["by_divergence_class"])),
        label_balance=dict(
            sorted((str(k), int(v)) for k, v in doc["by_symbolic_verdict"].items())
        ),
        provenance=(
            "Twelve real-library extraction families are transpiled by the actual "
            f"{doc['translator']} {doc['translator_version']} binary."
        ),
        construction=(
            "Checked-in C extraction units are regenerated into Rust with c2rust; "
            "the verifier records symbolic verdicts and source/target hashes."
        ),
        validation_commands=("make c2rust-corpus-check",),
        real_code_evidence=(
            "Focused tests compile every C source and generated Rust artifact; "
            "when c2rust is installed, checked-in Rust is byte-compared to fresh output."
        ),
        limitations=(
            "Extraction units are minimized families, not vendored full upstream trees.",
            "Compiler-backed regeneration requires a local c2rust binary.",
        ),
        content_hash=str(doc["content_hash"]),
    )


def _github_ports_record() -> DatasheetRecord:
    doc = github_port_miner.results_document()
    return DatasheetRecord(
        corpus_id="github-mined-ports",
        title="GitHub-mined active C-to-Rust port samples",
        kind="real-port extraction corpus",
        source_module="ub_oracle.github_port_miner",
        artifact_paths=(
            _rel(github_port_miner.RESULTS_PATH),
            _rel(github_port_miner.SAMPLE_DIR),
        ),
        population_size=int(doc["n_verified_samples"]),
        unit="mined port extraction sample",
        language_pairs=("c->rust",),
        divergence_classes=tuple(sorted(str(k) for k in doc["by_divergence_class"])),
        label_balance=dict(sorted((str(k), int(v)) for k, v in doc["by_symbolic_verdict"].items())),
        provenance=(
            "Seeded and optionally live-ranked public C-to-Rust port families "
            "including coreutils, sudo-rs, and zlib-rs."
        ),
        construction=(
            "Repository candidates are scored deterministically; checked-in "
            "extraction samples are verified through the same symbolic verdict layer."
        ),
        validation_commands=("make github-port-mining-check", "make real-bug-check"),
        real_code_evidence=(
            "The checked samples compile as real C/Rust code when toolchains are "
            "available; the divergent sudo/coreutils-style samples are "
            "compiler-confirmed through the verifier lane."
        ),
        limitations=(
            "Samples are extraction units derived from port families, not whole-repo audits.",
            "The live GitHub search path is optional and not part of byte-stable CI.",
        ),
        content_hash=str(doc["content_hash"]),
    )


def _findings_record() -> DatasheetRecord:
    records = divergence_findings.finding_records()
    return DatasheetRecord(
        corpus_id="responsible-findings",
        title="Evidence-tiered responsible divergence findings",
        kind="finding ledger",
        source_module="ub_oracle.divergence_findings",
        artifact_paths=(
            _rel(divergence_findings.RESULTS_PATH),
            "docs/divergence_findings.md",
            "docs/repro/",
        ),
        population_size=len(records),
        unit="responsible disclosure finding",
        language_pairs=tuple(sorted(f"c->{rec.target_lang}" for rec in records)),
        divergence_classes=tuple(sorted({rec.divergence_class for rec in records})),
        label_balance={"confirmed_extraction_unit": len(records)},
        provenance=(
            "Actively maintained public port families with maintainer-ready "
            "extraction-unit findings and explicit upstream-claim boundaries."
        ),
        construction=(
            "Symbolic witnesses are attached to frozen source hashes, safe controls, "
            "maintenance snapshots, and reproduction bundles."
        ),
        validation_commands=("make real-bug-check",),
        real_code_evidence=(
            "When clang/UBSan and rustc are available, every finding witness traps "
            "on C and is defined on the Rust target while the safe control stays silent."
        ),
        limitations=(
            "Findings are not cited as upstream CVEs or confirmed repository defects "
            "until a separate source audit confirms the same pattern upstream.",
        ),
        content_hash=_stable_hash_short(
            [
                {
                    "id": rec.finding_id,
                    "sample": rec.sample_id,
                    "class": rec.divergence_class,
                    "target": rec.target_lang,
                    "c_sha": rec.c_sha256,
                    "rust_sha": rec.rust_sha256,
                }
                for rec in records
            ]
        ),
    )


def _bug_regression_record() -> DatasheetRecord:
    doc = bug_regression_corpus.results_document()
    return DatasheetRecord(
        corpus_id="bug-regressions",
        title="Frozen regression corpus from caught divergences",
        kind="regression corpus",
        source_module="ub_oracle.bug_regression_corpus",
        artifact_paths=(
            _rel(bug_regression_corpus.RESULTS_PATH),
            _rel(bug_regression_corpus.DOC_PATH),
        ),
        population_size=int(doc["n_regressions"]),
        unit="frozen finding regression",
        language_pairs=tuple(sorted(f"c->{lang}" for lang in doc["by_target_lang"])),
        divergence_classes=tuple(sorted(str(k) for k in doc["by_divergence_class"])),
        label_balance={"regression": int(doc["n_regressions"])},
        provenance=str(doc["evidence_policy"]),
        construction=(
            "Every responsible finding is mirrored into a regression entry with "
            "source hashes, witness input, safe input, and a reproduction bundle."
        ),
        validation_commands=("make bug-regression-check",),
        real_code_evidence=(
            "The gate lints bundles, checks frozen source hashes, and replays the "
            "live witness when clang/UBSan plus the target compiler are available."
        ),
        limitations=(
            "The corpus size follows the number of responsibly recorded findings.",
            "It inherits the upstream-claim restrictions from the findings ledger.",
        ),
        content_hash=str(doc["content_hash"]),
    )


def _idiomatic_record() -> DatasheetRecord:
    items = tuple(idiomatic_corpus.CORPUS)
    pairs = sorted({f"c->{lang}" for item in items for lang in item.targets})
    class_counts = _counts(item.klass for item in items)
    target_units = sum(len(item.targets) for item in items)
    return DatasheetRecord(
        corpus_id="idiomatic-ports",
        title="Human-idiomatic port corpus",
        kind="idiomatic port corpus",
        source_module="ub_oracle.idiomatic_corpus",
        artifact_paths=("src/ub_oracle/idiomatic_corpus.py",),
        population_size=target_units,
        unit="source function x target language",
        language_pairs=tuple(pairs),
        divergence_classes=tuple(sorted(class_counts)),
        label_balance=_counts(
            item.declared_label
            for item in items
            for _lang in item.targets
        ),
        provenance=(
            "Real-world-shaped human ports, including coreutils/sudo-rs/zlib-rs "
            "classes, with non-literal Rust and Go translations."
        ),
        construction=(
            "Each item carries C source, idiomatic targets, a UB-triggering input, "
            "and a safe input; Step 161 expands the real-port family coverage."
        ),
        validation_commands=("make idiomatic-port-check",),
        real_code_evidence=(
            "Focused tests compile and replay Step 161 items through real "
            "clang/UBSan, rustc, and go when available."
        ),
        limitations=(
            "Functions are extraction units, not whole applications.",
            "The corpus emphasizes integer/runtime-contract divergences and true controls.",
        ),
        content_hash=_stable_hash_short(
            [
                {
                    "id": item.item_id,
                    "targets": sorted(item.targets),
                    "label": item.declared_label,
                    "klass": item.klass,
                }
                for item in items
            ]
        ),
        notes=("Step 161 families: " + ", ".join(idiomatic_corpus.STEP161_FAMILY_IDS),),
    )


def _llm_record() -> DatasheetRecord:
    items = llm_scale_study.generate_corpus()
    census = llm_scale_study.corpus_census(items)
    return DatasheetRecord(
        corpus_id="llm-translations",
        title="Frozen LLM-transpiler scale corpus",
        kind="LLM translation corpus",
        source_module="ub_oracle.llm_scale_study",
        artifact_paths=("src/ub_oracle/llm_scale_study.py",),
        population_size=int(census["n_items"]),
        unit="frozen C-to-Rust translation",
        language_pairs=("c->rust",),
        divergence_classes=tuple(sorted(str(k) for k in census["by_class"])),
        label_balance=dict(sorted((str(k), int(v)) for k, v in census["by_label"].items())),
        provenance=(
            "Frozen GitHub Copilot CLI/GPT-5.5 translations derived from the "
            "Tier-1 c2rust real-library extraction families."
        ),
        construction=(
            "Deterministic fixtures retain prompt/model provenance and are labeled "
            "only through real clang/UBSan plus rustc execution in the study gate."
        ),
        validation_commands=("make llm-scale-check",),
        real_code_evidence=(
            "The focused gate checks corpus size, hash stability, Wilson intervals, "
            "and a seeded live real-compiler sample."
        ),
        limitations=(
            "The translations are frozen fixtures, not an online model benchmark.",
            "Reported live metrics depend on local compiler availability.",
        ),
        content_hash=_stable_hash_short(sorted(item.content_hash for item in items)),
    )


def _historical_cve_record() -> DatasheetRecord:
    cases = tuple(cve_corpus.historical_cve_cases())
    templates = tuple(cve_corpus.HISTORICAL_TEMPLATES)
    template_by_id = {template.template_id: template for template in templates}
    langs = sorted({lang for template in templates for lang, _src in template.targets})
    return DatasheetRecord(
        corpus_id="historical-cve-replays",
        title="Historical CVE weakness-class replay corpus",
        kind="weakness replay corpus",
        source_module="ub_oracle.cve_corpus",
        artifact_paths=("src/ub_oracle/cve_corpus.py",),
        population_size=len(cases),
        unit="CVE-tagged weakness replay",
        language_pairs=tuple(_pairs_from_targets("c", langs)),
        divergence_classes=tuple(
            sorted({template_by_id[case.replay_template].divergence_class for case in cases})
        ),
        label_balance=_counts(template_by_id[case.replay_template].cwe for case in cases),
        provenance=(
            "CVE IDs and NVD CWE families are used as weakness-class provenance; "
            "replays are minimized examples, not original vulnerable vendor source."
        ),
        construction=(
            "Each historical CVE entry points to a replay template, NVD CWE query, "
            "per-language target source, witness inputs, and safe controls."
        ),
        validation_commands=("make historical-cve-check",),
        real_code_evidence=(
            "Representative templates and bundles are compiled and replayed against "
            "real clang/UBSan plus rustc/go when those toolchains are available."
        ),
        limitations=(
            "The corpus reproduces weakness classes associated with CVEs; it is "
            "not original vulnerable vendor source or vendored third-party source.",
        ),
        content_hash=_stable_hash_short(
            [
                {
                    "cve": case.cve_id,
                    "template": case.replay_template,
                    "cwe": case.nvd_cwe,
                    "scope": case.replay_scope,
                }
                for case in cases
            ]
        ),
    )


def _negative_record() -> DatasheetRecord:
    doc = negative_corpus.results_document()
    census = doc["census"]
    return DatasheetRecord(
        corpus_id="negative-true-equivalence",
        title="Verified-equivalent negative corpus",
        kind="negative corpus",
        source_module="ub_oracle.negative_corpus",
        artifact_paths=(
            _rel(negative_corpus.RESULTS_PATH),
            "docs/negative_corpus.md",
        ),
        population_size=int(census["n_items"]),
        unit="true-equivalence port",
        language_pairs=tuple(sorted(f"c->{lang}" for lang in census["by_target_lang"])),
        divergence_classes=tuple(sorted(str(k) for k in census["by_family"])),
        label_balance=dict(sorted((str(k), int(v)) for k, v in census["by_target_lang"].items())),
        provenance=(
            "Deterministically generated C-to-Rust/C-to-Go safe integer ports "
            "with declared operating ranges."
        ),
        construction=(
            "All items are covered by registered oracles, range-pruned, and checked "
            "against bounded proof inputs to bound false-positive behavior."
        ),
        validation_commands=("make negative-corpus-check",),
        real_code_evidence=(
            "A seeded sample is labeled equivalent by real clang/UBSan plus rustc/go "
            "when the toolchains are available."
        ),
        limitations=(
            "The corpus bounds false positives over its generated safe families; "
            "it is not a proof of arbitrary-program equivalence.",
        ),
        content_hash=str(doc["content_hash"]),
    )


def _adversarial_record() -> DatasheetRecord:
    doc = adversarial_corpus.results_document()
    census = doc["census"]
    return DatasheetRecord(
        corpus_id="adversarial-near-misses",
        title="Hand-crafted adversarial near-miss corpus",
        kind="adversarial corpus",
        source_module="ub_oracle.adversarial_corpus",
        artifact_paths=(
            _rel(adversarial_corpus.RESULTS_PATH),
            "docs/adversarial_corpus.md",
        ),
        population_size=int(census["n_cases"]),
        unit="near-miss verifier case",
        language_pairs=tuple(sorted(str(k) for k in census["by_pair"])),
        divergence_classes=tuple(sorted(str(k) for k in census["by_family"])),
        label_balance=dict(sorted((str(k), int(v)) for k, v in census["by_policy"].items())),
        provenance=(
            "Hand-crafted boundary cases from internal red-team pressure points: "
            "divergent controls, safe one-edit twins, and loud abstentions."
        ),
        construction=(
            "The committed manifest uses deterministic no-toolchain verdicts; live "
            "compiler replay separately ensures divergent controls do not degrade "
            "to safe or uncovered verdicts."
        ),
        validation_commands=("make adversarial-corpus-check",),
        real_code_evidence=(
            "When clang/UBSan and a target compiler are available, divergent "
            "controls are replayed and must remain flagged."
        ),
        limitations=(
            "Small by design; it stress-tests boundaries rather than estimating population rates.",
            "The byte-stable manifest intentionally excludes host-dependent live replay output.",
        ),
        content_hash=str(doc["content_hash"]),
    )


def _multipair_record() -> DatasheetRecord:
    items = tuple(multipair_corpus.CORPUS)
    target_units = sum(len(item.targets) for item in items)
    return DatasheetRecord(
        corpus_id="multi-pair-translations",
        title="Tier-3 multi-pair translation corpus",
        kind="multi-pair corpus",
        source_module="ub_oracle.multipair_corpus",
        artifact_paths=("src/ub_oracle/multipair_corpus.py",),
        population_size=target_units,
        unit="source function x target language",
        language_pairs=tuple(sorted({f"c->{lang}" for item in items for lang in item.targets})),
        divergence_classes=tuple(sorted({item.klass for item in items})),
        label_balance=_counts(
            item.declared_label
            for item in items
            for _lang in item.targets
        ),
        provenance=(
            "C functions translated into Rust, Go, and Swift in deliberately varied "
            "transpiler/LLM-style forms to test cross-pair invariance."
        ),
        construction=(
            "Divergent functions must flag on every available target; equivalent "
            "functions must stay silent across all target pairs."
        ),
        validation_commands=("python3 -m src.ub_oracle.multipair_corpus",),
        real_code_evidence=(
            "Live confirmation compiles and runs clang/UBSan plus every available "
            "target compiler and requires a stable verdict hash across runs."
        ),
        limitations=(
            "Swift coverage is opportunistic and depends on local toolchain availability.",
            "The corpus is a focused generality check rather than a large-scale scrape.",
        ),
        content_hash=_stable_hash_short(
            [
                {
                    "id": item.func_id,
                    "targets": sorted(item.targets),
                    "label": item.declared_label,
                    "klass": item.klass,
                }
                for item in items
            ]
        ),
    )


def _divergence_zoo_record() -> DatasheetRecord:
    exhibits = tuple(divergence_zoo.EXHIBITS)
    return DatasheetRecord(
        corpus_id="divergence-zoo",
        title="Machine-readable divergence zoo",
        kind="catalogue corpus",
        source_module="ub_oracle.divergence_zoo",
        artifact_paths=("docs/zoo.md",),
        population_size=len(exhibits),
        unit="zoo exhibit",
        language_pairs=tuple(sorted({exhibit.pair for exhibit in exhibits})),
        divergence_classes=tuple(sorted({exhibit.divergence_class for exhibit in exhibits})),
        label_balance=_counts(exhibit.declared_label for exhibit in exhibits),
        provenance=(
            "Generated from the live idiomatic and multi-pair corpora, preserving "
            "each exhibit's source, translation, witness, safe input, and provenance."
        ),
        construction=(
            "The zoo indexes divergent exhibits by class and pair and regenerates "
            "docs/zoo.md from corpus data."
        ),
        validation_commands=("python3 -m src.ub_oracle.divergence_zoo",),
        real_code_evidence=(
            "confirm_zoo re-runs every available divergent witness and safe input "
            "through the real compiler-backed harness."
        ),
        limitations=(
            "The zoo is a curated reference index, not a statistically sampled benchmark.",
            "Live reconfirmation is consistency-only when no target toolchain is present.",
        ),
        content_hash=divergence_zoo.content_hash(),
    )


def _large_scale_record() -> DatasheetRecord:
    items = large_scale_study.generate_corpus()
    census = large_scale_study.corpus_census(items)
    return DatasheetRecord(
        corpus_id="large-scale-generated",
        title="1M-LOC generated migration stress corpus",
        kind="large-scale stress corpus",
        source_module="ub_oracle.large_scale_study",
        artifact_paths=("src/ub_oracle/large_scale_study.py",),
        population_size=int(census["n_items"]),
        unit="generated translation pair",
        language_pairs=tuple(sorted(f"c->{lang}" for lang in census["pairs"])),
        divergence_classes=tuple(sorted(str(k) for k in census["by_class"])),
        label_balance=dict(sorted((str(k), int(v)) for k, v in census["by_label"].items())),
        provenance=(
            "Mechanically generated, distinct, compilable C-to-Rust/C-to-Go "
            "programs over catalogued divergence and safe-control families."
        ),
        construction=(
            "The full corpus census counts exact LOC, distinct-program hashes, "
            "label balance, pair balance, and class balance; live runs execute "
            "a deterministic seeded sample."
        ),
        validation_commands=("make large-scale",),
        real_code_evidence=(
            "The live sample is labeled by real clang/UBSan plus rustc/go when "
            "available, with a reproducible verdict-layer hash for a fixed seed."
        ),
        limitations=(
            "The scale population is generated rather than mined from third-party repositories.",
            "CI normally samples the population instead of compiling every item.",
        ),
        content_hash=_stable_hash_short(census),
        notes=(f"total_loc={census['total_loc']}",),
    )


def _existing_tools_record() -> DatasheetRecord:
    subjects = existing_tools_study.build_subjects()
    classes = _counts(subject.divergence_class for subject in subjects)
    expected = {
        "expected_divergent": sum(1 for subject in subjects if subject.expected_divergent),
        "safe_control": sum(1 for subject in subjects if not subject.expected_divergent),
    }
    return DatasheetRecord(
        corpus_id="existing-tools-head-to-head",
        title="Same-corpus existing-tools benchmark",
        kind="benchmark over c2rust-output corpus",
        source_module="ub_oracle.existing_tools_study",
        artifact_paths=("src/ub_oracle/existing_tools_study.py",),
        population_size=len(subjects),
        unit="c2rust subject reused for baseline comparison",
        language_pairs=("c->rust",),
        divergence_classes=tuple(sorted(classes)),
        label_balance=expected,
        provenance=(
            "Benchmark subjects are the checked Tier-1 c2rust-output corpus; this "
            "record is labeled as a benchmark to avoid double-counting the c2rust population."
        ),
        construction=(
            "Each subject is wrapped into runnable C/Rust programs and evaluated "
            "against SemRec, c2rust-style tests, optional Miri, and equal-budget fuzzing."
        ),
        validation_commands=("make existing-tools-check",),
        real_code_evidence=(
            "The focused gate confirms SemRec on the real c2rust subjects and runs "
            "the baseline checks with the same corpus inputs when toolchains exist."
        ),
        limitations=(
            "Miri is Rust-only and is recorded as structural applicability, not as "
            "a cross-language oracle.",
            "This benchmark reuses c2rust subjects and should not be counted as an independent corpus.",
        ),
        content_hash=_stable_hash_short(
            [
                {
                    "id": subject.item_id,
                    "class": subject.divergence_class,
                    "expected": subject.expected_symbolic_verdict,
                    "c_sha": subject.c_sha256,
                    "rust_sha": subject.rust_sha256,
                }
                for subject in subjects
            ]
        ),
    )


def records() -> Tuple[DatasheetRecord, ...]:
    """Return all corpus datasheet records in deterministic order."""

    recs = (
        _adversarial_record(),
        _bug_regression_record(),
        _c2rust_record(),
        _divergence_zoo_record(),
        _existing_tools_record(),
        _findings_record(),
        _github_ports_record(),
        _historical_cve_record(),
        _idiomatic_record(),
        _large_scale_record(),
        _llm_record(),
        _multipair_record(),
        _negative_record(),
    )
    return tuple(sorted(recs, key=lambda rec: rec.corpus_id))


def results_document(records_override: Optional[Sequence[DatasheetRecord]] = None) -> Dict[str, object]:
    recs = tuple(records() if records_override is None else records_override)
    record_dicts = [rec.to_dict() for rec in recs]
    aggregate_pairs = sorted({pair for rec in recs for pair in rec.language_pairs})
    aggregate_classes = sorted({klass for rec in recs for klass in rec.divergence_classes})
    by_kind = _counts(rec.kind for rec in recs)
    stable = {
        "schema": SCHEMA_VERSION,
        "scope": DATASET_SCOPE,
        "records": record_dicts,
    }
    return {
        "schema": SCHEMA_VERSION,
        "scope": DATASET_SCOPE,
        "content_hash": _sha256_obj(stable),
        "n_records": len(recs),
        "by_kind": by_kind,
        "language_pairs": aggregate_pairs,
        "divergence_classes": aggregate_classes,
        "records": record_dicts,
    }


def _record_from_doc(doc: Mapping[str, object], corpus_id: str) -> Mapping[str, object]:
    raw = doc.get("records")
    if not isinstance(raw, list):
        raise AssertionError("records must be a list")
    for record in raw:
        if isinstance(record, Mapping) and record.get("corpus_id") == corpus_id:
            return record
    raise AssertionError(f"missing datasheet record {corpus_id!r}")


def write_results(path: Path = RESULTS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(results_document(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_results(path: Path = RESULTS_PATH) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def check_results(path: Path = RESULTS_PATH) -> Tuple[bool, str]:
    if not path.exists():
        return False, f"{path} is missing"
    regenerated = json.dumps(results_document(), indent=2, sort_keys=True) + "\n"
    on_disk = path.read_text(encoding="utf-8")
    if on_disk != regenerated:
        return False, f"{path} does not match regenerated corpus datasheet"
    doc = json.loads(on_disk)
    if doc.get("schema") != SCHEMA_VERSION:
        return False, f"unexpected schema {doc.get('schema')!r}"
    for record in doc.get("records", []):
        if not isinstance(record, Mapping):
            return False, "record must be an object"
        required = (
            "corpus_id",
            "provenance",
            "construction",
            "validation_commands",
            "real_code_evidence",
            "limitations",
            "content_hash",
        )
        missing = [key for key in required if not record.get(key)]
        if missing:
            return False, f"{record.get('corpus_id', '<unknown>')}: missing {missing}"
    return True, "OK"


def _fmt_mapping(mapping: Mapping[str, object]) -> str:
    if not mapping:
        return "(none)"
    return ", ".join(f"`{key}`={mapping[key]}" for key in sorted(mapping))


def markdown_document(doc: Optional[Mapping[str, object]] = None) -> str:
    doc = results_document() if doc is None else doc
    lines: List[str] = [
        "# Corpus datasheet",
        "",
        "Generated by `ub_oracle.corpus_datasheet`; do not hand-edit the tables. "
        "Run `make corpus-datasheet-check` to assert this page and the JSON "
        "manifest are fresh.",
        "",
        DATASET_SCOPE,
        "",
        f"**Schema:** `{doc['schema']}`  ",
        f"**Content hash:** `{doc['content_hash']}`  ",
        f"**Records:** {doc['n_records']}  ",
        f"**Language pairs:** {len(doc['language_pairs'])}  ",
        f"**Divergence classes/families:** {len(doc['divergence_classes'])}",
        "",
        "| corpus | kind | population | pairs | balance | validation |",
        "|---|---:|---:|---|---|---|",
    ]
    for record in doc["records"]:
        pairs = ", ".join(f"`{pair}`" for pair in record["language_pairs"])
        validation = "<br>".join(f"`{cmd}`" for cmd in record["validation_commands"])
        lines.append(
            f"| `{record['corpus_id']}` | {record['kind']} | "
            f"{record['population_size']} {record['unit']} | {pairs} | "
            f"{_fmt_mapping(record['label_balance'])} | {validation} |"
        )
    lines.extend(["", "## Records", ""])
    for record in doc["records"]:
        lines.extend(
            [
                f"### `{record['corpus_id']}` -- {record['title']}",
                "",
                f"**Provenance.** {record['provenance']}",
                "",
                f"**Construction and balance.** {record['construction']} "
                f"Population: {record['population_size']} {record['unit']}; "
                f"labels/balance: {_fmt_mapping(record['label_balance'])}; "
                f"classes/families: {', '.join(f'`{klass}`' for klass in record['divergence_classes'])}.",
                "",
                f"**Validation.** {record['real_code_evidence']} Commands: "
                + ", ".join(f"`{cmd}`" for cmd in record["validation_commands"])
                + ".",
                "",
                "**Artifacts.** "
                + ", ".join(f"`{path}`" for path in record["artifact_paths"])
                + f". Hash: `{record['content_hash']}`.",
                "",
                "**Limitations.** "
                + " ".join(str(item) for item in record["limitations"]),
                "",
            ]
        )
        if record["notes"]:
            lines.extend(["**Notes.** " + " ".join(str(n) for n in record["notes"]), ""])
    return "\n".join(lines).rstrip() + "\n"


def write_markdown(path: Path = DOC_PATH) -> None:
    path.write_text(markdown_document(), encoding="utf-8")


def check_markdown(path: Path = DOC_PATH) -> Tuple[bool, str]:
    if not path.exists():
        return False, f"{path} is missing"
    expected = markdown_document()
    if path.read_text(encoding="utf-8") != expected:
        return False, f"{path} does not match regenerated corpus datasheet"
    return True, "OK"


def write_all() -> None:
    write_results()
    write_markdown()


def check_all() -> Tuple[bool, str]:
    ok, detail = check_results()
    if not ok:
        return ok, detail
    ok, detail = check_markdown()
    if not ok:
        return ok, detail
    return True, "OK"


__all__ = [
    "SCHEMA_VERSION",
    "RESULTS_PATH",
    "DOC_PATH",
    "DATASET_SCOPE",
    "DatasheetRecord",
    "records",
    "results_document",
    "write_results",
    "load_results",
    "check_results",
    "markdown_document",
    "write_markdown",
    "check_markdown",
    "write_all",
    "check_all",
]
