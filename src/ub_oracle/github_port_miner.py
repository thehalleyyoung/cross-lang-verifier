"""GitHub C->Rust port miner and verifier-backed sample corpus.

This is the Step-156 bridge between "toy benchmark" and "where users really
port code": it identifies public repositories that look like C-to-Rust ports or
rewrites, ranks them deterministically, and ties a small checked-in extraction
sample to the normal verifier/result-hash discipline.  The checked-in sources are
minimal extraction units derived from mined project families; they are not copied
third-party repository files.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .ir import assert_valid
from .verify import VerifyReport, VerifyVerdict, verify_unit

SCHEMA_VERSION = "github-port-mining/v1"
DISCOVERY_QUERY = (
    '("rust rewrite" OR "rust implementation" OR "memory safe implementation" '
    'OR "C dynamic library") (C OR "GNU" OR zlib OR sudo OR coreutils)'
)

_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = _ROOT / "experiments" / "github_ports"
SAMPLE_DIR = EXPERIMENT_DIR / "samples"
RESULTS_PATH = EXPERIMENT_DIR / "results.json"

_RUST_TERMS = ("rust", "cargo", "crate", "memory safe")
_C_PORT_TERMS = (
    "gnu",
    "libc",
    "c library",
    "c-abi",
    "c abi",
    "zlib",
    "sudo",
    "coreutils",
    "dynamic library",
    "ffi",
)
_PORT_TERMS = (
    "port",
    "rewrite",
    "reimplementation",
    "implementation",
    "translation",
    "transpile",
    "memory safe",
)


@dataclass(frozen=True)
class GitHubPortCandidate:
    owner_repo: str
    url: str
    description: str
    stars: int = 0
    topics: Tuple[str, ...] = ()
    evidence: Tuple[str, ...] = ()

    @property
    def score(self) -> int:
        text = _candidate_text(self.owner_repo, self.description, self.topics)
        score = 0
        score += 25 if _contains_any(text, _RUST_TERMS) else 0
        score += 25 if _contains_any(text, _C_PORT_TERMS) else 0
        score += 20 if _contains_any(text, _PORT_TERMS) else 0
        score += min(20, self.stars // 1000)
        score += min(10, len(self.evidence) * 2)
        return score


@dataclass(frozen=True)
class GitHubPortSample:
    sample_id: str
    candidate_repo: str
    source_family: str
    c_file: str
    rust_file: str
    divergence_class: str
    unit: Dict[str, object]
    expected_symbolic_verdict: str
    provenance: str
    note: str

    @property
    def c_path(self) -> Path:
        return SAMPLE_DIR / self.c_file

    @property
    def rust_path(self) -> Path:
        return SAMPLE_DIR / self.rust_file


SEED_CANDIDATES: Tuple[GitHubPortCandidate, ...] = (
    GitHubPortCandidate(
        owner_repo="uutils/coreutils",
        url="https://github.com/uutils/coreutils",
        description="Cross-platform Rust rewrite of the GNU coreutils",
        stars=23_373,
        topics=("rust", "coreutils", "gnu-coreutils", "cross-platform"),
        evidence=("Rust rewrite", "GNU coreutils lineage", "large public port"),
    ),
    GitHubPortCandidate(
        owner_repo="trifectatechfoundation/sudo-rs",
        url="https://github.com/trifectatechfoundation/sudo-rs",
        description="A memory safe implementation of sudo and su.",
        stars=4_390,
        topics=("rust", "sudo", "memory-safety"),
        evidence=("memory-safe implementation", "sudo lineage", "security-critical port"),
    ),
    GitHubPortCandidate(
        owner_repo="trifectatechfoundation/zlib-rs",
        url="https://github.com/trifectatechfoundation/zlib-rs",
        description=(
            "A zlib implementation in rust available as a C dynamic library "
            "and as a rust crate"
        ),
        stars=643,
        topics=("rust", "zlib", "c-abi"),
        evidence=("zlib lineage", "C dynamic library", "Rust crate"),
    ),
)


SAMPLES: Tuple[GitHubPortSample, ...] = (
    GitHubPortSample(
        sample_id="coreutils-size-accumulate",
        candidate_repo="uutils/coreutils",
        source_family="GNU coreutils size/accounting arithmetic",
        c_file="coreutils_size_accumulate.c",
        rust_file="coreutils_size_accumulate.rs",
        divergence_class="signed_overflow",
        unit={
            "name": "coreutils-size-accumulate",
            "kind": "binop_const",
            "op": "add",
            "const": 4096,
            "width": 32,
            "var": "bytes",
            "signed": True,
            "probe": "signed_overflow",
            "source_lang": "c",
            "target_lang": "rust",
        },
        expected_symbolic_verdict=VerifyVerdict.CANDIDATE.value,
        provenance=(
            "Mined from the uutils/coreutils Rust rewrite family; extraction unit "
            "models byte-count accumulation common in file-size utilities."
        ),
        note="Symbolic signed-overflow witness; real compiler confirmation is optional.",
    ),
    GitHubPortSample(
        sample_id="zlib-checksum-window",
        candidate_repo="trifectatechfoundation/zlib-rs",
        source_family="zlib checksum window arithmetic",
        c_file="zlib_checksum_window.c",
        rust_file="zlib_checksum_window.rs",
        divergence_class="signed_overflow",
        unit={
            "name": "zlib-checksum-window",
            "kind": "binop_const",
            "op": "add",
            "const": 65521,
            "width": 32,
            "var": "sum",
            "signed": True,
            "probe": "signed_overflow",
            "source_lang": "c",
            "target_lang": "rust",
            "x_range": [0, 1_000_000],
        },
        expected_symbolic_verdict=VerifyVerdict.NO_DIVERGENCE_FOUND.value,
        provenance=(
            "Mined from the zlib-rs C-ABI port family; extraction unit models a "
            "bounded checksum-window update."
        ),
        note="Safe-range control discharged by the interval pre-pass.",
    ),
    GitHubPortSample(
        sample_id="sudo-tty-rate",
        candidate_repo="trifectatechfoundation/sudo-rs",
        source_family="sudo terminal/accounting rate arithmetic",
        c_file="sudo_tty_rate.c",
        rust_file="sudo_tty_rate.rs",
        divergence_class="div_by_zero",
        unit={
            "name": "sudo-tty-rate",
            "kind": "div",
            "width": 32,
            "signed": True,
            "a": "bytes",
            "b": "elapsed",
            "probe": "div_by_zero",
            "source_lang": "c",
            "target_lang": "rust",
        },
        expected_symbolic_verdict=VerifyVerdict.CANDIDATE.value,
        provenance=(
            "Mined from the sudo-rs memory-safe sudo implementation family; "
            "extraction unit models rate computation with an elapsed-time divisor."
        ),
        note="Symbolic zero-divisor witness; real compiler confirmation is optional.",
    ),
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _canonical_bytes(obj: object) -> bytes:
    return json.dumps(
        obj, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _contains_any(text: str, needles: Sequence[str]) -> bool:
    return any(n in text for n in needles)


def _candidate_text(owner_repo: str, description: str, topics: Iterable[str]) -> str:
    return " ".join((owner_repo, description, *topics)).lower()


def _topic_names(raw: object) -> Tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, list):
        names = []
        for item in raw:
            if isinstance(item, str):
                names.append(item)
            elif isinstance(item, Mapping) and isinstance(item.get("name"), str):
                names.append(item["name"])
        return tuple(sorted({n for n in names if n}))
    return ()


def _candidate_from_record(record: Mapping[str, object]) -> Optional[GitHubPortCandidate]:
    owner_repo = record.get("full_name") or record.get("nameWithOwner")
    if not isinstance(owner_repo, str) or "/" not in owner_repo:
        return None
    if record.get("archived") is True or record.get("isArchived") is True:
        return None
    url = record.get("html_url") or record.get("url") or f"https://github.com/{owner_repo}"
    description = record.get("description") or ""
    stars = record.get("stargazers_count", record.get("stargazerCount", 0))
    topics = _topic_names(record.get("topics") or record.get("repositoryTopics"))
    if not isinstance(url, str):
        url = f"https://github.com/{owner_repo}"
    if not isinstance(description, str):
        description = ""
    if isinstance(stars, bool) or not isinstance(stars, int):
        stars = 0

    text = _candidate_text(owner_repo, description, topics)
    evidence = []
    for term in (*_RUST_TERMS, *_C_PORT_TERMS, *_PORT_TERMS):
        if term in text:
            evidence.append(term)
    if not (_contains_any(text, _RUST_TERMS) and _contains_any(text, _C_PORT_TERMS)):
        return None
    return GitHubPortCandidate(
        owner_repo=owner_repo,
        url=url,
        description=description,
        stars=stars,
        topics=topics,
        evidence=tuple(sorted(set(evidence))),
    )


def parse_github_search_response(payload: Mapping[str, object]) -> List[GitHubPortCandidate]:
    """Parse GitHub search JSON into ranked C->Rust port candidates.

    The parser accepts both REST repository-search payloads (``items``) and the
    compact GraphQL-like ``nodes`` shape used in tests.  It intentionally does not
    preserve response metadata, timestamps, or rate-limit headers.
    """

    raw_items = payload.get("items")
    if raw_items is None:
        raw_items = payload.get("nodes")
    if not isinstance(raw_items, list):
        return []
    candidates = []
    for item in raw_items:
        if isinstance(item, Mapping):
            cand = _candidate_from_record(item)
            if cand is not None:
                candidates.append(cand)
    return rank_candidates(candidates)


def rank_candidates(candidates: Iterable[GitHubPortCandidate]) -> List[GitHubPortCandidate]:
    dedup: Dict[str, GitHubPortCandidate] = {}
    for cand in candidates:
        old = dedup.get(cand.owner_repo)
        if old is None or cand.score > old.score or (
            cand.score == old.score and cand.stars > old.stars
        ):
            dedup[cand.owner_repo] = cand
    return sorted(
        dedup.values(), key=lambda c: (-c.score, -c.stars, c.owner_repo.lower())
    )


def seeded_candidates() -> List[GitHubPortCandidate]:
    return rank_candidates(SEED_CANDIDATES)


def mined_candidates(payload: Optional[Mapping[str, object]] = None) -> List[GitHubPortCandidate]:
    live = parse_github_search_response(payload) if payload is not None else []
    return rank_candidates((*SEED_CANDIDATES, *live))


def fetch_github_search(
    query: str = DISCOVERY_QUERY,
    *,
    token: Optional[str] = None,
    per_page: int = 20,
) -> Mapping[str, object]:
    """Fetch one repository-search page from GitHub.

    This is deliberately optional and not used by the reproducibility check; tests
    exercise the parser with fixtures.  Callers may pass ``token`` or set
    ``GITHUB_TOKEN`` to raise unauthenticated rate limits.
    """

    from urllib.parse import urlencode

    token = token or os.environ.get("GITHUB_TOKEN")
    qs = urlencode({"q": query, "sort": "stars", "order": "desc", "per_page": per_page})
    request = urllib.request.Request(
        f"https://api.github.com/search/repositories?{qs}",
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "cross-lang-verifier-github-port-miner",
            **({"Authorization": f"Bearer {token}"} if token else {}),
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def validate_corpus_shape(samples: Iterable[GitHubPortSample] = SAMPLES) -> None:
    samples = tuple(samples)
    if len(samples) < 3:
        raise AssertionError(f"expected at least 3 verified samples, got {len(samples)}")
    if len({s.candidate_repo for s in samples}) < 3:
        raise AssertionError("expected samples from at least 3 distinct mined repositories")
    candidate_repos = {c.owner_repo for c in SEED_CANDIDATES}
    for sample in samples:
        if sample.candidate_repo not in candidate_repos:
            raise AssertionError(f"{sample.sample_id}: unknown candidate repo")
        if sample.unit.get("probe") != sample.divergence_class:
            raise AssertionError(f"{sample.sample_id}: probe/class mismatch")
        assert_valid(sample.unit, label=sample.sample_id)
        if not sample.c_path.exists():
            raise AssertionError(f"{sample.c_path} is missing")
        if not sample.rust_path.exists():
            raise AssertionError(f"{sample.rust_path} is missing")


def confirm_sample(sample: GitHubPortSample) -> VerifyReport:
    """Run a live compiler-backed confirmation for ``sample``'s IR unit."""

    return verify_unit(dict(sample.unit), confirm=True)


def sample_record(sample: GitHubPortSample) -> Dict[str, object]:
    assert_valid(sample.unit, label=sample.sample_id)
    c_src = _read(sample.c_path)
    rust_src = _read(sample.rust_path)
    report = verify_unit(dict(sample.unit), confirm=False)
    return {
        "sample_id": sample.sample_id,
        "candidate_repo": sample.candidate_repo,
        "source_family": sample.source_family,
        "provenance": sample.provenance,
        "note": sample.note,
        "c_file": str(sample.c_path.relative_to(_ROOT)),
        "rust_file": str(sample.rust_path.relative_to(_ROOT)),
        "source_sha256": _sha256_text(c_src),
        "rust_sha256": _sha256_text(rust_src),
        "divergence_class": sample.divergence_class,
        "unit": sample.unit,
        "expected_symbolic_verdict": sample.expected_symbolic_verdict,
        "observed_symbolic_verdict": report.verdict.value,
        "verdict_matches_expectation": (
            report.verdict.value == sample.expected_symbolic_verdict
        ),
        "prepass_pruned": sorted(report.prepass_pruned),
    }


def verdict_layer() -> List[Dict[str, object]]:
    return [sample_record(sample) for sample in SAMPLES]


def candidate_records(candidates: Optional[Sequence[GitHubPortCandidate]] = None) -> List[Dict[str, object]]:
    candidates = seeded_candidates() if candidates is None else rank_candidates(candidates)
    return [
        {
            "owner_repo": cand.owner_repo,
            "url": cand.url,
            "description": cand.description,
            "stars": cand.stars,
            "topics": list(cand.topics),
            "evidence": list(cand.evidence),
            "score": cand.score,
        }
        for cand in candidates
    ]


def content_hash(cases: Optional[List[Dict[str, object]]] = None) -> str:
    cases = verdict_layer() if cases is None else cases
    stable = [
        {
            "sample_id": c["sample_id"],
            "candidate_repo": c["candidate_repo"],
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
    by_verdict: Dict[str, int] = {}
    by_class: Dict[str, int] = {}
    for case in cases:
        by_verdict[case["observed_symbolic_verdict"]] = (
            by_verdict.get(case["observed_symbolic_verdict"], 0) + 1
        )
        by_class[case["divergence_class"]] = by_class.get(case["divergence_class"], 0) + 1
    return {
        "schema": SCHEMA_VERSION,
        "discovery_query": DISCOVERY_QUERY,
        "content_hash": content_hash(cases),
        "n_candidates": len(SEED_CANDIDATES),
        "n_verified_samples": len(cases),
        "n_source_families": len({c["source_family"] for c in cases}),
        "all_verdicts_match_expectation": all(
            c["verdict_matches_expectation"] for c in cases
        ),
        "by_symbolic_verdict": by_verdict,
        "by_divergence_class": by_class,
        "candidates": candidate_records(),
        "samples": cases,
    }


def write_results(path: Path = RESULTS_PATH) -> None:
    doc = results_document()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def check_results(path: Path = RESULTS_PATH) -> Tuple[bool, str]:
    regenerated = json.dumps(results_document(), indent=2, sort_keys=True) + "\n"
    if not path.exists():
        return False, f"{path} is missing"
    if path.read_text(encoding="utf-8") != regenerated:
        return False, f"{path} does not match regenerated results"
    return True, "OK"


__all__ = [
    "SCHEMA_VERSION",
    "DISCOVERY_QUERY",
    "EXPERIMENT_DIR",
    "SAMPLE_DIR",
    "RESULTS_PATH",
    "GitHubPortCandidate",
    "GitHubPortSample",
    "SEED_CANDIDATES",
    "SAMPLES",
    "parse_github_search_response",
    "rank_candidates",
    "seeded_candidates",
    "mined_candidates",
    "fetch_github_search",
    "validate_corpus_shape",
    "confirm_sample",
    "sample_record",
    "verdict_layer",
    "candidate_records",
    "content_hash",
    "results_document",
    "write_results",
    "check_results",
]
