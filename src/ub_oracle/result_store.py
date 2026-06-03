"""Versioned result-store schema v2 (100_STEPS step 154).

Older experiment documents used a single ``content_hash`` over the verdict layer.
That hash is the reproducibility lemma: reruns may have different timings or
paths, but if the semantic verdicts are the same the verdict hash is unchanged.

Schema v2 preserves that invariant as ``hashes.verdict_hash`` and adds a separate
``hashes.store_hash`` for the environment-bound identity of a stored artifact
(schema, kind, producer, corpus hash, verdict hash, and toolchain fingerprint).
This separation lets v1 artifacts migrate without pretending that old documents
recorded toolchain or corpus metadata they did not have.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from .parallel_harness import _canonical_bytes

SCHEMA_VERSION = "result-store/v2"


class ResultStoreError(ValueError):
    """Raised when a result-store document is malformed or stale."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sorted_verdicts(verdicts: Iterable[Mapping[str, object]]) -> List[Dict[str, object]]:
    return sorted(
        (dict(v) for v in verdicts),
        key=lambda r: (str(r["item_id"]), str(r["lang"])),
    )


def verdict_hash(verdicts: Iterable[Mapping[str, object]]) -> str:
    """Stable hash over semantic verdicts only."""

    return _sha(_canonical_bytes(_sorted_verdicts(verdicts)))


def _fingerprint_from_provenance(provenance: Optional[Mapping[str, object]]) -> Dict[str, str]:
    if not provenance:
        return {}
    fp = provenance.get("fingerprint")
    if isinstance(fp, Mapping):
        return {str(k): str(v) for k, v in fp.items()}
    return {}


def _store_identity(
    *,
    artifact_kind: str,
    producer: str,
    corpus_hash: str,
    vhash: str,
    toolchain_fingerprint: Mapping[str, str],
) -> Dict[str, object]:
    return {
        "schema": SCHEMA_VERSION,
        "artifact_kind": artifact_kind,
        "producer": producer,
        "corpus_hash": corpus_hash,
        "verdict_hash": vhash,
        "toolchain_fingerprint": dict(sorted(toolchain_fingerprint.items())),
    }


def store_hash(
    *,
    artifact_kind: str,
    producer: str,
    corpus_hash: str,
    vhash: str,
    toolchain_fingerprint: Mapping[str, str],
) -> str:
    return _sha(
        _canonical_bytes(
            _store_identity(
                artifact_kind=artifact_kind,
                producer=producer,
                corpus_hash=corpus_hash,
                vhash=vhash,
                toolchain_fingerprint=toolchain_fingerprint,
            )
        )
    )


def build_result_store_doc(
    *,
    artifact_kind: str,
    producer: str,
    verdicts: Sequence[Mapping[str, object]],
    corpus_hash: str = "",
    toolchain_provenance: Optional[Mapping[str, object]] = None,
    measurements: Optional[Mapping[str, object]] = None,
    metadata: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    """Build a schema-v2 document.

    ``measurements`` and ``metadata`` are intentionally excluded from both hashes.
    They can record wall-clock timings, paths, notes, or table-friendly summaries
    without invalidating the semantic reproducibility lemma.
    """

    sorted_verdicts = _sorted_verdicts(verdicts)
    vhash = verdict_hash(sorted_verdicts)
    fingerprint = _fingerprint_from_provenance(toolchain_provenance)
    shash = store_hash(
        artifact_kind=artifact_kind,
        producer=producer,
        corpus_hash=corpus_hash,
        vhash=vhash,
        toolchain_fingerprint=fingerprint,
    )
    return {
        "schema": SCHEMA_VERSION,
        "artifact_kind": artifact_kind,
        "producer": producer,
        "corpus": {
            "hash": corpus_hash,
        },
        "environment": {
            "toolchain_fingerprint": fingerprint,
            "toolchain_provenance": dict(toolchain_provenance or {}),
        },
        "hashes": {
            "verdict_hash": vhash,
            "store_hash": shash,
        },
        "verdicts": sorted_verdicts,
        "measurements": dict(measurements or {}),
        "metadata": dict(metadata or {}),
        "reproducibility_lemma": (
            "hashes.verdict_hash is computed solely from the sorted verdict "
            "layer; measurements and metadata cannot change it"
        ),
    }


def validate_result_store_doc(doc: Mapping[str, object]) -> None:
    if doc.get("schema") != SCHEMA_VERSION:
        raise ResultStoreError("unsupported result-store schema")
    verdicts = doc.get("verdicts")
    if not isinstance(verdicts, list):
        raise ResultStoreError("result-store document missing verdicts")
    hashes = doc.get("hashes")
    if not isinstance(hashes, Mapping):
        raise ResultStoreError("result-store document missing hashes")
    corpus = doc.get("corpus")
    if not isinstance(corpus, Mapping):
        raise ResultStoreError("result-store document missing corpus")
    env = doc.get("environment")
    if not isinstance(env, Mapping):
        raise ResultStoreError("result-store document missing environment")

    vhash = verdict_hash(verdicts)
    if hashes.get("verdict_hash") != vhash:
        raise ResultStoreError("verdict hash mismatch")
    fingerprint_obj = env.get("toolchain_fingerprint")
    if not isinstance(fingerprint_obj, Mapping):
        raise ResultStoreError("result-store document missing toolchain fingerprint")
    expected_store_hash = store_hash(
        artifact_kind=str(doc.get("artifact_kind", "")),
        producer=str(doc.get("producer", "")),
        corpus_hash=str(corpus.get("hash", "")),
        vhash=vhash,
        toolchain_fingerprint={str(k): str(v) for k, v in fingerprint_obj.items()},
    )
    if hashes.get("store_hash") != expected_store_hash:
        raise ResultStoreError("store hash mismatch")


def migrate_scale_measure_v1(doc: Mapping[str, object]) -> Dict[str, object]:
    """Migrate a ``scale-measure/v1`` document into schema v2."""

    if doc.get("schema") != "scale-measure/v1":
        raise ResultStoreError("expected scale-measure/v1 source document")
    verdicts = doc.get("verdicts")
    if not isinstance(verdicts, list):
        raise ResultStoreError("scale-measure/v1 document missing verdicts")
    legacy_hash = doc.get("content_hash")
    computed = verdict_hash(verdicts)
    if legacy_hash != computed:
        raise ResultStoreError("legacy content_hash does not match verdict layer")
    migrated = build_result_store_doc(
        artifact_kind="scale-measure",
        producer="ub_oracle.scale_measure",
        verdicts=verdicts,
        measurements=dict(doc.get("measurements", {})),
        metadata={
            "source_schema": doc.get("schema"),
            "legacy_content_hash": legacy_hash,
            "n_items": doc.get("n_items"),
            "n_decided": doc.get("n_decided"),
            "n_abstained": doc.get("n_abstained"),
        },
    )
    validate_result_store_doc(migrated)
    return migrated


@dataclass(frozen=True)
class ReproducibilityLemma:
    ok: bool
    left_verdict_hash: str
    right_verdict_hash: str
    detail: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "ok": self.ok,
            "left_verdict_hash": self.left_verdict_hash,
            "right_verdict_hash": self.right_verdict_hash,
            "detail": self.detail,
        }


def prove_verdict_hash_stability(
    left: Mapping[str, object],
    right: Mapping[str, object],
) -> ReproducibilityLemma:
    """Check the v2 reproducibility lemma for two result-store documents."""

    validate_result_store_doc(left)
    validate_result_store_doc(right)
    left_hash = str(left["hashes"]["verdict_hash"])  # type: ignore[index]
    right_hash = str(right["hashes"]["verdict_hash"])  # type: ignore[index]
    ok = left_hash == right_hash
    return ReproducibilityLemma(
        ok=ok,
        left_verdict_hash=left_hash,
        right_verdict_hash=right_hash,
        detail="verdict hashes match" if ok else "verdict hashes differ",
    )


RESULT_STORE_SPI = {
    "build_result_store_doc": build_result_store_doc,
    "validate_result_store_doc": validate_result_store_doc,
    "migrate_scale_measure_v1": migrate_scale_measure_v1,
    "prove_verdict_hash_stability": prove_verdict_hash_stability,
}


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps({"schema": SCHEMA_VERSION}, sort_keys=True))
