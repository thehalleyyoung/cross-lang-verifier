"""
Language-agnostic, re-executable counterexample format (100_STEPS steps 9 & 132).

A ``Counterexample`` is the unit of evidence the oracle emits.  It is designed
to be *re-executed by an independent harness* (see ``reexec``) regardless of the
language pair, so that no verdict ships unconfirmed by ground truth.

The JSON schema is intentionally minimal and stable:

    {
      "schema_version": 1,
      "divergence_class": "signed_overflow",
      "source_lang": "c",
      "target_lang": "rust",
      "inputs": {"x": 2147483647},
      "source": {
          "snippet": "...C source...",
          "definedness": "undefined",      # status of the witnessed behavior in C
          "observed": {"O0": "0", "O2": "1"} # optional ground-truth observations
      },
      "target": {
          "snippet": "...Rust source...",
          "observed": "0"
      },
      "divergence_witness": "C signed overflow at x=INT_MAX; O0/O2 disagree, Rust defined=0",
      "definedness_witness": "all inputs are valid 32-bit ints (defined inputs)",
      "confirmed": true,
      "proof_certificate": {
          "schema_version": 1,
          "verdict": "divergent",
          "observation": {
              "ub_reached": true,
              "target_defined": true,
              "consequence": true
          },
          "kernel_theorem": "oracle_sound",
          "checker_scope": "final source-UB positive-claim inference over trusted run facts",
          "counterexample_hash": "sha256:...",
          "certificate_hash": "sha256:..."
      }
    }
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

REPLAY_SCHEMA_VERSION = 1
PROOF_CERTIFICATE_SCHEMA_VERSION = 1
CERTIFICATE_THEOREM = "oracle_sound"
CERTIFICATE_SCOPE = "final source-UB positive-claim inference over trusted run facts"
CERTIFICATE_ISSUER = "cross-lang-verifier"


def _canonical_json(value: Dict[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )


def _sha256_json(value: Dict[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _required_bool(mapping: Dict[str, Any], key: str) -> bool:
    value = mapping.get(key)
    if type(value) is not bool:
        raise ValueError(f"proof certificate field {key!r} must be a JSON boolean")
    return value


@dataclass
class ProofCertificate:
    """Machine-checkable certificate for a source-UB positive claim.

    The certificate binds the verified-checker observation to a canonical
    counterexample payload. The hash is an integrity/rebinding guard, not a
    cryptographic signature: the trusted facts are still the issuer's recorded
    re-execution facts, while the Lean checker proves the final
    ``productViolated -> UB divergence`` inference over those facts.
    """

    verdict: str
    ub_reached: bool
    target_defined: bool
    consequence: bool
    counterexample_hash: str
    certificate_hash: str = ""
    kernel_theorem: str = CERTIFICATE_THEOREM
    checker_scope: str = CERTIFICATE_SCOPE
    issuer: str = CERTIFICATE_ISSUER
    schema_version: int = PROOF_CERTIFICATE_SCHEMA_VERSION

    @property
    def observation(self) -> Dict[str, bool]:
        return {
            "ub_reached": bool(self.ub_reached),
            "target_defined": bool(self.target_defined),
            "consequence": bool(self.consequence),
        }

    def _hash_payload(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "verdict": self.verdict,
            "observation": self.observation,
            "kernel_theorem": self.kernel_theorem,
            "checker_scope": self.checker_scope,
            "issuer": self.issuer,
            "counterexample_hash": self.counterexample_hash,
        }

    def recompute_hash(self) -> str:
        return _sha256_json(self._hash_payload())

    def to_dict(self) -> Dict[str, Any]:
        return {
            **self._hash_payload(),
            "certificate_hash": self.certificate_hash,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProofCertificate":
        obs = d.get("observation", {})
        if not isinstance(obs, dict):
            raise ValueError("proof certificate observation must be an object")
        return cls(
            verdict=str(d["verdict"]),
            ub_reached=_required_bool(obs, "ub_reached"),
            target_defined=_required_bool(obs, "target_defined"),
            consequence=_required_bool(obs, "consequence"),
            counterexample_hash=str(d["counterexample_hash"]),
            certificate_hash=str(d.get("certificate_hash", "")),
            kernel_theorem=str(d.get("kernel_theorem", CERTIFICATE_THEOREM)),
            checker_scope=str(d.get("checker_scope", CERTIFICATE_SCOPE)),
            issuer=str(d.get("issuer", CERTIFICATE_ISSUER)),
            schema_version=int(d.get("schema_version", PROOF_CERTIFICATE_SCHEMA_VERSION)),
        )


@dataclass
class Counterexample:
    divergence_class: str
    source_lang: str
    target_lang: str
    inputs: Dict[str, Any]
    source_snippet: str
    target_snippet: str
    # the status of the witnessed behavior in the *source* language
    source_definedness: str = "undefined"
    divergence_witness: str = ""
    definedness_witness: str = ""
    # filled by the re-execution harness
    source_observed: Dict[str, Any] = field(default_factory=dict)
    target_observed: Optional[Any] = None
    confirmed: bool = False
    proof_certificate: Optional[ProofCertificate] = None
    schema_version: int = REPLAY_SCHEMA_VERSION

    def certificate_payload(self) -> Dict[str, Any]:
        """Canonical payload the proof certificate is bound to.

        The payload deliberately excludes ``proof_certificate`` itself to avoid a
        circular hash, and includes observed outputs so a certificate minted for
        one confirmed run cannot be rebound to a different replay transcript.
        """
        return {
            "schema_version": self.schema_version,
            "divergence_class": self.divergence_class,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "inputs": self.inputs,
            "source": {
                "snippet": self.source_snippet,
                "definedness": self.source_definedness,
                "observed": self.source_observed,
            },
            "target": {
                "snippet": self.target_snippet,
                "observed": self.target_observed,
            },
            "divergence_witness": self.divergence_witness,
            "definedness_witness": self.definedness_witness,
            "confirmed": self.confirmed,
        }

    def counterexample_hash(self) -> str:
        return _sha256_json(self.certificate_payload())

    def attach_proof_certificate(self, observation: Dict[str, bool]) -> ProofCertificate:
        cert = ProofCertificate(
            verdict="divergent",
            ub_reached=_required_bool(observation, "ub_reached"),
            target_defined=_required_bool(observation, "target_defined"),
            consequence=_required_bool(observation, "consequence"),
            counterexample_hash=self.counterexample_hash(),
        )
        cert.certificate_hash = cert.recompute_hash()
        self.proof_certificate = cert
        return cert

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "schema_version": self.schema_version,
            "divergence_class": self.divergence_class,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "inputs": self.inputs,
            "source": {
                "snippet": self.source_snippet,
                "definedness": self.source_definedness,
                "observed": self.source_observed,
            },
            "target": {
                "snippet": self.target_snippet,
                "observed": self.target_observed,
            },
            "divergence_witness": self.divergence_witness,
            "definedness_witness": self.definedness_witness,
            "confirmed": self.confirmed,
        }
        if self.proof_certificate is not None:
            out["proof_certificate"] = self.proof_certificate.to_dict()
        return out

    def to_json(self, *, indent: int = 2, sort_keys: bool = True) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=sort_keys)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Counterexample":
        src = d.get("source", {})
        tgt = d.get("target", {})
        raw_cert = d.get("proof_certificate")
        cert = None
        if raw_cert is not None:
            cert = ProofCertificate.from_dict(raw_cert)
        return cls(
            divergence_class=d["divergence_class"],
            source_lang=d["source_lang"],
            target_lang=d["target_lang"],
            inputs=d["inputs"],
            source_snippet=src.get("snippet", ""),
            target_snippet=tgt.get("snippet", ""),
            source_definedness=src.get("definedness", "undefined"),
            divergence_witness=d.get("divergence_witness", ""),
            definedness_witness=d.get("definedness_witness", ""),
            source_observed=src.get("observed", {}) or {},
            target_observed=tgt.get("observed"),
            confirmed=d.get("confirmed", False),
            proof_certificate=cert,
            schema_version=d.get("schema_version", REPLAY_SCHEMA_VERSION),
        )

    @classmethod
    def from_json(cls, s: str) -> "Counterexample":
        return cls.from_dict(json.loads(s))


def verify_certificate(counterexample: Counterexample) -> ProofCertificate:
    """Validate a proof-carrying counterexample without re-running compilers.

    This checks the durable certificate's shape, its binding to the serialized
    counterexample payload, and the same recorded-observable predicate consumed
    by the Lean verified checker. It intentionally does not re-establish the run
    facts; that remains the job of the ground-truth harness that minted the
    certificate.
    """
    cert = counterexample.proof_certificate
    if cert is None:
        raise ValueError("counterexample has no proof_certificate")
    if counterexample.source_definedness != "undefined":
        raise ValueError("proof certificates are scoped to source-UB witnesses")
    if cert.schema_version != PROOF_CERTIFICATE_SCHEMA_VERSION:
        raise ValueError(f"unsupported proof certificate schema: {cert.schema_version}")
    if cert.verdict != "divergent":
        raise ValueError(f"unsupported proof certificate verdict: {cert.verdict}")
    if cert.kernel_theorem != CERTIFICATE_THEOREM:
        raise ValueError(f"unexpected certificate theorem: {cert.kernel_theorem}")
    if cert.checker_scope != CERTIFICATE_SCOPE:
        raise ValueError(f"unexpected certificate scope: {cert.checker_scope}")
    if cert.issuer != CERTIFICATE_ISSUER:
        raise ValueError(f"unexpected certificate issuer: {cert.issuer}")
    if cert.counterexample_hash != counterexample.counterexample_hash():
        raise ValueError("proof certificate is not bound to this counterexample")
    if cert.certificate_hash != cert.recompute_hash():
        raise ValueError("proof certificate hash mismatch")
    if not (cert.ub_reached and cert.target_defined and cert.consequence):
        raise ValueError("certificate observation does not violate the product assertion")
    return cert
