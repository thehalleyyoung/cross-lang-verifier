"""
Language-agnostic, re-executable counterexample format (100_STEPS step 9).

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
      "confirmed": false
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

REPLAY_SCHEMA_VERSION = 1


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
    schema_version: int = REPLAY_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
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

    def to_json(self, *, indent: int = 2, sort_keys: bool = True) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=sort_keys)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Counterexample":
        src = d.get("source", {})
        tgt = d.get("target", {})
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
            schema_version=d.get("schema_version", REPLAY_SCHEMA_VERSION),
        )

    @classmethod
    def from_json(cls, s: str) -> "Counterexample":
        return cls.from_dict(json.loads(s))
