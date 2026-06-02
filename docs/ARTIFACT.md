# Artifact Appendix

This appendix maps `cross-lang-verifier` onto the three **ACM artifact badges**
and — crucially — makes each badge's criteria a *checked predicate* rather than a
prose promise. The same inspection an artifact-evaluation committee performs by
hand is encoded in `src/ub_oracle/artifact_eval.py` and re-run as part of the
traceability matrix (claim **C40-artifact-eval**).

```
python -m src.ub_oracle.artifact_eval
```

prints, for every badge, each criterion and whether it passed with *real
evidence* or *consistency-only* (the latter happens only when a toolchain is
absent, and never falsely claims a badge).

## Artifacts Available

The artifact is publicly archived under a stable identifier with an open licence.
Checked ingredients:

| Criterion | Evidence |
| --- | --- |
| `open_licence_present` | `LICENSE` (MIT) present with an OSI-style grant. |
| `archival_descriptor_names_public_repo` | `CITATION.cff` names `github.com/thehalleyyoung/cross-lang-verifier`. |
| `readme_present` | `README.md` present and non-trivial. |
| `version_consistent` | `pyproject.toml` version agrees with `CITATION.cff`. |

## Artifacts Evaluated — Functional

The artifact is documented, consistent, complete and **exercisable**.

| Criterion | Evidence |
| --- | --- |
| `documented` | README, CAPABILITIES, this appendix and TRACEABILITY all present. |
| `entry_points_exercisable` | the replication kit's files resolve and the corpus is ≥500 pairs / ≥2 languages. |
| `packaging_proof_present` | `scripts/verify_packaging.sh` (fresh-venv wheel install + console-script smoke test). |
| `live_oracle_runs` | with a C+UBSan+rustc toolchain, the **real** oracle catches a div-by-zero divergence (`10 0`) and stays silent on a safe input (`10 2`). |

The live check compiles and runs actual binaries; with no toolchain it is
reported as consistency-only and does not assert the badge on missing evidence.

## Artifacts Evaluated — Reproduced

An independent party can regenerate the central results.

| Criterion | Evidence |
| --- | --- |
| `trusted_results_byte_identical` | `experiments/ub_divergence/results.json` regenerates byte-for-byte (the credibility-guard property). |
| `replication_kit_hash_stable` | `replication.manifest(...)["kit_hash"]` is identical across two independent runs. |
| `scale_hash_reproducible` | the scale measurement's verdict content hash is stable across runs. |
| `generalization_hash_reproducible` | the (pair × style × class) generalization grid's content hash is stable across runs. |

## One-line confirmation

```python
from ub_oracle.artifact_eval import confirm_artifact_evaluation
c = confirm_artifact_evaluation()
assert c.ok                       # every criterion of every badge passes
print(c.earned_badges)            # ('available', 'functional', 'reproduced')
```
