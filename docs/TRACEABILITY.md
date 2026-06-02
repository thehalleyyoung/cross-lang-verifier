# Theory–implementation traceability

Every claim this project makes is backed by a **named code module and symbol**, and — where the claim is a checkable theorem — by a runnable check. The single source of truth for this mapping is `src/ub_oracle/traceability.py`; this page is generated from it and the test suite asserts the two never drift.

Run the machine check yourself:

```python
from src.ub_oracle.traceability import verify_traceability
assert verify_traceability() == []   # imports + symbols + theorems all pass
```

Each row whose **Executable check** is ✓ has a fast, toolchain-free theorem core that is *run* (not just referenced) by `verify_traceability` and by `test_traceability_every_claim_maps_to_code`.

| Claim id | Statement | Module | Symbols | Executable check | Docs |
| --- | --- | --- | --- | --- | --- |
| `C1-soundness` | A DIVERGENT verdict is only returned after ground-truth re-execution of real compiled programs confirms a UB-rooted divergence. | `src/ub_oracle/verify.py` | `verify_unit`, `VerifyVerdict`, `VerifyReport` | — | `README.md`, `docs/SEMANTICS.md` |
| `C2-divergence-semantics` | The witnessed property — divergence modulo source-undefinedness — is an executable predicate that is one-sided (positive only when source UB is reached). | `src/ub_oracle/semantics.py` | `is_divergence`, `judge`, `Observation`, `EXPLOITED`, `TRAP_VS_DEFINED` | ✓ | `docs/SEMANTICS.md` |
| `C3-semantics-coincides` | The formal divergence predicate coincides with the re-execution harness's confirmed decision on real programs. | `src/ub_oracle/semantics.py` | `coincides_with_harness`, `observation_from_reexec` | ✓ | `docs/SEMANTICS.md` |
| `C4-completeness-classes` | For the four integer classes the symbolic search is complete on a precisely-stated bounded fragment (no false negatives there). | `src/ub_oracle/completeness.py` | `FRAGMENTS`, `check_class_completeness`, `check_all_completeness`, `OUT_OF_FRAGMENT` | ✓ | `docs/COMPLETENESS.md` |
| `C5-completeness-pairs` | The completeness result holds across every registered language pair (C→Rust/Go/Swift). | `src/ub_oracle/completeness.py` | `check_pair_completeness` | ✓ | `docs/COMPLETENESS.md` |
| `C6-prepass-sound` | The abstract-interpretation pre-pass is a verdict-preserving accelerator: it never prunes a class that can actually reach UB on the unit's domain. | `src/ub_oracle/abstract_interp.py` | `prunable_classes`, `analyze_unit`, `Interval` | ✓ | `README.md` |
| `C7-ir-contract` | A single pair-agnostic IR contract is frozen and validated at the boundary; ill-formed units are rejected. | `src/ub_oracle/ir.py` | `validate_unit`, `assert_valid`, `KNOWN_KINDS`, `IRValidationError` | ✓ | `docs/IR.md` |
| `C8-pluggable-targets` | Target-language semantics are data-driven packs; the pipeline differs across pairs only by the declared target, and three pairs are registered. | `src/ub_oracle/target_semantics.py` | `TargetPack`, `PACKS`, `get_pack` | ✓ | `README.md`, `CAPABILITIES.md` |
| `C9-replay-format` | Confirmed counterexamples are captured in a versioned, replayable format. | `src/ub_oracle/replay.py` | `Counterexample`, `REPLAY_SCHEMA_VERSION` | — | `README.md` |
| `C10-ub-catalogue` | The supported undefined-behavior divergence classes are enumerated in a single catalogue. | `src/ub_oracle/catalogue.py` | `CATALOGUE`, `DivergenceClass`, `c_ub_classes` | ✓ | `README.md`, `CAPABILITIES.md` |
| `C11-redteam` | An internal red-team actively tries to make the oracle call a truly divergent pair equivalent, on every supported pair. | `src/ub_oracle/redteam.py` | `build_cases`, `run_redteam`, `RedTeamReport` | — | `README.md` |
| `C12-uninit-definedness` | The uninitialized-read class is decided by a real three-point definedness-lattice dataflow analysis that flags reads of slots not written on all paths and never flags a fully-initialized read. | `src/ub_oracle/oracles/uninit_read.py` | `analyze_definedness`, `uninitialized_read`, `UninitializedReadOracle` | ✓ | `README.md`, `CAPABILITIES.md` |
| `C13-cegar-refinement` | Guarded fragments the non-relational interval pre-pass cannot discharge are decided by a lazy predicate-abstraction CEGAR loop: it starts from the UB condition with no guards, refines one path-condition at a time on each spurious model, and is sound (its verdict matches exact enumeration of the UB region) while genuinely refining on path-sensitive fragments. | `src/ub_oracle/cegar.py` | `run_cegar`, `brute_force_witness`, `GuardedQuery` | ✓ | `README.md` |

## How to extend

Add a `Claim(...)` to `CLAIMS` in `src/ub_oracle/traceability.py` (with a fast `theorem=` core when the claim is checkable), then regenerate this table. The test `test_traceability_doc_lists_every_claim_id` fails if a claim id is added to the code but not cited here, or vice-versa.
