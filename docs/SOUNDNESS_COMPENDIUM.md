# Soundness compendium

This file is generated from `src/ub_oracle/soundness_compendium.py`, which joins `soundness_gate.SOUNDNESS_STATEMENTS`, `traceability.CLAIMS`, and the Lean theorem contract in `mechanized_soundness.REQUIRED_THEOREMS`. It is intentionally stable: it records declared theorem/claim references and concrete witness units, not host-dependent proof-assistant output.

- Registered oracle statements: **60**
- Traceability claims: **64**
- Required ProductSoundness Lean theorems: **20**

## Mechanized theorem surface

- `oracle_sound`
- `oracle_complete_rel`
- `oracle_decides`
- `equivalence_never_reported`
- `report_implies_ub`
- `pack_oracle_sound`
- `rust_oracle_sound`
- `product_program_preserves_divergence_witness`
- `product_program_emits_witness_iff_product_violated`
- `product_program_witness_iff_divergence`
- `strict_aliasing_oracle_sound`
- `strict_aliasing_report_implies_type_pun`
- `strict_aliasing_report_implies_optimizer_exploited`
- `pointer_provenance_oracle_sound`
- `pointer_provenance_report_implies_out_of_provenance`
- `pointer_provenance_report_implies_checked_target`
- `pointer_provenance_report_implies_trap_vs_defined`
- `minimizer_preserves_divergence`
- `minimizer_reduction_steps_preserve_divergence`
- `minimizer_certificate_sound`

## Oracle-to-evidence matrix

| Statement | Oracle | Mode | Source premise | Evidence refs | Witness unit |
| --- | --- | --- | --- | --- | --- |
| `S-c-cpp-signed-shift-sign-bit` | `c->cpp:signed_shift_sign_bit` | `trap_vs_defined` | `undefined` | `claim:C37-product-program`, `claim:C10-ub-catalogue` | `{"kind":"sign_bit_shift","probe":"signed_shift_sign_bit","source_lang":"c","target_lang":"cpp","width":32}` |
| `S-c-go-array-oob` | `c->go:array_oob` | `trap_vs_defined` | `undefined` | `claim:C37-product-program`, `claim:C31-cve-corpus` | `{"kind":"array_index","length":4,"probe":"array_oob","source_lang":"c","target_lang":"go"}` |
| `S-c-go-atomic-ordering` | `c->go:atomic_ordering` | `model_level_divergence` | `defined` | `claim:C37-product-program`, `claim:C37-product-program` | `{"kind":"atomic_litmus","pattern":"store_buffering","probe":"atomic_ordering","source_lang":"c","target_lang":"go"}` |
| `S-c-go-bitfield-layout` | `c->go:bitfield_layout` | `defined_divergence` | `impl_defined` | `claim:C37-product-program`, `claim:C15-abi-layout` | `{"kind":"bitfield_struct","probe":"bitfield_layout","source_lang":"c","target_lang":"go"}` |
| `S-c-go-div-by-zero` | `c->go:div_by_zero` | `trap_vs_defined` | `undefined` | `claim:C37-product-program`, `claim:C31-cve-corpus` | `{"kind":"div","probe":"div_by_zero","signed":true,"source_lang":"c","target_lang":"go","width":32}` |
| `S-c-go-enum-out-of-range` | `c->go:enum_out_of_range` | `defined_divergence` | `impl_defined` | `claim:C37-product-program`, `claim:C10-ub-catalogue` | `{"kind":"enum_cast","probe":"enum_out_of_range","source_lang":"c","target_lang":"go"}` |
| `S-c-go-fast-math-reassoc` | `c->go:fast_math_reassoc` | `optimizer_exploited` | `unspecified` | `claim:C37-product-program`, `claim:C60-fast-math-reassociation-divergence` | `{"kind":"fp_reassoc","probe":"fast_math_reassoc","source_lang":"c","target_lang":"go"}` |
| `S-c-go-float-cast-overflow` | `c->go:float_cast_overflow` | `trap_vs_defined` | `undefined` | `claim:C37-product-program`, `claim:C59-float-cast-overflow-divergence` | `{"kind":"float_cast","probe":"float_cast_overflow","source_lang":"c","target_lang":"go","width":32}` |
| `S-c-go-intmin-div-neg1` | `c->go:intmin_div_neg1` | `trap_vs_defined` | `undefined` | `claim:C37-product-program`, `claim:C31-cve-corpus` | `{"kind":"div","probe":"intmin_div_neg1","signed":true,"source_lang":"c","target_lang":"go","width":32}` |
| `S-c-go-longjmp-vla` | `c->go:longjmp_vla` | `libc_contract_trap_vs_defined` | `undefined` | `claim:C37-product-program`, `claim:C10-ub-catalogue` | `{"kind":"longjmp_vla","preferred_bound":4,"probe":"longjmp_vla","source_lang":"c","target_lang":"go"}` |
| `S-c-go-memcpy-overlap` | `c->go:memcpy_overlap` | `libc_contract_trap_vs_defined` | `undefined` | `claim:C37-product-program`, `claim:C24-libc-model` | `{"buffer_len":16,"kind":"memcpy_overlap","probe":"memcpy_overlap","source_lang":"c","target_lang":"go"}` |
| `S-c-go-pointer-provenance` | `c->go:pointer_provenance` | `trap_vs_defined` | `undefined` | `claim:C37-product-program`, `claim:C62-pointer-provenance-divergence`, `lean:pointer_provenance_oracle_sound` | `{"kind":"pointer_offset","probe":"pointer_provenance","source_lang":"c","target_lang":"go","width":32}` |
| `S-c-go-restrict-violation` | `c->go:restrict_violation` | `optimizer_exploited` | `undefined` | `claim:C37-product-program`, `claim:C61-restrict-violation-divergence` | `{"kind":"restrict_pair","probe":"restrict_violation","source_lang":"c","target_lang":"go"}` |
| `S-c-go-shift-oob` | `c->go:shift_oob` | `trap_vs_defined` | `undefined` | `claim:C37-product-program`, `claim:C31-cve-corpus` | `{"kind":"shift","probe":"shift_oob","source_lang":"c","target_lang":"go","value":1,"width":32}` |
| `S-c-go-signed-overflow` | `c->go:signed_overflow` | `exploited` | `undefined` | `claim:C37-product-program`, `claim:C37-product-program`, `lean:oracle_sound` | `{"const":1,"kind":"binop_const","op":"add","probe":"signed_overflow","signed":true,"source_lang":"c","target_lang":"go","width":32}` |
| `S-c-go-strict-aliasing` | `c->go:strict_aliasing` | `optimizer_exploited` | `undefined` | `claim:C37-product-program`, `claim:C41-mechanized-soundness`, `lean:strict_aliasing_oracle_sound` | `{"kind":"type_pun","probe":"strict_aliasing","source_lang":"c","target_lang":"go"}` |
| `S-c-go-uninit-padding` | `c->go:uninit_padding` | `uninit_padding` | `unspecified` | `claim:C37-product-program`, `claim:C10-ub-catalogue` | `{"kind":"uninit_padding","probe":"uninit_padding","source_lang":"c","target_lang":"go"}` |
| `S-c-go-uninit-read` | `c->go:uninit_read` | `optimizer_exploited` | `undefined` | `claim:C37-product-program`, `claim:C12-uninit-definedness` | `{"kind":"uninit_read","probe":"uninit_read","read":null,"source_lang":"c","storage":{"kind":"scalar"},"target_lang":"go","writes":[]}` |
| `S-c-go-vla-bound` | `c->go:vla_bound` | `trap_vs_defined` | `undefined` | `claim:C37-product-program`, `claim:C58-vla-bound-divergence` | `{"kind":"vla","probe":"vla_bound","source_lang":"c","target_lang":"go","width":32}` |
| `S-c-ocaml-array-oob` | `c->ocaml:array_oob` | `trap_vs_defined` | `undefined` | `claim:C37-product-program`, `claim:C31-cve-corpus` | `{"kind":"array_index","length":4,"probe":"array_oob","source_lang":"c","target_lang":"ocaml"}` |
| `S-c-ocaml-div-by-zero` | `c->ocaml:div_by_zero` | `trap_vs_defined` | `undefined` | `claim:C37-product-program`, `claim:C31-cve-corpus` | `{"kind":"div","probe":"div_by_zero","signed":true,"source_lang":"c","target_lang":"ocaml","width":32}` |
| `S-c-ocaml-intmin-div-neg1` | `c->ocaml:intmin_div_neg1` | `trap_vs_defined` | `undefined` | `claim:C37-product-program`, `claim:C31-cve-corpus` | `{"kind":"div","probe":"intmin_div_neg1","signed":true,"source_lang":"c","target_lang":"ocaml","width":32}` |
| `S-c-ocaml-signed-overflow` | `c->ocaml:signed_overflow` | `exploited` | `undefined` | `claim:C37-product-program`, `claim:C37-product-program`, `lean:oracle_sound` | `{"const":1,"kind":"binop_const","op":"add","probe":"signed_overflow","signed":true,"source_lang":"c","target_lang":"ocaml","width":32}` |
| `S-c-rust-array-oob` | `c->rust:array_oob` | `trap_vs_defined` | `undefined` | `claim:C37-product-program`, `claim:C31-cve-corpus` | `{"kind":"array_index","length":4,"probe":"array_oob","source_lang":"c","target_lang":"rust"}` |
| `S-c-rust-atomic-ordering` | `c->rust:atomic_ordering` | `model_level_divergence` | `defined` | `claim:C37-product-program`, `claim:C37-product-program` | `{"kind":"atomic_litmus","pattern":"store_buffering","probe":"atomic_ordering","source_lang":"c","target_lang":"rust"}` |
| `S-c-rust-bitfield-layout` | `c->rust:bitfield_layout` | `defined_divergence` | `impl_defined` | `claim:C37-product-program`, `claim:C15-abi-layout` | `{"kind":"bitfield_struct","probe":"bitfield_layout","source_lang":"c","target_lang":"rust"}` |
| `S-c-rust-div-by-zero` | `c->rust:div_by_zero` | `trap_vs_defined` | `undefined` | `claim:C37-product-program`, `claim:C31-cve-corpus` | `{"kind":"div","probe":"div_by_zero","signed":true,"source_lang":"c","target_lang":"rust","width":32}` |
| `S-c-rust-enum-out-of-range` | `c->rust:enum_out_of_range` | `defined_divergence` | `impl_defined` | `claim:C37-product-program`, `claim:C10-ub-catalogue` | `{"kind":"enum_cast","probe":"enum_out_of_range","source_lang":"c","target_lang":"rust"}` |
| `S-c-rust-eval-order` | `c->rust:eval_order` | `static_ub_vs_defined` | `undefined` | `claim:C37-product-program`, `claim:C30-eval-order` | `{"kind":"unsequenced","pattern":"postinc_read_add","probe":"eval_order","source_lang":"c","target_lang":"rust"}` |
| `S-c-rust-fast-math-reassoc` | `c->rust:fast_math_reassoc` | `optimizer_exploited` | `unspecified` | `claim:C37-product-program`, `claim:C60-fast-math-reassociation-divergence` | `{"kind":"fp_reassoc","probe":"fast_math_reassoc","source_lang":"c","target_lang":"rust"}` |
| `S-c-rust-float-cast-overflow` | `c->rust:float_cast_overflow` | `trap_vs_defined` | `undefined` | `claim:C37-product-program`, `claim:C59-float-cast-overflow-divergence` | `{"kind":"float_cast","probe":"float_cast_overflow","source_lang":"c","target_lang":"rust","width":32}` |
| `S-c-rust-fp-contraction` | `c->rust:fp_contraction` | `optimizer_exploited` | `unspecified` | `claim:C37-product-program`, `claim:C27-solver-portfolio` | `{"kind":"fp_fma","probe":"fp_contraction","source_lang":"c","target_lang":"rust"}` |
| `S-c-rust-intmin-div-neg1` | `c->rust:intmin_div_neg1` | `trap_vs_defined` | `undefined` | `claim:C37-product-program`, `claim:C31-cve-corpus` | `{"kind":"div","probe":"intmin_div_neg1","signed":true,"source_lang":"c","target_lang":"rust","width":32}` |
| `S-c-rust-longjmp-vla` | `c->rust:longjmp_vla` | `libc_contract_trap_vs_defined` | `undefined` | `claim:C37-product-program`, `claim:C10-ub-catalogue` | `{"kind":"longjmp_vla","preferred_bound":4,"probe":"longjmp_vla","source_lang":"c","target_lang":"rust"}` |
| `S-c-rust-memcpy-overlap` | `c->rust:memcpy_overlap` | `libc_contract_trap_vs_defined` | `undefined` | `claim:C37-product-program`, `claim:C24-libc-model` | `{"buffer_len":16,"kind":"memcpy_overlap","probe":"memcpy_overlap","source_lang":"c","target_lang":"rust"}` |
| `S-c-rust-pointer-provenance` | `c->rust:pointer_provenance` | `trap_vs_defined` | `undefined` | `claim:C37-product-program`, `claim:C62-pointer-provenance-divergence`, `lean:pointer_provenance_oracle_sound` | `{"kind":"pointer_offset","probe":"pointer_provenance","source_lang":"c","target_lang":"rust","width":32}` |
| `S-c-rust-restrict-violation` | `c->rust:restrict_violation` | `optimizer_exploited` | `undefined` | `claim:C37-product-program`, `claim:C61-restrict-violation-divergence` | `{"kind":"restrict_pair","probe":"restrict_violation","source_lang":"c","target_lang":"rust"}` |
| `S-c-rust-shift-oob` | `c->rust:shift_oob` | `trap_vs_defined` | `undefined` | `claim:C37-product-program`, `claim:C31-cve-corpus` | `{"kind":"shift","probe":"shift_oob","source_lang":"c","target_lang":"rust","value":1,"width":32}` |
| `S-c-rust-signed-overflow` | `c->rust:signed_overflow` | `exploited` | `undefined` | `claim:C37-product-program`, `claim:C37-product-program`, `lean:oracle_sound` | `{"const":1,"kind":"binop_const","op":"add","probe":"signed_overflow","signed":true,"source_lang":"c","target_lang":"rust","width":32}` |
| `S-c-rust-strict-aliasing` | `c->rust:strict_aliasing` | `optimizer_exploited` | `undefined` | `claim:C37-product-program`, `claim:C41-mechanized-soundness`, `lean:strict_aliasing_oracle_sound` | `{"kind":"type_pun","probe":"strict_aliasing","source_lang":"c","target_lang":"rust"}` |
| `S-c-rust-uninit-padding` | `c->rust:uninit_padding` | `uninit_padding` | `unspecified` | `claim:C37-product-program`, `claim:C10-ub-catalogue` | `{"kind":"uninit_padding","probe":"uninit_padding","source_lang":"c","target_lang":"rust"}` |
| `S-c-rust-uninit-read` | `c->rust:uninit_read` | `optimizer_exploited` | `undefined` | `claim:C37-product-program`, `claim:C12-uninit-definedness` | `{"kind":"uninit_read","probe":"uninit_read","read":null,"source_lang":"c","storage":{"kind":"scalar"},"target_lang":"rust","writes":[]}` |
| `S-c-rust-vla-bound` | `c->rust:vla_bound` | `trap_vs_defined` | `undefined` | `claim:C37-product-program`, `claim:C58-vla-bound-divergence` | `{"kind":"vla","probe":"vla_bound","source_lang":"c","target_lang":"rust","width":32}` |
| `S-c-swift-array-oob` | `c->swift:array_oob` | `trap_vs_defined` | `undefined` | `claim:C37-product-program`, `claim:C31-cve-corpus` | `{"kind":"array_index","length":4,"probe":"array_oob","source_lang":"c","target_lang":"swift"}` |
| `S-c-swift-div-by-zero` | `c->swift:div_by_zero` | `trap_vs_defined` | `undefined` | `claim:C37-product-program`, `claim:C31-cve-corpus` | `{"kind":"div","probe":"div_by_zero","signed":true,"source_lang":"c","target_lang":"swift","width":32}` |
| `S-c-swift-intmin-div-neg1` | `c->swift:intmin_div_neg1` | `trap_vs_defined` | `undefined` | `claim:C37-product-program`, `claim:C31-cve-corpus` | `{"kind":"div","probe":"intmin_div_neg1","signed":true,"source_lang":"c","target_lang":"swift","width":32}` |
| `S-c-swift-shift-oob` | `c->swift:shift_oob` | `trap_vs_defined` | `undefined` | `claim:C37-product-program`, `claim:C31-cve-corpus` | `{"kind":"shift","probe":"shift_oob","source_lang":"c","target_lang":"swift","value":1,"width":32}` |
| `S-c-swift-signed-overflow` | `c->swift:signed_overflow` | `exploited` | `undefined` | `claim:C37-product-program`, `claim:C37-product-program`, `lean:oracle_sound` | `{"const":1,"kind":"binop_const","op":"add","probe":"signed_overflow","signed":true,"source_lang":"c","target_lang":"swift","width":32}` |
| `S-c-swift-uninit-read` | `c->swift:uninit_read` | `optimizer_exploited` | `undefined` | `claim:C37-product-program`, `claim:C12-uninit-definedness` | `{"kind":"uninit_read","probe":"uninit_read","read":null,"source_lang":"c","storage":{"kind":"scalar"},"target_lang":"swift","writes":[]}` |
| `S-c-wasm-div-by-zero` | `c->wasm:div_by_zero` | `trap_vs_defined` | `undefined` | `claim:C37-product-program`, `claim:C31-cve-corpus` | `{"kind":"div","probe":"div_by_zero","signed":true,"source_lang":"c","target_lang":"wasm","width":32}` |
| `S-c-wasm-intmin-div-neg1` | `c->wasm:intmin_div_neg1` | `trap_vs_defined` | `undefined` | `claim:C37-product-program`, `claim:C31-cve-corpus` | `{"kind":"div","probe":"intmin_div_neg1","signed":true,"source_lang":"c","target_lang":"wasm","width":32}` |
| `S-c-wasm-shift-oob` | `c->wasm:shift_oob` | `trap_vs_defined` | `undefined` | `claim:C37-product-program`, `claim:C31-cve-corpus` | `{"kind":"shift","probe":"shift_oob","source_lang":"c","target_lang":"wasm","value":1,"width":32}` |
| `S-c-wasm-signed-overflow` | `c->wasm:signed_overflow` | `exploited` | `undefined` | `claim:C37-product-program`, `claim:C37-product-program`, `lean:oracle_sound` | `{"const":1,"kind":"binop_const","op":"add","probe":"signed_overflow","signed":true,"source_lang":"c","target_lang":"wasm","width":32}` |
| `S-c-zig-array-oob` | `c->zig:array_oob` | `trap_vs_defined` | `undefined` | `claim:C37-product-program`, `claim:C31-cve-corpus` | `{"kind":"array_index","length":4,"probe":"array_oob","source_lang":"c","target_lang":"zig"}` |
| `S-c-zig-div-by-zero` | `c->zig:div_by_zero` | `trap_vs_defined` | `undefined` | `claim:C37-product-program`, `claim:C31-cve-corpus` | `{"kind":"div","probe":"div_by_zero","signed":true,"source_lang":"c","target_lang":"zig","width":32}` |
| `S-c-zig-intmin-div-neg1` | `c->zig:intmin_div_neg1` | `trap_vs_defined` | `undefined` | `claim:C37-product-program`, `claim:C31-cve-corpus` | `{"kind":"div","probe":"intmin_div_neg1","signed":true,"source_lang":"c","target_lang":"zig","width":32}` |
| `S-c-zig-shift-oob` | `c->zig:shift_oob` | `trap_vs_defined` | `undefined` | `claim:C37-product-program`, `claim:C31-cve-corpus` | `{"kind":"shift","probe":"shift_oob","source_lang":"c","target_lang":"zig","value":1,"width":32}` |
| `S-c-zig-signed-overflow` | `c->zig:signed_overflow` | `exploited` | `undefined` | `claim:C37-product-program`, `claim:C37-product-program`, `lean:oracle_sound` | `{"const":1,"kind":"binop_const","op":"add","probe":"signed_overflow","signed":true,"source_lang":"c","target_lang":"zig","width":32}` |
| `S-go-rust-intmin-div-neg1` | `go->rust:intmin_div_neg1` | `defined_divergence` | `defined` | `claim:C37-product-program`, `claim:C31-cve-corpus` | `{"kind":"div","probe":"intmin_div_neg1","signed":true,"source_lang":"go","target_lang":"rust","width":32}` |
| `S-rust-c-intmin-div-neg1` | `rust->c:intmin_div_neg1` | `source_defined_target_ub` | `defined` | `claim:C37-product-program`, `claim:C31-cve-corpus` | `{"kind":"div","probe":"intmin_div_neg1","signed":true,"source_lang":"rust","target_lang":"c","width":32}` |

## Traceability claim index

| Claim | Module | Symbols |
| --- | --- | --- |
| `C1-soundness` | `ub_oracle.verify` | `verify_unit`, `VerifyVerdict`, `VerifyReport` |
| `C2-divergence-semantics` | `ub_oracle.semantics` | `is_divergence`, `judge`, `Observation`, `EXPLOITED`, `TRAP_VS_DEFINED` |
| `C3-semantics-coincides` | `ub_oracle.semantics` | `coincides_with_harness`, `observation_from_reexec` |
| `C4-completeness-classes` | `ub_oracle.completeness` | `FRAGMENTS`, `check_class_completeness`, `check_all_completeness`, `OUT_OF_FRAGMENT` |
| `C5-completeness-pairs` | `ub_oracle.completeness` | `check_pair_completeness` |
| `C6-prepass-sound` | `ub_oracle.abstract_interp` | `prunable_classes`, `analyze_unit`, `Interval` |
| `C7-ir-contract` | `ub_oracle.ir` | `validate_unit`, `assert_valid`, `KNOWN_KINDS`, `IRValidationError` |
| `C8-pluggable-targets` | `ub_oracle.target_semantics` | `TargetPack`, `PACKS`, `get_pack` |
| `C9-replay-format` | `ub_oracle.replay` | `Counterexample`, `REPLAY_SCHEMA_VERSION` |
| `C10-ub-catalogue` | `ub_oracle.catalogue` | `CATALOGUE`, `DivergenceClass`, `c_ub_classes` |
| `C11-redteam` | `ub_oracle.redteam` | `build_cases`, `run_redteam`, `RedTeamReport` |
| `C12-uninit-definedness` | `ub_oracle.oracles.uninit_read` | `analyze_definedness`, `uninitialized_read`, `UninitializedReadOracle` |
| `C13-cegar-refinement` | `ub_oracle.cegar` | `run_cegar`, `brute_force_witness`, `GuardedQuery` |
| `C14-kinduction-loops` | `ub_oracle.kinduction` | `prove`, `simulate`, `TransitionSystem`, `saturating_counter`, `accumulator_overflow` |
| `C15-abi-layout` | `ub_oracle.abi_layout` | `c_layout`, `optimized_layout`, `abi_divergence`, `union_layout`, `enum_abi_divergence`, `confirm_abi` |
| `C16-provenance-memory` | `ub_oracle.memory_model` | `simulate`, `first_fault`, `MemEvent`, `FaultKind`, `confirm_memory` |
| `C17-pointer-provenance` | `ub_oracle.provenance` | `simulate`, `first_fault`, `ProvEvent`, `ProvFault`, `PROVENANCE_INTERFACE`, `confirm_provenance` |
| `C18-ownership-facts` | `ub_oracle.ownership` | `PATTERNS`, `pattern`, `confirm_ownership`, `OWNERSHIP_INTERFACE` |
| `C19-unit-alignment` | `ub_oracle.unit_alignment` | `align`, `signature_score`, `types_compatible`, `name_only_align`, `alignment_accuracy` |
| `C20-foreign-frontier` | `ub_oracle.foreign_effects` | `scan_c_source`, `decide`, `FrontierVerdict`, `confirm_all`, `FOREIGN_FRONTIER` |
| `C21-concurrency-race` | `ub_oracle.concurrency` | `PATTERNS`, `pattern`, `confirm_race`, `RaceConfirmation`, `RustConfirmation`, `c_race_detector_available`, `go_race_detector_available`, `RACE_FRONTIER` |
| `C22-indirect-resolution` | `ub_oracle.indirect_calls` | `parse_unit`, `resolve_table_call`, `signature_compatible_targets`, `confirm_table_dispatch`, `table_is_well_typed` |
| `C23-real-preprocessing` | `ub_oracle.preprocess` | `preprocess`, `detect_unparenthesized_macros`, `confirm_macro_precedence_hazard`, `confirm_conditional_compilation`, `confirm_include_resolution` |
| `C24-libc-model` | `ub_oracle.libc_model` | `SPECS`, `confirm_spec`, `confirm_all`, `model_strcmp`, `LIBC_CONTRACTS` |
| `C25-ir-ingest` | `ub_oracle.ir_ingest` | `ingest_clang`, `ingest_rustc_mir`, `confirm_clang_ingest`, `confirm_mir_ingest`, `IRModule` |
| `C26-project-ingest` | `ub_oracle.project_ingest` | `ingest_compile_db`, `ingest_cargo_workspace`, `confirm_compile_db`, `confirm_cargo_workspace`, `ProjectModule` |
| `C27-solver-portfolio` | `ub_oracle.solver_portfolio` | `solve_portfolio`, `robustness_report`, `confirm_portfolio`, `available_solvers`, `PortfolioResult` |
| `C28-frontend-fuzz` | `ub_oracle.frontend_fuzz` | `fuzz_clang_frontend`, `confirm_fuzz`, `generate_program`, `GARBAGE_INPUTS`, `FuzzReport` |
| `C29-conformance` | `ub_oracle.conformance` | `run_conformance`, `confirm_conformance`, `ALL_CASES`, `ConformanceCase`, `CaseResult` |
| `C30-eval-order` | `ub_oracle.eval_order` | `detect_unsequenced`, `decide`, `confirm_sequencing`, `UnsequencedReport` |
| `C31-cve-corpus` | `ub_oracle.cve_corpus` | `run_corpus`, `confirm_corpus`, `coverage_table`, `CORPUS`, `CveCase` |
| `C32-ground-truth` | `ub_oracle.ground_truth` | `enumerate_corpus`, `label_item`, `establish_ground_truth`, `confirm_ground_truth`, `corpus_stats` |
| `C33-scale-measure` | `ub_oracle.scale_measure` | `run_scale`, `results_document`, `emit_results_json`, `content_hash`, `confirm_scale` |
| `C34-external-head-to-head` | `ub_oracle.external_baselines` | `run_head_to_head`, `confirm_head_to_head`, `applicability_table`, `CATEGORIES` |
| `C35-replication-kit` | `ub_oracle.replication` | `confirm_replication_kit`, `manifest`, `render` |
| `C36-statistical-rigor` | `ub_oracle.statistical_rigor` | `confirm_statistical_rigor`, `run_study`, `wilson_interval`, `PREREGISTERED_METRICS` |
| `C37-product-program` | `ub_oracle.product_program` | `confirm_product_program`, `product_violated`, `evaluate_clauses`, `build_product` |
| `C38-translation-validation` | `ub_oracle.translation_validation` | `validate`, `CounterexampleWitness`, `confirm_translation_validation` |
| `C39-generalization` | `ub_oracle.generalization` | `confirm_generalization`, `run_generalization`, `target_source` |
| `C40-artifact-eval` | `ub_oracle.artifact_eval` | `confirm_artifact_evaluation`, `evaluate_artifact`, `BADGES` |
| `C41-mechanized-soundness` | `ub_oracle.mechanized_soundness` | `confirm_mechanized_soundness`, `build_verified_checker`, `run_verified_checker`, `confirm_coq_crosscheck`, `confirm_mechanized_completeness_boundary`, `REQUIRED_THEOREMS`, `REQUIRED_COMPLETENESS_BOUNDARY_THEOREMS`, `REQUIRED_COQ_THEOREMS`, `LEAN_SOURCE`, `COMPLETENESS_BOUNDARY_SOURCE`, `COQ_SOURCE`, `CHECKER_SOURCE` |
| `C42-idiomatic-corpus` | `ub_oracle.idiomatic_corpus` | `confirm_idiomatic_corpus`, `run_corpus`, `CORPUS` |
| `C43-multipair-corpus` | `ub_oracle.multipair_corpus` | `confirm_multipair_corpus`, `run_corpus`, `CORPUS` |
| `C44-transpiler-recipes` | `ub_oracle.transpiler_recipes` | `confirm_transpiler_recipes`, `verify_transpiled`, `RECIPES` |
| `C45-web-playground` | `ub_oracle.playground` | `confirm_playground`, `evaluate`, `make_server` |
| `C46-docs-site` | `ub_oracle.docs_site` | `confirm_docs_site`, `generate_gallery` |
| `C47-vscode-extension` | `ub_oracle.vscode_ext` | `confirm_vscode_extension` |
| `C48-real-frontends` | `ub_oracle.frontends` | `confirm_real_frontends`, `ingest_treesitter`, `FRONTENDS` |
| `C49-single-binary` | `ub_oracle.single_binary` | `confirm_single_binary`, `build_pyz` |
| `C50-divergence-zoo` | `ub_oracle.divergence_zoo` | `confirm_zoo`, `index_by_class_and_pair`, `EXHIBITS` |
| `C51-paper-figures` | `ub_oracle.figures` | `confirm_figures`, `collect`, `generate_figures` |
| `C52-ecosystem-semver` | `ub_oracle.ecosystem` | `confirm_ecosystem`, `PUBLIC_API_V1`, `generate_artifacts` |
| `C53-claims-audit` | `ub_oracle.claims_audit` | `confirm_claims_audit`, `audit_text` |
| `C54-case-studies` | `ub_oracle.case_studies` | `confirm_case_studies`, `generate_case_studies`, `CaseResult` |
| `C55-responsible-disclosure` | `ub_oracle.disclosure` | `confirm_disclosures`, `reproduce_disclosure`, `DisclosureRecord` |
| `C56-true-green-ratchet` | `ub_oracle.test_ratchet_core` | `violations`, `enforce_counts`, `is_baselineable` |
| `C57-large-scale-study` | `ub_oracle.large_scale_study` | `generate_corpus`, `corpus_census`, `confirm_large_scale_study` |
| `C58-vla-bound-divergence` | `ub_oracle.oracles.vla_bound` | `VlaBoundOracle`, `GoVlaBoundOracle` |
| `C59-float-cast-overflow-divergence` | `ub_oracle.oracles.float_cast` | `FloatCastOverflowOracle`, `GoFloatCastOverflowOracle` |
| `C60-fast-math-reassociation-divergence` | `ub_oracle.oracles.fast_math` | `FastMathReassocOracle`, `GoFastMathReassocOracle` |
| `C61-restrict-violation-divergence` | `ub_oracle.oracles.restrict_alias` | `RestrictViolationOracle`, `GoRestrictViolationOracle` |
| `C62-pointer-provenance-divergence` | `ub_oracle.oracles.pointer_provenance` | `PointerProvenanceOracle`, `GoPointerProvenanceOracle` |
| `C63-soundness-regression-gate` | `ub_oracle.soundness_gate` | `SOUNDNESS_STATEMENTS`, `confirm_soundness_registry`, `audit_soundness_statements`, `SoundnessStatement` |
| `C64-soundness-compendium` | `ub_oracle.soundness_compendium` | `compendium_rows`, `render_compendium`, `confirm_compendium`, `SOUNDNESS_COMPENDIUM_DOC` |
