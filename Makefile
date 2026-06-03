# cross-lang-verifier — reproducibility & CI entry points (100_STEPS steps 1,2,7)
#
# The trusted, real (non-simulated) results live under experiments/ub_divergence/
# and are confirmed against real C (UBSan) and Rust compilers.

PYTHON ?= $(shell [ -x venv/bin/python ] && echo venv/bin/python || echo python3)

.PHONY: help reproduce reproduce-confirm reproduce-check reproduce-kit docker-build docker-reproduce guard test-ub ci cold-start-ci matrix matrix-confirm matrix-check cex-quality cex-quality-check perf perf-check memory-bound-check sharded-repro-check flaky-toolchain-check distributed-manifest-check result-store-check repro-hardening-check redteam redteam-check c2rust-corpus c2rust-corpus-check historical-cve-check github-port-mining-check package-check coverage coverage-check verified-check soundness-check pre-review-check ir-diff-check cross-arch-check scale-paper-section scale-paper-section-check demo-video

help:
	@echo "Targets:"
	@echo "  reproduce         regenerate trusted results (deterministic, no toolchain)"
	@echo "  reproduce-kit     full external replication kit (guard + regen + re-confirm + manifest)"
	@echo "  docker-build      build the hermetic replication image"
	@echo "  docker-reproduce  build + run the replication kit inside the image"
	@echo "  reproduce-confirm regenerate + confirm against real C/Rust compilers"
	@echo "  reproduce-check   assert results regenerate byte-identically"
	@echo "  matrix            regenerate the cross-pair regression matrix (deterministic)"
	@echo "  matrix-confirm    regenerate matrix + confirm every cell against real compilers"
	@echo "  matrix-check      assert the cross-pair matrix regenerates byte-identically"
	@echo "  cex-quality       minimize every anchor witness against real compilers (study)"
	@echo "  cex-quality-check assert the cex-quality baseline regenerates byte-identically"
	@echo "  perf              measure the symbolic-search scalability curves (study)"
	@echo "  perf-check        assert the perf grid reproduces and latency budgets hold"
	@echo "  memory-bound-check prove bounded/unbounded verdict equivalence"
	@echo "  sharded-repro-check prove shard hashes merge to the whole-run verdict hash"
	@echo "  flaky-toolchain-check quarantine unstable compiler/runtime evidence"
	@echo "  distributed-manifest-check prove distributed shard manifests merge deterministically"
	@echo "  result-store-check validate result-store v2 migration and reproducibility lemma"
	@echo "  cross-arch-check prove cross-architecture replay reporting/detection"
	@echo "  scale-paper-section regenerate the paper's migration-scale TeX section"
	@echo "  scale-paper-section-check assert generated scale paper section is fresh"
	@echo "  redteam           run the internal red-team against real compilers (study)"
	@echo "  redteam-check     assert the red-team adversarial grid regenerates byte-identically"
	@echo "  c2rust-corpus     regenerate the Tier-1 c2rust-output corpus artifacts"
	@echo "  c2rust-corpus-check assert c2rust corpus results and generated Rust reproduce"
	@echo "  historical-cve-check assert 50+ historical-CVE weakness replays and bundles"
	@echo "  github-port-mining-check assert GitHub-mined port samples reproduce"
	@echo "  package-check     build the wheel, install it in a fresh venv, run the CLI"
	@echo "  demo-video        regenerate the README-linked c2rust CWE-class demo video"
	@echo "  guard             run the credibility guard (no simulated results)"
	@echo "  cold-start-ci     full non-compiler suite + one real compiled sample (<5 min)"
	@echo "  test-ub           run the ub_oracle test suite only"
	@echo "  coverage          print the core-module branch-coverage table"
	@echo "  coverage-check    enforce the coverage ratchet floor (slow, ~4 min)"
	@echo "  verified-check    build the Lean boundary proof and smoke-test the checker"
	@echo "  soundness-check   enforce one soundness statement per registered oracle"
	@echo "  pre-review-check  validate the external pre-review/artifact-release packet"
	@echo "  ir-diff-check     prove clang-AST vs rustc-MIR divergence localization"
	@echo "  ci                guard + cold-start-ci + reproduce-check + matrix-check + cex-quality-check + perf-check + repro-hardening-check + redteam-check + verified-check + soundness-check + test-ub"

reproduce:
	$(PYTHON) -m experiments.ub_divergence.run

reproduce-kit:
	bash scripts/reproduce_kit.sh

docker-build:
	docker build -t cross-lang-verifier .

docker-reproduce: docker-build
	docker run --rm cross-lang-verifier

reproduce-confirm:
	$(PYTHON) -m experiments.ub_divergence.run --confirm

reproduce-check:
	$(PYTHON) -m experiments.ub_divergence.run --check

matrix:
	$(PYTHON) -m experiments.cross_pair_matrix.run --table

matrix-confirm:
	$(PYTHON) -m experiments.cross_pair_matrix.run --confirm

matrix-check:
	$(PYTHON) -m experiments.cross_pair_matrix.run --check

cex-quality:
	$(PYTHON) -m experiments.cex_quality.run --minimize --table

cex-quality-check:
	$(PYTHON) -m experiments.cex_quality.run --check

perf:
	$(PYTHON) -m experiments.perf_curves.run --table

perf-check:
	$(PYTHON) -m experiments.perf_curves.run --check --budget-check

memory-bound-check:
	$(PYTHON) -m pytest tests/test_memory_bounded_mode.py -q

sharded-repro-check:
	$(PYTHON) -m pytest tests/test_sharded_repro.py -q

flaky-toolchain-check:
	$(PYTHON) -m pytest tests/test_flaky_toolchain.py -q

distributed-manifest-check:
	$(PYTHON) -m pytest tests/test_distributed_manifest.py -q

result-store-check:
	$(PYTHON) -m pytest tests/test_result_store.py -q

repro-hardening-check: sharded-repro-check flaky-toolchain-check distributed-manifest-check result-store-check

cross-arch-check:
	$(PYTHON) -m pytest tests/test_arch_replay.py -q

scale-paper-section:
	PYTHONPATH=src $(PYTHON) -c "from ub_oracle.paper_scale_section import write_scale_section; print(write_scale_section())"

scale-paper-section-check:
	$(PYTHON) -m pytest tests/test_paper_scale_section.py -q

redteam:
	$(PYTHON) -m experiments.redteam.run --attack --table

redteam-check:
	$(PYTHON) -m experiments.redteam.run --check

c2rust-corpus:
	$(PYTHON) -m experiments.c2rust_corpus.run --regenerate

c2rust-corpus-check:
	$(PYTHON) -m experiments.c2rust_corpus.run --check --check-generated

historical-cve-check:
	$(PYTHON) -m pytest tests/test_historical_cve_corpus.py -q

github-port-mining-check:
	$(PYTHON) -m experiments.github_ports.run --check

package-check:
	bash scripts/verify_packaging.sh

demo-video:
	$(PYTHON) scripts/build_demo_video.py

coverage:
	$(PYTHON) scripts/coverage_gate.py --report

coverage-check:
	$(PYTHON) scripts/coverage_gate.py

verified-check:
	cd formal && lake build CompletenessBoundary
	cd formal && lake build SPIContract
	cd formal && lake build HashStability
	cd formal && lake build verified-checker
	cd formal && .lake/build/bin/verified-checker --verdict divergent --ub true --target-defined true --consequence true
	cd formal && ! .lake/build/bin/verified-checker --verdict divergent --ub false --target-defined true --consequence true

soundness-check:
	PYTHONPATH=src $(PYTHON) -m ub_oracle.soundness_gate --check

pre-review-check:
	$(PYTHON) -m pytest tests/test_pre_review.py -q

ir-diff-check:
	$(PYTHON) -m pytest tests/test_ub_oracle.py::test_compiler_ir_diff_localizes_signed_overflow_semantics_on_real_ir tests/test_ub_oracle.py::test_compiler_ir_diff_also_detects_plain_rust_overflow_assert_mir -q

guard:
	bash scripts/check_no_simulated_results.sh

test-ub:
	$(PYTHON) -m pytest tests/test_ub_oracle.py -q

green-check:
	$(PYTHON) scripts/test_ratchet.py --fast

cold-start-ci:
	$(PYTHON) scripts/cold_start_ci.py

large-scale:
	PYTHONPATH=src $(PYTHON) -m ub_oracle.large_scale_study

ci: guard cold-start-ci reproduce-check matrix-check cex-quality-check perf-check repro-hardening-check redteam-check verified-check soundness-check test-ub
	@echo "ci: PASSED"
