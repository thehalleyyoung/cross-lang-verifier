# cross-lang-verifier — reproducibility & CI entry points (100_STEPS steps 1,2,7)
#
# The trusted, real (non-simulated) results live under experiments/ub_divergence/
# and are confirmed against real C (UBSan) and Rust compilers.

PYTHON ?= $(shell [ -x venv/bin/python ] && echo venv/bin/python || echo python3)

.PHONY: help reproduce reproduce-confirm reproduce-check reproduce-kit docker-build docker-reproduce guard test-ub ci matrix matrix-confirm matrix-check cex-quality cex-quality-check perf perf-check redteam redteam-check c2rust-corpus c2rust-corpus-check package-check coverage coverage-check verified-check

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
	@echo "  perf-check        assert the perf scalability grid regenerates byte-identically"
	@echo "  redteam           run the internal red-team against real compilers (study)"
	@echo "  redteam-check     assert the red-team adversarial grid regenerates byte-identically"
	@echo "  c2rust-corpus     regenerate the Tier-1 c2rust-output corpus artifacts"
	@echo "  c2rust-corpus-check assert c2rust corpus results and generated Rust reproduce"
	@echo "  package-check     build the wheel, install it in a fresh venv, run the CLI"
	@echo "  guard             run the credibility guard (no simulated results)"
	@echo "  test-ub           run the ub_oracle test suite only"
	@echo "  coverage          print the core-module branch-coverage table"
	@echo "  coverage-check    enforce the coverage ratchet floor (slow, ~4 min)"
	@echo "  verified-check    build and smoke-test the Lean/Lake verified checker"
	@echo "  ci                guard + reproduce-check + matrix-check + cex-quality-check + perf-check + redteam-check + verified-check + test-ub"

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
	$(PYTHON) -m experiments.perf_curves.run --check

redteam:
	$(PYTHON) -m experiments.redteam.run --attack --table

redteam-check:
	$(PYTHON) -m experiments.redteam.run --check

c2rust-corpus:
	$(PYTHON) -m experiments.c2rust_corpus.run --regenerate

c2rust-corpus-check:
	$(PYTHON) -m experiments.c2rust_corpus.run --check --check-generated

package-check:
	bash scripts/verify_packaging.sh

coverage:
	$(PYTHON) scripts/coverage_gate.py --report

coverage-check:
	$(PYTHON) scripts/coverage_gate.py

verified-check:
	cd formal && lake build verified-checker
	cd formal && .lake/build/bin/verified-checker --verdict divergent --ub true --target-defined true --consequence true
	cd formal && ! .lake/build/bin/verified-checker --verdict divergent --ub false --target-defined true --consequence true

guard:
	bash scripts/check_no_simulated_results.sh

test-ub:
	$(PYTHON) -m pytest tests/test_ub_oracle.py -q

green-check:
	$(PYTHON) scripts/test_ratchet.py --fast

large-scale:
	PYTHONPATH=src $(PYTHON) -m ub_oracle.large_scale_study

ci: guard green-check reproduce-check matrix-check cex-quality-check perf-check redteam-check verified-check test-ub
	@echo "ci: PASSED"
