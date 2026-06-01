# cross-lang-verifier — reproducibility & CI entry points (100_STEPS steps 1,2,7)
#
# The trusted, real (non-simulated) results live under experiments/ub_divergence/
# and are confirmed against real C (UBSan) and Rust compilers.

PYTHON ?= $(shell [ -x venv/bin/python ] && echo venv/bin/python || echo python3)

.PHONY: help reproduce reproduce-confirm reproduce-check guard test-ub ci

help:
	@echo "Targets:"
	@echo "  reproduce         regenerate trusted results (deterministic, no toolchain)"
	@echo "  reproduce-confirm regenerate + confirm against real C/Rust compilers"
	@echo "  reproduce-check   assert results regenerate byte-identically"
	@echo "  guard             run the credibility guard (no simulated results)"
	@echo "  test-ub           run the ub_oracle test suite only"
	@echo "  ci                guard + reproduce-check + test-ub"

reproduce:
	$(PYTHON) -m experiments.ub_divergence.run

reproduce-confirm:
	$(PYTHON) -m experiments.ub_divergence.run --confirm

reproduce-check:
	$(PYTHON) -m experiments.ub_divergence.run --check

guard:
	bash scripts/check_no_simulated_results.sh

test-ub:
	$(PYTHON) -m pytest tests/test_ub_oracle.py -q

ci: guard reproduce-check test-ub
	@echo "ci: PASSED"
