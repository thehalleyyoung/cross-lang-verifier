#!/usr/bin/env bash
# Credibility guard (100_STEPS step 1).
#
# Fails the build if:
#   (1) any committed results artifact (*.json / *.csv) under benchmarks/ or
#       experiments/ contains a simulation marker (a *fabricated* number), OR
#   (2) a Python file under benchmarks/ or experiments/ introduces a simulation
#       marker without being listed in benchmarks/QUARANTINE.md (prevents silent
#       reintroduction of fake benchmarks), OR
#   (3) the trusted real-results artifact is missing or not byte-reproducible.
#
# The trusted artifact is experiments/ub_divergence/results.json.

set -euo pipefail
cd "$(dirname "$0")/.."

MARKER='simulat\|fabricat\|\bmock\b\|fake result'
QUARANTINE='benchmarks/QUARANTINE.md'
TRUSTED='experiments/ub_divergence/results.json'
fail=0

echo "==> [1/3] checking committed results artifacts are not simulated"
# Only inspect data artifacts, and exclude the quarantine doc itself.
results_files=$(grep -rIl --include='*.json' --include='*.csv' -e "$MARKER" \
                  benchmarks experiments 2>/dev/null \
                  | grep -v '__pycache__' || true)
if [ -n "$results_files" ]; then
  echo "ERROR: simulation markers found in results artifacts:" >&2
  echo "$results_files" | sed 's/^/   /' >&2
  fail=1
else
  echo "    ok: no simulated results artifacts"
fi

echo "==> [2/3] checking simulated benchmark scripts are quarantined"
sim_scripts=$(grep -rIl --include='*.py' -e 'simulat' benchmarks experiments 2>/dev/null \
                | grep -v '__pycache__' | sort || true)
for f in $sim_scripts; do
  if ! grep -qF "$f" "$QUARANTINE"; then
    echo "ERROR: $f contains a simulation marker but is not listed in $QUARANTINE" >&2
    fail=1
  fi
done
if [ "$fail" -eq 0 ]; then
  echo "    ok: all simulated scripts are quarantined"
fi

echo "==> [3/3] checking trusted real-results artifact reproduces"
if [ ! -f "$TRUSTED" ]; then
  echo "ERROR: missing trusted artifact $TRUSTED (run 'make reproduce')" >&2
  fail=1
else
  if grep -qi -e "$MARKER" "$TRUSTED"; then
    echo "ERROR: trusted artifact $TRUSTED contains a simulation marker" >&2
    fail=1
  fi
  PY="${PYTHON:-python3}"
  if [ -x venv/bin/python ]; then PY=venv/bin/python; fi
  if "$PY" -m experiments.ub_divergence.run --check >/dev/null 2>&1; then
    echo "    ok: $TRUSTED reproduces byte-identically"
  else
    echo "ERROR: $TRUSTED is not byte-reproducible (run 'make reproduce')" >&2
    fail=1
  fi
fi

if [ "$fail" -ne 0 ]; then
  echo "credibility guard: FAILED" >&2
  exit 1
fi
echo "credibility guard: PASSED"
