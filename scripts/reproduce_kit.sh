#!/usr/bin/env bash
# cross-lang-verifier — external replication kit entry point (Step 54).
#
# A stranger runs this (directly, via `make reproduce-kit`, or inside the Docker
# image) to regenerate every byte-reproducible table and re-confirm every oracle
# against the *real* toolchain, then prints a content-hash manifest. Exit code 0
# means the artifact reproduced; non-zero means a divergence from the recorded
# results — exactly the signal an artifact evaluator needs.
#
# It is intentionally self-contained and prints what it is doing at each step.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-$([ -x venv/bin/python ] && echo venv/bin/python || echo python3)}"
MANIFEST="${MANIFEST:-reproduction_manifest.json}"

echo "=== cross-lang-verifier replication kit ==="
echo "python: $($PYTHON --version 2>&1)"
for tool in clang rustc go z3 boolector; do
    if command -v "$tool" >/dev/null 2>&1; then
        echo "$tool: $(command -v "$tool")"
    else
        echo "$tool: (absent — toolchain-gated steps will consistency-check only)"
    fi
done
echo

echo "--- [1/4] credibility guard (no simulated results) ---"
bash scripts/check_no_simulated_results.sh

echo "--- [2/4] byte-identical regeneration of the trusted result tables ---"
"$PYTHON" -m experiments.ub_divergence.run --check
"$PYTHON" -m experiments.cross_pair_matrix.run --check

echo "--- [3/4] re-confirm the corpora & measurement layers against real code ---"
"$PYTHON" - <<'PYEOF'
import json, sys
from src.ub_oracle import replication
rep = replication.confirm_replication_kit(quick=False)
print(replication.render(rep))
manifest = replication.manifest(rep)
import os
path = os.environ.get("MANIFEST", "reproduction_manifest.json")
with open(path, "w") as f:
    json.dump(manifest, f, sort_keys=True, indent=2)
    f.write("\n")
print(f"wrote manifest -> {path}")
sys.exit(0 if rep.ok else 1)
PYEOF

echo "--- [4/4] manifest content-hash stability (verdict layers reproduce) ---"
"$PYTHON" - <<'PYEOF'
from src.ub_oracle import replication
a = replication.manifest(replication.confirm_replication_kit())
b = replication.manifest(replication.confirm_replication_kit())
ha, hb = a["kit_hash"], b["kit_hash"]
print(f"kit_hash run-1: {ha}")
print(f"kit_hash run-2: {hb}")
assert ha == hb, "replication kit hash is not stable across runs"
print("kit hash is stable.")
PYEOF

echo
echo "=== replication kit: PASSED ==="
