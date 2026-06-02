#!/usr/bin/env bash
# Packaging proof (100_STEPS step 59).
#
# Builds the wheel, installs it into a *fresh* virtualenv (PYTHONPATH unset, run
# from a neutral working directory so nothing can be imported from the source
# tree by accident), and exercises the real console script end-to-end:
#
#   1. `import ub_oracle` resolves to the INSTALLED top-level package.
#   2. both console scripts (`cross-lang-verify`, `cross-lang-verifier`) exist.
#   3. `--no-confirm` is a portable smoke test: exit 0, symbolic CANDIDATEs.
#   4. the default (ground-truth) run catches real DIVERGENCEs and exits 1 — the
#      CI gate — *iff* a C+UBSan+Rust toolchain is present (gated, not assumed).
#
# Exit 0 == packaging works.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
unset PYTHONPATH

PYTHON="${PYTHON:-python3}"
MANIFEST="$REPO/examples/units_manifest.json"

echo "==> [1/5] building wheel"
"$PYTHON" -m pip wheel "$REPO" --no-deps -w "$WORK/dist" >/dev/null
WHEEL="$(ls "$WORK"/dist/cross_lang_verifier-*.whl)"
echo "    built $(basename "$WHEEL")"

echo "==> [2/5] creating fresh venv and installing the wheel"
"$PYTHON" -m venv "$WORK/venv"
VPY="$WORK/venv/bin/python"
VBIN="$WORK/venv/bin"
"$VPY" -m pip install -q --upgrade pip >/dev/null 2>&1 || true
"$VPY" -m pip install -q "$WHEEL"

# From here on, run from a neutral directory so an `import ub_oracle` can only
# resolve to the installed package, never the ./src checkout.
cd "$WORK"

echo "==> [3/5] importing the INSTALLED top-level package"
"$VPY" -c "import ub_oracle, ub_oracle.cli, ub_oracle.regression_matrix; \
print('    import ub_oracle OK ->', ub_oracle.__file__)"
test -x "$VBIN/cross-lang-verify"   || { echo "ERROR: missing cross-lang-verify script" >&2; exit 1; }
test -x "$VBIN/cross-lang-verifier" || { echo "ERROR: missing cross-lang-verifier script" >&2; exit 1; }
echo "    both console scripts present"

echo "==> [4/5] portable smoke test (--no-confirm: expect exit 0, CANDIDATE)"
out="$("$VBIN/cross-lang-verify" --units "$MANIFEST" --no-confirm)"; rc=0
echo "$out" | grep -q "CANDIDATE" || { echo "ERROR: expected CANDIDATE verdicts" >&2; exit 1; }
echo "$out" | grep -q "DIVERGENT" && { echo "ERROR: --no-confirm must not report DIVERGENT" >&2; exit 1; }
echo "    smoke test OK (symbolic CANDIDATEs, no confirmed divergence)"

echo "==> [5/5] ground-truth run against real compilers (gated on toolchain)"
have_tools=1
command -v clang  >/dev/null 2>&1 || have_tools=0
command -v rustc  >/dev/null 2>&1 || have_tools=0
if [ "$have_tools" -eq 1 ]; then
  set +e
  out="$("$VBIN/cross-lang-verify" --units "$MANIFEST")"; rc=$?
  set -e
  echo "$out" | grep -q "DIVERGENT" || { echo "ERROR: expected confirmed DIVERGENT" >&2; echo "$out" >&2; exit 1; }
  if [ "$rc" -ne 1 ]; then
    echo "ERROR: expected exit 1 (CI gate) on confirmed divergence, got $rc" >&2; exit 1
  fi
  ndiv="$(echo "$out" | grep -c 'DIVERGENT')"
  echo "    confirmed $ndiv real divergence(s) against clang+UBSan + rustc; gate exit=1 OK"
else
  echo "    SKIPPED: clang/rustc not on PATH (packaging still verified above)"
fi

echo "packaging proof: PASSED"
