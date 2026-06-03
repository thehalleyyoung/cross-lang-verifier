#!/usr/bin/env bash
set -euo pipefail

repo_root="${GITHUB_WORKSPACE:-$PWD}"
workdir="${CLV_WORKING_DIRECTORY:-.}"

abs_path() {
  local base="$1"
  local path="$2"
  if [[ "$path" = /* ]]; then
    printf '%s\n' "$path"
  else
    printf '%s/%s\n' "$base" "$path"
  fi
}

write_output() {
  local key="$1"
  local value="$2"
  if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    printf '%s=%s\n' "$key" "$value" >> "$GITHUB_OUTPUT"
  fi
}

write_empty_sarif() {
  local path="$1"
  mkdir -p "$(dirname "$path")"
  printf '%s\n' \
    '{"$schema":"https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json","version":"2.1.0","runs":[{"tool":{"driver":{"name":"cross-lang-verifier","informationUri":"https://github.com/thehalleyyoung/cross-lang-verifier","rules":[]}},"results":[]}]}' \
    > "$path"
}

workdir_abs="$(abs_path "$repo_root" "$workdir")"
if [[ ! -d "$workdir_abs" ]]; then
  printf 'translation-equivalence-guard: working-directory not found: %s\n' "$workdir_abs" >&2
  sarif_abs="$(abs_path "$repo_root" "${CLV_SARIF:-cross-lang-verify.sarif}")"
  write_empty_sarif "$sarif_abs"
  write_output "sarif" "$sarif_abs"
  write_output "exit_code" "2"
  exit 2
fi

manifest="${CLV_MANIFEST:?CLV_MANIFEST is required}"
sarif="${CLV_SARIF:-cross-lang-verify.sarif}"
manifest_abs="$(abs_path "$workdir_abs" "$manifest")"
sarif_abs="$(abs_path "$workdir_abs" "$sarif")"

mkdir -p "$(dirname "$sarif_abs")"
if [[ ! -f "$manifest_abs" ]]; then
  printf 'translation-equivalence-guard: manifest not found: %s\n' "$manifest_abs" >&2
  write_empty_sarif "$sarif_abs"
  write_output "sarif" "$sarif_abs"
  write_output "exit_code" "2"
  exit 2
fi

fail_on="${CLV_FAIL_ON:-divergent}"
fail_tokens=()
read -r -a fail_tokens <<< "$fail_on"
fail_flags=()
for verdict in "${fail_tokens[@]}"; do
  [[ -n "$verdict" ]] && fail_flags+=(--fail-on "$verdict")
done

extra="${CLV_EXTRA_ARGS:-}"
extra_flags=()
if [[ -n "$extra" ]]; then
  read -r -a extra_flags <<< "$extra"
fi

no_confirm="${CLV_NO_CONFIRM:-false}"
confirm_flags=()
case "$no_confirm" in
  true|TRUE|1|yes|YES) confirm_flags+=(--no-confirm) ;;
  false|FALSE|0|no|NO|"") ;;
  *)
    printf 'translation-equivalence-guard: no-confirm must be true/false, got %s\n' "$no_confirm" >&2
    write_empty_sarif "$sarif_abs"
    write_output "sarif" "$sarif_abs"
    write_output "exit_code" "2"
    exit 2
    ;;
esac

cmd=()
if [[ -n "${CLV_PYTHON_MODULE:-}" ]]; then
  cmd=("${CLV_PYTHON:-python3}" -m "$CLV_PYTHON_MODULE")
else
  cmd=("${CLV_CLI:-cross-lang-verify}")
fi

set +e
set +u
(
  cd "$workdir_abs"
  "${cmd[@]}" \
    --units "$manifest_abs" \
    --sarif "$sarif_abs" \
    --color "${CLV_COLOR:-always}" \
    "${confirm_flags[@]}" \
    "${fail_flags[@]}" \
    "${extra_flags[@]}"
)
code=$?
set -e
set -u

if [[ ! -f "$sarif_abs" ]]; then
  write_empty_sarif "$sarif_abs"
fi

write_output "sarif" "$sarif_abs"
write_output "exit_code" "$code"

if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
  {
    printf '### Translation Equivalence Guard\n\n'
    printf '* Manifest: `%s`\n' "$manifest_abs"
    printf '* SARIF: `%s`\n' "$sarif_abs"
    printf '* Exit code: `%s`\n' "$code"
  } >> "$GITHUB_STEP_SUMMARY"
fi

exit "$code"
