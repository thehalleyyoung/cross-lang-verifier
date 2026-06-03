#!/usr/bin/env bash
# Reproduction for CLV-DIV-0001: coreutils-size-accumulate
# Evidence tier: confirmed extraction-unit finding, not an upstream CVE.
set -euo pipefail
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cat > "$WORK/case.c" <<'CEOF'
#include <stdint.h>

int32_t coreutils_size_accumulate(int32_t bytes) {
    return bytes + 4096;
}

#include <stdio.h>
#include <stdlib.h>
int main(int argc, char **argv) {
    if (argc < 2) return 2;
    int32_t bytes = (int32_t)strtol(argv[1], 0, 10);
    printf("%d\n", coreutils_size_accumulate(bytes));
    return 0;
}
CEOF
cat > "$WORK/case.rs" <<'REOF'
#[no_mangle]
pub extern "C" fn coreutils_size_accumulate(bytes: i32) -> i32 {
    bytes.wrapping_add(4096)
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    if argv.len() < 2 { std::process::exit(2); }
    let bytes: i32 = argv[1].parse().unwrap();
    println!("{}", coreutils_size_accumulate(bytes));
}
REOF
echo "== C extraction unit under UBSan on witness =="
clang -O1 -fsanitize=undefined -fno-sanitize-recover=all "$WORK/case.c" -o "$WORK/c_bin"
"$WORK/c_bin" 2147479552 || echo "  (C trapped / nonzero: UB is reachable)"
echo "== Rust extraction unit on the same witness =="
rustc -O "$WORK/case.rs" -o "$WORK/rs_bin"
"$WORK/rs_bin" 2147479552 || echo "  (Rust nonzero exit is checked by the target semantics pack)"
echo "== Safe input control =="
"$WORK/c_bin" 0
"$WORK/rs_bin" 0
