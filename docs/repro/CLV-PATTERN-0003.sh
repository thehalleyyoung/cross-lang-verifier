#!/usr/bin/env bash
# Reproduction for CLV-PATTERN-0003: Oversized-shift divergence in translated packed-bitfield extraction
# Self-contained: compiles the C with UBSan and the target, then runs
# both on the witnessing input to exhibit the divergence.
set -euo pipefail
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cat > "$WORK/case.c" <<'CEOF'
#include <stdio.h>
#include <stdlib.h>
static int field(int v,int w){return v<<w;}
int main(int argc,char**argv){int v=atoi(argv[1]);int w=atoi(argv[2]);printf("%d\n",field(v,w));return 0;}
CEOF
cat > "$WORK/case.rs" <<'TEOF'
fn field(v:i32,w:u32)->i32{ v.wrapping_shl(w) }
fn main(){
  let v: i32 = std::env::args().nth(1).unwrap().parse().unwrap();
  let w: u32 = std::env::args().nth(2).unwrap().parse().unwrap();
  println!("{}", field(v,w));
}
TEOF
echo "== C (UBSan) on witnessing input =="
clang -O0 -fsanitize=undefined -fno-sanitize-recover=all "$WORK/case.c" -o "$WORK/c_bin"
"$WORK/c_bin" 1 40 || echo "  (C trapped / nonzero: UB is reachable)"
echo "== target on the same input (defined) =="
rustc -O "$WORK/case.rs" -o "$WORK/t_bin" 2>/dev/null && "$WORK/t_bin" 1 40
echo "== both on the safe input (should agree) =="
