#!/usr/bin/env bash
# Reproduction for CLV-PATTERN-0001: Signed-overflow divergence in translated binary-search midpoint
# Self-contained: compiles the C with UBSan and the target, then runs
# both on the witnessing input to exhibit the divergence.
set -euo pipefail
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cat > "$WORK/case.c" <<'CEOF'
#include <stdio.h>
#include <stdlib.h>
static int midpoint(int lo,int hi){return (lo+hi)/2;}
int main(int argc,char**argv){int lo=atoi(argv[1]);int hi=atoi(argv[2]);printf("%d\n",midpoint(lo,hi));return 0;}
CEOF
cat > "$WORK/case.rs" <<'TEOF'
fn midpoint(lo:i32,hi:i32)->i32{ lo.wrapping_add(hi) / 2 }
fn main(){
  let lo: i32 = std::env::args().nth(1).unwrap().parse().unwrap();
  let hi: i32 = std::env::args().nth(2).unwrap().parse().unwrap();
  println!("{}", midpoint(lo,hi));
}
TEOF
echo "== C (UBSan) on witnessing input =="
clang -O0 -fsanitize=undefined -fno-sanitize-recover=all "$WORK/case.c" -o "$WORK/c_bin"
"$WORK/c_bin" 2000000000 2000000000 || echo "  (C trapped / nonzero: UB is reachable)"
echo "== target on the same input (defined) =="
rustc -O "$WORK/case.rs" -o "$WORK/t_bin" 2>/dev/null && "$WORK/t_bin" 2000000000 2000000000
echo "== both on the safe input (should agree) =="
