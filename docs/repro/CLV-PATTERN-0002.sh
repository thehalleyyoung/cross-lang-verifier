#!/usr/bin/env bash
# Reproduction for CLV-PATTERN-0002: Divide-by-zero divergence in translated rate/throughput computation
# Self-contained: compiles the C with UBSan and the target, then runs
# both on the witnessing input to exhibit the divergence.
set -euo pipefail
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cat > "$WORK/case.c" <<'CEOF'
#include <stdio.h>
#include <stdlib.h>
static int rate(int total,int count){return total/count;}
int main(int argc,char**argv){int t=atoi(argv[1]);int c=atoi(argv[2]);printf("%d\n",rate(t,c));return 0;}
CEOF
cat > "$WORK/case.go" <<'TEOF'
package main
import ("fmt";"os";"strconv")
func rate(total,count int)int{return total/count}
func main(){t,_:=strconv.Atoi(os.Args[1]);c,_:=strconv.Atoi(os.Args[2]);fmt.Println(rate(t,c))}
TEOF
echo "== C (UBSan) on witnessing input =="
clang -O0 -fsanitize=undefined -fno-sanitize-recover=all "$WORK/case.c" -o "$WORK/c_bin"
"$WORK/c_bin" 100 0 || echo "  (C trapped / nonzero: UB is reachable)"
echo "== target on the same input (defined) =="
(cd "$WORK" && go run case.go 100 0)
echo "== both on the safe input (should agree) =="
