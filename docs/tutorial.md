# Tutorial — catch your first divergence

This walks through catching a real UB-rooted divergence between a C function and
its translation, end to end against real compilers.

## Prerequisites

- `clang` (with UBSan) and the target compiler for the pair you want
  (`rustc`, `go`, or `swiftc`).
- The repo checked out with its `venv` (`pip install -r requirements.txt`).

The tool degrades **honestly**: if a toolchain is missing, the oracle reports
`available=false` rather than guessing.

## 1. The bug: integer division by zero

C division by zero is **undefined behaviour**. A faithful-looking Rust
translation is *defined* (it panics deterministically), so on the input that
makes the divisor zero the two programs diverge.

```c
// rate.c
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char **argv) {
  int a = atoi(argv[1]);
  int b = atoi(argv[2]);
  printf("%d\n", a / b);
  return 0;
}
```

```rust
// rate.rs
use std::env;
fn main() {
  let a: i32 = env::args().nth(1).unwrap().parse().unwrap();
  let b: i32 = env::args().nth(2).unwrap().parse().unwrap();
  println!("{}", a / b);
}
```

## 2. Ask the oracle

```python
from ub_oracle.playground import evaluate

c   = open("rate.c").read()
rs  = open("rate.rs").read()

bug  = evaluate(c, rs, ["10", "0"], "division_by_zero", "rust")
safe = evaluate(c, rs, ["10", "2"], "division_by_zero", "rust")

print(bug.diverged)   # True  — UB reachable in C, defined in Rust
print(safe.diverged)  # False — well-defined on both sides
print(bug.summary)
```

The oracle compiles both programs, runs them on the concrete input, observes
that the C UBSan build traps while Rust is defined, and reports the divergence
**with the witnessing input `["10","0"]`**. On the safe input it stays silent.

## 3. Try it in the browser

```bash
python -m ub_oracle.playground   # http://127.0.0.1:8000/
```

Paste the two sources, pick the target language from the dropdown, enter the
inputs, and hit **verify**. See the [playground docs](PLAYGROUND.md).

## 4. Verify a transpiler's output

Already using c2rust or an LLM transpiler? Wrap it as a
[recipe](TRANSPILER_RECIPES.md) — *translate with `$tool`, then verify with us* —
and the same oracle checks its output.

## 5. Where to go next

- The [gallery](gallery.md) of catalogued divergences (auto-generated from the
  live corpora).
- The [per-pair soundness statements](COMPLETENESS.md) and the
  [mechanized Lean proof](MECHANIZED_SOUNDNESS.md).
- The [claim→proof traceability matrix](TRACEABILITY.md).
