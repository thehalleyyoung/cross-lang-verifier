# Restrict-violation divergence class

> Claim **C61-restrict-violation-divergence** · module `src/ub_oracle/oracles/restrict_alias.py`

A `restrict`-qualified pointer is a **promise** to the compiler that, while the
pointer is live, the object it points at is accessed *only* through that pointer.
If two `restrict` pointers actually alias and at least one access is a store, the
behavior is **undefined** (C17 §6.7.3.1p4) — and the optimizer *relies* on the
promise:

```c
static int f(int *restrict a, int *restrict b){
    *a = 1;
    *b = 2;            /* if a==b this is UB; -O2 assumes a,b never alias */
    return *a + *b;    /* -O0 re-reads (=4 aliased); -O2 caches *a=1 (=3) */
}
```

Call `f` with two pointers to the **same** object and the result depends only on
the optimisation level:

```
$ clang -O0 r.c -o o0 && ./o0 1     # 4   (honest re-read of aliased memory)
$ clang -O2 r.c -o o2 && ./o2 1     # 3   (restrict lets -O2 cache *a = 1)
$ clang -O1 -fsanitize=undefined r.c -o san && ./san 1   # 3, exit 0 — no trap
```

No sanitizer traps a `restrict` violation: the evidence is the
optimisation-level disagreement itself (`4` vs `3`).

The idiomatic *safe* translation cannot reproduce the hazard:

- **Rust** `fn f(a: &mut i32, b: &mut i32)` — two live `&mut` to one object are
  rejected by the borrow checker, so the safe port is *non-aliasing by
  construction* and returns a single deterministic value (`3`).
- **Go** has no `restrict` qualifier and performs no restrict-based rewrite, so
  its pointer code is defined and deterministic whether or not it aliases.

This is exactly the `optimizer_exploited` shape the ground-truth harness confirms
against real `clang` (`-O0` vs `-O2`) + `rustc`/`go`.

## Witness search

The witnessing **selector** — the input that makes the two pointers alias — is
*found* with Z3 (the least-extreme non-zero selector, honouring any declared
range) rather than hard-coded, so the search honours an AI pre-pass's
constraints just like the other oracles.

## Soundness

The oracle is **sound for divergence**: it only emits `DIVERGENT` with a concrete
aliasing selector, and the real-compiler confirmation requires the *same* C
source to disagree across `-O0`/`-O2` while the safe target is defined and
deterministic. The structural witness is unconditional; the confirmation runs
whenever the toolchain is present, so the machine-checked theorem
`_thm_restrict_violation_oracle` is total.
