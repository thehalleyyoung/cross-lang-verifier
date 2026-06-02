# Fast-math reassociation divergence class

> Claim **C60-fast-math-reassociation-divergence** · module `src/ub_oracle/oracles/fast_math.py`

IEEE-754 floating-point arithmetic is **not associative**: rounding happens at
every step, so `(x + y) - x` is *not* algebraically equal to `y`. A standard,
conforming C compilation honours this:

```c
double g(double x, double y){ return (x + y) - x; }
```

For a large `x` that *swallows* a small `y` (i.e. `round(x + y) == x`), the
IEEE-strict result is exactly `0`. But under `-ffast-math` / `-Ofast` the
compiler is licensed to **reassociate** floating arithmetic as if it were over
the reals — it may fold `(x + y) - x` straight to `y`. Real builds bear this out:

```
$ clang -O2 -fno-fast-math g.c -o strict && ./strict -9011734740205572.0 1.0
0
$ clang -O2 -ffast-math    g.c -o fast   && ./fast   -9011734740205572.0 1.0
1
```

The **same** C source, on the **same** input, prints `0` or `1` depending only
on a compiler flag: the value is not fixed by the program.

The idiomatic *safe* translation never auto-reassociates — Rust and Go both
evaluate `(x + y) - x` IEEE-strict (fusion/reassociation only via explicit
intrinsics such as `f64::mul_add`):

```
$ ./g_rust -9011734740205572.0 1.0   # 0  (deterministic)
$ ./g_go   -9011734740205572.0 1.0   # 0  (deterministic)
```

So on the same input the C result is under-determined (it flips with `-Ofast`)
while the safe target is a single deterministic, defined value — exactly the
`optimizer_exploited` shape the ground-truth harness confirms against real
`clang` (`-fno-fast-math` vs `-ffast-math`) + `rustc`/`go`. No sanitizer is
needed or able to trap a `-ffast-math` value flip; the optimisation-flag
disagreement *is* the evidence.

## Witness search

The witnessing pair `(x, y)` is *found* with Z3's floating-point theory, not
hard-coded: it requires `round(x + y) == x` (so IEEE-strict `(x + y) - x` rounds
to `0`) together with `y != 0` (so the reassociated value is a non-zero `y`),
with both operands constrained to finite normals in a clean printable range so
the decimals round-trip identically through C `strtod`, Rust's `f64` parser and
Go's `strconv.ParseFloat`. The "y is entirely swallowed" witness maximises the
visible gap (`0` vs `y`), making the divergence robust and reproducible.

## Soundness

The oracle is **sound for divergence**: it only emits `DIVERGENT` when it has a
concrete `(x, y)` on which IEEE-strict evaluation yields `0` while the
reassociated value is non-zero, so the source result genuinely depends on the
(non-conforming) `-ffast-math` licence while the target is IEEE-deterministic.
The structural witness check (`round(x+y)==x`, `y!=0`) is unconditional; the
real-compiler confirmation runs whenever the toolchain is present, so the
machine-checked theorem `_thm_fast_math_reassoc_oracle` is total.
