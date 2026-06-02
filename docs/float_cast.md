# Float-cast-overflow divergence class

> Claim **C59-float-cast-overflow-divergence** · module `src/ub_oracle/oracles/float_cast.py`

A C narrowing conversion from a floating type to an integer type sizes nothing
and looks innocent:

```c
int g(double x) { return (int)x; }   /* x is rounded toward zero, then narrowed */
```

When the **rounded** value does not fit the destination integer type, the
behavior is **undefined** (C17 §6.3.1.4p1). Real builds bear this out:

- `clang -O0` returns a target-specific garbage value (on x86‑64, often
  `INT_MIN`).
- `clang -fsanitize=undefined` (which includes the `float-cast-overflow` check)
  **traps** deterministically:
  `runtime error: 2.14748e+09 is outside the range of representable values of type 'int'`.

The idiomatic *safe* translation keeps the cast — and the cast is **defined**:

- **Rust** `x as i32` is a **saturating** cast (defined since Rust 1.45): the
  out-of-range value clamps to `i32::MAX` / `i32::MIN`. A single deterministic
  value, no UB.
- **Go** `int32(x)` is defined: on overflow the result is *implementation
  specified* (Go spec, "Conversions"), but it never traps and is deterministic
  for a given build.

So on the **same** out-of-range input the C program is undefined while the safe
target is defined and deterministic — exactly the `trap_vs_defined` shape the
ground-truth harness confirms against real `clang`/UBSan + `rustc`/`go`.

## The conversion lattice — why this is the only divergent corner

Step 106 asks for the full conversion catalogue. The point of the catalogue is
that **most** of it does *not* diverge on mainstream two's-complement targets;
only the float→int UB corner does. Each row is backed by the C standard text:

| Conversion | C status | Rust / Go `as` | Diverges on real targets? |
|---|---|---|---|
| signed → unsigned (out of range) | **defined**, modular (§6.3.1.3p2) | same modular value | **no** |
| unsigned → signed (out of range) | implementation-defined (§6.3.1.3p3) | two's-complement wrap | **no** (every 2's-complement target) |
| right shift of a negative integer | implementation-defined (§6.5.7p5) | arithmetic shift | **no** (every mainstream target) |
| **float → int, rounded value out of range** | **undefined** (§6.3.1.4p1) | Rust saturates / Go impl-specified, both **defined** | **yes — witnessed, UBSan traps** |

The oracle therefore implements the one corner that genuinely produces a
cross-language divergence, and documents the rest as provably non-divergent.

## Witness search

The witnessing value is *found* with Z3, not hard-coded: it is the
least-extreme integer just past the destination maximum (e.g. `INT_MAX + 1 =
2147483648` for `int`, `LLONG_MAX + 1 = 9223372036854775808` for `long long`),
optionally constrained by a declared `value_range`. The least-extreme witness
keeps the counterexample clean and exactly reproducible.

## Soundness

The oracle is **sound for divergence**: it only emits `DIVERGENT` when it has a
concrete value on which the C cast is UB (so the source is undefined) while the
target conversion is defined. The structural witness check is unconditional; the
real-compiler confirmation runs whenever the toolchain is present, so the
machine-checked theorem `_thm_float_cast_overflow_oracle` is total.
