# Pointer-provenance divergence class

> Claim **C62-pointer-provenance-divergence** · module `src/ub_oracle/oracles/pointer_provenance.py`

A C pointer may only be computed within the array object it points into, plus
the one-past-the-end position. Its **provenance** is that object:

```c
int *p = a;
p = p + n;        /* defined only while p stays in a[0..len] */
```

When `n` pushes the result out of the object — and, in the limit, when the
address-space computation `n * sizeof(T)` **overflows** — the behavior is
**undefined** (C17 §6.5.6p8). The optimiser is entitled to assume the offset
stays in range, and real builds behave wildly:

- `clang -O0` forms a wild pointer and derives a garbage value from it;
- `clang -O1 -fsanitize=undefined` (which includes the `pointer-overflow` check)
  **traps** deterministically with
  `runtime error: ... offset ... overflowed to ...`.

## The divergence

The idiomatic *safe* translations never form a raw out-of-provenance pointer.
They keep an **index** and access through a **checked** operation, so a far-out
offset becomes a deterministic, defined value:

| Side | Code (sketch) | On `n = 2^62` |
| --- | --- | --- |
| C | `int *p = a; p = p + n; use(p - a);` | **UB** — UBSan `pointer-overflow` trap |
| Rust | `*a.get(n as usize).unwrap_or(&-1)` | `-1`, `rc = 0` (defined) |
| Go | `if n < 0 \|\| n >= len(a) { return -1 }; a[n]` | `-1`, `rc = 0` (defined) |

This is a textbook **`trap_vs_defined`** divergence: on the *same* concrete input
the C side is undefined while the target side is defined and deterministic.

## How the oracle works

`PointerProvenanceOracle` (C→Rust) and `GoPointerProvenanceOracle` (C→Go) both:

1. **Find** the witness with Z3 — the *least* offset `n` whose byte displacement
   `n * sizeof(T)` is guaranteed to overflow a 64-bit address space irrespective
   of the (ASLR-randomised, run-time) base address, i.e. `n * sizeof(T) >= 2**64`
   (so `n = 2**62` for 4-byte ints), honouring any declared `offset_range`.
2. **Emit** a C program that forms `a + n` and *uses* the resulting pointer (so
   it cannot be optimised away), plus the safe target port.
3. **Confirm** end-to-end through the real toolchain: the UBSan build must trap
   via `pointer-overflow` and the target binary must produce a defined,
   deterministic outcome.
4. **Certify** the recorded evidence through the Lean product-program theorem:
   `pointer_provenance_oracle_sound` proves that a compiler-confirmed
   `trap_vs_defined` witness is a genuine UB-rooted divergence in the
   recorded-observable abstraction.

```python
from ub_oracle import oracles                      # registers plugins
from ub_oracle.plugin import get_oracle_for
from ub_oracle.reexec import ReexecHarness

orc = get_oracle_for("pointer_provenance", "c", "rust")
res = orc.find_divergence({"kind": "pointer_offset", "width": 32})
rr  = orc.confirm(res, ReexecHarness()).reexec
assert rr.confirmed                                # clang/UBSan traps; Rust defined
```

Both pairs are validated against real compilers in
`tests/test_ub_oracle.py::test_pointer_provenance_*`.
