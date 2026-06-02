# VLA-bound divergence class

> Claim **C58-vla-bound-divergence** · module `src/ub_oracle/oracles/vla_bound.py`

A C **variable-length array** sizes a stack allocation from a runtime expression:

```c
T a[n];   /* n is evaluated at runtime */
```

When that size expression `n` is **not a positive value**, the behavior is
**undefined** (C17 §6.7.6.2p5). This is not a theoretical corner: the optimiser
is entitled to assume it never happens, and real builds behave wildly:

- `clang -O0` reserves a frame from a wild, negatively-sized computation → the
  program **segfaults**;
- `clang -O1 -fsanitize=undefined` (which includes the `vla-bound` check)
  **traps** deterministically with
  `runtime error: variable length array bound evaluates to non-positive value`.

## The divergence

The idiomatic *safe* translations have **no VLA at all** — a dynamically-sized
buffer becomes a heap container whose length comes from a **checked** conversion
of the signed input. A negative bound is therefore not undefined; it is a
deterministic, defined **panic**:

| Side | Code (sketch) | On `n = -1` |
| --- | --- | --- |
| C | `T a[n];` | **UB** — UBSan `vla-bound` trap / `-O0` segfault |
| Rust | `if n < 0 { panic!() } let a = vec![0; n as usize];` | panic, `rc = 101` (defined) |
| Go | `a := make([]T, n)` | `makeslice` panic, `rc = 2` (defined) |

This is a textbook **`trap_vs_defined`** divergence: on the *same* concrete input
the C side is undefined while the target side is defined and deterministic.

## How the oracle works

`VlaBoundOracle` (C→Rust) and `GoVlaBoundOracle` (C→Go) both:

1. **Find** the witness with Z3 — the *least extreme* non-positive bound (closest
   to zero, i.e. `-1`) consistent with any declared `bound_range`, so the search
   honours an AI pre-pass's constraints rather than hard-coding `-1`.
2. **Emit** a C program whose hot function declares `T a[n]` and *uses every
   element* (so the array cannot be optimised away), plus the safe target port.
3. **Confirm** end-to-end through the real toolchain: the UBSan build must trap
   and the target binary must produce a defined, deterministic outcome.

```python
from ub_oracle import oracles                      # registers plugins
from ub_oracle.plugin import get_oracle_for
from ub_oracle.reexec import ReexecHarness

orc = get_oracle_for("vla_bound", "c", "go")
res = orc.find_divergence({"kind": "vla", "width": 32})
rr  = orc.confirm(res, ReexecHarness()).reexec
assert rr.confirmed                                # clang/UBSan traps; Go panics rc=2
```

Both pairs are validated against real compilers in
`tests/test_ub_oracle.py::test_vla_*`.
