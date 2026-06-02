# Gallery of caught divergences

*Auto-generated from the in-repo corpora (`ub_oracle.idiomatic_corpus`, `ub_oracle.multipair_corpus`) — do not edit by hand; run `python -m ub_oracle.docs_site`.*

Each row is a real, compilable C function and its translation(s); the **verdict** is what the oracle proves live against clang/UBSan + the target compiler. *Divergent* rows are UB-rooted bugs the oracle flags; *equivalent* rows are safe translations it stays silent on.

## Tier-2 — idiomatic anchors (one target each)

| id | mirrors | class | target(s) | verdict |
|----|---------|-------|-----------|---------|
| `midpoint-overflow` | binary-search / merge midpoint `(lo+hi)/2` (the JDK / NIST-famous signed-overflow bug); idiomatic ports use wrapping (Rust) or 64-bit `int` (Go), so they are defined where C is UB. | `signed_overflow` | go, rust | **divergent** |
| `bitfield-shift` | bit-field / flag extraction `v << w` (as in packed-struct decoders); a width >= 32 is out-of-range UB in C, but Rust's `wrapping_shl` is defined. | `oversized_shift` | rust | **divergent** |
| `rate-divide` | throughput/rate `total/count` (as in coreutils-style accounting); a zero divisor is UB in C, a defined panic in Rust and a defined panic in Go. | `div_by_zero` | go, rust | **divergent** |
| `safe-average` | overflow-safe average widening to 64 bits before halving; well-defined on both sides — the idiomatic fix for the midpoint bug. | `none` | go, rust | **equivalent** |
| `clamp-byte` | saturating clamp to [0,255] (pixel/byte saturation); no UB on either side, must never be flagged. | `none` | go, rust | **equivalent** |
| `additive-checksum` | additive checksum mod 256 using unsigned arithmetic (Internet-checksum shaped); well-defined wrap-around on both sides. | `none` | go, rust | **equivalent** |

## Tier-3 — multi-pair (every target at once)

| id | mirrors | class | pairs | verdict |
|----|---------|-------|-------|---------|
| `midpoint` | binary-search/merge midpoint (lo+hi)/2 signed-overflow bug | `signed_overflow` | go, rust, swift | **divergent** |
| `rate` | coreutils-style throughput total/count (zero divisor) | `div_by_zero` | go, rust, swift | **divergent** |
| `bitfield` | packed-struct bit-field extraction v<<w (out-of-range shift) | `oversized_shift` | go, rust, swift | **divergent** |
| `clamp` | saturating clamp to [0,255] (no UB on any side) | `none` | go, rust, swift | **equivalent** |
| `checksum` | additive mod-256 checksum (well-defined wrap-around) | `none` | go, rust, swift | **equivalent** |

*6 catalogued UB-rooted divergences across 11 functions. Every verdict is reproduced live by the test-suite and the traceability check.*
