# Frontends — robust, real source ingestion + the frontend SPI

Hand-rolled parsers are fine for toy inputs but lose on real code: they
re-derive (badly) facts the language's own toolchain already computes exactly.
This project therefore puts **grammar-backed and compiler-backed frontends** on
the supported path and exposes a small **frontend SPI** so adding a language is a
bounded, documented task.

Implemented in `src/ub_oracle/frontends.py`.

## The SPI

```python
@runtime_checkable
class Frontend(Protocol):
    name: str
    language: str                       # "c", "rust", ...
    def available(self) -> bool: ...    # honest gating on its toolchain
    def ingest(self, src: str) -> Optional[IRModule]: ...
```

`ingest` returns the **shared** `ir_ingest.IRModule` (a `{name -> IRFunction}`
table with return type, parameters, and storage class), so every frontend feeds
the *same* downstream oracle. Adding a language frontend is: implement the
protocol, populate an `IRModule`, append it to `FRONTENDS`.

## Registered frontends

| name | language | parser / IR | role |
|------|----------|-------------|------|
| `treesitter-c` | C | **tree-sitter** grammar | supported parser for real code |
| `clang-ast-c` | C | clang AST (JSON) | highest-fidelity ground truth |
| `rustc-mir-rust` | Rust | rustc MIR (post-borrow-check) | ownership facts for free |

`frontends_for("c")` returns the C frontends; `get_frontend(name)` resolves one
by name (raising on unknown).

## Why tree-sitter for the supported path

[tree-sitter](https://tree-sitter.github.io/) is the same incremental,
error-tolerant parser editors use. It robustly handles the constructs that
defeat hand-rolled parsers — nested and pointer/array declarators, `(void)`
parameter lists, storage-class specifiers, `const`/qualified parameter types,
K&R-ish spacing — and is available as a small pip dependency
(`pip install -e ".[frontends]"`). When it is not installed the frontend reports
`available() is False` and is skipped; it never fabricates a parse.

## Cross-validation against the compiler

A frontend is only trustworthy if it agrees with the language's own toolchain.
`confirm_real_frontends()` parses several real-world-shaped C translation units
with **both** the tree-sitter frontend and the **clang AST** frontend and
requires the extracted function tables to **agree** — same function names, same
arities, same positional parameter names, same storage class — with the clang
AST as ground truth. On this machine all samples agree (8 functions across 5
units). This is the machine-checked claim `C48` behind the tree-sitter frontend.
