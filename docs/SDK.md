# Plugin SDK — adding a divergence class without forking the engine

The verifier is a **registry of divergence oracles** behind one uniform contract.
Every divergence class — signed overflow, out-of-range shift, OOB indexing,
FP contraction, … — is an independent plugin. New classes (and new language
pairs) are *additive*: you depend on the public SPI, you do **not** patch the
engine.

This document is the contract. A complete, runnable reference plugin lives at
[`examples/plugins/float_cast_overflow_oracle.py`](../examples/plugins/float_cast_overflow_oracle.py),
and is exercised end-to-end (including real-compiler confirmation) by the tests
named `test_external_plugin_*` in `tests/test_ub_oracle.py`.

## The SPI

Subclass `src.ub_oracle.plugin.DivergenceOracle` and implement two methods:

```python
from src.ub_oracle.plugin import DivergenceOracle, OracleResult, OracleVerdict, register

class MyOracle(DivergenceOracle):
    divergence_class = "my_class"      # stable key you own
    source_lang = "c"                  # the pair this plugin is validated on
    target_lang = "rust"
    confirmation_mode = "trap_vs_defined"  # how the harness confirms (see below)

    def applies_to(self, unit: dict) -> bool:
        ...                            # is this unit relevant to my class?

    def find_divergence(self, unit: dict) -> OracleResult:
        ...                            # search (symbolically!) for a witness

register(MyOracle())                   # importing your module wires it in
```

`register()` appends your oracle to the pair-agnostic backbone `ALL_ORACLES`
and, for the `c→rust` anchor pair, to the legacy `REGISTRY`. After import,
`verify_unit`, `oracles_for(...)`, and `get_oracle_for(...)` discover and run it
automatically — no engine edit, no fork.

### `applies_to(unit) -> bool`

Return `True` only for units your class understands. Honour an optional
`unit["probe"]` (a class key) so a unit can pin itself to a single class, and
respect the declared `source_lang`/`target_lang` (the engine already filters by
pair before calling you).

### `find_divergence(unit) -> OracleResult`

**Search, don't hard-code.** The reference plugin uses Z3's floating-point
theory to *find* a double whose `(int)` cast overflows; the core oracles use Z3
bitvector theory. A fixture is not an oracle — the value of this project is that
the witness is discovered from first principles and would be found even where a
human could not.

Return an `OracleResult` with:

* `verdict` — `DIVERGENT` (witness found), `NO_DIVERGENCE_FOUND`,
  `NOT_APPLICABLE`, or `UNKNOWN` (solver abstained).
* `divergence_class` — your key.
* `counterexample` — a `src.ub_oracle.replay.Counterexample` carrying the
  concrete inputs **and** self-contained, compilable `source_snippet` /
  `target_snippet` so the witness is re-executable.

### Confirmation modes

`DivergenceOracle.confirm()` is provided for you; you only pick a
`confirmation_mode` matching how your divergence becomes observable:

| `confirmation_mode`    | What the ground-truth harness checks |
|------------------------|--------------------------------------|
| `trap_vs_defined`      | C is UB on a defined input (UBSan **traps**) while the target is defined & deterministic. *(used by the reference plugin: `(int)1e30` traps; Rust `as i32` saturates)* |
| `exploited`            | The UB **flips an observable value** across `-O0` vs `-O2`. |
| `optimizer_exploited`  | The same C source yields different output under two conforming compilations (set `optimizer_flag_variants`, e.g. `-ffp-contract=off` vs `=fast`); the target is deterministic. |

Confirmation compiles and runs **real** `clang` (under
`-fsanitize=undefined -fno-sanitize-recover=all`) and the real target compiler.
Only a confirmed witness is ever reported as `DIVERGENT`; an unconfirmable
symbolic witness is downgraded to `CANDIDATE` (sound-for-divergence policy).

## Adding a new language pair

A divergence class is pair-parametric. To target a second language, set your
`target_lang` (e.g. `"go"`/`"swift"`) and emit the target snippet in that
language; the harness's `target_lang` plumbing and the per-pair
`target_semantics` packs handle compilation and the defined-outcome check.
Declaring an unsupported pair on a unit honestly yields `NOT_COVERED` rather
than being silently treated as `c→rust`.

## Checklist

1. Subclass `DivergenceOracle`; set `divergence_class`, the language pair, and
   `confirmation_mode`.
2. Implement `applies_to` and a *searching* `find_divergence`.
3. Emit a re-executable `Counterexample` (concrete inputs + compilable snippets).
4. Call `register(YourOracle())` at import.
5. Import your module → `verify_unit` finds and confirms it. Done.
