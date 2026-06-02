# The shared semantic IR contract

This is the operational specification of the **translation unit** — the single,
frozen, language-pair-agnostic data shape that *every* frontend lowers into and
*every* divergence oracle consumes. Freezing this contract is what makes the
engine general: the C→Rust, C→Go and C→Swift pipelines differ only in the
`target_lang` they declare and the target program a pair-oracle emits — never in
the IR.

The rules below are **executable**: they are enforced by
`src/ub_oracle/ir.py::validate_unit` and rejected at the frontend boundary by the
CLI manifest loader. The validator is *sound about rejection* — it flags a
lowering only on a genuine contract violation, never on a merely
*uncovered-but-well-formed* unit (an unknown `kind` or an unsupported pair is
valid IR; the engine reports it `NOT_COVERED` rather than rejecting it).

## A unit is an effect-bearing fragment

Conceptually a unit denotes a small, typed, side-effecting source fragment
`f(args)` over integer / floating-point / array operands, together with the
**operating range** (a memory/effect precondition) under which it runs and the
language pair it was translated across.

## Envelope (all kinds)

| field | type | required | meaning |
| --- | --- | --- | --- |
| `kind` | string | **yes** | the structural operator family (see below) |
| `source_lang` | `"c"` (default) / known lang | no | source language of the fragment |
| `target_lang` | `"rust"` (default) / known lang | no | target language it was translated to |
| `probe` | divergence-class key | no | restrict the unit to a single class |
| `signed` | bool | no | integer signedness (default `true`) |
| `name` / `id` | string | no | human label used in diagnostics |

Known language tokens: `c`, `rust`, `go`, `swift`. A well-formed pair may still
be *unsupported* by the registered oracles — that is a coverage matter
(`NOT_COVERED`), not an IR error.

### Operating ranges (optional, per integer operand)

A unit may declare a closed interval precondition on each integer operand:
`x_range`, `a_range`, `b_range`, `shift_range`, each a `[lo, hi]` pair of
integers with `lo <= hi`. These are the memory/effect preconditions used both by
the abstract-interpretation pre-pass (`abstract_interp.py`, to prove a class's
UB unreachable and skip SMT) and by the oracles' SMT searches (so a reported
witness is real *under the declared range*). Omitting a range means the operand
is unconstrained over its full signed type.

## Operator families (`kind`)

| `kind` | required operands | optional | classes |
| --- | --- | --- | --- |
| `binop_const` | `op` (string), `const` (int) | `width∈{32,64}`, `var`, `x_range` | `signed_overflow` |
| `shift` | — | `width∈{32,64}`, `var`, `shift_var`, `value`, `shift_range` | `shift_oob` |
| `div` / `rem` | — | `width∈{32,64}`, `a`, `b`, `dividend`, `a_range`, `b_range` | `div_by_zero`, `intmin_div_neg1` |
| `array_index` | `length` (int > 0) | `index_var` | `array_oob` |
| `type_pun` | — | — | `strict_aliasing` |
| `fp_fma` | — | — | `fp_contraction` |

For the four integer kinds (`binop_const`, `shift`, `div`, `rem`) `width`
defaults to `32` and must be one of `{32, 64}`.

## What the validator rejects

`validate_unit(unit)` returns a list of `IRError(field, message)`; an empty list
means valid. `assert_valid(unit)` raises `IRValidationError` on any violation.
Rejected cases:

* the unit is not a JSON object, or `kind` is missing / non-string;
* an integer kind declares an unsupported `width`;
* `binop_const` omits `op` or `const`, or mistypes them;
* `array_index` omits `length` or declares a non-positive length;
* any `*_range` is not a `[lo, hi]` integer pair, or has `lo > hi`;
* `source_lang` / `target_lang` is a non-string or an unknown language token;
* `probe` names a divergence class not in the catalogue;
* `signed` is non-boolean.

Pass `require_known_kind=True` to additionally reject a `kind` outside the
shipped set (used where the engine wants to refuse forward-compatible kinds it
cannot yet lower).

## Enforcement points

* **CLI** (`cross-lang-verify`): every manifest unit is validated on load;
  ill-formed lowerings are reported and the run exits non-zero. Pass
  `--no-validate` to bypass (e.g. when intentionally probing engine behavior on
  raw input).
* **Library**: call `assert_valid(unit)` (or `is_valid(unit)`) before handing a
  unit to `verify_unit`.
