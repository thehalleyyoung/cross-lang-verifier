# cross-lang-verifier

A **C→{Rust, Go, Swift} cross-language semantic-divergence oracle** rooted in C
**undefined behaviour**. When you migrate C to a memory-safe language — by hand,
with [c2rust](https://github.com/immunant/c2rust), or with an LLM — the
translation can silently change behaviour exactly where the original C relied on
undefined behaviour. This tool **catches those divergences and proves them
against real compiled code**: it actually compiles and runs both programs
(clang/UBSan + the target compiler) and reports a divergence only with a concrete
witnessing input.

## What makes it different

- **No mocked results.** Every verdict is backed by a real compile-and-run. A
  built-in *credibility guard* rejects any simulated oracle.
- **Three real target languages** (`rust`, `go`, `swift`) behind one
  language-pair-agnostic oracle and a [frozen shared-IR contract](IR.md).
- **Soundness you can check.** [Per-pair soundness statements](COMPLETENESS.md),
  a [mechanized Lean proof](MECHANIZED_SOUNDNESS.md) of the decision procedure,
  and a [claim→proof traceability](TRACEABILITY.md) matrix where every published
  claim is tied to a live, machine-checked theorem.
- **Built to integrate.** [Transpiler recipes](TRANSPILER_RECIPES.md)
  (*translate with `$tool`, then verify with us*), a [pre-commit hook](pre_commit.md),
  an interactive [web playground](PLAYGROUND.md), and a [plugin SDK](SDK.md).

## Start here

- New here? Read the [tutorial](tutorial.md).
- Want to see real bugs it catches? The [gallery](gallery.md) is auto-generated
  from the live corpora.
- Curious how soundness is established? Start with the
  [product-program oracle](PRODUCT_PROGRAM.md).
