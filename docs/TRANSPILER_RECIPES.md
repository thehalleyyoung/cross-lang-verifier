# Transpiler Integration Recipes

The workflow the tool is built for is **"translate your C with `$tool`, then
verify the translation with us."** Integrations are *pluggable* so new
transpilers and new language pairs slot in without touching the oracle.

Implemented in `src/ub_oracle/transpiler_recipes.py`.

## The integration point — `Translator`

```python
class Translator(Protocol):
    name: str
    target_lang: str
    def available(self) -> bool: ...
    def translate(self, c_src: str, divergence_class: str) -> Optional[str]: ...
```

Two concrete kinds ship:

* **`ReferenceTranslator(target_lang)`** — a built-in, fully-compilable
  translator for the catalogued divergence classes (Rust/Go/Swift). It is the
  *known-good translation baseline* a recipe validates the pipeline against, and
  is always `available()`.
* **`ExternalCommandTranslator(name, target_lang, binary, arg_template)`** —
  shells out to a real transpiler binary, with `{in}`/`{out}` placeholders. It is
  **gated** on the binary being installed: when it is absent, `available()` is
  `False` and `translate(...)` returns `None` — it never fabricates output.

## Recipes are data

```python
RECIPES = (
  Recipe("reference-rust",  "rust",  ..., lambda: ReferenceTranslator("rust")),
  Recipe("reference-go",    "go",    ..., lambda: ReferenceTranslator("go")),
  Recipe("reference-swift", "swift", ..., lambda: ReferenceTranslator("swift")),
  Recipe("c2rust",          "rust",  ..., lambda: ExternalCommandTranslator(
              "c2rust", "rust", "c2rust", ("transpile", "{in}", "-o", "{out}"))),
  Recipe("llm-transpiler",  "rust",  ..., lambda: ExternalCommandTranslator(
              "llm-transpiler", "rust",
              os.environ.get("LLM_TRANSPILER", "llm-transpiler"), ("{in}", "{out}"))),
)
```

Adding a transpiler is appending a `Recipe`. Adding a language pair is adding a
reference generator (or pointing an `ExternalCommandTranslator` at a different
`target_lang`).

### c2rust

```
c2rust transpile your_unit.c -o out.rs    # produced by the recipe template
```

then the oracle is run on `out.rs`. Install c2rust to enable; without it the
recipe is reported `unavailable` (never fabricated).

### LLM transpilers

Point `ExternalCommandTranslator` at any CLI that reads a C file and writes
target source (`$LLM_TRANSPILER {in} {out}`), set `LLM_TRANSPILER`, and the same
`translate → verify` step applies.

## The end-to-end step

```python
from ub_oracle.transpiler_recipes import ReferenceTranslator, verify_transpiled
v = verify_transpiled(c_src, ReferenceTranslator("rust"),
                      ["10", "0"], "div_by_zero")
assert v.translated and v.diverged        # UB-rooted divergence flagged
```

`confirm_transpiler_recipes()` exercises every available reference translator
live (clang/UBSan + rustc/go/swiftc): the translate→verify step flags a
div-by-zero divergence on the UB input and stays silent on a safe input, proving
the recipe pipeline preserves the oracle's guarantees end to end. External
recipes are confirmed to be registered and correctly gated.
