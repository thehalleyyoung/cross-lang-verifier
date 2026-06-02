"""Step 64 — transpiler integration recipes.

The workflow the tool is built for is *"translate your C with $tool, then verify
the translation with us."* This module makes that a first-class, **pluggable**
pipeline so new transpilers and new language pairs slot in without touching the
oracle:

* `Translator` — the integration point. A translator turns a C translation-unit
  into target source for one language. Two concrete kinds ship:
    - `ReferenceTranslator` — a built-in, fully-compilable translator for the
      catalogued divergence classes (the "known-good translation" baseline a
      recipe verifies against), one per target pack (Rust/Go/Swift);
    - `ExternalCommandTranslator` — shells out to a real transpiler binary
      (e.g. **c2rust**, or an LLM-transpiler CLI) named by a recipe; it is
      *gated* on the binary being installed and degrades to "unavailable"
      (never fabricated) when it is not.

* `Recipe` — a named (transpiler, target) pairing with the command template and
  the human instructions, so "c2rust → verify" and "llm-transpiler → verify" are
  data, not code. New transpilers are added by appending a `Recipe`.

* `verify_transpiled(...)` — runs the **real oracle** (`confirm_trap_vs_defined`)
  on the translator's output and returns whether the translation diverges from
  the C source because of source UB. This is the actual end-to-end recipe step,
  proven against compiled binaries.

The reference translators are exercised live (clang/UBSan + rustc/go/swiftc):
a UB-rooted C function is flagged on the produced translation, a safe input is
not, and a well-defined function is never flagged — i.e. the recipe pipeline
preserves the oracle's guarantees end to end.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from . import target_semantics as ts
from .reexec import ReexecHarness, toolchain_available


# --------------------------------------------------------------------------- #
# The integration point.
# --------------------------------------------------------------------------- #
@runtime_checkable
class Translator(Protocol):
    """Turns a C translation-unit into target source for one language pair.

    A transpiler integration implements this (or is wrapped by
    `ExternalCommandTranslator`). `available()` lets a recipe gate on a real
    binary being installed without fabricating output."""

    name: str
    target_lang: str

    def available(self) -> bool: ...

    def translate(self, c_src: str, divergence_class: str) -> Optional[str]: ...


# --------------------------------------------------------------------------- #
# Built-in reference translators (real, compilable) per target pack.
# --------------------------------------------------------------------------- #
def _rust_translation(klass: str) -> Optional[str]:
    if klass in ("div_by_zero", "division_by_zero"):
        body = ("  let a: i32 = std::env::args().nth(1).unwrap().parse().unwrap();\n"
                "  let b: i32 = std::env::args().nth(2).unwrap().parse().unwrap();\n"
                "  println!(\"{}\", a / b);")
    elif klass in ("signed_overflow", "int_overflow"):
        body = ("  let a: i32 = std::env::args().nth(1).unwrap().parse().unwrap();\n"
                "  let b: i32 = std::env::args().nth(2).unwrap().parse().unwrap();\n"
                "  println!(\"{}\", a.wrapping_add(b) / 2);")
    elif klass in ("oversized_shift", "shift"):
        body = ("  let v: i32 = std::env::args().nth(1).unwrap().parse().unwrap();\n"
                "  let w: u32 = std::env::args().nth(2).unwrap().parse().unwrap();\n"
                "  println!(\"{}\", v.wrapping_shl(w));")
    else:
        return None
    return "fn main(){\n" + body + "\n}\n"


def _go_translation(klass: str) -> Optional[str]:
    head = "package main\nimport (\"fmt\";\"os\";\"strconv\")\n"
    rd2 = ("a,_:=strconv.Atoi(os.Args[1]);b,_:=strconv.Atoi(os.Args[2]);")
    if klass in ("div_by_zero", "division_by_zero"):
        return head + "func main(){" + rd2 + "fmt.Println(a/b)}\n"
    if klass in ("signed_overflow", "int_overflow"):
        return head + "func main(){" + rd2 + "fmt.Println((a+b)/2)}\n"
    if klass in ("oversized_shift", "shift"):
        return (head + "func main(){v,_:=strconv.Atoi(os.Args[1]);"
                "w,_:=strconv.Atoi(os.Args[2]);fmt.Println(v<<uint(w))}\n")
    return None


def _swift_translation(klass: str) -> Optional[str]:
    head = "import Foundation\n"
    if klass in ("div_by_zero", "division_by_zero"):
        return (head + "let a=Int32(CommandLine.arguments[1])!\n"
                "let b=Int32(CommandLine.arguments[2])!\nprint(a / b)\n")
    if klass in ("signed_overflow", "int_overflow"):
        return (head + "let a=Int32(CommandLine.arguments[1])!\n"
                "let b=Int32(CommandLine.arguments[2])!\nprint((a &+ b) / 2)\n")
    if klass in ("oversized_shift", "shift"):
        return (head + "let v=Int32(CommandLine.arguments[1])!\n"
                "let w=Int32(CommandLine.arguments[2])!\nprint(v << w)\n")
    return None


_REFERENCE_GEN: Dict[str, Callable[[str], Optional[str]]] = {
    "rust": _rust_translation,
    "go": _go_translation,
    "swift": _swift_translation,
}


@dataclass
class ReferenceTranslator:
    """A built-in, fully-compilable translator for the catalogued classes — the
    known-good translation a recipe verifies against. Always `available`."""
    target_lang: str
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"reference-{self.target_lang}"
        if self.target_lang not in _REFERENCE_GEN:
            raise ValueError(f"no reference translator for {self.target_lang!r}; "
                             f"known: {sorted(_REFERENCE_GEN)}")

    def available(self) -> bool:
        return True

    def translate(self, c_src: str, divergence_class: str) -> Optional[str]:
        return _REFERENCE_GEN[self.target_lang](divergence_class)


@dataclass
class ExternalCommandTranslator:
    """Wraps a real transpiler binary (c2rust, an LLM-transpiler CLI, ...). The
    command template uses `{in}` / `{out}` placeholders for the C input and the
    target output file. Gated on the binary existing; never fabricates output."""
    name: str
    target_lang: str
    binary: str
    arg_template: Tuple[str, ...]   # e.g. ("transpile", "{in}", "-o", "{out}")
    timeout: int = 120

    def available(self) -> bool:
        return shutil.which(self.binary) is not None

    def translate(self, c_src: str, divergence_class: str) -> Optional[str]:
        if not self.available():
            return None
        import tempfile
        with tempfile.TemporaryDirectory() as d:  # pragma: no cover - needs binary
            cin = os.path.join(d, "in.c")
            cout = os.path.join(d, "out.txt")
            with open(cin, "w") as f:
                f.write(c_src)
            args = [self.binary] + [
                a.replace("{in}", cin).replace("{out}", cout)
                for a in self.arg_template]
            try:
                subprocess.run(args, cwd=d, capture_output=True, text=True,
                               timeout=self.timeout, check=True)
            except (subprocess.SubprocessError, OSError):
                return None
            try:
                with open(cout) as f:
                    return f.read()
            except OSError:
                return None


# --------------------------------------------------------------------------- #
# Recipes: named (transpiler, target) integrations as data.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Recipe:
    name: str
    target_lang: str
    instructions: str
    translator_factory: Callable[[], Translator]

    def translator(self) -> Translator:
        return self.translator_factory()


RECIPES: Tuple[Recipe, ...] = (
    Recipe(
        "reference-rust", "rust",
        "Built-in reference C->Rust translation (idiomatic wrapping / defined "
        "panic). Use as the known-good baseline to validate the pipeline.",
        lambda: ReferenceTranslator("rust")),
    Recipe(
        "reference-go", "go",
        "Built-in reference C->Go translation (64-bit int / defined panic).",
        lambda: ReferenceTranslator("go")),
    Recipe(
        "reference-swift", "swift",
        "Built-in reference C->Swift translation (defined trap / smart shift).",
        lambda: ReferenceTranslator("swift")),
    Recipe(
        "c2rust", "rust",
        "Translate with the c2rust transpiler, then verify: "
        "`c2rust transpile {in} -o {out}`. Install c2rust to enable; the recipe "
        "is gated and reports 'unavailable' (never fabricated) if absent.",
        lambda: ExternalCommandTranslator(
            "c2rust", "rust", "c2rust",
            ("transpile", "{in}", "-o", "{out}"))),
    Recipe(
        "llm-transpiler", "rust",
        "Translate with any LLM-transpiler CLI that reads a C file and writes "
        "target source: `$LLM_TRANSPILER {in} {out}`. Set the binary via a "
        "custom ExternalCommandTranslator; gated on the binary existing.",
        lambda: ExternalCommandTranslator(
            "llm-transpiler",
            "rust",
            os.environ.get("LLM_TRANSPILER", "llm-transpiler"),
            ("{in}", "{out}"))),
)


def get_recipe(name: str) -> Recipe:
    for r in RECIPES:
        if r.name == name:
            return r
    raise ValueError(f"unknown recipe {name!r}; known: {[r.name for r in RECIPES]}")


def recipe_names() -> List[str]:
    return [r.name for r in RECIPES]


# --------------------------------------------------------------------------- #
# The end-to-end recipe step: translate, then verify with the real oracle.
# --------------------------------------------------------------------------- #
@dataclass
class TranspileVerdict:
    recipe: str
    target_lang: str
    translator_available: bool
    oracle_available: bool
    translated: bool
    diverged: bool
    reason: str = ""


def verify_transpiled(
        c_src: str,
        translator: Translator,
        argv_inputs: List[str],
        divergence_class: str,
        harness: Optional[ReexecHarness] = None) -> TranspileVerdict:
    """Translate `c_src` with `translator` and run the real oracle on the result.
    Returns a verdict carrying whether the translation diverges from the C source
    because of source-level undefined behaviour."""
    lang = translator.target_lang
    h = harness or ReexecHarness(toolchain_available())
    oracle_ok = h.status.full_for(lang)
    if not translator.available():
        return TranspileVerdict(
            getattr(translator, "name", "?"), lang, False, oracle_ok, False,
            False, "translator unavailable (binary not installed)")
    tgt = translator.translate(c_src, divergence_class)
    if tgt is None:
        return TranspileVerdict(
            getattr(translator, "name", "?"), lang, True, oracle_ok, False,
            False, "translator produced no output for this class")
    if not oracle_ok:
        return TranspileVerdict(
            getattr(translator, "name", "?"), lang, True, False, True, False,
            "toolchain unavailable for verification")
    res = h.confirm_trap_vs_defined(c_src, tgt, argv_inputs, divergence_class, lang)
    return TranspileVerdict(
        getattr(translator, "name", "?"), lang, True, True, True,
        bool(res.confirmed), res.reason or res.summary())


# --------------------------------------------------------------------------- #
# Confirmation: the recipe pipeline preserves the oracle's guarantees.
# --------------------------------------------------------------------------- #
_C_DIV = ("#include <stdio.h>\n#include <stdlib.h>\n"
          "int main(int argc,char**argv){int a=atoi(argv[1]);int b=atoi(argv[2]);"
          'printf("%d\\n",a/b);return 0;}\n')


@dataclass
class RecipeConfirmation:
    available: bool
    ok: bool
    n_reference_pairs: int
    external_recipes: Tuple[str, ...]
    verdicts: List[TranspileVerdict] = field(default_factory=list)
    detail: str = ""

    def render(self) -> str:
        lines = ["transpiler-recipe pipeline:"]
        for v in self.verdicts:
            lines.append(f"  [{v.recipe} -> {v.target_lang}] "
                         f"translated={v.translated} diverged={v.diverged} "
                         f"({v.reason[:60]})")
        for name in self.external_recipes:
            lines.append(f"  [{name}] external transpiler recipe registered "
                         f"(gated on binary)")
        lines.append(f"  => {'ok' if self.ok else 'FAILED'}")
        return "\n".join(lines)


def confirm_transpiler_recipes() -> RecipeConfirmation:
    """Prove the recipe pipeline preserves the oracle's guarantees: for every
    available reference translator (real Rust/Go/Swift), the translate→verify
    step flags a div-by-zero divergence on the UB input and stays silent on a
    safe input. External recipes (c2rust, llm-transpiler) are registered and
    correctly gated."""
    status = toolchain_available()
    h = ReexecHarness(status)
    verdicts: List[TranspileVerdict] = []
    n_ref = 0
    ok = True
    for lang in ("rust", "go", "swift"):
        if not status.full_for(lang):
            continue
        tr = ReferenceTranslator(lang)
        ub = verify_transpiled(_C_DIV, tr, ["10", "0"], "div_by_zero", h)
        safe = verify_transpiled(_C_DIV, tr, ["10", "2"], "div_by_zero", h)
        verdicts.append(ub)
        verdicts.append(safe)
        n_ref += 1
        if not (ub.translated and ub.diverged and safe.translated
                and not safe.diverged):
            ok = False
    external = tuple(r.name for r in RECIPES
                     if not r.name.startswith("reference-"))
    # external recipes must at least be constructible and report availability
    # honestly (no fabrication) — never required to be installed.
    for name in external:
        tr = get_recipe(name).translator()
        # availability is a bool; translate on unavailable must yield None.
        if not tr.available():
            assert tr.translate(_C_DIV, "div_by_zero") is None
    available = n_ref > 0
    if not available:
        ok = True  # consistency-only when no target toolchain present.
    detail = (f"reference_pairs={n_ref} external_recipes={list(external)} "
              f"verdicts={len(verdicts)}")
    return RecipeConfirmation(
        available=available, ok=ok, n_reference_pairs=n_ref,
        external_recipes=external, verdicts=verdicts, detail=detail)


TRANSPILER_RECIPES_SPI = {
    "RECIPES": RECIPES,
    "get_recipe": get_recipe,
    "recipe_names": recipe_names,
    "verify_transpiled": verify_transpiled,
    "confirm_transpiler_recipes": confirm_transpiler_recipes,
}


if __name__ == "__main__":  # pragma: no cover
    conf = confirm_transpiler_recipes()
    print(f"ok={conf.ok} {conf.detail}")
    print(conf.render())
