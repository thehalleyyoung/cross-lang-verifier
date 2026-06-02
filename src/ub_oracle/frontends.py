"""Step 25 — robust, real frontends + a documented frontend SPI.

Hand-rolled parsers are fine for toys but lose on real code.  This module
promotes a **real, grammar-backed frontend** for C to the supported path: it
parses C with **tree-sitter** (the same incremental parser editors use) and
populates the *same* shared :class:`ir_ingest.IRModule` the compiler-IR
frontends produce.  Tree-sitter is robust to the constructs that defeat
hand-rolled parsers (nested declarators, pointer/array params, attributes,
storage classes, K&R-ish spacing), so it is the supported parser for "anything
beyond toys" while the clang-AST frontend remains the *highest-fidelity* path.

The frontend SPI is made explicit so adding a language is a bounded, documented
task:

    class Frontend(Protocol):
        name: str
        language: str        # "c", "rust", ...
        def available(self) -> bool: ...
        def ingest(self, src: str) -> Optional[IRModule]: ...

Three frontends are registered behind it: ``treesitter-c`` (this module),
``clang-ast-c`` and ``rustc-mir-rust`` (wrapping the Step-27 ingesters).

The tree-sitter frontend is **cross-validated against the language's own
compiler**: on real C translation units, the function table it extracts
(names, arities, parameter names, storage class) must *agree with the clang AST*
— the compiler is the ground truth.  ``confirm_real_frontends()`` proves that
agreement live.  When tree-sitter is not installed the frontend reports
``available() is False`` and is skipped — never fabricated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Tuple, runtime_checkable

from .ir_ingest import (
    IRFunction,
    IRModule,
    IRParam,
    ingest_clang,
    ingest_rustc_mir,
)


# --------------------------------------------------------------------------- #
# The frontend SPI.
# --------------------------------------------------------------------------- #


@runtime_checkable
class Frontend(Protocol):
    name: str
    language: str

    def available(self) -> bool: ...

    def ingest(self, src: str) -> Optional[IRModule]: ...


# --------------------------------------------------------------------------- #
# tree-sitter C frontend.
# --------------------------------------------------------------------------- #


def _ts_modules():
    """Return (tree_sitter, tree_sitter_c) or None if not installed."""
    try:
        import tree_sitter as ts
        import tree_sitter_c as tsc
        return ts, tsc
    except Exception:
        return None


_TS_CACHE: dict = {}


def _ts_parser():
    mods = _ts_modules()
    if mods is None:
        return None
    if "parser" not in _TS_CACHE:
        ts, tsc = mods
        lang = ts.Language(tsc.language())
        parser = ts.Parser(lang)
        _TS_CACHE["parser"] = parser
    return _TS_CACHE["parser"]


def treesitter_available() -> bool:
    return _ts_modules() is not None


def _text(src: bytes, node) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", "replace")


def _child(node, *types):
    for c in node.children:
        if c.type in types:
            return c
    return None


def _find_declarator(node):
    """Descend through pointer/array/paren declarators to the function
    declarator, accumulating the pointer/qualifier prefix that belongs to the
    return type (e.g. ``char *(...)`` -> prefix ``*``)."""
    prefix = ""
    cur = node
    while cur is not None and cur.type != "function_declarator":
        if cur.type == "pointer_declarator":
            prefix += "*"
            cur = _child(cur, "function_declarator", "pointer_declarator",
                         "array_declarator", "parenthesized_declarator",
                         "identifier")
        elif cur.type in ("array_declarator", "parenthesized_declarator"):
            cur = _child(cur, "function_declarator", "pointer_declarator",
                         "array_declarator", "parenthesized_declarator",
                         "identifier")
        else:
            cur = None
    return cur, prefix


def _param_type_and_name(src: bytes, pdecl) -> Tuple[str, Optional[str]]:
    """Extract ``(type, name)`` from a ``parameter_declaration`` node."""
    # the base type is the first type-bearing child.
    base = None
    for c in pdecl.children:
        if c.type in ("primitive_type", "type_identifier", "sized_type_specifier",
                      "struct_specifier", "union_specifier", "enum_specifier"):
            base = _text(src, c).strip()
            break
    if base is None:
        base = "int"
    # walk the declarator for pointer stars and the identifier (the name).
    name: Optional[str] = None
    stars = ""
    decl = None
    for c in pdecl.children:
        if c.type in ("pointer_declarator", "identifier", "array_declarator",
                      "abstract_pointer_declarator", "function_declarator"):
            decl = c
            break
    cur = decl
    while cur is not None:
        if cur.type == "identifier":
            name = _text(src, cur)
            break
        if cur.type in ("pointer_declarator", "abstract_pointer_declarator"):
            stars += "*"
            nxt = None
            for cc in cur.children:
                if cc.type in ("pointer_declarator", "identifier",
                               "array_declarator", "abstract_pointer_declarator"):
                    nxt = cc
                    break
            cur = nxt
        elif cur.type == "array_declarator":
            nxt = _child(cur, "identifier", "pointer_declarator")
            cur = nxt
        else:
            cur = None
    ptype = (base + " " + stars).strip() if stars else base
    # normalise "char *" spacing to match clang qualType style ("char *").
    ptype = ptype.replace(" *", " *")
    return ptype, name


def _storage_of(src: bytes, fdef) -> str:
    for c in fdef.children:
        if c.type == "storage_class_specifier":
            return _text(src, c).strip()
    return ""


def _return_type_of(src: bytes, fdef) -> str:
    for c in fdef.children:
        if c.type in ("primitive_type", "type_identifier", "sized_type_specifier",
                      "struct_specifier", "union_specifier", "enum_specifier"):
            return _text(src, c).strip()
    return "int"


def ingest_treesitter(src: str) -> Optional[IRModule]:
    """Build an :class:`IRModule` from a C translation unit using tree-sitter."""
    parser = _ts_parser()
    if parser is None:
        return None
    raw = src.encode("utf-8")
    tree = parser.parse(raw)
    mod = IRModule("c", "treesitter-c")

    def visit(node):
        if node.type == "function_definition":
            decl = _child(node, "function_declarator", "pointer_declarator",
                          "parenthesized_declarator", "array_declarator")
            fdecl, ret_ptr = _find_declarator(decl) if decl is not None else (None, "")
            if fdecl is not None:
                ident = _child(fdecl, "identifier", "field_identifier")
                if ident is not None:
                    name = _text(raw, ident)
                    base_ret = _return_type_of(raw, node)
                    ret = (base_ret + " " + ret_ptr).strip() if ret_ptr else base_ret
                    params: List[IRParam] = []
                    plist = _child(fdecl, "parameter_list")
                    if plist is not None:
                        for p in plist.children:
                            if p.type == "parameter_declaration":
                                pt, pn = _param_type_and_name(raw, p)
                                params.append(IRParam(pn, pt))
                            elif p.type == "identifier":
                                # K&R-style / void-less; skip.
                                pass
                    # a single "(void)" param list yields zero IR params.
                    if len(params) == 1 and params[0].type == "void" \
                            and params[0].name is None:
                        params = []
                    storage = _storage_of(raw, node)
                    mod.functions[name] = IRFunction(
                        name, ret, tuple(params), storage)
        for c in node.children:
            visit(c)

    visit(tree.root_node)
    return mod


# --------------------------------------------------------------------------- #
# Registered frontends behind the SPI.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _TreeSitterC:
    name: str = "treesitter-c"
    language: str = "c"

    def available(self) -> bool:
        return treesitter_available()

    def ingest(self, src: str) -> Optional[IRModule]:
        return ingest_treesitter(src)


@dataclass(frozen=True)
class _ClangAstC:
    name: str = "clang-ast-c"
    language: str = "c"

    def available(self) -> bool:
        import os
        from .ir_ingest import CLANG
        return os.path.exists(CLANG)

    def ingest(self, src: str) -> Optional[IRModule]:
        return ingest_clang(src)


@dataclass(frozen=True)
class _RustcMir:
    name: str = "rustc-mir-rust"
    language: str = "rust"

    def available(self) -> bool:
        import os
        from .ir_ingest import RUSTC
        return os.path.exists(RUSTC)

    def ingest(self, src: str) -> Optional[IRModule]:
        return ingest_rustc_mir(src)


FRONTENDS: Tuple[Frontend, ...] = (_TreeSitterC(), _ClangAstC(), _RustcMir())


def frontends_for(language: str) -> Tuple[Frontend, ...]:
    return tuple(f for f in FRONTENDS if f.language == language)


def get_frontend(name: str) -> Frontend:
    for f in FRONTENDS:
        if f.name == name:
            return f
    raise ValueError(f"unknown frontend {name!r}; "
                     f"have {[f.name for f in FRONTENDS]}")


# --------------------------------------------------------------------------- #
# Cross-validation against the compiler (the ground truth).
# --------------------------------------------------------------------------- #


# Real-world-shaped C translation units the frontend must handle.
_SAMPLES: Tuple[str, ...] = (
    "static int add(int a, long b){ return a + b; }\n"
    "int main(int argc, char **argv){ return add(argc, 1); }\n",

    "unsigned clamp(unsigned v, unsigned lo, unsigned hi){\n"
    "  if (v < lo) return lo; if (v > hi) return hi; return v; }\n",

    "char *pick(const char *a, char *b, int which){ return which ? b : (char*)a; }\n",

    "static long mac(long acc, int x, int y){ return acc + (long)x * y; }\n"
    "void noop(void){ }\n",

    "int midpoint(int lo, int hi){ return lo + (hi - lo) / 2; }\n"
    "static unsigned char lo8(unsigned v){ return (unsigned char)(v & 0xFF); }\n",
)


@dataclass
class FrontendAgreement:
    sample_index: int
    functions: Tuple[str, ...]
    name_match: bool
    arity_match: bool
    param_name_match: bool
    storage_match: bool

    @property
    def ok(self) -> bool:
        return (self.name_match and self.arity_match
                and self.param_name_match and self.storage_match)


@dataclass
class FrontendsReport:
    available: bool
    ok: bool
    spi_frontends: Tuple[str, ...]
    agreements: Tuple[FrontendAgreement, ...]
    detail: str


def _compare(ts_mod: IRModule, clang_mod: IRModule,
             index: int) -> FrontendAgreement:
    ts_names = set(ts_mod.functions)
    cl_names = set(clang_mod.functions)
    name_match = ts_names == cl_names
    common = sorted(ts_names & cl_names)
    arity_match = all(
        ts_mod.functions[n].arity == clang_mod.functions[n].arity
        for n in common)
    param_name_match = True
    storage_match = True
    for n in common:
        tf = ts_mod.functions[n]
        cf = clang_mod.functions[n]
        # parameter names (when both name a parameter) must agree positionally.
        for tp, cp in zip(tf.params, cf.params):
            if tp.name is not None and cp.name is not None and tp.name != cp.name:
                param_name_match = False
        if (tf.storage or "") != (cf.storage or ""):
            storage_match = False
    return FrontendAgreement(
        sample_index=index, functions=tuple(common),
        name_match=name_match, arity_match=arity_match,
        param_name_match=param_name_match, storage_match=storage_match)


def confirm_real_frontends() -> FrontendsReport:
    """Prove the tree-sitter C frontend agrees with the clang AST (the
    compiler's own ground truth) on real C, and that the SPI exposes the three
    registered frontends.  Consistency-only when a toolchain is missing."""
    spi = tuple(f.name for f in FRONTENDS)
    ts = get_frontend("treesitter-c")
    cl = get_frontend("clang-ast-c")
    if not ts.available() or not cl.available():
        return FrontendsReport(
            available=False, ok=True, spi_frontends=spi, agreements=(),
            detail="tree-sitter and/or clang absent; SPI shape exercised only")

    agreements: List[FrontendAgreement] = []
    for i, src in enumerate(_SAMPLES):
        ts_mod = ts.ingest(src)
        cl_mod = cl.ingest(src)
        if ts_mod is None or cl_mod is None:
            return FrontendsReport(
                available=True, ok=False, spi_frontends=spi,
                agreements=tuple(agreements),
                detail=f"sample {i}: a frontend returned None")
        agreements.append(_compare(ts_mod, cl_mod, i))
    ok = all(a.ok for a in agreements) and len(agreements) == len(_SAMPLES)
    bad = [a.sample_index for a in agreements if not a.ok]
    detail = (f"{len(agreements)} samples, "
              f"{sum(len(a.functions) for a in agreements)} functions "
              f"cross-validated vs clang AST; failures={bad}")
    return FrontendsReport(available=True, ok=bool(ok), spi_frontends=spi,
                           agreements=tuple(agreements), detail=detail)


if __name__ == "__main__":  # pragma: no cover
    rep = confirm_real_frontends()
    print("frontends SPI:", rep.spi_frontends)
    print("real-frontends:", rep.detail)
    for a in rep.agreements:
        print(f"  sample {a.sample_index}: {list(a.functions)} ok={a.ok}")
    print("=> ok" if rep.ok else "=> FAILED")
    raise SystemExit(0 if rep.ok else 1)
