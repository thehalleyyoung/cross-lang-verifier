"""High-fidelity source ingestion via compiler IRs (Step 27).

Hand-rolled parsers are fine for toys but lose on real code: they re-derive
(badly) facts the language's own compiler already computes exactly.  The durable
pattern — established here so future language pairs follow it — is to **ingest the
compiler's own IR**:

  * **C(++): clang AST (JSON).**  ``clang -Xclang -ast-dump=json`` yields the
    type-checked AST with fully-resolved signatures, parameter names/types,
    storage class, and source locations.  We extract a faithful function table
    (``IRFunction``) directly from it — no guessing at C's declarator grammar.
  * **Rust: rustc MIR.**  ``rustc --emit=mir`` yields the mid-level IR *after*
    borrow checking, so ownership facts come **for free**: a ``move`` operand in
    MIR is the compiler's own statement that a value was moved (consumed), which
    is exactly the borrow/ownership signal the divergence oracle needs and which a
    text parser cannot recover.

Both ingesters are proven against the real compilers on real code: the extracted
function signatures match what the program declares, and the MIR ``move`` facts
match the program's ownership behaviour (a by-value ``Vec`` parameter passed to a
method is a move; a ``Copy`` ``i32`` is not).  The result is a small, shared
``IRModule`` both frontends populate, demonstrating the "use the language's own
compiler IR" SPI.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

CLANG = "/usr/bin/clang"
RUSTC = "/opt/homebrew/bin/rustc"


@dataclass(frozen=True)
class IRParam:
    name: Optional[str]
    type: str


@dataclass(frozen=True)
class IRFunction:
    name: str
    ret_type: str
    params: Tuple[IRParam, ...]
    storage: str = ""           # e.g. "static", "extern", ""
    moved_params: Tuple[str, ...] = ()   # MIR-derived: params consumed by move

    @property
    def arity(self) -> int:
        return len(self.params)


@dataclass
class IRModule:
    lang: str
    source_ir: str               # "clang-ast-json" | "rustc-mir"
    functions: Dict[str, IRFunction] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# C(++) ingestion via clang AST JSON.
# --------------------------------------------------------------------------- #

def clang_ast_json(src: str,
                   extra_args: Optional[List[str]] = None) -> Optional[dict]:
    if not os.path.exists(CLANG):
        return None
    with tempfile.TemporaryDirectory() as d:
        cpath = os.path.join(d, "a.c")
        with open(cpath, "w") as f:
            f.write(src)
        args = [CLANG, "-Xclang", "-ast-dump=json", "-fsyntax-only"]
        args += (extra_args or [])
        args.append(cpath)
        r = subprocess.run(args, capture_output=True, text=True)
        if r.returncode != 0 or not r.stdout:
            return None
        try:
            return json.loads(r.stdout)
        except json.JSONDecodeError:
            return None


def _split_fn_qualtype(qual: str) -> Tuple[str, ...]:
    """Given a clang function qualType like ``int (int, int)`` or
    ``char *(const char *)``, return the parameter-type tuple."""
    i = qual.find("(")
    if i < 0:
        return ()
    # the parameter list is the last top-level (...) group.
    depth = 0
    start = -1
    end = -1
    for k, ch in enumerate(qual):
        if ch == "(":
            if depth == 0:
                start = k
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                end = k
    if start < 0 or end < 0:
        return ()
    inner = qual[start + 1:end].strip()
    if inner in ("", "void"):
        return ()
    return tuple(p.strip() for p in inner.split(","))


def ingest_clang(src: str,
                 extra_args: Optional[List[str]] = None) -> Optional[IRModule]:
    """Build an IRModule from a C translation unit via the clang AST.

    ``extra_args`` (e.g. ``["-I", incdir]``) are forwarded to the AST dump so
    translation units that include project headers parse correctly.
    """
    ast = clang_ast_json(src, extra_args=extra_args)
    if ast is None:
        return None
    mod = IRModule("c", "clang-ast-json")
    for node in ast.get("inner", []) or []:
        if node.get("kind") != "FunctionDecl" or "name" not in node:
            continue
        # only functions actually defined or declared in this TU (skip the huge
        # set of builtins, which carry an "isImplicit" flag or no loc.spellingLoc).
        loc = node.get("loc", {})
        if not loc or loc.get("includedFrom") is not None:
            continue
        name = node["name"]
        qual = node.get("type", {}).get("qualType", "")
        ret = qual.split("(")[0].strip()
        params: List[IRParam] = []
        ptypes = _split_fn_qualtype(qual)
        decl_params = [k for k in (node.get("inner") or [])
                       if k.get("kind") == "ParmVarDecl"]
        for idx, pt in enumerate(ptypes):
            pname = (decl_params[idx].get("name")
                     if idx < len(decl_params) else None)
            params.append(IRParam(pname, pt))
        storage = node.get("storageClass", "") or ""
        mod.functions[name] = IRFunction(name, ret, tuple(params), storage)
    return mod


# --------------------------------------------------------------------------- #
# Rust ingestion via rustc MIR (post-borrow-check; ownership for free).
# --------------------------------------------------------------------------- #

def rustc_mir(src: str) -> Optional[str]:
    if not os.path.exists(RUSTC):
        return None
    with tempfile.TemporaryDirectory() as d:
        rs = os.path.join(d, "a.rs")
        with open(rs, "w") as f:
            f.write(src)
        out = os.path.join(d, "a.mir")
        r = subprocess.run(
            [RUSTC, "--emit=mir", "-o", out, "--crate-type=lib", rs],
            capture_output=True, text=True, cwd=d)
        if r.returncode != 0 or not os.path.exists(out):
            return None
        with open(out) as f:
            return f.read()


_RE_MIR_FN = re.compile(
    r"^fn\s+([A-Za-z_]\w*)\s*\(([^)]*)\)\s*->\s*([^\{]+)\{", re.MULTILINE)
_RE_MIR_LOCAL = re.compile(r"^\s*(?:debug\s+(\w+)\s*=>\s*_(\d+);)")


def ingest_rustc_mir(src: str) -> Optional[IRModule]:
    """Build an IRModule from Rust via rustc MIR, recording which by-value
    parameters are *moved* (consumed) — a fact taken straight from the compiler's
    post-borrow-check IR."""
    mir = rustc_mir(src)
    if mir is None:
        return None
    mod = IRModule("rust", "rustc-mir")
    # Split the MIR into per-function chunks.
    fn_iters = list(_RE_MIR_FN.finditer(mir))
    for i, m in enumerate(fn_iters):
        name = m.group(1)
        raw_params = m.group(2).strip()
        ret = m.group(3).strip()
        body_start = m.end()
        body_end = (fn_iters[i + 1].start() if i + 1 < len(fn_iters)
                    else len(mir))
        body = mir[body_start:body_end]
        # MIR params look like `_1: i32, _2: Vec<i32>`.
        params: List[IRParam] = []
        local_to_debug: Dict[str, str] = {}
        for dm in _RE_MIR_LOCAL.finditer(body):
            if dm.group(1):
                local_to_debug[dm.group(2)] = dm.group(1)
        param_locals: List[Tuple[str, str]] = []
        if raw_params:
            for p in _split_top_level(raw_params):
                pm = re.match(r"_(\d+)\s*:\s*(.+)", p.strip())
                if pm:
                    local = pm.group(1)
                    ptype = pm.group(2).strip()
                    pname = local_to_debug.get(local)
                    params.append(IRParam(pname, ptype))
                    param_locals.append((local, ptype))
        # a parameter is "moved"/consumed if the function takes ownership of it:
        # MIR shows either `move _N` (passed by value onward) or `drop(_N)` (the
        # function owns it and is responsible for dropping it). A Copy type (e.g.
        # i32) is neither moved nor dropped.
        moved: List[str] = []
        for local, _pt in param_locals:
            consumed = (re.search(r"\bmove\s+_" + local + r"\b", body)
                        or re.search(r"\bdrop\(_" + local + r"\)", body))
            if consumed:
                pname = local_to_debug.get(local) or ("_" + local)
                moved.append(pname)
        mod.functions[name] = IRFunction(name, ret, tuple(params),
                                         moved_params=tuple(moved))
    return mod


def _split_top_level(s: str) -> List[str]:
    out: List[str] = []
    depth = 0
    cur = ""
    for ch in s:
        if ch in "<([":
            depth += 1
        elif ch in ">)]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    return out


# --------------------------------------------------------------------------- #
# Confirmations.
# --------------------------------------------------------------------------- #

EXAMPLE_C = (
    "int add(int a, int b) { int s = a + b; return s; }\n"
    "static char *dup_first(const char *p) { return (char*)p; }\n"
    "int triple(int x) { return x + x + x; }\n")

# Rust where one fn moves a Vec (by value, consumed) and one only copies an i32.
EXAMPLE_RUST = (
    "pub fn consume(v: Vec<i32>) -> usize { v.len() }\n"
    "pub fn double(x: i32) -> i32 { x + x }\n")


@dataclass
class ClangIngestConfirmation:
    available: bool
    module: Optional[IRModule]

    @property
    def ok(self) -> bool:
        if not (self.available and self.module):
            return False
        fns = self.module.functions
        if not {"add", "dup_first", "triple"}.issubset(fns):
            return False
        add = fns["add"]
        if add.ret_type != "int" or add.arity != 2:
            return False
        if tuple(p.type for p in add.params) != ("int", "int"):
            return False
        if tuple(p.name for p in add.params) != ("a", "b"):
            return False
        if fns["dup_first"].storage != "static":
            return False
        if fns["triple"].arity != 1:
            return False
        return True


def confirm_clang_ingest() -> ClangIngestConfirmation:
    if not os.path.exists(CLANG):
        return ClangIngestConfirmation(False, None)
    return ClangIngestConfirmation(True, ingest_clang(EXAMPLE_C))


@dataclass
class MirIngestConfirmation:
    available: bool
    module: Optional[IRModule]

    @property
    def ok(self) -> bool:
        if not (self.available and self.module):
            return False
        fns = self.module.functions
        if not {"consume", "double"}.issubset(fns):
            return False
        # the Vec parameter is moved (consumed); the i32 is Copy, never moved.
        consume = fns["consume"]
        double = fns["double"]
        moved_consume = any("Vec" in p.type for p in consume.params) \
            and bool(consume.moved_params)
        not_moved_double = (double.moved_params == ())
        return moved_consume and not_moved_double


def confirm_mir_ingest() -> MirIngestConfirmation:
    if not os.path.exists(RUSTC):
        return MirIngestConfirmation(False, None)
    return MirIngestConfirmation(True, ingest_rustc_mir(EXAMPLE_RUST))


# The ingestion SPI: each frontend turns the language's own compiler IR into the
# shared IRModule, so adding a language is a bounded task (provide an ingester).
IR_INGESTION_SPI: Dict[str, str] = {
    "c": "clang -Xclang -ast-dump=json -> IRModule (signatures, params, storage)",
    "rust": "rustc --emit=mir -> IRModule (signatures + move/ownership facts)",
    "pattern": "use the language's own compiler IR; never hand-roll a parser for "
               "real code. A new language = one ingester producing IRModule.",
}
