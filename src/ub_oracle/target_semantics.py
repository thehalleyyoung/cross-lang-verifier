"""
Pluggable target-semantics packs (100_STEPS step 39).

A *target-semantics pack* encodes everything the engine needs to know about a
target language's **defined behaviour**, as data:

  * how to compile a single-file program in that language (``compile_argv`` plus
    any hermetic ``compile_env``) and the source-file ``source_suffix``;
  * which process return codes count as a language-*defined* outcome
    (``defined_returncodes``) — a value *or* a guaranteed, deterministic abort;
  * a human/data description of how the target *resolves* each divergence class
    that C leaves undefined (``class_resolution``).

This is the abstraction that makes the project's generality claim real: adding a
new target language is *configuration* (a new :class:`TargetPack`, plus the small
per-class source templates the oracles render) rather than new harness, verifier
or oracle-engine code.

Return codes are **as observed by Python's** :mod:`subprocess`, which reports a
process killed by signal *N* as the *negative* code ``-N`` (not the shell's
``128+N``). The three packs below were each grounded against real compilers:

  ===========  ======================  ===========================================
  target       compiler                defined return codes (python subprocess)
  ===========  ======================  ===========================================
  rust         ``rustc -O``            0 (value), 101 (clean unwinding panic)
  go           ``go build``            0 (value), 2 (runtime panic, os.Exit(2))
  swift        ``swiftc -O``           0 (value), -5 (SIGTRAP runtime trap)
  ===========  ======================  ===========================================

This module deliberately imports nothing from the rest of the package, so the
re-execution harness can consume it without any import cycle.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class TargetPack:
    """The defined-behaviour contract for one target language, as data."""

    #: stable language key used throughout the engine (e.g. "rust", "go").
    name: str
    #: candidate compiler executables, tried in order via ``shutil.which``.
    compiler_candidates: Tuple[str, ...]
    #: extension for the emitted single-file program, e.g. ".rs".
    source_suffix: str
    #: process return codes (as Python's subprocess reports them) that are a
    #: language-*defined* outcome: a value, or a guaranteed deterministic abort.
    defined_returncodes: Tuple[int, ...]
    #: build the compiler argv from (compiler_path, src_path, out_path).
    compile_argv: Callable[[str, str, str], List[str]]
    #: extra environment for the *compile* step, derived from the work dir
    #: (used e.g. to give Go a hermetic, workdir-local build cache).
    compile_env: Callable[[str], Dict[str, str]] = field(
        default=lambda workdir: {})
    #: how this target *defines* each UB class C leaves undefined. Pure data,
    #: keyed by catalogue divergence-class key -> short description.
    class_resolution: Dict[str, str] = field(default_factory=dict)

    def is_defined_returncode(self, rc: int) -> bool:
        return rc in self.defined_returncodes


# ── compiler argv / env builders ─────────────────────────────────────────────

def _rust_argv(cc: str, src: str, out: str) -> List[str]:
    return [cc, "-O", "-o", out, src]


def _go_argv(cc: str, src: str, out: str) -> List[str]:
    return [cc, "build", "-o", out, src]


def _swift_argv(cc: str, src: str, out: str) -> List[str]:
    return [cc, "-O", "-o", out, src]


def _cpp_argv(cc: str, src: str, out: str) -> List[str]:
    # -O2 exercises the optimizer on the *target* too; -std=c++20 is what makes
    # the otherwise-C-UB constructs (e.g. shifting a 1 into the sign bit) defined.
    return [cc, "-std=c++20", "-O2", "-o", out, src]


def _ocaml_argv(cc: str, src: str, out: str) -> List[str]:
    # ocamlopt produces a native binary; -O3 turns on flambda-style native
    # optimisation where available (harmless otherwise). Int32/Int64 arithmetic
    # is *modular* (defined) and the array/division operations raise defined
    # exceptions, so the optimiser never has C-style UB latitude on the target.
    return [cc, "-O3", "-o", out, src]


def _ocaml_env(workdir: str) -> Dict[str, str]:
    # ocamlopt drops .cmi/.cmx/.o artefacts beside the source; keep them (and any
    # temporary objects) inside the hermetic work dir.
    return {"TMPDIR": workdir}


def _go_env(workdir: str) -> Dict[str, str]:
    # A single ``package main`` file builds standalone, but we still pin a
    # workdir-local cache + module mode so the build is hermetic and never
    # touches a surrounding project module or the global cache.
    return {"GOCACHE": os.path.join(workdir, ".gocache"), "GOFLAGS": "-mod=mod"}


# ── the registry of packs ────────────────────────────────────────────────────

_RUST = TargetPack(
    name="rust",
    compiler_candidates=("rustc",),
    source_suffix=".rs",
    defined_returncodes=(0, 101),
    compile_argv=_rust_argv,
    class_resolution={
        "signed_overflow": "wrapping/checked arithmetic (wrapping_* gives a defined value)",
        "shift_oob": "wrapping_shl masks the shift amount to a defined value",
        "div_by_zero": "guaranteed panic (a defined, deterministic abort)",
        "intmin_div_neg1": "guaranteed panic on the overflowing division",
        "array_oob": "bounds-checked index; guaranteed panic",
        "memcpy_overlap": "slice::copy_within has defined memmove semantics under overlap",
        "eval_order": "explicit statements evaluate in one deterministic order",
    },
)

_GO = TargetPack(
    name="go",
    compiler_candidates=("go",),
    source_suffix=".go",
    defined_returncodes=(0, 2),
    compile_argv=_go_argv,
    compile_env=_go_env,
    class_resolution={
        "signed_overflow": "two's-complement wraparound is defined (a defined value)",
        "shift_oob": "shift counts >= width yield a defined 0",
        "div_by_zero": "runtime panic, os.Exit(2) (a defined, deterministic abort)",
        "intmin_div_neg1": "x/-1 == x for the most-negative x (a defined value)",
        "array_oob": "bounds-checked index; runtime panic (defined abort)",
        "memcpy_overlap": "built-in copy on slices is defined for overlapping ranges",
    },
)

_SWIFT = TargetPack(
    name="swift",
    compiler_candidates=("swiftc",),
    source_suffix=".swift",
    defined_returncodes=(0, -5),  # value, or SIGTRAP runtime trap (python: -5)
    compile_argv=_swift_argv,
    class_resolution={
        "signed_overflow": "&+/&- wrapping operators give a defined value",
        "shift_oob": "smart shift `<<` is defined for any amount (0 on overshift)",
        "div_by_zero": "guaranteed runtime trap (SIGTRAP; a defined, deterministic abort)",
        "intmin_div_neg1": "guaranteed overflow trap on the overflowing division",
        "array_oob": "bounds-checked index; guaranteed runtime trap",
    },
)

# C++ is the *defined-subset* target (100_STEPS step 117): the byte-identical
# source token that is undefined in C can be well-defined under C++ rules. Only
# a value is a defined outcome here (the witnessing construct does not abort).
_CPP = TargetPack(
    name="cpp",
    compiler_candidates=("clang++", "g++", "c++"),
    source_suffix=".cpp",
    defined_returncodes=(0,),
    compile_argv=_cpp_argv,
    class_resolution={
        "signed_shift_sign_bit": "C++20 mandates two's-complement and defines "
                                 "`1 << 31` as INT_MIN by modular wraparound "
                                 "([expr.shift]/2) — a defined value",
    },
)

# OCaml is the GC'd, exception-based target (100_STEPS step 121): a strict
# native-compiled functional language whose fixed-width Int32/Int64 arithmetic is
# *modular* (defined) and whose array / division faults raise exceptions that, if
# uncaught, abort the process deterministically with exit code 2. So every C-UB
# class either becomes a defined value (modular overflow) or a defined,
# deterministic abort (uncaught Division_by_zero / Invalid_argument).
_OCAML = TargetPack(
    name="ocaml",
    compiler_candidates=("ocamlopt", "ocamlopt.opt"),
    source_suffix=".ml",
    defined_returncodes=(0, 2),  # value, or uncaught-exception abort (OCaml: 2)
    compile_argv=_ocaml_argv,
    compile_env=_ocaml_env,
    class_resolution={
        "signed_overflow": "Int32/Int64 arithmetic is modular (defined wraparound)",
        "div_by_zero": "raises Division_by_zero; uncaught -> deterministic exit 2",
        "intmin_div_neg1": "Int32.div min_int (-1) wraps to a defined value",
        "array_oob": "bounds-checked `a.(i)` raises Invalid_argument; "
                     "uncaught -> deterministic exit 2",
    },
)

PACKS: Dict[str, TargetPack] = {p.name: p for p in
                                (_RUST, _GO, _SWIFT, _CPP, _OCAML)}


def get_pack(name: str) -> TargetPack:
    """Return the pack for ``name`` or raise loudly for an unknown target."""
    try:
        return PACKS[name]
    except KeyError:
        raise ValueError(
            f"unknown target language {name!r}; known targets: "
            f"{sorted(PACKS)}") from None


def defined_returncodes(name: str) -> Tuple[int, ...]:
    return get_pack(name).defined_returncodes


def target_names() -> List[str]:
    return list(PACKS.keys())
