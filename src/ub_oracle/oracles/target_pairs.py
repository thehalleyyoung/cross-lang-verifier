"""
Data-driven target-pair oracles (100_STEPS step 39).

Each divergence oracle for the anchor pair (C -> Rust) already separates *finding*
a divergence witness (a Z3 search) from *emitting* the witnessing program. This
module turns "support a new target language" into pure configuration:

  * the target's defined-behaviour contract lives in
    :mod:`~src.ub_oracle.target_semantics` as a :class:`TargetPack` (compiler,
    source suffix, defined return codes, hermetic build env);
  * the only per-(target, class) code is a small **source emitter** that renders
    the equivalent program in the target language.

The :func:`make_pair_oracle` factory then synthesises a fully-functional oracle
for ``(anchor class, target)`` by reusing the anchor oracle's witness search
verbatim and swapping in the target source. Both the C -> Go and C -> Swift pairs
below are generated from the *same* declarative table — no bespoke oracle classes.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

from ..plugin import DivergenceOracle, register
from ..replay import Counterexample
from ..target_semantics import PACKS
from .signed_overflow import SignedOverflowOracle
from .integer_ub import (
    ShiftOutOfRangeOracle,
    DivisionByZeroOracle,
    IntMinDivNeg1Oracle,
)
from .memory_shape import ArrayOutOfBoundsOracle

# width -> (target type name, strconv/parse bit-size)
_INT = {32: 32, 64: 64}
_GO_TYPE = {32: "int32", 64: "int64"}
_SWIFT_TYPE = {32: "Int32", 64: "Int64"}


# ── the factory ──────────────────────────────────────────────────────────────

def make_pair_oracle(anchor_cls, build_method: str, target_lang: str,
                     emitter: Callable[..., Tuple[str, str]]) -> DivergenceOracle:
    """Synthesise a ``(anchor_cls, target_lang)`` oracle.

    ``emitter`` receives exactly the same arguments the anchor's ``build_method``
    receives, and returns ``(target_source, divergence_note)``. The generated
    oracle reuses the anchor's ``find_divergence`` (hence the identical C source
    and Z3 witness) and only swaps the emitted target program + ``target_lang``.
    """

    def _wrapped(self, *args, _bm=build_method, _emit=emitter,
                 _tl=target_lang, _base=anchor_cls):
        # Call the anchor's own builder to get the (identical) C source + the
        # witness inputs, then retarget it to this language.
        ce: Counterexample = getattr(_base, _bm)(self, *args)
        src, note = _emit(*args)
        ce.target_lang = _tl
        ce.target_snippet = src
        ce.divergence_witness = note
        return ce

    name = f"{target_lang.capitalize()}{anchor_cls.__name__}"
    cls = type(name, (anchor_cls,),
               {"target_lang": target_lang, build_method: _wrapped})
    return cls()


# ── Go source emitters (one per divergence class) ────────────────────────────

def _go_signed(op, c, width, var, witness):
    gtype = _GO_TYPE[width]
    c_op = "+" if op == "add" else "-"
    cmp = ">" if op == "add" else "<"
    src = (
        "package main\n"
        "import (\n\t\"fmt\"\n\t\"os\"\n\t\"strconv\"\n)\n"
        f"func f({var} {gtype}) {gtype} {{\n"
        f"\tif {var} {c_op} {c} {cmp} {var} {{\n\t\treturn 1\n\t}}\n\treturn 0\n}}\n"
        "func main() {\n"
        f"\tv, _ := strconv.ParseInt(os.Args[1], 10, {_INT[width]})\n"
        f"\tfmt.Println(f({gtype}(v)))\n"
        "}\n"
    )
    note = (f"C signed {op} overflow at {var}={witness} (UB; optimiser may assume "
            f"`{var} {c_op} {c} {cmp} {var}` always holds), whereas Go's "
            f"`{var} {c_op} {c}` wraps deterministically to a defined value.")
    return src, note


def _go_two_arg(width, expr, a, b, bbits=None):
    gtype = _GO_TYPE[width]
    bits = _INT[width]
    bbits = bbits or bits
    return (
        "package main\n"
        "import (\n\t\"fmt\"\n\t\"os\"\n\t\"strconv\"\n)\n"
        f"func f({a} {gtype}, {b} {gtype}) {gtype} {{\n\treturn {expr}\n}}\n"
        "func main() {\n"
        f"\tav, _ := strconv.ParseInt(os.Args[1], 10, {bits})\n"
        f"\tbv, _ := strconv.ParseInt(os.Args[2], 10, {bbits})\n"
        f"\tfmt.Println(f({gtype}(av), {gtype}(bv)))\n"
        "}\n"
    )


def _go_shift(width, var, svar, x_val, shift_amt):
    src = _go_two_arg(width, f"{var} << uint({svar})", var, svar, bbits=32)
    note = (f"C `{var} << {svar}` with {svar}={shift_amt} >= width {width} is UB; "
            f"Go masks/widens the count and yields a defined value (0).")
    return src, note


def _go_div(width, op, avar, bvar, a_val, b_val):
    go_op = "/" if op == "div" else "%"
    src = _go_two_arg(width, f"{avar} {go_op} {bvar}", avar, bvar)
    note = (f"C `{avar} {go_op} {bvar}` with {bvar}=0 is UB; Go panics "
            f"deterministically at runtime (a defined, observable outcome).")
    return src, note


def _go_intmin(width, op, avar, bvar, a_val, b_val):
    go_op = "/" if op == "div" else "%"
    src = _go_two_arg(width, f"{avar} {go_op} {bvar}", avar, bvar)
    note = (f"C `{avar} {go_op} {bvar}` with {avar}={a_val}, {bvar}={b_val} overflows "
            f"signed division (UB); Go defines x/-1 == x for the most-negative x.")
    return src, note


def _go_oob(length, var, idx):
    elems = ", ".join(str(10 + k) for k in range(length))
    src = (
        "package main\n"
        "import (\n\t\"fmt\"\n\t\"os\"\n\t\"strconv\"\n)\n"
        f"var a = [{length}]int32{{{elems}}}\n"
        f"func f({var} int) int32 {{ return a[{var}] }}\n"
        "func main() {\n"
        f"\t{var}, _ := strconv.Atoi(os.Args[1])\n"
        f"\tfmt.Println(f({var}))\n"
        "}\n"
    )
    note = (f"C `a[{var}]` with {var}={idx} on a length-{length} array is UB; "
            f"Go bounds-checks and panics deterministically (a defined outcome).")
    return src, note


# ── Swift source emitters (one per divergence class) ─────────────────────────

def _swift_signed(op, c, width, var, witness):
    stype = _SWIFT_TYPE[width]
    amp = "&+" if op == "add" else "&-"
    op_word = "+" if op == "add" else "-"
    cmp = ">" if op == "add" else "<"
    src = (
        "import Foundation\n"
        f"func f(_ {var}: {stype}) -> {stype} {{ return ({var} {amp} {c} {cmp} {var}) ? 1 : 0 }}\n"
        f"let {var} = {stype}(CommandLine.arguments[1])!\n"
        f"print(f({var}))\n"
    )
    note = (f"C signed {op} overflow at {var}={witness} (UB; optimiser may assume "
            f"`{var} {op_word} {c} {cmp} {var}` always holds), whereas Swift's "
            f"`{amp}` wrapping operator gives a defined value.")
    return src, note


def _swift_two_arg(width, expr, a, b, atype=None, btype=None):
    stype = _SWIFT_TYPE[width]
    atype = atype or stype
    btype = btype or stype
    return (
        "import Foundation\n"
        f"func f(_ {a}: {atype}, _ {b}: {btype}) -> {stype} {{ return {expr} }}\n"
        f"let {a} = {atype}(CommandLine.arguments[1])!\n"
        f"let {b} = {btype}(CommandLine.arguments[2])!\n"
        f"print(f({a}, {b}))\n"
    )


def _swift_shift(width, var, svar, x_val, shift_amt):
    # Swift's `<<` is a smart shift: defined for any amount (0 on overshift).
    src = _swift_two_arg(width, f"{var} << {svar}", var, svar)
    note = (f"C `{var} << {svar}` with {svar}={shift_amt} >= width {width} is UB; "
            f"Swift's smart shift `<<` is defined for any amount and yields 0.")
    return src, note


def _swift_div(width, op, avar, bvar, a_val, b_val):
    sw_op = "/" if op == "div" else "%"
    src = _swift_two_arg(width, f"{avar} {sw_op} {bvar}", avar, bvar)
    note = (f"C `{avar} {sw_op} {bvar}` with {bvar}=0 is UB; Swift traps "
            f"deterministically at runtime (a defined, observable abort).")
    return src, note


def _swift_intmin(width, op, avar, bvar, a_val, b_val):
    sw_op = "/" if op == "div" else "%"
    src = _swift_two_arg(width, f"{avar} {sw_op} {bvar}", avar, bvar)
    note = (f"C `{avar} {sw_op} {bvar}` with {avar}={a_val}, {bvar}={b_val} overflows "
            f"signed division (UB); Swift traps deterministically on the overflow.")
    return src, note


def _swift_oob(length, var, idx):
    elems = ", ".join(str(10 + k) for k in range(length))
    src = (
        "import Foundation\n"
        f"let a: [Int32] = [{elems}]\n"
        f"func f(_ {var}: Int) -> Int32 {{ return a[{var}] }}\n"
        f"let {var} = Int(CommandLine.arguments[1])!\n"
        f"print(f({var}))\n"
    )
    note = (f"C `a[{var}]` with {var}={idx} on a length-{length} array is UB; "
            f"Swift bounds-checks and traps deterministically (a defined outcome).")
    return src, note


# ── OCaml source emitters (one per divergence class) ─────────────────────────
# OCaml's fixed-width integers live in the Int32/Int64 modules: arithmetic is
# modular (defined), division by zero raises Division_by_zero, and array indexing
# is bounds-checked (Invalid_argument). Each emitted program reads its operands
# from argv, applies the operation, and prints one integer — mirroring the C/Go
# witnesses exactly so the same Z3-found input drives all targets.
_OCAML_MOD = {32: ("Int32", "l", "%ld"), 64: ("Int64", "L", "%Ld")}


def _ocaml_signed(op, c, width, var, witness):
    mod, suf, _ = _OCAML_MOD[width]
    fn = "add" if op == "add" else "sub"
    op_word = "+" if op == "add" else "-"
    cmp = ">" if op == "add" else "<"
    src = (
        f"let f {var} = if ({mod}.compare ({mod}.{fn} {var} {c}{suf}) {var}) {cmp} 0 "
        f"then 1 else 0\n"
        "let () =\n"
        f"  let {var} = {mod}.of_string Sys.argv.(1) in\n"
        f"  Printf.printf \"%d\\n\" (f {var})\n"
    )
    note = (f"C signed {op} overflow at {var}={witness} (UB; optimiser may assume "
            f"`{var} {op_word} {c} {cmp} {var}` always holds), whereas OCaml's "
            f"`{mod}.{fn}` is modular and yields a defined value.")
    return src, note


def _ocaml_binop(width, expr_mod, fn, avar, bvar):
    mod, _, fmt = _OCAML_MOD[width]
    return (
        f"let f {avar} {bvar} = {expr_mod}.{fn} {avar} {bvar}\n"
        "let () =\n"
        f"  let {avar} = {mod}.of_string Sys.argv.(1) in\n"
        f"  let {bvar} = {mod}.of_string Sys.argv.(2) in\n"
        f"  Printf.printf \"{fmt}\\n\" (f {avar} {bvar})\n"
    )


def _ocaml_div(width, op, avar, bvar, a_val, b_val):
    mod = _OCAML_MOD[width][0]
    fn = "div" if op == "div" else "rem"
    glyph = "/" if op == "div" else "%"
    src = _ocaml_binop(width, mod, fn, avar, bvar)
    note = (f"C `{avar} {glyph} {bvar}` with {bvar}=0 is UB; OCaml's `{mod}.{fn}` "
            f"raises Division_by_zero, aborting deterministically (exit 2) — a "
            f"defined, observable outcome.")
    return src, note


def _ocaml_intmin(width, op, avar, bvar, a_val, b_val):
    mod = _OCAML_MOD[width][0]
    fn = "div" if op == "div" else "rem"
    glyph = "/" if op == "div" else "%"
    src = _ocaml_binop(width, mod, fn, avar, bvar)
    note = (f"C `{avar} {glyph} {bvar}` with {avar}={a_val}, {bvar}={b_val} overflows "
            f"signed division (UB); OCaml's `{mod}.{fn}` wraps modularly to a "
            f"defined value.")
    return src, note


def _ocaml_oob(length, var, idx):
    mod, suf, fmt = _OCAML_MOD[32]
    elems = "; ".join(f"{10 + k}{suf}" for k in range(length))
    src = (
        f"let a = [| {elems} |]\n"
        f"let f {var} = a.({var})\n"
        "let () =\n"
        f"  let {var} = int_of_string Sys.argv.(1) in\n"
        f"  Printf.printf \"{fmt}\\n\" (f {var})\n"
    )
    note = (f"C `a[{var}]` with {var}={idx} on a length-{length} array is UB; "
            f"OCaml bounds-checks `a.({var})` and raises Invalid_argument, "
            f"aborting deterministically (exit 2) — a defined outcome.")
    return src, note


# ── the declarative table: (anchor class, build method, class key) ───────────

_SPECS = [
    (SignedOverflowOracle, "_build_counterexample", "signed_overflow"),
    (ShiftOutOfRangeOracle, "_build", "shift_oob"),
    (DivisionByZeroOracle, "_build", "div_by_zero"),
    (IntMinDivNeg1Oracle, "_build", "intmin_div_neg1"),
    (ArrayOutOfBoundsOracle, "_build", "array_oob"),
]

# per-target source emitters keyed by divergence-class key. Adding a target is
# exactly this much configuration (plus its TargetPack) — no new oracle code.
_EMITTERS: Dict[str, Dict[str, Callable]] = {
    "go": {
        "signed_overflow": _go_signed,
        "shift_oob": _go_shift,
        "div_by_zero": _go_div,
        "intmin_div_neg1": _go_intmin,
        "array_oob": _go_oob,
    },
    "swift": {
        "signed_overflow": _swift_signed,
        "shift_oob": _swift_shift,
        "div_by_zero": _swift_div,
        "intmin_div_neg1": _swift_intmin,
        "array_oob": _swift_oob,
    },
    # OCaml supports every class except the bit-shift family: OCaml itself leaves
    # `lsl`/`Int32.shift_left` by an out-of-range amount *unspecified*, so it is
    # not a sound "defined target" for shift_oob and is deliberately omitted.
    "ocaml": {
        "signed_overflow": _ocaml_signed,
        "div_by_zero": _ocaml_div,
        "intmin_div_neg1": _ocaml_intmin,
        "array_oob": _ocaml_oob,
    },
}

#: every generated oracle, keyed by (target_lang, divergence_class).
GENERATED: Dict[Tuple[str, str], DivergenceOracle] = {}


def _build_all():
    for target_lang, emitters in _EMITTERS.items():
        if target_lang not in PACKS:  # pragma: no cover - guarded by tests
            raise ValueError(f"no TargetPack registered for {target_lang!r}")
        for anchor_cls, build_method, class_key in _SPECS:
            emitter = emitters.get(class_key)
            if emitter is None:
                continue
            oracle = make_pair_oracle(anchor_cls, build_method, target_lang, emitter)
            register(oracle)
            GENERATED[(target_lang, class_key)] = oracle


_build_all()
