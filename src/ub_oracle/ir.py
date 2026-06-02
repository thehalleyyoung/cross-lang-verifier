"""
Frozen shared semantic-IR contract + validator (100_STEPS step 6).

Every frontend — regardless of source/target language — lowers a translation
*unit* into the single dictionary shape this module specifies, and every oracle
consumes exactly that shape. Freezing this contract (and validating against it)
is what keeps the engine **language-pair-agnostic**: the C->Rust, C->Go and
C->Swift pipelines differ only in which ``target_lang`` they declare and which
target program a pair-oracle emits — never in the IR itself.

A unit is an *effect-bearing arithmetic/memory fragment* over typed integer,
floating-point or array operands, tagged with:

* ``kind`` — the structural operator family (see :data:`KNOWN_KINDS`);
* the operands/immediates that operator needs (e.g. ``op``/``const`` for a
  binary-op-with-constant, ``length`` for an array index);
* an optional **operating range** per integer operand (``x_range``/``a_range``/
  ``b_range``/``shift_range``) — the memory/effect precondition under which the
  fragment runs, used both by the abstract-interpretation pre-pass and by the
  oracles' SMT searches;
* an optional ``source_lang``/``target_lang`` declaring the language pair
  (defaulting to the C->Rust anchor) and an optional ``probe`` restricting the
  unit to a single divergence class.

This module is the *operational spec*: the rules below are executable. The
companion prose lives in ``docs/IR.md``. The validator is **sound about
rejection**: it flags a lowering as ill-formed only on a genuine contract
violation (wrong envelope type, an integer kind missing/mistyping a required
operand, an out-of-domain width, or a malformed range), never on a merely
*uncovered-but-well-formed* unit (an unknown ``kind`` or an unsupported pair is
valid IR — the engine reports it ``NOT_COVERED`` rather than rejecting it).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

#: the frozen set of structural operator families the IR defines. Frontends may
#: emit other kinds for forward-compatibility (they validate as well-formed but
#: are reported NOT_COVERED until an oracle claims them); these are the kinds the
#: shipped oracles understand.
KNOWN_KINDS = frozenset({
    "binop_const",   # f(x) = x <op> const           (signed_overflow)
    "shift",         # f(x, s) = x << s               (shift_oob)
    "div",           # f(a, b) = a / b                (div_by_zero, intmin_div_neg1)
    "rem",           # f(a, b) = a % b                (div_by_zero, intmin_div_neg1)
    "array_index",   # f(i) = arr[i]                  (array_oob)
    "type_pun",      # reinterpret bytes              (strict_aliasing)
    "fp_fma",        # a*b + c contraction            (fp_contraction)
    "uninit_read",   # read of uninitialized storage  (uninit_read)
})

#: integer kinds carry a machine width and must use a supported one.
_INTEGER_KINDS = frozenset({"binop_const", "shift", "div", "rem"})
_SUPPORTED_WIDTHS = frozenset({32, 64})

#: the per-operand operating-range keys and which kinds they are meaningful for.
_RANGE_KEYS = ("x_range", "a_range", "b_range", "shift_range")

#: known source/target language tokens (a *well-formed* pair may still be
#: unsupported by the registered oracles — that is a coverage matter, not an IR
#: error — but a non-string or unknown token is a contract violation).
_KNOWN_LANGS = frozenset({"c", "rust", "go", "swift"})


@dataclass(frozen=True)
class IRError:
    """A single contract violation, located by the field that violated it."""

    field: str
    message: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.field}: {self.message}"


class IRValidationError(ValueError):
    """Raised by :func:`assert_valid` when a unit violates the IR contract."""

    def __init__(self, label: str, errors: List[IRError]) -> None:
        self.label = label
        self.errors = errors
        joined = "; ".join(str(e) for e in errors)
        super().__init__(f"ill-formed IR unit {label!r}: {joined}")


def _check_int(unit: Dict, key: str, errors: List[IRError], *,
               required: bool = False) -> None:
    if key not in unit:
        if required:
            errors.append(IRError(key, "required field is missing"))
        return
    v = unit[key]
    if isinstance(v, bool) or not isinstance(v, int):
        errors.append(IRError(key, f"must be an integer, got {type(v).__name__}"))


def _check_str(unit: Dict, key: str, errors: List[IRError], *,
               required: bool = False) -> None:
    if key not in unit:
        if required:
            errors.append(IRError(key, "required field is missing"))
        return
    if not isinstance(unit[key], str) or not unit[key]:
        errors.append(IRError(key, "must be a non-empty string"))


def _check_range(unit: Dict, key: str, errors: List[IRError]) -> None:
    if key not in unit:
        return
    raw = unit[key]
    if (not isinstance(raw, (list, tuple))) or len(raw) != 2:
        errors.append(IRError(key, "operating range must be a [lo, hi] pair"))
        return
    lo, hi = raw
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (lo, hi)):
        errors.append(IRError(key, "range bounds must be integers"))
        return
    if lo > hi:
        errors.append(IRError(key, f"range has lo > hi ({lo} > {hi})"))


def validate_unit(unit: Dict, *, require_known_kind: bool = False) -> List[IRError]:
    """Return the list of contract violations for ``unit`` (empty == valid).

    With ``require_known_kind`` the unit's ``kind`` must be one the shipped
    oracles understand; by default an unknown-but-well-formed kind is accepted
    (it lowers fine; the engine just reports it ``NOT_COVERED``).
    """
    errors: List[IRError] = []

    if not isinstance(unit, dict):
        return [IRError("<unit>", f"must be a JSON object, got "
                                  f"{type(unit).__name__}")]

    kind = unit.get("kind")
    if kind is None:
        errors.append(IRError("kind", "required field is missing"))
    elif not isinstance(kind, str):
        errors.append(IRError("kind", "must be a string"))
    elif require_known_kind and kind not in KNOWN_KINDS:
        errors.append(IRError("kind", f"unknown kind {kind!r}; expected one of "
                                      f"{sorted(KNOWN_KINDS)}"))

    # language-pair envelope.
    for lk in ("source_lang", "target_lang"):
        if lk in unit:
            v = unit[lk]
            if not isinstance(v, str):
                errors.append(IRError(lk, "must be a string"))
            elif v not in _KNOWN_LANGS:
                errors.append(IRError(lk, f"unknown language {v!r}; known: "
                                          f"{sorted(_KNOWN_LANGS)}"))

    # probe must name a real divergence class if present.
    if "probe" in unit and unit["probe"] is not None:
        from .catalogue import CATALOGUE
        if unit["probe"] not in CATALOGUE:
            errors.append(IRError("probe", f"unknown divergence class "
                                           f"{unit['probe']!r}"))

    if "signed" in unit and not isinstance(unit["signed"], bool):
        errors.append(IRError("signed", "must be a boolean"))

    # width domain for the integer kinds.
    if isinstance(kind, str) and kind in _INTEGER_KINDS:
        w = unit.get("width", 32)
        if isinstance(w, bool) or not isinstance(w, int):
            errors.append(IRError("width", "must be an integer"))
        elif w not in _SUPPORTED_WIDTHS:
            errors.append(IRError("width", f"unsupported width {w}; "
                                           f"expected one of {sorted(_SUPPORTED_WIDTHS)}"))

    # per-kind required operands.
    if kind == "binop_const":
        _check_str(unit, "op", errors, required=True)
        _check_int(unit, "const", errors, required=True)
    elif kind == "array_index":
        if "length" not in unit:
            errors.append(IRError("length", "required field is missing"))
        else:
            _check_int(unit, "length", errors)
            if not any(e.field == "length" for e in errors) and unit["length"] <= 0:
                errors.append(IRError("length", "array length must be positive"))
    elif kind == "uninit_read":
        storage = unit.get("storage")
        if not isinstance(storage, dict):
            errors.append(IRError("storage", "required dict field is missing"))
        else:
            skind = storage.get("kind")
            if skind not in ("scalar", "array", "struct"):
                errors.append(IRError("storage.kind",
                                      f"unknown storage kind {skind!r}; expected "
                                      "scalar/array/struct"))
            valid_slots = None
            if skind == "array":
                if isinstance(storage.get("length"), bool) or \
                        not isinstance(storage.get("length"), int) or \
                        storage.get("length", 0) <= 0:
                    errors.append(IRError("storage.length",
                                          "array storage needs a positive length"))
                else:
                    valid_slots = set(range(storage["length"]))
            elif skind == "struct":
                fields = storage.get("fields")
                if not isinstance(fields, list) or not fields or \
                        not all(isinstance(f, str) for f in fields):
                    errors.append(IRError("storage.fields",
                                          "struct storage needs a non-empty list "
                                          "of field-name strings"))
                else:
                    valid_slots = set(fields)
            elif skind == "scalar":
                valid_slots = {None}
            writes = unit.get("writes", [])
            if not isinstance(writes, list):
                errors.append(IRError("writes", "must be a list of write specs"))
            elif valid_slots is not None:
                for i, w in enumerate(writes):
                    if not isinstance(w, dict) or "slot" not in w:
                        errors.append(IRError(f"writes[{i}]",
                                              "each write needs a 'slot'"))
                    elif w["slot"] not in valid_slots:
                        errors.append(IRError(f"writes[{i}].slot",
                                              f"slot {w['slot']!r} not in storage"))
            if "read" not in unit:
                errors.append(IRError("read", "required field is missing"))
            elif valid_slots is not None and unit["read"] not in valid_slots:
                errors.append(IRError("read",
                                      f"read slot {unit['read']!r} not in storage"))

    # operating ranges are always optional but must be well-formed if present.
    for rk in _RANGE_KEYS:
        _check_range(unit, rk, errors)

    return errors


def is_valid(unit: Dict, *, require_known_kind: bool = False) -> bool:
    return not validate_unit(unit, require_known_kind=require_known_kind)


def assert_valid(unit: Dict, *, label: Optional[str] = None,
                 require_known_kind: bool = False) -> None:
    """Raise :class:`IRValidationError` if ``unit`` violates the IR contract."""
    errors = validate_unit(unit, require_known_kind=require_known_kind)
    if errors:
        lbl = label or str(unit.get("name") or unit.get("id") or unit.get("kind")
                           or "<unit>")
        raise IRValidationError(lbl, errors)
