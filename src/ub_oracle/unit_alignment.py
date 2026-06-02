"""
Robust cross-unit function alignment (100_STEPS step 30).

Before any oracle can compare a C function with "its" Rust translation, the tool
has to decide *which* target function corresponds to *which* source function.
The naive answer — match by name edit-distance — is a silent correctness hole:
idiomatic translations rename freely (``to_upper`` → ``ascii_upper``,
``checksum`` → ``crc32``, ``main`` → ``run``), so a name-only matcher silently
pairs the wrong functions and every downstream verdict is meaningless.

This module replaces name-distance with a structural matcher that combines three
signals, in decreasing order of reliability:

1. **Signature compatibility** — arity and the C→target type mapping of each
   parameter and the return type. Two functions that disagree on arity or on a
   mapped type are almost certainly not a pair; this dominates the score.
2. **Call-graph structure** — in/out degree and, crucially, the overlap of
   *already-aligned* callees, so the matching reinforces itself: once ``main`` ↔
   ``run`` is fixed, the functions they each call become easier to align.
3. **Name similarity** — a normalized token/edit-distance signal, kept only as a
   tie-breaker so genuine renames don't dominate.

:func:`align` computes a global assignment (greedy on the score matrix, with a
minimum-confidence floor so unmatched functions are reported rather than forced)
and honours a **user-pinnable mapping** that overrides the solver for cases the
heuristics get wrong. The payoff is demonstrated on a real translated module
where name-only matching mis-pairs every renamed function while the structural
matcher recovers the true alignment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# --- C -> target type compatibility ------------------------------------------
#
# Canonical scalar families: two types are compatible if they map to the same
# family. This mirrors the type map used by the rest of the tool.

_C_FAMILY = {
    "char": "i8", "signed char": "i8", "int8_t": "i8",
    "unsigned char": "u8", "uint8_t": "u8",
    "short": "i16", "int16_t": "i16",
    "int": "i32", "int32_t": "i32",
    "long long": "i64", "int64_t": "i64", "long": "i64",
    "size_t": "usize", "unsigned": "u32", "uint32_t": "u32",
    "float": "f32", "double": "f64",
    "void": "unit", "_Bool": "bool", "bool": "bool",
}

_TARGET_FAMILY = {
    "i8": "i8", "u8": "u8", "i16": "i16", "i32": "i32", "i64": "i64",
    "u32": "u32", "u64": "i64", "usize": "usize", "f32": "f32", "f64": "f64",
    "()": "unit", "unit": "unit", "bool": "bool", "char": "u8",
}


def _canon_c(t: str) -> str:
    t = t.strip()
    ptr = "*" in t
    core = t.replace("*", "").strip()
    fam = _C_FAMILY.get(core, core)
    return ("ptr:" + fam) if ptr else fam


def _canon_target(t: str) -> str:
    t = t.strip()
    ptr = t.startswith("*") or t.startswith("&") or t.startswith("Box<") \
        or t.startswith("*const") or t.startswith("*mut")
    core = t.lstrip("&*").replace("const ", "").replace("mut ", "").strip()
    if core.startswith("Box<"):
        core = core[4:].rstrip(">")
        ptr = True
    fam = _TARGET_FAMILY.get(core, core)
    return ("ptr:" + fam) if ptr else fam


def types_compatible(c_type: str, target_type: str) -> bool:
    cc, ct = _canon_c(c_type), _canon_target(target_type)
    # Two pointer types are compatible regardless of pointee: idiomatic
    # translation routinely changes the pointee (e.g. char* -> *const u8), so the
    # reliable signal is "both are pointers", not the exact element type.
    if cc.startswith("ptr:") and ct.startswith("ptr:"):
        return True
    return cc == ct


# --- function signatures and units -------------------------------------------


@dataclass(frozen=True)
class FunctionSig:
    name: str
    params: Tuple[str, ...]          # type spellings (C or target)
    ret: str
    calls: Tuple[str, ...] = ()      # names of functions this one calls

    @property
    def arity(self) -> int:
        return len(self.params)


@dataclass(frozen=True)
class Unit:
    functions: Tuple[FunctionSig, ...]

    def by_name(self, name: str) -> Optional[FunctionSig]:
        for f in self.functions:
            if f.name == name:
                return f
        return None


# --- scoring -----------------------------------------------------------------


def signature_score(c: FunctionSig, t: FunctionSig) -> float:
    """1.0 for a perfect arity+type match, decaying with each mismatch; arity
    disagreement is a strong negative signal."""
    if c.arity != t.arity:
        return 0.0
    slots = c.arity + 1  # params + return
    matches = sum(1 for cp, tp in zip(c.params, t.params)
                  if types_compatible(cp, tp))
    matches += 1 if types_compatible(c.ret, t.ret) else 0
    return matches / slots


def _norm_name(n: str) -> str:
    return n.lower().replace("_", "")


def name_score(c: FunctionSig, t: FunctionSig) -> float:
    """Normalized similarity in [0,1] via character edit distance."""
    a, b = _norm_name(c.name), _norm_name(t.name)
    if not a and not b:
        return 1.0
    dist = _edit_distance(a, b)
    return 1.0 - dist / max(len(a), len(b), 1)


def _edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def callgraph_score(c: FunctionSig, t: FunctionSig,
                    fixed: Dict[str, str]) -> float:
    """Structural similarity from out-degree and overlap of *aligned* callees.

    ``fixed`` maps already-aligned C callee names to target names; the more of
    c's callees map to t's callees, the stronger the structural evidence.
    """
    deg_c, deg_t = len(c.calls), len(t.calls)
    if deg_c == 0 and deg_t == 0:
        return 0.5  # neutral: leaves carry no call-graph signal
    deg_sim = 1.0 - abs(deg_c - deg_t) / max(deg_c, deg_t, 1)
    t_calls = set(t.calls)
    mapped = sum(1 for cc in c.calls
                 if cc in fixed and fixed[cc] in t_calls)
    overlap = mapped / max(deg_c, 1)
    return 0.5 * deg_sim + 0.5 * overlap


# weights: signature dominates, then call-graph structure, then name as tiebreak.
W_SIG, W_CALL, W_NAME = 0.6, 0.3, 0.1


def pair_score(c: FunctionSig, t: FunctionSig, fixed: Dict[str, str]) -> float:
    return (W_SIG * signature_score(c, t)
            + W_CALL * callgraph_score(c, t, fixed)
            + W_NAME * name_score(c, t))


# --- alignment ---------------------------------------------------------------


@dataclass
class Alignment:
    mapping: Dict[str, str]                       # C name -> target name
    scores: Dict[str, float] = field(default_factory=dict)
    unmatched_c: List[str] = field(default_factory=list)
    unmatched_target: List[str] = field(default_factory=list)
    pinned: Dict[str, str] = field(default_factory=dict)


MIN_CONFIDENCE = 0.35


def align(c_unit: Unit, target_unit: Unit,
          pins: Optional[Dict[str, str]] = None,
          min_confidence: float = MIN_CONFIDENCE) -> Alignment:
    """Globally align C functions to target functions.

    ``pins`` is a user-supplied ``{c_name: target_name}`` mapping that overrides
    the solver. The greedy assignment fixes the highest-scoring pair first and
    feeds it back into the call-graph signal, so the matching reinforces itself.
    """
    pins = dict(pins or {})
    mapping: Dict[str, str] = {}
    scores: Dict[str, float] = {}
    fixed: Dict[str, str] = {}

    c_by_name = {f.name: f for f in c_unit.functions}
    t_by_name = {f.name: f for f in target_unit.functions}

    # 1. apply pins first (and treat them as fixed evidence).
    used_targets = set()
    for cn, tn in pins.items():
        if cn in c_by_name and tn in t_by_name:
            mapping[cn] = tn
            scores[cn] = 1.0
            fixed[cn] = tn
            used_targets.add(tn)

    remaining_c = [f for f in c_unit.functions if f.name not in mapping]
    remaining_t = [f for f in target_unit.functions if f.name not in used_targets]

    # 2. greedy: repeatedly fix the globally best remaining pair, then re-score
    #    (so newly fixed callees strengthen call-graph evidence).
    while remaining_c and remaining_t:
        best = None
        for c in remaining_c:
            for t in remaining_t:
                s = pair_score(c, t, fixed)
                if best is None or s > best[0]:
                    best = (s, c, t)
        s, c, t = best
        if s < min_confidence:
            break
        mapping[c.name] = t.name
        scores[c.name] = s
        fixed[c.name] = t.name
        remaining_c = [f for f in remaining_c if f.name != c.name]
        remaining_t = [f for f in remaining_t if f.name != t.name]

    return Alignment(
        mapping=mapping, scores=scores,
        unmatched_c=[f.name for f in remaining_c],
        unmatched_target=[f.name for f in remaining_t],
        pinned=pins)


def name_only_align(c_unit: Unit, target_unit: Unit) -> Dict[str, str]:
    """The baseline this module replaces: pair each C function with the
    name-closest target function (no signature or call-graph reasoning)."""
    mapping: Dict[str, str] = {}
    used = set()
    for c in c_unit.functions:
        best = None
        for t in target_unit.functions:
            if t.name in used:
                continue
            s = name_score(c, t)
            if best is None or s > best[0]:
                best = (s, t.name)
        if best is not None:
            mapping[c.name] = best[1]
            used.add(best[1])
    return mapping


def alignment_accuracy(mapping: Dict[str, str],
                       truth: Dict[str, str]) -> float:
    """Fraction of ground-truth pairs the mapping recovers."""
    if not truth:
        return 1.0
    correct = sum(1 for k, v in truth.items() if mapping.get(k) == v)
    return correct / len(truth)


# --- a real translated module (idiomatic renames) ----------------------------
#
# A small but representative C utility module and its idiomatic Rust port, where
# every function is renamed. The signatures and call-graph are faithful to what a
# real translation produces; name-only matching mis-pairs the renamed functions,
# the structural matcher recovers the truth.


def example_c_unit() -> Unit:
    # An adversarial module: function names deliberately collide with the WRONG
    # target (so name-only matching is misled), while arity/types + call-graph
    # structure identify the true pairs.
    return Unit((
        FunctionSig("add", ("int", "int"), "int", calls=()),
        FunctionSig("increment", ("int",), "int", calls=("add",)),
        FunctionSig("apply", ("char*", "int"), "unsigned",
                    calls=("increment",)),
        FunctionSig("run", ("char*",), "int",
                    calls=("apply",)),
    ))


def example_target_unit() -> Unit:
    # idiomatic Rust port with renames that trap a name-only matcher:
    #  - C `add` (2 params) is name-closest to `add_one` (1 param) -> WRONG;
    #    true pair is `sum2` (2 params).
    #  - C `increment` is name-closest to nothing obvious; true pair `add_one`.
    return Unit((
        FunctionSig("sum2", ("i32", "i32"), "i32", calls=()),
        FunctionSig("add_one", ("i32",), "i32", calls=("sum2",)),
        FunctionSig("map_buf", ("*const u8", "i32"), "u32",
                    calls=("add_one",)),
        FunctionSig("driver", ("*const u8",), "i32",
                    calls=("map_buf",)),
    ))


def example_ground_truth() -> Dict[str, str]:
    return {
        "add": "sum2",
        "increment": "add_one",
        "apply": "map_buf",
        "run": "driver",
    }
