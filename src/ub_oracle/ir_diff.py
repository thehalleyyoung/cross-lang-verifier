"""Compiler-output IR diffing for semantic-divergence localization (Step 151).

The verifier already proves verdicts by re-executing real programs.  This module
adds a complementary debugging view: for one translated unit, ingest the source
compiler's IR (clang AST JSON) and the target compiler's IR (rustc MIR), normalize
the relevant semantic operations, and report the exact IR nodes/statements where
the two languages' semantics stop lining up.

For the C->Rust anchor, the first localization target is the bug class reviewers
always ask about: C signed arithmetic in clang's AST versus Rust's defined
overflow behavior in MIR (wrapping operations or MIR overflow assertions).  The
evidence is intentionally compiler-produced; if the compilers are missing or fail
to emit IR, the report is unavailable rather than fabricated.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .ir_ingest import (
    CLANG,
    RUSTC,
    IRModule,
    clang_ast_json,
    ingest_clang,
    ingest_rustc_mir,
    rustc_mir,
)


_C_ARITHMETIC = {"+": "add", "-": "sub", "*": "mul"}
_C_COMPARISONS = {
    ">": "gt",
    "<": "lt",
    ">=": "ge",
    "<=": "le",
    "==": "eq",
    "!=": "ne",
}
_RUST_WRAPPING = {
    "wrapping_add": "add",
    "wrapping_sub": "sub",
    "wrapping_mul": "mul",
}
_RUST_OVERFLOW = {
    "AddWithOverflow": "add",
    "SubWithOverflow": "sub",
    "MulWithOverflow": "mul",
}
_RUST_COMPARISONS = {
    "Gt": "gt",
    "Lt": "lt",
    "Ge": "ge",
    "Le": "le",
    "Eq": "eq",
    "Ne": "ne",
}
_RE_MIR_FN = re.compile(r"^fn\s+([A-Za-z_]\w*)\s*\(")
_RE_MIR_BLOCK = re.compile(r"^\s*(bb\d+):")


@dataclass(frozen=True)
class IRFact:
    """One normalized semantic fact extracted from real compiler IR."""

    language: str
    ir_kind: str
    function: str
    semantic: str
    op: str
    type: str
    line: Optional[int]
    column: Optional[int]
    block: Optional[str]
    evidence: str

    def to_dict(self) -> Dict:
        return {
            "language": self.language,
            "ir_kind": self.ir_kind,
            "function": self.function,
            "semantic": self.semantic,
            "op": self.op,
            "type": self.type,
            "line": self.line,
            "column": self.column,
            "block": self.block,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class CompilerIREvidence:
    """Normalized evidence from one real compiler IR dump."""

    language: str
    tool: str
    command: Tuple[str, ...]
    version: str
    ir_kind: str
    functions: Tuple[str, ...]
    facts: Tuple[IRFact, ...]
    normalized_hash: str

    def to_dict(self) -> Dict:
        return {
            "language": self.language,
            "tool": self.tool,
            "command": list(self.command),
            "version": self.version,
            "ir_kind": self.ir_kind,
            "functions": list(self.functions),
            "facts": [f.to_dict() for f in self.facts],
            "normalized_hash": self.normalized_hash,
        }


@dataclass(frozen=True)
class IRDiffHunk:
    """A localized mismatch between source and target compiler IR facts."""

    kind: str
    function: str
    severity: str
    source_fact: IRFact
    target_fact: IRFact
    explanation: str

    def to_dict(self) -> Dict:
        return {
            "kind": self.kind,
            "function": self.function,
            "severity": self.severity,
            "source_fact": self.source_fact.to_dict(),
            "target_fact": self.target_fact.to_dict(),
            "explanation": self.explanation,
        }


@dataclass(frozen=True)
class IRDiffReport:
    """Result of diffing one C->Rust translated unit at compiler-IR level."""

    available: bool
    source_evidence: Optional[CompilerIREvidence]
    target_evidence: Optional[CompilerIREvidence]
    hunks: Tuple[IRDiffHunk, ...]
    detail: str
    pair: str = "c->rust"

    @property
    def ok(self) -> bool:
        return (
            self.available
            and self.source_evidence is not None
            and self.target_evidence is not None
        )

    @property
    def has_semantic_mismatch(self) -> bool:
        return any(h.kind == "arithmetic_semantics_mismatch" for h in self.hunks)

    def to_dict(self) -> Dict:
        return {
            "available": self.available,
            "ok": self.ok,
            "pair": self.pair,
            "detail": self.detail,
            "source_evidence": (
                self.source_evidence.to_dict()
                if self.source_evidence is not None else None
            ),
            "target_evidence": (
                self.target_evidence.to_dict()
                if self.target_evidence is not None else None
            ),
            "hunks": [h.to_dict() for h in self.hunks],
        }


@dataclass(frozen=True)
class IRDiffConfirmation:
    """Live proof object for the Step-151 signed-overflow localization sample."""

    available: bool
    report: Optional[IRDiffReport]

    @property
    def ok(self) -> bool:
        return (
            self.available
            and self.report is not None
            and self.report.ok
            and self.report.has_semantic_mismatch
        )


def localize_compiler_ir_divergence(
    c_src: str,
    rust_src: str,
    *,
    function: Optional[str] = None,
) -> IRDiffReport:
    """Diff clang-AST and rustc-MIR facts for the same translated unit.

    ``function`` optionally restricts the comparison to one same-named function.
    Without it, every common function name is compared.  Missing compilers or
    malformed IR produce an unavailable report rather than a guessed result.
    """

    source = _clang_evidence(c_src, function=function)
    target = _rust_mir_evidence(rust_src, function=function)
    if source is None or target is None:
        return IRDiffReport(
            available=False,
            source_evidence=source,
            target_evidence=target,
            hunks=(),
            detail="clang AST and/or rustc MIR evidence unavailable",
        )

    functions = _functions_to_compare(source.functions, target.functions, function)
    hunks = tuple(_semantic_hunks(source.facts, target.facts, functions))
    detail = (
        f"diffed {len(functions)} common function(s) from real compiler IR; "
        f"localized {len(hunks)} semantic mismatch hunk(s)"
    )
    return IRDiffReport(
        available=True,
        source_evidence=source,
        target_evidence=target,
        hunks=hunks,
        detail=detail,
    )


def confirm_compiler_ir_diff() -> IRDiffConfirmation:
    """Run the canonical Step-151 localization against live clang and rustc."""

    c_src = "int f(int x){ return x + 1 > x; }\n"
    rust_src = (
        "pub fn f(x:i32)->i32 { ((x.wrapping_add(1) > x) as i32) }\n"
    )
    if not (os.path.exists(CLANG) and os.path.exists(RUSTC)):
        return IRDiffConfirmation(False, None)
    report = localize_compiler_ir_divergence(c_src, rust_src, function="f")
    return IRDiffConfirmation(report.available, report)


def _clang_evidence(src: str, *, function: Optional[str]) -> Optional[CompilerIREvidence]:
    if not os.path.exists(CLANG):
        return None
    ast = clang_ast_json(src)
    module = ingest_clang(src)
    if ast is None or module is None:
        return None
    facts = tuple(_clang_facts(ast, function=function))
    functions = _selected_functions(module, function)
    return _evidence(
        language="c",
        tool=CLANG,
        command=(CLANG, "-Xclang", "-ast-dump=json", "-fsyntax-only", "<unit>.c"),
        ir_kind="clang-ast-json",
        functions=functions,
        facts=facts,
    )


def _rust_mir_evidence(src: str, *, function: Optional[str]) -> Optional[CompilerIREvidence]:
    if not os.path.exists(RUSTC):
        return None
    mir = rustc_mir(src)
    module = ingest_rustc_mir(src)
    if mir is None or module is None:
        return None
    facts = tuple(_rust_mir_facts(mir, function=function))
    functions = _selected_functions(module, function)
    return _evidence(
        language="rust",
        tool=RUSTC,
        command=(RUSTC, "--emit=mir", "--crate-type=lib", "<unit>.rs"),
        ir_kind="rustc-mir",
        functions=functions,
        facts=facts,
    )


def _selected_functions(module: IRModule, function: Optional[str]) -> Tuple[str, ...]:
    names = sorted(module.functions)
    if function is not None:
        return tuple(n for n in names if n == function)
    return tuple(names)


def _evidence(
    *,
    language: str,
    tool: str,
    command: Tuple[str, ...],
    ir_kind: str,
    functions: Tuple[str, ...],
    facts: Tuple[IRFact, ...],
) -> CompilerIREvidence:
    ordered_facts = tuple(sorted(facts, key=_fact_key))
    payload = {
        "language": language,
        "ir_kind": ir_kind,
        "functions": list(functions),
        "facts": [f.to_dict() for f in ordered_facts],
    }
    return CompilerIREvidence(
        language=language,
        tool=tool,
        command=command,
        version=_compiler_version(tool),
        ir_kind=ir_kind,
        functions=functions,
        facts=ordered_facts,
        normalized_hash=_stable_hash(payload),
    )


def _compiler_version(path: str) -> str:
    try:
        r = subprocess.run(
            [path, "--version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    line = (r.stdout or r.stderr).splitlines()
    return line[0].strip() if line else "unavailable"


def _stable_hash(payload: Dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _fact_key(fact: IRFact) -> Tuple:
    return (
        fact.function,
        fact.semantic,
        fact.op,
        fact.type,
        fact.line if fact.line is not None else -1,
        fact.column if fact.column is not None else -1,
        fact.block or "",
        fact.evidence,
    )


def _clang_facts(ast: Dict, *, function: Optional[str]) -> Iterable[IRFact]:
    for node in ast.get("inner", []) or []:
        if node.get("kind") != "FunctionDecl" or "name" not in node:
            continue
        loc = node.get("loc", {}) or {}
        if not loc or loc.get("includedFrom") is not None:
            continue
        fn = str(node["name"])
        if function is not None and fn != function:
            continue
        yield from _walk_clang_node(node, fn, loc.get("line"), loc.get("col"))


def _walk_clang_node(
    node: Dict,
    function: str,
    inherited_line: Optional[int],
    inherited_col: Optional[int],
) -> Iterable[IRFact]:
    loc = node.get("loc", {}) or {}
    rng = node.get("range", {}) or {}
    begin = rng.get("begin", {}) or {}
    line = loc.get("line", begin.get("line", inherited_line))
    col = loc.get("col", begin.get("col", inherited_col))

    if node.get("kind") == "BinaryOperator":
        fact = _clang_binary_fact(node, function, line, col)
        if fact is not None:
            yield fact

    for child in node.get("inner", []) or []:
        yield from _walk_clang_node(child, function, line, col)


def _clang_binary_fact(
    node: Dict,
    function: str,
    line: Optional[int],
    col: Optional[int],
) -> Optional[IRFact]:
    opcode = str(node.get("opcode") or "")
    qual = str((node.get("type") or {}).get("qualType") or "")
    if opcode in _C_ARITHMETIC and _is_signed_integer_type(qual):
        op = _C_ARITHMETIC[opcode]
        semantic = f"c.signed_{op}"
    elif opcode in _C_COMPARISONS:
        op = _C_COMPARISONS[opcode]
        semantic = f"c.compare_{op}"
    else:
        return None
    return IRFact(
        language="c",
        ir_kind="clang-ast-json",
        function=function,
        semantic=semantic,
        op=op,
        type=qual,
        line=_int_or_none(line),
        column=_int_or_none(col),
        block=None,
        evidence=f"BinaryOperator opcode={opcode!r} qualType={qual!r}",
    )


def _is_signed_integer_type(qual: str) -> bool:
    q = " ".join(qual.replace("_Atomic", " ").split()).lower()
    if not q or "unsigned" in q or "*" in q:
        return False
    if any(tok in q for tok in ("float", "double", "_bool", "bool")):
        return False
    return bool(re.search(r"\b(signed|char|short|int|long)\b", q))


def _rust_mir_facts(mir: str, *, function: Optional[str]) -> Iterable[IRFact]:
    current_fn: Optional[str] = None
    current_block: Optional[str] = None
    brace_depth = 0

    for line_no, line in enumerate(mir.splitlines(), start=1):
        fn_match = _RE_MIR_FN.match(line)
        if fn_match is not None:
            current_fn = fn_match.group(1)
            current_block = None
            brace_depth = line.count("{") - line.count("}")
            continue

        if current_fn is None:
            continue

        stripped = line.strip()
        block_match = _RE_MIR_BLOCK.match(line)
        if block_match is not None:
            current_block = block_match.group(1)

        if function is None or current_fn == function:
            fact = _rust_mir_line_fact(stripped, current_fn, line_no, current_block)
            if fact is not None:
                yield fact

        brace_depth += line.count("{") - line.count("}")
        if brace_depth <= 0:
            current_fn = None
            current_block = None


def _rust_mir_line_fact(
    line: str,
    function: str,
    line_no: int,
    block: Optional[str],
) -> Optional[IRFact]:
    for name, op in _RUST_WRAPPING.items():
        if name in line:
            return IRFact(
                language="rust",
                ir_kind="rustc-mir",
                function=function,
                semantic=f"rust.{name}",
                op=op,
                type="",
                line=line_no,
                column=None,
                block=block,
                evidence=line,
            )
    for name, op in _RUST_OVERFLOW.items():
        if name in line:
            return IRFact(
                language="rust",
                ir_kind="rustc-mir",
                function=function,
                semantic=f"rust.{op}_with_overflow_assert",
                op=op,
                type="",
                line=line_no,
                column=None,
                block=block,
                evidence=line,
            )
    for name, op in _RUST_COMPARISONS.items():
        if re.search(r"\b" + re.escape(name) + r"\s*\(", line):
            return IRFact(
                language="rust",
                ir_kind="rustc-mir",
                function=function,
                semantic=f"rust.compare_{op}",
                op=op,
                type="bool",
                line=line_no,
                column=None,
                block=block,
                evidence=line,
            )
    return None


def _functions_to_compare(
    source_functions: Sequence[str],
    target_functions: Sequence[str],
    function: Optional[str],
) -> Tuple[str, ...]:
    if function is not None:
        return (function,) if function in source_functions and function in target_functions else ()
    return tuple(sorted(set(source_functions) & set(target_functions)))


def _semantic_hunks(
    source_facts: Sequence[IRFact],
    target_facts: Sequence[IRFact],
    functions: Sequence[str],
) -> Iterable[IRDiffHunk]:
    functions_set = set(functions)
    target_by_key: Dict[Tuple[str, str], List[IRFact]] = {}
    for fact in target_facts:
        if fact.function in functions_set:
            target_by_key.setdefault((fact.function, fact.op), []).append(fact)

    for source in source_facts:
        if source.function not in functions_set:
            continue
        if not source.semantic.startswith("c.signed_"):
            continue
        targets = [
            t for t in target_by_key.get((source.function, source.op), ())
            if t.semantic.startswith("rust.wrapping_")
            or t.semantic.endswith("_with_overflow_assert")
        ]
        for target in targets:
            yield IRDiffHunk(
                kind="arithmetic_semantics_mismatch",
                function=source.function,
                severity="high",
                source_fact=source,
                target_fact=target,
                explanation=_arithmetic_explanation(source, target),
            )


def _arithmetic_explanation(source: IRFact, target: IRFact) -> str:
    if target.semantic.startswith("rust.wrapping_"):
        rust_behavior = "wraps in Rust with a defined result"
    else:
        rust_behavior = "is guarded by Rust MIR overflow checking and can panic"
    return (
        f"clang AST exposes {source.semantic} at this C BinaryOperator; signed "
        f"{source.op} overflow is undefined in C. rustc MIR for the translated "
        f"function uses {target.semantic}, which {rust_behavior}. The semantic "
        "gap is localized to these two compiler-IR facts."
    )


def _int_or_none(value: object) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":  # pragma: no cover
    confirmation = confirm_compiler_ir_diff()
    if confirmation.report is None:
        print(json.dumps({"available": False, "ok": False}, indent=2))
        raise SystemExit(1)
    print(json.dumps(confirmation.report.to_dict(), indent=2, sort_keys=True))
    raise SystemExit(0 if confirmation.ok else 1)
