"""
Cross-language equivalence verification engine for comparing C and Rust code.
Performs structural, semantic, and behavioral comparison of parsed ASTs.
"""

from __future__ import annotations
import re
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class Severity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Divergence:
    kind: str
    location_c: Optional[str]
    location_rust: Optional[str]
    description: str
    severity: str  # "low" / "medium" / "high" / "critical"
    suggestion: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "location_c": self.location_c,
            "location_rust": self.location_rust,
            "description": self.description,
            "severity": self.severity,
            "suggestion": self.suggestion,
        }


@dataclass
class FunctionMapping:
    c_name: str
    rust_name: str
    score: float  # 0-1 similarity
    method: str   # "exact", "snake_case", "levenshtein", "signature"


@dataclass
class TypeMapping:
    c_type: str
    rust_type: str
    compatible: bool
    notes: str = ""


@dataclass
class EquivalenceResult:
    equivalent: bool
    divergences: List[Divergence]
    confidence: float  # 0-1
    function_mappings: List[FunctionMapping]
    type_mappings: List[TypeMapping]

    def summary(self) -> str:
        status = "EQUIVALENT" if self.equivalent else "NOT EQUIVALENT"
        n_div = len(self.divergences)
        crit = sum(1 for d in self.divergences if d.severity == "critical")
        return (
            f"{status} (confidence={self.confidence:.2f}, "
            f"divergences={n_div}, critical={crit})"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "equivalent": self.equivalent,
            "confidence": self.confidence,
            "divergences": [d.to_dict() for d in self.divergences],
            "function_mappings": [
                {"c": fm.c_name, "rust": fm.rust_name,
                 "score": fm.score, "method": fm.method}
                for fm in self.function_mappings
            ],
            "type_mappings": [
                {"c": tm.c_type, "rust": tm.rust_type,
                 "compatible": tm.compatible, "notes": tm.notes}
                for tm in self.type_mappings
            ],
        }


# ---------------------------------------------------------------------------
# Control-flow graph
# ---------------------------------------------------------------------------

@dataclass
class CFGEdge:
    source_id: int
    target_id: int
    label: str = ""  # "true", "false", "unconditional", "case:X", "default"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CFGEdge):
            return NotImplemented
        return (self.source_id == other.source_id
                and self.target_id == other.target_id
                and self.label == other.label)

    def __hash__(self) -> int:
        return hash((self.source_id, self.target_id, self.label))


@dataclass
class CFGNode:
    node_id: int
    kind: str  # "entry", "exit", "basic", "branch", "loop_header", "switch", "match"
    statements: List[Any] = field(default_factory=list)
    edges_out: List[CFGEdge] = field(default_factory=list)
    edges_in: List[CFGEdge] = field(default_factory=list)
    label: Optional[str] = None  # for goto targets / labeled blocks

    @property
    def successors(self) -> List[int]:
        return [e.target_id for e in self.edges_out]

    @property
    def predecessors(self) -> List[int]:
        return [e.source_id for e in self.edges_in]

    def is_empty(self) -> bool:
        return len(self.statements) == 0 and self.kind == "basic"


@dataclass
class CFG:
    nodes: Dict[int, CFGNode] = field(default_factory=dict)
    entry_id: int = 0
    exit_id: int = -1

    def add_node(self, node: CFGNode) -> None:
        self.nodes[node.node_id] = node

    def add_edge(self, src: int, dst: int, label: str = "") -> None:
        edge = CFGEdge(src, dst, label)
        if src in self.nodes:
            self.nodes[src].edges_out.append(edge)
        if dst in self.nodes:
            self.nodes[dst].edges_in.append(edge)

    def remove_node(self, nid: int) -> None:
        if nid not in self.nodes:
            return
        node = self.nodes[nid]
        for e_in in list(node.edges_in):
            src_node = self.nodes.get(e_in.source_id)
            if src_node:
                src_node.edges_out = [
                    e for e in src_node.edges_out if e.target_id != nid
                ]
                for e_out in node.edges_out:
                    new_label = e_in.label if e_in.label else e_out.label
                    self.add_edge(e_in.source_id, e_out.target_id, new_label)
        for e_out in list(node.edges_out):
            tgt_node = self.nodes.get(e_out.target_id)
            if tgt_node:
                tgt_node.edges_in = [
                    e for e in tgt_node.edges_in if e.source_id != nid
                ]
        del self.nodes[nid]

    def node_count(self) -> int:
        return len(self.nodes)

    def edge_count(self) -> int:
        return sum(len(n.edges_out) for n in self.nodes.values())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_levenshtein_distance(s: str, t: str) -> int:
    n, m = len(s), len(t)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            cost = 0 if s[i - 1] == t[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,       # deletion
                curr[j - 1] + 1,   # insertion
                prev[j - 1] + cost # substitution
            )
        prev, curr = curr, prev
    return prev[m]


def normalize_identifier(name: str) -> str:
    """Convert CamelCase / mixedCase to snake_case and lower."""
    result: List[str] = []
    for i, ch in enumerate(name):
        if ch.isupper():
            if i > 0 and name[i - 1].islower():
                result.append("_")
            elif (i > 0 and i + 1 < len(name)
                  and name[i - 1].isupper() and name[i + 1].islower()):
                result.append("_")
            result.append(ch.lower())
        else:
            result.append(ch)
    return "".join(result).strip("_")


def _camel_to_snake(name: str) -> str:
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


# ---------------------------------------------------------------------------
# Type equivalence tables
# ---------------------------------------------------------------------------

_C_TO_RUST_PRIMITIVE: Dict[str, List[str]] = {
    "int": ["i32"],
    "signed int": ["i32"],
    "unsigned int": ["u32"],
    "unsigned": ["u32"],
    "short": ["i16"],
    "unsigned short": ["u16"],
    "long": ["i64"],
    "long long": ["i64"],
    "unsigned long": ["u64"],
    "unsigned long long": ["u64"],
    "char": ["i8", "u8"],
    "signed char": ["i8"],
    "unsigned char": ["u8"],
    "float": ["f32"],
    "double": ["f64"],
    "long double": ["f64"],
    "void": ["()"],
    "_Bool": ["bool"],
    "bool": ["bool"],
    "int8_t": ["i8"],
    "int16_t": ["i16"],
    "int32_t": ["i32"],
    "int64_t": ["i64"],
    "uint8_t": ["u8"],
    "uint16_t": ["u16"],
    "uint32_t": ["u32"],
    "uint64_t": ["u64"],
    "size_t": ["usize"],
    "ssize_t": ["isize"],
    "ptrdiff_t": ["isize"],
    "intptr_t": ["isize"],
    "uintptr_t": ["usize"],
}

_C_POINTER_TO_RUST: Dict[str, List[str]] = {
    "char*": ["&str", "String", "CStr", "CString", "*const c_char", "*mut c_char"],
    "const char*": ["&str", "String", "CStr", "*const c_char"],
    "void*": ["*mut u8", "*const u8", "*mut c_void", "*const c_void"],
    "const void*": ["*const u8", "*const c_void"],
}

_C_ALLOC_TO_RUST: Dict[str, List[str]] = {
    "malloc": ["Box::new", "Vec::with_capacity", "vec![]", "alloc::alloc"],
    "calloc": ["vec![0;", "Vec::with_capacity", "Box::new"],
    "realloc": ["Vec::reserve", "Vec::resize", "vec![]"],
    "free": ["drop", "std::mem::drop"],
}

_C_ERROR_PATTERNS = {
    "return -1": "error_code_minus1",
    "return NULL": "error_null",
    "return 0": "error_zero_failure",
    "errno": "errno_set",
    "goto cleanup": "goto_cleanup",
    "goto error": "goto_error",
    "goto fail": "goto_fail",
}

_RUST_ERROR_PATTERNS = {
    "Err(": "result_err",
    "None": "option_none",
    "?": "question_mark",
    "unwrap()": "unwrap_call",
    "expect(": "expect_call",
    "panic!": "panic_macro",
}


# ---------------------------------------------------------------------------
# Expression simplifier
# ---------------------------------------------------------------------------

@dataclass
class Expr:
    kind: str  # "literal", "ident", "binop", "unop", "call", "index", "member", "cast"
    value: Any = None
    children: List["Expr"] = field(default_factory=list)
    op: str = ""
    type_info: str = ""


def simplify_expression(expr: Expr) -> Expr:
    """Algebraic simplification: constant folding, identity removal, etc."""
    if expr.kind == "literal":
        return expr
    if expr.kind == "ident":
        return expr

    simplified_children = [simplify_expression(c) for c in expr.children]
    expr = Expr(
        kind=expr.kind, value=expr.value,
        children=simplified_children, op=expr.op,
        type_info=expr.type_info,
    )

    if expr.kind == "binop" and len(expr.children) == 2:
        left, right = expr.children
        # constant folding
        if left.kind == "literal" and right.kind == "literal":
            lv, rv = left.value, right.value
            if isinstance(lv, (int, float)) and isinstance(rv, (int, float)):
                result = _eval_binop(expr.op, lv, rv)
                if result is not None:
                    return Expr(kind="literal", value=result)

        # identity: x + 0 = x, x - 0 = x
        if expr.op in ("+", "-") and right.kind == "literal" and right.value == 0:
            return left
        # identity: 0 + x = x
        if expr.op == "+" and left.kind == "literal" and left.value == 0:
            return right
        # identity: x * 1 = x
        if expr.op == "*" and right.kind == "literal" and right.value == 1:
            return left
        if expr.op == "*" and left.kind == "literal" and left.value == 1:
            return right
        # x * 0 = 0
        if expr.op == "*" and (
            (right.kind == "literal" and right.value == 0)
            or (left.kind == "literal" and left.value == 0)
        ):
            return Expr(kind="literal", value=0)
        # x / 1 = x
        if expr.op == "/" and right.kind == "literal" and right.value == 1:
            return left
        # x ^ 0 = 0, x & 0 = 0
        if expr.op == "^" and _exprs_equal(left, right):
            return Expr(kind="literal", value=0)
        if expr.op == "&" and right.kind == "literal" and right.value == 0:
            return Expr(kind="literal", value=0)
        # x | 0 = x
        if expr.op == "|" and right.kind == "literal" and right.value == 0:
            return left
        # x && true = x
        if expr.op == "&&" and right.kind == "literal" and right.value is True:
            return left
        # x || false = x
        if expr.op == "||" and right.kind == "literal" and right.value is False:
            return left

    if expr.kind == "unop" and len(expr.children) == 1:
        child = expr.children[0]
        # double negation: !!x = x (boolean context)
        if expr.op == "!" and child.kind == "unop" and child.op == "!":
            return child.children[0] if child.children else expr
        # -(-x) = x
        if expr.op == "-" and child.kind == "unop" and child.op == "-":
            return child.children[0] if child.children else expr
        # constant fold unary
        if child.kind == "literal" and isinstance(child.value, (int, float)):
            if expr.op == "-":
                return Expr(kind="literal", value=-child.value)
            if expr.op == "~" and isinstance(child.value, int):
                return Expr(kind="literal", value=~child.value)
            if expr.op == "!":
                return Expr(kind="literal", value=not child.value)

    return expr


def _eval_binop(op: str, lv: Any, rv: Any) -> Any:
    try:
        if op == "+":
            return lv + rv
        if op == "-":
            return lv - rv
        if op == "*":
            return lv * rv
        if op == "/" and rv != 0:
            if isinstance(lv, int) and isinstance(rv, int):
                return lv // rv
            return lv / rv
        if op == "%" and rv != 0 and isinstance(lv, int) and isinstance(rv, int):
            return lv % rv
        if op == "<<" and isinstance(lv, int) and isinstance(rv, int) and rv >= 0:
            return lv << rv
        if op == ">>" and isinstance(lv, int) and isinstance(rv, int) and rv >= 0:
            return lv >> rv
        if op == "&" and isinstance(lv, int) and isinstance(rv, int):
            return lv & rv
        if op == "|" and isinstance(lv, int) and isinstance(rv, int):
            return lv | rv
        if op == "^" and isinstance(lv, int) and isinstance(rv, int):
            return lv ^ rv
        if op == "==":
            return lv == rv
        if op == "!=":
            return lv != rv
        if op == "<":
            return lv < rv
        if op == ">":
            return lv > rv
        if op == "<=":
            return lv <= rv
        if op == ">=":
            return lv >= rv
        if op == "&&":
            return bool(lv) and bool(rv)
        if op == "||":
            return bool(lv) or bool(rv)
    except (OverflowError, ZeroDivisionError, ValueError):
        return None
    return None


def _exprs_equal(a: Expr, b: Expr) -> bool:
    if a.kind != b.kind or a.op != b.op or a.value != b.value:
        return False
    if len(a.children) != len(b.children):
        return False
    return all(_exprs_equal(ac, bc) for ac, bc in zip(a.children, b.children))


def _exprs_commutative_equal(a: Expr, b: Expr) -> bool:
    """Check equality allowing commutativity of +, *, &, |, ^, ==, !=, &&, ||."""
    if a.kind != b.kind or a.op != b.op:
        return False
    if a.kind == "literal":
        return a.value == b.value
    if a.kind == "ident":
        return a.value == b.value
    if len(a.children) != len(b.children):
        return False
    if a.kind == "binop" and a.op in ("+", "*", "&", "|", "^", "==", "!=", "&&", "||"):
        if len(a.children) == 2 and len(b.children) == 2:
            direct = (_exprs_commutative_equal(a.children[0], b.children[0])
                      and _exprs_commutative_equal(a.children[1], b.children[1]))
            swapped = (_exprs_commutative_equal(a.children[0], b.children[1])
                       and _exprs_commutative_equal(a.children[1], b.children[0]))
            return direct or swapped
    return all(
        _exprs_commutative_equal(ac, bc)
        for ac, bc in zip(a.children, b.children)
    )


# ---------------------------------------------------------------------------
# CFG builder
# ---------------------------------------------------------------------------

class _CFGBuilder:
    def __init__(self) -> None:
        self._next_id = 0
        self.cfg = CFG()

    def _new_id(self) -> int:
        nid = self._next_id
        self._next_id += 1
        return nid

    def _make_node(self, kind: str = "basic", stmts: Optional[List[Any]] = None,
                   label: Optional[str] = None) -> CFGNode:
        nid = self._new_id()
        node = CFGNode(node_id=nid, kind=kind,
                       statements=stmts if stmts else [], label=label)
        self.cfg.add_node(node)
        return node

    def build(self, ast_body: List[Dict[str, Any]], language: str = "c") -> CFG:
        entry = self._make_node("entry")
        self.cfg.entry_id = entry.node_id
        exit_node = self._make_node("exit")
        self.cfg.exit_id = exit_node.node_id

        if not ast_body:
            self.cfg.add_edge(entry.node_id, exit_node.node_id, "unconditional")
            return self.cfg

        last_nodes = self._process_stmts(ast_body, [entry.node_id], language)
        for nid in last_nodes:
            self.cfg.add_edge(nid, exit_node.node_id, "unconditional")
        return self.cfg

    def _process_stmts(self, stmts: List[Dict[str, Any]],
                       predecessors: List[int],
                       language: str) -> List[int]:
        current_preds = list(predecessors)
        for stmt in stmts:
            current_preds = self._process_stmt(stmt, current_preds, language)
            if not current_preds:
                break
        return current_preds

    def _process_stmt(self, stmt: Dict[str, Any],
                      predecessors: List[int],
                      language: str) -> List[int]:
        kind = stmt.get("kind", "expr")

        if kind in ("if", "if_else"):
            return self._process_if(stmt, predecessors, language)
        if kind in ("while", "for", "do_while", "loop"):
            return self._process_loop(stmt, predecessors, language)
        if kind in ("switch", "match"):
            return self._process_switch(stmt, predecessors, language)
        if kind == "return":
            node = self._make_node("basic", [stmt])
            for p in predecessors:
                self.cfg.add_edge(p, node.node_id, "unconditional")
            self.cfg.add_edge(node.node_id, self.cfg.exit_id, "unconditional")
            return []
        if kind == "goto":
            node = self._make_node("basic", [stmt])
            for p in predecessors:
                self.cfg.add_edge(p, node.node_id, "unconditional")
            return []
        if kind == "break":
            node = self._make_node("basic", [stmt])
            for p in predecessors:
                self.cfg.add_edge(p, node.node_id, "unconditional")
            return []
        if kind == "continue":
            node = self._make_node("basic", [stmt])
            for p in predecessors:
                self.cfg.add_edge(p, node.node_id, "unconditional")
            return []
        if kind == "block":
            return self._process_stmts(
                stmt.get("body", []), predecessors, language
            )
        if kind == "label":
            node = self._make_node("basic", [stmt], label=stmt.get("name"))
            for p in predecessors:
                self.cfg.add_edge(p, node.node_id, "unconditional")
            inner = stmt.get("body", [])
            if inner:
                return self._process_stmts(inner, [node.node_id], language)
            return [node.node_id]

        # default: simple statement
        node = self._make_node("basic", [stmt])
        for p in predecessors:
            self.cfg.add_edge(p, node.node_id, "unconditional")
        return [node.node_id]

    def _process_if(self, stmt: Dict[str, Any],
                    predecessors: List[int],
                    language: str) -> List[int]:
        branch = self._make_node("branch", [stmt.get("condition", {})])
        for p in predecessors:
            self.cfg.add_edge(p, branch.node_id, "unconditional")

        then_body = stmt.get("then", stmt.get("body", []))
        if not isinstance(then_body, list):
            then_body = [then_body]
        then_exits = self._process_stmts(then_body, [branch.node_id], language)
        # add "true" label on the edge from branch to first then-node
        for e in branch.edges_out:
            if e.target_id != branch.node_id:
                e.label = "true"
                break

        else_body = stmt.get("else", [])
        if else_body:
            if not isinstance(else_body, list):
                else_body = [else_body]
            else_exits = self._process_stmts(else_body, [branch.node_id], language)
            for e in branch.edges_out:
                if e.label != "true":
                    e.label = "false"
                    break
        else:
            else_exits = [branch.node_id]

        return then_exits + else_exits

    def _process_loop(self, stmt: Dict[str, Any],
                      predecessors: List[int],
                      language: str) -> List[int]:
        header = self._make_node("loop_header", [stmt.get("condition", {})])
        for p in predecessors:
            self.cfg.add_edge(p, header.node_id, "unconditional")

        body = stmt.get("body", [])
        if not isinstance(body, list):
            body = [body]
        body_exits = self._process_stmts(body, [header.node_id], language)
        for nid in body_exits:
            self.cfg.add_edge(nid, header.node_id, "back_edge")

        after = self._make_node("basic")
        self.cfg.add_edge(header.node_id, after.node_id, "false")
        return [after.node_id]

    def _process_switch(self, stmt: Dict[str, Any],
                        predecessors: List[int],
                        language: str) -> List[int]:
        sw_kind = "switch" if language == "c" else "match"
        switch_node = self._make_node(sw_kind, [stmt.get("expr", {})])
        for p in predecessors:
            self.cfg.add_edge(p, switch_node.node_id, "unconditional")

        exits: List[int] = []
        cases = stmt.get("cases", stmt.get("arms", []))
        for case in cases:
            label = case.get("value", case.get("pattern", "default"))
            case_body = case.get("body", [])
            if not isinstance(case_body, list):
                case_body = [case_body]
            case_exits = self._process_stmts(
                case_body, [switch_node.node_id], language
            )
            for e in switch_node.edges_out:
                if e.label == "":
                    e.label = f"case:{label}"
                    break
            exits.extend(case_exits)

        if not exits:
            exits = [switch_node.node_id]
        return exits


def build_cfg(ast_body: List[Dict[str, Any]], language: str = "c") -> CFG:
    builder = _CFGBuilder()
    return builder.build(ast_body, language)


def _normalize_cfg(cfg: CFG) -> CFG:
    """Remove empty basic blocks and merge linear chains."""
    changed = True
    while changed:
        changed = False
        to_remove: List[int] = []
        for nid, node in list(cfg.nodes.items()):
            if nid in (cfg.entry_id, cfg.exit_id):
                continue
            if node.is_empty() and len(node.edges_out) == 1:
                to_remove.append(nid)
                changed = True
        for nid in to_remove:
            cfg.remove_node(nid)

    # merge linear chains: A->B where A has single successor B and B has single predecessor A
    changed = True
    while changed:
        changed = False
        for nid, node in list(cfg.nodes.items()):
            if nid not in cfg.nodes:
                continue
            if len(node.edges_out) != 1:
                continue
            succ_id = node.edges_out[0].target_id
            if succ_id == nid or succ_id not in cfg.nodes:
                continue
            succ = cfg.nodes[succ_id]
            if len(succ.edges_in) != 1:
                continue
            if succ_id in (cfg.entry_id, cfg.exit_id):
                continue
            if succ.kind not in ("basic",):
                continue
            node.statements.extend(succ.statements)
            node.edges_out = []
            for e in succ.edges_out:
                cfg.add_edge(nid, e.target_id, e.label)
                tgt = cfg.nodes.get(e.target_id)
                if tgt:
                    tgt.edges_in = [
                        ei for ei in tgt.edges_in if ei.source_id != succ_id
                    ]
            del cfg.nodes[succ_id]
            changed = True

    return cfg


# ---------------------------------------------------------------------------
# CFG isomorphism checker
# ---------------------------------------------------------------------------

class _CFGIsomorphismChecker:
    def __init__(self, cfg_a: CFG, cfg_b: CFG) -> None:
        self.cfg_a = cfg_a
        self.cfg_b = cfg_b

    def check(self) -> Tuple[bool, float]:
        if self.cfg_a.node_count() != self.cfg_b.node_count():
            ratio = 1.0 - abs(
                self.cfg_a.node_count() - self.cfg_b.node_count()
            ) / max(self.cfg_a.node_count(), self.cfg_b.node_count(), 1)
            return False, max(0.0, ratio * 0.5)

        if self.cfg_a.edge_count() != self.cfg_b.edge_count():
            ratio = 1.0 - abs(
                self.cfg_a.edge_count() - self.cfg_b.edge_count()
            ) / max(self.cfg_a.edge_count(), self.cfg_b.edge_count(), 1)
            return False, max(0.0, ratio * 0.5)

        sig_a = self._signature(self.cfg_a)
        sig_b = self._signature(self.cfg_b)

        if sig_a == sig_b:
            return True, 1.0

        common = sum(min(sig_a.get(k, 0), sig_b.get(k, 0)) for k in sig_a.keys() | sig_b.keys())
        total = sum(sig_a.values()) + sum(sig_b.values())
        similarity = (2 * common / total) if total > 0 else 1.0
        return similarity > 0.9, similarity

    def _signature(self, cfg: CFG) -> Dict[str, int]:
        sig: Dict[str, int] = {}
        for node in cfg.nodes.values():
            key = f"{node.kind}:in{len(node.edges_in)}:out{len(node.edges_out)}"
            sig[key] = sig.get(key, 0) + 1
        return sig


# ---------------------------------------------------------------------------
# Operator mapping C ↔ Rust
# ---------------------------------------------------------------------------

_OPERATOR_MAP_C_TO_RUST: Dict[str, str] = {
    "!": "!",
    "&&": "&&",
    "||": "||",
    "==": "==",
    "!=": "!=",
    "<": "<",
    ">": ">",
    "<=": "<=",
    ">=": ">=",
    "+": "+",
    "-": "-",
    "*": "*",
    "/": "/",
    "%": "%",
    "&": "&",
    "|": "|",
    "^": "^",
    "~": "!",  # bitwise NOT: C ~ vs Rust !
    "<<": "<<",
    ">>": ">>",
    "++": "+= 1",
    "--": "-= 1",
    "->": ".",  # member access through pointer
    "sizeof": "std::mem::size_of",
}


def _map_c_operator_to_rust(c_op: str) -> str:
    return _OPERATOR_MAP_C_TO_RUST.get(c_op, c_op)


# ---------------------------------------------------------------------------
# Main checker
# ---------------------------------------------------------------------------

class EquivalenceChecker:
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.strict_mode = self.config.get("strict", False)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        self.levenshtein_threshold = self.config.get("levenshtein_threshold", 3)

    # -- public API --

    @staticmethod
    def _type_val_to_str(t: Any) -> str:
        """Convert a type value (str, dict, or object) to a string."""
        if isinstance(t, str):
            return t
        if isinstance(t, dict):
            name = t.get("name", "")
            ptr = t.get("pointer_depth", 0)
            return name + "*" * ptr if ptr else name
        return str(t) if t is not None else ""

    @staticmethod
    def _ast_to_dict(ast_obj: Any) -> Dict[str, Any]:
        """Convert a CAST or RustAST object to the dict format expected internally."""
        result: Dict[str, Any] = {"functions": [], "types": {}, "globals": []}

        if isinstance(ast_obj, dict):
            # Normalize an asdict-produced or raw dict
            for f in ast_obj.get("functions", []):
                fd: Dict[str, Any] = {"name": f.get("name", "")}
                fd["return_type"] = EquivalenceChecker._type_val_to_str(
                    f.get("return_type", "void"))
                params = f.get("params", f.get("parameters", []))
                fd["params"] = []
                for p in (params or []):
                    pd: Dict[str, Any] = {"name": p.get("name", "")}
                    pt = p.get("type_spec", p.get("type", ""))
                    pd["type"] = EquivalenceChecker._type_val_to_str(pt)
                    fd["params"].append(pd)
                fd["body"] = f.get("body", "")
                result["functions"].append(fd)
            for key in ("types",):
                if key in ast_obj and isinstance(ast_obj[key], dict):
                    result[key] = ast_obj[key]
            for st in ast_obj.get("structs", []):
                name = st.get("name", st.get("tag", ""))
                if name:
                    result["types"][name] = {"name": name, "kind": "struct"}
            result["globals"] = ast_obj.get("globals", [])
            return result

        # Handle AST objects
        funcs = getattr(ast_obj, 'functions', [])
        for f in funcs:
            fd2: Dict[str, Any] = {"name": getattr(f, 'name', '')}
            rt = getattr(f, 'return_type', None)
            fd2["return_type"] = EquivalenceChecker._type_val_to_str(rt) if rt else "void"
            params = getattr(f, 'params', []) or getattr(f, 'parameters', []) or []
            fd2["params"] = []
            for p in params:
                pd2: Dict[str, Any] = {"name": getattr(p, 'name', '')}
                pt = getattr(p, 'type_spec', None) or getattr(p, 'type_annotation', None)
                pd2["type"] = EquivalenceChecker._type_val_to_str(pt) if pt else ""
                fd2["params"].append(pd2)
            body = getattr(f, 'body', None)
            fd2["body"] = str(body) if body else ""
            result["functions"].append(fd2)
        types_list = getattr(ast_obj, 'type_definitions', []) or getattr(ast_obj, 'structs', []) or []
        for t in types_list:
            tname = getattr(t, 'name', str(t))
            result["types"][tname] = {"name": tname, "kind": type(t).__name__}
        for g in getattr(ast_obj, 'global_vars', []) or getattr(ast_obj, 'consts', []) or []:
            result["globals"].append({"name": getattr(g, 'name', str(g))})
        return result

    def check(self, c_ast: Any,
              rust_ast: Any) -> EquivalenceResult:
        divergences: List[Divergence] = []
        func_mappings: List[FunctionMapping] = []
        type_maps: List[TypeMapping] = []

        c_dict = self._ast_to_dict(c_ast)
        rust_dict = self._ast_to_dict(rust_ast)
        c_funcs = c_dict.get("functions", [])
        rust_funcs = rust_dict.get("functions", [])
        func_mappings = self._match_functions(c_funcs, rust_funcs)
        unmatched_c, unmatched_rust = self._find_unmatched(
            c_funcs, rust_funcs, func_mappings
        )
        for name in unmatched_c:
            divergences.append(Divergence(
                kind="missing_function",
                location_c=name, location_rust=None,
                description=f"C function '{name}' has no Rust equivalent",
                severity="high",
                suggestion=f"Implement a Rust function matching '{name}'",
            ))
        for name in unmatched_rust:
            divergences.append(Divergence(
                kind="extra_function",
                location_c=None, location_rust=name,
                description=f"Rust function '{name}' has no C equivalent",
                severity="medium",
                suggestion="Verify this function is intentionally new",
            ))

        c_types = c_dict.get("types", {})
        rust_types = rust_dict.get("types", {})
        type_maps, type_divs = self._check_type_equivalences(c_types, rust_types)
        divergences.extend(type_divs)

        for fm in func_mappings:
            c_func = self._find_func(c_funcs, fm.c_name)
            r_func = self._find_func(rust_funcs, fm.rust_name)
            if c_func and r_func:
                divergences.extend(
                    self._compare_functions(c_func, r_func, fm)
                )

        c_globals = c_dict.get("globals", [])
        rust_globals = rust_dict.get("globals", [])
        divergences.extend(self._check_global_equivalence(c_globals, rust_globals))

        confidence = self._compute_confidence(divergences, func_mappings)
        equivalent = (
            len([d for d in divergences if d.severity in ("high", "critical")]) == 0
            and confidence >= self.confidence_threshold
        )

        return EquivalenceResult(
            equivalent=equivalent,
            divergences=divergences,
            confidence=confidence,
            function_mappings=func_mappings,
            type_mappings=type_maps,
        )

    # -- function matching --

    def _match_functions(self, c_funcs: List[Dict[str, Any]],
                         rust_funcs: List[Dict[str, Any]]) -> List[FunctionMapping]:
        mappings: List[FunctionMapping] = []
        used_rust: Set[str] = set()

        c_names = [f.get("name", "") for f in c_funcs]
        rust_names = [f.get("name", "") for f in rust_funcs]

        # pass 1: exact match
        for cn in c_names:
            if cn in rust_names and cn not in used_rust:
                mappings.append(FunctionMapping(cn, cn, 1.0, "exact"))
                used_rust.add(cn)

        # pass 2: snake_case conversion
        remaining_c = [n for n in c_names if not any(m.c_name == n for m in mappings)]
        for cn in remaining_c:
            snake = _camel_to_snake(cn)
            norm = normalize_identifier(cn)
            for rn in rust_names:
                if rn in used_rust:
                    continue
                if rn == snake or rn == norm:
                    mappings.append(FunctionMapping(cn, rn, 0.95, "snake_case"))
                    used_rust.add(rn)
                    break

        # pass 3: Levenshtein distance
        remaining_c = [n for n in c_names if not any(m.c_name == n for m in mappings)]
        for cn in remaining_c:
            best_rn = ""
            best_dist = self.levenshtein_threshold + 1
            cn_norm = normalize_identifier(cn)
            for rn in rust_names:
                if rn in used_rust:
                    continue
                dist = compute_levenshtein_distance(cn_norm, normalize_identifier(rn))
                if dist < best_dist:
                    best_dist = dist
                    best_rn = rn
            if best_rn and best_dist <= self.levenshtein_threshold:
                max_len = max(len(cn_norm), len(normalize_identifier(best_rn)), 1)
                score = 1.0 - best_dist / max_len
                mappings.append(FunctionMapping(cn, best_rn, score, "levenshtein"))
                used_rust.add(best_rn)

        # pass 4: signature similarity for still-unmatched functions
        remaining_c = [n for n in c_names if not any(m.c_name == n for m in mappings)]
        for cn in remaining_c:
            c_func = self._find_func_in_list(c_funcs, cn)
            if not c_func:
                continue
            best_rn = ""
            best_score = 0.0
            for rn in rust_names:
                if rn in used_rust:
                    continue
                r_func = self._find_func_in_list(rust_funcs, rn)
                if not r_func:
                    continue
                score = self._signature_similarity(c_func, r_func)
                if score > best_score and score >= 0.6:
                    best_score = score
                    best_rn = rn
            if best_rn:
                mappings.append(FunctionMapping(cn, best_rn, best_score, "signature"))
                used_rust.add(best_rn)

        return mappings

    def _signature_similarity(self, c_func: Dict[str, Any],
                              r_func: Dict[str, Any]) -> float:
        c_params = c_func.get("params", [])
        r_params = r_func.get("params", [])

        if len(c_params) == 0 and len(r_params) == 0:
            count_score = 1.0
        elif len(c_params) == len(r_params):
            count_score = 1.0
        else:
            diff = abs(len(c_params) - len(r_params))
            max_p = max(len(c_params), len(r_params))
            count_score = 1.0 - diff / max_p

        type_score = 0.0
        min_len = min(len(c_params), len(r_params))
        if min_len > 0:
            matches = 0
            for i in range(min_len):
                ct = c_params[i].get("type", "")
                rt = r_params[i].get("type", "")
                if self._types_compatible(ct, rt):
                    matches += 1
            type_score = matches / min_len
        else:
            type_score = 1.0 if len(c_params) == len(r_params) else 0.5

        c_ret = c_func.get("return_type", "void")
        r_ret = r_func.get("return_type", "()")
        ret_score = 1.0 if self._types_compatible(c_ret, r_ret) else 0.3

        return 0.3 * count_score + 0.4 * type_score + 0.3 * ret_score

    def _find_func(self, funcs: List[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
        return self._find_func_in_list(funcs, name)

    def _find_func_in_list(self, funcs: List[Dict[str, Any]],
                           name: str) -> Optional[Dict[str, Any]]:
        for f in funcs:
            if f.get("name") == name:
                return f
        return None

    def _find_unmatched(self, c_funcs: List[Dict[str, Any]],
                        rust_funcs: List[Dict[str, Any]],
                        mappings: List[FunctionMapping]) -> Tuple[List[str], List[str]]:
        matched_c = {m.c_name for m in mappings}
        matched_r = {m.rust_name for m in mappings}
        uc = [f.get("name", "") for f in c_funcs if f.get("name", "") not in matched_c]
        ur = [f.get("name", "") for f in rust_funcs if f.get("name", "") not in matched_r]
        return uc, ur

    # -- type equivalence --

    def _types_compatible(self, c_type, rust_type) -> bool:
        # Coerce to string if dict/object passed
        if isinstance(c_type, dict):
            c_type = c_type.get("name", c_type.get("type", str(c_type)))
        if isinstance(rust_type, dict):
            rust_type = rust_type.get("name", rust_type.get("type", str(rust_type)))
        c_type = str(c_type).strip()
        rust_type = str(rust_type).strip()

        if c_type == rust_type:
            return True

        # check primitive table
        rust_options = _C_TO_RUST_PRIMITIVE.get(c_type, [])
        if rust_type in rust_options:
            return True

        # check pointer table
        c_ptr = c_type.replace(" ", "").rstrip("*") + "*" if c_type.endswith("*") else ""
        if not c_ptr and "*" in c_type:
            c_ptr = c_type.replace(" ", "")
        for key, vals in _C_POINTER_TO_RUST.items():
            if c_ptr == key.replace(" ", "") or c_type.replace(" ", "") == key.replace(" ", ""):
                if rust_type in vals:
                    return True

        # const pointer
        if c_type.startswith("const "):
            base = c_type[6:].strip()
            return self._types_compatible(base, rust_type)

        # pointer to known type: e.g. int* -> *mut i32
        if c_type.endswith("*"):
            base_c = c_type[:-1].strip()
            if rust_type.startswith("*mut ") or rust_type.startswith("*const "):
                base_r = rust_type.split(" ", 1)[1] if " " in rust_type else ""
                return self._types_compatible(base_c, base_r)
            if rust_type.startswith("&mut ") or rust_type.startswith("&"):
                base_r = rust_type.lstrip("&").lstrip("mut ").strip()
                return self._types_compatible(base_c, base_r)

        # array: int[N] -> [i32; N]
        arr_match = re.match(r"(.+)\[(\d+)\]", c_type)
        if arr_match:
            base_c = arr_match.group(1).strip()
            size = arr_match.group(2)
            rust_arr = re.match(r"\[(.+);\s*(\d+)\]", rust_type)
            if rust_arr:
                return (self._types_compatible(base_c, rust_arr.group(1).strip())
                        and size == rust_arr.group(2))

        # function pointer: int (*)(int, int) -> fn(i32, i32) -> i32
        if "(*)" in c_type or c_type.startswith("fn("):
            return True  # approximate match for fn pointers

        return False

    def _check_type_equivalences(
        self, c_types: Dict[str, Any], rust_types: Dict[str, Any]
    ) -> Tuple[List[TypeMapping], List[Divergence]]:
        mappings: List[TypeMapping] = []
        divs: List[Divergence] = []

        for c_name, c_def in c_types.items():
            r_name = c_name
            snake_name = _camel_to_snake(c_name)
            norm_name = normalize_identifier(c_name)

            r_def = (rust_types.get(r_name)
                     or rust_types.get(snake_name)
                     or rust_types.get(norm_name))

            if r_def is None:
                mappings.append(TypeMapping(c_name, "", False, "No Rust equivalent found"))
                divs.append(Divergence(
                    kind="missing_type",
                    location_c=c_name, location_rust=None,
                    description=f"C type '{c_name}' has no Rust equivalent",
                    severity="high",
                    suggestion=f"Define a Rust type corresponding to '{c_name}'",
                ))
                continue

            actual_r_name = r_name if r_name in rust_types else (
                snake_name if snake_name in rust_types else norm_name
            )
            c_kind = c_def.get("kind", "")
            r_kind = r_def.get("kind", "")

            if c_kind == "struct" and r_kind == "struct":
                compat, notes, field_divs = self._compare_structs(
                    c_name, c_def, actual_r_name, r_def
                )
                mappings.append(TypeMapping(c_name, actual_r_name, compat, notes))
                divs.extend(field_divs)
            elif c_kind == "enum" and r_kind == "enum":
                compat, notes, enum_divs = self._compare_enums(
                    c_name, c_def, actual_r_name, r_def
                )
                mappings.append(TypeMapping(c_name, actual_r_name, compat, notes))
                divs.extend(enum_divs)
            elif c_kind == "typedef":
                base_c = c_def.get("base_type", "")
                base_r = r_def.get("base_type", r_def.get("type", ""))
                compat = self._types_compatible(base_c, base_r)
                mappings.append(TypeMapping(c_name, actual_r_name, compat,
                                           f"typedef {base_c} -> {base_r}"))
            else:
                mappings.append(TypeMapping(c_name, actual_r_name, c_kind == r_kind,
                                           f"kind mismatch: {c_kind} vs {r_kind}"))

        return mappings, divs

    def _compare_structs(
        self, c_name: str, c_def: Dict[str, Any],
        r_name: str, r_def: Dict[str, Any]
    ) -> Tuple[bool, str, List[Divergence]]:
        divs: List[Divergence] = []
        c_fields = c_def.get("fields", [])
        r_fields = r_def.get("fields", [])
        compatible = True
        notes_parts: List[str] = []

        if len(c_fields) != len(r_fields):
            compatible = False
            notes_parts.append(
                f"field count: C={len(c_fields)} vs Rust={len(r_fields)}"
            )
            divs.append(Divergence(
                kind="struct_field_count",
                location_c=c_name, location_rust=r_name,
                description=f"Struct field count differs: {len(c_fields)} vs {len(r_fields)}",
                severity="high",
                suggestion="Ensure all fields are translated",
            ))

        c_field_map = {f.get("name", ""): f for f in c_fields}
        r_field_map = {f.get("name", ""): f for f in r_fields}

        for fname, fdef in c_field_map.items():
            rfield = r_field_map.get(fname) or r_field_map.get(_camel_to_snake(fname))
            if rfield is None:
                compatible = False
                divs.append(Divergence(
                    kind="struct_missing_field",
                    location_c=f"{c_name}.{fname}",
                    location_rust=r_name,
                    description=f"C struct field '{fname}' not found in Rust struct",
                    severity="high",
                    suggestion=f"Add field '{_camel_to_snake(fname)}' to Rust struct",
                ))
                continue
            ct = fdef.get("type", "")
            rt = rfield.get("type", "")
            if not self._types_compatible(ct, rt):
                compatible = False
                divs.append(Divergence(
                    kind="struct_field_type",
                    location_c=f"{c_name}.{fname}",
                    location_rust=f"{r_name}.{rfield.get('name', fname)}",
                    description=f"Field type mismatch: C '{ct}' vs Rust '{rt}'",
                    severity="medium",
                    suggestion=f"Change Rust field type to match C type '{ct}'",
                ))

        return compatible, "; ".join(notes_parts) if notes_parts else "OK", divs

    def _compare_enums(
        self, c_name: str, c_def: Dict[str, Any],
        r_name: str, r_def: Dict[str, Any]
    ) -> Tuple[bool, str, List[Divergence]]:
        divs: List[Divergence] = []
        c_variants = c_def.get("variants", c_def.get("values", []))
        r_variants = r_def.get("variants", r_def.get("values", []))
        compatible = True
        notes_parts: List[str] = []

        c_names_set = set()
        r_names_set = set()
        for v in c_variants:
            name = v.get("name", v) if isinstance(v, dict) else str(v)
            c_names_set.add(normalize_identifier(name))
        for v in r_variants:
            name = v.get("name", v) if isinstance(v, dict) else str(v)
            r_names_set.add(normalize_identifier(name))

        missing_in_rust = c_names_set - r_names_set
        extra_in_rust = r_names_set - c_names_set

        if missing_in_rust:
            compatible = False
            for m in missing_in_rust:
                divs.append(Divergence(
                    kind="enum_missing_variant",
                    location_c=f"{c_name}::{m}",
                    location_rust=r_name,
                    description=f"C enum variant '{m}' missing in Rust",
                    severity="high",
                    suggestion=f"Add variant '{m}' to Rust enum",
                ))
        if extra_in_rust:
            notes_parts.append(f"extra Rust variants: {extra_in_rust}")

        if len(c_variants) != len(r_variants):
            notes_parts.append(
                f"variant count: C={len(c_variants)} vs Rust={len(r_variants)}"
            )

        return compatible, "; ".join(notes_parts) if notes_parts else "OK", divs

    # -- function body comparison --

    def _compare_functions(self, c_func: Dict[str, Any],
                           r_func: Dict[str, Any],
                           mapping: FunctionMapping) -> List[Divergence]:
        divs: List[Divergence] = []

        # return type
        c_ret = c_func.get("return_type", "void")
        r_ret = r_func.get("return_type", "()")
        if not self._types_compatible(c_ret, r_ret):
            if not self._is_error_return_pattern(c_ret, r_ret):
                divs.append(Divergence(
                    kind="return_type_mismatch",
                    location_c=mapping.c_name,
                    location_rust=mapping.rust_name,
                    description=f"Return type mismatch: C '{c_ret}' vs Rust '{r_ret}'",
                    severity="medium",
                    suggestion="Verify return type translation is intentional",
                ))

        # parameter types
        c_params = c_func.get("params", [])
        r_params = r_func.get("params", [])
        divs.extend(self._compare_params(c_params, r_params, mapping))

        # control flow
        c_body = c_func.get("body", [])
        r_body = r_func.get("body", [])
        divs.extend(self._compare_control_flow(c_body, r_body, mapping))

        # memory management
        divs.extend(self._check_memory_equivalence(c_body, r_body, mapping))

        # error handling
        divs.extend(self._check_error_handling(c_func, r_func, mapping))

        # overflow behavior
        divs.extend(self._check_overflow_behavior(c_func, r_func, mapping))

        # pointer/reference patterns
        divs.extend(self._check_pointer_reference(c_func, r_func, mapping))

        return divs

    def _compare_params(self, c_params: List[Dict[str, Any]],
                        r_params: List[Dict[str, Any]],
                        mapping: FunctionMapping) -> List[Divergence]:
        divs: List[Divergence] = []
        if len(c_params) != len(r_params):
            divs.append(Divergence(
                kind="param_count_mismatch",
                location_c=mapping.c_name,
                location_rust=mapping.rust_name,
                description=f"Parameter count: C has {len(c_params)}, Rust has {len(r_params)}",
                severity="medium",
                suggestion="Verify parameter list translation",
            ))
        min_len = min(len(c_params), len(r_params))
        for i in range(min_len):
            ct = c_params[i].get("type", "")
            rt = r_params[i].get("type", "")
            if not self._types_compatible(ct, rt):
                divs.append(Divergence(
                    kind="param_type_mismatch",
                    location_c=f"{mapping.c_name}:param[{i}]",
                    location_rust=f"{mapping.rust_name}:param[{i}]",
                    description=f"Param {i} type mismatch: C '{ct}' vs Rust '{rt}'",
                    severity="medium",
                    suggestion=f"Change Rust param type to match C '{ct}'",
                ))
        return divs

    def _compare_control_flow(self, c_body: List[Any],
                              r_body: List[Any],
                              mapping: FunctionMapping) -> List[Divergence]:
        divs: List[Divergence] = []
        if not c_body and not r_body:
            return divs

        c_cfg = build_cfg(c_body if isinstance(c_body, list) else [c_body], "c")
        r_cfg = build_cfg(r_body if isinstance(r_body, list) else [r_body], "rust")

        c_cfg = _normalize_cfg(c_cfg)
        r_cfg = _normalize_cfg(r_cfg)

        checker = _CFGIsomorphismChecker(c_cfg, r_cfg)
        iso, similarity = checker.check()

        if not iso:
            sev = "low" if similarity > 0.8 else ("medium" if similarity > 0.5 else "high")
            divs.append(Divergence(
                kind="control_flow_divergence",
                location_c=mapping.c_name,
                location_rust=mapping.rust_name,
                description=(
                    f"Control flow graphs differ (similarity={similarity:.2f}). "
                    f"C has {c_cfg.node_count()} nodes/{c_cfg.edge_count()} edges, "
                    f"Rust has {r_cfg.node_count()} nodes/{r_cfg.edge_count()} edges."
                ),
                severity=sev,
                suggestion="Review control flow translation for correctness",
            ))

        # check switch vs match
        c_switches = [n for n in c_cfg.nodes.values() if n.kind == "switch"]
        r_matches = [n for n in r_cfg.nodes.values() if n.kind == "match"]
        if len(c_switches) != len(r_matches):
            divs.append(Divergence(
                kind="switch_match_count",
                location_c=mapping.c_name,
                location_rust=mapping.rust_name,
                description=(
                    f"C has {len(c_switches)} switch statements, "
                    f"Rust has {len(r_matches)} match expressions"
                ),
                severity="medium",
                suggestion="Ensure all switch statements are translated to match",
            ))

        # check goto usage in C
        c_gotos = self._count_stmt_kind(c_body, "goto")
        if c_gotos > 0:
            r_labels = self._count_stmt_kind(r_body, "label") + self._count_stmt_kind(r_body, "loop")
            if r_labels == 0:
                divs.append(Divergence(
                    kind="goto_translation",
                    location_c=mapping.c_name,
                    location_rust=mapping.rust_name,
                    description=f"C uses {c_gotos} goto(s) but Rust has no labeled blocks/loops",
                    severity="medium",
                    suggestion="Translate goto patterns to labeled blocks or loops in Rust",
                ))

        return divs

    def _count_stmt_kind(self, body: Any, kind: str) -> int:
        count = 0
        if isinstance(body, dict):
            if body.get("kind") == kind:
                count += 1
            for val in body.values():
                count += self._count_stmt_kind(val, kind)
        elif isinstance(body, list):
            for item in body:
                count += self._count_stmt_kind(item, kind)
        return count

    # -- memory model --

    def _check_memory_equivalence(self, c_body: Any, r_body: Any,
                                  mapping: FunctionMapping) -> List[Divergence]:
        divs: List[Divergence] = []

        c_allocs = self._find_calls(c_body, {"malloc", "calloc", "realloc"})
        c_frees = self._find_calls(c_body, {"free"})
        r_box_new = self._find_calls(r_body, {"Box::new"})
        r_vec_calls = self._find_calls(r_body, {"Vec::new", "Vec::with_capacity", "vec!"})
        r_drops = self._find_calls(r_body, {"drop", "std::mem::drop"})

        total_c_alloc = len(c_allocs)
        total_r_alloc = len(r_box_new) + len(r_vec_calls)

        if total_c_alloc > 0 and total_r_alloc == 0:
            divs.append(Divergence(
                kind="memory_alloc_missing",
                location_c=mapping.c_name,
                location_rust=mapping.rust_name,
                description=(
                    f"C has {total_c_alloc} heap allocation(s) "
                    "but Rust has no Box/Vec allocations"
                ),
                severity="medium",
                suggestion="Verify Rust uses stack allocation or different allocation strategy",
            ))

        if len(c_frees) > 0 and len(r_drops) == 0:
            # Rust may rely on implicit drop, which is fine
            divs.append(Divergence(
                kind="memory_free_implicit",
                location_c=mapping.c_name,
                location_rust=mapping.rust_name,
                description=(
                    f"C has {len(c_frees)} free() call(s); "
                    "Rust relies on implicit drop"
                ),
                severity="low",
                suggestion="Implicit drop is idiomatic Rust; verify ownership is correct",
            ))

        # unbalanced alloc/free in C
        if total_c_alloc > 0 and len(c_frees) != total_c_alloc:
            diff = total_c_alloc - len(c_frees)
            if diff > 0:
                divs.append(Divergence(
                    kind="memory_leak_c",
                    location_c=mapping.c_name,
                    location_rust=mapping.rust_name,
                    description=f"Potential C memory leak: {total_c_alloc} allocs, {len(c_frees)} frees",
                    severity="high",
                    suggestion="Check for missing free() calls in C code",
                ))

        return divs

    def _find_calls(self, body: Any, names: Set[str]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        if isinstance(body, dict):
            if body.get("kind") == "call" and body.get("name", "") in names:
                results.append(body)
            callee = body.get("callee", "")
            if isinstance(callee, str) and callee in names:
                results.append(body)
            for val in body.values():
                results.extend(self._find_calls(val, names))
        elif isinstance(body, list):
            for item in body:
                results.extend(self._find_calls(item, names))
        return results

    # -- error handling --

    def _check_error_handling(self, c_func: Dict[str, Any],
                              r_func: Dict[str, Any],
                              mapping: FunctionMapping) -> List[Divergence]:
        divs: List[Divergence] = []
        c_body_str = str(c_func.get("body", ""))
        r_body_str = str(r_func.get("body", ""))

        c_patterns_found: List[str] = []
        for pattern, label in _C_ERROR_PATTERNS.items():
            if pattern in c_body_str:
                c_patterns_found.append(label)

        r_patterns_found: List[str] = []
        for pattern, label in _RUST_ERROR_PATTERNS.items():
            if pattern in r_body_str:
                r_patterns_found.append(label)

        c_has_error = len(c_patterns_found) > 0
        r_has_error = len(r_patterns_found) > 0

        if c_has_error and not r_has_error:
            divs.append(Divergence(
                kind="error_handling_missing",
                location_c=mapping.c_name,
                location_rust=mapping.rust_name,
                description=(
                    f"C uses error patterns ({', '.join(c_patterns_found)}) "
                    "but Rust has no error handling"
                ),
                severity="high",
                suggestion="Add Result/Option error handling in Rust",
            ))
        elif c_has_error and r_has_error:
            # Check goto cleanup vs ? operator
            if "goto_cleanup" in c_patterns_found or "goto_error" in c_patterns_found:
                if "question_mark" not in r_patterns_found:
                    divs.append(Divergence(
                        kind="error_handling_style",
                        location_c=mapping.c_name,
                        location_rust=mapping.rust_name,
                        description="C uses goto-based cleanup but Rust doesn't use ? operator",
                        severity="low",
                        suggestion="Consider using ? operator for idiomatic Rust error handling",
                    ))

        # C return -1/NULL → Rust Result::Err
        c_ret = c_func.get("return_type", "void")
        r_ret = r_func.get("return_type", "()")
        if self._is_error_return_pattern(c_ret, r_ret):
            if "error_code_minus1" in c_patterns_found or "error_null" in c_patterns_found:
                if "result_err" not in r_patterns_found and "option_none" not in r_patterns_found:
                    divs.append(Divergence(
                        kind="error_return_translation",
                        location_c=mapping.c_name,
                        location_rust=mapping.rust_name,
                        description=(
                            f"C returns error codes ({c_ret}) but Rust return type "
                            f"'{r_ret}' may not handle errors"
                        ),
                        severity="medium",
                        suggestion="Use Result<T, E> or Option<T> as Rust return type",
                    ))

        return divs

    def _is_error_return_pattern(self, c_ret: str, r_ret: str) -> bool:
        if "Result" in r_ret or "Option" in r_ret:
            return True
        if c_ret in ("int", "long", "ssize_t") and "Result" in r_ret:
            return True
        if c_ret.endswith("*") and "Option" in r_ret:
            return True
        return False

    # -- overflow behavior --

    def _check_overflow_behavior(self, c_func: Dict[str, Any],
                                 r_func: Dict[str, Any],
                                 mapping: FunctionMapping) -> List[Divergence]:
        divs: List[Divergence] = []
        c_body = c_func.get("body", [])
        r_body = r_func.get("body", [])

        c_arith = self._find_arithmetic_ops(c_body)
        r_arith = self._find_arithmetic_ops(r_body)

        if not c_arith and not r_arith:
            return divs

        # look for wrapping/checked/saturating in Rust
        r_body_str = str(r_body)
        has_wrapping = "wrapping_" in r_body_str
        has_checked = "checked_" in r_body_str
        has_saturating = "saturating_" in r_body_str
        has_overflowing = "overflowing_" in r_body_str

        if c_arith and not (has_wrapping or has_checked or has_saturating or has_overflowing):
            c_signed_ops = [op for op in c_arith if op.get("signed", True)]
            if c_signed_ops:
                divs.append(Divergence(
                    kind="overflow_semantics",
                    location_c=mapping.c_name,
                    location_rust=mapping.rust_name,
                    description=(
                        f"C has {len(c_signed_ops)} signed arithmetic op(s) with "
                        "undefined overflow behavior. Rust will panic in debug "
                        "mode or wrap in release mode."
                    ),
                    severity="medium",
                    suggestion=(
                        "Use wrapping_add/checked_add/saturating_add in Rust "
                        "to match intended C overflow behavior"
                    ),
                ))

        c_unsigned_ops = [op for op in c_arith if not op.get("signed", True)]
        r_unsigned = [op for op in r_arith if not op.get("signed", True)]
        if c_unsigned_ops and not r_unsigned and not has_wrapping:
            divs.append(Divergence(
                kind="unsigned_overflow",
                location_c=mapping.c_name,
                location_rust=mapping.rust_name,
                description=(
                    "C unsigned arithmetic wraps on overflow; "
                    "ensure Rust uses appropriate unsigned types or wrapping ops"
                ),
                severity="low",
                suggestion="Use Wrapping<u32> or .wrapping_add() for C-compatible unsigned math",
            ))

        return divs

    def _find_arithmetic_ops(self, body: Any) -> List[Dict[str, Any]]:
        ops: List[Dict[str, Any]] = []
        if isinstance(body, dict):
            if body.get("kind") == "binop" and body.get("op") in ("+", "-", "*", "<<"):
                t = body.get("type", "int")
                signed = not t.startswith("u") and "unsigned" not in t
                ops.append({"op": body["op"], "type": t, "signed": signed})
            for val in body.values():
                ops.extend(self._find_arithmetic_ops(val))
        elif isinstance(body, list):
            for item in body:
                ops.extend(self._find_arithmetic_ops(item))
        return ops

    # -- pointer/reference --

    def _check_pointer_reference(self, c_func: Dict[str, Any],
                                 r_func: Dict[str, Any],
                                 mapping: FunctionMapping) -> List[Divergence]:
        divs: List[Divergence] = []
        c_body = c_func.get("body", [])
        r_body = r_func.get("body", [])

        c_ptr_arith = self._find_pointer_arithmetic(c_body)
        r_index_ops = self._find_index_operations(r_body)

        if c_ptr_arith and not r_index_ops:
            divs.append(Divergence(
                kind="pointer_arithmetic",
                location_c=mapping.c_name,
                location_rust=mapping.rust_name,
                description=(
                    f"C uses {len(c_ptr_arith)} pointer arithmetic operation(s) "
                    "but Rust has no corresponding index operations"
                ),
                severity="medium",
                suggestion="Translate pointer arithmetic to slice/array indexing in Rust",
            ))

        c_arrow = self._count_operator(c_body, "->")
        r_dot_ref = self._count_operator(r_body, ".")
        if c_arrow > 0 and r_dot_ref == 0:
            divs.append(Divergence(
                kind="member_access_translation",
                location_c=mapping.c_name,
                location_rust=mapping.rust_name,
                description=f"C uses {c_arrow} '->' member access(es) with no '.' in Rust",
                severity="low",
                suggestion="Rust auto-dereferences; '.' should replace '->'",
            ))

        # array access
        c_array = self._find_array_access(c_body)
        r_array = self._find_array_access(r_body)
        if len(c_array) > 0 and len(r_array) == 0 and not r_index_ops:
            divs.append(Divergence(
                kind="array_access_missing",
                location_c=mapping.c_name,
                location_rust=mapping.rust_name,
                description=f"C has {len(c_array)} array access(es) not reflected in Rust",
                severity="medium",
                suggestion="Translate C array accesses to Rust indexing or iterators",
            ))

        return divs

    def _find_pointer_arithmetic(self, body: Any) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        if isinstance(body, dict):
            if body.get("kind") == "binop" and body.get("op") in ("+", "-"):
                if body.get("type", "").endswith("*") or body.get("is_pointer"):
                    results.append(body)
            for val in body.values():
                results.extend(self._find_pointer_arithmetic(val))
        elif isinstance(body, list):
            for item in body:
                results.extend(self._find_pointer_arithmetic(item))
        return results

    def _find_index_operations(self, body: Any) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        if isinstance(body, dict):
            if body.get("kind") == "index":
                results.append(body)
            for val in body.values():
                results.extend(self._find_index_operations(val))
        elif isinstance(body, list):
            for item in body:
                results.extend(self._find_index_operations(item))
        return results

    def _find_array_access(self, body: Any) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        if isinstance(body, dict):
            if body.get("kind") in ("index", "subscript", "array_access"):
                results.append(body)
            for val in body.values():
                results.extend(self._find_array_access(val))
        elif isinstance(body, list):
            for item in body:
                results.extend(self._find_array_access(item))
        return results

    def _count_operator(self, body: Any, op: str) -> int:
        count = 0
        if isinstance(body, dict):
            if body.get("op") == op:
                count += 1
            for val in body.values():
                count += self._count_operator(val, op)
        elif isinstance(body, list):
            for item in body:
                count += self._count_operator(item, op)
        return count

    # -- global equivalence --

    def _check_global_equivalence(self, c_globals: List[Dict[str, Any]],
                                  rust_globals: List[Dict[str, Any]]) -> List[Divergence]:
        divs: List[Divergence] = []
        c_names = {g.get("name", ""): g for g in c_globals}
        r_names = {g.get("name", ""): g for g in rust_globals}
        r_snake = {_camel_to_snake(n): n for n in r_names}

        for cn, cdef in c_names.items():
            rn = cn if cn in r_names else r_snake.get(_camel_to_snake(cn))
            if rn is None:
                divs.append(Divergence(
                    kind="missing_global",
                    location_c=cn, location_rust=None,
                    description=f"C global '{cn}' has no Rust equivalent",
                    severity="medium",
                    suggestion=f"Add a static/const in Rust for '{cn}'",
                ))
                continue
            rdef = r_names.get(rn, {})
            ct = cdef.get("type", "")
            rt = rdef.get("type", "")
            if ct and rt and not self._types_compatible(ct, rt):
                divs.append(Divergence(
                    kind="global_type_mismatch",
                    location_c=cn, location_rust=rn,
                    description=f"Global type mismatch: C '{ct}' vs Rust '{rt}'",
                    severity="medium",
                    suggestion="Align global variable types",
                ))

            c_mut = cdef.get("mutable", True)
            r_mut = rdef.get("mutable", False)
            if c_mut and not r_mut:
                divs.append(Divergence(
                    kind="global_mutability",
                    location_c=cn, location_rust=rn,
                    description=f"C global '{cn}' is mutable but Rust equivalent is const",
                    severity="low",
                    suggestion="Use static mut or Mutex/RwLock for mutable globals in Rust",
                ))

        return divs

    # -- confidence --

    def _compute_confidence(self, divergences: List[Divergence],
                            mappings: List[FunctionMapping]) -> float:
        if not mappings:
            return 0.0

        base = sum(m.score for m in mappings) / len(mappings)

        penalty = 0.0
        severity_weights = {
            "low": 0.02,
            "medium": 0.05,
            "high": 0.10,
            "critical": 0.25,
        }
        for d in divergences:
            penalty += severity_weights.get(d.severity, 0.05)

        confidence = max(0.0, min(1.0, base - penalty))
        return round(confidence, 4)

    # -- expression comparison --

    def compare_expressions(self, c_expr: Dict[str, Any],
                            r_expr: Dict[str, Any]) -> Tuple[bool, float, str]:
        ce = self._dict_to_expr(c_expr, "c")
        re_ = self._dict_to_expr(r_expr, "rust")

        ce = simplify_expression(ce)
        re_ = simplify_expression(re_)

        if _exprs_commutative_equal(ce, re_):
            return True, 1.0, "Expressions are equivalent"

        if _exprs_equal(ce, re_):
            return True, 1.0, "Expressions are structurally identical"

        similarity = self._expr_similarity(ce, re_)
        if similarity > 0.9:
            return True, similarity, "Expressions are nearly equivalent"

        return False, similarity, "Expressions differ"

    def _dict_to_expr(self, d: Dict[str, Any], lang: str) -> Expr:
        kind = d.get("kind", "ident")
        if kind == "literal":
            return Expr(kind="literal", value=d.get("value"))
        if kind in ("ident", "identifier", "name"):
            return Expr(kind="ident", value=d.get("name", d.get("value", "")))
        if kind == "binop":
            op = d.get("op", "+")
            if lang == "c":
                op = _map_c_operator_to_rust(op)
            children = []
            if "left" in d:
                children.append(self._dict_to_expr(d["left"], lang))
            if "right" in d:
                children.append(self._dict_to_expr(d["right"], lang))
            return Expr(kind="binop", op=op, children=children)
        if kind == "unop":
            op = d.get("op", "!")
            if lang == "c":
                op = _map_c_operator_to_rust(op)
            children = []
            if "operand" in d:
                children.append(self._dict_to_expr(d["operand"], lang))
            return Expr(kind="unop", op=op, children=children)
        if kind == "call":
            children = [
                self._dict_to_expr(a, lang) for a in d.get("args", [])
            ]
            return Expr(kind="call", value=d.get("name", ""), children=children)
        if kind == "index":
            children = []
            if "base" in d:
                children.append(self._dict_to_expr(d["base"], lang))
            if "index" in d:
                children.append(self._dict_to_expr(d["index"], lang))
            return Expr(kind="index", children=children)
        if kind == "member":
            children = []
            if "base" in d:
                children.append(self._dict_to_expr(d["base"], lang))
            op = d.get("op", ".")
            return Expr(kind="member", value=d.get("field", ""), op=op, children=children)
        if kind == "cast":
            children = []
            if "expr" in d:
                children.append(self._dict_to_expr(d["expr"], lang))
            return Expr(kind="cast", type_info=d.get("type", ""), children=children)
        return Expr(kind="ident", value=str(d))

    def _expr_similarity(self, a: Expr, b: Expr) -> float:
        if a.kind != b.kind:
            return 0.1
        if a.kind == "literal":
            return 1.0 if a.value == b.value else 0.0
        if a.kind == "ident":
            if a.value == b.value:
                return 1.0
            na = normalize_identifier(str(a.value))
            nb = normalize_identifier(str(b.value))
            if na == nb:
                return 0.95
            dist = compute_levenshtein_distance(na, nb)
            max_len = max(len(na), len(nb), 1)
            return max(0.0, 1.0 - dist / max_len) * 0.8

        if len(a.children) == 0 and len(b.children) == 0:
            return 1.0 if a.op == b.op else 0.3

        if len(a.children) != len(b.children):
            max_ch = max(len(a.children), len(b.children))
            min_ch = min(len(a.children), len(b.children))
            ratio = min_ch / max_ch if max_ch > 0 else 1.0
            child_sim = 0.0
            for i in range(min_ch):
                child_sim += self._expr_similarity(a.children[i], b.children[i])
            child_sim = child_sim / max_ch if max_ch > 0 else 1.0
            op_sim = 1.0 if a.op == b.op else 0.5
            return 0.5 * child_sim + 0.3 * op_sim + 0.2 * ratio

        child_sim = sum(
            self._expr_similarity(ac, bc)
            for ac, bc in zip(a.children, b.children)
        ) / len(a.children) if a.children else 1.0
        op_sim = 1.0 if a.op == b.op else 0.5
        return 0.7 * child_sim + 0.3 * op_sim

    # -- batch / convenience API --

    def check_files(self, c_ast: Dict[str, Any],
                    rust_ast: Dict[str, Any]) -> EquivalenceResult:
        """Convenience wrapper identical to check(); accepts file-level ASTs."""
        return self.check(c_ast, rust_ast)

    def check_function_pair(self, c_func: Dict[str, Any],
                            r_func: Dict[str, Any]) -> List[Divergence]:
        """Compare a single C/Rust function pair directly."""
        mapping = FunctionMapping(
            c_name=c_func.get("name", "<anon_c>"),
            rust_name=r_func.get("name", "<anon_rust>"),
            score=1.0,
            method="manual",
        )
        return self._compare_functions(c_func, r_func, mapping)

    def check_type_pair(self, c_type: str, rust_type: str) -> TypeMapping:
        compat = self._types_compatible(c_type, rust_type)
        rust_options = _C_TO_RUST_PRIMITIVE.get(c_type, [])
        note = ""
        if compat:
            note = "Compatible"
        elif rust_options:
            note = f"Expected one of {rust_options}, got '{rust_type}'"
        else:
            note = "No known mapping"
        return TypeMapping(c_type, rust_type, compat, note)
