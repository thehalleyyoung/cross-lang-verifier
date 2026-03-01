"""AST-level semantic diff tool for C-to-Rust migration.

Provides structural comparison of C and Rust source code at the AST,
control-flow graph, data-flow, call-graph, and type-hierarchy levels.
Produces machine-readable diffs and human-readable visualisations.
"""

from __future__ import annotations

import re
import html
import math
import itertools
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)


# ─── Enums ──────────────────────────────────────────────────────────────────

class DiffCategory(Enum):
    STRUCTURAL = auto()
    SEMANTIC = auto()
    SYNTACTIC = auto()
    TYPE_MISMATCH = auto()
    CONTROL_FLOW = auto()
    DATA_FLOW = auto()
    CALL_GRAPH = auto()
    NAMING = auto()
    MISSING = auto()
    EXTRA = auto()


class NodeKind(Enum):
    FUNCTION = "function"
    VARIABLE = "variable"
    IF = "if"
    ELSE = "else"
    WHILE = "while"
    FOR = "for"
    SWITCH = "switch"
    MATCH = "match"
    RETURN = "return"
    CALL = "call"
    ASSIGNMENT = "assignment"
    BINARY_OP = "binary_op"
    UNARY_OP = "unary_op"
    STRUCT = "struct"
    ENUM = "enum"
    TYPEDEF = "typedef"
    IMPL = "impl"
    TRAIT = "trait"
    BLOCK = "block"
    LITERAL = "literal"
    PARAMETER = "parameter"
    FIELD = "field"
    UNKNOWN = "unknown"


# ─── Data classes ───────────────────────────────────────────────────────────

@dataclass
class ASTNode:
    """A single node in a parsed abstract syntax tree."""
    kind: NodeKind
    name: str = ""
    value: str = ""
    type_annotation: str = ""
    children: List[ASTNode] = field(default_factory=list)
    line: int = 0
    col: int = 0
    source_span: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- helpers ---------------------------------------------------------------

    @property
    def depth(self) -> int:
        if not self.children:
            return 0
        return 1 + max(c.depth for c in self.children)

    @property
    def size(self) -> int:
        return 1 + sum(c.size for c in self.children)

    def flatten(self) -> List[ASTNode]:
        """Pre-order traversal returning all nodes."""
        result: List[ASTNode] = [self]
        for child in self.children:
            result.extend(child.flatten())
        return result

    def find(self, kind: NodeKind) -> List[ASTNode]:
        return [n for n in self.flatten() if n.kind == kind]

    def signature(self) -> str:
        parts = [self.kind.value]
        if self.name:
            parts.append(self.name)
        if self.type_annotation:
            parts.append(self.type_annotation)
        return ":".join(parts)


@dataclass
class DiffEntry:
    """A single difference between two ASTs."""
    category: DiffCategory
    message: str
    c_node: Optional[ASTNode] = None
    rust_node: Optional[ASTNode] = None
    severity: float = 0.5          # 0 = informational, 1 = critical
    suggestion: str = ""

    @property
    def label(self) -> str:
        return f"[{self.category.name}] {self.message}"


@dataclass
class ASTDiff:
    """Result of comparing two ASTs."""
    entries: List[DiffEntry] = field(default_factory=list)
    c_root: Optional[ASTNode] = None
    rust_root: Optional[ASTNode] = None

    @property
    def total(self) -> int:
        return len(self.entries)

    def by_category(self, cat: DiffCategory) -> List[DiffEntry]:
        return [e for e in self.entries if e.category == cat]

    def critical(self) -> List[DiffEntry]:
        return [e for e in self.entries if e.severity >= 0.8]


@dataclass
class CFGNode:
    """A node in a control-flow graph."""
    node_id: int
    label: str
    kind: str = "basic"
    successors: List[int] = field(default_factory=list)
    predecessors: List[int] = field(default_factory=list)
    source_line: int = 0


@dataclass
class CFGDiff:
    """Diff between two control-flow graphs."""
    c_nodes: List[CFGNode] = field(default_factory=list)
    rust_nodes: List[CFGNode] = field(default_factory=list)
    added_edges: List[Tuple[int, int]] = field(default_factory=list)
    removed_edges: List[Tuple[int, int]] = field(default_factory=list)
    structural_changes: List[str] = field(default_factory=list)

    @property
    def is_isomorphic(self) -> bool:
        return not self.added_edges and not self.removed_edges and not self.structural_changes


@dataclass
class DataFlowDiff:
    """Diff of data-flow (def-use) information."""
    c_chains: Dict[str, List[Tuple[int, str]]] = field(default_factory=dict)
    rust_chains: Dict[str, List[Tuple[int, str]]] = field(default_factory=dict)
    missing_defs: List[str] = field(default_factory=list)
    extra_defs: List[str] = field(default_factory=list)
    changed_uses: List[Tuple[str, str]] = field(default_factory=list)


@dataclass
class CallGraphDiff:
    """Diff between two call graphs."""
    c_edges: Set[Tuple[str, str]] = field(default_factory=set)
    rust_edges: Set[Tuple[str, str]] = field(default_factory=set)
    added_calls: Set[Tuple[str, str]] = field(default_factory=set)
    removed_calls: Set[Tuple[str, str]] = field(default_factory=set)
    renamed_functions: Dict[str, str] = field(default_factory=dict)


@dataclass
class TypeDiff:
    """Diff between type hierarchies."""
    c_types: List[str] = field(default_factory=list)
    rust_types: List[str] = field(default_factory=list)
    matched: List[Tuple[str, str, float]] = field(default_factory=list)
    unmatched_c: List[str] = field(default_factory=list)
    unmatched_rust: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ─── C type → Rust type canonical mapping ───────────────────────────────────

_C_TO_RUST_TYPE: Dict[str, str] = {
    "int": "i32",
    "unsigned int": "u32",
    "unsigned": "u32",
    "long": "i64",
    "unsigned long": "u64",
    "long long": "i64",
    "unsigned long long": "u64",
    "short": "i16",
    "unsigned short": "u16",
    "char": "i8",
    "unsigned char": "u8",
    "signed char": "i8",
    "float": "f32",
    "double": "f64",
    "long double": "f64",
    "void": "()",
    "bool": "bool",
    "_Bool": "bool",
    "size_t": "usize",
    "ssize_t": "isize",
    "int8_t": "i8",
    "int16_t": "i16",
    "int32_t": "i32",
    "int64_t": "i64",
    "uint8_t": "u8",
    "uint16_t": "u16",
    "uint32_t": "u32",
    "uint64_t": "u64",
    "intptr_t": "isize",
    "uintptr_t": "usize",
    "ptrdiff_t": "isize",
}


def _normalise_c_type(raw: str) -> str:
    """Strip qualifiers, collapse whitespace, remove pointer stars."""
    cleaned = re.sub(r"\b(const|volatile|restrict|static|extern|inline)\b", "", raw)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _map_c_type_to_rust(c_type: str) -> str:
    """Best-effort mapping of a C type string to its Rust equivalent."""
    norm = _normalise_c_type(c_type)
    ptr_depth = norm.count("*")
    base = norm.replace("*", "").strip()
    rust_base = _C_TO_RUST_TYPE.get(base, base)
    for _ in range(ptr_depth):
        rust_base = f"*mut {rust_base}"
    return rust_base


# ─── Regex-based C parser ───────────────────────────────────────────────────

_C_FUNC_RE = re.compile(
    r"(?P<ret>[\w\s\*]+?)\s+(?P<name>\w+)\s*\((?P<params>[^)]*)\)\s*\{",
    re.MULTILINE,
)
_C_STRUCT_RE = re.compile(
    r"\bstruct\s+(?P<name>\w+)\s*\{(?P<body>[^}]*)\}",
    re.MULTILINE | re.DOTALL,
)
_C_ENUM_RE = re.compile(
    r"\benum\s+(?P<name>\w+)\s*\{(?P<body>[^}]*)\}",
    re.MULTILINE | re.DOTALL,
)
_C_TYPEDEF_RE = re.compile(
    r"\btypedef\s+(?P<orig>.+?)\s+(?P<alias>\w+)\s*;",
    re.MULTILINE,
)
_C_VAR_DECL_RE = re.compile(
    r"(?P<type>[\w\s\*]+?)\s+(?P<name>\w+)\s*(?:=\s*(?P<init>[^;]+))?\s*;",
)
_C_IF_RE = re.compile(r"\bif\s*\((?P<cond>[^)]+)\)")
_C_ELSE_RE = re.compile(r"\belse\b")
_C_WHILE_RE = re.compile(r"\bwhile\s*\((?P<cond>[^)]+)\)")
_C_FOR_RE = re.compile(
    r"\bfor\s*\(\s*(?P<init>[^;]*)\s*;\s*(?P<cond>[^;]*)\s*;\s*(?P<step>[^)]*)\)",
)
_C_SWITCH_RE = re.compile(r"\bswitch\s*\((?P<expr>[^)]+)\)")
_C_RETURN_RE = re.compile(r"\breturn\s+(?P<val>[^;]*)\s*;")
_C_CALL_RE = re.compile(r"\b(?P<name>\w+)\s*\((?P<args>[^)]*)\)")


def _extract_c_params(raw: str) -> List[ASTNode]:
    """Parse a C parameter list into parameter ASTNodes."""
    params: List[ASTNode] = []
    if not raw.strip() or raw.strip() == "void":
        return params
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        tokens = part.rsplit(None, 1)
        if len(tokens) == 2:
            p_type, p_name = tokens
        else:
            p_type, p_name = part, ""
        p_name = p_name.strip("* ")
        params.append(ASTNode(
            kind=NodeKind.PARAMETER,
            name=p_name,
            type_annotation=_normalise_c_type(p_type),
        ))
    return params


def _parse_c_body(body: str, start_line: int = 0) -> List[ASTNode]:
    """Parse C function/block body into child ASTNodes."""
    children: List[ASTNode] = []
    line_offset = start_line

    for m in _C_IF_RE.finditer(body):
        children.append(ASTNode(
            kind=NodeKind.IF,
            value=m.group("cond").strip(),
            line=line_offset + body[:m.start()].count("\n"),
            source_span=m.group(0),
        ))

    for m in _C_ELSE_RE.finditer(body):
        children.append(ASTNode(
            kind=NodeKind.ELSE,
            line=line_offset + body[:m.start()].count("\n"),
            source_span=m.group(0),
        ))

    for m in _C_WHILE_RE.finditer(body):
        children.append(ASTNode(
            kind=NodeKind.WHILE,
            value=m.group("cond").strip(),
            line=line_offset + body[:m.start()].count("\n"),
            source_span=m.group(0),
        ))

    for m in _C_FOR_RE.finditer(body):
        children.append(ASTNode(
            kind=NodeKind.FOR,
            value=f"{m.group('init')}; {m.group('cond')}; {m.group('step')}",
            line=line_offset + body[:m.start()].count("\n"),
            source_span=m.group(0),
            metadata={
                "init": m.group("init").strip(),
                "cond": m.group("cond").strip(),
                "step": m.group("step").strip(),
            },
        ))

    for m in _C_SWITCH_RE.finditer(body):
        children.append(ASTNode(
            kind=NodeKind.SWITCH,
            value=m.group("expr").strip(),
            line=line_offset + body[:m.start()].count("\n"),
            source_span=m.group(0),
        ))

    for m in _C_RETURN_RE.finditer(body):
        children.append(ASTNode(
            kind=NodeKind.RETURN,
            value=m.group("val").strip(),
            line=line_offset + body[:m.start()].count("\n"),
            source_span=m.group(0),
        ))

    for m in _C_CALL_RE.finditer(body):
        fname = m.group("name")
        if fname in ("if", "while", "for", "switch", "return", "sizeof"):
            continue
        children.append(ASTNode(
            kind=NodeKind.CALL,
            name=fname,
            value=m.group("args").strip(),
            line=line_offset + body[:m.start()].count("\n"),
            source_span=m.group(0),
        ))

    for m in _C_VAR_DECL_RE.finditer(body):
        vtype = _normalise_c_type(m.group("type"))
        vname = m.group("name")
        if vname in ("if", "else", "while", "for", "switch", "return", "struct", "enum"):
            continue
        children.append(ASTNode(
            kind=NodeKind.VARIABLE,
            name=vname,
            type_annotation=vtype,
            value=m.group("init").strip() if m.group("init") else "",
            line=line_offset + body[:m.start()].count("\n"),
            source_span=m.group(0),
        ))

    children.sort(key=lambda n: n.line)
    return children


def parse_c_ast(c_code: str) -> ASTNode:
    """Parse C source code into an AST using regex-based heuristic parsing."""
    root = ASTNode(kind=NodeKind.BLOCK, name="<c_root>")

    # structs
    for m in _C_STRUCT_RE.finditer(c_code):
        fields: List[ASTNode] = []
        for fm in _C_VAR_DECL_RE.finditer(m.group("body")):
            fields.append(ASTNode(
                kind=NodeKind.FIELD,
                name=fm.group("name"),
                type_annotation=_normalise_c_type(fm.group("type")),
            ))
        root.children.append(ASTNode(
            kind=NodeKind.STRUCT,
            name=m.group("name"),
            children=fields,
            line=c_code[:m.start()].count("\n") + 1,
            source_span=m.group(0),
        ))

    # enums
    for m in _C_ENUM_RE.finditer(c_code):
        variants: List[ASTNode] = []
        for v in m.group("body").split(","):
            v = v.strip()
            if not v:
                continue
            parts = v.split("=", 1)
            variants.append(ASTNode(
                kind=NodeKind.FIELD,
                name=parts[0].strip(),
                value=parts[1].strip() if len(parts) > 1 else "",
            ))
        root.children.append(ASTNode(
            kind=NodeKind.ENUM,
            name=m.group("name"),
            children=variants,
            line=c_code[:m.start()].count("\n") + 1,
            source_span=m.group(0),
        ))

    # typedefs
    for m in _C_TYPEDEF_RE.finditer(c_code):
        root.children.append(ASTNode(
            kind=NodeKind.TYPEDEF,
            name=m.group("alias"),
            type_annotation=_normalise_c_type(m.group("orig")),
            line=c_code[:m.start()].count("\n") + 1,
            source_span=m.group(0),
        ))

    # functions
    for m in _C_FUNC_RE.finditer(c_code):
        fn_line = c_code[:m.start()].count("\n") + 1
        params = _extract_c_params(m.group("params"))
        brace_depth = 1
        body_start = m.end()
        idx = body_start
        while idx < len(c_code) and brace_depth > 0:
            if c_code[idx] == "{":
                brace_depth += 1
            elif c_code[idx] == "}":
                brace_depth -= 1
            idx += 1
        body_text = c_code[body_start:idx - 1]
        body_children = _parse_c_body(body_text, start_line=fn_line)

        func_node = ASTNode(
            kind=NodeKind.FUNCTION,
            name=m.group("name"),
            type_annotation=_normalise_c_type(m.group("ret")),
            children=params + body_children,
            line=fn_line,
            source_span=c_code[m.start():idx],
        )
        root.children.append(func_node)

    return root


# ─── Regex-based Rust parser ────────────────────────────────────────────────

_RUST_FN_RE = re.compile(
    r"(?:pub\s+)?fn\s+(?P<name>\w+)\s*"
    r"(?:<[^>]*>)?"
    r"\s*\((?P<params>[^)]*)\)"
    r"(?:\s*->\s*(?P<ret>[^\{]+))?"
    r"\s*\{",
    re.MULTILINE,
)
_RUST_STRUCT_RE = re.compile(
    r"(?:pub\s+)?struct\s+(?P<name>\w+)(?:<[^>]*>)?\s*\{(?P<body>[^}]*)\}",
    re.MULTILINE | re.DOTALL,
)
_RUST_ENUM_RE = re.compile(
    r"(?:pub\s+)?enum\s+(?P<name>\w+)(?:<[^>]*>)?\s*\{(?P<body>[^}]*)\}",
    re.MULTILINE | re.DOTALL,
)
_RUST_IMPL_RE = re.compile(
    r"impl(?:<[^>]*>)?\s+(?P<name>\w+)(?:<[^>]*>)?\s*\{",
    re.MULTILINE,
)
_RUST_TRAIT_RE = re.compile(
    r"(?:pub\s+)?trait\s+(?P<name>\w+)(?:<[^>]*>)?\s*\{",
    re.MULTILINE,
)
_RUST_IF_RE = re.compile(r"\bif\s+(?P<cond>[^{]+)\s*\{")
_RUST_ELSE_RE = re.compile(r"\belse\b")
_RUST_WHILE_RE = re.compile(r"\bwhile\s+(?P<cond>[^{]+)\s*\{")
_RUST_FOR_RE = re.compile(r"\bfor\s+(?P<var>\w+)\s+in\s+(?P<iter>[^{]+)\s*\{")
_RUST_MATCH_RE = re.compile(r"\bmatch\s+(?P<expr>[^{]+)\s*\{")
_RUST_RETURN_RE = re.compile(r"\breturn\s+(?P<val>[^;]*)\s*;")
_RUST_LET_RE = re.compile(
    r"\blet\s+(?:mut\s+)?(?P<name>\w+)\s*(?::\s*(?P<type>[^=;]+))?\s*(?:=\s*(?P<init>[^;]+))?\s*;",
)
_RUST_CALL_RE = re.compile(r"\b(?P<name>\w+)\s*\((?P<args>[^)]*)\)")
_RUST_FIELD_RE = re.compile(
    r"(?:pub\s+)?(?P<name>\w+)\s*:\s*(?P<type>[^,}]+)",
)


def _extract_rust_params(raw: str) -> List[ASTNode]:
    params: List[ASTNode] = []
    if not raw.strip():
        return params
    for part in raw.split(","):
        part = part.strip()
        if not part or part == "self" or part == "&self" or part == "&mut self":
            continue
        halves = part.split(":", 1)
        p_name = halves[0].strip().lstrip("&").strip()
        p_type = halves[1].strip() if len(halves) > 1 else ""
        params.append(ASTNode(
            kind=NodeKind.PARAMETER,
            name=p_name.replace("mut ", "").strip(),
            type_annotation=p_type,
        ))
    return params


def _parse_rust_body(body: str, start_line: int = 0) -> List[ASTNode]:
    children: List[ASTNode] = []
    line_offset = start_line

    for m in _RUST_IF_RE.finditer(body):
        children.append(ASTNode(
            kind=NodeKind.IF,
            value=m.group("cond").strip(),
            line=line_offset + body[:m.start()].count("\n"),
            source_span=m.group(0),
        ))

    for m in _RUST_ELSE_RE.finditer(body):
        children.append(ASTNode(
            kind=NodeKind.ELSE,
            line=line_offset + body[:m.start()].count("\n"),
            source_span=m.group(0),
        ))

    for m in _RUST_WHILE_RE.finditer(body):
        children.append(ASTNode(
            kind=NodeKind.WHILE,
            value=m.group("cond").strip(),
            line=line_offset + body[:m.start()].count("\n"),
            source_span=m.group(0),
        ))

    for m in _RUST_FOR_RE.finditer(body):
        children.append(ASTNode(
            kind=NodeKind.FOR,
            value=f"{m.group('var')} in {m.group('iter').strip()}",
            line=line_offset + body[:m.start()].count("\n"),
            source_span=m.group(0),
            metadata={"var": m.group("var"), "iter": m.group("iter").strip()},
        ))

    for m in _RUST_MATCH_RE.finditer(body):
        children.append(ASTNode(
            kind=NodeKind.MATCH,
            value=m.group("expr").strip(),
            line=line_offset + body[:m.start()].count("\n"),
            source_span=m.group(0),
        ))

    for m in _RUST_RETURN_RE.finditer(body):
        children.append(ASTNode(
            kind=NodeKind.RETURN,
            value=m.group("val").strip(),
            line=line_offset + body[:m.start()].count("\n"),
            source_span=m.group(0),
        ))

    for m in _RUST_CALL_RE.finditer(body):
        fname = m.group("name")
        if fname in ("if", "while", "for", "match", "return", "let", "fn", "pub"):
            continue
        children.append(ASTNode(
            kind=NodeKind.CALL,
            name=fname,
            value=m.group("args").strip(),
            line=line_offset + body[:m.start()].count("\n"),
            source_span=m.group(0),
        ))

    for m in _RUST_LET_RE.finditer(body):
        children.append(ASTNode(
            kind=NodeKind.VARIABLE,
            name=m.group("name"),
            type_annotation=m.group("type").strip() if m.group("type") else "",
            value=m.group("init").strip() if m.group("init") else "",
            line=line_offset + body[:m.start()].count("\n"),
            source_span=m.group(0),
        ))

    children.sort(key=lambda n: n.line)
    return children


def _extract_rust_block(code: str, start: int) -> str:
    """Return body text between braces starting at *start*."""
    brace_depth = 1
    idx = start
    while idx < len(code) and brace_depth > 0:
        if code[idx] == "{":
            brace_depth += 1
        elif code[idx] == "}":
            brace_depth -= 1
        idx += 1
    return code[start:idx - 1]


def parse_rust_ast(rust_code: str) -> ASTNode:
    """Parse Rust source code into an AST using regex-based heuristic parsing."""
    root = ASTNode(kind=NodeKind.BLOCK, name="<rust_root>")

    # structs
    for m in _RUST_STRUCT_RE.finditer(rust_code):
        fields: List[ASTNode] = []
        for fm in _RUST_FIELD_RE.finditer(m.group("body")):
            fields.append(ASTNode(
                kind=NodeKind.FIELD,
                name=fm.group("name"),
                type_annotation=fm.group("type").strip().rstrip(","),
            ))
        root.children.append(ASTNode(
            kind=NodeKind.STRUCT,
            name=m.group("name"),
            children=fields,
            line=rust_code[:m.start()].count("\n") + 1,
            source_span=m.group(0),
        ))

    # enums
    for m in _RUST_ENUM_RE.finditer(rust_code):
        variants: List[ASTNode] = []
        for v in m.group("body").split(","):
            v = v.strip()
            if not v:
                continue
            vname = v.split("(")[0].split("{")[0].strip()
            if vname:
                variants.append(ASTNode(kind=NodeKind.FIELD, name=vname, value=v))
        root.children.append(ASTNode(
            kind=NodeKind.ENUM,
            name=m.group("name"),
            children=variants,
            line=rust_code[:m.start()].count("\n") + 1,
            source_span=m.group(0),
        ))

    # impl blocks
    for m in _RUST_IMPL_RE.finditer(rust_code):
        body = _extract_rust_block(rust_code, m.end())
        impl_children = _parse_rust_body(body, start_line=rust_code[:m.start()].count("\n") + 1)
        root.children.append(ASTNode(
            kind=NodeKind.IMPL,
            name=m.group("name"),
            children=impl_children,
            line=rust_code[:m.start()].count("\n") + 1,
            source_span=m.group(0) + body + "}",
        ))

    # trait blocks
    for m in _RUST_TRAIT_RE.finditer(rust_code):
        body = _extract_rust_block(rust_code, m.end())
        root.children.append(ASTNode(
            kind=NodeKind.TRAIT,
            name=m.group("name"),
            children=_parse_rust_body(body, start_line=rust_code[:m.start()].count("\n") + 1),
            line=rust_code[:m.start()].count("\n") + 1,
            source_span=m.group(0) + body + "}",
        ))

    # functions
    for m in _RUST_FN_RE.finditer(rust_code):
        fn_line = rust_code[:m.start()].count("\n") + 1
        params = _extract_rust_params(m.group("params"))
        body = _extract_rust_block(rust_code, m.end())
        body_children = _parse_rust_body(body, start_line=fn_line)
        ret = m.group("ret").strip() if m.group("ret") else "()"
        root.children.append(ASTNode(
            kind=NodeKind.FUNCTION,
            name=m.group("name"),
            type_annotation=ret,
            children=params + body_children,
            line=fn_line,
            source_span=rust_code[m.start():m.end()] + body + "}",
        ))

    return root


# ─── AST diff ───────────────────────────────────────────────────────────────

def _levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(a) < len(b):
        return _levenshtein(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


def _name_similarity(a: str, b: str) -> float:
    """Normalised name similarity (1.0 = identical)."""
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    # handle snake_case vs camelCase
    a_parts = set(re.split(r"[_\s]", a.lower()))
    b_parts = set(re.split(r"[_\s]", b.lower()))
    if a_parts and b_parts:
        jaccard = len(a_parts & b_parts) / len(a_parts | b_parts)
        if jaccard > 0.5:
            return jaccard
    max_len = max(len(a), len(b))
    return 1.0 - _levenshtein(a.lower(), b.lower()) / max_len


def _match_children(
    c_children: List[ASTNode],
    r_children: List[ASTNode],
) -> List[Tuple[Optional[ASTNode], Optional[ASTNode], float]]:
    """Greedy best-match pairing of child node lists by kind + name similarity."""
    used_r: Set[int] = set()
    pairs: List[Tuple[Optional[ASTNode], Optional[ASTNode], float]] = []

    for c_node in c_children:
        best_score = -1.0
        best_idx = -1
        for ri, r_node in enumerate(r_children):
            if ri in used_r:
                continue
            if c_node.kind == r_node.kind or (
                c_node.kind == NodeKind.SWITCH and r_node.kind == NodeKind.MATCH
            ) or (
                c_node.kind == NodeKind.FOR and r_node.kind == NodeKind.FOR
            ):
                score = _name_similarity(c_node.name, r_node.name)
                if c_node.kind == r_node.kind:
                    score += 0.3
                if score > best_score:
                    best_score = score
                    best_idx = ri
        if best_idx >= 0 and best_score > 0.1:
            used_r.add(best_idx)
            pairs.append((c_node, r_children[best_idx], best_score))
        else:
            pairs.append((c_node, None, 0.0))

    for ri, r_node in enumerate(r_children):
        if ri not in used_r:
            pairs.append((None, r_node, 0.0))

    return pairs


def _compare_nodes(
    c_node: ASTNode,
    r_node: ASTNode,
    entries: List[DiffEntry],
    depth: int = 0,
) -> None:
    """Recursively compare two matched AST nodes and record differences."""
    # kind mismatch (e.g. switch → match)
    if c_node.kind != r_node.kind:
        if c_node.kind == NodeKind.SWITCH and r_node.kind == NodeKind.MATCH:
            entries.append(DiffEntry(
                category=DiffCategory.STRUCTURAL,
                message=f"C switch translated to Rust match (line C:{c_node.line} / R:{r_node.line})",
                c_node=c_node,
                rust_node=r_node,
                severity=0.2,
            ))
        else:
            entries.append(DiffEntry(
                category=DiffCategory.STRUCTURAL,
                message=f"Node kind differs: C {c_node.kind.value} vs Rust {r_node.kind.value}",
                c_node=c_node,
                rust_node=r_node,
                severity=0.7,
            ))

    # name mismatch
    if c_node.name and r_node.name:
        sim = _name_similarity(c_node.name, r_node.name)
        if sim < 1.0 and sim > 0.3:
            entries.append(DiffEntry(
                category=DiffCategory.NAMING,
                message=f"Name changed: '{c_node.name}' → '{r_node.name}' (similarity {sim:.2f})",
                c_node=c_node,
                rust_node=r_node,
                severity=0.1,
            ))
        elif sim <= 0.3:
            entries.append(DiffEntry(
                category=DiffCategory.NAMING,
                message=f"Significant rename: '{c_node.name}' → '{r_node.name}'",
                c_node=c_node,
                rust_node=r_node,
                severity=0.4,
            ))

    # type mismatch
    if c_node.type_annotation and r_node.type_annotation:
        expected_rust = _map_c_type_to_rust(c_node.type_annotation)
        if expected_rust != r_node.type_annotation.strip():
            entries.append(DiffEntry(
                category=DiffCategory.TYPE_MISMATCH,
                message=(
                    f"Type mismatch for '{c_node.name}': "
                    f"C '{c_node.type_annotation}' (expected Rust '{expected_rust}') "
                    f"vs actual Rust '{r_node.type_annotation}'"
                ),
                c_node=c_node,
                rust_node=r_node,
                severity=0.6,
            ))

    # recurse into children
    child_pairs = _match_children(c_node.children, r_node.children)
    for c_child, r_child, _score in child_pairs:
        if c_child is not None and r_child is not None:
            _compare_nodes(c_child, r_child, entries, depth + 1)
        elif c_child is not None:
            entries.append(DiffEntry(
                category=DiffCategory.MISSING,
                message=f"C node '{c_child.kind.value}:{c_child.name}' has no Rust counterpart",
                c_node=c_child,
                severity=0.7,
            ))
        elif r_child is not None:
            entries.append(DiffEntry(
                category=DiffCategory.EXTRA,
                message=f"Rust node '{r_child.kind.value}:{r_child.name}' has no C counterpart",
                rust_node=r_child,
                severity=0.3,
            ))


def ast_diff(c_ast: ASTNode, rust_ast: ASTNode) -> ASTDiff:
    """Produce a structural diff between a C AST and a Rust AST."""
    entries: List[DiffEntry] = []
    _compare_nodes(c_ast, rust_ast, entries)
    return ASTDiff(entries=entries, c_root=c_ast, rust_root=rust_ast)


# ─── Control-flow graph extraction & diff ───────────────────────────────────

def _build_cfg(code: str, lang: str) -> List[CFGNode]:
    """Build a simplified CFG from source code."""
    nodes: List[CFGNode] = []
    node_id = 0

    entry = CFGNode(node_id=node_id, label="entry", kind="entry")
    nodes.append(entry)
    node_id += 1

    if_pat = _C_IF_RE if lang == "c" else _RUST_IF_RE
    while_pat = _C_WHILE_RE if lang == "c" else _RUST_WHILE_RE
    for_pat = _C_FOR_RE if lang == "c" else _RUST_FOR_RE
    ret_pat = _C_RETURN_RE if lang == "c" else _RUST_RETURN_RE

    prev_id = 0

    for m in if_pat.finditer(code):
        cond_val = m.group(1) if m.lastindex else ""
        n = CFGNode(
            node_id=node_id,
            label=f"if({cond_val.strip()[:40]})",
            kind="branch",
            source_line=code[:m.start()].count("\n") + 1,
        )
        nodes[prev_id].successors.append(node_id)
        n.predecessors.append(prev_id)
        nodes.append(n)
        prev_id = node_id
        node_id += 1

    for m in while_pat.finditer(code):
        cond_val = m.group(1) if m.lastindex else ""
        n = CFGNode(
            node_id=node_id,
            label=f"while({cond_val.strip()[:40]})",
            kind="loop",
            source_line=code[:m.start()].count("\n") + 1,
        )
        nodes[prev_id].successors.append(node_id)
        n.predecessors.append(prev_id)
        n.successors.append(node_id)  # back-edge
        nodes.append(n)
        prev_id = node_id
        node_id += 1

    for m in for_pat.finditer(code):
        n = CFGNode(
            node_id=node_id,
            label=f"for(...)",
            kind="loop",
            source_line=code[:m.start()].count("\n") + 1,
        )
        nodes[prev_id].successors.append(node_id)
        n.predecessors.append(prev_id)
        n.successors.append(node_id)
        nodes.append(n)
        prev_id = node_id
        node_id += 1

    for m in ret_pat.finditer(code):
        n = CFGNode(
            node_id=node_id,
            label=f"return",
            kind="return",
            source_line=code[:m.start()].count("\n") + 1,
        )
        nodes[prev_id].successors.append(node_id)
        n.predecessors.append(prev_id)
        nodes.append(n)
        prev_id = node_id
        node_id += 1

    # match / switch
    if lang == "c":
        for m in _C_SWITCH_RE.finditer(code):
            n = CFGNode(
                node_id=node_id,
                label=f"switch({m.group('expr').strip()[:30]})",
                kind="branch",
                source_line=code[:m.start()].count("\n") + 1,
            )
            nodes[prev_id].successors.append(node_id)
            n.predecessors.append(prev_id)
            nodes.append(n)
            prev_id = node_id
            node_id += 1
    else:
        for m in _RUST_MATCH_RE.finditer(code):
            n = CFGNode(
                node_id=node_id,
                label=f"match({m.group('expr').strip()[:30]})",
                kind="branch",
                source_line=code[:m.start()].count("\n") + 1,
            )
            nodes[prev_id].successors.append(node_id)
            n.predecessors.append(prev_id)
            nodes.append(n)
            prev_id = node_id
            node_id += 1

    exit_node = CFGNode(node_id=node_id, label="exit", kind="exit")
    nodes[prev_id].successors.append(node_id)
    exit_node.predecessors.append(prev_id)
    nodes.append(exit_node)

    return nodes


def _cfg_edges(nodes: List[CFGNode]) -> Set[Tuple[str, str]]:
    edges: Set[Tuple[str, str]] = set()
    node_map = {n.node_id: n for n in nodes}
    for n in nodes:
        for s in n.successors:
            if s in node_map:
                edges.add((n.label, node_map[s].label))
    return edges


def control_flow_diff(c_code: str, rust_code: str) -> CFGDiff:
    """Compare control-flow graphs extracted from C and Rust source."""
    c_nodes = _build_cfg(c_code, "c")
    r_nodes = _build_cfg(rust_code, "rust")

    c_edges = _cfg_edges(c_nodes)
    r_edges = _cfg_edges(r_nodes)

    added = r_edges - c_edges
    removed = c_edges - r_edges

    changes: List[str] = []

    c_kinds = [n.kind for n in c_nodes if n.kind not in ("entry", "exit")]
    r_kinds = [n.kind for n in r_nodes if n.kind not in ("entry", "exit")]
    if len(c_kinds) != len(r_kinds):
        changes.append(
            f"CFG node count differs: C has {len(c_kinds)} interior nodes, "
            f"Rust has {len(r_kinds)}"
        )

    c_loops = sum(1 for k in c_kinds if k == "loop")
    r_loops = sum(1 for k in r_kinds if k == "loop")
    if c_loops != r_loops:
        changes.append(f"Loop count differs: C={c_loops}, Rust={r_loops}")

    c_branches = sum(1 for k in c_kinds if k == "branch")
    r_branches = sum(1 for k in r_kinds if k == "branch")
    if c_branches != r_branches:
        changes.append(f"Branch count differs: C={c_branches}, Rust={r_branches}")

    return CFGDiff(
        c_nodes=c_nodes,
        rust_nodes=r_nodes,
        added_edges=[(hash(a), hash(b)) for a, b in added],
        removed_edges=[(hash(a), hash(b)) for a, b in removed],
        structural_changes=changes,
    )


# ─── Data-flow analysis ────────────────────────────────────────────────────

def _extract_def_use_c(code: str) -> Dict[str, List[Tuple[int, str]]]:
    """Extract variable def-use chains from C code."""
    chains: Dict[str, List[Tuple[int, str]]] = {}

    for m in _C_VAR_DECL_RE.finditer(code):
        vname = m.group("name")
        if vname in ("if", "else", "while", "for", "switch", "return", "struct"):
            continue
        line = code[:m.start()].count("\n") + 1
        chains.setdefault(vname, []).append((line, "def"))

    assign_re = re.compile(r"\b(?P<name>\w+)\s*(?<!=)=(?!=)\s*(?P<val>[^;]+);")
    for m in assign_re.finditer(code):
        vname = m.group("name")
        if vname in chains:
            line = code[:m.start()].count("\n") + 1
            chains[vname].append((line, "def"))

    for vname in list(chains.keys()):
        use_re = re.compile(r"\b" + re.escape(vname) + r"\b")
        for m in use_re.finditer(code):
            line = code[:m.start()].count("\n") + 1
            if not any(e == (line, "def") for e in chains[vname]):
                chains[vname].append((line, "use"))
        chains[vname].sort(key=lambda t: t[0])

    return chains


def _extract_def_use_rust(code: str) -> Dict[str, List[Tuple[int, str]]]:
    """Extract variable def-use chains from Rust code."""
    chains: Dict[str, List[Tuple[int, str]]] = {}

    for m in _RUST_LET_RE.finditer(code):
        vname = m.group("name")
        line = code[:m.start()].count("\n") + 1
        chains.setdefault(vname, []).append((line, "def"))

    assign_re = re.compile(r"\b(?P<name>\w+)\s*=\s*(?P<val>[^;]+);")
    for m in assign_re.finditer(code):
        vname = m.group("name")
        if vname in chains:
            line = code[:m.start()].count("\n") + 1
            if not any(e == (line, "def") for e in chains.get(vname, [])):
                chains[vname].append((line, "def"))

    for vname in list(chains.keys()):
        use_re = re.compile(r"\b" + re.escape(vname) + r"\b")
        for m in use_re.finditer(code):
            line = code[:m.start()].count("\n") + 1
            if not any(e == (line, "def") for e in chains.get(vname, [])):
                chains[vname].append((line, "use"))
        chains[vname].sort(key=lambda t: t[0])

    return chains


def data_flow_diff(c_code: str, rust_code: str) -> DataFlowDiff:
    """Compare data-flow (def-use chains) between C and Rust source."""
    c_chains = _extract_def_use_c(c_code)
    r_chains = _extract_def_use_rust(rust_code)

    c_vars = set(c_chains.keys())
    r_vars = set(r_chains.keys())

    missing_defs = sorted(c_vars - r_vars)
    extra_defs = sorted(r_vars - c_vars)

    changed_uses: List[Tuple[str, str]] = []
    for var in c_vars & r_vars:
        c_use_count = sum(1 for _, act in c_chains[var] if act == "use")
        r_use_count = sum(1 for _, act in r_chains[var] if act == "use")
        if c_use_count != r_use_count:
            changed_uses.append((
                var,
                f"C has {c_use_count} uses, Rust has {r_use_count} uses",
            ))

    return DataFlowDiff(
        c_chains=c_chains,
        rust_chains=r_chains,
        missing_defs=missing_defs,
        extra_defs=extra_defs,
        changed_uses=changed_uses,
    )


# ─── Call graph extraction & diff ───────────────────────────────────────────

def _extract_call_graph_c(code: str) -> Set[Tuple[str, str]]:
    """Extract caller→callee edges from C source."""
    edges: Set[Tuple[str, str]] = set()
    current_fn: Optional[str] = None

    for m in _C_FUNC_RE.finditer(code):
        current_fn = m.group("name")
        brace_depth = 1
        idx = m.end()
        while idx < len(code) and brace_depth > 0:
            if code[idx] == "{":
                brace_depth += 1
            elif code[idx] == "}":
                brace_depth -= 1
            idx += 1
        body = code[m.end():idx - 1]
        for cm in _C_CALL_RE.finditer(body):
            callee = cm.group("name")
            if callee not in (
                "if", "while", "for", "switch", "return", "sizeof", "printf",
                "fprintf", "sprintf", "snprintf",
            ):
                edges.add((current_fn, callee))
    return edges


def _extract_call_graph_rust(code: str) -> Set[Tuple[str, str]]:
    """Extract caller→callee edges from Rust source."""
    edges: Set[Tuple[str, str]] = set()

    for m in _RUST_FN_RE.finditer(code):
        current_fn = m.group("name")
        body = _extract_rust_block(code, m.end())
        for cm in _RUST_CALL_RE.finditer(body):
            callee = cm.group("name")
            if callee not in (
                "if", "while", "for", "match", "return", "let", "fn", "pub",
                "println", "eprintln", "format", "panic", "todo", "unimplemented",
            ):
                edges.add((current_fn, callee))
    return edges


def _fuzzy_match_functions(
    c_names: Set[str], r_names: Set[str],
) -> Dict[str, str]:
    """Best-effort fuzzy matching of C function names to Rust names."""
    mapping: Dict[str, str] = {}
    used: Set[str] = set()
    for cn in sorted(c_names):
        best_sim = 0.0
        best_rn = ""
        for rn in sorted(r_names):
            if rn in used:
                continue
            sim = _name_similarity(cn, rn)
            if sim > best_sim:
                best_sim = sim
                best_rn = rn
        if best_sim >= 0.4 and best_rn:
            mapping[cn] = best_rn
            used.add(best_rn)
    return mapping


def call_graph_diff(c_project: str, rust_project: str) -> CallGraphDiff:
    """Compare function call graphs between C and Rust source code."""
    c_edges = _extract_call_graph_c(c_project)
    r_edges = _extract_call_graph_rust(rust_project)

    c_fns = {caller for caller, _ in c_edges} | {callee for _, callee in c_edges}
    r_fns = {caller for caller, _ in r_edges} | {callee for _, callee in r_edges}
    renamed = _fuzzy_match_functions(c_fns, r_fns)

    normalised_c: Set[Tuple[str, str]] = set()
    for caller, callee in c_edges:
        nc = renamed.get(caller, caller)
        ne = renamed.get(callee, callee)
        normalised_c.add((nc, ne))

    added = r_edges - normalised_c
    removed = normalised_c - r_edges

    return CallGraphDiff(
        c_edges=c_edges,
        rust_edges=r_edges,
        added_calls=added,
        removed_calls=removed,
        renamed_functions=renamed,
    )


# ─── Type hierarchy diff ───────────────────────────────────────────────────

def _parse_type_name(raw: str) -> Tuple[str, str]:
    """Return (kind, name) from a type declaration string."""
    raw = raw.strip()
    if raw.startswith("struct "):
        return ("struct", raw[7:].strip().rstrip("{; "))
    if raw.startswith("enum "):
        return ("enum", raw[5:].strip().rstrip("{; "))
    if raw.startswith("typedef "):
        parts = raw[8:].rsplit(None, 1)
        return ("typedef", parts[-1].rstrip("; ") if parts else raw)
    if raw.startswith("impl "):
        return ("impl", raw[5:].strip().rstrip("{; "))
    if raw.startswith("trait "):
        return ("trait", raw[6:].strip().rstrip("{; "))
    return ("type", raw.rstrip("; "))


def type_hierarchy_diff(
    c_types: List[str], rust_types: List[str],
) -> TypeDiff:
    """Compare type hierarchies between C and Rust declarations."""
    c_parsed = [_parse_type_name(t) for t in c_types]
    r_parsed = [_parse_type_name(t) for t in rust_types]

    matched: List[Tuple[str, str, float]] = []
    unmatched_c: List[str] = []
    unmatched_rust: List[str] = []
    warnings: List[str] = []

    used_r: Set[int] = set()
    for ck, cn in c_parsed:
        best_score = 0.0
        best_idx = -1
        for ri, (rk, rn) in enumerate(r_parsed):
            if ri in used_r:
                continue
            sim = _name_similarity(cn, rn)
            kind_bonus = 0.2 if ck == rk else 0.0
            if ck == "struct" and rk == "struct":
                kind_bonus = 0.3
            score = sim + kind_bonus
            if score > best_score:
                best_score = score
                best_idx = ri
        if best_idx >= 0 and best_score >= 0.4:
            rk2, rn2 = r_parsed[best_idx]
            used_r.add(best_idx)
            matched.append((cn, rn2, best_score))
            if ck != rk2:
                warnings.append(
                    f"Type kind changed: C {ck} '{cn}' → Rust {rk2} '{rn2}'"
                )
        else:
            unmatched_c.append(cn)

    for ri, (rk, rn) in enumerate(r_parsed):
        if ri not in used_r:
            unmatched_rust.append(rn)

    c_mapped = set()
    for ct in c_types:
        norm = _normalise_c_type(ct.split()[-1].rstrip(";{ "))
        rust_equiv = _map_c_type_to_rust(norm)
        if rust_equiv != norm:
            c_mapped.add((norm, rust_equiv))
    for cn, rn in c_mapped:
        if rn not in [r for _, r, _ in matched]:
            warnings.append(
                f"Primitive type mapping: C '{cn}' should map to Rust '{rn}'"
            )

    return TypeDiff(
        c_types=c_types,
        rust_types=rust_types,
        matched=matched,
        unmatched_c=unmatched_c,
        unmatched_rust=unmatched_rust,
        warnings=warnings,
    )


# ─── Similarity score ──────────────────────────────────────────────────────

def compute_similarity_score(diff: ASTDiff) -> float:
    """Compute a 0–1 similarity metric from an ASTDiff.

    Higher values mean the two ASTs are more similar.  Weights each diff
    entry by its severity.
    """
    if not diff.entries:
        return 1.0

    total_nodes = 0
    if diff.c_root:
        total_nodes += diff.c_root.size
    if diff.rust_root:
        total_nodes += diff.rust_root.size
    total_nodes = max(total_nodes, 1)

    weighted_penalty = sum(e.severity for e in diff.entries)
    raw = 1.0 - (weighted_penalty / total_nodes)
    return max(0.0, min(1.0, raw))


# ─── DOT graph visualisation ───────────────────────────────────────────────

def generate_dot_graph(diff: ASTDiff) -> str:
    """Generate a Graphviz DOT representation of the diff."""
    lines: List[str] = [
        "digraph ASTDiff {",
        '  rankdir=TB;',
        '  node [shape=box, fontname="Courier"];',
        '  edge [fontname="Courier", fontsize=10];',
        "",
    ]

    def _dot_id(prefix: str, node: ASTNode) -> str:
        return f"{prefix}_{id(node)}"

    def _emit_tree(node: ASTNode, prefix: str) -> None:
        nid = _dot_id(prefix, node)
        label = f"{node.kind.value}"
        if node.name:
            label += f"\\n{node.name}"
        if node.type_annotation:
            label += f"\\n: {node.type_annotation}"
        colour = "#a0c4ff" if prefix == "c" else "#bdb2ff"
        lines.append(f'  {nid} [label="{label}", style=filled, fillcolor="{colour}"];')
        for child in node.children:
            cid = _dot_id(prefix, child)
            lines.append(f"  {nid} -> {cid};")
            _emit_tree(child, prefix)

    if diff.c_root:
        lines.append("  subgraph cluster_c {")
        lines.append('    label="C AST";')
        _emit_tree(diff.c_root, "c")
        lines.append("  }")
        lines.append("")

    if diff.rust_root:
        lines.append("  subgraph cluster_rust {")
        lines.append('    label="Rust AST";')
        _emit_tree(diff.rust_root, "rust")
        lines.append("  }")
        lines.append("")

    for entry in diff.entries:
        if entry.c_node and entry.rust_node:
            c_id = _dot_id("c", entry.c_node)
            r_id = _dot_id("rust", entry.rust_node)
            colour = "#ff6b6b" if entry.severity >= 0.7 else "#ffd93d"
            esc_msg = entry.message[:60].replace('"', '\\"')
            lines.append(
                f'  {c_id} -> {r_id} [label="{esc_msg}", '
                f'color="{colour}", style=dashed, constraint=false];'
            )

    lines.append("}")
    return "\n".join(lines)


# ─── HTML side-by-side diff ────────────────────────────────────────────────

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>AST Diff: C → Rust</title>
<style>
  body {{ font-family: "Segoe UI", system-ui, sans-serif; margin: 0; padding: 20px; background: #1e1e2e; color: #cdd6f4; }}
  h1 {{ text-align: center; color: #89b4fa; }}
  .container {{ display: flex; gap: 16px; }}
  .pane {{ flex: 1; background: #181825; border-radius: 8px; padding: 16px; overflow-x: auto; }}
  .pane h2 {{ margin-top: 0; color: #a6e3a1; }}
  pre {{ white-space: pre-wrap; font-size: 13px; line-height: 1.5; }}
  .line {{ display: block; }}
  .line-num {{ display: inline-block; width: 3em; color: #585b70; text-align: right; padding-right: 1em; user-select: none; }}
  .diff-entry {{ padding: 6px 10px; margin: 4px 0; border-radius: 4px; font-size: 13px; }}
  .sev-high {{ background: #45243644; border-left: 3px solid #f38ba8; }}
  .sev-med  {{ background: #40354044; border-left: 3px solid #fab387; }}
  .sev-low  {{ background: #2d3a3a44; border-left: 3px solid #a6e3a1; }}
  .summary {{ text-align: center; padding: 12px; font-size: 15px; }}
  .score {{ font-size: 28px; font-weight: bold; }}
</style>
</head>
<body>
<h1>AST Diff: C &rarr; Rust</h1>
<div class="summary">
  Similarity: <span class="score">{score:.0%}</span>
  &nbsp;|&nbsp; {total} differences found
</div>
<div class="container">
  <div class="pane">
    <h2>C Source</h2>
    <pre>{c_html}</pre>
  </div>
  <div class="pane">
    <h2>Rust Source</h2>
    <pre>{r_html}</pre>
  </div>
</div>
<h2 style="padding-left:16px;">Differences</h2>
<div style="padding: 0 16px 40px;">
{diff_html}
</div>
</body>
</html>
"""


def _source_to_html(source: str) -> str:
    """Convert source code to numbered HTML lines."""
    escaped = html.escape(source)
    out_lines: List[str] = []
    for i, line in enumerate(escaped.splitlines(), 1):
        out_lines.append(
            f'<span class="line"><span class="line-num">{i}</span>{line}</span>'
        )
    return "\n".join(out_lines)


def _severity_class(sev: float) -> str:
    if sev >= 0.7:
        return "sev-high"
    if sev >= 0.4:
        return "sev-med"
    return "sev-low"


def generate_html_diff(
    c_code: str, rust_code: str, diff: ASTDiff,
) -> str:
    """Generate an HTML side-by-side view of the C/Rust diff."""
    score = compute_similarity_score(diff)
    c_html = _source_to_html(c_code)
    r_html = _source_to_html(rust_code)

    diff_items: List[str] = []
    for entry in diff.entries:
        cls = _severity_class(entry.severity)
        msg = html.escape(entry.label)
        loc_parts: List[str] = []
        if entry.c_node and entry.c_node.line:
            loc_parts.append(f"C line {entry.c_node.line}")
        if entry.rust_node and entry.rust_node.line:
            loc_parts.append(f"Rust line {entry.rust_node.line}")
        loc = f" ({', '.join(loc_parts)})" if loc_parts else ""
        diff_items.append(f'<div class="diff-entry {cls}">{msg}{html.escape(loc)}</div>')

    diff_html = "\n".join(diff_items) if diff_items else '<div class="diff-entry sev-low">No differences found.</div>'

    return _HTML_TEMPLATE.format(
        score=score,
        total=diff.total,
        c_html=c_html,
        r_html=r_html,
        diff_html=diff_html,
    )
