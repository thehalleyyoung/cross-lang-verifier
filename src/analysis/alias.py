"""
Points-to / alias analysis for the Cross-Language Equivalence Verifier.

Provides Andersen-style (subset-based) and Steensgaard-style (union-find)
points-to analyses, field-sensitive analysis, pointer provenance tracking,
and an alias query interface (must-alias, may-alias, no-alias).
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from ..ir.basic_block import BasicBlock
from ..ir.function import Function
from ..ir.module import Module
from ..ir.instructions import (
    Instruction,
    AllocaInst,
    LoadInst,
    StoreInst,
    GetElementPtrInst,
    CastInst,
    CallInst,
    PhiInst,
    SelectInst,
    Value,
    Constant,
    Argument,
    CastKind,
)
from ..ir.types import (
    IRType,
    PointerType,
    StructType,
    ArrayType,
    VoidType,
)


# ── Alias result ─────────────────────────────────────────────────────────

class AliasResult(Enum):
    """Result of an alias query."""
    MUST_ALIAS = auto()    # Definitely the same location
    MAY_ALIAS = auto()     # Might be the same location
    NO_ALIAS = auto()      # Definitely different locations

    def __str__(self) -> str:
        return self.name

    @property
    def may_overlap(self) -> bool:
        return self is not AliasResult.NO_ALIAS

    @property
    def must_overlap(self) -> bool:
        return self is AliasResult.MUST_ALIAS


# ── Abstract memory locations ────────────────────────────────────────────

@dataclass(frozen=True)
class MemoryLocation:
    """An abstract memory location (allocation site or field thereof)."""
    base_id: int          # Value.id of the base allocation
    field_path: tuple[int, ...] = ()  # Sequence of field indices for field-sensitivity
    name: str = ""

    @property
    def is_field(self) -> bool:
        return len(self.field_path) > 0

    @property
    def base_location(self) -> "MemoryLocation":
        return MemoryLocation(self.base_id, name=self.name)

    def with_field(self, field_idx: int) -> "MemoryLocation":
        return MemoryLocation(self.base_id, self.field_path + (field_idx,), self.name)

    def __str__(self) -> str:
        name = self.name or f"loc_{self.base_id}"
        if self.field_path:
            fields = ".".join(str(f) for f in self.field_path)
            return f"{name}.{fields}"
        return name


# ── Points-to set ────────────────────────────────────────────────────────

class PointsToSet:
    """A set of memory locations that a pointer may point to."""

    def __init__(self, locations: set[MemoryLocation] | None = None) -> None:
        self._locations: set[MemoryLocation] = locations or set()

    def __contains__(self, loc: MemoryLocation) -> bool:
        return loc in self._locations

    def __iter__(self) -> Iterator[MemoryLocation]:
        return iter(self._locations)

    def __len__(self) -> int:
        return len(self._locations)

    def __bool__(self) -> bool:
        return bool(self._locations)

    def add(self, loc: MemoryLocation) -> None:
        self._locations.add(loc)

    def union(self, other: "PointsToSet") -> "PointsToSet":
        return PointsToSet(self._locations | other._locations)

    def intersection(self, other: "PointsToSet") -> "PointsToSet":
        return PointsToSet(self._locations & other._locations)

    def is_subset(self, other: "PointsToSet") -> bool:
        return self._locations.issubset(other._locations)

    def overlaps(self, other: "PointsToSet") -> bool:
        return bool(self._locations & other._locations)

    def copy(self) -> "PointsToSet":
        return PointsToSet(set(self._locations))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PointsToSet):
            return NotImplemented
        return self._locations == other._locations

    def __hash__(self) -> int:
        return hash(frozenset(self._locations))

    def __repr__(self) -> str:
        if not self._locations:
            return "∅"
        return "{" + ", ".join(str(l) for l in sorted(self._locations, key=str)) + "}"


# ── Constraint types for Andersen's analysis ─────────────────────────────

class ConstraintKind(Enum):
    ADDR_OF = auto()     # p = &q  → pts(p) ⊇ {q}
    COPY = auto()        # p = q   → pts(p) ⊇ pts(q)
    LOAD = auto()        # p = *q  → ∀ r ∈ pts(q): pts(p) ⊇ pts(r)
    STORE = auto()       # *p = q  → ∀ r ∈ pts(p): pts(r) ⊇ pts(q)
    GEP = auto()         # p = &q[i] → pts(p) ⊇ {q.field(i)}


@dataclass(frozen=True)
class Constraint:
    kind: ConstraintKind
    lhs: int          # Value.id of the left-hand side
    rhs: int          # Value.id of the right-hand side
    field_idx: int = 0  # For GEP constraints

    def __str__(self) -> str:
        return f"{self.kind.name}(lhs=%{self.lhs}, rhs=%{self.rhs})"


# ── Andersen-style points-to analysis ────────────────────────────────────

class AndersenAnalysis:
    """Andersen's subset-based flow-insensitive points-to analysis.

    Computes, for each pointer variable, the set of abstract memory
    locations it may point to.  Handles: alloca, load, store, GEP,
    pointer casts, phi nodes, select instructions, and function calls
    (conservatively).

    Complexity: O(n³) worst case, but typically much better.
    """

    def __init__(self, field_sensitive: bool = True) -> None:
        self.field_sensitive = field_sensitive
        self._pts: dict[int, PointsToSet] = defaultdict(PointsToSet)
        self._constraints: list[Constraint] = []
        self._locations: dict[int, MemoryLocation] = {}  # alloc id → location

    @property
    def points_to(self) -> dict[int, PointsToSet]:
        return dict(self._pts)

    def get_points_to(self, value: Value) -> PointsToSet:
        return self._pts.get(value.id, PointsToSet())

    def analyze_function(self, function: Function) -> None:
        """Run the analysis on a single function."""
        self._collect_constraints(function)
        self._solve()

    def analyze_module(self, module: Module) -> None:
        """Run the analysis on all functions in a module."""
        for func in module.iter_functions():
            self._collect_constraints(func)
        # Handle globals
        for gv in module.iter_globals():
            loc = MemoryLocation(base_id=id(gv), name=gv.name)
            self._locations[id(gv)] = loc
        self._solve()

    def alias_query(self, p: Value, q: Value) -> AliasResult:
        """Query the alias relationship between two pointer values."""
        pts_p = self._pts.get(p.id, PointsToSet())
        pts_q = self._pts.get(q.id, PointsToSet())
        return _alias_from_pts(pts_p, pts_q)

    # ── Constraint collection ────────────────────────────────────────────

    def _collect_constraints(self, function: Function) -> None:
        for block in function.blocks:
            for inst in block:
                self._process_instruction(inst)

        # Arguments are unknown pointers
        for arg in function.arguments:
            if isinstance(arg.type, PointerType):
                loc = MemoryLocation(base_id=arg.id, name=arg.name)
                self._locations[arg.id] = loc
                self._pts[arg.id].add(loc)

    def _process_instruction(self, inst: Instruction) -> None:
        if isinstance(inst, AllocaInst):
            loc = MemoryLocation(base_id=inst.id, name=inst.name)
            self._locations[inst.id] = loc
            self._pts[inst.id].add(loc)

        elif isinstance(inst, StoreInst):
            # *address = value → STORE constraint
            addr = inst.address
            val = inst.value
            if isinstance(val.type, PointerType):
                self._constraints.append(Constraint(
                    kind=ConstraintKind.STORE,
                    lhs=addr.id,
                    rhs=val.id,
                ))

        elif isinstance(inst, LoadInst):
            # result = *address → LOAD constraint
            addr = inst.address
            if isinstance(inst.type, PointerType):
                self._constraints.append(Constraint(
                    kind=ConstraintKind.LOAD,
                    lhs=inst.id,
                    rhs=addr.id,
                ))

        elif isinstance(inst, GetElementPtrInst):
            base = inst.operands[0] if inst.operands else None
            if base is not None:
                if self.field_sensitive and len(inst.operands) > 2:
                    # Field-sensitive: try to extract field index
                    idx_val = inst.operands[-1]
                    field_idx = 0
                    if isinstance(idx_val, Constant) and isinstance(idx_val.value, int):
                        field_idx = idx_val.value
                    self._constraints.append(Constraint(
                        kind=ConstraintKind.GEP,
                        lhs=inst.id,
                        rhs=base.id,
                        field_idx=field_idx,
                    ))
                else:
                    # Field-insensitive: treat as copy
                    self._constraints.append(Constraint(
                        kind=ConstraintKind.COPY,
                        lhs=inst.id,
                        rhs=base.id,
                    ))

        elif isinstance(inst, CastInst):
            if inst.cast_kind is CastKind.BITCAST:
                if isinstance(inst.type, PointerType):
                    self._constraints.append(Constraint(
                        kind=ConstraintKind.COPY,
                        lhs=inst.id,
                        rhs=inst.operand.id,
                    ))
            elif inst.cast_kind is CastKind.INTTOPTR:
                # Conservative: points to unknown
                pass

        elif isinstance(inst, PhiInst):
            if isinstance(inst.type, PointerType):
                for val, _block in inst.incoming:
                    self._constraints.append(Constraint(
                        kind=ConstraintKind.COPY,
                        lhs=inst.id,
                        rhs=val.id,
                    ))

        elif isinstance(inst, SelectInst):
            if isinstance(inst.type, PointerType):
                self._constraints.append(Constraint(
                    kind=ConstraintKind.COPY,
                    lhs=inst.id,
                    rhs=inst.true_value.id,
                ))
                self._constraints.append(Constraint(
                    kind=ConstraintKind.COPY,
                    lhs=inst.id,
                    rhs=inst.false_value.id,
                ))

        elif isinstance(inst, CallInst):
            # Conservative: return value may point to anything passed in
            if isinstance(inst.type, PointerType):
                for op in inst.operands[1:]:
                    if isinstance(op.type, PointerType):
                        self._constraints.append(Constraint(
                            kind=ConstraintKind.COPY,
                            lhs=inst.id,
                            rhs=op.id,
                        ))

    # ── Solver ───────────────────────────────────────────────────────────

    def _solve(self) -> None:
        """Iterative worklist solver for Andersen's constraints."""
        # First apply all ADDR_OF and direct GEP constraints
        for c in self._constraints:
            if c.kind is ConstraintKind.GEP:
                rhs_pts = self._pts.get(c.rhs, PointsToSet())
                for loc in rhs_pts:
                    new_loc = loc.with_field(c.field_idx) if self.field_sensitive else loc
                    self._pts[c.lhs].add(new_loc)

        # Worklist: all variables that have non-empty points-to sets
        worklist: deque[int] = deque()
        in_worklist: set[int] = set()
        for vid, pts in self._pts.items():
            if pts:
                worklist.append(vid)
                in_worklist.add(vid)

        max_iter = len(self._constraints) * 100 + 1000
        iterations = 0

        while worklist and iterations < max_iter:
            vid = worklist.popleft()
            in_worklist.discard(vid)
            iterations += 1

            for c in self._constraints:
                changed = False

                if c.kind is ConstraintKind.COPY and c.rhs == vid:
                    old_size = len(self._pts[c.lhs])
                    for loc in self._pts[vid]:
                        self._pts[c.lhs].add(loc)
                    if len(self._pts[c.lhs]) > old_size:
                        changed = True
                        if c.lhs not in in_worklist:
                            worklist.append(c.lhs)
                            in_worklist.add(c.lhs)

                elif c.kind is ConstraintKind.LOAD and c.rhs == vid:
                    # p = *q: for each r in pts(q), pts(p) ⊇ pts(r)
                    for loc in self._pts[vid]:
                        old_size = len(self._pts[c.lhs])
                        for target_loc in self._pts.get(loc.base_id, PointsToSet()):
                            self._pts[c.lhs].add(target_loc)
                        if len(self._pts[c.lhs]) > old_size:
                            if c.lhs not in in_worklist:
                                worklist.append(c.lhs)
                                in_worklist.add(c.lhs)

                elif c.kind is ConstraintKind.STORE and c.lhs == vid:
                    # *p = q: for each r in pts(p), pts(r) ⊇ pts(q)
                    for loc in self._pts[vid]:
                        old_size = len(self._pts.get(loc.base_id, PointsToSet()))
                        for src_loc in self._pts.get(c.rhs, PointsToSet()):
                            self._pts[loc.base_id].add(src_loc)
                        new_size = len(self._pts.get(loc.base_id, PointsToSet()))
                        if new_size > old_size:
                            if loc.base_id not in in_worklist:
                                worklist.append(loc.base_id)
                                in_worklist.add(loc.base_id)

                elif c.kind is ConstraintKind.GEP and c.rhs == vid:
                    old_size = len(self._pts[c.lhs])
                    for loc in self._pts[vid]:
                        new_loc = loc.with_field(c.field_idx) if self.field_sensitive else loc
                        self._pts[c.lhs].add(new_loc)
                    if len(self._pts[c.lhs]) > old_size:
                        if c.lhs not in in_worklist:
                            worklist.append(c.lhs)
                            in_worklist.add(c.lhs)

    def summary(self) -> str:
        lines = [f"Andersen analysis ({len(self._pts)} pointer variables):"]
        for vid in sorted(self._pts.keys()):
            pts = self._pts[vid]
            if pts:
                lines.append(f"  %{vid} → {pts}")
        return "\n".join(lines)


# ── Steensgaard-style (union-find) analysis ──────────────────────────────

class _UnionFind:
    """Union-Find data structure for Steensgaard's analysis."""

    def __init__(self) -> None:
        self._parent: dict[int, int] = {}
        self._rank: dict[int, int] = {}

    def make_set(self, x: int) -> None:
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0

    def find(self, x: int) -> int:
        if x not in self._parent:
            self.make_set(x)
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])  # Path compression
        return self._parent[x]

    def union(self, x: int, y: int) -> int:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return rx
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1
        return rx

    def same_set(self, x: int, y: int) -> bool:
        return self.find(x) == self.find(y)


class SteensgaardAnalysis:
    """Steensgaard's union-find-based points-to analysis.

    A fast approximation (nearly linear time) that unifies pointer
    targets: if p may point to x, and p may point to y, then x and y
    are unified into the same equivalence class.

    Less precise than Andersen but much faster for large programs.
    """

    def __init__(self) -> None:
        self._uf = _UnionFind()
        self._locations: dict[int, MemoryLocation] = {}
        self._pts_representatives: dict[int, set[int]] = defaultdict(set)

    def analyze_function(self, function: Function) -> None:
        for block in function.blocks:
            for inst in block:
                self._process_instruction(inst)

        for arg in function.arguments:
            self._uf.make_set(arg.id)

    def _process_instruction(self, inst: Instruction) -> None:
        self._uf.make_set(inst.id)

        if isinstance(inst, AllocaInst):
            loc = MemoryLocation(base_id=inst.id, name=inst.name)
            self._locations[inst.id] = loc

        elif isinstance(inst, StoreInst):
            self._uf.make_set(inst.address.id)
            self._uf.make_set(inst.value.id)
            if isinstance(inst.value.type, PointerType):
                # Unify stored value with targets of address
                self._pts_representatives[self._uf.find(inst.address.id)].add(
                    self._uf.find(inst.value.id)
                )

        elif isinstance(inst, LoadInst):
            self._uf.make_set(inst.address.id)
            if isinstance(inst.type, PointerType):
                self._pts_representatives[self._uf.find(inst.address.id)].add(
                    self._uf.find(inst.id)
                )

        elif isinstance(inst, GetElementPtrInst):
            if inst.operands:
                base = inst.operands[0]
                self._uf.make_set(base.id)
                self._uf.union(inst.id, base.id)

        elif isinstance(inst, CastInst):
            if isinstance(inst.type, PointerType) and isinstance(inst.operand.type, PointerType):
                self._uf.make_set(inst.operand.id)
                self._uf.union(inst.id, inst.operand.id)

        elif isinstance(inst, PhiInst):
            if isinstance(inst.type, PointerType):
                for val, _ in inst.incoming:
                    self._uf.make_set(val.id)
                    self._uf.union(inst.id, val.id)

        elif isinstance(inst, SelectInst):
            if isinstance(inst.type, PointerType):
                self._uf.make_set(inst.true_value.id)
                self._uf.make_set(inst.false_value.id)
                self._uf.union(inst.id, inst.true_value.id)
                self._uf.union(inst.id, inst.false_value.id)

    def alias_query(self, p: Value, q: Value) -> AliasResult:
        """Check alias relationship using equivalence classes."""
        p_rep = self._uf.find(p.id)
        q_rep = self._uf.find(q.id)
        if p_rep == q_rep:
            return AliasResult.MAY_ALIAS
        # Check if they share any points-to targets
        p_targets = self._pts_representatives.get(p_rep, set())
        q_targets = self._pts_representatives.get(q_rep, set())
        if p_targets & q_targets:
            return AliasResult.MAY_ALIAS
        return AliasResult.NO_ALIAS

    def same_class(self, p: Value, q: Value) -> bool:
        return self._uf.same_set(p.id, q.id)

    def summary(self) -> str:
        # Group variables by representative
        classes: dict[int, list[int]] = defaultdict(list)
        for vid in self._uf._parent:
            classes[self._uf.find(vid)].append(vid)
        lines = [f"Steensgaard analysis ({len(classes)} equivalence classes):"]
        for rep, members in sorted(classes.items()):
            member_strs = [f"%{m}" for m in sorted(members)]
            lines.append(f"  [{rep}] = {{{', '.join(member_strs)}}}")
        return "\n".join(lines)


# ── Alias query interface ────────────────────────────────────────────────

class AliasQuery:
    """High-level alias query interface that wraps an analysis backend."""

    def __init__(
        self,
        function: Function | None = None,
        use_andersen: bool = True,
        field_sensitive: bool = True,
    ) -> None:
        self._function = function
        if use_andersen:
            self._backend = AndersenAnalysis(field_sensitive=field_sensitive)
        else:
            self._backend = SteensgaardAnalysis()

        if function is not None:
            if isinstance(self._backend, AndersenAnalysis):
                self._backend.analyze_function(function)
            else:
                self._backend.analyze_function(function)

    def query(self, p: Value, q: Value) -> AliasResult:
        """Query the alias relationship between p and q."""
        return self._backend.alias_query(p, q)

    def must_alias(self, p: Value, q: Value) -> bool:
        return self.query(p, q) is AliasResult.MUST_ALIAS

    def may_alias(self, p: Value, q: Value) -> bool:
        return self.query(p, q).may_overlap

    def no_alias(self, p: Value, q: Value) -> bool:
        return self.query(p, q) is AliasResult.NO_ALIAS

    def points_to(self, p: Value) -> PointsToSet:
        """Return the points-to set for p."""
        if isinstance(self._backend, AndersenAnalysis):
            return self._backend.get_points_to(p)
        return PointsToSet()  # Steensgaard doesn't track explicit sets


# ── Helper ───────────────────────────────────────────────────────────────

def _alias_from_pts(pts_a: PointsToSet, pts_b: PointsToSet) -> AliasResult:
    """Determine alias result from two points-to sets."""
    if not pts_a or not pts_b:
        return AliasResult.MAY_ALIAS  # Unknown → conservative

    if len(pts_a) == 1 and len(pts_b) == 1:
        a_loc = next(iter(pts_a))
        b_loc = next(iter(pts_b))
        if a_loc == b_loc:
            return AliasResult.MUST_ALIAS
        return AliasResult.NO_ALIAS

    if pts_a.overlaps(pts_b):
        return AliasResult.MAY_ALIAS

    return AliasResult.NO_ALIAS
